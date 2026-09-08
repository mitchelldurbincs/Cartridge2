use super::*;
use crate::types::*;

const HISTORY_LIMIT: usize = 256;

impl GameSession {
    pub fn with_model_info(mut self, info: Arc<RwLock<crate::ModelInfo>>) -> Self {
        self.model_info = info;
        self
    }

    pub fn position_key(&self) -> PositionKey {
        PositionKey { session_id: self.session_id.clone(), revision: self.revision }
    }

    pub fn matches_position(&self, key: &PositionKey) -> bool {
        self.session_id == key.session_id && self.revision == key.revision
    }

    pub(super) fn record_position(&mut self) -> Result<()> {
        self.history.push_back(PositionRecord { state: self.to_response()?, decision: None });
        while self.history.len() > HISTORY_LIMIT { self.history.pop_front(); }
        Ok(())
    }

    pub(super) fn apply_decision(&mut self, action: u32, analysis: DecisionAnalysis) -> Result<()> {
        let previous = self.revision;
        self.make_move(action)?;
        if let Some(record) = self.history.iter_mut().find(|r| r.state.revision == previous) {
            record.decision = Some(analysis);
        }
        Ok(())
    }

    pub(super) fn decision(&self, action: u32, source: DecisionSource) -> DecisionAnalysis {
        DecisionAnalysis {
            schema_version: 1,
            analysis_id: format!("{}:{}", self.session_id, self.revision),
            position: self.position_key(),
            env_id: self.ctx.metadata().id,
            env_contract_version: self.ctx.capabilities().contract_version,
            algorithm_id: "alphazero_board_v1".into(),
            actor: u32::from(self.current_player()),
            source,
            checkpoint: None,
            selected_action: ActionReference::Discrete { index: action },
            network_value: None,
            search_value: None,
            actions: Vec::new(),
            search: None,
        }
    }

    pub(super) fn random_analysis(&self, action: u32, legal: &[u32]) -> DecisionAnalysis {
        let mut analysis = self.decision(action, DecisionSource::Random);
        analysis.actions = legal.iter().map(|&index| ActionAssessment {
            action: ActionReference::Discrete { index },
            network_prior: None, search_prior: None, visit_share: None,
            selection_probability: Some(1.0 / legal.len() as f32),
            visits: None, q_value: None, expanded: None,
        }).collect();
        analysis
    }

    #[cfg(feature = "onnx")]
    pub(super) fn search_analysis(&self, result: &mcts::SearchResult, diagnostics: &mcts::RootDiagnostics,
        checkpoint: Option<CheckpointIdentity>) -> DecisionAnalysis {
        let mut analysis = self.decision(result.action, DecisionSource::AlphaZeroMcts);
        let estimate = |value| ValueEstimate { value,
            perspective_agent: analysis.actor, quantity: ValueQuantity::ExpectedOutcome,
            bounds: Some([-1.0, 1.0]) };
        analysis.network_value = Some(estimate(diagnostics.network_value));
        analysis.search_value = Some(estimate(result.value));
        analysis.checkpoint = checkpoint;
        analysis.actions = diagnostics.actions.iter().map(|a| ActionAssessment {
            action: ActionReference::Discrete { index: a.action },
            network_prior: Some(a.network_prior), search_prior: a.search_prior,
            visit_share: Some(a.visit_share), selection_probability: Some(a.selection_probability),
            visits: Some(a.visits), q_value: a.q_value, expanded: Some(a.expanded),
        }).collect();
        analysis.search = Some(SearchEffort {
            completed_simulations: diagnostics.completed_simulations,
            root_visits: diagnostics.root_visits,
            neural_evaluations: diagnostics.neural_evaluations,
            total_time_us: result.stats.total_time_us,
            temperature: diagnostics.temperature,
        });
        analysis
    }

    /// Bounded pages; default to the most recent page. Requests can explicitly
    /// page older retained positions. GET never starts inference.
    pub fn history(&self, from_revision: Option<u64>, limit: usize) -> HistoryResponse {
        let limit = limit.clamp(1, 64);
        let first = self.history.front().map_or(self.revision, |r| r.state.revision);
        let from = from_revision.unwrap_or_else(|| self.revision.saturating_sub(limit as u64 - 1));
        let records: Vec<_> = self.history.iter().filter(|r| r.state.revision >= from)
            .take(limit).cloned().collect();
        let next_revision = records.last().and_then(|r| {
            (r.state.revision < self.revision).then_some(r.state.revision + 1)
        });
        HistoryResponse { schema_version: 1, session_id: self.session_id.clone(),
            first_available_revision: first, current_revision: self.revision, records, next_revision }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_human_and_bot_decisions_on_their_pre_move_positions() {
        engine_games::register_all_environments();
        let mut session = GameSession::new("tictactoe").unwrap();
        let initial = session.position_key();
        session.player_move(4).unwrap();
        session.bot_move().unwrap();
        let history = session.history(None, 64);
        assert_eq!(history.records.len(), 3);
        assert_eq!(history.records[0].state.cells[4].owner, 0);
        assert!(matches!(history.records[0].decision.as_ref().unwrap().source, DecisionSource::Human));
        let before_bot = &history.records[1];
        assert_eq!(before_bot.state.cells[4].owner, 1);
        assert_eq!(before_bot.state.current_player, 2);
        let decision = before_bot.decision.as_ref().unwrap();
        assert_eq!(decision.position.revision, 1);
        assert_eq!(decision.position.session_id, initial.session_id);
        assert!(matches!(decision.source, DecisionSource::Random));
        assert!(decision.network_value.is_none());
        assert!(decision.search_value.is_none());
        assert!(decision.checkpoint.is_none());
        assert!(decision.search.is_none());
        assert_eq!(decision.actions.len(), 8);
        assert!(history.records[2].decision.is_none());
        assert!(!session.matches_position(&initial));
    }

    #[test]
    fn bot_first_records_actor_one_and_human_seat_two() {
        engine_games::register_all_environments();
        let mut session = GameSession::new("connect4").unwrap();
        session.set_human_player(2).unwrap();
        session.bot_move().unwrap();
        let history = session.history(None, 64);
        assert_eq!(history.records[0].state.human_player, 2);
        assert_eq!(history.records[0].decision.as_ref().unwrap().actor, 1);
        assert_eq!(history.records[1].state.current_player, 2);
    }

    #[test]
    fn history_is_bounded_paged_and_reset_has_a_new_identity() {
        engine_games::register_all_environments();
        let mut session = GameSession::new("tictactoe").unwrap();
        // Exercise trace retention without making up game transitions.
        for revision in 1..300 {
            session.revision = revision;
            session.record_position().unwrap();
        }
        let oldest = session.history(Some(0), usize::MAX);
        assert_eq!(session.history.len(), 256);
        assert_eq!(oldest.first_available_revision, 44);
        assert_eq!(oldest.records.len(), 64);
        assert_eq!(oldest.next_revision, Some(108));
        let latest = session.history(None, 64);
        assert_eq!(latest.records.last().unwrap().state.revision, 299);
        assert!(latest.next_revision.is_none());
        assert_ne!(session.position_key().session_id,
            GameSession::new("tictactoe").unwrap().position_key().session_id);
    }
}
