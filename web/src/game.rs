//! Game session management
//!
//! Wraps the EngineContext to provide a convenient API for the web server.

use algorithm_core::BuiltinAlgorithm;
use anyhow::{anyhow, Result};
use engine_core::board_profile::{BoardGameMetadata, BoardView};
use engine_core::{AgentId, EngineContext, EpisodeStatus, ErasedTimestep};
#[cfg(feature = "onnx")]
use mcts::{MctsConfig, MctsSearch, SharedOnnxEvaluator};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
// Note: Uses std::sync::RwLock (not tokio) because this is shared with model_watcher
// crate which requires std::sync::RwLock. bot_move() only clones the current
// evaluator handle out of the lock and releases it before searching, so a hot
// reload never waits on MCTS and MCTS never blocks a reload.
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(feature = "onnx")]
use tracing::debug;

use crate::types::{GameStateResponse, PositionRecord};
use std::collections::VecDeque;
mod analysis;
mod validation;
#[cfg(not(feature = "onnx"))]
use crate::SharedOnnxEvaluator;
use validation::{active_observation, validate_position, ExpectedTransition};

// =============================================================================
// Configuration Constants
// =============================================================================

/// Number of MCTS simulations for web play (less than training, more than eval)
#[cfg(feature = "onnx")]
const MCTS_SIMULATIONS: u32 = 200;

/// MCTS temperature for web play (some randomness but not too much)
#[cfg(feature = "onnx")]
const MCTS_TEMPERATURE: f32 = 0.5;

/// Default human player number (1 = goes first)
const DEFAULT_HUMAN_PLAYER: u8 = 1;

// =============================================================================
// Game Session
// =============================================================================

/// A game session tracking current state
pub struct GameSession {
    session_id: String,
    revision: u64,
    history: VecDeque<PositionRecord>,
    model_info: Arc<RwLock<crate::ModelInfo>>,
    ctx: EngineContext,
    /// Explicit board-game profile required by this AlphaZero web cartridge.
    board: BoardGameMetadata,
    /// Current encoded state
    state: Vec<u8>,
    /// Complete algorithm-neutral transition envelope. The web adapter
    /// validates and narrows this to one active AlphaZero board player.
    timestep: ErasedTimestep,
    /// The engine's display projection of `state` — board contents, player to
    /// act, and winner. Never decoded here: state byte layout is private to
    /// each game.
    view: BoardView,
    /// Which player the human is (1 or 2). Set when game starts based on who goes first.
    human_player: u8,
    /// RNG for bot moves
    rng: ChaCha20Rng,
    /// Shared evaluator for MCTS (loaded from model file)
    #[cfg(feature = "onnx")]
    evaluator: Arc<RwLock<Option<SharedOnnxEvaluator>>>,
    /// Stub evaluator when ONNX is disabled
    #[cfg(not(feature = "onnx"))]
    #[allow(dead_code)]
    evaluator: Arc<RwLock<Option<SharedOnnxEvaluator>>>,
    /// MCTS configuration for bot play
    #[cfg(feature = "onnx")]
    mcts_config: MctsConfig,
    /// Reusable simulation context for MCTS (avoids repeated registry lookups)
    /// Separate from `ctx` because MCTS needs its own context for simulations
    #[cfg(feature = "onnx")]
    mcts_sim_ctx: Option<EngineContext>,
}

impl GameSession {
    /// Create a new game session with default (empty) evaluator.
    /// Used in tests; production code uses `with_evaluator` for hot-reloading.
    #[cfg(test)]
    pub fn new(env_id: &str) -> Result<Self> {
        Self::with_evaluator(env_id, Arc::new(RwLock::new(None)))
    }

    /// Create a new game session with a shared evaluator (for hot-reloading).
    /// Without the `onnx` feature the evaluator is a stub and MCTS is skipped.
    pub fn with_evaluator(
        env_id: &str,
        evaluator: Arc<RwLock<Option<SharedOnnxEvaluator>>>,
    ) -> Result<Self> {
        let mut ctx = EngineContext::new(env_id)
            .map_err(|error| anyhow!("Environment '{env_id}' is unavailable: {error}"))?;

        BuiltinAlgorithm::AlphaZeroBoardV1
            .compatibility(&ctx)
            .require_compatible()?;

        let metadata = ctx.metadata();
        let board = metadata.require_board()?.clone();

        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let reset = ctx.reset(seed, &[])?;
        let view = validate_position(
            &ctx,
            &reset.state,
            &reset.timestep,
            &board,
            ExpectedTransition::Reset,
        )?;

        // Configure MCTS for playing (less exploration than training)
        #[cfg(feature = "onnx")]
        let mcts_config = MctsConfig::for_evaluation()
            .with_simulations(MCTS_SIMULATIONS)
            .with_temperature(MCTS_TEMPERATURE);

        // Pre-create simulation context for MCTS (avoids repeated registry lookups)
        #[cfg(feature = "onnx")]
        let mcts_sim_ctx = Some(
            EngineContext::new(env_id)
                .map_err(|error| anyhow!("Environment '{env_id}' is unavailable: {error}"))?,
        );

        let mut session = Self {
            session_id: format!("{:032x}", rand::random::<u128>()),
            revision: 0,
            history: VecDeque::new(),
            model_info: Arc::new(RwLock::new(crate::ModelInfo::default())),
            ctx,
            board,
            state: reset.state,
            timestep: reset.timestep,
            view,
            human_player: DEFAULT_HUMAN_PLAYER,
            rng: ChaCha20Rng::seed_from_u64(seed),
            evaluator,
            #[cfg(feature = "onnx")]
            mcts_config,
            #[cfg(feature = "onnx")]
            mcts_sim_ctx,
        };
        session.record_position()?;
        Ok(session)
    }

    /// Player to act (1 or 2)
    pub fn current_player(&self) -> u8 {
        self.view.current_player
    }

    /// Winner (0=ongoing, 1, 2, 3=draw)
    pub fn winner(&self) -> u8 {
        self.view.winner
    }

    /// Get legal moves by extracting from observation using metadata
    pub fn legal_moves(&self) -> Result<Vec<u32>> {
        if self.is_game_over() {
            return Ok(Vec::new());
        }

        Ok(active_observation(&self.timestep)?
            .2
            .iter_ones()
            .map(|i| i as u32)
            .collect())
    }

    /// Check if a move is legal by extracting from observation using metadata
    pub fn is_legal_move(&self, position: u32) -> Result<bool> {
        if self.is_game_over() {
            return Ok(false);
        }

        Ok(active_observation(&self.timestep)?
            .2
            .is_legal(position as usize))
    }

    /// Check if game is over
    pub fn is_game_over(&self) -> bool {
        self.timestep.episode.is_done()
    }

    /// Set which player the human is (called when game starts)
    pub fn set_human_player(&mut self, player: u8) -> Result<()> {
        if !matches!(player, 1 | 2) {
            return Err(anyhow!(
                "AlphaZero web serving only supports board seats 1 and 2, got {player}"
            ));
        }
        self.human_player = player;
        if let Some(record) = self.history.back_mut() {
            record.state.human_player = player;
        }
        // Refresh the status text as well as the selected seat.
        let state = self.to_response()?;
        if let Some(record) = self.history.back_mut() {
            record.state = state;
        }
        Ok(())
    }

    /// Check if it's the human's turn
    pub fn is_human_turn(&self) -> bool {
        self.timestep.episode == EpisodeStatus::Running
            && self.current_player() == self.human_player
    }

    /// Make a player move
    pub fn player_move(&mut self, position: u32) -> Result<()> {
        let analysis = self.decision(position, crate::types::DecisionSource::Human);
        self.apply_decision(position, analysis)
    }

    /// Make a bot move using MCTS if model is available, otherwise random
    #[cfg(feature = "onnx")]
    pub fn bot_move(&mut self) -> Result<u32> {
        let legal = self.legal_moves()?;
        if legal.is_empty() {
            return Err(anyhow!("No legal moves available"));
        }

        // Snapshot the current model and release the reload lock before the
        // search. The clone pins this move to one model generation while a
        // concurrent hot reload can proceed immediately.
        let (evaluator, checkpoint) = {
            let guard = self
                .evaluator
                .read()
                .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
            // The filesystem and S3 writers acquire evaluator then model_info.
            // Holding both read guards gives one consistent model generation.
            let info = self
                .model_info
                .read()
                .map_err(|e| anyhow!("Model info lock: {e}"))?;
            let checkpoint = if guard.is_some() {
                Some(crate::types::CheckpointIdentity {
                    checkpoint_id: info
                        .checkpoint_id
                        .clone()
                        .ok_or_else(|| anyhow!("Loaded evaluator has no checkpoint identity"))?,
                    training_step: info.training_step,
                })
            } else {
                None
            };
            (guard.clone(), checkpoint)
        };

        let (position, analysis) = if let Some(evaluator) = evaluator {
            // Try to use MCTS with neural network
            debug!("Attempting MCTS for bot move");

            let mcts_result = (|| -> Result<_> {
                // Use pre-created simulation context (avoids repeated registry lookups)
                let sim_ctx = self
                    .mcts_sim_ctx
                    .as_mut()
                    .ok_or_else(|| anyhow!("Simulation context not available"))?;

                let mut search = MctsSearch::new(
                    sim_ctx,
                    &evaluator,
                    self.mcts_config.clone(),
                    self.state.clone(),
                    self.timestep.clone(),
                )?;
                let (result, diagnostics) = search.run_with_diagnostics(&mut self.rng)?;

                debug!(
                    action = result.action,
                    value = result.value,
                    simulations = result.simulations,
                    "MCTS selected move"
                );

                Ok((result, diagnostics))
            })();

            let (result, diagnostics) = mcts_result.map_err(|error| {
                anyhow!(
                    "Loaded model failed during MCTS; refusing to hide the runtime error: {error}"
                )
            })?;
            let analysis = self.search_analysis(&result, &diagnostics, checkpoint);
            (result.action, analysis)
        } else {
            // Fall back to random move
            debug!("No model loaded, using random move");
            use rand::seq::SliceRandom;
            let position = *legal.choose(&mut self.rng).unwrap();
            (position, self.random_analysis(position, &legal))
        };

        self.apply_decision(position, analysis)?;
        Ok(position)
    }

    /// Make a bot move using random selection (when ONNX is disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn bot_move(&mut self) -> Result<u32> {
        let legal = self.legal_moves()?;
        if legal.is_empty() {
            return Err(anyhow!("No legal moves available"));
        }

        // Random move selection
        use rand::seq::SliceRandom;
        let position = *legal.choose(&mut self.rng).unwrap();

        let analysis = self.random_analysis(position, &legal);
        self.apply_decision(position, analysis)?;
        Ok(position)
    }

    /// Internal move execution
    fn make_move(&mut self, position: u32) -> Result<()> {
        if !self.is_legal_move(position)? {
            return Err(anyhow!("illegal board action {position}"));
        }
        let actor = active_observation(&self.timestep)?.0;

        // Encode action as u32 little-endian
        let action = position.to_le_bytes().to_vec();

        let step = self.ctx.step(&self.state, &action)?;
        let view = validate_position(
            &self.ctx,
            &step.state,
            &step.timestep,
            &self.board,
            ExpectedTransition::Agent(actor),
        )?;

        // Commit the new session position only after the complete transition
        // and presentation pass the AlphaZero-board boundary checks.
        self.state = step.state;
        self.timestep = step.timestep;
        self.view = view;
        self.revision += 1;
        self.record_position()?;

        Ok(())
    }

    /// Convert to API response
    pub fn to_response(&self) -> Result<GameStateResponse> {
        let human_symbol = &self.board.players[(self.human_player - 1) as usize].symbol;
        let bot_symbol = &self.board.players[(2 - self.human_player) as usize].symbol;

        let message = match self.timestep.episode {
            EpisodeStatus::Running => {
                if self.is_human_turn() {
                    format!("Your turn ({})", human_symbol)
                } else {
                    format!("Bot's turn ({})", bot_symbol)
                }
            }
            EpisodeStatus::Truncated => "Game truncated".to_string(),
            EpisodeStatus::Terminated => match self.winner() {
                w if w == self.human_player => "You win!".to_string(),
                3 => "It's a draw!".to_string(),
                _ => "Bot wins!".to_string(),
            },
        };

        Ok(GameStateResponse {
            session_id: self.session_id.clone(),
            revision: self.revision,
            actions: self
                .legal_moves()?
                .into_iter()
                .map(|action| {
                    self.ctx
                        .describe_discrete_action(AgentId::from(self.current_player()), action)
                        .unwrap_or(engine_core::ActionPresentation {
                            action,
                            label: format!("Action {action}"),
                            target: None,
                        })
                })
                .collect(),
            cells: self.view.cells.clone(),
            current_player: self.current_player(),
            human_player: self.human_player,
            winner: self.winner(),
            game_over: self.is_game_over(),
            legal_moves: self.legal_moves()?,
            message,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game_session() {
        engine_games::register_all_environments();

        let session = GameSession::new("tictactoe").unwrap();

        assert_eq!(session.view.owners(), vec![0u8; 9]);
        assert_eq!(session.current_player(), 1);
        assert_eq!(session.winner(), 0);
        assert_eq!(session.legal_moves().unwrap().len(), 9);
    }

    #[test]
    fn test_player_move() {
        engine_games::register_all_environments();

        let mut session = GameSession::new("tictactoe").unwrap();
        session.player_move(4).unwrap(); // Center

        assert_eq!(session.view.cells[4].owner, 1); // X placed
        assert_eq!(session.current_player(), 2); // Now O's turn
        assert!(!session.legal_moves().unwrap().contains(&4));
    }

    #[test]
    fn test_bot_move() {
        engine_games::register_all_environments();

        let mut session = GameSession::new("tictactoe").unwrap();
        session.player_move(4).unwrap();

        let bot_pos = session.bot_move().unwrap();

        assert!(bot_pos < 9);
        assert_ne!(bot_pos, 4);
        assert_eq!(session.view.cells[bot_pos as usize].owner, 2); // O placed
        assert_eq!(session.current_player(), 1); // Back to X
    }

    #[test]
    fn test_illegal_move() {
        engine_games::register_all_environments();

        let mut session = GameSession::new("tictactoe").unwrap();
        session.player_move(4).unwrap();

        // Position 4 is now occupied
        assert!(!session.is_legal_move(4).unwrap());
        let before = serde_json::to_value(session.history(Some(0), 64)).unwrap();
        let state = session.state.clone();
        let timestep = session.timestep.clone();

        assert_eq!(
            session.player_move(4).unwrap_err().to_string(),
            "illegal board action 4"
        );
        assert_eq!(session.state, state);
        assert_eq!(session.timestep, timestep);
        assert_eq!(
            serde_json::to_value(session.history(Some(0), 64)).unwrap(),
            before
        );
    }

    #[test]
    fn rejected_presentation_does_not_commit_position_or_history() {
        engine_games::register_all_environments();
        let mut session = GameSession::new("tictactoe").unwrap();
        let response = serde_json::to_value(session.to_response().unwrap()).unwrap();
        let history = serde_json::to_value(session.history(Some(0), 64)).unwrap();
        let state = session.state.clone();
        let timestep = session.timestep.clone();

        // The engine step succeeds, but its projection cannot match this
        // deliberately inconsistent session metadata.
        session.board.width = 4;
        assert_eq!(
            session.player_move(4).unwrap_err().to_string(),
            "board presentation has 9 cells, metadata declares 4x3 (12 cells)"
        );

        assert_eq!(session.state, state);
        assert_eq!(session.timestep, timestep);
        assert_eq!(
            serde_json::to_value(session.to_response().unwrap()).unwrap(),
            response
        );
        assert_eq!(
            serde_json::to_value(session.history(Some(0), 64)).unwrap(),
            history
        );
    }

    #[test]
    fn terminal_board_view_agrees_with_per_agent_outcomes() {
        engine_games::register_all_environments();

        let mut session = GameSession::new("tictactoe").unwrap();
        for action in [0, 3, 1, 4, 2] {
            session.player_move(action).unwrap();
        }

        assert_eq!(session.timestep.episode, EpisodeStatus::Terminated);
        assert_eq!(session.winner(), 1);
        assert_eq!(session.timestep.reward_for(AgentId(1)), Some(1.0));
        assert_eq!(session.timestep.reward_for(AgentId(2)), Some(-1.0));
        assert!(session.is_game_over());
        assert!(session.legal_moves().unwrap().is_empty());
    }
}
