//! Game session management
//!
//! Wraps the EngineContext to provide a convenient API for the web server.

use algorithm_core::BuiltinAlgorithm;
use anyhow::{anyhow, Result};
use engine_core::board_profile::{BoardGameMetadata, BoardView};
use engine_core::{
    AgentId, Decision, EngineContext, EpisodeStatus, ErasedTimestep, Presentation, TransitionSource,
};
#[cfg(feature = "onnx")]
use mcts::{run_mcts, MctsConfig, OnnxEvaluator};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
// Note: Uses std::sync::RwLock (not tokio) because this is shared with model_watcher
// crate which requires std::sync::RwLock. The lock is only held briefly during
// synchronous bot_move() calls, never across await points.
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(feature = "onnx")]
use tracing::debug;

use crate::types::GameStateResponse;
#[cfg(not(feature = "onnx"))]
use crate::OnnxEvaluator;

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
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// Stub evaluator when ONNX is disabled
    #[cfg(not(feature = "onnx"))]
    #[allow(dead_code)]
    evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
    /// MCTS configuration for bot play
    #[cfg(feature = "onnx")]
    mcts_config: MctsConfig,
    /// Reusable simulation context for MCTS (avoids repeated registry lookups)
    /// Separate from `ctx` because MCTS needs its own context for simulations
    #[cfg(feature = "onnx")]
    mcts_sim_ctx: Option<EngineContext>,
}

#[derive(Debug, Clone, Copy)]
enum ExpectedTransition {
    Reset,
    Agent(AgentId),
}

/// Narrow one generic timestep to the position shape this serving cartridge
/// understands. Chance and simultaneous decisions are rejected here rather
/// than being assigned a synthetic board player.
fn active_observation(timestep: &ErasedTimestep) -> Result<(AgentId, &[u8])> {
    if timestep.episode != EpisodeStatus::Running {
        return Err(anyhow!(
            "AlphaZero board action selection requires a running episode, got {:?}",
            timestep.episode
        ));
    }

    let active_agent = match &timestep.decision {
        Decision::Agents { agent_ids } if agent_ids.len() == 1 => agent_ids[0],
        decision => anyhow::bail!(
            "AlphaZero board serving requires exactly one active decision agent, got {decision:?}"
        ),
    };
    if !matches!(active_agent, AgentId(1) | AgentId(2)) {
        return Err(anyhow!(
            "AlphaZero board serving only supports seats 1 and 2, got {}",
            active_agent.0
        ));
    }

    let observation = timestep.sole_observation()?;
    if observation.agent_id != active_agent {
        return Err(anyhow!(
            "sole observation belongs to agent {}, but active decision belongs to agent {}",
            observation.agent_id.0,
            active_agent.0
        ));
    }
    Ok((active_agent, observation.data.as_slice()))
}

fn validate_two_player_outcomes(timestep: &ErasedTimestep) -> Result<()> {
    if timestep.outcomes.len() != 2 {
        return Err(anyhow!(
            "AlphaZero board timestep requires exactly two per-agent outcomes, got {}",
            timestep.outcomes.len()
        ));
    }

    for agent_id in [AgentId(1), AgentId(2)] {
        let matches = timestep
            .outcomes
            .iter()
            .filter(|outcome| outcome.agent_id == agent_id)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(anyhow!(
                "AlphaZero board timestep requires exactly one outcome for agent {}, got {}",
                agent_id.0,
                matches.len()
            ));
        }
        let outcome = matches[0];
        if !outcome.reward.is_finite() {
            return Err(anyhow!(
                "AlphaZero board timestep has a non-finite reward for agent {}",
                agent_id.0
            ));
        }
        let flags_match = match timestep.episode {
            EpisodeStatus::Running => !outcome.terminated && !outcome.truncated,
            EpisodeStatus::Terminated => outcome.terminated && !outcome.truncated,
            EpisodeStatus::Truncated => !outcome.terminated && outcome.truncated,
        };
        if !flags_match {
            return Err(anyhow!(
                "outcome flags for agent {} disagree with episode status {:?}",
                agent_id.0,
                timestep.episode
            ));
        }
    }

    let seat_one = timestep
        .reward_for(AgentId(1))
        .expect("validated outcome for seat 1");
    let seat_two = timestep
        .reward_for(AgentId(2))
        .expect("validated outcome for seat 2");
    match timestep.episode {
        EpisodeStatus::Running | EpisodeStatus::Truncated if seat_one != 0.0 || seat_two != 0.0 => {
            Err(anyhow!(
                "AlphaZero terminal-only reward contract emitted ({seat_one}, {seat_two}) for {:?}",
                timestep.episode
            ))
        }
        EpisodeStatus::Terminated if (seat_one + seat_two).abs() > 1e-6 => Err(anyhow!(
            "AlphaZero terminal rewards must be zero-sum, got ({seat_one}, {seat_two})"
        )),
        _ => Ok(()),
    }
}

fn validate_timestep(timestep: &ErasedTimestep, expected: ExpectedTransition) -> Result<()> {
    match (expected, &timestep.source) {
        (ExpectedTransition::Reset, TransitionSource::Reset) => {}
        (ExpectedTransition::Agent(expected_actor), TransitionSource::Agents { agent_ids })
            if agent_ids.as_slice() == [expected_actor] => {}
        (ExpectedTransition::Reset, source) => {
            return Err(anyhow!(
                "reset timestep has invalid transition source {source:?}"
            ))
        }
        (ExpectedTransition::Agent(expected_actor), source) => {
            return Err(anyhow!(
                "step by agent {} has invalid transition source {source:?}",
                expected_actor.0
            ))
        }
    }

    validate_two_player_outcomes(timestep)?;
    match timestep.episode {
        EpisodeStatus::Running => {
            let (next_agent, _) = active_observation(timestep)?;
            if let ExpectedTransition::Agent(actor) = expected {
                if next_agent == actor {
                    return Err(anyhow!(
                        "alternating-turn board transition kept agent {} active",
                        actor.0
                    ));
                }
            }
        }
        EpisodeStatus::Terminated | EpisodeStatus::Truncated => {
            if timestep.decision != Decision::None {
                return Err(anyhow!(
                    "completed board timestep must have no next decision, got {:?}",
                    timestep.decision
                ));
            }
            let observation = timestep.sole_observation()?;
            if !matches!(observation.agent_id, AgentId(1) | AgentId(2)) {
                return Err(anyhow!(
                    "completed board observation belongs to unsupported agent {}",
                    observation.agent_id.0
                ));
            }
        }
    }
    Ok(())
}

fn require_board_view(
    ctx: &EngineContext,
    state: &[u8],
    timestep: &ErasedTimestep,
    board: &BoardGameMetadata,
) -> Result<BoardView> {
    let view = require_board_presentation(ctx.presentation(state)?)?;

    let expected_cells = board.board_size()?;
    if view.cells.len() != expected_cells {
        return Err(anyhow!(
            "board presentation has {} cells, metadata declares {}x{} ({expected_cells} cells)",
            view.cells.len(),
            board.width,
            board.height
        ));
    }
    if !matches!(view.current_player, 1 | 2) {
        return Err(anyhow!(
            "board presentation current player must be seat 1 or 2, got {}",
            view.current_player
        ));
    }
    if view.cells.iter().any(|cell| cell.owner > 2) {
        return Err(anyhow!(
            "board presentation contains an owner outside seats 1 and 2"
        ));
    }
    let observation = timestep.sole_observation()?;
    if observation.agent_id.0 != u32::from(view.current_player) {
        return Err(anyhow!(
            "board presentation current player {} disagrees with sole observation agent {}",
            view.current_player,
            observation.agent_id.0
        ));
    }

    match timestep.episode {
        EpisodeStatus::Running => {
            if view.winner != 0 {
                return Err(anyhow!(
                    "running episode has terminal board winner {}",
                    view.winner
                ));
            }
            let (active_agent, _) = active_observation(timestep)?;
            if u32::from(view.current_player) != active_agent.0 {
                return Err(anyhow!(
                    "board presentation current player {} disagrees with active agent {}",
                    view.current_player,
                    active_agent.0
                ));
            }
        }
        EpisodeStatus::Terminated => {
            if !matches!(view.winner, 1..=3) {
                return Err(anyhow!(
                    "terminated episode requires winner 1, 2, or draw marker 3, got {}",
                    view.winner
                ));
            }
            let seat_one = timestep
                .reward_for(AgentId(1))
                .ok_or_else(|| anyhow!("terminal board timestep is missing seat 1 reward"))?;
            let seat_two = timestep
                .reward_for(AgentId(2))
                .ok_or_else(|| anyhow!("terminal board timestep is missing seat 2 reward"))?;
            let rewards_match_winner = match view.winner {
                1 => seat_one > 0.0 && seat_two < 0.0,
                2 => seat_one < 0.0 && seat_two > 0.0,
                3 => seat_one == 0.0 && seat_two == 0.0,
                _ => unreachable!("winner range validated above"),
            };
            if !rewards_match_winner {
                return Err(anyhow!(
                    "board winner {} disagrees with per-agent rewards ({seat_one}, {seat_two})",
                    view.winner
                ));
            }
        }
        EpisodeStatus::Truncated => {
            if view.winner != 0 {
                return Err(anyhow!(
                    "truncated episode must not fabricate a winner, got {}",
                    view.winner
                ));
            }
        }
    }

    Ok(view)
}

fn require_board_presentation(presentation: Option<Presentation>) -> Result<BoardView> {
    match presentation {
        Some(Presentation::Board { view }) => Ok(view),
        Some(Presentation::Custom { contract, .. }) => Err(anyhow!(
            "AlphaZero web serving requires a board presentation, got custom contract '{contract}'"
        )),
        None => Err(anyhow!(
            "AlphaZero web serving requires the environment to expose a board presentation"
        )),
    }
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
        evaluator: Arc<RwLock<Option<OnnxEvaluator>>>,
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
        validate_timestep(&reset.timestep, ExpectedTransition::Reset)?;
        let view = require_board_view(&ctx, &reset.state, &reset.timestep, &board)?;

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

        Ok(Self {
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
        })
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

        Ok(self
            .board
            .extract_legal_moves(active_observation(&self.timestep)?.1)
            .map_err(|error| anyhow!(error))?
            .into_iter()
            .map(|i| i as u32)
            .collect())
    }

    /// Check if a move is legal by extracting from observation using metadata
    pub fn is_legal_move(&self, position: u32) -> Result<bool> {
        if self.is_game_over() {
            return Ok(false);
        }

        self.board
            .is_action_legal(active_observation(&self.timestep)?.1, position as usize)
            .map_err(|error| anyhow!(error))
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
        Ok(())
    }

    /// Check if it's the human's turn
    pub fn is_human_turn(&self) -> bool {
        self.timestep.episode == EpisodeStatus::Running
            && self.current_player() == self.human_player
    }

    /// Make a player move
    pub fn player_move(&mut self, position: u32) -> Result<()> {
        self.make_move(position)
    }

    /// Make a bot move using MCTS if model is available, otherwise random
    #[cfg(feature = "onnx")]
    pub fn bot_move(&mut self) -> Result<u32> {
        let legal = self.legal_moves()?;
        if legal.is_empty() {
            return Err(anyhow!("No legal moves available"));
        }

        // Check if we have a model
        let has_model = {
            let guard = self
                .evaluator
                .read()
                .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
            guard.is_some()
        };

        let position = if has_model {
            // Try to use MCTS with neural network
            debug!("Attempting MCTS for bot move");

            let mcts_result = (|| -> Result<u32> {
                let guard = self
                    .evaluator
                    .read()
                    .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;
                let evaluator = guard.as_ref().unwrap();

                // Use pre-created simulation context (avoids repeated registry lookups)
                let sim_ctx = self
                    .mcts_sim_ctx
                    .as_mut()
                    .ok_or_else(|| anyhow!("Simulation context not available"))?;

                let result = run_mcts(
                    sim_ctx,
                    evaluator,
                    self.mcts_config.clone(),
                    self.state.clone(),
                    self.timestep.clone(),
                    &mut self.rng,
                )?;

                debug!(
                    action = result.action,
                    value = result.value,
                    simulations = result.simulations,
                    "MCTS selected move"
                );

                Ok(result.action)
            })();

            mcts_result.map_err(|error| {
                anyhow!(
                    "Loaded model failed during MCTS; refusing to hide the runtime error: {error}"
                )
            })?
        } else {
            // Fall back to random move
            debug!("No model loaded, using random move");
            use rand::seq::SliceRandom;
            *legal.choose(&mut self.rng).unwrap()
        };

        self.make_move(position)?;
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

        self.make_move(position)?;
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
        validate_timestep(&step.timestep, ExpectedTransition::Agent(actor))?;
        let view = require_board_view(&self.ctx, &step.state, &step.timestep, &self.board)?;

        // Commit the new session position only after the complete transition
        // and presentation pass the AlphaZero-board boundary checks.
        self.state = step.state;
        self.timestep = step.timestep;
        self.view = view;

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

    #[test]
    fn test_legal_moves_handles_short_obs() {
        engine_games::register_all_environments();

        let mut session = GameSession::new("tictactoe").unwrap();
        // Corrupt the encoded observation to simulate a mismatch with metadata.
        session.timestep.observations[0].data.truncate(4);

        assert!(session.legal_moves().is_err());
        assert!(session.is_legal_move(0).is_err());
    }

    #[test]
    fn active_position_rejects_chance_and_simultaneous_decisions() {
        engine_games::register_all_environments();
        let mut session = GameSession::new("tictactoe").unwrap();

        session.timestep.decision = Decision::Chance;
        assert!(session
            .legal_moves()
            .unwrap_err()
            .to_string()
            .contains("Chance"));

        session.timestep.decision = Decision::Agents {
            agent_ids: vec![AgentId(1), AgentId(2)],
        };
        assert!(session
            .legal_moves()
            .unwrap_err()
            .to_string()
            .contains("exactly one active decision agent"));
    }

    #[test]
    fn active_position_requires_one_matching_observation() {
        engine_games::register_all_environments();
        let mut session = GameSession::new("tictactoe").unwrap();

        session.timestep.observations[0].agent_id = AgentId(2);
        assert!(session
            .legal_moves()
            .unwrap_err()
            .to_string()
            .contains("sole observation belongs to agent 2"));

        session.timestep.observations.clear();
        assert!(session
            .legal_moves()
            .unwrap_err()
            .to_string()
            .contains("exactly one observation"));
    }

    #[test]
    fn timestep_validation_requires_per_agent_outcomes_matching_episode_status() {
        engine_games::register_all_environments();
        let session = GameSession::new("tictactoe").unwrap();
        let mut timestep = session.timestep.clone();

        timestep.outcomes.pop();
        assert!(validate_timestep(&timestep, ExpectedTransition::Reset)
            .unwrap_err()
            .to_string()
            .contains("exactly two per-agent outcomes"));

        let mut timestep = session.timestep.clone();
        timestep.outcomes[0].terminated = true;
        assert!(validate_timestep(&timestep, ExpectedTransition::Reset)
            .unwrap_err()
            .to_string()
            .contains("disagree with episode status"));
    }

    #[test]
    fn board_serving_rejects_custom_or_missing_presentations() {
        let custom = Presentation::Custom {
            contract: "counter_text_v1".to_string(),
            payload: Vec::new(),
        };
        assert!(require_board_presentation(Some(custom))
            .unwrap_err()
            .to_string()
            .contains("custom contract 'counter_text_v1'"));
        assert!(require_board_presentation(None)
            .unwrap_err()
            .to_string()
            .contains("requires the environment to expose a board presentation"));
    }
}
