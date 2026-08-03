//! Narrow adapter contract and shared types for two-player board games.
//!
//! This module provides reusable observation types that eliminate code
//! duplication between similar board games like TicTacToe and Connect4.

use crate::board_game_utils::encode_f32_slices;
use crate::board_view::{BoardView, Presentation};
use crate::legal_mask::LegalMask;
use crate::metadata::EnvironmentMetadata;
use crate::typed::{
    AgentId, AgentObservation, AgentOutcome, Capabilities, Decision, DecodeError, EncodeError,
    EngineId, Environment, EnvironmentError, EpisodeStatus, Timestep, TransitionSource,
};
use rand_chacha::ChaCha20Rng;

/// Result of one transition in the explicitly narrow board-game contract.
///
/// The scalar is named for its perspective here; the generic environment ABI
/// receives a reward for every agent after adaptation.
#[derive(Debug, Clone, PartialEq)]
pub struct BoardTransition<O> {
    pub observation: O,
    pub actor_reward: f32,
    pub terminated: bool,
}

/// Convenience trait for the bundled deterministic two-seat board games.
///
/// This is deliberately *not* the engine's generic environment contract. The
/// [`BoardGameEnvironment`] adapter makes its assumptions explicit and maps it
/// into per-agent [`Timestep`] values.
pub trait BoardGame: Send + Sync + std::fmt::Debug + 'static {
    type State: Send + Sync + 'static;
    type Action: Send + Sync + 'static;
    type Observation: Send + Sync + 'static;

    fn engine_id(&self) -> EngineId;
    fn capabilities(&self) -> Capabilities;
    fn metadata(&self) -> EnvironmentMetadata;
    /// Start an episode, rejecting unsupported hints or invalid generated state.
    fn reset(
        &mut self,
        rng: &mut ChaCha20Rng,
        hint: &[u8],
    ) -> Result<(Self::State, Self::Observation), EnvironmentError>;
    /// Apply one legal action.
    ///
    /// Actions outside the current observation's legal mask must return
    /// [`EnvironmentError::InvalidAction`] without mutating `state`.
    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        rng: &mut ChaCha20Rng,
    ) -> Result<BoardTransition<Self::Observation>, EnvironmentError>;
    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError>;
    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError>;
    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError>;
    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError>;
    fn encode_observation(
        observation: &Self::Observation,
        out: &mut Vec<u8>,
    ) -> Result<(), EncodeError>;
    fn view(state: &Self::State) -> BoardView;
}

/// Adapts a narrow board game into the generic typed [`Environment`] ABI.
#[derive(Debug)]
pub struct BoardGameEnvironment<G: BoardGame> {
    game: G,
}

impl<G: BoardGame> BoardGameEnvironment<G> {
    pub fn new(game: G) -> Self {
        Self { game }
    }

    fn outcome(agent_id: u32, reward: f32, terminated: bool) -> AgentOutcome {
        AgentOutcome {
            agent_id: AgentId(agent_id),
            reward,
            terminated,
            truncated: false,
        }
    }

    fn validate_profile(&self) -> Result<(), EnvironmentError> {
        let metadata = self.game.metadata();
        let board = metadata.board.ok_or_else(|| {
            EnvironmentError::Transition(
                "BoardGame implementation did not publish board metadata".to_string(),
            )
        })?;
        if board.players.len() != 2 {
            return Err(EnvironmentError::Transition(format!(
                "BoardGame requires exactly two seats, got {}",
                board.players.len()
            )));
        }
        Ok(())
    }
}

impl<G: BoardGame> Environment for BoardGameEnvironment<G> {
    type State = G::State;
    type Action = G::Action;
    type Observation = G::Observation;

    fn engine_id(&self) -> EngineId {
        self.game.engine_id()
    }

    fn capabilities(&self) -> Capabilities {
        self.game.capabilities()
    }

    fn metadata(&self) -> EnvironmentMetadata {
        self.game.metadata()
    }

    fn reset(
        &mut self,
        rng: &mut ChaCha20Rng,
        hint: &[u8],
    ) -> Result<(Self::State, Timestep<Self::Observation>), EnvironmentError> {
        self.validate_profile()?;
        let (state, observation) = self.game.reset(rng, hint)?;
        let view = G::view(&state);
        let terminated = view.game_over();
        let current = AgentId::from(view.current_player);
        Ok((
            state,
            Timestep {
                agents: vec![AgentId(1), AgentId(2)],
                observations: vec![AgentObservation {
                    agent_id: current,
                    observation,
                }],
                outcomes: vec![
                    Self::outcome(1, 0.0, terminated),
                    Self::outcome(2, 0.0, terminated),
                ],
                decision: if terminated {
                    Decision::None
                } else {
                    Decision::Agents {
                        agent_ids: vec![current],
                    }
                },
                episode: if terminated {
                    EpisodeStatus::Terminated
                } else {
                    EpisodeStatus::Running
                },
                source: TransitionSource::Reset,
                info: Vec::new(),
            },
        ))
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        rng: &mut ChaCha20Rng,
    ) -> Result<Timestep<Self::Observation>, EnvironmentError> {
        let actor = AgentId::from(G::view(state).current_player);
        let transition = self.game.step(state, action, rng)?;
        let view = G::view(state);
        let observer = AgentId::from(view.current_player);
        let other = if actor == AgentId(1) {
            AgentId(2)
        } else {
            AgentId(1)
        };
        Ok(Timestep {
            agents: vec![AgentId(1), AgentId(2)],
            observations: vec![AgentObservation {
                agent_id: observer,
                observation: transition.observation,
            }],
            outcomes: vec![
                AgentOutcome {
                    agent_id: actor,
                    reward: transition.actor_reward,
                    terminated: transition.terminated,
                    truncated: false,
                },
                AgentOutcome {
                    agent_id: other,
                    reward: -transition.actor_reward,
                    terminated: transition.terminated,
                    truncated: false,
                },
            ],
            decision: if transition.terminated {
                Decision::None
            } else {
                Decision::Agents {
                    agent_ids: vec![observer],
                }
            },
            episode: if transition.terminated {
                EpisodeStatus::Terminated
            } else {
                EpisodeStatus::Running
            },
            source: TransitionSource::Agents {
                agent_ids: vec![actor],
            },
            info: Vec::new(),
        })
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        G::encode_state(state, out)
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        G::decode_state(buf)
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        G::encode_action(action, out)
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        G::decode_action(buf)
    }

    fn encode_observation(
        observation: &Self::Observation,
        out: &mut Vec<u8>,
    ) -> Result<(), EncodeError> {
        G::encode_observation(observation, out)
    }

    fn presentation(state: &Self::State) -> Option<Presentation> {
        Some(Presentation::Board {
            view: G::view(state),
        })
    }
}

/// Neural network observation for two-player board games.
///
/// Generic over board view size and number of actions to support different board sizes.
/// - `BOARD_VIEW_SIZE`: Total size of one-hot board encoding (board_size * 2 for two players)
/// - `NUM_ACTIONS`: Number of possible actions (board positions or columns)
#[derive(Debug, Clone, PartialEq)]
pub struct TwoPlayerObs<const BOARD_VIEW_SIZE: usize, const NUM_ACTIONS: usize> {
    /// One-hot encoding of board: [player1_positions, player2_positions]
    pub board_view: [f32; BOARD_VIEW_SIZE],
    /// Legal moves mask (1.0 = legal, 0.0 = illegal)
    pub legal_moves: [f32; NUM_ACTIONS],
    /// Current player indicator: [is_player1, is_player2]
    pub current_player: [f32; 2],
}

impl<const BOARD_VIEW_SIZE: usize, const NUM_ACTIONS: usize>
    TwoPlayerObs<BOARD_VIEW_SIZE, NUM_ACTIONS>
{
    /// Create an observation from a board and a validated dynamic-width legal mask.
    ///
    /// - `board`: Slice of cell values (0=empty, 1=player1, 2=player2)
    /// - `legal_mask`: One bit per action; its width must equal `NUM_ACTIONS`
    /// - `current_player`: Current player (1 or 2)
    pub fn from_board(
        board: &[u8],
        legal_mask: &LegalMask,
        current_player: u8,
    ) -> Result<Self, TwoPlayerObsError> {
        if !BOARD_VIEW_SIZE.is_multiple_of(2) {
            return Err(TwoPlayerObsError::InvalidBoardViewSize {
                board_view_size: BOARD_VIEW_SIZE,
            });
        }
        let board_size = BOARD_VIEW_SIZE / 2;
        if board.len() != board_size {
            return Err(TwoPlayerObsError::InvalidBoardLength {
                expected: board_size,
                actual: board.len(),
            });
        }
        if legal_mask.num_actions() != NUM_ACTIONS {
            return Err(TwoPlayerObsError::InvalidLegalMaskWidth {
                expected: NUM_ACTIONS,
                actual: legal_mask.num_actions(),
            });
        }
        if !(1..=2).contains(&current_player) {
            return Err(TwoPlayerObsError::InvalidCurrentPlayer { current_player });
        }

        let mut obs = Self {
            board_view: [0.0; BOARD_VIEW_SIZE],
            legal_moves: [0.0; NUM_ACTIONS],
            current_player: [0.0; 2],
        };

        // Encode board state (one-hot for each player)
        for (i, &cell) in board.iter().enumerate() {
            match cell {
                0 => {}
                1 => obs.board_view[i] = 1.0,
                2 => obs.board_view[i + board_size] = 1.0,
                value => {
                    return Err(TwoPlayerObsError::InvalidBoardCell { index: i, value });
                }
            }
        }

        // Encode legal moves
        for (pos, slot) in obs.legal_moves.iter_mut().enumerate() {
            if legal_mask.is_legal(pos) {
                *slot = 1.0;
            }
        }

        // Encode current player
        if current_player == 1 {
            obs.current_player[0] = 1.0;
        } else {
            obs.current_player[1] = 1.0;
        }

        Ok(obs)
    }

    /// Create an observation from an iterator of legal action indices.
    ///
    /// Every index is bounds-checked before the mask is mutated, so malformed
    /// environment state is returned as an error rather than panicking.
    pub fn from_board_with_legal_actions(
        board: &[u8],
        legal_actions: impl IntoIterator<Item = usize>,
        current_player: u8,
    ) -> Result<Self, TwoPlayerObsError> {
        let mut legal_mask = LegalMask::new(NUM_ACTIONS);
        for action in legal_actions {
            if action >= NUM_ACTIONS {
                return Err(TwoPlayerObsError::InvalidLegalAction {
                    action,
                    num_actions: NUM_ACTIONS,
                });
            }
            legal_mask.set(action);
        }
        Self::from_board(board, &legal_mask, current_player)
    }

    /// Encode observation as bytes for neural network input.
    pub fn encode(&self, out: &mut Vec<u8>) {
        encode_f32_slices(
            out,
            [
                &self.board_view[..],
                &self.legal_moves[..],
                &self.current_player[..],
            ],
        );
    }

    /// Total observation size in floats.
    pub const fn obs_size() -> usize {
        BOARD_VIEW_SIZE + NUM_ACTIONS + 2
    }
}

/// Invalid data supplied while building a [`TwoPlayerObs`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TwoPlayerObsError {
    #[error("board view size {board_view_size} must be even (one plane per player)")]
    InvalidBoardViewSize { board_view_size: usize },
    #[error("board length must be exactly {expected}, got {actual}")]
    InvalidBoardLength { expected: usize, actual: usize },
    #[error("board cell {index} must be 0, 1, or 2, got {value}")]
    InvalidBoardCell { index: usize, value: u8 },
    #[error("legal mask width must be exactly {expected}, got {actual}")]
    InvalidLegalMaskWidth { expected: usize, actual: usize },
    #[error("legal action {action} is outside action space of size {num_actions}")]
    InvalidLegalAction { action: usize, num_actions: usize },
    #[error("current player must be 1 or 2, got {current_player}")]
    InvalidCurrentPlayer { current_player: u8 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tictactoe_obs() {
        // TicTacToe: 9 positions * 2 players = 18, 9 actions
        type TicTacToeObs = TwoPlayerObs<18, 9>;

        let board = [1, 0, 2, 0, 1, 0, 0, 0, 0u8];
        let legal_mask = LegalMask::from_u64(0b111101010, 9); // positions 1, 3, 5, 6, 7, 8
        let obs = TicTacToeObs::from_board(&board, &legal_mask, 2).unwrap();

        // Player 1 at positions 0, 4
        assert_eq!(obs.board_view[0], 1.0);
        assert_eq!(obs.board_view[4], 1.0);
        // Player 2 at position 2
        assert_eq!(obs.board_view[9 + 2], 1.0);
        // Current player is 2
        assert_eq!(obs.current_player, [0.0, 1.0]);
        // Check obs size
        assert_eq!(TicTacToeObs::obs_size(), 29);
    }

    #[test]
    fn test_connect4_obs() {
        // Connect4: 42 positions * 2 players = 84, 7 actions
        type Connect4Obs = TwoPlayerObs<84, 7>;

        let mut board = [0u8; 42];
        board[3] = 1; // Red at column 3, row 0
        let legal_mask = LegalMask::all_legal(7);
        let obs = Connect4Obs::from_board(&board, &legal_mask, 1).unwrap();

        assert_eq!(obs.board_view[3], 1.0);
        assert_eq!(obs.current_player, [1.0, 0.0]);
        assert_eq!(obs.legal_moves, [1.0; 7]);
        assert_eq!(Connect4Obs::obs_size(), 93);
    }

    #[test]
    fn observation_supports_action_spaces_wider_than_u64() {
        type WideObs = TwoPlayerObs<4, 130>;
        let obs = WideObs::from_board_with_legal_actions(&[1, 2], [0, 64, 129], 1).unwrap();

        assert_eq!(obs.legal_moves[0], 1.0);
        assert_eq!(obs.legal_moves[64], 1.0);
        assert_eq!(obs.legal_moves[129], 1.0);
        assert_eq!(
            obs.legal_moves
                .iter()
                .filter(|&&value| value == 1.0)
                .count(),
            3
        );
    }

    #[test]
    fn observation_rejects_malformed_inputs_without_panicking() {
        type Obs = TwoPlayerObs<4, 3>;
        assert_eq!(
            Obs::from_board(&[0], &LegalMask::new(3), 1),
            Err(TwoPlayerObsError::InvalidBoardLength {
                expected: 2,
                actual: 1,
            })
        );
        assert_eq!(
            Obs::from_board(&[0, 9], &LegalMask::new(3), 1),
            Err(TwoPlayerObsError::InvalidBoardCell { index: 1, value: 9 })
        );
        assert_eq!(
            Obs::from_board(&[0, 0], &LegalMask::new(2), 1),
            Err(TwoPlayerObsError::InvalidLegalMaskWidth {
                expected: 3,
                actual: 2,
            })
        );
        assert_eq!(
            Obs::from_board(&[0, 0], &LegalMask::new(3), 3),
            Err(TwoPlayerObsError::InvalidCurrentPlayer { current_player: 3 })
        );
        assert_eq!(
            Obs::from_board_with_legal_actions(&[0, 0], [3], 1),
            Err(TwoPlayerObsError::InvalidLegalAction {
                action: 3,
                num_actions: 3,
            })
        );
    }

    #[test]
    fn observation_rejects_odd_board_view_size() {
        type Obs = TwoPlayerObs<3, 1>;
        assert_eq!(
            Obs::from_board(&[0], &LegalMask::new(1), 1),
            Err(TwoPlayerObsError::InvalidBoardViewSize { board_view_size: 3 })
        );
    }
}
