//! TicTacToe game implementation for the Cartridge engine
//!
//! This crate provides a complete reference implementation of TicTacToe
//! demonstrating the narrow BoardGame adapter contract.
//!
//! # Usage
//!
//! ```rust
//! use games_tictactoe::{TicTacToe, register_tictactoe};
//! use engine_core::EngineContext;
//!
//! // Register the game with the global registry
//! register_tictactoe();
//!
//! // Create a context to play
//! let mut ctx = EngineContext::new("tictactoe").expect("tictactoe should be registered");
//! let reset = ctx.reset(42, &[]).unwrap();
//! ```

use engine_core::board_profile::{
    calculate_reward, decode_action_u32, opponent, register_board_game, validate_board_cells,
    validate_player_and_winner, BoardGame, BoardGameMetadata, BoardPlayerMetadata, BoardTransition,
    BoardView, TwoPlayerObs, TwoPlayerObsError,
};
use engine_core::typed::{
    ActionSpace, AgentId, AgentModel, Capabilities, DecodeError, EncodeError, Encoding, EngineId,
    EnvironmentSemantics, TensorSpec,
};
use engine_core::{EnvironmentError, EnvironmentMetadata, LegalMask};
use rand_chacha::ChaCha20Rng;

/// Immutable environment contract revision for wire formats and semantics.
pub const ENV_CONTRACT_VERSION: u32 = 2;

/// Register TicTacToe with the global game registry
///
/// Call this function once at startup to make TicTacToe available
/// via `EngineContext::new("tictactoe")`.
pub fn register_tictactoe() {
    register_board_game::<TicTacToe>().expect("tictactoe environment must only be registered once");
}

/// TicTacToe game state
///
/// Represents the complete state of a TicTacToe game including the board,
/// current player, and winner information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State {
    /// Board representation: 0=empty, 1=X, 2=O
    board: [u8; 9],
    /// Current player: 1=X, 2=O
    current_player: u8,
    /// Winner: 0=none/ongoing, 1=X, 2=O, 3=draw
    winner: u8,
}

impl State {
    /// Create a new initial game state
    pub fn new() -> Self {
        Self {
            board: [0; 9],
            current_player: 1, // X goes first
            winner: 0,
        }
    }

    /// Check if the game is over
    pub fn is_done(&self) -> bool {
        self.winner != 0
    }

    /// Get legal moves (empty positions)
    pub fn legal_moves(&self) -> Vec<u8> {
        if self.is_done() {
            return Vec::new();
        }

        (0..9u8)
            .filter(|&pos| self.board[pos as usize] == 0)
            .collect()
    }

    /// Bit-mask representation of legal moves.
    ///
    /// Bits 0-8 correspond to board positions 0-8. A bit set to 1 indicates the
    /// position is currently legal. When the game is finished the mask is zeroed.
    pub fn legal_moves_mask(&self) -> u16 {
        if self.is_done() {
            return 0;
        }

        self.board
            .iter()
            .enumerate()
            .fold(0u16, |mask, (idx, cell)| {
                if *cell == 0 {
                    mask | (1u16 << idx)
                } else {
                    mask
                }
            })
    }

    /// Make a move and return the new state
    pub fn make_move(&self, position: u8) -> State {
        if self.is_done() || position >= 9 || self.board[position as usize] != 0 {
            return *self; // Invalid move, return unchanged state
        }

        let mut new_state = *self;
        new_state.board[position as usize] = self.current_player;

        // Check for winner
        new_state.winner = Self::check_winner(&new_state.board);

        // Switch player if game not over
        if new_state.winner == 0 {
            new_state.current_player = opponent(self.current_player);
        }

        new_state
    }

    /// Check for winner on the board
    fn check_winner(board: &[u8; 9]) -> u8 {
        // Winning positions (rows, columns, diagonals)
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8], // rows
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8], // columns
            [0, 4, 8],
            [2, 4, 6], // diagonals
        ];

        for line in &LINES {
            let [a, b, c] = *line;
            if board[a] != 0 && board[a] == board[b] && board[b] == board[c] {
                return board[a]; // Return the winning player
            }
        }

        // Check for draw (board full but no winner)
        if board.iter().all(|&cell| cell != 0) {
            return 3; // Draw
        }

        0 // Game ongoing
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

/// TicTacToe action - position to place a piece (0-8)
pub type Action = u8;

/// Player-relative TicTacToe observation: two 3x3 occupancy planes.
pub type Observation = TwoPlayerObs<18>;

/// Create observation from game state
pub fn observation_from_state(state: &State) -> Result<Observation, TwoPlayerObsError> {
    TwoPlayerObs::from_board(&state.board, state.current_player)
}

/// TicTacToe game implementation
#[derive(Debug)]
pub struct TicTacToe;

impl TicTacToe {
    /// Create a new TicTacToe game
    pub fn new() -> Self {
        Self
    }
}

impl Default for TicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl BoardGame for TicTacToe {
    type State = State;
    type Action = Action;
    type Observation = Observation;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "tictactoe".to_string(),
            build_id: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: ENV_CONTRACT_VERSION,
            encoding: Encoding::discrete_u32_le(
                "tictactoe_state:v1",
                TensorSpec::f32_fixed([("channel", 2), ("row", 3), ("column", 3)]),
            ),
            semantics:
                EnvironmentSemantics::deterministic_alternating_perfect_information_terminal_zero_sum(),
            max_horizon: Some(9),
            agents: AgentModel::fixed_homogeneous_masked(
                [AgentId(1), AgentId(2)],
                ActionSpace::discrete(9),
            ),
            preferred_batch: 64,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_description("Get three in a row to win!")
            .with_board(BoardGameMetadata::new(3, 3).with_players(vec![
                BoardPlayerMetadata::new("X", "X"),
                BoardPlayerMetadata::new("O", "O"),
            ]))
    }

    fn describe_discrete_action(&self, action: u32) -> Option<engine_core::ActionPresentation> {
        (action < 9).then(|| engine_core::ActionPresentation::cell(action, 3))
    }

    // reset/step mirror games-connect4 and games-othello; shared reward and
    // Validation helpers live in the explicit engine_core::board_profile API.
    fn reset(
        &mut self,
        _rng: &mut ChaCha20Rng,
        _hint: &[u8],
    ) -> Result<(Self::State, Self::Observation), EnvironmentError> {
        let state = State::new();
        let obs = observation_from_state(&state)
            .map_err(|error| EnvironmentError::InvalidState(error.to_string()))?;
        Ok((state, obs))
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> Result<BoardTransition<Self::Observation>, EnvironmentError> {
        if state.is_done() || action >= 9 || state.board[action as usize] != 0 {
            return Err(EnvironmentError::InvalidAction(format!(
                "position {action} is not legal in the current Tic-Tac-Toe state"
            )));
        }
        let previous_player = state.current_player;
        *state = state.make_move(action);

        let obs = observation_from_state(state)
            .map_err(|error| EnvironmentError::InvalidState(error.to_string()))?;
        let reward = calculate_reward(state.winner, previous_player);
        let done = state.is_done();
        Ok(BoardTransition {
            observation: obs,
            actor_reward: reward,
            terminated: done,
        })
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        // Simple binary encoding: board (9 bytes) + current_player (1 byte) + winner (1 byte)
        out.extend_from_slice(&state.board);
        out.push(state.current_player);
        out.push(state.winner);
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        if buf.len() != 11 {
            return Err(DecodeError::InvalidLength {
                expected: 11,
                actual: buf.len(),
            });
        }

        let mut board = [0u8; 9];
        board.copy_from_slice(&buf[0..9]);

        let current_player = buf[9];
        let winner = buf[10];

        // Validate the state
        validate_player_and_winner(current_player, winner)?;
        validate_board_cells(&board)?;

        Ok(State {
            board,
            current_player,
            winner,
        })
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        if *action >= 9 {
            return Err(EncodeError::InvalidData(format!(
                "Invalid action position: {}",
                action
            )));
        }
        // Encode as u32 in little-endian format (4 bytes)
        out.extend_from_slice(&(*action as u32).to_le_bytes());
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        let position = decode_action_u32(buf)?;
        if position >= 9 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid action position: {}",
                position
            )));
        }

        Ok(position as u8)
    }

    fn encode_observation(obs: &Self::Observation, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        obs.encode(out);
        Ok(())
    }

    fn legal_actions(state: &Self::State) -> Result<LegalMask, EnvironmentError> {
        Ok(LegalMask::from_u64(state.legal_moves_mask() as u64, 9))
    }

    fn view(state: &Self::State) -> BoardView {
        BoardView::from_owners(&state.board, state.current_player, state.winner)
    }
}

#[cfg(test)]
mod tests;
