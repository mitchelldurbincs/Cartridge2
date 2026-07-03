//! TicTacToe game implementation for the Cartridge engine
//!
//! This crate provides a complete reference implementation of TicTacToe
//! demonstrating how to implement the Game trait for the engine framework.
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

use engine_core::game_utils::{
    calculate_reward, decode_action_u32, info_bits, opponent, validate_board_cells,
    validate_player_and_winner,
};
use engine_core::typed::{
    ActionSpace, Capabilities, DecodeError, EncodeError, Encoding, EngineId, Game,
};
use engine_core::{register_game, GameAdapter, GameMetadata, TwoPlayerObs};
use rand_chacha::ChaCha20Rng;

/// Register TicTacToe with the global game registry
///
/// Call this function once at startup to make TicTacToe available
/// via `EngineContext::new("tictactoe")`.
pub fn register_tictactoe() {
    register_game("tictactoe".to_string(), || {
        Box::new(GameAdapter::new(TicTacToe::new()))
    });
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

/// TicTacToe observation (18 board view + 9 legal moves + 2 current player = 29 floats)
pub type Observation = TwoPlayerObs<18, 9>;

/// Create observation from game state
pub fn observation_from_state(state: &State) -> Observation {
    TwoPlayerObs::from_board(
        &state.board,
        state.legal_moves_mask() as u64,
        state.current_player,
    )
}

/// TicTacToe game implementation
#[derive(Debug)]
pub struct TicTacToe;

impl TicTacToe {
    /// Create a new TicTacToe game
    pub fn new() -> Self {
        Self
    }

    /// Pack auxiliary information about the state into a u64 bit-field.
    ///
    /// Uses the standard layout from `engine_core::game_utils::info_bits`:
    /// * Bits 0-8  : Legal move mask
    /// * Bits 16-19: Current player (1 = X, 2 = O)
    /// * Bits 20-23: Winner (0 = none, 1 = X, 2 = O, 3 = draw)
    /// * Bits 24-31: Moves played so far (0-9)
    fn compute_info_bits(state: &State) -> u64 {
        let moves_played = state.board.iter().filter(|&&cell| cell != 0).count() as u64;
        info_bits::compute_info_bits(
            state.legal_moves_mask() as u64,
            state.current_player,
            state.winner,
            moves_played,
        )
    }
}

impl Default for TicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl Game for TicTacToe {
    type State = State;
    type Action = Action;
    type Obs = Observation;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "tictactoe".to_string(),
            build_id: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            encoding: Encoding {
                state: "tictactoe_state:v1".to_string(),
                action: "discrete_position:v1".to_string(),
                obs: "f32x29:v1".to_string(), // 18 + 9 + 2 = 29 floats
                schema_version: 1,
            },
            max_horizon: 9,                         // Maximum 9 moves in TicTacToe
            action_space: ActionSpace::Discrete(9), // 9 possible positions
            preferred_batch: 64,
        }
    }

    fn metadata(&self) -> GameMetadata {
        GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_board(3, 3)
            .with_actions(9)
            .with_observation(29, 18) // 29 floats, legal mask starts at index 18
            .with_players(2, vec!["X".to_string(), "O".to_string()], vec!['X', 'O'])
            .with_description("Get three in a row to win!")
    }

    // reset/step mirror games-connect4 and games-othello; the shared pieces
    // (reward, info bits, validation) live in engine_core::game_utils.
    fn reset(&mut self, _rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
        let state = State::new();
        let obs = observation_from_state(&state);
        (state, obs)
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> (Self::Obs, f32, bool, u64) {
        let previous_player = state.current_player;
        *state = state.make_move(action);

        let obs = observation_from_state(state);
        let reward = calculate_reward(state.winner, previous_player);
        let done = state.is_done();
        let info = Self::compute_info_bits(state);

        (obs, reward, done, info)
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

    fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        obs.encode(out);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
