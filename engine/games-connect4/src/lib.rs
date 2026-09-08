//! Connect 4 game implementation for the Cartridge engine
//!
//! Connect 4 is a two-player connection game where players drop colored discs
//! into a 7-column, 6-row vertically suspended grid. The objective is to be
//! the first to form a horizontal, vertical, or diagonal line of four discs.
//!
//! # Board Layout
//!
//! The board is stored in row-major order, with row 0 at the bottom:
//! ```text
//! Row 5: [35][36][37][38][39][40][41]  <- Top
//! Row 4: [28][29][30][31][32][33][34]
//! Row 3: [21][22][23][24][25][26][27]
//! Row 2: [14][15][16][17][18][19][20]
//! Row 1: [ 7][ 8][ 9][10][11][12][13]
//! Row 0: [ 0][ 1][ 2][ 3][ 4][ 5][ 6]  <- Bottom
//!    Col   0   1   2   3   4   5   6
//! ```
//!
//! # Usage
//!
//! ```rust
//! use games_connect4::{Connect4, register_connect4};
//! use engine_core::EngineContext;
//!
//! // Register the game with the global registry
//! register_connect4();
//!
//! // Create a context to play
//! let mut ctx = EngineContext::new("connect4").expect("connect4 should be registered");
//! let reset = ctx.reset(42, &[]).unwrap();
//! ```

use engine_core::board_profile::{
    calculate_reward, decode_action_u32, opponent, register_board_game, validate_board_cells,
    validate_player_and_winner, BoardGame, BoardGameMetadata, BoardPlayerMetadata, BoardRenderer,
    BoardTransition, BoardView, TwoPlayerObs, TwoPlayerObsError,
};
use engine_core::typed::{
    ActionSpace, AgentId, AgentModel, Capabilities, DecodeError, EncodeError, Encoding, EngineId,
    EnvironmentSemantics, TensorSpec,
};
use engine_core::{EnvironmentError, EnvironmentMetadata, LegalMask};
use rand_chacha::ChaCha20Rng;

/// Board dimensions
pub const COLS: usize = 7;
pub const ROWS: usize = 6;
pub const BOARD_SIZE: usize = COLS * ROWS; // 42

/// Immutable environment contract revision for wire formats and semantics.
pub const ENV_CONTRACT_VERSION: u32 = 2;

/// Register Connect4 with the global game registry
///
/// Call this function once at startup to make Connect4 available
/// via `EngineContext::new("connect4")`.
pub fn register_connect4() {
    register_board_game::<Connect4>().expect("connect4 environment must only be registered once");
}

/// Connect4 game state
///
/// Represents the complete state of a Connect4 game including the board,
/// current player, and winner information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    /// Board representation: 0=empty, 1=Red (player 1), 2=Yellow (player 2)
    /// Stored in row-major order with row 0 at the bottom
    board: [u8; BOARD_SIZE],
    /// Current player: 1=Red, 2=Yellow
    current_player: u8,
    /// Winner: 0=none/ongoing, 1=Red, 2=Yellow, 3=draw
    winner: u8,
    /// Height of each column (0-6 means number of pieces in column)
    column_heights: [u8; COLS],
}

impl State {
    /// Create a new initial game state
    pub fn new() -> Self {
        Self {
            board: [0; BOARD_SIZE],
            current_player: 1, // Red goes first
            winner: 0,
            column_heights: [0; COLS],
        }
    }

    /// Check if the game is over
    pub fn is_done(&self) -> bool {
        self.winner != 0
    }

    /// Get legal moves (columns that are not full)
    pub fn legal_moves(&self) -> Vec<u8> {
        if self.is_done() {
            return Vec::new();
        }

        (0..COLS as u8)
            .filter(|&col| self.column_heights[col as usize] < ROWS as u8)
            .collect()
    }

    /// Bit-mask representation of legal moves.
    ///
    /// Bits 0-6 correspond to columns 0-6. A bit set to 1 indicates the
    /// column is not full and a piece can be dropped there.
    pub fn legal_moves_mask(&self) -> u8 {
        if self.is_done() {
            return 0;
        }

        self.column_heights
            .iter()
            .enumerate()
            .fold(0u8, |mask, (col, &height)| {
                if height < ROWS as u8 {
                    mask | (1u8 << col)
                } else {
                    mask
                }
            })
    }

    /// Convert column and row to board index
    #[inline]
    fn pos(col: usize, row: usize) -> usize {
        row * COLS + col
    }

    /// Drop a piece in the given column and return the new state
    pub fn drop_piece(&self, column: u8) -> State {
        let col = column as usize;

        // Check if move is valid
        if self.is_done() || col >= COLS || self.column_heights[col] >= ROWS as u8 {
            return self.clone(); // Invalid move, return unchanged state
        }

        let mut new_state = self.clone();
        let row = self.column_heights[col] as usize;
        let pos = Self::pos(col, row);

        // Place the piece
        new_state.board[pos] = self.current_player;
        new_state.column_heights[col] += 1;

        // Check for winner
        new_state.winner = new_state.check_winner_at(col, row);

        // Switch player if game not over
        if new_state.winner == 0 {
            new_state.current_player = opponent(self.current_player);
        }

        new_state
    }

    /// Check if the piece at (col, row) creates a winning line
    fn check_winner_at(&self, col: usize, row: usize) -> u8 {
        let player = self.board[Self::pos(col, row)];
        if player == 0 {
            return 0;
        }

        // Direction vectors: horizontal, vertical, diagonal /, diagonal \
        let directions: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];

        for (dc, dr) in directions {
            let mut count = 1; // Count the piece we just placed

            // Count in positive direction
            let (mut c, mut r) = (col as i32 + dc, row as i32 + dr);
            while c >= 0 && c < COLS as i32 && r >= 0 && r < ROWS as i32 {
                if self.board[Self::pos(c as usize, r as usize)] == player {
                    count += 1;
                    c += dc;
                    r += dr;
                } else {
                    break;
                }
            }

            // Count in negative direction
            let (mut c, mut r) = (col as i32 - dc, row as i32 - dr);
            while c >= 0 && c < COLS as i32 && r >= 0 && r < ROWS as i32 {
                if self.board[Self::pos(c as usize, r as usize)] == player {
                    count += 1;
                    c -= dc;
                    r -= dr;
                } else {
                    break;
                }
            }

            if count >= 4 {
                return player;
            }
        }

        // Check for draw (board full but no winner)
        if self.column_heights.iter().all(|&h| h >= ROWS as u8) {
            return 3; // Draw
        }

        0 // Game ongoing
    }

    /// Get the row where the last piece was placed in a column
    pub fn last_row_in_column(&self, col: usize) -> Option<usize> {
        if self.column_heights[col] == 0 {
            None
        } else {
            Some((self.column_heights[col] - 1) as usize)
        }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

/// Connect4 action - column to drop a piece in (0-6)
pub type Action = u8;

/// Player-relative Connect4 observation: two 6x7 occupancy planes.
pub type Observation = TwoPlayerObs<84>;

/// Create observation from game state
pub fn observation_from_state(state: &State) -> Result<Observation, TwoPlayerObsError> {
    TwoPlayerObs::from_board(&state.board, state.current_player)
}

/// Connect4 game implementation
#[derive(Debug)]
pub struct Connect4;

impl Connect4 {
    /// Create a new Connect4 game
    pub fn new() -> Self {
        Self
    }
}

impl Default for Connect4 {
    fn default() -> Self {
        Self::new()
    }
}

/// Observation size: two player-relative board planes.
impl BoardGame for Connect4 {
    type State = State;
    type Action = Action;
    type Observation = Observation;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "connect4".to_string(),
            build_id: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: ENV_CONTRACT_VERSION,
            encoding: Encoding::discrete_u32_le(
                "connect4_state:v1",
                TensorSpec::f32_fixed([
                    ("channel", 2),
                    ("row", ROWS as u32),
                    ("column", COLS as u32),
                ]),
            ),
            semantics:
                EnvironmentSemantics::deterministic_alternating_perfect_information_terminal_zero_sum(),
            max_horizon: Some(BOARD_SIZE as u32),
            agents: AgentModel::fixed_homogeneous_masked(
                [AgentId(1), AgentId(2)],
                ActionSpace::discrete(COLS as u32),
            ),
            preferred_batch: 64,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("connect4", "Connect 4")
            .with_description("Drop discs to connect four in a row!")
            .with_board(
                BoardGameMetadata::new(COLS, ROWS)
                    .with_players(vec![
                        BoardPlayerMetadata::new("Red", "🔴"),
                        BoardPlayerMetadata::new("Yellow", "🟡"),
                    ])
                    .with_renderer(BoardRenderer::DropColumn),
            )
    }

    fn describe_discrete_action(&self, action: u32) -> Option<engine_core::ActionPresentation> {
        (action < COLS as u32).then(|| {
            engine_core::ActionPresentation::new(
                action,
                format!("Column {}", action + 1),
                engine_core::ActionTarget::Column {
                    index: action as usize,
                },
            )
        })
    }

    // reset/step mirror games-tictactoe and games-othello; shared reward and
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
        let column = action as usize;
        if state.is_done() || column >= COLS || state.column_heights[column] >= ROWS as u8 {
            return Err(EnvironmentError::InvalidAction(format!(
                "column {action} is not legal in the current Connect 4 state"
            )));
        }
        let previous_player = state.current_player;
        *state = state.drop_piece(action);

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
        // Binary encoding: board (42 bytes) + current_player (1 byte) + winner (1 byte)
        // Note: column_heights can be reconstructed from board
        out.extend_from_slice(&state.board);
        out.push(state.current_player);
        out.push(state.winner);
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        let expected_len = BOARD_SIZE + 2; // board + current_player + winner
        if buf.len() != expected_len {
            return Err(DecodeError::InvalidLength {
                expected: expected_len,
                actual: buf.len(),
            });
        }

        let mut board = [0u8; BOARD_SIZE];
        board.copy_from_slice(&buf[0..BOARD_SIZE]);

        let current_player = buf[BOARD_SIZE];
        let winner = buf[BOARD_SIZE + 1];

        // Validate the state
        validate_player_and_winner(current_player, winner)?;
        validate_board_cells(&board)?;

        // Reconstruct column heights from board
        let mut column_heights = [0u8; COLS];
        for col in 0..COLS {
            for row in 0..ROWS {
                if board[State::pos(col, row)] != 0 {
                    column_heights[col] = (row + 1) as u8;
                }
            }
        }

        Ok(State {
            board,
            current_player,
            winner,
            column_heights,
        })
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        if *action as usize >= COLS {
            return Err(EncodeError::InvalidData(format!(
                "Invalid action column: {}",
                action
            )));
        }
        // Encode as u32 in little-endian format (4 bytes)
        out.extend_from_slice(&(*action as u32).to_le_bytes());
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        let column = decode_action_u32(buf)?;
        if column as usize >= COLS {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid action column: {}",
                column
            )));
        }

        Ok(column as u8)
    }

    fn encode_observation(obs: &Self::Observation, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        obs.encode(out);
        Ok(())
    }

    fn legal_actions(state: &Self::State) -> Result<LegalMask, EnvironmentError> {
        Ok(LegalMask::from_u64(state.legal_moves_mask() as u64, COLS))
    }

    fn view(state: &Self::State) -> BoardView {
        BoardView::from_owners(&state.board, state.current_player, state.winner)
    }
}

#[cfg(test)]
mod tests;
