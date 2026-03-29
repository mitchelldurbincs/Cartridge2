//! Othello game implementation for the Cartridge engine
//!
//! Othello (also known as Reversi) is a two-player strategy board game played on
//! an 8×8 uncheckered board. Players take turns placing pieces on the board, and
//! each piece placed must flip at least one opponent piece. The game ends when
//! neither player can make a valid move. The player with the most pieces wins.
//!
//! # Board Layout
//!
//! The board is stored in row-major order, with row 0 at the top:
//! ```text
//! Row 0: [ 0][ 1][ 2][ 3][ 4][ 5][ 6][ 7]  <- Top
//! Row 1: [ 8][ 9][10][11][12][13][14][15]
//! Row 2: [16][17][18][19][20][21][22][23]
//! Row 3: [24][25][26][27][28][29][30][31]
//! Row 4: [32][33][34][35][36][37][38][39]
//! Row 5: [40][41][42][43][44][45][46][47]
//! Row 6: [48][49][50][51][52][53][54][55]
//! Row 7: [56][57][58][59][60][61][62][63]  <- Bottom
//!    Col   0   1   2   3   4   5   6   7
//! ```
//!
//! # Usage
//!
//! ```rust
//! use games_othello::{Othello, register_othello};
//! use engine_core::EngineContext;
//!
//! // Register the game with the global registry
//! register_othello();
//!
//! // Create a context to play
//! let mut ctx = EngineContext::new("othello").expect("othello should be registered");
//! let reset = ctx.reset(42, &[]).unwrap();
//! ```

use engine_core::game_utils::{calculate_reward, info_bits};
use engine_core::typed::{
    ActionSpace, Capabilities, DecodeError, EncodeError, Encoding, EngineId, Game,
};
use engine_core::{register_game, GameAdapter, GameMetadata};
use rand_chacha::ChaCha20Rng;

/// Board dimensions
pub const COLS: usize = 8;
pub const ROWS: usize = 8;
pub const BOARD_SIZE: usize = COLS * ROWS; // 64

/// Number of actions: 64 board positions + 1 pass
pub const NUM_ACTIONS: usize = 65;

/// Pass action index
pub const PASS_ACTION: u32 = 64;

/// Register Othello with the global game registry
///
/// Call this function once at startup to make Othello available
/// via `EngineContext::new("othello")`.
pub fn register_othello() {
    register_game("othello".to_string(), || {
        Box::new(GameAdapter::new(Othello::new()))
    });
}

/// Othello game state
///
/// Represents the complete state of an Othello game including the board,
/// current player, winner information, and pass tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct State {
    /// Board representation: 0=empty, 1=Black, 2=White
    /// Stored in row-major order with row 0 at the top
    board: [u8; BOARD_SIZE],
    /// Current player: 1=Black, 2=White (Black goes first)
    current_player: u8,
    /// Winner: 0=none/ongoing, 1=Black, 2=White, 3=draw
    winner: u8,
    /// Number of consecutive passes (0, 1, or 2). Game ends at 2.
    pass_count: u8,
}

impl State {
    /// Create a new initial game state
    pub fn new() -> Self {
        let mut board = [0u8; BOARD_SIZE];
        // Standard Othello starting position: 4 pieces in the center
        // Black at (3, 3) and (4, 4), White at (3, 4) and (4, 3)
        board[Self::pos(3, 3)] = 1; // Black at D4
        board[Self::pos(4, 4)] = 1; // Black at E5
        board[Self::pos(3, 4)] = 2; // White at E4
        board[Self::pos(4, 3)] = 2; // White at D5

        Self {
            board,
            current_player: 1, // Black goes first
            winner: 0,
            pass_count: 0,
        }
    }

    /// Check if the game is over
    pub fn is_done(&self) -> bool {
        self.winner != 0
    }

    /// Get the current player (1=Black, 2=White)
    pub fn current_player(&self) -> u8 {
        self.current_player
    }

    /// Convert column and row to board index
    #[inline]
    fn pos(col: usize, row: usize) -> usize {
        row * COLS + col
    }

    /// Convert index to (col, row)
    #[inline]
    fn idx_to_pos(idx: usize) -> (usize, usize) {
        (idx % COLS, idx / COLS)
    }

    /// Get the opponent of a player
    #[inline]
    fn opponent(player: u8) -> u8 {
        if player == 1 {
            2
        } else {
            1
        }
    }

    /// Check if a move is valid (must flip at least one opponent piece)
    fn is_valid_move(&self, pos: usize) -> bool {
        if self.board[pos] != 0 {
            return false;
        }

        let (col, row) = Self::idx_to_pos(pos);
        let player = self.current_player;
        let opponent = Self::opponent(player);

        // Check all 8 directions
        let directions: [(isize, isize); 8] = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];

        for (dc, dr) in directions {
            let mut c = col as isize + dc;
            let mut r = row as isize + dr;
            let mut found_opponent = false;

            // Move in this direction, looking for opponent pieces
            while c >= 0 && c < COLS as isize && r >= 0 && r < ROWS as isize {
                let cell = self.board[Self::pos(c as usize, r as usize)];
                if cell == opponent {
                    found_opponent = true;
                    c += dc;
                    r += dr;
                } else if cell == player && found_opponent {
                    // Found player piece after opponent pieces - valid move!
                    return true;
                } else {
                    // Empty or no sandwich - invalid in this direction
                    break;
                }
            }
        }

        false
    }

    /// Get legal moves (positions that are empty and would flip at least one piece)
    pub fn legal_moves(&self) -> Vec<u32> {
        if self.is_done() {
            return Vec::new();
        }

        let mut moves: Vec<u32> = (0..BOARD_SIZE)
            .filter(|&pos| self.is_valid_move(pos))
            .map(|pos| pos as u32)
            .collect();

        // If no board moves are available, pass is the only legal move
        if moves.is_empty() {
            moves.push(PASS_ACTION);
        }

        moves
    }

    /// Bit-mask representation of legal moves for board positions (0-63).
    /// Does not include pass action - use `is_pass_legal()` for that.
    pub fn legal_moves_mask(&self) -> u64 {
        if self.is_done() {
            return 0;
        }

        let mut mask: u64 = 0;

        for pos in 0..BOARD_SIZE {
            if self.is_valid_move(pos) {
                mask |= 1u64 << pos;
            }
        }

        mask
    }

    /// Check if pass action is legal (only when no board moves available)
    pub fn is_pass_legal(&self) -> bool {
        if self.is_done() {
            return false;
        }
        !self.has_any_legal_moves()
    }

    /// Get legal actions mask including pass.
    /// Returns (board_mask: u64, pass_legal: bool)
    pub fn legal_actions(&self) -> (u64, bool) {
        (self.legal_moves_mask(), self.is_pass_legal())
    }

    /// Make a move and return the new state
    pub fn make_move(&self, action: u32) -> State {
        if self.is_done() {
            return self.clone();
        }

        // Handle pass action
        if action == PASS_ACTION {
            let mut new_state = self.clone();
            new_state.pass_count += 1;
            new_state.current_player = Self::opponent(self.current_player);

            // Check if game should end (two consecutive passes)
            if new_state.pass_count >= 2 {
                new_state.determine_winner();
            }
            // If only one pass so far, game continues
            // The frontend will detect no moves and show pass button

            return new_state;
        }

        let pos = action as usize;
        if pos >= BOARD_SIZE || self.board[pos] != 0 || !self.is_valid_move(pos) {
            return self.clone(); // Invalid move
        }

        let (col, row) = Self::idx_to_pos(pos);
        let player = self.current_player;
        let opponent = Self::opponent(player);

        let mut new_state = self.clone();
        new_state.board[pos] = player;
        new_state.pass_count = 0; // Reset pass count on any board move

        // Flip pieces in all 8 directions
        let directions: [(isize, isize); 8] = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];

        for (dc, dr) in directions {
            let mut to_flip: Vec<usize> = Vec::new();
            let mut c = col as isize + dc;
            let mut r = row as isize + dr;

            // Collect opponent pieces in this direction
            while c >= 0 && c < COLS as isize && r >= 0 && r < ROWS as isize {
                let check_pos = Self::pos(c as usize, r as usize);
                let cell = self.board[check_pos];

                if cell == opponent {
                    to_flip.push(check_pos);
                    c += dc;
                    r += dr;
                } else if cell == player && !to_flip.is_empty() {
                    // Found sandwich - flip all collected pieces
                    for flip_pos in to_flip {
                        new_state.board[flip_pos] = player;
                    }
                    break;
                } else {
                    break;
                }
            }
        }

        // Switch player
        new_state.current_player = opponent;
        new_state.pass_count = 0; // Reset pass count on any board move

        // Check if the new current player has any legal moves
        if !new_state.has_any_legal_moves() {
            // Current player has no moves - they must pass
            // Check if the opponent (previous player) also has no moves
            let opponent_has_moves = {
                let temp_state = State {
                    board: new_state.board,
                    current_player: Self::opponent(new_state.current_player),
                    winner: 0,
                    pass_count: 0,
                };
                temp_state.has_any_legal_moves()
            };

            if !opponent_has_moves {
                // Both players have no moves - game ends
                new_state.determine_winner();
            } else {
                // Current player must pass, but opponent had moves
                // This is the first pass - increment and continue
                new_state.pass_count = 1;
                // Note: The frontend/web layer should detect this and auto-pass
                // or the next move attempt will be PASS_ACTION
            }
        }

        new_state
    }

    /// Check if the current player has any legal moves
    fn has_any_legal_moves(&self) -> bool {
        for pos in 0..BOARD_SIZE {
            if self.is_valid_move(pos) {
                return true;
            }
        }
        false
    }

    /// Determine winner by disc count and set winner field
    fn determine_winner(&mut self) {
        let black_count = self.board.iter().filter(|&&c| c == 1).count();
        let white_count = self.board.iter().filter(|&&c| c == 2).count();

        self.winner = if black_count > white_count {
            1 // Black wins
        } else if white_count > black_count {
            2 // White wins
        } else {
            3 // Draw
        };
    }

    /// Count pieces for each player
    pub fn piece_counts(&self) -> (usize, usize) {
        let black = self.board.iter().filter(|&&c| c == 1).count();
        let white = self.board.iter().filter(|&&c| c == 2).count();
        (black, white)
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

/// Othello action - board position (0-63) or pass (64)
pub type Action = u32;

/// Custom observation type for Othello with 65 actions.
///
/// Unlike TwoPlayerObs which uses u64 for legal mask, this supports
/// more than 64 actions by using a separate array.
#[derive(Debug, Clone, PartialEq)]
pub struct OthelloObs {
    /// Board encoding: [64 Black positions, 64 White positions]
    pub board_view: [f32; 128],
    /// Legal moves: [64 board positions, 1 pass action]
    pub legal_moves: [f32; 65],
    /// Current player indicator: [is_black, is_white]
    pub current_player: [f32; 2],
}

impl OthelloObs {
    /// Create a new empty observation.
    pub fn new() -> Self {
        Self {
            board_view: [0.0; 128],
            legal_moves: [0.0; 65],
            current_player: [0.0; 2],
        }
    }

    /// Create observation from game state.
    ///
    /// - `board`: Board array (0=empty, 1=Black, 2=White)
    /// - `legal_mask`: u64 bitmask of legal board positions (0-63)
    /// - `pass_legal`: Whether pass action (64) is legal
    /// - `current_player`: Current player (1=Black, 2=White)
    pub fn from_board(board: &[u8], legal_mask: u64, pass_legal: bool, current_player: u8) -> Self {
        let mut obs = Self::new();

        // Encode board state (one-hot for each player)
        for (i, &cell) in board.iter().enumerate() {
            if cell == 1 {
                obs.board_view[i] = 1.0; // Black in first 64 positions
            } else if cell == 2 {
                obs.board_view[i + 64] = 1.0; // White in second 64 positions
            }
        }

        // Encode legal moves for board positions (0-63)
        for pos in 0..64 {
            if (legal_mask & (1u64 << pos)) != 0 {
                obs.legal_moves[pos] = 1.0;
            }
        }

        // Encode pass action (64)
        if pass_legal {
            obs.legal_moves[64] = 1.0;
        }

        // Encode current player
        if current_player == 1 {
            obs.current_player[0] = 1.0;
        } else {
            obs.current_player[1] = 1.0;
        }

        obs
    }

    /// Encode observation as bytes for neural network input.
    pub fn encode(&self, out: &mut Vec<u8>) {
        use engine_core::game_utils::encode_f32_slices;
        encode_f32_slices(
            out,
            [
                &self.board_view[..],
                &self.legal_moves[..],
                &self.current_player[..],
            ],
        );
    }
}

impl Default for OthelloObs {
    fn default() -> Self {
        Self::new()
    }
}

/// Create observation from game state
pub fn observation_from_state(state: &State) -> OthelloObs {
    let (legal_mask, pass_legal) = state.legal_actions();
    OthelloObs::from_board(&state.board, legal_mask, pass_legal, state.current_player)
}

/// Othello game implementation
#[derive(Debug)]
pub struct Othello;

impl Othello {
    /// Create a new Othello game
    pub fn new() -> Self {
        Self
    }

    /// Pack auxiliary information about the state into a u64 bit-field.
    ///
    /// Uses the standard layout from `engine_core::game_utils::info_bits`:
    /// * Bits 0-63 : Legal move mask (64 board positions)
    /// * Bits 16-19: Current player (1 = Black, 2 = White)
    /// * Bits 20-23: Winner (0 = none, 1 = Black, 2 = White, 3 = draw)
    /// * Bits 24-31: Consecutive passes (0-2)
    fn compute_info_bits(state: &State) -> u64 {
        let legal_mask = state.legal_moves_mask();
        let player = state.current_player;
        let winner = state.winner;
        let passes = state.pass_count as u64;

        info_bits::compute_info_bits(legal_mask, player, winner, passes)
    }
}

impl Default for Othello {
    fn default() -> Self {
        Self::new()
    }
}

/// Observation size: 64 (Black) + 64 (White) + 65 (legal) + 2 (player) = 195
const OBS_SIZE: usize = BOARD_SIZE * 2 + NUM_ACTIONS + 2;

impl Game for Othello {
    type State = State;
    type Action = Action;
    type Obs = OthelloObs;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "othello".to_string(),
            build_id: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            encoding: Encoding {
                state: "othello_state:v1".to_string(),
                action: "discrete_position:v1".to_string(),
                obs: format!("f32x{}:v1", OBS_SIZE), // 195 floats
                schema_version: 1,
            },
            max_horizon: BOARD_SIZE as u32, // Maximum 64 moves (unlikely in practice)
            action_space: ActionSpace::Discrete(NUM_ACTIONS as u32), // 65 possible actions
            preferred_batch: 64,
        }
    }

    fn metadata(&self) -> GameMetadata {
        GameMetadata::new("othello", "Othello")
            .with_board(COLS, ROWS)
            .with_actions(NUM_ACTIONS)
            .with_observation(OBS_SIZE, BOARD_SIZE * 2) // legal mask starts after board views
            .with_players(
                2,
                vec!["Black".to_string(), "White".to_string()],
                vec!['⚫', '⚪'], // Black circle, White circle emoji
            )
            .with_description("Flip opponent pieces to dominate the board!")
            .with_board_type("grid")
    }

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
        // Binary encoding: board (64 bytes) + current_player (1 byte) + winner (1 byte) + pass_count (1 byte)
        out.extend_from_slice(&state.board);
        out.push(state.current_player);
        out.push(state.winner);
        out.push(state.pass_count);
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        let expected_len = BOARD_SIZE + 3; // board + current_player + winner + pass_count
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
        let pass_count = buf[BOARD_SIZE + 2];

        // Validate the state
        if current_player != 1 && current_player != 2 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid current_player: {}",
                current_player
            )));
        }

        if winner > 3 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid winner: {}",
                winner
            )));
        }

        if pass_count > 2 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid pass_count: {}",
                pass_count
            )));
        }

        for &cell in &board {
            if cell > 2 {
                return Err(DecodeError::CorruptedData(format!(
                    "Invalid board cell: {}",
                    cell
                )));
            }
        }

        Ok(State {
            board,
            current_player,
            winner,
            pass_count,
        })
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        if *action >= NUM_ACTIONS as u32 {
            return Err(EncodeError::InvalidData(format!(
                "Invalid action: {}. Must be 0-{}",
                action,
                NUM_ACTIONS - 1
            )));
        }
        // Encode as u32 in little-endian format (4 bytes)
        out.extend_from_slice(&action.to_le_bytes());
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        if buf.len() != 4 {
            return Err(DecodeError::InvalidLength {
                expected: 4,
                actual: buf.len(),
            });
        }

        let action = u32::from_le_bytes(buf.try_into().unwrap());
        if action >= NUM_ACTIONS as u32 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid action: {}. Must be 0-{}",
                action,
                NUM_ACTIONS - 1
            )));
        }

        Ok(action)
    }

    fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        obs.encode(out);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
