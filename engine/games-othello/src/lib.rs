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

/// Board dimensions
pub const COLS: usize = 8;
pub const ROWS: usize = 8;
pub const BOARD_SIZE: usize = COLS * ROWS; // 64

/// Immutable environment contract revision for wire formats and semantics.
pub const ENV_CONTRACT_VERSION: u32 = 2;

/// Number of actions: 64 board positions + 1 pass
pub const NUM_ACTIONS: usize = 65;

/// Pass action index
pub const PASS_ACTION: u32 = 64;

/// The 8 direction vectors (dc, dr) used for move validation and flipping.
const DIRECTIONS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Register Othello with the global game registry
///
/// Call this function once at startup to make Othello available
/// via `EngineContext::new("othello")`.
pub fn register_othello() {
    register_board_game::<Othello>().expect("othello environment must only be registered once");
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

    /// Check if a move is valid (must flip at least one opponent piece)
    fn is_valid_move(&self, pos: usize) -> bool {
        if self.board[pos] != 0 {
            return false;
        }

        let (col, row) = Self::idx_to_pos(pos);
        let player = self.current_player;
        let opponent = opponent(player);

        // Check all 8 directions
        for (dc, dr) in DIRECTIONS {
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

    /// Make a move and return the new state
    pub fn make_move(&self, action: u32) -> State {
        if self.is_done() {
            return self.clone();
        }

        // Handle pass action
        if action == PASS_ACTION {
            let mut new_state = self.clone();
            new_state.pass_count += 1;
            new_state.current_player = opponent(self.current_player);

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
        let opponent = opponent(player);

        let mut new_state = self.clone();
        new_state.board[pos] = player;
        new_state.pass_count = 0; // Reset pass count on any board move

        // Flip pieces in all 8 directions
        for (dc, dr) in DIRECTIONS {
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

        // Check if the new current player has any legal moves
        if !new_state.has_any_legal_moves() {
            // Current player has no moves - they must pass
            // Check if the opponent (previous player) also has no moves
            let opponent_has_moves = {
                let temp_state = State {
                    board: new_state.board,
                    current_player: player,
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
        let (black_count, white_count) = self.piece_counts();

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

/// Player-relative Othello observation: two 8x8 occupancy planes.
pub type OthelloObs = TwoPlayerObs<128>;

/// Create observation from game state
pub fn observation_from_state(state: &State) -> Result<OthelloObs, TwoPlayerObsError> {
    OthelloObs::from_board(&state.board, state.current_player)
}

/// Othello game implementation
#[derive(Debug)]
pub struct Othello;

impl Othello {
    /// Create a new Othello game
    pub fn new() -> Self {
        Self
    }
}

impl Default for Othello {
    fn default() -> Self {
        Self::new()
    }
}

/// Observation size: two player-relative board planes.
impl BoardGame for Othello {
    type State = State;
    type Action = Action;
    type Observation = OthelloObs;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "othello".to_string(),
            build_id: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: ENV_CONTRACT_VERSION,
            encoding: Encoding::discrete_u32_le(
                "othello_state:v1",
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
                ActionSpace::discrete(NUM_ACTIONS as u32),
            ),
            preferred_batch: 64,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("othello", "Othello")
            .with_description("Flip opponent pieces to dominate the board!")
            .with_board(BoardGameMetadata::new(COLS, ROWS).with_players(vec![
                BoardPlayerMetadata::new("Black", "⚫"),
                BoardPlayerMetadata::new("White", "⚪"),
            ]))
    }

    // reset/step mirror games-tictactoe and games-connect4; shared reward and
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
        let legal = !state.is_done()
            && if action == PASS_ACTION {
                state.is_pass_legal()
            } else {
                usize::try_from(action)
                    .ok()
                    .filter(|&position| position < BOARD_SIZE)
                    .is_some_and(|position| state.is_valid_move(position))
            };
        if !legal {
            return Err(EnvironmentError::InvalidAction(format!(
                "action {action} is not legal in the current Othello state"
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
        validate_player_and_winner(current_player, winner)?;

        if pass_count > 2 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid pass_count: {}",
                pass_count
            )));
        }

        validate_board_cells(&board)?;

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
        let action = decode_action_u32(buf)?;
        if action >= NUM_ACTIONS as u32 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid action: {}. Must be 0-{}",
                action,
                NUM_ACTIONS - 1
            )));
        }

        Ok(action)
    }

    fn encode_observation(obs: &Self::Observation, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        obs.encode(out);
        Ok(())
    }

    fn legal_actions(state: &Self::State) -> Result<LegalMask, EnvironmentError> {
        let mut mask = LegalMask::new(NUM_ACTIONS);
        for action in state.legal_moves() {
            mask.set(action as usize);
        }
        Ok(mask)
    }

    fn view(state: &Self::State) -> BoardView {
        BoardView::from_owners(&state.board, state.current_player, state.winner)
    }
}

#[cfg(test)]
mod tests;
