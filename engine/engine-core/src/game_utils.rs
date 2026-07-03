//! Shared utilities for two-player game implementations
//!
//! This module provides common functionality used across multiple game implementations
//! to reduce code duplication and ensure consistent behavior.

use crate::typed::DecodeError;

/// Get the opponent of a player in a two-player game (1 <-> 2).
///
/// # Example
/// ```
/// use engine_core::game_utils::opponent;
///
/// assert_eq!(opponent(1), 2);
/// assert_eq!(opponent(2), 1);
/// ```
#[inline]
pub fn opponent(player: u8) -> u8 {
    if player == 1 {
        2
    } else {
        1
    }
}

/// Validate the `current_player` and `winner` fields of a decoded two-player game state.
///
/// Shared by game `decode_state` implementations. Accepts `current_player` of 1 or 2
/// and `winner` of 0 (ongoing), 1, 2, or 3 (draw).
///
/// # Errors
/// Returns `DecodeError::CorruptedData` if either field is out of range.
pub fn validate_player_and_winner(current_player: u8, winner: u8) -> Result<(), DecodeError> {
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

    Ok(())
}

/// Validate that every cell of a decoded board is 0 (empty), 1 (player 1), or 2 (player 2).
///
/// # Errors
/// Returns `DecodeError::CorruptedData` on the first out-of-range cell.
pub fn validate_board_cells(board: &[u8]) -> Result<(), DecodeError> {
    for &cell in board {
        if cell > 2 {
            return Err(DecodeError::CorruptedData(format!(
                "Invalid board cell: {}",
                cell
            )));
        }
    }
    Ok(())
}

/// Decode a discrete action encoded as a little-endian u32 (4 bytes).
///
/// Shared by game `decode_action` implementations; callers apply their own
/// game-specific range check on the returned value.
///
/// # Errors
/// Returns `DecodeError::InvalidLength` if `buf` is not exactly 4 bytes.
pub fn decode_action_u32(buf: &[u8]) -> Result<u32, DecodeError> {
    if buf.len() != 4 {
        return Err(DecodeError::InvalidLength {
            expected: 4,
            actual: buf.len(),
        });
    }
    Ok(u32::from_le_bytes(buf.try_into().unwrap()))
}

/// Calculate reward for a two-player zero-sum game.
///
/// Returns the reward from the perspective of the player who just moved.
///
/// # Arguments
/// * `winner` - Winner indicator: 0=ongoing, 1=player1 wins, 2=player2 wins, 3=draw
/// * `previous_player` - The player who made the move (1 or 2)
///
/// # Returns
/// * `1.0` if the previous player won
/// * `-1.0` if the previous player lost
/// * `0.0` for draws or ongoing games
///
/// # Example
/// ```
/// use engine_core::game_utils::calculate_reward;
///
/// // Player 1 wins, viewed from player 1's perspective
/// assert_eq!(calculate_reward(1, 1), 1.0);
///
/// // Player 1 wins, viewed from player 2's perspective
/// assert_eq!(calculate_reward(1, 2), -1.0);
///
/// // Draw
/// assert_eq!(calculate_reward(3, 1), 0.0);
///
/// // Game ongoing
/// assert_eq!(calculate_reward(0, 1), 0.0);
/// ```
#[inline]
pub fn calculate_reward(winner: u8, previous_player: u8) -> f32 {
    match winner {
        0 => 0.0, // Game ongoing
        1 => {
            if previous_player == 1 {
                1.0
            } else {
                -1.0
            }
        } // Player 1 wins
        2 => {
            if previous_player == 2 {
                1.0
            } else {
                -1.0
            }
        } // Player 2 wins
        3 => 0.0, // Draw
        _ => 0.0, // Shouldn't happen
    }
}

/// Standard bit-field layout constants for game info encoding.
///
/// The info u64 is laid out as follows (little endian bit numbering):
/// * Bits 0-15  : Legal move mask (varies by game: 9 bits for TicTacToe, 7 for Connect4)
/// * Bits 16-19 : Current player (1 = player1, 2 = player2)
/// * Bits 20-23 : Winner (0 = none, 1 = player1, 2 = player2, 3 = draw)
/// * Bits 24-31 : Moves played so far
pub mod info_bits {
    /// Bit position for current player field
    pub const CURRENT_PLAYER_SHIFT: u32 = 16;
    /// Bit position for winner field
    pub const WINNER_SHIFT: u32 = 20;
    /// Bit position for moves played counter
    pub const MOVES_PLAYED_SHIFT: u32 = 24;

    /// Pack game state auxiliary information into a u64 bit-field.
    ///
    /// # Arguments
    /// * `legal_moves_mask` - Bit mask of legal moves (game-specific width)
    /// * `current_player` - Current player (1 or 2)
    /// * `winner` - Winner indicator (0=ongoing, 1=p1, 2=p2, 3=draw)
    /// * `moves_played` - Number of moves played so far
    ///
    /// # Example
    /// ```
    /// use engine_core::game_utils::info_bits::compute_info_bits;
    ///
    /// // TicTacToe: all 9 positions legal, player 1 to move, no winner yet, 0 moves
    /// let info = compute_info_bits(0x1FF, 1, 0, 0);
    /// assert_eq!(info & 0x1FF, 0x1FF);  // Legal moves mask
    /// assert_eq!((info >> 16) & 0xF, 1); // Current player
    /// assert_eq!((info >> 20) & 0xF, 0); // Winner
    /// assert_eq!((info >> 24) & 0xFF, 0); // Moves played
    /// ```
    #[inline]
    pub fn compute_info_bits(
        legal_moves_mask: u64,
        current_player: u8,
        winner: u8,
        moves_played: u64,
    ) -> u64 {
        let mut info = legal_moves_mask;
        info |= (current_player as u64) << CURRENT_PLAYER_SHIFT;
        info |= (winner as u64) << WINNER_SHIFT;
        info |= moves_played << MOVES_PLAYED_SHIFT;
        info
    }

    /// Extract the legal moves mask from info bits
    #[inline]
    pub fn extract_legal_mask(info: u64, mask_width: u32) -> u64 {
        info & ((1u64 << mask_width) - 1)
    }

    /// Extract the current player from info bits
    #[inline]
    pub fn extract_current_player(info: u64) -> u8 {
        ((info >> CURRENT_PLAYER_SHIFT) & 0xF) as u8
    }

    /// Extract the winner from info bits
    #[inline]
    pub fn extract_winner(info: u64) -> u8 {
        ((info >> WINNER_SHIFT) & 0xF) as u8
    }

    /// Extract the moves played count from info bits
    #[inline]
    pub fn extract_moves_played(info: u64) -> u64 {
        (info >> MOVES_PLAYED_SHIFT) & 0xFF
    }
}

/// Encode multiple f32 slices to bytes in little-endian format.
///
/// This is a common pattern for encoding observations that consist of
/// multiple float arrays (board view, legal moves, current player).
///
/// # Arguments
/// * `out` - Output buffer to append bytes to
/// * `slices` - Iterator of f32 slices to encode
///
/// # Example
/// ```
/// use engine_core::game_utils::encode_f32_slices;
///
/// let board = [1.0f32, 0.0, 0.0];
/// let legal = [1.0f32, 1.0, 1.0];
/// let player = [1.0f32, 0.0];
///
/// let mut buf = Vec::new();
/// encode_f32_slices(&mut buf, [&board[..], &legal[..], &player[..]]);
///
/// // Should be 8 floats * 4 bytes = 32 bytes
/// assert_eq!(buf.len(), 32);
/// ```
pub fn encode_f32_slices<'a>(out: &mut Vec<u8>, slices: impl IntoIterator<Item = &'a [f32]>) {
    for slice in slices {
        for &value in slice {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_reward_player1_wins() {
        // Player 1 wins, from player 1's perspective
        assert_eq!(calculate_reward(1, 1), 1.0);
        // Player 1 wins, from player 2's perspective
        assert_eq!(calculate_reward(1, 2), -1.0);
    }

    #[test]
    fn test_calculate_reward_player2_wins() {
        // Player 2 wins, from player 2's perspective
        assert_eq!(calculate_reward(2, 2), 1.0);
        // Player 2 wins, from player 1's perspective
        assert_eq!(calculate_reward(2, 1), -1.0);
    }

    #[test]
    fn test_calculate_reward_draw() {
        assert_eq!(calculate_reward(3, 1), 0.0);
        assert_eq!(calculate_reward(3, 2), 0.0);
    }

    #[test]
    fn test_calculate_reward_ongoing() {
        assert_eq!(calculate_reward(0, 1), 0.0);
        assert_eq!(calculate_reward(0, 2), 0.0);
    }

    #[test]
    fn test_compute_info_bits_tictactoe() {
        // TicTacToe: all positions legal (9 bits), player 1, no winner, 0 moves
        let info = info_bits::compute_info_bits(0x1FF, 1, 0, 0);

        assert_eq!(info_bits::extract_legal_mask(info, 9), 0x1FF);
        assert_eq!(info_bits::extract_current_player(info), 1);
        assert_eq!(info_bits::extract_winner(info), 0);
        assert_eq!(info_bits::extract_moves_played(info), 0);
    }

    #[test]
    fn test_compute_info_bits_connect4() {
        // Connect4: all columns legal (7 bits), player 2, player 1 won, 10 moves
        let info = info_bits::compute_info_bits(0x7F, 2, 1, 10);

        assert_eq!(info_bits::extract_legal_mask(info, 7), 0x7F);
        assert_eq!(info_bits::extract_current_player(info), 2);
        assert_eq!(info_bits::extract_winner(info), 1);
        assert_eq!(info_bits::extract_moves_played(info), 10);
    }

    #[test]
    fn test_encode_f32_slices() {
        let board = [1.0f32, 0.0];
        let legal = [1.0f32];
        let player = [0.0f32, 1.0];

        let mut buf = Vec::new();
        encode_f32_slices(&mut buf, [&board[..], &legal[..], &player[..]]);

        // 5 floats * 4 bytes = 20 bytes
        assert_eq!(buf.len(), 20);

        // Verify first float is 1.0
        let first = f32::from_le_bytes(buf[0..4].try_into().unwrap());
        assert_eq!(first, 1.0);

        // Verify last float is 1.0
        let last = f32::from_le_bytes(buf[16..20].try_into().unwrap());
        assert_eq!(last, 1.0);
    }

    #[test]
    fn test_encode_f32_slices_empty() {
        let mut buf = Vec::new();
        encode_f32_slices(&mut buf, std::iter::empty::<&[f32]>());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_opponent() {
        assert_eq!(opponent(1), 2);
        assert_eq!(opponent(2), 1);
    }

    #[test]
    fn test_validate_player_and_winner_accepts_valid_combinations() {
        for player in [1u8, 2] {
            for winner in [0u8, 1, 2, 3] {
                assert!(validate_player_and_winner(player, winner).is_ok());
            }
        }
    }

    #[test]
    fn test_validate_player_and_winner_rejects_invalid_player() {
        for player in [0u8, 3, 255] {
            let err = validate_player_and_winner(player, 0).unwrap_err();
            assert!(matches!(err, DecodeError::CorruptedData(_)));
        }
    }

    #[test]
    fn test_validate_player_and_winner_rejects_invalid_winner() {
        for winner in [4u8, 255] {
            let err = validate_player_and_winner(1, winner).unwrap_err();
            assert!(matches!(err, DecodeError::CorruptedData(_)));
        }
    }

    #[test]
    fn test_validate_board_cells() {
        assert!(validate_board_cells(&[0, 1, 2, 1, 0]).is_ok());
        assert!(validate_board_cells(&[]).is_ok());

        let err = validate_board_cells(&[0, 1, 3]).unwrap_err();
        assert!(matches!(err, DecodeError::CorruptedData(_)));
    }

    #[test]
    fn test_decode_action_u32_roundtrip() {
        for value in [0u32, 4, 63, 64, u32::MAX] {
            let buf = value.to_le_bytes();
            assert_eq!(decode_action_u32(&buf).unwrap(), value);
        }
    }

    #[test]
    fn test_decode_action_u32_rejects_wrong_length() {
        for bad in [&[][..], &[1][..], &[1, 2, 3][..], &[1, 2, 3, 4, 5][..]] {
            let err = decode_action_u32(bad).unwrap_err();
            assert!(matches!(
                err,
                DecodeError::InvalidLength { expected: 4, .. }
            ));
        }
    }
}
