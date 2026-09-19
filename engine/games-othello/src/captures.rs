//! Othello's bracketed-disc rule, shared by move legality and application.
//!
//! Masks contain only captured opponent discs, never the destination or the
//! closing friendly disc. Turn, pass, and winner handling remain with `State`.

use engine_core::board_profile::opponent;

use crate::{BOARD_SIZE, COLS, ROWS};

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

/// Whether an in-bounds destination captures any discs. Legality queries stop
/// at the first bracketed ray; they do not need to collect every capture.
pub(super) fn has_capture(board: &[u8; BOARD_SIZE], player: u8, position: usize) -> bool {
    board[position] == 0
        && DIRECTIONS
            .into_iter()
            .any(|direction| captured_in_direction(board, player, position, direction) != 0)
}

/// All discs captured by an in-bounds placement, or zero for an illegal one.
pub(super) fn captured_discs(board: &[u8; BOARD_SIZE], player: u8, position: usize) -> u64 {
    if board[position] != 0 {
        return 0;
    }
    DIRECTIONS.into_iter().fold(0, |captured, direction| {
        captured | captured_in_direction(board, player, position, direction)
    })
}

fn captured_in_direction(
    board: &[u8; BOARD_SIZE],
    player: u8,
    position: usize,
    (dc, dr): (isize, isize),
) -> u64 {
    let mut col = (position % COLS) as isize + dc;
    let mut row = (position / COLS) as isize + dr;
    let enemy = opponent(player);
    let mut captured = 0;

    while (0..COLS as isize).contains(&col) && (0..ROWS as isize).contains(&row) {
        let square = row as usize * COLS + col as usize;
        let cell = board[square];
        if cell == enemy {
            captured |= 1u64 << square;
            col += dc;
            row += dr;
        } else {
            // An empty square breaks the bracket. An adjacent friendly disc
            // returns the still-empty mask, so it cannot make a move legal.
            return if cell == player { captured } else { 0 };
        }
    }

    // Opponents reaching the board edge have no closing friendly disc.
    0
}
