//! Action encoding and move validity.
//!
//! Canonical action space (matches the Go rules calculator
//! `rules/legal_moves.go`, which is authoritative over the Go experience
//! serializer's diverging order):
//!
//! ```text
//! index = (y * WIDTH + x) * 4 + direction
//! direction: 0 = up (y-1), 1 = right (x+1), 2 = down (y+1), 3 = left (x-1)
//! index 256 = wait (no-op)
//! ```
//!
//! Half-moves are not part of this ruleset: every move sends `army - 1`.

use crate::board::{idx, in_bounds, xy, Tile};
use crate::params::{NUM_ACTIONS, WAIT_ACTION};

/// Direction offsets in canonical order: up, right, down, left.
pub const DIR_DX: [isize; 4] = [0, 1, 0, -1];
pub const DIR_DY: [isize; 4] = [-1, 0, 1, 0];

/// A decoded action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    /// Move all-but-one army from the tile at `from` in `dir` (0-3).
    Step {
        from: usize,
        dir: usize,
    },
    Wait,
}

/// Decode a raw action index. Returns `None` if out of range.
pub fn decode_move(action: u32) -> Option<Move> {
    if action == WAIT_ACTION {
        return Some(Move::Wait);
    }
    if action as usize >= NUM_ACTIONS {
        return None;
    }
    Some(Move::Step {
        from: action as usize / 4,
        dir: action as usize % 4,
    })
}

/// Encode a (from, dir) move as an action index.
pub fn encode_move(from: usize, dir: usize) -> u32 {
    (from * 4 + dir) as u32
}

/// Target tile index for moving from `from` in `dir`, if it stays on-board.
pub fn move_target(from: usize, dir: usize) -> Option<usize> {
    let (x, y) = xy(from);
    let nx = x as isize + DIR_DX[dir];
    let ny = y as isize + DIR_DY[dir];
    in_bounds(nx, ny).then(|| idx(nx as usize, ny as usize))
}

/// Allocation-free move validity, ported from Go `core.IsValidMove`:
/// source owned by `player` with more than 1 army, target on-board and not
/// a mountain. Returns the target index when valid.
pub fn valid_move_target(tiles: &[Tile], player: u8, from: usize, dir: usize) -> Option<usize> {
    let from_tile = &tiles[from];
    if from_tile.owner != player || from_tile.army <= 1 {
        return None;
    }
    let to = move_target(from, dir)?;
    (!tiles[to].is_mountain()).then_some(to)
}
