//! Board and tile types, ported from the Go engine's `core/board.go`.
//!
//! Differences from the Go source:
//! - Owner is `0 = neutral, 1 = player 1, 2 = player 2` (Cartridge2's
//!   two-player convention) instead of Go's `-1` neutral / 0-based IDs.
//! - The per-player visibility bitfields are not ported: this ruleset is
//!   full-information, and the fog variant will need observation history
//!   anyway (see the fog notes in `lib.rs`).

use crate::params::{BOARD_SIZE, HEIGHT, WIDTH};

/// Neutral (unowned) tile owner value.
pub const NEUTRAL: u8 = 0;

/// Tile terrain/structure kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TileKind {
    Normal = 0,
    General = 1,
    City = 2,
    Mountain = 3,
}

impl TileKind {
    /// Decode from the wire byte; `None` for out-of-range values.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Normal),
            1 => Some(Self::General),
            2 => Some(Self::City),
            3 => Some(Self::Mountain),
            _ => None,
        }
    }
}

/// A single board tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    /// 0 = neutral, 1/2 = players.
    pub owner: u8,
    pub army: u32,
    pub kind: TileKind,
}

impl Tile {
    pub fn neutral() -> Self {
        Self {
            owner: NEUTRAL,
            army: 0,
            kind: TileKind::Normal,
        }
    }

    pub fn is_neutral(&self) -> bool {
        self.owner == NEUTRAL
    }

    pub fn is_mountain(&self) -> bool {
        self.kind == TileKind::Mountain
    }
}

/// Row-major tile index for (x, y).
#[inline]
pub fn idx(x: usize, y: usize) -> usize {
    y * WIDTH + x
}

/// Inverse of [`idx`].
#[inline]
pub fn xy(i: usize) -> (usize, usize) {
    (i % WIDTH, i / WIDTH)
}

/// Whether (x, y) lies on the board. Takes signed coords so callers can
/// probe neighbors without underflow checks.
#[inline]
pub fn in_bounds(x: isize, y: isize) -> bool {
    x >= 0 && (x as usize) < WIDTH && y >= 0 && (y as usize) < HEIGHT
}

/// Manhattan distance between two tile indices.
pub fn manhattan(a: usize, b: usize) -> usize {
    let (ax, ay) = xy(a);
    let (bx, by) = xy(b);
    ax.abs_diff(bx) + ay.abs_diff(by)
}

/// A fresh all-neutral board.
pub fn new_board() -> Vec<Tile> {
    vec![Tile::neutral(); BOARD_SIZE]
}
