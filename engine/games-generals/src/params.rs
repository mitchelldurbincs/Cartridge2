//! Game balance parameters for the `generals_8x8` ruleset.
//!
//! Values mirror the Go engine's config defaults
//! (`GeneralsReinforcementLearning/internal/game/constants.go`), with the
//! ruleset simplifications documented in `lib.rs` (no half-moves, strictly
//! alternating turns).

/// Board width in tiles.
pub const WIDTH: usize = 8;
/// Board height in tiles.
pub const HEIGHT: usize = 8;
/// Total tiles on the board.
pub const BOARD_SIZE: usize = WIDTH * HEIGHT;

/// Move actions: one per (tile, direction) pair.
pub const NUM_MOVE_ACTIONS: usize = BOARD_SIZE * 4;
/// The wait (no-op) action index. Always legal so a player with no moves
/// can never deadlock the game.
pub const WAIT_ACTION: u32 = NUM_MOVE_ACTIONS as u32;
/// Total actions: 256 moves + 1 wait.
pub const NUM_ACTIONS: usize = NUM_MOVE_ACTIONS + 1;

/// One city is placed per this many tiles.
pub const CITY_RATIO: usize = 20;
/// Starting army on a neutral city.
pub const CITY_START_ARMY: u32 = 40;
/// Minimum Manhattan distance between generals.
pub const MIN_GENERAL_SPACING: usize = 5;
/// Starting army on each general (2 so players can move on round 1).
pub const GENERAL_START_ARMY: u32 = 2;

/// Armies produced per round by a general.
pub const GENERAL_PRODUCTION: u32 = 1;
/// Armies produced per round by a city (owned only).
pub const CITY_PRODUCTION: u32 = 1;
/// Armies produced by normal owned tiles on growth rounds.
pub const NORMAL_PRODUCTION: u32 = 1;
/// Normal tiles grow every this many rounds.
pub const NORMAL_GROW_INTERVAL: u32 = 25;

/// Rounds (both players moved) before the game is adjudicated by territory
/// (more tiles wins; armies tiebreak; a truly even position is a draw).
/// Kept short so self-play games are dense in decisive outcomes — the
/// original 500-round pure-draw cap collapsed training into 100% draws.
pub const MAX_TURNS: u32 = 200;

/// Mountain vein generation (Go `DefaultMapConfig` formulas for 8x8).
pub const NUM_MOUNTAIN_VEINS: usize = BOARD_SIZE / 50; // 1
pub const MIN_VEIN_LENGTH: usize = 3;
pub const MAX_VEIN_LENGTH: usize = WIDTH / 4; // 2 (< min => veins are min length)

/// Army normalization ceiling for observation encoding (log1p scale).
pub const MAX_ARMY_NORM: f32 = 1000.0;
