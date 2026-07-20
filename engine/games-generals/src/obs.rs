//! Observation tensor encoding — schema `generals_obs:v1`.
//!
//! Player-relative, full-information. Layout (all f32, flattened):
//!
//! ```text
//! [ 9 spatial channels x 64 tiles = 576 ]
//!   ch0 own territory        (1.0 where owner == me)
//!   ch1 enemy territory
//!   ch2 neutral passable     (unowned, not mountain)
//!   ch3 own armies           log1p(army) / log1p(MAX_ARMY_NORM)
//!   ch4 enemy armies         log1p(army) / log1p(MAX_ARMY_NORM)
//!   ch5 cities               (any owner)
//!   ch6 mountains
//!   ch7 generals             (+1.0 own, -1.0 enemy)
//!   ch8 turn progress        (constant plane: round / MAX_TURNS)
//! [ legal-move mask: 257 ]   legal_mask_offset = 576
//! [ current-player one-hot: 2 ]
//! obs_size = 576 + 257 + 2 = 835
//! ```
//!
//! "Own"/"enemy" are relative to the player to act (`current_player`), so
//! the same network plays both seats. Fog channels are intentionally
//! absent; a fog variant gets a new schema version, not a reinterpretation
//! of this one.

use engine_core::game_utils::encode_f32_slices;

use crate::board::{Tile, TileKind};
use crate::params::{BOARD_SIZE, MAX_ARMY_NORM, MAX_TURNS, NUM_ACTIONS};
use crate::rules::fill_legal_moves;

/// Number of spatial channels.
pub const NUM_CHANNELS: usize = 9;
/// Flattened spatial section length.
pub const CHANNELS_LEN: usize = NUM_CHANNELS * BOARD_SIZE;
/// Total observation length in floats.
pub const OBS_SIZE: usize = CHANNELS_LEN + NUM_ACTIONS + 2;
/// Float index where the legal-move plane starts.
pub const LEGAL_MASK_OFFSET: usize = CHANNELS_LEN;

/// Generals observation.
#[derive(Debug, Clone)]
pub struct GeneralsObs {
    pub channels: [f32; CHANNELS_LEN],
    pub legal_moves: [f32; NUM_ACTIONS],
    pub current_player: [f32; 2],
}

impl GeneralsObs {
    /// Build the observation for `player` (1 or 2) from the tile array.
    pub fn from_tiles(tiles: &[Tile], player: u8, alive: [bool; 2], round: u32) -> Self {
        let mut channels = [0.0f32; CHANNELS_LEN];
        let norm = (1.0 + MAX_ARMY_NORM).ln();
        let turn_progress = (round as f32 / MAX_TURNS as f32).min(1.0);

        for (i, tile) in tiles.iter().enumerate().take(BOARD_SIZE) {
            let ch = |c: usize| c * BOARD_SIZE + i;
            let own = tile.owner == player;
            let enemy = !tile.is_neutral() && !own;

            if own {
                channels[ch(0)] = 1.0;
                channels[ch(3)] = (1.0 + tile.army as f32).ln() / norm;
            } else if enemy {
                channels[ch(1)] = 1.0;
                channels[ch(4)] = (1.0 + tile.army as f32).ln() / norm;
            } else if !tile.is_mountain() {
                channels[ch(2)] = 1.0;
            }

            match tile.kind {
                TileKind::City => channels[ch(5)] = 1.0,
                TileKind::Mountain => channels[ch(6)] = 1.0,
                TileKind::General => channels[ch(7)] = if own { 1.0 } else { -1.0 },
                TileKind::Normal => {}
            }
            channels[ch(8)] = turn_progress;
        }

        let mut legal_moves = [0.0f32; NUM_ACTIONS];
        let player_alive = alive[player as usize - 1];
        fill_legal_moves(tiles, player, player_alive, &mut legal_moves);

        let mut current_player = [0.0f32; 2];
        current_player[player as usize - 1] = 1.0;

        Self {
            channels,
            legal_moves,
            current_player,
        }
    }

    /// Encode as little-endian f32 bytes for the neural network.
    pub fn encode(&self, out: &mut Vec<u8>) {
        encode_f32_slices(
            out,
            [
                &self.channels[..],
                &self.legal_moves[..],
                &self.current_player[..],
            ],
        );
    }
}
