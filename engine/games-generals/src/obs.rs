//! Observation tensor encoding — schema `generals_obs:v2`.
//!
//! Player-relative, full-information. Layout (all f32, flattened):
//!
//! ```text
//! [ 10 spatial channels x 64 tiles = 640 ]
//!   ch0 own territory        (1.0 where owner == me)
//!   ch1 enemy territory
//!   ch2 neutral passable     (unowned, not mountain)
//!   ch3 own armies           log1p(army) / log1p(MAX_ARMY_NORM)
//!   ch4 enemy armies         log1p(army) / log1p(MAX_ARMY_NORM)
//!   ch5 cities               (any owner)
//!   ch6 mountains
//!   ch7 generals             (+1.0 own, -1.0 enemy)
//!   ch8 turn progress        (constant plane: round / MAX_TURNS)
//!   ch9 plies remaining      (constant plane: remaining / (2 * MAX_TURNS),
//!                              zero after termination)
//! Action availability and the observing agent are carried by the generic
//! timestep decision envelope, not duplicated in this tensor.
//! obs_size = 640
//! ```
//!
//! "Own"/"enemy" are relative to the player to act (`current_player`), so
//! the same network plays both seats. Fog channels are intentionally
//! absent; a fog variant gets a new schema version, not a reinterpretation
//! of this one.

use engine_core::board_profile::encode_f32_slices;

use crate::board::TileKind;
use crate::params::{BOARD_SIZE, MAX_ARMY_NORM, MAX_TURNS};
use crate::State;

/// Number of spatial channels.
pub const NUM_CHANNELS: usize = 10;
/// Flattened spatial section length.
pub const CHANNELS_LEN: usize = NUM_CHANNELS * BOARD_SIZE;
/// Total observation length in floats.
pub const OBS_SIZE: usize = CHANNELS_LEN;

/// Generals observation.
#[derive(Debug, Clone)]
pub struct GeneralsObs {
    pub channels: [f32; CHANNELS_LEN],
}

impl GeneralsObs {
    /// Build the complete player-relative observation from authoritative state.
    pub fn from_state(state: &State) -> Self {
        let mut channels = [0.0f32; CHANNELS_LEN];
        let norm = (1.0 + MAX_ARMY_NORM).ln();
        let turn_progress = (state.round as f32 / MAX_TURNS as f32).min(1.0);
        let completed_plies = if state.winner == 0 {
            state
                .round
                .saturating_mul(2)
                .saturating_add(u32::from(state.current_player == 2))
        } else {
            state.cap_plies as u32
        };
        let plies_remaining = (state.cap_plies as u32).saturating_sub(completed_plies) as f32
            / (MAX_TURNS * 2) as f32;

        channels[8 * BOARD_SIZE..9 * BOARD_SIZE].fill(turn_progress);
        channels[9 * BOARD_SIZE..10 * BOARD_SIZE].fill(plies_remaining);

        for (i, tile) in state.tiles.iter().enumerate().take(BOARD_SIZE) {
            let ch = |c: usize| c * BOARD_SIZE + i;
            let own = tile.owner == state.current_player;
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
        }

        Self { channels }
    }

    /// Encode as little-endian f32 bytes for the neural network.
    pub fn encode(&self, out: &mut Vec<u8>) {
        encode_f32_slices(out, [&self.channels[..]]);
    }
}
