//! Game registration for Cartridge engine
//!
//! This crate provides a single initialization point for registering all
//! available games with the engine-core registry.
//!
//! # Usage
//!
//! ```rust
//! use engine_games::register_all_games;
//!
//! // Call once at startup - safe to call multiple times
//! register_all_games();
//! ```

use std::sync::Once;

pub mod manifest;

static INIT: Once = Once::new();

/// Register all available games with the engine-core registry.
///
/// This function uses `std::sync::Once` to ensure registration only
/// happens once, even if called multiple times. Safe to call from
/// multiple threads.
///
/// Currently registers:
/// - TicTacToe (`"tictactoe"`)
/// - Connect 4 (`"connect4"`)
/// - Othello (`"othello"`)
/// - Generals (`"generals_8x8"`)
pub fn register_all_games() {
    INIT.call_once(|| {
        games_tictactoe::register_tictactoe();
        games_connect4::register_connect4();
        games_othello::register_othello();
        games_generals::register_generals();
    });
}

// Re-export individual registration functions for advanced use cases
pub use games_connect4::register_connect4;
pub use games_generals::register_generals;
pub use games_othello::register_othello;
pub use games_tictactoe::register_tictactoe;

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{is_registered, list_registered_games};

    #[test]
    fn test_register_all_games() {
        register_all_games();

        assert!(is_registered("tictactoe"));
        assert!(is_registered("connect4"));
        assert!(is_registered("othello"));
        assert!(is_registered("generals_8x8"));
    }

    #[test]
    fn test_register_all_games_idempotent() {
        register_all_games();
        register_all_games();
        register_all_games();

        let games = list_registered_games();
        let tictactoe_count = games.iter().filter(|g| *g == "tictactoe").count();
        let connect4_count = games.iter().filter(|g| *g == "connect4").count();
        let othello_count = games.iter().filter(|g| *g == "othello").count();

        assert_eq!(tictactoe_count, 1);
        assert_eq!(connect4_count, 1);
        assert_eq!(othello_count, 1);
    }

    /// Every game's observation must be laid out as
    /// `[obs_channels * board_size][legal mask: num_actions][player one-hot: 2]`.
    ///
    /// Both halves matter downstream: the trainer reshapes the leading slice
    /// using `obs_channels`, and reads the player indicator at
    /// `legal_mask_offset + num_actions`. A game that declares an encoding
    /// inconsistent with its own obs_size would mistrain silently, so assert
    /// it here for every registered game rather than per-crate.
    #[test]
    fn test_observation_layout_invariants_hold_for_every_game() {
        register_all_games();

        for env_id in list_registered_games() {
            let meta = engine_core::create_game(&env_id)
                .unwrap_or_else(|| panic!("{env_id} registered but not constructible"))
                .metadata();
            let board_size = meta.board_size();

            assert!(
                meta.obs_channels > 0,
                "{env_id}: obs_channels must be declared (got 0)"
            );
            assert!(board_size > 0, "{env_id}: board_size must be non-zero");
            assert_eq!(
                meta.legal_mask_offset,
                meta.obs_channels * board_size,
                "{env_id}: legal mask must start immediately after the board planes \
                 (obs_channels={} * board_size={})",
                meta.obs_channels,
                board_size,
            );
            assert_eq!(
                meta.obs_size,
                meta.legal_mask_offset + meta.num_actions + 2,
                "{env_id}: obs_size must be planes + legal mask + 2-element player one-hot",
            );
        }
    }
}
