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

    /// Every game currently bundled conforms to the **AlphaZero spatial
    /// observation profile**:
    /// `[obs_channels * board_size][legal mask: num_actions][player one-hot: 2]`.
    ///
    /// Both halves matter downstream: the trainer reshapes the leading slice
    /// using `obs_channels`, and reads the player indicator at
    /// `legal_mask_offset + num_actions`. A game that declares an encoding
    /// inconsistent with its own obs_size would mistrain silently, so assert
    /// it here for every registered game rather than per-crate.
    ///
    /// **This is a profile the current trainer requires, not a property of
    /// `Game`.** It holds for every game in the registry today because every
    /// game today is a two-player perfect-information board game trained by
    /// AlphaZero. An environment that is not — a vector/scalar observation, a
    /// recurrent or fog-of-war encoding, a simultaneous-turn adapter, anything
    /// driven by PPO — will legitimately violate it, and the answer then is to
    /// scope this assertion to the games claiming the profile rather than to
    /// force the new environment into a board-shaped layout. Deliberately not
    /// building that opt-out yet: there is no second algorithm to design it
    /// against, and guessing the seam before one exists is how it ends up
    /// fitting neither.
    #[test]
    fn test_bundled_games_conform_to_the_alphazero_spatial_profile() {
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

    /// `Game::view` is what every out-of-engine consumer (the web server, the
    /// evaluation harness' position dump) uses instead of decoding state bytes
    /// itself, so a game whose view is the wrong shape, or is a constant stub,
    /// breaks them silently rather than failing to compile. Assert both here,
    /// registry-wide, so a newly added game is covered without remembering to.
    #[test]
    fn test_every_game_projects_a_well_formed_view() {
        register_all_games();

        for env_id in list_registered_games() {
            let mut ctx = engine_core::EngineContext::new(&env_id)
                .unwrap_or_else(|| panic!("{env_id} registered but not constructible"));
            let meta = ctx.metadata();
            let reset = ctx.reset(7, &[]).expect("reset");

            let view = ctx
                .view(&reset.state)
                .expect("view of a freshly reset state");

            assert_eq!(
                view.cells.len(),
                meta.board_size(),
                "{env_id}: view must have one cell per board position",
            );
            assert!(
                (1..=meta.player_count as u8).contains(&view.current_player),
                "{env_id}: current_player {} out of range",
                view.current_player,
            );
            assert_eq!(view.winner, 0, "{env_id}: a fresh game cannot be decided");
            for (i, cell) in view.cells.iter().enumerate() {
                assert!(
                    cell.owner as usize <= meta.player_count,
                    "{env_id}: cell {i} owner {} is not a player or neutral",
                    cell.owner,
                );
            }

            // A constant stub would satisfy everything above. Playing a legal
            // move must move the view.
            let action = meta
                .extract_legal_moves(&reset.obs)
                .first()
                .copied()
                .unwrap_or_else(|| panic!("{env_id}: fresh game has no legal move"));
            let step = ctx
                .step(&reset.state, &(action as u32).to_le_bytes())
                .expect("step");
            let after = ctx.view(&step.state).expect("view after a move");

            assert_ne!(
                view, after,
                "{env_id}: view did not change after a legal move — is it a stub?",
            );
        }
    }
}
