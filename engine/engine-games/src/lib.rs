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
pub fn register_all_games() {
    INIT.call_once(|| {
        games_tictactoe::register_tictactoe();
        games_connect4::register_connect4();
        games_othello::register_othello();
    });
}

// Re-export individual registration functions for advanced use cases
pub use games_connect4::register_connect4;
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
}
