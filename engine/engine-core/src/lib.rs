//! Core traits and types for the Cartridge game engine
//!
//! This crate provides the fundamental abstractions for game simulation:
//! - `Game`: Typed trait for ergonomic game development
//! - `ErasedGame`: Runtime interface that works only with bytes
//! - `GameAdapter`: Automatic conversion from typed to erased interface
//! - `Registry`: Static registration system for games
//! - `EngineContext`: High-level API for running game simulations

pub mod adapter;
pub mod board_game;
pub mod context;
pub mod erased;
pub mod game_utils;
pub mod legal_mask;
pub mod metadata;
pub mod registry;
pub mod typed;

// Re-export main types for convenience
pub use adapter::GameAdapter;
pub use board_game::TwoPlayerObs;
pub use context::{EngineContext, ResetResult, StepResult};
pub use erased::ErasedGame;
pub use legal_mask::LegalMask;
pub use metadata::GameMetadata;
pub use registry::{
    clear_registry, create_game, is_registered, list_registered_games, register_game, GameFactory,
};
pub use typed::{ActionSpace, Game};

/// Test utilities (internal use only)
#[cfg(test)]
pub(crate) mod test_utils {
    use once_cell::sync::Lazy;
    use std::sync::Mutex;

    /// Global mutex to serialize all registry-dependent tests
    pub static REGISTRY_TEST_MUTEX: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));
}
