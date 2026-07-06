//! Static game registry for compile-time game registration
//!
//! This module provides a thread-safe registry system that allows games to be
//! registered at compile-time and looked up at runtime by their env_id.

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

use tracing::warn;

use crate::erased::ErasedGame;

/// Factory function type for creating game instances
pub type GameFactory = fn() -> Box<dyn ErasedGame>;

/// Thread-safe registry mapping env_id to game factory functions
static REGISTRY: Lazy<Mutex<HashMap<String, GameFactory>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Register a game with the global registry
///
/// This function should typically be called from game crate initialization
/// or using the `register_game!` macro.
///
/// # Arguments
///
/// * `env_id` - Unique environment identifier (e.g., "tictactoe")
/// * `factory` - Function that creates new instances of the game
///
/// # Example
///
/// ```rust
/// # use engine_core::adapter::GameAdapter;
/// # use engine_core::erased::ErasedGame;
/// # use engine_core::registry::*;
/// # use engine_core::typed::{self, ActionSpace, Capabilities, DecodeError, EncodeError, EngineId, Game};
/// # use rand_chacha::ChaCha20Rng;
/// #
/// # #[derive(Debug)]
/// # struct MyGame;
/// # impl Game for MyGame {
/// #     type State = ();
/// #     type Action = ();
/// #     type Obs = ();
/// #
/// #     fn engine_id(&self) -> EngineId {
/// #         EngineId {
/// #             env_id: "example".into(),
/// #             build_id: "test".into(),
/// #         }
/// #     }
/// #
/// #     fn capabilities(&self) -> Capabilities {
/// #         Capabilities {
/// #             id: self.engine_id(),
/// #             encoding: typed::Encoding {
/// #                 state: "state".into(),
/// #                 action: "action".into(),
/// #                 obs: "obs".into(),
/// #                 schema_version: 1,
/// #             },
/// #             max_horizon: 1,
/// #             action_space: ActionSpace::Discrete(1),
/// #             preferred_batch: 1,
/// #         }
/// #     }
/// #
/// #     fn metadata(&self) -> engine_core::GameMetadata {
/// #         engine_core::GameMetadata::new("example", "Example")
/// #     }
/// #
/// #     fn reset(&mut self, _rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
/// #         ((), ())
/// #     }
/// #
/// #     fn step(
/// #         &mut self,
/// #         _state: &mut Self::State,
/// #         _action: Self::Action,
/// #         _rng: &mut ChaCha20Rng,
/// #     ) -> (Self::Obs, f32, bool, u64) {
/// #         ((), 0.0, true, 0)
/// #     }
/// #
/// #     fn encode_state(_state: &Self::State, _out: &mut Vec<u8>) -> Result<(), EncodeError> {
/// #         Ok(())
/// #     }
/// #
/// #     fn decode_state(_buf: &[u8]) -> Result<Self::State, DecodeError> {
/// #         Ok(())
/// #     }
/// #
/// #     fn encode_action(_action: &Self::Action, _out: &mut Vec<u8>) -> Result<(), EncodeError> {
/// #         Ok(())
/// #     }
/// #
/// #     fn decode_action(_buf: &[u8]) -> Result<Self::Action, DecodeError> {
/// #         Ok(())
/// #     }
/// #
/// #     fn encode_obs(_obs: &Self::Obs, _out: &mut Vec<u8>) -> Result<(), EncodeError> {
/// #         Ok(())
/// #     }
/// # }
///
/// fn my_game_factory() -> Box<dyn ErasedGame> {
///     Box::new(GameAdapter::new(MyGame))
/// }
///
/// register_game("my_game".to_string(), my_game_factory);
/// ```
pub fn register_game(env_id: String, factory: GameFactory) {
    let mut registry = REGISTRY.lock().unwrap();
    if registry.contains_key(&env_id) {
        warn!(env_id = %env_id, "Overriding existing game registration");
    }
    registry.insert(env_id, factory);
}

/// Create a new game instance by env_id
///
/// # Arguments
///
/// * `env_id` - Environment identifier to look up
///
/// # Returns
///
/// Returns `Some(game)` if the env_id is registered, `None` otherwise.
///
/// # Example
///
/// ```rust
/// # use engine_core::registry::*;
///
/// match create_game("tictactoe") {
///     Some(game) => {
///         println!("Created game: {}", game.engine_id().env_id);
///     }
///     None => {
///         println!("Game 'tictactoe' not found");
///     }
/// }
/// ```
pub fn create_game(env_id: &str) -> Option<Box<dyn ErasedGame>> {
    let registry = REGISTRY.lock().unwrap();
    match registry.get(env_id) {
        Some(factory) => Some(factory()),
        None => {
            warn!(env_id = %env_id, "Attempted to create unregistered game");
            None
        }
    }
}

/// Get list of all registered environment IDs
///
/// This is useful for debugging and listing available games.
///
/// # Returns
///
/// A vector of all registered env_id strings.
pub fn list_registered_games() -> Vec<String> {
    let registry = REGISTRY.lock().unwrap();
    registry.keys().cloned().collect()
}

/// Check if a game is registered
///
/// # Arguments
///
/// * `env_id` - Environment identifier to check
///
/// # Returns
///
/// `true` if the game is registered, `false` otherwise.
pub fn is_registered(env_id: &str) -> bool {
    let registry = REGISTRY.lock().unwrap();
    registry.contains_key(env_id)
}

/// Clear all registered games (mainly for testing)
///
/// This function removes all registered games from the registry.
/// It should primarily be used in test scenarios.
pub fn clear_registry() {
    let mut registry = REGISTRY.lock().unwrap();
    registry.clear();
}

/// Convenience macro for registering games
///
/// This macro simplifies the registration process by automatically creating
/// the factory function and calling register_game.
///
/// # Example
///
/// ```ignore
/// register_game!(TicTacToe, "tictactoe");
/// ```
#[macro_export]
macro_rules! register_game {
    ($game_type:ty, $env_id:expr) => {{
        fn factory() -> Box<dyn $crate::erased::ErasedGame> {
            Box::new($crate::adapter::GameAdapter::new(<$game_type>::default()))
        }
        $crate::registry::register_game($env_id.to_string(), factory);
    }};
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
