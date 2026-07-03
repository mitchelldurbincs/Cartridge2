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
mod tests {
    use super::*;
    use crate::adapter::GameAdapter;
    use crate::test_utils::REGISTRY_TEST_MUTEX;
    use crate::typed::{
        ActionSpace, Capabilities, DecodeError, EncodeError, Encoding, EngineId, Game,
    };
    use rand_chacha::ChaCha20Rng;

    // Test game implementation
    #[derive(Debug, Default)]
    // Similar dummy-game fixtures exist in typed.rs and context.rs tests;
    // each module keeps its own variant (different state/action shapes).
    struct TestGame {
        name: String,
    }

    impl TestGame {
        fn new(name: String) -> Self {
            Self { name }
        }
    }

    impl Game for TestGame {
        type State = u32;
        type Action = u8;
        type Obs = Vec<f32>;

        fn engine_id(&self) -> EngineId {
            EngineId {
                env_id: self.name.clone(),
                build_id: "0.1.0".to_string(),
            }
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities {
                id: self.engine_id(),
                encoding: Encoding {
                    state: "u32:v1".to_string(),
                    action: "u8:v1".to_string(),
                    obs: "f32_vec:v1".to_string(),
                    schema_version: 1,
                },
                max_horizon: 100,
                action_space: ActionSpace::Discrete(4),
                preferred_batch: 32,
            }
        }

        fn metadata(&self) -> crate::metadata::GameMetadata {
            crate::metadata::GameMetadata::new(&self.name, "Test Game")
                .with_board(2, 2)
                .with_actions(4)
                .with_observation(1, 0)
        }

        fn reset(&mut self, _rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
            (0, vec![0.0])
        }

        fn step(
            &mut self,
            state: &mut Self::State,
            action: Self::Action,
            _rng: &mut ChaCha20Rng,
        ) -> (Self::Obs, f32, bool, u64) {
            *state += action as u32;
            (vec![*state as f32], 1.0, *state >= 10, *state as u64)
        }

        fn encode_state(
            state: &Self::State,
            out: &mut Vec<u8>,
        ) -> Result<(), crate::typed::EncodeError> {
            out.extend_from_slice(&state.to_le_bytes());
            Ok(())
        }

        fn decode_state(buf: &[u8]) -> Result<Self::State, crate::typed::DecodeError> {
            if buf.len() != 4 {
                return Err(crate::typed::DecodeError::InvalidLength {
                    expected: 4,
                    actual: buf.len(),
                });
            }
            Ok(u32::from_le_bytes(buf.try_into().unwrap()))
        }

        fn encode_action(
            action: &Self::Action,
            out: &mut Vec<u8>,
        ) -> Result<(), crate::typed::EncodeError> {
            out.push(*action);
            Ok(())
        }

        fn decode_action(buf: &[u8]) -> Result<Self::Action, crate::typed::DecodeError> {
            if buf.len() != 1 {
                return Err(crate::typed::DecodeError::InvalidLength {
                    expected: 1,
                    actual: buf.len(),
                });
            }
            Ok(buf[0])
        }

        fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), crate::typed::EncodeError> {
            for &value in obs {
                out.extend_from_slice(&value.to_le_bytes());
            }
            Ok(())
        }
    }

    #[derive(Debug)]
    struct OverrideTestGame {
        env_id: &'static str,
        build_id: &'static str,
    }

    impl OverrideTestGame {
        const fn new(env_id: &'static str, build_id: &'static str) -> Self {
            Self { env_id, build_id }
        }
    }

    impl Game for OverrideTestGame {
        type State = u8;
        type Action = u8;
        type Obs = ();

        fn engine_id(&self) -> EngineId {
            EngineId {
                env_id: self.env_id.to_string(),
                build_id: self.build_id.to_string(),
            }
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities {
                id: self.engine_id(),
                encoding: Encoding {
                    state: "u8:v1".into(),
                    action: "u8:v1".into(),
                    obs: "unit".into(),
                    schema_version: 1,
                },
                max_horizon: 1,
                action_space: ActionSpace::Discrete(2),
                preferred_batch: 1,
            }
        }

        fn metadata(&self) -> crate::metadata::GameMetadata {
            crate::metadata::GameMetadata::new(self.env_id, "Override Test Game")
                .with_board(1, 1)
                .with_actions(2)
                .with_observation(0, 0)
        }

        fn reset(&mut self, _rng: &mut ChaCha20Rng, _hint: &[u8]) -> (Self::State, Self::Obs) {
            (0, ())
        }

        fn step(
            &mut self,
            state: &mut Self::State,
            action: Self::Action,
            _rng: &mut ChaCha20Rng,
        ) -> (Self::Obs, f32, bool, u64) {
            *state = state.wrapping_add(action);
            ((), 0.0, true, *state as u64)
        }

        fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
            out.push(*state);
            Ok(())
        }

        fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
            if buf.len() != 1 {
                return Err(DecodeError::InvalidLength {
                    expected: 1,
                    actual: buf.len(),
                });
            }
            Ok(buf[0])
        }

        fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
            out.push(*action);
            Ok(())
        }

        fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
            if buf.len() != 1 {
                return Err(DecodeError::InvalidLength {
                    expected: 1,
                    actual: buf.len(),
                });
            }
            Ok(buf[0])
        }

        fn encode_obs(_obs: &Self::Obs, _out: &mut Vec<u8>) -> Result<(), EncodeError> {
            Ok(())
        }
    }

    #[test]
    fn test_register_and_create_game() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        clear_registry();

        fn test_factory() -> Box<dyn ErasedGame> {
            Box::new(GameAdapter::new(TestGame::new("test_game".to_string())))
        }

        register_game("test_game".to_string(), test_factory);

        let game = create_game("test_game");
        assert!(game.is_some());

        let game = game.unwrap();
        assert_eq!(game.engine_id().env_id, "test_game");
    }

    #[test]
    fn test_create_nonexistent_game() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        clear_registry();

        let game = create_game("nonexistent");
        assert!(game.is_none());
    }

    #[test]
    fn test_list_registered_games() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        clear_registry();

        fn factory1() -> Box<dyn ErasedGame> {
            Box::new(GameAdapter::new(TestGame::new("game1".to_string())))
        }
        fn factory2() -> Box<dyn ErasedGame> {
            Box::new(GameAdapter::new(TestGame::new("game2".to_string())))
        }

        register_game("game1".to_string(), factory1);
        register_game("game2".to_string(), factory2);

        let mut games = list_registered_games();
        games.sort();

        assert_eq!(games, vec!["game1".to_string(), "game2".to_string()]);
    }

    #[test]
    fn test_is_registered() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        clear_registry();

        fn factory() -> Box<dyn ErasedGame> {
            Box::new(GameAdapter::new(TestGame::new(
                "registered_game".to_string(),
            )))
        }

        assert!(!is_registered("registered_game"));

        register_game("registered_game".to_string(), factory);
        assert!(is_registered("registered_game"));
        assert!(!is_registered("unregistered_game"));
    }

    #[test]
    fn test_clear_registry() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        clear_registry();

        fn factory() -> Box<dyn ErasedGame> {
            Box::new(GameAdapter::new(TestGame::new("temp_game".to_string())))
        }

        register_game("temp_game".to_string(), factory);
        assert!(is_registered("temp_game"));

        clear_registry();
        assert!(!is_registered("temp_game"));
        assert!(list_registered_games().is_empty());
    }

    #[test]
    fn test_register_game_overrides_existing_factory() {
        let _guard = REGISTRY_TEST_MUTEX.lock().unwrap();
        clear_registry();

        fn factory_old() -> Box<dyn ErasedGame> {
            Box::new(GameAdapter::new(OverrideTestGame::new(
                "override_env",
                "build_old",
            )))
        }

        fn factory_new() -> Box<dyn ErasedGame> {
            Box::new(GameAdapter::new(OverrideTestGame::new(
                "override_env",
                "build_new",
            )))
        }

        register_game("override_env".to_string(), factory_old);
        let initial_build = create_game("override_env")
            .expect("initial factory should produce a game")
            .engine_id()
            .build_id;
        assert_eq!(initial_build, "build_old");

        register_game("override_env".to_string(), factory_new);
        let updated_build = create_game("override_env")
            .expect("overridden factory should still produce a game")
            .engine_id()
            .build_id;
        assert_eq!(updated_build, "build_new");

        let registered = list_registered_games();
        assert_eq!(registered, vec!["override_env".to_string()]);
    }
}
