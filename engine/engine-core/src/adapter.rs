//! Adapter layer converting typed games to erased interface
//!
//! This module provides the `GameAdapter` struct that automatically converts
//! any typed `Game` implementation to the `ErasedGame` interface, handling
//! all encoding/decoding and random number generation management.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::erased::{ErasedGame, ErasedGameError};
use crate::metadata::GameMetadata;
use crate::typed::{Capabilities, EngineId, Game};

/// Adapter that converts typed games to erased interface
///
/// This struct wraps any typed `Game` implementation and provides the `ErasedGame`
/// interface by handling all encoding/decoding operations and managing the
/// random number generator state.
///
/// The adapter maintains its own RNG instance that gets re-seeded on each reset,
/// ensuring deterministic behavior while providing the stateless, bytes-only
/// interface expected by the registry and `EngineContext`.
///
/// # Example
///
/// ```rust
/// # use engine_core::adapter::GameAdapter;
/// # use engine_core::erased::ErasedGame;
/// # use engine_core::typed::{ActionSpace, Capabilities, DecodeError, EncodeError, EngineId, Game};
/// # use rand_chacha::ChaCha20Rng;
/// #
/// # #[derive(Debug, Default)]
/// # struct MyGame;
/// # impl Game for MyGame {
/// #     type State = u32;
/// #     type Action = u8;
/// #     type Obs = Vec<f32>;
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
/// #             encoding: engine_core::typed::Encoding {
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
/// #         (0, vec![])
/// #     }
/// #
/// #     fn step(
/// #         &mut self,
/// #         state: &mut Self::State,
/// #         _action: Self::Action,
/// #         _rng: &mut ChaCha20Rng,
/// #     ) -> (Self::Obs, f32, bool, u64) {
/// #         *state += 1;
/// #         (vec![], 0.0, true, *state as u64)
/// #     }
/// #
/// #     fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
/// #         out.extend_from_slice(&state.to_le_bytes());
/// #         Ok(())
/// #     }
/// #
/// #     fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
/// #         let mut arr = [0_u8; 4];
/// #         arr.copy_from_slice(&buf[..4]);
/// #         Ok(u32::from_le_bytes(arr))
/// #     }
/// #
/// #     fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
/// #         out.push(*action);
/// #         Ok(())
/// #     }
/// #
/// #     fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
/// #         Ok(buf[0])
/// #     }
/// #
/// #     fn encode_obs(_obs: &Self::Obs, _out: &mut Vec<u8>) -> Result<(), EncodeError> {
/// #         Ok(())
/// #     }
/// # }
///
/// let typed_game = MyGame::default();
/// let mut erased_game: Box<dyn ErasedGame> = Box::new(GameAdapter::new(typed_game));
///
/// // Now you can use the erased interface
/// let engine_id = erased_game.engine_id();
/// println!("Game: {}", engine_id.env_id);
/// ```
#[derive(Debug)]
pub struct GameAdapter<T: Game> {
    game: T,
    rng: ChaCha20Rng,
}

impl<T: Game> GameAdapter<T> {
    /// Create a new adapter wrapping the given game
    ///
    /// The adapter starts with a default-seeded RNG that will be re-seeded
    /// on the first reset call.
    pub fn new(game: T) -> Self {
        Self {
            game,
            rng: ChaCha20Rng::seed_from_u64(0), // Will be re-seeded on reset
        }
    }

    /// Get a reference to the underlying game
    pub fn game(&self) -> &T {
        &self.game
    }

    /// Get a mutable reference to the underlying game
    pub fn game_mut(&mut self) -> &mut T {
        &mut self.game
    }

    /// Consume the adapter and return the underlying game
    pub fn into_inner(self) -> T {
        self.game
    }
}

impl<T: Game> ErasedGame for GameAdapter<T> {
    fn engine_id(&self) -> EngineId {
        self.game.engine_id()
    }

    fn capabilities(&self) -> Capabilities {
        self.game.capabilities()
    }

    fn metadata(&self) -> GameMetadata {
        self.game.metadata()
    }

    fn reset(
        &mut self,
        seed: u64,
        hint: &[u8],
        out_state: &mut Vec<u8>,
        out_obs: &mut Vec<u8>,
    ) -> Result<(), ErasedGameError> {
        // Re-seed the RNG for deterministic behavior
        self.rng = ChaCha20Rng::seed_from_u64(seed);

        // Clear output buffers
        out_state.clear();
        out_obs.clear();

        // Call the typed reset method
        let (state, obs) = self.game.reset(&mut self.rng, hint);

        // Encode the results
        T::encode_state(&state, out_state).map_err(|e| ErasedGameError::Encoding(e.to_string()))?;

        T::encode_obs(&obs, out_obs).map_err(|e| ErasedGameError::Encoding(e.to_string()))?;

        Ok(())
    }

    fn step(
        &mut self,
        state: &[u8],
        action: &[u8],
        out_state: &mut Vec<u8>,
        out_obs: &mut Vec<u8>,
    ) -> Result<(f32, bool, u64), ErasedGameError> {
        // Clear output buffers
        out_state.clear();
        out_obs.clear();

        // Decode the inputs
        let mut state =
            T::decode_state(state).map_err(|e| ErasedGameError::Decoding(e.to_string()))?;

        let action =
            T::decode_action(action).map_err(|e| ErasedGameError::Decoding(e.to_string()))?;

        // Call the typed step method
        let (obs, reward, done, info) = self.game.step(&mut state, action, &mut self.rng);

        // Encode the results
        T::encode_state(&state, out_state).map_err(|e| ErasedGameError::Encoding(e.to_string()))?;

        T::encode_obs(&obs, out_obs).map_err(|e| ErasedGameError::Encoding(e.to_string()))?;

        Ok((reward, done, info))
    }
}

#[cfg(test)]
#[path = "adapter_tests.rs"]
mod tests;
