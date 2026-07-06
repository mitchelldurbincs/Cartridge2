//! Typed Game trait providing ergonomic interface for game developers
//!
//! This trait allows game implementations to work with strongly-typed state,
//! action, and observation types while maintaining compile-time type safety.

use crate::metadata::GameMetadata;
use rand_chacha::ChaCha20Rng;

/// Engine identification information
#[derive(Debug, Clone, PartialEq)]
pub struct EngineId {
    pub env_id: String,
    pub build_id: String,
}

/// Encoding format specifications
#[derive(Debug, Clone, PartialEq)]
pub struct Encoding {
    pub state: String,
    pub action: String,
    pub obs: String,
    pub schema_version: u32,
}

/// Action space variants
#[derive(Debug, Clone, PartialEq)]
pub enum ActionSpace {
    Discrete(u32),
    MultiDiscrete(Vec<u32>),
    Continuous {
        low: Vec<f32>,
        high: Vec<f32>,
        shape: Vec<u32>,
    },
}

/// Game capabilities and configuration
#[derive(Debug, Clone, PartialEq)]
pub struct Capabilities {
    pub id: EngineId,
    pub encoding: Encoding,
    pub max_horizon: u32,
    pub action_space: ActionSpace,
    pub preferred_batch: u32,
}

/// Main trait for game implementations
///
/// Games should implement this trait with their specific types for State, Action, and Obs.
/// The trait provides compile-time type safety while allowing conversion to the erased
/// interface for runtime polymorphism.
///
/// # Type Parameters
///
/// * `State` - Game state type, should be POD-like for efficient copying
/// * `Action` - Action type, should be small and Copy or compact
/// * `Obs` - Observation type, often contiguous arrays of f32
///
/// # Example
///
/// ```rust
/// # use engine_core::typed::*;
/// # use rand_chacha::ChaCha20Rng;
///
/// #[derive(Debug, Clone, Copy)]
/// struct TicTacToeState {
///     board: [u8; 9],
///     current_player: u8,
/// }
///
/// #[derive(Debug, Clone, Copy)]
/// enum TicTacToeAction {
///     Place(u8),
/// }
///
/// #[derive(Debug, Clone)]
/// struct TicTacToeObs {
///     board_view: [f32; 18],
///     legal_moves: [f32; 9],
/// }
///
/// #[derive(Debug)]
/// struct TicTacToe;
///
/// impl Game for TicTacToe {
///     type State = TicTacToeState;
///     type Action = TicTacToeAction;
///     type Obs = TicTacToeObs;
///     
///     // Implementation methods...
/// #   fn engine_id(&self) -> EngineId { todo!() }
/// #   fn capabilities(&self) -> Capabilities { todo!() }
/// #   fn metadata(&self) -> engine_core::GameMetadata { todo!() }
/// #   fn reset(&mut self, rng: &mut ChaCha20Rng, hint: &[u8]) -> (Self::State, Self::Obs) { todo!() }
/// #   fn step(
/// #       &mut self,
/// #       state: &mut Self::State,
/// #       action: Self::Action,
/// #       rng: &mut ChaCha20Rng,
/// #   ) -> (Self::Obs, f32, bool, u64) { todo!() }
/// #   fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> { todo!() }
/// #   fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> { todo!() }
/// #   fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> { todo!() }
/// #   fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> { todo!() }
/// #   fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError> { todo!() }
/// }
/// ```
pub trait Game: Send + Sync + std::fmt::Debug + 'static {
    /// Game state type - should be efficiently copyable
    type State: Send + Sync + 'static;

    /// Action type - should be small and Copy or compact
    type Action: Send + Sync + 'static;

    /// Observation type - often contiguous arrays of f32
    type Obs: Send + Sync + 'static;

    /// Get engine identification information
    fn engine_id(&self) -> EngineId;

    /// Get game capabilities and configuration
    fn capabilities(&self) -> Capabilities;

    /// Get game metadata for UI and configuration
    ///
    /// Returns display-oriented metadata about the game including board dimensions,
    /// player information, and observation format details needed by actors and trainers.
    fn metadata(&self) -> GameMetadata;

    /// Reset the game to initial state
    ///
    /// # Arguments
    ///
    /// * `rng` - Deterministic random number generator for reproducible resets
    /// * `hint` - Optional hint data for environment setup
    ///
    /// # Returns
    ///
    /// A tuple of (initial_state, initial_observation)
    fn reset(&mut self, rng: &mut ChaCha20Rng, hint: &[u8]) -> (Self::State, Self::Obs);

    /// Perform one simulation step
    ///
    /// # Arguments
    ///
    /// * `state` - Current game state (mutable for in-place updates)
    /// * `action` - Action to take
    /// * `rng` - Random number generator for stochastic elements
    ///
    /// # Returns
    ///
    /// A tuple of (observation, reward, done, info)
    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        rng: &mut ChaCha20Rng,
    ) -> (Self::Obs, f32, bool, u64);

    // Encoding/Decoding hooks for serialization

    /// Encode state to bytes
    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError>;

    /// Decode state from bytes
    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError>;

    /// Encode action to bytes
    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError>;

    /// Decode action from bytes
    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError>;

    /// Encode observation to bytes
    fn encode_obs(obs: &Self::Obs, out: &mut Vec<u8>) -> Result<(), EncodeError>;
}

/// Error type for encoding operations
#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("Failed to encode data: {0}")]
    SerializationError(String),
    #[error("Buffer too small, needed {needed} bytes but got {available}")]
    BufferTooSmall { needed: usize, available: usize },
    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Error type for decoding operations  
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("Failed to decode data: {0}")]
    DeserializationError(String),
    #[error("Invalid buffer length: expected {expected} but got {actual}")]
    InvalidLength { expected: usize, actual: usize },
    #[error("Corrupted data: {0}")]
    CorruptedData(String),
    #[error("Unsupported version: {version}")]
    UnsupportedVersion { version: u32 },
}

#[cfg(test)]
#[path = "typed_tests.rs"]
mod tests;
