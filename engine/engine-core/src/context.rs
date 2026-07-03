//! Engine context providing a high-level API for game simulation
//!
//! This module provides `EngineContext`, a convenient wrapper for running
//! game simulations without dealing with raw bytes and registry lookups.

use crate::erased::{ErasedGame, ErasedGameError};
use crate::metadata::GameMetadata;
use crate::registry::create_game;
use crate::typed::{ActionSpace, Capabilities, EngineId};

/// High-level context for running game simulations
///
/// `EngineContext` wraps an `ErasedGame` instance and provides convenient
/// methods for reset/step operations while managing internal buffers.
///
/// # Example
///
/// ```rust,ignore
/// use engine_core::context::EngineContext;
///
/// // Create context for TicTacToe
/// let mut ctx = EngineContext::new("tictactoe").expect("game not found");
///
/// // Reset and get initial state
/// let reset = ctx.reset(42, &[]).unwrap();
///
/// // Take a step with an action
/// let action = 4u32.to_le_bytes().to_vec(); // Center position
/// let result = ctx.step(&reset.state, &action).unwrap();
/// println!("Reward: {}, Done: {}", result.reward, result.done);
/// ```
#[derive(Debug)]
pub struct EngineContext {
    game: Box<dyn ErasedGame>,
    state_buf: Vec<u8>,
    obs_buf: Vec<u8>,
}

/// Result of a step operation
#[derive(Debug, Clone)]
pub struct StepResult {
    /// New state after the action
    pub state: Vec<u8>,
    /// Observation for the new state
    pub obs: Vec<u8>,
    /// Reward received from this step
    pub reward: f32,
    /// Whether the episode has terminated
    pub done: bool,
    /// Additional packed info bits
    pub info: u64,
}

/// Result of a reset operation
#[derive(Debug, Clone)]
pub struct ResetResult {
    /// Initial state
    pub state: Vec<u8>,
    /// Initial observation
    pub obs: Vec<u8>,
}

impl EngineContext {
    /// Create a new engine context for the specified game
    ///
    /// # Arguments
    ///
    /// * `env_id` - Environment identifier (e.g., "tictactoe")
    ///
    /// # Returns
    ///
    /// Returns `Some(context)` if the game is registered, `None` otherwise.
    pub fn new(env_id: &str) -> Option<Self> {
        let game = create_game(env_id)?;
        Some(Self {
            game,
            state_buf: Vec::with_capacity(256),
            obs_buf: Vec::with_capacity(512),
        })
    }

    /// Create a new engine context from an existing game instance
    pub fn from_game(game: Box<dyn ErasedGame>) -> Self {
        Self {
            game,
            state_buf: Vec::with_capacity(256),
            obs_buf: Vec::with_capacity(512),
        }
    }

    /// Get the engine ID for this game
    pub fn engine_id(&self) -> EngineId {
        self.game.engine_id()
    }

    /// Get the capabilities of this game
    pub fn capabilities(&self) -> Capabilities {
        self.game.capabilities()
    }

    /// Get the action space for this game
    pub fn action_space(&self) -> ActionSpace {
        self.game.capabilities().action_space
    }

    /// Get the metadata for this game
    pub fn metadata(&self) -> GameMetadata {
        self.game.metadata()
    }

    /// Reset the game to initial state
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for deterministic reset
    /// * `hint` - Optional hint data for environment setup
    ///
    /// # Returns
    ///
    /// Returns `ResetResult` containing initial state and observation.
    pub fn reset(&mut self, seed: u64, hint: &[u8]) -> Result<ResetResult, ErasedGameError> {
        self.state_buf.clear();
        self.obs_buf.clear();

        self.game
            .reset(seed, hint, &mut self.state_buf, &mut self.obs_buf)?;

        Ok(ResetResult {
            state: self.state_buf.clone(),
            obs: self.obs_buf.clone(),
        })
    }

    /// Perform one simulation step
    ///
    /// # Arguments
    ///
    /// * `state` - Current state encoded as bytes
    /// * `action` - Action to take encoded as bytes
    ///
    /// # Returns
    ///
    /// Returns `StepResult` containing new state, observation, reward, done flag, and info.
    pub fn step(&mut self, state: &[u8], action: &[u8]) -> Result<StepResult, ErasedGameError> {
        self.state_buf.clear();
        self.obs_buf.clear();

        let (reward, done, info) =
            self.game
                .step(state, action, &mut self.state_buf, &mut self.obs_buf)?;

        Ok(StepResult {
            state: self.state_buf.clone(),
            obs: self.obs_buf.clone(),
            reward,
            done,
            info,
        })
    }

    /// Sample a random action from the action space
    ///
    /// Uses the provided RNG to sample a valid action for this game's action space.
    pub fn sample_random_action(&self, rng: &mut rand_chacha::ChaCha20Rng) -> Vec<u8> {
        use rand::Rng;

        let mut out = Vec::new();
        let caps = self.game.capabilities();

        match &caps.action_space {
            ActionSpace::Discrete(n) => {
                let action = rng.gen_range(0..*n);
                out.extend_from_slice(&action.to_le_bytes());
            }
            ActionSpace::MultiDiscrete(nvec) => {
                for &n in nvec {
                    let action = rng.gen_range(0..n);
                    out.extend_from_slice(&action.to_le_bytes());
                }
            }
            ActionSpace::Continuous { low, high, .. } => {
                for i in 0..low.len() {
                    let value = rng.gen_range(low[i]..high[i]);
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
        }

        out
    }

    /// Get access to the underlying erased game
    pub fn game(&self) -> &dyn ErasedGame {
        self.game.as_ref()
    }

    /// Get mutable access to the underlying erased game
    pub fn game_mut(&mut self) -> &mut dyn ErasedGame {
        self.game.as_mut()
    }

    // =========================================================================
    // Zero-copy API for hot paths
    // =========================================================================

    /// Reset the game to initial state, writing directly to caller-provided buffers.
    ///
    /// This is a zero-allocation alternative to `reset()` for use in hot paths
    /// like training loops. The caller provides buffers that are cleared and filled
    /// with the new state and observation.
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed for deterministic reset
    /// * `hint` - Optional hint data for environment setup
    /// * `state` - Buffer to receive the initial state (cleared first)
    /// * `obs` - Buffer to receive the initial observation (cleared first)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut state = Vec::with_capacity(64);
    /// let mut obs = Vec::with_capacity(512);
    ///
    /// ctx.reset_into(42, &[], &mut state, &mut obs)?;
    /// // state and obs now contain initial game state - no allocations after warmup
    /// ```
    pub fn reset_into(
        &mut self,
        seed: u64,
        hint: &[u8],
        state: &mut Vec<u8>,
        obs: &mut Vec<u8>,
    ) -> Result<(), ErasedGameError> {
        state.clear();
        obs.clear();
        self.game.reset(seed, hint, state, obs)
    }

    /// Perform one simulation step, writing directly to caller-provided buffers.
    ///
    /// This is a zero-allocation alternative to `step()` for use in hot paths
    /// like training loops. The caller provides buffers that are cleared and filled
    /// with the new state and observation.
    ///
    /// # Arguments
    ///
    /// * `state` - Current state encoded as bytes
    /// * `action` - Action to take encoded as bytes
    /// * `state_out` - Buffer to receive the new state (cleared first)
    /// * `obs_out` - Buffer to receive the new observation (cleared first)
    ///
    /// # Returns
    ///
    /// Returns `(reward, done, info)` tuple on success.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Allocate buffers once
    /// let mut state = Vec::with_capacity(64);
    /// let mut obs = Vec::with_capacity(512);
    /// let mut next_state = Vec::with_capacity(64);
    /// let mut next_obs = Vec::with_capacity(512);
    ///
    /// ctx.reset_into(seed, &[], &mut state, &mut obs)?;
    ///
    /// loop {
    ///     let (reward, done, info) = ctx.step_into(
    ///         &state, &action, &mut next_state, &mut next_obs
    ///     )?;
    ///     std::mem::swap(&mut state, &mut next_state);
    ///     std::mem::swap(&mut obs, &mut next_obs);
    ///     if done { break; }
    /// }
    /// ```
    pub fn step_into(
        &mut self,
        state: &[u8],
        action: &[u8],
        state_out: &mut Vec<u8>,
        obs_out: &mut Vec<u8>,
    ) -> Result<(f32, bool, u64), ErasedGameError> {
        state_out.clear();
        obs_out.clear();
        self.game.step(state, action, state_out, obs_out)
    }
}

#[cfg(test)]
#[path = "context_tests.rs"]
mod tests;
