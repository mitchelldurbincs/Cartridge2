//! MCTS-based policy using neural network evaluation
//!
//! This module provides a policy that uses Monte Carlo Tree Search with
//! an ONNX neural network evaluator to select actions.

use anyhow::{anyhow, Result};
use engine_core::{EngineContext, ErasedTimestep};
use mcts::{
    run_mcts, MctsConfig, SearchResult, SearchStats, SharedOnnxEvaluator, UniformEvaluator,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tracing::{debug, warn};

/// Result from MCTS policy selection, including the policy training target.
pub struct MctsPolicyResult {
    /// Selected action as bytes
    pub action: Vec<u8>,
    /// Policy distribution from MCTS (for training).
    ///
    /// The raw root visit distribution, unaffected by the temperature
    /// schedule below — that only decides which action gets played.
    pub policy: Vec<f32>,
    /// Performance statistics from the MCTS search
    pub stats: SearchStats,
}

/// MCTS-based policy that uses neural network for evaluation
pub struct MctsPolicy {
    /// Environment ID for creating simulation contexts
    env_id: String,
    /// MCTS configuration
    config: MctsConfig,
    /// Base temperature (used for early moves)
    base_temperature: f32,
    /// Temperature for late-game moves (after threshold)
    late_temperature: f32,
    /// Move number after which to use late_temperature
    temp_threshold: u32,
    /// Number of actions in the game
    num_actions: usize,
    /// Observation size for the neural network
    obs_size: usize,
    /// Shared evaluator that can be hot-swapped
    evaluator: Arc<RwLock<Option<SharedOnnxEvaluator>>>,
    /// RNG for action sampling
    rng: ChaCha20Rng,
    /// Reusable simulation context for MCTS (avoids repeated registry lookups)
    sim_ctx: Option<EngineContext>,
}

impl std::fmt::Debug for MctsPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MctsPolicy")
            .field("env_id", &self.env_id)
            .field("num_actions", &self.num_actions)
            .field("obs_size", &self.obs_size)
            .field("temp_threshold", &self.temp_threshold)
            .field(
                "has_model",
                &self.evaluator.read().map(|e| e.is_some()).unwrap_or(false),
            )
            .finish()
    }
}

impl MctsPolicy {
    /// Create a new MCTS policy without a model loaded
    pub fn new(env_id: String, num_actions: usize, obs_size: usize) -> Self {
        let config = MctsConfig::for_training();
        let base_temp = config.temperature;
        Self {
            env_id,
            base_temperature: base_temp,
            // Scheduling is disabled until the caller supplies an explicit
            // threshold and late temperature.
            late_temperature: base_temp,
            temp_threshold: 0, // Disabled by default (0 = no threshold)
            config,
            num_actions,
            obs_size,
            evaluator: Arc::new(RwLock::new(None)),
            rng: ChaCha20Rng::from_entropy(),
            sim_ctx: None,
        }
    }

    /// Create with a specific seed for determinism (used in tests)
    #[allow(dead_code)]
    pub fn with_seed(env_id: String, num_actions: usize, obs_size: usize, seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            ..Self::new(env_id, num_actions, obs_size)
        }
    }

    /// Set the MCTS configuration
    pub fn with_config(mut self, config: MctsConfig) -> Self {
        self.base_temperature = config.temperature;
        self.config = config;
        self
    }

    /// Set the temperature schedule for move-dependent exploration
    ///
    /// After `threshold` moves, temperature drops from base to `late_temp`.
    /// This encourages exploration early and exploitation late in games.
    ///
    /// Set threshold to 0 to disable (always use base temperature).
    pub fn with_temp_schedule(mut self, threshold: u32, late_temp: f32) -> Self {
        self.temp_threshold = threshold;
        self.late_temperature = late_temp;
        self
    }

    /// Check if a model is loaded (used for debugging/logging)
    #[allow(dead_code)]
    pub fn has_model(&self) -> bool {
        self.evaluator
            .read()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Get shared evaluator reference for hot-reloading
    pub fn evaluator_ref(&self) -> Arc<RwLock<Option<SharedOnnxEvaluator>>> {
        Arc::clone(&self.evaluator)
    }

    /// Select an action using MCTS
    ///
    /// # Arguments
    /// * `state` - Current game state bytes
    /// * `timestep` - Current algorithm-neutral timestep; the MCTS cartridge
    ///   validates and extracts its single active-agent observation (including
    ///   the authoritative legal-action mask)
    /// * `move_number` - Current move number in the game (0-indexed)
    ///
    /// # Returns
    /// `MctsPolicyResult` with action, policy distribution, and value estimate
    pub fn select_action(
        &mut self,
        state: &[u8],
        timestep: &ErasedTimestep,
        move_number: u32,
    ) -> Result<MctsPolicyResult> {
        // Snapshot the current model and release the reload lock before the
        // search. The clone pins one model generation for this entire search
        // even if a hot reload lands mid-way.
        let evaluator = self
            .evaluator
            .read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?
            .clone();

        if self.sim_ctx.is_none() {
            self.sim_ctx = Some(EngineContext::new(&self.env_id).map_err(|error| {
                anyhow!("environment '{}' is unavailable: {error}", self.env_id)
            })?);
        }
        let sim_ctx = self
            .sim_ctx
            .as_mut()
            .expect("simulation context was initialized above");

        // Apply temperature schedule: use lower temperature for late-game moves
        let mut config = self.config.clone();
        if self.temp_threshold > 0 && move_number >= self.temp_threshold {
            config.temperature = self.late_temperature;
        }

        // Run MCTS search with timing
        let mcts_start = Instant::now();
        let result: SearchResult = match &evaluator {
            Some(model) => run_mcts(
                sim_ctx,
                model,
                config,
                state.to_vec(),
                timestep.clone(),
                &mut self.rng,
            ),
            None => {
                // Root collection has no RunHead model yet. Search with
                // uniform priors so stored policy targets are real visit
                // distributions instead of uniform placeholders.
                debug!("No model loaded, running MCTS with the uniform evaluator");
                run_mcts(
                    sim_ctx,
                    &UniformEvaluator::new(),
                    config,
                    state.to_vec(),
                    timestep.clone(),
                    &mut self.rng,
                )
            }
        }
        .map_err(|e| anyhow!("MCTS search failed: {}", e))?;
        let mcts_elapsed_ms = mcts_start.elapsed().as_millis();

        // Log detailed stats at debug level
        let stats = &result.stats;
        debug!(
            action = result.action,
            value = result.value,
            simulations = result.simulations,
            move_number = move_number,
            mcts_ms = mcts_elapsed_ms,
            inference_ms = stats.inference_time_us as f64 / 1000.0,
            expansion_ms = stats.expansion_time_us as f64 / 1000.0,
            game_steps = stats.game_steps,
            num_batches = stats.num_batches,
            "MCTS selected action"
        );

        // Warn if MCTS is taking too long (> 2 seconds indicates a problem)
        if mcts_elapsed_ms > 2000 {
            warn!(
                mcts_ms = mcts_elapsed_ms,
                inference_pct =
                    (stats.inference_time_us as f64 / stats.total_time_us as f64 * 100.0) as u32,
                expansion_pct =
                    (stats.expansion_time_us as f64 / stats.total_time_us as f64 * 100.0) as u32,
                game_steps = stats.game_steps,
                num_batches = stats.num_batches,
                avg_batch_size = stats
                    .total_evals
                    .checked_div(stats.num_batches)
                    .unwrap_or(0),
                "MCTS step took >2s - performance issue detected"
            );
        }

        // Convert action index to bytes (u32 little-endian)
        let action_bytes = result.action.to_le_bytes().to_vec();

        Ok(MctsPolicyResult {
            action: action_bytes,
            policy: result.policy,
            stats: result.stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() {
        engine_games::register_all_environments();
    }

    #[test]
    fn test_mcts_policy_creation() {
        let policy = MctsPolicy::new("tictactoe".into(), 9, 29);
        assert!(!policy.has_model());
        assert_eq!(policy.num_actions, 9);
    }

    #[test]
    fn test_mcts_policy_without_model_runs_uniform_evaluator_mcts() {
        setup();

        let mut policy = MctsPolicy::with_seed("tictactoe".into(), 9, 29, 42);
        let mut context = EngineContext::new("tictactoe").unwrap();
        let reset = context.reset(42, &[]).unwrap();

        // Without a model, MCTS still runs, backed by the uniform evaluator.
        let result = policy
            .select_action(&reset.state, &reset.timestep, 0)
            .unwrap();
        assert_eq!(result.action.len(), 4); // u32

        // Action should be in valid range
        let action = u32::from_le_bytes(result.action.try_into().unwrap());
        assert!(action < 9);

        // Policy is a real visit distribution over the 9 legal opening moves.
        let sum: f32 = result.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(result.stats.total_evals > 0, "search evaluated no leaves");
        assert!(result.stats.game_steps > 0, "search never stepped the game");
    }

    #[test]
    fn test_mcts_without_model_assigns_no_mass_to_occupied_cells() {
        setup();

        let mut policy = MctsPolicy::with_seed("tictactoe".into(), 9, 29, 42);
        let mut context = EngineContext::new("tictactoe").unwrap();
        let reset = context.reset(42, &[]).unwrap();
        // Occupy the center so action 4 becomes illegal.
        let step = context.step(&reset.state, &4u32.to_le_bytes()).unwrap();

        let result = policy
            .select_action(&step.state, &step.timestep, 1)
            .unwrap();

        let action = u32::from_le_bytes(result.action.try_into().unwrap());
        assert_ne!(action, 4, "selected an occupied cell");
        assert_eq!(result.policy[4], 0.0, "policy mass on an occupied cell");
        let sum: f32 = result.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ========================================
    // Temperature scheduling tests
    // ========================================

    #[test]
    fn test_temperature_scheduling_disabled() {
        // With temp_threshold = 0, scheduling is disabled
        let policy = MctsPolicy::new("tictactoe".into(), 9, 29).with_temp_schedule(0, 0.1);

        // Threshold of 0 means temperature scheduling is disabled
        assert_eq!(policy.temp_threshold, 0);
    }

    #[test]
    fn test_temperature_scheduling_configuration() {
        let policy = MctsPolicy::new("tictactoe".into(), 9, 29).with_temp_schedule(10, 0.2);

        assert_eq!(policy.temp_threshold, 10);
        assert!((policy.late_temperature - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_with_config_updates_base_temperature() {
        let config = MctsConfig::for_training().with_temperature(0.5);

        let policy = MctsPolicy::new("tictactoe".into(), 9, 29).with_config(config);

        assert!((policy.base_temperature - 0.5).abs() < 1e-6);
    }

    // ========================================
    // Action byte encoding tests
    // ========================================

    #[test]
    fn test_action_roundtrip_encoding() {
        // Test that we can encode and decode all possible TicTacToe actions
        for action in 0..9u32 {
            let bytes = action.to_le_bytes().to_vec();
            let decoded = u32::from_le_bytes(bytes.try_into().unwrap());
            assert_eq!(action, decoded);
        }
    }

    // ========================================
    // Debug trait and evaluator reference tests
    // ========================================

    #[test]
    fn test_debug_implementation() {
        let policy = MctsPolicy::new("tictactoe".into(), 9, 29);
        let debug_str = format!("{:?}", policy);

        // Should contain key fields
        assert!(debug_str.contains("MctsPolicy"));
        assert!(debug_str.contains("tictactoe"));
        assert!(debug_str.contains("num_actions: 9"));
        assert!(debug_str.contains("has_model: false"));
    }

    #[test]
    fn test_evaluator_ref_returns_valid_arc() {
        let policy = MctsPolicy::new("tictactoe".into(), 9, 29);
        let eval_ref = policy.evaluator_ref();

        // Should be able to read the evaluator
        let guard = eval_ref.read().unwrap();
        assert!(guard.is_none()); // No model loaded initially
    }

    #[test]
    fn test_has_model_returns_false_initially() {
        let policy = MctsPolicy::new("tictactoe".into(), 9, 29);
        assert!(!policy.has_model());
    }

    // ========================================
    // Connect4 tests (different game configuration)
    // ========================================

    #[test]
    fn test_mcts_policy_connect4_configuration() {
        // Connect4 has 7 columns (actions) and different obs size
        let policy = MctsPolicy::new("connect4".into(), 7, 127);

        assert_eq!(policy.num_actions, 7);
        assert_eq!(policy.obs_size, 127);
        assert_eq!(policy.env_id, "connect4");
    }

    #[test]
    fn test_uniform_mcts_connect4() {
        setup();

        let mut policy = MctsPolicy::with_seed("connect4".into(), 7, 127, 42);
        let mut context = EngineContext::new("connect4").unwrap();
        let reset = context.reset(42, &[]).unwrap();

        let result = policy
            .select_action(&reset.state, &reset.timestep, 0)
            .unwrap();

        let action = u32::from_le_bytes(result.action.try_into().unwrap());
        assert!(action < 7, "Connect4 action should be 0-6, got {}", action);

        // Policy should sum to 1
        let sum: f32 = result.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ========================================
    // Seeded determinism tests
    // ========================================

    #[test]
    fn test_seeded_policy_is_deterministic() {
        setup();

        // Create two policies with the same seed
        let mut policy1 = MctsPolicy::with_seed("tictactoe".into(), 9, 29, 12345);
        let mut policy2 = MctsPolicy::with_seed("tictactoe".into(), 9, 29, 12345);
        let mut context = EngineContext::new("tictactoe").unwrap();
        let reset = context.reset(7, &[]).unwrap();

        // They should produce the same actions and policy targets
        for move_number in 0..5 {
            let r1 = policy1
                .select_action(&reset.state, &reset.timestep, move_number)
                .unwrap();
            let r2 = policy2
                .select_action(&reset.state, &reset.timestep, move_number)
                .unwrap();
            assert_eq!(r1.action, r2.action);
            assert_eq!(r1.policy, r2.policy);
        }
    }
}
