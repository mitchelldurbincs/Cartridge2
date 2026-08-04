//! MCTS configuration parameters.

use thiserror::Error;

/// Configuration errors rejected before a search allocates or evaluates a tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MctsConfigError {
    #[error("num_simulations must be greater than zero")]
    ZeroSimulations,
    #[error("c_puct must be finite and non-negative")]
    InvalidCPuct,
    #[error("dirichlet_alpha must be finite and non-negative")]
    InvalidDirichletAlpha,
    #[error("dirichlet_epsilon must be finite and between zero and one")]
    InvalidDirichletEpsilon,
    #[error("dirichlet_alpha and dirichlet_epsilon must both be zero to disable noise")]
    InconsistentDirichletNoise,
    #[error("temperature must be finite and non-negative")]
    InvalidTemperature,
    #[error("virtual_loss must be finite and non-negative")]
    InvalidVirtualLoss,
    #[error("eval_batch_size must be greater than zero")]
    ZeroEvalBatchSize,
}

/// Configuration for Monte Carlo Tree Search.
#[derive(Debug, Clone)]
pub struct MctsConfig {
    /// Number of simulations to run per search.
    pub num_simulations: u32,

    /// Exploration constant for UCB formula (c_puct in AlphaZero).
    /// Higher values encourage exploration, lower values favor exploitation.
    /// Typical range: 1.0 - 4.0, AlphaZero uses ~1.25
    pub c_puct: f32,

    /// Dirichlet noise alpha for root node exploration.
    /// Scaled by 10/avg_legal_moves. For games with ~10 legal moves, use ~0.3.
    /// Set this and `dirichlet_epsilon` to 0.0 to disable noise.
    pub dirichlet_alpha: f32,

    /// Fraction of prior that comes from Dirichlet noise at root.
    /// AlphaZero uses 0.25, meaning 75% prior + 25% noise.
    pub dirichlet_epsilon: f32,

    /// Temperature for action selection after search.
    /// 1.0 = sample proportional to visit counts
    /// 0.0 = always pick most-visited (argmax)
    /// AlphaZero uses 1.0 for first 30 moves, then 0.0
    pub temperature: f32,

    /// Magnitude of the virtual loss applied to a leaf while it waits in a
    /// pending evaluation batch. The leaf's value sum is RAISED by this
    /// amount (and its visit count incremented) when selected — nodes store
    /// values from the opponent-of-parent perspective and UCB negates them,
    /// so a higher stored value makes the node less attractive to its
    /// parent. Restored before backpropagation.
    pub virtual_loss: f32,

    /// Upper bound for neural-network evaluation batches.
    /// Search may reduce this to one quarter of the simulation budget so
    /// evaluated values feed back into later selection rounds. Higher values
    /// can improve throughput but increase latency per effective batch.
    /// Set to 1 to disable batching (original behavior).
    pub eval_batch_size: usize,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            c_puct: 1.25,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            temperature: 1.0,
            virtual_loss: 1.0,
            eval_batch_size: 32,
        }
    }
}

impl MctsConfig {
    /// Reject values that make search undefined or numerically invalid.
    pub fn validate(&self) -> Result<(), MctsConfigError> {
        if self.num_simulations == 0 {
            return Err(MctsConfigError::ZeroSimulations);
        }
        if !self.c_puct.is_finite() || self.c_puct < 0.0 {
            return Err(MctsConfigError::InvalidCPuct);
        }
        if !self.dirichlet_alpha.is_finite() || self.dirichlet_alpha < 0.0 {
            return Err(MctsConfigError::InvalidDirichletAlpha);
        }
        if !self.dirichlet_epsilon.is_finite() || !(0.0..=1.0).contains(&self.dirichlet_epsilon) {
            return Err(MctsConfigError::InvalidDirichletEpsilon);
        }
        if (self.dirichlet_alpha == 0.0) != (self.dirichlet_epsilon == 0.0) {
            return Err(MctsConfigError::InconsistentDirichletNoise);
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(MctsConfigError::InvalidTemperature);
        }
        if !self.virtual_loss.is_finite() || self.virtual_loss < 0.0 {
            return Err(MctsConfigError::InvalidVirtualLoss);
        }
        if self.eval_batch_size == 0 {
            return Err(MctsConfigError::ZeroEvalBatchSize);
        }
        Ok(())
    }

    /// Create config for training (with exploration noise).
    pub fn for_training() -> Self {
        Self::default()
    }

    /// Create config for evaluation/inference (no noise, greedy selection).
    pub fn for_evaluation() -> Self {
        Self {
            num_simulations: 800,
            c_puct: 1.25,
            dirichlet_alpha: 0.0, // No noise
            dirichlet_epsilon: 0.0,
            temperature: 0.0, // Greedy
            virtual_loss: 1.0,
            eval_batch_size: 32,
        }
    }

    /// Create a fast config for testing.
    pub fn for_testing() -> Self {
        Self {
            num_simulations: 50,
            c_puct: 1.25,
            dirichlet_alpha: 0.0,
            dirichlet_epsilon: 0.0,
            temperature: 0.0,
            virtual_loss: 1.0,
            eval_batch_size: 8,
        }
    }

    /// Builder pattern: set number of simulations.
    pub fn with_simulations(mut self, n: u32) -> Self {
        self.num_simulations = n;
        self
    }

    /// Builder pattern: set c_puct exploration constant.
    pub fn with_c_puct(mut self, c: f32) -> Self {
        self.c_puct = c;
        self
    }

    /// Builder pattern: set temperature.
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Builder pattern: set evaluation batch size.
    pub fn with_eval_batch_size(mut self, size: usize) -> Self {
        self.eval_batch_size = size;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MctsConfig::default();
        assert_eq!(config.num_simulations, 800);
        assert!((config.c_puct - 1.25).abs() < 1e-6);
    }

    #[test]
    fn test_builder_pattern() {
        let config = MctsConfig::default()
            .with_simulations(100)
            .with_temperature(0.5);

        assert_eq!(config.num_simulations, 100);
        assert!((config.temperature - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_evaluation_config() {
        let config = MctsConfig::for_evaluation();
        assert!((config.dirichlet_alpha).abs() < 1e-6);
        assert!((config.temperature).abs() < 1e-6);
    }

    #[test]
    fn validation_rejects_zero_search_and_batch_budgets() {
        let mut config = MctsConfig::for_training();
        config.num_simulations = 0;
        assert_eq!(config.validate(), Err(MctsConfigError::ZeroSimulations));

        config.num_simulations = 1;
        config.eval_batch_size = 0;
        assert_eq!(config.validate(), Err(MctsConfigError::ZeroEvalBatchSize));
    }

    #[test]
    fn validation_rejects_invalid_float_domains() {
        let mut config = MctsConfig::for_training();
        config.c_puct = f32::NAN;
        assert_eq!(config.validate(), Err(MctsConfigError::InvalidCPuct));

        config = MctsConfig::for_training();
        config.dirichlet_alpha = -0.1;
        assert_eq!(
            config.validate(),
            Err(MctsConfigError::InvalidDirichletAlpha)
        );

        config = MctsConfig::for_training();
        config.dirichlet_epsilon = 1.1;
        assert_eq!(
            config.validate(),
            Err(MctsConfigError::InvalidDirichletEpsilon)
        );

        config = MctsConfig::for_training();
        config.dirichlet_alpha = 0.0;
        assert_eq!(
            config.validate(),
            Err(MctsConfigError::InconsistentDirichletNoise)
        );

        config = MctsConfig::for_training();
        config.temperature = f32::INFINITY;
        assert_eq!(config.validate(), Err(MctsConfigError::InvalidTemperature));

        config = MctsConfig::for_training();
        config.virtual_loss = -1.0;
        assert_eq!(config.validate(), Err(MctsConfigError::InvalidVirtualLoss));
    }
}
