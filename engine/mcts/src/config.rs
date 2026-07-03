//! MCTS configuration parameters.

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
    /// Set to 0.0 to disable noise (for evaluation/inference).
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
    /// pending evaluation batch. The leaf's value sum is reduced by this amount
    /// (and its visit count incremented) when selected, then restored before
    /// backpropagation, discouraging the same leaf from being selected twice
    /// within one batch.
    pub virtual_loss: f32,

    /// Batch size for neural network evaluation.
    /// Leaves are collected until this batch size is reached, then evaluated together.
    /// Higher values improve throughput but increase latency per batch.
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
}
