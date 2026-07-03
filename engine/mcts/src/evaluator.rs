//! Evaluator trait for position evaluation.
//!
//! The evaluator provides policy (action probabilities) and value estimates
//! for game states. In AlphaZero, this is a neural network. For testing,
//! we provide a uniform evaluator that returns equal priors.

use thiserror::Error;

/// Errors that can occur during evaluation.
#[derive(Debug, Error)]
pub enum EvaluatorError {
    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Model error: {0}")]
    ModelError(String),
}

/// Result of evaluating a game state.
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Policy: probability distribution over actions.
    /// Index i corresponds to action i, values should sum to ~1.0.
    /// For illegal moves, the value should be 0.0.
    pub policy: Vec<f32>,

    /// Value estimate for the current player.
    /// Range: -1.0 (certain loss) to +1.0 (certain win).
    pub value: f32,
}

/// Trait for position evaluators.
///
/// Implementations:
/// - UniformEvaluator: Returns uniform policy (for testing)
/// - OnnxEvaluator: Neural network inference (for training/play, `onnx` feature)
pub trait Evaluator: Send + Sync {
    /// Evaluate a single game observation.
    ///
    /// # Arguments
    /// * `obs` - Encoded observation bytes (neural network input format)
    /// * `legal_moves_mask` - Bit mask of legal actions (from info bits)
    /// * `num_actions` - Total number of possible actions
    ///
    /// # Returns
    /// Policy distribution and value estimate
    fn evaluate(
        &self,
        obs: &[u8],
        legal_moves_mask: u64,
        num_actions: usize,
    ) -> Result<EvalResult, EvaluatorError>;

    /// Batch evaluate multiple observations (optional optimization).
    /// Default implementation calls evaluate() in a loop.
    fn evaluate_batch(
        &self,
        observations: &[&[u8]],
        legal_moves_masks: &[u64],
        num_actions: usize,
    ) -> Result<Vec<EvalResult>, EvaluatorError> {
        observations
            .iter()
            .zip(legal_moves_masks.iter())
            .map(|(obs, mask)| self.evaluate(obs, *mask, num_actions))
            .collect()
    }
}

/// Uniform evaluator that assigns equal probability to all legal moves.
/// Value is always 0.0 (neutral). Useful for testing MCTS without a model.
#[derive(Debug, Clone, Default)]
pub struct UniformEvaluator;

impl UniformEvaluator {
    pub fn new() -> Self {
        Self
    }
}

impl Evaluator for UniformEvaluator {
    fn evaluate(
        &self,
        _obs: &[u8],
        legal_moves_mask: u64,
        num_actions: usize,
    ) -> Result<EvalResult, EvaluatorError> {
        let mut policy = vec![0.0; num_actions];

        // Count legal moves
        let num_legal = legal_moves_mask.count_ones() as f32;

        if num_legal == 0.0 {
            // No legal moves - terminal state
            return Ok(EvalResult { policy, value: 0.0 });
        }

        // Uniform distribution over legal moves
        let prob = 1.0 / num_legal;
        for (i, p) in policy.iter_mut().enumerate().take(num_actions) {
            if (legal_moves_mask >> i) & 1 == 1 {
                *p = prob;
            }
        }

        Ok(EvalResult { policy, value: 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_evaluator() {
        let eval = UniformEvaluator::new();

        // 3 legal moves out of 9 (mask: 0b101010001 = positions 0, 4, 6, 8)
        let mask = 0b101010001u64;
        let result = eval.evaluate(&[], mask, 9).unwrap();

        // Should have 4 legal moves
        let num_legal = mask.count_ones();
        assert_eq!(num_legal, 4);

        // Check probabilities
        let expected_prob = 1.0 / 4.0;
        assert!((result.policy[0] - expected_prob).abs() < 1e-6);
        assert!((result.policy[4] - expected_prob).abs() < 1e-6);
        assert!((result.policy[6] - expected_prob).abs() < 1e-6);
        assert!((result.policy[8] - expected_prob).abs() < 1e-6);

        // Illegal moves should be 0
        assert!((result.policy[1]).abs() < 1e-6);
        assert!((result.policy[2]).abs() < 1e-6);

        // Value should be neutral
        assert!((result.value).abs() < 1e-6);
    }

    #[test]
    fn test_uniform_evaluator_no_legal_moves() {
        let eval = UniformEvaluator::new();

        let result = eval.evaluate(&[], 0, 9).unwrap();

        // All zeros
        for p in &result.policy {
            assert!(p.abs() < 1e-6);
        }
        assert!((result.value).abs() < 1e-6);
    }

    #[test]
    fn test_uniform_evaluator_all_legal() {
        let eval = UniformEvaluator::new();

        // All 9 moves legal
        let mask = 0b111111111u64;
        let result = eval.evaluate(&[], mask, 9).unwrap();

        let expected_prob = 1.0 / 9.0;
        for p in &result.policy {
            assert!((p - expected_prob).abs() < 1e-6);
        }
    }
}
