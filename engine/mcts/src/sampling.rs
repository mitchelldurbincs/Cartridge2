//! Free helper functions for sampling during MCTS search.
//!
//! Extracted from `search.rs` and re-exported there so the public API is
//! unchanged.

use rand::Rng;
use rand_chacha::ChaCha20Rng;

use crate::types::SearchError;

/// Sample an action from a probability distribution.
pub(crate) fn sample_action(policy: &[f32], rng: &mut ChaCha20Rng) -> Result<u32, SearchError> {
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &p) in policy.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return Ok(i as u32);
        }
    }

    // Fallback to last non-zero action (handles floating point issues)
    for (i, &p) in policy.iter().enumerate().rev() {
        if p > 0.0 {
            return Ok(i as u32);
        }
    }

    Err(SearchError::NoLegalMoves)
}

/// Generate Dirichlet-distributed noise using Gamma variates.
pub(crate) fn dirichlet_noise(n: usize, alpha: f32, rng: &mut ChaCha20Rng) -> Vec<f32> {
    use rand_distr::{Distribution, Gamma};

    let gamma = Gamma::new(alpha as f64, 1.0).unwrap();
    let mut samples: Vec<f32> = (0..n).map(|_| gamma.sample(rng) as f32).collect();

    // Normalize
    let sum: f32 = samples.iter().sum();
    if sum > 0.0 {
        for s in &mut samples {
            *s /= sum;
        }
    }

    samples
}
