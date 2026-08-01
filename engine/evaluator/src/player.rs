//! The two things a seat can be: a random baseline, or a model.

use anyhow::{anyhow, Result};
use engine_core::{EngineContext, LegalMask};
use mcts::{run_mcts, Evaluator, MctsConfig, OnnxEvaluator};
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use std::path::Path;

/// How a model picks moves.
///
/// `simulations == 0` means "no search": sample straight from the policy head,
/// which is what the trainer's Python evaluator did and therefore what keeps
/// existing eval numbers comparable. Anything above 0 runs MCTS, which is the
/// honest measure of the system's strength — the policy head alone understates
/// it (a 50-sim search went 1-19 vs random where 1-sim went 10-10 on generals,
/// before the search fixes).
pub struct ModelPlayer {
    evaluator: OnnxEvaluator,
    temperature: f32,
    simulations: u32,
    /// MCTS needs a context of its own for rollouts, separate from the one
    /// stepping the real game.
    sim_ctx: EngineContext,
    label: String,
}

/// A seat in an evaluation match.
pub enum Player {
    Random,
    Model(Box<ModelPlayer>),
}

impl Player {
    /// Load a model player for `env_id`.
    pub fn model(
        env_id: &str,
        model_path: &str,
        temperature: f32,
        simulations: u32,
        intra_threads: usize,
    ) -> Result<Self> {
        let ctx =
            EngineContext::new(env_id).ok_or_else(|| anyhow!("Game '{env_id}' not registered"))?;
        let obs_size = ctx.metadata().obs_size;
        let evaluator = OnnxEvaluator::load(model_path, obs_size, intra_threads)
            .map_err(|e| anyhow!("Failed to load model '{model_path}': {e}"))?;

        Ok(Player::Model(Box::new(ModelPlayer {
            evaluator,
            temperature,
            simulations,
            sim_ctx: ctx,
            // Matches Python's ModelPlayer.name. Eval records and W&B runs key
            // off these strings, so the two must not drift.
            label: format!(
                "ONNX({})",
                Path::new(model_path)
                    .file_name()
                    .unwrap_or_else(|| model_path.as_ref())
                    .to_string_lossy()
            ),
        })))
    }

    /// Name reported in the results, mirroring the Python policy names so
    /// eval records stay comparable across the migration.
    pub fn name(&self) -> String {
        match self {
            Player::Random => "Random".to_string(),
            Player::Model(m) => m.label.clone(),
        }
    }

    /// Choose an action for the current position.
    pub fn select_action(
        &mut self,
        state: &[u8],
        obs: &[u8],
        mask: &LegalMask,
        num_actions: usize,
        rng: &mut ChaCha20Rng,
    ) -> Result<u32> {
        let legal: Vec<u32> = mask.iter_ones().map(|i| i as u32).collect();
        if legal.is_empty() {
            return Err(anyhow!("No legal moves available"));
        }

        match self {
            Player::Random => Ok(legal[rng.gen_range(0..legal.len())]),
            Player::Model(m) if m.simulations == 0 => {
                let result = m
                    .evaluator
                    .evaluate(obs, mask, num_actions)
                    .map_err(|e| anyhow!("Model evaluation failed: {e}"))?;
                Ok(sample_policy(&result.policy, &legal, m.temperature, rng))
            }
            Player::Model(m) => {
                let config = MctsConfig::for_evaluation()
                    .with_simulations(m.simulations)
                    // Evaluations interleave in waves rather than all leaves
                    // being selected before any result returns; a batch larger
                    // than a quarter of the budget makes visit counts carry no
                    // value information at all.
                    .with_eval_batch_size((m.simulations as usize / 4).max(1))
                    .with_temperature(m.temperature);
                let result = run_mcts(
                    &mut m.sim_ctx,
                    &m.evaluator,
                    config,
                    state.to_vec(),
                    obs.to_vec(),
                    mask.clone(),
                    rng,
                )?;
                Ok(result.action)
            }
        }
    }
}

/// Sample an action from `policy`, restricted to `legal`.
///
/// Temperature 0 is greedy. Above 0 the legal probabilities are raised to
/// `1/temperature` and renormalized — the standard AlphaZero play-temperature,
/// and the reason head-to-head evals do not replay one identical game.
fn sample_policy(policy: &[f32], legal: &[u32], temperature: f32, rng: &mut ChaCha20Rng) -> u32 {
    if temperature <= 0.0 {
        return *legal
            .iter()
            .max_by(|&&a, &&b| {
                policy[a as usize]
                    .partial_cmp(&policy[b as usize])
                    .expect("policy has no NaN")
            })
            .expect("legal is non-empty");
    }

    let inv = 1.0 / temperature;
    let weights: Vec<f32> = legal
        .iter()
        .map(|&a| policy[a as usize].max(0.0).powf(inv))
        .collect();
    let total: f32 = weights.iter().sum();

    // A uniformly zero policy over the legal moves (an untrained or degenerate
    // head) would otherwise sample from nothing. NaN fails both comparisons, so
    // check finiteness explicitly rather than relying on `!(total > 0.0)`.
    if total <= 0.0 || !total.is_finite() {
        return legal[rng.gen_range(0..legal.len())];
    }

    let mut point = rng.gen_range(0.0..total);
    for (i, w) in weights.iter().enumerate() {
        point -= w;
        if point <= 0.0 {
            return legal[i];
        }
    }
    *legal.last().expect("legal is non-empty")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> ChaCha20Rng {
        ChaCha20Rng::seed_from_u64(1)
    }

    #[test]
    fn temperature_zero_picks_the_best_legal_action() {
        let policy = vec![0.1, 0.7, 0.2];
        // Action 1 is the argmax overall but illegal here.
        assert_eq!(sample_policy(&policy, &[0, 2], 0.0, &mut rng()), 2);
        assert_eq!(sample_policy(&policy, &[0, 1, 2], 0.0, &mut rng()), 1);
    }

    #[test]
    fn sampling_only_ever_returns_legal_actions() {
        let policy = vec![0.9, 0.05, 0.05];
        let mut r = rng();
        for _ in 0..200 {
            let action = sample_policy(&policy, &[1, 2], 1.0, &mut r);
            assert!(action == 1 || action == 2, "picked illegal action {action}");
        }
    }

    #[test]
    fn an_all_zero_policy_falls_back_to_uniform_instead_of_dividing_by_zero() {
        let policy = vec![0.0, 0.0, 0.0];
        let mut r = rng();
        for _ in 0..50 {
            let action = sample_policy(&policy, &[0, 2], 1.0, &mut r);
            assert!(action == 0 || action == 2);
        }
    }

    #[test]
    fn low_temperature_concentrates_on_the_favourite() {
        let policy = vec![0.6, 0.4];
        let mut r = rng();
        let favourite = (0..400)
            .filter(|_| sample_policy(&policy, &[0, 1], 0.1, &mut r) == 0)
            .count();
        assert!(
            favourite > 380,
            "temperature 0.1 should be near-greedy, got {favourite}/400"
        );
    }
}
