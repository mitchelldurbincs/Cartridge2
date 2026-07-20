//! Probe the MCTS policy targets generated for generals self-play.
//!
//! Reproduces the actor's exact search configuration against a checkpoint
//! and prints the visit-count policy that would be stored as a training
//! target. Diagnosing the wait-collapse: targets in the replay buffer are
//! one-hot on wait; this shows whether the search itself produces them.
//!
//! Run: cargo run -p mcts --features onnx --example generals_policy_probe \
//!        --release -- <model.onnx> [num_sims]

use engine_core::EngineContext;
use mcts::{run_mcts, MctsConfig, OnnxEvaluator, UniformEvaluator};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn print_policy(label: &str, policy: &[f32]) {
    let mut top: Vec<(usize, f32)> = policy
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, p)| *p > 0.0)
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let support = top.len();
    let wait_mass = policy[256];
    let head: Vec<String> = top
        .iter()
        .take(6)
        .map(|(a, p)| format!("{}:{:.3}", a, p))
        .collect();
    println!(
        "{label}: support={support} wait_mass={wait_mass:.3} top=[{}]",
        head.join(", ")
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("usage: probe <model.onnx> [sims]");
    let sims: u32 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(35);

    engine_games::register_all_games();
    let mut ctx = EngineContext::new("generals_8x8").unwrap();
    let meta = ctx.metadata();

    let config = MctsConfig::for_training()
        .with_simulations(sims)
        .with_eval_batch_size(64)
        .with_temperature(1.0);

    let evaluator = OnnxEvaluator::load(model_path, meta.obs_size, 1).unwrap();
    let uniform = UniformEvaluator::new();

    for seed in [42u64, 7, 13] {
        let reset = ctx.reset(seed, &[]).unwrap();
        let mask = meta.legal_mask_from_obs(&reset.obs);
        println!("--- seed {seed}: {} legal actions ---", mask.count_ones());

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let result = run_mcts(
            &mut ctx,
            &evaluator,
            config.clone(),
            reset.state.clone(),
            reset.obs.clone(),
            mask.clone(),
            &mut rng,
        )
        .unwrap();
        print_policy("onnx   ", &result.policy);

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let result_u = run_mcts(
            &mut ctx,
            &uniform,
            config.clone(),
            reset.state,
            reset.obs,
            mask,
            &mut rng,
        )
        .unwrap();
        print_policy("uniform", &result_u.policy);
    }
}
