//! Measure the trained generals model's playing strength: MCTS + ONNX
//! versus a uniform-random opponent, alternating seats.
//!
//! This is the fair strength test — the trainer's built-in eval plays the
//! raw policy argmax with no search, which understates the system.
//!
//! Run: cargo run -p mcts --features onnx --example generals_strength_probe \
//!        --release -- <model.onnx> [games] [sims]

use engine_core::EngineContext;
use mcts::{run_mcts, MctsConfig, OnnxEvaluator};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .expect("usage: probe <model.onnx> [games] [sims]");
    let games: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20);
    let sims: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(50);

    engine_games::register_all_games();
    let mut ctx = EngineContext::new("generals_8x8").unwrap();
    let meta = ctx.metadata();
    let evaluator = OnnxEvaluator::load(model_path, meta.obs_size, 1).unwrap();

    // Evaluation config: greedy, no exploration noise
    let config = MctsConfig::for_evaluation()
        .with_simulations(sims)
        .with_eval_batch_size(64)
        .with_temperature(0.0);

    let mut model_wins = 0u32;
    let mut random_wins = 0u32;
    let mut draws = 0u32;

    for game_idx in 0..games {
        // Alternate which seat the model plays
        let model_seat: u8 = if game_idx % 2 == 0 { 1 } else { 2 };
        let reset = ctx.reset(1000 + game_idx, &[]).unwrap();
        let mut state = reset.state;
        let mut obs = reset.obs;
        let mut rng = ChaCha20Rng::seed_from_u64(game_idx);
        let mut current: u8 = 1;

        loop {
            let mask = meta.legal_mask_from_obs(&obs);
            let action: u32 = if current == model_seat {
                let mut search_rng = ChaCha20Rng::seed_from_u64(game_idx * 10_000);
                run_mcts(
                    &mut ctx,
                    &evaluator,
                    config.clone(),
                    state.clone(),
                    obs.clone(),
                    mask,
                    &mut search_rng,
                )
                .unwrap()
                .action
            } else {
                let legal: Vec<usize> = mask.iter_ones().collect();
                legal[rng.gen_range(0..legal.len())] as u32
            };

            let step = ctx.step(&state, &action.to_le_bytes()).unwrap();
            state = step.state;
            obs = step.obs;

            if step.done {
                // Winner from info bits (mask field unused for generals)
                let winner = engine_core::game_utils::info_bits::extract_winner(step.info);
                if winner == model_seat {
                    model_wins += 1;
                } else if winner == 3 || winner == 0 {
                    draws += 1;
                } else {
                    random_wins += 1;
                }
                break;
            }
            current = if current == 1 { 2 } else { 1 };
        }
        println!(
            "game {game_idx}: model as P{model_seat} -> running score model={model_wins} random={random_wins} draws={draws}"
        );
    }

    println!(
        "\nFINAL: model {}-{}-{} vs random ({} sims) -> {:.0}% win rate",
        model_wins,
        random_wins,
        draws,
        sims,
        100.0 * model_wins as f64 / games as f64
    );
}
