//! Measure the trained generals model's playing strength: MCTS + ONNX
//! versus a uniform-random opponent, alternating seats.
//!
//! This is the fair strength test — the trainer's built-in eval plays the
//! raw policy argmax with no search, which understates the system.
//!
//! Run: cargo run -p mcts --features onnx --example generals_strength_probe \
//!        --release -- <model.onnx> [games] [sims]

use algorithm_core::{resolve_algorithm, ALPHAZERO_BOARD_V1_ID};
use engine_core::{
    ActionAvailability, ActionSpace, AgentId, EngineContext, EpisodeStatus, ErasedTimestep,
    ObservationEncoding,
};
use mcts::{run_mcts, MctsConfig, OnnxEvaluator};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn active_agent_and_mask(timestep: &ErasedTimestep) -> (u8, &engine_core::LegalMask) {
    let active = timestep.decision.sole_agent().expect("expected one agent");
    let ActionAvailability::DiscreteMask { mask } = &active.availability else {
        panic!("expected a discrete legal-action mask");
    };
    (
        u8::try_from(active.agent_id.0).expect("board seat fits u8"),
        mask,
    )
}

fn terminal_winner(timestep: &ErasedTimestep) -> u8 {
    timestep
        .outcomes
        .iter()
        .find(|outcome| outcome.reward > 0.0)
        .map(|outcome| u8::try_from(outcome.agent_id.0).unwrap())
        .unwrap_or(3)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .expect("usage: probe <model.onnx> [games] [sims]");
    let games: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20);
    let sims: u32 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(50);

    engine_games::register_all_environments();
    let mut ctx = EngineContext::new("generals_8x8").unwrap();
    let capabilities = ctx.capabilities();
    let ObservationEncoding::Tensor { spec } = &capabilities.encoding.observation else {
        panic!("expected tensor observations");
    };
    let obs_size = spec.fixed_elements().expect("fixed observation shape");
    let ActionSpace::Discrete { size: action_count } = capabilities
        .action_space(AgentId(1))
        .expect("agent 1 action space")
    else {
        panic!("expected discrete actions");
    };
    let action_count = usize::try_from(*action_count).expect("action count fits usize");
    let model_contract = resolve_algorithm(ALPHAZERO_BOARD_V1_ID)
        .unwrap()
        .descriptor()
        .model_artifact_contract("generals_8x8", ctx.capabilities().contract_version);
    let evaluator =
        OnnxEvaluator::load_from_file(model_path, obs_size, action_count, 1, &model_contract)
            .unwrap();

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
        let mut timestep = reset.timestep;
        let mut rng = ChaCha20Rng::seed_from_u64(game_idx);

        loop {
            let (current, mask) = active_agent_and_mask(&timestep);
            let action: u32 = if current == model_seat {
                let mut search_rng = ChaCha20Rng::seed_from_u64(game_idx * 10_000);
                run_mcts(
                    &mut ctx,
                    &evaluator,
                    config.clone(),
                    state.clone(),
                    timestep.clone(),
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
            timestep = step.timestep;

            if timestep.episode != EpisodeStatus::Running {
                let winner = terminal_winner(&timestep);
                if winner == model_seat {
                    model_wins += 1;
                } else if winner == 3 || winner == 0 {
                    draws += 1;
                } else {
                    random_wins += 1;
                }
                break;
            }
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
