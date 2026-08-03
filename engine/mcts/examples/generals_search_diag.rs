//! Diagnostic: how much signal do generals self-play policy targets carry?
//!
//! Part A measures the game's shape under random play (branching factor,
//! episode length, terminal type). Part B runs the actor's search config at
//! several simulation budgets and reports how the root visit counts spread.
//!
//! Part B prints two targets per row. `tau=1` is what the search now stores
//! for training. `tau=0.1` is what it used to store, when the late-game play
//! temperature was also applied to the training target — kept here to show
//! how much of the visit distribution that discarded.
//!
//! Run: cargo run -p mcts --example generals_search_diag --release

use engine_core::{Decision, EngineContext, EpisodeStatus, ErasedTimestep};
use mcts::{run_mcts, MctsConfig, UniformEvaluator};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

const CAP_PLIES: u32 = 400; // 2 * MAX_TURNS

fn active_observation(timestep: &ErasedTimestep) -> &[u8] {
    let agent = match &timestep.decision {
        Decision::Agents { agent_ids } if agent_ids.len() == 1 => agent_ids[0],
        decision => panic!("expected one active agent, got {decision:?}"),
    };
    timestep.observation_for(agent).unwrap()
}

fn terminal_winner(timestep: &ErasedTimestep) -> usize {
    timestep
        .outcomes
        .iter()
        .find(|outcome| outcome.reward > 0.0)
        .map(|outcome| outcome.agent_id.0 as usize)
        .unwrap_or(3)
}

fn percentile(sorted: &[u32], p: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

/// Training target after the actor applies `root_policy(temperature)`.
///
/// Mirrors `MctsTree::root_policy`: raise raw VISIT COUNTS to 1/tau, then
/// normalize. Done in f64 — raising already-normalized f32 probabilities
/// underflows once the distribution is wide.
fn sharpen(visits: &[f64], temperature: f64) -> Vec<f64> {
    let raised: Vec<f64> = visits.iter().map(|v| v.powf(1.0 / temperature)).collect();
    let total: f64 = raised.iter().sum();
    raised.iter().map(|v| v / total.max(1e-300)).collect()
}

/// exp(entropy) — the effective number of moves the distribution spreads over.
fn perplexity(policy: &[f64]) -> f64 {
    let h: f64 = policy
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    h.exp()
}

fn top1(policy: &[f64]) -> f64 {
    policy.iter().copied().fold(0.0, f64::max)
}

fn normalize(visits: &[f64]) -> Vec<f64> {
    let total: f64 = visits.iter().sum();
    visits.iter().map(|v| v / total.max(1e-300)).collect()
}

fn main() {
    engine_games::register_all_environments();
    let mut ctx = EngineContext::new("generals_8x8").unwrap();
    let meta = ctx.metadata().clone();
    let board = meta.require_board().unwrap();

    // ---------------- Part A: game shape under random play ----------------
    let games = 200u64;
    let mut plies_per_game: Vec<u32> = Vec::new();
    let mut legal_counts: Vec<u32> = Vec::new();
    let mut wait_only = 0u64;
    let mut total_plies = 0u64;
    let mut adjudicated = 0u32;
    let mut winners = [0u32; 4];

    for seed in 0..games {
        let reset = ctx.reset(seed, &[]).unwrap();
        let mut state = reset.state;
        let mut timestep = reset.timestep;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut plies = 0u32;

        loop {
            let mask = board
                .legal_mask_from_obs(active_observation(&timestep))
                .unwrap();
            let legal: Vec<usize> = mask.iter_ones().collect();
            legal_counts.push(legal.len() as u32);
            if legal.len() == 1 {
                wait_only += 1;
            }
            let action = legal[rng.gen_range(0..legal.len())] as u32;
            let step = ctx.step(&state, &action.to_le_bytes()).unwrap();
            state = step.state;
            timestep = step.timestep;
            plies += 1;
            total_plies += 1;
            if timestep.episode != EpisodeStatus::Running {
                winners[terminal_winner(&timestep).min(3)] += 1;
                if plies >= CAP_PLIES - 1 {
                    adjudicated += 1;
                }
                break;
            }
        }
        plies_per_game.push(plies);
    }

    plies_per_game.sort_unstable();
    legal_counts.sort_unstable();

    println!("=== Part A: generals_8x8 under uniform-random self-play ({games} games) ===");
    println!(
        "episode length (plies): mean={:.0} p10={} p50={} p90={} max={}",
        total_plies as f64 / games as f64,
        percentile(&plies_per_game, 0.10),
        percentile(&plies_per_game, 0.50),
        percentile(&plies_per_game, 0.90),
        plies_per_game.last().unwrap(),
    );
    println!(
        "legal moves per ply:    mean={:.1} p10={} p50={} p90={} max={}",
        legal_counts.iter().map(|c| *c as f64).sum::<f64>() / legal_counts.len() as f64,
        percentile(&legal_counts, 0.10),
        percentile(&legal_counts, 0.50),
        percentile(&legal_counts, 0.90),
        legal_counts.last().unwrap(),
    );
    println!(
        "terminal: by general capture={} by cap adjudication={} ({:.0}%)",
        games as u32 - adjudicated,
        adjudicated,
        100.0 * adjudicated as f64 / games as f64
    );
    println!(
        "winners: p1={} p2={} draw={} | wait-only plies={:.2}%",
        winners[1],
        winners[2],
        winners[3],
        100.0 * wait_only as f64 / total_plies as f64
    );

    // ---------------- Part B: search signal vs simulation budget ----------------
    // Sample positions along one random rollout, then search each with the
    // actor's training config at several budgets.
    let probe_plies = [0u32, 10, 30, 80, 200];
    let sim_budgets = [50u32, 100, 250, 800, 2000];

    println!("\n=== Part B: root visit spread (UniformEvaluator, actor training config) ===");
    println!(
        "ply  legal | sims | visited  ms/search | stored now (tau=1): top1 perplex | was (tau=0.1): top1 perplex"
    );

    for &probe in &probe_plies {
        // Replay a fixed random rollout to the probe ply.
        let reset = ctx.reset(7, &[]).unwrap();
        let mut state = reset.state;
        let mut timestep = reset.timestep;
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let mut reached = true;
        for _ in 0..probe {
            let mask = board
                .legal_mask_from_obs(active_observation(&timestep))
                .unwrap();
            let legal: Vec<usize> = mask.iter_ones().collect();
            let action = legal[rng.gen_range(0..legal.len())] as u32;
            let step = ctx.step(&state, &action.to_le_bytes()).unwrap();
            state = step.state;
            timestep = step.timestep;
            if timestep.episode != EpisodeStatus::Running {
                reached = false;
                break;
            }
        }
        if !reached {
            continue;
        }

        let mask = board
            .legal_mask_from_obs(active_observation(&timestep))
            .unwrap();
        let n_legal = mask.count_ones();

        for &sims in &sim_budgets {
            // Mirror the actor: for_training(), eval batch capped at sims/4.
            let config = MctsConfig::for_training()
                .with_simulations(sims)
                .with_eval_batch_size(64)
                .with_temperature(1.0);
            let evaluator = UniformEvaluator::new();
            let mut search_rng = ChaCha20Rng::seed_from_u64(99);
            let t0 = std::time::Instant::now();
            let result = run_mcts(
                &mut ctx,
                &evaluator,
                config,
                state.clone(),
                timestep.clone(),
                &mut search_rng,
            )
            .unwrap();
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

            // result.policy at temperature 1.0 is visits/total; recover counts.
            let visits: Vec<f64> = result
                .policy
                .iter()
                .map(|p| (*p as f64) * result.simulations as f64)
                .collect();
            let visited = visits.iter().filter(|v| **v > 0.0).count();
            let raw = normalize(&visits);
            let stored = sharpen(&visits, 0.1);

            println!(
                "{:>3}  {:>5} | {:>4} | {:>7}  {:>9.1} | {:>17.3} {:>7.1} | {:>17.3} {:>7.1}",
                probe,
                n_legal,
                sims,
                visited,
                elapsed_ms,
                top1(&raw),
                perplexity(&raw),
                top1(&stored),
                perplexity(&stored),
            );
        }
    }
}
