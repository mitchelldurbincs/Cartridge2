//! Outcome distribution of uniform-random self-play, for comparing the
//! Rust engine's playout dynamics against the pure-Python mirror.
//!
//! Run: cargo run -p games-generals --example random_playout --release

use engine_core::board_profile::BoardGame;
use games_generals::params::MAX_TURNS;
use games_generals::Generals;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

fn main() {
    let games = 50;
    let mut wins = [0u32; 2];
    let mut draws = 0u32;
    let mut total_plies = 0u64;

    for seed in 0..games {
        let mut game = Generals::new();
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let (mut state, _) = game.reset(&mut rng, &[]).unwrap();
        let mut plies = 0u32;

        loop {
            let legal: Vec<u32> = Generals::legal_actions(&state)
                .unwrap()
                .iter_ones()
                .map(|action| action as u32)
                .collect();
            let action = legal[rng.gen_range(0..legal.len())];
            let transition = game.step(&mut state, action, &mut rng).unwrap();
            plies += 1;
            if transition.terminated {
                break;
            }
            assert!(plies <= MAX_TURNS * 2);
        }

        total_plies += plies as u64;
        match state.winner {
            1 => wins[0] += 1,
            2 => wins[1] += 1,
            _ => draws += 1,
        }
    }

    println!(
        "rust: games={} p1={} p2={} draws={} avg_plies={:.0}",
        games,
        wins[0],
        wins[1],
        draws,
        total_plies as f64 / games as f64
    );
}
