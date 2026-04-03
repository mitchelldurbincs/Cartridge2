use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use engine_core::Game;
use games_othello::{observation_from_state, Othello, State};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn bench_reset(c: &mut Criterion) {
    let mut group = c.benchmark_group("othello_reset");
    group.bench_function("reset", |b| {
        let mut game = Othello::new();
        b.iter_batched(
            || ChaCha20Rng::seed_from_u64(42),
            |mut rng| {
                let _ = game.reset(&mut rng, &[]);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("othello_step");

    // Step from initial position (Black plays D3 = position 19)
    group.bench_function("step_opening", |b| {
        let mut game = Othello::new();
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let (base_state, _) = game.reset(&mut rng, &[]);
        b.iter_batched(
            || base_state.clone(),
            |mut state| {
                let action = 19u32; // D3, a legal opening move
                let _ = game.step(&mut state, action, &mut rng);
            },
            BatchSize::SmallInput,
        );
    });

    // Step from a mid-game position (more pieces on board = more flipping)
    group.bench_function("step_midgame", |b| {
        let mut game = Othello::new();
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let (mut state, _) = game.reset(&mut rng, &[]);

        // Play a few moves to reach a mid-game state
        let opening_moves = [19u32, 18, 10, 11, 2, 34];
        for &m in &opening_moves {
            let legal = state.legal_moves();
            if legal.contains(&m) {
                state = state.make_move(m);
            } else if let Some(&first_legal) = legal.first() {
                state = state.make_move(first_legal);
            }
        }

        let midgame_state = state;
        b.iter_batched(
            || midgame_state.clone(),
            |mut state| {
                let legal = state.legal_moves();
                if let Some(&action) = legal.first() {
                    let _ = game.step(&mut state, action, &mut rng);
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_make_move(c: &mut Criterion) {
    let mut group = c.benchmark_group("othello_make_move");

    group.bench_function("make_move_opening", |b| {
        let state = State::new();
        b.iter_batched(
            || state.clone(),
            |s| {
                let _ = s.make_move(19); // D3
            },
            BatchSize::SmallInput,
        );
    });

    // Mid-game make_move with more pieces to flip
    group.bench_function("make_move_midgame", |b| {
        let mut state = State::new();
        let opening_moves = [19u32, 18, 10, 11, 2, 34];
        for &m in &opening_moves {
            let legal = state.legal_moves();
            if legal.contains(&m) {
                state = state.make_move(m);
            } else if let Some(&first_legal) = legal.first() {
                state = state.make_move(first_legal);
            }
        }

        let midgame_state = state;
        b.iter_batched(
            || midgame_state.clone(),
            |s| {
                let legal = s.legal_moves();
                if let Some(&action) = legal.first() {
                    let _ = s.make_move(action);
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_legal_moves(c: &mut Criterion) {
    let mut group = c.benchmark_group("othello_legal_moves");

    // Legal moves from initial position
    group.bench_function("legal_moves_opening", |b| {
        let state = State::new();
        b.iter(|| state.legal_moves());
    });

    // Legal moves mask (bitmask variant)
    group.bench_function("legal_moves_mask_opening", |b| {
        let state = State::new();
        b.iter(|| state.legal_moves_mask());
    });

    // Legal moves from mid-game (more pieces = more positions to check)
    group.bench_function("legal_moves_midgame", |b| {
        let mut state = State::new();
        let opening_moves = [19u32, 18, 10, 11, 2, 34];
        for &m in &opening_moves {
            let legal = state.legal_moves();
            if legal.contains(&m) {
                state = state.make_move(m);
            } else if let Some(&first_legal) = legal.first() {
                state = state.make_move(first_legal);
            }
        }
        b.iter(|| state.legal_moves());
    });

    group.bench_function("legal_moves_mask_midgame", |b| {
        let mut state = State::new();
        let opening_moves = [19u32, 18, 10, 11, 2, 34];
        for &m in &opening_moves {
            let legal = state.legal_moves();
            if legal.contains(&m) {
                state = state.make_move(m);
            } else if let Some(&first_legal) = legal.first() {
                state = state.make_move(first_legal);
            }
        }
        b.iter(|| state.legal_moves_mask());
    });

    group.finish();
}

fn bench_encode_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("othello_encoding");

    group.bench_function("state_roundtrip", |b| {
        let state = State::new();
        b.iter_batched(
            || Vec::with_capacity(128),
            |mut buffer| {
                Othello::encode_state(&state, &mut buffer).unwrap();
                let _ = Othello::decode_state(&buffer).unwrap();
                buffer
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("observation_encode", |b| {
        let obs = observation_from_state(&State::new());
        b.iter_batched(
            || Vec::with_capacity(1024),
            |mut buffer| {
                Othello::encode_obs(&obs, &mut buffer).unwrap();
                buffer
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_reset,
    bench_step,
    bench_make_move,
    bench_legal_moves,
    bench_encode_decode
);
criterion_main!(benches);
