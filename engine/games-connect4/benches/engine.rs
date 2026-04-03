use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use engine_core::Game;
use games_connect4::{observation_from_state, Connect4, State, COLS, ROWS};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn bench_reset(c: &mut Criterion) {
    let mut group = c.benchmark_group("connect4_reset");
    group.bench_function("reset", |b| {
        let mut game = Connect4::new();
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
    let mut group = c.benchmark_group("connect4_step");

    // Step from initial position (drop in center column)
    group.bench_function("step_center", |b| {
        let mut game = Connect4::new();
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let (base_state, _) = game.reset(&mut rng, &[]);
        b.iter_batched(
            || base_state.clone(),
            |mut state| {
                let action = 3u8; // Center column
                let _ = game.step(&mut state, action, &mut rng);
            },
            BatchSize::SmallInput,
        );
    });

    // Step from mid-game position
    group.bench_function("step_midgame", |b| {
        let mut game = Connect4::new();
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let (mut state, _) = game.reset(&mut rng, &[]);

        // Play several moves to reach mid-game
        let moves = [3u8, 3, 4, 4, 2, 2, 5, 5, 1, 1];
        for &m in &moves {
            state = state.drop_piece(m);
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
    let mut group = c.benchmark_group("connect4_make_move");

    group.bench_function("drop_piece_opening", |b| {
        let state = State::new();
        b.iter_batched(
            || state.clone(),
            |s| {
                let _ = s.drop_piece(3); // Center column
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("drop_piece_midgame", |b| {
        let mut state = State::new();
        let moves = [3u8, 3, 4, 4, 2, 2, 5, 5, 1, 1];
        for &m in &moves {
            state = state.drop_piece(m);
        }

        let midgame_state = state;
        b.iter_batched(
            || midgame_state.clone(),
            |s| {
                let legal = s.legal_moves();
                if let Some(&action) = legal.first() {
                    let _ = s.drop_piece(action);
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_legal_moves(c: &mut Criterion) {
    let mut group = c.benchmark_group("connect4_legal_moves");

    group.bench_function("legal_moves_opening", |b| {
        let state = State::new();
        b.iter(|| state.legal_moves());
    });

    group.bench_function("legal_moves_mask_opening", |b| {
        let state = State::new();
        b.iter(|| state.legal_moves_mask());
    });

    // Near-full board where few columns remain
    group.bench_function("legal_moves_near_full", |b| {
        let mut state = State::new();
        // Fill most columns
        for col in 0..COLS as u8 - 1 {
            for _ in 0..ROWS {
                state = state.drop_piece(col);
            }
        }
        b.iter(|| state.legal_moves());
    });

    group.bench_function("legal_moves_mask_near_full", |b| {
        let mut state = State::new();
        for col in 0..COLS as u8 - 1 {
            for _ in 0..ROWS {
                state = state.drop_piece(col);
            }
        }
        b.iter(|| state.legal_moves_mask());
    });

    group.finish();
}

fn bench_encode_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("connect4_encoding");

    group.bench_function("state_roundtrip", |b| {
        let state = State::new();
        b.iter_batched(
            || Vec::with_capacity(64),
            |mut buffer| {
                Connect4::encode_state(&state, &mut buffer).unwrap();
                let _ = Connect4::decode_state(&buffer).unwrap();
                buffer
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("observation_encode", |b| {
        let obs = observation_from_state(&State::new());
        b.iter_batched(
            || Vec::with_capacity(512),
            |mut buffer| {
                Connect4::encode_obs(&obs, &mut buffer).unwrap();
                buffer
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_column_heights_redundancy(c: &mut Criterion) {
    let mut group = c.benchmark_group("connect4_column_heights");

    // Benchmark: reconstructing column_heights from board (as decode_state does)
    // vs reading the cached column_heights field
    let mut state = State::new();
    let moves = [3u8, 3, 4, 4, 2, 2, 5, 5, 1, 1, 0, 6, 3, 4];
    for &m in &moves {
        state = state.drop_piece(m);
    }

    // Encode the state (which strips column_heights)
    let mut encoded = Vec::new();
    Connect4::encode_state(&state, &mut encoded).unwrap();

    group.bench_function("decode_reconstructs_heights", |b| {
        b.iter(|| {
            let _ = Connect4::decode_state(&encoded).unwrap();
        });
    });

    // Compare: legal_moves using cached column_heights (normal path)
    group.bench_function("legal_moves_cached_heights", |b| {
        b.iter(|| state.legal_moves_mask());
    });

    // Compare: computing column heights from board manually
    group.bench_function("compute_heights_from_board", |b| {
        let board = &encoded[..COLS * ROWS];
        b.iter(|| {
            let mut heights = [0u8; COLS];
            for col in 0..COLS {
                for row in 0..ROWS {
                    if board[row * COLS + col] != 0 {
                        heights[col] = (row + 1) as u8;
                    }
                }
            }
            heights
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_reset,
    bench_step,
    bench_make_move,
    bench_legal_moves,
    bench_encode_decode,
    bench_column_heights_redundancy
);
criterion_main!(benches);
