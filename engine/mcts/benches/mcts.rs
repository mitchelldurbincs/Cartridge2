//! MCTS benchmarks for performance profiling.
//!
//! Run with: `cargo bench -p mcts`
//!
//! These benchmarks measure:
//! - Full MCTS search with varying simulation counts
//! - Tree operations (selection, backpropagation, policy extraction)
//! - Search from different game states (opening, midgame, near-terminal)
//! - Game comparison (TicTacToe vs Connect4)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use engine_core::EngineContext;
use mcts::{MctsConfig, MctsSearch, MctsTree, UniformEvaluator};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Register all games and return TicTacToe context.
fn setup_tictactoe() -> EngineContext {
    engine_games::register_all_games();
    EngineContext::new("tictactoe").unwrap()
}

/// Register all games and return Connect4 context.
fn setup_connect4() -> EngineContext {
    engine_games::register_all_games();
    EngineContext::new("connect4").unwrap()
}

/// Helper to create a game state after playing a sequence of moves.
fn play_moves(ctx: &mut EngineContext, seed: u64, moves: &[u32]) -> (Vec<u8>, Vec<u8>, u64) {
    let reset = ctx.reset(seed, &[]).unwrap();
    let mut state = reset.state;
    let mut obs = reset.obs;
    let mut info = 0b111111111u64; // All positions legal initially

    for &m in moves {
        let action = m.to_le_bytes().to_vec();
        let step = ctx.step(&state, &action).unwrap();
        state = step.state;
        obs = step.obs;
        info = step.info;
    }

    let legal_mask = info & 0x1FF;
    (state, obs, legal_mask)
}

// =============================================================================
// Full MCTS Search Benchmarks
// =============================================================================

fn bench_mcts_search_simulations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_search_simulations");

    // Test different simulation counts (including 1600 for strong play)
    for sims in [50, 100, 200, 400, 800, 1600] {
        group.throughput(Throughput::Elements(sims as u64));
        group.bench_with_input(BenchmarkId::new("tictactoe", sims), &sims, |b, &sims| {
            let mut ctx = setup_tictactoe();
            let evaluator = UniformEvaluator::new();
            let config = MctsConfig::for_testing().with_simulations(sims);

            b.iter(|| {
                let reset = ctx.reset(42, &[]).unwrap();
                let legal_mask = 0b111111111u64;
                let mut rng = ChaCha20Rng::seed_from_u64(42);

                let mut search = MctsSearch::new(
                    &mut ctx,
                    &evaluator,
                    config.clone(),
                    reset.state,
                    reset.obs,
                    legal_mask,
                )
                .unwrap();

                black_box(search.run(&mut rng).unwrap())
            });
        });
    }

    group.finish();
}

// =============================================================================
// Connect4 MCTS Search Benchmarks
// =============================================================================

fn bench_mcts_connect4(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_connect4");

    // Test different simulation counts for Connect4 (larger state space)
    for sims in [50, 100, 200, 400, 800] {
        group.throughput(Throughput::Elements(sims as u64));
        group.bench_with_input(BenchmarkId::new("opening", sims), &sims, |b, &sims| {
            let mut ctx = setup_connect4();
            let evaluator = UniformEvaluator::new();
            let config = MctsConfig::for_testing().with_simulations(sims);

            b.iter(|| {
                let reset = ctx.reset(42, &[]).unwrap();
                let legal_mask = 0b1111111u64; // All 7 columns legal
                let mut rng = ChaCha20Rng::seed_from_u64(42);

                let mut search = MctsSearch::new(
                    &mut ctx,
                    &evaluator,
                    config.clone(),
                    reset.state,
                    reset.obs,
                    legal_mask,
                )
                .unwrap();

                black_box(search.run(&mut rng).unwrap())
            });
        });
    }

    group.finish();
}

// =============================================================================
// Game Comparison Benchmarks (TicTacToe vs Connect4)
// =============================================================================

fn bench_game_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("game_comparison");
    let sims = 400u32;

    // TicTacToe baseline
    group.bench_function("tictactoe_400_sims", |b| {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_testing().with_simulations(sims);

        b.iter(|| {
            let reset = ctx.reset(42, &[]).unwrap();
            let legal_mask = 0b111111111u64;
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search = MctsSearch::new(
                &mut ctx,
                &evaluator,
                config.clone(),
                reset.state,
                reset.obs,
                legal_mask,
            )
            .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    // Connect4 comparison (larger state space, deeper games)
    group.bench_function("connect4_400_sims", |b| {
        let mut ctx = setup_connect4();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_testing().with_simulations(sims);

        b.iter(|| {
            let reset = ctx.reset(42, &[]).unwrap();
            let legal_mask = 0b1111111u64;
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search = MctsSearch::new(
                &mut ctx,
                &evaluator,
                config.clone(),
                reset.state,
                reset.obs,
                legal_mask,
            )
            .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    group.finish();
}

fn bench_mcts_game_phases(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_game_phases");
    let sims = 200u32;

    // Opening position (all 9 moves available)
    group.bench_function("opening", |b| {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_testing().with_simulations(sims);

        b.iter(|| {
            let reset = ctx.reset(42, &[]).unwrap();
            let legal_mask = 0b111111111u64;
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search = MctsSearch::new(
                &mut ctx,
                &evaluator,
                config.clone(),
                reset.state,
                reset.obs,
                legal_mask,
            )
            .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    // Midgame position (5 moves available)
    // Board: X at 4, O at 0, X at 2, O at 6
    group.bench_function("midgame", |b| {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_testing().with_simulations(sims);

        b.iter(|| {
            let (state, obs, legal_mask) = play_moves(&mut ctx, 42, &[4, 0, 2, 6]);
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search =
                MctsSearch::new(&mut ctx, &evaluator, config.clone(), state, obs, legal_mask)
                    .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    // Near-terminal position (winning move available)
    // Board: X at 0, O at 3, X at 1, O at 4 -> X can win at 2
    group.bench_function("near_terminal", |b| {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_testing().with_simulations(sims);

        b.iter(|| {
            let (state, obs, legal_mask) = play_moves(&mut ctx, 42, &[0, 3, 1, 4]);
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search =
                MctsSearch::new(&mut ctx, &evaluator, config.clone(), state, obs, legal_mask)
                    .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    group.finish();
}

// =============================================================================
// Tree Operation Benchmarks
// =============================================================================

fn bench_tree_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_tree_ops");

    // Benchmark node allocation
    group.bench_function("allocate_node", |b| {
        b.iter(|| {
            let mut tree = MctsTree::new(0b111111111);

            // Allocate 100 child nodes
            for i in 0..100u8 {
                tree.add_child(tree.root(), i % 9, 0.11, 0b111111111, false, 0.0);
            }

            black_box(tree.len())
        });
    });

    // Benchmark child selection (UCB calculation)
    group.bench_function("select_child", |b| {
        // Pre-build a tree with children
        let mut tree = MctsTree::new(0b111111111);

        // Add 9 children with varying priors and visit counts
        for i in 0..9u8 {
            let child_id = tree.add_child(
                tree.root(),
                i,
                (i as f32 + 1.0) / 45.0, // Varying priors
                0b111111111,
                false,
                0.0,
            );
            // Simulate some visits
            let child = tree.get_mut(child_id);
            child.visit_count = (i as u32 + 1) * 10;
            child.value_sum = (i as f32 - 4.0) * 0.1 * child.visit_count as f32;
        }

        // Update root visit count
        tree.get_mut(tree.root()).visit_count = 450;

        b.iter(|| black_box(tree.select_child(tree.root(), 1.25)));
    });

    // Benchmark backpropagation
    group.bench_function("backpropagate_depth_5", |b| {
        b.iter_batched(
            || {
                // Setup: create a tree with depth 5
                let mut tree = MctsTree::new(0b111111111);
                let mut parent = tree.root();

                for i in 0..5 {
                    let child = tree.add_child(
                        parent,
                        i,
                        0.5,
                        0b111111111,
                        i == 4, // Last one is terminal
                        if i == 4 { 1.0 } else { 0.0 },
                    );
                    parent = child;
                }

                (tree, parent)
            },
            |(mut tree, leaf)| {
                tree.backpropagate(leaf, 1.0);
                black_box(tree)
            },
            criterion::BatchSize::SmallInput,
        );
    });

    // Benchmark policy extraction
    group.bench_function("root_policy", |b| {
        // Pre-build a tree with children
        let mut tree = MctsTree::new(0b111111111);

        for i in 0..9u8 {
            let child_id = tree.add_child(tree.root(), i, 1.0 / 9.0, 0b111111111, false, 0.0);
            tree.get_mut(child_id).visit_count = (i as u32 + 1) * 50;
        }

        b.iter(|| black_box(tree.root_policy(9, 1.0)));
    });

    // Benchmark policy extraction with temperature scaling
    group.bench_function("root_policy_temperature", |b| {
        let mut tree = MctsTree::new(0b111111111);

        for i in 0..9u8 {
            let child_id = tree.add_child(tree.root(), i, 1.0 / 9.0, 0b111111111, false, 0.0);
            tree.get_mut(child_id).visit_count = (i as u32 + 1) * 50;
        }

        b.iter(|| black_box(tree.root_policy(9, 0.5)));
    });

    group.finish();
}

// =============================================================================
// Configuration Comparison Benchmarks
// =============================================================================

fn bench_mcts_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_configs");
    let sims = 200u32;

    // Training config (with Dirichlet noise)
    group.bench_function("training_config", |b| {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_training().with_simulations(sims);

        b.iter(|| {
            let reset = ctx.reset(42, &[]).unwrap();
            let legal_mask = 0b111111111u64;
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search = MctsSearch::new(
                &mut ctx,
                &evaluator,
                config.clone(),
                reset.state,
                reset.obs,
                legal_mask,
            )
            .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    // Evaluation config (no noise, greedy)
    group.bench_function("evaluation_config", |b| {
        let mut ctx = setup_tictactoe();
        let evaluator = UniformEvaluator::new();
        let config = MctsConfig::for_evaluation().with_simulations(sims);

        b.iter(|| {
            let reset = ctx.reset(42, &[]).unwrap();
            let legal_mask = 0b111111111u64;
            let mut rng = ChaCha20Rng::seed_from_u64(42);

            let mut search = MctsSearch::new(
                &mut ctx,
                &evaluator,
                config.clone(),
                reset.state,
                reset.obs,
                legal_mask,
            )
            .unwrap();

            black_box(search.run(&mut rng).unwrap())
        });
    });

    // Different c_puct values
    for c_puct in [0.5, 1.25, 2.5, 4.0] {
        group.bench_with_input(BenchmarkId::new("c_puct", c_puct), &c_puct, |b, &c_puct| {
            let mut ctx = setup_tictactoe();
            let evaluator = UniformEvaluator::new();
            let config = MctsConfig::for_testing()
                .with_simulations(sims)
                .with_c_puct(c_puct);

            b.iter(|| {
                let reset = ctx.reset(42, &[]).unwrap();
                let legal_mask = 0b111111111u64;
                let mut rng = ChaCha20Rng::seed_from_u64(42);

                let mut search = MctsSearch::new(
                    &mut ctx,
                    &evaluator,
                    config.clone(),
                    reset.state,
                    reset.obs,
                    legal_mask,
                )
                .unwrap();

                black_box(search.run(&mut rng).unwrap())
            });
        });
    }

    group.finish();
}

// =============================================================================
// Batch Size Benchmarks
// =============================================================================

fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_batch_sizes");
    let sims = 400u32;

    // Test different batch sizes
    // With UniformEvaluator this mainly tests batching overhead
    // With real ONNX models, larger batches amortize inference cost
    for batch_size in [1, 4, 8, 16, 32, 64] {
        group.bench_with_input(
            BenchmarkId::new("tictactoe", batch_size),
            &batch_size,
            |b, &batch_size| {
                let mut ctx = setup_tictactoe();
                let evaluator = UniformEvaluator::new();
                let config = MctsConfig::for_testing()
                    .with_simulations(sims)
                    .with_eval_batch_size(batch_size);

                b.iter(|| {
                    let reset = ctx.reset(42, &[]).unwrap();
                    let legal_mask = 0b111111111u64;
                    let mut rng = ChaCha20Rng::seed_from_u64(42);

                    let mut search = MctsSearch::new(
                        &mut ctx,
                        &evaluator,
                        config.clone(),
                        reset.state,
                        reset.obs,
                        legal_mask,
                    )
                    .unwrap();

                    black_box(search.run(&mut rng).unwrap())
                });
            },
        );
    }

    // Also test batch sizes with Connect4 (more complex game)
    for batch_size in [1, 8, 32] {
        group.bench_with_input(
            BenchmarkId::new("connect4", batch_size),
            &batch_size,
            |b, &batch_size| {
                let mut ctx = setup_connect4();
                let evaluator = UniformEvaluator::new();
                let config = MctsConfig::for_testing()
                    .with_simulations(sims)
                    .with_eval_batch_size(batch_size);

                b.iter(|| {
                    let reset = ctx.reset(42, &[]).unwrap();
                    let legal_mask = 0b1111111u64;
                    let mut rng = ChaCha20Rng::seed_from_u64(42);

                    let mut search = MctsSearch::new(
                        &mut ctx,
                        &evaluator,
                        config.clone(),
                        reset.state,
                        reset.obs,
                        legal_mask,
                    )
                    .unwrap();

                    black_box(search.run(&mut rng).unwrap())
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_mcts_search_simulations,
    bench_mcts_connect4,
    bench_game_comparison,
    bench_mcts_game_phases,
    bench_tree_operations,
    bench_mcts_configs,
    bench_batch_sizes,
);

criterion_main!(benches);
