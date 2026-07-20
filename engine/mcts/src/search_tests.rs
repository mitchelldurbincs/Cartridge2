//! Tests for the MCTS search implementation.

use super::*;
use crate::evaluator::UniformEvaluator;
use rand::SeedableRng;
use tracing::trace;

fn setup_tictactoe() -> EngineContext {
    engine_games::register_all_games();
    EngineContext::new("tictactoe").unwrap()
}

#[test]
fn test_mcts_basic_search() {
    let mut ctx = setup_tictactoe();
    let evaluator = UniformEvaluator::new();
    let config = MctsConfig::for_testing();

    let reset = ctx.reset(42, &[]).unwrap();
    // All 9 positions are legal at the start of TicTacToe
    let legal_mask = LegalMask::from_u64(0b111111111, 9);

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let result = run_mcts(
        &mut ctx,
        &evaluator,
        config,
        reset.state,
        reset.obs,
        legal_mask,
        &mut rng,
    );

    assert!(result.is_ok());
    let result = result.unwrap();

    // Should pick a valid action (0-8)
    assert!(result.action < 9);

    // Policy should sum to ~1.0
    let sum: f32 = result.policy.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);

    // Should have run simulations
    assert!(result.simulations > 0);
}

#[test]
fn test_mcts_finds_winning_move() {
    // Set up a position where X can win immediately
    // Board:
    // X | X | _
    // O | O | _
    // _ | _ | _
    //
    // X should play position 2 to win

    let mut ctx = setup_tictactoe();
    let evaluator = UniformEvaluator::new();
    let config = MctsConfig::for_testing().with_simulations(200);

    // Start fresh
    let reset = ctx.reset(42, &[]).unwrap();

    // Play moves: X at 0, O at 3, X at 1, O at 4
    let moves = [0u32, 3, 1, 4];
    let mut state = reset.state;
    let mut obs = reset.obs;

    for m in moves {
        let action = m.to_le_bytes().to_vec();
        let step = ctx.step(&state, &action).unwrap();
        state = step.state;
        obs = step.obs;
    }

    // Now it's X's turn, position 2 wins
    let legal_mask = LegalMask::from_u64(0b111100100, 9); // positions 2, 5, 6, 7, 8 are legal

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let result = run_mcts(
        &mut ctx, &evaluator, config, state, obs, legal_mask, &mut rng,
    )
    .unwrap();

    // With enough simulations, MCTS should find the winning move
    // Note: With uniform evaluator, it may not always find it, but should favor it
    assert!(result.action < 9);
    assert!(result.policy[2] > 0.0); // Position 2 should have some probability
}

#[test]
fn test_mcts_winning_move_has_positive_value() {
    // Critical test: verify that an immediate winning move results in
    // positive value at the root (not negative due to sign bugs)
    //
    // Board setup where X can win immediately:
    // X | X | _
    // O | O | _
    // _ | _ | _
    //
    // Position 2 is an instant win for X

    let mut ctx = setup_tictactoe();
    let evaluator = UniformEvaluator::new();
    // Use greedy selection (temperature=0) so we deterministically pick best move
    // Increase simulations to give MCTS enough time to discover the winning move
    let config = MctsConfig::for_testing()
        .with_simulations(800)
        .with_temperature(0.0);

    // Start fresh
    let reset = ctx.reset(42, &[]).unwrap();

    // Play moves: X at 0, O at 3, X at 1, O at 4
    let moves = [0u32, 3, 1, 4];
    let mut state = reset.state;
    let mut obs = reset.obs;

    for m in moves {
        let action = m.to_le_bytes().to_vec();
        let step = ctx.step(&state, &action).unwrap();
        state = step.state;
        obs = step.obs;
    }

    // Extract legal mask from the observation (the authoritative source)
    let legal_mask = ctx.metadata().legal_mask_from_obs(&obs);

    // Verify position 2 is legal
    assert!(
        legal_mask.is_legal(2),
        "Position 2 should be legal, mask={:?}",
        legal_mask
    );

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut search = MctsSearch::new(&mut ctx, &evaluator, config, state, obs, legal_mask).unwrap();
    let result = search.run(&mut rng).unwrap();

    // Check the tree directly
    let tree = search.tree();
    let root = tree.get(tree.root());

    // Debug output: log all children's values
    for (action, child_id) in &root.children {
        let child = tree.get(*child_id);
        let q_value = if child.visit_count > 0 {
            child.value_sum / child.visit_count as f32
        } else {
            0.0
        };
        trace!(
            action,
            terminal = child.is_terminal,
            visits = child.visit_count,
            value_sum = child.value_sum,
            q_value,
            prior = child.prior,
            "Child node stats"
        );
    }

    // Find the child for action 2
    let winning_child_id = root
        .children
        .iter()
        .find(|(action, _)| *action == 2)
        .map(|(_, id)| *id)
        .expect("Child for action 2 should exist");

    let winning_child = tree.get(winning_child_id);

    // The winning child should be terminal
    assert!(
        winning_child.is_terminal,
        "Position 2 should lead to terminal state (X wins)"
    );

    // terminal_value should be -1 (negated +1 reward for winner)
    assert!(
        (winning_child.terminal_value - (-1.0)).abs() < 0.01,
        "Terminal value should be -1.0 (negated win), got {}",
        winning_child.terminal_value
    );

    // The winning child should have been visited
    assert!(
        winning_child.visit_count > 0,
        "Winning action should have visits"
    );

    // The root's mean_value should be positive (we have a winning move)
    trace!(
        visits = root.visit_count,
        value_sum = root.value_sum,
        mean_value = root.mean_value(),
        "Root node stats"
    );

    // The winning move (position 2) should be selected
    assert_eq!(result.action, 2, "Should select winning move at position 2");

    // The root value should be POSITIVE because we have a winning move
    assert!(
        result.value > 0.0,
        "Root value should be positive when winning move exists, got {}",
        result.value
    );

    // Policy should strongly favor position 2
    assert!(
        result.policy[2] > 0.5,
        "Policy should favor winning move, got {}",
        result.policy[2]
    );
}

#[test]
fn test_mcts_generals_257_actions() {
    // Regression test for the u64 mask ceiling: generals_8x8 has 257 actions
    // (256 moves + wait), which cannot fit in a u64 mask or in info_bits.
    engine_games::register_all_games();
    let mut ctx = EngineContext::new("generals_8x8").unwrap();
    let evaluator = UniformEvaluator::new();
    let config = MctsConfig::for_testing().with_simulations(50);

    let reset = ctx.reset(42, &[]).unwrap();
    let meta = ctx.metadata();
    assert_eq!(meta.num_actions, 257);
    let legal_mask = meta.legal_mask_from_obs(&reset.obs);
    // Wait (action 256, past the u64 boundary) must be legal at the root
    assert!(legal_mask.is_legal(256));

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let result = run_mcts(
        &mut ctx,
        &evaluator,
        config,
        reset.state,
        reset.obs.clone(),
        legal_mask.clone(),
        &mut rng,
    )
    .unwrap();

    // The chosen action must be legal and the policy confined to legal moves
    assert!(legal_mask.is_legal(result.action as usize));
    assert_eq!(result.policy.len(), 257);
    for (a, &p) in result.policy.iter().enumerate() {
        if p > 0.0 {
            assert!(
                legal_mask.is_legal(a),
                "policy mass on illegal action {}",
                a
            );
        }
    }
    assert!(result.simulations > 0);
}

#[test]
fn test_single_batch_search_spreads_visits() {
    // Regression test for the virtual-loss sign bug: when all simulations
    // fit in one evaluation batch (sims <= eval_batch_size), virtual loss
    // with the wrong sign made every simulation pile onto one child (the
    // last-index tie-winner), producing one-hot policy targets and
    // collapsing self-play training.
    engine_games::register_all_games();
    let mut ctx = EngineContext::new("tictactoe").unwrap();
    let evaluator = UniformEvaluator::new();
    let config = MctsConfig::for_testing()
        .with_simulations(9)
        .with_eval_batch_size(64) // all sims in a single batch
        .with_temperature(1.0); // visit-proportional policy, not greedy

    let reset = ctx.reset(42, &[]).unwrap();
    let legal_mask = LegalMask::from_u64(0b111111111, 9);

    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let result = run_mcts(
        &mut ctx,
        &evaluator,
        config,
        reset.state,
        reset.obs,
        legal_mask,
        &mut rng,
    )
    .unwrap();

    let support = result.policy.iter().filter(|&&p| p > 0.0).count();
    assert!(
        support >= 5,
        "single-batch search must spread visits over children, support={} policy={:?}",
        support,
        result.policy
    );
}

#[test]
fn test_sample_action() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let policy = vec![0.0, 0.5, 0.3, 0.2, 0.0];

    // Sample many times and check distribution
    let mut counts = [0u32; 5];
    for _ in 0..1000 {
        let action = sample_action(&policy, &mut rng).unwrap();
        counts[action as usize] += 1;
    }

    // Action 0 and 4 should never be selected
    assert_eq!(counts[0], 0);
    assert_eq!(counts[4], 0);

    // Action 1 should be most common (~500), action 2 (~300), action 3 (~200)
    assert!(counts[1] > counts[2]);
    assert!(counts[2] > counts[3]);
}

#[test]
fn test_dirichlet_noise() {
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let noise = dirichlet_noise(5, 0.3, &mut rng);

    // Should sum to 1.0
    let sum: f32 = noise.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);

    // All values should be positive
    for &n in &noise {
        assert!(n >= 0.0);
    }
}
