use super::*;

#[test]
fn test_observation_legal_mask_exactly_matches_step_acceptance() {
    let mut state = State::new();

    for ply in 0..32 {
        let legal = Othello::legal_actions(&state).unwrap();
        for action in 0..NUM_ACTIONS as u32 {
            let mut candidate = state.clone();
            let before = candidate.clone();
            let accepted = Othello::new()
                .step(&mut candidate, action, &mut ChaCha20Rng::seed_from_u64(1))
                .is_ok();
            assert_eq!(
                accepted,
                legal.is_legal(action as usize),
                "ply {ply}, action {action} disagrees with the decision mask"
            );
            if !accepted {
                assert_eq!(candidate, before, "rejected action mutated the state");
            }
        }

        if state.is_done() {
            break;
        }
        let action = state.legal_moves()[0];
        Othello::new()
            .step(&mut state, action, &mut ChaCha20Rng::seed_from_u64(2))
            .unwrap();
    }

    // A terminal marker closes the action set even if the underlying board
    // shape would otherwise admit flips.
    state.winner = 1;
    let legal = Othello::legal_actions(&state).unwrap();
    for action in 0..NUM_ACTIONS as u32 {
        let mut candidate = state.clone();
        assert!(!legal.is_legal(action as usize));
        assert!(Othello::new()
            .step(&mut candidate, action, &mut ChaCha20Rng::seed_from_u64(3),)
            .is_err());
        assert_eq!(candidate, state, "rejected terminal action mutated state");
    }
}

#[test]
fn test_game_metadata() {
    let game = Othello::new();
    let metadata = game.metadata();

    assert_eq!(metadata.id, "othello");
    let board = metadata.require_board().unwrap();
    assert_eq!(board.width, 8);
    assert_eq!(board.height, 8);
    assert_eq!(board.players.len(), 2);
    assert_eq!(
        board.renderer,
        engine_core::board_profile::BoardRenderer::Grid
    );
}

#[test]
fn test_engine_id() {
    let game = Othello::new();
    let id = game.engine_id();

    assert_eq!(id.env_id, "othello");
    assert!(!id.build_id.is_empty());
}

#[test]
fn test_capabilities() {
    let game = Othello::new();
    let caps = game.capabilities();

    assert_eq!(
        caps.action_space(AgentId(1)),
        Some(&ActionSpace::Discrete { size: 65 })
    );
    assert_eq!(caps.max_horizon, Some(64));
}

#[test]
fn test_reset_and_step() {
    let mut game = Othello::new();
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let (mut state, _obs) = game.reset(&mut rng, &[]).unwrap();

    assert_eq!(state.current_player, 1);
    assert!(!state.is_done());

    // Get a legal move and make it
    let legal = state.legal_moves();
    assert!(!legal.is_empty());
    let action = legal[0];

    let transition = game.step(&mut state, action, &mut rng).unwrap();

    // The position we moved to should now show as occupied (not legal).
    assert!(!Othello::legal_actions(&state)
        .unwrap()
        .is_legal(action as usize));
    assert_eq!(transition.actor_reward, 0.0); // No winner yet
    assert!(!transition.terminated);
    assert_eq!(state.current_player, 2); // Turn switched
}

#[test]
fn test_full_game() {
    let mut game = Othello::new();
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let (state, _) = game.reset(&mut rng, &[]).unwrap();

    // Play random moves until game ends
    let mut move_count = 0;
    let mut current_state = state;
    while !current_state.is_done() && move_count < 100 {
        let legal = current_state.legal_moves();
        if legal.is_empty() {
            break;
        }

        // Pick first legal move
        let action = legal[0];
        let transition = game.step(&mut current_state, action, &mut rng).unwrap();
        move_count += 1;

        if transition.terminated {
            break;
        }
    }

    // Game should have ended or we hit move limit
    assert!(move_count < 100, "Game didn't end within 100 moves");

    // Winner should be determined
    assert!(current_state.winner != 0 || current_state.is_done());
}

#[test]
fn test_random_games_invariants() {
    use rand::Rng;

    let mut rng = ChaCha20Rng::seed_from_u64(12345);

    for _ in 0..100 {
        let mut game = Othello::new();
        let (mut state, _) = game
            .reset(&mut ChaCha20Rng::seed_from_u64(rng.gen()), &[])
            .unwrap();

        // Play random game
        for _ in 0..200 {
            if state.is_done() {
                break;
            }

            let legal = state.legal_moves();
            if legal.is_empty() {
                break;
            }

            let action = legal[rng.gen::<usize>() % legal.len()];
            let transition = game
                .step(&mut state, action, &mut ChaCha20Rng::seed_from_u64(0))
                .unwrap();

            if transition.terminated {
                break;
            }
        }

        // Invariants:
        // 1. Board only has values 0, 1, 2
        for &cell in &state.board {
            assert!(cell <= 2);
        }

        // 2. Winner is valid
        assert!(state.winner <= 3);

        // 3. Pass count is valid
        assert!(state.pass_count <= 2);

        // 4. Current player is valid
        assert!(state.current_player == 1 || state.current_player == 2);
    }
}
