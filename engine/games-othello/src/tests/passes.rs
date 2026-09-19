use super::*;

#[test]
fn test_pass_action() {
    // Black cannot bracket anything at A1; White can capture B1 from A1.
    let mut board = [2; BOARD_SIZE];
    board[0] = 0;
    board[1] = 1;
    let mut state = State {
        board,
        current_player: 1,
        winner: 0,
        pass_count: 0,
    };
    assert_eq!(state.legal_moves(), [PASS_ACTION]);
    let transition = Othello::new()
        .step(&mut state, PASS_ACTION, &mut ChaCha20Rng::seed_from_u64(0))
        .unwrap();
    assert_eq!(state.board, board);
    assert_eq!(state.current_player, 2);
    assert_eq!(state.pass_count, 1);
    assert!(!transition.terminated);
    assert_eq!(transition.actor_reward, 0.0);
    assert_eq!(state.legal_moves(), [0]);
}

#[test]
fn test_two_consecutive_passes_ends_game() {
    // Create a state where both players have no legal moves
    // This should end the game
    let mut state = State {
        board: {
            let mut b = [0u8; BOARD_SIZE];
            // Fill with alternating pattern so no moves possible
            for (i, cell) in b.iter_mut().enumerate() {
                *cell = ((i % 2) + 1) as u8;
            }
            b
        },
        current_player: 1,
        winner: 0,
        pass_count: 0,
    };

    // Both players must pass
    state = state.make_move(PASS_ACTION);
    assert_eq!(state.pass_count, 1);
    assert!(!state.is_done());

    state = state.make_move(PASS_ACTION);
    assert!(state.is_done()); // Two consecutive passes end the game
    assert!(state.winner != 0);
}

#[test]
fn test_game_continues_when_one_player_has_moves() {
    // Bug fix test: Game should NOT end when one player has no moves
    // but the other player still has legal moves available.
    // This was the bug where the game ended prematurely.

    // Test the core logic: after a move, if opponent has no moves,
    // the game should set pass_count = 1 but NOT end the game

    let mut game = Othello::new();
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let (state, _) = game.reset(&mut rng, &[]).unwrap();

    // Verify initial state has moves for both players
    assert!(!state.is_done());

    // The key fix is in the make_move logic:
    // - When a player has no moves, they must pass
    // - Game only ends when pass_count >= 2 (both players passed)
    // - Game does NOT end when only one player has no moves

    // Test pass action increments counter but doesn't end game immediately
    let pass_state = state.make_move(PASS_ACTION);
    assert_eq!(pass_state.pass_count, 1);
    assert!(!pass_state.is_done(), "One pass should not end game");

    // After a regular move, pass_count should reset
    let legal_moves = state.legal_moves();
    assert!(!legal_moves.is_empty());
    let action = legal_moves[0];
    let mut new_state = state.clone();
    let _transition = game.step(&mut new_state, action, &mut rng).unwrap();
    assert_eq!(
        new_state.pass_count, 0,
        "Pass count should reset after board move"
    );
}

#[test]
fn test_winner_by_piece_count() {
    // Set up an endgame position
    let mut state = State {
        board: {
            let mut b = [0u8; BOARD_SIZE];
            // Black has 33, White has 31
            for cell in b.iter_mut().take(33) {
                *cell = 1;
            }
            for cell in b.iter_mut().skip(33) {
                *cell = 2;
            }
            b
        },
        current_player: 1,
        winner: 0,
        pass_count: 2, // Ready to end
    };

    state.determine_winner();
    assert_eq!(state.winner, 1); // Black wins

    // Test draw
    let mut draw_state = State {
        board: {
            let mut b = [0u8; BOARD_SIZE];
            // 32 each
            for cell in b.iter_mut().take(32) {
                *cell = 1;
            }
            for cell in b.iter_mut().skip(32) {
                *cell = 2;
            }
            b
        },
        current_player: 1,
        winner: 0,
        pass_count: 2,
    };

    draw_state.determine_winner();
    assert_eq!(draw_state.winner, 3); // Draw
}

#[test]
fn test_legal_moves_mask_with_pass() {
    // Create a state with no legal board moves
    let mut state = State::new();

    // Fill the board completely - no moves for anyone
    for i in 0..BOARD_SIZE {
        state.board[i] = ((i % 2) + 1) as u8;
    }

    let mask = state.legal_moves_mask();

    // Board mask should be empty
    assert_eq!(mask, 0);
    // But pass should be legal
    assert!(state.is_pass_legal());
}
