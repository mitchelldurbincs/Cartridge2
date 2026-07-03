use super::*;
use rand::SeedableRng;

#[test]
fn test_initial_state() {
    let state = State::new();
    assert_eq!(state.board, [0; BOARD_SIZE]);
    assert_eq!(state.current_player, 1);
    assert_eq!(state.winner, 0);
    assert_eq!(state.column_heights, [0; COLS]);
    assert!(!state.is_done());
}

#[test]
fn test_legal_moves() {
    let state = State::new();
    let legal = state.legal_moves();
    assert_eq!(legal, (0..COLS as u8).collect::<Vec<_>>());
    assert_eq!(state.legal_moves_mask(), 0x7Fu8); // All 7 columns

    // After one move
    let state = state.drop_piece(3); // Center column
    let legal = state.legal_moves();
    assert_eq!(legal.len(), 7); // All columns still available
    assert!(legal.contains(&3));
}

#[test]
fn test_drop_piece() {
    let state = State::new();
    let new_state = state.drop_piece(3); // Red drops in center

    // Piece should be at bottom of column 3
    assert_eq!(new_state.board[State::pos(3, 0)], 1);
    assert_eq!(new_state.column_heights[3], 1);
    assert_eq!(new_state.current_player, 2); // Now Yellow's turn
    assert!(!new_state.is_done());
}

#[test]
fn test_stacking_pieces() {
    let mut state = State::new();

    // Stack pieces in column 0
    for i in 0..ROWS {
        state = state.drop_piece(0);
        assert_eq!(state.column_heights[0], (i + 1) as u8);
    }

    // Column 0 is now full
    assert!(!state.legal_moves().contains(&0));
    assert_eq!(state.legal_moves_mask() & 1, 0);
}

#[test]
fn test_invalid_move_full_column() {
    let mut state = State::new();

    // Fill column 0
    for _ in 0..ROWS {
        state = state.drop_piece(0);
    }

    let before = state.clone();
    let after = state.drop_piece(0); // Try to drop in full column

    // State should be unchanged
    assert_eq!(before, after);
}

#[test]
fn test_horizontal_win() {
    let mut state = State::new();

    // Red: 0, 1, 2, 3 (bottom row)
    // Yellow: 0, 1, 2 (second row)
    state = state.drop_piece(0); // Red at (0,0)
    state = state.drop_piece(0); // Yellow at (0,1)
    state = state.drop_piece(1); // Red at (1,0)
    state = state.drop_piece(1); // Yellow at (1,1)
    state = state.drop_piece(2); // Red at (2,0)
    state = state.drop_piece(2); // Yellow at (2,1)
    state = state.drop_piece(3); // Red at (3,0) - WINS

    assert_eq!(state.winner, 1); // Red wins
    assert!(state.is_done());
    assert!(state.legal_moves().is_empty());
}

#[test]
fn test_vertical_win() {
    let mut state = State::new();

    // Red stacks in column 0, Yellow in column 1
    state = state.drop_piece(0); // Red
    state = state.drop_piece(1); // Yellow
    state = state.drop_piece(0); // Red
    state = state.drop_piece(1); // Yellow
    state = state.drop_piece(0); // Red
    state = state.drop_piece(1); // Yellow
    state = state.drop_piece(0); // Red - WINS

    assert_eq!(state.winner, 1); // Red wins
    assert!(state.is_done());
}

#[test]
fn test_diagonal_win_ascending() {
    let mut state = State::new();

    // Build ascending diagonal for Red: (0,0), (1,1), (2,2), (3,3)
    // Yellow plays defensively in columns 5 and 6 to avoid horizontal wins
    state = state.drop_piece(0); // Red at (0,0)    - Red's turn
    state = state.drop_piece(5); // Yellow at (5,0) - Yellow's turn
    state = state.drop_piece(1); // Red at (1,0)    - Red's turn (need base for col 1)
    state = state.drop_piece(6); // Yellow at (6,0) - Yellow's turn
    state = state.drop_piece(1); // Red at (1,1)    - Red's turn
    state = state.drop_piece(5); // Yellow at (5,1) - Yellow's turn
    state = state.drop_piece(2); // Red at (2,0)    - Red's turn (need base for col 2)
    state = state.drop_piece(6); // Yellow at (6,1) - Yellow's turn
    state = state.drop_piece(2); // Red at (2,1)    - Red's turn
    state = state.drop_piece(5); // Yellow at (5,2) - Yellow's turn
    state = state.drop_piece(2); // Red at (2,2)    - Red's turn
    state = state.drop_piece(6); // Yellow at (6,2) - Yellow's turn
    state = state.drop_piece(3); // Red at (3,0)    - Red's turn (need base for col 3)
    state = state.drop_piece(5); // Yellow at (5,3) - Yellow's turn
    state = state.drop_piece(3); // Red at (3,1)    - Red's turn
    state = state.drop_piece(6); // Yellow at (6,3) - Yellow's turn
    state = state.drop_piece(3); // Red at (3,2)    - Red's turn
    state = state.drop_piece(5); // Yellow at (5,4) - Yellow's turn
    state = state.drop_piece(3); // Red at (3,3)    - Red's turn - WINS

    assert_eq!(state.winner, 1); // Red wins
}

#[test]
fn test_diagonal_win_descending() {
    let mut state = State::new();

    // Build descending diagonal for Red
    // Red at: (3,0), (2,1), (1,2), (0,3)
    state = state.drop_piece(3); // Red at (3,0)
    state = state.drop_piece(2); // Yellow at (2,0)
    state = state.drop_piece(2); // Red at (2,1)
    state = state.drop_piece(1); // Yellow at (1,0)
    state = state.drop_piece(1); // Red at (1,1)
    state = state.drop_piece(0); // Yellow at (0,0)
    state = state.drop_piece(1); // Red at (1,2)
    state = state.drop_piece(0); // Yellow at (0,1)
    state = state.drop_piece(0); // Red at (0,2)
    state = state.drop_piece(4); // Yellow at (4,0)
    state = state.drop_piece(0); // Red at (0,3) - WINS

    assert_eq!(state.winner, 1); // Red wins
}

#[test]
fn test_draw_game() {
    // Creating a draw requires filling all 42 positions without a win
    // This is tricky to construct manually, so we'll test the draw detection logic
    let mut state = State::new();

    // Fill the board in a pattern that creates a draw
    // Pattern that avoids 4-in-a-row:
    // Row 0: R Y R Y R Y R
    // Row 1: R Y R Y R Y R
    // Row 2: Y R Y R Y R Y
    // Row 3: Y R Y R Y R Y
    // Row 4: R Y R Y R Y R
    // Row 5: R Y R Y R Y R

    let pattern = [
        // Column 0: R R Y Y R R
        [1, 1, 2, 2, 1, 1],
        // Column 1: Y Y R R Y Y
        [2, 2, 1, 1, 2, 2],
        // Column 2: R R Y Y R R
        [1, 1, 2, 2, 1, 1],
        // Column 3: Y Y R R Y Y
        [2, 2, 1, 1, 2, 2],
        // Column 4: R R Y Y R R
        [1, 1, 2, 2, 1, 1],
        // Column 5: Y Y R R Y Y
        [2, 2, 1, 1, 2, 2],
        // Column 6: R R Y Y R R
        [1, 1, 2, 2, 1, 1],
    ];

    // Build the board directly
    let mut board = [0u8; BOARD_SIZE];
    for col in 0..COLS {
        for row in 0..ROWS {
            board[State::pos(col, row)] = pattern[col][row];
        }
    }

    state.board = board;
    state.column_heights = [6; COLS];
    state.winner = state.check_winner_at(0, 0); // Check any position

    // Verify it's a draw (no winner but board is full)
    assert_eq!(state.winner, 3);
    assert!(state.is_done());
}

#[test]
fn test_observation_encoding() {
    let state = State::new();
    let obs = observation_from_state(&state);

    // All board positions should be 0 initially
    assert_eq!(obs.board_view, [0.0; BOARD_SIZE * 2]);
    // All columns should be legal
    assert_eq!(obs.legal_moves, [1.0; COLS]);
    // Red should be current player
    assert_eq!(obs.current_player, [1.0, 0.0]);
}

#[test]
fn test_game_trait_implementation() {
    let mut game = Connect4::new();
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let (state, _obs) = game.reset(&mut rng, &[]);
    assert_eq!(state, State::new());

    let action: Action = 3;
    let (_new_obs, reward, done, info) = game.step(&mut state.clone(), action, &mut rng);

    // Should not be done after one move
    assert!(!done);
    // Reward should be 0 for ongoing game
    assert_eq!(reward, 0.0);

    // All columns should still be legal
    assert_eq!(info & 0x7F, 0x7Fu64);
    // Next player should be Yellow (value 2)
    assert_eq!((info >> 16) & 0xF, 2);
}

#[test]
fn test_state_encoding_roundtrip() {
    let mut original_state = State::new();
    original_state = original_state.drop_piece(3);
    original_state = original_state.drop_piece(3);
    original_state = original_state.drop_piece(4);

    let mut buf = Vec::new();
    Connect4::encode_state(&original_state, &mut buf).unwrap();
    let decoded_state = Connect4::decode_state(&buf).unwrap();

    assert_eq!(original_state, decoded_state);
}

#[test]
fn test_action_encoding_roundtrip() {
    for col in 0..COLS as u8 {
        let action: Action = col;

        let mut buf = Vec::new();
        Connect4::encode_action(&action, &mut buf).unwrap();
        let decoded_action = Connect4::decode_action(&buf).unwrap();

        assert_eq!(action, decoded_action);
    }
}

#[test]
fn test_observation_byte_encoding() {
    let mut state = State::new();
    state = state.drop_piece(3);

    let obs = observation_from_state(&state);

    let mut buf = Vec::new();
    Connect4::encode_obs(&obs, &mut buf).unwrap();

    // Should be OBS_SIZE * 4 bytes (OBS_SIZE f32 values)
    assert_eq!(buf.len(), OBS_SIZE * 4);
}

#[test]
fn test_engine_capabilities() {
    let game = Connect4::new();
    let caps = game.capabilities();

    assert_eq!(caps.id.env_id, "connect4");
    assert_eq!(caps.max_horizon, BOARD_SIZE as u32);

    match caps.action_space {
        ActionSpace::Discrete(n) => assert_eq!(n, COLS as u32),
        ref other => {
            panic!("Expected discrete action space, but got {:?}", other);
        }
    }
}

#[test]
fn test_invalid_state_decoding() {
    // Test wrong length
    let buf = vec![1, 2, 3]; // Too short
    let result = Connect4::decode_state(&buf);
    assert!(result.is_err());

    // Test invalid current_player
    let mut buf = vec![0; BOARD_SIZE + 2];
    buf[BOARD_SIZE] = 5; // Invalid player
    let result = Connect4::decode_state(&buf);
    assert!(result.is_err());

    // Test invalid winner
    let mut buf = vec![0; BOARD_SIZE + 2];
    buf[BOARD_SIZE] = 1; // Valid player
    buf[BOARD_SIZE + 1] = 5; // Invalid winner
    let result = Connect4::decode_state(&buf);
    assert!(result.is_err());
}

#[test]
fn test_invalid_action_decoding() {
    // Test wrong length
    let buf = vec![1, 2]; // Too short
    let result = Connect4::decode_action(&buf);
    assert!(result.is_err());

    // Test invalid column
    let buf = (7u32).to_le_bytes().to_vec(); // Column out of bounds
    let result = Connect4::decode_action(&buf);
    assert!(result.is_err());
}

#[test]
fn test_metadata() {
    let game = Connect4::new();
    let meta = game.metadata();

    assert_eq!(meta.env_id, "connect4");
    assert_eq!(meta.display_name, "Connect 4");
    assert_eq!(meta.board_width, COLS);
    assert_eq!(meta.board_height, ROWS);
    assert_eq!(meta.num_actions, COLS);
    assert_eq!(meta.player_count, 2);
}

#[test]
fn test_random_games_invariants() {
    use rand::Rng;

    for seed in 0..20 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut game = Connect4::new();
        let (mut state, _) = game.reset(&mut rng, &[]);

        let mut move_count = 0;
        let max_moves = BOARD_SIZE;

        while !state.is_done() && move_count < max_moves {
            let legal = state.legal_moves();
            assert!(
                !legal.is_empty(),
                "Non-done game must have legal moves (seed={}, moves={})",
                seed,
                move_count
            );

            // Pick random legal move
            let action: Action = legal[rng.gen_range(0..legal.len())];

            let prev_player = state.current_player;
            let (_, reward, done, info) = game.step(&mut state, action, &mut rng);

            move_count += 1;

            // Invariants
            if done {
                assert!(
                    state.winner != 0,
                    "Done game must have winner (seed={})",
                    seed
                );
                assert!(
                    state.legal_moves().is_empty(),
                    "Done game must have no legal moves (seed={})",
                    seed
                );
                // Winner should match reward
                if state.winner == prev_player {
                    assert_eq!(reward, 1.0, "Winner should get +1 reward (seed={})", seed);
                } else if state.winner == 3 {
                    assert_eq!(reward, 0.0, "Draw should give 0 reward (seed={})", seed);
                }
            } else {
                assert_eq!(
                    reward, 0.0,
                    "Ongoing game should have 0 reward (seed={})",
                    seed
                );
                // Player should have switched
                assert_ne!(
                    state.current_player, prev_player,
                    "Player should switch after move (seed={})",
                    seed
                );
            }

            // Info bits should match state
            let mask_from_info = (info & 0x7F) as u8;
            assert_eq!(
                mask_from_info,
                state.legal_moves_mask(),
                "Info mask should match state (seed={})",
                seed
            );
        }

        // Game should finish within max_moves
        assert!(
            state.is_done() || move_count == max_moves,
            "Game should finish within {} moves (seed={})",
            max_moves,
            seed
        );
    }
}

/// State encoding must roundtrip at every point of a game. This also
/// exercises the column-height reconstruction in `decode_state`, which
/// rebuilds `column_heights` from the raw board bytes.
#[test]
fn test_state_encoding_roundtrip_random_games() {
    use rand::Rng;

    for seed in 0..10u64 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut state = State::new();

        while !state.is_done() {
            let mut buf = Vec::new();
            Connect4::encode_state(&state, &mut buf).unwrap();
            let decoded = Connect4::decode_state(&buf).unwrap();
            assert_eq!(state, decoded, "State should roundtrip (seed={})", seed);

            let legal = state.legal_moves();
            let action = legal[rng.gen_range(0..legal.len())];
            state = state.drop_piece(action);
        }

        // Terminal state must roundtrip too
        let mut buf = Vec::new();
        Connect4::encode_state(&state, &mut buf).unwrap();
        let decoded = Connect4::decode_state(&buf).unwrap();
        assert_eq!(
            state, decoded,
            "Terminal state should roundtrip (seed={})",
            seed
        );
    }
}
