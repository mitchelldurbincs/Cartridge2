use super::*;
use rand::SeedableRng;

#[test]
fn test_initial_state() {
    let state = State::new();
    assert_eq!(state.board, [0; 9]);
    assert_eq!(state.current_player, 1);
    assert_eq!(state.winner, 0);
    assert!(!state.is_done());
}

#[test]
fn test_legal_moves() {
    let state = State::new();
    let legal = state.legal_moves();
    assert_eq!(legal, (0..9).collect::<Vec<_>>());
    assert_eq!(state.legal_moves_mask(), 0x1FFu16);

    // After one move
    let state = state.make_move(4); // Center
    let legal = state.legal_moves();
    assert_eq!(legal.len(), 8);
    assert!(!legal.contains(&4));
    assert_eq!(state.legal_moves_mask(), 0x1FFu16 & !(1u16 << 4));
}

#[test]
fn test_make_move() {
    let state = State::new();
    let new_state = state.make_move(4); // X places in center

    assert_eq!(new_state.board[4], 1);
    assert_eq!(new_state.current_player, 2); // Now O's turn
    assert!(!new_state.is_done());
}

#[test]
fn test_invalid_move() {
    let state = State::new();
    let state_with_move = state.make_move(4);

    // Try to place in same position
    let invalid_state = state_with_move.make_move(4);
    assert_eq!(invalid_state, state_with_move); // Should be unchanged
}

#[test]
fn test_winning_game() {
    let mut state = State::new();

    // X wins with top row
    state = state.make_move(0); // X
    state = state.make_move(3); // O
    state = state.make_move(1); // X
    state = state.make_move(4); // O
    state = state.make_move(2); // X wins

    assert_eq!(state.winner, 1);
    assert!(state.is_done());
    assert!(state.legal_moves().is_empty());
}

#[test]
fn test_draw_game() {
    // Create a draw state manually since getting the exact move sequence is tricky
    // Board: X O X / O X O / O X O
    let state = State {
        board: [1, 2, 1, 2, 1, 2, 2, 1, 2], // X=1, O=2
        current_player: 1,                  // Doesn't matter since game is over
        winner: 3,                          // This should be detected as a draw
    };

    // Verify this is actually a draw by checking the game logic
    let detected_winner = State::check_winner(&state.board);
    assert_eq!(detected_winner, 3); // Should be draw
    assert!(state.is_done());
}

#[test]
fn test_observation_encoding() {
    let state = State::new();
    let obs = observation_from_state(&state).unwrap();

    // All board positions should be 0 initially
    assert_eq!(obs.board_view, [0.0; 18]);
    let legal = TicTacToe::legal_actions(&state).unwrap();
    assert_eq!(legal.count_ones(), 9);
}

#[test]
fn test_game_trait_implementation() {
    let mut game = TicTacToe::new();
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    let (state, _obs) = game.reset(&mut rng, &[]).unwrap();
    assert_eq!(state, State::new());

    let action: Action = 4;
    let transition = game.step(&mut state.clone(), action, &mut rng).unwrap();

    // Should not be done after one move
    assert!(!transition.terminated);
    // Reward should be 0 for ongoing game
    assert_eq!(transition.actor_reward, 0.0);

    assert_eq!(transition.observation.board_view[9 + 4], 1.0);
    let legal = TicTacToe::legal_actions(&state.make_move(4)).unwrap();
    assert!(!legal.is_legal(4));
}

#[test]
fn test_state_encoding_roundtrip() {
    let original_state = State {
        board: [1, 0, 2, 0, 1, 0, 2, 0, 0],
        current_player: 2,
        winner: 0,
    };

    let mut buf = Vec::new();
    TicTacToe::encode_state(&original_state, &mut buf).unwrap();
    let decoded_state = TicTacToe::decode_state(&buf).unwrap();

    assert_eq!(original_state, decoded_state);
}

#[test]
fn test_action_encoding_roundtrip() {
    let action: Action = 5;

    let mut buf = Vec::new();
    TicTacToe::encode_action(&action, &mut buf).unwrap();
    let decoded_action = TicTacToe::decode_action(&buf).unwrap();

    assert_eq!(action, decoded_action);
}

#[test]
fn test_observation_byte_encoding() {
    let state = State {
        board: [1, 0, 2, 0, 0, 0, 0, 0, 0],
        current_player: 2,
        winner: 0,
    };
    let obs = observation_from_state(&state).unwrap();

    let mut buf = Vec::new();
    TicTacToe::encode_observation(&obs, &mut buf).unwrap();

    // Two 3x3 player-relative planes.
    assert_eq!(buf.len(), 18 * 4);
}

#[test]
fn test_engine_capabilities() {
    let game = TicTacToe::new();
    let caps = game.capabilities();

    assert_eq!(caps.id.env_id, "tictactoe");
    assert_eq!(caps.max_horizon, Some(9));

    match caps.action_space(AgentId(1)) {
        Some(ActionSpace::Discrete { size }) => assert_eq!(*size, 9),
        ref other => {
            panic!("Expected discrete action space, but got {:?}", other);
        }
    }
}

#[test]
fn test_invalid_state_decoding() {
    // Test wrong length
    let buf = vec![1, 2, 3]; // Too short
    let result = TicTacToe::decode_state(&buf);
    assert!(result.is_err());

    // Test invalid current_player
    let mut buf = vec![0; 11];
    buf[9] = 5; // Invalid player
    let result = TicTacToe::decode_state(&buf);
    assert!(result.is_err());

    // Test invalid winner
    let mut buf = vec![0; 11];
    buf[9] = 1; // Valid player
    buf[10] = 5; // Invalid winner
    let result = TicTacToe::decode_state(&buf);
    assert!(result.is_err());
}

#[test]
fn test_invalid_action_decoding() {
    // Test wrong length
    let buf = vec![1, 2]; // Too short
    let result = TicTacToe::decode_action(&buf);
    assert!(result.is_err());

    // Test invalid position
    let buf = 9u32.to_le_bytes().to_vec(); // Position out of bounds
    let result = TicTacToe::decode_action(&buf);
    assert!(result.is_err());
}

// =========================================================================
// Property tests for TicTacToe rules
// =========================================================================

/// All 8 winning lines should be detected correctly
#[test]
fn test_all_winning_lines() {
    // Rows
    let row_lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8]];
    // Columns
    let col_lines = [[0, 3, 6], [1, 4, 7], [2, 5, 8]];
    // Diagonals
    let diag_lines = [[0, 4, 8], [2, 4, 6]];

    let all_lines: Vec<[usize; 3]> = row_lines
        .iter()
        .chain(col_lines.iter())
        .chain(diag_lines.iter())
        .copied()
        .collect();

    assert_eq!(all_lines.len(), 8, "Should have 8 winning lines");

    for (line_idx, line) in all_lines.iter().enumerate() {
        // Test X wins on this line
        let mut board_x = [0u8; 9];
        for &pos in line {
            board_x[pos] = 1; // X
        }
        let winner = State::check_winner(&board_x);
        assert_eq!(winner, 1, "X should win on line {}: {:?}", line_idx, line);

        // Test O wins on this line
        let mut board_o = [0u8; 9];
        for &pos in line {
            board_o[pos] = 2; // O
        }
        let winner = State::check_winner(&board_o);
        assert_eq!(winner, 2, "O should win on line {}: {:?}", line_idx, line);
    }
}

/// Legal move mask should match legal_moves vector
#[test]
fn test_legal_moves_mask_consistency() {
    // Test various board configurations
    let boards = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0], // Empty
        [1, 0, 0, 0, 0, 0, 0, 0, 0], // One move
        [1, 2, 1, 2, 0, 0, 0, 0, 0], // Four moves
        [1, 2, 1, 2, 1, 2, 0, 0, 0], // Six moves
        [1, 2, 1, 2, 1, 2, 2, 1, 0], // Eight moves
    ];

    for board in &boards {
        let state = State {
            board: *board,
            current_player: 1,
            winner: 0,
        };

        let legal_vec = state.legal_moves();
        let legal_mask = state.legal_moves_mask();

        // Verify mask has correct bits set
        for pos in 0..9u8 {
            let is_in_vec = legal_vec.contains(&pos);
            let is_in_mask = (legal_mask & (1u16 << pos)) != 0;
            assert_eq!(
                is_in_vec, is_in_mask,
                "Mismatch at pos {} for board {:?}",
                pos, board
            );
        }

        // Verify counts match
        assert_eq!(
            legal_vec.len(),
            legal_mask.count_ones() as usize,
            "Count mismatch for board {:?}",
            board
        );
    }
}

#[test]
fn test_observation_legal_mask_exactly_matches_step_acceptance() {
    let states = [
        State::new(),
        State::new().make_move(4).make_move(0).make_move(8),
        State::new()
            .make_move(0)
            .make_move(3)
            .make_move(1)
            .make_move(4)
            .make_move(2),
    ];

    for state in states {
        let legal = TicTacToe::legal_actions(&state).unwrap();
        for action in 0..9u8 {
            let mut candidate = state;
            let before = candidate;
            let accepted = TicTacToe::new()
                .step(&mut candidate, action, &mut ChaCha20Rng::seed_from_u64(1))
                .is_ok();
            assert_eq!(
                accepted,
                legal.is_legal(action as usize),
                "action {action} disagrees with the decision mask for {state:?}"
            );
            if !accepted {
                assert_eq!(candidate, before, "rejected action mutated the state");
            }
        }
    }
}

/// Draw detection: full board with no winner
#[test]
fn test_draw_detection_comprehensive() {
    // Several known draw configurations
    let draw_boards = [
        [1, 2, 1, 1, 2, 2, 2, 1, 1], // X O X / X O O / O X X
        [1, 2, 1, 2, 1, 1, 2, 1, 2], // X O X / O X X / O X O
        [2, 1, 2, 2, 1, 1, 1, 2, 2], // O X O / O X X / X O O
    ];

    for board in &draw_boards {
        let winner = State::check_winner(board);
        assert_eq!(winner, 3, "Should detect draw for board {:?}", board);

        let state = State {
            board: *board,
            current_player: 1,
            winner,
        };
        assert!(state.is_done());
        assert!(state.legal_moves().is_empty());
        assert_eq!(state.legal_moves_mask(), 0);
    }
}

/// Reward symmetry: X winning gives +1 to X, -1 to O (and vice versa)
#[test]
fn test_reward_symmetry() {
    let mut game = TicTacToe::new();
    let mut rng = ChaCha20Rng::seed_from_u64(123);

    // Play game where X wins (top row)
    let (mut state, _) = game.reset(&mut rng, &[]).unwrap();

    // X at 0
    let transition = game.step(&mut state, 0, &mut rng).unwrap();
    assert_eq!(transition.actor_reward, 0.0); // No winner yet

    // O at 3
    let transition = game.step(&mut state, 3, &mut rng).unwrap();
    assert_eq!(transition.actor_reward, 0.0);

    // X at 1
    let transition = game.step(&mut state, 1, &mut rng).unwrap();
    assert_eq!(transition.actor_reward, 0.0);

    // O at 4
    let transition = game.step(&mut state, 4, &mut rng).unwrap();
    assert_eq!(transition.actor_reward, 0.0);

    // X at 2 - X wins!
    let transition = game.step(&mut state, 2, &mut rng).unwrap();
    assert!(transition.terminated);
    assert_eq!(
        transition.actor_reward, 1.0,
        "X (previous player) should get +1 for winning"
    );
}

/// No moves allowed on finished game
#[test]
fn test_no_moves_after_game_over() {
    // Create a state where X has won
    let state = State {
        board: [1, 1, 1, 2, 2, 0, 0, 0, 0],
        current_player: 2, // Doesn't matter
        winner: 1,         // X won
    };

    assert!(state.is_done());
    assert!(state.legal_moves().is_empty());
    assert_eq!(state.legal_moves_mask(), 0);

    // Attempting a move should return unchanged state
    let new_state = state.make_move(5);
    assert_eq!(new_state, state);
}

// =========================================================================
// Encode/decode roundtrip tests
// =========================================================================

/// Test state encoding roundtrip for all valid states
#[test]
fn test_state_encoding_roundtrip_comprehensive() {
    let test_states = [
        State::new(),
        State {
            board: [1, 0, 0, 0, 0, 0, 0, 0, 0],
            current_player: 2,
            winner: 0,
        },
        State {
            board: [1, 2, 1, 2, 1, 2, 0, 0, 0],
            current_player: 1,
            winner: 0,
        },
        State {
            board: [1, 1, 1, 2, 2, 0, 0, 0, 0],
            current_player: 2,
            winner: 1,
        },
        State {
            board: [1, 2, 1, 1, 2, 2, 2, 1, 1],
            current_player: 1,
            winner: 3,
        },
    ];

    for original in &test_states {
        let mut buf = Vec::new();
        TicTacToe::encode_state(original, &mut buf).expect("encode should succeed");
        assert_eq!(buf.len(), 11, "State should encode to 11 bytes");

        let decoded = TicTacToe::decode_state(&buf).expect("decode should succeed");
        assert_eq!(original, &decoded, "Roundtrip should preserve state");
    }
}

/// Test action encoding roundtrip for all 9 positions
#[test]
fn test_action_encoding_roundtrip_all_positions() {
    for pos in 0..9u8 {
        let action: Action = pos;
        let mut buf = Vec::new();
        TicTacToe::encode_action(&action, &mut buf).expect("encode should succeed");
        assert_eq!(buf.len(), 4, "Action should encode to 4 bytes (u32)");

        let decoded = TicTacToe::decode_action(&buf).expect("decode should succeed");
        assert_eq!(
            action, decoded,
            "Roundtrip should preserve action at pos {}",
            pos
        );
    }
}

/// Test observation encoding roundtrip
#[test]
fn test_observation_encoding_roundtrip() {
    let states = [
        State::new(),
        State {
            board: [1, 2, 0, 0, 1, 0, 0, 0, 2],
            current_player: 1,
            winner: 0,
        },
    ];

    for state in &states {
        let obs = observation_from_state(state).unwrap();
        let mut buf = Vec::new();
        TicTacToe::encode_observation(&obs, &mut buf).expect("encode should succeed");
        assert_eq!(
            buf.len(),
            72,
            "Observation should encode to 72 bytes (18 * 4)"
        );

        // Decode manually and verify
        let decoded_floats: Vec<f32> = buf
            .chunks(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(decoded_floats.len(), 18);

        // Verify board_view
        for (i, &decoded_val) in decoded_floats.iter().enumerate().take(18) {
            assert_eq!(decoded_val, obs.board_view[i], "board_view[{}] mismatch", i);
        }
    }
}

// =========================================================================
// Fuzz-style tests with random seeds/actions
// =========================================================================

/// Play many random games and verify invariants hold
/// (mirrors games-connect4's test of the same name; the game-specific
/// constants — legal-mask width, max moves — differ)
#[test]
fn test_random_games_invariants() {
    use rand::Rng;

    for seed in 0..50 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut game = TicTacToe::new();
        let (mut state, _) = game.reset(&mut rng, &[]).unwrap();

        let mut move_count = 0;
        let max_moves = 9; // TicTacToe has at most 9 moves

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
            let transition = game.step(&mut state, action, &mut rng).unwrap();
            let reward = transition.actor_reward;
            let done = transition.terminated;

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

            let legal = TicTacToe::legal_actions(&state).unwrap();
            let encoded = legal
                .iter_ones()
                .fold(0u16, |mask, index| mask | (1u16 << index));
            assert_eq!(
                encoded,
                state.legal_moves_mask(),
                "Decision mask should match state (seed={})",
                seed
            );
        }

        // Game should finish within 9 moves
        assert!(
            state.is_done(),
            "Game should finish within 9 moves (seed={})",
            seed
        );
    }
}

/// Verify encode/decode doesn't panic for random valid states
#[test]
fn test_encoding_no_panic_random_states() {
    use rand::Rng;

    for seed in 0..100 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Generate random board (not necessarily valid game state)
        let mut board = [0u8; 9];
        for cell in board.iter_mut() {
            *cell = rng.gen_range(0..=2);
        }
        let current_player = rng.gen_range(1..=2);
        let winner = State::check_winner(&board);

        let state = State {
            board,
            current_player,
            winner,
        };

        // Encode should not panic
        let mut buf = Vec::new();
        let encode_result = TicTacToe::encode_state(&state, &mut buf);
        assert!(
            encode_result.is_ok(),
            "Encode should succeed for seed {}",
            seed
        );

        // Decode should roundtrip
        let decode_result = TicTacToe::decode_state(&buf);
        assert!(
            decode_result.is_ok(),
            "Decode should succeed for seed {}",
            seed
        );
        assert_eq!(state, decode_result.unwrap());
    }
}

/// Test that invalid action positions are rejected
#[test]
fn test_invalid_action_positions() {
    for invalid_pos in [9, 10, 100, 255] {
        let buf = (invalid_pos as u32).to_le_bytes().to_vec();
        let result = TicTacToe::decode_action(&buf);
        assert!(
            result.is_err(),
            "Position {} should be invalid",
            invalid_pos
        );
    }
}
