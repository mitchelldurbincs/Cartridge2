use super::*;

const RAYS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

fn ray_square(dc: isize, dr: isize, distance: isize) -> usize {
    let col = match dc {
        -1 => 7,
        0 => 3,
        _ => 0,
    };
    let row = match dr {
        -1 => 7,
        0 => 3,
        _ => 0,
    };
    State::pos(
        (col + dc * distance) as usize,
        (row + dr * distance) as usize,
    )
}

#[test]
fn bracketed_runs_flip_exactly_the_opponents_in_each_direction() {
    for player in [1, 2] {
        for (dc, dr) in RAYS {
            for length in 1..=6 {
                let mut board = [0; BOARD_SIZE];
                let mut expected = board;
                for distance in 1..=length {
                    let square = ray_square(dc, dr, distance);
                    board[square] = opponent(player);
                    expected[square] = player;
                }
                let closing = ray_square(dc, dr, length + 1);
                board[closing] = player;
                expected[closing] = player;
                let action = ray_square(dc, dr, 0) as u32;
                expected[action as usize] = player;
                let state = State {
                    board,
                    current_player: player,
                    winner: 0,
                    pass_count: 1,
                };
                assert!(state.legal_moves().contains(&action));
                let after = state.make_move(action);
                assert_eq!(
                    after.board, expected,
                    "player {player}, ray ({dc}, {dr}), length {length}"
                );
                assert_eq!(after.current_player, opponent(player));
                assert_eq!(after.pass_count, 0);
            }
        }
    }
}

#[test]
fn unbracketed_runs_and_occupied_destinations_are_not_moves() {
    for player in [1, 2] {
        for (dc, dr) in RAYS {
            let enemy = opponent(player);
            for (case, destination_owner, ray) in [
                ("empty termination", 0, vec![enemy]),
                ("gap before bracket", 0, vec![enemy, 0, player]),
                ("adjacent own disc", 0, vec![player, enemy, player]),
                ("board-edge termination", 0, vec![enemy; 7]),
                ("occupied destination", player, vec![enemy, player]),
            ] {
                let mut board = [0; BOARD_SIZE];
                board[ray_square(dc, dr, 0)] = destination_owner;
                for (index, owner) in ray.into_iter().enumerate() {
                    board[ray_square(dc, dr, index as isize + 1)] = owner;
                }
                let action = ray_square(dc, dr, 0) as u32;
                let state = State {
                    board,
                    current_player: player,
                    winner: 0,
                    pass_count: 0,
                };
                assert!(
                    !state.legal_moves().contains(&action),
                    "ray ({dc}, {dr}), case {case}"
                );
                assert_eq!(state.make_move(action), state);
            }
        }
    }
}

#[test]
fn captures_do_not_wrap_across_rows() {
    for player in [1, 2] {
        for (destination, adjacent, closing) in [(7, 8, 9), (8, 7, 6), (8, 15, 22), (15, 8, 1)] {
            let mut board = [0; BOARD_SIZE];
            board[adjacent] = opponent(player);
            board[closing] = player;
            let state = State {
                board,
                current_player: player,
                winner: 0,
                pass_count: 0,
            };
            assert!(!state.legal_moves().contains(&destination));
            assert_eq!(state.make_move(destination), state);
        }
    }
}

#[test]
fn test_initial_state() {
    let state = State::new();

    // Standard Othello starting position
    assert_eq!(state.board[State::pos(3, 3)], 1); // Black at D4
    assert_eq!(state.board[State::pos(4, 4)], 1); // Black at E5
    assert_eq!(state.board[State::pos(3, 4)], 2); // White at E4
    assert_eq!(state.board[State::pos(4, 3)], 2); // White at D5

    // All other cells are empty
    assert_eq!(state.board[State::pos(0, 0)], 0);
    assert_eq!(state.board[State::pos(7, 7)], 0);

    assert_eq!(state.current_player, 1); // Black goes first
    assert_eq!(state.winner, 0);
    assert_eq!(state.pass_count, 0);
    assert!(!state.is_done());
}

#[test]
fn test_initial_legal_moves() {
    assert_eq!(State::new().legal_moves(), [20, 29, 34, 43]);
}

#[test]
fn test_make_move_and_flip() {
    let state = State::new();

    // Get a legal move from the initial state
    let legal = state.legal_moves();
    assert!(
        !legal.is_empty(),
        "Should have legal moves in initial state"
    );

    // Make the first legal move
    let action = legal[0];
    let new_state = state.make_move(action);

    // The position we moved to should now have our piece
    assert_eq!(new_state.board[action as usize], 1); // Black

    // At least one opponent piece should have been flipped
    let (orig_black, orig_white) = state.piece_counts();
    let (new_black, new_white) = new_state.piece_counts();

    // Black should have gained pieces (original + placed + flipped)
    assert!(new_black > orig_black);
    // White should have lost pieces (some flipped to Black)
    assert!(new_white < orig_white);

    assert_eq!(new_state.current_player, 2); // White's turn
    assert!(!new_state.is_done());
}

#[test]
fn test_multi_direction_flip() {
    // Two opponents bracketed by an own disc in every direction from D4.
    // The white/black disc at H8 must remain untouched.
    for player in [1, 2] {
        let mut board = [0; BOARD_SIZE];
        let mut expected = board;
        for (dc, dr) in RAYS {
            for distance in 1..=3 {
                let col = (3 + dc * distance) as usize;
                let row = (3 + dr * distance) as usize;
                board[State::pos(col, row)] = if distance == 3 {
                    player
                } else {
                    opponent(player)
                };
                expected[State::pos(col, row)] = player;
            }
        }
        board[63] = opponent(player);
        expected[63] = opponent(player);
        expected[27] = player;
        let state = State {
            board,
            current_player: player,
            winner: 0,
            pass_count: 0,
        };
        assert!(state.legal_moves().contains(&27));
        let after = state.make_move(27);
        assert_eq!(after.board, expected, "player {player}");
        assert_eq!(after.current_player, opponent(player));
    }
}

#[test]
fn test_invalid_move_unchanged() {
    let state = State::new();

    // Try to move on an occupied square
    let before = state.clone();
    let after = state.make_move(27); // D4 is occupied by Black

    // State should be unchanged
    assert_eq!(before.board, after.board);
    assert_eq!(before.current_player, after.current_player);
}

#[test]
fn test_legal_moves_mask() {
    let state = State::new();
    let mask = state.legal_moves_mask();

    // Should have exactly 4 bits set for Black's opening moves
    assert_eq!(mask.count_ones(), 4);

    // Check that pass is not legal (there are board moves available)
    assert!(!state.is_pass_legal());
}

#[test]
fn test_corner_moves() {
    // Test that moves in corners work correctly
    // Set up position where corner (0,0) is a legal move
    let mut board = [0u8; BOARD_SIZE];

    // Create a diagonal sandwich: White at (1,1) between Black at (0,0)'s perspective
    // Actually, to flip (1,1) by playing at (0,0), we need Black at (2,2)
    // Pattern: (0,0)=empty, (1,1)=White, (2,2)=Black
    board[State::pos(1, 1)] = 2; // White
    board[State::pos(2, 2)] = 1; // Black - this sandwiches White at (1,1)

    let state = State {
        board,
        current_player: 1,
        winner: 0,
        pass_count: 0,
    };

    // Check if corner is legal
    let legal = state.legal_moves();
    assert!(
        legal.contains(&0),
        "Corner (0,0) should be legal - can flip White at (1,1) via diagonal"
    );

    // Make the move and verify
    let new_state = state.make_move(0);
    assert_eq!(new_state.board[0], 1); // Now Black
    assert_eq!(new_state.board[9], 1); // (1,1) flipped to Black - position 9
    assert_eq!(new_state.board[18], 1); // (2,2) still Black - position 18
}
