use super::*;

#[test]
fn test_state_encoding_roundtrip() {
    let original = State::new();

    let mut encoded = Vec::new();
    Othello::encode_state(&original, &mut encoded).unwrap();

    // Should be 67 bytes: 64 board + 1 player + 1 winner + 1 pass
    assert_eq!(encoded.len(), 67);

    let decoded = Othello::decode_state(&encoded).unwrap();

    assert_eq!(original.board, decoded.board);
    assert_eq!(original.current_player, decoded.current_player);
    assert_eq!(original.winner, decoded.winner);
    assert_eq!(original.pass_count, decoded.pass_count);
}

#[test]
fn test_action_encoding_roundtrip() {
    for action in [0u32, 32, 63, PASS_ACTION] {
        let mut encoded = Vec::new();
        Othello::encode_action(&action, &mut encoded).unwrap();

        assert_eq!(encoded.len(), 4);

        let decoded = Othello::decode_action(&encoded).unwrap();
        assert_eq!(action, decoded);
    }
}

#[test]
fn test_invalid_action_encoding() {
    let mut encoded = Vec::new();
    let result = Othello::encode_action(&65, &mut encoded);
    assert!(result.is_err());
}

#[test]
fn test_decode_invalid_length() {
    // State too short
    let result = Othello::decode_state(&[0u8; 10]);
    assert!(result.is_err());

    // Action too short
    let result = Othello::decode_action(&[0u8; 2]);
    assert!(result.is_err());
}

#[test]
fn test_decode_invalid_state_data() {
    // Invalid player
    let mut buf = vec![0u8; 67];
    buf[64] = 3; // Invalid player
    let result = Othello::decode_state(&buf);
    assert!(result.is_err());

    // Invalid winner (current_player must be valid so the winner check is reached)
    let mut buf = vec![0u8; 67];
    buf[64] = 1; // Valid player
    buf[65] = 5; // Invalid winner
    let result = Othello::decode_state(&buf);
    assert!(result.is_err());

    // Invalid pass count (player and winner valid so the pass check is reached)
    let mut buf = vec![0u8; 67];
    buf[64] = 1; // Valid player
    buf[66] = 3; // Invalid pass count
    let result = Othello::decode_state(&buf);
    assert!(result.is_err());

    // Invalid cell value (other fields valid so the cell check is reached)
    let mut buf = vec![0u8; 67];
    buf[0] = 5; // Invalid cell
    buf[64] = 1; // Valid player
    let result = Othello::decode_state(&buf);
    assert!(result.is_err());
}

#[test]
fn test_observation_encoding() {
    let state = State::new();
    let obs = observation_from_state(&state).unwrap();

    let mut encoded = Vec::new();
    Othello::encode_observation(&obs, &mut encoded).unwrap();

    // Two player-relative 8x8 planes.
    assert_eq!(encoded.len(), 128 * 4);
}

/// State encoding must roundtrip at every point of a game, not just the
/// initial position (mid-game boards, flipped pieces, terminal states).
#[test]
fn test_state_encoding_roundtrip_random_games() {
    use rand::Rng;

    for seed in 0..10u64 {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut state = State::new();

        for _ in 0..200 {
            let mut buf = Vec::new();
            Othello::encode_state(&state, &mut buf).unwrap();
            let decoded = Othello::decode_state(&buf).unwrap();
            assert_eq!(state, decoded, "State should roundtrip (seed={})", seed);

            if state.is_done() {
                break;
            }

            let legal = state.legal_moves();
            let action = legal[rng.gen_range(0..legal.len())];
            state = state.make_move(action);
        }
    }
}

/// A state with a pending pass (pass_count = 1) must roundtrip, since
/// pass_count is part of the encoded state.
#[test]
fn test_state_encoding_roundtrip_pass_state() {
    let state = State::new().make_move(PASS_ACTION);
    assert_eq!(state.pass_count, 1);

    let mut buf = Vec::new();
    Othello::encode_state(&state, &mut buf).unwrap();
    let decoded = Othello::decode_state(&buf).unwrap();
    assert_eq!(state, decoded);

    // Second pass ends the game; the terminal state must roundtrip too.
    let terminal = state.make_move(PASS_ACTION);
    assert!(terminal.is_done());

    let mut buf = Vec::new();
    Othello::encode_state(&terminal, &mut buf).unwrap();
    let decoded = Othello::decode_state(&buf).unwrap();
    assert_eq!(terminal, decoded);
}
