//! Request types for the web API.

use serde::Deserialize;

/// Which side makes the first move in an interactive game.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FirstPlayer {
    #[default]
    Player,
    Bot,
}

/// Request to start a new game.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NewGameRequest {
    /// Optional for legacy clients; the interactive UI always supplies this.
    #[serde(default)]
    pub expected: Option<super::PositionKey>,
    /// Who plays first: "player" or "bot"
    #[serde(default)]
    pub first: FirstPlayer,
    /// Game to play (e.g., "tictactoe", "connect4")
    #[serde(default)]
    pub game: Option<String>,
}

/// Request to make a move.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoveRequest {
    #[serde(default)]
    pub expected: Option<super::PositionKey>,
    /// Action index to play (game-specific: 0-8 for TicTacToe, 0-6 for
    /// Connect4, 0-256 for Generals — which is why this is not a u8)
    pub position: u32,
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================
    // NewGameRequest Tests
    // ========================================

    #[test]
    fn test_new_game_request_default_first() {
        assert_eq!(FirstPlayer::default(), FirstPlayer::Player);
    }

    #[test]
    fn test_new_game_request_deserialization_empty() {
        let json = r#"{}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();

        // Should use defaults
        assert_eq!(request.first, FirstPlayer::Player);
        assert!(request.game.is_none());
    }

    #[test]
    fn test_new_game_request_deserialization_with_first() {
        let json = r#"{"first": "bot"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.first, FirstPlayer::Bot);
        assert!(request.game.is_none());
    }

    #[test]
    fn test_new_game_request_deserialization_with_game() {
        let json = r#"{"first": "player", "game": "tictactoe"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.first, FirstPlayer::Player);
        assert_eq!(request.game, Some("tictactoe".to_string()));
    }

    #[test]
    fn test_new_game_request_deserialization_full() {
        let json = r#"{"first": "bot", "game": "connect4"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.first, FirstPlayer::Bot);
        assert_eq!(request.game, Some("connect4".to_string()));
    }

    #[test]
    fn test_new_game_request_valid_first_values() {
        // Test that "player" is valid
        let json = r#"{"first": "player"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.first, FirstPlayer::Player);

        // Test that "bot" is valid
        let json = r#"{"first": "bot"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.first, FirstPlayer::Bot);

        let json = r#"{"first": "invalid"}"#;
        assert!(serde_json::from_str::<NewGameRequest>(json).is_err());
    }

    // ========================================
    // MoveRequest Tests
    // ========================================

    #[test]
    fn test_move_request_deserialization() {
        let json = r#"{"position": 4}"#;
        let request: MoveRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.position, 4);
    }

    #[test]
    fn test_move_request_deserialization_zero() {
        let json = r#"{"position": 0}"#;
        let request: MoveRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.position, 0);
    }

    #[test]
    fn test_move_request_accepts_generals_action_indices() {
        // Generals has 257 actions, so anything that fit in a u8 is not enough.
        let json = r#"{"position": 256}"#;
        let request: MoveRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.position, 256);
    }

    #[test]
    fn test_move_request_valid_positions() {
        // Test typical TicTacToe positions (0-8)
        for pos in 0..9u32 {
            let json = format!(r#"{{"position": {}}}"#, pos);
            let request: MoveRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(request.position, pos);
        }

        // Test typical Connect4 positions (0-6)
        for pos in 0..7u32 {
            let json = format!(r#"{{"position": {}}}"#, pos);
            let request: MoveRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(request.position, pos);
        }
    }

    // ========================================
    // Edge Cases and Error Handling
    // ========================================

    #[test]
    fn test_new_game_request_rejects_extra_fields() {
        let json = r#"{"first": "player", "unknown_field": "value"}"#;
        assert!(serde_json::from_str::<NewGameRequest>(json).is_err());
    }

    #[test]
    fn test_move_request_rejects_extra_fields() {
        let json = r#"{"position": 4, "extra": "ignored"}"#;
        assert!(serde_json::from_str::<MoveRequest>(json).is_err());
    }

    #[test]
    fn test_new_game_request_null_game() {
        // Explicit null should work
        let json = r#"{"first": "bot", "game": null}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.first, FirstPlayer::Bot);
        assert!(request.game.is_none());
    }

    #[test]
    fn test_move_request_missing_field_fails() {
        // Position is required - missing it should fail
        let json = r#"{}"#;
        let result: Result<MoveRequest, _> = serde_json::from_str(json);

        assert!(result.is_err());
    }
}
