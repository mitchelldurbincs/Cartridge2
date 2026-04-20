//! Request types for the web API.

use serde::Deserialize;

/// Request to start a new game.
#[derive(Deserialize)]
pub struct NewGameRequest {
    /// Who plays first: "player" or "bot"
    #[serde(default = "default_first")]
    pub first: String,
    /// Game to play (e.g., "tictactoe", "connect4")
    #[serde(default)]
    pub game: Option<String>,
}

fn default_first() -> String {
    "player".to_string()
}

/// Request to make a move.
#[derive(Deserialize)]
pub struct MoveRequest {
    /// Position or column to play (game-specific: 0-8 for TicTacToe, 0-6 for Connect4)
    pub position: u8,
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
        // The default_first function should return "player"
        assert_eq!(default_first(), "player");
    }

    #[test]
    fn test_new_game_request_deserialization_empty() {
        let json = r#"{}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();

        // Should use defaults
        assert_eq!(request.first, "player");
        assert!(request.game.is_none());
    }

    #[test]
    fn test_new_game_request_deserialization_with_first() {
        let json = r#"{"first": "bot"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.first, "bot");
        assert!(request.game.is_none());
    }

    #[test]
    fn test_new_game_request_deserialization_with_game() {
        let json = r#"{"first": "player", "game": "tictactoe"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.first, "player");
        assert_eq!(request.game, Some("tictactoe".to_string()));
    }

    #[test]
    fn test_new_game_request_deserialization_full() {
        let json = r#"{"first": "bot", "game": "connect4"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.first, "bot");
        assert_eq!(request.game, Some("connect4".to_string()));
    }

    #[test]
    fn test_new_game_request_valid_first_values() {
        // Test that "player" is valid
        let json = r#"{"first": "player"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.first, "player");

        // Test that "bot" is valid
        let json = r#"{"first": "bot"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.first, "bot");

        // Any string value is technically valid (validation happens elsewhere)
        let json = r#"{"first": "invalid"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.first, "invalid");
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
    fn test_move_request_deserialization_max_u8() {
        let json = r#"{"position": 255}"#;
        let request: MoveRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.position, 255);
    }

    #[test]
    fn test_move_request_valid_positions() {
        // Test typical TicTacToe positions (0-8)
        for pos in 0..9u8 {
            let json = format!(r#"{{"position": {}}}"#, pos);
            let request: MoveRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(request.position, pos);
        }

        // Test typical Connect4 positions (0-6)
        for pos in 0..7u8 {
            let json = format!(r#"{{"position": {}}}"#, pos);
            let request: MoveRequest = serde_json::from_str(&json).unwrap();
            assert_eq!(request.position, pos);
        }
    }

    // ========================================
    // Edge Cases and Error Handling
    // ========================================

    #[test]
    fn test_new_game_request_extra_fields_ignored() {
        // Serde ignores unknown fields by default
        let json = r#"{"first": "player", "unknown_field": "value"}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.first, "player");
    }

    #[test]
    fn test_move_request_extra_fields_ignored() {
        let json = r#"{"position": 4, "extra": "ignored"}"#;
        let request: MoveRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.position, 4);
    }

    #[test]
    fn test_new_game_request_null_game() {
        // Explicit null should work
        let json = r#"{"first": "bot", "game": null}"#;
        let request: NewGameRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.first, "bot");
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
