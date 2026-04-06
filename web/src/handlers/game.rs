//! Game-related handlers.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use engine_core::create_game;
use std::sync::Arc;
use std::time::Instant;

use crate::game::GameSession;
use crate::metrics;
use crate::types::{
    GameInfoResponse, GameStateResponse, GamesListResponse, MoveRequest, MoveResponse,
    NewGameRequest,
};
use crate::AppState;

/// List available games.
/// Only returns the currently configured game to prevent users from
/// selecting games that don't match the loaded model.
pub async fn list_games(State(state): State<Arc<AppState>>) -> Json<GamesListResponse> {
    // Only return the current game - users can't play other games
    // since the model is trained for a specific game
    let current_game = state.current_game.read().await;
    Json(GamesListResponse {
        games: vec![current_game.clone()],
    })
}

/// Get metadata for the current game.
/// Only returns info for the currently configured game to ensure
/// the frontend only shows the game the model is trained for.
pub async fn get_game_info(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<GameInfoResponse>, (StatusCode, String)> {
    // Only allow access to the current game
    let current_game = state.current_game.read().await;
    if id != *current_game {
        return Err((
            StatusCode::FORBIDDEN,
            format!(
                "Cannot access game '{}': only the current game '{}' is available",
                id, current_game
            ),
        ));
    }

    let game = create_game(&id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Game not found: {}", id),
        )
    })?;

    let metadata = game.metadata();
    Ok(Json(metadata.into()))
}

/// Get current game state.
pub async fn get_game_state(
    State(state): State<Arc<AppState>>,
) -> Result<Json<GameStateResponse>, (StatusCode, String)> {
    let session = state.session.lock().await;
    Ok(Json(session.to_response()))
}

/// Start a new game.
/// Only allows creating the currently configured game type.
/// Rejects requests to switch to a different game since the model
/// is trained for a specific game.
pub async fn new_game(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NewGameRequest>,
) -> Result<Json<GameStateResponse>, (StatusCode, String)> {
    let mut session = state.session.lock().await;

    // Record metrics for new game
    metrics::GAMES_CREATED.inc();
    metrics::GAMES_ACTIVE.inc();

    // Get the current configured game
    let current_game = state.current_game.read().await.clone();

    // Reject requests trying to switch to a different game
    if let Some(ref requested_game) = req.game {
        if requested_game != &current_game {
            return Err((
                StatusCode::FORBIDDEN,
                format!(
                    "Cannot switch to game '{}': only the current game '{}' is available",
                    requested_game, current_game
                ),
            ));
        }
    }

    // Use the current game (cannot be changed)
    let game_id = current_game;

    // Reset the game with shared evaluator (for hot-reloading)
    *session =
        GameSession::with_evaluator(&game_id, Arc::clone(&state.evaluator)).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create game '{}': {}", game_id, e),
            )
        })?;

    // If bot goes first, bot is player 1, human is player 2
    // If player goes first, human is player 1, bot is player 2
    if req.first == "bot" {
        session.set_human_player(2); // Human plays as O (player 2)

        // Time bot move
        let bot_start = Instant::now();
        session.bot_move().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Bot move failed: {}", e),
            )
        })?;
        metrics::BOT_MOVE_SECONDS.observe(bot_start.elapsed().as_secs_f64());
    } else {
        session.set_human_player(1); // Human plays as X (player 1) - default
    }

    Ok(Json(session.to_response()))
}

/// Make a move (player + bot response).
pub async fn make_move(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MoveRequest>,
) -> Result<Json<MoveResponse>, (StatusCode, String)> {
    let mut session = state.session.lock().await;

    // Check if game is over
    if session.is_game_over() {
        return Err((StatusCode::BAD_REQUEST, "Game is already over".to_string()));
    }

    // Check if it's the human's turn
    if !session.is_human_turn() {
        return Err((StatusCode::BAD_REQUEST, "Not your turn".to_string()));
    }

    // Check if move is legal (this handles position validation based on game type)
    if !session.is_legal_move(req.position) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Illegal move: position/column {} is not valid",
                req.position
            ),
        ));
    }

    // Make player's move
    session.player_move(req.position).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Move failed: {}", e),
        )
    })?;
    metrics::MOVES_PLAYED.inc();

    // If game is not over, bot makes a move
    let bot_move = if !session.is_game_over() {
        let bot_start = Instant::now();
        let pos = session.bot_move().map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Bot move failed: {}", e),
            )
        })?;
        metrics::BOT_MOVE_SECONDS.observe(bot_start.elapsed().as_secs_f64());
        metrics::MOVES_PLAYED.inc(); // Count bot move too
        Some(pos)
    } else {
        // Game ended - record completion
        metrics::GAMES_COMPLETED.inc();
        metrics::GAMES_ACTIVE.dec();
        None
    };

    // Check if game is now over after bot move
    if bot_move.is_some() && session.is_game_over() {
        metrics::GAMES_COMPLETED.inc();
        metrics::GAMES_ACTIVE.dec();
    }

    Ok(Json(MoveResponse {
        state: session.to_response(),
        bot_move,
    }))
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{GameInfoResponse, GameStateResponse, GamesListResponse, MoveRequest, MoveResponse, NewGameRequest};
    use engine_core::GameMetadata;

    #[test]
    fn test_games_list_response_creation() {
        let response = GamesListResponse {
            games: vec!["tictactoe".to_string(), "connect4".to_string()],
        };

        assert_eq!(response.games.len(), 2);
        assert_eq!(response.games[0], "tictactoe");
        assert_eq!(response.games[1], "connect4");
    }

    #[test]
    fn test_games_list_response_empty() {
        let response = GamesListResponse { games: vec![] };
        assert!(response.games.is_empty());
    }

    #[test]
    fn test_game_info_response_from_metadata() {
        let metadata = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
            .with_board(3, 3)
            .with_actions(9)
            .with_observation(29, 18)
            .with_players(2, vec!["X".to_string(), "O".to_string()], vec!['X', 'O']);

        let response: GameInfoResponse = metadata.into();

        assert_eq!(response.env_id, "tictactoe");
        assert_eq!(response.display_name, "Tic-Tac-Toe");
        assert_eq!(response.board_width, 3);
        assert_eq!(response.board_height, 3);
        assert_eq!(response.num_actions, 9);
        assert_eq!(response.obs_size, 29);
        assert_eq!(response.legal_mask_offset, 18);
        assert_eq!(response.player_count, 2);
        assert_eq!(response.player_names, vec!["X", "O"]);
        assert_eq!(response.player_symbols, vec!['X', 'O']);
    }

    #[test]
    fn test_game_info_response_connect4() {
        let metadata = GameMetadata::new("connect4", "Connect Four")
            .with_board(7, 6)
            .with_actions(7)
            .with_observation(93, 84)
            .with_players(2, vec!["Red".to_string(), "Yellow".to_string()], vec!['R', 'Y']);

        let response: GameInfoResponse = metadata.into();

        assert_eq!(response.env_id, "connect4");
        assert_eq!(response.display_name, "Connect Four");
        assert_eq!(response.board_width, 7);
        assert_eq!(response.board_height, 6);
        assert_eq!(response.num_actions, 7);
    }

    #[test]
    fn test_game_state_response_default() {
        let response = GameStateResponse {
            board: vec![0u8; 9],
            current_player: 1,
            human_player: 1,
            winner: 0,
            game_over: false,
            legal_moves: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            message: "Your turn (X)".to_string(),
        };

        assert_eq!(response.board, vec![0u8; 9]);
        assert_eq!(response.current_player, 1);
        assert_eq!(response.human_player, 1);
        assert_eq!(response.winner, 0);
        assert!(!response.game_over);
        assert_eq!(response.legal_moves.len(), 9);
        assert_eq!(response.message, "Your turn (X)");
    }

    #[test]
    fn test_game_state_response_game_over() {
        let response = GameStateResponse {
            board: vec![1, 1, 1, 0, 2, 0, 0, 0, 0],
            current_player: 2,
            human_player: 1,
            winner: 1,
            game_over: true,
            legal_moves: vec![],
            message: "You win!".to_string(),
        };

        assert!(response.game_over);
        assert_eq!(response.winner, 1);
        assert!(response.legal_moves.is_empty());
        assert_eq!(response.message, "You win!");
    }

    #[test]
    fn test_new_game_request_defaults() {
        let req = NewGameRequest {
            first: "player".to_string(),
            game: None,
        };

        assert_eq!(req.first, "player");
        assert!(req.game.is_none());
    }

    #[test]
    fn test_new_game_request_with_game() {
        let req = NewGameRequest {
            first: "bot".to_string(),
            game: Some("tictactoe".to_string()),
        };

        assert_eq!(req.first, "bot");
        assert_eq!(req.game, Some("tictactoe".to_string()));
    }

    #[test]
    fn test_move_request_creation() {
        let req = MoveRequest { position: 4 };
        assert_eq!(req.position, 4);
    }

    #[test]
    fn test_move_response_creation() {
        let state = GameStateResponse {
            board: vec![1, 0, 0, 0, 2, 0, 0, 0, 0],
            current_player: 1,
            human_player: 1,
            winner: 0,
            game_over: false,
            legal_moves: vec![1, 2, 3, 5, 6, 7, 8],
            message: "Your turn (X)".to_string(),
        };

        let response = MoveResponse {
            state,
            bot_move: Some(4),
        };

        assert_eq!(response.bot_move, Some(4));
        assert_eq!(response.state.board[0], 1); // Player move
        assert_eq!(response.state.board[4], 2); // Bot move
    }

    #[test]
    fn test_move_response_no_bot_move() {
        let state = GameStateResponse {
            board: vec![1, 1, 1, 0, 2, 0, 0, 0, 0],
            current_player: 2,
            human_player: 1,
            winner: 1,
            game_over: true,
            legal_moves: vec![],
            message: "You win!".to_string(),
        };

        let response = MoveResponse {
            state,
            bot_move: None, // Game ended before bot could move
        };

        assert!(response.bot_move.is_none());
        assert!(response.state.game_over);
    }

    #[test]
    fn test_game_state_response_serialization() {
        let response = GameStateResponse {
            board: vec![0u8; 9],
            current_player: 1,
            human_player: 1,
            winner: 0,
            game_over: false,
            legal_moves: vec![0, 1, 2],
            message: "Test".to_string(),
        };

        let json = serde_json::to_string(&response);
        assert!(json.is_ok());
        
        let json_str = json.unwrap();
        assert!(json_str.contains("board"));
        assert!(json_str.contains("current_player"));
        assert!(json_str.contains("winner"));
        assert!(json_str.contains("game_over"));
        assert!(json_str.contains("legal_moves"));
        assert!(json_str.contains("message"));
    }

    #[test]
    fn test_move_request_deserialization() {
        let json = r#"{"position": 4}"#;
        let result: Result<MoveRequest, _> = serde_json::from_str(json);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap().position, 4);
    }

    #[test]
    fn test_new_game_request_deserialization_default() {
        // Test that deserializing with default values works
        let json = r#"{}"#;
        let result: Result<NewGameRequest, _> = serde_json::from_str(json);
        
        assert!(result.is_ok());
        let req = result.unwrap();
        assert_eq!(req.first, "player"); // Default value
        assert!(req.game.is_none());
    }

    #[test]
    fn test_new_game_request_deserialization_with_fields() {
        let json = r#"{"first": "bot", "game": "connect4"}"#;
        let result: Result<NewGameRequest, _> = serde_json::from_str(json);
        
        assert!(result.is_ok());
        let req = result.unwrap();
        assert_eq!(req.first, "bot");
        assert_eq!(req.game, Some("connect4".to_string()));
    }

    #[test]
    fn test_move_response_serialization() {
        let state = GameStateResponse {
            board: vec![1, 0, 0, 0, 2, 0, 0, 0, 0],
            current_player: 1,
            human_player: 1,
            winner: 0,
            game_over: false,
            legal_moves: vec![1, 2, 3, 5, 6, 7, 8],
            message: "Your turn".to_string(),
        };

        let response = MoveResponse {
            state,
            bot_move: Some(4),
        };

        let json = serde_json::to_string(&response);
        assert!(json.is_ok());
        
        // The response should be flattened with state fields at top level
        let json_str = json.unwrap();
        assert!(json_str.contains("bot_move"));
        assert!(json_str.contains("board"));
    }
}
