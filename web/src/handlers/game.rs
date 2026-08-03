//! Game-related handlers.

use algorithm_core::BuiltinAlgorithm;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use engine_core::EngineContext;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::OwnedMutexGuard;

use crate::game::GameSession;
use crate::metrics;
use crate::types::{
    FirstPlayer, GameInfoResponse, GameStateResponse, GamesListResponse, MoveRequest, MoveResponse,
    NewGameRequest,
};
use crate::AppState;

/// Map an internal failure to a 500 response tuple.
fn internal_error(context: &str, e: impl std::fmt::Display) -> (StatusCode, String) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("{}: {}", context, e),
    )
}

/// Run the bot's synchronous MCTS on a blocking worker thread so it never
/// stalls the async runtime, recording its latency and converting errors to
/// a 500. Takes and returns the owned session guard because the guard must
/// move onto the worker thread with the search.
async fn timed_bot_move(
    mut session: OwnedMutexGuard<GameSession>,
) -> Result<(OwnedMutexGuard<GameSession>, u32), (StatusCode, String)> {
    tokio::task::spawn_blocking(move || {
        let bot_start = Instant::now();
        let pos = session
            .bot_move()
            .map_err(|e| internal_error("Bot move failed", e))?;
        metrics::BOT_MOVE_SECONDS.observe(bot_start.elapsed().as_secs_f64());
        Ok((session, pos))
    })
    .await
    .map_err(|e| internal_error("Bot move task failed", e))?
}

/// Reconcile the active-session gauge with the single session slot.
fn record_session_activity(session: &GameSession) {
    metrics::GAMES_ACTIVE.set(i64::from(!session.is_game_over()));
}

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

    let context = EngineContext::new(&id).map_err(|error| {
        (
            StatusCode::NOT_FOUND,
            format!("Environment '{id}' is unavailable: {error}"),
        )
    })?;
    BuiltinAlgorithm::AlphaZeroBoardV1
        .compatibility(&context)
        .require_compatible()
        .map_err(|error| {
            internal_error(
                "Environment is not compatible with AlphaZero board serving",
                error,
            )
        })?;

    let metadata = context.metadata();
    Ok(Json(GameInfoResponse::try_from(metadata).map_err(
        |error| {
            internal_error(
                "Environment is not compatible with AlphaZero board serving",
                error,
            )
        },
    )?))
}

/// Get current game state.
pub async fn get_game_state(
    State(state): State<Arc<AppState>>,
) -> Result<Json<GameStateResponse>, (StatusCode, String)> {
    let session = state.session.lock().await;
    Ok(Json(session.to_response().map_err(|error| {
        internal_error("Invalid game observation", error)
    })?))
}

/// Start a new game.
/// Only allows creating the currently configured game type.
/// Rejects requests to switch to a different game since the model
/// is trained for a specific game.
pub async fn new_game(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NewGameRequest>,
) -> Result<Json<GameStateResponse>, (StatusCode, String)> {
    // Get the current configured game
    let current_game = state.current_game.read().await.clone();

    // Reject requests trying to switch to a different game before touching
    // the session or any metric
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

    let mut session = Arc::clone(&state.session).lock_owned().await;

    // Reset the game with shared evaluator (for hot-reloading)
    *session = GameSession::with_evaluator(&game_id, Arc::clone(&state.evaluator))
        .map_err(|e| internal_error(&format!("Failed to create game '{}'", game_id), e))?;
    metrics::GAMES_CREATED.inc();

    // If bot goes first, bot is player 1, human is player 2
    // If player goes first, human is player 1, bot is player 2
    if req.first == FirstPlayer::Bot {
        session
            .set_human_player(2)
            .map_err(|e| internal_error("Invalid human board seat", e))?; // Human plays as O (player 2)
        (session, _) = timed_bot_move(session).await?;
    } else {
        session
            .set_human_player(1)
            .map_err(|e| internal_error("Invalid human board seat", e))?; // Human plays as X (player 1) - default
    }
    record_session_activity(&session);

    Ok(Json(session.to_response().map_err(|error| {
        internal_error("Invalid game observation", error)
    })?))
}

/// Make a move (player + bot response).
pub async fn make_move(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MoveRequest>,
) -> Result<Json<MoveResponse>, (StatusCode, String)> {
    let mut session = Arc::clone(&state.session).lock_owned().await;

    // Check if game is over
    if session.is_game_over() {
        return Err((StatusCode::BAD_REQUEST, "Game is already over".to_string()));
    }

    // Check if it's the human's turn
    if !session.is_human_turn() {
        return Err((StatusCode::BAD_REQUEST, "Not your turn".to_string()));
    }

    // Check if move is legal (this handles position validation based on game type)
    if !session
        .is_legal_move(req.position)
        .map_err(|error| internal_error("Invalid game observation", error))?
    {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Illegal move: position/column {} is not valid",
                req.position
            ),
        ));
    }

    // Make player's move
    session
        .player_move(req.position)
        .map_err(|e| internal_error("Move failed", e))?;
    metrics::MOVES_PLAYED.inc();

    // If game is not over, bot makes a move
    let bot_move = if !session.is_game_over() {
        let pos;
        (session, pos) = timed_bot_move(session).await?;
        metrics::MOVES_PLAYED.inc(); // Count bot move too
        Some(pos)
    } else {
        None
    };

    // The game was running when this request started, so reaching a terminal
    // state now is exactly one completion regardless of who ended it.
    if session.is_game_over() {
        metrics::GAMES_COMPLETED.inc();
    }
    record_session_activity(&session);

    Ok(Json(MoveResponse {
        state: session
            .to_response()
            .map_err(|error| internal_error("Invalid game observation", error))?,
        bot_move,
    }))
}

// ============================================================================
// Unit Tests
// ============================================================================

// These construction/serialization tests overlap with the suites in
// types/requests.rs and types/responses.rs; they exercise the shapes the
// handlers above actually return.
#[cfg(test)]
mod tests {
    use crate::types::{
        FirstPlayer, GameInfoResponse, GameStateResponse, GamesListResponse, MoveRequest,
        MoveResponse, NewGameRequest,
    };
    use engine_core::board_profile::{BoardGameMetadata, BoardPlayerMetadata, BoardRenderer};
    use engine_core::EnvironmentMetadata;

    fn board_metadata(
        id: &str,
        display_name: &str,
        dimensions: (usize, usize),
        action_count: usize,
        observation: (usize, usize),
        players: [(&str, &str); 2],
        renderer: BoardRenderer,
    ) -> EnvironmentMetadata {
        let (width, height) = dimensions;
        let (elements, legal_actions_offset) = observation;
        EnvironmentMetadata::new(id, display_name).with_board(
            BoardGameMetadata::new(width, height, action_count)
                .with_observation(elements, 2, legal_actions_offset, false)
                .with_players(
                    players
                        .into_iter()
                        .map(|(name, symbol)| BoardPlayerMetadata::new(name, symbol))
                        .collect(),
                )
                .with_renderer(renderer),
        )
    }

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
        let metadata = board_metadata(
            "tictactoe",
            "Tic-Tac-Toe",
            (3, 3),
            9,
            (29, 18),
            [("X", "X"), ("O", "O")],
            BoardRenderer::Grid,
        );

        let response = GameInfoResponse::try_from(metadata).unwrap();

        assert_eq!(response.env_id, "tictactoe");
        assert_eq!(response.display_name, "Tic-Tac-Toe");
        assert_eq!(response.board_width, 3);
        assert_eq!(response.board_height, 3);
        assert_eq!(response.num_actions, 9);
        assert_eq!(response.obs_size, 29);
        assert_eq!(response.legal_mask_offset, 18);
        assert_eq!(response.player_count, 2);
        assert_eq!(response.player_names, vec!["X", "O"]);
        assert_eq!(response.player_symbols, vec!["X", "O"]);
    }

    #[test]
    fn test_game_info_response_connect4() {
        let metadata = board_metadata(
            "connect4",
            "Connect Four",
            (7, 6),
            7,
            (93, 84),
            [("Red", "🔴"), ("Yellow", "🟡")],
            BoardRenderer::DropColumn,
        );

        let response = GameInfoResponse::try_from(metadata).unwrap();

        assert_eq!(response.env_id, "connect4");
        assert_eq!(response.display_name, "Connect Four");
        assert_eq!(response.board_width, 7);
        assert_eq!(response.board_height, 6);
        assert_eq!(response.num_actions, 7);
    }

    #[test]
    fn test_game_state_response_default() {
        let response = GameStateResponse {
            cells: engine_core::board_profile::BoardView::from_owners(&[0u8; 9], 1, 0).cells,
            current_player: 1,
            human_player: 1,
            winner: 0,
            game_over: false,
            legal_moves: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
            message: "Your turn (X)".to_string(),
        };

        assert_eq!(response.cells.len(), 9);
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
            cells: engine_core::board_profile::BoardView::from_owners(
                &[1, 1, 1, 0, 2, 0, 0, 0, 0],
                1,
                0,
            )
            .cells,
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
            first: FirstPlayer::Player,
            game: None,
        };

        assert_eq!(req.first, FirstPlayer::Player);
        assert!(req.game.is_none());
    }

    #[test]
    fn test_new_game_request_with_game() {
        let req = NewGameRequest {
            first: FirstPlayer::Bot,
            game: Some("tictactoe".to_string()),
        };

        assert_eq!(req.first, FirstPlayer::Bot);
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
            cells: engine_core::board_profile::BoardView::from_owners(
                &[1, 0, 0, 0, 2, 0, 0, 0, 0],
                1,
                0,
            )
            .cells,
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
        assert_eq!(response.state.cells[0].owner, 1); // Player move
        assert_eq!(response.state.cells[4].owner, 2); // Bot move
    }

    #[test]
    fn test_move_response_no_bot_move() {
        let state = GameStateResponse {
            cells: engine_core::board_profile::BoardView::from_owners(
                &[1, 1, 1, 0, 2, 0, 0, 0, 0],
                1,
                0,
            )
            .cells,
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
            cells: engine_core::board_profile::BoardView::from_owners(&[0u8; 9], 1, 0).cells,
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
        assert!(json_str.contains("cells"));
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
        assert_eq!(req.first, FirstPlayer::Player); // Default value
        assert!(req.game.is_none());
    }

    #[test]
    fn test_new_game_request_deserialization_with_fields() {
        let json = r#"{"first": "bot", "game": "connect4"}"#;
        let result: Result<NewGameRequest, _> = serde_json::from_str(json);

        assert!(result.is_ok());
        let req = result.unwrap();
        assert_eq!(req.first, FirstPlayer::Bot);
        assert_eq!(req.game, Some("connect4".to_string()));
    }

    #[test]
    fn test_move_response_serialization() {
        let state = GameStateResponse {
            cells: engine_core::board_profile::BoardView::from_owners(
                &[1, 0, 0, 0, 2, 0, 0, 0, 0],
                1,
                0,
            )
            .cells,
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
        assert!(json_str.contains("cells"));
    }
}
