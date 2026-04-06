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

    let game = create_game(&id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Game not found: {}", id)))?;

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
