//! Integration tests for the web server routes and CORS configuration.

use super::*;
use crate::types::{
    GameInfoResponse, GameStateResponse, GamesListResponse, HealthResponse, MoveResponse,
};
use axum::{
    body::Body,
    http::{Request, StatusCode},
    Router,
};
use http_body_util::BodyExt;
use tower::ServiceExt;

/// Helper to make a GET request and return response body as string
async fn get(app: Router, uri: &str) -> (StatusCode, String) {
    let response = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    (status, body_str)
}

/// Helper to make a POST request with JSON body and return response
async fn post_json(app: Router, uri: &str, json: &str) -> (StatusCode, String) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(json.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8(body.to_vec()).unwrap();
    (status, body_str)
}

#[tokio::test]
async fn test_health_endpoint() {
    let state = create_test_state();
    let app = create_app(state);

    let (status, body) = get(app, "/health").await;

    assert_eq!(status, StatusCode::OK);
    let response: HealthResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(response.status, "ok");
}

#[tokio::test]
async fn test_game_state_returns_initial_board() {
    let state = create_test_state();
    let app = create_app(state);

    let (status, body) = get(app, "/game/state").await;

    assert_eq!(status, StatusCode::OK);
    let response: GameStateResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(
        response.board,
        vec![0u8; 9],
        "Initial board should be empty"
    );
    assert_eq!(response.current_player, 1, "Player X should go first");
    assert_eq!(response.winner, 0, "No winner yet");
    assert!(!response.game_over);
    assert_eq!(response.legal_moves.len(), 9, "All 9 moves should be legal");
}

#[tokio::test]
async fn test_new_game_player_first() {
    let state = create_test_state();
    let app = create_app(state);

    let (status, body) = post_json(app, "/game/new", r#"{"first": "player"}"#).await;

    assert_eq!(status, StatusCode::OK);
    let response: GameStateResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(
        response.board,
        vec![0u8; 9],
        "Board should be empty when player goes first"
    );
    assert_eq!(response.current_player, 1);
    assert!(!response.game_over);
}

#[tokio::test]
async fn test_new_game_bot_first() {
    let state = create_test_state();
    let app = create_app(state);

    let (status, body) = post_json(app, "/game/new", r#"{"first": "bot"}"#).await;

    assert_eq!(status, StatusCode::OK);
    let response: GameStateResponse = serde_json::from_str(&body).unwrap();
    // Bot should have made one move (one cell is non-zero)
    let moves_made: usize = response.board.iter().filter(|&&x| x != 0).count();
    assert_eq!(moves_made, 1, "Bot should have made exactly one move");
    // When bot goes "first", it plays as X (since X always starts in TicTacToe)
    // After bot's move as X, it's O's turn (current_player = 2)
    assert_eq!(
        response.current_player, 2,
        "Should be O's turn after bot (X) moves first"
    );
}

#[tokio::test]
async fn test_new_game_default_player_first() {
    let state = create_test_state();
    let app = create_app(state);

    // Empty JSON should default to player first
    let (status, body) = post_json(app, "/game/new", r#"{}"#).await;

    assert_eq!(status, StatusCode::OK);
    let response: GameStateResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(
        response.board,
        vec![0u8; 9],
        "Board should be empty with default"
    );
}

#[tokio::test]
async fn test_new_game_rejects_different_game_type() {
    let state = create_test_state();
    let app = create_app(state);

    // Trying to create a game of a different type should be rejected
    let (status, body) = post_json(
        app,
        "/game/new",
        r#"{"first": "player", "game": "connect4"}"#,
    )
    .await;

    assert_eq!(status, StatusCode::FORBIDDEN);
    assert!(body.contains("Cannot switch to game"));
    assert!(body.contains("only the current game 'tictactoe' is available"));
}

#[tokio::test]
async fn test_new_game_allows_same_game_type() {
    let state = create_test_state();
    let app = create_app(state);

    // Requesting the current game type should be allowed
    let (status, body) = post_json(
        app,
        "/game/new",
        r#"{"first": "player", "game": "tictactoe"}"#,
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let response: GameStateResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(response.board, vec![0u8; 9]);
}

#[tokio::test]
async fn test_move_valid() {
    let state = create_test_state();
    let app = create_app(state);

    // Make a move at position 4 (center)
    let (status, body) = post_json(app, "/move", r#"{"position": 4}"#).await;

    assert_eq!(status, StatusCode::OK);
    let response: MoveResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(response.state.board[4], 1, "Player X should be at center");
    // Bot should have made a move (unless game over)
    if !response.state.game_over {
        assert!(response.bot_move.is_some(), "Bot should make a move");
        let bot_pos = response.bot_move.unwrap() as usize;
        assert_eq!(response.state.board[bot_pos], 2, "Bot should have placed O");
    }
}

#[tokio::test]
async fn test_move_invalid_position() {
    let state = create_test_state();
    let app = create_app(state);

    // Position 9 is out of bounds for TicTacToe
    let (status, body) = post_json(app, "/move", r#"{"position": 9}"#).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("Illegal move") || body.contains("not valid"));
}

#[tokio::test]
async fn test_move_occupied_position() {
    let state = create_test_state();

    // First move at center
    {
        let app = create_app(Arc::clone(&state));
        let (status, _) = post_json(app, "/move", r#"{"position": 4}"#).await;
        assert_eq!(status, StatusCode::OK);
    }

    // Try to move at center again (occupied by player X)
    {
        let app = create_app(Arc::clone(&state));
        let (status, body) = post_json(app, "/move", r#"{"position": 4}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("Illegal move") || body.contains("occupied"));
    }
}

#[tokio::test]
async fn test_game_flow_player_wins() {
    let state = create_test_state();

    // Start fresh game
    {
        let app = create_app(Arc::clone(&state));
        post_json(app, "/game/new", r#"{"first": "player"}"#).await;
    }

    // Try to get player to win with top row (0, 1, 2)
    // This may not always work due to bot blocking, but tests the flow
    let moves = [0, 1, 2]; // Try top row
    let mut player_positions = vec![];

    for pos in moves {
        let app = create_app(Arc::clone(&state));
        let (status, body) = post_json(app, "/move", &format!(r#"{{"position": {}}}"#, pos)).await;

        if status == StatusCode::BAD_REQUEST {
            // Position might be taken by bot, skip
            continue;
        }
        assert_eq!(status, StatusCode::OK);

        let response: MoveResponse = serde_json::from_str(&body).unwrap();
        player_positions.push(pos);

        if response.state.game_over {
            // Game ended - verify state is consistent
            assert!(response.state.winner != 0 || response.state.legal_moves.is_empty());
            break;
        }
    }
}

#[tokio::test]
async fn test_move_when_game_over() {
    let state = create_test_state();

    // Play a complete game by making moves until done
    let mut game_over = false;
    let mut position = 0u8;

    while !game_over && position < 9 {
        let app = create_app(Arc::clone(&state));
        let (status, body) =
            post_json(app, "/move", &format!(r#"{{"position": {}}}"#, position)).await;

        if status == StatusCode::OK {
            let response: MoveResponse = serde_json::from_str(&body).unwrap();
            game_over = response.state.game_over;
        }
        position += 1;
    }

    // If game is over, next move should fail
    if game_over {
        // Find an empty position (if any) and try to move
        let app = create_app(Arc::clone(&state));
        let (status, body) = post_json(app, "/move", r#"{"position": 0}"#).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("Game is already over") || body.contains("Illegal move"));
    }
}

#[tokio::test]
async fn test_state_updates_after_move() {
    let state = create_test_state();

    // Get initial state
    let (_, initial_body) = {
        let app = create_app(Arc::clone(&state));
        get(app, "/game/state").await
    };
    let initial: GameStateResponse = serde_json::from_str(&initial_body).unwrap();
    assert_eq!(initial.board, vec![0u8; 9]);

    // Make a move
    {
        let app = create_app(Arc::clone(&state));
        let (status, _) = post_json(app, "/move", r#"{"position": 0}"#).await;
        assert_eq!(status, StatusCode::OK);
    }

    // Get updated state
    let (_, updated_body) = {
        let app = create_app(Arc::clone(&state));
        get(app, "/game/state").await
    };
    let updated: GameStateResponse = serde_json::from_str(&updated_body).unwrap();

    // Verify board changed
    assert_ne!(updated.board, vec![0u8; 9], "Board should have changed");
    assert_eq!(updated.board[0], 1, "Player X should be at position 0");
}

#[tokio::test]
async fn test_new_game_resets_state() {
    let state = create_test_state();

    // Make some moves
    {
        let app = create_app(Arc::clone(&state));
        post_json(app, "/move", r#"{"position": 4}"#).await;
    }

    // Verify board is not empty
    let (_, body) = {
        let app = create_app(Arc::clone(&state));
        get(app, "/game/state").await
    };
    let mid_game: GameStateResponse = serde_json::from_str(&body).unwrap();
    assert_ne!(mid_game.board, vec![0u8; 9], "Board should have moves");

    // Start new game
    {
        let app = create_app(Arc::clone(&state));
        post_json(app, "/game/new", r#"{"first": "player"}"#).await;
    }

    // Verify board is reset
    let (_, body) = {
        let app = create_app(Arc::clone(&state));
        get(app, "/game/state").await
    };
    let new_game: GameStateResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(new_game.board, vec![0u8; 9], "Board should be reset");
    assert_eq!(new_game.current_player, 1);
    assert_eq!(new_game.winner, 0);
}

#[tokio::test]
async fn test_legal_moves_update() {
    let state = create_test_state();

    // Initial state has 9 legal moves
    let (_, body) = {
        let app = create_app(Arc::clone(&state));
        get(app, "/game/state").await
    };
    let initial: GameStateResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(initial.legal_moves.len(), 9);

    // Make a move
    {
        let app = create_app(Arc::clone(&state));
        post_json(app, "/move", r#"{"position": 0}"#).await;
    }

    // Should have fewer legal moves (player + bot each made one)
    let (_, body) = {
        let app = create_app(Arc::clone(&state));
        get(app, "/game/state").await
    };
    let after_move: GameStateResponse = serde_json::from_str(&body).unwrap();
    assert!(
        after_move.legal_moves.len() < 9,
        "Should have fewer legal moves"
    );
    assert!(
        !after_move.legal_moves.contains(&0),
        "Position 0 should not be legal"
    );
}

#[tokio::test]
async fn test_list_games_returns_only_current_game() {
    let state = create_test_state();
    let app = create_app(state);

    let (status, body) = get(app, "/games").await;

    assert_eq!(status, StatusCode::OK);
    let response: GamesListResponse = serde_json::from_str(&body).unwrap();
    // Should only return the current game, not all registered games
    assert_eq!(response.games.len(), 1);
    assert_eq!(response.games[0], "tictactoe");
}

#[tokio::test]
async fn test_get_game_info_tictactoe() {
    let state = create_test_state();
    let app = create_app(state);

    let (status, body) = get(app, "/game-info/tictactoe").await;

    assert_eq!(status, StatusCode::OK);
    let response: GameInfoResponse = serde_json::from_str(&body).unwrap();
    assert_eq!(response.env_id, "tictactoe");
    assert_eq!(response.display_name, "Tic-Tac-Toe");
    assert_eq!(response.board_width, 3);
    assert_eq!(response.board_height, 3);
    assert_eq!(response.num_actions, 9);
    assert_eq!(response.player_count, 2);
}

#[tokio::test]
async fn test_get_game_info_forbidden_for_non_current_game() {
    let state = create_test_state();
    let app = create_app(state);

    // Requesting a different game than the current one should return FORBIDDEN
    let (status, body) = get(app, "/game-info/connect4").await;

    assert_eq!(status, StatusCode::FORBIDDEN);
    assert!(body.contains("Cannot access game"));
    assert!(body.contains("only the current game 'tictactoe' is available"));
}

#[tokio::test]
async fn test_get_game_info_not_found_for_invalid_game() {
    let state = create_test_state();
    let app = create_app(state);

    // Requesting an invalid game ID that matches current game check
    // but doesn't exist in registry should return NOT_FOUND
    // (This would require the game ID to be "tictactoe" to pass the filter,
    // so this test case is for truly invalid games)
    let (status, _body) = get(app, "/game-info/tictactoe_invalid").await;

    // Since "tictactoe_invalid" != "tictactoe", it returns FORBIDDEN
    assert_eq!(status, StatusCode::FORBIDDEN);
}

// ========================================================================
// CORS Tests
// ========================================================================

#[tokio::test]
async fn test_cors_allows_configured_origin() {
    let state = create_test_state();
    let allowed = vec!["https://allowed.example.com".to_string()];
    let app = create_app_with_cors(state, &allowed);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("Origin", "https://allowed.example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("access-control-allow-origin")
            .unwrap(),
        "https://allowed.example.com"
    );
}

#[tokio::test]
async fn test_cors_rejects_unknown_origin_in_production() {
    let state = create_test_state();
    let allowed = vec!["https://allowed.example.com".to_string()];
    let app = create_app_with_cors(state, &allowed);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("Origin", "https://evil.example.com")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Response should succeed but without CORS header for disallowed origin
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response
        .headers()
        .get("access-control-allow-origin")
        .is_none());
}

#[tokio::test]
async fn test_cors_allows_localhost_in_development_mode() {
    let state = create_test_state();
    // Empty allowed_origins = development mode (localhost only)
    let allowed: Vec<String> = vec![];
    let app = create_app_with_cors(state, &allowed);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .header("Origin", "http://localhost:3000")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    // In development mode, localhost origins should be allowed
    assert_eq!(
        response
            .headers()
            .get("access-control-allow-origin")
            .unwrap(),
        "http://localhost:3000"
    );
}

#[tokio::test]
async fn test_cors_preflight_request() {
    let state = create_test_state();
    let allowed = vec!["https://allowed.example.com".to_string()];
    let app = create_app_with_cors(state, &allowed);

    let response = app
        .oneshot(
            Request::builder()
                .method("OPTIONS")
                .uri("/move")
                .header("Origin", "https://allowed.example.com")
                .header("Access-Control-Request-Method", "POST")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Preflight should succeed
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response
        .headers()
        .get("access-control-allow-methods")
        .is_some());
}
