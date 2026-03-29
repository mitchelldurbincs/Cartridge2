//! Cartridge2 Web Server
//!
//! Minimal HTTP server exposing game API for the Svelte frontend.
//! Endpoints:
//! - GET  /health        - Health check
//! - GET  /games         - List available games
//! - GET  /game-info/:id - Get metadata for a specific game
//! - POST /game/new      - Start a new game
//! - GET  /game/state    - Get current game state
//! - POST /move          - Make a move (player action + bot response)
//! - GET  /stats         - Read training stats from data/stats.json
//! - GET  /actor-stats   - Read actor self-play stats from data/actor_stats.json
//! - GET  /model         - Get info about currently loaded model

use axum::{
    http::{header, HeaderValue, Method},
    routing::{get, post},
    Router,
};
#[cfg(feature = "onnx")]
use mcts::OnnxEvaluator;
use std::sync::Arc;
// Note: We use std::sync::RwLock (aliased as StdRwLock) for `evaluator` and `model_info`
// because they are shared with the model_watcher crate which requires std::sync::RwLock.
// These locks are only held briefly (no await points while held) so blocking is minimal.
// `current_game` uses tokio::sync::RwLock since it's owned entirely by AppState.
use std::sync::RwLock as StdRwLock;
use tokio::sync::{Mutex, RwLock};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use tracing::{info, warn};

mod game;
mod handlers;
mod metrics;
#[cfg(feature = "onnx")]
mod model_watcher;
mod types;

use engine_config::load_config;
use game::GameSession;
use handlers::{
    get_actor_stats, get_game_info, get_game_state, get_model_info, get_stats, health, list_games,
    make_move, metrics_handler, new_game,
};
#[cfg(feature = "onnx")]
use model_watcher::{ModelInfo, ModelWatcher};

/// Stub evaluator type when ONNX is disabled (for testing)
#[cfg(not(feature = "onnx"))]
pub type OnnxEvaluator = ();

/// Stub ModelInfo when ONNX is disabled
#[cfg(not(feature = "onnx"))]
#[derive(Default, Clone)]
pub struct ModelInfo {
    pub loaded: bool,
    pub path: Option<String>,
    pub file_modified: Option<u64>,
    pub loaded_at: Option<u64>,
    pub training_step: Option<u32>,
}

/// Shared application state
pub struct AppState {
    /// Current game session (tokio async Mutex - held across awaits in handlers)
    pub session: Mutex<GameSession>,
    /// Current game ID (tokio async RwLock - owned by AppState)
    pub current_game: RwLock<String>,
    /// Data directory for stats.json
    pub data_dir: String,
    /// Shared evaluator for MCTS (std RwLock - shared with model_watcher crate)
    /// Only read briefly in sync code, never held across await points.
    pub evaluator: Arc<StdRwLock<Option<OnnxEvaluator>>>,
    /// Model info (std RwLock - shared with model_watcher crate)
    /// Only read briefly, never held across await points.
    pub model_info: Arc<StdRwLock<ModelInfo>>,
}

/// Configure CORS based on allowed origins.
///
/// If `allowed_origins` is empty, allows all origins (development mode) with a warning.
/// Otherwise, restricts to the specified origins (production mode).
fn configure_cors(allowed_origins: &[String]) -> CorsLayer {
    if allowed_origins.is_empty() {
        // Development mode: allow all origins with a warning
        warn!(
            component = "web",
            event = "cors_insecure",
            "CORS: No allowed_origins configured - allowing all origins (insecure for production)"
        );
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([header::CONTENT_TYPE, header::ACCEPT])
    } else {
        // Production mode: restrict to configured origins
        let origins: Vec<HeaderValue> = allowed_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();

        info!(
            component = "web",
            event = "cors_configured",
            origins = ?allowed_origins,
            "CORS: Allowing configured origins"
        );

        CorsLayer::new()
            .allow_origin(AllowOrigin::list(origins))
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([header::CONTENT_TYPE, header::ACCEPT])
            .allow_credentials(true)
    }
}

/// Create the application router with the given state and allowed origins.
/// This is separated out for testing purposes.
pub fn create_app_with_cors(state: Arc<AppState>, allowed_origins: &[String]) -> Router {
    let cors = configure_cors(allowed_origins);

    Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        .route("/games", get(list_games))
        .route("/game-info/:id", get(get_game_info))
        .route("/game/new", post(new_game))
        .route("/game/state", get(get_game_state))
        .route("/move", post(make_move))
        .route("/stats", get(get_stats))
        .route("/actor-stats", get(get_actor_stats))
        .route("/model", get(get_model_info))
        .layer(cors)
        .with_state(state)
}

/// Create the application router with the given state.
/// Uses permissive CORS (empty allowed_origins = development mode).
pub fn create_app(state: Arc<AppState>) -> Router {
    create_app_with_cors(state, &[])
}

/// Create application state for testing (no model watcher, no logging)
#[cfg(test)]
pub fn create_test_state() -> Arc<AppState> {
    engine_games::register_all_games();
    let evaluator: Arc<StdRwLock<Option<OnnxEvaluator>>> = Arc::new(StdRwLock::new(None));
    let model_info = Arc::new(StdRwLock::new(ModelInfo::default()));
    let session = GameSession::with_evaluator("tictactoe", Arc::clone(&evaluator))
        .expect("Failed to create game session");

    Arc::new(AppState {
        session: Mutex::new(session),
        current_game: RwLock::new("tictactoe".to_string()),
        data_dir: "./test_data".to_string(),
        evaluator,
        model_info,
    })
}

/// Initialize tracing with optional JSON format for cloud deployments.
///
/// Supports CARTRIDGE_LOGGING_FORMAT environment variable override:
/// - "text" (default): Human-readable format for local development
/// - "json": Structured JSON format for Google Cloud Logging
fn init_tracing(logging_config: &engine_config::LoggingConfig) {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"))
        .add_directive("web=info".parse().unwrap())
        .add_directive("ort=warn".parse().unwrap())
        .add_directive("h2=warn".parse().unwrap())
        .add_directive("hyper=warn".parse().unwrap());

    // Check for environment variable override
    let json_format = std::env::var("CARTRIDGE_LOGGING_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or_else(|_| logging_config.is_json());

    let registry = tracing_subscriber::registry().with(filter);

    if json_format {
        // JSON format for Google Cloud Logging
        registry
            .with(
                fmt::layer()
                    .json()
                    .with_current_span(true)
                    .with_span_list(false)
                    .with_file(false)
                    .with_line_number(false)
                    .flatten_event(true)
                    .with_target(logging_config.include_target),
            )
            .init();
    } else {
        // Human-readable format for local development
        registry.with(fmt::layer()).init();
    }
}

/// Creates a future that completes when a shutdown signal is received.
/// Handles Ctrl+C on all platforms.
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    info!(
        component = "web",
        event = "shutdown_signal",
        "Shutdown signal received, stopping server"
    );
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load configuration first (needed for logging config)
    let config = load_config();

    // Initialize tracing with JSON support for cloud deployments
    init_tracing(&config.logging);

    // Initialize Prometheus metrics
    metrics::init_metrics();
    info!(component = "web", "Prometheus metrics initialized");

    // Register all games
    engine_games::register_all_games();
    info!(component = "web", "Registered all games");

    let data_dir = config.common.data_dir.clone();
    let default_game = config.common.env_id.clone();
    info!(
        component = "web",
        data_dir = %data_dir,
        default_game = %default_game,
        host = %config.web.host,
        port = config.web.port,
        "Web server configuration loaded"
    );

    // Set up shared evaluator for model hot-reloading
    // Uses std::sync::RwLock because it's shared with model_watcher crate
    let evaluator: Arc<StdRwLock<Option<OnnxEvaluator>>> = Arc::new(StdRwLock::new(None));

    #[cfg(feature = "onnx")]
    let model_info = {
        use engine_core::EngineContext;
        use tracing::warn;
        let model_dir = format!("{}/models", data_dir);
        // Get obs_size from the configured game, falling back to tictactoe if not found
        let obs_size = EngineContext::new(&default_game)
            .or_else(|| {
                warn!(
                    component = "web",
                    game = %default_game,
                    "Game not found, falling back to tictactoe for obs_size"
                );
                EngineContext::new("tictactoe")
            })
            .map(|ctx| ctx.metadata().obs_size)
            .expect("At least tictactoe should be registered");
        info!(
            component = "web",
            obs_size = obs_size,
            game = %default_game,
            "Model watcher initialized"
        );

        // Create model watcher with metadata tracking for the web UI
        // Use 1 intra-op thread since web server does single-threaded inference for play
        let model_watcher = ModelWatcher::new(
            &model_dir,
            "latest.onnx",
            obs_size,
            1,
            Arc::clone(&evaluator),
        )
        .with_metadata();

        // Try to load existing model
        match model_watcher.try_load_existing() {
            Ok(true) => info!(
                component = "web",
                event = "model_loaded",
                model_path = %format!("{}/latest.onnx", model_dir),
                "Loaded existing model"
            ),
            Ok(false) => info!(
                component = "web",
                event = "model_not_found",
                model_path = %format!("{}/latest.onnx", model_dir),
                "No model found - bot will play randomly"
            ),
            Err(e) => warn!(
                component = "web",
                event = "model_load_error",
                error = %e,
                "Failed to load existing model - bot will play randomly"
            ),
        }

        // Get model info reference before moving watcher
        let model_info = model_watcher.model_info();

        // Start watching for model updates
        let mut model_rx = model_watcher.start_watching().await?;

        // Spawn task to log model updates
        tokio::spawn(async move {
            while model_rx.recv().await.is_some() {
                info!(
                    component = "web",
                    event = "model_updated",
                    "Model updated - bot will use new model for future games"
                );
            }
        });

        model_info
    };

    #[cfg(not(feature = "onnx"))]
    let model_info = Arc::new(StdRwLock::new(ModelInfo::default()));

    // Create initial game session with shared evaluator
    let session = GameSession::with_evaluator(&default_game, Arc::clone(&evaluator))?;

    let state = Arc::new(AppState {
        session: Mutex::new(session),
        current_game: RwLock::new(default_game),
        data_dir,
        evaluator,
        model_info,
    });

    // Build router with CORS configuration
    let app = create_app_with_cors(state, &config.web.allowed_origins);

    let addr = format!("{}:{}", config.web.host, config.web.port);
    info!(
        component = "web",
        event = "server_start",
        address = %addr,
        "Starting web server"
    );

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!(
        component = "web",
        event = "shutdown_complete",
        "Server shut down gracefully"
    );
    Ok(())
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        GameInfoResponse, GameStateResponse, GamesListResponse, HealthResponse, MoveResponse,
    };
    use axum::{
        body::Body,
        http::{Request, StatusCode},
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
            let (status, body) =
                post_json(app, "/move", &format!(r#"{{"position": {}}}"#, pos)).await;

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
    async fn test_list_games() {
        let state = create_test_state();
        let app = create_app(state);

        let (status, body) = get(app, "/games").await;

        assert_eq!(status, StatusCode::OK);
        let response: GamesListResponse = serde_json::from_str(&body).unwrap();
        assert!(response.games.contains(&"tictactoe".to_string()));
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
    async fn test_get_game_info_not_found() {
        let state = create_test_state();
        let app = create_app(state);

        let (status, body) = get(app, "/game-info/nonexistent").await;

        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.contains("Game not found"));
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
    async fn test_cors_allows_any_origin_in_development_mode() {
        let state = create_test_state();
        // Empty allowed_origins = development mode
        let allowed: Vec<String> = vec![];
        let app = create_app_with_cors(state, &allowed);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header("Origin", "https://any-origin.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        // In development mode, any origin should be allowed
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-origin")
                .unwrap(),
            "*"
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
}
