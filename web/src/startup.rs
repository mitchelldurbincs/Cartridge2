//! Server plumbing: shared application state, router/CORS construction,
//! tracing initialization, and shutdown signal handling.
//!
//! This keeps `main.rs` a thin entrypoint. The types and constructor
//! functions here are re-exported from the crate root so that their public
//! paths (`crate::AppState`, `crate::create_app`, ...) remain stable.

use axum::{
    http::{header, HeaderValue, Method},
    routing::{get, post},
    Router,
};
#[cfg(feature = "onnx")]
pub use mcts::OnnxEvaluator;
use std::sync::Arc;
// Note: We use std::sync::RwLock (aliased as StdRwLock) for `evaluator` and `model_info`
// because they are shared with the model_watcher crate which requires std::sync::RwLock.
// These locks are only held briefly (no await points while held) so blocking is minimal.
// `current_game` uses tokio::sync::RwLock since it's owned entirely by AppState.
use std::sync::RwLock as StdRwLock;
use tokio::sync::{Mutex, RwLock};
use tower_http::cors::{AllowOrigin, CorsLayer};
use tracing::{info, warn};

use crate::game::GameSession;
use crate::handlers::{
    get_actor_stats, get_game_info, get_game_state, get_model_info, get_stats, health, list_games,
    make_move, metrics_handler, new_game,
};
#[cfg(feature = "onnx")]
pub use crate::model_watcher::ModelInfo;

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
/// If `allowed_origins` is empty, only allows localhost origins (secure development mode).
/// Otherwise, restricts to the specified origins (production mode).
/// This is deny-by-default behavior to prevent accidental insecure configurations.
fn configure_cors(allowed_origins: &[String]) -> CorsLayer {
    if allowed_origins.is_empty() {
        // Deny-by-default: only allow localhost origins when none configured
        warn!(
            component = "web",
            event = "cors_localhost_fallback",
            "CORS: No allowed_origins configured - restricting to localhost only"
        );
        let localhost_origins = vec![
            "http://localhost".parse().ok(),
            "http://localhost:3000".parse().ok(),
            "http://localhost:5173".parse().ok(),
            "http://localhost:8080".parse().ok(),
            "http://127.0.0.1".parse().ok(),
            "http://127.0.0.1:3000".parse().ok(),
            "http://127.0.0.1:5173".parse().ok(),
            "http://127.0.0.1:8080".parse().ok(),
        ];
        let origins: Vec<HeaderValue> = localhost_origins.into_iter().flatten().collect();

        CorsLayer::new()
            .allow_origin(AllowOrigin::list(origins))
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([header::CONTENT_TYPE, header::ACCEPT])
            .allow_credentials(true)
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
pub fn init_tracing(logging_config: &engine_config::LoggingConfig) {
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
pub async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    info!(
        component = "web",
        event = "shutdown_signal",
        "Shutdown signal received, stopping server"
    );
}
