//! Server plumbing: shared application state, router/CORS construction,
//! and shutdown signal handling (tracing setup lives in engine_config::init_tracing).
//!
//! This keeps `main.rs` a thin entrypoint. The crate root re-exports the
//! plumbing used by handlers and route tests.

use algorithm_core::{resolve_algorithm, BuiltinAlgorithm, ModelArtifactContract};
use anyhow::{anyhow, Result};
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
    get_game_info, get_game_state, get_model_info, get_stats, health, list_games, make_move,
    metrics_handler, new_game,
};
#[cfg(feature = "onnx")]
pub use model_watcher::ModelInfo;

/// Algorithm/environment binding resolved before the web server watches a
/// model or constructs its first game session.
pub struct StartupProfile {
    pub algorithm: BuiltinAlgorithm,
    pub obs_size: usize,
    pub num_actions: usize,
    pub max_horizon: u32,
    pub env_contract_version: u32,
    pub model_contract: ModelArtifactContract,
}

const SUPPORTED_SERVING_COMPONENT: &str = "alphazero_mcts_web_v1";

/// Resolve the configured algorithm and exact environment without a fallback.
pub fn resolve_startup_profile(algorithm_id: &str, env_id: &str) -> Result<StartupProfile> {
    let algorithm = resolve_algorithm(algorithm_id)?;
    let context = engine_core::EngineContext::new(env_id)
        .map_err(|error| anyhow!("Environment '{env_id}' is unavailable: {error}"))?;
    algorithm.compatibility(&context).require_compatible()?;
    let descriptor = algorithm.descriptor();
    if descriptor.components.serving != SUPPORTED_SERVING_COMPONENT {
        return Err(anyhow!(
            "Algorithm '{}' requires unsupported serving component '{}'",
            descriptor.id,
            descriptor.components.serving
        ));
    }
    let capabilities = context.capabilities();
    let max_horizon = capabilities
        .max_horizon
        .filter(|value| *value > 0)
        .ok_or_else(|| anyhow!("Serving environment '{env_id}' must declare max_horizon"))?;
    let metadata = context.metadata();
    let board = metadata.require_board()?;

    Ok(StartupProfile {
        algorithm,
        obs_size: board.observation.elements,
        num_actions: board.action_count,
        max_horizon,
        env_contract_version: capabilities.contract_version,
        model_contract: descriptor.model_artifact_contract(env_id, capabilities.contract_version),
    })
}

/// Stub evaluator type when ONNX is disabled (for testing)
#[cfg(not(feature = "onnx"))]
pub type OnnxEvaluator = ();

/// Stub ModelInfo when ONNX is disabled
#[cfg(not(feature = "onnx"))]
#[derive(Default, Clone)]
pub struct ModelInfo {
    pub loaded: bool,
    pub checkpoint_id: Option<String>,
    pub model_sha256: Option<String>,
    pub path: Option<String>,
    pub loaded_at: Option<u64>,
    pub training_step: Option<u64>,
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
fn configure_cors(allowed_origins: &[String]) -> Result<CorsLayer> {
    if allowed_origins.is_empty() {
        // Deny-by-default: only allow localhost origins when none configured
        warn!(
            component = "web",
            event = "cors_localhost_fallback",
            "CORS: No allowed_origins configured - restricting to localhost only"
        );
        let origins = [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:8080",
        ]
        .into_iter()
        .map(HeaderValue::from_static)
        .collect::<Vec<_>>();

        Ok(CorsLayer::new()
            .allow_origin(AllowOrigin::list(origins))
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([header::CONTENT_TYPE, header::ACCEPT])
            .allow_credentials(true))
    } else {
        // Production mode: restrict to configured origins
        let origins = allowed_origins
            .iter()
            .map(|origin| {
                origin.parse::<HeaderValue>().map_err(|error| {
                    anyhow!("web.allowed_origins contains invalid HTTP origin {origin:?}: {error}")
                })
            })
            .collect::<Result<Vec<_>>>()?;

        info!(
            component = "web",
            event = "cors_configured",
            origins = ?allowed_origins,
            "CORS: Allowing configured origins"
        );

        Ok(CorsLayer::new()
            .allow_origin(AllowOrigin::list(origins))
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            .allow_headers([header::CONTENT_TYPE, header::ACCEPT])
            .allow_credentials(true))
    }
}

/// Create the application router with the given state and allowed origins.
/// This is separated out for testing purposes.
pub fn create_app_with_cors(state: Arc<AppState>, allowed_origins: &[String]) -> Result<Router> {
    let cors = configure_cors(allowed_origins)?;

    Ok(Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        .route("/games", get(list_games))
        .route("/game-info/:id", get(get_game_info))
        .route("/game/new", post(new_game))
        .route("/game/state", get(get_game_state))
        .route("/move", post(make_move))
        .route("/stats", get(get_stats))
        .route("/model", get(get_model_info))
        .layer(cors)
        .with_state(state))
}

/// Create the application router with the given state.
/// Uses permissive CORS (empty allowed_origins = development mode).
pub fn create_app(state: Arc<AppState>) -> Router {
    create_app_with_cors(state, &[]).expect("hard-coded localhost CORS origins are valid")
}

/// Create application state for testing (no model watcher, no logging)
#[cfg(test)]
pub fn create_test_state() -> Arc<AppState> {
    engine_games::register_all_environments();
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

#[cfg(test)]
mod startup_profile_tests {
    use super::*;

    #[test]
    fn configured_algorithm_and_environment_resolve_to_a_model_contract() {
        engine_games::register_all_environments();
        let profile =
            resolve_startup_profile(algorithm_core::ALPHAZERO_BOARD_V1_ID, "connect4").unwrap();

        assert_eq!(profile.model_contract.env_id, "connect4");
        assert_eq!(
            profile.model_contract.model_contract,
            "onnx_policy_value_v1"
        );
        assert_eq!(profile.obs_size, 93);
        assert_eq!(profile.num_actions, 7);
        assert_eq!(profile.env_contract_version, 1);
        assert_eq!(profile.model_contract.env_contract_version, 1);
    }

    #[test]
    fn unknown_environment_is_rejected_instead_of_substituted() {
        engine_games::register_all_environments();
        let error =
            resolve_startup_profile(algorithm_core::ALPHAZERO_BOARD_V1_ID, "not-a-real-game")
                .err()
                .expect("unknown environment must fail")
                .to_string();

        assert!(error.contains("not-a-real-game"));
        assert!(error.contains("not registered"));
    }
}
