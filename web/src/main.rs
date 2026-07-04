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

use std::sync::Arc;
use std::sync::RwLock as StdRwLock;
use tokio::sync::{Mutex, RwLock};
use tracing::info;

mod game;
mod handlers;
mod metrics;
mod startup;
mod types;

use engine_config::load_config;
use game::GameSession;
#[cfg(feature = "onnx")]
use model_watcher::ModelWatcher;

use startup::shutdown_signal;
// Re-export server plumbing so the public paths (`crate::AppState`,
// `crate::create_app`, `crate::ModelInfo`, ...) stay stable for handlers and tests.
#[cfg(test)]
pub use startup::create_test_state;
pub use startup::{create_app, create_app_with_cors, AppState, ModelInfo, OnnxEvaluator};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load configuration first (needed for logging config)
    let config = load_config();

    // Initialize tracing with JSON support for cloud deployments
    engine_config::init_tracing("info", &["web=info"], &config.logging);

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
#[path = "main_tests.rs"]
mod tests;
