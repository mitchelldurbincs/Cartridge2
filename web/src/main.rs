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
#[cfg(all(feature = "onnx", feature = "s3"))]
use model_watcher::s3::{S3Config, S3ModelWatcher};
#[cfg(feature = "onnx")]
use model_watcher::{ModelLoadSpec, ModelSelection, ModelWatcher};

use algorithm_core::RuntimeProfile;

use startup::{resolve_startup_profile, shutdown_signal};
// Re-export the server plumbing used by handlers and route tests.
#[cfg(test)]
pub use startup::create_test_state;
pub use startup::{create_app, create_app_with_cors, AppState, ModelInfo, OnnxEvaluator};

#[cfg(feature = "onnx")]
async fn initialize_model_watcher(
    storage: &engine_config::StorageConfig,
    profile: &RuntimeProfile,
    data_root: &str,
    startup_profile: &startup::StartupProfile,
    evaluator: Arc<StdRwLock<Option<OnnxEvaluator>>>,
) -> anyhow::Result<Arc<StdRwLock<ModelInfo>>> {
    let obs_size = startup_profile.obs_size;
    let num_actions = startup_profile.num_actions;
    let identity = startup_profile.model_contract.clone();
    let model_spec = ModelLoadSpec::new(
        obs_size,
        num_actions,
        1,
        startup_profile.max_horizon,
        identity,
    )?;

    let (model_info, mut updates) = match storage.model_backend.as_str() {
        "filesystem" => {
            let model_dir = profile.model_dir(data_root);
            tokio::fs::create_dir_all(&model_dir).await?;
            let watcher = ModelWatcher::new(
                &model_dir,
                model_spec,
                ModelSelection::ChampionOrLatest,
                evaluator,
            );
            let loaded = watcher.try_load_existing()?;
            info!(
                component = "web",
                event = if loaded { "model_loaded" } else { "model_not_found" },
                channel = %model_dir.join("channels/current.json").display(),
                "Filesystem model startup check complete"
            );
            let model_info = watcher.model_info();
            let updates = watcher.start_watching().await?;
            (model_info, updates)
        }
        "s3" => {
            #[cfg(feature = "s3")]
            {
                let bucket = storage.s3_bucket.clone().ok_or_else(|| {
                    anyhow::anyhow!("storage.s3_bucket is required for S3 model watching")
                })?;
                let prefix = profile.model_prefix();
                let watcher = S3ModelWatcher::new(
                    S3Config {
                        bucket: bucket.clone(),
                        prefix: prefix.clone(),
                        endpoint_url: storage.s3_endpoint.clone(),
                        region: None,
                        cache_dir: std::env::temp_dir()
                            .join("cartridge-model-cache")
                            .join(profile.storage_prefix()),
                    },
                    model_spec,
                    ModelSelection::ChampionOrLatest,
                    evaluator,
                )
                .await?;
                let loaded = watcher.try_load_existing().await?;
                info!(
                    component = "web",
                    event = if loaded { "model_loaded" } else { "model_not_found" },
                    channel = %format!("s3://{bucket}/{prefix}/channels/current.json"),
                    "S3 model startup check complete"
                );
                let model_info = watcher.model_info();
                let updates = watcher.start_watching().await?;
                (model_info, updates)
            }
            #[cfg(not(feature = "s3"))]
            {
                anyhow::bail!(
                    "storage.model_backend is 's3' but the web binary was built without the s3 feature"
                )
            }
        }
        backend => anyhow::bail!("unsupported model storage backend '{backend}'"),
    };

    tokio::spawn(async move {
        while updates.recv().await.is_some() {
            info!(
                component = "web",
                event = "model_updated",
                "Model updated - future games will use the new model"
            );
        }
    });
    Ok(model_info)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load configuration first (needed for logging config)
    let config = load_config()?;

    // Initialize tracing with JSON support for cloud deployments
    engine_config::init_tracing("info", &["web=info"], &config.logging);

    // Initialize Prometheus metrics
    metrics::init_metrics();
    info!(component = "web", "Prometheus metrics initialized");

    // Register all games
    engine_games::register_all_environments();
    info!(component = "web", "Registered all games");

    let data_root = config.common.data_dir.clone();
    let default_game = config.common.env_id.clone();
    let startup_profile = resolve_startup_profile(&config.algorithm.id, &default_game)?;
    let algorithm = startup_profile.algorithm.descriptor();
    let runtime_profile = RuntimeProfile::new(
        algorithm.id,
        default_game.clone(),
        startup_profile.env_contract_version,
    )?;
    let data_dir = runtime_profile.data_dir(&data_root).display().to_string();
    info!(
        component = "web",
        data_root = %data_root,
        data_dir = %data_dir,
        default_game = %default_game,
        algorithm = algorithm.id,
        model_contract = %startup_profile.model_contract.model_contract,
        observation_elements = startup_profile.obs_size,
        action_count = startup_profile.num_actions,
        max_horizon = startup_profile.max_horizon,
        host = %config.web.host,
        port = config.web.port,
        "Web server configuration loaded"
    );

    // Set up shared evaluator for model hot-reloading
    // Uses std::sync::RwLock because it's shared with model_watcher crate
    let evaluator: Arc<StdRwLock<Option<OnnxEvaluator>>> = Arc::new(StdRwLock::new(None));

    #[cfg(feature = "onnx")]
    let model_info = initialize_model_watcher(
        &config.storage,
        &runtime_profile,
        &data_root,
        &startup_profile,
        Arc::clone(&evaluator),
    )
    .await?;

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
    let app = create_app_with_cors(state, &config.web.allowed_origins)?;

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
