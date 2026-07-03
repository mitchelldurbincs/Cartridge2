//! Configuration loading logic.
//!
//! Handles loading config from files and applying environment variable overrides.

use crate::CentralConfig;
use std::path::PathBuf;
use tracing::{debug, info, warn};

/// Standard locations to search for config.toml
pub const CONFIG_SEARCH_PATHS: &[&str] = &[
    "config.toml",      // Current directory
    "../config.toml",   // Parent directory (when running from subdirectory)
    "/app/config.toml", // Docker container
];

/// Load the central configuration from config.toml.
///
/// Searches for config.toml in the following order:
/// 1. Path specified by CARTRIDGE_CONFIG environment variable
/// 2. Current directory (config.toml)
/// 3. Parent directory (../config.toml)
/// 4. Docker container path (/app/config.toml)
///
/// After loading, environment variable overrides are applied.
pub fn load_config() -> CentralConfig {
    // Check for explicit config path
    if let Ok(path) = std::env::var("CARTRIDGE_CONFIG") {
        let path = PathBuf::from(&path);
        if path.exists() {
            info!("Loading config from CARTRIDGE_CONFIG: {}", path.display());
            return load_from_path(&path);
        }
        warn!(
            "CARTRIDGE_CONFIG={} not found, searching defaults",
            path.display()
        );
    }

    // Search default locations
    for path_str in CONFIG_SEARCH_PATHS {
        let path = PathBuf::from(path_str);
        if path.exists() {
            info!("Loading config from {}", path.display());
            return load_from_path(&path);
        }
    }

    // Fall back to defaults
    debug!("No config.toml found, using built-in defaults");
    apply_env_overrides(CentralConfig::default())
}

/// Load configuration from a specific path.
pub fn load_from_path(path: &PathBuf) -> CentralConfig {
    match std::fs::read_to_string(path) {
        Ok(content) => match toml::from_str(&content) {
            Ok(config) => apply_env_overrides(config),
            Err(e) => {
                warn!("Failed to parse {}: {}, using defaults", path.display(), e);
                apply_env_overrides(CentralConfig::default())
            }
        },
        Err(e) => {
            warn!("Failed to read {}: {}, using defaults", path.display(), e);
            apply_env_overrides(CentralConfig::default())
        }
    }
}

/// Macro to reduce env override boilerplate
macro_rules! env_override {
    // String field
    ($config:expr, $section:ident . $field:ident, $key:expr) => {
        if let Ok(v) = std::env::var($key) {
            $config.$section.$field = v;
        }
    };
    // Parseable field (i32, u64, f64, etc.)
    ($config:expr, $section:ident . $field:ident, $key:expr, parse) => {
        if let Ok(v) =
            std::env::var($key).and_then(|s| s.parse().map_err(|_| std::env::VarError::NotPresent))
        {
            $config.$section.$field = v;
        }
    };
    // Optional string field
    ($config:expr, $section:ident . $field:ident, $key:expr, optional) => {
        if let Ok(v) = std::env::var($key) {
            $config.$section.$field = Some(v);
        }
    };
    // Optional parseable field (Option<i32>, Option<u64>, etc.)
    ($config:expr, $section:ident . $field:ident, $key:expr, optional_parse) => {
        if let Ok(v) =
            std::env::var($key).and_then(|s| s.parse().map_err(|_| std::env::VarError::NotPresent))
        {
            $config.$section.$field = Some(v);
        }
    };
}

/// Apply environment variable overrides to a configuration.
///
/// Environment variables follow the pattern: CARTRIDGE_<SECTION>_<KEY>
pub fn apply_env_overrides(mut config: CentralConfig) -> CentralConfig {
    // Common
    env_override!(config, common.env_id, "CARTRIDGE_COMMON_ENV_ID");
    env_override!(config, common.data_dir, "CARTRIDGE_COMMON_DATA_DIR");
    env_override!(config, common.log_level, "CARTRIDGE_COMMON_LOG_LEVEL");

    // Training
    env_override!(
        config,
        training.iterations,
        "CARTRIDGE_TRAINING_ITERATIONS",
        parse
    );
    env_override!(
        config,
        training.start_iteration,
        "CARTRIDGE_TRAINING_START_ITERATION",
        parse
    );
    env_override!(
        config,
        training.episodes_per_iteration,
        "CARTRIDGE_TRAINING_EPISODES_PER_ITERATION",
        parse
    );
    env_override!(
        config,
        training.steps_per_iteration,
        "CARTRIDGE_TRAINING_STEPS_PER_ITERATION",
        parse
    );
    env_override!(
        config,
        training.batch_size,
        "CARTRIDGE_TRAINING_BATCH_SIZE",
        parse
    );
    env_override!(
        config,
        training.learning_rate,
        "CARTRIDGE_TRAINING_LEARNING_RATE",
        parse
    );
    env_override!(
        config,
        training.weight_decay,
        "CARTRIDGE_TRAINING_WEIGHT_DECAY",
        parse
    );
    env_override!(
        config,
        training.grad_clip_norm,
        "CARTRIDGE_TRAINING_GRAD_CLIP_NORM",
        parse
    );
    env_override!(config, training.device, "CARTRIDGE_TRAINING_DEVICE");
    env_override!(
        config,
        training.checkpoint_interval,
        "CARTRIDGE_TRAINING_CHECKPOINT_INTERVAL",
        parse
    );
    env_override!(
        config,
        training.max_checkpoints,
        "CARTRIDGE_TRAINING_MAX_CHECKPOINTS",
        parse
    );

    // Evaluation
    env_override!(
        config,
        evaluation.interval,
        "CARTRIDGE_EVALUATION_INTERVAL",
        parse
    );
    env_override!(
        config,
        evaluation.games,
        "CARTRIDGE_EVALUATION_GAMES",
        parse
    );

    // Actor
    env_override!(config, actor.actor_id, "CARTRIDGE_ACTOR_ACTOR_ID");
    env_override!(
        config,
        actor.max_episodes,
        "CARTRIDGE_ACTOR_MAX_EPISODES",
        parse
    );
    env_override!(
        config,
        actor.episode_timeout_secs,
        "CARTRIDGE_ACTOR_EPISODE_TIMEOUT_SECS",
        parse
    );
    env_override!(
        config,
        actor.flush_interval_secs,
        "CARTRIDGE_ACTOR_FLUSH_INTERVAL_SECS",
        parse
    );
    env_override!(
        config,
        actor.log_interval,
        "CARTRIDGE_ACTOR_LOG_INTERVAL",
        parse
    );

    // Web
    env_override!(config, web.host, "CARTRIDGE_WEB_HOST");
    env_override!(config, web.port, "CARTRIDGE_WEB_PORT", parse);

    // MCTS
    env_override!(config, mcts.c_puct, "CARTRIDGE_MCTS_C_PUCT", parse);
    env_override!(
        config,
        mcts.temperature,
        "CARTRIDGE_MCTS_TEMPERATURE",
        parse
    );
    env_override!(
        config,
        mcts.dirichlet_alpha,
        "CARTRIDGE_MCTS_DIRICHLET_ALPHA",
        parse
    );
    env_override!(
        config,
        mcts.dirichlet_weight,
        "CARTRIDGE_MCTS_DIRICHLET_WEIGHT",
        parse
    );
    env_override!(
        config,
        mcts.eval_batch_size,
        "CARTRIDGE_MCTS_EVAL_BATCH_SIZE",
        parse
    );
    env_override!(
        config,
        mcts.onnx_intra_threads,
        "CARTRIDGE_MCTS_ONNX_INTRA_THREADS",
        parse
    );
    env_override!(config, mcts.start_sims, "CARTRIDGE_MCTS_START_SIMS", parse);
    env_override!(config, mcts.max_sims, "CARTRIDGE_MCTS_MAX_SIMS", parse);
    env_override!(
        config,
        mcts.sim_ramp_rate,
        "CARTRIDGE_MCTS_SIM_RAMP_RATE",
        parse
    );

    // Storage
    env_override!(
        config,
        storage.model_backend,
        "CARTRIDGE_STORAGE_MODEL_BACKEND"
    );
    env_override!(
        config,
        storage.postgres_url,
        "CARTRIDGE_STORAGE_POSTGRES_URL",
        optional
    );
    env_override!(
        config,
        storage.s3_bucket,
        "CARTRIDGE_STORAGE_S3_BUCKET",
        optional
    );
    env_override!(
        config,
        storage.s3_endpoint,
        "CARTRIDGE_STORAGE_S3_ENDPOINT",
        optional
    );
    env_override!(
        config,
        storage.pool_max_size,
        "CARTRIDGE_STORAGE_POOL_MAX_SIZE",
        parse
    );
    env_override!(
        config,
        storage.pool_connect_timeout,
        "CARTRIDGE_STORAGE_POOL_CONNECT_TIMEOUT",
        parse
    );
    env_override!(
        config,
        storage.pool_idle_timeout,
        "CARTRIDGE_STORAGE_POOL_IDLE_TIMEOUT",
        optional_parse
    );

    config
}
