//! Configuration loading logic.
//!
//! Handles loading config from files and applying environment variable overrides.

use crate::CentralConfig;
use std::fmt;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Standard locations to search for config.toml
pub const CONFIG_SEARCH_PATHS: &[&str] = &[
    "config.toml",      // Current directory
    "../config.toml",   // Parent directory (when running from subdirectory)
    "/app/config.toml", // Docker container
];

/// A config file was found but could not be used.
///
/// Deliberately not raised when no config file exists at all: running on the
/// built-in defaults is a normal production state (the Kubernetes manifests
/// mount no `config.toml`, and the images ship only `config.defaults.toml`).
/// The distinction that matters is *found but broken*, which is always an
/// operator mistake.
#[derive(Debug)]
pub enum ConfigError {
    /// The file exists but could not be read.
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The file was read but is not valid TOML, or does not match the schema.
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
}

impl ConfigError {
    /// The config file the error refers to.
    pub fn path(&self) -> &Path {
        match self {
            ConfigError::Read { path, .. } | ConfigError::Parse { path, .. } => path,
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Read { path, source } => {
                write!(f, "failed to read config file {}: {source}", path.display())
            }
            ConfigError::Parse { path, source } => {
                write!(
                    f,
                    "failed to parse config file {}: {source}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Read { source, .. } => Some(source),
            ConfigError::Parse { source, .. } => Some(source),
        }
    }
}

/// Locate the config file, if there is one.
///
/// Search order:
/// 1. Path specified by CARTRIDGE_CONFIG environment variable
/// 2. Current directory (config.toml)
/// 3. Parent directory (../config.toml)
/// 4. Docker container path (/app/config.toml)
fn find_config_file() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("CARTRIDGE_CONFIG") {
        let path = PathBuf::from(&path);
        if path.exists() {
            info!("Loading config from CARTRIDGE_CONFIG: {}", path.display());
            return Some(path);
        }
        warn!(
            "CARTRIDGE_CONFIG={} not found, searching defaults",
            path.display()
        );
    }

    for path_str in CONFIG_SEARCH_PATHS {
        let path = PathBuf::from(path_str);
        if path.exists() {
            info!("Loading config from {}", path.display());
            return Some(path);
        }
    }

    None
}

/// Load the central configuration, reporting a broken config file as an error.
///
/// This is the entry point binaries should use: a `config.toml` that exists but
/// cannot be parsed is an operator mistake that must stop startup, not
/// something to paper over. Silently substituting defaults means `env_id`
/// reverts from whatever the operator configured to the built-in default, and
/// self-play then fills the replay buffer with the wrong game -- discoverable
/// only hours later, from the model.
///
/// No config file at all is *not* an error: it returns the built-in defaults
/// (from the compile-time embed of `config.defaults.toml`) with environment
/// overrides applied.
pub fn try_load_config() -> Result<CentralConfig, ConfigError> {
    match find_config_file() {
        Some(path) => load_from_path(&path),
        None => {
            debug!("No config.toml found, using built-in defaults");
            Ok(apply_env_overrides(CentralConfig::default()))
        }
    }
}

/// Load the central configuration from config.toml, falling back to defaults.
///
/// Retained for callers that cannot propagate an error -- notably the actor's
/// `Lazy<CentralConfig>`, which is forced from clap `default_value_t`
/// expressions. Those callers must run [`try_load_config`] first, at a point
/// where they *can* fail, so a broken file has already aborted startup by the
/// time this runs. Prefer [`try_load_config`] everywhere else.
pub fn load_config() -> CentralConfig {
    match try_load_config() {
        Ok(config) => config,
        Err(e) => {
            warn!("{e}; using defaults");
            apply_env_overrides(CentralConfig::default())
        }
    }
}

/// Load configuration from a specific path.
pub fn load_from_path(path: &Path) -> Result<CentralConfig, ConfigError> {
    let content = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;

    let config = toml::from_str(&content).map_err(|source| ConfigError::Parse {
        path: path.to_path_buf(),
        source,
    })?;

    Ok(apply_env_overrides(config))
}

/// Macro to reduce env override boilerplate.
///
/// The `parse` arms deliberately warn rather than abort on an unparseable
/// value: environment variables are set by orchestration layers (compose,
/// Kubernetes, the Python loop) where refusing to start is more disruptive
/// than continuing on the configured value. They must not be *silent* though —
/// `CARTRIDGE_TRAINING_BATCH_SIZE=sixty` previously did nothing at all, with
/// no way to tell it apart from not having set it.
macro_rules! env_override {
    // String field
    ($config:expr, $section:ident . $field:ident, $key:expr) => {
        if let Ok(v) = std::env::var($key) {
            $config.$section.$field = v;
        }
    };
    // Parseable field (i32, u64, f64, etc.)
    ($config:expr, $section:ident . $field:ident, $key:expr, parse) => {
        if let Ok(raw) = std::env::var($key) {
            match raw.parse() {
                Ok(v) => $config.$section.$field = v,
                Err(_) => warn!(
                    "Ignoring {}={:?}: not a valid {}",
                    $key,
                    raw,
                    std::stringify!($field)
                ),
            }
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
        if let Ok(raw) = std::env::var($key) {
            match raw.parse() {
                Ok(v) => $config.$section.$field = Some(v),
                Err(_) => warn!(
                    "Ignoring {}={:?}: not a valid {}",
                    $key,
                    raw,
                    std::stringify!($field)
                ),
            }
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
    env_override!(
        config,
        mcts.num_simulations,
        "CARTRIDGE_MCTS_NUM_SIMULATIONS",
        parse
    );
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
