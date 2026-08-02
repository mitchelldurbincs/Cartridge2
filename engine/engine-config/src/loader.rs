//! Configuration loading logic.
//!
//! Handles loading config from files and applying environment variable overrides.

use crate::CentralConfig;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, info};

/// Errors that make the requested process configuration unusable.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("CARTRIDGE_CONFIG points to missing file {path}")]
    MissingExplicitPath { path: PathBuf },
    #[error("failed to read configuration file {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse configuration file {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("invalid value {value:?} for environment variable {key}: {reason}")]
    InvalidEnvironment {
        key: &'static str,
        value: String,
        reason: String,
    },
    #[error("invalid configuration value {field}: {reason}")]
    InvalidValue {
        field: &'static str,
        reason: &'static str,
    },
}

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
pub fn load_config() -> Result<CentralConfig, ConfigError> {
    // Check for explicit config path
    if let Some(path) = std::env::var_os("CARTRIDGE_CONFIG") {
        if !path.is_empty() {
            let path = PathBuf::from(path);
            if !path.exists() {
                return Err(ConfigError::MissingExplicitPath { path });
            }
            info!("Loading config from CARTRIDGE_CONFIG: {}", path.display());
            return load_from_path(&path);
        }
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
pub fn load_from_path(path: &Path) -> Result<CentralConfig, ConfigError> {
    let content = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let config = toml::from_str(&content).map_err(|source| ConfigError::Parse {
        path: path.to_path_buf(),
        source,
    })?;
    apply_env_overrides(config)
}

fn env_value(key: &'static str) -> Result<Option<String>, ConfigError> {
    match std::env::var(key) {
        Ok(value) if value.is_empty() => Ok(None),
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(ConfigError::InvalidEnvironment {
            key,
            value: "<non-Unicode>".to_string(),
            reason: "expected valid Unicode".to_string(),
        }),
    }
}

fn parsed_env_value<T>(key: &'static str) -> Result<Option<T>, ConfigError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(value) = env_value(key)? else {
        return Ok(None);
    };

    value
        .parse()
        .map(Some)
        .map_err(|error: T::Err| ConfigError::InvalidEnvironment {
            key,
            value,
            reason: error.to_string(),
        })
}

fn string_list_env_value(key: &'static str) -> Result<Option<Vec<String>>, ConfigError> {
    let Some(value) = env_value(key)? else {
        return Ok(None);
    };
    serde_json::from_str(&value)
        .map(Some)
        .map_err(|error| ConfigError::InvalidEnvironment {
            key,
            value,
            reason: format!("expected a JSON array of strings: {error}"),
        })
}

fn bool_env_value(key: &'static str) -> Result<Option<bool>, ConfigError> {
    let Some(value) = env_value(key)? else {
        return Ok(None);
    };
    let parsed = match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => true,
        "false" | "0" | "no" | "off" => false,
        _ => {
            return Err(ConfigError::InvalidEnvironment {
                key,
                value,
                reason: "expected true/false, 1/0, yes/no, or on/off".to_string(),
            })
        }
    };
    Ok(Some(parsed))
}

/// Macro to reduce env override boilerplate
macro_rules! env_override {
    // String field
    ($config:expr, $section:ident . $field:ident, $key:expr) => {
        if let Some(v) = env_value($key)? {
            $config.$section.$field = v;
        }
    };
    // Parseable field (i32, u64, f64, etc.)
    ($config:expr, $section:ident . $field:ident, $key:expr, parse) => {
        if let Some(v) = parsed_env_value($key)? {
            $config.$section.$field = v;
        }
    };
    // Optional string field
    ($config:expr, $section:ident . $field:ident, $key:expr, optional) => {
        if let Some(v) = env_value($key)? {
            $config.$section.$field = Some(v);
        }
    };
    // Optional parseable field (Option<i32>, Option<u64>, etc.)
    ($config:expr, $section:ident . $field:ident, $key:expr, optional_parse) => {
        if let Some(v) = parsed_env_value($key)? {
            $config.$section.$field = Some(v);
        }
    };
}

/// Apply environment variable overrides to a configuration.
///
/// Environment variables follow the pattern: CARTRIDGE_<SECTION>_<KEY>
pub fn apply_env_overrides(mut config: CentralConfig) -> Result<CentralConfig, ConfigError> {
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
    env_override!(
        config,
        training.num_actors,
        "CARTRIDGE_TRAINING_NUM_ACTORS",
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
    env_override!(
        config,
        evaluation.win_threshold,
        "CARTRIDGE_EVALUATION_WIN_THRESHOLD",
        parse
    );
    if let Some(value) = bool_env_value("CARTRIDGE_EVALUATION_EVAL_VS_RANDOM")? {
        config.evaluation.eval_vs_random = value;
    }

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
    env_override!(
        config,
        actor.health_port,
        "CARTRIDGE_ACTOR_HEALTH_PORT",
        parse
    );

    // Web
    env_override!(config, web.host, "CARTRIDGE_WEB_HOST");
    env_override!(config, web.port, "CARTRIDGE_WEB_PORT", parse);
    if let Some(value) = string_list_env_value("CARTRIDGE_WEB_ALLOWED_ORIGINS")? {
        config.web.allowed_origins = value;
    }

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
        mcts.temp_threshold,
        "CARTRIDGE_MCTS_TEMP_THRESHOLD",
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

    // Logging
    env_override!(config, logging.format, "CARTRIDGE_LOGGING_FORMAT");
    if let Some(value) = bool_env_value("CARTRIDGE_LOGGING_INCLUDE_TIMESTAMPS")? {
        config.logging.include_timestamps = value;
    }
    if let Some(value) = bool_env_value("CARTRIDGE_LOGGING_INCLUDE_TARGET")? {
        config.logging.include_target = value;
    }

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

    validate_config(&config)?;
    Ok(config)
}

fn invalid(field: &'static str, reason: &'static str) -> ConfigError {
    ConfigError::InvalidValue { field, reason }
}

/// Validate values whose invalid ranges have unambiguous failure modes.
fn validate_config(config: &CentralConfig) -> Result<(), ConfigError> {
    if config.common.data_dir.trim().is_empty() {
        return Err(invalid("common.data_dir", "must not be empty"));
    }
    if config.common.env_id.trim().is_empty() {
        return Err(invalid("common.env_id", "must not be empty"));
    }
    if !matches!(
        config.common.log_level.to_ascii_lowercase().as_str(),
        "trace" | "debug" | "info" | "warn" | "error"
    ) {
        return Err(invalid(
            "common.log_level",
            "must be one of trace, debug, info, warn, error",
        ));
    }

    for (field, value) in [
        ("training.iterations", config.training.iterations),
        ("training.start_iteration", config.training.start_iteration),
        (
            "training.episodes_per_iteration",
            config.training.episodes_per_iteration,
        ),
        (
            "training.steps_per_iteration",
            config.training.steps_per_iteration,
        ),
        ("training.batch_size", config.training.batch_size),
        (
            "training.checkpoint_interval",
            config.training.checkpoint_interval,
        ),
        ("training.num_actors", config.training.num_actors),
    ] {
        if value <= 0 {
            return Err(invalid(field, "must be greater than zero"));
        }
    }
    if config.training.max_checkpoints < 0 {
        return Err(invalid(
            "training.max_checkpoints",
            "must be zero or greater",
        ));
    }
    if !config.training.learning_rate.is_finite() || config.training.learning_rate <= 0.0 {
        return Err(invalid(
            "training.learning_rate",
            "must be finite and greater than zero",
        ));
    }
    if !config.training.weight_decay.is_finite() || config.training.weight_decay < 0.0 {
        return Err(invalid(
            "training.weight_decay",
            "must be finite and zero or greater",
        ));
    }
    if !config.training.grad_clip_norm.is_finite() || config.training.grad_clip_norm < 0.0 {
        return Err(invalid(
            "training.grad_clip_norm",
            "must be finite and zero or greater",
        ));
    }
    if !matches!(
        config.training.device.to_ascii_lowercase().as_str(),
        "auto" | "cpu" | "cuda" | "mps"
    ) {
        return Err(invalid(
            "training.device",
            "must be one of auto, cpu, cuda, mps",
        ));
    }

    if config.evaluation.interval < 0 {
        return Err(invalid("evaluation.interval", "must be zero or greater"));
    }
    if config.evaluation.games <= 0 {
        return Err(invalid("evaluation.games", "must be greater than zero"));
    }
    if !config.evaluation.win_threshold.is_finite()
        || !(0.0..=1.0).contains(&config.evaluation.win_threshold)
    {
        return Err(invalid(
            "evaluation.win_threshold",
            "must be finite and between zero and one",
        ));
    }

    if config.actor.actor_id.trim().is_empty() {
        return Err(invalid("actor.actor_id", "must not be empty"));
    }
    if config.actor.max_episodes != -1 && config.actor.max_episodes <= 0 {
        return Err(invalid(
            "actor.max_episodes",
            "must be -1 (unlimited) or greater than zero",
        ));
    }
    if config.actor.episode_timeout_secs == 0 {
        return Err(invalid(
            "actor.episode_timeout_secs",
            "must be greater than zero",
        ));
    }
    if config.actor.flush_interval_secs == 0 {
        return Err(invalid(
            "actor.flush_interval_secs",
            "must be greater than zero",
        ));
    }
    if config.actor.health_port == 0 {
        return Err(invalid("actor.health_port", "must be greater than zero"));
    }

    if config.web.host.trim().is_empty() {
        return Err(invalid("web.host", "must not be empty"));
    }
    if config.web.port == 0 {
        return Err(invalid("web.port", "must be greater than zero"));
    }
    if config
        .web
        .allowed_origins
        .iter()
        .any(|value| value.trim().is_empty())
    {
        return Err(invalid(
            "web.allowed_origins",
            "must not contain empty origins",
        ));
    }

    if config.mcts.num_simulations == 0 {
        return Err(invalid("mcts.num_simulations", "must be greater than zero"));
    }
    for (field, value) in [
        ("mcts.c_puct", config.mcts.c_puct),
        ("mcts.temperature", config.mcts.temperature),
        ("mcts.dirichlet_alpha", config.mcts.dirichlet_alpha),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(invalid(field, "must be finite and zero or greater"));
        }
    }
    if !config.mcts.dirichlet_weight.is_finite()
        || !(0.0..=1.0).contains(&config.mcts.dirichlet_weight)
    {
        return Err(invalid(
            "mcts.dirichlet_weight",
            "must be finite and between zero and one",
        ));
    }
    if config.mcts.eval_batch_size == 0 {
        return Err(invalid("mcts.eval_batch_size", "must be greater than zero"));
    }
    if config.mcts.start_sims == 0 || config.mcts.max_sims == 0 {
        return Err(invalid(
            "mcts.start_sims/mcts.max_sims",
            "must be greater than zero",
        ));
    }
    if config.mcts.start_sims > config.mcts.max_sims {
        return Err(invalid("mcts.start_sims", "must not exceed mcts.max_sims"));
    }

    if !matches!(
        config.logging.format.to_ascii_lowercase().as_str(),
        "text" | "json"
    ) {
        return Err(invalid("logging.format", "must be either text or json"));
    }
    if !matches!(
        config.storage.model_backend.to_ascii_lowercase().as_str(),
        "filesystem" | "s3"
    ) {
        return Err(invalid(
            "storage.model_backend",
            "must be either filesystem or s3",
        ));
    }
    if config.storage.pool_max_size == 0 {
        return Err(invalid(
            "storage.pool_max_size",
            "must be greater than zero",
        ));
    }
    if config.storage.pool_connect_timeout == 0 {
        return Err(invalid(
            "storage.pool_connect_timeout",
            "must be greater than zero",
        ));
    }
    if config.storage.pool_idle_timeout == Some(0) {
        return Err(invalid(
            "storage.pool_idle_timeout",
            "must be greater than zero when set",
        ));
    }

    Ok(())
}
