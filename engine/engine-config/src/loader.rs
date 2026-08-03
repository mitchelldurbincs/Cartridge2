//! Configuration loading logic.
//!
//! Handles loading config from files and applying environment variable overrides.

use crate::CentralConfig;
use std::any::type_name;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tracing::{debug, info};

/// Configuration errors are fatal: falling back would risk selecting a
/// different environment, algorithm, or storage namespace than requested.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("CARTRIDGE_CONFIG points to missing file: {0}")]
    ExplicitPathMissing(PathBuf),
    #[error("failed to read configuration {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse configuration {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error("environment variable {key} is not valid Unicode")]
    NonUnicodeEnv { key: &'static str },
    #[error("invalid value {value:?} for {key}; expected {expected}")]
    InvalidEnv {
        key: &'static str,
        value: String,
        expected: &'static str,
    },
    #[error("unknown configuration environment override: {0}")]
    UnknownEnv(String),
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

const CONFIG_ENV_SECTION_PREFIXES: &[&str] = &[
    "COMMON_",
    "ALGORITHM_",
    "TRAINING_",
    "EVALUATION_",
    "ACTOR_",
    "WEB_",
    "MCTS_",
    "LOGGING_",
    "STORAGE_",
    "WANDB_",
];

const KNOWN_CONFIG_ENV_KEYS: &[&str] = &[
    "CARTRIDGE_COMMON_ENV_ID",
    "CARTRIDGE_COMMON_DATA_DIR",
    "CARTRIDGE_COMMON_LOG_LEVEL",
    "CARTRIDGE_ALGORITHM_ID",
    "CARTRIDGE_TRAINING_ITERATIONS",
    "CARTRIDGE_TRAINING_EPISODES_PER_ITERATION",
    "CARTRIDGE_TRAINING_STEPS_PER_ITERATION",
    "CARTRIDGE_TRAINING_BATCH_SIZE",
    "CARTRIDGE_TRAINING_LEARNING_RATE",
    "CARTRIDGE_TRAINING_WEIGHT_DECAY",
    "CARTRIDGE_TRAINING_GRAD_CLIP_NORM",
    "CARTRIDGE_TRAINING_DEVICE",
    "CARTRIDGE_TRAINING_CHECKPOINT_INTERVAL",
    "CARTRIDGE_TRAINING_NUM_ACTORS",
    "CARTRIDGE_EVALUATION_INTERVAL",
    "CARTRIDGE_EVALUATION_GAMES",
    "CARTRIDGE_EVALUATION_WIN_THRESHOLD",
    "CARTRIDGE_EVALUATION_EVAL_VS_RANDOM",
    "CARTRIDGE_EVALUATION_SIMULATIONS",
    "CARTRIDGE_EVALUATION_TEMPERATURE",
    "CARTRIDGE_EVALUATION_SOLVER_GAMES",
    "CARTRIDGE_EVALUATION_EVALUATION_SEED",
    "CARTRIDGE_EVALUATION_PROMOTION_METRIC",
    "CARTRIDGE_EVALUATION_PROMOTION_MARGIN",
    "CARTRIDGE_ACTOR_ACTOR_ID",
    "CARTRIDGE_ACTOR_EPISODE_TIMEOUT_SECS",
    "CARTRIDGE_ACTOR_LOG_INTERVAL",
    "CARTRIDGE_WEB_HOST",
    "CARTRIDGE_WEB_PORT",
    "CARTRIDGE_WEB_ALLOWED_ORIGINS",
    "CARTRIDGE_MCTS_C_PUCT",
    "CARTRIDGE_MCTS_TEMPERATURE",
    "CARTRIDGE_MCTS_LATE_TEMPERATURE",
    "CARTRIDGE_MCTS_TEMP_THRESHOLD",
    "CARTRIDGE_MCTS_DIRICHLET_ALPHA",
    "CARTRIDGE_MCTS_DIRICHLET_WEIGHT",
    "CARTRIDGE_MCTS_EVAL_BATCH_SIZE",
    "CARTRIDGE_MCTS_ONNX_INTRA_THREADS",
    "CARTRIDGE_MCTS_START_SIMS",
    "CARTRIDGE_MCTS_MAX_SIMS",
    "CARTRIDGE_MCTS_SIM_RAMP_RATE",
    "CARTRIDGE_LOGGING_FORMAT",
    "CARTRIDGE_LOGGING_INCLUDE_TIMESTAMPS",
    "CARTRIDGE_LOGGING_INCLUDE_TARGET",
    "CARTRIDGE_STORAGE_MODEL_BACKEND",
    "CARTRIDGE_STORAGE_POSTGRES_URL",
    "CARTRIDGE_STORAGE_S3_BUCKET",
    "CARTRIDGE_STORAGE_S3_ENDPOINT",
    "CARTRIDGE_STORAGE_POOL_MAX_SIZE",
    "CARTRIDGE_STORAGE_POOL_CONNECT_TIMEOUT",
    "CARTRIDGE_STORAGE_POOL_IDLE_TIMEOUT",
    "CARTRIDGE_WANDB_ENABLED",
    "CARTRIDGE_WANDB_REQUIRED",
    "CARTRIDGE_WANDB_PROJECT",
    "CARTRIDGE_WANDB_ENTITY",
    "CARTRIDGE_WANDB_GROUP",
    "CARTRIDGE_WANDB_TAGS",
    "CARTRIDGE_WANDB_INIT_TIMEOUT_SECONDS",
];

fn reject_unknown_config_env_vars() -> Result<(), ConfigError> {
    for (key, _) in std::env::vars_os() {
        let Some(key) = key.to_str() else {
            continue;
        };
        let Some(suffix) = key.strip_prefix("CARTRIDGE_") else {
            continue;
        };
        let is_config_section = CONFIG_ENV_SECTION_PREFIXES
            .iter()
            .any(|prefix| suffix.starts_with(prefix));
        if is_config_section && !KNOWN_CONFIG_ENV_KEYS.contains(&key) {
            return Err(ConfigError::UnknownEnv(key.to_string()));
        }
    }
    Ok(())
}

fn validate_nonnegative_f32(name: &str, value: f32) -> Result<(), ConfigError> {
    if !value.is_finite() || value < 0.0 {
        return Err(ConfigError::InvalidConfig(format!(
            "{name} must be a finite nonnegative f32"
        )));
    }
    Ok(())
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
    if let Ok(path) = std::env::var("CARTRIDGE_CONFIG") {
        let path = PathBuf::from(&path);
        if path.is_file() {
            info!("Loading config from CARTRIDGE_CONFIG: {}", path.display());
            return load_from_path(&path);
        }
        return Err(ConfigError::ExplicitPathMissing(path));
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

fn env_string(key: &'static str) -> Result<Option<String>, ConfigError> {
    match std::env::var(key) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(ConfigError::NonUnicodeEnv { key }),
    }
}

fn env_value<T>(key: &'static str) -> Result<Option<T>, ConfigError>
where
    T: FromStr,
{
    let Some(value) = env_string(key)? else {
        return Ok(None);
    };
    value
        .parse()
        .map(Some)
        .map_err(|_| ConfigError::InvalidEnv {
            key,
            value,
            expected: type_name::<T>(),
        })
}

fn env_string_list(key: &'static str) -> Result<Option<Vec<String>>, ConfigError> {
    let Some(value) = env_string(key)? else {
        return Ok(None);
    };
    if value.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let items = value
        .split(',')
        .map(str::trim)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if items.iter().any(String::is_empty) {
        return Err(ConfigError::InvalidEnv {
            key,
            value,
            expected: "a comma-separated list of non-empty strings",
        });
    }
    Ok(Some(items))
}

/// Macro to reduce env override boilerplate
macro_rules! env_override {
    // String field
    ($config:expr, $section:ident . $field:ident, $key:expr) => {
        if let Some(v) = env_string($key)? {
            $config.$section.$field = v;
        }
    };
    // Parseable field (i32, u64, f64, etc.)
    ($config:expr, $section:ident . $field:ident, $key:expr, parse) => {
        if let Some(v) = env_value($key)? {
            $config.$section.$field = v;
        }
    };
    // Optional string field
    ($config:expr, $section:ident . $field:ident, $key:expr, optional) => {
        if let Some(v) = env_string($key)? {
            $config.$section.$field = Some(v);
        }
    };
    // Optional parseable field (Option<i32>, Option<u64>, etc.)
    ($config:expr, $section:ident . $field:ident, $key:expr, optional_parse) => {
        if let Some(v) = env_value($key)? {
            $config.$section.$field = Some(v);
        }
    };
    // Comma-separated string list
    ($config:expr, $section:ident . $field:ident, $key:expr, list) => {
        if let Some(v) = env_string_list($key)? {
            $config.$section.$field = v;
        }
    };
}

/// Apply environment variable overrides to a configuration.
///
/// Environment variables follow the pattern: CARTRIDGE_<SECTION>_<KEY>
pub fn apply_env_overrides(mut config: CentralConfig) -> Result<CentralConfig, ConfigError> {
    reject_unknown_config_env_vars()?;

    // Common
    env_override!(config, common.env_id, "CARTRIDGE_COMMON_ENV_ID");
    env_override!(config, common.data_dir, "CARTRIDGE_COMMON_DATA_DIR");
    env_override!(config, common.log_level, "CARTRIDGE_COMMON_LOG_LEVEL");

    // Algorithm
    env_override!(config, algorithm.id, "CARTRIDGE_ALGORITHM_ID");

    // Training
    env_override!(
        config,
        training.iterations,
        "CARTRIDGE_TRAINING_ITERATIONS",
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
    env_override!(
        config,
        evaluation.eval_vs_random,
        "CARTRIDGE_EVALUATION_EVAL_VS_RANDOM",
        parse
    );
    env_override!(
        config,
        evaluation.simulations,
        "CARTRIDGE_EVALUATION_SIMULATIONS",
        parse
    );
    env_override!(
        config,
        evaluation.temperature,
        "CARTRIDGE_EVALUATION_TEMPERATURE",
        parse
    );
    env_override!(
        config,
        evaluation.solver_games,
        "CARTRIDGE_EVALUATION_SOLVER_GAMES",
        parse
    );
    env_override!(
        config,
        evaluation.evaluation_seed,
        "CARTRIDGE_EVALUATION_EVALUATION_SEED",
        parse
    );
    env_override!(
        config,
        evaluation.promotion_metric,
        "CARTRIDGE_EVALUATION_PROMOTION_METRIC"
    );
    env_override!(
        config,
        evaluation.promotion_margin,
        "CARTRIDGE_EVALUATION_PROMOTION_MARGIN",
        parse
    );

    // Actor
    env_override!(config, actor.actor_id, "CARTRIDGE_ACTOR_ACTOR_ID");
    env_override!(
        config,
        actor.episode_timeout_secs,
        "CARTRIDGE_ACTOR_EPISODE_TIMEOUT_SECS",
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
    env_override!(
        config,
        web.allowed_origins,
        "CARTRIDGE_WEB_ALLOWED_ORIGINS",
        list
    );

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
        mcts.late_temperature,
        "CARTRIDGE_MCTS_LATE_TEMPERATURE",
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
    env_override!(
        config,
        logging.include_timestamps,
        "CARTRIDGE_LOGGING_INCLUDE_TIMESTAMPS",
        parse
    );
    env_override!(
        config,
        logging.include_target,
        "CARTRIDGE_LOGGING_INCLUDE_TARGET",
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

    // W&B
    env_override!(config, wandb.enabled, "CARTRIDGE_WANDB_ENABLED", parse);
    env_override!(config, wandb.required, "CARTRIDGE_WANDB_REQUIRED", parse);
    env_override!(config, wandb.project, "CARTRIDGE_WANDB_PROJECT");
    env_override!(config, wandb.entity, "CARTRIDGE_WANDB_ENTITY");
    env_override!(config, wandb.group, "CARTRIDGE_WANDB_GROUP");
    env_override!(config, wandb.tags, "CARTRIDGE_WANDB_TAGS", list);
    env_override!(
        config,
        wandb.init_timeout_seconds,
        "CARTRIDGE_WANDB_INIT_TIMEOUT_SECONDS",
        parse
    );

    if config.algorithm.id.trim().is_empty() {
        return Err(ConfigError::InvalidConfig(
            "algorithm.id cannot be empty".to_string(),
        ));
    }
    if config.common.env_id.trim().is_empty() {
        return Err(ConfigError::InvalidConfig(
            "common.env_id cannot be empty".to_string(),
        ));
    }
    if config.common.data_dir.trim().is_empty() {
        return Err(ConfigError::InvalidConfig(
            "common.data_dir cannot be empty".to_string(),
        ));
    }
    if config.common.log_level.parse::<tracing::Level>().is_err() {
        return Err(ConfigError::InvalidConfig(format!(
            "common.log_level is invalid: {:?}",
            config.common.log_level
        )));
    }
    match config.storage.model_backend.as_str() {
        "filesystem" => {}
        "s3" => {
            if config
                .storage
                .s3_bucket
                .as_deref()
                .is_none_or(|bucket| bucket.trim().is_empty())
            {
                return Err(ConfigError::InvalidConfig(
                    "storage.s3_bucket is required when model_backend = 's3'".to_string(),
                ));
            }
        }
        backend => {
            return Err(ConfigError::InvalidConfig(format!(
                "storage.model_backend must be 'filesystem' or 's3', got '{backend}'"
            )));
        }
    }
    if config
        .storage
        .postgres_url
        .as_deref()
        .is_some_and(|url| url.trim().is_empty())
    {
        return Err(ConfigError::InvalidConfig(
            "storage.postgres_url cannot be empty when set".to_string(),
        ));
    }
    if config.storage.pool_max_size == 0 || config.storage.pool_connect_timeout == 0 {
        return Err(ConfigError::InvalidConfig(
            "storage pool size and connect timeout must be greater than zero".to_string(),
        ));
    }
    if !matches!(config.logging.format.as_str(), "text" | "json") {
        return Err(ConfigError::InvalidConfig(format!(
            "logging.format must be 'text' or 'json', got {:?}",
            config.logging.format
        )));
    }
    if !matches!(
        config.evaluation.promotion_metric.as_str(),
        "win_rate" | "solver_optimal"
    ) {
        return Err(ConfigError::InvalidConfig(format!(
            "evaluation.promotion_metric must be 'win_rate' or 'solver_optimal', got {:?}",
            config.evaluation.promotion_metric
        )));
    }
    if config.training.iterations == 0
        || config.training.episodes_per_iteration == 0
        || config.training.steps_per_iteration == 0
        || config.training.batch_size == 0
        || config.training.checkpoint_interval == 0
        || config.training.num_actors == 0
    {
        return Err(ConfigError::InvalidConfig(
            "training count and cadence fields must be positive".to_string(),
        ));
    }
    if config.training.num_actors > config.training.episodes_per_iteration {
        return Err(ConfigError::InvalidConfig(
            "training.num_actors cannot exceed training.episodes_per_iteration".to_string(),
        ));
    }
    if config
        .training
        .iterations
        .checked_mul(config.training.steps_per_iteration)
        .is_none()
    {
        return Err(ConfigError::InvalidConfig(
            "training.iterations * training.steps_per_iteration exceeds u64".to_string(),
        ));
    }
    if config.evaluation.games == 0 {
        return Err(ConfigError::InvalidConfig(
            "evaluation.games must be positive".to_string(),
        ));
    }
    if config.evaluation.solver_games > 0 && config.common.env_id != "connect4" {
        return Err(ConfigError::InvalidConfig(
            "evaluation.solver_games may be nonzero only when common.env_id is 'connect4'"
                .to_string(),
        ));
    }
    let largest_evaluation_run = config.evaluation.games.max(config.evaluation.solver_games);
    if u64::from(largest_evaluation_run)
        .checked_sub(1)
        .and_then(|last_index| config.evaluation.evaluation_seed.checked_add(last_index))
        .is_none()
    {
        return Err(ConfigError::InvalidConfig(
            "evaluation seed schedule exceeds u64".to_string(),
        ));
    }
    if config.evaluation.promotion_metric == "solver_optimal"
        && (config.common.env_id != "connect4" || config.evaluation.solver_games == 0)
    {
        return Err(ConfigError::InvalidConfig(
            "evaluation.promotion_metric='solver_optimal' requires connect4 with evaluation.solver_games > 0"
                .to_string(),
        ));
    }
    validate_nonnegative_f32("evaluation.temperature", config.evaluation.temperature)?;
    for (name, value) in [
        ("evaluation.win_threshold", config.evaluation.win_threshold),
        (
            "evaluation.promotion_margin",
            config.evaluation.promotion_margin,
        ),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(ConfigError::InvalidConfig(format!(
                "{name} must be a finite rate in [0, 1]"
            )));
        }
    }
    match config.evaluation.promotion_metric.as_str() {
        "win_rate" if config.evaluation.promotion_margin != 0.0 => {
            return Err(ConfigError::InvalidConfig(
                "evaluation.promotion_margin must be zero when promotion_metric is 'win_rate'"
                    .to_string(),
            ));
        }
        "solver_optimal" if config.evaluation.win_threshold != 0.0 => {
            return Err(ConfigError::InvalidConfig(
                "evaluation.win_threshold must be zero when promotion_metric is 'solver_optimal'"
                    .to_string(),
            ));
        }
        _ => {}
    }
    for (name, value) in [
        ("mcts.c_puct", config.mcts.c_puct),
        ("mcts.temperature", config.mcts.temperature),
        ("mcts.late_temperature", config.mcts.late_temperature),
        ("mcts.dirichlet_alpha", config.mcts.dirichlet_alpha),
        ("mcts.dirichlet_weight", config.mcts.dirichlet_weight),
    ] {
        validate_nonnegative_f32(name, value)?;
    }
    if config.mcts.dirichlet_weight > 1.0 {
        return Err(ConfigError::InvalidConfig(
            "mcts.dirichlet_weight must be a rate in [0, 1]".to_string(),
        ));
    }
    if (config.mcts.dirichlet_alpha == 0.0) != (config.mcts.dirichlet_weight == 0.0) {
        return Err(ConfigError::InvalidConfig(
            "mcts.dirichlet_alpha and mcts.dirichlet_weight must both be zero to disable noise"
                .to_string(),
        ));
    }
    if config.mcts.temp_threshold == 0 {
        if config.mcts.late_temperature != config.mcts.temperature {
            return Err(ConfigError::InvalidConfig(
                "mcts.late_temperature must equal mcts.temperature when mcts.temp_threshold is zero"
                    .to_string(),
            ));
        }
    } else if config.mcts.late_temperature == config.mcts.temperature {
        return Err(ConfigError::InvalidConfig(
            "mcts.late_temperature must differ from mcts.temperature when the schedule is enabled"
                .to_string(),
        ));
    }
    if config.mcts.start_sims == 0 || config.mcts.max_sims == 0 {
        return Err(ConfigError::InvalidConfig(
            "mcts.start_sims and mcts.max_sims must be positive".to_string(),
        ));
    }
    if config.mcts.start_sims > config.mcts.max_sims {
        return Err(ConfigError::InvalidConfig(
            "mcts.start_sims cannot exceed mcts.max_sims".to_string(),
        ));
    }
    if config.mcts.start_sims == config.mcts.max_sims {
        if config.mcts.sim_ramp_rate != 0 {
            return Err(ConfigError::InvalidConfig(
                "mcts.sim_ramp_rate must be zero when start_sims equals max_sims".to_string(),
            ));
        }
    } else {
        let delta = config.mcts.max_sims - config.mcts.start_sims;
        if config.mcts.sim_ramp_rate == 0 || config.mcts.sim_ramp_rate > delta {
            return Err(ConfigError::InvalidConfig(
                "ramped MCTS requires sim_ramp_rate in [1, max_sims - start_sims]".to_string(),
            ));
        }
        let steps_to_cap = u64::from(delta.div_ceil(config.mcts.sim_ramp_rate));
        if config.training.iterations - 1 < steps_to_cap {
            return Err(ConfigError::InvalidConfig(
                "MCTS simulation schedule must reach max_sims within training.iterations"
                    .to_string(),
            ));
        }
    }
    if config.mcts.eval_batch_size == 0 || config.mcts.onnx_intra_threads == 0 {
        return Err(ConfigError::InvalidConfig(
            "mcts.eval_batch_size and mcts.onnx_intra_threads must be positive".to_string(),
        ));
    }
    if !config.wandb.init_timeout_seconds.is_finite() || config.wandb.init_timeout_seconds <= 0.0 {
        return Err(ConfigError::InvalidConfig(
            "wandb.init_timeout_seconds must be finite and greater than zero".to_string(),
        ));
    }

    Ok(config)
}
