use std::any::type_name;
use std::str::FromStr;

use crate::CentralConfig;

use super::validation::validate_config;
use super::ConfigError;

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

macro_rules! env_override {
    ($config:expr, $section:ident . $field:ident, $key:expr) => {
        if let Some(value) = env_string($key)? {
            $config.$section.$field = value;
        }
    };
    ($config:expr, $section:ident . $field:ident, $key:expr, parse) => {
        if let Some(value) = env_value($key)? {
            $config.$section.$field = value;
        }
    };
    ($config:expr, $section:ident . $field:ident, $key:expr, optional) => {
        if let Some(value) = env_string($key)? {
            $config.$section.$field = Some(value);
        }
    };
    ($config:expr, $section:ident . $field:ident, $key:expr, optional_parse) => {
        if let Some(value) = env_value($key)? {
            $config.$section.$field = Some(value);
        }
    };
    ($config:expr, $section:ident . $field:ident, $key:expr, list) => {
        if let Some(value) = env_string_list($key)? {
            $config.$section.$field = value;
        }
    };
}

/// Apply every supported environment override, then validate the result.
pub fn apply_env_overrides(mut config: CentralConfig) -> Result<CentralConfig, ConfigError> {
    reject_unknown_config_env_vars()?;
    apply_identity_overrides(&mut config)?;
    apply_training_overrides(&mut config)?;
    apply_evaluation_overrides(&mut config)?;
    apply_runtime_overrides(&mut config)?;
    apply_mcts_overrides(&mut config)?;
    apply_service_overrides(&mut config)?;
    validate_config(&config)?;
    Ok(config)
}

fn apply_identity_overrides(config: &mut CentralConfig) -> Result<(), ConfigError> {
    env_override!(config, common.env_id, "CARTRIDGE_COMMON_ENV_ID");
    env_override!(config, common.data_dir, "CARTRIDGE_COMMON_DATA_DIR");
    env_override!(config, common.log_level, "CARTRIDGE_COMMON_LOG_LEVEL");
    env_override!(config, algorithm.id, "CARTRIDGE_ALGORITHM_ID");
    Ok(())
}

fn apply_training_overrides(config: &mut CentralConfig) -> Result<(), ConfigError> {
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
    Ok(())
}

fn apply_evaluation_overrides(config: &mut CentralConfig) -> Result<(), ConfigError> {
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
    Ok(())
}

fn apply_runtime_overrides(config: &mut CentralConfig) -> Result<(), ConfigError> {
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
    env_override!(config, web.host, "CARTRIDGE_WEB_HOST");
    env_override!(config, web.port, "CARTRIDGE_WEB_PORT", parse);
    env_override!(
        config,
        web.allowed_origins,
        "CARTRIDGE_WEB_ALLOWED_ORIGINS",
        list
    );
    Ok(())
}

fn apply_mcts_overrides(config: &mut CentralConfig) -> Result<(), ConfigError> {
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
    Ok(())
}

fn apply_service_overrides(config: &mut CentralConfig) -> Result<(), ConfigError> {
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
    Ok(())
}
