use crate::CentralConfig;

use super::ConfigError;

pub(super) fn validate_config(config: &CentralConfig) -> Result<(), ConfigError> {
    validate_identity(config)?;
    validate_storage_and_logging(config)?;
    validate_promotion_metric_name(config)?;
    validate_training(config)?;
    validate_evaluation(config)?;
    validate_mcts(config)?;
    validate_wandb(config)
}

fn validate_identity(config: &CentralConfig) -> Result<(), ConfigError> {
    for (name, value) in [
        ("algorithm.id", config.algorithm.id.as_str()),
        ("common.env_id", config.common.env_id.as_str()),
        ("common.data_dir", config.common.data_dir.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(ConfigError::InvalidConfig(format!(
                "{name} cannot be empty"
            )));
        }
    }
    if config.common.log_level.parse::<tracing::Level>().is_err() {
        return Err(ConfigError::InvalidConfig(format!(
            "common.log_level is invalid: {:?}",
            config.common.log_level
        )));
    }
    Ok(())
}

fn validate_storage_and_logging(config: &CentralConfig) -> Result<(), ConfigError> {
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
    if config.storage.replay_retained_scopes == 0 {
        return Err(ConfigError::InvalidConfig(
            "storage.replay_retained_scopes must be greater than zero".to_string(),
        ));
    }
    if !matches!(config.logging.format.as_str(), "text" | "json") {
        return Err(ConfigError::InvalidConfig(format!(
            "logging.format must be 'text' or 'json', got {:?}",
            config.logging.format
        )));
    }
    Ok(())
}

fn validate_training(config: &CentralConfig) -> Result<(), ConfigError> {
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
    Ok(())
}

fn validate_evaluation(config: &CentralConfig) -> Result<(), ConfigError> {
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
    Ok(())
}

fn validate_promotion_metric_name(config: &CentralConfig) -> Result<(), ConfigError> {
    if !matches!(
        config.evaluation.promotion_metric.as_str(),
        "win_rate" | "solver_optimal"
    ) {
        return Err(ConfigError::InvalidConfig(format!(
            "evaluation.promotion_metric must be 'win_rate' or 'solver_optimal', got {:?}",
            config.evaluation.promotion_metric
        )));
    }
    Ok(())
}

fn validate_mcts(config: &CentralConfig) -> Result<(), ConfigError> {
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
    validate_temperature_schedule(config)?;
    validate_simulation_schedule(config)?;
    if config.mcts.eval_batch_size == 0 || config.mcts.onnx_intra_threads == 0 {
        return Err(ConfigError::InvalidConfig(
            "mcts.eval_batch_size and mcts.onnx_intra_threads must be positive".to_string(),
        ));
    }
    Ok(())
}

fn validate_temperature_schedule(config: &CentralConfig) -> Result<(), ConfigError> {
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
    Ok(())
}

fn validate_simulation_schedule(config: &CentralConfig) -> Result<(), ConfigError> {
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
        return Ok(());
    }
    let delta = config.mcts.max_sims - config.mcts.start_sims;
    if config.mcts.sim_ramp_rate == 0 || config.mcts.sim_ramp_rate > delta {
        return Err(ConfigError::InvalidConfig(
            "ramped MCTS requires sim_ramp_rate in [1, max_sims - start_sims]".to_string(),
        ));
    }
    let steps_to_cap = u64::from(delta.div_ceil(config.mcts.sim_ramp_rate));
    if config.training.iterations - 1 < steps_to_cap {
        return Err(ConfigError::InvalidConfig(
            "MCTS simulation schedule must reach max_sims within training.iterations".to_string(),
        ));
    }
    Ok(())
}

fn validate_wandb(config: &CentralConfig) -> Result<(), ConfigError> {
    if !config.wandb.init_timeout_seconds.is_finite() || config.wandb.init_timeout_seconds <= 0.0 {
        return Err(ConfigError::InvalidConfig(
            "wandb.init_timeout_seconds must be finite and greater than zero".to_string(),
        ));
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
