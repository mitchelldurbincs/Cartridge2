//! Configuration struct definitions.
//!
//! Missing sections and fields use the section `Default` implementations,
//! which read canonical values from `config.defaults.toml` via `defaults`.
//! Keep this mapping in one place; field-level Serde defaults are unnecessary.

use crate::defaults;
use serde::Deserialize;

/// Root configuration structure matching config.toml
#[derive(Debug, Deserialize, Default, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(deny_unknown_fields)]
pub struct CentralConfig {
    #[serde(default)]
    pub common: CommonConfig,
    #[serde(default)]
    pub algorithm: AlgorithmConfig,
    #[serde(default)]
    pub training: TrainingConfig,
    #[serde(default)]
    pub evaluation: EvaluationConfig,
    #[serde(default)]
    pub actor: ActorConfig,
    #[serde(default)]
    pub web: WebConfig,
    #[serde(default)]
    pub mcts: MctsConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub wandb: WandbConfig,
}

/// Algorithm cartridge selected for collection, learning, and evaluation.
#[derive(Debug, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(default, deny_unknown_fields)]
pub struct AlgorithmConfig {
    pub id: String,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            id: defaults::algorithm_id().into(),
        }
    }
}

/// Common configuration shared by all components
#[derive(Debug, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(default, deny_unknown_fields)]
pub struct CommonConfig {
    pub data_dir: String,
    pub env_id: String,
    pub log_level: String,
}

impl Default for CommonConfig {
    fn default() -> Self {
        Self {
            data_dir: defaults::data_dir().into(),
            env_id: defaults::env_id().into(),
            log_level: defaults::log_level().into(),
        }
    }
}

/// Training configuration for the trainer
#[derive(Debug, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(default, deny_unknown_fields)]
pub struct TrainingConfig {
    pub iterations: u64,
    pub episodes_per_iteration: u32,
    pub steps_per_iteration: u64,
    pub batch_size: u64,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub grad_clip_norm: f64,
    pub device: String,
    pub checkpoint_interval: u64,
    pub num_actors: u32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            iterations: defaults::iterations(),
            episodes_per_iteration: defaults::episodes_per_iteration(),
            steps_per_iteration: defaults::steps_per_iteration(),
            batch_size: defaults::batch_size(),
            learning_rate: defaults::learning_rate(),
            weight_decay: defaults::weight_decay(),
            grad_clip_norm: defaults::grad_clip_norm(),
            device: defaults::device().into(),
            checkpoint_interval: defaults::checkpoint_interval(),
            num_actors: defaults::num_actors(),
        }
    }
}

/// Evaluation configuration
#[derive(Debug, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(default, deny_unknown_fields)]
pub struct EvaluationConfig {
    pub interval: u64,
    pub games: u32,
    pub win_threshold: f64,
    pub eval_vs_random: bool,
    pub simulations: u32,
    pub temperature: f32,
    pub solver_games: u32,
    pub evaluation_seed: u64,
    pub promotion_metric: String,
    pub promotion_margin: f64,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            interval: defaults::eval_interval(),
            games: defaults::eval_games(),
            win_threshold: defaults::win_threshold(),
            eval_vs_random: defaults::eval_vs_random(),
            simulations: defaults::eval_simulations(),
            temperature: defaults::eval_temperature(),
            solver_games: defaults::solver_games(),
            evaluation_seed: defaults::evaluation_seed(),
            promotion_metric: defaults::promotion_metric().into(),
            promotion_margin: defaults::promotion_margin(),
        }
    }
}

/// Actor (self-play) configuration
#[derive(Debug, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(default, deny_unknown_fields)]
pub struct ActorConfig {
    pub actor_id: String,
    pub episode_timeout_secs: u64,
    pub log_interval: u32,
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            actor_id: defaults::actor_id().into(),
            episode_timeout_secs: defaults::episode_timeout_secs(),
            log_interval: defaults::log_interval(),
        }
    }
}

/// Web server configuration
#[derive(Debug, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(default, deny_unknown_fields)]
pub struct WebConfig {
    pub host: String,
    pub port: u16,
    /// CORS allowed origins. Empty = localhost-only development allowlist.
    /// Set to specific domains in production (e.g., ["https://your-domain.com"]).
    pub allowed_origins: Vec<String>,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            host: defaults::host().into(),
            port: defaults::port(),
            allowed_origins: defaults::allowed_origins().to_vec(),
        }
    }
}

/// MCTS (Monte Carlo Tree Search) configuration
#[derive(Debug, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(default, deny_unknown_fields)]
pub struct MctsConfig {
    pub c_puct: f32,
    pub temperature: f32,
    pub late_temperature: f32,
    pub temp_threshold: u32,
    pub dirichlet_alpha: f32,
    pub dirichlet_weight: f32,
    pub eval_batch_size: u32,
    pub onnx_intra_threads: u32,
    pub start_sims: u32,
    pub max_sims: u32,
    pub sim_ramp_rate: u32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            c_puct: defaults::c_puct(),
            temperature: defaults::temperature(),
            late_temperature: defaults::late_temperature(),
            temp_threshold: defaults::temp_threshold(),
            dirichlet_alpha: defaults::dirichlet_alpha(),
            dirichlet_weight: defaults::dirichlet_weight(),
            eval_batch_size: defaults::eval_batch_size(),
            onnx_intra_threads: defaults::onnx_intra_threads(),
            start_sims: defaults::start_sims(),
            max_sims: defaults::max_sims(),
            sim_ramp_rate: defaults::sim_ramp_rate(),
        }
    }
}

/// Logging configuration for structured logging output
#[derive(Debug, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    /// Log output format: "text" for human-readable, "json" for structured JSON
    pub format: String,
    /// Include timestamps in log output (set false if cloud logging adds them)
    pub include_timestamps: bool,
    /// Include module target in log output
    pub include_target: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            format: defaults::logging_format().into(),
            include_timestamps: defaults::logging_include_timestamps(),
            include_target: defaults::logging_include_target(),
        }
    }
}

impl LoggingConfig {
    /// Check if JSON format is enabled
    pub fn is_json(&self) -> bool {
        self.format.eq_ignore_ascii_case("json")
    }
}

/// Storage backend configuration
#[derive(Debug, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(default, deny_unknown_fields)]
pub struct StorageConfig {
    pub model_backend: String,
    pub postgres_url: Option<String>,
    pub s3_bucket: Option<String>,
    pub s3_endpoint: Option<String>,
    /// Maximum number of connections in the PostgreSQL pool
    pub pool_max_size: usize,
    /// Timeout in seconds to wait for a connection from the pool
    pub pool_connect_timeout: u64,
    /// Idle timeout for connections in seconds (None = no timeout)
    pub pool_idle_timeout: Option<u64>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            model_backend: defaults::model_backend().into(),
            postgres_url: Some(defaults::postgres_url().into()),
            s3_bucket: None,
            s3_endpoint: None,
            pool_max_size: defaults::pool_max_size(),
            pool_connect_timeout: defaults::pool_connect_timeout(),
            pool_idle_timeout: Some(defaults::pool_idle_timeout()),
        }
    }
}

/// Weights & Biases settings consumed by Python orchestration.
///
/// Rust keeps this section typed so every process accepts and validates the
/// same canonical configuration document instead of maintaining partial
/// per-language schemas.
#[derive(Debug, Deserialize, Clone)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(default, deny_unknown_fields)]
pub struct WandbConfig {
    pub enabled: bool,
    pub required: bool,
    pub project: String,
    pub entity: String,
    pub group: String,
    pub tags: Vec<String>,
    pub init_timeout_seconds: f64,
}

impl Default for WandbConfig {
    fn default() -> Self {
        Self {
            enabled: defaults::wandb_enabled(),
            required: defaults::wandb_required(),
            project: defaults::wandb_project().into(),
            entity: defaults::wandb_entity().into(),
            group: defaults::wandb_group().into(),
            tags: defaults::wandb_tags().to_vec(),
            init_timeout_seconds: defaults::wandb_init_timeout_seconds(),
        }
    }
}
