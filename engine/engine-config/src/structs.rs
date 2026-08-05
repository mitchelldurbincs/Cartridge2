//! Configuration struct definitions.
//!
//! All config structs with serde deserialization support and default values.

use crate::defaults;
use serde::Deserialize;

// ============================================================================
// Serde default functions (required for #[serde(default = "...")])
// These call the accessor functions from defaults module
// ============================================================================

fn d_data_dir() -> String {
    defaults::data_dir().into()
}
fn d_env_id() -> String {
    defaults::env_id().into()
}
fn d_log_level() -> String {
    defaults::log_level().into()
}
fn d_algorithm_id() -> String {
    defaults::algorithm_id().into()
}
fn d_iterations() -> u64 {
    defaults::iterations()
}
fn d_episodes() -> u32 {
    defaults::episodes_per_iteration()
}
fn d_steps() -> u64 {
    defaults::steps_per_iteration()
}
fn d_batch_size() -> u64 {
    defaults::batch_size()
}
fn d_lr() -> f64 {
    defaults::learning_rate()
}
fn d_weight_decay() -> f64 {
    defaults::weight_decay()
}
fn d_grad_clip() -> f64 {
    defaults::grad_clip_norm()
}
fn d_device() -> String {
    defaults::device().into()
}
fn d_ckpt_interval() -> u64 {
    defaults::checkpoint_interval()
}
fn d_num_actors() -> u32 {
    defaults::num_actors()
}
fn d_eval_interval() -> u64 {
    defaults::eval_interval()
}
fn d_eval_games() -> u32 {
    defaults::eval_games()
}
fn d_win_threshold() -> f64 {
    defaults::win_threshold()
}
fn d_eval_vs_random() -> bool {
    defaults::eval_vs_random()
}
fn d_eval_simulations() -> u32 {
    defaults::eval_simulations()
}
fn d_eval_temperature() -> f32 {
    defaults::eval_temperature()
}
fn d_solver_games() -> u32 {
    defaults::solver_games()
}
fn d_evaluation_seed() -> u64 {
    defaults::evaluation_seed()
}
fn d_promotion_metric() -> String {
    defaults::promotion_metric().into()
}
fn d_promotion_margin() -> f64 {
    defaults::promotion_margin()
}
fn d_actor_id() -> String {
    defaults::actor_id().into()
}
fn d_episode_timeout() -> u64 {
    defaults::episode_timeout_secs()
}
fn d_log_interval() -> u32 {
    defaults::log_interval()
}
fn d_host() -> String {
    defaults::host().into()
}
fn d_port() -> u16 {
    defaults::port()
}
fn d_allowed_origins() -> Vec<String> {
    defaults::allowed_origins().to_vec()
}
fn d_c_puct() -> f32 {
    defaults::c_puct()
}
fn d_temperature() -> f32 {
    defaults::temperature()
}
fn d_late_temperature() -> f32 {
    defaults::late_temperature()
}
fn d_temp_threshold() -> u32 {
    defaults::temp_threshold()
}
fn d_dirichlet_alpha() -> f32 {
    defaults::dirichlet_alpha()
}
fn d_dirichlet_weight() -> f32 {
    defaults::dirichlet_weight()
}
fn d_eval_batch_size() -> u32 {
    defaults::eval_batch_size()
}
fn d_onnx_intra_threads() -> u32 {
    defaults::onnx_intra_threads()
}
fn d_start_sims() -> u32 {
    defaults::start_sims()
}
fn d_max_sims() -> u32 {
    defaults::max_sims()
}
fn d_sim_ramp_rate() -> u32 {
    defaults::sim_ramp_rate()
}
fn d_logging_format() -> String {
    defaults::logging_format().into()
}
fn d_logging_include_timestamps() -> bool {
    defaults::logging_include_timestamps()
}
fn d_logging_include_target() -> bool {
    defaults::logging_include_target()
}
fn d_model_backend() -> String {
    defaults::model_backend().into()
}
fn d_postgres_url() -> Option<String> {
    Some(defaults::postgres_url().into())
}
fn d_pool_max_size() -> usize {
    defaults::pool_max_size()
}
fn d_pool_connect_timeout() -> u64 {
    defaults::pool_connect_timeout()
}
fn d_pool_idle_timeout() -> Option<u64> {
    Some(defaults::pool_idle_timeout())
}
fn d_replay_retained_scopes() -> u32 {
    defaults::replay_retained_scopes()
}
fn d_wandb_enabled() -> bool {
    defaults::wandb_enabled()
}
fn d_wandb_required() -> bool {
    defaults::wandb_required()
}
fn d_wandb_project() -> String {
    defaults::wandb_project().into()
}
fn d_wandb_entity() -> String {
    defaults::wandb_entity().into()
}
fn d_wandb_group() -> String {
    defaults::wandb_group().into()
}
fn d_wandb_tags() -> Vec<String> {
    defaults::wandb_tags().to_vec()
}
fn d_wandb_init_timeout_seconds() -> f64 {
    defaults::wandb_init_timeout_seconds()
}

// ============================================================================
// Configuration Structs
// ============================================================================

/// Root configuration structure matching config.toml
#[derive(Debug, Deserialize, Default, Clone)]
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
#[serde(default, deny_unknown_fields)]
pub struct AlgorithmConfig {
    #[serde(default = "d_algorithm_id")]
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
#[serde(default, deny_unknown_fields)]
pub struct CommonConfig {
    #[serde(default = "d_data_dir")]
    pub data_dir: String,
    #[serde(default = "d_env_id")]
    pub env_id: String,
    #[serde(default = "d_log_level")]
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
#[serde(default, deny_unknown_fields)]
pub struct TrainingConfig {
    #[serde(default = "d_iterations")]
    pub iterations: u64,
    #[serde(default = "d_episodes")]
    pub episodes_per_iteration: u32,
    #[serde(default = "d_steps")]
    pub steps_per_iteration: u64,
    #[serde(default = "d_batch_size")]
    pub batch_size: u64,
    #[serde(default = "d_lr")]
    pub learning_rate: f64,
    #[serde(default = "d_weight_decay")]
    pub weight_decay: f64,
    #[serde(default = "d_grad_clip")]
    pub grad_clip_norm: f64,
    #[serde(default = "d_device")]
    pub device: String,
    #[serde(default = "d_ckpt_interval")]
    pub checkpoint_interval: u64,
    #[serde(default = "d_num_actors")]
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
#[serde(default, deny_unknown_fields)]
pub struct EvaluationConfig {
    #[serde(default = "d_eval_interval")]
    pub interval: u64,
    #[serde(default = "d_eval_games")]
    pub games: u32,
    #[serde(default = "d_win_threshold")]
    pub win_threshold: f64,
    #[serde(default = "d_eval_vs_random")]
    pub eval_vs_random: bool,
    #[serde(default = "d_eval_simulations")]
    pub simulations: u32,
    #[serde(default = "d_eval_temperature")]
    pub temperature: f32,
    #[serde(default = "d_solver_games")]
    pub solver_games: u32,
    #[serde(default = "d_evaluation_seed")]
    pub evaluation_seed: u64,
    #[serde(default = "d_promotion_metric")]
    pub promotion_metric: String,
    #[serde(default = "d_promotion_margin")]
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
#[serde(default, deny_unknown_fields)]
pub struct ActorConfig {
    #[serde(default = "d_actor_id")]
    pub actor_id: String,
    #[serde(default = "d_episode_timeout")]
    pub episode_timeout_secs: u64,
    #[serde(default = "d_log_interval")]
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
#[serde(default, deny_unknown_fields)]
pub struct WebConfig {
    #[serde(default = "d_host")]
    pub host: String,
    #[serde(default = "d_port")]
    pub port: u16,
    /// CORS allowed origins. Empty = localhost-only development allowlist.
    /// Set to specific domains in production (e.g., ["https://your-domain.com"]).
    #[serde(default = "d_allowed_origins")]
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
#[serde(default, deny_unknown_fields)]
pub struct MctsConfig {
    #[serde(default = "d_c_puct")]
    pub c_puct: f32,
    #[serde(default = "d_temperature")]
    pub temperature: f32,
    #[serde(default = "d_late_temperature")]
    pub late_temperature: f32,
    #[serde(default = "d_temp_threshold")]
    pub temp_threshold: u32,
    #[serde(default = "d_dirichlet_alpha")]
    pub dirichlet_alpha: f32,
    #[serde(default = "d_dirichlet_weight")]
    pub dirichlet_weight: f32,
    #[serde(default = "d_eval_batch_size")]
    pub eval_batch_size: u32,
    #[serde(default = "d_onnx_intra_threads")]
    pub onnx_intra_threads: u32,
    #[serde(default = "d_start_sims")]
    pub start_sims: u32,
    #[serde(default = "d_max_sims")]
    pub max_sims: u32,
    #[serde(default = "d_sim_ramp_rate")]
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
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    /// Log output format: "text" for human-readable, "json" for structured JSON
    #[serde(default = "d_logging_format")]
    pub format: String,
    /// Include timestamps in log output (set false if cloud logging adds them)
    #[serde(default = "d_logging_include_timestamps")]
    pub include_timestamps: bool,
    /// Include module target in log output
    #[serde(default = "d_logging_include_target")]
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
#[serde(default, deny_unknown_fields)]
pub struct StorageConfig {
    #[serde(default = "d_model_backend")]
    pub model_backend: String,
    #[serde(default = "d_postgres_url")]
    pub postgres_url: Option<String>,
    #[serde(default)]
    pub s3_bucket: Option<String>,
    #[serde(default)]
    pub s3_endpoint: Option<String>,
    /// Maximum number of connections in the PostgreSQL pool
    #[serde(default = "d_pool_max_size")]
    pub pool_max_size: usize,
    /// Timeout in seconds to wait for a connection from the pool
    #[serde(default = "d_pool_connect_timeout")]
    pub pool_connect_timeout: u64,
    /// Idle timeout for connections in seconds (None = no timeout)
    #[serde(default = "d_pool_idle_timeout")]
    pub pool_idle_timeout: Option<u64>,
    /// Newest collection scopes kept per profile by the Python scope reaper
    #[serde(default = "d_replay_retained_scopes")]
    pub replay_retained_scopes: u32,
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
            replay_retained_scopes: defaults::replay_retained_scopes(),
        }
    }
}

/// Weights & Biases settings consumed by Python orchestration.
///
/// Rust keeps this section typed so every process accepts and validates the
/// same canonical configuration document instead of maintaining partial
/// per-language schemas.
#[derive(Debug, Deserialize, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct WandbConfig {
    #[serde(default = "d_wandb_enabled")]
    pub enabled: bool,
    #[serde(default = "d_wandb_required")]
    pub required: bool,
    #[serde(default = "d_wandb_project")]
    pub project: String,
    #[serde(default = "d_wandb_entity")]
    pub entity: String,
    #[serde(default = "d_wandb_group")]
    pub group: String,
    #[serde(default = "d_wandb_tags")]
    pub tags: Vec<String>,
    #[serde(default = "d_wandb_init_timeout_seconds")]
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
