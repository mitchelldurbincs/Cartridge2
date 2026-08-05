//! Default configuration values loaded from config.defaults.toml.
//!
//! This module loads defaults from the shared TOML file at compile time,
//! ensuring Rust and Python use identical default values.

use once_cell::sync::Lazy;
use serde::Deserialize;

/// The embedded defaults TOML file (loaded at compile time)
const DEFAULTS_TOML: &str = include_str!("../../../config.defaults.toml");

/// Parsed defaults structure (parsed once at first use)
static DEFAULTS: Lazy<DefaultsConfig> =
    Lazy::new(|| toml::from_str(DEFAULTS_TOML).expect("config.defaults.toml should be valid TOML"));

// ============================================================================
// Internal structs for parsing config.defaults.toml
// ============================================================================

#[derive(Debug, Deserialize)]
struct DefaultsConfig {
    common: CommonDefaults,
    algorithm: AlgorithmDefaults,
    training: TrainingDefaults,
    evaluation: EvaluationDefaults,
    actor: ActorDefaults,
    web: WebDefaults,
    mcts: MctsDefaults,
    logging: LoggingDefaults,
    storage: StorageDefaults,
    wandb: WandbDefaults,
}

#[derive(Debug, Deserialize)]
struct CommonDefaults {
    data_dir: String,
    env_id: String,
    log_level: String,
}

#[derive(Debug, Deserialize)]
struct AlgorithmDefaults {
    id: String,
}

#[derive(Debug, Deserialize)]
struct TrainingDefaults {
    iterations: u64,
    episodes_per_iteration: u32,
    steps_per_iteration: u64,
    batch_size: u64,
    learning_rate: f64,
    weight_decay: f64,
    grad_clip_norm: f64,
    device: String,
    checkpoint_interval: u64,
    num_actors: u32,
}

#[derive(Debug, Deserialize)]
struct EvaluationDefaults {
    interval: u64,
    games: u32,
    win_threshold: f64,
    eval_vs_random: bool,
    simulations: u32,
    temperature: f32,
    solver_games: u32,
    evaluation_seed: u64,
    promotion_metric: String,
    promotion_margin: f64,
}

#[derive(Debug, Deserialize)]
struct ActorDefaults {
    actor_id: String,
    episode_timeout_secs: u64,
    log_interval: u32,
}

#[derive(Debug, Deserialize)]
struct WebDefaults {
    host: String,
    port: u16,
    #[serde(default)]
    allowed_origins: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct MctsDefaults {
    c_puct: f32,
    temperature: f32,
    late_temperature: f32,
    temp_threshold: u32,
    dirichlet_alpha: f32,
    dirichlet_weight: f32,
    eval_batch_size: u32,
    onnx_intra_threads: u32,
    start_sims: u32,
    max_sims: u32,
    sim_ramp_rate: u32,
}

#[derive(Debug, Deserialize)]
struct LoggingDefaults {
    format: String,
    include_timestamps: bool,
    include_target: bool,
}

#[derive(Debug, Deserialize)]
struct StorageDefaults {
    model_backend: String,
    postgres_url: String,
    pool_max_size: usize,
    pool_connect_timeout: u64,
    pool_idle_timeout: u64,
    replay_retained_scopes: u32,
    learner_state_retained_checkpoints: u32,
}

#[derive(Debug, Deserialize)]
struct WandbDefaults {
    enabled: bool,
    required: bool,
    project: String,
    entity: String,
    group: String,
    tags: Vec<String>,
    init_timeout_seconds: f64,
}

// ============================================================================
// Public accessor functions
// ============================================================================

// Common
pub fn data_dir() -> &'static str {
    &DEFAULTS.common.data_dir
}
pub fn env_id() -> &'static str {
    &DEFAULTS.common.env_id
}
pub fn log_level() -> &'static str {
    &DEFAULTS.common.log_level
}

// Algorithm
pub fn algorithm_id() -> &'static str {
    &DEFAULTS.algorithm.id
}

// Training
pub fn iterations() -> u64 {
    DEFAULTS.training.iterations
}
pub fn episodes_per_iteration() -> u32 {
    DEFAULTS.training.episodes_per_iteration
}
pub fn steps_per_iteration() -> u64 {
    DEFAULTS.training.steps_per_iteration
}
pub fn batch_size() -> u64 {
    DEFAULTS.training.batch_size
}
pub fn learning_rate() -> f64 {
    DEFAULTS.training.learning_rate
}
pub fn weight_decay() -> f64 {
    DEFAULTS.training.weight_decay
}
pub fn grad_clip_norm() -> f64 {
    DEFAULTS.training.grad_clip_norm
}
pub fn device() -> &'static str {
    &DEFAULTS.training.device
}
pub fn checkpoint_interval() -> u64 {
    DEFAULTS.training.checkpoint_interval
}
pub fn num_actors() -> u32 {
    DEFAULTS.training.num_actors
}

// Evaluation
pub fn eval_interval() -> u64 {
    DEFAULTS.evaluation.interval
}
pub fn eval_games() -> u32 {
    DEFAULTS.evaluation.games
}
pub fn win_threshold() -> f64 {
    DEFAULTS.evaluation.win_threshold
}
pub fn eval_vs_random() -> bool {
    DEFAULTS.evaluation.eval_vs_random
}
pub fn eval_simulations() -> u32 {
    DEFAULTS.evaluation.simulations
}
pub fn eval_temperature() -> f32 {
    DEFAULTS.evaluation.temperature
}
pub fn solver_games() -> u32 {
    DEFAULTS.evaluation.solver_games
}
pub fn evaluation_seed() -> u64 {
    DEFAULTS.evaluation.evaluation_seed
}
pub fn promotion_metric() -> &'static str {
    &DEFAULTS.evaluation.promotion_metric
}
pub fn promotion_margin() -> f64 {
    DEFAULTS.evaluation.promotion_margin
}

// Actor
pub fn actor_id() -> &'static str {
    &DEFAULTS.actor.actor_id
}
pub fn episode_timeout_secs() -> u64 {
    DEFAULTS.actor.episode_timeout_secs
}
pub fn log_interval() -> u32 {
    DEFAULTS.actor.log_interval
}

// Web
pub fn host() -> &'static str {
    &DEFAULTS.web.host
}
pub fn port() -> u16 {
    DEFAULTS.web.port
}
pub fn allowed_origins() -> &'static [String] {
    &DEFAULTS.web.allowed_origins
}

// MCTS
pub fn c_puct() -> f32 {
    DEFAULTS.mcts.c_puct
}
pub fn temperature() -> f32 {
    DEFAULTS.mcts.temperature
}
pub fn late_temperature() -> f32 {
    DEFAULTS.mcts.late_temperature
}
pub fn temp_threshold() -> u32 {
    DEFAULTS.mcts.temp_threshold
}
pub fn dirichlet_alpha() -> f32 {
    DEFAULTS.mcts.dirichlet_alpha
}
pub fn dirichlet_weight() -> f32 {
    DEFAULTS.mcts.dirichlet_weight
}
pub fn eval_batch_size() -> u32 {
    DEFAULTS.mcts.eval_batch_size
}
pub fn onnx_intra_threads() -> u32 {
    DEFAULTS.mcts.onnx_intra_threads
}
pub fn start_sims() -> u32 {
    DEFAULTS.mcts.start_sims
}
pub fn max_sims() -> u32 {
    DEFAULTS.mcts.max_sims
}
pub fn sim_ramp_rate() -> u32 {
    DEFAULTS.mcts.sim_ramp_rate
}

// Logging
pub fn logging_format() -> &'static str {
    &DEFAULTS.logging.format
}
pub fn logging_include_timestamps() -> bool {
    DEFAULTS.logging.include_timestamps
}
pub fn logging_include_target() -> bool {
    DEFAULTS.logging.include_target
}

// Storage
pub fn model_backend() -> &'static str {
    &DEFAULTS.storage.model_backend
}
pub fn postgres_url() -> &'static str {
    &DEFAULTS.storage.postgres_url
}
pub fn pool_max_size() -> usize {
    DEFAULTS.storage.pool_max_size
}
pub fn pool_connect_timeout() -> u64 {
    DEFAULTS.storage.pool_connect_timeout
}
pub fn pool_idle_timeout() -> u64 {
    DEFAULTS.storage.pool_idle_timeout
}
pub fn replay_retained_scopes() -> u32 {
    DEFAULTS.storage.replay_retained_scopes
}
pub fn learner_state_retained_checkpoints() -> u32 {
    DEFAULTS.storage.learner_state_retained_checkpoints
}

// W&B (parsed by Rust so the canonical config has one strict schema even
// though only the Python orchestration component consumes these values).
pub fn wandb_enabled() -> bool {
    DEFAULTS.wandb.enabled
}
pub fn wandb_required() -> bool {
    DEFAULTS.wandb.required
}
pub fn wandb_project() -> &'static str {
    &DEFAULTS.wandb.project
}
pub fn wandb_entity() -> &'static str {
    &DEFAULTS.wandb.entity
}
pub fn wandb_group() -> &'static str {
    &DEFAULTS.wandb.group
}
pub fn wandb_tags() -> &'static [String] {
    &DEFAULTS.wandb.tags
}
pub fn wandb_init_timeout_seconds() -> f64 {
    DEFAULTS.wandb.init_timeout_seconds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults_parse() {
        // Just accessing these will verify the TOML parses correctly
        assert_eq!(data_dir(), "./data");
        assert_eq!(env_id(), "tictactoe");
        assert_eq!(log_level(), "info");
        assert_eq!(algorithm_id(), "alphazero_board_v1");
    }

    #[test]
    fn test_training_defaults() {
        assert_eq!(iterations(), 100);
        assert_eq!(batch_size(), 64);
        assert!((learning_rate() - 0.001).abs() < f64::EPSILON);
        assert_eq!(num_actors(), 1);
    }

    #[test]
    fn test_mcts_defaults() {
        assert!((c_puct() - 1.4).abs() < f32::EPSILON);
        assert!((late_temperature() - 1.0).abs() < f32::EPSILON);
        assert_eq!(temp_threshold(), 0);
        assert_eq!(start_sims(), 50);
        assert_eq!(max_sims(), 400);
        assert_eq!(sim_ramp_rate(), 20);
    }

    #[test]
    fn test_evaluation_defaults() {
        assert_eq!(eval_interval(), 1);
        assert_eq!(eval_games(), 50);
        assert!((win_threshold() - 0.55).abs() < f64::EPSILON);
        assert!(eval_vs_random());
        assert!((eval_temperature() - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_storage_defaults() {
        assert_eq!(model_backend(), "filesystem");
        assert_eq!(pool_max_size(), 16);
    }

    #[test]
    fn test_logging_defaults() {
        assert_eq!(logging_format(), "text");
        assert!(logging_include_timestamps());
        assert!(logging_include_target());
    }
}
