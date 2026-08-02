//! Configuration for the Actor service
//!
//! Configuration is loaded from config.toml with environment variable overrides.
//! CLI arguments take highest priority, followed by env vars, then config.toml.

use anyhow::{anyhow, Result};
use clap::Parser;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::str::FromStr;
use std::time::Duration;
use tracing::level_filters::LevelFilter;

use crate::storage::PoolConfig;
use engine_config::{load_config, CentralConfig};

// Load central config once at startup
static CENTRAL_CONFIG: OnceCell<CentralConfig> = OnceCell::new();

/// Load and validate central configuration before clap evaluates its defaults.
pub fn initialize_central_config() -> Result<&'static CentralConfig> {
    CENTRAL_CONFIG.get_or_try_init(|| load_config().map_err(anyhow::Error::from))
}

/// Central config.toml settings, loaded once per process.
pub fn central_config() -> &'static CentralConfig {
    initialize_central_config().expect("central configuration must be valid")
}

fn env_string(key: &'static str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) if value.is_empty() => None,
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("invalid value for environment variable {key}: expected valid Unicode")
        }
    }
}

fn env_or<T>(key: &'static str, fallback: T) -> T
where
    T: FromStr,
    T::Err: Display,
{
    let Some(value) = env_string(key) else {
        return fallback;
    };
    parse_env_value(key, &value).unwrap_or_else(|message| panic!("{message}"))
}

fn parse_env_value<T>(key: &'static str, value: &str) -> std::result::Result<T, String>
where
    T: FromStr,
    T::Err: Display,
{
    value
        .parse()
        .map_err(|error| format!("invalid value {value:?} for environment variable {key}: {error}"))
}

// Default value functions - env var -> central config fallback
fn default_actor_id() -> String {
    env_string("ACTOR_ACTOR_ID").unwrap_or_else(|| central_config().actor.actor_id.clone())
}

fn default_env_id() -> String {
    env_string("ACTOR_ENV_ID").unwrap_or_else(|| central_config().common.env_id.clone())
}

fn default_max_episodes() -> i32 {
    env_or("ACTOR_MAX_EPISODES", central_config().actor.max_episodes)
}

fn default_episode_timeout() -> u64 {
    env_or(
        "ACTOR_EPISODE_TIMEOUT",
        central_config().actor.episode_timeout_secs,
    )
}

fn default_flush_interval() -> u64 {
    env_or(
        "ACTOR_FLUSH_INTERVAL",
        central_config().actor.flush_interval_secs,
    )
}

fn default_log_level() -> String {
    env_string("ACTOR_LOG_LEVEL").unwrap_or_else(|| central_config().common.log_level.clone())
}

fn default_log_interval() -> u32 {
    env_or("ACTOR_LOG_INTERVAL", central_config().actor.log_interval)
}

fn default_postgres_url() -> String {
    env_string("CARTRIDGE_STORAGE_POSTGRES_URL")
        .or_else(|| env_string("ACTOR_POSTGRES_URL"))
        .unwrap_or_else(|| {
            central_config()
                .storage
                .postgres_url
                .clone()
                .unwrap_or_else(|| {
                    "postgresql://cartridge:cartridge@localhost:5432/cartridge".to_string()
                })
        })
}

fn default_data_dir() -> String {
    env_string("ACTOR_DATA_DIR").unwrap_or_else(|| central_config().common.data_dir.clone())
}

fn default_num_simulations() -> u32 {
    env_or(
        "ACTOR_NUM_SIMULATIONS",
        central_config().mcts.num_simulations,
    )
}

fn default_temp_threshold() -> u32 {
    env_or("ACTOR_TEMP_THRESHOLD", central_config().mcts.temp_threshold)
}

fn default_eval_batch_size() -> usize {
    env_or(
        "ACTOR_EVAL_BATCH_SIZE",
        central_config().mcts.eval_batch_size,
    )
}

fn default_onnx_intra_threads() -> usize {
    env_or(
        "ACTOR_ONNX_INTRA_THREADS",
        central_config().mcts.onnx_intra_threads,
    )
}

fn default_health_port() -> u16 {
    if let Some(value) = env_string("ACTOR_HEALTH_PORT") {
        return parse_env_value("ACTOR_HEALTH_PORT", &value)
            .unwrap_or_else(|message| panic!("{message}"));
    }
    env_or("HEALTH_PORT", central_config().actor.health_port)
}

#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
#[command(name = "actor")]
#[command(about = "Cartridge2 Actor - Self-play episode runner")]
pub struct Config {
    #[arg(long, default_value_t = default_actor_id())]
    pub actor_id: String,

    #[arg(long, default_value_t = default_env_id())]
    pub env_id: String,

    #[arg(long, default_value_t = default_max_episodes())]
    pub max_episodes: i32,

    #[arg(long, default_value_t = default_episode_timeout())]
    pub episode_timeout_secs: u64,

    #[arg(long, default_value_t = default_flush_interval())]
    pub flush_interval_secs: u64,

    #[arg(long, default_value_t = default_log_level())]
    pub log_level: String,

    #[arg(long, default_value_t = default_log_interval())]
    pub log_interval: u32,

    #[arg(long, default_value_t = default_data_dir())]
    pub data_dir: String,

    #[arg(long, default_value_t = default_num_simulations())]
    pub num_simulations: u32,

    #[arg(long, default_value_t = default_temp_threshold())]
    pub temp_threshold: u32,

    #[arg(long, default_value_t = default_eval_batch_size())]
    pub eval_batch_size: usize,

    #[arg(long, default_value_t = default_onnx_intra_threads())]
    pub onnx_intra_threads: usize,

    #[arg(long, default_value_t = default_postgres_url())]
    pub postgres_url: String,

    /// Disable model file watching (load model once at startup).
    /// Use this for orchestrator mode where models are pre-loaded before actors start.
    #[arg(long, default_value_t = false)]
    pub no_watch: bool,

    /// Port for health check HTTP server (Kubernetes liveness/readiness probes).
    /// If the port is already in use (e.g. several actors on one host), the
    /// server scans forward from this port and binds the next free one.
    #[arg(long, default_value_t = default_health_port())]
    pub health_port: u16,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            actor_id: default_actor_id(),
            env_id: default_env_id(),
            max_episodes: default_max_episodes(),
            episode_timeout_secs: default_episode_timeout(),
            flush_interval_secs: default_flush_interval(),
            log_level: default_log_level(),
            log_interval: default_log_interval(),
            data_dir: default_data_dir(),
            num_simulations: default_num_simulations(),
            temp_threshold: default_temp_threshold(),
            eval_batch_size: default_eval_batch_size(),
            onnx_intra_threads: default_onnx_intra_threads(),
            postgres_url: default_postgres_url(),
            no_watch: false,
            health_port: default_health_port(),
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.actor_id.trim().is_empty() {
            return Err(anyhow!("actor_id cannot be empty"));
        }
        if self.env_id.trim().is_empty() {
            return Err(anyhow!("env_id cannot be empty"));
        }
        if self.max_episodes != -1 && self.max_episodes <= 0 {
            return Err(anyhow!(
                "max_episodes must be -1 (unlimited) or greater than 0"
            ));
        }
        if self.episode_timeout_secs == 0 {
            return Err(anyhow!("episode_timeout_secs must be greater than 0"));
        }
        if self.flush_interval_secs == 0 {
            return Err(anyhow!("flush_interval_secs must be greater than 0"));
        }
        if self.log_level.parse::<LevelFilter>().is_err() {
            return Err(anyhow!(
                "invalid log level '{}', expected one of trace, debug, info, warn, error",
                self.log_level
            ));
        }
        if self.data_dir.trim().is_empty() {
            return Err(anyhow!("data_dir cannot be empty"));
        }
        if self.num_simulations == 0 {
            return Err(anyhow!("num_simulations must be greater than 0"));
        }
        if self.eval_batch_size == 0 {
            return Err(anyhow!("eval_batch_size must be greater than 0"));
        }
        if self.postgres_url.trim().is_empty() {
            return Err(anyhow!("postgres_url cannot be empty"));
        }
        if self.health_port == 0 {
            return Err(anyhow!("health_port must be greater than 0"));
        }

        Ok(())
    }

    #[cfg(test)]
    pub fn episode_timeout(&self) -> Duration {
        Duration::from_secs(self.episode_timeout_secs)
    }

    pub fn flush_interval(&self) -> Duration {
        Duration::from_secs(self.flush_interval_secs)
    }

    #[cfg(test)]
    pub fn model_path(&self) -> String {
        format!("{}/models/latest.onnx", self.data_dir)
    }

    /// Get the connection pool configuration from central config.
    pub fn pool_config(&self) -> PoolConfig {
        PoolConfig {
            max_size: central_config().storage.pool_max_size,
            connect_timeout_secs: central_config().storage.pool_connect_timeout,
            idle_timeout_secs: central_config().storage.pool_idle_timeout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> Config {
        Config {
            actor_id: "actor".into(),
            env_id: "tictactoe".into(),
            max_episodes: 1,
            episode_timeout_secs: 30,
            flush_interval_secs: 5,
            log_level: "info".into(),
            log_interval: 10,
            data_dir: "../data".into(),
            num_simulations: 100,
            temp_threshold: 0,
            eval_batch_size: 32,
            onnx_intra_threads: 1,
            postgres_url: "postgresql://test:test@localhost:5432/test".into(),
            no_watch: false,
            health_port: 8081,
        }
    }

    #[test]
    fn validate_accepts_valid_configuration() {
        assert!(base_config().validate().is_ok());
    }

    #[test]
    fn validate_rejects_empty_actor_id() {
        let mut cfg = base_config();
        cfg.actor_id.clear();
        assert!(cfg.validate().unwrap_err().to_string().contains("actor_id"));
    }

    #[test]
    fn validate_rejects_empty_env_id() {
        let mut cfg = base_config();
        cfg.env_id.clear();
        assert!(cfg.validate().unwrap_err().to_string().contains("env_id"));
    }

    #[test]
    fn validate_rejects_invalid_log_level() {
        let mut cfg = base_config();
        cfg.log_level = "nope".into();
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("invalid log level"));
    }

    #[test]
    fn validate_rejects_zero_episode_timeout() {
        let mut cfg = base_config();
        cfg.episode_timeout_secs = 0;
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("episode_timeout_secs"));
    }

    #[test]
    fn validate_rejects_zero_flush_interval() {
        let mut cfg = base_config();
        cfg.flush_interval_secs = 0;
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("flush_interval_secs"));
    }

    #[test]
    fn validate_rejects_empty_postgres_url() {
        let mut cfg = base_config();
        cfg.postgres_url.clear();
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("postgres_url"));
    }

    #[test]
    fn validate_accepts_negative_max_episodes() {
        let mut cfg = base_config();
        cfg.max_episodes = -1;
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_other_non_positive_max_episodes() {
        for value in [-2, 0] {
            let mut cfg = base_config();
            cfg.max_episodes = value;
            assert!(cfg
                .validate()
                .unwrap_err()
                .to_string()
                .contains("max_episodes"));
        }
    }

    #[test]
    fn validate_rejects_zero_mcts_work_sizes() {
        let mut cfg = base_config();
        cfg.num_simulations = 0;
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("num_simulations"));

        let mut cfg = base_config();
        cfg.eval_batch_size = 0;
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("eval_batch_size"));
    }

    #[test]
    fn invalid_legacy_numeric_value_is_rejected_instead_of_falling_back() {
        let error = parse_env_value::<u32>("ACTOR_NUM_SIMULATIONS", "many").unwrap_err();
        assert!(error.contains("ACTOR_NUM_SIMULATIONS"));
        assert!(error.contains("many"));
    }

    #[test]
    fn episode_timeout_returns_correct_duration() {
        assert_eq!(base_config().episode_timeout(), Duration::from_secs(30));
    }

    #[test]
    fn flush_interval_returns_correct_duration() {
        assert_eq!(base_config().flush_interval(), Duration::from_secs(5));
    }

    #[test]
    fn model_path_constructs_correctly() {
        assert_eq!(base_config().model_path(), "../data/models/latest.onnx");
    }
}
