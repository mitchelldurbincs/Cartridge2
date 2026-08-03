//! Configuration for the bounded one-shot collector process.
//!
//! Configuration is loaded from config.toml with environment variable overrides.
//! CLI arguments take highest priority, followed by env vars, then config.toml.

use anyhow::{anyhow, Result};
use clap::Parser;
use once_cell::sync::Lazy;
#[cfg(test)]
use std::time::Duration;
use tracing::level_filters::LevelFilter;

use crate::storage::PoolConfig;
use algorithm_core::resolve_algorithm;
use engine_config::{load_config, CentralConfig};

// Load central config once at startup
static CENTRAL_CONFIG: Lazy<CentralConfig> = Lazy::new(|| {
    load_config().unwrap_or_else(|error| panic!("failed to load Cartridge config: {error}"))
});

/// Central config.toml settings, loaded once per process.
pub fn central_config() -> &'static CentralConfig {
    &CENTRAL_CONFIG
}

// CLI defaults that are intentionally inherited from central configuration.
fn default_actor_id() -> String {
    CENTRAL_CONFIG.actor.actor_id.clone()
}

fn default_log_level() -> String {
    CENTRAL_CONFIG.common.log_level.clone()
}

fn default_log_interval() -> u32 {
    CENTRAL_CONFIG.actor.log_interval
}

fn default_postgres_url() -> String {
    CENTRAL_CONFIG
        .storage
        .postgres_url
        .clone()
        .expect("storage.postgres_url is required by the actor")
}

fn default_data_dir() -> String {
    CENTRAL_CONFIG.common.data_dir.clone()
}

#[derive(Parser, Debug, Clone)]
#[command(name = "actor")]
#[command(about = "Cartridge2 Actor - Self-play episode runner")]
pub struct Config {
    #[arg(long, default_value_t = default_actor_id())]
    pub actor_id: String,

    #[arg(long)]
    pub env_id: String,

    /// Algorithm cartridge used for self-play collection.
    #[arg(long = "algorithm")]
    pub algorithm_id: String,

    /// Exact positive episode quota for this one-shot collector process.
    #[arg(long)]
    pub max_episodes: u32,

    /// Required lowercase SHA-256-shaped identity for this collection attempt.
    #[arg(long)]
    pub collection_scope_id: String,

    /// Exact checkpoint generation used for collection; omitted only at run root.
    #[arg(long)]
    pub source_checkpoint_id: Option<String>,

    #[arg(long)]
    pub episode_timeout_secs: u64,

    #[arg(long, default_value_t = default_log_level())]
    pub log_level: String,

    #[arg(long, default_value_t = default_log_interval())]
    pub log_interval: u32,

    #[arg(long, default_value_t = default_data_dir())]
    pub data_dir: String,

    /// Exact positive search budget supplied by the synchronized orchestrator.
    #[arg(long)]
    pub num_simulations: u32,

    #[arg(long)]
    pub c_puct: f32,

    #[arg(long)]
    pub temperature: f32,

    #[arg(long)]
    pub late_temperature: f32,

    #[arg(long)]
    pub temp_threshold: u32,

    #[arg(long)]
    pub dirichlet_alpha: f32,

    #[arg(long)]
    pub dirichlet_weight: f32,

    #[arg(long)]
    pub eval_batch_size: u32,

    #[arg(long)]
    pub onnx_intra_threads: u32,

    #[arg(long, default_value_t = default_postgres_url())]
    pub postgres_url: String,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.actor_id.is_empty() {
            return Err(anyhow!("actor_id cannot be empty"));
        }
        if self.env_id.is_empty() {
            return Err(anyhow!("env_id cannot be empty"));
        }
        resolve_algorithm(&self.algorithm_id).map_err(|error| anyhow!(error))?;
        if self.max_episodes == 0 {
            return Err(anyhow!("max_episodes must be greater than 0"));
        }
        crate::storage::validate_replay_digest(&self.collection_scope_id, "collection_scope_id")?;
        if let Some(source_checkpoint_id) = &self.source_checkpoint_id {
            crate::storage::validate_replay_digest(source_checkpoint_id, "source_checkpoint_id")?;
        }
        if self.episode_timeout_secs == 0 {
            return Err(anyhow!("episode_timeout_secs must be greater than 0"));
        }
        if self.num_simulations == 0 {
            return Err(anyhow!("num_simulations must be greater than 0"));
        }
        if self.eval_batch_size == 0 {
            return Err(anyhow!("eval_batch_size must be greater than 0"));
        }
        if self.onnx_intra_threads == 0 {
            return Err(anyhow!("onnx_intra_threads must be greater than 0"));
        }
        for (name, value) in [
            ("c_puct", self.c_puct),
            ("temperature", self.temperature),
            ("late_temperature", self.late_temperature),
            ("dirichlet_alpha", self.dirichlet_alpha),
            ("dirichlet_weight", self.dirichlet_weight),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(anyhow!("{name} must be finite and nonnegative"));
            }
        }
        if self.dirichlet_weight > 1.0 {
            return Err(anyhow!("dirichlet_weight must be in [0, 1]"));
        }
        if (self.dirichlet_alpha == 0.0) != (self.dirichlet_weight == 0.0) {
            return Err(anyhow!(
                "dirichlet_alpha and dirichlet_weight must both be zero to disable noise"
            ));
        }
        if self.temp_threshold == 0 {
            if self.late_temperature != self.temperature {
                return Err(anyhow!(
                    "late_temperature must equal temperature when temp_threshold is zero"
                ));
            }
        } else if self.late_temperature == self.temperature {
            return Err(anyhow!(
                "late_temperature must differ from temperature when the schedule is enabled"
            ));
        }
        if self.log_level.parse::<LevelFilter>().is_err() {
            return Err(anyhow!(
                "invalid log level '{}', expected one of trace, debug, info, warn, error",
                self.log_level
            ));
        }
        if self.postgres_url.is_empty() {
            return Err(anyhow!("postgres_url cannot be empty"));
        }

        Ok(())
    }

    #[cfg(test)]
    pub fn episode_timeout(&self) -> Duration {
        Duration::from_secs(self.episode_timeout_secs)
    }

    /// Get the connection pool configuration from central config.
    pub fn pool_config(&self) -> PoolConfig {
        PoolConfig {
            max_size: CENTRAL_CONFIG.storage.pool_max_size,
            connect_timeout_secs: CENTRAL_CONFIG.storage.pool_connect_timeout,
            idle_timeout_secs: CENTRAL_CONFIG.storage.pool_idle_timeout,
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
            algorithm_id: algorithm_core::ALPHAZERO_BOARD_V1_ID.into(),
            max_episodes: 1,
            collection_scope_id: "a".repeat(64),
            source_checkpoint_id: Some("b".repeat(64)),
            episode_timeout_secs: 30,
            log_level: "info".into(),
            log_interval: 10,
            data_dir: "../data".into(),
            num_simulations: 100,
            c_puct: 1.4,
            temperature: 1.0,
            late_temperature: 1.0,
            temp_threshold: 0,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
            eval_batch_size: 32,
            onnx_intra_threads: 1,
            postgres_url: "postgresql://test:test@localhost:5432/test".into(),
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
    fn validate_rejects_unknown_algorithm_with_available_profile() {
        let mut cfg = base_config();
        cfg.algorithm_id = "ppo".into();
        let error = cfg.validate().unwrap_err().to_string();
        assert!(error.contains("ppo"));
        assert!(error.contains(algorithm_core::ALPHAZERO_BOARD_V1_ID));
    }

    #[test]
    fn validate_rejects_zero_episode_quota() {
        let mut cfg = base_config();
        cfg.max_episodes = 0;
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("max_episodes"));
    }

    #[test]
    fn validate_rejects_invalid_replay_selection_digests() {
        let mut cfg = base_config();
        cfg.collection_scope_id = "A".repeat(64);
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("collection_scope_id"));

        let mut cfg = base_config();
        cfg.source_checkpoint_id = Some("b".repeat(63));
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("source_checkpoint_id"));

        let mut root = base_config();
        root.source_checkpoint_id = None;
        assert!(root.validate().is_ok());
    }

    #[test]
    fn parser_requires_every_recipe_owned_argument() {
        let required = [
            ("--env-id", "tictactoe"),
            ("--algorithm", algorithm_core::ALPHAZERO_BOARD_V1_ID),
            ("--max-episodes", "1"),
            (
                "--collection-scope-id",
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            ),
            ("--episode-timeout-secs", "30"),
            ("--num-simulations", "10"),
            ("--c-puct", "1.4"),
            ("--temperature", "1.0"),
            ("--late-temperature", "1.0"),
            ("--temp-threshold", "0"),
            ("--dirichlet-alpha", "0.3"),
            ("--dirichlet-weight", "0.25"),
            ("--eval-batch-size", "32"),
            ("--onnx-intra-threads", "1"),
        ];
        let complete = || {
            std::iter::once("actor".to_string())
                .chain(
                    required
                        .iter()
                        .flat_map(|(flag, value)| [(*flag).to_string(), (*value).to_string()]),
                )
                .collect::<Vec<_>>()
        };
        assert!(Config::try_parse_from(complete()).is_ok());

        for (missing, _) in required {
            let mut args = complete();
            let index = args.iter().position(|value| value == missing).unwrap();
            args.drain(index..=index + 1);
            let error = Config::try_parse_from(args).unwrap_err().to_string();
            assert!(error.contains(missing), "{missing}: {error}");
        }
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
    fn validate_rejects_zero_mcts_budgets() {
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

        let mut cfg = base_config();
        cfg.onnx_intra_threads = 0;
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("onnx_intra_threads"));
    }

    #[test]
    fn validate_rejects_invalid_search_floats_and_inactive_noise_values() {
        for (field, mutate) in [
            (
                "c_puct",
                (|cfg: &mut Config| cfg.c_puct = f32::NAN) as fn(&mut Config),
            ),
            ("temperature", |cfg: &mut Config| {
                cfg.temperature = f32::INFINITY
            }),
            ("late_temperature", |cfg: &mut Config| {
                cfg.late_temperature = -0.1
            }),
            ("dirichlet_weight", |cfg: &mut Config| {
                cfg.dirichlet_weight = 1.1
            }),
            ("both be zero", |cfg: &mut Config| cfg.dirichlet_alpha = 0.0),
            ("must differ", |cfg: &mut Config| cfg.temp_threshold = 1),
        ] {
            let mut cfg = base_config();
            mutate(&mut cfg);
            assert!(cfg.validate().unwrap_err().to_string().contains(field));
        }
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
    fn episode_timeout_returns_correct_duration() {
        assert_eq!(base_config().episode_timeout(), Duration::from_secs(30));
    }
}
