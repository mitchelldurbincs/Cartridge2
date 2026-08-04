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

    /// Algorithm-owned, versioned JSON configuration for the selected collector.
    #[arg(long = "collector-config")]
    pub collector_config: String,

    #[arg(long)]
    pub episode_timeout_secs: u64,

    #[arg(long, default_value_t = default_log_level())]
    pub log_level: String,

    #[arg(long, default_value_t = default_log_interval())]
    pub log_interval: u32,

    #[arg(long, default_value_t = default_data_dir())]
    pub data_dir: String,

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
        let recipe: serde_json::Value = serde_json::from_str(&self.collector_config)
            .map_err(|error| anyhow!("collector_config must be valid JSON: {error}"))?;
        if !recipe.is_object() {
            return Err(anyhow!("collector_config must be a JSON object"));
        }
        if self.episode_timeout_secs == 0 {
            return Err(anyhow!("episode_timeout_secs must be greater than 0"));
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
            collector_config: r#"{"schema_version":1}"#.into(),
            episode_timeout_secs: 30,
            log_level: "info".into(),
            log_interval: 10,
            data_dir: "../data".into(),
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
            ("--collector-config", r#"{"schema_version":1}"#),
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
    fn validate_rejects_malformed_collector_config() {
        let mut cfg = base_config();
        cfg.collector_config = "[]".into();
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("JSON object"));

        cfg.collector_config = "{".into();
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("valid JSON"));
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
