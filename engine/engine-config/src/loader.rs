//! Configuration loading logic.

mod env;
mod validation;

use crate::CentralConfig;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

pub use env::apply_env_overrides;

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

/// Standard locations to search for config.toml.
pub const CONFIG_SEARCH_PATHS: &[&str] = &["config.toml", "../config.toml", "/app/config.toml"];

/// Load the central configuration and then apply environment overrides.
pub fn load_config() -> Result<CentralConfig, ConfigError> {
    if let Ok(path) = std::env::var("CARTRIDGE_CONFIG") {
        let path = PathBuf::from(&path);
        if path.is_file() {
            info!("Loading config from CARTRIDGE_CONFIG: {}", path.display());
            return load_from_path(&path);
        }
        return Err(ConfigError::ExplicitPathMissing(path));
    }

    for path_str in CONFIG_SEARCH_PATHS {
        let path = PathBuf::from(path_str);
        if path.exists() {
            info!("Loading config from {}", path.display());
            return load_from_path(&path);
        }
    }

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
