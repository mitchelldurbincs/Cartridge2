//! Centralized configuration loading from config.toml.
//!
//! This crate provides configuration structs and loading logic shared
//! across all Rust components (actor, web).
//!
//! # Configuration Priority
//!
//! Settings are loaded with the following priority (highest to lowest):
//! 1. Environment variables (`CARTRIDGE_<SECTION>_<KEY>`)
//! 2. config.toml file
//! 3. Built-in defaults (a compile-time embed of `config.defaults.toml`)
//!
//! # Missing vs. broken config
//!
//! No `config.toml` at all is a normal state — the Kubernetes manifests mount
//! none and the images ship only `config.defaults.toml` — and yields the
//! built-in defaults. A `config.toml` that *exists but cannot be parsed* is an
//! operator mistake and is reported by [`try_load_config`], which binaries
//! should use so startup aborts instead of silently running on defaults.
//!
//! # Environment Variable Override Pattern
//!
//! ```text
//! CARTRIDGE_<SECTION>_<KEY>=value
//!
//! Examples:
//!     CARTRIDGE_COMMON_ENV_ID=connect4
//!     CARTRIDGE_COMMON_DATA_DIR=/data
//!     CARTRIDGE_WEB_HOST=127.0.0.1
//!     CARTRIDGE_WEB_PORT=3000
//!     CARTRIDGE_TRAINING_ITERATIONS=50
//! ```

mod defaults;
mod loader;
mod logging;
mod structs;

pub use defaults::*;
pub use loader::{
    apply_env_overrides, load_config, load_from_path, try_load_config, ConfigError,
    CONFIG_SEARCH_PATHS,
};
pub use logging::init_tracing;
pub use structs::*;

#[cfg(test)]
mod tests;
