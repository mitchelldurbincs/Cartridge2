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
//! 3. Built-in defaults
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
pub use loader::{apply_env_overrides, load_config, load_from_path, CONFIG_SEARCH_PATHS};
pub use logging::init_tracing;
pub use structs::*;

#[cfg(test)]
mod tests;
