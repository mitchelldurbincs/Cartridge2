//! Shared tracing initialization for the Rust binaries (actor, web).
//!
//! Both binaries configure `tracing_subscriber` identically: an env-filter
//! seeded from `RUST_LOG` (falling back to a default level plus quiet
//! directives for noisy dependencies), and a JSON or human-readable format
//! chosen from [`LoggingConfig`] with a `CARTRIDGE_LOGGING_FORMAT` override.

use crate::structs::LoggingConfig;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialize tracing with optional JSON format for cloud deployments.
///
/// The filter defaults to `default_level` when `RUST_LOG` is unset, always
/// quiets the `ort`/`h2`/`hyper` dependencies, and applies any
/// `extra_directives` (e.g. `"web=info"`) on top.
///
/// Supports the `CARTRIDGE_LOGGING_FORMAT` environment variable override:
/// - `"text"` (default): Human-readable format for local development
/// - `"json"`: Structured JSON format for Google Cloud Logging
///
/// # Panics
///
/// Panics if an entry in `extra_directives` is not a valid tracing directive,
/// or if a global subscriber is already set. Both indicate programmer error
/// at binary startup.
pub fn init_tracing(default_level: &str, extra_directives: &[&str], config: &LoggingConfig) {
    let mut filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default_level))
        .add_directive("ort=warn".parse().unwrap())
        .add_directive("h2=warn".parse().unwrap())
        .add_directive("hyper=warn".parse().unwrap());
    for directive in extra_directives {
        filter = filter.add_directive(directive.parse().unwrap());
    }

    // Check for environment variable override
    let json_format = std::env::var("CARTRIDGE_LOGGING_FORMAT")
        .map(|v| v.eq_ignore_ascii_case("json"))
        .unwrap_or_else(|_| config.is_json());

    let registry = tracing_subscriber::registry().with(filter);

    if json_format {
        // JSON format for Google Cloud Logging
        registry
            .with(
                fmt::layer()
                    .json()
                    .with_current_span(true)
                    .with_span_list(false)
                    .with_file(false)
                    .with_line_number(false)
                    .flatten_event(true)
                    .with_target(config.include_target),
            )
            .init();
    } else {
        // Human-readable format for local development
        registry.with(fmt::layer()).init();
    }
}
