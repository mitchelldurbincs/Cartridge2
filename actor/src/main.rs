//! Actor - Self-play episode runner for Cartridge2
//!
//! A long-running process that:
//! 1. Watches `./data/models/latest.onnx` for updates
//! 2. Runs MCTS self-play loops using the Engine library
//! 3. Saves completed games to the PostgreSQL replay buffer
//! 4. Exposes a health check HTTP server for Kubernetes probes

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info};

mod actor;
mod config;
mod game_config;
mod health;
mod mcts_policy;
mod metrics;
mod stats;
mod storage;

use crate::actor::Actor;
use crate::config::Config;
use crate::health::{start_health_server, HealthState};

/// Get trace context from environment variables for distributed tracing.
///
/// Reads CARTRIDGE_TRACE_ID and CARTRIDGE_TRACE_PARENT from environment.
/// These are set by the orchestrator when launching actor processes.
fn get_trace_context() -> (Option<String>, Option<String>) {
    let trace_id = std::env::var("CARTRIDGE_TRACE_ID").ok();
    let parent_span = std::env::var("CARTRIDGE_TRACE_PARENT").ok();
    (trace_id, parent_span)
}

/// Generate a span ID for this actor process.
fn generate_span_id() -> String {
    uuid::Uuid::new_v4().to_string()[..16].to_string()
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse configuration
    let config = Config::parse();

    // Validate configuration
    config.validate()?;

    // Initialize tracing with JSON support for cloud deployments
    // (logging settings come from the central config loaded by config::Config)
    engine_config::init_tracing(&config.log_level, &[], &config::central_config().logging);

    // Get trace context from environment (set by orchestrator)
    let (trace_id, parent_span) = get_trace_context();
    let span_id = generate_span_id();

    // Log startup with trace context
    info!(
        log_level = %config.log_level,
        component = "actor",
        env_id = %config.env_id,
        actor_id = %config.actor_id,
        trace_id = trace_id.as_deref().unwrap_or("none"),
        span_id = %span_id,
        parent_span = parent_span.as_deref().unwrap_or("none"),
        "Actor service starting"
    );

    // Initialize Prometheus metrics
    metrics::init_metrics();
    metrics::set_actor_info(&config.env_id, &config.actor_id);
    info!(component = "actor", "Prometheus metrics initialized");

    // Log the max_episodes setting
    info!(
        component = "actor",
        max_episodes = config.max_episodes,
        env_id = %config.env_id,
        actor_id = %config.actor_id,
        num_simulations = config.num_simulations,
        temp_threshold = config.temp_threshold,
        "Actor configuration loaded"
    );

    // Create shared health state for Kubernetes probes
    let health_state = HealthState::new();

    // Start health server in background
    let health_port = config.health_port;
    let health_handle = {
        let state = health_state.clone();
        tokio::spawn(async move {
            if let Err(e) = start_health_server(health_port, state).await {
                error!(component = "actor", port = health_port, error = %e, "Health server error");
            }
        })
    };

    // Create actor instance
    let actor = Actor::new(config).await?;
    let actor = Arc::new(actor);

    // Mark as ready once initialization is complete
    health_state.set_ready();

    // Setup graceful shutdown
    let shutdown_actor = Arc::clone(&actor);
    let shutdown_handle = tokio::spawn(async move {
        if let Err(e) = signal::ctrl_c().await {
            error!(component = "actor", error = %e, "Failed to listen for ctrl+c signal");
            return;
        }
        info!(
            component = "actor",
            event = "shutdown_signal",
            "Shutdown signal received, stopping actor"
        );
        shutdown_actor.shutdown();
    });

    // Run the actor
    let run_result = actor.run(&health_state).await;

    // Wait for shutdown to complete
    shutdown_handle.abort();
    health_handle.abort();

    match run_result {
        Ok(_) => {
            info!(
                component = "actor",
                event = "shutdown_complete",
                "Actor completed successfully"
            );
            Ok(())
        }
        Err(e) => {
            error!(component = "actor", event = "actor_failed", error = %e, "Actor failed");
            Err(e)
        }
    }
}
