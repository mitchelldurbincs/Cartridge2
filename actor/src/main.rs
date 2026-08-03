//! Actor - Self-play episode runner for Cartridge2
//!
//! A bounded one-shot process that:
//! 1. Pins the profile's exact source model from the content-addressed RunHead
//! 2. Runs MCTS self-play loops using the Engine library
//! 3. Saves opaque cartridge records to one exact PostgreSQL replay selection

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info};

mod actor;
mod algorithms;
mod config;
mod mcts_policy;
mod resources;
mod stats;
mod storage;

use crate::algorithms::build_collector;
use crate::config::Config;

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
        algorithm = %config.algorithm_id,
        actor_id = %config.actor_id,
        collection_scope_id = %config.collection_scope_id,
        source_checkpoint_id = config.source_checkpoint_id.as_deref().unwrap_or("root"),
        trace_id = trace_id.as_deref().unwrap_or("none"),
        span_id = %span_id,
        parent_span = parent_span.as_deref().unwrap_or("none"),
        "Actor worker starting"
    );

    // Log the max_episodes setting
    info!(
        component = "actor",
        max_episodes = config.max_episodes,
        env_id = %config.env_id,
        algorithm = %config.algorithm_id,
        actor_id = %config.actor_id,
        collection_scope_id = %config.collection_scope_id,
        source_checkpoint_id = config.source_checkpoint_id.as_deref().unwrap_or("root"),
        num_simulations = config.num_simulations,
        c_puct = config.c_puct,
        temperature = config.temperature,
        late_temperature = config.late_temperature,
        temp_threshold = config.temp_threshold,
        dirichlet_alpha = config.dirichlet_alpha,
        dirichlet_weight = config.dirichlet_weight,
        eval_batch_size = config.eval_batch_size,
        onnx_intra_threads = config.onnx_intra_threads,
        "Actor configuration loaded"
    );

    // Create actor instance
    let actor = build_collector(config).await?;
    let actor = Arc::new(actor);

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
    let run_result = actor.run().await;

    // Wait for shutdown to complete
    shutdown_handle.abort();

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
