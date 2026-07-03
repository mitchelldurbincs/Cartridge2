//! Prometheus metrics for the actor component.
//!
//! This module provides comprehensive metrics for monitoring actor performance,
//! episode generation throughput, MCTS search efficiency, and storage operations.
//!
//! The metric definitions below are actor-specific (its sibling is
//! `web/src/metrics.rs`); the shared registration/encoding plumbing lives in
//! the `metrics-common` crate.

use lazy_static::lazy_static;
use prometheus::{Histogram, HistogramOpts, IntCounter, IntGauge, IntGaugeVec, Opts, Registry};
use std::sync::Once;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // ========== Episode Counters ==========

    /// Total number of self-play episodes completed
    pub static ref EPISODES_TOTAL: IntCounter = IntCounter::with_opts(
        Opts::new("actor_episodes_total", "Total number of self-play episodes completed")
    ).unwrap();

    /// Total episodes where player 1 won
    pub static ref PLAYER1_WINS: IntCounter = IntCounter::with_opts(
        Opts::new("actor_player1_wins_total", "Total episodes where player 1 won")
    ).unwrap();

    /// Total episodes where player 2 won
    pub static ref PLAYER2_WINS: IntCounter = IntCounter::with_opts(
        Opts::new("actor_player2_wins_total", "Total episodes where player 2 won")
    ).unwrap();

    /// Total episodes ending in a draw
    pub static ref DRAWS: IntCounter = IntCounter::with_opts(
        Opts::new("actor_draws_total", "Total episodes ending in a draw")
    ).unwrap();

    // ========== Throughput Gauges ==========

    /// Current episode generation throughput
    pub static ref EPISODES_PER_SECOND: prometheus::Gauge = prometheus::Gauge::with_opts(
        Opts::new("actor_episodes_per_second", "Current episode generation throughput")
    ).unwrap();

    // ========== Episode Histograms ==========

    /// Time to complete one episode
    pub static ref EPISODE_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new("actor_episode_duration_seconds", "Time to complete one episode")
            .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    ).unwrap();

    /// Number of game steps per episode
    pub static ref EPISODE_STEPS: Histogram = Histogram::with_opts(
        HistogramOpts::new("actor_episode_steps", "Number of game steps per episode")
            .buckets(vec![5.0, 10.0, 20.0, 30.0, 42.0, 50.0, 60.0, 80.0, 100.0])
    ).unwrap();

    // ========== MCTS Metrics ==========

    /// Total MCTS searches performed
    pub static ref MCTS_SEARCHES_TOTAL: IntCounter = IntCounter::with_opts(
        Opts::new("actor_mcts_searches_total", "Total MCTS searches performed")
    ).unwrap();

    /// Neural network inference time per MCTS search (seconds)
    pub static ref MCTS_INFERENCE_SECONDS: Histogram = Histogram::with_opts(
        HistogramOpts::new("actor_mcts_inference_seconds", "Neural network inference time per MCTS search")
            .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    ).unwrap();

    /// Total MCTS time per search (seconds)
    pub static ref MCTS_SEARCH_SECONDS: Histogram = Histogram::with_opts(
        HistogramOpts::new("actor_mcts_search_seconds", "Total MCTS time per search")
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    ).unwrap();

    /// MCTS simulations per search
    pub static ref MCTS_SIMULATIONS_PER_SEARCH: Histogram = Histogram::with_opts(
        HistogramOpts::new("actor_mcts_simulations_per_search", "Number of MCTS simulations per search")
            .buckets(vec![50.0, 100.0, 200.0, 400.0, 800.0, 1600.0])
    ).unwrap();

    // ========== Storage Metrics ==========

    /// Total transitions written to replay buffer
    pub static ref TRANSITIONS_STORED: IntCounter = IntCounter::with_opts(
        Opts::new("actor_transitions_stored_total", "Total transitions written to replay buffer")
    ).unwrap();

    /// Database write latency (seconds)
    pub static ref DB_WRITE_SECONDS: Histogram = Histogram::with_opts(
        HistogramOpts::new("actor_db_write_seconds", "Database write latency")
            .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    ).unwrap();

    /// Database connection pool - current size
    pub static ref DB_POOL_SIZE: IntGauge = IntGauge::with_opts(
        Opts::new("actor_db_pool_size", "Current number of connections in the pool")
    ).unwrap();

    /// Database connection pool - available connections
    pub static ref DB_POOL_AVAILABLE: IntGauge = IntGauge::with_opts(
        Opts::new("actor_db_pool_available", "Number of available connections in the pool")
    ).unwrap();

    /// Database connection pool - waiting tasks
    pub static ref DB_POOL_WAITING: IntGauge = IntGauge::with_opts(
        Opts::new("actor_db_pool_waiting", "Number of tasks waiting for a connection")
    ).unwrap();

    // ========== Model Metrics ==========

    /// Number of model hot-reload events
    pub static ref MODEL_RELOADS: IntCounter = IntCounter::with_opts(
        Opts::new("actor_model_reloads_total", "Number of model hot-reload events")
    ).unwrap();

    /// Time to load ONNX model (seconds)
    pub static ref MODEL_LOAD_SECONDS: Histogram = Histogram::with_opts(
        HistogramOpts::new("actor_model_load_seconds", "Time to load ONNX model")
            .buckets(vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    ).unwrap();

    /// Whether a neural network model is currently loaded (0=no, 1=yes)
    pub static ref MODEL_LOADED: IntGauge = IntGauge::with_opts(
        Opts::new("actor_model_loaded", "Whether a neural network model is currently loaded")
    ).unwrap();

    // ========== Resource Metrics ==========

    /// Resident set size in bytes
    pub static ref MEMORY_RSS_BYTES: IntGauge = IntGauge::with_opts(
        Opts::new("actor_memory_rss_bytes", "Resident set size in bytes")
    ).unwrap();

    // ========== Info Metrics ==========

    /// Actor metadata with labels for game type and actor ID
    pub static ref ACTOR_INFO: IntGaugeVec = IntGaugeVec::new(
        Opts::new("actor_info", "Actor metadata"),
        &["game", "actor_id"]
    ).unwrap();
}

static INIT: Once = Once::new();

/// Initialize and register all metrics with the registry.
/// Safe to call multiple times - only initializes once.
pub fn init_metrics() {
    INIT.call_once(|| {
        metrics_common::register_all(
            &REGISTRY,
            vec![
                Box::new(EPISODES_TOTAL.clone()),
                Box::new(PLAYER1_WINS.clone()),
                Box::new(PLAYER2_WINS.clone()),
                Box::new(DRAWS.clone()),
                Box::new(EPISODES_PER_SECOND.clone()),
                Box::new(EPISODE_DURATION.clone()),
                Box::new(EPISODE_STEPS.clone()),
                Box::new(MCTS_SEARCHES_TOTAL.clone()),
                Box::new(MCTS_INFERENCE_SECONDS.clone()),
                Box::new(MCTS_SEARCH_SECONDS.clone()),
                Box::new(MCTS_SIMULATIONS_PER_SEARCH.clone()),
                Box::new(TRANSITIONS_STORED.clone()),
                Box::new(DB_WRITE_SECONDS.clone()),
                Box::new(DB_POOL_SIZE.clone()),
                Box::new(DB_POOL_AVAILABLE.clone()),
                Box::new(DB_POOL_WAITING.clone()),
                Box::new(MODEL_RELOADS.clone()),
                Box::new(MODEL_LOAD_SECONDS.clone()),
                Box::new(MODEL_LOADED.clone()),
                Box::new(MEMORY_RSS_BYTES.clone()),
                Box::new(ACTOR_INFO.clone()),
            ],
        );
    });
}

/// Set actor info labels (call once at startup after initializing)
pub fn set_actor_info(game: &str, actor_id: &str) {
    ACTOR_INFO.with_label_values(&[game, actor_id]).set(1);
}

/// Encode all metrics to Prometheus text format
pub fn encode_metrics() -> String {
    metrics_common::encode_metrics(&REGISTRY)
}

/// Read the current resident set size in kB from /proc/self/status.
/// Returns None where /proc is unavailable (e.g. macOS) or on parse failure.
fn read_rss_kb() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = contents.lines().find(|l| l.starts_with("VmRSS:"))?;
    // Format: "VmRSS:    12345 kB"
    line.split_whitespace().nth(1)?.parse().ok()
}

/// Current resident set size in MB, if available on this platform.
pub fn rss_mb() -> Option<f64> {
    read_rss_kb().map(|kb| kb as f64 / 1024.0)
}

/// Update memory RSS gauge from /proc/self/status (no-op where unavailable)
pub fn update_memory_metrics() {
    if let Some(kb) = read_rss_kb() {
        MEMORY_RSS_BYTES.set(kb as i64 * 1024);
    }
}

/// Record a game outcome
pub fn record_outcome(reward: f32) {
    // Reward convention: +1 = player 1 wins, -1 = player 2 wins, 0 = draw
    if reward > 0.5 {
        PLAYER1_WINS.inc();
    } else if reward < -0.5 {
        PLAYER2_WINS.inc();
    } else {
        DRAWS.inc();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_init() {
        // This tests that metrics can be initialized without panicking
        init_metrics();
    }

    #[test]
    fn test_encode_metrics() {
        init_metrics();
        let output = encode_metrics();
        assert!(output.contains("actor_episodes_total"));
        assert!(output.contains("actor_mcts_searches_total"));
    }

    #[test]
    fn test_record_outcome() {
        // Reset counters by reading them
        let _ = PLAYER1_WINS.get();
        let _ = PLAYER2_WINS.get();
        let _ = DRAWS.get();

        record_outcome(1.0);
        record_outcome(-1.0);
        record_outcome(0.0);

        // Just verify no panic
    }
}
