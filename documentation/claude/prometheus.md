# Prometheus Metrics Implementation Plan

## Executive Summary

This document outlines the implementation plan for adding Prometheus metrics endpoints to Cartridge-2. Without metrics, we cannot optimize resource utilization or identify performance bottlenecks, leading to wasted cloud spend.

**Current State:** No `/metrics` endpoints exist (actor has a placeholder returning only `actor_up 1`).

**Target State:** All three components (web server, actor, trainer) expose Prometheus-compatible `/metrics` endpoints with comprehensive instrumentation.

---

## Why This Matters for Cost

| Problem | Impact | Metrics Solution |
|---------|--------|------------------|
| Actor underutilization | Paying for 4 replicas when 2 suffice | `actor_episodes_per_second` shows actual throughput |
| Slow model inference | GPU time wasted on inefficient batching | `actor_mcts_inference_seconds` histogram |
| Training stalls | Compute running but no progress | `trainer_steps_total` counter flatlines |
| Memory leaks | OOM kills → pod restarts → lost work | `*_memory_rss_bytes` gauge trending up |
| Database bottlenecks | Actor waiting on PostgreSQL writes | `actor_db_write_seconds` histogram |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Prometheus Server                            │
│              (scrapes all /metrics endpoints)                    │
└───────────────┬──────────────────┬──────────────────┬───────────┘
                │                  │                  │
         ┌──────▼──────┐   ┌───────▼───────┐   ┌─────▼─────┐
         │ Web Server  │   │    Actor      │   │  Trainer  │
         │ :8080       │   │  :8081        │   │  :8082    │
         │ /metrics    │   │  /metrics     │   │  /metrics │
         └─────────────┘   └───────────────┘   └───────────┘
```

Each component exposes metrics on its own port for independent scraping.

---

## Phase 1: Actor Metrics (Highest Priority)

The actor generates training data and is the primary cost driver. Without visibility into actor performance, you cannot right-size replicas.

### 1.1 Add Dependencies

**File:** `actor/Cargo.toml`

```toml
[dependencies]
# Add these
prometheus = "0.13"
lazy_static = "1.4"
```

### 1.2 Create Metrics Module

**File:** `actor/src/metrics.rs` (new file)

```rust
use lazy_static::lazy_static;
use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec,
    IntCounter, IntCounterVec, IntGauge, Registry, TextEncoder, Encoder,
    opts, register_counter, register_counter_vec, register_gauge, register_gauge_vec,
    register_histogram, register_histogram_vec, register_int_counter, register_int_gauge,
};

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // Episode Counters
    pub static ref EPISODES_TOTAL: IntCounter = register_int_counter!(
        "actor_episodes_total",
        "Total number of self-play episodes completed"
    ).unwrap();

    pub static ref PLAYER1_WINS: IntCounter = register_int_counter!(
        "actor_player1_wins_total",
        "Total episodes where player 1 won"
    ).unwrap();

    pub static ref PLAYER2_WINS: IntCounter = register_int_counter!(
        "actor_player2_wins_total",
        "Total episodes where player 2 won"
    ).unwrap();

    pub static ref DRAWS: IntCounter = register_int_counter!(
        "actor_draws_total",
        "Total episodes ending in a draw"
    ).unwrap();

    // Throughput Gauges
    pub static ref EPISODES_PER_SECOND: Gauge = register_gauge!(
        "actor_episodes_per_second",
        "Current episode generation throughput"
    ).unwrap();

    // Episode Histograms
    pub static ref EPISODE_DURATION: Histogram = register_histogram!(
        "actor_episode_duration_seconds",
        "Time to complete one episode",
        vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    ).unwrap();

    pub static ref EPISODE_STEPS: Histogram = register_histogram!(
        "actor_episode_steps",
        "Number of game steps per episode",
        vec![5.0, 10.0, 20.0, 30.0, 42.0, 50.0, 60.0, 80.0, 100.0]
    ).unwrap();

    // MCTS Metrics
    pub static ref MCTS_SEARCHES_TOTAL: IntCounter = register_int_counter!(
        "actor_mcts_searches_total",
        "Total MCTS searches performed"
    ).unwrap();

    pub static ref MCTS_INFERENCE_SECONDS: Histogram = register_histogram!(
        "actor_mcts_inference_seconds",
        "Neural network inference time per MCTS search",
        vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    ).unwrap();

    pub static ref MCTS_SIMULATIONS_PER_SEARCH: Histogram = register_histogram!(
        "actor_mcts_simulations_per_search",
        "Number of MCTS simulations per search",
        vec![50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
    ).unwrap();

    // Storage Metrics
    pub static ref TRANSITIONS_STORED: IntCounter = register_int_counter!(
        "actor_transitions_stored_total",
        "Total transitions written to replay buffer"
    ).unwrap();

    pub static ref DB_WRITE_SECONDS: Histogram = register_histogram!(
        "actor_db_write_seconds",
        "Database write latency",
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    ).unwrap();

    // Model Metrics
    pub static ref MODEL_RELOADS: IntCounter = register_int_counter!(
        "actor_model_reloads_total",
        "Number of model hot-reload events"
    ).unwrap();

    pub static ref MODEL_LOAD_SECONDS: Histogram = register_histogram!(
        "actor_model_load_seconds",
        "Time to load ONNX model",
        vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ).unwrap();

    // Resource Metrics
    pub static ref MEMORY_RSS_BYTES: IntGauge = register_int_gauge!(
        "actor_memory_rss_bytes",
        "Resident set size in bytes"
    ).unwrap();

    // Info Metric (labels for game, actor_id)
    pub static ref ACTOR_INFO: IntGaugeVec = IntGaugeVec::new(
        opts!("actor_info", "Actor metadata"),
        &["game", "actor_id"]
    ).unwrap();
}

/// Encode all metrics to Prometheus text format
pub fn encode_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

/// Update memory RSS gauge (call periodically)
pub fn update_memory_metrics() {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<i64>() {
                            MEMORY_RSS_BYTES.set(kb * 1024);
                        }
                    }
                    break;
                }
            }
        }
    }
}
```

### 1.3 Update Health Server

**File:** `actor/src/health.rs`

Replace the placeholder `metrics_handler`:

```rust
use crate::metrics;

async fn metrics_handler() -> impl IntoResponse {
    // Update memory metrics before responding
    metrics::update_memory_metrics();

    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        metrics::encode_metrics(),
    )
}
```

### 1.4 Instrument the Actor Loop

**File:** `actor/src/actor.rs`

Add metric recording at key points:

```rust
use crate::metrics;

// In run_episode():
pub fn run_episode(...) -> ... {
    let episode_timer = metrics::EPISODE_DURATION.start_timer();

    // ... existing episode logic ...

    // After each MCTS search:
    metrics::MCTS_SEARCHES_TOTAL.inc();
    metrics::MCTS_INFERENCE_SECONDS.observe(search_stats.inference_time_secs);

    // After episode completes:
    metrics::EPISODES_TOTAL.inc();
    metrics::EPISODE_STEPS.observe(steps as f64);

    match outcome {
        Outcome::Player1Win => metrics::PLAYER1_WINS.inc(),
        Outcome::Player2Win => metrics::PLAYER2_WINS.inc(),
        Outcome::Draw => metrics::DRAWS.inc(),
    }

    episode_timer.observe_duration();
}

// In main loop (periodic throughput update):
metrics::EPISODES_PER_SECOND.set(
    episode_count as f64 / start_time.elapsed().as_secs_f64()
);
```

### 1.5 Instrument Storage

**File:** `actor/src/storage/sqlite.rs`

```rust
use crate::metrics;

impl ReplayStore for SqliteStore {
    fn store_transition(&mut self, ...) -> Result<()> {
        let timer = metrics::DB_WRITE_SECONDS.start_timer();
        // ... existing logic ...
        timer.observe_duration();
        metrics::TRANSITIONS_STORED.inc();
        Ok(())
    }
}
```

### 1.6 Instrument Model Watcher

**File:** `actor/src/model_watcher.rs`

```rust
use crate::metrics;

fn reload_model(&mut self) -> Result<()> {
    let timer = metrics::MODEL_LOAD_SECONDS.start_timer();
    // ... existing load logic ...
    timer.observe_duration();
    metrics::MODEL_RELOADS.inc();
    Ok(())
}
```

---

## Phase 2: Web Server Metrics

The web server handles user gameplay and serves the frontend. Metrics help identify latency issues and usage patterns.

### 2.1 Add Dependencies

**File:** `web/Cargo.toml`

```toml
[dependencies]
prometheus = "0.13"
lazy_static = "1.4"
```

### 2.2 Create Metrics Module

**File:** `web/src/metrics.rs` (new file)

```rust
use lazy_static::lazy_static;
use prometheus::{
    Counter, Histogram, HistogramOpts, IntCounter, IntGauge, Registry,
    TextEncoder, Encoder, register_histogram, register_int_counter, register_int_gauge,
};

lazy_static! {
    // Game Session Metrics
    pub static ref GAMES_CREATED: IntCounter = register_int_counter!(
        "web_games_created_total",
        "Total game sessions created"
    ).unwrap();

    pub static ref GAMES_ACTIVE: IntGauge = register_int_gauge!(
        "web_games_active",
        "Currently active game sessions"
    ).unwrap();

    pub static ref MOVES_PLAYED: IntCounter = register_int_counter!(
        "web_moves_played_total",
        "Total moves played across all games"
    ).unwrap();

    // Request Latency
    pub static ref REQUEST_LATENCY: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "web_request_duration_seconds",
            "HTTP request latency"
        ).buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]),
        &["endpoint", "method"]
    ).unwrap();

    // Bot Inference
    pub static ref BOT_MOVE_SECONDS: Histogram = register_histogram!(
        "web_bot_move_seconds",
        "Time for bot to compute move",
        vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    ).unwrap();

    // Model Status
    pub static ref MODEL_LOADED: IntGauge = register_int_gauge!(
        "web_model_loaded",
        "Whether a valid model is loaded (0/1)"
    ).unwrap();
}

pub fn encode_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}
```

### 2.3 Add Metrics Endpoint

**File:** `web/src/main.rs`

```rust
mod metrics;

// Add route
let app = Router::new()
    // ... existing routes ...
    .route("/metrics", get(metrics_handler));

async fn metrics_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        metrics::encode_metrics(),
    )
}
```

### 2.4 Instrument Handlers

**File:** `web/src/handlers/game.rs`

```rust
use crate::metrics;

pub async fn new_game(...) -> ... {
    metrics::GAMES_CREATED.inc();
    metrics::GAMES_ACTIVE.inc();
    // ... existing logic ...
}

pub async fn make_move(...) -> ... {
    metrics::MOVES_PLAYED.inc();

    let bot_timer = metrics::BOT_MOVE_SECONDS.start_timer();
    // ... bot move computation ...
    bot_timer.observe_duration();

    // ... existing logic ...
}
```

---

## Phase 3: Trainer Metrics

The trainer is the most complex component. Metrics help identify training stalls, convergence issues, and resource utilization.

### 3.1 Add Dependencies

**File:** `trainer/pyproject.toml`

```toml
dependencies = [
    # ... existing deps ...
    "prometheus-client>=0.20.0",
]
```

### 3.2 Create Metrics Module

**File:** `trainer/src/trainer/metrics.py` (new file)

```python
"""Prometheus metrics for the trainer."""
from prometheus_client import (
    Counter, Gauge, Histogram, Info,
    start_http_server, generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY
)
import threading

# Training Progress
STEPS_TOTAL = Counter(
    'trainer_steps_total',
    'Total training steps completed'
)

SAMPLES_SEEN = Counter(
    'trainer_samples_seen_total',
    'Total unique samples processed'
)

ITERATIONS_TOTAL = Counter(
    'trainer_iterations_total',
    'Total training iterations completed'
)

# Loss Metrics
LOSS_TOTAL = Gauge(
    'trainer_loss_total',
    'Combined loss value'
)

LOSS_VALUE = Gauge(
    'trainer_loss_value',
    'Value head loss'
)

LOSS_POLICY = Gauge(
    'trainer_loss_policy',
    'Policy head loss'
)

# Learning Rate
LEARNING_RATE = Gauge(
    'trainer_learning_rate',
    'Current learning rate'
)

# Replay Buffer
REPLAY_BUFFER_SIZE = Gauge(
    'trainer_replay_buffer_size',
    'Current replay buffer size'
)

# Step Timing
STEP_DURATION = Histogram(
    'trainer_step_duration_seconds',
    'Time per training step',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

BATCH_LOAD_DURATION = Histogram(
    'trainer_batch_load_seconds',
    'Time to load a batch from replay buffer',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
)

# Evaluation Metrics
EVAL_WIN_RATE = Gauge(
    'trainer_eval_win_rate',
    'Win rate against baseline'
)

EVAL_DRAW_RATE = Gauge(
    'trainer_eval_draw_rate',
    'Draw rate against baseline'
)

EVAL_LOSS_RATE = Gauge(
    'trainer_eval_loss_rate',
    'Loss rate against baseline'
)

EVAL_GAMES_TOTAL = Counter(
    'trainer_eval_games_total',
    'Total evaluation games played'
)

# Checkpoint Metrics
CHECKPOINTS_SAVED = Counter(
    'trainer_checkpoints_saved_total',
    'Total checkpoints saved'
)

CHECKPOINT_DURATION = Histogram(
    'trainer_checkpoint_duration_seconds',
    'Time to save checkpoint + ONNX export',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Model Info
MODEL_INFO = Info(
    'trainer_model',
    'Model metadata'
)

# Gradient Metrics
GRADIENT_NORM = Gauge(
    'trainer_gradient_norm',
    'Gradient norm before clipping'
)

# Memory (if torch available)
GPU_MEMORY_ALLOCATED = Gauge(
    'trainer_gpu_memory_allocated_bytes',
    'GPU memory currently allocated'
)

GPU_MEMORY_RESERVED = Gauge(
    'trainer_gpu_memory_reserved_bytes',
    'GPU memory currently reserved'
)


def start_metrics_server(port: int = 8082):
    """Start Prometheus metrics HTTP server in background thread."""
    start_http_server(port)
    print(f"Prometheus metrics available at http://0.0.0.0:{port}/metrics")


def update_gpu_memory():
    """Update GPU memory metrics (call periodically)."""
    try:
        import torch
        if torch.cuda.is_available():
            GPU_MEMORY_ALLOCATED.set(torch.cuda.memory_allocated())
            GPU_MEMORY_RESERVED.set(torch.cuda.memory_reserved())
    except ImportError:
        pass
```

### 3.3 Instrument Training Loop

**File:** `trainer/src/trainer/trainer.py`

```python
from trainer import metrics
import time

def train(self, steps: int) -> dict:
    for step in range(steps):
        step_start = time.perf_counter()

        # Batch loading
        batch_start = time.perf_counter()
        batch = self.replay.sample(self.config.batch_size)
        metrics.BATCH_LOAD_DURATION.observe(time.perf_counter() - batch_start)

        # Forward + backward
        loss_dict = self._train_step(batch)

        # Record losses
        metrics.LOSS_TOTAL.set(loss_dict["loss/total"])
        metrics.LOSS_VALUE.set(loss_dict["loss/value"])
        metrics.LOSS_POLICY.set(loss_dict["loss/policy"])

        # Record LR
        metrics.LEARNING_RATE.set(self.optimizer.param_groups[0]["lr"])

        # Record gradient norm (if computed)
        if "gradient_norm" in loss_dict:
            metrics.GRADIENT_NORM.set(loss_dict["gradient_norm"])

        # Increment counters
        metrics.STEPS_TOTAL.inc()
        metrics.SAMPLES_SEEN.inc(self.config.batch_size)

        # Record step duration
        metrics.STEP_DURATION.observe(time.perf_counter() - step_start)

        # Periodic GPU memory update
        if step % 100 == 0:
            metrics.update_gpu_memory()
```

### 3.4 Instrument Evaluation

**File:** `trainer/src/trainer/evaluator.py`

```python
from trainer import metrics

def evaluate(self, games: int) -> EvalStats:
    # ... run evaluation games ...

    # Record results
    metrics.EVAL_WIN_RATE.set(stats.win_rate)
    metrics.EVAL_DRAW_RATE.set(stats.draw_rate)
    metrics.EVAL_LOSS_RATE.set(stats.loss_rate)
    metrics.EVAL_GAMES_TOTAL.inc(games)

    return stats
```

### 3.5 Instrument Checkpointing

**File:** `trainer/src/trainer/checkpoint.py`

```python
from trainer import metrics
import time

def save_checkpoint(self, path: str):
    start = time.perf_counter()
    # ... save logic ...
    metrics.CHECKPOINT_DURATION.observe(time.perf_counter() - start)
    metrics.CHECKPOINTS_SAVED.inc()
```

### 3.6 Start Metrics Server

**File:** `trainer/src/trainer/__main__.py`

```python
from trainer import metrics

def main():
    # Start metrics server early
    metrics.start_metrics_server(port=8082)

    # ... rest of CLI handling ...
```

---

## Phase 4: Prometheus Configuration

### 4.1 prometheus.yml

**File:** `prometheus.yml` (new file in project root)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'web'
    static_configs:
      - targets: ['web:8080']
    metrics_path: /metrics

  - job_name: 'actor'
    static_configs:
      - targets: ['actor:8081']
    metrics_path: /metrics

  - job_name: 'trainer'
    static_configs:
      - targets: ['trainer:8082']
    metrics_path: /metrics
```

### 4.2 Docker Compose Addition

**File:** `docker-compose.yml` (add to existing)

```yaml
services:
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: cartridge-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=7d'
    depends_on:
      - web
      - actor
      - trainer

volumes:
  prometheus_data:
```

---

## Key Metrics for Cost Optimization

### Actor Right-Sizing Dashboard

```promql
# Episodes per second per actor
rate(actor_episodes_total[5m])

# CPU utilization proxy (if episodes/sec is low, actor is underutilized)
actor_episodes_per_second / 10  # Assumes 10 eps/s is full utilization

# Memory trend (detect leaks)
rate(actor_memory_rss_bytes[1h])

# Replica recommendation (target 80% utilization)
ceil(sum(rate(actor_episodes_total[5m])) / 8)
```

### Training Efficiency Dashboard

```promql
# Steps per second
rate(trainer_steps_total[5m])

# Loss trend (should decrease)
trainer_loss_total

# Learning rate schedule (verify cosine annealing)
trainer_learning_rate

# GPU utilization proxy
trainer_step_duration_seconds / 0.1  # Assumes 0.1s is GPU-bound
```

### Alerts

```yaml
# Alert: Actor throughput dropped
- alert: ActorThroughputLow
  expr: rate(actor_episodes_total[5m]) < 1
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Actor throughput is very low"

# Alert: Training stalled
- alert: TrainingStalled
  expr: increase(trainer_steps_total[10m]) == 0
  for: 15m
  labels:
    severity: critical
  annotations:
    summary: "Training has not progressed in 15 minutes"

# Alert: Memory leak
- alert: MemoryLeak
  expr: rate(actor_memory_rss_bytes[1h]) > 10000000  # 10MB/hour growth
  for: 2h
  labels:
    severity: warning
  annotations:
    summary: "Potential memory leak detected"
```

---

## Implementation Order

| Phase | Component | Effort | Impact |
|-------|-----------|--------|--------|
| 1 | Actor metrics | 2-3 hours | HIGH - Primary cost driver visibility |
| 2 | Web server metrics | 1-2 hours | MEDIUM - User-facing latency visibility |
| 3 | Trainer metrics | 2-3 hours | HIGH - Training efficiency visibility |
| 4 | Prometheus setup | 1 hour | Required for scraping |
| 5 | Grafana dashboards | 2-3 hours | Visualization (optional but recommended) |

**Total Estimated Effort:** 8-12 hours

---

## Testing the Implementation

### Manual Verification

```bash
# Actor metrics
curl http://localhost:8081/metrics | grep actor_

# Web metrics
curl http://localhost:8080/metrics | grep web_

# Trainer metrics
curl http://localhost:8082/metrics | grep trainer_
```

### Expected Output (Actor)

```
# HELP actor_episodes_total Total number of self-play episodes completed
# TYPE actor_episodes_total counter
actor_episodes_total 1234

# HELP actor_episode_duration_seconds Time to complete one episode
# TYPE actor_episode_duration_seconds histogram
actor_episode_duration_seconds_bucket{le="0.01"} 0
actor_episode_duration_seconds_bucket{le="0.05"} 12
actor_episode_duration_seconds_bucket{le="0.1"} 156
...
actor_episode_duration_seconds_sum 89.234
actor_episode_duration_seconds_count 1234

# HELP actor_episodes_per_second Current episode generation throughput
# TYPE actor_episodes_per_second gauge
actor_episodes_per_second 8.5
```

---

## Files to Create/Modify Summary

### New Files
- `actor/src/metrics.rs`
- `web/src/metrics.rs`
- `trainer/src/trainer/metrics.py`
- `prometheus.yml`

### Modified Files
- `actor/Cargo.toml` - Add prometheus, lazy_static
- `actor/src/lib.rs` - Add `mod metrics;`
- `actor/src/health.rs` - Implement metrics_handler
- `actor/src/actor.rs` - Instrument episode loop
- `actor/src/storage/sqlite.rs` - Instrument storage
- `actor/src/model_watcher.rs` - Instrument reloads
- `web/Cargo.toml` - Add prometheus, lazy_static
- `web/src/main.rs` - Add metrics endpoint
- `web/src/handlers/game.rs` - Instrument handlers
- `trainer/pyproject.toml` - Add prometheus-client
- `trainer/src/trainer/trainer.py` - Instrument training
- `trainer/src/trainer/evaluator.py` - Instrument evaluation
- `trainer/src/trainer/__main__.py` - Start metrics server
- `docker-compose.yml` - Add prometheus service

---

## Future Enhancements

1. **Grafana Dashboards** - Pre-built dashboards for each component
2. **Alertmanager Integration** - PagerDuty/Slack alerts
3. **OpenTelemetry Migration** - Unified traces + metrics
4. **Custom Exporters** - PostgreSQL replay buffer stats
5. **Push Gateway** - For short-lived batch jobs
