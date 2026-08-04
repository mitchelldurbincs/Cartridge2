"""Prometheus metrics for the trainer component.

This module provides comprehensive metrics for monitoring training progress,
including loss values, learning rate, throughput, and evaluation results.
"""

import logging
import threading

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    start_http_server,
)

logger = logging.getLogger(__name__)

# ========== Training Metrics ==========

# Current training step
TRAINING_STEP = Gauge(
    "trainer_training_step",
    "Current training step",
)

# Total loss value
TOTAL_LOSS = Gauge(
    "trainer_loss_total",
    "Current total loss value",
)

# Value loss
VALUE_LOSS = Gauge(
    "trainer_loss_value",
    "Current value loss",
)

# Policy loss
POLICY_LOSS = Gauge(
    "trainer_loss_policy",
    "Current policy loss",
)

# Learning rate
LEARNING_RATE = Gauge(
    "trainer_learning_rate",
    "Current learning rate",
)

# ========== Training Step Histogram ==========

STEP_DURATION = Histogram(
    "trainer_step_duration_seconds",
    "Time per training step",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

# ========== Throughput Metrics ==========

# Samples processed counter
SAMPLES_PROCESSED = Counter(
    "trainer_samples_processed_total",
    "Total number of samples processed",
)

# Throughput gauge
SAMPLES_PER_SECOND = Gauge(
    "trainer_samples_per_second",
    "Current training throughput (samples/second)",
)

# Steps per second
STEPS_PER_SECOND = Gauge(
    "trainer_steps_per_second",
    "Current training throughput (steps/second)",
)

# ========== Replay Store Metrics ==========

REPLAY_RECORD_COUNT = Gauge(
    "trainer_replay_record_count",
    "Number of opaque records in the exact replay selection",
)

# ========== Checkpoint Metrics ==========

CHECKPOINTS_SAVED = Counter(
    "trainer_checkpoints_saved_total",
    "Total number of checkpoints saved",
)

CHECKPOINT_DURATION = Histogram(
    "trainer_checkpoint_duration_seconds",
    "Time to save checkpoint",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# ========== Evaluation Metrics ==========

EVALUATION_WIN_RATE = Gauge(
    "trainer_eval_win_rate",
    "Win rate against random baseline",
)

EVALUATION_DRAW_RATE = Gauge(
    "trainer_eval_draw_rate",
    "Draw rate against random baseline",
)

EVALUATION_GAMES = Counter(
    "trainer_eval_games_total",
    "Total evaluation games played",
)

EVALUATION_DURATION = Histogram(
    "trainer_evaluation_duration_seconds",
    "Time to run evaluation",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# ========== GPU/Memory Metrics ==========

GPU_MEMORY_USED_BYTES = Gauge(
    "trainer_gpu_memory_used_bytes",
    "GPU memory used in bytes",
)

GPU_MEMORY_CACHED_BYTES = Gauge(
    "trainer_gpu_memory_cached_bytes",
    "GPU memory cached in bytes",
)

# ========== Info Metric ==========

TRAINER_INFO = Info(
    "trainer",
    "Trainer metadata",
)


# Server state
_metrics_server_started = False
_metrics_server_lock = threading.Lock()


def start_metrics_server(port: int = 9090) -> bool:
    """Start the Prometheus metrics HTTP server.

    Args:
        port: Port to listen on (default 9090).

    Returns:
        True if server started, False if already running.
    """
    global _metrics_server_started

    with _metrics_server_lock:
        if _metrics_server_started:
            logger.debug("Metrics server already running")
            return False

        try:
            start_http_server(port)
            _metrics_server_started = True
            logger.info(f"Prometheus metrics server started on port {port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to start metrics server on port {port}: {e}")
            return False


def set_trainer_info(env_id: str, device: str, batch_size: int) -> None:
    """Set trainer info labels.

    Args:
        env_id: Game environment ID.
        device: Training device (cpu, cuda, mps).
        batch_size: Training batch size.
    """
    TRAINER_INFO.info(
        {
            "env_id": env_id,
            "device": device,
            "batch_size": str(batch_size),
        }
    )


def record_training_step(
    step: int,
    total_loss: float,
    value_loss: float,
    policy_loss: float,
    learning_rate: float,
    duration_seconds: float,
    batch_size: int,
) -> None:
    """Record metrics for a training step.

    Args:
        step: Current training step.
        total_loss: Total loss value.
        value_loss: Value head loss.
        policy_loss: Policy head loss.
        learning_rate: Current learning rate.
        duration_seconds: Time taken for this step.
        batch_size: Number of samples in batch.
    """
    TRAINING_STEP.set(step)
    TOTAL_LOSS.set(total_loss)
    VALUE_LOSS.set(value_loss)
    POLICY_LOSS.set(policy_loss)
    LEARNING_RATE.set(learning_rate)
    STEP_DURATION.observe(duration_seconds)
    SAMPLES_PROCESSED.inc(batch_size)

    # Calculate throughput
    if duration_seconds > 0:
        SAMPLES_PER_SECOND.set(batch_size / duration_seconds)
        STEPS_PER_SECOND.set(1.0 / duration_seconds)


def record_checkpoint(duration_seconds: float) -> None:
    """Record checkpoint save metrics.

    Args:
        duration_seconds: Time to save checkpoint.
    """
    CHECKPOINTS_SAVED.inc()
    CHECKPOINT_DURATION.observe(duration_seconds)


def record_evaluation(
    win_rate: float,
    draw_rate: float,
    games_played: int,
    duration_seconds: float,
) -> None:
    """Record evaluation results.

    Args:
        win_rate: Win rate against baseline.
        draw_rate: Draw rate.
        games_played: Number of games played.
        duration_seconds: Time to run evaluation.
    """
    EVALUATION_WIN_RATE.set(win_rate)
    EVALUATION_DRAW_RATE.set(draw_rate)
    EVALUATION_GAMES.inc(games_played)
    EVALUATION_DURATION.observe(duration_seconds)


def update_replay_record_count(count: int) -> None:
    """Update the exact replay selection's record-count metric.

    Args:
        count: Number of opaque replay records.
    """
    REPLAY_RECORD_COUNT.set(count)


def update_gpu_memory() -> None:
    """Update GPU memory metrics (if available)."""
    try:
        import torch

        if torch.cuda.is_available():
            GPU_MEMORY_USED_BYTES.set(torch.cuda.memory_allocated())
            GPU_MEMORY_CACHED_BYTES.set(torch.cuda.memory_reserved())
    except Exception:
        pass  # Silently ignore if GPU not available
