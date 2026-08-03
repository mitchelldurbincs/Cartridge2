"""Per-step metrics recording and content-addressed stats publication.

Functions take the owning ``AlphaZeroLearner`` as their first argument and
operate on its state.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from . import metrics as prom_metrics
from .stats import write_ephemeral_stats_projection

if TYPE_CHECKING:
    from .trainer import AlphaZeroLearner

logger = logging.getLogger(__name__)


def record_step_metrics(
    trainer: "AlphaZeroLearner",
    step: int,
    global_step: int,
    metrics: dict[str, float],
    step_duration: float,
    batch_size: int,
    replay,
    env_id: str,
) -> None:
    """Update stats, Prometheus metrics, rolling losses, and log progress.

    Args:
        trainer: The owning AlphaZero learner.
        step: Local step within this training run.
        global_step: Global step across all training runs.
        metrics: Loss metrics from the training step.
        step_duration: Wall-clock time for the training step.
        batch_size: Number of samples in the batch.
        replay: Replay store (for periodic size updates).
        env_id: Environment identifier.
    """
    # Update stats
    trainer.stats.step = global_step
    trainer.stats.total_loss = metrics["loss/total"]
    trainer.stats.value_loss = metrics["loss/value"]
    trainer.stats.policy_loss = metrics["loss/policy"]
    trainer.stats.learning_rate = trainer.optimizer.param_groups[0]["lr"]
    trainer.stats.samples_seen = trainer.samples_seen
    # Wall clocks can move backwards across hosts or after NTP correction.
    # Keep the persisted run chronology logical and nondecreasing.
    trainer.stats.timestamp = max(time.time(), trainer.stats.timestamp)

    # Record Prometheus metrics
    prom_metrics.record_training_step(
        step=global_step,
        total_loss=metrics["loss/total"],
        value_loss=metrics["loss/value"],
        policy_loss=metrics["loss/policy"],
        learning_rate=trainer.optimizer.param_groups[0]["lr"],
        duration_seconds=step_duration,
        batch_size=batch_size,
    )

    # Update the cached profile record count periodically.
    if step % trainer._replay_record_count_update_interval == 0:
        trainer._replay_record_count_cache = replay.count()
        prom_metrics.update_replay_record_count(trainer._replay_record_count_cache)
        prom_metrics.update_gpu_memory()
    trainer.stats.replay_record_count = trainer._replay_record_count_cache

    # Track recent losses for rolling average
    trainer._recent_losses.append(
        {
            "total": metrics["loss/total"],
            "value": metrics["loss/value"],
            "policy": metrics["loss/policy"],
        }
    )
    if len(trainer._recent_losses) > trainer._rolling_window:
        trainer._recent_losses.pop(0)

    # Log progress
    if step % trainer.config.log_interval == 0:
        lr = trainer.optimizer.param_groups[0]["lr"]
        n = len(trainer._recent_losses)
        avg_total = sum(x["total"] for x in trainer._recent_losses) / n
        avg_value = sum(x["value"] for x in trainer._recent_losses) / n
        avg_policy = sum(x["policy"] for x in trainer._recent_losses) / n
        logger.info(
            f"Step {global_step} ({step}/{trainer.config.total_steps}): "
            f"loss={metrics['loss/total']:.4f} "
            f"(v={metrics['loss/value']:.4f}, p={metrics['loss/policy']:.4f}) "
            f"avg100={avg_total:.4f} (v={avg_value:.4f}, p={avg_policy:.4f}) "
            f"lr={lr:.2e}"
        )

    # Save stats (uses bounded append)
    if step % trainer.config.stats_interval == 0:
        history_entry = {
            "step": global_step,
            "total_loss": metrics["loss/total"],
            "value_loss": metrics["loss/value"],
            "policy_loss": metrics["loss/policy"],
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            "grad_norm": metrics.get("grad_norm"),
        }
        trainer.stats.append_history(history_entry)
        write_ephemeral_stats_projection(
            trainer.stats,
            trainer.config.stats_path,
        )

        if trainer.config.metrics_hook is not None:
            try:
                payload = dict(history_entry)
                payload["samples_seen"] = trainer.samples_seen
                trainer.config.metrics_hook(payload, global_step)
            except Exception as e:
                logger.warning(f"metrics_hook failed: {e}")
