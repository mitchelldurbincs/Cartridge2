"""Checkpoint saving and evaluation orchestration.

Extracted from ``trainer.py`` as pure code motion. Functions take the owning
``Trainer`` instance as their first argument and operate on its state. Builds
on the lower-level utilities in ``checkpoint.py``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from . import metrics as prom_metrics
from .checkpoint import (
    cleanup_old_checkpoints,
    cleanup_temp_onnx_data,
    save_onnx_checkpoint,
    save_pytorch_checkpoint,
)
from .evaluator import OnnxPolicy, RandomPolicy, evaluate
from .stats import EvalStats
from .step_metrics import write_stats

if TYPE_CHECKING:
    from .trainer import Trainer

logger = logging.getLogger(__name__)


def discover_existing_checkpoints(trainer: "Trainer") -> list[Path]:
    """Scan model directory for existing checkpoint files.

    This ensures checkpoint cleanup works correctly across training restarts
    by discovering checkpoints from previous runs.

    Returns:
        List of checkpoint paths sorted by step number (oldest first).
    """
    model_dir = Path(trainer.config.model_dir)
    if not model_dir.exists():
        return []

    # Find all model_step_*.onnx files
    checkpoints = list(model_dir.glob("model_step_*.onnx"))

    # Sort by step number (extracted from filename)
    def extract_step(path: Path) -> int:
        # Filename format: model_step_000100.onnx
        try:
            return int(path.stem.split("_")[-1])
        except (ValueError, IndexError):
            return 0

    checkpoints.sort(key=extract_step)

    if checkpoints:
        logger.info(
            f"Discovered {len(checkpoints)} existing checkpoints "
            f"(steps {extract_step(checkpoints[0])}-{extract_step(checkpoints[-1])})"
        )

    return checkpoints


def handle_checkpoint_and_eval(trainer: "Trainer", step: int, global_step: int) -> None:
    """Save checkpoint and run evaluation if due.

    Args:
        trainer: The owning Trainer instance.
        step: Local step within this training run.
        global_step: Global step across all training runs.
    """
    checkpoint_path: Path | None = None

    if step % trainer.config.checkpoint_interval == 0:
        ckpt_start = time.time()
        checkpoint_path = save_checkpoint(trainer, global_step)
        ckpt_duration = time.time() - ckpt_start
        prom_metrics.record_checkpoint(ckpt_duration)
        trainer.stats.last_checkpoint = str(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    if (
        trainer.config.eval_interval > 0
        and global_step % trainer.config.eval_interval == 0
    ):
        if checkpoint_path is None:
            checkpoint_path = save_checkpoint(trainer, global_step)
            trainer.stats.last_checkpoint = str(checkpoint_path)
            logger.info(f"Saved checkpoint for evaluation: {checkpoint_path}")
        evaluate_checkpoint(trainer, checkpoint_path, global_step)
        write_stats(trainer)


def evaluate_checkpoint(trainer: "Trainer", checkpoint_path: Path, step: int) -> None:
    """Run evaluation on a checkpoint and record results.

    Args:
        trainer: The owning Trainer instance.
        checkpoint_path: Path to the ONNX checkpoint to evaluate.
        step: Current training step for recording.
    """
    logger.info(
        f"Running evaluation at step {step} ({trainer.config.eval_games} games)..."
    )

    eval_start = time.time()
    try:
        model_policy = OnnxPolicy(str(checkpoint_path), temperature=0.0)
        random_policy = RandomPolicy()

        results = evaluate(
            player1=model_policy,
            player2=random_policy,
            env_id=trainer.config.env_id,
            config=trainer.game_config,
            num_games=trainer.config.eval_games,
            verbose=False,
        )

        eval_duration = time.time() - eval_start

        eval_stats = EvalStats(
            step=step,
            win_rate=results.player1_win_rate,
            draw_rate=results.draw_rate,
            loss_rate=results.player2_win_rate,
            games_played=results.games_played,
            avg_game_length=results.avg_game_length,
            timestamp=time.time(),
        )

        trainer.stats.append_eval(eval_stats)

        # Record Prometheus evaluation metrics
        prom_metrics.record_evaluation(
            win_rate=results.player1_win_rate,
            draw_rate=results.draw_rate,
            games_played=results.games_played,
            duration_seconds=eval_duration,
        )

        logger.info(
            f"Evaluation complete: win_rate={eval_stats.win_rate:.1%}, "
            f"draw_rate={eval_stats.draw_rate:.1%}, "
            f"avg_length={eval_stats.avg_game_length:.1f}"
        )

    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")


def save_checkpoint(trainer: "Trainer", step: int) -> Path:
    """Save model checkpoint with atomic write-then-rename.

    Saves both ONNX (for actor inference) and PyTorch (for training continuity).

    Args:
        trainer: The owning Trainer instance.
        step: Current training step.

    Returns:
        Path to the saved ONNX checkpoint.
    """
    model_dir = Path(trainer.config.model_dir)

    # Save ONNX checkpoint (for actor inference)
    checkpoint_path = save_onnx_checkpoint(
        network=trainer.network,
        obs_size=trainer.network.obs_size,
        step=step,
        model_dir=model_dir,
        device=trainer.device,
    )

    # Save PyTorch checkpoint (for training continuity across iterations)
    save_pytorch_checkpoint(
        network=trainer.network,
        optimizer=trainer.optimizer,
        step=step,
        model_dir=model_dir,
        scheduler=trainer.lr_scheduler,
    )

    # Track checkpoints for cleanup
    trainer.checkpoints.append(checkpoint_path)
    trainer.checkpoints = cleanup_old_checkpoints(
        trainer.checkpoints, trainer.config.max_checkpoints
    )

    # Clean up orphaned .onnx.data files from PyTorch exporter
    cleanup_temp_onnx_data(model_dir)

    return checkpoint_path
