"""Training loop with ONNX export and stats tracking.

This module provides the main training loop that:
1. Loads batches from the PostgreSQL replay buffer
2. Trains the AlphaZero-style network
3. Exports ONNX checkpoints using atomic write-then-rename
4. Writes stats.json for web visualization

Training targets:
    - Policy targets: MCTS visit count distributions (soft targets) from the actor.
    - Value targets: Game outcomes (win=+1, loss=-1, draw=0) propagated from
      terminal states. Each position is labeled with the final outcome from
      that player's perspective, giving meaningful signal at every position.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.utils as nn_utils
import torch.optim as optim

from . import metrics as prom_metrics
from .backoff import LOG_EVERY_N_WAITS, WaitTimeout, wait_with_backoff
from .checkpoint import (
    cleanup_old_checkpoints,
    cleanup_temp_onnx_data,
    load_pytorch_checkpoint,
    save_onnx_checkpoint,
    save_pytorch_checkpoint,
)
from .config import TrainerConfig
from .evaluator import OnnxPolicy, RandomPolicy, evaluate
from .game_config import GameConfig, get_config
from .lr_scheduler import LRConfig, WarmupCosineScheduler
from .network import AlphaZeroLoss, create_network
from .stats import EvalStats, TrainerStats, load_stats, write_stats
from .storage import create_replay_buffer

# Re-export for convenience (used by tests and external callers)
__all__ = ["Trainer", "TrainerConfig", "EvalStats", "TrainerStats", "WaitTimeout"]

logger = logging.getLogger(__name__)


class Trainer:
    """AlphaZero-style trainer for game agents."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Get game configuration
        self.game_config = get_config(config.env_id)

        # Create model directory
        Path(config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(config.stats_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize network
        self.network = create_network(config.env_id)
        self.network.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Try to load existing checkpoint (critical for training continuity!)
        self._checkpoint_loaded = False
        self._loaded_step: int | None = None
        self._loaded_scheduler_state: dict | None = None
        checkpoint_result = load_pytorch_checkpoint(
            self.network,
            self.optimizer,
            Path(config.model_dir),
            self.device,
        )
        if checkpoint_result is not None:
            loaded_step, self._loaded_scheduler_state = checkpoint_result
            self._loaded_step = loaded_step
            self._checkpoint_loaded = True
            logger.info(f"Resuming training from checkpoint (step {loaded_step})")

        # Initialize LR scheduler (warmup + cosine annealing)
        # Use lr_total_steps for continuous decay across iterations if set
        lr_horizon = (
            config.lr_total_steps if config.lr_total_steps > 0 else config.total_steps
        )
        lr_config = LRConfig(
            target_lr=config.learning_rate,
            warmup_steps=config.lr_warmup_steps,
            warmup_start_ratio=config.lr_warmup_start_ratio,
            min_ratio=config.lr_min_ratio,
            total_steps=lr_horizon,
            enabled=config.use_lr_scheduler,
        )
        self.lr_scheduler = WarmupCosineScheduler(
            self.optimizer,
            lr_config,
            from_checkpoint=self._checkpoint_loaded,
        )

        # Restore scheduler state from checkpoint if available
        if self._loaded_scheduler_state is not None:
            try:
                self.lr_scheduler.load_state_dict(self._loaded_scheduler_state)
            except Exception as e:
                logger.warning(f"Failed to restore scheduler state: {e}")

        # Initialize loss function
        self.loss_fn = AlphaZeroLoss(
            value_weight=config.value_loss_weight,
            policy_weight=config.policy_loss_weight,
        )

        # Stats tracking - load existing stats to preserve eval history
        self.stats = load_stats(config.stats_path)
        self.stats.total_steps = config.total_steps
        self.stats.env_id = config.env_id
        self.stats._max_history = config.max_history_length
        self.samples_seen = 0

        # Replay maintenance
        self._replay_cleanup_every = (
            config.replay_cleanup_interval
            if config.replay_cleanup_interval > 0
            else config.stats_interval
        )

        # Rolling window for averaging (last 100 steps)
        self._recent_losses: list[dict[str, float]] = []
        self._rolling_window = 100

        # Buffer size caching (avoid expensive count() calls every step)
        self._buffer_size_cache: int = 0
        self._buffer_size_update_interval: int = 100  # Update every 100 steps

        # Checkpoint tracking - discover existing checkpoints on disk
        self.checkpoints: list[Path] = self._discover_existing_checkpoints()
        self.latest_checkpoint: Path | None = None

    def _discover_existing_checkpoints(self) -> list[Path]:
        """Scan model directory for existing checkpoint files.

        This ensures checkpoint cleanup works correctly across training restarts
        by discovering checkpoints from previous runs.

        Returns:
            List of checkpoint paths sorted by step number (oldest first).
        """
        model_dir = Path(self.config.model_dir)
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

    def _wait_with_backoff(
        self, condition_fn, description: str, check_interval: float | None = None
    ) -> None:
        """Wait for a condition with periodic checks and timeout.

        Args:
            condition_fn: Callable returning True when condition is met.
            description: Human-readable description for logging.
            check_interval: Override default wait interval.

        Raises:
            WaitTimeout: If max_wait is exceeded (and max_wait > 0).
        """
        interval = check_interval or self.config.wait_interval
        wait_with_backoff(
            condition_fn=condition_fn,
            description=description,
            interval=interval,
            max_wait=self.config.max_wait,
            logger=logger,
        )

    def _create_replay_buffer(self):
        """Create PostgreSQL replay buffer using the factory.

        Returns:
            PostgresReplayBuffer instance.
        """
        logger.info("Connecting to PostgreSQL replay buffer...")
        return create_replay_buffer()

    def _setup_replay(self, replay, env_id: str) -> None:
        """Set up the replay buffer: clear if needed, load metadata, wait for data.

        Args:
            replay: The replay buffer instance.
            env_id: Environment identifier for filtering.

        Raises:
            WaitTimeout: If max_wait is exceeded waiting for data.
        """
        if self.config.clear_replay_on_start:
            deleted = replay.clear_transitions()
            logger.info(
                f"Cleared {deleted} transitions from replay buffer before training"
            )

        # Try to get game metadata from database (preferred, self-describing)
        db_metadata = replay.get_metadata(env_id)
        if db_metadata:
            logger.info(f"Using game metadata from database for {env_id}")
            self.game_config = GameConfig(
                env_id=db_metadata.env_id,
                display_name=db_metadata.display_name,
                board_width=db_metadata.board_width,
                board_height=db_metadata.board_height,
                num_actions=db_metadata.num_actions,
                obs_size=db_metadata.obs_size,
                legal_mask_offset=db_metadata.legal_mask_offset,
            )
        else:
            logger.warning(
                f"No metadata in database for {env_id}, using fallback config"
            )

        buffer_size = replay.count(env_id=env_id)
        logger.info(f"Replay buffer contains {buffer_size} transitions for {env_id}")

        # Wait for enough data with proper backoff
        if buffer_size < self.config.batch_size:
            self._wait_with_backoff(
                lambda: replay.count(env_id=env_id) >= self.config.batch_size,
                f"sufficient data ({self.config.batch_size} samples for {env_id})",
            )
            buffer_size = replay.count(env_id=env_id)
            logger.info(f"Replay buffer now has {buffer_size} transitions for {env_id}")

        self._buffer_size_cache = buffer_size
        self.stats.replay_buffer_size = buffer_size

    def _record_step_metrics(
        self,
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
            step: Local step within this training run.
            global_step: Global step across all training runs.
            metrics: Loss metrics from the training step.
            step_duration: Wall-clock time for the training step.
            batch_size: Number of samples in the batch.
            replay: Replay buffer (for periodic buffer size updates).
            env_id: Environment identifier.
        """
        # Update stats
        self.stats.total_loss = metrics["loss/total"]
        self.stats.value_loss = metrics["loss/value"]
        self.stats.policy_loss = metrics["loss/policy"]
        self.stats.learning_rate = self.optimizer.param_groups[0]["lr"]
        self.stats.samples_seen = self.samples_seen
        self.stats.timestamp = time.time()

        # Record Prometheus metrics
        prom_metrics.record_training_step(
            step=global_step,
            total_loss=metrics["loss/total"],
            value_loss=metrics["loss/value"],
            policy_loss=metrics["loss/policy"],
            learning_rate=self.optimizer.param_groups[0]["lr"],
            duration_seconds=step_duration,
            batch_size=batch_size,
        )

        # Update cached buffer size periodically
        if step % self._buffer_size_update_interval == 0:
            self._buffer_size_cache = replay.count(env_id=env_id)
            prom_metrics.update_replay_buffer_size(self._buffer_size_cache)
            prom_metrics.update_gpu_memory()
        self.stats.replay_buffer_size = self._buffer_size_cache

        # Track recent losses for rolling average
        self._recent_losses.append(
            {
                "total": metrics["loss/total"],
                "value": metrics["loss/value"],
                "policy": metrics["loss/policy"],
            }
        )
        if len(self._recent_losses) > self._rolling_window:
            self._recent_losses.pop(0)

        # Log progress
        if step % self.config.log_interval == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            n = len(self._recent_losses)
            avg_total = sum(x["total"] for x in self._recent_losses) / n
            avg_value = sum(x["value"] for x in self._recent_losses) / n
            avg_policy = sum(x["policy"] for x in self._recent_losses) / n
            logger.info(
                f"Step {global_step} ({step}/{self.config.total_steps}): "
                f"loss={metrics['loss/total']:.4f} "
                f"(v={metrics['loss/value']:.4f}, p={metrics['loss/policy']:.4f}) "
                f"avg100={avg_total:.4f} (v={avg_value:.4f}, p={avg_policy:.4f}) "
                f"lr={lr:.2e}"
            )

        # Save stats (uses bounded append)
        if step % self.config.stats_interval == 0:
            history_entry = {
                "step": global_step,
                "total_loss": metrics["loss/total"],
                "value_loss": metrics["loss/value"],
                "policy_loss": metrics["loss/policy"],
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            if "grad_norm" in metrics:
                history_entry["grad_norm"] = metrics["grad_norm"]
            self.stats.append_history(history_entry)
            self._write_stats()

    def _handle_replay_cleanup(self, global_step: int, replay, env_id: str) -> None:
        """Clean up old replay transitions if configured.

        Args:
            global_step: Current global step.
            replay: Replay buffer instance.
            env_id: Environment identifier.
        """
        if (
            self.config.replay_window > 0
            and global_step % self._replay_cleanup_every == 0
        ):
            deleted = replay.cleanup(self.config.replay_window)
            if deleted > 0:
                logger.info(
                    f"Replay cleanup removed {deleted} old transitions "
                    f"(window={self.config.replay_window})"
                )
            self._buffer_size_cache = replay.count(env_id=env_id)
            self.stats.replay_buffer_size = self._buffer_size_cache

    def _handle_checkpoint_and_eval(self, step: int, global_step: int) -> None:
        """Save checkpoint and run evaluation if due.

        Args:
            step: Local step within this training run.
            global_step: Global step across all training runs.
        """
        checkpoint_path: Path | None = None

        if step % self.config.checkpoint_interval == 0:
            ckpt_start = time.time()
            checkpoint_path = self._save_checkpoint(global_step)
            ckpt_duration = time.time() - ckpt_start
            prom_metrics.record_checkpoint(ckpt_duration)
            self.stats.last_checkpoint = str(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        if (
            self.config.eval_interval > 0
            and global_step % self.config.eval_interval == 0
        ):
            if checkpoint_path is None:
                checkpoint_path = self._save_checkpoint(global_step)
                self.stats.last_checkpoint = str(checkpoint_path)
                logger.info(f"Saved checkpoint for evaluation: {checkpoint_path}")
            self._evaluate_checkpoint(checkpoint_path, global_step)
            self._write_stats()

    def train(self) -> TrainerStats:
        """Run the training loop.

        Returns:
            Final training statistics.

        Raises:
            WaitTimeout: If max_wait is exceeded waiting for database or data.
        """
        logger.info(f"Starting training for {self.config.total_steps} steps")
        if self.config.grad_clip_norm > 0:
            logger.info(
                f"Gradient clipping enabled: max_norm={self.config.grad_clip_norm}"
            )
        if self.lr_scheduler.config.enabled:
            logger.info(f"LR scheduler: {self.lr_scheduler}")

        prom_metrics.set_trainer_info(
            env_id=self.config.env_id,
            device=self.config.device,
            batch_size=self.config.batch_size,
        )

        replay = self._create_replay_buffer()
        try:
            env_id = self.config.env_id
            self._setup_replay(replay, env_id)

            # Training loop — while loop so step only increments after successful training
            consecutive_skips = 0
            start_step = self.config.start_step
            step = 0
            while step < self.config.total_steps:
                if self.config.shutdown_check and self.config.shutdown_check():
                    logger.info("Shutdown requested, stopping training early")
                    break

                batch = replay.sample_batch_tensors(
                    self.config.batch_size,
                    num_actions=self.game_config.num_actions,
                    env_id=env_id,
                )
                if batch is None:
                    consecutive_skips += 1
                    if consecutive_skips % LOG_EVERY_N_WAITS == 1:
                        logger.warning(
                            f"Not enough data for batch (need {self.config.batch_size}), "
                            f"sleeping {self.config.wait_interval}s... "
                            f"(skip {consecutive_skips}"
                            f"/{self.config.max_consecutive_empty_batches or 'inf'})"
                        )
                    if (
                        self.config.max_consecutive_empty_batches > 0
                        and consecutive_skips
                        >= self.config.max_consecutive_empty_batches
                    ):
                        raise RuntimeError(
                            f"Replay buffer returned {consecutive_skips} consecutive "
                            f"empty batches. The buffer may be empty or corrupted. "
                            f"Increase --max-empty-batches or check data pipeline."
                        )
                    time.sleep(self.config.wait_interval)
                    continue

                step += 1
                global_step = start_step + step
                self.stats.step = global_step
                consecutive_skips = 0
                observations, policy_targets, value_targets = batch
                self.samples_seen += len(observations)

                step_start = time.time()
                metrics = self._train_step(observations, policy_targets, value_targets)
                step_duration = time.time() - step_start

                self.lr_scheduler.step()

                self._record_step_metrics(
                    step,
                    global_step,
                    metrics,
                    step_duration,
                    len(observations),
                    replay,
                    env_id,
                )
                self._handle_replay_cleanup(global_step, replay, env_id)
                self._handle_checkpoint_and_eval(step, global_step)

            # Final checkpoint
            final_global_step = start_step + self.config.total_steps
            self._save_checkpoint(final_global_step, is_final=True)
            self._write_stats()
        finally:
            replay.close()

        logger.info("Training complete")
        return self.stats

    def _train_step(
        self,
        observations: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
    ) -> dict[str, float]:
        """Perform a single training step.

        Args:
            observations: Game observations (batch, obs_size)
            policy_targets: MCTS policy distributions (batch, action_size)
            value_targets: Value targets (batch,)
        """
        self.network.train()
        self.optimizer.zero_grad()

        # Convert to tensors
        obs_t = torch.from_numpy(observations).to(self.device)
        policy_targets_t = torch.from_numpy(policy_targets).to(self.device)
        value_targets_t = torch.from_numpy(value_targets).to(self.device)

        # Extract legal mask from observations using game-specific offsets
        legal_mask = self.game_config.extract_legal_mask(obs_t)

        # Forward pass
        policy_logits, value_pred = self.network(obs_t)

        # Compute loss with soft policy targets
        loss, metrics = self.loss_fn(
            policy_logits, value_pred, policy_targets_t, value_targets_t, legal_mask
        )

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        if self.config.grad_clip_norm > 0:
            grad_norm = nn_utils.clip_grad_norm_(
                self.network.parameters(), self.config.grad_clip_norm
            )
            metrics["grad_norm"] = grad_norm.item()

        self.optimizer.step()

        return metrics

    def _evaluate_checkpoint(self, checkpoint_path: Path, step: int) -> None:
        """Run evaluation on a checkpoint and record results.

        Args:
            checkpoint_path: Path to the ONNX checkpoint to evaluate.
            step: Current training step for recording.
        """
        logger.info(
            f"Running evaluation at step {step} ({self.config.eval_games} games)..."
        )

        eval_start = time.time()
        try:
            model_policy = OnnxPolicy(str(checkpoint_path), temperature=0.0)
            random_policy = RandomPolicy()

            results = evaluate(
                player1=model_policy,
                player2=random_policy,
                env_id=self.config.env_id,
                config=self.game_config,
                num_games=self.config.eval_games,
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

            self.stats.append_eval(eval_stats)

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

    def _save_checkpoint(self, step: int, is_final: bool = False) -> Path:
        """Save model checkpoint with atomic write-then-rename.

        Saves both ONNX (for actor inference) and PyTorch (for training continuity).

        Args:
            step: Current training step.
            is_final: Whether this is the final checkpoint (unused, for future use).

        Returns:
            Path to the saved ONNX checkpoint.
        """
        model_dir = Path(self.config.model_dir)

        # Save ONNX checkpoint (for actor inference)
        checkpoint_path = save_onnx_checkpoint(
            network=self.network,
            obs_size=self.network.obs_size,
            step=step,
            model_dir=model_dir,
            device=self.device,
        )

        # Save PyTorch checkpoint (for training continuity across iterations)
        save_pytorch_checkpoint(
            network=self.network,
            optimizer=self.optimizer,
            step=step,
            model_dir=model_dir,
            scheduler=self.lr_scheduler,
        )

        # Track checkpoints for cleanup
        self.checkpoints.append(checkpoint_path)
        self.checkpoints = cleanup_old_checkpoints(
            self.checkpoints, self.config.max_checkpoints
        )
        self.latest_checkpoint = checkpoint_path

        # Clean up orphaned .onnx.data files from PyTorch exporter
        cleanup_temp_onnx_data(model_dir)

        return checkpoint_path

    def _write_stats(self) -> None:
        """Write stats.json for web polling (atomic write)."""
        write_stats(self.stats, self.config.stats_path)
