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

from . import checkpoint_runner, replay_setup, step_metrics
from . import metrics as prom_metrics
from .backoff import LOG_EVERY_N_WAITS, WaitTimeout
from .checkpoint import load_pytorch_checkpoint
from .config import TrainerConfig
from .game_config import get_config
from .lr_scheduler import LRConfig, WarmupCosineScheduler
from .network import AlphaZeroLoss, create_network
from .stats import EvalStats, TrainerStats, load_stats
from .storage import create_replay_buffer

# Re-export for convenience (used by tests and external callers)
__all__ = ["Trainer", "TrainerConfig", "EvalStats", "TrainerStats", "WaitTimeout"]

logger = logging.getLogger(__name__)


class Trainer:
    """AlphaZero-style trainer for game agents."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        resolved_device = config.resolve_device()
        self.device = torch.device(resolved_device)

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

    def _discover_existing_checkpoints(self) -> list[Path]:
        return checkpoint_runner.discover_existing_checkpoints(self)

    def _wait_with_backoff(
        self, condition_fn, description: str, check_interval: float | None = None
    ) -> None:
        replay_setup.wait_with_backoff(self, condition_fn, description, check_interval)

    def _create_replay_buffer(self):
        """Create PostgreSQL replay buffer using the factory.

        Returns:
            PostgresReplayBuffer instance.
        """
        logger.info("Connecting to PostgreSQL replay buffer...")
        return create_replay_buffer()

    def _setup_replay(self, replay, env_id: str) -> None:
        replay_setup.setup_replay(self, replay, env_id)

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
        step_metrics.record_step_metrics(
            self,
            step,
            global_step,
            metrics,
            step_duration,
            batch_size,
            replay,
            env_id,
        )

    def _handle_replay_cleanup(self, global_step: int, replay, env_id: str) -> None:
        replay_setup.handle_replay_cleanup(self, global_step, replay, env_id)

    def _handle_checkpoint_and_eval(self, step: int, global_step: int) -> None:
        checkpoint_runner.handle_checkpoint_and_eval(self, step, global_step)

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
            self._save_checkpoint(final_global_step)
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
        checkpoint_runner.evaluate_checkpoint(self, checkpoint_path, step)

    def _save_checkpoint(self, step: int) -> Path:
        return checkpoint_runner.save_checkpoint(self, step)

    def _write_stats(self) -> None:
        step_metrics.write_stats(self)
