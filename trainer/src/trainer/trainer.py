"""Training loop with ONNX export and stats tracking.

This module provides the main training loop that:
1. Samples opaque records from the profile-bound PostgreSQL replay store
2. Trains the AlphaZero-style network
3. Stages validated content-addressed ONNX and learner-state checkpoints
4. Commits an embedded stats snapshot through RunHead and rebuilds stats.json

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
from .algorithms.alphazero_board_v1 import (
    ALGORITHM_ID,
    DESCRIPTOR,
    decode_replay_batch,
    get_game_config,
)
from .checkpoint import learner_config_sha256, restore_learner_state
from .config import AlphaZeroLearnerConfig
from .environment_catalog import get_environment
from .lr_scheduler import LRConfig, WarmupCosineScheduler
from .network import AlphaZeroLoss, create_network
from .stats import (
    PreparedStatsSnapshotV2,
    TrainerStats,
    decode_stats_snapshot,
    prepare_stats_snapshot,
    write_ephemeral_stats_projection,
    write_stats_projection,
)
from .storage import ReplayProfile, ReplaySelection, create_replay_store
from .storage.evaluation import create_evaluation_repository
from .storage.publisher import (
    ArtifactValidationError,
    CheckpointRef,
    OnnxArtifactContract,
    create_checkpoint_publisher,
)
from .storage.run_commit import RunCommitRepository, RunCommitV1

__all__ = ["AlphaZeroLearner"]

logger = logging.getLogger(__name__)


class AlphaZeroLearner:
    """AlphaZero-style trainer for game agents."""

    def __init__(self, config: AlphaZeroLearnerConfig):
        if config.total_steps <= 0:
            raise ValueError("total_steps must be greater than zero")
        if not isinstance(config.defer_run_commit, bool):
            raise TypeError("defer_run_commit must be a boolean")
        self.config = config
        resolved_device = config.resolve_device()
        self.device = torch.device(resolved_device)

        # Get game configuration
        environment = get_environment(config.env_id)
        self.game_config = get_game_config(config.env_id)
        self.replay_profile = ReplayProfile(
            env_id=config.env_id,
            env_contract_version=environment.contract_version,
            algorithm_id=ALGORITHM_ID,
            experience_schema=DESCRIPTOR.components.experience_schema,
        )
        if not isinstance(config.replay_selection, ReplaySelection):
            raise ValueError(
                "Training requires an explicit ReplaySelection with collection "
                "scope and source checkpoint"
            )
        if config.replay_selection.profile != self.replay_profile:
            raise ArtifactValidationError(
                "Learner replay selection profile does not match its cartridge"
            )
        self.replay_selection = config.replay_selection
        self.model_contract = DESCRIPTOR.components.model_contract
        self.artifact_contract = OnnxArtifactContract(
            algorithm_id=self.replay_profile.algorithm_id,
            env_id=config.env_id,
            env_contract_version=environment.contract_version,
            model_artifact_schema_version=DESCRIPTOR.model_artifact_schema_version,
            model_contract=self.model_contract,
            obs_size=self.game_config.obs_size,
            num_actions=self.game_config.num_actions,
        )
        self.config_sha256 = learner_config_sha256(config)

        # Create model directory
        Path(config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(config.stats_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize network
        self.network = create_network(config.env_id, config=self.game_config)
        self.network.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.checkpoint_publisher = create_checkpoint_publisher(
            self.artifact_contract,
            Path(config.model_dir),
        )
        self.evaluation_repository = create_evaluation_repository(
            self.checkpoint_publisher
        )
        self.run_commit_repository = RunCommitRepository(
            self.checkpoint_publisher,
            self.evaluation_repository,
        )

        # Resolve one indivisible checkpoint/stats state through RunHeadV2.
        self._checkpoint_loaded = False
        self._loaded_step: int | None = None
        self._loaded_scheduler_state: dict | None = None
        self.start_step = config.start_step
        run_head = self.checkpoint_publisher.resolve_run_head()
        self.current_run_commit: RunCommitV1 | None = None
        self.current_run_commit_id: str | None = None
        checkpoint_ref: CheckpointRef | None = None
        stats = TrainerStats(env_id=self.artifact_contract.env_id)
        if run_head is not None:
            run_chain = self.run_commit_repository.resolve_chain(run_head.run_commit_id)
            if not run_chain or run_chain[-1].run_commit_id != run_head.run_commit_id:
                raise ArtifactValidationError(
                    "RunHead does not resolve to a complete RunCommit lineage"
                )
            run_commit = run_chain[-1].commit
            if run_commit.checkpoint_id != run_head.checkpoint_id:
                raise ArtifactValidationError(
                    "RunHead checkpoint does not match its RunCommit"
                )
            if run_commit.profile != self.artifact_contract.profile:
                raise ArtifactValidationError("RunCommit learner profile mismatch")
            if run_commit.config_sha256 != self.config_sha256:
                raise ArtifactValidationError("RunCommit learner config mismatch")
            if run_commit.run_recipe is not None and not config.defer_run_commit:
                raise ArtifactValidationError(
                    "Standalone learner cannot continue a recipe-owned run; "
                    "select a new profile data root"
                )
            checkpoint_ref = self.checkpoint_publisher.resolve_checkpoint(
                run_head.checkpoint_id,
                expected_config_sha256=self.config_sha256,
            )
            # The RunCommit bytes remain immutable authority. Training receives
            # a separate mutable projection so its live counters cannot become
            # an unauthenticated view of the selected snapshot.
            stats = decode_stats_snapshot(
                run_commit.stats_snapshot.data,
                expected_stats_id=run_commit.stats_id,
                expected_binding=run_commit.stats_snapshot.binding,
            ).stats
            self.current_run_commit = run_commit
            self.current_run_commit_id = run_head.run_commit_id

        expected_source_checkpoint_id = (
            run_head.checkpoint_id if run_head is not None else None
        )
        if self.replay_selection.source_checkpoint_id != expected_source_checkpoint_id:
            raise ArtifactValidationError(
                "Learner replay source checkpoint does not match RunHead"
            )

        self.last_checkpoint_ref: CheckpointRef | None = checkpoint_ref
        self.last_prepared_stats: PreparedStatsSnapshotV2 | None = None
        self.parent_checkpoint_id: str | None = (
            checkpoint_ref.checkpoint_id if checkpoint_ref is not None else None
        )
        if checkpoint_ref is not None:
            self._loaded_scheduler_state = restore_learner_state(
                self.network,
                self.optimizer,
                checkpoint_ref.learner_state_path,
                self.device,
                checkpoint_ref.manifest,
                self.artifact_contract,
            )
            loaded_step = checkpoint_ref.manifest.step
            self._loaded_step = loaded_step
            self._checkpoint_loaded = True
            if config.start_step not in (0, loaded_step):
                raise ValueError(
                    "Configured start_step does not match the checkpoint: "
                    f"got {config.start_step}, checkpoint is step {loaded_step}"
                )
            self.start_step = loaded_step
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
            self.lr_scheduler.load_state_dict(self._loaded_scheduler_state)

        # Initialize loss function
        self.loss_fn = AlphaZeroLoss(
            value_weight=config.value_loss_weight,
            policy_weight=config.policy_loss_weight,
        )

        self.stats = stats
        self.stats.total_steps = self.start_step + config.total_steps
        self.samples_seen = self.stats.samples_seen
        # stats.json is never authority. Rebuild it even for a fresh run so a
        # stale projection cannot survive after an absent RunHeadV2.
        write_ephemeral_stats_projection(self.stats, config.stats_path)

        # Replay maintenance
        self._replay_cleanup_every = (
            config.replay_cleanup_interval
            if config.replay_cleanup_interval > 0
            else config.stats_interval
        )

        # Rolling window for averaging (last 100 steps)
        self._recent_losses: list[dict[str, float]] = []
        self._rolling_window = 100

        # Replay-count caching (avoid expensive count() calls every step)
        self._replay_record_count_cache = self.stats.replay_record_count
        self._replay_record_count_update_interval: int = 100

    def _wait_with_backoff(
        self, condition_fn, description: str, check_interval: float | None = None
    ) -> None:
        replay_setup.wait_with_backoff(self, condition_fn, description, check_interval)

    def _create_replay_store(self):
        """Create the profile-bound PostgreSQL replay store.

        Returns:
            PostgresReplayStore instance.
        """
        logger.info("Connecting to PostgreSQL replay store...")
        return create_replay_store(self.replay_selection)

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

    def _handle_checkpoint(self, step: int, global_step: int) -> None:
        checkpoint_runner.handle_checkpoint(self, step, global_step)

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

        replay = self._create_replay_store()
        try:
            env_id = self.config.env_id
            self._setup_replay(replay, env_id)

            # Sampling fills with replacement as needed, so one usable replay
            # record is sufficient for every positive minibatch size.
            start_step = self.start_step
            step = 0
            last_global_step = start_step
            while step < self.config.total_steps:
                if self.config.shutdown_check and self.config.shutdown_check():
                    logger.info("Shutdown requested, stopping training early")
                    break

                records = replay.sample(self.config.batch_size)
                if len(records) != self.config.batch_size:
                    raise RuntimeError(
                        "ReplayStore.sample contract violation: "
                        f"requested {self.config.batch_size} records, "
                        f"received {len(records)}"
                    )
                batch = decode_replay_batch(
                    records,
                    selection=self.replay_selection,
                    obs_size=self.game_config.obs_size,
                    num_actions=self.game_config.num_actions,
                )

                step += 1
                global_step = start_step + step
                last_global_step = global_step
                self.stats.step = global_step
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
                self._handle_checkpoint(step, global_step)

            # Final checkpoint
            final_checkpoint = self._save_checkpoint(last_global_step)
            self.stats.step = last_global_step
            self.stats.last_checkpoint = final_checkpoint.checkpoint_id
            if self.config.defer_run_commit:
                self._prepare_run_state(final_checkpoint)
            else:
                self._publish_run_state(final_checkpoint)
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

    def _save_checkpoint(self, step: int) -> CheckpointRef:
        return checkpoint_runner.save_checkpoint(self, step)

    def _prepare_run_state(self, checkpoint: CheckpointRef) -> PreparedStatsSnapshotV2:
        """Bind current in-memory stats to one staged checkpoint."""
        self.stats.last_checkpoint = checkpoint.checkpoint_id
        prepared = prepare_stats_snapshot(self.stats, checkpoint)
        self.last_checkpoint_ref = checkpoint
        self.last_prepared_stats = prepared
        return prepared

    def _publish_run_state(self, checkpoint: CheckpointRef) -> RunCommitV1:
        """Publish one standalone RunCommit and atomically advance RunHeadV2."""
        if self.config.defer_run_commit:
            raise RuntimeError(
                "Deferred loop learners cannot commit RunHead from inside training"
            )
        prepared = self._prepare_run_state(checkpoint)
        parent = self.current_run_commit
        parent_id = self.current_run_commit_id
        if (parent is None) != (parent_id is None):
            raise ArtifactValidationError("In-memory RunCommit identity is incomplete")

        # Overlapping stats/checkpoint/evaluation cadences can call this twice
        # without changing any state. Do not manufacture a no-op child commit.
        if (
            parent is not None
            and parent.checkpoint_id == checkpoint.checkpoint_id
            and parent.stats_id == prepared.stats_id
        ):
            self.parent_checkpoint_id = checkpoint.checkpoint_id
            write_stats_projection(prepared, self.config.stats_path)
            return parent

        commit = RunCommitV1(
            profile=self.artifact_contract.profile,
            config_sha256=self.config_sha256,
            parent_run_commit_id=parent_id,
            checkpoint_id=checkpoint.checkpoint_id,
            stats_snapshot=prepared,
            champion=parent.champion if parent is not None else None,
            evaluation_head_id=(
                parent.evaluation_head_id if parent is not None else None
            ),
            orchestration=None,
        )
        reference = self.run_commit_repository.publish(commit)
        head = self.checkpoint_publisher.commit_run_head(
            checkpoint_id=checkpoint.checkpoint_id,
            run_commit_id=reference.run_commit_id,
            expected_run_commit_id=parent_id,
        )
        if (
            head.checkpoint_id != checkpoint.checkpoint_id
            or head.run_commit_id != reference.run_commit_id
        ):
            raise ArtifactValidationError(
                "RunHead advanced to a descendant while committing this learner; "
                "stop and reload authoritative state"
            )

        # Update lineage before rebuilding the derived projection. If the
        # projection write fails, retry/resume still follows the committed head.
        self.current_run_commit = commit
        self.current_run_commit_id = reference.run_commit_id
        self.parent_checkpoint_id = checkpoint.checkpoint_id
        write_stats_projection(prepared, self.config.stats_path)
        return commit
