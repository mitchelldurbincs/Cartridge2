"""Bounded off-policy orchestration owned by the DQN cartridge."""

from __future__ import annotations

import logging
import math
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..environment_catalog import get_environment
from ..runtime_profile import resolve_runtime_profile
from ..storage import ReplayProfile, ReplaySelection
from ..storage.publisher import create_checkpoint_publisher
from .dqn_application import format_evaluation_result
from .dqn_requests import DqnCollectRequest, DqnEvaluateRequest, DqnTrainRequest

if TYPE_CHECKING:
    from ..environment_catalog import EnvironmentDescriptor
    from .dqn_v1 import DqnV1

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DqnLoopConfig:
    env_id: str
    iterations: int
    episodes_per_iteration: int
    steps_per_iteration: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gamma: float
    target_sync_interval: int
    hidden_size: int
    grad_clip_norm: float
    device: str
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    seed: int
    onnx_intra_threads: int
    evaluation_episodes: int
    episode_timeout_secs: int
    actor_binary: Path | None
    eval_binary: Path | None
    data_root: Path
    log_level: str

    def __post_init__(self) -> None:
        for name in (
            "iterations",
            "episodes_per_iteration",
            "steps_per_iteration",
            "batch_size",
            "target_sync_interval",
            "hidden_size",
            "onnx_intra_threads",
            "episode_timeout_secs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.evaluation_episodes, bool)
            or not isinstance(self.evaluation_episodes, int)
            or self.evaluation_episodes < 0
        ):
            raise ValueError("evaluation_episodes must be a nonnegative integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 1 << 64
        ):
            raise ValueError("seed must be a u64")
        for name in ("epsilon_start", "epsilon_end"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("epsilon_end cannot exceed epsilon_start")
        if not math.isfinite(self.epsilon_decay) or not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be finite and in (0, 1]")
        if not isinstance(self.data_root, Path):
            raise TypeError("data_root must be a pathlib.Path")
        if not isinstance(self.log_level, str) or not self.log_level.strip():
            raise ValueError("log_level must be non-empty")


class DqnLoop:
    def __init__(self, cartridge: "DqnV1", config: DqnLoopConfig):
        self.cartridge = cartridge
        self.config = config

    def run(self) -> int:
        environment = get_environment(self.config.env_id)
        self.cartridge.compatibility(environment).require_compatible()
        profile_dir = resolve_runtime_profile(
            self.cartridge.descriptor.id, environment.env_id
        ).data_dir(self.config.data_root)
        model_dir = profile_dir / "models"
        checkpoints = create_checkpoint_publisher(
            self.cartridge.artifact_contract(environment), model_dir
        )

        for iteration in range(1, self.config.iterations + 1):
            source_checkpoint_id = self._run_iteration(
                environment, profile_dir, checkpoints, iteration
            )
            logger.info(
                "DQN iteration %s/%s committed checkpoint %s",
                iteration,
                self.config.iterations,
                source_checkpoint_id,
            )
        return 0

    def _run_iteration(self, environment, profile_dir: Path, checkpoints, iteration: int) -> str:
        head = checkpoints.resolve_run_head()
        source_checkpoint_id = head.checkpoint_id if head is not None else None
        selection = self._selection(environment, source_checkpoint_id)
        collect_request = self._collect_request(environment, selection, iteration)
        if self.cartridge.collect(collect_request) != 0:
            raise RuntimeError("DQN collector exited unsuccessfully")
        self.cartridge.train(self._train_request(environment, profile_dir, selection))
        advanced = checkpoints.resolve_run_head()
        if advanced is None or advanced.checkpoint_id == source_checkpoint_id:
            raise RuntimeError("DQN learner did not advance the authoritative RunHead")
        if self.config.evaluation_episodes > 0:
            result = self.cartridge.evaluate(
                self._evaluate_request(environment, profile_dir, advanced.checkpoint_id, iteration)
            )
            print(format_evaluation_result(result))
        return advanced.checkpoint_id

    def _selection(
        self, environment: "EnvironmentDescriptor", source_checkpoint_id: str | None
    ) -> ReplaySelection:
        return ReplaySelection(
            profile=ReplayProfile(
                env_id=environment.env_id,
                env_contract_version=environment.contract_version,
                algorithm_id=self.cartridge.descriptor.id,
                experience_schema=self.cartridge.descriptor.components.experience_schema,
            ),
            collection_scope_id=secrets.token_hex(32),
            source_checkpoint_id=source_checkpoint_id,
        )

    def _collect_request(
        self,
        environment: "EnvironmentDescriptor",
        selection: ReplaySelection,
        iteration: int,
    ) -> DqnCollectRequest:
        source_checkpoint_id = selection.source_checkpoint_id
        epsilon = (
            1.0
            if source_checkpoint_id is None
            else max(
                self.config.epsilon_end,
                self.config.epsilon_start * self.config.epsilon_decay ** (iteration - 1),
            )
        )
        return DqnCollectRequest(
            env_id=environment.env_id,
            episodes=self.config.episodes_per_iteration,
            collection_scope_id=selection.collection_scope_id,
            source_checkpoint_id=source_checkpoint_id,
            epsilon=epsilon,
            seed=(self.config.seed + iteration - 1) % (1 << 64),
            onnx_intra_threads=self.config.onnx_intra_threads,
            actor_id=f"dqn-loop-{iteration}",
            episode_timeout_secs=self.config.episode_timeout_secs,
            actor_binary=self.config.actor_binary,
            data_root=self.config.data_root,
            log_level=self.config.log_level,
        )

    def _train_request(
        self,
        environment: "EnvironmentDescriptor",
        profile_dir: Path,
        selection: ReplaySelection,
    ) -> DqnTrainRequest:
        return DqnTrainRequest(
            env_id=environment.env_id,
            replay_selection=selection,
            model_dir=profile_dir / "models",
            stats_path=profile_dir / "stats.json",
            total_steps=self.config.steps_per_iteration,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            gamma=self.config.gamma,
            target_sync_interval=self.config.target_sync_interval,
            hidden_size=self.config.hidden_size,
            grad_clip_norm=self.config.grad_clip_norm,
            device=self.config.device,
            log_level=self.config.log_level,
        )

    def _evaluate_request(
        self,
        environment: "EnvironmentDescriptor",
        profile_dir: Path,
        checkpoint_id: str,
        iteration: int,
    ) -> DqnEvaluateRequest:
        return DqnEvaluateRequest(
            env_id=environment.env_id,
            episodes=self.config.evaluation_episodes,
            seed=(self.config.seed + iteration - 1) % (1 << 64),
            checkpoint_id=checkpoint_id,
            model_dir=profile_dir / "models",
            eval_binary=self.config.eval_binary,
            onnx_intra_threads=self.config.onnx_intra_threads,
        )
