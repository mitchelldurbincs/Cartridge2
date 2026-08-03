"""Bounded off-policy orchestration owned by the DQN cartridge."""

from __future__ import annotations

import argparse
import logging
import math
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..central_config import get_config
from ..environment_catalog import get_environment
from ..runtime_profile import resolve_runtime_profile
from ..storage.publisher import create_checkpoint_publisher
from .dqn_config import DqnLearnerConfig

if TYPE_CHECKING:
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
    actor_binary: str | None
    eval_binary: str | None
    data_dir: str | None

    def __post_init__(self) -> None:
        for name in (
            "iterations",
            "episodes_per_iteration",
            "steps_per_iteration",
            "batch_size",
            "target_sync_interval",
            "hidden_size",
            "onnx_intra_threads",
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

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DqnLoopConfig":
        return cls(
            env_id=args.env_id,
            iterations=args.iterations,
            episodes_per_iteration=args.episodes_per_iteration,
            steps_per_iteration=args.steps_per_iteration,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gamma=args.gamma,
            target_sync_interval=args.target_sync_interval,
            hidden_size=args.hidden_size,
            grad_clip_norm=args.grad_clip,
            device=args.device,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed,
            onnx_intra_threads=args.onnx_intra_threads,
            evaluation_episodes=args.evaluation_episodes,
            actor_binary=args.actor_binary,
            eval_binary=args.eval_binary,
            data_dir=args.data_dir,
        )


def configure_dqn_loop_parser(parser: argparse.ArgumentParser) -> None:
    learner = DqnLearnerConfig()
    parser.add_argument("--env-id", default="counter")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--episodes-per-iteration", type=int, default=100)
    parser.add_argument("--steps-per-iteration", type=int, default=learner.total_steps)
    parser.add_argument("--batch-size", type=int, default=learner.batch_size)
    parser.add_argument("--learning-rate", type=float, default=learner.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=learner.weight_decay)
    parser.add_argument("--gamma", type=float, default=learner.gamma)
    parser.add_argument("--target-sync-interval", type=int, default=learner.target_sync_interval)
    parser.add_argument("--hidden-size", type=int, default=learner.hidden_size)
    parser.add_argument("--grad-clip", type=float, default=learner.grad_clip_norm)
    parser.add_argument("--device", default=learner.device)
    parser.add_argument("--epsilon-start", type=float, default=0.25)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--onnx-intra-threads", type=int, default=1)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--actor-binary")
    parser.add_argument("--eval-binary")
    parser.add_argument("--data-dir")
    parser.add_argument("--log-level", default="INFO")


class DqnLoop:
    def __init__(self, cartridge: "DqnV1", config: DqnLoopConfig):
        self.cartridge = cartridge
        self.config = config

    def run(self) -> int:
        environment = get_environment(self.config.env_id)
        self.cartridge.compatibility(environment).require_compatible()
        data_root = Path(self.config.data_dir or get_config().data_root)
        profile_dir = resolve_runtime_profile(
            self.cartridge.descriptor.id, environment.env_id
        ).data_dir(data_root)
        model_dir = profile_dir / "models"
        stats_path = profile_dir / "stats.json"
        data_dir = str(data_root)
        checkpoints = create_checkpoint_publisher(
            self.cartridge.artifact_contract(environment), model_dir
        )

        for iteration in range(1, self.config.iterations + 1):
            head = checkpoints.resolve_run_head()
            source_checkpoint_id = head.checkpoint_id if head is not None else None
            collection_scope_id = secrets.token_hex(32)
            epsilon = (
                1.0
                if source_checkpoint_id is None
                else max(
                    self.config.epsilon_end,
                    self.config.epsilon_start * self.config.epsilon_decay ** (iteration - 1),
                )
            )
            source = {
                "source_checkpoint_id": source_checkpoint_id,
                "source_root": source_checkpoint_id is None,
            }
            collect_args = argparse.Namespace(
                env_id=environment.env_id,
                episodes=self.config.episodes_per_iteration,
                collection_scope_id=collection_scope_id,
                epsilon=epsilon,
                seed=(self.config.seed + iteration - 1) % (1 << 64),
                onnx_intra_threads=self.config.onnx_intra_threads,
                actor_id=f"dqn-loop-{iteration}",
                episode_timeout_secs=30,
                actor_binary=self.config.actor_binary,
                data_dir=data_dir,
                log_level="INFO",
                **source,
            )
            if self.cartridge._run_collect(collect_args) != 0:
                return 1
            train_args = argparse.Namespace(
                env_id=environment.env_id,
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                steps=self.config.steps_per_iteration,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                gamma=self.config.gamma,
                target_sync_interval=self.config.target_sync_interval,
                hidden_size=self.config.hidden_size,
                grad_clip=self.config.grad_clip_norm,
                device=self.config.device,
                collection_scope_id=collection_scope_id,
                log_level="INFO",
                **source,
            )
            if self.cartridge._run_train(train_args) != 0:
                return 1
            advanced = checkpoints.resolve_run_head()
            if advanced is None or advanced.checkpoint_id == source_checkpoint_id:
                raise RuntimeError("DQN learner did not advance the authoritative RunHead")
            if self.config.evaluation_episodes > 0:
                evaluate_args = argparse.Namespace(
                    env_id=environment.env_id,
                    episodes=self.config.evaluation_episodes,
                    seed=(self.config.seed + iteration - 1) % (1 << 64),
                    checkpoint_id=advanced.checkpoint_id,
                    random=False,
                    model_dir=str(model_dir),
                    eval_binary=self.config.eval_binary,
                    onnx_intra_threads=self.config.onnx_intra_threads,
                )
                if self.cartridge._run_evaluate(evaluate_args) != 0:
                    return 1
            logger.info(
                "DQN iteration %s/%s committed checkpoint %s",
                iteration,
                self.config.iterations,
                advanced.checkpoint_id,
            )
        return 0
