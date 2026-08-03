"""Python bindings owned by the single-agent discrete DQN cartridge."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..environment_catalog import (
    AlgorithmDescriptor,
    CompatibilityReport,
    EnvironmentDescriptor,
    get_algorithm_descriptor,
    get_environment,
)
from ..runtime_profile import resolve_runtime_profile
from ..storage.base import ReplayRecord, ReplaySelection
from ..storage.publisher import OnnxArtifactContract, OnnxTensorSpec
from .base import AlgorithmCommand
from .dqn_config import DqnLearnerConfig

ALGORITHM_ID = "dqn_v1"
DESCRIPTOR = get_algorithm_descriptor(ALGORITHM_ID)

EXPECTED_COMPONENTS = {
    "collector": "dqn_epsilon_greedy_v1",
    "learner": "dqn_q_learning_v1",
    "orchestration": "off_policy_dqn_v1",
    "experience_schema": "dqn_transition_v1",
    "model_contract": "onnx_q_values_v1",
    "evaluation_suite": "single_agent_return_v1",
    "serving": "dqn_greedy_v1",
}


@dataclass(frozen=True)
class DqnReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_availability: np.ndarray


@dataclass(frozen=True)
class DqnEvaluationResults:
    env_id: str
    player_name: str
    episodes_played: int
    terminated_episodes: int
    truncated_episodes: int
    mean_return: float
    min_return: float
    max_return: float
    avg_episode_length: float

    @classmethod
    def from_json(cls, value: object) -> "DqnEvaluationResults":
        fields = {
            "env_id",
            "player_name",
            "episodes_played",
            "terminated_episodes",
            "truncated_episodes",
            "mean_return",
            "min_return",
            "max_return",
            "avg_episode_length",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("DQN evaluation result fields do not match the exact contract")
        result = cls(**value)
        if not result.env_id or not result.player_name:
            raise ValueError("DQN evaluation identity fields must be non-empty")
        for name in ("episodes_played", "terminated_episodes", "truncated_episodes"):
            item = getattr(result, name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"DQN evaluation {name} must be a nonnegative integer")
        if result.episodes_played <= 0:
            raise ValueError("DQN evaluation must contain at least one episode")
        if result.terminated_episodes + result.truncated_episodes != result.episodes_played:
            raise ValueError("DQN evaluation completion counts do not partition its episodes")
        for name in ("mean_return", "min_return", "max_return", "avg_episode_length"):
            item = getattr(result, name)
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"DQN evaluation {name} must be finite")
            if not math.isfinite(float(item)):
                raise ValueError(f"DQN evaluation {name} must be finite")
        if not result.min_return <= result.mean_return <= result.max_return:
            raise ValueError("DQN evaluation return aggregates are inconsistent")
        if result.avg_episode_length <= 0:
            raise ValueError("DQN evaluation average episode length must be positive")
        return result


def decode_replay_batch(
    records: list[ReplayRecord],
    *,
    selection: ReplaySelection,
    obs_size: int,
    num_actions: int,
) -> DqnReplayBatch:
    """Decode exact ``dqn_transition_v1`` opaque replay records."""
    if selection.profile.algorithm_id != ALGORITHM_ID:
        raise ValueError("DQN replay selection uses the wrong algorithm")
    if selection.profile.experience_schema != DESCRIPTOR.components.experience_schema:
        raise ValueError("DQN replay selection uses the wrong experience schema")
    expected_bytes = obs_size * 8 + 10 + num_actions
    observations = np.empty((len(records), obs_size), dtype=np.float32)
    actions = np.empty(len(records), dtype=np.int64)
    rewards = np.empty(len(records), dtype=np.float32)
    next_observations = np.empty((len(records), obs_size), dtype=np.float32)
    terminated = np.empty(len(records), dtype=np.bool_)
    truncated = np.empty(len(records), dtype=np.bool_)
    next_availability = np.empty((len(records), num_actions), dtype=np.bool_)
    observation_bytes = obs_size * 4

    for index, record in enumerate(records):
        if not selection.matches(record):
            raise ValueError(f"DQN replay record {record.id!r} crosses its selection fence")
        payload = record.payload
        if len(payload) != expected_bytes:
            raise ValueError(
                f"DQN replay record {record.id!r} has {len(payload)} bytes, "
                f"expected {expected_bytes}"
            )
        cursor = 0
        observations[index] = np.frombuffer(payload, dtype="<f4", count=obs_size, offset=cursor)
        cursor += observation_bytes
        action = int.from_bytes(payload[cursor : cursor + 4], "little")
        cursor += 4
        if action >= num_actions:
            raise ValueError(f"DQN replay record {record.id!r} has invalid action {action}")
        actions[index] = action
        reward = np.frombuffer(payload, dtype="<f4", count=1, offset=cursor)[0]
        cursor += 4
        if not np.isfinite(reward):
            raise ValueError(f"DQN replay record {record.id!r} has non-finite reward")
        rewards[index] = reward
        next_observations[index] = np.frombuffer(
            payload, dtype="<f4", count=obs_size, offset=cursor
        )
        cursor += observation_bytes
        flags = payload[cursor : cursor + 2]
        cursor += 2
        if any(flag not in (0, 1) for flag in flags) or flags == b"\x01\x01":
            raise ValueError(f"DQN replay record {record.id!r} has invalid completion flags")
        terminated[index] = bool(flags[0])
        truncated[index] = bool(flags[1])
        availability = np.frombuffer(payload, dtype=np.uint8, offset=cursor)
        if np.any(availability > 1):
            raise ValueError(f"DQN replay record {record.id!r} has invalid availability")
        next_availability[index] = availability.astype(np.bool_)
        done = bool(flags[0] or flags[1])
        if done == bool(availability.any()):
            expectation = "empty" if done else "non-empty"
            raise ValueError(
                f"DQN replay record {record.id!r} requires {expectation} next availability"
            )
        if (
            not np.isfinite(observations[index]).all()
            or not np.isfinite(next_observations[index]).all()
        ):
            raise ValueError(f"DQN replay record {record.id!r} has non-finite observation")

    return DqnReplayBatch(
        observations,
        actions,
        rewards,
        next_observations,
        terminated,
        truncated,
        next_availability,
    )


class DqnV1:
    """Every Python binding owned by ``dqn_v1``."""

    descriptor: AlgorithmDescriptor = DESCRIPTOR

    def __init__(self) -> None:
        mismatches = [
            f"{name}={getattr(self.descriptor.components, name)!r} (expected {expected!r})"
            for name, expected in EXPECTED_COMPONENTS.items()
            if getattr(self.descriptor.components, name) != expected
        ]
        if mismatches:
            raise RuntimeError(
                f"Python implementation does not match algorithm '{ALGORITHM_ID}': "
                + "; ".join(mismatches)
            )

    def commands(self) -> tuple[AlgorithmCommand, ...]:
        return (
            AlgorithmCommand(
                name="collect",
                help="Collect epsilon-greedy DQN transitions",
                configure_parser=self._configure_collect_parser,
                run=self._run_collect,
            ),
            AlgorithmCommand(
                name="train",
                help="Train a Q-network from an exact DQN replay collection",
                configure_parser=self._configure_train_parser,
                run=self._run_train,
            ),
            AlgorithmCommand(
                name="evaluate",
                help="Evaluate a greedy Q-policy by single-agent episode return",
                configure_parser=self._configure_evaluate_parser,
                run=self._run_evaluate,
            ),
            AlgorithmCommand(
                name="loop",
                help="Run bounded off-policy collect/train/evaluate iterations",
                configure_parser=self._configure_loop_parser,
                run=self._run_loop,
            ),
        )

    def compatibility(self, environment: EnvironmentDescriptor) -> CompatibilityReport:
        return environment.compatibility(self.descriptor.id)

    def artifact_contract(self, environment: EnvironmentDescriptor) -> OnnxArtifactContract:
        self.compatibility(environment).require_compatible()
        tensor = environment.capabilities.encoding.observation_tensor
        agents = environment.capabilities.agents.agents
        if tensor is None or tensor.fixed_elements is None:
            raise ValueError(
                f"Environment '{environment.env_id}' has no fixed DQN observation tensor"
            )
        if len(agents) != 1 or agents[0].action_space.discrete_size is None:
            raise ValueError(
                f"Environment '{environment.env_id}' has no single-agent discrete action space"
            )
        action_count = agents[0].action_space.discrete_size
        return OnnxArtifactContract(
            algorithm_id=self.descriptor.id,
            env_id=environment.env_id,
            env_contract_version=environment.contract_version,
            model_artifact_schema_version=(self.descriptor.model_artifact_schema_version),
            model_contract=self.descriptor.components.model_contract,
            inputs=(
                OnnxTensorSpec(
                    "observation",
                    "float32",
                    ("batch_size", tensor.fixed_elements),
                ),
            ),
            outputs=(
                OnnxTensorSpec(
                    "q_values",
                    "float32",
                    ("batch_size", action_count),
                ),
            ),
        )

    def build_learner(self, config: DqnLearnerConfig):
        from .dqn_learner import DqnLearner

        return DqnLearner(config)

    def build_collector_runner(self, config: Any, shutdown_check: Callable[[], bool] | None = None):
        from ..orchestrator.actor_runner import ActorRunner

        return ActorRunner(
            config,
            collector_config_builder=self.collector_config,
            shutdown_check=shutdown_check,
        )

    def collector_config(self, config: Any, _search_budget: int) -> dict:
        epsilon = getattr(config, "epsilon", 1.0)
        seed = getattr(config, "seed", 0)
        onnx_threads = getattr(config, "actor_onnx_intra_threads", 1)
        return {
            "schema_version": 1,
            "epsilon": epsilon,
            "seed": seed,
            "onnx_intra_threads": onnx_threads,
        }

    @staticmethod
    def _configure_train_parser(parser: argparse.ArgumentParser) -> None:
        DqnLearnerConfig.configure_parser(parser)

    @staticmethod
    def _configure_collect_parser(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--env-id", default="counter")
        parser.add_argument("--episodes", type=int, required=True)
        parser.add_argument("--collection-scope-id", required=True)
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--source-checkpoint-id")
        source.add_argument("--source-root", action="store_true")
        parser.add_argument("--epsilon", type=float, default=1.0)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--onnx-intra-threads", type=int, default=1)
        parser.add_argument("--actor-id", default="dqn-collector")
        parser.add_argument("--episode-timeout-secs", type=int, default=30)
        parser.add_argument("--actor-binary")
        parser.add_argument("--data-dir")
        parser.add_argument("--log-level", default="INFO")

    @staticmethod
    def _configure_evaluate_parser(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--env-id", default="counter")
        parser.add_argument("--episodes", type=int, default=100)
        parser.add_argument("--seed", type=int, default=42)
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--checkpoint-id")
        source.add_argument("--random", action="store_true")
        parser.add_argument("--model-dir")
        parser.add_argument("--eval-binary")
        parser.add_argument("--onnx-intra-threads", type=int, default=1)

    @staticmethod
    def _configure_loop_parser(parser: argparse.ArgumentParser) -> None:
        from .dqn_loop import configure_dqn_loop_parser

        configure_dqn_loop_parser(parser)

    def _run_collect(self, args: argparse.Namespace) -> int:
        from ..central_config import get_config
        from ..orchestrator.actor_runner import _BINARY_CANDIDATES

        try:
            environment = get_environment(args.env_id)
            self.compatibility(environment).require_compatible()
            if args.episodes <= 0:
                raise ValueError("episodes must be positive")
            if not 0.0 <= args.epsilon <= 1.0:
                raise ValueError("epsilon must be in [0, 1]")
            if args.seed < 0 or args.seed >= 1 << 64:
                raise ValueError("seed must be a u64")
            candidates = []
            if args.actor_binary:
                candidates.append(args.actor_binary)
            if os.environ.get("ACTOR_BINARY"):
                candidates.append(os.environ["ACTOR_BINARY"])
            candidates.extend(str(path) for path in _BINARY_CANDIDATES)
            actor_binary = next(
                (candidate for candidate in candidates if Path(candidate).is_file()),
                None,
            )
            if actor_binary is None:
                raise ValueError("actor binary not found; build it or pass --actor-binary")
            collector_config = {
                "schema_version": 1,
                "epsilon": args.epsilon,
                "seed": args.seed,
                "onnx_intra_threads": args.onnx_intra_threads,
            }
            command = [
                actor_binary,
                "--algorithm",
                self.descriptor.id,
                "--env-id",
                environment.env_id,
                "--max-episodes",
                str(args.episodes),
                "--collection-scope-id",
                args.collection_scope_id,
                "--collector-config",
                json.dumps(
                    collector_config,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                "--actor-id",
                args.actor_id,
                "--episode-timeout-secs",
                str(args.episode_timeout_secs),
                "--data-dir",
                args.data_dir or str(get_config().data_root),
                "--log-level",
                args.log_level.lower(),
            ]
            if args.source_checkpoint_id:
                command.extend(
                    ["--source-checkpoint-id", args.source_checkpoint_id]
                )
            return subprocess.run(command, check=False).returncode
        except Exception:
            import logging

            logging.getLogger(__name__).exception("DQN collection failed")
            return 1

    def _run_train(self, args: argparse.Namespace) -> int:
        from ..central_config import get_config
        from ..storage import ReplayProfile

        try:
            environment = get_environment(args.env_id)
            self.compatibility(environment).require_compatible()
            profile_dir = resolve_runtime_profile(self.descriptor.id, environment.env_id).data_dir(
                get_config().data_root
            )
            if args.model_dir is None:
                args.model_dir = str(profile_dir / "models")
            if args.stats_path is None:
                args.stats_path = str(profile_dir / "stats.json")
            config = DqnLearnerConfig.from_args(args)
            config.replay_selection = ReplaySelection(
                profile=ReplayProfile(
                    env_id=environment.env_id,
                    env_contract_version=environment.contract_version,
                    algorithm_id=self.descriptor.id,
                    experience_schema=self.descriptor.components.experience_schema,
                ),
                collection_scope_id=args.collection_scope_id,
                source_checkpoint_id=args.source_checkpoint_id,
            )
            self.build_learner(config).train()
            return 0
        except Exception:
            import logging

            logging.getLogger(__name__).exception("DQN training failed")
            return 1

    def _run_evaluate(self, args: argparse.Namespace) -> int:
        from ..central_config import get_config
        from ..evaluator import _BINARY_CANDIDATES, EVAL_BINARY_ENV
        from ..storage.publisher import create_checkpoint_publisher

        try:
            environment = get_environment(args.env_id)
            self.compatibility(environment).require_compatible()
            if args.episodes <= 0 or args.episodes >= 1 << 32:
                raise ValueError("episodes must be a positive u32")
            if args.seed < 0 or args.seed >= 1 << 64:
                raise ValueError("seed must be a u64")
            if args.seed > (1 << 64) - args.episodes:
                raise ValueError("seed plus episode index exceeds u64")
            if args.onnx_intra_threads <= 0:
                raise ValueError("onnx-intra-threads must be positive")
            profile_dir = resolve_runtime_profile(self.descriptor.id, environment.env_id).data_dir(
                get_config().data_root
            )
            model_dir = Path(args.model_dir) if args.model_dir else profile_dir / "models"
            player = "random"
            if args.checkpoint_id:
                repository = create_checkpoint_publisher(
                    self.artifact_contract(environment), model_dir
                )
                checkpoint = repository.resolve_checkpoint(args.checkpoint_id)
                player = str(checkpoint.onnx_path)
            candidates = [args.eval_binary, os.environ.get(EVAL_BINARY_ENV)]
            candidates.extend(str(path) for path in _BINARY_CANDIDATES)
            binary = next(
                (candidate for candidate in candidates if candidate and Path(candidate).is_file()),
                None,
            )
            if binary is None:
                raise ValueError("cartridge-eval binary not found; build it or pass --eval-binary")
            command = [
                binary,
                "--algorithm",
                self.descriptor.id,
                "--env-id",
                environment.env_id,
                "--p1",
                player,
                "--games",
                str(args.episodes),
                "--seed",
                str(args.seed),
                "--onnx-intra-threads",
                str(args.onnx_intra_threads),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                message = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(f"cartridge-eval exited with {completed.returncode}: {message}")
            result = DqnEvaluationResults.from_json(json.loads(completed.stdout))
            print(json.dumps(result.__dict__, sort_keys=True, separators=(",", ":")))
            return 0
        except Exception:
            logging.getLogger(__name__).exception("DQN evaluation failed")
            return 1

    def _run_loop(self, args: argparse.Namespace) -> int:
        from .dqn_loop import DqnLoop, DqnLoopConfig

        try:
            return DqnLoop(self, DqnLoopConfig.from_args(args)).run()
        except Exception:
            logging.getLogger(__name__).exception("DQN loop failed")
            return 1
