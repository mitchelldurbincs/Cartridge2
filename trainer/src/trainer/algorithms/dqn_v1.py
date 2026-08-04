"""Python bindings owned by the single-agent discrete DQN cartridge."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..environment_catalog import (
    AlgorithmDescriptor,
    CompatibilityReport,
    EnvironmentDescriptor,
    get_algorithm_descriptor,
)
from ..storage.base import ReplayRecord, ReplaySelection
from ..storage.publisher import OnnxArtifactContract, OnnxTensorSpec
from .base import AlgorithmCommand
from .dqn_application import DqnApplication
from .dqn_config import DqnLearnerConfig
from .dqn_requests import DqnCollectRequest, DqnEvaluateRequest, DqnTrainRequest

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
        self.application = DqnApplication(self)

    def commands(self) -> tuple[AlgorithmCommand, ...]:
        from .dqn_commands import commands

        return commands(self)

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

    def collect(self, request: DqnCollectRequest) -> int:
        return self.application.collect(request)

    def train(self, request: DqnTrainRequest) -> None:
        self.application.train(request)

    def evaluate(self, request: DqnEvaluateRequest) -> DqnEvaluationResults:
        return self.application.evaluate(request)
