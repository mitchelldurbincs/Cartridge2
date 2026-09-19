"""Python bindings owned by the single-agent discrete DQN cartridge."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..environment_catalog import (
    AlgorithmDescriptor,
    CompatibilityReport,
    EnvironmentDescriptor,
)
from ..storage.publisher import OnnxArtifactContract, OnnxTensorSpec
from .base import AlgorithmCommand
from .dqn_application import DqnApplication
from .dqn_config import DqnLearnerConfig
from .dqn_contract import (
    ALGORITHM_ID,
    DESCRIPTOR,
    EXPECTED_COMPONENTS,
)
from .dqn_replay import DqnReplayBatch as DqnReplayBatch
from .dqn_replay import decode_replay_batch as decode_replay_batch
from .dqn_requests import DqnCollectRequest, DqnEvaluateRequest, DqnTrainRequest


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
