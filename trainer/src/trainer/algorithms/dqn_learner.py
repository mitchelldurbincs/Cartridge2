"""Single-agent DQN learner and Q-network implementation."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ..checkpoint import (
    LearnerStateContract,
    export_onnx_artifact,
    learner_config_sha256,
    restore_learner_state,
    write_learner_state_artifact,
)
from ..environment_catalog import get_environment
from ..stats import (
    TrainerStats,
    decode_stats_snapshot,
    prepare_stats_snapshot,
    retain_training_history,
    write_stats_projection,
)
from ..storage import ReplayProfile, ReplaySelection, create_replay_store
from ..storage.evaluation import create_evaluation_repository
from ..storage.publisher import ArtifactValidationError, create_checkpoint_publisher
from ..storage.run_commit import RunCommitRepository, RunCommitV1
from .dqn_config import DqnLearnerConfig
from .dqn_v1 import ALGORITHM_ID, DESCRIPTOR, decode_replay_batch


class DqnQNetwork(nn.Module):
    """Online and target Q-networks with one inference-only forward boundary."""

    def __init__(self, obs_size: int, action_count: int, hidden_size: int):
        super().__init__()
        if min(obs_size, action_count, hidden_size) <= 0:
            raise ValueError("DQN network dimensions must be positive")
        self.obs_size = obs_size
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.online = self._mlp(obs_size, action_count, hidden_size)
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)

    @staticmethod
    def _mlp(obs_size: int, action_count: int, hidden_size: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.online(observation)

    def target_values(self, observation: torch.Tensor) -> torch.Tensor:
        return self.target(observation)

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())


class DqnLearner:
    """Train one root DQN checkpoint from an exact replay selection."""

    def __init__(self, config: DqnLearnerConfig):
        if not isinstance(config.replay_selection, ReplaySelection):
            raise ValueError("DQN training requires an exact ReplaySelection")
        self.config = config
        self.device = torch.device(config.resolve_device())
        environment = get_environment(config.env_id)
        environment.compatibility(ALGORITHM_ID).require_compatible()
        tensor = environment.capabilities.encoding.observation_tensor
        agents = environment.capabilities.agents.agents
        if tensor is None or tensor.fixed_elements is None or len(agents) != 1:
            raise ValueError("DQN requires one agent and a fixed observation tensor")
        action_count = agents[0].action_space.discrete_size
        if action_count is None:
            raise ValueError("DQN requires a discrete action space")
        self.obs_size = tensor.fixed_elements
        self.action_count = action_count
        self.replay_profile = ReplayProfile(
            env_id=environment.env_id,
            env_contract_version=environment.contract_version,
            algorithm_id=ALGORITHM_ID,
            experience_schema=DESCRIPTOR.components.experience_schema,
        )
        if config.replay_selection.profile != self.replay_profile:
            raise ArtifactValidationError("DQN replay selection profile mismatch")
        self.replay_selection = config.replay_selection
        from . import get_algorithm

        self.artifact_contract = get_algorithm(ALGORITHM_ID).artifact_contract(environment)
        self.config_sha256 = learner_config_sha256(config)
        self.network = DqnQNetwork(self.obs_size, self.action_count, config.hidden_size).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(
            self.network.online.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        Path(config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(config.stats_path).parent.mkdir(parents=True, exist_ok=True)
        self.checkpoints = create_checkpoint_publisher(
            self.artifact_contract, Path(config.model_dir)
        )
        self.evaluations = create_evaluation_repository(self.checkpoints)
        self.run_commits = RunCommitRepository(self.checkpoints, self.evaluations)
        self.parent_commit = None
        self.parent_run_commit_id = None
        self.parent_checkpoint_id = None
        self.start_step = 0
        self.stats = TrainerStats(env_id=config.env_id)
        run_head = self.checkpoints.resolve_run_head()
        expected_source = run_head.checkpoint_id if run_head is not None else None
        if self.replay_selection.source_checkpoint_id != expected_source:
            raise ArtifactValidationError(
                "DQN replay source checkpoint does not match the authoritative RunHead"
            )
        if run_head is not None:
            chain = self.run_commits.resolve_chain(run_head.run_commit_id)
            if not chain or chain[-1].run_commit_id != run_head.run_commit_id:
                raise ArtifactValidationError("DQN RunHead has an incomplete RunCommit chain")
            parent = chain[-1]
            if parent.commit.run_recipe is not None or parent.commit.orchestration is not None:
                raise ArtifactValidationError("DQN cannot resume an AlphaZero recipe-owned run")
            if parent.commit.config_sha256 != self.config_sha256:
                raise ArtifactValidationError("DQN learner configuration changed within the run")
            checkpoint = self.checkpoints.resolve_checkpoint(
                run_head.checkpoint_id,
                expected_config_sha256=self.config_sha256,
            )
            restore_learner_state(
                self.network,
                self.optimizer,
                checkpoint.learner_state_path,
                self.device,
                checkpoint.manifest,
                self.artifact_contract,
            )
            self.stats = decode_stats_snapshot(
                parent.commit.stats_snapshot.data,
                expected_stats_id=parent.commit.stats_id,
                expected_binding=parent.commit.stats_snapshot.binding,
            ).stats
            self.parent_commit = parent.commit
            self.parent_run_commit_id = parent.run_commit_id
            self.parent_checkpoint_id = checkpoint.checkpoint_id
            self.start_step = checkpoint.manifest.step

    def train_step(self, batch) -> tuple[float, float | None]:
        observations = torch.from_numpy(batch.observations).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.device)
        next_observations = torch.from_numpy(batch.next_observations).to(self.device)
        terminated = torch.from_numpy(batch.terminated).to(self.device)
        truncated = torch.from_numpy(batch.truncated).to(self.device)
        next_availability = torch.from_numpy(batch.next_availability).to(self.device)

        self.network.train()
        predicted = self.network(observations).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_q = self.network.target_values(next_observations)
            next_q = next_q.masked_fill(~next_availability, float("-inf"))
            done = terminated | truncated
            next_value = next_q.max(dim=1).values
            next_value[done] = 0.0
            target = rewards + self.config.gamma * next_value
        loss = F.smooth_l1_loss(predicted, target)
        if not torch.isfinite(loss):
            raise RuntimeError("DQN loss became non-finite")
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = None
        if self.config.grad_clip_norm > 0:
            grad_norm = float(
                clip_grad_norm_(self.network.online.parameters(), self.config.grad_clip_norm).item()
            )
        self.optimizer.step()
        return float(loss.item()), grad_norm

    def train(self) -> TrainerStats:
        replay = create_replay_store(self.replay_selection)
        samples_seen = 0
        final_loss = 0.0
        final_grad_norm = None
        history = []
        try:
            if replay.count() == 0:
                raise ValueError("DQN replay selection is empty")
            for step in range(1, self.config.total_steps + 1):
                global_step = self.start_step + step
                records = replay.sample(self.config.batch_size)
                batch = decode_replay_batch(
                    records,
                    selection=self.replay_selection,
                    obs_size=self.obs_size,
                    num_actions=self.action_count,
                )
                final_loss, final_grad_norm = self.train_step(batch)
                samples_seen += len(records)
                if global_step % self.config.target_sync_interval == 0:
                    self.network.sync_target()
                history.append(
                    {
                        "step": global_step,
                        "metrics": {"loss/td": final_loss},
                        "learning_rate": self.config.learning_rate,
                        "grad_norm": final_grad_norm,
                    }
                )
            replay_count = replay.count()
        finally:
            replay.close()

        model_root = Path(self.config.model_dir)
        with tempfile.TemporaryDirectory(prefix=".dqn-checkpoint-", dir=model_root) as staging:
            staging_dir = Path(staging)
            learner_contract = LearnerStateContract(self.network, self.optimizer, None)
            onnx_path = export_onnx_artifact(
                self.network,
                staging_dir / "model.onnx",
                self.device,
                self.artifact_contract,
            )
            learner_path = write_learner_state_artifact(
                self.network,
                self.optimizer,
                self.start_step + self.config.total_steps,
                staging_dir / "learner.pt",
                self.artifact_contract,
                self.config_sha256,
            )
            checkpoint = self.checkpoints.stage_checkpoint(
                onnx_path,
                learner_path,
                step=self.start_step + self.config.total_steps,
                parent_checkpoint_id=self.parent_checkpoint_id,
                config_sha256=self.config_sha256,
                learner_state_contract=learner_contract,
            )

        final_step = self.start_step + self.config.total_steps
        self.stats.step = final_step
        self.stats.total_steps = final_step
        self.stats.metrics = {"loss/td": final_loss}
        self.stats.learning_rate = self.config.learning_rate
        self.stats.samples_seen += samples_seen
        self.stats.replay_record_count = replay_count
        self.stats.last_checkpoint = checkpoint.checkpoint_id
        self.stats.history = retain_training_history([*self.stats.history, *history], final_step)
        prepared = prepare_stats_snapshot(self.stats, checkpoint)
        commit = RunCommitV1(
            profile=self.artifact_contract.profile,
            config_sha256=self.config_sha256,
            parent_run_commit_id=self.parent_run_commit_id,
            checkpoint_id=checkpoint.checkpoint_id,
            stats_snapshot=prepared,
            champion=(self.parent_commit.champion if self.parent_commit else None),
            evaluation_head_id=(
                self.parent_commit.evaluation_head_id if self.parent_commit else None
            ),
            orchestration=None,
        )
        reference = self.run_commits.publish(commit)
        self.checkpoints.commit_run_head(
            checkpoint_id=checkpoint.checkpoint_id,
            run_commit_id=reference.run_commit_id,
            expected_run_commit_id=self.parent_run_commit_id,
        )
        write_stats_projection(prepared, self.config.stats_path)
        return self.stats
