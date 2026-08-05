"""DQN application operations, independent of the command-line parser."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from ..environment_catalog import get_environment
from ..storage import ReplayProfile
from ..storage.publisher import create_checkpoint_publisher
from .dqn_config import DqnLearnerConfig
from .dqn_requests import DqnCollectRequest, DqnEvaluateRequest, DqnTrainRequest

if TYPE_CHECKING:
    from .dqn_v1 import DqnEvaluationResults, DqnV1


class DqnApplication:
    """Execute complete typed requests for one DQN cartridge."""

    def __init__(self, cartridge: "DqnV1") -> None:
        self.cartridge = cartridge

    def collect(self, request: DqnCollectRequest) -> int:
        from ..orchestrator.actor_runner import _BINARY_CANDIDATES

        environment = get_environment(request.env_id)
        self.cartridge.compatibility(environment).require_compatible()
        actor_binary = _find_binary(
            request.actor_binary,
            os.environ.get("ACTOR_BINARY"),
            _BINARY_CANDIDATES,
        )
        if actor_binary is None:
            raise ValueError("actor binary not found; build it or pass --actor-binary")
        collector_config = {
            "schema_version": 1,
            "epsilon": request.epsilon,
            "seed": request.seed,
            "onnx_intra_threads": request.onnx_intra_threads,
        }
        command = [
            str(actor_binary),
            "--algorithm",
            self.cartridge.descriptor.id,
            "--env-id",
            environment.env_id,
            "--max-episodes",
            str(request.episodes),
            "--collection-scope-id",
            request.collection_scope_id,
            "--collector-config",
            json.dumps(collector_config, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "--actor-id",
            request.actor_id,
            "--episode-timeout-secs",
            str(request.episode_timeout_secs),
            "--data-dir",
            str(request.data_root),
            "--log-level",
            request.log_level.lower(),
        ]
        if request.source_checkpoint_id is not None:
            command.extend(["--source-checkpoint-id", request.source_checkpoint_id])
        return subprocess.run(command, check=False).returncode

    def train(self, request: DqnTrainRequest) -> None:
        environment = get_environment(request.env_id)
        self.cartridge.compatibility(environment).require_compatible()
        expected_profile = ReplayProfile(
            env_id=environment.env_id,
            env_contract_version=environment.contract_version,
            algorithm_id=self.cartridge.descriptor.id,
            experience_schema=self.cartridge.descriptor.components.experience_schema,
        )
        if request.replay_selection.profile != expected_profile:
            raise ValueError("DQN replay selection does not match the requested environment")
        config = DqnLearnerConfig(
            env_id=request.env_id,
            model_dir=str(request.model_dir),
            stats_path=str(request.stats_path),
            replay_selection=request.replay_selection,
            total_steps=request.total_steps,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            gamma=request.gamma,
            target_sync_interval=request.target_sync_interval,
            hidden_size=request.hidden_size,
            grad_clip_norm=request.grad_clip_norm,
            device=request.device,
        )
        self.cartridge.build_learner(config).train()

    def evaluate(self, request: DqnEvaluateRequest) -> "DqnEvaluationResults":
        from ..evaluator import _BINARY_CANDIDATES, EVAL_BINARY_ENV
        from .dqn_v1 import DqnEvaluationResults

        environment = get_environment(request.env_id)
        self.cartridge.compatibility(environment).require_compatible()
        player = "random"
        if request.checkpoint_id is not None:
            repository = create_checkpoint_publisher(
                self.cartridge.artifact_contract(environment), request.model_dir
            )
            checkpoint = repository.resolve_checkpoint(
                request.checkpoint_id, require_learner_state=False
            )
            player = str(checkpoint.onnx_path)
        binary = _find_binary(
            request.eval_binary,
            os.environ.get(EVAL_BINARY_ENV),
            _BINARY_CANDIDATES,
        )
        if binary is None:
            raise ValueError("cartridge-eval binary not found; build it or pass --eval-binary")
        command = [
            str(binary),
            "--algorithm",
            self.cartridge.descriptor.id,
            "--env-id",
            environment.env_id,
            "--p1",
            player,
            "--games",
            str(request.episodes),
            "--seed",
            str(request.seed),
            "--onnx-intra-threads",
            str(request.onnx_intra_threads),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"cartridge-eval exited with {completed.returncode}: {message}")
        return DqnEvaluationResults.from_json(json.loads(completed.stdout))


def format_evaluation_result(result: "DqnEvaluationResults") -> str:
    return json.dumps(result.__dict__, sort_keys=True, separators=(",", ":"))


def _find_binary(
    explicit: Path | None,
    environment_value: str | None,
    candidates: tuple[Path, ...] | list[Path],
) -> Path | None:
    paths = [explicit]
    if environment_value:
        paths.append(Path(environment_value))
    paths.extend(candidates)
    return next(
        (Path(path) for path in paths if path is not None and Path(path).is_file()),
        None,
    )
