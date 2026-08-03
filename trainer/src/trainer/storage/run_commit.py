"""The single content-addressed authority for a Cartridge2 training run."""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol

from ..stats import (
    DEFAULT_MAX_EVAL_HISTORY,
    EvaluationStats,
    LoadedStatsSnapshotV3,
    PreparedStatsSnapshotV3,
    StatsBindingV1,
    decode_stats_snapshot,
    retain_training_history,
)
from .evaluation import (
    ChampionReferenceV1,
    EvaluationArtifactV2,
    EvaluationRepository,
)
from .publisher import (
    ArtifactValidationError,
    CheckpointProfileV1,
    CheckpointPublisher,
    canonical_json_bytes,
    sha256_bytes,
    validate_sha256_digest,
)

_MAX_U64 = (1 << 64) - 1
_MAX_U32 = (1 << 32) - 1
_MAX_F32 = float.fromhex("0x1.fffffep+127")

_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
_PROFILE_FIELDS = frozenset(
    {
        "algorithm_id",
        "env_id",
        "env_contract_version",
        "model_artifact_schema_version",
        "model_contract",
    }
)
_CHAMPION_FIELDS = frozenset({"checkpoint_id", "evaluation_id"})
_ORCHESTRATION_FIELDS = frozenset(
    {
        "iteration",
        "episodes_generated",
        "transitions_generated",
        "training_steps",
        "actor_time_seconds",
        "trainer_time_seconds",
        "eval_time_seconds",
        "total_time_seconds",
        "eval_win_rate",
        "eval_draw_rate",
        "timestamp",
        "evaluation_id",
        "collector_simulations",
        "collector_seed",
        "evaluation_seed",
        "collection_scope_id",
        "source_checkpoint_id",
    }
)
_RUN_RECIPE_FIELDS = frozenset(
    {
        "schema_version",
        "learner_recipe",
        "learner_config_sha256",
        "total_iterations",
        "episodes_per_iteration",
        "training_steps_per_iteration",
        "num_actors",
        "collector_episode_timeout_seconds",
        "collector_eval_batch_size",
        "collector_onnx_intra_threads",
        "mcts_start_simulations",
        "mcts_max_simulations",
        "mcts_simulation_ramp",
        "collector_c_puct",
        "collector_temperature",
        "collector_late_temperature",
        "temperature_move_threshold",
        "collector_dirichlet_alpha",
        "collector_dirichlet_weight",
        "collector_seed_strategy",
        "replay_policy",
        "evaluation_interval",
        "evaluation_games",
        "evaluation_simulations",
        "evaluation_temperature",
        "evaluation_win_threshold",
        "evaluation_vs_random",
        "solver_games",
        "evaluation_seed",
        "promotion_metric",
        "promotion_margin",
    }
)
_LEARNER_RECIPE_FIELDS = frozenset(
    {
        "schema_version",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "value_loss_weight",
        "policy_loss_weight",
        "grad_clip_norm",
        "use_lr_scheduler",
        "lr_min_ratio",
        "lr_warmup_steps",
        "lr_warmup_start_ratio",
        "lr_horizon_steps",
        "training_steps",
        "clear_replay_on_start",
        "replay_window",
        "replay_cleanup_cadence",
        "model_architecture",
    }
)
_MODEL_ARCHITECTURE_FIELDS = frozenset(
    {
        "schema_version",
        "implementation",
        "network_type",
        "observation_elements",
        "action_count",
        "hidden_size",
        "board_width",
        "board_height",
        "observation_spatial_channels",
        "residual_blocks",
        "residual_filters",
    }
)
_RUN_COMMIT_FIELDS = frozenset(
    {
        "schema_version",
        "profile",
        "config_sha256",
        "run_recipe_id",
        "run_recipe",
        "parent_run_commit_id",
        "checkpoint_id",
        "stats_id",
        "stats_snapshot",
        "champion",
        "evaluation_head_id",
        "orchestration",
    }
)


def _exact(value: object, fields: frozenset[str], *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != fields:
        actual = frozenset(value) if isinstance(value, dict) else frozenset()
        raise ArtifactValidationError(
            f"{context} fields must be exact "
            f"(missing={sorted(fields - actual)}, extra={sorted(actual - fields)})"
        )
    return value


def _decode_canonical(data: bytes, *, context: str) -> Mapping[str, Any]:
    if not isinstance(data, bytes):
        raise ArtifactValidationError(f"{context} must be bytes")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"{context} is not valid UTF-8 JSON") from exc
    if canonical_json_bytes(value) != data:
        raise ArtifactValidationError(f"{context} is not canonical JSON")
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{context} must be a JSON object")
    return value


def _integer(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > _MAX_U64:
        qualifier = "positive" if positive else "nonnegative"
        raise ArtifactValidationError(f"{field} must be a {qualifier} integer")
    return value


def _u32(value: object, *, field: str, positive: bool = False) -> int:
    result = _integer(value, field=field, positive=positive)
    if result > _MAX_U32:
        raise ArtifactValidationError(f"{field} exceeds u32")
    return result


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{field} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ArtifactValidationError(f"{field} must be a finite nonnegative number")
    return 0.0 if result == 0.0 else result


def _f32_number(value: object, *, field: str) -> float:
    result = _number(value, field=field)
    if result > _MAX_F32:
        raise ArtifactValidationError(f"{field} exceeds f32")
    return float(struct.unpack("!f", struct.pack("!f", result))[0])


def _optional_rate(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    result = _number(value, field=field)
    if result > 1.0:
        raise ArtifactValidationError(f"{field} must be between zero and one")
    return result


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{field} must be a nonempty string")
    return value


def _timestamp(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ArtifactValidationError(f"{field} must be a UTC timestamp with six fractional digits")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactValidationError(f"{field} is not a valid timestamp") from exc
    return value


@dataclass(frozen=True, init=False)
class LearnerRecipeV1:
    """Mutation-safe canonical algorithm-owned learner semantics."""

    _data: bytes

    def __init__(self, value: Mapping[str, Any]):
        if not isinstance(value, Mapping) or not value:
            raise ArtifactValidationError("learner_recipe must be a nonempty object")
        data = canonical_json_bytes(dict(value))
        decoded = _decode_canonical(data, context="learner recipe")
        object.__setattr__(self, "_data", canonical_json_bytes(dict(decoded)))

    def to_dict(self) -> dict[str, object]:
        return dict(_decode_canonical(self._data, context="learner recipe"))

    @property
    def config_sha256(self) -> str:
        return sha256_bytes(self._data)


def _validate_alphazero_learner_recipe(
    value: Mapping[str, Any],
    *,
    total_iterations: int,
    training_steps_per_iteration: int,
    replay_policy: str,
) -> None:
    fields = _exact(value, _LEARNER_RECIPE_FIELDS, context="AlphaZero learner recipe")
    if fields["schema_version"] != 2 or isinstance(fields["schema_version"], bool):
        raise ArtifactValidationError("learner_recipe.schema_version must be exactly 2")
    _integer(fields["batch_size"], field="learner_recipe.batch_size", positive=True)
    for field in (
        "learning_rate",
        "weight_decay",
        "value_loss_weight",
        "policy_loss_weight",
        "grad_clip_norm",
    ):
        _number(fields[field], field=f"learner_recipe.{field}")
    if not isinstance(fields["use_lr_scheduler"], bool):
        raise ArtifactValidationError("learner_recipe.use_lr_scheduler must be boolean")
    for field in ("lr_min_ratio", "lr_warmup_start_ratio"):
        rate = _optional_rate(fields[field], field=f"learner_recipe.{field}")
        assert rate is not None
    _integer(fields["lr_warmup_steps"], field="learner_recipe.lr_warmup_steps")
    training_steps = _integer(
        fields["training_steps"],
        field="learner_recipe.training_steps",
        positive=True,
    )
    if training_steps != training_steps_per_iteration:
        raise ArtifactValidationError("learner_recipe.training_steps disagrees with the run recipe")
    if total_iterations > _MAX_U64 // training_steps_per_iteration:
        raise ArtifactValidationError("run recipe global LR horizon exceeds u64")
    horizon = _integer(
        fields["lr_horizon_steps"],
        field="learner_recipe.lr_horizon_steps",
        positive=True,
    )
    if horizon != total_iterations * training_steps_per_iteration:
        raise ArtifactValidationError(
            "learner_recipe.lr_horizon_steps disagrees with the global run target"
        )
    if not isinstance(fields["clear_replay_on_start"], bool):
        raise ArtifactValidationError("learner_recipe.clear_replay_on_start must be boolean")
    replay_window = _integer(fields["replay_window"], field="learner_recipe.replay_window")
    cleanup = _integer(
        fields["replay_cleanup_cadence"],
        field="learner_recipe.replay_cleanup_cadence",
    )
    if replay_policy == "scoped_fresh_iteration_v1" and (
        fields["clear_replay_on_start"] or replay_window != 0 or cleanup != 0
    ):
        raise ArtifactValidationError(
            "scoped_fresh_iteration_v1 requires no learner-owned replay reset or window"
        )
    if replay_window == 0 and cleanup != 0:
        raise ArtifactValidationError("learner replay cleanup requires a nonzero replay window")

    model = _exact(
        fields["model_architecture"],
        _MODEL_ARCHITECTURE_FIELDS,
        context="AlphaZero model architecture",
    )
    if model["schema_version"] != 2 or isinstance(model["schema_version"], bool):
        raise ArtifactValidationError("model_architecture.schema_version must be exactly 2")
    if model["implementation"] != "alphazero_policy_value_network_v1":
        raise ArtifactValidationError("model_architecture implementation is unsupported")
    if model["network_type"] not in {"mlp", "resnet"}:
        raise ArtifactValidationError("model_architecture.network_type is invalid")
    for field in (
        "observation_elements",
        "action_count",
        "hidden_size",
        "board_width",
        "board_height",
        "observation_spatial_channels",
        "residual_blocks",
        "residual_filters",
    ):
        _u32(model[field], field=f"model_architecture.{field}", positive=True)


@dataclass(frozen=True)
class RunRecipeV1:
    """Canonical, immutable semantics for one synchronized training run."""

    learner_recipe: LearnerRecipeV1
    learner_config_sha256: str
    total_iterations: int
    episodes_per_iteration: int
    training_steps_per_iteration: int
    num_actors: int
    collector_episode_timeout_seconds: int
    collector_eval_batch_size: int
    collector_onnx_intra_threads: int
    mcts_start_simulations: int
    mcts_max_simulations: int
    mcts_simulation_ramp: int
    collector_c_puct: float
    collector_temperature: float
    collector_late_temperature: float
    temperature_move_threshold: int
    collector_dirichlet_alpha: float
    collector_dirichlet_weight: float
    collector_seed_strategy: str
    replay_policy: str
    evaluation_interval: int
    evaluation_games: int
    evaluation_simulations: int
    evaluation_temperature: float
    evaluation_win_threshold: float
    evaluation_vs_random: bool
    solver_games: int
    evaluation_seed: int
    promotion_metric: str
    promotion_margin: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ArtifactValidationError("run_recipe.schema_version must be exactly 1")
        if not isinstance(self.learner_recipe, LearnerRecipeV1):
            raise ArtifactValidationError("run_recipe.learner_recipe is invalid")
        validate_sha256_digest(
            self.learner_config_sha256,
            field="run_recipe.learner_config_sha256",
        )
        if self.learner_config_sha256 != self.learner_recipe.config_sha256:
            raise ArtifactValidationError("run_recipe learner digest disagrees with learner_recipe")
        _integer(self.total_iterations, field="run_recipe.total_iterations", positive=True)
        _u32(
            self.episodes_per_iteration,
            field="run_recipe.episodes_per_iteration",
            positive=True,
        )
        _integer(
            self.training_steps_per_iteration,
            field="run_recipe.training_steps_per_iteration",
            positive=True,
        )
        _u32(self.num_actors, field="run_recipe.num_actors", positive=True)
        if self.num_actors > self.episodes_per_iteration:
            raise ArtifactValidationError(
                "run_recipe.num_actors cannot exceed episodes_per_iteration"
            )
        _integer(
            self.collector_episode_timeout_seconds,
            field="run_recipe.collector_episode_timeout_seconds",
            positive=True,
        )
        _u32(
            self.collector_eval_batch_size,
            field="run_recipe.collector_eval_batch_size",
            positive=True,
        )
        _u32(
            self.collector_onnx_intra_threads,
            field="run_recipe.collector_onnx_intra_threads",
            positive=True,
        )
        start = _u32(
            self.mcts_start_simulations,
            field="run_recipe.mcts_start_simulations",
            positive=True,
        )
        maximum = _u32(
            self.mcts_max_simulations,
            field="run_recipe.mcts_max_simulations",
            positive=True,
        )
        _u32(
            self.mcts_simulation_ramp,
            field="run_recipe.mcts_simulation_ramp",
        )
        for field in (
            "collector_c_puct",
            "collector_temperature",
            "collector_late_temperature",
            "collector_dirichlet_alpha",
        ):
            normalized = _f32_number(getattr(self, field), field=f"run_recipe.{field}")
            object.__setattr__(self, field, normalized)
        dirichlet_weight = _f32_number(
            self.collector_dirichlet_weight,
            field="run_recipe.collector_dirichlet_weight",
        )
        if dirichlet_weight > 1.0:
            raise ArtifactValidationError(
                "run_recipe.collector_dirichlet_weight must be between zero and one"
            )
        object.__setattr__(self, "collector_dirichlet_weight", dirichlet_weight)
        if (self.collector_dirichlet_alpha == 0.0) != (dirichlet_weight == 0.0):
            raise ArtifactValidationError(
                "run_recipe collector Dirichlet alpha and weight must both be zero to disable noise"
            )
        _u32(
            self.temperature_move_threshold,
            field="run_recipe.temperature_move_threshold",
        )
        if self.temperature_move_threshold == 0:
            if self.collector_late_temperature != self.collector_temperature:
                raise ArtifactValidationError(
                    "run_recipe collector late temperature must equal base "
                    "temperature when its move threshold is zero"
                )
        elif self.collector_late_temperature == self.collector_temperature:
            raise ArtifactValidationError(
                "run_recipe collector late temperature must differ from base "
                "temperature when its schedule is enabled"
            )
        if start > maximum:
            raise ArtifactValidationError("run_recipe MCTS start simulations exceed its maximum")
        if start == maximum:
            if self.mcts_simulation_ramp != 0:
                raise ArtifactValidationError(
                    "run_recipe constant MCTS schedule requires a zero ramp"
                )
        else:
            delta = maximum - start
            if self.mcts_simulation_ramp == 0 or self.mcts_simulation_ramp > delta:
                raise ArtifactValidationError(
                    "run_recipe ramped MCTS schedule has a noncanonical ramp"
                )
            steps_to_cap = (delta + self.mcts_simulation_ramp - 1) // self.mcts_simulation_ramp
            if self.total_iterations - 1 < steps_to_cap:
                raise ArtifactValidationError(
                    "run_recipe MCTS schedule does not reach its cap within the run"
                )
        if self.collector_seed_strategy != "system_entropy_v1":
            raise ArtifactValidationError("run_recipe.collector_seed_strategy is unsupported")
        if self.replay_policy != "scoped_fresh_iteration_v1":
            raise ArtifactValidationError("run_recipe.replay_policy is unsupported")
        _integer(
            self.evaluation_interval,
            field="run_recipe.evaluation_interval",
        )
        games = _u32(
            self.evaluation_games,
            field="run_recipe.evaluation_games",
            positive=True,
        )
        _u32(
            self.evaluation_simulations,
            field="run_recipe.evaluation_simulations",
        )
        temperature = _f32_number(
            self.evaluation_temperature,
            field="run_recipe.evaluation_temperature",
        )
        object.__setattr__(self, "evaluation_temperature", temperature)
        threshold = _optional_rate(
            self.evaluation_win_threshold,
            field="run_recipe.evaluation_win_threshold",
        )
        margin = _optional_rate(
            self.promotion_margin,
            field="run_recipe.promotion_margin",
        )
        assert threshold is not None and margin is not None
        object.__setattr__(self, "evaluation_win_threshold", threshold)
        object.__setattr__(self, "promotion_margin", margin)
        if not isinstance(self.evaluation_vs_random, bool):
            raise ArtifactValidationError("run_recipe.evaluation_vs_random must be boolean")
        solver_games = _u32(
            self.solver_games,
            field="run_recipe.solver_games",
        )
        seed = _integer(
            self.evaluation_seed,
            field="run_recipe.evaluation_seed",
        )
        largest_run = max(games, solver_games)
        if seed > _MAX_U64 - (largest_run - 1):
            raise ArtifactValidationError("run_recipe.evaluation_seed plus game index exceeds u64")
        if self.promotion_metric not in {"win_rate", "solver_optimal"}:
            raise ArtifactValidationError("run_recipe.promotion_metric is invalid")
        if self.promotion_metric == "win_rate" and self.promotion_margin != 0.0:
            raise ArtifactValidationError(
                "run_recipe.promotion_margin must be zero for win_rate promotion"
            )
        if self.promotion_metric == "solver_optimal" and self.evaluation_win_threshold != 0.0:
            raise ArtifactValidationError(
                "run_recipe.evaluation_win_threshold must be zero for solver_optimal promotion"
            )
        if self.promotion_metric == "solver_optimal" and solver_games == 0:
            raise ArtifactValidationError("solver_optimal promotion requires solver games")
        _validate_alphazero_learner_recipe(
            self.learner_recipe.to_dict(),
            total_iterations=self.total_iterations,
            training_steps_per_iteration=self.training_steps_per_iteration,
            replay_policy=self.replay_policy,
        )

    def simulations_for(self, iteration: int) -> int:
        _integer(iteration, field="iteration", positive=True)
        ramped = self.mcts_start_simulations + (iteration - 1) * self.mcts_simulation_ramp
        return min(ramped, self.mcts_max_simulations)

    def evaluation_scheduled(self, iteration: int) -> bool:
        _integer(iteration, field="iteration", positive=True)
        return self.evaluation_interval > 0 and iteration % self.evaluation_interval == 0

    def to_dict(self) -> dict[str, object]:
        value = {field: getattr(self, field) for field in _RUN_RECIPE_FIELDS}
        value["learner_recipe"] = self.learner_recipe.to_dict()
        return value

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def run_recipe_id(self) -> str:
        return sha256_bytes(self.to_bytes())

    @classmethod
    def from_dict(cls, value: object) -> "RunRecipeV1":
        fields = _exact(value, _RUN_RECIPE_FIELDS, context="run recipe")
        kwargs = dict(fields)
        kwargs["learner_recipe"] = LearnerRecipeV1(fields["learner_recipe"])
        return cls(**kwargs)


@dataclass(frozen=True)
class OrchestrationCommitV1:
    iteration: int
    episodes_generated: int
    transitions_generated: int
    training_steps: int
    actor_time_seconds: float
    trainer_time_seconds: float
    eval_time_seconds: float
    total_time_seconds: float
    eval_win_rate: float | None
    eval_draw_rate: float | None
    timestamp: str
    evaluation_id: str | None
    collector_simulations: int
    collector_seed: int | None
    evaluation_seed: int | None
    collection_scope_id: str
    source_checkpoint_id: str | None

    def __post_init__(self) -> None:
        _integer(self.iteration, field="orchestration.iteration", positive=True)
        for field in (
            "episodes_generated",
            "transitions_generated",
            "training_steps",
        ):
            _integer(getattr(self, field), field=f"orchestration.{field}")
        _u32(
            self.collector_simulations,
            field="orchestration.collector_simulations",
        )
        if self.collector_seed is not None:
            _integer(self.collector_seed, field="orchestration.collector_seed")
        if self.evaluation_seed is not None:
            _integer(self.evaluation_seed, field="orchestration.evaluation_seed")
        validate_sha256_digest(
            self.collection_scope_id,
            field="orchestration.collection_scope_id",
        )
        if self.source_checkpoint_id is not None:
            validate_sha256_digest(
                self.source_checkpoint_id,
                field="orchestration.source_checkpoint_id",
            )
        phase_total = 0.0
        for field in (
            "actor_time_seconds",
            "trainer_time_seconds",
            "eval_time_seconds",
        ):
            normalized = _number(getattr(self, field), field=f"orchestration.{field}")
            object.__setattr__(self, field, normalized)
            phase_total += normalized
        total = _number(self.total_time_seconds, field="orchestration.total_time_seconds")
        object.__setattr__(self, "total_time_seconds", total)
        if total + 1e-12 < phase_total:
            raise ArtifactValidationError(
                "orchestration.total_time_seconds is shorter than its phases"
            )
        win_rate = _optional_rate(self.eval_win_rate, field="orchestration.eval_win_rate")
        draw_rate = _optional_rate(self.eval_draw_rate, field="orchestration.eval_draw_rate")
        object.__setattr__(self, "eval_win_rate", win_rate)
        object.__setattr__(self, "eval_draw_rate", draw_rate)
        if (win_rate is None) != (draw_rate is None):
            raise ArtifactValidationError("Orchestration evaluation rates are incomplete")
        if win_rate is not None and win_rate + draw_rate > 1.0:
            raise ArtifactValidationError("Orchestration evaluation rates exceed one")
        _timestamp(self.timestamp, field="orchestration.timestamp")
        if self.evaluation_id is None:
            if (
                self.eval_time_seconds != 0.0
                or win_rate is not None
                or self.evaluation_seed is not None
            ):
                raise ArtifactValidationError(
                    "A non-evaluation commit cannot contain evaluation metrics"
                )
        else:
            validate_sha256_digest(self.evaluation_id, field="orchestration.evaluation_id")
            if self.evaluation_seed is None:
                raise ArtifactValidationError(
                    "An evaluation commit requires evaluation_seed provenance"
                )

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in _ORCHESTRATION_FIELDS}

    @classmethod
    def from_dict(cls, value: object) -> "OrchestrationCommitV1":
        fields = _exact(value, _ORCHESTRATION_FIELDS, context="orchestration commit")
        return cls(**dict(fields))


StatsSnapshotV2 = PreparedStatsSnapshotV3 | LoadedStatsSnapshotV3


@dataclass(frozen=True)
class RunCommitV1:
    profile: CheckpointProfileV1
    config_sha256: str
    parent_run_commit_id: str | None
    checkpoint_id: str
    stats_snapshot: StatsSnapshotV2
    champion: ChampionReferenceV1 | None
    evaluation_head_id: str | None
    orchestration: OrchestrationCommitV1 | None
    run_recipe: RunRecipeV1 | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ArtifactValidationError("run_commit.schema_version must be exactly 1")
        if not isinstance(self.profile, CheckpointProfileV1):
            raise ArtifactValidationError("run_commit.profile is invalid")
        validate_sha256_digest(self.config_sha256, field="run_commit.config_sha256")
        if self.run_recipe is not None:
            if not isinstance(self.run_recipe, RunRecipeV1):
                raise ArtifactValidationError("run_commit.run_recipe is invalid")
            if self.run_recipe.learner_config_sha256 != self.config_sha256:
                raise ArtifactValidationError(
                    "RunCommit recipe learner digest disagrees with config_sha256"
                )
        if self.parent_run_commit_id is not None:
            validate_sha256_digest(
                self.parent_run_commit_id,
                field="run_commit.parent_run_commit_id",
            )
        validate_sha256_digest(self.checkpoint_id, field="run_commit.checkpoint_id")
        if not isinstance(self.stats_snapshot, (PreparedStatsSnapshotV3, LoadedStatsSnapshotV3)):
            raise ArtifactValidationError("run_commit.stats_snapshot is invalid")
        binding = self.stats_snapshot.binding
        if (
            binding.profile != self.profile
            or binding.config_sha256 != self.config_sha256
            or binding.checkpoint_id != self.checkpoint_id
        ):
            raise ArtifactValidationError(
                "RunCommit fields do not match the embedded stats binding"
            )
        if self.evaluation_head_id is not None:
            validate_sha256_digest(self.evaluation_head_id, field="run_commit.evaluation_head_id")
        if self.champion is not None and self.evaluation_head_id is None:
            raise ArtifactValidationError("A RunCommit champion requires an evaluation head")
        if self.orchestration is not None and not isinstance(
            self.orchestration, OrchestrationCommitV1
        ):
            raise ArtifactValidationError("run_commit.orchestration is invalid")

    @property
    def stats_id(self) -> str:
        return self.stats_snapshot.stats_id

    @property
    def run_recipe_id(self) -> str | None:
        return self.run_recipe.run_recipe_id if self.run_recipe is not None else None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile.to_dict(),
            "config_sha256": self.config_sha256,
            "run_recipe_id": self.run_recipe_id,
            "run_recipe": (self.run_recipe.to_dict() if self.run_recipe is not None else None),
            "parent_run_commit_id": self.parent_run_commit_id,
            "checkpoint_id": self.checkpoint_id,
            "stats_id": self.stats_id,
            "stats_snapshot": dict(
                _decode_canonical(
                    self.stats_snapshot.data,
                    context="run commit stats snapshot",
                )
            ),
            "champion": self.champion.to_dict() if self.champion is not None else None,
            "evaluation_head_id": self.evaluation_head_id,
            "orchestration": (
                self.orchestration.to_dict() if self.orchestration is not None else None
            ),
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def run_commit_id(self) -> str:
        return sha256_bytes(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "RunCommitV1":
        raw = _decode_canonical(data, context="run commit")
        fields = _exact(raw, _RUN_COMMIT_FIELDS, context="run commit")
        if fields["schema_version"] != 1 or isinstance(fields["schema_version"], bool):
            raise ArtifactValidationError("run_commit.schema_version must be exactly 1")
        profile_fields = _exact(fields["profile"], _PROFILE_FIELDS, context="run commit profile")
        profile = CheckpointProfileV1.from_dict(dict(profile_fields))
        config_sha256 = validate_sha256_digest(
            fields["config_sha256"], field="run_commit.config_sha256"
        )
        run_recipe_value = fields["run_recipe"]
        run_recipe = (
            RunRecipeV1.from_dict(run_recipe_value) if run_recipe_value is not None else None
        )
        run_recipe_id = fields["run_recipe_id"]
        if (run_recipe is None) != (run_recipe_id is None):
            raise ArtifactValidationError(
                "run_commit.run_recipe and run_recipe_id must both be null or present"
            )
        if run_recipe is not None:
            run_recipe_id = validate_sha256_digest(run_recipe_id, field="run_commit.run_recipe_id")
            if run_recipe_id != run_recipe.run_recipe_id:
                raise ArtifactValidationError("RunCommit recipe SHA-256 mismatch")
        checkpoint_id = validate_sha256_digest(
            fields["checkpoint_id"], field="run_commit.checkpoint_id"
        )
        snapshot_bytes = canonical_json_bytes(fields["stats_snapshot"])
        snapshot_binding = StatsBindingV1.from_fields(fields["stats_snapshot"])
        expected_binding = StatsBindingV1(
            profile=profile,
            config_sha256=config_sha256,
            checkpoint_id=checkpoint_id,
            step=snapshot_binding.step,
        )
        stats = decode_stats_snapshot(
            snapshot_bytes,
            expected_stats_id=validate_sha256_digest(
                fields["stats_id"], field="run_commit.stats_id"
            ),
            expected_binding=expected_binding,
        )
        parent = fields["parent_run_commit_id"]
        if parent is not None:
            parent = validate_sha256_digest(parent, field="run_commit.parent_run_commit_id")
        evaluation_head = fields["evaluation_head_id"]
        if evaluation_head is not None:
            evaluation_head = validate_sha256_digest(
                evaluation_head, field="run_commit.evaluation_head_id"
            )
        commit = cls(
            schema_version=1,
            profile=profile,
            config_sha256=config_sha256,
            run_recipe=run_recipe,
            parent_run_commit_id=parent,
            checkpoint_id=checkpoint_id,
            stats_snapshot=stats,
            champion=(
                ChampionReferenceV1.from_dict(fields["champion"])
                if fields["champion"] is not None
                else None
            ),
            evaluation_head_id=evaluation_head,
            orchestration=(
                OrchestrationCommitV1.from_dict(fields["orchestration"])
                if fields["orchestration"] is not None
                else None
            ),
        )
        if commit.to_bytes() != data:
            raise ArtifactValidationError("RunCommit is not normalized canonical JSON")
        return commit


@dataclass(frozen=True)
class RunCommitRef:
    run_commit_id: str
    commit: RunCommitV1

    def __post_init__(self) -> None:
        validate_sha256_digest(self.run_commit_id, field="run_commit_id")
        if self.run_commit_id != self.commit.run_commit_id:
            raise ArtifactValidationError("RunCommitRef ID does not match its bytes")


def _loaded_stats(commit: RunCommitV1):
    return decode_stats_snapshot(
        commit.stats_snapshot.data,
        expected_stats_id=commit.stats_id,
        expected_binding=commit.stats_snapshot.binding,
    ).stats


def _validate_stats_continuity(
    commit: RunCommitV1,
    parent: RunCommitV1 | None,
) -> tuple[object, object | None]:
    current = _loaded_stats(commit)
    if parent is None:
        return current, None
    previous = _loaded_stats(parent)
    if commit.checkpoint_id == parent.checkpoint_id:
        if commit.stats_id != parent.stats_id:
            raise ArtifactValidationError(
                "A same-checkpoint RunCommit must carry the exact stats snapshot"
            )
        return current, previous
    if current.samples_seen < previous.samples_seen:
        raise ArtifactValidationError("RunCommit stats samples_seen regressed")
    if current.total_steps < previous.total_steps:
        raise ArtifactValidationError("RunCommit stats total_steps regressed")
    if current.timestamp < previous.timestamp:
        raise ArtifactValidationError("RunCommit stats timestamp regressed")

    new_history = [entry for entry in current.history if entry["step"] > previous.step]
    expected_history = retain_training_history(
        [*previous.history, *new_history],
        current.step,
    )
    if current.history != expected_history:
        raise ArtifactValidationError(
            "RunCommit stats training history is not the exact retained continuation"
        )
    return current, previous


def _validate_eval_stats_continuity(current, previous, artifact) -> None:
    inherited = previous.evaluation_history if previous is not None else []
    result = artifact.results.vs_random if artifact is not None else None
    if result is None:
        if current.evaluation_history != inherited:
            raise ArtifactValidationError(
                "RunCommit stats evaluation history changed without random evidence"
            )
        return
    completed = datetime.fromisoformat(artifact.completed_at.replace("Z", "+00:00"))
    appended = EvaluationStats(
        step=current.step,
        metrics={
            "outcome/win_rate": result.candidate_win_rate,
            "outcome/draw_rate": result.draw_rate,
            "outcome/loss_rate": 1.0 - result.candidate_win_rate - result.draw_rate,
        },
        episodes=result.games_played,
        mean_episode_length=result.average_game_length,
        timestamp=completed.timestamp(),
    ).to_dict()
    expected = [*inherited, appended][-DEFAULT_MAX_EVAL_HISTORY:]
    if current.evaluation_history != expected or current.last_evaluation is None:
        raise ArtifactValidationError(
            "RunCommit stats do not contain the exact random-evaluation projection"
        )


def _validate_evaluation_recipe(
    artifact: EvaluationArtifactV2,
    recipe: RunRecipeV1,
) -> None:
    requested = artifact.recipe.requested_games
    has_champion = artifact.champion_before is not None
    solver_enabled = artifact.profile.env_id == "connect4" and recipe.solver_games > 0
    expected_requested = (
        recipe.evaluation_games if has_champion else 0,
        recipe.evaluation_games if recipe.evaluation_vs_random else 0,
        recipe.solver_games if solver_enabled else 0,
        recipe.solver_games if solver_enabled and has_champion else 0,
    )
    actual_requested = (
        requested.vs_champion,
        requested.vs_random,
        requested.candidate_solver,
        requested.champion_solver,
    )
    if (
        artifact.recipe.simulations != recipe.evaluation_simulations
        or artifact.recipe.temperature != recipe.evaluation_temperature
        or artifact.recipe.promotion_metric != recipe.promotion_metric
        or artifact.recipe.promotion_margin != recipe.promotion_margin
        or artifact.recipe.win_threshold != recipe.evaluation_win_threshold
        or artifact.recipe.seed != recipe.evaluation_seed
        or actual_requested != expected_requested
    ):
        raise ArtifactValidationError("Evaluation evidence disagrees with the immutable run recipe")


def validate_transition(
    commit: RunCommitV1,
    parent: RunCommitV1 | None,
    *,
    checkpoints: CheckpointPublisher,
    evaluations: EvaluationRepository,
    prepared_evaluation: EvaluationArtifactV2 | None = None,
) -> None:
    """Validate one full RunCommit edge and every referenced immutable object."""
    if not isinstance(commit, RunCommitV1):
        raise TypeError("commit must be RunCommitV1")
    if parent is not None and not isinstance(parent, RunCommitV1):
        raise TypeError("parent must be RunCommitV1 or None")
    expected_parent_id = parent.run_commit_id if parent is not None else None
    if commit.parent_run_commit_id != expected_parent_id:
        raise ArtifactValidationError("RunCommit parent identity is inconsistent")

    if parent is None:
        if commit.orchestration is None and commit.run_recipe is not None:
            raise ArtifactValidationError(
                "A standalone prefix cannot introduce a synchronized run recipe"
            )
    elif parent.run_recipe is None:
        if commit.orchestration is not None or commit.run_recipe is not None:
            raise ArtifactValidationError("Standalone and synchronized run modes cannot be mixed")
    else:
        if commit.run_recipe != parent.run_recipe:
            raise ArtifactValidationError("RunCommit changed the immutable run recipe")
        if commit.orchestration is None:
            raise ArtifactValidationError("A recipe-owned run cannot contain standalone commits")

    checkpoint = checkpoints.read_checkpoint_manifest_exact(commit.checkpoint_id)
    if (
        checkpoint.profile != commit.profile
        or checkpoint.config_sha256 != commit.config_sha256
        or checkpoint.step != commit.stats_snapshot.binding.step
    ):
        raise ArtifactValidationError(
            "RunCommit checkpoint does not match its profile/config/stats binding"
        )
    current_stats, previous_stats = _validate_stats_continuity(commit, parent)

    inherited_champion = parent.champion if parent is not None else None
    inherited_evaluation = parent.evaluation_head_id if parent is not None else None
    if parent is None:
        if checkpoint.parent_checkpoint_id is not None:
            raise ArtifactValidationError(
                "The first RunCommit checkpoint must start a checkpoint lineage"
            )
    else:
        if commit.profile != parent.profile or commit.config_sha256 != parent.config_sha256:
            raise ArtifactValidationError("RunCommit profile/config changed within a run")
        parent_checkpoint = checkpoints.read_checkpoint_manifest_exact(parent.checkpoint_id)
        if commit.checkpoint_id == parent.checkpoint_id:
            raise ArtifactValidationError("A RunCommit must select a new direct-child checkpoint")
        if (
            checkpoint.parent_checkpoint_id != parent.checkpoint_id
            or checkpoint.step <= parent_checkpoint.step
        ):
            raise ArtifactValidationError(
                "RunCommit checkpoint must be a strictly newer direct child"
            )

    orchestration = commit.orchestration
    if orchestration is None:
        if (
            commit.champion != inherited_champion
            or commit.evaluation_head_id != inherited_evaluation
        ):
            raise ArtifactValidationError(
                "A standalone RunCommit must carry evaluation state unchanged"
            )
        _validate_eval_stats_continuity(current_stats, previous_stats, None)
        return

    expected_source_checkpoint_id = parent.checkpoint_id if parent is not None else None
    if orchestration.source_checkpoint_id != expected_source_checkpoint_id:
        raise ArtifactValidationError(
            "Orchestration replay source does not match its parent checkpoint"
        )

    recipe = commit.run_recipe
    if recipe is None:
        raise ArtifactValidationError("An orchestration RunCommit requires an immutable run recipe")
    if commit.profile.env_id != "connect4" and (
        recipe.solver_games > 0 or recipe.promotion_metric == "solver_optimal"
    ):
        raise ArtifactValidationError(
            "RunCommit solver evaluation settings require the connect4 profile"
        )
    from ..environment_catalog import get_environment

    max_horizon = get_environment(commit.profile.env_id).capabilities.max_horizon
    if max_horizon is None:
        raise ArtifactValidationError(
            "Synchronized RunCommit environment must declare a finite max_horizon"
        )
    if recipe.temperature_move_threshold != 0 and recipe.temperature_move_threshold >= max_horizon:
        raise ArtifactValidationError(
            "RunCommit temperature threshold is unreachable for its environment"
        )
    model_architecture = recipe.learner_recipe.to_dict()["model_architecture"]
    if not isinstance(model_architecture, dict):
        # RunRecipeV1 already rejects this shape. Keep the transition validator
        # total in case a non-canonical instance reaches this boundary.
        raise ArtifactValidationError("RunCommit learner model architecture is invalid")
    if checkpoints.contract.input("observation").shape != (
        "batch_size",
        model_architecture["observation_elements"],
    ) or checkpoints.contract.output("policy_logits").shape != (
        "batch_size",
        model_architecture["action_count"],
    ):
        raise ArtifactValidationError(
            "RunCommit learner model dimensions disagree with the checkpoint contract"
        )
    if orchestration.iteration > recipe.total_iterations:
        raise ArtifactValidationError("Orchestration iteration exceeds the run's total target")
    if (
        orchestration.episodes_generated != recipe.episodes_per_iteration
        or orchestration.training_steps != recipe.training_steps_per_iteration
        or orchestration.collector_simulations != recipe.simulations_for(orchestration.iteration)
        or orchestration.collector_seed is not None
    ):
        raise ArtifactValidationError("Orchestration facts disagree with the immutable run recipe")
    learner_recipe = recipe.learner_recipe.to_dict()
    batch_size = learner_recipe.get("batch_size")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or batch_size > _MAX_U64
    ):
        raise ArtifactValidationError(
            "run_recipe learner batch_size must be a positive u64 integer"
        )
    if orchestration.training_steps > _MAX_U64 // batch_size:
        raise ArtifactValidationError("orchestration training_steps times batch_size exceeds u64")
    previous_samples = previous_stats.samples_seen if previous_stats is not None else 0
    if current_stats.samples_seen - previous_samples != orchestration.training_steps * batch_size:
        raise ArtifactValidationError(
            "RunCommit samples_seen delta disagrees with training_steps times batch_size"
        )
    if current_stats.total_steps != current_stats.step:
        raise ArtifactValidationError(
            "Orchestration stats total_steps must equal the selected checkpoint step"
        )
    scheduled = recipe.evaluation_scheduled(orchestration.iteration)
    if scheduled != (orchestration.evaluation_id is not None):
        raise ArtifactValidationError(
            "Orchestration evaluation presence disagrees with the run schedule"
        )

    parent_step = (
        checkpoints.read_checkpoint_manifest_exact(parent.checkpoint_id).step
        if parent is not None
        else 0
    )
    if orchestration.training_steps == 0:
        raise ArtifactValidationError(
            "An orchestration RunCommit must record positive training_steps"
        )
    if parent is not None and commit.checkpoint_id == parent.checkpoint_id:
        raise ArtifactValidationError(
            "An orchestration RunCommit must select a newly trained direct child"
        )
    if checkpoint.step - parent_step != orchestration.training_steps:
        raise ArtifactValidationError(
            "RunCommit checkpoint step delta must equal orchestration training_steps"
        )

    if orchestration.evaluation_id is None:
        if (
            commit.champion != inherited_champion
            or commit.evaluation_head_id != inherited_evaluation
        ):
            raise ArtifactValidationError(
                "A non-evaluation iteration must carry evaluation state unchanged"
            )
        _validate_eval_stats_continuity(current_stats, previous_stats, None)
        return

    if prepared_evaluation is not None:
        if prepared_evaluation.evaluation_id != orchestration.evaluation_id:
            raise ArtifactValidationError("Prepared evaluation ID disagrees with its RunCommit")
        evaluations.validate_evidence(prepared_evaluation)
        artifact = prepared_evaluation
        evaluation_id = artifact.evaluation_id
    else:
        evaluation = evaluations.resolve_evaluation(orchestration.evaluation_id)
        artifact = evaluation.artifact
        evaluation_id = evaluation.evaluation_id
    _validate_evaluation_recipe(artifact, recipe)
    if (
        artifact.profile != commit.profile
        or artifact.iteration != orchestration.iteration
        or artifact.candidate_checkpoint_id != commit.checkpoint_id
        or artifact.previous_evaluation_id != inherited_evaluation
        or artifact.champion_before != inherited_champion
        or commit.evaluation_head_id != evaluation_id
        or orchestration.evaluation_seed != artifact.recipe.seed
    ):
        raise ArtifactValidationError(
            "RunCommit evaluation does not match its authoritative transition"
        )
    if artifact.completed_at > orchestration.timestamp:
        raise ArtifactValidationError("RunCommit timestamp precedes evaluation completion")
    observed_win = (
        artifact.results.vs_champion.candidate_win_rate
        if artifact.results.vs_champion is not None
        else None
    )
    observed_draw = (
        artifact.results.vs_champion.draw_rate if artifact.results.vs_champion is not None else None
    )
    if orchestration.eval_win_rate != observed_win or orchestration.eval_draw_rate != observed_draw:
        raise ArtifactValidationError("RunCommit evaluation rates disagree with immutable evidence")
    expected_champion = (
        ChampionReferenceV1(
            checkpoint_id=commit.checkpoint_id,
            evaluation_id=evaluation_id,
        )
        if artifact.decision.promoted
        else inherited_champion
    )
    if commit.champion != expected_champion:
        raise ArtifactValidationError("RunCommit champion disagrees with the evaluation decision")
    if prepared_evaluation is None:
        lineage = evaluations.list_evaluations(commit.evaluation_head_id)
        if not lineage or lineage[-1].evaluation_id != evaluation_id:
            raise ArtifactValidationError("RunCommit evaluation lineage is incomplete")
    _validate_eval_stats_continuity(current_stats, previous_stats, artifact)


class _RunCommitStorage(Protocol):
    def publish_run_commit_bytes(self, run_commit_id: str, data: bytes) -> None: ...

    def read_run_commit_bytes(self, run_commit_id: str) -> bytes: ...


class RunCommitRepository:
    """Resolve, validate, and materialize immutable RunCommit chains."""

    def __init__(
        self,
        checkpoints: CheckpointPublisher,
        evaluations: EvaluationRepository,
    ) -> None:
        self.checkpoints = checkpoints
        self.evaluations = evaluations

    def resolve(self, run_commit_id: str) -> RunCommitRef:
        validate_sha256_digest(run_commit_id, field="run_commit_id")
        storage: _RunCommitStorage = self.checkpoints  # type: ignore[assignment]
        data = storage.read_run_commit_bytes(run_commit_id)
        if sha256_bytes(data) != run_commit_id:
            raise ArtifactValidationError("RunCommit SHA-256 mismatch")
        return RunCommitRef(run_commit_id, RunCommitV1.from_bytes(data))

    def validate_prepared(
        self,
        commit: RunCommitV1,
        evaluation: EvaluationArtifactV2 | None = None,
    ) -> RunCommitV1:
        """Fully validate and normalize an intent without writing any artifact."""
        if not isinstance(commit, RunCommitV1):
            raise TypeError("commit must be RunCommitV1")
        canonical = RunCommitV1.from_bytes(commit.to_bytes())
        if canonical.stats_snapshot.binding != commit.stats_snapshot.binding:
            raise ArtifactValidationError(
                "RunCommit stats wrapper disagrees with its canonical snapshot"
            )
        parent_chain = self.resolve_chain(canonical.parent_run_commit_id)
        parent = parent_chain[-1].commit if parent_chain else None
        if canonical.orchestration is not None:
            prior_scopes = {
                reference.commit.orchestration.collection_scope_id
                for reference in parent_chain
                if reference.commit.orchestration is not None
            }
            if canonical.orchestration.collection_scope_id in prior_scopes:
                raise ArtifactValidationError(
                    "Orchestration replay collection scopes must be unique"
                )
            latest = next(
                (
                    reference.commit.orchestration
                    for reference in reversed(parent_chain)
                    if reference.commit.orchestration is not None
                ),
                None,
            )
            expected_iteration = latest.iteration + 1 if latest is not None else 1
            if canonical.orchestration.iteration != expected_iteration:
                raise ArtifactValidationError(
                    "Orchestration iterations must be contiguous from one"
                )
            if latest is not None and canonical.orchestration.timestamp <= latest.timestamp:
                raise ArtifactValidationError(
                    "Orchestration timestamps must be strictly increasing"
                )
        elif evaluation is not None:
            raise ArtifactValidationError(
                "Standalone RunCommit cannot carry prepared evaluation evidence"
            )
        validate_transition(
            canonical,
            parent,
            checkpoints=self.checkpoints,
            evaluations=self.evaluations,
            prepared_evaluation=evaluation,
        )
        return canonical

    def publish(self, commit: RunCommitV1) -> RunCommitRef:
        commit = self.validate_prepared(commit)
        storage: _RunCommitStorage = self.checkpoints  # type: ignore[assignment]
        storage.publish_run_commit_bytes(commit.run_commit_id, commit.to_bytes())
        return RunCommitRef(commit.run_commit_id, commit)

    def resolve_chain(self, run_commit_id: str | None) -> list[RunCommitRef]:
        if run_commit_id is None:
            return []
        reversed_chain: list[RunCommitRef] = []
        seen: set[str] = set()
        current_id: str | None = run_commit_id
        while current_id is not None:
            if current_id in seen:
                raise ArtifactValidationError("RunCommit lineage contains a cycle")
            seen.add(current_id)
            reference = self.resolve(current_id)
            reversed_chain.append(reference)
            current_id = reference.commit.parent_run_commit_id
        chain = list(reversed(reversed_chain))
        parent: RunCommitV1 | None = None
        latest_orchestration: OrchestrationCommitV1 | None = None
        collection_scopes: set[str] = set()
        for reference in chain:
            validate_transition(
                reference.commit,
                parent,
                checkpoints=self.checkpoints,
                evaluations=self.evaluations,
            )
            orchestration = reference.commit.orchestration
            if orchestration is not None:
                if orchestration.collection_scope_id in collection_scopes:
                    raise ArtifactValidationError(
                        "Orchestration replay collection scopes must be unique"
                    )
                collection_scopes.add(orchestration.collection_scope_id)
                expected_iteration = (
                    latest_orchestration.iteration + 1 if latest_orchestration is not None else 1
                )
                if orchestration.iteration != expected_iteration:
                    raise ArtifactValidationError(
                        "Orchestration iterations must be contiguous from one"
                    )
                if (
                    latest_orchestration is not None
                    and orchestration.timestamp <= latest_orchestration.timestamp
                ):
                    raise ArtifactValidationError(
                        "Orchestration timestamps must be strictly increasing"
                    )
                latest_orchestration = orchestration
            parent = reference.commit
        return chain


__all__ = [
    "OrchestrationCommitV1",
    "LearnerRecipeV1",
    "RunCommitRef",
    "RunCommitRepository",
    "RunCommitV1",
    "RunRecipeV1",
    "validate_transition",
]
