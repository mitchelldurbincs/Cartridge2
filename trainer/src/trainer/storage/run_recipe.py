"""Immutable learner and run recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .artifact_codec import (
    ArtifactValidationError,
    canonical_json_bytes,
    sha256_bytes,
    validate_sha256_digest,
)
from .run_commit_codec import (
    _LEARNER_RECIPE_FIELDS,
    _MAX_U64,
    _MODEL_ARCHITECTURE_FIELDS,
    _RUN_RECIPE_FIELDS,
    _decode_canonical,
    _exact,
    _f32_number,
    _integer,
    _number,
    _optional_rate,
    _u32,
)


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
        self._validate_identity_and_counts()
        self._normalize_collection_recipe()
        self._normalize_evaluation_recipe()
        _validate_alphazero_learner_recipe(
            self.learner_recipe.to_dict(),
            total_iterations=self.total_iterations,
            training_steps_per_iteration=self.training_steps_per_iteration,
            replay_policy=self.replay_policy,
        )

    def _validate_identity_and_counts(self) -> None:
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
        if self.collector_seed_strategy != "system_entropy_v1":
            raise ArtifactValidationError("run_recipe.collector_seed_strategy is unsupported")
        if self.replay_policy != "scoped_fresh_iteration_v1":
            raise ArtifactValidationError("run_recipe.replay_policy is unsupported")

    def _normalize_collection_recipe(self) -> None:
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
        _u32(self.mcts_simulation_ramp, field="run_recipe.mcts_simulation_ramp")
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
        _u32(self.temperature_move_threshold, field="run_recipe.temperature_move_threshold")
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

    def _normalize_evaluation_recipe(self) -> None:
        _integer(self.evaluation_interval, field="run_recipe.evaluation_interval")
        games = _u32(
            self.evaluation_games,
            field="run_recipe.evaluation_games",
            positive=True,
        )
        _u32(self.evaluation_simulations, field="run_recipe.evaluation_simulations")
        temperature = _f32_number(
            self.evaluation_temperature,
            field="run_recipe.evaluation_temperature",
        )
        object.__setattr__(self, "evaluation_temperature", temperature)
        threshold = _optional_rate(
            self.evaluation_win_threshold,
            field="run_recipe.evaluation_win_threshold",
        )
        margin = _optional_rate(self.promotion_margin, field="run_recipe.promotion_margin")
        assert threshold is not None and margin is not None
        object.__setattr__(self, "evaluation_win_threshold", threshold)
        object.__setattr__(self, "promotion_margin", margin)
        if not isinstance(self.evaluation_vs_random, bool):
            raise ArtifactValidationError("run_recipe.evaluation_vs_random must be boolean")
        solver_games = _u32(self.solver_games, field="run_recipe.solver_games")
        seed = _integer(self.evaluation_seed, field="run_recipe.evaluation_seed")
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
