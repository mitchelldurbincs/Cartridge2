"""Canonical RunCommit validation primitives."""

from __future__ import annotations

import json
import math
import re
import struct
from datetime import datetime
from typing import Any, Mapping

from .artifact_codec import ArtifactValidationError, canonical_json_bytes

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
