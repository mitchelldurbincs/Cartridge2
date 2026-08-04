"""Primitive validation for evaluation artifacts."""

from __future__ import annotations

import math
import re
import struct
from datetime import datetime, timezone
from typing import Any, Mapping

from .artifact_codec import ArtifactValidationError, canonical_json_bytes

_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_MAX_F32 = float.fromhex("0x1.fffffep+127")
_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")
_PROFILE_FIELDS = frozenset(
    {
        "algorithm_id",
        "env_id",
        "env_contract_version",
        "model_artifact_schema_version",
        "model_contract",
    }
)
_REFERENCE_FIELDS = frozenset({"checkpoint_id", "evaluation_id"})
_REQUESTED_GAMES_FIELDS = frozenset(
    {"vs_champion", "vs_random", "candidate_solver", "champion_solver"}
)
_RECIPE_FIELDS = frozenset(
    {
        "simulations",
        "temperature",
        "promotion_metric",
        "promotion_margin",
        "win_threshold",
        "seed",
        "seat_schedule",
        "requested_games",
    }
)
_SEAT_SCHEDULE = {
    "kind": "alternating_v1",
    "candidate_first": "even_game_indices",
    "seed_rule": "base_plus_game_index",
}
_HEAD_TO_HEAD_FIELDS = frozenset(
    {
        "games_played",
        "candidate_wins",
        "opponent_wins",
        "draws",
        "candidate_wins_as_first",
        "candidate_wins_as_second",
        "opponent_wins_while_candidate_first",
        "opponent_wins_while_candidate_second",
        "average_game_length",
    }
)
_BUCKET_FIELDS = frozenset(
    {
        "positions",
        "value_optimal",
        "exact_best",
        "blunders_win_to_draw",
        "blunders_win_to_loss",
        "blunders_draw_to_loss",
        "forced",
    }
)
_SOLVER_FIELDS = frozenset(
    {
        "games_played",
        "candidate_wins",
        "opponent_wins",
        "draws",
        "average_game_length",
        "overall",
        "by_ply",
        "by_seat",
        "solver_queries",
        "solver_cache_hits",
        "solver_time_seconds",
        "wall_time_seconds",
        "solver_version",
    }
)
_RESULTS_FIELDS = frozenset({"vs_champion", "vs_random", "candidate_solver", "champion_solver"})
_DECISION_FIELDS = frozenset({"promoted", "reason"})
_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "profile",
        "iteration",
        "candidate_checkpoint_id",
        "previous_evaluation_id",
        "champion_before",
        "recipe",
        "results",
        "decision",
        "started_at",
        "completed_at",
    }
)


def utc_timestamp() -> str:
    """Return the strict UTC timestamp representation stored in artifacts."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _exact(value: object, fields: frozenset[str], *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{context} must be a JSON object")
    actual = frozenset(value)
    if actual != fields:
        raise ArtifactValidationError(
            f"{context} fields must be exact "
            f"(missing={sorted(fields - actual)}, extra={sorted(actual - fields)})"
        )
    return value


def _integer(
    value: object,
    *,
    field: str,
    positive: bool = False,
    maximum: int = _MAX_U64,
) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        qualifier = "positive" if positive else "nonnegative"
        raise ArtifactValidationError(f"{field} must be a {qualifier} integer")
    return value


def _finite(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{field} must be a finite number")
    number = float(value)
    if (
        not math.isfinite(number)
        or (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
    ):
        raise ArtifactValidationError(f"{field} is outside its valid range")
    # JSON distinguishes ``-0.0`` from ``0.0`` even though the values have the
    # same domain meaning.  Collapse both spellings before content-addressing.
    return 0.0 if number == 0.0 else number


def _finite_f32(value: object, *, field: str, minimum: float = 0.0) -> float:
    number = _finite(value, field=field, minimum=minimum, maximum=_MAX_F32)
    return float(struct.unpack("!f", struct.pack("!f", number))[0])


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{field} must be a non-empty string")
    return value


def _timestamp(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ArtifactValidationError(f"{field} must be a UTC timestamp with six fractional digits")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactValidationError(f"{field} is not a valid timestamp") from exc
    return value


def _decode_canonical(data: bytes, *, context: str) -> object:
    import json

    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"{context} is not valid UTF-8 JSON") from exc
    if canonical_json_bytes(value) != data:
        raise ArtifactValidationError(f"{context} is not canonical JSON")
    return value
