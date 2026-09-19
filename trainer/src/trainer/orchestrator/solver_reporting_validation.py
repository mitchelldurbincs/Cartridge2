"""Pure schema validation for disposable solver-history projections.

These checks keep generated reports internally consistent. Immutable evaluation
artifacts and the selected RunCommit chain remain the only evaluation authority.
"""

from __future__ import annotations

import math
from datetime import datetime

from ..storage.publisher import validate_sha256_digest

_SOLVER_BUCKET_FIELDS = frozenset(
    {
        "positions",
        "value_optimal",
        "exact_best",
        "blunders_win_to_draw",
        "blunders_win_to_loss",
        "blunders_draw_to_loss",
        "forced",
        "value_optimal_rate",
        "exact_best_rate",
        "blunder_rate",
        "forced_rate",
    }
)
_SOLVER_COUNT_FIELDS = (
    "positions",
    "value_optimal",
    "exact_best",
    "blunders_win_to_draw",
    "blunders_win_to_loss",
    "blunders_draw_to_loss",
    "forced",
)
_SOLVER_ENTRY_FIELDS = frozenset(
    {
        "model",
        "model_path",
        "checkpoint_id",
        "step",
        "env_id",
        "opponent",
        "games",
        "seed",
        "temperature",
        "model_wins",
        "model_losses",
        "draws",
        "avg_game_length",
        "positions_scored",
        "forced_moves",
        "forced_move_rate",
        "value_optimal_rate",
        "exact_best_rate",
        "blunder_rate",
        "blunders_win_to_draw",
        "blunders_win_to_loss",
        "blunders_draw_to_loss",
        "by_ply",
        "by_seat",
        "solver_queries",
        "solver_cache_hits",
        "solver_cache_hit_rate",
        "solver_time_seconds",
        "wall_time_seconds",
        "bitbully_version",
        "timestamp",
        "iteration",
        "global_step",
        "context",
        "evaluation_id",
    }
)
SOLVER_PLY_BUCKETS = ("ply_1_8", "ply_9_20", "ply_21_plus")
SOLVER_SEAT_BUCKETS = ("first", "second")


def _exact_dict(value: object, fields: frozenset[str], *, context: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{context} has an invalid schema")
    return value


def _integer(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _number(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ValueError(f"{field} is outside its valid range")
    return result


def _rate(value: object, *, field: str) -> float:
    return _number(value, field=field, minimum=0.0, maximum=1.0)


def _require_equal_rate(actual: object, expected: float, *, field: str) -> None:
    value = _rate(actual, field=field)
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field} does not match its source counts")


def _validate_solver_bucket(value: object, *, field: str) -> dict[str, int]:
    bucket = _exact_dict(value, _SOLVER_BUCKET_FIELDS, context=field)
    counts = {
        name: _integer(bucket[name], field=f"{field}.{name}") for name in _SOLVER_COUNT_FIELDS
    }
    positions = counts["positions"]
    if any(count > positions for name, count in counts.items() if name != "positions"):
        raise ValueError(f"{field} count exceeds positions")
    if counts["exact_best"] > counts["value_optimal"]:
        raise ValueError(f"{field}.exact_best cannot exceed value_optimal")
    if counts["forced"] > counts["exact_best"]:
        raise ValueError(f"{field}.forced cannot exceed exact_best")
    blunders = sum(
        counts[name]
        for name in (
            "blunders_win_to_draw",
            "blunders_win_to_loss",
            "blunders_draw_to_loss",
        )
    )
    if blunders != positions - counts["value_optimal"]:
        raise ValueError(f"{field} blunders do not partition non-optimal positions")

    def expected(count: int) -> float:
        return count / positions if positions else 0.0

    _require_equal_rate(
        bucket["value_optimal_rate"],
        expected(counts["value_optimal"]),
        field=f"{field}.value_optimal_rate",
    )
    _require_equal_rate(
        bucket["exact_best_rate"],
        expected(counts["exact_best"]),
        field=f"{field}.exact_best_rate",
    )
    _require_equal_rate(
        bucket["blunder_rate"],
        expected(blunders),
        field=f"{field}.blunder_rate",
    )
    _require_equal_rate(
        bucket["forced_rate"],
        expected(counts["forced"]),
        field=f"{field}.forced_rate",
    )
    return counts


def validate_solver_entry(value: object, index: int | str) -> dict:
    """Validate one projected record without changing its fields or values."""
    field = f"solver_evaluation[{index}]"
    entry = _exact_dict(value, _SOLVER_ENTRY_FIELDS, context=field)
    for name in ("model", "model_path", "env_id", "opponent"):
        if not isinstance(entry[name], str) or not entry[name]:
            raise ValueError(f"{field}.{name} must be a nonempty string")
    if entry["context"] != "loop":
        raise ValueError(f"{field}.context must be exactly 'loop'")
    for name in ("checkpoint_id", "evaluation_id"):
        try:
            validate_sha256_digest(entry[name], field=f"{field}.{name}")
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    _integer(entry["iteration"], field=f"{field}.iteration", positive=True)
    step = _integer(entry["step"], field=f"{field}.step")
    global_step = _integer(entry["global_step"], field=f"{field}.global_step")
    if step != global_step:
        raise ValueError(f"{field}.step and global_step must match")
    games = _integer(entry["games"], field=f"{field}.games", positive=True)
    outcomes = sum(
        _integer(entry[name], field=f"{field}.{name}")
        for name in ("model_wins", "model_losses", "draws")
    )
    if outcomes != games:
        raise ValueError(f"{field} game outcomes must partition games")
    seed = _integer(entry["seed"], field=f"{field}.seed")
    if seed > 2**64 - 1:
        raise ValueError(f"{field}.seed exceeds u64")
    if seed > 2**64 - games:
        raise ValueError(f"{field}.seed plus game index exceeds u64")
    _number(entry["temperature"], field=f"{field}.temperature", minimum=0.0)
    _number(
        entry["avg_game_length"],
        field=f"{field}.avg_game_length",
        minimum=0.0,
    )

    by_ply = _exact_dict(entry["by_ply"], frozenset(SOLVER_PLY_BUCKETS), context=f"{field}.by_ply")
    by_seat = _exact_dict(
        entry["by_seat"], frozenset(SOLVER_SEAT_BUCKETS), context=f"{field}.by_seat"
    )
    ply_counts = {
        name: _validate_solver_bucket(by_ply[name], field=f"{field}.by_ply.{name}")
        for name in SOLVER_PLY_BUCKETS
    }
    seat_counts = {
        name: _validate_solver_bucket(by_seat[name], field=f"{field}.by_seat.{name}")
        for name in SOLVER_SEAT_BUCKETS
    }
    overall: dict[str, int] = {}
    for count_field in _SOLVER_COUNT_FIELDS:
        from_ply = sum(bucket[count_field] for bucket in ply_counts.values())
        from_seat = sum(bucket[count_field] for bucket in seat_counts.values())
        if from_ply != from_seat:
            raise ValueError(f"{field} solver slices disagree on {count_field}")
        overall[count_field] = from_ply

    positions = overall["positions"]
    if _integer(entry["positions_scored"], field=f"{field}.positions_scored") != positions:
        raise ValueError(f"{field}.positions_scored does not match solver slices")
    if _integer(entry["forced_moves"], field=f"{field}.forced_moves") != overall["forced"]:
        raise ValueError(f"{field}.forced_moves does not match solver slices")
    for name in (
        "blunders_win_to_draw",
        "blunders_win_to_loss",
        "blunders_draw_to_loss",
    ):
        if _integer(entry[name], field=f"{field}.{name}") != overall[name]:
            raise ValueError(f"{field}.{name} does not match solver slices")

    def expected(count: int) -> float:
        return count / positions if positions else 0.0

    _require_equal_rate(
        entry["forced_move_rate"],
        expected(overall["forced"]),
        field=f"{field}.forced_move_rate",
    )
    _require_equal_rate(
        entry["value_optimal_rate"],
        expected(overall["value_optimal"]),
        field=f"{field}.value_optimal_rate",
    )
    _require_equal_rate(
        entry["exact_best_rate"],
        expected(overall["exact_best"]),
        field=f"{field}.exact_best_rate",
    )
    blunders = positions - overall["value_optimal"]
    _require_equal_rate(
        entry["blunder_rate"],
        expected(blunders),
        field=f"{field}.blunder_rate",
    )

    queries = _integer(entry["solver_queries"], field=f"{field}.solver_queries")
    if queries != positions:
        raise ValueError(f"{field}.solver_queries must equal positions_scored")
    hits = _integer(entry["solver_cache_hits"], field=f"{field}.solver_cache_hits")
    if hits > queries:
        raise ValueError(f"{field}.solver_cache_hits cannot exceed solver_queries")
    _require_equal_rate(
        entry["solver_cache_hit_rate"],
        hits / queries if queries else 0.0,
        field=f"{field}.solver_cache_hit_rate",
    )
    _number(
        entry["solver_time_seconds"],
        field=f"{field}.solver_time_seconds",
        minimum=0.0,
    )
    _number(
        entry["wall_time_seconds"],
        field=f"{field}.wall_time_seconds",
        minimum=0.0,
    )
    version = entry["bitbully_version"]
    if version is not None and (not isinstance(version, str) or not version):
        raise ValueError(f"{field}.bitbully_version must be null or nonempty")
    timestamp = entry["timestamp"]
    if not isinstance(timestamp, str) or not timestamp:
        raise ValueError(f"{field}.timestamp must be a nonempty timestamp")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field}.timestamp is invalid") from exc
    return entry


def validate_solver_history(value: object) -> list[dict]:
    """Validate records, unique evidence IDs, and strictly increasing iterations."""
    if not isinstance(value, list):
        raise ValueError("solver_evaluations must be a list")
    entries = [validate_solver_entry(entry, index) for index, entry in enumerate(value)]
    evaluation_ids = [entry["evaluation_id"] for entry in entries]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("Solver evaluation identifiers must be unique")
    iterations = [entry["iteration"] for entry in entries]
    if any(current <= previous for previous, current in zip(iterations, iterations[1:])):
        raise ValueError("Solver evaluation iterations must be strictly increasing")
    return entries


__all__ = [
    "SOLVER_PLY_BUCKETS",
    "SOLVER_SEAT_BUCKETS",
    "validate_solver_entry",
    "validate_solver_history",
]
