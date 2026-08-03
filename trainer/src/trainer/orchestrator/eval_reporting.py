"""Reporting projections derived from immutable evaluation artifacts."""

from __future__ import annotations

import math
from datetime import datetime

from ..storage.evaluation import EvaluationRef, SolverBucketV1
from ..storage.publisher import ArtifactValidationError, validate_sha256_digest

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
_PLY_BUCKETS = ("ply_1_8", "ply_9_20", "ply_21_plus")
_SEAT_BUCKETS = ("first", "second")


def _exact_dict(value: object, fields: frozenset[str], *, context: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{context} has an invalid schema")
    return value


def _integer(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{field} must be a {'positive' if positive else 'nonnegative'} integer"
        )
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
        name: _integer(bucket[name], field=f"{field}.{name}")
        for name in _SOLVER_COUNT_FIELDS
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


def _validate_solver_entry(value: object, index: int | str) -> dict:
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

    by_ply = _exact_dict(
        entry["by_ply"], frozenset(_PLY_BUCKETS), context=f"{field}.by_ply"
    )
    by_seat = _exact_dict(
        entry["by_seat"], frozenset(_SEAT_BUCKETS), context=f"{field}.by_seat"
    )
    ply_counts = {
        name: _validate_solver_bucket(by_ply[name], field=f"{field}.by_ply.{name}")
        for name in _PLY_BUCKETS
    }
    seat_counts = {
        name: _validate_solver_bucket(by_seat[name], field=f"{field}.by_seat.{name}")
        for name in _SEAT_BUCKETS
    }
    overall: dict[str, int] = {}
    for count_field in _SOLVER_COUNT_FIELDS:
        from_ply = sum(bucket[count_field] for bucket in ply_counts.values())
        from_seat = sum(bucket[count_field] for bucket in seat_counts.values())
        if from_ply != from_seat:
            raise ValueError(f"{field} solver slices disagree on {count_field}")
        overall[count_field] = from_ply

    positions = overall["positions"]
    if (
        _integer(entry["positions_scored"], field=f"{field}.positions_scored")
        != positions
    ):
        raise ValueError(f"{field}.positions_scored does not match solver slices")
    if (
        _integer(entry["forced_moves"], field=f"{field}.forced_moves")
        != overall["forced"]
    ):
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


def _validate_solver_history(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("solver_evaluations must be a list")
    entries = [
        _validate_solver_entry(entry, index) for index, entry in enumerate(value)
    ]
    evaluation_ids = [entry["evaluation_id"] for entry in entries]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("Solver evaluation identifiers must be unique")
    iterations = [entry["iteration"] for entry in entries]
    if any(
        current <= previous for previous, current in zip(iterations, iterations[1:])
    ):
        raise ValueError("Solver evaluation iterations must be strictly increasing")
    return entries


class EvalReportingMixin:
    """Project durable evidence into UI history, solver history, and W&B."""

    def _build_eval_record(self, evaluation: EvaluationRef) -> dict:
        artifact = evaluation.artifact
        vs_champion = artifact.results.vs_champion
        vs_random = artifact.results.vs_random
        solver = artifact.results.candidate_solver
        candidate_checkpoint = self.checkpoints.read_checkpoint_manifest_exact(
            artifact.candidate_checkpoint_id
        )
        champion_reference = artifact.champion_before
        champion_evaluation = None
        if champion_reference is not None:
            champion_evaluation = self.evaluations.resolve_evaluation(
                champion_reference.evaluation_id
            )
            if (
                champion_evaluation.artifact.candidate_checkpoint_id
                != champion_reference.checkpoint_id
            ):
                raise ArtifactValidationError(
                    "Evaluation record champion reference is inconsistent"
                )
        return {
            "iteration": artifact.iteration,
            "step": candidate_checkpoint.step,
            "candidate_checkpoint_id": artifact.candidate_checkpoint_id,
            "evaluation_id": evaluation.evaluation_id,
            "vs_champion_checkpoint_id": (
                champion_reference.checkpoint_id
                if champion_reference is not None
                else None
            ),
            "vs_champion_evaluation_id": (
                champion_reference.evaluation_id
                if champion_reference is not None
                else None
            ),
            "vs_champion_win_rate": (
                vs_champion.candidate_win_rate if vs_champion is not None else None
            ),
            "vs_champion_draw_rate": (
                vs_champion.draw_rate if vs_champion is not None else None
            ),
            "vs_champion_average_game_length": (
                vs_champion.average_game_length if vs_champion is not None else None
            ),
            "vs_champion_iteration": (
                champion_evaluation.artifact.iteration
                if champion_evaluation is not None
                else None
            ),
            "promoted": artifact.decision.promoted,
            "promotion_reason": artifact.decision.reason,
            "vs_random_win_rate": (
                vs_random.candidate_win_rate if vs_random is not None else None
            ),
            "vs_random_draw_rate": (
                vs_random.draw_rate if vs_random is not None else None
            ),
            "vs_random_average_game_length": (
                vs_random.average_game_length if vs_random is not None else None
            ),
            "solver_value_optimal_rate": (
                solver.overall.value_optimal_rate if solver is not None else None
            ),
            "solver_exact_best_rate": (
                solver.overall.exact_best / solver.overall.positions
                if solver is not None and solver.overall.positions
                else (0.0 if solver is not None else None)
            ),
            "solver_blunder_rate": (
                (
                    solver.overall.blunders_win_to_draw
                    + solver.overall.blunders_win_to_loss
                    + solver.overall.blunders_draw_to_loss
                )
                / solver.overall.positions
                if solver is not None and solver.overall.positions
                else (0.0 if solver is not None else None)
            ),
            "solver_positions": (
                solver.overall.positions if solver is not None else None
            ),
            "promotion_metric": artifact.recipe.promotion_metric,
            "requested_vs_champion_games": (
                artifact.recipe.requested_games.vs_champion
            ),
            "requested_vs_random_games": artifact.recipe.requested_games.vs_random,
            "timestamp": artifact.completed_at,
        }

    @staticmethod
    def _solver_bucket_projection(bucket: SolverBucketV1) -> dict[str, int | float]:
        positions = bucket.positions
        blunders = (
            bucket.blunders_win_to_draw
            + bucket.blunders_win_to_loss
            + bucket.blunders_draw_to_loss
        )

        def rate(count: int) -> float:
            return count / positions if positions else 0.0

        return {
            **bucket.to_dict(),
            "value_optimal_rate": rate(bucket.value_optimal),
            "exact_best_rate": rate(bucket.exact_best),
            "blunder_rate": rate(blunders),
            "forced_rate": rate(bucket.forced),
        }

    def _build_solver_record(self, evaluation: EvaluationRef) -> dict | None:
        artifact = evaluation.artifact
        solver = artifact.results.candidate_solver
        if solver is None:
            return None
        checkpoint = self.checkpoints.read_checkpoint_manifest_exact(
            artifact.candidate_checkpoint_id
        )
        overall = solver.overall
        blunders = (
            overall.blunders_win_to_draw
            + overall.blunders_win_to_loss
            + overall.blunders_draw_to_loss
        )

        def rate(count: int) -> float:
            return count / overall.positions if overall.positions else 0.0

        entry = {
            "model": f"checkpoint:{artifact.candidate_checkpoint_id}",
            "model_path": f"checkpoint:{artifact.candidate_checkpoint_id}",
            "checkpoint_id": artifact.candidate_checkpoint_id,
            "step": checkpoint.step,
            "env_id": artifact.profile.env_id,
            "opponent": "random_v1",
            "games": solver.games_played,
            "seed": artifact.recipe.seed,
            "temperature": 0.0,
            "model_wins": solver.candidate_wins,
            "model_losses": solver.opponent_wins,
            "draws": solver.draws,
            "avg_game_length": solver.average_game_length,
            "positions_scored": overall.positions,
            "forced_moves": overall.forced,
            "forced_move_rate": rate(overall.forced),
            "value_optimal_rate": rate(overall.value_optimal),
            "exact_best_rate": rate(overall.exact_best),
            "blunder_rate": rate(blunders),
            "blunders_win_to_draw": overall.blunders_win_to_draw,
            "blunders_win_to_loss": overall.blunders_win_to_loss,
            "blunders_draw_to_loss": overall.blunders_draw_to_loss,
            "by_ply": {
                name: self._solver_bucket_projection(solver.by_ply[name])
                for name in _PLY_BUCKETS
            },
            "by_seat": {
                name: self._solver_bucket_projection(solver.by_seat[name])
                for name in _SEAT_BUCKETS
            },
            "solver_queries": solver.solver_queries,
            "solver_cache_hits": solver.solver_cache_hits,
            "solver_cache_hit_rate": (
                solver.solver_cache_hits / solver.solver_queries
                if solver.solver_queries
                else 0.0
            ),
            "solver_time_seconds": solver.solver_time_seconds,
            "wall_time_seconds": solver.wall_time_seconds,
            "bitbully_version": solver.solver_version,
            "timestamp": artifact.completed_at,
            "iteration": artifact.iteration,
            "global_step": checkpoint.manifest.step,
            "context": "loop",
            "evaluation_id": evaluation.evaluation_id,
        }
        _validate_solver_entry(entry, "projection")
        return entry

    def _build_solver_history(self, evaluations: list[EvaluationRef]) -> list[dict]:
        records = [
            record
            for evaluation in evaluations
            if (record := self._build_solver_record(evaluation)) is not None
        ]
        return _validate_solver_history(records)

    def _log_eval_to_wandb(self, eval_record: dict) -> None:
        if self.wandb_logger is None:
            return
        metrics = {
            "eval/vs_champion_win_rate": eval_record["vs_champion_win_rate"],
            "eval/vs_champion_draw_rate": eval_record["vs_champion_draw_rate"],
            "eval/promoted": int(eval_record["promoted"]),
            "eval/champion_iteration": self.champion_iteration,
            "eval/vs_random_win_rate": eval_record["vs_random_win_rate"],
            "eval/vs_random_draw_rate": eval_record["vs_random_draw_rate"],
            "solver/value_optimal_rate": eval_record["solver_value_optimal_rate"],
            "solver/exact_best_rate": eval_record["solver_exact_best_rate"],
            "solver/blunder_rate": eval_record["solver_blunder_rate"],
            "solver/positions_scored": eval_record["solver_positions"],
        }
        self.wandb_logger.log(
            {key: value for key, value in metrics.items() if value is not None},
            step=eval_record["step"],
        )


__all__ = ["EvalReportingMixin"]
