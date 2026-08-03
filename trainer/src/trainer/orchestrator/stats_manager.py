"""Strict projections from durable loop/evaluation evidence."""

from __future__ import annotations

import math
import re
from datetime import datetime

from crucible.orchestrator.config import IterationStats

from ..stats import (
    LoadedStatsSnapshotV2,
    PreparedStatsSnapshotV2,
    write_stats_projection,
)
from ..storage.publisher import validate_sha256_digest
from .config import LoopConfig
from .durable_json import write_json_durable
from .eval_reporting import _validate_solver_history

_EVAL_RECORD_FIELDS = frozenset(
    {
        "iteration",
        "step",
        "candidate_checkpoint_id",
        "evaluation_id",
        "vs_champion_checkpoint_id",
        "vs_champion_evaluation_id",
        "vs_champion_win_rate",
        "vs_champion_draw_rate",
        "vs_champion_average_game_length",
        "vs_champion_iteration",
        "promoted",
        "promotion_reason",
        "vs_random_win_rate",
        "vs_random_draw_rate",
        "vs_random_average_game_length",
        "solver_value_optimal_rate",
        "solver_exact_best_rate",
        "solver_blunder_rate",
        "solver_positions",
        "promotion_metric",
        "requested_vs_champion_games",
        "requested_vs_random_games",
        "timestamp",
    }
)
_UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")


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
    number = float(value)
    if (
        not math.isfinite(number)
        or (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
    ):
        raise ValueError(f"{field} is outside its valid range")
    return number


def _optional_rate(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _number(value, field=field, minimum=0.0, maximum=1.0)


def _digest(value: object, *, field: str) -> str:
    try:
        return validate_sha256_digest(value, field=field)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def _timestamp(value: object, *, field: str, require_utc: bool) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a timestamp string")
    if require_utc and _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"{field} must use the evaluation UTC timestamp format")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid timestamp") from exc
    return value


def _validate_eval_record(record: dict, index: int) -> None:
    prefix = f"evaluation[{index}]"
    _integer(record["iteration"], field=f"{prefix}.iteration", positive=True)
    _integer(record["step"], field=f"{prefix}.step")
    _digest(
        record["candidate_checkpoint_id"], field=f"{prefix}.candidate_checkpoint_id"
    )
    _digest(record["evaluation_id"], field=f"{prefix}.evaluation_id")
    champion_checkpoint = record["vs_champion_checkpoint_id"]
    champion_evaluation = record["vs_champion_evaluation_id"]
    if (champion_checkpoint is None) != (champion_evaluation is None):
        raise ValueError(f"{prefix} champion identifiers must both be null or present")
    if champion_checkpoint is not None:
        _digest(champion_checkpoint, field=f"{prefix}.vs_champion_checkpoint_id")
        _digest(champion_evaluation, field=f"{prefix}.vs_champion_evaluation_id")
    champion_win = _optional_rate(
        record["vs_champion_win_rate"], field=f"{prefix}.vs_champion_win_rate"
    )
    champion_draw = _optional_rate(
        record["vs_champion_draw_rate"], field=f"{prefix}.vs_champion_draw_rate"
    )
    champion_length = record["vs_champion_average_game_length"]
    champion_iteration = record["vs_champion_iteration"]
    champion_games = _integer(
        record["requested_vs_champion_games"],
        field=f"{prefix}.requested_vs_champion_games",
    )
    champion_values = (
        champion_win,
        champion_draw,
        champion_length,
        champion_iteration,
    )
    if champion_checkpoint is None:
        if champion_games != 0 or any(value is not None for value in champion_values):
            raise ValueError(f"{prefix} has champion results without a champion")
    else:
        if champion_games == 0 or any(value is None for value in champion_values):
            raise ValueError(f"{prefix} has incomplete champion results")
        _number(
            champion_length,
            field=f"{prefix}.vs_champion_average_game_length",
            minimum=0.0,
        )
        _integer(
            champion_iteration, field=f"{prefix}.vs_champion_iteration", positive=True
        )
        if champion_win + champion_draw > 1.0:
            raise ValueError(f"{prefix} champion rates exceed one")

    random_win = _optional_rate(
        record["vs_random_win_rate"], field=f"{prefix}.vs_random_win_rate"
    )
    random_draw = _optional_rate(
        record["vs_random_draw_rate"], field=f"{prefix}.vs_random_draw_rate"
    )
    random_length = record["vs_random_average_game_length"]
    random_games = _integer(
        record["requested_vs_random_games"],
        field=f"{prefix}.requested_vs_random_games",
    )
    if random_games == 0:
        if any(value is not None for value in (random_win, random_draw, random_length)):
            raise ValueError(f"{prefix} has random results with zero requested games")
    else:
        if any(value is None for value in (random_win, random_draw, random_length)):
            raise ValueError(f"{prefix} has incomplete random results")
        _number(
            random_length, field=f"{prefix}.vs_random_average_game_length", minimum=0.0
        )
        if random_win + random_draw > 1.0:
            raise ValueError(f"{prefix} random rates exceed one")

    if not isinstance(record["promoted"], bool):
        raise ValueError(f"{prefix}.promoted must be boolean")
    if (
        not isinstance(record["promotion_reason"], str)
        or not record["promotion_reason"].strip()
    ):
        raise ValueError(f"{prefix}.promotion_reason must be nonempty")
    if record["promotion_metric"] not in {"win_rate", "solver_optimal"}:
        raise ValueError(f"{prefix}.promotion_metric is invalid")
    solver_rate = _optional_rate(
        record["solver_value_optimal_rate"],
        field=f"{prefix}.solver_value_optimal_rate",
    )
    solver_exact = _optional_rate(
        record["solver_exact_best_rate"], field=f"{prefix}.solver_exact_best_rate"
    )
    solver_blunder = _optional_rate(
        record["solver_blunder_rate"], field=f"{prefix}.solver_blunder_rate"
    )
    solver_positions = record["solver_positions"]
    if solver_positions is None:
        if any(
            value is not None for value in (solver_rate, solver_exact, solver_blunder)
        ):
            raise ValueError(f"{prefix} has incomplete solver results")
    else:
        positions = _integer(solver_positions, field=f"{prefix}.solver_positions")
        if any(value is None for value in (solver_rate, solver_exact, solver_blunder)):
            raise ValueError(f"{prefix} has incomplete solver results")
        if solver_exact > solver_rate:
            raise ValueError(f"{prefix} solver exact-best rate exceeds optimal rate")
        if positions == 0 and (solver_rate != 0.0 or solver_exact != 0.0):
            raise ValueError(f"{prefix} zero-position solver rates must be zero")
        expected_blunder = 1.0 - solver_rate if positions else 0.0
        if not math.isclose(
            solver_blunder, expected_blunder, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{prefix} solver blunder rate does not complement optimal rate"
            )
    _timestamp(record["timestamp"], field=f"{prefix}.timestamp", require_utc=True)


def _validate_eval_lineage(records: list[dict]) -> None:
    champion: dict | None = None
    for index, record in enumerate(records):
        if champion is None:
            if record["vs_champion_checkpoint_id"] is not None:
                raise ValueError(
                    f"Evaluation record {index} refers to a champion outside history"
                )
            if not record["promoted"]:
                raise ValueError(
                    f"Evaluation record {index} must establish the first champion"
                )
        else:
            expected = (
                champion["candidate_checkpoint_id"],
                champion["evaluation_id"],
                champion["iteration"],
            )
            actual = (
                record["vs_champion_checkpoint_id"],
                record["vs_champion_evaluation_id"],
                record["vs_champion_iteration"],
            )
            if actual != expected:
                raise ValueError(
                    f"Evaluation record {index} does not reference the latest champion"
                )
        if record["promoted"]:
            champion = record


def _require_eval_history(value: object) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("Evaluation history must be a list")
    records: list[dict] = []
    for index, record in enumerate(value):
        if not isinstance(record, dict) or set(record) != _EVAL_RECORD_FIELDS:
            raise ValueError(f"Evaluation record {index} has an invalid schema")
        _validate_eval_record(record, index)
        records.append(record)
    evaluation_ids = [record["evaluation_id"] for record in records]
    if len(evaluation_ids) != len(set(evaluation_ids)):
        raise ValueError("Evaluation identifiers must be unique")
    eval_iterations = [record["iteration"] for record in records]
    if any(
        current <= previous
        for previous, current in zip(eval_iterations, eval_iterations[1:])
    ):
        raise ValueError("Evaluation iterations must be strictly increasing")
    _validate_eval_lineage(records)
    return records


def _validate_resume_relationship(
    history: list[IterationStats], eval_history: list[dict]
) -> None:
    loop_by_iteration = {item.iteration: item for item in history}
    if len(loop_by_iteration) != len(history):
        raise ValueError("Loop iteration identities must be unique")

    for index, record in enumerate(eval_history):
        loop_record = loop_by_iteration.get(record["iteration"])
        if loop_record is None:
            raise ValueError(
                f"Evaluation record {index} has no completed loop iteration"
            )
        expected_rates = (
            record["vs_champion_win_rate"],
            record["vs_champion_draw_rate"],
        )
        observed_rates = (
            loop_record.eval_win_rate,
            loop_record.eval_draw_rate,
        )
        if observed_rates != expected_rates:
            raise ValueError(
                f"Evaluation record {index} disagrees with its loop iteration"
            )


class StatsManager:
    """Rebuild disposable projections from a validated RunCommit chain."""

    def __init__(self, config: LoopConfig) -> None:
        self.config = config

    def save_loop_stats(self, history: list[IterationStats]) -> None:
        stats = {
            "config": {
                "env_id": self.config.env_id,
                "episodes_per_iteration": self.config.episodes_per_iteration,
                "steps_per_iteration": self.config.steps_per_iteration,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "eval_interval": self.config.eval_interval,
                "eval_games": self.config.eval_games,
            },
            "iterations": [
                {
                    "iteration": item.iteration,
                    "episodes": item.episodes_generated,
                    "transitions": item.transitions_generated,
                    "steps": item.training_steps,
                    "actor_time": item.actor_time_seconds,
                    "trainer_time": item.trainer_time_seconds,
                    "eval_time": item.eval_time_seconds,
                    "total_time": item.total_time_seconds,
                    "eval_win_rate": item.eval_win_rate,
                    "eval_draw_rate": item.eval_draw_rate,
                    "timestamp": item.timestamp,
                }
                for item in history
            ],
        }
        write_json_durable(self.config.loop_stats_path, stats)

    def save_eval_stats(self, eval_history: list[dict]) -> None:
        records = _require_eval_history(eval_history)
        write_json_durable(self.config.eval_stats_path, {"evaluations": records})

    def save_solver_stats(self, solver_history: list[dict]) -> None:
        records = _validate_solver_history(solver_history)
        write_json_durable(
            self.config.solver_stats_path,
            {"solver_evaluations": records},
        )

    def rebuild_projections(
        self,
        *,
        history: list[IterationStats],
        eval_history: list[dict],
        solver_history: list[dict],
        stats_snapshot: PreparedStatsSnapshotV2 | LoadedStatsSnapshotV2,
    ) -> None:
        """Rebuild every mutable view from one validated RunCommit chain."""
        records = _require_eval_history(eval_history)
        _validate_resume_relationship(
            history,
            records,
        )
        self.save_loop_stats(history)
        self.save_eval_stats(records)
        self.save_solver_stats(solver_history)
        write_stats_projection(stats_snapshot, self.config.stats_path)


__all__ = ["StatsManager"]
