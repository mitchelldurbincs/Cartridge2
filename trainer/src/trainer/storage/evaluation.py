"""Immutable evaluation evidence selected atomically by a RunCommit."""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from .publisher import (
    ArtifactValidationError,
    CheckpointProfileV1,
    CheckpointPublisher,
    FilesystemCheckpointPublisher,
    S3CheckpointPublisher,
    _create_or_verify,
    canonical_json_bytes,
    sha256_bytes,
    validate_sha256_digest,
)

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
_RESULTS_FIELDS = frozenset(
    {"vs_champion", "vs_random", "candidate_solver", "champion_solver"}
)
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
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


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
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
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
        raise ArtifactValidationError(
            f"{field} must be a UTC timestamp with six fractional digits"
        )
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


@dataclass(frozen=True)
class ChampionReferenceV1:
    checkpoint_id: str
    evaluation_id: str

    def __post_init__(self) -> None:
        validate_sha256_digest(self.checkpoint_id, field="champion.checkpoint_id")
        validate_sha256_digest(self.evaluation_id, field="champion.evaluation_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "evaluation_id": self.evaluation_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ChampionReferenceV1":
        data = _exact(value, _REFERENCE_FIELDS, context="champion reference")
        return cls(
            checkpoint_id=validate_sha256_digest(
                data["checkpoint_id"], field="champion.checkpoint_id"
            ),
            evaluation_id=validate_sha256_digest(
                data["evaluation_id"], field="champion.evaluation_id"
            ),
        )


@dataclass(frozen=True)
class RequestedGamesV1:
    vs_champion: int
    vs_random: int
    candidate_solver: int
    champion_solver: int

    def __post_init__(self) -> None:
        for field in _REQUESTED_GAMES_FIELDS:
            _integer(
                getattr(self, field),
                field=f"requested_games.{field}",
                maximum=_MAX_U32,
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "vs_champion": self.vs_champion,
            "vs_random": self.vs_random,
            "candidate_solver": self.candidate_solver,
            "champion_solver": self.champion_solver,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RequestedGamesV1":
        data = _exact(value, _REQUESTED_GAMES_FIELDS, context="requested games")
        return cls(
            **{
                field: _integer(
                    data[field],
                    field=f"requested_games.{field}",
                    maximum=_MAX_U32,
                )
                for field in _REQUESTED_GAMES_FIELDS
            }
        )


@dataclass(frozen=True)
class EvaluationRecipeV1:
    simulations: int
    temperature: float
    promotion_metric: str
    promotion_margin: float
    win_threshold: float
    seed: int
    requested_games: RequestedGamesV1

    def __post_init__(self) -> None:
        _integer(self.simulations, field="recipe.simulations", maximum=_MAX_U32)
        object.__setattr__(
            self,
            "temperature",
            _finite_f32(
                self.temperature,
                field="recipe.temperature",
            ),
        )
        if self.promotion_metric not in {"win_rate", "solver_optimal"}:
            raise ArtifactValidationError(
                "recipe.promotion_metric must be 'win_rate' or 'solver_optimal'"
            )
        object.__setattr__(
            self,
            "promotion_margin",
            _finite(
                self.promotion_margin,
                field="recipe.promotion_margin",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "win_threshold",
            _finite(
                self.win_threshold,
                field="recipe.win_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        if self.promotion_metric == "win_rate":
            if self.promotion_margin != 0.0:
                raise ArtifactValidationError(
                    "recipe.promotion_margin must be zero for win_rate promotion"
                )
        elif self.win_threshold != 0.0:
            raise ArtifactValidationError(
                "recipe.win_threshold must be zero for solver_optimal promotion"
            )
        _integer(self.seed, field="recipe.seed")
        largest_run = max(
            self.requested_games.vs_champion,
            self.requested_games.vs_random,
            self.requested_games.candidate_solver,
            self.requested_games.champion_solver,
        )
        if largest_run and self.seed > _MAX_U64 - (largest_run - 1):
            raise ArtifactValidationError("recipe.seed plus the game index exceeds u64")
        if (
            self.requested_games.vs_champion
            + self.requested_games.vs_random
            + self.requested_games.candidate_solver
            == 0
        ):
            raise ArtifactValidationError(
                "An evaluation recipe must request candidate evidence"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "simulations": self.simulations,
            "temperature": self.temperature,
            "promotion_metric": self.promotion_metric,
            "promotion_margin": self.promotion_margin,
            "win_threshold": self.win_threshold,
            "seed": self.seed,
            "seat_schedule": dict(_SEAT_SCHEDULE),
            "requested_games": self.requested_games.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "EvaluationRecipeV1":
        data = _exact(value, _RECIPE_FIELDS, context="evaluation recipe")
        schedule = _exact(
            data["seat_schedule"], frozenset(_SEAT_SCHEDULE), context="seat schedule"
        )
        if dict(schedule) != _SEAT_SCHEDULE:
            raise ArtifactValidationError(
                "seat schedule must be deterministic alternating_v1"
            )
        metric = _string(data["promotion_metric"], field="recipe.promotion_metric")
        return cls(
            simulations=_integer(
                data["simulations"],
                field="recipe.simulations",
                maximum=_MAX_U32,
            ),
            temperature=_finite_f32(
                data["temperature"],
                field="recipe.temperature",
            ),
            promotion_metric=metric,
            promotion_margin=_finite(
                data["promotion_margin"],
                field="recipe.promotion_margin",
                minimum=0.0,
                maximum=1.0,
            ),
            win_threshold=_finite(
                data["win_threshold"],
                field="recipe.win_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
            seed=_integer(data["seed"], field="recipe.seed"),
            requested_games=RequestedGamesV1.from_dict(data["requested_games"]),
        )


@dataclass(frozen=True)
class HeadToHeadResultV1:
    games_played: int
    candidate_wins: int
    opponent_wins: int
    draws: int
    candidate_wins_as_first: int
    candidate_wins_as_second: int
    opponent_wins_while_candidate_first: int
    opponent_wins_while_candidate_second: int
    average_game_length: float

    def __post_init__(self) -> None:
        for field in _HEAD_TO_HEAD_FIELDS - {"average_game_length"}:
            _integer(getattr(self, field), field=f"head_to_head.{field}")
        object.__setattr__(
            self,
            "average_game_length",
            _finite(
                self.average_game_length,
                field="head_to_head.average_game_length",
                minimum=0.0,
            ),
        )
        if (self.games_played == 0) != (self.average_game_length == 0.0):
            raise ArtifactValidationError(
                "Head-to-head average game length must be zero exactly when no games exist"
            )
        if self.candidate_wins + self.opponent_wins + self.draws != self.games_played:
            raise ArtifactValidationError(
                "Head-to-head wins, losses, and draws must partition games_played"
            )
        if (
            self.candidate_wins_as_first + self.candidate_wins_as_second
            != self.candidate_wins
        ):
            raise ArtifactValidationError("Candidate seat wins do not sum to wins")
        if (
            self.opponent_wins_while_candidate_first
            + self.opponent_wins_while_candidate_second
            != self.opponent_wins
        ):
            raise ArtifactValidationError("Opponent seat wins do not sum to wins")
        candidate_first_games = (self.games_played + 1) // 2
        candidate_second_games = self.games_played // 2
        if (
            self.candidate_wins_as_first + self.opponent_wins_while_candidate_first
            > candidate_first_games
        ):
            raise ArtifactValidationError(
                "First-seat outcomes exceed the deterministic seat schedule"
            )
        if (
            self.candidate_wins_as_second + self.opponent_wins_while_candidate_second
            > candidate_second_games
        ):
            raise ArtifactValidationError(
                "Second-seat outcomes exceed the deterministic seat schedule"
            )

    @property
    def candidate_win_rate(self) -> float:
        return self.candidate_wins / self.games_played if self.games_played else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games_played if self.games_played else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field)
            for field in (
                "games_played",
                "candidate_wins",
                "opponent_wins",
                "draws",
                "candidate_wins_as_first",
                "candidate_wins_as_second",
                "opponent_wins_while_candidate_first",
                "opponent_wins_while_candidate_second",
                "average_game_length",
            )
        }

    @classmethod
    def from_dict(cls, value: object) -> "HeadToHeadResultV1":
        data = _exact(value, _HEAD_TO_HEAD_FIELDS, context="head-to-head result")
        return cls(
            **{
                field: (
                    _finite(
                        data[field],
                        field="head_to_head.average_game_length",
                        minimum=0.0,
                    )
                    if field == "average_game_length"
                    else _integer(data[field], field=f"head_to_head.{field}")
                )
                for field in _HEAD_TO_HEAD_FIELDS
            }
        )

    @classmethod
    def from_results(cls, results: object) -> "HeadToHeadResultV1":
        return cls(
            games_played=results.games_played,
            candidate_wins=results.player1_wins,
            opponent_wins=results.player2_wins,
            draws=results.draws,
            candidate_wins_as_first=results.player1_wins_as_first,
            candidate_wins_as_second=results.player1_wins_as_second,
            opponent_wins_while_candidate_first=results.player2_wins_as_second,
            opponent_wins_while_candidate_second=results.player2_wins_as_first,
            average_game_length=results.avg_game_length,
        )


@dataclass(frozen=True)
class SolverBucketV1:
    positions: int
    value_optimal: int
    exact_best: int
    blunders_win_to_draw: int
    blunders_win_to_loss: int
    blunders_draw_to_loss: int
    forced: int

    def __post_init__(self) -> None:
        for field in _BUCKET_FIELDS:
            value = _integer(getattr(self, field), field=f"solver_bucket.{field}")
            if field != "positions" and value > self.positions:
                raise ArtifactValidationError(
                    f"solver_bucket.{field} cannot exceed positions"
                )
        if self.exact_best > self.value_optimal:
            raise ArtifactValidationError(
                "solver_bucket.exact_best cannot exceed value_optimal"
            )
        if self.forced > self.exact_best:
            raise ArtifactValidationError(
                "solver_bucket.forced cannot exceed exact_best"
            )
        blunders = (
            self.blunders_win_to_draw
            + self.blunders_win_to_loss
            + self.blunders_draw_to_loss
        )
        if blunders != self.positions - self.value_optimal:
            raise ArtifactValidationError(
                "solver_bucket blunders must partition non-value-optimal positions"
            )

    @property
    def value_optimal_rate(self) -> float:
        return self.value_optimal / self.positions if self.positions else 0.0

    def to_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in _BUCKET_FIELDS}

    @classmethod
    def from_dict(cls, value: object) -> "SolverBucketV1":
        data = _exact(value, _BUCKET_FIELDS, context="solver bucket")
        return cls(
            **{
                field: _integer(data[field], field=f"solver_bucket.{field}")
                for field in _BUCKET_FIELDS
            }
        )

    @classmethod
    def from_results(cls, results: object) -> "SolverBucketV1":
        return cls(**{field: getattr(results, field) for field in _BUCKET_FIELDS})


@dataclass(frozen=True)
class SolverResultV1:
    games_played: int
    candidate_wins: int
    opponent_wins: int
    draws: int
    average_game_length: float
    overall: SolverBucketV1
    by_ply: Mapping[str, SolverBucketV1]
    by_seat: Mapping[str, SolverBucketV1]
    solver_queries: int
    solver_cache_hits: int
    solver_time_seconds: float
    wall_time_seconds: float
    solver_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.overall, SolverBucketV1):
            raise ArtifactValidationError("solver.overall must be a solver bucket")
        if not isinstance(self.by_ply, Mapping) or not isinstance(
            self.by_seat, Mapping
        ):
            raise ArtifactValidationError("solver slice collections must be mappings")
        by_ply = dict(self.by_ply)
        by_seat = dict(self.by_seat)
        if not all(isinstance(bucket, SolverBucketV1) for bucket in by_ply.values()):
            raise ArtifactValidationError("solver.by_ply values must be solver buckets")
        if not all(isinstance(bucket, SolverBucketV1) for bucket in by_seat.values()):
            raise ArtifactValidationError(
                "solver.by_seat values must be solver buckets"
            )
        object.__setattr__(self, "by_ply", MappingProxyType(by_ply))
        object.__setattr__(self, "by_seat", MappingProxyType(by_seat))
        for field in (
            "games_played",
            "candidate_wins",
            "opponent_wins",
            "draws",
            "solver_queries",
            "solver_cache_hits",
        ):
            _integer(getattr(self, field), field=f"solver.{field}")
        if self.candidate_wins + self.opponent_wins + self.draws != self.games_played:
            raise ArtifactValidationError("Solver game outcomes must partition games")
        if self.solver_cache_hits > self.solver_queries:
            raise ArtifactValidationError("Solver cache hits cannot exceed queries")
        if self.solver_queries != self.overall.positions:
            raise ArtifactValidationError(
                "Solver queries must equal the number of scored positions"
            )
        object.__setattr__(
            self,
            "average_game_length",
            _finite(
                self.average_game_length,
                field="solver.average_game_length",
                minimum=0.0,
            ),
        )
        if (self.games_played == 0) != (self.average_game_length == 0.0):
            raise ArtifactValidationError(
                "Solver average game length must be zero exactly when no games exist"
            )
        object.__setattr__(
            self,
            "solver_time_seconds",
            _finite(
                self.solver_time_seconds,
                field="solver.solver_time_seconds",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "wall_time_seconds",
            _finite(
                self.wall_time_seconds,
                field="solver.wall_time_seconds",
                minimum=0.0,
            ),
        )
        if set(self.by_ply) != {"ply_1_8", "ply_9_20", "ply_21_plus"}:
            raise ArtifactValidationError("solver.by_ply fields must be exact")
        if set(self.by_seat) != {"first", "second"}:
            raise ArtifactValidationError("solver.by_seat fields must be exact")
        for field in _BUCKET_FIELDS:
            overall_value = getattr(self.overall, field)
            if sum(getattr(bucket, field) for bucket in self.by_ply.values()) != (
                overall_value
            ):
                raise ArtifactValidationError(
                    f"solver.by_ply does not partition overall.{field}"
                )
            if sum(getattr(bucket, field) for bucket in self.by_seat.values()) != (
                overall_value
            ):
                raise ArtifactValidationError(
                    f"solver.by_seat does not partition overall.{field}"
                )
        _string(self.solver_version, field="solver.solver_version")

    def to_dict(self) -> dict[str, object]:
        return {
            "games_played": self.games_played,
            "candidate_wins": self.candidate_wins,
            "opponent_wins": self.opponent_wins,
            "draws": self.draws,
            "average_game_length": self.average_game_length,
            "overall": self.overall.to_dict(),
            "by_ply": {
                name: self.by_ply[name].to_dict()
                for name in ("ply_1_8", "ply_9_20", "ply_21_plus")
            },
            "by_seat": {
                name: self.by_seat[name].to_dict() for name in ("first", "second")
            },
            "solver_queries": self.solver_queries,
            "solver_cache_hits": self.solver_cache_hits,
            "solver_time_seconds": self.solver_time_seconds,
            "wall_time_seconds": self.wall_time_seconds,
            "solver_version": self.solver_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SolverResultV1":
        data = _exact(value, _SOLVER_FIELDS, context="solver result")
        by_ply_data = _exact(
            data["by_ply"],
            frozenset({"ply_1_8", "ply_9_20", "ply_21_plus"}),
            context="solver.by_ply",
        )
        by_seat_data = _exact(
            data["by_seat"],
            frozenset({"first", "second"}),
            context="solver.by_seat",
        )
        version = _string(data["solver_version"], field="solver.solver_version")
        return cls(
            games_played=_integer(data["games_played"], field="solver.games_played"),
            candidate_wins=_integer(
                data["candidate_wins"], field="solver.candidate_wins"
            ),
            opponent_wins=_integer(data["opponent_wins"], field="solver.opponent_wins"),
            draws=_integer(data["draws"], field="solver.draws"),
            average_game_length=_finite(
                data["average_game_length"],
                field="solver.average_game_length",
                minimum=0.0,
            ),
            overall=SolverBucketV1.from_dict(data["overall"]),
            by_ply={
                name: SolverBucketV1.from_dict(by_ply_data[name])
                for name in by_ply_data
            },
            by_seat={
                name: SolverBucketV1.from_dict(by_seat_data[name])
                for name in by_seat_data
            },
            solver_queries=_integer(
                data["solver_queries"], field="solver.solver_queries"
            ),
            solver_cache_hits=_integer(
                data["solver_cache_hits"], field="solver.solver_cache_hits"
            ),
            solver_time_seconds=_finite(
                data["solver_time_seconds"],
                field="solver.solver_time_seconds",
                minimum=0.0,
            ),
            wall_time_seconds=_finite(
                data["wall_time_seconds"],
                field="solver.wall_time_seconds",
                minimum=0.0,
            ),
            solver_version=version,
        )

    @classmethod
    def from_results(cls, results: object) -> "SolverResultV1":
        return cls(
            games_played=results.games,
            candidate_wins=results.model_wins,
            opponent_wins=results.model_losses,
            draws=results.draws,
            average_game_length=results.avg_game_length,
            overall=SolverBucketV1.from_results(results.overall),
            by_ply={
                name: SolverBucketV1.from_results(bucket)
                for name, bucket in results.by_ply.items()
            },
            by_seat={
                name: SolverBucketV1.from_results(bucket)
                for name, bucket in results.by_seat.items()
            },
            solver_queries=results.solver_queries,
            solver_cache_hits=results.solver_cache_hits,
            solver_time_seconds=results.solver_time_seconds,
            wall_time_seconds=results.wall_time_seconds,
            solver_version=results.bitbully_version,
        )


@dataclass(frozen=True)
class ObservedResultsV1:
    vs_champion: HeadToHeadResultV1 | None
    vs_random: HeadToHeadResultV1 | None
    candidate_solver: SolverResultV1 | None
    champion_solver: SolverResultV1 | None

    def to_dict(self) -> dict[str, object]:
        return {
            "vs_champion": (
                self.vs_champion.to_dict() if self.vs_champion is not None else None
            ),
            "vs_random": (
                self.vs_random.to_dict() if self.vs_random is not None else None
            ),
            "candidate_solver": (
                self.candidate_solver.to_dict()
                if self.candidate_solver is not None
                else None
            ),
            "champion_solver": (
                self.champion_solver.to_dict()
                if self.champion_solver is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ObservedResultsV1":
        data = _exact(value, _RESULTS_FIELDS, context="observed results")
        return cls(
            vs_champion=(
                HeadToHeadResultV1.from_dict(data["vs_champion"])
                if data["vs_champion"] is not None
                else None
            ),
            vs_random=(
                HeadToHeadResultV1.from_dict(data["vs_random"])
                if data["vs_random"] is not None
                else None
            ),
            candidate_solver=(
                SolverResultV1.from_dict(data["candidate_solver"])
                if data["candidate_solver"] is not None
                else None
            ),
            champion_solver=(
                SolverResultV1.from_dict(data["champion_solver"])
                if data["champion_solver"] is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PromotionDecisionV1:
    promoted: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.promoted, bool):
            raise ArtifactValidationError("decision.promoted must be boolean")
        _string(self.reason, field="decision.reason")

    def to_dict(self) -> dict[str, object]:
        return {"promoted": self.promoted, "reason": self.reason}

    @classmethod
    def from_dict(cls, value: object) -> "PromotionDecisionV1":
        data = _exact(value, _DECISION_FIELDS, context="promotion decision")
        return cls(promoted=data["promoted"], reason=data["reason"])


@dataclass(frozen=True)
class EvaluationArtifactV2:
    profile: CheckpointProfileV1
    iteration: int
    candidate_checkpoint_id: str
    previous_evaluation_id: str | None
    champion_before: ChampionReferenceV1 | None
    recipe: EvaluationRecipeV1
    results: ObservedResultsV1
    decision: PromotionDecisionV1
    started_at: str
    completed_at: str
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2 or isinstance(self.schema_version, bool):
            raise ArtifactValidationError("evaluation.schema_version must be exactly 2")
        _integer(self.iteration, field="evaluation.iteration", positive=True)
        validate_sha256_digest(
            self.candidate_checkpoint_id, field="evaluation.candidate_checkpoint_id"
        )
        if self.previous_evaluation_id is not None:
            validate_sha256_digest(
                self.previous_evaluation_id,
                field="evaluation.previous_evaluation_id",
            )
        _timestamp(self.started_at, field="evaluation.started_at")
        _timestamp(self.completed_at, field="evaluation.completed_at")
        if self.completed_at <= self.started_at:
            raise ArtifactValidationError(
                "Evaluation completion must be strictly after its start"
            )
        expected_pairs = (
            (
                self.recipe.requested_games.vs_champion,
                self.results.vs_champion,
                "vs_champion",
            ),
            (
                self.recipe.requested_games.vs_random,
                self.results.vs_random,
                "vs_random",
            ),
            (
                self.recipe.requested_games.candidate_solver,
                self.results.candidate_solver,
                "candidate_solver",
            ),
            (
                self.recipe.requested_games.champion_solver,
                self.results.champion_solver,
                "champion_solver",
            ),
        )
        for requested, result, field in expected_pairs:
            if requested == 0 and result is not None:
                raise ArtifactValidationError(
                    f"{field} result exists with zero request"
                )
            if requested > 0 and result is None:
                raise ArtifactValidationError(f"{field} result missing for request")
            if result is not None and result.games_played != requested:
                raise ArtifactValidationError(
                    f"{field} games_played does not match requested games"
                )
        if self.champion_before is None and self.results.vs_champion is not None:
            raise ArtifactValidationError("vs_champion result requires a champion")
        if self.champion_before is None and self.results.champion_solver is not None:
            raise ArtifactValidationError("champion_solver result requires a champion")
        if (
            self.recipe.promotion_metric == "win_rate"
            and self.recipe.requested_games.champion_solver != 0
        ):
            raise ArtifactValidationError(
                "win_rate promotion cannot request champion solver evidence"
            )
        if self.champion_before is None:
            if self.previous_evaluation_id is not None:
                raise ArtifactValidationError(
                    "An evaluation after the first must have a champion"
                )
            if not self.decision.promoted:
                raise ArtifactValidationError(
                    "The first valid candidate must establish champion state"
                )
        elif self.previous_evaluation_id is None:
            raise ArtifactValidationError(
                "The first evaluation cannot refer to an existing champion"
            )
        elif self.recipe.requested_games.vs_champion == 0:
            raise ArtifactValidationError(
                "A candidate with an existing champion must compare against it"
            )
        elif self.candidate_checkpoint_id == self.champion_before.checkpoint_id:
            raise ArtifactValidationError(
                "A checkpoint cannot be evaluated for promotion against itself"
            )
        if (
            self.champion_before is not None
            and self.recipe.promotion_metric == "win_rate"
            and self.results.vs_champion is not None
        ):
            expected = (
                self.results.vs_champion.candidate_win_rate > self.recipe.win_threshold
            )
            if self.decision.promoted != expected:
                raise ArtifactValidationError(
                    "Promotion decision disagrees with the observed champion win rate"
                )
        if (
            self.champion_before is not None
            and self.recipe.promotion_metric == "solver_optimal"
        ):
            candidate_solver = self.results.candidate_solver
            champion_solver = self.results.champion_solver
            if candidate_solver is None or champion_solver is None:
                raise ArtifactValidationError(
                    "solver_optimal promotion requires fresh candidate and "
                    "champion solver evidence"
                )
            if (
                self.recipe.requested_games.candidate_solver
                != self.recipe.requested_games.champion_solver
            ):
                raise ArtifactValidationError(
                    "solver_optimal candidate and champion game counts must match"
                )
            if candidate_solver.solver_version != champion_solver.solver_version:
                raise ArtifactValidationError(
                    "solver_optimal candidate and champion solver versions must match"
                )
            expected = (
                candidate_solver.overall.value_optimal_rate
                > champion_solver.overall.value_optimal_rate
                + self.recipe.promotion_margin
            )
            if self.decision.promoted != expected:
                raise ArtifactValidationError(
                    "Promotion decision disagrees with solver comparison evidence"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile.to_dict(),
            "iteration": self.iteration,
            "candidate_checkpoint_id": self.candidate_checkpoint_id,
            "previous_evaluation_id": self.previous_evaluation_id,
            "champion_before": (
                self.champion_before.to_dict()
                if self.champion_before is not None
                else None
            ),
            "recipe": self.recipe.to_dict(),
            "results": self.results.to_dict(),
            "decision": self.decision.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def evaluation_id(self) -> str:
        return sha256_bytes(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "EvaluationArtifactV2":
        value = _decode_canonical(data, context="evaluation artifact")
        fields = _exact(value, _ARTIFACT_FIELDS, context="evaluation artifact")
        if fields["schema_version"] != 2 or isinstance(fields["schema_version"], bool):
            raise ArtifactValidationError("evaluation.schema_version must be exactly 2")
        profile_data = _exact(
            fields["profile"], _PROFILE_FIELDS, context="evaluation profile"
        )
        champion = fields["champion_before"]
        artifact = cls(
            schema_version=2,
            profile=CheckpointProfileV1.from_dict(dict(profile_data)),
            iteration=_integer(
                fields["iteration"], field="evaluation.iteration", positive=True
            ),
            candidate_checkpoint_id=validate_sha256_digest(
                fields["candidate_checkpoint_id"],
                field="evaluation.candidate_checkpoint_id",
            ),
            previous_evaluation_id=(
                validate_sha256_digest(
                    fields["previous_evaluation_id"],
                    field="evaluation.previous_evaluation_id",
                )
                if fields["previous_evaluation_id"] is not None
                else None
            ),
            champion_before=(
                ChampionReferenceV1.from_dict(champion)
                if champion is not None
                else None
            ),
            recipe=EvaluationRecipeV1.from_dict(fields["recipe"]),
            results=ObservedResultsV1.from_dict(fields["results"]),
            decision=PromotionDecisionV1.from_dict(fields["decision"]),
            started_at=_timestamp(fields["started_at"], field="evaluation.started_at"),
            completed_at=_timestamp(
                fields["completed_at"], field="evaluation.completed_at"
            ),
        )
        if artifact.to_bytes() != data:
            raise ArtifactValidationError(
                "evaluation artifact is not normalized canonical JSON"
            )
        return artifact


@dataclass(frozen=True)
class EvaluationRef:
    evaluation_id: str
    artifact: EvaluationArtifactV2
    path: Path

    def __post_init__(self) -> None:
        validate_sha256_digest(self.evaluation_id, field="evaluation_id")
        if self.evaluation_id != self.artifact.evaluation_id:
            raise ArtifactValidationError("EvaluationRef ID does not match artifact")


class EvaluationRepository(Protocol):
    def validate_evidence(self, artifact: EvaluationArtifactV2) -> None: ...

    def publish_evidence(self, artifact: EvaluationArtifactV2) -> EvaluationRef: ...

    def resolve_evaluation(self, evaluation_id: str) -> EvaluationRef: ...

    def list_evaluations(
        self, evaluation_head_id: str | None
    ) -> list[EvaluationRef]: ...


class FilesystemEvaluationRepository:
    def __init__(self, checkpoints: FilesystemCheckpointPublisher):
        self.checkpoints = checkpoints
        self.model_root = checkpoints.model_root
        self.profile = checkpoints.contract.profile

    def _evaluation_path(self, evaluation_id: str) -> Path:
        return (
            self.model_root
            / "evaluations"
            / "manifests"
            / "sha256"
            / f"{evaluation_id}.json"
        )

    def _validate_v2_evidence(self, artifact: EvaluationArtifactV2) -> None:
        if artifact.profile != self.profile:
            raise ArtifactValidationError("Evaluation profile mismatch")
        self.checkpoints.read_checkpoint_manifest_exact(
            artifact.candidate_checkpoint_id
        )
        prior_lineage = self.list_evaluations(artifact.previous_evaluation_id)
        expected_champion: ChampionReferenceV1 | None = None
        for reference in prior_lineage:
            if reference.artifact.decision.promoted:
                expected_champion = ChampionReferenceV1(
                    checkpoint_id=reference.artifact.candidate_checkpoint_id,
                    evaluation_id=reference.evaluation_id,
                )
        if artifact.champion_before != expected_champion:
            raise ArtifactValidationError(
                "Evaluation champion_before does not match its evidence lineage"
            )
        if artifact.previous_evaluation_id is not None:
            previous = prior_lineage[-1]
            if artifact.iteration <= previous.artifact.iteration:
                raise ArtifactValidationError(
                    "Evaluation iterations must be strictly increasing"
                )
            if artifact.started_at <= previous.artifact.completed_at:
                raise ArtifactValidationError(
                    "Evaluation timestamps must be strictly increasing"
                )
        if expected_champion is not None:
            self.checkpoints.read_checkpoint_manifest_exact(
                expected_champion.checkpoint_id
            )

    def validate_evidence(self, artifact: EvaluationArtifactV2) -> None:
        """Validate canonical evidence and its full ancestry without writing it."""
        if not isinstance(artifact, EvaluationArtifactV2):
            raise TypeError("artifact must be EvaluationArtifactV2")
        normalized = EvaluationArtifactV2.from_bytes(artifact.to_bytes())
        if normalized != artifact:
            raise ArtifactValidationError(
                "Evaluation evidence is not its normalized canonical value"
            )
        self._validate_v2_evidence(normalized)

    def publish_evidence(self, artifact: EvaluationArtifactV2) -> EvaluationRef:
        """Persist validated evidence without changing any mutable authority."""
        self.validate_evidence(artifact)
        evaluation_id = artifact.evaluation_id
        path = self._evaluation_path(evaluation_id)
        _create_or_verify(path, artifact.to_bytes())
        return EvaluationRef(evaluation_id, artifact, path)

    def resolve_evaluation(self, evaluation_id: str) -> EvaluationRef:
        validate_sha256_digest(evaluation_id, field="evaluation_id")
        path = self._evaluation_path(evaluation_id)
        if not path.is_file():
            raise ArtifactValidationError(f"Evaluation artifact does not exist: {path}")
        data = path.read_bytes()
        if sha256_bytes(data) != evaluation_id:
            raise ArtifactValidationError("Evaluation artifact SHA-256 mismatch")
        artifact = EvaluationArtifactV2.from_bytes(data)
        if artifact.profile != self.profile:
            raise ArtifactValidationError("Evaluation profile mismatch")
        return EvaluationRef(evaluation_id, artifact, path)

    def list_evaluations(self, evaluation_head_id: str | None) -> list[EvaluationRef]:
        """Resolve and fully validate an immutable evaluation chain."""
        if evaluation_head_id is None:
            return []
        validate_sha256_digest(evaluation_head_id, field="evaluation_head_id")
        reversed_lineage: list[EvaluationRef] = []
        seen: set[str] = set()
        current_id: str | None = evaluation_head_id
        while current_id is not None:
            if current_id in seen:
                raise ArtifactValidationError("Evaluation lineage contains a cycle")
            seen.add(current_id)
            reference = self.resolve_evaluation(current_id)
            reversed_lineage.append(reference)
            current_id = reference.artifact.previous_evaluation_id
        lineage = list(reversed(reversed_lineage))

        previous: EvaluationRef | None = None
        champion: ChampionReferenceV1 | None = None
        for reference in lineage:
            artifact = reference.artifact
            expected_previous = previous.evaluation_id if previous is not None else None
            if artifact.previous_evaluation_id != expected_previous:
                raise ArtifactValidationError("Evaluation lineage is disconnected")
            if artifact.champion_before != champion:
                raise ArtifactValidationError(
                    "Evaluation lineage has an inconsistent champion"
                )
            if previous is not None:
                if artifact.iteration <= previous.artifact.iteration:
                    raise ArtifactValidationError(
                        "Evaluation iterations must be strictly increasing"
                    )
                if artifact.started_at <= previous.artifact.completed_at:
                    raise ArtifactValidationError(
                        "Evaluation timestamps must be strictly increasing"
                    )
            self.checkpoints.read_checkpoint_manifest_exact(
                artifact.candidate_checkpoint_id
            )
            if artifact.decision.promoted:
                champion = ChampionReferenceV1(
                    checkpoint_id=artifact.candidate_checkpoint_id,
                    evaluation_id=reference.evaluation_id,
                )
            previous = reference
        return lineage


class S3EvaluationRepository(FilesystemEvaluationRepository):
    def __init__(self, checkpoints: S3CheckpointPublisher):
        super().__init__(checkpoints)
        self.checkpoints: S3CheckpointPublisher = checkpoints

    def _evaluation_key(self, evaluation_id: str) -> str:
        return self.checkpoints._key(
            f"evaluations/manifests/sha256/{evaluation_id}.json"
        )

    def publish_evidence(self, artifact: EvaluationArtifactV2) -> EvaluationRef:
        """Persist exact evidence in both the remote store and local cache."""
        self.validate_evidence(artifact)
        evaluation_id = artifact.evaluation_id
        data = artifact.to_bytes()
        path = self._evaluation_path(evaluation_id)
        _create_or_verify(path, data)
        self.checkpoints._put_immutable(
            self._evaluation_key(evaluation_id), data, "application/json"
        )
        return EvaluationRef(evaluation_id, artifact, path)

    def resolve_evaluation(self, evaluation_id: str) -> EvaluationRef:
        validate_sha256_digest(evaluation_id, field="evaluation_id")
        data = self.checkpoints._get(self._evaluation_key(evaluation_id))
        if data is None:
            raise ArtifactValidationError("S3 evaluation artifact does not exist")
        if sha256_bytes(data) != evaluation_id:
            raise ArtifactValidationError("S3 evaluation artifact SHA-256 mismatch")
        artifact = EvaluationArtifactV2.from_bytes(data)
        if artifact.profile != self.profile:
            raise ArtifactValidationError("Evaluation profile mismatch")
        path = self._evaluation_path(evaluation_id)
        _create_or_verify(path, data)
        return EvaluationRef(evaluation_id, artifact, path)


def create_evaluation_repository(
    checkpoints: CheckpointPublisher,
) -> EvaluationRepository:
    if isinstance(checkpoints, S3CheckpointPublisher):
        return S3EvaluationRepository(checkpoints)
    if isinstance(checkpoints, FilesystemCheckpointPublisher):
        return FilesystemEvaluationRepository(checkpoints)
    raise TypeError("Unsupported checkpoint repository implementation")


__all__ = [
    "ChampionReferenceV1",
    "EvaluationArtifactV2",
    "EvaluationRecipeV1",
    "EvaluationRef",
    "EvaluationRepository",
    "FilesystemEvaluationRepository",
    "HeadToHeadResultV1",
    "ObservedResultsV1",
    "PromotionDecisionV1",
    "RequestedGamesV1",
    "S3EvaluationRepository",
    "SolverBucketV1",
    "SolverResultV1",
    "create_evaluation_repository",
    "utc_timestamp",
]
