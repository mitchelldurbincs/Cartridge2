"""Observed evaluation results and promotion decisions."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .artifact_codec import ArtifactValidationError
from .evaluation_validation import (
    _BUCKET_FIELDS,
    _DECISION_FIELDS,
    _HEAD_TO_HEAD_FIELDS,
    _RESULTS_FIELDS,
    _SOLVER_FIELDS,
    _exact,
    _finite,
    _integer,
    _string,
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
        if self.candidate_wins_as_first + self.candidate_wins_as_second != self.candidate_wins:
            raise ArtifactValidationError("Candidate seat wins do not sum to wins")
        if (
            self.opponent_wins_while_candidate_first + self.opponent_wins_while_candidate_second
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
                raise ArtifactValidationError(f"solver_bucket.{field} cannot exceed positions")
        if self.exact_best > self.value_optimal:
            raise ArtifactValidationError("solver_bucket.exact_best cannot exceed value_optimal")
        if self.forced > self.exact_best:
            raise ArtifactValidationError("solver_bucket.forced cannot exceed exact_best")
        blunders = (
            self.blunders_win_to_draw + self.blunders_win_to_loss + self.blunders_draw_to_loss
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
        if not isinstance(self.by_ply, Mapping) or not isinstance(self.by_seat, Mapping):
            raise ArtifactValidationError("solver slice collections must be mappings")
        by_ply = dict(self.by_ply)
        by_seat = dict(self.by_seat)
        if not all(isinstance(bucket, SolverBucketV1) for bucket in by_ply.values()):
            raise ArtifactValidationError("solver.by_ply values must be solver buckets")
        if not all(isinstance(bucket, SolverBucketV1) for bucket in by_seat.values()):
            raise ArtifactValidationError("solver.by_seat values must be solver buckets")
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
            if sum(getattr(bucket, field) for bucket in self.by_ply.values()) != (overall_value):
                raise ArtifactValidationError(f"solver.by_ply does not partition overall.{field}")
            if sum(getattr(bucket, field) for bucket in self.by_seat.values()) != (overall_value):
                raise ArtifactValidationError(f"solver.by_seat does not partition overall.{field}")
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
                name: self.by_ply[name].to_dict() for name in ("ply_1_8", "ply_9_20", "ply_21_plus")
            },
            "by_seat": {name: self.by_seat[name].to_dict() for name in ("first", "second")},
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
            candidate_wins=_integer(data["candidate_wins"], field="solver.candidate_wins"),
            opponent_wins=_integer(data["opponent_wins"], field="solver.opponent_wins"),
            draws=_integer(data["draws"], field="solver.draws"),
            average_game_length=_finite(
                data["average_game_length"],
                field="solver.average_game_length",
                minimum=0.0,
            ),
            overall=SolverBucketV1.from_dict(data["overall"]),
            by_ply={name: SolverBucketV1.from_dict(by_ply_data[name]) for name in by_ply_data},
            by_seat={name: SolverBucketV1.from_dict(by_seat_data[name]) for name in by_seat_data},
            solver_queries=_integer(data["solver_queries"], field="solver.solver_queries"),
            solver_cache_hits=_integer(data["solver_cache_hits"], field="solver.solver_cache_hits"),
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
                name: SolverBucketV1.from_results(bucket) for name, bucket in results.by_ply.items()
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
            "vs_champion": (self.vs_champion.to_dict() if self.vs_champion is not None else None),
            "vs_random": (self.vs_random.to_dict() if self.vs_random is not None else None),
            "candidate_solver": (
                self.candidate_solver.to_dict() if self.candidate_solver is not None else None
            ),
            "champion_solver": (
                self.champion_solver.to_dict() if self.champion_solver is not None else None
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
