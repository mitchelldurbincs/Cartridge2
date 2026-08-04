"""Evaluation recipe and champion identity types."""

from __future__ import annotations

from dataclasses import dataclass

from .artifact_codec import ArtifactValidationError, validate_sha256_digest
from .evaluation_validation import (
    _MAX_U32,
    _MAX_U64,
    _RECIPE_FIELDS,
    _REFERENCE_FIELDS,
    _REQUESTED_GAMES_FIELDS,
    _SEAT_SCHEDULE,
    _exact,
    _finite,
    _finite_f32,
    _integer,
    _string,
)


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
            raise ArtifactValidationError("An evaluation recipe must request candidate evidence")

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
        schedule = _exact(data["seat_schedule"], frozenset(_SEAT_SCHEDULE), context="seat schedule")
        if dict(schedule) != _SEAT_SCHEDULE:
            raise ArtifactValidationError("seat schedule must be deterministic alternating_v1")
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
