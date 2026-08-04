"""Content-addressed evaluation artifact."""

from __future__ import annotations

from dataclasses import dataclass

from .artifact_codec import (
    ArtifactValidationError,
    canonical_json_bytes,
    sha256_bytes,
    validate_sha256_digest,
)
from .checkpoint_types import CheckpointProfileV1
from .evaluation_recipe import ChampionReferenceV1, EvaluationRecipeV1
from .evaluation_results import ObservedResultsV1, PromotionDecisionV1
from .evaluation_validation import (
    _ARTIFACT_FIELDS,
    _PROFILE_FIELDS,
    _decode_canonical,
    _exact,
    _integer,
    _timestamp,
)


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
        self._validate_identity()
        self._validate_requested_results()
        self._validate_champion_state()
        self._validate_promotion_decision()

    def _validate_identity(self) -> None:
        if self.schema_version != 2 or isinstance(self.schema_version, bool):
            raise ArtifactValidationError("evaluation.schema_version must be exactly 2")
        _integer(self.iteration, field="evaluation.iteration", positive=True)
        validate_sha256_digest(
            self.candidate_checkpoint_id,
            field="evaluation.candidate_checkpoint_id",
        )
        if self.previous_evaluation_id is not None:
            validate_sha256_digest(
                self.previous_evaluation_id,
                field="evaluation.previous_evaluation_id",
            )
        _timestamp(self.started_at, field="evaluation.started_at")
        _timestamp(self.completed_at, field="evaluation.completed_at")
        if self.completed_at <= self.started_at:
            raise ArtifactValidationError("Evaluation completion must be strictly after its start")

    def _validate_requested_results(self) -> None:
        expected_pairs = (
            (self.recipe.requested_games.vs_champion, self.results.vs_champion, "vs_champion"),
            (self.recipe.requested_games.vs_random, self.results.vs_random, "vs_random"),
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
                raise ArtifactValidationError(f"{field} result exists with zero request")
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

    def _validate_champion_state(self) -> None:
        if self.champion_before is None:
            if self.previous_evaluation_id is not None:
                raise ArtifactValidationError("An evaluation after the first must have a champion")
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

    def _validate_promotion_decision(self) -> None:
        if (
            self.champion_before is not None
            and self.recipe.promotion_metric == "win_rate"
            and self.results.vs_champion is not None
        ):
            expected = self.results.vs_champion.candidate_win_rate > self.recipe.win_threshold
            if self.decision.promoted != expected:
                raise ArtifactValidationError(
                    "Promotion decision disagrees with the observed champion win rate"
                )
        if self.champion_before is None or self.recipe.promotion_metric != "solver_optimal":
            return
        candidate_solver = self.results.candidate_solver
        champion_solver = self.results.champion_solver
        if candidate_solver is None or champion_solver is None:
            raise ArtifactValidationError(
                "solver_optimal promotion requires fresh candidate and champion solver evidence"
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
            > champion_solver.overall.value_optimal_rate + self.recipe.promotion_margin
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
                self.champion_before.to_dict() if self.champion_before is not None else None
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
        profile_data = _exact(fields["profile"], _PROFILE_FIELDS, context="evaluation profile")
        champion = fields["champion_before"]
        artifact = cls(
            schema_version=2,
            profile=CheckpointProfileV1.from_dict(dict(profile_data)),
            iteration=_integer(fields["iteration"], field="evaluation.iteration", positive=True),
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
                ChampionReferenceV1.from_dict(champion) if champion is not None else None
            ),
            recipe=EvaluationRecipeV1.from_dict(fields["recipe"]),
            results=ObservedResultsV1.from_dict(fields["results"]),
            decision=PromotionDecisionV1.from_dict(fields["decision"]),
            started_at=_timestamp(fields["started_at"], field="evaluation.started_at"),
            completed_at=_timestamp(fields["completed_at"], field="evaluation.completed_at"),
        )
        if artifact.to_bytes() != data:
            raise ArtifactValidationError("evaluation artifact is not normalized canonical JSON")
        return artifact
