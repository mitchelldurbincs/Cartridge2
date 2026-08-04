"""Compatibility façade for immutable evaluation evidence."""

from .evaluation_artifact import EvaluationArtifactV2
from .evaluation_recipe import ChampionReferenceV1, EvaluationRecipeV1, RequestedGamesV1
from .evaluation_repository import (
    EvaluationRef,
    EvaluationRepository,
    FilesystemEvaluationRepository,
    S3EvaluationRepository,
    create_evaluation_repository,
)
from .evaluation_results import (
    HeadToHeadResultV1,
    ObservedResultsV1,
    PromotionDecisionV1,
    SolverBucketV1,
    SolverResultV1,
)
from .evaluation_validation import utc_timestamp

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
