"""Crash-durable prepared RunCommit intents keyed by their authoritative parent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from ..storage.evaluation import EvaluationArtifactV2
from ..storage.publisher import (
    ArtifactValidationError,
    CheckpointPublisher,
    canonical_json_bytes,
    validate_sha256_digest,
)
from ..storage.run_commit import RunCommitRepository, RunCommitV1
from .eval_runner import PreparedEvaluation

_PREPARATION_FIELDS = frozenset({"schema_version", "run_commit_id", "run_commit", "evaluation"})
_PREPARED_EVALUATION_FIELDS = frozenset(
    {
        "evaluation_id",
        "artifact",
        "win_rate",
        "draw_rate",
        "elapsed_seconds",
    }
)


def _decode_canonical(data: bytes, *, context: str) -> dict:
    if not isinstance(data, bytes):
        raise ArtifactValidationError(f"{context} must be bytes")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"{context} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != data:
        raise ArtifactValidationError(f"{context} must be canonical JSON")
    return value


@dataclass(frozen=True)
class PreparedRunV1:
    """Exact run intent written before evaluation evidence or RunHead mutation."""

    run_commit: RunCommitV1
    evaluation: PreparedEvaluation | None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ArtifactValidationError("run_preparation.schema_version must be exactly 1")
        if not isinstance(self.run_commit, RunCommitV1):
            raise ArtifactValidationError("run_preparation.run_commit is invalid")
        orchestration = self.run_commit.orchestration
        evaluation_id = orchestration.evaluation_id if orchestration is not None else None
        if (evaluation_id is None) != (self.evaluation is None):
            raise ArtifactValidationError(
                "Run preparation evaluation presence disagrees with its RunCommit"
            )
        if self.evaluation is not None:
            if not isinstance(self.evaluation, PreparedEvaluation):
                raise ArtifactValidationError("run_preparation.evaluation is invalid")
            if evaluation_id != self.evaluation.evaluation_id:
                raise ArtifactValidationError(
                    "Run preparation evaluation ID disagrees with its RunCommit"
                )
            if self.evaluation.artifact.candidate_checkpoint_id != (self.run_commit.checkpoint_id):
                raise ArtifactValidationError(
                    "Run preparation candidate disagrees with its RunCommit"
                )
            if orchestration is None or (
                orchestration.eval_win_rate,
                orchestration.eval_draw_rate,
                orchestration.eval_time_seconds,
            ) != (
                self.evaluation.win_rate,
                self.evaluation.draw_rate,
                self.evaluation.elapsed_seconds,
            ):
                raise ArtifactValidationError(
                    "Run preparation evaluation metrics disagree with its RunCommit"
                )

    @property
    def parent_run_commit_id(self) -> str | None:
        return self.run_commit.parent_run_commit_id

    @property
    def run_commit_id(self) -> str:
        return self.run_commit.run_commit_id

    def to_dict(self) -> dict[str, object]:
        prepared_evaluation = None
        if self.evaluation is not None:
            prepared_evaluation = {
                "evaluation_id": self.evaluation.evaluation_id,
                "artifact": self.evaluation.artifact.to_dict(),
                "win_rate": self.evaluation.win_rate,
                "draw_rate": self.evaluation.draw_rate,
                "elapsed_seconds": self.evaluation.elapsed_seconds,
            }
        return {
            "schema_version": self.schema_version,
            "run_commit_id": self.run_commit_id,
            "run_commit": self.run_commit.to_dict(),
            "evaluation": prepared_evaluation,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_bytes(cls, data: bytes) -> "PreparedRunV1":
        value = _decode_canonical(data, context="run preparation")
        if frozenset(value) != _PREPARATION_FIELDS:
            raise ArtifactValidationError("Run preparation fields must be exact")
        if value["schema_version"] != 1 or isinstance(value["schema_version"], bool):
            raise ArtifactValidationError("run_preparation.schema_version must be exactly 1")
        run_commit = RunCommitV1.from_bytes(canonical_json_bytes(value["run_commit"]))
        run_commit_id = validate_sha256_digest(
            value["run_commit_id"], field="run_preparation.run_commit_id"
        )
        if run_commit_id != run_commit.run_commit_id:
            raise ArtifactValidationError(
                "Run preparation ID does not match its canonical RunCommit"
            )
        prepared_evaluation = value["evaluation"]
        evaluation = None
        if prepared_evaluation is not None:
            if (
                not isinstance(prepared_evaluation, dict)
                or frozenset(prepared_evaluation) != _PREPARED_EVALUATION_FIELDS
            ):
                raise ArtifactValidationError("Prepared evaluation fields must be exact")
            artifact = EvaluationArtifactV2.from_bytes(
                canonical_json_bytes(prepared_evaluation["artifact"])
            )
            evaluation = PreparedEvaluation(
                artifact=artifact,
                evaluation_id=validate_sha256_digest(
                    prepared_evaluation["evaluation_id"],
                    field="run_preparation.evaluation_id",
                ),
                win_rate=prepared_evaluation["win_rate"],
                draw_rate=prepared_evaluation["draw_rate"],
                elapsed_seconds=prepared_evaluation["elapsed_seconds"],
            )
        prepared = cls(run_commit=run_commit, evaluation=evaluation)
        if prepared.to_bytes() != data:
            raise ArtifactValidationError("Run preparation is not normalized canonical JSON")
        return prepared


class _PreparationStorage(Protocol):
    def publish_run_preparation_bytes(
        self, parent_run_commit_id: str | None, data: bytes
    ) -> None: ...

    def read_run_preparation_bytes(self, parent_run_commit_id: str | None) -> bytes | None: ...


class RunJournal:
    """Store and recover the sole next intent for a particular RunHead parent."""

    def __init__(
        self,
        checkpoints: CheckpointPublisher,
        run_commits: RunCommitRepository,
    ):
        self.checkpoints = checkpoints
        self.run_commits = run_commits

    def publish(self, prepared: PreparedRunV1) -> None:
        if not isinstance(prepared, PreparedRunV1):
            raise TypeError("prepared must be PreparedRunV1")
        canonical = self.run_commits.validate_prepared(
            prepared.run_commit,
            prepared.evaluation.artifact if prepared.evaluation is not None else None,
        )
        prepared = PreparedRunV1(canonical, prepared.evaluation)
        storage: _PreparationStorage = self.checkpoints  # type: ignore[assignment]
        storage.publish_run_preparation_bytes(prepared.parent_run_commit_id, prepared.to_bytes())

    def resolve(self, parent_run_commit_id: str | None) -> PreparedRunV1 | None:
        storage: _PreparationStorage = self.checkpoints  # type: ignore[assignment]
        data = storage.read_run_preparation_bytes(parent_run_commit_id)
        if data is None:
            return None
        prepared = PreparedRunV1.from_bytes(data)
        if prepared.parent_run_commit_id != parent_run_commit_id:
            raise ArtifactValidationError("Run preparation is stored under the wrong parent")
        self.run_commits.validate_prepared(
            prepared.run_commit,
            prepared.evaluation.artifact if prepared.evaluation is not None else None,
        )
        return prepared


__all__ = ["PreparedRunV1", "RunJournal"]
