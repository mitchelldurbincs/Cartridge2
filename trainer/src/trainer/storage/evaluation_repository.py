"""Evaluation repositories for filesystem and S3 storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .artifact_codec import ArtifactValidationError, sha256_bytes, validate_sha256_digest
from .checkpoint_types import CheckpointPublisher, OnnxArtifactContract
from .evaluation_artifact import EvaluationArtifactV2
from .evaluation_recipe import ChampionReferenceV1
from .filesystem_backend import create_or_verify


@runtime_checkable
class _FilesystemCheckpointStorage(CheckpointPublisher, Protocol):
    model_root: Path
    contract: OnnxArtifactContract


@runtime_checkable
class _S3CheckpointStorage(_FilesystemCheckpointStorage, Protocol):
    bucket: str

    def _key(self, relative: str) -> str: ...

    def _get(self, key: str) -> bytes | None: ...

    def _put_immutable(self, key: str, data: bytes, content_type: str) -> None: ...


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

    def list_evaluations(self, evaluation_head_id: str | None) -> list[EvaluationRef]: ...


class FilesystemEvaluationRepository:
    def __init__(self, checkpoints: _FilesystemCheckpointStorage):
        self.checkpoints = checkpoints
        self.model_root = checkpoints.model_root
        self.profile = checkpoints.contract.profile

    def _evaluation_path(self, evaluation_id: str) -> Path:
        return self.model_root / "evaluations" / "manifests" / "sha256" / f"{evaluation_id}.json"

    def _validate_v2_evidence(self, artifact: EvaluationArtifactV2) -> None:
        if artifact.profile != self.profile:
            raise ArtifactValidationError("Evaluation profile mismatch")
        self.checkpoints.read_checkpoint_manifest_exact(artifact.candidate_checkpoint_id)
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
                raise ArtifactValidationError("Evaluation iterations must be strictly increasing")
            if artifact.started_at <= previous.artifact.completed_at:
                raise ArtifactValidationError("Evaluation timestamps must be strictly increasing")
        if expected_champion is not None:
            self.checkpoints.read_checkpoint_manifest_exact(expected_champion.checkpoint_id)

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
        create_or_verify(path, artifact.to_bytes())
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
                raise ArtifactValidationError("Evaluation lineage has an inconsistent champion")
            if previous is not None:
                if artifact.iteration <= previous.artifact.iteration:
                    raise ArtifactValidationError(
                        "Evaluation iterations must be strictly increasing"
                    )
                if artifact.started_at <= previous.artifact.completed_at:
                    raise ArtifactValidationError(
                        "Evaluation timestamps must be strictly increasing"
                    )
            self.checkpoints.read_checkpoint_manifest_exact(artifact.candidate_checkpoint_id)
            if artifact.decision.promoted:
                champion = ChampionReferenceV1(
                    checkpoint_id=artifact.candidate_checkpoint_id,
                    evaluation_id=reference.evaluation_id,
                )
            previous = reference
        return lineage


class S3EvaluationRepository(FilesystemEvaluationRepository):
    def __init__(self, checkpoints: _S3CheckpointStorage):
        super().__init__(checkpoints)
        self.checkpoints: _S3CheckpointStorage = checkpoints

    def _evaluation_key(self, evaluation_id: str) -> str:
        return self.checkpoints._key(f"evaluations/manifests/sha256/{evaluation_id}.json")

    def publish_evidence(self, artifact: EvaluationArtifactV2) -> EvaluationRef:
        """Persist exact evidence in both the remote store and local cache."""
        self.validate_evidence(artifact)
        evaluation_id = artifact.evaluation_id
        data = artifact.to_bytes()
        path = self._evaluation_path(evaluation_id)
        create_or_verify(path, data)
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
        create_or_verify(path, data)
        return EvaluationRef(evaluation_id, artifact, path)


def create_evaluation_repository(
    checkpoints: CheckpointPublisher,
) -> EvaluationRepository:
    if isinstance(checkpoints, _S3CheckpointStorage):
        return S3EvaluationRepository(checkpoints)
    if isinstance(checkpoints, _FilesystemCheckpointStorage):
        return FilesystemEvaluationRepository(checkpoints)
    raise TypeError("Unsupported checkpoint repository implementation")
