"""RunCommit repository and recovery operations."""

from __future__ import annotations

from typing import Protocol

from .artifact_codec import ArtifactValidationError, sha256_bytes, validate_sha256_digest
from .checkpoint_types import CheckpointPublisher
from .evaluation_artifact import EvaluationArtifactV2
from .evaluation_repository import EvaluationRepository
from .run_commit_transition import validate_transition
from .run_commit_types import OrchestrationCommitV1, RunCommitRef, RunCommitV1
from .run_lineage_cache import lineage_cache_for


class _RunCommitStorage(Protocol):
    def publish_run_commit_bytes(self, run_commit_id: str, data: bytes) -> None: ...

    def read_run_commit_bytes(self, run_commit_id: str) -> bytes: ...


class RunCommitRepository:
    """Resolve, validate, and materialize immutable RunCommit chains."""

    def __init__(
        self,
        checkpoints: CheckpointPublisher,
        evaluations: EvaluationRepository,
    ) -> None:
        self.checkpoints = checkpoints
        self.evaluations = evaluations
        self._cache = lineage_cache_for(checkpoints)

    def resolve(self, run_commit_id: str) -> RunCommitRef:
        validate_sha256_digest(run_commit_id, field="run_commit_id")
        storage: _RunCommitStorage = self.checkpoints  # type: ignore[assignment]
        data = storage.read_run_commit_bytes(run_commit_id)
        if sha256_bytes(data) != run_commit_id:
            raise ArtifactValidationError("RunCommit SHA-256 mismatch")
        return RunCommitRef(run_commit_id, RunCommitV1.from_bytes(data))

    def validate_prepared(
        self,
        commit: RunCommitV1,
        evaluation: EvaluationArtifactV2 | None = None,
    ) -> RunCommitV1:
        """Fully validate and normalize an intent without writing any artifact."""
        if not isinstance(commit, RunCommitV1):
            raise TypeError("commit must be RunCommitV1")
        canonical = RunCommitV1.from_bytes(commit.to_bytes())
        if canonical.stats_snapshot.binding != commit.stats_snapshot.binding:
            raise ArtifactValidationError(
                "RunCommit stats wrapper disagrees with its canonical snapshot"
            )
        parent_chain = self.resolve_chain(canonical.parent_run_commit_id)
        parent = parent_chain[-1].commit if parent_chain else None
        if canonical.orchestration is not None:
            prior_scopes = {
                reference.commit.orchestration.collection_scope_id
                for reference in parent_chain
                if reference.commit.orchestration is not None
            }
            if canonical.orchestration.collection_scope_id in prior_scopes:
                raise ArtifactValidationError(
                    "Orchestration replay collection scopes must be unique"
                )
            latest = next(
                (
                    reference.commit.orchestration
                    for reference in reversed(parent_chain)
                    if reference.commit.orchestration is not None
                ),
                None,
            )
            expected_iteration = latest.iteration + 1 if latest is not None else 1
            if canonical.orchestration.iteration != expected_iteration:
                raise ArtifactValidationError(
                    "Orchestration iterations must be contiguous from one"
                )
            if latest is not None and canonical.orchestration.timestamp <= latest.timestamp:
                raise ArtifactValidationError(
                    "Orchestration timestamps must be strictly increasing"
                )
        elif evaluation is not None:
            raise ArtifactValidationError(
                "Standalone RunCommit cannot carry prepared evaluation evidence"
            )
        validate_transition(
            canonical,
            parent,
            checkpoints=self.checkpoints,
            evaluations=self.evaluations,
            prepared_evaluation=evaluation,
        )
        return canonical

    def publish(self, commit: RunCommitV1) -> RunCommitRef:
        commit = self.validate_prepared(commit)
        storage: _RunCommitStorage = self.checkpoints  # type: ignore[assignment]
        storage.publish_run_commit_bytes(commit.run_commit_id, commit.to_bytes())
        return RunCommitRef(commit.run_commit_id, commit)

    def resolve_chain(self, run_commit_id: str | None) -> list[RunCommitRef]:
        if run_commit_id is None:
            return []
        cached = self._cache.chains.get(run_commit_id)
        if cached is not None:
            return list(cached)
        # Walk toward the root, stopping at the deepest already-validated
        # ancestor: only the suffix beyond it needs reads and validation.
        reversed_suffix: list[RunCommitRef] = []
        seen: set[str] = set()
        prefix: tuple[RunCommitRef, ...] = ()
        current_id: str | None = run_commit_id
        while current_id is not None:
            if current_id in seen:
                raise ArtifactValidationError("RunCommit lineage contains a cycle")
            cached_prefix = self._cache.chains.get(current_id)
            if cached_prefix is not None:
                prefix = cached_prefix
                break
            seen.add(current_id)
            reference = self.resolve(current_id)
            reversed_suffix.append(reference)
            current_id = reference.commit.parent_run_commit_id
        if seen & {reference.run_commit_id for reference in prefix}:
            raise ArtifactValidationError("RunCommit lineage contains a cycle")
        chain = [*prefix, *reversed(reversed_suffix)]
        # Seed the fold state from the validated prefix entirely in memory —
        # no reads, no hashing, no re-validation of settled history.
        parent: RunCommitV1 | None = prefix[-1].commit if prefix else None
        latest_orchestration: OrchestrationCommitV1 | None = None
        collection_scopes: set[str] = set()
        for reference in prefix:
            orchestration = reference.commit.orchestration
            if orchestration is not None:
                collection_scopes.add(orchestration.collection_scope_id)
                latest_orchestration = orchestration
        for index in range(len(prefix), len(chain)):
            reference = chain[index]
            validate_transition(
                reference.commit,
                parent,
                checkpoints=self.checkpoints,
                evaluations=self.evaluations,
            )
            orchestration = reference.commit.orchestration
            if orchestration is not None:
                if orchestration.collection_scope_id in collection_scopes:
                    raise ArtifactValidationError(
                        "Orchestration replay collection scopes must be unique"
                    )
                collection_scopes.add(orchestration.collection_scope_id)
                expected_iteration = (
                    latest_orchestration.iteration + 1 if latest_orchestration is not None else 1
                )
                if orchestration.iteration != expected_iteration:
                    raise ArtifactValidationError(
                        "Orchestration iterations must be contiguous from one"
                    )
                if (
                    latest_orchestration is not None
                    and orchestration.timestamp <= latest_orchestration.timestamp
                ):
                    raise ArtifactValidationError(
                        "Orchestration timestamps must be strictly increasing"
                    )
                latest_orchestration = orchestration
            parent = reference.commit
            # Every prefix of a validated chain is itself a validated chain.
            self._cache.chains[reference.run_commit_id] = tuple(chain[: index + 1])
        return chain
