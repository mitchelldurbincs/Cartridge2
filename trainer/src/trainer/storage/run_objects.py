"""Pure naming and byte-envelope rules for immutable run objects.

RunCommits are content-addressed; preparations are indexed by their parent.
These checks preserve exact bytes. Typed schema and lineage validation remain
the responsibility of RunCommitRepository and RunJournal.
"""

from __future__ import annotations

from .artifact_codec import (
    ArtifactValidationError,
    decode_canonical_json,
    require_digest,
    sha256_bytes,
)


def run_commit_relative_path(run_commit_id: str) -> str:
    require_digest(run_commit_id, field="run_commit_id")
    return f"run-commits/sha256/{run_commit_id}.json"


def run_preparation_relative_path(parent_run_commit_id: str | None) -> str:
    if parent_run_commit_id is not None:
        require_digest(parent_run_commit_id, field="parent_run_commit_id")
    name = "root" if parent_run_commit_id is None else parent_run_commit_id
    return f"run-preparations/by-parent/{name}.json"


def validate_run_commit_publication(run_commit_id: str, data: bytes) -> None:
    """Reject invalid caller bytes before checking their proposed identity."""
    if not isinstance(data, bytes):
        raise ArtifactValidationError("Run commit must be bytes")
    decode_canonical_json(data, context="run commit")
    if sha256_bytes(data) != run_commit_id:
        raise ArtifactValidationError("Run commit SHA-256 does not match its ID")


def validate_stored_run_commit(run_commit_id: str, data: bytes) -> None:
    """Authenticate stored bytes before interpreting their canonical encoding."""
    if sha256_bytes(data) != run_commit_id:
        raise ArtifactValidationError("Run commit SHA-256 does not match its ID")
    decode_canonical_json(data, context="run commit")


def validate_run_preparation_bytes(data: bytes) -> None:
    if not isinstance(data, bytes):
        raise ArtifactValidationError("Run preparation must be bytes")
    decode_canonical_json(data, context="run preparation")
