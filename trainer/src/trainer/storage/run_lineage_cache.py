"""In-process memo of fully validated immutable lineages.

Every artifact ID in these chains is the SHA-256 of the artifact's canonical
bytes, and resolution re-verifies that digest on every read. Bytes therefore
cannot change under a cached ID, and a commit's bytes fix its parent's ID, so
a chain validated once stays valid for the process lifetime: the memo never
needs invalidation. Without it, every head resolution re-reads, re-hashes,
and re-validates the entire history — O(N²) per iteration and worse on S3,
where each read is a network GET.

The cache lives on the long-lived ``CheckpointPublisher`` because
``RunCommitRepository`` and the evaluation repositories are constructed fresh
for many individual operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluation_repository import EvaluationRef
    from .run_commit_types import RunCommitRef


class RunLineageCache:
    """Validated ``head id -> full chain`` memos for commits and evaluations."""

    def __init__(self) -> None:
        self.chains: dict[str, tuple[RunCommitRef, ...]] = {}
        self.eval_lineages: dict[str, tuple[EvaluationRef, ...]] = {}


def lineage_cache_for(checkpoints: object) -> RunLineageCache:
    """Return the publisher's shared cache, or a private one for bare fakes."""
    cache = getattr(checkpoints, "_lineage_cache", None)
    return cache if isinstance(cache, RunLineageCache) else RunLineageCache()
