"""Prune learner-state blobs that no future resume can need.

Every checkpoint stores two immutable blobs: the ONNX model and a
learner-state ``.pt`` (model + optimizer + scheduler, typically 2-3x larger).
The ONNX is needed forever — tournaments and solver-eval deliberately rate
every historical checkpoint — but a ``.pt`` exists only to resume training
from that exact checkpoint, and only the newest few will ever be resumed
from. Without pruning, the largest blob in the repository accumulates once
per iteration for the life of the profile.

Safety comes from what is *kept*, computed conservatively:

- the newest ``retained_checkpoints`` checkpoints of the current chain,
- the current champion (cheap insurance, though inference needs only ONNX),
- every checkpoint referenced by a manifest *outside* the current chain —
  standalone or unknown lineages are never touched.

Deleting an old ``.pt`` breaks nothing else by construction: ancestor lineage
validation reads only manifests, and inference-only consumers resolve with
``require_learner_state=False`` (absent is permitted, corrupt never is).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def prune_learner_state_blobs(
    publisher,
    chain,
    *,
    retained_checkpoints: int,
) -> int:
    """Delete prunable ``.pt`` blobs; return how many digests were removed."""
    if isinstance(retained_checkpoints, bool) or not isinstance(retained_checkpoints, int):
        raise ValueError("retained_checkpoints must be a positive integer")
    if retained_checkpoints < 1:
        raise ValueError("retained_checkpoints must be a positive integer")
    chain_ids = [reference.commit.checkpoint_id for reference in chain]
    if not chain_ids:
        return 0
    keep = set(chain_ids[-retained_checkpoints:])
    champion = chain[-1].commit.champion
    if champion is not None:
        keep.add(champion.checkpoint_id)
    chain_set = set(chain_ids)

    protected_digests: set[str] = set()
    prunable_digests: set[str] = set()
    for checkpoint_id in publisher.list_checkpoint_manifest_ids():
        manifest = publisher.read_checkpoint_manifest_exact(checkpoint_id)
        digest = manifest.learner_state.sha256
        if checkpoint_id in keep or checkpoint_id not in chain_set:
            protected_digests.add(digest)
        else:
            prunable_digests.add(digest)

    deleted = sum(
        1
        for digest in sorted(prunable_digests - protected_digests)
        if publisher.delete_learner_state_blob(digest)
    )
    if deleted:
        logger.info(
            "Pruned %d learner-state blob(s); retained the newest %d "
            "checkpoint(s), the champion, and all out-of-chain lineages",
            deleted,
            retained_checkpoints,
        )
    return deleted
