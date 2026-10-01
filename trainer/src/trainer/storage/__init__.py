"""Opaque replay and immutable checkpoint storage contracts.

Storage adapters depend on trainer contracts such as stats and RunCommit types.
Keep this package initializer free of eager imports so low-level contracts can
be imported without loading every repository and adapter in the package.
Public exports remain available through lazy attribute lookup.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    **{
        name: "base"
        for name in (
            "EmptyReplaySelectionError",
            "ReplayProfile",
            "ReplayRecord",
            "ReplaySelection",
            "ReplayStore",
        )
    },
    "PostgresReplayStore": "postgres",
    "create_replay_store": "factory",
    "ArtifactValidationError": "artifact_codec",
    "canonical_json_bytes": "artifact_codec",
    "sha256_bytes": "artifact_codec",
    "validate_sha256_digest": "artifact_codec",
    **{
        name: "checkpoint_types"
        for name in (
            "BlobDescriptorV1",
            "CheckpointManifestV1",
            "CheckpointProfileV1",
            "CheckpointPublisher",
            "CheckpointRef",
            "OnnxArtifactContract",
            "OnnxTensorSpec",
            "RunHeadV2",
        )
    },
    **{
        name: "publisher"
        for name in (
            "FilesystemCheckpointPublisher",
            "S3CheckpointPublisher",
            "create_checkpoint_publisher",
        )
    },
    "validate_onnx_checkpoint": "checkpoint_validation",
}

__all__ = [
    "EmptyReplaySelectionError",
    "ReplayProfile",
    "ReplayRecord",
    "ReplaySelection",
    "ReplayStore",
    "PostgresReplayStore",
    "create_replay_store",
    "ArtifactValidationError",
    "BlobDescriptorV1",
    "CheckpointManifestV1",
    "CheckpointProfileV1",
    "CheckpointPublisher",
    "CheckpointRef",
    "FilesystemCheckpointPublisher",
    "OnnxArtifactContract",
    "OnnxTensorSpec",
    "RunHeadV2",
    "S3CheckpointPublisher",
    "canonical_json_bytes",
    "create_checkpoint_publisher",
    "sha256_bytes",
    "validate_sha256_digest",
    "validate_onnx_checkpoint",
]


def __getattr__(name: str) -> Any:
    """Load public storage exports only when a caller requests them."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
