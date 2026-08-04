"""Compatibility façade for checkpoint publication."""

# ruff: noqa: I001 -- publish immutable types before adapters to break legacy import cycles.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifact_codec import (
    ArtifactValidationError,
    canonical_json_bytes,
    sha256_bytes,
    validate_sha256_digest,
)
from .checkpoint_types import (
    BlobDescriptorV1,
    CheckpointManifestV1,
    CheckpointProfileV1,
    CheckpointPublisher,
    CheckpointRef,
    OnnxArtifactContract,
    OnnxTensorSpec,
    RunHeadV2,
)
from .checkpoint_filesystem import FilesystemCheckpointPublisher
from .checkpoint_s3 import S3CheckpointPublisher
from .checkpoint_validation import validate_onnx_checkpoint

if TYPE_CHECKING:
    from ..central_config import StorageConfig


def create_checkpoint_publisher(
    contract: OnnxArtifactContract,
    model_root: str | Path,
    storage_config: StorageConfig | None = None,
    *,
    s3_client: Any | None = None,
) -> CheckpointPublisher:
    """Build the configured repository for a profile's model root."""
    if storage_config is None:
        from ..central_config import get_config

        storage_config = get_config().storage
    if storage_config.model_backend == "filesystem":
        return FilesystemCheckpointPublisher(model_root=model_root, contract=contract)
    if storage_config.model_backend == "s3":
        if not storage_config.s3_bucket:
            raise ValueError(
                "S3 checkpoint publication requires storage.s3_bucket or "
                "CARTRIDGE_STORAGE_S3_BUCKET"
            )
        return S3CheckpointPublisher(
            model_root=model_root,
            bucket=storage_config.s3_bucket,
            endpoint=storage_config.s3_endpoint,
            contract=contract,
            client=s3_client,
        )
    raise ValueError(
        f"Unknown checkpoint publication backend '{storage_config.model_backend}'; "
        "expected 'filesystem' or 's3'"
    )


__all__ = [
    "ArtifactValidationError",
    "BlobDescriptorV1",
    "RunHeadV2",
    "CheckpointManifestV1",
    "CheckpointProfileV1",
    "CheckpointPublisher",
    "CheckpointRef",
    "FilesystemCheckpointPublisher",
    "OnnxArtifactContract",
    "OnnxTensorSpec",
    "S3CheckpointPublisher",
    "canonical_json_bytes",
    "create_checkpoint_publisher",
    "sha256_bytes",
    "validate_sha256_digest",
    "validate_onnx_checkpoint",
]
