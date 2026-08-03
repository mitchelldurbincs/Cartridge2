"""Opaque replay persistence and strict checkpoint publication."""

from trainer.storage.base import (
    EmptyReplaySelectionError,
    ReplayProfile,
    ReplayRecord,
    ReplaySelection,
    ReplayStore,
)
from trainer.storage.factory import create_replay_store
from trainer.storage.postgres import PostgresReplayStore
from trainer.storage.publisher import (
    ArtifactValidationError,
    BlobDescriptorV1,
    CheckpointManifestV1,
    CheckpointProfileV1,
    CheckpointPublisher,
    CheckpointRef,
    FilesystemCheckpointPublisher,
    OnnxArtifactContract,
    RunHeadV2,
    S3CheckpointPublisher,
    canonical_json_bytes,
    create_checkpoint_publisher,
    sha256_bytes,
    validate_onnx_checkpoint,
    validate_sha256_digest,
)

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
    "RunHeadV2",
    "S3CheckpointPublisher",
    "canonical_json_bytes",
    "create_checkpoint_publisher",
    "sha256_bytes",
    "validate_sha256_digest",
    "validate_onnx_checkpoint",
]
