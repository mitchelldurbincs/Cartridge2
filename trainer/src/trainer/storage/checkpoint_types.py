"""Immutable checkpoint contracts and repository interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..runtime_profile import RuntimeProfile
from .artifact_codec import (
    MAX_U32,
    MAX_U64,
    ArtifactValidationError,
    canonical_json_bytes,
    decode_canonical_json,
    require_digest,
    require_exact_fields,
    require_nonempty_string,
    require_nonnegative_integer,
    require_positive_integer,
    sha256_bytes,
)

_PROFILE_FIELDS = frozenset(
    {
        "algorithm_id",
        "env_id",
        "env_contract_version",
        "model_artifact_schema_version",
        "model_contract",
    }
)
_BLOB_FIELDS = frozenset({"sha256", "size_bytes"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "profile",
        "step",
        "parent_checkpoint_id",
        "config_sha256",
        "onnx",
        "learner_state",
    }
)
_HEAD_FIELDS = frozenset({"schema_version", "checkpoint_id", "run_commit_id"})


@dataclass(frozen=True)
class OnnxTensorSpec:
    name: str
    dtype: str
    shape: tuple[str | int, ...]

    def __post_init__(self) -> None:
        require_nonempty_string(self.name, field="OnnxTensorSpec.name")
        if self.dtype != "float32":
            raise ValueError(f"OnnxTensorSpec.dtype {self.dtype!r} is not supported")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise ValueError("OnnxTensorSpec.shape must be a non-empty tuple")
        for index, dimension in enumerate(self.shape):
            field = f"OnnxTensorSpec.shape[{index}]"
            if isinstance(dimension, bool):
                raise ValueError(f"{field} must be a positive integer or symbol")
            if isinstance(dimension, int):
                if dimension <= 0:
                    raise ValueError(f"{field} must be positive")
            elif not isinstance(dimension, str) or not dimension.isidentifier():
                raise ValueError(f"{field} must be a valid non-empty symbol")


@dataclass(frozen=True)
class OnnxArtifactContract:
    algorithm_id: str
    env_id: str
    env_contract_version: int
    model_artifact_schema_version: int
    model_contract: str
    inputs: tuple[OnnxTensorSpec, ...]
    outputs: tuple[OnnxTensorSpec, ...]

    def __post_init__(self) -> None:
        for field_name in ("algorithm_id", "env_id", "model_contract"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"OnnxArtifactContract.{field_name} must be a non-empty string")
        for field_name in ("env_contract_version", "model_artifact_schema_version"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"OnnxArtifactContract.{field_name} must be a positive integer")
            if value > MAX_U32:
                raise ValueError(f"OnnxArtifactContract.{field_name} exceeds u32")
        for field_name in ("inputs", "outputs"):
            tensors = getattr(self, field_name)
            if not isinstance(tensors, tuple) or not tensors:
                raise ValueError(f"OnnxArtifactContract.{field_name} must be a non-empty tuple")
            if any(not isinstance(tensor, OnnxTensorSpec) for tensor in tensors):
                raise TypeError(f"OnnxArtifactContract.{field_name} must contain OnnxTensorSpec")
            names = [tensor.name for tensor in tensors]
            if len(names) != len(set(names)):
                raise ValueError(f"OnnxArtifactContract.{field_name} names must be unique")
        RuntimeProfile(self.algorithm_id, self.env_id, self.env_contract_version)

    def input(self, name: str) -> OnnxTensorSpec:
        return self._tensor(self.inputs, name, boundary="input")

    def output(self, name: str) -> OnnxTensorSpec:
        return self._tensor(self.outputs, name, boundary="output")

    @staticmethod
    def _tensor(tensors: tuple[OnnxTensorSpec, ...], name: str, *, boundary: str) -> OnnxTensorSpec:
        matches = [tensor for tensor in tensors if tensor.name == name]
        if not matches:
            raise ValueError(f"ONNX contract has no {boundary} named {name!r}")
        return matches[0]

    @property
    def profile(self) -> "CheckpointProfileV1":
        return CheckpointProfileV1(
            algorithm_id=self.algorithm_id,
            env_id=self.env_id,
            env_contract_version=self.env_contract_version,
            model_artifact_schema_version=self.model_artifact_schema_version,
            model_contract=self.model_contract,
        )


@dataclass(frozen=True)
class CheckpointProfileV1:
    algorithm_id: str
    env_id: str
    env_contract_version: int
    model_artifact_schema_version: int
    model_contract: str

    def __post_init__(self) -> None:
        RuntimeProfile(self.algorithm_id, self.env_id, self.env_contract_version)
        require_positive_integer(
            self.env_contract_version,
            field="profile.env_contract_version",
            maximum=MAX_U32,
        )
        require_positive_integer(
            self.model_artifact_schema_version,
            field="profile.model_artifact_schema_version",
            maximum=MAX_U32,
        )
        require_nonempty_string(self.model_contract, field="profile.model_contract")

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm_id": self.algorithm_id,
            "env_id": self.env_id,
            "env_contract_version": self.env_contract_version,
            "model_artifact_schema_version": self.model_artifact_schema_version,
            "model_contract": self.model_contract,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CheckpointProfileV1":
        data = require_exact_fields(value, _PROFILE_FIELDS, context="profile")
        try:
            return cls(
                algorithm_id=require_nonempty_string(
                    data["algorithm_id"], field="profile.algorithm_id"
                ),
                env_id=require_nonempty_string(data["env_id"], field="profile.env_id"),
                env_contract_version=require_positive_integer(
                    data["env_contract_version"],
                    field="profile.env_contract_version",
                    maximum=MAX_U32,
                ),
                model_artifact_schema_version=require_positive_integer(
                    data["model_artifact_schema_version"],
                    field="profile.model_artifact_schema_version",
                    maximum=MAX_U32,
                ),
                model_contract=require_nonempty_string(
                    data["model_contract"], field="profile.model_contract"
                ),
            )
        except ValueError as exc:
            raise ArtifactValidationError(f"Invalid checkpoint profile: {exc}") from exc


@dataclass(frozen=True)
class BlobDescriptorV1:
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        require_digest(self.sha256, field="blob.sha256")
        require_positive_integer(self.size_bytes, field="blob.size_bytes", maximum=MAX_U64)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BlobDescriptorV1":
        return cls(sha256=sha256_bytes(data), size_bytes=len(data))

    def to_dict(self) -> dict[str, object]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, value: object, *, field: str) -> "BlobDescriptorV1":
        data = require_exact_fields(value, _BLOB_FIELDS, context=field)
        return cls(
            sha256=require_digest(data["sha256"], field=f"{field}.sha256"),
            size_bytes=require_positive_integer(
                data["size_bytes"], field=f"{field}.size_bytes", maximum=MAX_U64
            ),
        )


@dataclass(frozen=True)
class CheckpointManifestV1:
    profile: CheckpointProfileV1
    step: int
    parent_checkpoint_id: str | None
    config_sha256: str
    onnx: BlobDescriptorV1
    learner_state: BlobDescriptorV1
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ArtifactValidationError("manifest.schema_version must be exactly 1")
        require_nonnegative_integer(self.step, field="manifest.step", maximum=MAX_U64)
        if self.parent_checkpoint_id is not None:
            require_digest(self.parent_checkpoint_id, field="manifest.parent_checkpoint_id")
        require_digest(self.config_sha256, field="manifest.config_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile.to_dict(),
            "step": self.step,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "config_sha256": self.config_sha256,
            "onnx": self.onnx.to_dict(),
            "learner_state": self.learner_state.to_dict(),
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def checkpoint_id(self) -> str:
        return sha256_bytes(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "CheckpointManifestV1":
        value = decode_canonical_json(data, context="checkpoint manifest")
        fields = require_exact_fields(value, _MANIFEST_FIELDS, context="checkpoint manifest")
        if fields["schema_version"] != 1 or isinstance(fields["schema_version"], bool):
            raise ArtifactValidationError("manifest.schema_version must be exactly 1")
        parent = fields["parent_checkpoint_id"]
        if parent is not None:
            parent = require_digest(parent, field="manifest.parent_checkpoint_id")
        return cls(
            schema_version=1,
            profile=CheckpointProfileV1.from_dict(fields["profile"]),
            step=require_nonnegative_integer(
                fields["step"], field="manifest.step", maximum=MAX_U64
            ),
            parent_checkpoint_id=parent,
            config_sha256=require_digest(fields["config_sha256"], field="manifest.config_sha256"),
            onnx=BlobDescriptorV1.from_dict(fields["onnx"], field="manifest.onnx"),
            learner_state=BlobDescriptorV1.from_dict(
                fields["learner_state"], field="manifest.learner_state"
            ),
        )


@dataclass(frozen=True)
class RunHeadV2:
    checkpoint_id: str
    run_commit_id: str
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2 or isinstance(self.schema_version, bool):
            raise ArtifactValidationError("head.schema_version must be exactly 2")
        require_digest(self.checkpoint_id, field="head.checkpoint_id")
        require_digest(self.run_commit_id, field="head.run_commit_id")

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema_version": self.schema_version,
                "checkpoint_id": self.checkpoint_id,
                "run_commit_id": self.run_commit_id,
            }
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "RunHeadV2":
        fields = require_exact_fields(
            decode_canonical_json(data, context="run head"),
            _HEAD_FIELDS,
            context="run head",
        )
        if fields["schema_version"] != 2 or isinstance(fields["schema_version"], bool):
            raise ArtifactValidationError("head.schema_version must be exactly 2")
        return cls(
            checkpoint_id=require_digest(fields["checkpoint_id"], field="head.checkpoint_id"),
            run_commit_id=require_digest(fields["run_commit_id"], field="head.run_commit_id"),
        )


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_id: str
    manifest: CheckpointManifestV1
    onnx_path: Path
    learner_state_path: Path

    def __post_init__(self) -> None:
        require_digest(self.checkpoint_id, field="checkpoint_id")
        if self.manifest.checkpoint_id != self.checkpoint_id:
            raise ArtifactValidationError("CheckpointRef ID does not match its canonical manifest")


class CheckpointPublisher(Protocol):
    contract: OnnxArtifactContract

    def stage_checkpoint(
        self,
        onnx_path: Path,
        learner_state_path: Path,
        *,
        step: int,
        parent_checkpoint_id: str | None,
        config_sha256: str,
        learner_state_contract: object,
    ) -> CheckpointRef: ...

    def resolve_head(
        self, *, expected_config_sha256: str | None = None
    ) -> CheckpointRef | None: ...

    def resolve_run_head(self) -> RunHeadV2 | None: ...

    def publish_run_commit_bytes(self, run_commit_id: str, data: bytes) -> None: ...

    def read_run_commit_bytes(self, run_commit_id: str) -> bytes: ...

    def commit_run_head(
        self,
        *,
        checkpoint_id: str,
        run_commit_id: str,
        expected_run_commit_id: str | None,
    ) -> RunHeadV2: ...

    def resolve_checkpoint(
        self,
        checkpoint_id: str,
        *,
        expected_config_sha256: str | None = None,
    ) -> CheckpointRef: ...

    def resolve_checkpoint_manifest(
        self,
        checkpoint_id: str,
        *,
        expected_config_sha256: str | None = None,
    ) -> CheckpointManifestV1: ...

    def read_checkpoint_manifest_exact(
        self,
        checkpoint_id: str,
        *,
        expected_config_sha256: str | None = None,
    ) -> CheckpointManifestV1: ...

    def list_checkpoints(self) -> list[CheckpointRef]: ...
