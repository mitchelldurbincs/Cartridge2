"""Content-addressed checkpoint storage for learner and inference artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

import onnx
from onnx import TensorProto

from ..runtime_profile import RuntimeProfile

if TYPE_CHECKING:
    from ..central_config import StorageConfig

logger = logging.getLogger(__name__)

_DIGEST_LENGTH = 64
_DIGEST_CHARACTERS = frozenset("0123456789abcdef")
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
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
_HEAD_FIELDS = frozenset({"schema_version", "checkpoint_id", "run_commit_id"})
_S3_CONDITIONAL_WRITE_ATTEMPTS = 5


class ArtifactValidationError(ValueError):
    """An artifact or checkpoint metadata object violates its contract."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode the one canonical JSON representation used for IDs and storage.

    Non-ASCII text is emitted as raw UTF-8 (``ensure_ascii=False``) because
    Rust consumers canonicalize with serde_json, which never writes ``\\uXXXX``
    escapes. Both languages must agree byte-for-byte or content-addressed IDs
    diverge across the boundary.
    """
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"Value is not canonical JSON: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_exact_fields(
    value: object, expected: frozenset[str], *, context: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{context} must be a JSON object")
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ArtifactValidationError(
            f"{context} fields must be exact (missing={missing}, extra={extra})"
        )
    return value


def _require_positive_integer(value: object, *, field: str, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or (maximum is not None and value > maximum)
    ):
        raise ArtifactValidationError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_integer(value: object, *, field: str, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise ArtifactValidationError(f"{field} must be a nonnegative integer")
    return value


def _require_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{field} must be a non-empty string")
    return value


def _require_digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in _DIGEST_CHARACTERS for character in value)
    ):
        raise ArtifactValidationError(f"{field} must be a lowercase 64-character SHA-256 digest")
    return value


def validate_sha256_digest(value: object, *, field: str) -> str:
    """Validate the digest grammar shared by manifests, channels, and config IDs."""
    return _require_digest(value, field=field)


def _decode_json(data: bytes, *, context: str) -> object:
    try:
        text = data.decode("utf-8")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"{context} is not valid UTF-8 JSON") from exc
    if canonical_json_bytes(value) != data:
        raise ArtifactValidationError(f"{context} is not canonical JSON")
    return value


@dataclass(frozen=True)
class OnnxTensorSpec:
    """One exact named tensor at an ONNX model boundary."""

    name: str
    dtype: str
    shape: tuple[str | int, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, field="OnnxTensorSpec.name")
        if self.dtype not in {"float32"}:
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
                continue
            if not isinstance(dimension, str) or not dimension.isidentifier():
                raise ValueError(f"{field} must be a valid non-empty symbol")


@dataclass(frozen=True)
class OnnxArtifactContract:
    """Identity and exact tensor interface for one model artifact profile."""

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
        for field_name in (
            "env_contract_version",
            "model_artifact_schema_version",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"OnnxArtifactContract.{field_name} must be a positive integer")
        if self.env_contract_version > _MAX_U32:
            raise ValueError("OnnxArtifactContract.env_contract_version exceeds u32")
        if self.model_artifact_schema_version > _MAX_U32:
            raise ValueError("OnnxArtifactContract.model_artifact_schema_version exceeds u32")
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
        _require_positive_integer(
            self.env_contract_version,
            field="profile.env_contract_version",
            maximum=_MAX_U32,
        )
        _require_positive_integer(
            self.model_artifact_schema_version,
            field="profile.model_artifact_schema_version",
            maximum=_MAX_U32,
        )
        _require_nonempty_string(self.model_contract, field="profile.model_contract")

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
        data = _require_exact_fields(value, _PROFILE_FIELDS, context="profile")
        try:
            return cls(
                algorithm_id=_require_nonempty_string(
                    data["algorithm_id"], field="profile.algorithm_id"
                ),
                env_id=_require_nonempty_string(data["env_id"], field="profile.env_id"),
                env_contract_version=_require_positive_integer(
                    data["env_contract_version"],
                    field="profile.env_contract_version",
                    maximum=_MAX_U32,
                ),
                model_artifact_schema_version=_require_positive_integer(
                    data["model_artifact_schema_version"],
                    field="profile.model_artifact_schema_version",
                    maximum=_MAX_U32,
                ),
                model_contract=_require_nonempty_string(
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
        _require_digest(self.sha256, field="blob.sha256")
        _require_positive_integer(self.size_bytes, field="blob.size_bytes", maximum=_MAX_U64)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BlobDescriptorV1":
        return cls(sha256=sha256_bytes(data), size_bytes=len(data))

    def to_dict(self) -> dict[str, object]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, value: object, *, field: str) -> "BlobDescriptorV1":
        data = _require_exact_fields(value, _BLOB_FIELDS, context=field)
        return cls(
            sha256=_require_digest(data["sha256"], field=f"{field}.sha256"),
            size_bytes=_require_positive_integer(
                data["size_bytes"], field=f"{field}.size_bytes", maximum=_MAX_U64
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
        _require_nonnegative_integer(self.step, field="manifest.step", maximum=_MAX_U64)
        if self.parent_checkpoint_id is not None:
            _require_digest(self.parent_checkpoint_id, field="manifest.parent_checkpoint_id")
        _require_digest(self.config_sha256, field="manifest.config_sha256")

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
        value = _decode_json(data, context="checkpoint manifest")
        fields = _require_exact_fields(value, _MANIFEST_FIELDS, context="checkpoint manifest")
        if fields["schema_version"] != 1 or isinstance(fields["schema_version"], bool):
            raise ArtifactValidationError("manifest.schema_version must be exactly 1")
        parent = fields["parent_checkpoint_id"]
        if parent is not None:
            parent = _require_digest(parent, field="manifest.parent_checkpoint_id")
        return cls(
            schema_version=1,
            profile=CheckpointProfileV1.from_dict(fields["profile"]),
            step=_require_nonnegative_integer(
                fields["step"], field="manifest.step", maximum=_MAX_U64
            ),
            parent_checkpoint_id=parent,
            config_sha256=_require_digest(fields["config_sha256"], field="manifest.config_sha256"),
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
        _require_digest(self.checkpoint_id, field="head.checkpoint_id")
        _require_digest(self.run_commit_id, field="head.run_commit_id")

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
        value = _decode_json(data, context="run head")
        fields = _require_exact_fields(value, _HEAD_FIELDS, context="run head")
        if fields["schema_version"] != 2 or isinstance(fields["schema_version"], bool):
            raise ArtifactValidationError("head.schema_version must be exactly 2")
        return cls(
            checkpoint_id=_require_digest(fields["checkpoint_id"], field="head.checkpoint_id"),
            run_commit_id=_require_digest(fields["run_commit_id"], field="head.run_commit_id"),
        )


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_id: str
    manifest: CheckpointManifestV1
    onnx_path: Path
    learner_state_path: Path

    def __post_init__(self) -> None:
        _require_digest(self.checkpoint_id, field="checkpoint_id")
        if self.manifest.checkpoint_id != self.checkpoint_id:
            raise ArtifactValidationError("CheckpointRef ID does not match its canonical manifest")


class CheckpointPublisher(Protocol):
    """Immutable checkpoint repository with one authoritative mutable head."""

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


def _fail(path: Path, message: str) -> ArtifactValidationError:
    return ArtifactValidationError(f"Invalid ONNX checkpoint {path}: {message}")


def _validate_tensor(
    path: Path,
    value_info: Any,
    *,
    spec: OnnxTensorSpec,
) -> None:
    name = spec.name
    value_type = value_info.type
    if value_type.WhichOneof("value") != "tensor_type":
        raise _fail(path, f"'{name}' must be a tensor")
    tensor_type = value_type.tensor_type
    expected_element_type = {"float32": TensorProto.FLOAT}[spec.dtype]
    if tensor_type.elem_type != expected_element_type:
        raise _fail(path, f"'{name}' must use {spec.dtype} elements")
    dims = list(tensor_type.shape.dim)
    if len(dims) != len(spec.shape):
        raise _fail(
            path,
            f"'{name}' must have rank {len(spec.shape)}, got rank {len(dims)}",
        )
    for index, (actual, expected) in enumerate(zip(dims, spec.shape, strict=True)):
        if isinstance(expected, int):
            valid = actual.WhichOneof("value") == "dim_value" and actual.dim_value == expected
        else:
            valid = actual.WhichOneof("value") == "dim_param" and actual.dim_param == expected
        if not valid:
            value = actual.dim_param or actual.dim_value
            raise _fail(
                path,
                f"'{name}' dimension {index} must be {expected!r}, got {value!r}",
            )


def validate_onnx_checkpoint(checkpoint_path: str | Path, contract: OnnxArtifactContract) -> None:
    """Require exact identity and the cartridge-declared tensor interface."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise _fail(path, "file does not exist")
    try:
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as exc:
        raise _fail(path, f"model is corrupt or structurally invalid ({exc})") from exc
    external_initializers = [
        initializer.name
        for initializer in model.graph.initializer
        if initializer.data_location == TensorProto.EXTERNAL or initializer.external_data
    ]
    if external_initializers:
        raise _fail(
            path,
            "external tensor data is not allowed: " + ", ".join(external_initializers),
        )
    metadata: dict[str, str] = {}
    for entry in model.metadata_props:
        if entry.key in metadata and entry.key.startswith("cartridge."):
            raise _fail(path, f"duplicate metadata key '{entry.key}'")
        metadata[entry.key] = entry.value
    expected_metadata = {
        "cartridge.schema_version": str(contract.model_artifact_schema_version),
        "cartridge.algorithm_id": contract.algorithm_id,
        "cartridge.model_contract": contract.model_contract,
        "cartridge.env_id": contract.env_id,
        "cartridge.env_contract_version": str(contract.env_contract_version),
    }
    actual_contract_keys = {key for key in metadata if key.startswith("cartridge.")}
    if actual_contract_keys != set(expected_metadata):
        raise _fail(
            path,
            "cartridge metadata keys must be exact: "
            f"got {sorted(actual_contract_keys)}, "
            f"expected {sorted(expected_metadata)}",
        )
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if actual != expected:
            raise _fail(path, f"metadata '{key}' is {actual!r}, expected {expected!r}")
    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    expected_inputs = [tensor.name for tensor in contract.inputs]
    expected_outputs = [tensor.name for tensor in contract.outputs]
    if [value.name for value in inputs] != expected_inputs:
        raise _fail(path, f"graph inputs must be exactly {expected_inputs!r}")
    if [value.name for value in outputs] != expected_outputs:
        raise _fail(path, f"graph outputs must be exactly {expected_outputs!r}")
    for value, spec in zip(inputs, contract.inputs, strict=True):
        _validate_tensor(path, value, spec=spec)
    for value, spec in zip(outputs, contract.outputs, strict=True):
        _validate_tensor(path, value, spec=spec)


def _verified_blob(data: bytes, descriptor: BlobDescriptorV1, *, name: str) -> None:
    if len(data) != descriptor.size_bytes:
        raise ArtifactValidationError(
            f"{name} size mismatch: got {len(data)}, expected {descriptor.size_bytes}"
        )
    digest = sha256_bytes(data)
    if digest != descriptor.sha256:
        raise ArtifactValidationError(
            f"{name} SHA-256 mismatch: got {digest}, expected {descriptor.sha256}"
        )


def _validate_learner_state(
    path: Path,
    *,
    contract: OnnxArtifactContract,
    step: int,
    config_sha256: str,
    learner_state_contract: object | None = None,
) -> None:
    # Import lazily because checkpoint.py defines the envelope and imports this
    # module's repository types.  Publication still validates before any write.
    from ..checkpoint import validate_learner_state_artifact

    validate_learner_state_artifact(
        path,
        artifact_contract=contract,
        step=step,
        config_sha256=config_sha256,
        learner_state_contract=learner_state_contract,
    )


def _snapshot_checkpoint_artifacts(
    onnx_path: Path,
    learner_state_path: Path,
    *,
    contract: OnnxArtifactContract,
    step: int,
    parent_checkpoint_id: str | None,
    config_sha256: str,
    learner_state_contract: object,
) -> tuple[CheckpointManifestV1, bytes, bytes]:
    """Validate private snapshots and return the exact bytes to publish."""
    onnx_data = onnx_path.read_bytes()
    learner_data = learner_state_path.read_bytes()
    with tempfile.TemporaryDirectory(prefix="cartridge-checkpoint-validation-") as root:
        snapshot_root = Path(root)
        onnx_snapshot = snapshot_root / "model.onnx"
        learner_snapshot = snapshot_root / "learner.pt"
        onnx_snapshot.write_bytes(onnx_data)
        learner_snapshot.write_bytes(learner_data)
        validate_onnx_checkpoint(onnx_snapshot, contract)
        _validate_learner_state(
            learner_snapshot,
            contract=contract,
            step=step,
            config_sha256=config_sha256,
            learner_state_contract=learner_state_contract,
        )
    manifest = CheckpointManifestV1(
        profile=contract.profile,
        step=step,
        parent_checkpoint_id=parent_checkpoint_id,
        config_sha256=config_sha256,
        onnx=BlobDescriptorV1.from_bytes(onnx_data),
        learner_state=BlobDescriptorV1.from_bytes(learner_data),
    )
    return manifest, onnx_data, learner_data


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_or_verify(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ArtifactValidationError(
                f"Immutable checkpoint object exists with different bytes: {path}"
            )
        return
    file_descriptor, temp_name = tempfile.mkstemp(prefix=".staging-", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != data:
                raise ArtifactValidationError(
                    f"Immutable checkpoint object raced with different bytes: {path}"
                )
        else:
            _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(prefix=".pointer-", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


@contextmanager
def _directory_lock(path: Path):
    """Hold an advisory lock on a repository directory without extra objects."""
    import fcntl

    path.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class FilesystemCheckpointPublisher:
    """Checkpoint repository rooted at one profile's models directory."""

    def __init__(self, *, model_root: str | Path, contract: OnnxArtifactContract):
        self.model_root = Path(model_root)
        self.contract = contract

    def _blob_path(self, descriptor: BlobDescriptorV1, extension: str) -> Path:
        return self.model_root / "blobs" / "sha256" / f"{descriptor.sha256}.{extension}"

    def _manifest_path(self, checkpoint_id: str) -> Path:
        return self.model_root / "manifests" / "sha256" / f"{checkpoint_id}.json"

    @property
    def _head_path(self) -> Path:
        return self.model_root / "channels" / "current.json"

    def _run_commit_path(self, run_commit_id: str) -> Path:
        return self.model_root / "run-commits" / "sha256" / f"{run_commit_id}.json"

    def _run_preparation_path(self, parent_run_commit_id: str | None) -> Path:
        name = "root" if parent_run_commit_id is None else parent_run_commit_id
        return self.model_root / "run-preparations" / "by-parent" / f"{name}.json"

    def run_head_commit_guard(self):
        """Serialize the local read/validate/replace RunHead transaction."""
        return _directory_lock(self.model_root)

    def _read_head_version(self) -> tuple[bytes, str | None] | None:
        if not self._head_path.exists():
            return None
        if not self._head_path.is_file():
            raise ArtifactValidationError(
                f"Checkpoint head must be a regular file: {self._head_path}"
            )
        return self._head_path.read_bytes(), None

    def _read_head_bytes(self) -> bytes | None:
        version = self._read_head_version()
        return None if version is None else version[0]

    def _compare_and_set_head_version(
        self,
        *,
        expected: tuple[bytes, str | None] | None,
        target: bytes,
    ) -> RunHeadV2:
        current = self._read_head_version()
        if current != expected:
            raise ArtifactValidationError("Run head changed during compare-and-set")
        _atomic_replace(self._head_path, target)
        return RunHeadV2.from_bytes(target)

    def publish_run_commit_bytes(self, run_commit_id: str, data: bytes) -> None:
        _require_digest(run_commit_id, field="run_commit_id")
        if not isinstance(data, bytes):
            raise ArtifactValidationError("Run commit must be bytes")
        _decode_json(data, context="run commit")
        if sha256_bytes(data) != run_commit_id:
            raise ArtifactValidationError("Run commit SHA-256 does not match its ID")
        _create_or_verify(self._run_commit_path(run_commit_id), data)

    def read_run_commit_bytes(self, run_commit_id: str) -> bytes:
        _require_digest(run_commit_id, field="run_commit_id")
        path = self._run_commit_path(run_commit_id)
        if not path.is_file():
            raise ArtifactValidationError(f"Run commit does not exist: {run_commit_id}")
        data = path.read_bytes()
        if sha256_bytes(data) != run_commit_id:
            raise ArtifactValidationError("Run commit SHA-256 does not match its ID")
        _decode_json(data, context="run commit")
        return data

    def publish_run_preparation_bytes(self, parent_run_commit_id: str | None, data: bytes) -> None:
        if parent_run_commit_id is not None:
            _require_digest(parent_run_commit_id, field="parent_run_commit_id")
        if not isinstance(data, bytes):
            raise ArtifactValidationError("Run preparation must be bytes")
        _decode_json(data, context="run preparation")
        _create_or_verify(self._run_preparation_path(parent_run_commit_id), data)

    def read_run_preparation_bytes(self, parent_run_commit_id: str | None) -> bytes | None:
        if parent_run_commit_id is not None:
            _require_digest(parent_run_commit_id, field="parent_run_commit_id")
        path = self._run_preparation_path(parent_run_commit_id)
        if not path.exists():
            return None
        if not path.is_file():
            raise ArtifactValidationError(f"Run preparation must be a regular file: {path}")
        data = path.read_bytes()
        _decode_json(data, context="run preparation")
        return data

    def _read_manifest_bytes(self, checkpoint_id: str) -> bytes | None:
        path = self._manifest_path(checkpoint_id)
        return path.read_bytes() if path.is_file() else None

    def _load_manifest(
        self,
        checkpoint_id: str,
        *,
        expected_config_sha256: str | None = None,
    ) -> CheckpointManifestV1:
        _require_digest(checkpoint_id, field="checkpoint_id")
        if expected_config_sha256 is not None:
            _require_digest(expected_config_sha256, field="expected_config_sha256")
        manifest_data = self._read_manifest_bytes(checkpoint_id)
        if manifest_data is None:
            raise ArtifactValidationError(f"Checkpoint manifest does not exist for {checkpoint_id}")
        if sha256_bytes(manifest_data) != checkpoint_id:
            raise ArtifactValidationError(
                "Checkpoint manifest SHA-256 does not match checkpoint ID"
            )
        manifest = CheckpointManifestV1.from_bytes(manifest_data)
        self._validate_manifest_profile(manifest, expected_config_sha256)
        return manifest

    def _validate_manifest_lineage(self, manifest: CheckpointManifestV1) -> None:
        """Verify that every declared parent exists and steps strictly increase."""
        child = manifest
        seen = {manifest.checkpoint_id}
        while child.parent_checkpoint_id is not None:
            parent_id = child.parent_checkpoint_id
            if parent_id in seen:
                raise ArtifactValidationError("Checkpoint lineage contains a cycle")
            parent = self._load_manifest(
                parent_id,
                expected_config_sha256=manifest.config_sha256,
            )
            if parent.step >= child.step:
                raise ArtifactValidationError(
                    "Checkpoint lineage steps must strictly increase: "
                    f"parent {parent_id} has step {parent.step}, child has step {child.step}"
                )
            seen.add(parent_id)
            child = parent

    def _validate_staged_checkpoint(self, manifest: CheckpointManifestV1) -> None:
        """Validate immutable checkpoint ancestry independently of mutable state.

        Staging is deliberately allowed to leave an orphan.  Only committing a
        RunHead decides whether that checkpoint is an inherited checkpoint or a
        direct child of the authoritative checkpoint.
        """
        if manifest.parent_checkpoint_id is not None:
            parent = self.read_checkpoint_manifest_exact(
                manifest.parent_checkpoint_id,
                expected_config_sha256=manifest.config_sha256,
            )
            if manifest.step <= parent.step:
                raise ArtifactValidationError(
                    "Checkpoint step must strictly increase from its parent: "
                    f"got {manifest.step}, parent step is {parent.step}"
                )

    def _materialize_immutables(
        self,
        manifest: CheckpointManifestV1,
        onnx_data: bytes,
        learner_data: bytes,
    ) -> CheckpointRef:
        checkpoint_id = manifest.checkpoint_id
        onnx_path = self._blob_path(manifest.onnx, "onnx")
        learner_path = self._blob_path(manifest.learner_state, "pt")
        _create_or_verify(onnx_path, onnx_data)
        _create_or_verify(learner_path, learner_data)
        _create_or_verify(self._manifest_path(checkpoint_id), manifest.to_bytes())
        return CheckpointRef(checkpoint_id, manifest, onnx_path, learner_path)

    def stage_checkpoint(
        self,
        onnx_path: Path,
        learner_state_path: Path,
        *,
        step: int,
        parent_checkpoint_id: str | None,
        config_sha256: str,
        learner_state_contract: object,
    ) -> CheckpointRef:
        onnx_path = Path(onnx_path)
        learner_state_path = Path(learner_state_path)
        manifest, onnx_data, learner_data = _snapshot_checkpoint_artifacts(
            onnx_path,
            learner_state_path,
            contract=self.contract,
            step=step,
            parent_checkpoint_id=parent_checkpoint_id,
            config_sha256=config_sha256,
            learner_state_contract=learner_state_contract,
        )
        self._validate_staged_checkpoint(manifest)
        checkpoint = self._materialize_immutables(manifest, onnx_data, learner_data)
        logger.info("Staged immutable checkpoint %s", checkpoint.checkpoint_id)
        return checkpoint

    @staticmethod
    def _decode_typed_run_commit(data: bytes):
        try:
            from .run_commit import RunCommitV1
        except ImportError as exc:
            raise ArtifactValidationError("RunCommitV1 support is unavailable") from exc
        try:
            return RunCommitV1.from_bytes(data)
        except ArtifactValidationError:
            raise
        except Exception as exc:
            raise ArtifactValidationError(f"Invalid run commit: {exc}") from exc

    def _resolve_run_commit_chain(self, run_commit_id: str):
        from .evaluation import create_evaluation_repository
        from .run_commit import RunCommitRepository

        repository = RunCommitRepository(
            self,
            create_evaluation_repository(self),
        )
        return repository.resolve_chain(run_commit_id)

    def commit_run_head(
        self,
        *,
        checkpoint_id: str,
        run_commit_id: str,
        expected_run_commit_id: str | None,
    ) -> RunHeadV2:
        """Atomically select one validated RunCommit and its checkpoint."""
        _require_digest(checkpoint_id, field="checkpoint_id")
        _require_digest(run_commit_id, field="run_commit_id")
        if expected_run_commit_id is not None:
            _require_digest(expected_run_commit_id, field="expected_run_commit_id")
        run_commit = self._decode_typed_run_commit(self.read_run_commit_bytes(run_commit_id))
        if run_commit.run_commit_id != run_commit_id:
            raise ArtifactValidationError("Run commit identity mismatch")
        if run_commit.checkpoint_id != checkpoint_id:
            raise ArtifactValidationError(
                "Run commit checkpoint_id does not match the proposed run head"
            )
        if run_commit.profile != self.contract.profile:
            raise ArtifactValidationError("Run commit profile mismatch")
        checkpoint = self.resolve_checkpoint(
            checkpoint_id,
            expected_config_sha256=run_commit.config_sha256,
        )
        target = RunHeadV2(
            checkpoint_id=checkpoint.checkpoint_id,
            run_commit_id=run_commit_id,
        )
        target_data = target.to_bytes()

        with self.run_head_commit_guard():
            current_version = self._read_head_version()
            current = None if current_version is None else RunHeadV2.from_bytes(current_version[0])
            chain = self._resolve_run_commit_chain(run_commit_id)
            if not chain or chain[-1].run_commit_id != run_commit_id:
                raise ArtifactValidationError("RunCommit lineage is incomplete")
            if current == target:
                return target
            actual_parent_commit_id = None if current is None else current.run_commit_id
            if actual_parent_commit_id != expected_run_commit_id:
                raise ArtifactValidationError(
                    "Run head changed before commit: "
                    f"got {actual_parent_commit_id!r}, expected "
                    f"{expected_run_commit_id!r}"
                )
            if run_commit.parent_run_commit_id != actual_parent_commit_id:
                raise ArtifactValidationError(
                    "Run commit parent must equal the authoritative run head"
                )
            if current is not None:
                current_manifest = self.read_checkpoint_manifest_exact(
                    current.checkpoint_id,
                    expected_config_sha256=run_commit.config_sha256,
                )
                if checkpoint.checkpoint_id == current.checkpoint_id:
                    raise ArtifactValidationError(
                        "A RunCommit must select a new direct-child checkpoint"
                    )
                if checkpoint.manifest.parent_checkpoint_id != current.checkpoint_id:
                    raise ArtifactValidationError(
                        "A new RunCommit checkpoint must directly extend the current "
                        "RunHead checkpoint"
                    )
                if checkpoint.manifest.step <= current_manifest.step:
                    raise ArtifactValidationError(
                        "RunCommit checkpoint step must strictly increase"
                    )
            elif checkpoint.manifest.parent_checkpoint_id is not None:
                raise ArtifactValidationError("The first RunCommit must select a root checkpoint")

            observed = self._compare_and_set_head_version(
                expected=current_version,
                target=target_data,
            )
        return observed

    def resolve_run_head(self) -> RunHeadV2 | None:
        head_data = self._read_head_bytes()
        if head_data is None:
            return None
        head = RunHeadV2.from_bytes(head_data)
        chain = self._resolve_run_commit_chain(head.run_commit_id)
        if not chain:
            raise ArtifactValidationError("Run head has no RunCommit lineage")
        run_commit = chain[-1].commit
        if run_commit.checkpoint_id != head.checkpoint_id:
            raise ArtifactValidationError(
                "Run head checkpoint does not match its immutable RunCommit"
            )
        self.resolve_checkpoint(
            head.checkpoint_id,
            expected_config_sha256=run_commit.config_sha256,
        )
        return head

    def resolve_head(self, *, expected_config_sha256: str | None = None) -> CheckpointRef | None:
        if expected_config_sha256 is not None:
            _require_digest(expected_config_sha256, field="expected_config_sha256")
        head = self.resolve_run_head()
        if head is None:
            return None
        return self.resolve_checkpoint(
            head.checkpoint_id,
            expected_config_sha256=expected_config_sha256,
        )

    def resolve_checkpoint(
        self,
        checkpoint_id: str,
        *,
        expected_config_sha256: str | None = None,
    ) -> CheckpointRef:
        manifest = self._load_manifest(
            checkpoint_id,
            expected_config_sha256=expected_config_sha256,
        )
        self._validate_manifest_lineage(manifest)
        onnx_path = self._blob_path(manifest.onnx, "onnx")
        learner_path = self._blob_path(manifest.learner_state, "pt")
        self._verify_local_blob(onnx_path, manifest.onnx, name="ONNX blob")
        self._verify_local_blob(learner_path, manifest.learner_state, name="learner-state blob")
        validate_onnx_checkpoint(onnx_path, self.contract)
        _validate_learner_state(
            learner_path,
            contract=self.contract,
            step=manifest.step,
            config_sha256=manifest.config_sha256,
        )
        return CheckpointRef(checkpoint_id, manifest, onnx_path, learner_path)

    def resolve_checkpoint_manifest(
        self,
        checkpoint_id: str,
        *,
        expected_config_sha256: str | None = None,
    ) -> CheckpointManifestV1:
        """Resolve strict manifest ancestry without reading historical blobs."""
        manifest = self._load_manifest(
            checkpoint_id,
            expected_config_sha256=expected_config_sha256,
        )
        self._validate_manifest_lineage(manifest)
        return manifest

    def read_checkpoint_manifest_exact(
        self,
        checkpoint_id: str,
        *,
        expected_config_sha256: str | None = None,
    ) -> CheckpointManifestV1:
        """Read and validate exactly one immutable manifest, without ancestry."""
        return self._load_manifest(
            checkpoint_id,
            expected_config_sha256=expected_config_sha256,
        )

    def list_checkpoints(self) -> list[CheckpointRef]:
        directory = self.model_root / "manifests" / "sha256"
        if not directory.exists():
            return []
        if not directory.is_dir():
            raise ArtifactValidationError(
                f"Checkpoint manifest store must be a directory: {directory}"
            )
        checkpoint_ids: list[str] = []
        for path in directory.iterdir():
            if not path.is_file() or path.suffix != ".json":
                raise ArtifactValidationError(
                    f"Unexpected object in checkpoint manifest store: {path}"
                )
            checkpoint_id = path.stem
            _require_digest(checkpoint_id, field="manifest filename checkpoint_id")
            checkpoint_ids.append(checkpoint_id)
        checkpoints = [self.resolve_checkpoint(checkpoint_id) for checkpoint_id in checkpoint_ids]
        checkpoints.sort(
            key=lambda checkpoint: (
                checkpoint.manifest.step,
                checkpoint.checkpoint_id,
            )
        )
        return checkpoints

    def _validate_manifest_profile(
        self,
        manifest: CheckpointManifestV1,
        expected_config_sha256: str | None,
    ) -> None:
        if manifest.profile != self.contract.profile:
            raise ArtifactValidationError(
                "Checkpoint profile mismatch: "
                f"got {manifest.profile.to_dict()}, expected {self.contract.profile.to_dict()}"
            )
        if expected_config_sha256 is not None and manifest.config_sha256 != expected_config_sha256:
            raise ArtifactValidationError(
                "Checkpoint config_sha256 mismatch: "
                f"got {manifest.config_sha256}, expected {expected_config_sha256}"
            )

    @staticmethod
    def _verify_local_blob(path: Path, descriptor: BlobDescriptorV1, *, name: str) -> None:
        if not path.is_file():
            raise ArtifactValidationError(f"{name} does not exist: {path}")
        _verified_blob(path.read_bytes(), descriptor, name=name)


def _is_s3_missing(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error", {})
    return isinstance(error, dict) and error.get("Code") in {
        "404",
        "NoSuchKey",
        "NotFound",
    }


def _s3_error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error", {})
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return code if isinstance(code, str) else None


def _is_s3_conflict(exc: Exception) -> bool:
    return _s3_error_code(exc) in {"409", "ConditionalRequestConflict"}


def _is_s3_precondition_failed(exc: Exception) -> bool:
    return _s3_error_code(exc) in {
        "409",
        "412",
        "ConditionalRequestConflict",
        "PreconditionFailed",
    }


class S3CheckpointPublisher(FilesystemCheckpointPublisher):
    """S3 repository with a verified local content-addressed artifact cache."""

    def __init__(
        self,
        *,
        model_root: str | Path,
        bucket: str,
        contract: OnnxArtifactContract,
        endpoint: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not isinstance(bucket, str) or not bucket.strip():
            raise ValueError("S3 checkpoint publisher requires a non-empty bucket")
        super().__init__(model_root=model_root, contract=contract)
        self.bucket = bucket
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "S3 checkpoint publication requires the declared boto3 dependency"
                ) from exc
            client = boto3.client("s3", endpoint_url=endpoint)
        self._client = client

    @property
    def prefix(self) -> str:
        return RuntimeProfile(
            self.contract.algorithm_id,
            self.contract.env_id,
            self.contract.env_contract_version,
        ).model_prefix

    def _key(self, relative: str) -> str:
        return f"{self.prefix}/{relative}"

    @property
    def _head_key(self) -> str:
        return self._key("channels/current.json")

    def _get(self, key: str) -> bytes | None:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _is_s3_missing(exc):
                return None
            raise
        body = response["Body"]
        data = body.read()
        if not isinstance(data, bytes):
            raise ArtifactValidationError(f"S3 object body is not bytes: {key}")
        return data

    def _get_versioned(self, key: str) -> tuple[bytes, str] | None:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _is_s3_missing(exc):
                return None
            raise
        body = response.get("Body")
        data = body.read() if body is not None else None
        etag = response.get("ETag")
        if not isinstance(data, bytes) or not isinstance(etag, str) or not etag:
            raise ArtifactValidationError(
                f"S3 object lacks a byte body or ETag required for CAS: {key}"
            )
        return data, etag

    def _read_head_version(self) -> tuple[bytes, str] | None:
        return self._get_versioned(self._head_key)

    def _read_manifest_bytes(self, checkpoint_id: str) -> bytes | None:
        return self._get(self._key(f"manifests/sha256/{checkpoint_id}.json"))

    def publish_run_commit_bytes(self, run_commit_id: str, data: bytes) -> None:
        _require_digest(run_commit_id, field="run_commit_id")
        if not isinstance(data, bytes):
            raise ArtifactValidationError("Run commit must be bytes")
        _decode_json(data, context="run commit")
        if sha256_bytes(data) != run_commit_id:
            raise ArtifactValidationError("Run commit SHA-256 does not match its ID")
        self._put_immutable(
            self._key(f"run-commits/sha256/{run_commit_id}.json"),
            data,
            "application/json",
        )

    def read_run_commit_bytes(self, run_commit_id: str) -> bytes:
        _require_digest(run_commit_id, field="run_commit_id")
        data = self._get(self._key(f"run-commits/sha256/{run_commit_id}.json"))
        if data is None:
            raise ArtifactValidationError(f"Run commit does not exist: {run_commit_id}")
        if sha256_bytes(data) != run_commit_id:
            raise ArtifactValidationError("Run commit SHA-256 does not match its ID")
        _decode_json(data, context="run commit")
        return data

    def publish_run_preparation_bytes(self, parent_run_commit_id: str | None, data: bytes) -> None:
        if parent_run_commit_id is not None:
            _require_digest(parent_run_commit_id, field="parent_run_commit_id")
        if not isinstance(data, bytes):
            raise ArtifactValidationError("Run preparation must be bytes")
        _decode_json(data, context="run preparation")
        name = "root" if parent_run_commit_id is None else parent_run_commit_id
        self._put_immutable(
            self._key(f"run-preparations/by-parent/{name}.json"),
            data,
            "application/json",
        )

    def read_run_preparation_bytes(self, parent_run_commit_id: str | None) -> bytes | None:
        if parent_run_commit_id is not None:
            _require_digest(parent_run_commit_id, field="parent_run_commit_id")
        name = "root" if parent_run_commit_id is None else parent_run_commit_id
        data = self._get(self._key(f"run-preparations/by-parent/{name}.json"))
        if data is not None:
            _decode_json(data, context="run preparation")
        return data

    def run_head_commit_guard(self):
        """S3 RunHead serialization is its exact ETag conditional write."""
        return nullcontext()

    def _put_immutable(self, key: str, data: bytes, content_type: str) -> None:
        existing = self._get(key)
        if existing is not None:
            if existing != data:
                raise ArtifactValidationError(
                    f"Immutable S3 checkpoint object differs: s3://{self.bucket}/{key}"
                )
            return
        for attempt in range(_S3_CONDITIONAL_WRITE_ATTEMPTS):
            try:
                self._client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=data,
                    ContentType=content_type,
                    IfNoneMatch="*",
                )
                return
            except Exception as exc:
                raced = self._get(key)
                if raced == data:
                    return
                if not _is_s3_precondition_failed(exc):
                    raise
                if (
                    _is_s3_conflict(exc)
                    and raced is None
                    and attempt + 1 < _S3_CONDITIONAL_WRITE_ATTEMPTS
                ):
                    time.sleep(0.01 * (2**attempt))
                    continue
                raise ArtifactValidationError(
                    f"Immutable S3 checkpoint object raced with different bytes: {key}"
                ) from exc

    def _compare_and_set_head_version(
        self,
        *,
        expected: tuple[bytes, str | None] | None,
        target: bytes,
    ) -> RunHeadV2:
        if expected is not None and expected[1] is None:
            raise ArtifactValidationError("S3 run-head CAS requires an ETag")
        condition = {"IfNoneMatch": "*"} if expected is None else {"IfMatch": expected[1]}
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=self._head_key,
                Body=target,
                ContentType="application/json",
                **condition,
            )
        except Exception as exc:
            confirmed = self._get_versioned(self._head_key)
            # The write can have committed even when the response was lost.
            if confirmed is not None and self._head_selects_or_descends_from(confirmed[0], target):
                return RunHeadV2.from_bytes(confirmed[0])
            if _is_s3_precondition_failed(exc):
                raise ArtifactValidationError(
                    "S3 run head changed during conditional compare-and-set"
                ) from exc
            raise
        confirmed = self._get_versioned(self._head_key)
        if confirmed is None or not self._head_selects_or_descends_from(confirmed[0], target):
            raise ArtifactValidationError("S3 run head update could not be confirmed")
        return RunHeadV2.from_bytes(confirmed[0])

    def _head_selects_or_descends_from(self, confirmed_data: bytes, target_data: bytes) -> bool:
        if confirmed_data == target_data:
            return True
        confirmed = RunHeadV2.from_bytes(confirmed_data)
        target = RunHeadV2.from_bytes(target_data)
        chain = self._resolve_run_commit_chain(confirmed.run_commit_id)
        if not chain or chain[-1].commit.checkpoint_id != confirmed.checkpoint_id:
            raise ArtifactValidationError(
                "Confirmed S3 run head does not match its RunCommit checkpoint"
            )
        return any(
            reference.run_commit_id == target.run_commit_id
            and reference.commit.checkpoint_id == target.checkpoint_id
            for reference in chain
        )

    def stage_checkpoint(
        self,
        onnx_path: Path,
        learner_state_path: Path,
        *,
        step: int,
        parent_checkpoint_id: str | None,
        config_sha256: str,
        learner_state_contract: object,
    ) -> CheckpointRef:
        onnx_path = Path(onnx_path)
        learner_state_path = Path(learner_state_path)
        manifest, onnx_data, learner_data = _snapshot_checkpoint_artifacts(
            onnx_path,
            learner_state_path,
            contract=self.contract,
            step=step,
            parent_checkpoint_id=parent_checkpoint_id,
            config_sha256=config_sha256,
            learner_state_contract=learner_state_contract,
        )
        self._validate_staged_checkpoint(manifest)
        checkpoint = self._materialize_immutables(manifest, onnx_data, learner_data)
        checkpoint_id = checkpoint.checkpoint_id
        self._put_immutable(
            self._key(f"blobs/sha256/{manifest.onnx.sha256}.onnx"),
            onnx_data,
            "application/octet-stream",
        )
        self._put_immutable(
            self._key(f"blobs/sha256/{manifest.learner_state.sha256}.pt"),
            learner_data,
            "application/octet-stream",
        )
        self._put_immutable(
            self._key(f"manifests/sha256/{checkpoint_id}.json"),
            manifest.to_bytes(),
            "application/json",
        )
        return checkpoint

    def resolve_checkpoint(
        self,
        checkpoint_id: str,
        *,
        expected_config_sha256: str | None = None,
    ) -> CheckpointRef:
        manifest = self._load_manifest(
            checkpoint_id,
            expected_config_sha256=expected_config_sha256,
        )
        self._validate_manifest_lineage(manifest)
        onnx_data = self._get(self._key(f"blobs/sha256/{manifest.onnx.sha256}.onnx"))
        learner_data = self._get(self._key(f"blobs/sha256/{manifest.learner_state.sha256}.pt"))
        if onnx_data is None or learner_data is None:
            raise ArtifactValidationError("S3 checkpoint is missing a blob")
        _verified_blob(onnx_data, manifest.onnx, name="S3 ONNX blob")
        _verified_blob(learner_data, manifest.learner_state, name="S3 learner-state blob")
        checkpoint = self._materialize_immutables(manifest, onnx_data, learner_data)
        validate_onnx_checkpoint(checkpoint.onnx_path, self.contract)
        _validate_learner_state(
            checkpoint.learner_state_path,
            contract=self.contract,
            step=manifest.step,
            config_sha256=manifest.config_sha256,
        )
        return checkpoint

    def resolve_checkpoint_manifest(
        self,
        checkpoint_id: str,
        *,
        expected_config_sha256: str | None = None,
    ) -> CheckpointManifestV1:
        """Resolve strict S3 manifest ancestry without downloading model blobs."""
        manifest = self._load_manifest(
            checkpoint_id,
            expected_config_sha256=expected_config_sha256,
        )
        self._validate_manifest_lineage(manifest)
        return manifest

    def read_checkpoint_manifest_exact(
        self,
        checkpoint_id: str,
        *,
        expected_config_sha256: str | None = None,
    ) -> CheckpointManifestV1:
        """Read one strict S3 manifest without fetching its ancestry or blobs."""
        return self._load_manifest(
            checkpoint_id,
            expected_config_sha256=expected_config_sha256,
        )

    def list_checkpoints(self) -> list[CheckpointRef]:
        prefix = self._key("manifests/sha256/")
        checkpoint_ids: list[str] = []
        continuation_token: str | None = None
        while True:
            request: dict[str, object] = {
                "Bucket": self.bucket,
                "Prefix": prefix,
            }
            if continuation_token is not None:
                request["ContinuationToken"] = continuation_token
            response = self._client.list_objects_v2(**request)
            contents = response.get("Contents", [])
            if not isinstance(contents, list):
                raise ArtifactValidationError("S3 checkpoint listing Contents is invalid")
            for item in contents:
                if not isinstance(item, dict) or not isinstance(item.get("Key"), str):
                    raise ArtifactValidationError("S3 checkpoint listing entry is invalid")
                key = item["Key"]
                relative = key.removeprefix(prefix)
                if not key.startswith(prefix) or "/" in relative or not relative.endswith(".json"):
                    raise ArtifactValidationError(
                        f"Unexpected S3 checkpoint manifest object: {key}"
                    )
                checkpoint_id = relative.removesuffix(".json")
                _require_digest(checkpoint_id, field="S3 manifest filename checkpoint_id")
                checkpoint_ids.append(checkpoint_id)
            truncated = response.get("IsTruncated", False)
            if not isinstance(truncated, bool):
                raise ArtifactValidationError("S3 checkpoint listing IsTruncated is invalid")
            if not truncated:
                break
            next_token = response.get("NextContinuationToken")
            if not isinstance(next_token, str) or not next_token:
                raise ArtifactValidationError(
                    "Truncated S3 checkpoint listing has no continuation token"
                )
            continuation_token = next_token
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ArtifactValidationError("S3 checkpoint listing contains duplicate objects")
        checkpoints = [self.resolve_checkpoint(checkpoint_id) for checkpoint_id in checkpoint_ids]
        checkpoints.sort(
            key=lambda checkpoint: (
                checkpoint.manifest.step,
                checkpoint.checkpoint_id,
            )
        )
        return checkpoints


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
