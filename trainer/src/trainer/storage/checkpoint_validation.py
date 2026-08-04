"""ONNX, learner-state, and checkpoint snapshot validation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto

from .artifact_codec import ArtifactValidationError, sha256_bytes
from .checkpoint_types import (
    BlobDescriptorV1,
    CheckpointManifestV1,
    OnnxArtifactContract,
    OnnxTensorSpec,
)


def _fail(path: Path, message: str) -> ArtifactValidationError:
    return ArtifactValidationError(f"Invalid ONNX checkpoint {path}: {message}")


def _validate_tensor(path: Path, value_info: Any, *, spec: OnnxTensorSpec) -> None:
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
        raise _fail(path, f"'{name}' must have rank {len(spec.shape)}, got rank {len(dims)}")
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
            f"got {sorted(actual_contract_keys)}, expected {sorted(expected_metadata)}",
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


def verified_blob(data: bytes, descriptor: BlobDescriptorV1, *, name: str) -> None:
    if len(data) != descriptor.size_bytes:
        raise ArtifactValidationError(
            f"{name} size mismatch: got {len(data)}, expected {descriptor.size_bytes}"
        )
    digest = sha256_bytes(data)
    if digest != descriptor.sha256:
        raise ArtifactValidationError(
            f"{name} SHA-256 mismatch: got {digest}, expected {descriptor.sha256}"
        )


def validate_learner_state(
    path: Path,
    *,
    contract: OnnxArtifactContract,
    step: int,
    config_sha256: str,
    learner_state_contract: object | None = None,
) -> None:
    from ..checkpoint import validate_learner_state_artifact

    validate_learner_state_artifact(
        path,
        artifact_contract=contract,
        step=step,
        config_sha256=config_sha256,
        learner_state_contract=learner_state_contract,
    )


def snapshot_checkpoint_artifacts(
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
        validate_learner_state(
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
