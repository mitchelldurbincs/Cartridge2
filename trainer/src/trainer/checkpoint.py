"""Create and restore the two immutable blobs in a checkpoint manifest."""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from crucible.atomic_io import atomic_write

from .storage.publisher import (
    ArtifactValidationError,
    CheckpointManifestV1,
    OnnxArtifactContract,
    canonical_json_bytes,
    sha256_bytes,
    validate_onnx_checkpoint,
    validate_sha256_digest,
)

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer

_LEARNER_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "profile",
        "step",
        "config_sha256",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }
)
# Deep float32 convolution stacks accumulate backend-specific reduction error
# near 1e-4 while still representing the same policy/value function.
_ONNX_EQUIVALENCE_RTOL = 1e-4
_ONNX_EQUIVALENCE_ATOL = 1e-4


def _assert_same_state_structure(expected: Any, actual: Any, *, path: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise ArtifactValidationError(f"{path} must be a tensor")
        if tuple(actual.shape) != tuple(expected.shape):
            raise ArtifactValidationError(
                f"{path} shape mismatch: got {tuple(actual.shape)}, "
                f"expected {tuple(expected.shape)}"
            )
        if actual.dtype != expected.dtype:
            raise ArtifactValidationError(
                f"{path} dtype mismatch: got {actual.dtype}, expected {expected.dtype}"
            )
        if not torch.equal(actual, expected):
            raise ArtifactValidationError(f"{path} values do not match the live learner")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ArtifactValidationError(f"{path} must be a mapping")
        if set(actual) != set(expected):
            raise ArtifactValidationError(
                f"{path} keys mismatch (missing={sorted(set(expected) - set(actual), key=repr)}, "
                f"extra={sorted(set(actual) - set(expected), key=repr)})"
            )
        for key, expected_value in expected.items():
            _assert_same_state_structure(
                expected_value,
                actual[key],
                path=f"{path}[{key!r}]",
            )
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise ArtifactValidationError(
                f"{path} sequence type/length does not match the live learner"
            )
        for index, (expected_value, actual_value) in enumerate(zip(expected, actual)):
            _assert_same_state_structure(
                expected_value,
                actual_value,
                path=f"{path}[{index}]",
            )
        return
    if expected is None:
        if actual is not None:
            raise ArtifactValidationError(f"{path} must be null")
        return
    if type(actual) is not type(expected):
        raise ArtifactValidationError(
            f"{path} type mismatch: got {type(actual).__name__}, expected {type(expected).__name__}"
        )
    if actual != expected:
        raise ArtifactValidationError(f"{path} value does not match the live learner")


def _clone_state_to_cpu(value: Any) -> Any:
    """Snapshot nested state without retaining accelerator allocations."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, Mapping):
        return {key: _clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


class LearnerStateContract:
    """Exact CPU snapshot used to prove a staged artifact matches live state."""

    def __init__(
        self,
        network: "nn.Module",
        optimizer: "Optimizer",
        scheduler: Any | None = None,
    ) -> None:
        try:
            self._model_state = _clone_state_to_cpu(network.state_dict())
            self._optimizer_state = _clone_state_to_cpu(optimizer.state_dict())
            self._scheduler_state = (
                _clone_state_to_cpu(scheduler.state_dict()) if scheduler is not None else None
            )
        except Exception as exc:
            raise ArtifactValidationError(
                f"Cannot snapshot the live learner-state contract: {exc}"
            ) from exc

    def validate(self, state: Mapping[str, Any]) -> None:
        _assert_same_state_structure(
            self._model_state,
            state["model_state_dict"],
            path="Learner model_state_dict",
        )
        _assert_same_state_structure(
            self._optimizer_state,
            state["optimizer_state_dict"],
            path="Learner optimizer_state_dict",
        )
        _assert_same_state_structure(
            self._scheduler_state,
            state["scheduler_state_dict"],
            path="Learner scheduler_state_dict",
        )


def learner_config_recipe(config: object) -> dict[str, object]:
    """Return the algorithm-owned canonical semantic learner recipe."""
    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise TypeError("learner config requires a dataclass instance")
    builder = getattr(config, "learner_recipe", None)
    if not callable(builder):
        raise TypeError("learner config must define learner_recipe()")
    value = builder()
    if not isinstance(value, dict) or not value:
        raise TypeError("learner_recipe() must return a nonempty object")
    # Round-trip through the canonical encoder now so invalid/non-finite values
    # fail at config construction rather than checkpoint publication.
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def learner_config_sha256(config: object) -> str:
    """Hash only the exact semantic learner recipe."""
    return sha256_bytes(canonical_json_bytes(learner_config_recipe(config)))


def _validate_onnx_runtime_equivalence(
    network: "nn.Module",
    checkpoint_path: str | Path,
    *,
    artifact_contract: OnnxArtifactContract,
    device: torch.device,
) -> None:
    """Fail closed unless ONNX Runtime matches the live eval-mode network."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        session = ort.InferenceSession(
            str(checkpoint_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise ArtifactValidationError(
            f"ONNX Runtime cannot load exported checkpoint: {exc}"
        ) from exc

    output_names = [spec.name for spec in artifact_contract.outputs]
    for batch_size in (1, 3):
        inputs = _deterministic_inputs(artifact_contract, batch_size, device)
        try:
            with torch.no_grad():
                expected_outputs = network(*inputs)
        except Exception as exc:
            raise ArtifactValidationError(
                f"Live network failed deterministic validation at batch {batch_size}: {exc}"
            ) from exc
        if isinstance(expected_outputs, torch.Tensor):
            expected_outputs = (expected_outputs,)
        if not isinstance(expected_outputs, (tuple, list)) or len(expected_outputs) != len(
            output_names
        ):
            raise ArtifactValidationError(
                "Live network output count does not match the artifact contract"
            )
        try:
            actual_outputs = session.run(
                output_names,
                {
                    spec.name: value.detach().cpu().numpy()
                    for spec, value in zip(artifact_contract.inputs, inputs, strict=True)
                },
            )
        except Exception as exc:
            raise ArtifactValidationError(
                f"ONNX Runtime execution failed at batch {batch_size}: {exc}"
            ) from exc
        if not isinstance(actual_outputs, (tuple, list)) or len(actual_outputs) != len(
            output_names
        ):
            raise ArtifactValidationError(
                "ONNX Runtime output count does not match the artifact contract"
            )

        for name, expected, actual in zip(
            output_names, expected_outputs, actual_outputs, strict=True
        ):
            if not isinstance(expected, torch.Tensor):
                raise ArtifactValidationError(f"Live network output '{name}' must be a tensor")
            expected_array = expected.detach().cpu().numpy()
            actual_array = np.asarray(actual)
            if actual_array.shape != expected_array.shape:
                raise ArtifactValidationError(
                    f"ONNX Runtime output '{name}' shape mismatch at batch {batch_size}: "
                    f"got {actual_array.shape}, expected {expected_array.shape}"
                )
            if actual_array.dtype != np.dtype(np.float32):
                raise ArtifactValidationError(
                    f"ONNX Runtime output '{name}' must use float32, got {actual_array.dtype}"
                )
            if not np.isfinite(expected_array).all():
                raise ArtifactValidationError(
                    f"Live network output '{name}' is non-finite at batch {batch_size}"
                )
            if not np.isfinite(actual_array).all():
                raise ArtifactValidationError(
                    f"ONNX Runtime output '{name}' is non-finite at batch {batch_size}"
                )
            if not np.allclose(
                actual_array,
                expected_array,
                rtol=_ONNX_EQUIVALENCE_RTOL,
                atol=_ONNX_EQUIVALENCE_ATOL,
                equal_nan=False,
            ):
                max_abs_diff = float(np.max(np.abs(actual_array - expected_array)))
                raise ArtifactValidationError(
                    f"ONNX Runtime output '{name}' differs from the live network at "
                    f"batch {batch_size} (max_abs_diff={max_abs_diff:.8g})"
                )


def _deterministic_inputs(
    contract: OnnxArtifactContract,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    inputs = []
    for spec in contract.inputs:
        if (
            not spec.shape
            or spec.shape[0] != "batch_size"
            or any(not isinstance(dimension, int) for dimension in spec.shape[1:])
        ):
            raise ValueError(
                f"ONNX exporter requires {spec.name!r} shape to be "
                "('batch_size', <fixed dimensions...>)"
            )
        fixed_shape = tuple(int(dimension) for dimension in spec.shape[1:])
        elements = math.prod(fixed_shape)
        value = torch.linspace(
            -1.0,
            1.0,
            steps=batch_size * elements,
            dtype=torch.float32,
            device=device,
        ).reshape(batch_size, *fixed_shape)
        inputs.append(value)
    return tuple(inputs)


def _validate_live_outputs(
    network: "nn.Module",
    inputs: tuple[torch.Tensor, ...],
    contract: OnnxArtifactContract,
) -> tuple[torch.Tensor, ...]:
    try:
        with torch.no_grad():
            outputs = network(*inputs)
    except Exception as exc:
        raise ArtifactValidationError(
            f"Live network failed artifact-contract validation: {exc}"
        ) from exc
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    if not isinstance(outputs, (tuple, list)) or len(outputs) != len(contract.outputs):
        raise ArtifactValidationError(
            "Live network output count does not match the artifact contract"
        )
    batch_size = inputs[0].shape[0]
    validated = []
    for spec, output in zip(contract.outputs, outputs, strict=True):
        if not isinstance(output, torch.Tensor):
            raise ArtifactValidationError(f"Live network output '{spec.name}' must be a tensor")
        expected_shape = tuple(
            batch_size if dimension == "batch_size" else dimension for dimension in spec.shape
        )
        if tuple(output.shape) != expected_shape:
            raise ArtifactValidationError(
                f"Live network output '{spec.name}' shape is {tuple(output.shape)}, "
                f"expected {expected_shape}"
            )
        if output.dtype != torch.float32:
            raise ArtifactValidationError(f"Live network output '{spec.name}' must use float32")
        if not torch.isfinite(output).all():
            raise ArtifactValidationError(
                f"Live network output '{spec.name}' contains non-finite values"
            )
        validated.append(output)
    return tuple(validated)


def export_onnx_artifact(
    network: "nn.Module",
    output_path: str | Path,
    device: torch.device,
    artifact_contract: OnnxArtifactContract,
) -> Path:
    """Export one validated ONNX blob to a staging path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    training_modes = [(module, module.training) for module in network.modules()]
    network.eval()
    # torch.export specializes dimensions whose example value is 0 or 1. A
    # two-row example is therefore required for the declared batch dimension
    # to remain symbolic in convolutional cartridges as well as MLPs.
    dummy_inputs = _deterministic_inputs(artifact_contract, 2, device)
    _validate_live_outputs(network, dummy_inputs, artifact_contract)
    input_names = [spec.name for spec in artifact_contract.inputs]
    output_names = [spec.name for spec in artifact_contract.outputs]

    def _export(temp_path: str) -> None:
        torch.onnx.export(
            network,
            dummy_inputs,
            temp_path,
            export_params=True,
            opset_version=18,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=tuple({0: torch.export.Dim("batch_size")} for _ in dummy_inputs),
            dynamo=True,
            external_data=False,
            # Cartridge validation below is fail-closed. PyTorch's verify mode
            # only logs checker/runtime/numeric failures and still returns an
            # ONNX program, so it is not a publication gate.
            verify=False,
            optimize=True,
            verbose=False,
        )
        model = onnx.load(temp_path, load_external_data=False)
        metadata = {
            entry.key: entry.value
            for entry in model.metadata_props
            if not entry.key.startswith("cartridge.")
        }
        metadata.update(
            {
                "cartridge.schema_version": str(artifact_contract.model_artifact_schema_version),
                "cartridge.algorithm_id": artifact_contract.algorithm_id,
                "cartridge.model_contract": artifact_contract.model_contract,
                "cartridge.env_id": artifact_contract.env_id,
                "cartridge.env_contract_version": str(artifact_contract.env_contract_version),
            }
        )
        del model.metadata_props[:]
        for key, value in sorted(metadata.items()):
            entry = model.metadata_props.add()
            entry.key = key
            entry.value = value
        onnx.save_model(model, temp_path, save_as_external_data=False)
        validate_onnx_checkpoint(temp_path, artifact_contract)
        _validate_onnx_runtime_equivalence(
            network,
            temp_path,
            artifact_contract=artifact_contract,
            device=device,
        )

    try:
        atomic_write(path, _export)
    finally:
        for module, training in training_modes:
            module.training = training
    return path


def write_learner_state_artifact(
    network: "nn.Module",
    optimizer: "Optimizer",
    step: int,
    output_path: str | Path,
    artifact_contract: OnnxArtifactContract,
    config_sha256: str,
    scheduler: Any | None = None,
) -> Path:
    """Serialize learner continuity state to a staging path."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Learner checkpoint step must be a nonnegative integer")
    validate_sha256_digest(config_sha256, field="config_sha256")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": 1,
        "profile": artifact_contract.profile.to_dict(),
        "step": step,
        "config_sha256": config_sha256,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
    }
    atomic_write(path, lambda temp_path: torch.save(state, temp_path))
    return path


def _validate_learner_state_bytes(
    data: bytes,
    *,
    artifact_contract: OnnxArtifactContract,
    step: int,
    config_sha256: str,
    device: torch.device | str = "cpu",
    learner_state_contract: LearnerStateContract | None = None,
) -> dict[str, Any]:
    """Safely decode and validate the exact supplied learner-state bytes."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ArtifactValidationError("Learner-state expected step must be a nonnegative integer")
    validate_sha256_digest(config_sha256, field="config_sha256")
    try:
        state = torch.load(io.BytesIO(data), map_location=device, weights_only=True)
    except Exception as exc:
        raise ArtifactValidationError(
            f"Learner-state artifact cannot be safely decoded: {exc}"
        ) from exc
    if not isinstance(state, dict):
        raise ArtifactValidationError("Learner-state payload must be a mapping")
    actual_fields = frozenset(state)
    if actual_fields != _LEARNER_STATE_FIELDS:
        raise ArtifactValidationError(
            "Learner-state fields must be exact "
            f"(missing={sorted(_LEARNER_STATE_FIELDS - actual_fields)}, "
            f"extra={sorted(actual_fields - _LEARNER_STATE_FIELDS)})"
        )
    expected_scalars = {
        "schema_version": 1,
        "profile": artifact_contract.profile.to_dict(),
        "step": step,
        "config_sha256": config_sha256,
    }
    for field, expected in expected_scalars.items():
        actual = state[field]
        if actual != expected or (field in {"schema_version", "step"} and isinstance(actual, bool)):
            raise ArtifactValidationError(
                f"Learner-state {field} mismatch: got {actual!r}, expected {expected!r}"
            )
    for field in ("model_state_dict", "optimizer_state_dict"):
        if not isinstance(state[field], Mapping):
            raise ArtifactValidationError(f"Learner-state {field} must be a mapping")
    scheduler_state = state["scheduler_state_dict"]
    if scheduler_state is not None and not isinstance(scheduler_state, Mapping):
        raise ArtifactValidationError(
            "Learner-state scheduler_state_dict must be a mapping or null"
        )
    if learner_state_contract is not None:
        if not isinstance(learner_state_contract, LearnerStateContract):
            raise ArtifactValidationError("learner_state_contract must be LearnerStateContract")
        learner_state_contract.validate(state)
    return state


def validate_learner_state_artifact(
    checkpoint_path: str | Path,
    *,
    artifact_contract: OnnxArtifactContract,
    step: int,
    config_sha256: str,
    device: torch.device | str = "cpu",
    learner_state_contract: LearnerStateContract | None = None,
) -> dict[str, Any]:
    """Safely decode and validate a staged learner-state envelope.

    Publication calls this before creating any immutable object or advancing the
    repository head.  ``weights_only=True`` deliberately rejects arbitrary
    pickled objects at this trust boundary.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise ArtifactValidationError(f"Learner-state artifact does not exist: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ArtifactValidationError(f"Learner-state artifact cannot be read: {exc}") from exc
    return _validate_learner_state_bytes(
        data,
        artifact_contract=artifact_contract,
        step=step,
        config_sha256=config_sha256,
        device=device,
        learner_state_contract=learner_state_contract,
    )


def restore_learner_state(
    network: "nn.Module",
    optimizer: "Optimizer",
    checkpoint_path: str | Path,
    device: torch.device,
    manifest: CheckpointManifestV1,
    artifact_contract: OnnxArtifactContract,
) -> dict | None:
    """Restore a hash-verified learner blob after checking its embedded contract."""
    path = Path(checkpoint_path)
    # The repository verifies this before calling torch.load; repeat at the trust
    # boundary so direct callers cannot accidentally bypass content identity.
    data = path.read_bytes()
    if len(data) != manifest.learner_state.size_bytes:
        raise ArtifactValidationError("Learner-state blob size does not match manifest")
    if sha256_bytes(data) != manifest.learner_state.sha256:
        raise ArtifactValidationError("Learner-state blob SHA-256 does not match manifest")
    if manifest.profile != artifact_contract.profile:
        raise ArtifactValidationError("Learner-state manifest profile mismatch")

    state = _validate_learner_state_bytes(
        data,
        artifact_contract=artifact_contract,
        step=manifest.step,
        config_sha256=manifest.config_sha256,
        device=device,
    )
    network.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler_state = state["scheduler_state_dict"]
    return scheduler_state


__all__ = [
    "LearnerStateContract",
    "export_onnx_artifact",
    "learner_config_sha256",
    "learner_config_recipe",
    "restore_learner_state",
    "validate_learner_state_artifact",
    "write_learner_state_artifact",
]
