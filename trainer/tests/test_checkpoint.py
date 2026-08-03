"""Tests for staging and restoring immutable checkpoint blobs."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim import Adam

from trainer.checkpoint import (
    _validate_onnx_runtime_equivalence,
    export_onnx_artifact,
    learner_config_recipe,
    learner_config_sha256,
    restore_learner_state,
    write_learner_state_artifact,
)
from trainer.config import AlphaZeroLearnerConfig
from trainer.lr_scheduler import LRConfig, WarmupCosineScheduler
from trainer.network import PolicyValueNetwork
from trainer.storage.publisher import (
    ArtifactValidationError,
    BlobDescriptorV1,
    CheckpointManifestV1,
    OnnxArtifactContract,
)

CONTRACT = OnnxArtifactContract(
    algorithm_id="alphazero_board_v1",
    env_id="tictactoe",
    env_contract_version=1,
    model_artifact_schema_version=1,
    model_contract="onnx_policy_value_v1",
    obs_size=29,
    num_actions=9,
)


def _manifest(onnx_path, learner_path, config_sha256, *, step=100):
    onnx_data = onnx_path.read_bytes()
    learner_data = learner_path.read_bytes()
    return CheckpointManifestV1(
        profile=CONTRACT.profile,
        step=step,
        parent_checkpoint_id=None,
        config_sha256=config_sha256,
        onnx=BlobDescriptorV1.from_bytes(onnx_data),
        learner_state=BlobDescriptorV1.from_bytes(learner_data),
    )


def test_export_onnx_artifact_writes_only_requested_staging_path(tmp_path, monkeypatch):
    output = tmp_path / "staging" / "model.onnx"
    network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
    network.train()
    network.fc1.eval()
    real_export = torch.onnx.export
    export_call = {}

    def recording_export(*args, **kwargs):
        export_call["args"] = args
        export_call["kwargs"] = kwargs
        return real_export(*args, **kwargs)

    monkeypatch.setattr(torch.onnx, "export", recording_export)

    result = export_onnx_artifact(
        network,
        29,
        output,
        torch.device("cpu"),
        CONTRACT,
    )

    assert result == output
    assert output.is_file()
    assert sorted(path.name for path in output.parent.iterdir()) == ["model.onnx"]
    assert network.training
    assert not network.fc1.training
    export_args = export_call["args"]
    export_kwargs = export_call["kwargs"]
    assert tuple(export_args[1][0].shape) == (2, 29)
    assert export_kwargs["dynamo"] is True
    assert export_kwargs["external_data"] is False
    assert export_kwargs["verify"] is False
    assert export_kwargs["optimize"] is True
    assert export_kwargs["verbose"] is False
    assert "dynamic_axes" not in export_kwargs
    assert export_kwargs["dynamic_shapes"][0][0].__name__ == "batch_size"


def test_export_has_exact_runtime_metadata(tmp_path):
    onnx = pytest.importorskip("onnx")
    path = export_onnx_artifact(
        PolicyValueNetwork(obs_size=29, action_size=9),
        29,
        tmp_path / "model.onnx",
        torch.device("cpu"),
        CONTRACT,
    )

    model = onnx.load(path)
    assert {item.key: item.value for item in model.metadata_props} == {
        "cartridge.schema_version": "1",
        "cartridge.algorithm_id": "alphazero_board_v1",
        "cartridge.model_contract": "onnx_policy_value_v1",
        "cartridge.env_id": "tictactoe",
        "cartridge.env_contract_version": "1",
    }
    assert [item.name for item in model.graph.input] == ["observation"]
    assert [item.name for item in model.graph.output] == ["policy_logits", "value"]
    tensors = [*model.graph.input, *model.graph.output]
    assert [
        [dim.dim_param or dim.dim_value for dim in item.type.tensor_type.shape.dim]
        for item in tensors
    ] == [["batch_size", 29], ["batch_size", 9], ["batch_size", 1]]
    assert not any(initializer.external_data for initializer in model.graph.initializer)
    assert not list(tmp_path.glob("*.data"))


def test_export_failure_preserves_existing_artifact_and_module_modes(
    tmp_path, monkeypatch
):
    output = tmp_path / "model.onnx"
    output.write_bytes(b"previous artifact")
    network = PolicyValueNetwork(obs_size=29, action_size=9)
    network.train()
    network.fc1.eval()

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("dynamo export failed")

    monkeypatch.setattr(torch.onnx, "export", fail_export)

    with pytest.raises(RuntimeError, match="dynamo export failed"):
        export_onnx_artifact(
            network,
            29,
            output,
            torch.device("cpu"),
            CONTRACT,
        )

    assert output.read_bytes() == b"previous artifact"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model.onnx"]
    assert network.training
    assert not network.fc1.training


def test_runtime_equivalence_rejects_shape_and_nonfinite_outputs(tmp_path, monkeypatch):
    network = PolicyValueNetwork(obs_size=29, action_size=9)
    network.eval()
    cases = [
        (
            lambda batch_size: [
                np.zeros((batch_size + 1, 9), dtype=np.float32),
                np.zeros((batch_size, 1), dtype=np.float32),
            ],
            "shape mismatch",
        ),
        (
            lambda batch_size: [
                np.full((batch_size, 9), np.nan, dtype=np.float32),
                np.zeros((batch_size, 1), dtype=np.float32),
            ],
            "non-finite",
        ),
    ]

    for outputs, expected_message in cases:

        class FakeSession:
            def __init__(self, *_args, **_kwargs):
                pass

            def run(self, _output_names, feeds):
                return outputs(feeds["observation"].shape[0])

        monkeypatch.setattr("trainer.checkpoint.ort.InferenceSession", FakeSession)
        with pytest.raises(ArtifactValidationError, match=expected_message):
            _validate_onnx_runtime_equivalence(
                network,
                tmp_path / "model.onnx",
                obs_size=29,
                device=torch.device("cpu"),
            )


def test_runtime_equivalence_rejects_meaningful_numeric_corruption(
    tmp_path, monkeypatch
):
    network = PolicyValueNetwork(obs_size=29, action_size=9)
    network.eval()

    class CorruptingSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, _output_names, feeds):
            observation = torch.from_numpy(feeds["observation"])
            with torch.no_grad():
                policy_logits, value = network(observation)
            return [policy_logits.numpy() + 1e-2, value.numpy()]

    monkeypatch.setattr("trainer.checkpoint.ort.InferenceSession", CorruptingSession)

    with pytest.raises(ArtifactValidationError, match="differs from the live network"):
        _validate_onnx_runtime_equivalence(
            network,
            tmp_path / "model.onnx",
            obs_size=29,
            device=torch.device("cpu"),
        )


def test_runtime_validation_failure_is_atomic(tmp_path, monkeypatch):
    output = tmp_path / "model.onnx"
    output.write_bytes(b"previous artifact")
    network = PolicyValueNetwork(obs_size=29, action_size=9)
    network.train()

    def reject_runtime(*_args, **_kwargs):
        raise ArtifactValidationError("runtime equivalence failed")

    monkeypatch.setattr(
        "trainer.checkpoint._validate_onnx_runtime_equivalence", reject_runtime
    )

    with pytest.raises(ArtifactValidationError, match="runtime equivalence failed"):
        export_onnx_artifact(
            network,
            29,
            output,
            torch.device("cpu"),
            CONTRACT,
        )

    assert output.read_bytes() == b"previous artifact"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model.onnx"]
    assert network.training


def test_export_rejects_observation_contract_mismatch(tmp_path):
    with pytest.raises(ValueError, match="obs_size"):
        export_onnx_artifact(
            PolicyValueNetwork(obs_size=29, action_size=9),
            28,
            tmp_path / "model.onnx",
            torch.device("cpu"),
            CONTRACT,
        )


def test_config_hash_is_deterministic_and_excludes_paths_and_callbacks(tmp_path):
    base = AlphaZeroLearnerConfig(
        env_id="tictactoe",
        model_dir=str(tmp_path / "first-models"),
        stats_path=str(tmp_path / "first-stats.json"),
        shutdown_check=lambda: False,
        metrics_hook=lambda *_: None,
    )
    moved = replace(
        base,
        model_dir=str(tmp_path / "second-models"),
        stats_path=str(tmp_path / "second-stats.json"),
        shutdown_check=lambda: True,
        metrics_hook=lambda *_: None,
        start_step=50,
        defer_run_commit=True,
        device="cpu",
        checkpoint_interval=7,
        stats_interval=3,
        log_interval=2,
    )

    assert learner_config_sha256(base) == learner_config_sha256(moved)
    assert learner_config_sha256(base) != learner_config_sha256(
        replace(base, learning_rate=0.25)
    )
    assert learner_config_sha256(base) != learner_config_sha256(
        replace(base, env_id="connect4")
    )


def test_config_recipe_returns_a_mutation_safe_semantic_copy():
    config = AlphaZeroLearnerConfig(env_id="tictactoe")
    expected_hash = learner_config_sha256(config)
    recipe = learner_config_recipe(config)
    architecture = recipe["model_architecture"]
    assert isinstance(architecture, dict)
    architecture["action_count"] = 999

    fresh = learner_config_recipe(config)

    assert fresh["model_architecture"]["action_count"] == 9
    assert learner_config_sha256(config) == expected_hash


def test_config_hash_canonicalizes_signed_zero():
    positive = AlphaZeroLearnerConfig(
        weight_decay=0.0,
        grad_clip_norm=0.0,
        lr_min_ratio=0.0,
    )
    negative = AlphaZeroLearnerConfig(
        weight_decay=-0.0,
        grad_clip_norm=-0.0,
        lr_min_ratio=-0.0,
    )

    assert learner_config_sha256(negative) == learner_config_sha256(positive)


def test_config_rejects_nan_before_hashing():
    with pytest.raises(ValueError, match="learning_rate"):
        AlphaZeroLearnerConfig(learning_rate=float("nan"))


def test_learner_state_round_trip_restores_model_optimizer_and_scheduler(tmp_path):
    config_hash = "a" * 64
    network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
    optimizer = Adam(network.parameters(), lr=0.001)
    scheduler = WarmupCosineScheduler(
        optimizer, LRConfig(target_lr=0.001, total_steps=100)
    )
    with torch.no_grad():
        network.policy_fc.weight.fill_(0.5)
    loss = network(torch.randn(1, 29))[1].sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    onnx_path = export_onnx_artifact(
        network,
        29,
        tmp_path / "model.onnx",
        torch.device("cpu"),
        CONTRACT,
    )
    learner_path = write_learner_state_artifact(
        network,
        optimizer,
        100,
        tmp_path / "learner.pt",
        CONTRACT,
        config_hash,
        scheduler,
    )
    manifest = _manifest(onnx_path, learner_path, config_hash)
    restored_network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
    restored_optimizer = Adam(restored_network.parameters(), lr=0.001)

    scheduler_state = restore_learner_state(
        restored_network,
        restored_optimizer,
        learner_path,
        torch.device("cpu"),
        manifest,
        CONTRACT,
    )

    assert scheduler_state == scheduler.state_dict()
    assert torch.equal(network.policy_fc.weight, restored_network.policy_fc.weight)
    assert restored_optimizer.state_dict()["state"]


def test_restore_verifies_hash_before_torch_load(tmp_path, monkeypatch):
    config_hash = "b" * 64
    network = PolicyValueNetwork(obs_size=29, action_size=9)
    optimizer = Adam(network.parameters(), lr=0.001)
    onnx_path = export_onnx_artifact(
        network,
        29,
        tmp_path / "model.onnx",
        torch.device("cpu"),
        CONTRACT,
    )
    learner_path = write_learner_state_artifact(
        network,
        optimizer,
        1,
        tmp_path / "learner.pt",
        CONTRACT,
        config_hash,
    )
    manifest = _manifest(onnx_path, learner_path, config_hash, step=1)
    learner_path.write_bytes(b"corrupt")
    called = False

    def forbidden_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="size does not match"):
        restore_learner_state(
            network,
            optimizer,
            learner_path,
            torch.device("cpu"),
            manifest,
            CONTRACT,
        )
    assert not called


def test_restore_decodes_the_exact_verified_bytes(tmp_path, monkeypatch):
    config_hash = "d" * 64
    network = PolicyValueNetwork(obs_size=29, action_size=9)
    optimizer = Adam(network.parameters(), lr=0.001)
    with torch.no_grad():
        network.policy_fc.weight.fill_(0.25)
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"manifest-only ONNX fixture")
    learner_path = write_learner_state_artifact(
        network,
        optimizer,
        1,
        tmp_path / "learner.pt",
        CONTRACT,
        config_hash,
    )
    manifest = _manifest(onnx_path, learner_path, config_hash, step=1)
    restored_network = PolicyValueNetwork(obs_size=29, action_size=9)
    restored_optimizer = Adam(restored_network.parameters(), lr=0.001)
    original_read_bytes = Path.read_bytes
    mutated = False

    def read_then_mutate(path):
        nonlocal mutated
        data = original_read_bytes(path)
        if path == learner_path and not mutated:
            learner_path.write_bytes(b"mutated after verified read")
            mutated = True
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_mutate)

    restore_learner_state(
        restored_network,
        restored_optimizer,
        learner_path,
        torch.device("cpu"),
        manifest,
        CONTRACT,
    )

    assert mutated
    assert learner_path.read_bytes() == b"mutated after verified read"
    assert torch.equal(network.policy_fc.weight, restored_network.policy_fc.weight)


def test_restore_rejects_manifest_profile_mismatch(tmp_path):
    config_hash = "c" * 64
    network = PolicyValueNetwork(obs_size=29, action_size=9)
    optimizer = Adam(network.parameters(), lr=0.001)
    onnx_path = export_onnx_artifact(
        network,
        29,
        tmp_path / "model.onnx",
        torch.device("cpu"),
        CONTRACT,
    )
    learner_path = write_learner_state_artifact(
        network,
        optimizer,
        1,
        tmp_path / "learner.pt",
        CONTRACT,
        config_hash,
    )
    manifest = _manifest(onnx_path, learner_path, config_hash, step=1)

    with pytest.raises(ValueError, match="profile mismatch"):
        restore_learner_state(
            network,
            optimizer,
            learner_path,
            torch.device("cpu"),
            manifest,
            replace(CONTRACT, env_id="connect4"),
        )
