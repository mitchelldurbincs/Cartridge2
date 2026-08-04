"""Content-addressed checkpoint repository contract tests."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from unittest.mock import patch

import onnx
import pytest
import torch
from onnx import TensorProto, helper
from torch.optim import Adam

from trainer.algorithms.alphazero_board_v1 import policy_value_artifact_contract
from trainer.central_config import StorageConfig
from trainer.checkpoint import (
    LearnerStateContract,
    export_onnx_artifact,
    write_learner_state_artifact,
)
from trainer.network import PolicyValueNetwork
from trainer.stats import TrainerStats, prepare_stats_snapshot
from trainer.storage.evaluation import create_evaluation_repository
from trainer.storage.publisher import (
    ArtifactValidationError,
    CheckpointManifestV1,
    FilesystemCheckpointPublisher,
    OnnxArtifactContract,
    OnnxTensorSpec,
    RunHeadV2,
    S3CheckpointPublisher,
    canonical_json_bytes,
    create_checkpoint_publisher,
    sha256_bytes,
    validate_onnx_checkpoint,
)
from trainer.storage.run_commit import RunCommitRepository, RunCommitV1

CONTRACT = policy_value_artifact_contract(
    algorithm_id="alphazero_board_v1",
    env_id="tictactoe",
    env_contract_version=1,
    model_artifact_schema_version=1,
    model_contract="onnx_policy_value_v1",
    obs_size=29,
    num_actions=9,
)
CONFIG_SHA256 = "a" * 64


@pytest.fixture(scope="module")
def staged_blobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("staged-checkpoint")
    network = PolicyValueNetwork(obs_size=29, action_size=9)
    optimizer = Adam(network.parameters(), lr=0.001)
    onnx_path = export_onnx_artifact(
        network,
        root / "model.onnx",
        torch.device("cpu"),
        CONTRACT,
    )
    learner_path = write_learner_state_artifact(
        network,
        optimizer,
        100,
        root / "learner.pt",
        CONTRACT,
        CONFIG_SHA256,
    )
    return onnx_path, learner_path, LearnerStateContract(network, optimizer)


def _stage(publisher, staged_blobs, **overrides):
    onnx_path, learner_path, learner_state_contract = staged_blobs
    values = {
        "step": 100,
        "parent_checkpoint_id": None,
        "config_sha256": CONFIG_SHA256,
    }
    values.update(overrides)
    if values["step"] != 100 or values["config_sha256"] != CONFIG_SHA256:
        staged_state_path = learner_path.with_name(
            f"learner-{values['step']}-{values['config_sha256'][:12]}.pt"
        )
        if not staged_state_path.exists():
            state = torch.load(learner_path, map_location="cpu", weights_only=True)
            state["step"] = values["step"]
            state["config_sha256"] = values["config_sha256"]
            torch.save(state, staged_state_path)
        learner_path = staged_state_path
    return publisher.stage_checkpoint(
        onnx_path,
        learner_path,
        learner_state_contract=learner_state_contract,
        **values,
    )


def _publish(publisher, staged_blobs, **overrides):
    checkpoint = _stage(publisher, staged_blobs, **overrides)
    head = publisher.resolve_run_head()
    stats = TrainerStats(
        step=checkpoint.manifest.step,
        total_steps=checkpoint.manifest.step,
        last_checkpoint=checkpoint.checkpoint_id,
        env_id=checkpoint.manifest.profile.env_id,
    )
    commit = RunCommitV1(
        profile=checkpoint.manifest.profile,
        config_sha256=checkpoint.manifest.config_sha256,
        parent_run_commit_id=head.run_commit_id if head is not None else None,
        checkpoint_id=checkpoint.checkpoint_id,
        stats_snapshot=prepare_stats_snapshot(stats, checkpoint),
        champion=None,
        evaluation_head_id=None,
        orchestration=None,
    )
    RunCommitRepository(publisher, create_evaluation_repository(publisher)).publish(commit)
    publisher.commit_run_head(
        checkpoint_id=checkpoint.checkpoint_id,
        run_commit_id=commit.run_commit_id,
        expected_run_commit_id=head.run_commit_id if head is not None else None,
    )
    return checkpoint


def test_generated_checkpoint_satisfies_strict_onnx_contract(staged_blobs):
    validate_onnx_checkpoint(staged_blobs[0], CONTRACT)


def test_validator_accepts_a_cartridge_declared_q_value_interface(tmp_path):
    contract = OnnxArtifactContract(
        algorithm_id="dqn_v1",
        env_id="counter",
        env_contract_version=2,
        model_artifact_schema_version=1,
        model_contract="onnx_q_values_v1",
        inputs=(OnnxTensorSpec("observation", "float32", ("batch_size", 2)),),
        outputs=(OnnxTensorSpec("q_values", "float32", ("batch_size", 2)),),
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["observation"], ["q_values"])],
        "dqn_q_values",
        [helper.make_tensor_value_info("observation", TensorProto.FLOAT, ["batch_size", 2])],
        [helper.make_tensor_value_info("q_values", TensorProto.FLOAT, ["batch_size", 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    metadata = {
        "cartridge.schema_version": "1",
        "cartridge.algorithm_id": "dqn_v1",
        "cartridge.model_contract": "onnx_q_values_v1",
        "cartridge.env_id": "counter",
        "cartridge.env_contract_version": "2",
    }
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    path = tmp_path / "q-values.onnx"
    onnx.save_model(model, path)

    validate_onnx_checkpoint(path, contract)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"algorithm_id": "../escape"}, "runtime profile"),
        ({"env_id": "Connect4"}, "runtime profile"),
        ({"model_contract": "  "}, "model_contract"),
        ({"env_contract_version": True}, "positive integer"),
        ({"model_artifact_schema_version": "1"}, "positive integer"),
        ({"inputs": ()}, "non-empty tuple"),
        ({"outputs": (CONTRACT.outputs[0], CONTRACT.outputs[0])}, "unique"),
    ],
)
def test_artifact_contract_rejects_unsafe_identity(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(CONTRACT, **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": ""}, "non-empty"),
        ({"dtype": "float64"}, "not supported"),
        ({"shape": ()}, "non-empty tuple"),
        ({"shape": ("batch-size", 9)}, "valid non-empty symbol"),
        ({"shape": ("batch_size", 0)}, "must be positive"),
    ],
)
def test_onnx_tensor_spec_rejects_invalid_boundaries(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(CONTRACT.inputs[0], **changes)


def test_filesystem_publication_has_exact_content_addressed_layout(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    checkpoint = _publish(publisher, staged_blobs)
    head = publisher.resolve_run_head()
    assert head is not None

    expected = {
        f"blobs/sha256/{checkpoint.manifest.onnx.sha256}.onnx",
        f"blobs/sha256/{checkpoint.manifest.learner_state.sha256}.pt",
        f"manifests/sha256/{checkpoint.checkpoint_id}.json",
        f"run-commits/sha256/{head.run_commit_id}.json",
        "channels/current.json",
    }
    assert {
        str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()
    } == expected
    assert checkpoint.onnx_path == (
        tmp_path / "blobs" / "sha256" / f"{checkpoint.manifest.onnx.sha256}.onnx"
    )
    assert not list(tmp_path.rglob("latest.*"))
    assert not list(tmp_path.rglob("model_step_*"))


def test_manifest_and_channels_are_exact_canonical_json(tmp_path, staged_blobs):
    checkpoint = _publish(
        FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT),
        staged_blobs,
    )
    manifest_path = tmp_path / "manifests" / "sha256" / f"{checkpoint.checkpoint_id}.json"
    data = manifest_path.read_bytes()
    decoded = json.loads(data)

    assert data == canonical_json_bytes(decoded)
    assert not data.endswith(b"\n")
    assert set(decoded) == {
        "schema_version",
        "profile",
        "step",
        "parent_checkpoint_id",
        "config_sha256",
        "onnx",
        "learner_state",
    }
    assert "checkpoint_id" not in decoded
    assert sha256_bytes(data) == checkpoint.checkpoint_id
    assert CheckpointManifestV1.from_bytes(data) == checkpoint.manifest
    pointer_bytes = (tmp_path / "channels" / "current.json").read_bytes()
    head = RunHeadV2.from_bytes(pointer_bytes)
    assert head.checkpoint_id == checkpoint.checkpoint_id
    assert json.loads(pointer_bytes) == {
        "schema_version": 2,
        "checkpoint_id": checkpoint.checkpoint_id,
        "run_commit_id": head.run_commit_id,
    }


def test_republishing_identical_checkpoint_deduplicates_immutables(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    first = _stage(publisher, staged_blobs)
    first_manifest_stat = (tmp_path / "manifests" / "sha256" / f"{first.checkpoint_id}.json").stat()
    second = _stage(publisher, staged_blobs)

    assert second == first
    assert (
        tmp_path / "manifests" / "sha256" / f"{first.checkpoint_id}.json"
    ).stat().st_ino == first_manifest_stat.st_ino
    assert len(list((tmp_path / "manifests" / "sha256").iterdir())) == 1


def test_staging_publishes_the_exact_private_snapshot(tmp_path, staged_blobs):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    onnx_source = source_root / "model.onnx"
    learner_source = source_root / "learner.pt"
    original_onnx = staged_blobs[0].read_bytes()
    original_learner = staged_blobs[1].read_bytes()
    onnx_source.write_bytes(original_onnx)
    learner_source.write_bytes(original_learner)

    def mutate_caller_owned_sources(snapshot_path, contract):
        onnx_source.write_bytes(b"mutated after snapshot")
        learner_source.write_bytes(b"mutated after snapshot")
        validate_onnx_checkpoint(snapshot_path, contract)

    publisher = FilesystemCheckpointPublisher(model_root=tmp_path / "models", contract=CONTRACT)
    with patch(
        "trainer.storage.checkpoint_validation.validate_onnx_checkpoint",
        side_effect=mutate_caller_owned_sources,
    ):
        checkpoint = publisher.stage_checkpoint(
            onnx_source,
            learner_source,
            step=100,
            parent_checkpoint_id=None,
            config_sha256=CONFIG_SHA256,
            learner_state_contract=staged_blobs[2],
        )

    assert checkpoint.onnx_path.read_bytes() == original_onnx
    assert checkpoint.learner_state_path.read_bytes() == original_learner
    assert onnx_source.read_bytes() != original_onnx
    assert learner_source.read_bytes() != original_learner


def test_parent_checkpoint_id_changes_manifest_identity(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    first = _publish(publisher, staged_blobs)
    second = _publish(
        publisher,
        staged_blobs,
        step=101,
        parent_checkpoint_id=first.checkpoint_id,
    )

    assert second.checkpoint_id != first.checkpoint_id
    assert second.manifest.parent_checkpoint_id == first.checkpoint_id


def test_publication_requires_exact_head_parent_and_increasing_step(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    root = _publish(publisher, staged_blobs, step=10)

    with pytest.raises(ArtifactValidationError, match="strictly newer direct child"):
        _publish(publisher, staged_blobs, step=11, parent_checkpoint_id=None)
    with pytest.raises(ArtifactValidationError, match="strictly increase"):
        _publish(
            publisher,
            staged_blobs,
            step=10,
            parent_checkpoint_id=root.checkpoint_id,
        )

    child = _publish(
        publisher,
        staged_blobs,
        step=11,
        parent_checkpoint_id=root.checkpoint_id,
    )
    with pytest.raises(ArtifactValidationError, match="strictly newer direct child"):
        _publish(
            publisher,
            staged_blobs,
            step=12,
            parent_checkpoint_id=root.checkpoint_id,
        )

    assert publisher.resolve_head() == child
    # Failed CAS/transition attempts may leave valid immutable checkpoints;
    # authority remains exclusively in channels/current.json.
    assert len(list((tmp_path / "manifests" / "sha256").iterdir())) == 4


def test_publication_requires_root_when_no_head(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)

    with pytest.raises(ArtifactValidationError, match="manifest does not exist"):
        _publish(
            publisher,
            staged_blobs,
            step=1,
            parent_checkpoint_id="f" * 64,
        )

    assert not list((tmp_path / "manifests" / "sha256").glob("*.json"))
    assert not (tmp_path / "channels").exists()


def test_publication_requires_parent_configuration_identity(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    root = _publish(publisher, staged_blobs, step=10)

    with pytest.raises(ArtifactValidationError, match="config_sha256 mismatch"):
        _publish(
            publisher,
            staged_blobs,
            step=11,
            parent_checkpoint_id=root.checkpoint_id,
            config_sha256="b" * 64,
        )

    assert publisher.resolve_head() == root
    assert len(list((tmp_path / "manifests" / "sha256").iterdir())) == 1


def test_invalid_learner_envelope_creates_no_repository_objects(tmp_path, staged_blobs):
    model_root = tmp_path / "models"
    invalid = tmp_path / "invalid-learner.pt"
    invalid.write_bytes(b"not a safe learner-state artifact")
    publisher = FilesystemCheckpointPublisher(model_root=model_root, contract=CONTRACT)

    with pytest.raises(ArtifactValidationError, match="safely decoded"):
        publisher.stage_checkpoint(
            staged_blobs[0],
            invalid,
            step=100,
            parent_checkpoint_id=None,
            config_sha256=CONFIG_SHA256,
            learner_state_contract=staged_blobs[2],
        )

    assert not model_root.exists()


def test_learner_envelope_must_match_publication_step(tmp_path, staged_blobs):
    model_root = tmp_path / "models"
    publisher = FilesystemCheckpointPublisher(model_root=model_root, contract=CONTRACT)

    with pytest.raises(ArtifactValidationError, match="step mismatch"):
        publisher.stage_checkpoint(
            staged_blobs[0],
            staged_blobs[1],
            step=101,
            parent_checkpoint_id=None,
            config_sha256=CONFIG_SHA256,
            learner_state_contract=staged_blobs[2],
        )

    assert not model_root.exists()


@pytest.mark.parametrize("mutation", ["model", "optimizer", "scheduler"])
def test_live_learner_contract_rejects_restore_incompatible_state_before_staging(
    tmp_path, staged_blobs, mutation
):
    model_root = tmp_path / "models"
    invalid = tmp_path / f"invalid-{mutation}.pt"
    state = torch.load(staged_blobs[1], map_location="cpu", weights_only=True)
    if mutation == "model":
        state["model_state_dict"].pop(next(iter(state["model_state_dict"])))
    elif mutation == "optimizer":
        state["optimizer_state_dict"]["param_groups"][0]["params"].pop()
    else:
        state["scheduler_state_dict"] = {"unexpected": 1}
    torch.save(state, invalid)
    publisher = FilesystemCheckpointPublisher(model_root=model_root, contract=CONTRACT)

    with pytest.raises(ArtifactValidationError, match="Learner (model|optimizer|scheduler)"):
        publisher.stage_checkpoint(
            staged_blobs[0],
            invalid,
            step=100,
            parent_checkpoint_id=None,
            config_sha256=CONFIG_SHA256,
            learner_state_contract=staged_blobs[2],
        )

    assert not model_root.exists()


def test_live_learner_contract_snapshots_every_tensor_on_cpu(staged_blobs):
    contract = staged_blobs[2]

    def tensor_devices(value):
        if isinstance(value, torch.Tensor):
            return {value.device.type}
        if isinstance(value, dict):
            return set().union(*(tensor_devices(item) for item in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(tensor_devices(item) for item in value))
        return set()

    assert tensor_devices(contract._model_state) == {"cpu"}
    assert tensor_devices(contract._optimizer_state) <= {"cpu"}


def test_resolve_head_and_list_checkpoints_are_strict_and_step_sorted(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    earlier = _publish(publisher, staged_blobs, step=10)
    later = _publish(
        publisher,
        staged_blobs,
        step=20,
        parent_checkpoint_id=earlier.checkpoint_id,
    )

    assert publisher.resolve_head() == later
    assert publisher.resolve_checkpoint(earlier.checkpoint_id) == earlier
    assert publisher.list_checkpoints() == [earlier, later]


def test_list_checkpoints_rejects_unexpected_or_corrupt_manifest(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    _publish(publisher, staged_blobs)
    unexpected = tmp_path / "manifests" / "sha256" / "README"
    unexpected.write_text("not an artifact")

    with pytest.raises(ArtifactValidationError, match="Unexpected object"):
        publisher.list_checkpoints()


def test_immutable_object_with_different_bytes_is_never_overwritten(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    onnx_data = staged_blobs[0].read_bytes()
    digest = sha256_bytes(onnx_data)
    target = tmp_path / "blobs" / "sha256" / f"{digest}.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"occupied by corrupt bytes")

    with pytest.raises(ArtifactValidationError, match="different bytes"):
        _publish(publisher, staged_blobs)
    assert target.read_bytes() == b"occupied by corrupt bytes"
    assert not (tmp_path / "channels").exists()


def test_failure_before_manifest_does_not_advance_channels(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    from trainer.storage import checkpoint_filesystem as publisher_module

    original = publisher_module.create_or_verify
    calls = 0

    def fail_manifest(path, data):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("manifest storage failed")
        original(path, data)

    with patch.object(publisher_module, "create_or_verify", side_effect=fail_manifest):
        with pytest.raises(RuntimeError, match="manifest storage failed"):
            _publish(publisher, staged_blobs)

    assert not (tmp_path / "channels").exists()


def test_head_rejects_corrupt_blob(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    checkpoint = _publish(publisher, staged_blobs)
    checkpoint.learner_state_path.write_bytes(b"corrupt")

    with pytest.raises(ArtifactValidationError, match="size mismatch"):
        publisher.resolve_head(expected_config_sha256=CONFIG_SHA256)


def test_head_rejects_profile_and_config_mismatch(tmp_path, staged_blobs):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    _publish(publisher, staged_blobs)

    with pytest.raises(ArtifactValidationError, match="config_sha256 mismatch"):
        publisher.resolve_head(expected_config_sha256="b" * 64)
    wrong_profile = FilesystemCheckpointPublisher(
        model_root=tmp_path,
        contract=replace(CONTRACT, env_id="connect4"),
    )
    with pytest.raises(ArtifactValidationError, match="profile mismatch"):
        wrong_profile.resolve_head(expected_config_sha256=CONFIG_SHA256)


def test_strict_json_rejects_extra_fields_and_noncanonical_encoding(tmp_path, staged_blobs):
    checkpoint = _publish(
        FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT),
        staged_blobs,
    )
    value = checkpoint.manifest.to_dict()
    value["checkpoint_id"] = checkpoint.checkpoint_id
    with pytest.raises(ArtifactValidationError, match="fields must be exact"):
        CheckpointManifestV1.from_bytes(canonical_json_bytes(value))
    with pytest.raises(ArtifactValidationError, match="not canonical JSON"):
        CheckpointManifestV1.from_bytes(
            json.dumps(checkpoint.manifest.to_dict(), indent=2).encode()
        )


class S3Error(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.puts = []
        self.fail_key = None

    def get_object(self, *, Bucket, Key):
        del Bucket
        if Key not in self.objects:
            raise S3Error("NoSuchKey")
        data = self.objects[Key]
        return {
            "Body": io.BytesIO(data),
            "ETag": f'"{sha256_bytes(data)}"',
        }

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        if key == self.fail_key:
            raise RuntimeError("S3 unavailable")
        data = kwargs["Body"]
        if hasattr(data, "read"):
            data = data.read()
        if kwargs.get("IfNoneMatch") == "*" and key in self.objects:
            raise S3Error("PreconditionFailed")
        if "IfMatch" in kwargs:
            existing = self.objects.get(key)
            etag = f'"{sha256_bytes(existing)}"' if existing is not None else None
            if kwargs["IfMatch"] != etag:
                raise S3Error("PreconditionFailed")
        self.objects[key] = bytes(data)
        self.puts.append(kwargs)

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        del Bucket
        if ContinuationToken is not None:
            raise AssertionError("fake listing is not paginated")
        return {
            "Contents": [{"Key": key} for key in sorted(self.objects) if key.startswith(Prefix)],
            "IsTruncated": False,
        }


class ConflictOnceS3(FakeS3):
    def __init__(self):
        super().__init__()
        self.conflict_key = None
        self.conflict_code = None
        self.store_before_error = False

    def put_object(self, **kwargs):
        if kwargs["Key"] == self.conflict_key and self.conflict_code is not None:
            code = self.conflict_code
            self.conflict_code = None
            if self.store_before_error:
                self.objects[kwargs["Key"]] = bytes(kwargs["Body"])
            raise S3Error(code)
        super().put_object(**kwargs)


def test_s3_publishes_verified_immutables_before_authoritative_head(tmp_path, staged_blobs):
    client = FakeS3()
    publisher = S3CheckpointPublisher(
        model_root=tmp_path,
        bucket="models-bucket",
        contract=CONTRACT,
        client=client,
    )
    checkpoint = _publish(publisher, staged_blobs)

    put_keys = [item["Key"] for item in client.puts]
    prefix = "profiles/alphazero_board_v1/tictactoe/v1/models"
    assert put_keys[-1:] == [f"{prefix}/channels/current.json"]
    assert put_keys[:3] == [
        f"{prefix}/blobs/sha256/{checkpoint.manifest.onnx.sha256}.onnx",
        f"{prefix}/blobs/sha256/{checkpoint.manifest.learner_state.sha256}.pt",
        f"{prefix}/manifests/sha256/{checkpoint.checkpoint_id}.json",
    ]
    assert all("IfNoneMatch" in item for item in client.puts)
    assert not any("/_coordination/" in key for key in client.objects)


def test_s3_deduplicates_and_can_resolve_into_a_fresh_local_cache(tmp_path, staged_blobs):
    client = FakeS3()
    source = S3CheckpointPublisher(
        model_root=tmp_path / "source",
        bucket="models-bucket",
        contract=CONTRACT,
        client=client,
    )
    first = _publish(source, staged_blobs)
    assert _stage(source, staged_blobs) == first
    immutable_writes = [
        item for item in client.puts if "/blobs/" in item["Key"] or "/manifests/" in item["Key"]
    ]
    assert len([item for item in immutable_writes if "IfNoneMatch" in item]) == 3

    receiver = S3CheckpointPublisher(
        model_root=tmp_path / "receiver",
        bucket="models-bucket",
        contract=CONTRACT,
        client=client,
    )
    resolved = receiver.resolve_head(expected_config_sha256=CONFIG_SHA256)

    assert resolved is not None
    assert resolved.checkpoint_id == first.checkpoint_id
    assert resolved.onnx_path.is_file()
    assert resolved.learner_state_path.is_file()
    assert receiver.resolve_head() == resolved
    assert receiver.list_checkpoints() == [resolved]


@pytest.mark.parametrize("code", ["409", "ConditionalRequestConflict"])
def test_s3_immutable_put_retries_transient_conditional_conflict(tmp_path, code):
    client = ConflictOnceS3()
    publisher = S3CheckpointPublisher(
        model_root=tmp_path,
        bucket="models-bucket",
        contract=CONTRACT,
        client=client,
    )
    key = publisher._key(f"blobs/sha256/{'f' * 64}.onnx")
    client.conflict_key = key
    client.conflict_code = code

    publisher._put_immutable(key, b"immutable", "application/octet-stream")

    assert client.objects[key] == b"immutable"


def test_s3_ambiguous_immutable_put_is_reconciled_by_strong_read(tmp_path):
    client = ConflictOnceS3()
    publisher = S3CheckpointPublisher(
        model_root=tmp_path,
        bucket="models-bucket",
        contract=CONTRACT,
        client=client,
    )
    key = publisher._key(f"blobs/sha256/{'e' * 64}.onnx")
    client.conflict_key = key
    client.conflict_code = "RequestTimeout"
    client.store_before_error = True

    publisher._put_immutable(key, b"immutable", "application/octet-stream")

    assert client.objects[key] == b"immutable"


def test_s3_head_publication_uses_conditional_compare_and_set(tmp_path, staged_blobs):
    client = ConflictOnceS3()
    publisher = S3CheckpointPublisher(
        model_root=tmp_path,
        bucket="models-bucket",
        contract=CONTRACT,
        client=client,
    )
    client.conflict_key = publisher._head_key
    client.conflict_code = "PreconditionFailed"

    with pytest.raises(ArtifactValidationError, match="head changed"):
        _publish(publisher, staged_blobs)

    assert publisher._head_key not in client.objects


def test_s3_ambiguous_head_write_reconciles_exact_committed_bytes(tmp_path, staged_blobs):
    client = ConflictOnceS3()
    publisher = S3CheckpointPublisher(
        model_root=tmp_path,
        bucket="models-bucket",
        contract=CONTRACT,
        client=client,
    )
    client.conflict_key = publisher._head_key
    client.conflict_code = "RequestTimeout"
    client.store_before_error = True

    checkpoint = _publish(publisher, staged_blobs)

    assert publisher.resolve_head() == checkpoint
    assert not any("/_coordination/" in key for key in client.objects)


def test_s3_head_confirmation_accepts_a_valid_descendant(tmp_path, staged_blobs):
    client = FakeS3()
    publisher = S3CheckpointPublisher(
        model_root=tmp_path,
        bucket="models-bucket",
        contract=CONTRACT,
        client=client,
    )
    root = _publish(publisher, staged_blobs, step=10)
    root_head_bytes = client.objects[publisher._head_key]
    child = _publish(
        publisher,
        staged_blobs,
        step=11,
        parent_checkpoint_id=root.checkpoint_id,
    )
    child_head_bytes = client.objects[publisher._head_key]

    # Model an ambiguous root write whose confirming GET observes that another
    # writer has already advanced root -> child. The desired root was selected
    # in that valid immutable lineage, so reporting failure would be false.
    observed = publisher._compare_and_set_head_version(expected=None, target=root_head_bytes)

    assert client.objects[publisher._head_key] == child_head_bytes
    assert observed == RunHeadV2.from_bytes(child_head_bytes)
    assert publisher.resolve_head() == child


def test_s3_head_confirmation_rejects_a_mismatched_checkpoint_binding(tmp_path, staged_blobs):
    client = FakeS3()
    publisher = S3CheckpointPublisher(
        model_root=tmp_path,
        bucket="models-bucket",
        contract=CONTRACT,
        client=client,
    )
    root = _publish(publisher, staged_blobs, step=10)
    root_head_bytes = client.objects[publisher._head_key]
    child = _publish(
        publisher,
        staged_blobs,
        step=11,
        parent_checkpoint_id=root.checkpoint_id,
    )
    child_head = publisher.resolve_run_head()
    assert child_head is not None
    malformed = RunHeadV2(
        checkpoint_id=root.checkpoint_id,
        run_commit_id=child_head.run_commit_id,
    )

    with pytest.raises(ArtifactValidationError, match="does not match"):
        publisher._head_selects_or_descends_from(malformed.to_bytes(), root_head_bytes)

    assert child.checkpoint_id != root.checkpoint_id


def test_s3_failure_before_manifest_does_not_publish_pointers(tmp_path, staged_blobs):
    client = FakeS3()
    publisher = S3CheckpointPublisher(
        model_root=tmp_path,
        bucket="models-bucket",
        contract=CONTRACT,
        client=client,
    )
    onnx_data = staged_blobs[0].read_bytes()
    learner_data = staged_blobs[1].read_bytes()
    # Derive the ID without mutating the destination.
    shadow = FilesystemCheckpointPublisher(model_root=tmp_path / "shadow", contract=CONTRACT)
    shadow_ref = _publish(shadow, staged_blobs)
    del onnx_data, learner_data
    client.fail_key = publisher._key(f"manifests/sha256/{shadow_ref.checkpoint_id}.json")

    with pytest.raises(RuntimeError, match="S3 unavailable"):
        _publish(publisher, staged_blobs)
    assert not any("/channels/" in key for key in client.objects)


def test_factory_uses_canonical_storage_config(tmp_path):
    client = FakeS3()
    publisher = create_checkpoint_publisher(
        CONTRACT,
        tmp_path,
        StorageConfig(
            model_backend="s3",
            s3_bucket="canonical-bucket",
            s3_endpoint="http://minio:9000",
        ),
        s3_client=client,
    )
    assert isinstance(publisher, S3CheckpointPublisher)
    assert publisher.bucket == "canonical-bucket"
    assert publisher._client is client


def test_factory_rejects_incomplete_or_unknown_config(tmp_path):
    with pytest.raises(ValueError, match="CARTRIDGE_STORAGE_S3_BUCKET"):
        create_checkpoint_publisher(CONTRACT, tmp_path, StorageConfig(model_backend="s3"))
    with pytest.raises(ValueError, match="Unknown checkpoint publication backend"):
        create_checkpoint_publisher(CONTRACT, tmp_path, StorageConfig(model_backend="legacy"))


def test_canonical_json_is_raw_utf8_cross_language_golden():
    """Python and Rust must produce byte-identical canonical JSON.

    serde_json (used by model-watcher/actor consumers) parses then
    re-serializes canonical objects and rejects any byte difference, and it
    never emits ``\\uXXXX`` escapes for non-ASCII text. These golden bytes are
    asserted verbatim on the Rust side in
    engine/model-watcher/src/artifact.rs.
    """
    value = {"name": "café", "piece": "♟", "z": 1}
    golden = '{"name":"café","piece":"♟","z":1}'.encode("utf-8")

    encoded = canonical_json_bytes(value)
    assert encoded == golden
    assert b"\\u" not in encoded
    assert (
        sha256_bytes(encoded) == "2e4360a215d64b8654fc51e28743d8761b5816bef04a30ceefa3314a1f151189"
    )

    # ASCII identity fields are unaffected by the raw-UTF-8 rule.
    assert canonical_json_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
