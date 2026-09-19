"""Byte storage contracts shared by local and remote run authority objects."""

from dataclasses import replace
from hashlib import sha256

import pytest

from trainer.algorithms.alphazero_board_v1 import policy_value_artifact_contract
from trainer.storage.publisher import (
    ArtifactValidationError,
    FilesystemCheckpointPublisher,
    S3CheckpointPublisher,
)

from .fake_s3 import FakeS3, S3Error

CONTRACT = policy_value_artifact_contract(
    algorithm_id="alphazero_board_v1",
    env_id="tictactoe",
    env_contract_version=1,
    model_artifact_schema_version=1,
    model_contract="onnx_policy_value_v1",
    obs_size=29,
    num_actions=9,
)


@pytest.fixture(params=["filesystem", "s3"])
def publisher(request, tmp_path):
    kwargs = {"model_root": tmp_path / "models", "contract": CONTRACT}
    if request.param == "s3":
        return S3CheckpointPublisher(**kwargs, bucket="test-bucket", client=FakeS3())
    return FilesystemCheckpointPublisher(**kwargs)


def stored_objects(publisher):
    if isinstance(publisher, S3CheckpointPublisher):
        prefix = "profiles/alphazero_board_v1/tictactoe/v1/models/"
        return {key.removeprefix(prefix): data for key, data in publisher._client.objects.items()}
    return {
        path.relative_to(publisher.model_root).as_posix(): path.read_bytes()
        for path in publisher.model_root.rglob("*")
        if path.is_file()
    }


def write_raw(publisher, relative, data):
    """Inject corrupt authority without using the validation being tested."""
    if isinstance(publisher, S3CheckpointPublisher):
        prefix = "profiles/alphazero_board_v1/tictactoe/v1/models/"
        publisher._client.objects[prefix + relative] = data
    else:
        path = publisher.model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


@pytest.mark.parametrize("data", ['{"label":"寻迹"}'.encode(), b"null"])
def test_run_commit_preserves_exact_bytes_and_deduplicates(publisher, data):
    # Byte storage accepts canonical JSON; typed schema checks belong to repositories.
    commit_id = sha256(data).hexdigest()
    publisher.publish_run_commit_bytes(commit_id, data)
    publisher.publish_run_commit_bytes(commit_id, data)

    assert publisher.read_run_commit_bytes(commit_id) == data
    assert stored_objects(publisher) == {f"run-commits/sha256/{commit_id}.json": data}
    if isinstance(publisher, S3CheckpointPublisher):
        assert len(publisher._client.puts) == 1
        assert publisher._client.puts[0]["IfNoneMatch"] == "*"


@pytest.mark.parametrize("parent", [None, "a" * 64])
def test_preparation_is_immutable_by_parent(publisher, parent):
    assert publisher.read_run_preparation_bytes(parent) is None
    publisher.publish_run_preparation_bytes(parent, b"null")
    publisher.publish_run_preparation_bytes(parent, b"null")
    with pytest.raises(ArtifactValidationError, match="different bytes|differs"):
        publisher.publish_run_preparation_bytes(parent, b"false")

    assert publisher.read_run_preparation_bytes(parent) == b"null"
    name = "root" if parent is None else parent
    assert stored_objects(publisher) == {f"run-preparations/by-parent/{name}.json": b"null"}
    if isinstance(publisher, S3CheckpointPublisher):
        assert len(publisher._client.puts) == 1
        assert publisher._client.puts[0]["IfNoneMatch"] == "*"


@pytest.mark.parametrize("identifier", ["../escape", "A" * 64, "a" * 63, ""])
def test_invalid_ids_fail_before_storage_access(publisher, identifier, monkeypatch):
    def unexpected_access(*args, **kwargs):
        pytest.fail("Invalid ID reached storage")

    if isinstance(publisher, S3CheckpointPublisher):
        monkeypatch.setattr(publisher._client, "get_object", unexpected_access)
        monkeypatch.setattr(publisher._client, "put_object", unexpected_access)
    else:
        monkeypatch.setattr(type(publisher.model_root), "is_file", unexpected_access)
        monkeypatch.setattr(type(publisher.model_root), "exists", unexpected_access)

    with pytest.raises(ArtifactValidationError, match="run_commit_id"):
        publisher.publish_run_commit_bytes(identifier, None)
    with pytest.raises(ArtifactValidationError, match="run_commit_id"):
        publisher.read_run_commit_bytes(identifier)
    with pytest.raises(ArtifactValidationError, match="parent_run_commit_id"):
        publisher.publish_run_preparation_bytes(identifier, None)
    with pytest.raises(ArtifactValidationError, match="parent_run_commit_id"):
        publisher.read_run_preparation_bytes(identifier)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (None, "must be bytes"),
        ("null", "must be bytes"),
        (bytearray(b"null"), "must be bytes"),
        (b"\xff", "not valid UTF-8 JSON"),
        (b"{broken", "not valid UTF-8 JSON"),
        (b"null\n", "not canonical JSON"),
        (b'{"b":2,"a":1}', "not canonical JSON"),
        (b"NaN", "not canonical JSON"),
    ],
)
def test_publication_validates_bytes_and_canonical_encoding_before_digest(publisher, data, message):
    with pytest.raises(ArtifactValidationError, match=message):
        publisher.publish_run_commit_bytes("a" * 64, data)
    with pytest.raises(ArtifactValidationError, match=message):
        publisher.publish_run_preparation_bytes(None, data)
    assert stored_objects(publisher) == {}


def test_commit_publication_rejects_wrong_digest_without_writes(publisher):
    with pytest.raises(ArtifactValidationError, match="SHA-256 does not match"):
        publisher.publish_run_commit_bytes("a" * 64, b"null")
    assert stored_objects(publisher) == {}


@pytest.mark.parametrize("data", [b"{broken", b"null\n"])
@pytest.mark.parametrize("matching_digest", [False, True])
def test_commit_read_checks_digest_before_canonical_encoding(publisher, data, matching_digest):
    commit_id = sha256(data).hexdigest() if matching_digest else "a" * 64
    write_raw(publisher, f"run-commits/sha256/{commit_id}.json", data)
    message = "JSON" if matching_digest else "SHA-256 does not match"
    with pytest.raises(ArtifactValidationError, match=message):
        publisher.read_run_commit_bytes(commit_id)


def test_missing_commit_raises(publisher):
    with pytest.raises(ArtifactValidationError, match="Run commit does not exist"):
        publisher.read_run_commit_bytes("a" * 64)


@pytest.mark.parametrize("data", [b"{broken", b"null\n"])
def test_present_invalid_preparation_raises(publisher, data):
    write_raw(publisher, "run-preparations/by-parent/root.json", data)
    with pytest.raises(ArtifactValidationError, match="JSON"):
        publisher.read_run_preparation_bytes(None)


def test_conflicting_commit_is_never_overwritten(publisher):
    commit_id = sha256(b"null").hexdigest()
    relative = f"run-commits/sha256/{commit_id}.json"
    write_raw(publisher, relative, b"corrupt existing object")
    with pytest.raises(ArtifactValidationError, match="different bytes|differs"):
        publisher.publish_run_commit_bytes(commit_id, b"null")
    assert stored_objects(publisher) == {relative: b"corrupt existing object"}


def test_preparation_directory_is_invalid(tmp_path):
    publisher = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    (tmp_path / "run-preparations/by-parent/root.json").mkdir(parents=True)
    with pytest.raises(ArtifactValidationError, match="must be a regular file"):
        publisher.read_run_preparation_bytes(None)


def test_s3_reads_remote_authority_without_local_fallback(tmp_path):
    publisher = S3CheckpointPublisher(
        model_root=tmp_path, contract=CONTRACT, bucket="test-bucket", client=FakeS3()
    )
    local = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    commit_id = sha256(b"null").hexdigest()
    local.publish_run_commit_bytes(commit_id, b"null")
    local.publish_run_preparation_bytes(None, b"null")

    with pytest.raises(ArtifactValidationError, match="Run commit does not exist"):
        publisher.read_run_commit_bytes(commit_id)
    assert publisher.read_run_preparation_bytes(None) is None


def test_s3_profile_namespaces_remain_separate(tmp_path):
    client = FakeS3()
    first = S3CheckpointPublisher(
        model_root=tmp_path / "first", contract=CONTRACT, bucket="test-bucket", client=client
    )
    other = S3CheckpointPublisher(
        model_root=tmp_path / "other",
        contract=replace(CONTRACT, env_contract_version=2),
        bucket="test-bucket",
        client=client,
    )
    commit_id = sha256(b"null").hexdigest()
    first.publish_run_commit_bytes(commit_id, b"null")
    first.publish_run_preparation_bytes(None, b"null")
    with pytest.raises(ArtifactValidationError, match="Run commit does not exist"):
        other.read_run_commit_bytes(commit_id)
    assert other.read_run_preparation_bytes(None) is None


def test_s3_access_errors_are_not_absence(tmp_path, monkeypatch):
    client = FakeS3()
    publisher = S3CheckpointPublisher(
        model_root=tmp_path, contract=CONTRACT, bucket="test-bucket", client=client
    )

    def deny_access(**kwargs):
        raise S3Error("AccessDenied")

    monkeypatch.setattr(client, "get_object", deny_access)
    with pytest.raises(S3Error, match="AccessDenied"):
        publisher.read_run_commit_bytes("a" * 64)
    with pytest.raises(S3Error, match="AccessDenied"):
        publisher.read_run_preparation_bytes(None)
