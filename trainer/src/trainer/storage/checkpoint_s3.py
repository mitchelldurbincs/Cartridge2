"""S3 checkpoint repository backed by a verified local cache."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ..runtime_profile import RuntimeProfile
from .artifact_codec import (
    ArtifactValidationError,
    decode_canonical_json,
    require_digest,
    sha256_bytes,
)
from .checkpoint_filesystem import FilesystemCheckpointPublisher
from .checkpoint_types import CheckpointManifestV1, CheckpointRef, OnnxArtifactContract, RunHeadV2
from .checkpoint_validation import (
    snapshot_checkpoint_artifacts,
    validate_learner_state,
    validate_onnx_checkpoint,
    verified_blob,
)

_CONDITIONAL_WRITE_ATTEMPTS = 5


def _error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error", {})
    if not isinstance(error, dict):
        return None
    code = error.get("Code")
    return code if isinstance(code, str) else None


def _is_missing(exc: Exception) -> bool:
    return _error_code(exc) in {"404", "NoSuchKey", "NotFound"}


def _is_conflict(exc: Exception) -> bool:
    return _error_code(exc) in {"409", "ConditionalRequestConflict"}


def _is_precondition_failed(exc: Exception) -> bool:
    return _error_code(exc) in {
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
            if _is_missing(exc):
                return None
            raise
        data = response["Body"].read()
        if not isinstance(data, bytes):
            raise ArtifactValidationError(f"S3 object body is not bytes: {key}")
        return data

    def _get_versioned(self, key: str) -> tuple[bytes, str] | None:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _is_missing(exc):
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
        require_digest(run_commit_id, field="run_commit_id")
        if not isinstance(data, bytes):
            raise ArtifactValidationError("Run commit must be bytes")
        decode_canonical_json(data, context="run commit")
        if sha256_bytes(data) != run_commit_id:
            raise ArtifactValidationError("Run commit SHA-256 does not match its ID")
        self._put_immutable(
            self._key(f"run-commits/sha256/{run_commit_id}.json"),
            data,
            "application/json",
        )

    def read_run_commit_bytes(self, run_commit_id: str) -> bytes:
        require_digest(run_commit_id, field="run_commit_id")
        data = self._get(self._key(f"run-commits/sha256/{run_commit_id}.json"))
        if data is None:
            raise ArtifactValidationError(f"Run commit does not exist: {run_commit_id}")
        if sha256_bytes(data) != run_commit_id:
            raise ArtifactValidationError("Run commit SHA-256 does not match its ID")
        decode_canonical_json(data, context="run commit")
        return data

    def publish_run_preparation_bytes(self, parent_run_commit_id: str | None, data: bytes) -> None:
        if parent_run_commit_id is not None:
            require_digest(parent_run_commit_id, field="parent_run_commit_id")
        if not isinstance(data, bytes):
            raise ArtifactValidationError("Run preparation must be bytes")
        decode_canonical_json(data, context="run preparation")
        name = "root" if parent_run_commit_id is None else parent_run_commit_id
        self._put_immutable(
            self._key(f"run-preparations/by-parent/{name}.json"),
            data,
            "application/json",
        )

    def read_run_preparation_bytes(self, parent_run_commit_id: str | None) -> bytes | None:
        if parent_run_commit_id is not None:
            require_digest(parent_run_commit_id, field="parent_run_commit_id")
        name = "root" if parent_run_commit_id is None else parent_run_commit_id
        data = self._get(self._key(f"run-preparations/by-parent/{name}.json"))
        if data is not None:
            decode_canonical_json(data, context="run preparation")
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
        for attempt in range(_CONDITIONAL_WRITE_ATTEMPTS):
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
                if not _is_precondition_failed(exc):
                    raise
                if (
                    _is_conflict(exc)
                    and raced is None
                    and attempt + 1 < _CONDITIONAL_WRITE_ATTEMPTS
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
            if confirmed is not None and self._head_selects_or_descends_from(confirmed[0], target):
                return RunHeadV2.from_bytes(confirmed[0])
            if _is_precondition_failed(exc):
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
        manifest, onnx_data, learner_data = snapshot_checkpoint_artifacts(
            Path(onnx_path),
            Path(learner_state_path),
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

    def _local_blob_or_download(self, descriptor, extension: str, *, name: str) -> bytes:
        """Return verified blob bytes, preferring the local content-addressed cache.

        The cached file is re-hashed against the manifest descriptor on every
        use (fail-closed against local corruption); only on absence or
        mismatch is the object actually downloaded. This is what keeps
        commit_run_head from re-downloading both full blobs per iteration.
        """
        path = self._blob_path(descriptor, extension)
        if path.is_file():
            cached = path.read_bytes()
            if len(cached) == descriptor.size_bytes and sha256_bytes(cached) == descriptor.sha256:
                return cached
        data = self._get(self._key(f"blobs/sha256/{descriptor.sha256}.{extension}"))
        if data is None:
            raise ArtifactValidationError("S3 checkpoint is missing a blob")
        verified_blob(data, descriptor, name=name)
        return data

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
        onnx_data = self._local_blob_or_download(manifest.onnx, "onnx", name="S3 ONNX blob")
        learner_data = self._local_blob_or_download(
            manifest.learner_state, "pt", name="S3 learner-state blob"
        )
        checkpoint = self._materialize_immutables(manifest, onnx_data, learner_data)
        validate_onnx_checkpoint(checkpoint.onnx_path, self.contract)
        validate_learner_state(
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
        return self._load_manifest(
            checkpoint_id,
            expected_config_sha256=expected_config_sha256,
        )

    def list_checkpoints(self) -> list[CheckpointRef]:
        prefix = self._key("manifests/sha256/")
        checkpoint_ids: list[str] = []
        continuation_token: str | None = None
        while True:
            request: dict[str, object] = {"Bucket": self.bucket, "Prefix": prefix}
            if continuation_token is not None:
                request["ContinuationToken"] = continuation_token
            response = self._client.list_objects_v2(**request)
            contents = response.get("Contents", [])
            if not isinstance(contents, list):
                raise ArtifactValidationError("S3 checkpoint listing Contents is invalid")
            checkpoint_ids.extend(self._checkpoint_ids_from_listing(contents, prefix))
            truncated = response.get("IsTruncated", False)
            if not isinstance(truncated, bool):
                raise ArtifactValidationError("S3 checkpoint listing IsTruncated is invalid")
            if not truncated:
                break
            continuation_token = response.get("NextContinuationToken")
            if not isinstance(continuation_token, str) or not continuation_token:
                raise ArtifactValidationError(
                    "Truncated S3 checkpoint listing has no continuation token"
                )
        if len(checkpoint_ids) != len(set(checkpoint_ids)):
            raise ArtifactValidationError("S3 checkpoint listing contains duplicate objects")
        checkpoints = [self.resolve_checkpoint(item) for item in checkpoint_ids]
        checkpoints.sort(key=lambda item: (item.manifest.step, item.checkpoint_id))
        return checkpoints

    @staticmethod
    def _checkpoint_ids_from_listing(contents: list[object], prefix: str) -> list[str]:
        checkpoint_ids: list[str] = []
        for item in contents:
            if not isinstance(item, dict) or not isinstance(item.get("Key"), str):
                raise ArtifactValidationError("S3 checkpoint listing entry is invalid")
            key = item["Key"]
            relative = key.removeprefix(prefix)
            if not key.startswith(prefix) or "/" in relative or not relative.endswith(".json"):
                raise ArtifactValidationError(f"Unexpected S3 checkpoint manifest object: {key}")
            checkpoint_id = relative.removesuffix(".json")
            require_digest(checkpoint_id, field="S3 manifest filename checkpoint_id")
            checkpoint_ids.append(checkpoint_id)
        return checkpoint_ids
