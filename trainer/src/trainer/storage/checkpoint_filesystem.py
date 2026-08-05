"""Filesystem checkpoint repository."""

from __future__ import annotations

import logging
from pathlib import Path

from .artifact_codec import (
    ArtifactValidationError,
    decode_canonical_json,
    require_digest,
    sha256_bytes,
)
from .checkpoint_types import (
    BlobDescriptorV1,
    CheckpointManifestV1,
    CheckpointRef,
    OnnxArtifactContract,
    RunHeadV2,
)
from .checkpoint_validation import (
    snapshot_checkpoint_artifacts,
    validate_learner_state,
    validate_onnx_checkpoint,
    verified_blob,
)
from .evaluation_repository import create_evaluation_repository
from .filesystem_backend import atomic_replace, create_or_verify, directory_lock
from .run_commit_repository import RunCommitRepository
from .run_commit_types import RunCommitV1
from .run_lineage_cache import RunLineageCache

logger = logging.getLogger(__name__)


class FilesystemCheckpointPublisher:
    """Checkpoint repository rooted at one profile's models directory."""

    def __init__(self, *, model_root: str | Path, contract: OnnxArtifactContract):
        self.model_root = Path(model_root)
        self.contract = contract
        # Shared by every repository built over this publisher; see
        # run_lineage_cache for why cached chains never need invalidation.
        self._lineage_cache = RunLineageCache()

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
        return directory_lock(self.model_root)

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
        atomic_replace(self._head_path, target)
        return RunHeadV2.from_bytes(target)

    def publish_run_commit_bytes(self, run_commit_id: str, data: bytes) -> None:
        require_digest(run_commit_id, field="run_commit_id")
        if not isinstance(data, bytes):
            raise ArtifactValidationError("Run commit must be bytes")
        decode_canonical_json(data, context="run commit")
        if sha256_bytes(data) != run_commit_id:
            raise ArtifactValidationError("Run commit SHA-256 does not match its ID")
        create_or_verify(self._run_commit_path(run_commit_id), data)

    def read_run_commit_bytes(self, run_commit_id: str) -> bytes:
        require_digest(run_commit_id, field="run_commit_id")
        path = self._run_commit_path(run_commit_id)
        if not path.is_file():
            raise ArtifactValidationError(f"Run commit does not exist: {run_commit_id}")
        data = path.read_bytes()
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
        create_or_verify(self._run_preparation_path(parent_run_commit_id), data)

    def read_run_preparation_bytes(self, parent_run_commit_id: str | None) -> bytes | None:
        if parent_run_commit_id is not None:
            require_digest(parent_run_commit_id, field="parent_run_commit_id")
        path = self._run_preparation_path(parent_run_commit_id)
        if not path.exists():
            return None
        if not path.is_file():
            raise ArtifactValidationError(f"Run preparation must be a regular file: {path}")
        data = path.read_bytes()
        decode_canonical_json(data, context="run preparation")
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
        require_digest(checkpoint_id, field="checkpoint_id")
        if expected_config_sha256 is not None:
            require_digest(expected_config_sha256, field="expected_config_sha256")
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
        create_or_verify(onnx_path, onnx_data)
        create_or_verify(learner_path, learner_data)
        create_or_verify(self._manifest_path(checkpoint_id), manifest.to_bytes())
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
        logger.info("Staged immutable checkpoint %s", checkpoint.checkpoint_id)
        return checkpoint

    @staticmethod
    def _decode_typed_run_commit(data: bytes):
        try:
            return RunCommitV1.from_bytes(data)
        except ArtifactValidationError:
            raise
        except Exception as exc:
            raise ArtifactValidationError(f"Invalid run commit: {exc}") from exc

    def _resolve_run_commit_chain(self, run_commit_id: str):
        repository = RunCommitRepository(self, create_evaluation_repository(self))
        return repository.resolve_chain(run_commit_id)

    def commit_run_head(
        self,
        *,
        checkpoint_id: str,
        run_commit_id: str,
        expected_run_commit_id: str | None,
    ) -> RunHeadV2:
        """Atomically select one validated RunCommit and its checkpoint."""
        require_digest(checkpoint_id, field="checkpoint_id")
        require_digest(run_commit_id, field="run_commit_id")
        if expected_run_commit_id is not None:
            require_digest(expected_run_commit_id, field="expected_run_commit_id")
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
        target = RunHeadV2(checkpoint_id=checkpoint.checkpoint_id, run_commit_id=run_commit_id)
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
                    f"got {actual_parent_commit_id!r}, expected {expected_run_commit_id!r}"
                )
            if run_commit.parent_run_commit_id != actual_parent_commit_id:
                raise ArtifactValidationError(
                    "Run commit parent must equal the authoritative run head"
                )
            self._validate_new_head_checkpoint(checkpoint, current, run_commit.config_sha256)
            observed = self._compare_and_set_head_version(
                expected=current_version,
                target=target.to_bytes(),
            )
        return observed

    def _validate_new_head_checkpoint(
        self,
        checkpoint: CheckpointRef,
        current: RunHeadV2 | None,
        config_sha256: str,
    ) -> None:
        if current is None:
            if checkpoint.manifest.parent_checkpoint_id is not None:
                raise ArtifactValidationError("The first RunCommit must select a root checkpoint")
            return
        current_manifest = self.read_checkpoint_manifest_exact(
            current.checkpoint_id,
            expected_config_sha256=config_sha256,
        )
        if checkpoint.checkpoint_id == current.checkpoint_id:
            raise ArtifactValidationError("A RunCommit must select a new direct-child checkpoint")
        if checkpoint.manifest.parent_checkpoint_id != current.checkpoint_id:
            raise ArtifactValidationError(
                "A new RunCommit checkpoint must directly extend the current RunHead checkpoint"
            )
        if checkpoint.manifest.step <= current_manifest.step:
            raise ArtifactValidationError("RunCommit checkpoint step must strictly increase")

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
            require_digest(expected_config_sha256, field="expected_config_sha256")
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
        validate_learner_state(
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
            require_digest(checkpoint_id, field="manifest filename checkpoint_id")
            checkpoint_ids.append(checkpoint_id)
        checkpoints = [self.resolve_checkpoint(item) for item in checkpoint_ids]
        checkpoints.sort(key=lambda item: (item.manifest.step, item.checkpoint_id))
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
        verified_blob(path.read_bytes(), descriptor, name=name)
