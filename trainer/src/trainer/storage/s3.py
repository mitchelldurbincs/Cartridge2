"""S3 backend for model storage.

This backend is designed for Kubernetes deployments where models
need to be shared across pods without a shared filesystem.

Supports S3-compatible storage including:
- AWS S3
- MinIO
- Google Cloud Storage (via S3 compatibility)
- DigitalOcean Spaces

Requires: boto3

Environment variables:
    CARTRIDGE_S3_BUCKET: S3 bucket name
    CARTRIDGE_S3_ENDPOINT: S3-compatible endpoint (for MinIO, etc.)
    CARTRIDGE_S3_PREFIX: Key prefix for all objects (default: "models/")
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_REGION: AWS region (default: us-east-1)

Sibling: FilesystemModelStore in filesystem.py. Naming, best-model
serialization, and checkpoint rotation shared by both live in base.py.
"""

import logging
import os
import tempfile
import time
from pathlib import Path

import torch

from trainer.storage.base import (
    ModelInfo,
    ModelStore,
    checkpoint_filename,
    decode_best_model_metadata,
    encode_best_model_metadata,
    parse_checkpoint_step,
)

logger = logging.getLogger(__name__)


class S3ModelStore(ModelStore):
    """S3-backed model storage.

    Stores models in an S3 bucket with the following key structure:
        {prefix}/
            latest.onnx
            best.onnx
            latest.pt
            checkpoints/
                model_step_000100.onnx
                model_step_000200.onnx
            metadata/
                best_model.json

    Uses versioning/ETags for efficient change detection.
    """

    def __init__(
        self,
        bucket: str,
        endpoint: str | None = None,
        prefix: str = "models",
        region: str | None = None,
        local_cache_dir: str | None = None,
    ):
        """Initialize S3 model store.

        Args:
            bucket: S3 bucket name.
            endpoint: S3-compatible endpoint URL (for MinIO, etc.).
            prefix: Key prefix for all objects.
            region: AWS region (default: from env or us-east-1).
            local_cache_dir: Local directory for caching models.

        Raises:
            ImportError: If boto3 is not installed.
        """
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise ImportError(
                "S3 backend requires boto3. Install with: pip install boto3"
            ) from e

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._best_step: int | None = None

        # Set up local cache
        if local_cache_dir:
            self._cache_dir = Path(local_cache_dir)
        else:
            self._cache_dir = Path(tempfile.gettempdir()) / "cartridge_model_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Configure S3 client
        config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=30,
        )

        client_kwargs = {"config": config}

        if endpoint:
            client_kwargs["endpoint_url"] = endpoint

        if region:
            client_kwargs["region_name"] = region
        elif "AWS_REGION" not in os.environ:
            client_kwargs["region_name"] = "us-east-1"

        self._s3 = boto3.client("s3", **client_kwargs)

        # Load best model metadata
        self._load_best_metadata()

    def _key(self, name: str) -> str:
        """Build full S3 key from name."""
        return f"{self.prefix}/{name}"

    def _load_best_metadata(self) -> None:
        """Load best model metadata from S3."""
        try:
            response = self._s3.get_object(
                Bucket=self.bucket,
                Key=self._key("metadata/best_model.json"),
            )
            self._best_step = decode_best_model_metadata(
                response["Body"].read().decode("utf-8")
            )
        except self._s3.exceptions.NoSuchKey:
            pass
        except Exception as e:
            logger.debug(f"Could not load best model metadata: {e}")

    def _save_best_metadata(self, step: int) -> None:
        """Save best model metadata to S3."""
        data = encode_best_model_metadata(step)
        self._s3.put_object(
            Bucket=self.bucket,
            Key=self._key("metadata/best_model.json"),
            Body=data.encode("utf-8"),
            ContentType="application/json",
        )

    def save_onnx(
        self,
        model_bytes: bytes,
        step: int,
        is_latest: bool = True,
    ) -> ModelInfo:
        """Save an ONNX model checkpoint to S3."""
        # Upload checkpoint
        checkpoint_key = self._key(f"checkpoints/{checkpoint_filename(step)}")
        self._s3.put_object(
            Bucket=self.bucket,
            Key=checkpoint_key,
            Body=model_bytes,
            ContentType="application/octet-stream",
        )

        if is_latest:
            # Also upload as latest.onnx
            latest_key = self._key("latest.onnx")
            self._s3.put_object(
                Bucket=self.bucket,
                Key=latest_key,
                Body=model_bytes,
                ContentType="application/octet-stream",
                Metadata={"step": str(step)},
            )

        logger.debug(f"Saved ONNX model to s3://{self.bucket}/{checkpoint_key}")

        return ModelInfo(
            path=f"s3://{self.bucket}/{checkpoint_key}",
            step=step,
            timestamp=time.time(),
            is_latest=is_latest,
        )

    def save_pytorch(
        self,
        state_dict: dict,
        step: int,
    ) -> str:
        """Save PyTorch training state to S3."""
        # Save to local temp file first, then upload
        state_dict["step"] = step

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(state_dict, f.name)
            f.flush()

            key = self._key("latest.pt")
            self._s3.upload_file(f.name, self.bucket, key)

        os.unlink(f.name)
        logger.debug(f"Saved PyTorch checkpoint to s3://{self.bucket}/{key}")
        return f"s3://{self.bucket}/{key}"

    def load_pytorch(self) -> tuple[dict, int] | None:
        """Load the latest PyTorch training state from S3."""
        key = self._key("latest.pt")

        try:
            # Download to local cache
            local_path = self._cache_dir / "latest.pt"
            self._s3.download_file(self.bucket, key, str(local_path))

            checkpoint = torch.load(local_path, map_location="cpu", weights_only=True)
            step = checkpoint.get("step", 0)
            logger.info(f"Loaded PyTorch checkpoint from s3://{self.bucket}/{key}")
            return checkpoint, step

        except self._s3.exceptions.NoSuchKey:
            logger.debug(f"No PyTorch checkpoint found at s3://{self.bucket}/{key}")
            return None
        except Exception as e:
            logger.warning(f"Failed to load PyTorch checkpoint: {e}")
            return None

    def get_latest_info(self) -> ModelInfo | None:
        """Get info about the latest model."""
        key = self._key("latest.onnx")

        try:
            response = self._s3.head_object(Bucket=self.bucket, Key=key)

            # Try to get step from metadata
            step = 0
            if "Metadata" in response and "step" in response["Metadata"]:
                step = int(response["Metadata"]["step"])

            return ModelInfo(
                path=f"s3://{self.bucket}/{key}",
                step=step,
                timestamp=response["LastModified"].timestamp(),
                is_latest=True,
            )
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.debug(f"Could not get latest model info: {e}")
            return None

    def get_latest_version(self) -> int | None:
        """Get the version/ETag of the latest model.

        Uses ETag as a version indicator for efficient change detection.
        """
        key = self._key("latest.onnx")

        try:
            response = self._s3.head_object(Bucket=self.bucket, Key=key)
            # Convert ETag to int for comparison
            etag = response["ETag"].strip('"')
            return hash(etag)
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.debug(f"Could not get latest model version: {e}")
            return None

    def load_latest_onnx(self) -> bytes | None:
        """Load the latest ONNX model bytes from S3."""
        key = self._key("latest.onnx")

        try:
            response = self._s3.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.warning(f"Failed to load latest ONNX model: {e}")
            return None

    def list_checkpoints(self) -> list[ModelInfo]:
        """List all available model checkpoints."""
        prefix = self._key("checkpoints/model_step_")

        checkpoints = []
        paginator = self._s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                step = parse_checkpoint_step(key)
                if step is not None:
                    checkpoints.append(
                        ModelInfo(
                            path=f"s3://{self.bucket}/{key}",
                            step=step,
                            timestamp=obj["LastModified"].timestamp(),
                            is_best=(step == self._best_step),
                        )
                    )

        return sorted(checkpoints, key=lambda x: x.step)

    def _delete_checkpoint(self, checkpoint: ModelInfo) -> None:
        """Delete a checkpoint object (rotation lives in ModelStore)."""
        # Extract key from s3:// path
        key = checkpoint.path.replace(f"s3://{self.bucket}/", "")
        self._s3.delete_object(Bucket=self.bucket, Key=key)

    def mark_as_best(self, step: int) -> None:
        """Mark a specific checkpoint as the 'best' model."""
        src_key = self._key(f"checkpoints/{checkpoint_filename(step)}")
        dst_key = self._key("best.onnx")

        try:
            # Copy checkpoint to best.onnx
            self._s3.copy_object(
                Bucket=self.bucket,
                Key=dst_key,
                CopySource={"Bucket": self.bucket, "Key": src_key},
            )
            self._best_step = step
            self._save_best_metadata(step)
            logger.info(f"Marked step {step} as best model")
        except Exception as e:
            logger.warning(f"Failed to mark step {step} as best: {e}")

    def get_best_info(self) -> ModelInfo | None:
        """Get info about the best model."""
        key = self._key("best.onnx")

        try:
            response = self._s3.head_object(Bucket=self.bucket, Key=key)
            return ModelInfo(
                path=f"s3://{self.bucket}/{key}",
                step=self._best_step or 0,
                timestamp=response["LastModified"].timestamp(),
                is_best=True,
            )
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.debug(f"Could not get best model info: {e}")
            return None

    def download_to_local(self, s3_path: str, local_path: Path) -> Path:
        """Download a model from S3 to local filesystem.

        Useful for loading into ONNX runtime.

        Args:
            s3_path: Full s3:// path or just the key.
            local_path: Local destination path.

        Returns:
            Path to downloaded file.
        """
        if s3_path.startswith("s3://"):
            # Parse s3://bucket/key format
            parts = s3_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            bucket = self.bucket
            key = s3_path

        self._s3.download_file(bucket, key, str(local_path))
        return local_path
