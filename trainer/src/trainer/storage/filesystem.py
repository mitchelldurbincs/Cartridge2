"""Filesystem backend for model storage.

This is the default backend for local development, storing ONNX models
and PyTorch checkpoints in the local filesystem.

Sibling: S3ModelStore in s3.py. Naming, best-model serialization, and
checkpoint rotation shared by both live in base.py; atomic-write helpers
are shared with trainer/checkpoint.py via trainer.atomic_io.
"""

import glob
import logging
import os
import time
from pathlib import Path

import torch

from trainer.atomic_io import atomic_copy, atomic_write
from trainer.checkpoint import cleanup_temp_onnx_data as _cleanup_temp_onnx_data
from trainer.storage.base import (
    ModelInfo,
    ModelStore,
    checkpoint_filename,
    decode_best_model_metadata,
    encode_best_model_metadata,
    parse_checkpoint_step,
)

logger = logging.getLogger(__name__)

# PyTorch checkpoint filename
PYTORCH_CHECKPOINT_NAME = "latest.pt"


class FilesystemModelStore(ModelStore):
    """Filesystem-backed model storage.

    Stores models in a local directory with atomic write-then-rename
    for safe checkpointing.

    Directory structure:
        {model_dir}/
            latest.onnx          - Current best model (hot-reloaded)
            best.onnx            - Best model from evaluation
            latest.pt            - PyTorch training state
            model_step_000100.onnx
            model_step_000200.onnx
            ...
    """

    def __init__(self, model_dir: str | Path):
        """Initialize filesystem model store.

        Args:
            model_dir: Directory to store models. Created if it doesn't exist.
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Track best model step (loaded from metadata file if exists)
        self._best_step: int | None = None
        self._load_best_metadata()

    def _load_best_metadata(self) -> None:
        """Load best model metadata from file."""
        meta_path = self.model_dir / "best_model.json"
        if meta_path.exists():
            try:
                self._best_step = decode_best_model_metadata(meta_path.read_text())
            except OSError:
                pass

    def _save_best_metadata(self, step: int) -> None:
        """Save best model metadata to file."""
        meta_path = self.model_dir / "best_model.json"
        payload = encode_best_model_metadata(step)
        atomic_write(meta_path, lambda temp_path: Path(temp_path).write_text(payload))

    def save_onnx(
        self,
        model_bytes: bytes,
        step: int,
        is_latest: bool = True,
    ) -> ModelInfo:
        """Save an ONNX model checkpoint with atomic write."""
        checkpoint_path = self.model_dir / checkpoint_filename(step)
        atomic_write(
            checkpoint_path, lambda temp_path: Path(temp_path).write_bytes(model_bytes)
        )

        if is_latest:
            # Copy to latest.onnx atomically
            atomic_copy(checkpoint_path, self.model_dir / "latest.onnx")

        return ModelInfo(
            path=str(checkpoint_path),
            step=step,
            timestamp=time.time(),
            is_latest=is_latest,
        )

    def save_pytorch(
        self,
        state_dict: dict,
        step: int,
    ) -> str:
        """Save PyTorch training state with atomic write."""
        checkpoint_path = self.model_dir / PYTORCH_CHECKPOINT_NAME

        # Add step to state dict
        state_dict["step"] = step
        atomic_write(
            checkpoint_path, lambda temp_path: torch.save(state_dict, temp_path)
        )
        logger.debug(f"Saved PyTorch checkpoint: {checkpoint_path}")
        return str(checkpoint_path)

    def load_pytorch(self) -> tuple[dict, int] | None:
        """Load the latest PyTorch training state."""
        checkpoint_path = self.model_dir / PYTORCH_CHECKPOINT_NAME

        if not checkpoint_path.exists():
            logger.debug(f"No PyTorch checkpoint found at {checkpoint_path}")
            return None

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            step = checkpoint.get("step", 0)
            logger.info(
                f"Loaded PyTorch checkpoint from step {step}: {checkpoint_path}"
            )
            return checkpoint, step

        except Exception as e:
            logger.warning(f"Failed to load PyTorch checkpoint: {e}")
            return None

    def get_latest_info(self) -> ModelInfo | None:
        """Get info about the latest model."""
        latest_path = self.model_dir / "latest.onnx"
        if not latest_path.exists():
            return None

        # Try to extract step from the actual checkpoint file
        step = self._extract_latest_step()

        return ModelInfo(
            path=str(latest_path),
            step=step or 0,
            timestamp=latest_path.stat().st_mtime,
            is_latest=True,
        )

    def _extract_latest_step(self) -> int | None:
        """Extract step number from the latest checkpoint file."""
        # Look at checkpoints to find the highest step
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[-1].step
        return None

    def get_latest_version(self) -> int | None:
        """Get the version/step of the latest model."""
        latest_path = self.model_dir / "latest.onnx"
        if not latest_path.exists():
            return None

        # Use mtime as a simple version indicator
        # For file-based storage, consumers can compare this
        return int(latest_path.stat().st_mtime * 1000)

    def load_latest_onnx(self) -> bytes | None:
        """Load the latest ONNX model bytes."""
        latest_path = self.model_dir / "latest.onnx"
        if not latest_path.exists():
            return None

        with open(latest_path, "rb") as f:
            return f.read()

    def list_checkpoints(self) -> list[ModelInfo]:
        """List all available model checkpoints."""
        pattern = str(self.model_dir / "model_step_*.onnx")
        files = glob.glob(pattern)

        checkpoints = []
        for filepath in sorted(files):
            step = parse_checkpoint_step(filepath)
            if step is not None:
                path = Path(filepath)
                checkpoints.append(
                    ModelInfo(
                        path=filepath,
                        step=step,
                        timestamp=path.stat().st_mtime,
                        is_best=(step == self._best_step),
                    )
                )

        return sorted(checkpoints, key=lambda x: x.step)

    def _delete_checkpoint(self, checkpoint: ModelInfo) -> None:
        """Delete a checkpoint file (rotation lives in ModelStore)."""
        os.unlink(checkpoint.path)

    def mark_as_best(self, step: int) -> None:
        """Mark a specific checkpoint as the 'best' model."""
        # Find the checkpoint
        checkpoint_path = self.model_dir / checkpoint_filename(step)
        best_path = self.model_dir / "best.onnx"

        if checkpoint_path.exists():
            atomic_copy(checkpoint_path, best_path)
            self._best_step = step
            self._save_best_metadata(step)
            logger.info(f"Marked step {step} as best model")
        else:
            logger.warning(f"Checkpoint not found for step {step}")

    def get_best_info(self) -> ModelInfo | None:
        """Get info about the best model."""
        best_path = self.model_dir / "best.onnx"
        if not best_path.exists():
            return None

        return ModelInfo(
            path=str(best_path),
            step=self._best_step or 0,
            timestamp=best_path.stat().st_mtime,
            is_best=True,
        )

    def cleanup_temp_onnx_data(self) -> None:
        """Remove orphaned tmp*.onnx.data files from PyTorch ONNX exporter."""
        _cleanup_temp_onnx_data(self.model_dir)
