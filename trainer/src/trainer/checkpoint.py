"""Model checkpointing with atomic writes and cleanup.

This module handles ONNX model export and checkpoint management:
- Atomic write-then-rename for safe checkpointing (via trainer.atomic_io)
- PyTorch state dict saving/loading for training continuity
- Checkpoint rotation to limit disk usage
- Cleanup of orphaned temporary files from PyTorch ONNX exporter

Sibling: FilesystemModelStore in storage/filesystem.py implements the same
conventions behind the ModelStore interface (taking model bytes rather than
exporting a live network).
"""

import glob
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import torch

from trainer.atomic_io import atomic_copy, atomic_write

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer

logger = logging.getLogger(__name__)

# PyTorch checkpoint filename (kept alongside ONNX checkpoints)
PYTORCH_CHECKPOINT_NAME = "latest.pt"


def save_onnx_checkpoint(
    network: "nn.Module",
    obs_size: int,
    step: int,
    model_dir: Path,
    device: torch.device,
) -> Path:
    """Export network to ONNX checkpoint with atomic write-then-rename.

    Exports ONNX once, then copies to latest.onnx to avoid duplicate export work.

    Args:
        network: The neural network to export.
        obs_size: Size of the observation input.
        step: Current training step (used for checkpoint naming).
        model_dir: Directory to save checkpoints.
        device: Device the network is on.

    Returns:
        Path to the saved checkpoint file.

    Raises:
        Exception: If ONNX export fails.
    """
    network.eval()

    # Create deterministic dummy input for ONNX export
    dummy_input = torch.zeros(1, obs_size, device=device)

    def _export(temp_path: str) -> None:
        torch.onnx.export(
            network,
            dummy_input,
            temp_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["observation"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "policy_logits": {0: "batch_size"},
                "value": {0: "batch_size"},
            },
            dynamo=False,
        )

        # If PyTorch emitted external data, inline it so the checkpoint is
        # self-contained and survives renames.
        data_sidecar = f"{temp_path}.data"
        if os.path.exists(data_sidecar):
            model = onnx.load(temp_path, load_external_data=True)
            onnx.save_model(model, temp_path, save_as_external_data=False)
            os.unlink(data_sidecar)

    # Same filename convention as trainer.storage.base.checkpoint_filename
    checkpoint_path = model_dir / f"model_step_{step:06d}.onnx"
    atomic_write(checkpoint_path, _export)

    # Copy to latest.onnx (instead of exporting twice)
    atomic_copy(checkpoint_path, model_dir / "latest.onnx")

    return checkpoint_path


def save_pytorch_checkpoint(
    network: "nn.Module",
    optimizer: "Optimizer",
    step: int,
    model_dir: Path,
    scheduler: "Any | None" = None,
) -> Path:
    """Save PyTorch model and optimizer state for training continuity.

    Uses atomic write-then-rename to prevent corruption.

    Args:
        network: The neural network to save.
        optimizer: The optimizer with its state.
        step: Current training step.
        model_dir: Directory to save checkpoint.
        scheduler: Optional LR scheduler to save state for.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_path = model_dir / PYTORCH_CHECKPOINT_NAME

    checkpoint_data = {
        "step": step,
        "model_state_dict": network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    # Save scheduler state to prevent LR jumps on resume
    if scheduler is not None:
        checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

    atomic_write(
        checkpoint_path, lambda temp_path: torch.save(checkpoint_data, temp_path)
    )
    logger.debug(f"Saved PyTorch checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_pytorch_checkpoint(
    network: "nn.Module",
    optimizer: "Optimizer",
    model_dir: Path,
    device: torch.device,
) -> tuple[int, dict | None] | None:
    """Load PyTorch model and optimizer state from checkpoint.

    Args:
        network: The neural network to load weights into.
        optimizer: The optimizer to load state into.
        model_dir: Directory containing the checkpoint.
        device: Device to map tensors to.

    Returns:
        Tuple of (step, scheduler_state_dict) if checkpoint exists,
        where scheduler_state_dict may be None if not saved.
        Returns None if no checkpoint exists.
    """
    checkpoint_path = model_dir / PYTORCH_CHECKPOINT_NAME

    if not checkpoint_path.exists():
        logger.debug(f"No PyTorch checkpoint found at {checkpoint_path}")
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = checkpoint.get("step", 0)
        scheduler_state = checkpoint.get("scheduler_state_dict")
        logger.info(f"Loaded PyTorch checkpoint from step {step}: {checkpoint_path}")
        return step, scheduler_state

    except Exception as e:
        logger.warning(f"Failed to load PyTorch checkpoint: {e}")
        return None


def cleanup_old_checkpoints(
    checkpoints: list[Path],
    max_keep: int,
) -> list[Path]:
    """Remove old checkpoints to save disk space.

    Removes checkpoints from the front of the list (oldest first) until
    the list is at most max_keep items long.

    Args:
        checkpoints: List of checkpoint paths (oldest first).
        max_keep: Maximum number of checkpoints to retain.

    Returns:
        Updated list of checkpoints after cleanup.
    """
    while len(checkpoints) > max_keep:
        old_checkpoint = checkpoints.pop(0)
        if old_checkpoint.exists():
            old_checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {old_checkpoint}")

    return checkpoints


def cleanup_temp_onnx_data(model_dir: Path) -> None:
    """Remove orphaned tmp*.onnx.data files created by PyTorch ONNX exporter.

    These files can accumulate when ONNX export creates external data files
    that are later inlined or when exports fail partway through.

    Args:
        model_dir: Directory to clean up.
    """
    pattern = str(model_dir / "tmp*.onnx.data")
    for data_file in glob.glob(pattern):
        try:
            os.unlink(data_file)
            logger.debug(f"Removed orphaned ONNX data file: {data_file}")
        except OSError as e:
            logger.warning(f"Failed to remove {data_file}: {e}")
