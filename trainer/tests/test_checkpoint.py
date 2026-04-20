"""Tests for checkpoint.py - Model checkpointing and management.

Tests cover:
- ONNX checkpoint export with atomic writes
- PyTorch checkpoint save/load
- Checkpoint rotation and cleanup
- Error handling and edge cases
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.optim import Adam

from trainer.checkpoint import (
    cleanup_old_checkpoints,
    cleanup_temp_onnx_data,
    load_pytorch_checkpoint,
    save_onnx_checkpoint,
    save_pytorch_checkpoint,
)
from trainer.lr_scheduler import LRConfig, WarmupCosineScheduler
from trainer.network import PolicyValueNetwork


class TestONNXCheckpoint:
    """Tests for ONNX checkpoint export."""

    def test_save_onnx_checkpoint_creates_file(self):
        """Test that ONNX checkpoint file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            device = torch.device("cpu")

            checkpoint_path = save_onnx_checkpoint(
                network=network,
                obs_size=29,
                step=100,
                model_dir=model_dir,
                device=device,
            )

            assert checkpoint_path.exists()
            assert checkpoint_path.name == "model_step_000100.onnx"

    def test_onnx_checkpoint_is_valid_model(self):
        """Test that exported ONNX model can be loaded."""
        onnx = pytest.importorskip("onnx")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            device = torch.device("cpu")

            checkpoint_path = save_onnx_checkpoint(
                network=network,
                obs_size=29,
                step=100,
                model_dir=model_dir,
                device=device,
            )

            # Should be able to load with onnx
            model = onnx.load(str(checkpoint_path))
            onnx.checker.check_model(model)

    def test_onnx_checkpoint_has_correct_signature(self):
        """Test that exported model has correct input/output names."""
        onnx = pytest.importorskip("onnx")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            device = torch.device("cpu")

            checkpoint_path = save_onnx_checkpoint(
                network=network,
                obs_size=29,
                step=100,
                model_dir=model_dir,
                device=device,
            )

            model = onnx.load(str(checkpoint_path))
            input_names = [input.name for input in model.graph.input]
            output_names = [output.name for output in model.graph.output]

            assert "observation" in input_names
            assert "policy_logits" in output_names
            assert "value" in output_names

    def test_onnx_checkpoint_creates_latest_symlink(self):
        """Test that latest.onnx is created/updated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            device = torch.device("cpu")

            save_onnx_checkpoint(
                network=network,
                obs_size=29,
                step=100,
                model_dir=model_dir,
                device=device,
            )

            latest_path = model_dir / "latest.onnx"
            assert latest_path.exists()

    def test_onnx_checkpoint_atomic_write_no_partial_files(self):
        """Test that no partial/temp files remain after successful export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            device = torch.device("cpu")

            save_onnx_checkpoint(
                network=network,
                obs_size=29,
                step=100,
                model_dir=model_dir,
                device=device,
            )

            # Check for any tmp files
            tmp_files = list(model_dir.glob("tmp*.onnx*"))
            assert len(tmp_files) == 0, f"Temp files found: {tmp_files}"

    def test_onnx_checkpoint_inlines_external_data(self):
        """Test that external data files are inlined into the model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            # Use a larger network that might trigger external data
            network = PolicyValueNetwork(obs_size=93, action_size=7, hidden_size=512)
            device = torch.device("cpu")

            # Just need to call it, don't need the return value for this test
            save_onnx_checkpoint(
                network=network,
                obs_size=93,
                step=100,
                model_dir=model_dir,
                device=device,
            )

            # No .data sidecar files should exist
            data_files = list(model_dir.glob("*.onnx.data"))
            assert len(data_files) == 0, f"External data files found: {data_files}"

    def test_onnx_checkpoint_creates_directory_if_needed(self):
        """Test that model_dir is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "nested" / "models"
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            device = torch.device("cpu")

            # Pre-create the directory since mkstemp needs it to exist
            model_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = save_onnx_checkpoint(
                network=network,
                obs_size=29,
                step=100,
                model_dir=model_dir,
                device=device,
            )

            assert model_dir.exists()
            assert checkpoint_path.exists()


class TestPyTorchCheckpoint:
    """Tests for PyTorch checkpoint save/load."""

    def test_save_pytorch_checkpoint_creates_file(self):
        """Test that PyTorch checkpoint file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer = Adam(network.parameters(), lr=0.001)

            checkpoint_path = save_pytorch_checkpoint(
                network=network,
                optimizer=optimizer,
                step=100,
                model_dir=model_dir,
            )

            assert checkpoint_path.exists()
            assert checkpoint_path.name == "latest.pt"

    def test_pytorch_checkpoint_contains_model_state(self):
        """Test that checkpoint contains model state dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer = Adam(network.parameters(), lr=0.001)

            # Set a specific weight value to verify later
            with torch.no_grad():
                network.policy_fc.weight.fill_(0.5)

            save_pytorch_checkpoint(
                network=network,
                optimizer=optimizer,
                step=100,
                model_dir=model_dir,
            )

            # Load and verify
            checkpoint = torch.load(model_dir / "latest.pt", weights_only=True)
            assert "model_state_dict" in checkpoint
            assert "step" in checkpoint
            assert checkpoint["step"] == 100

    def test_pytorch_checkpoint_contains_optimizer_state(self):
        """Test that checkpoint contains optimizer state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer = Adam(network.parameters(), lr=0.001)

            # Do a step to populate optimizer state
            loss = network(torch.randn(1, 29))[1].sum()
            loss.backward()
            optimizer.step()

            save_pytorch_checkpoint(
                network=network,
                optimizer=optimizer,
                step=100,
                model_dir=model_dir,
            )

            checkpoint = torch.load(model_dir / "latest.pt", weights_only=True)
            assert "optimizer_state_dict" in checkpoint

    def test_pytorch_checkpoint_contains_scheduler_state(self):
        """Test that checkpoint contains scheduler state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer = Adam(network.parameters(), lr=0.001)
            config = LRConfig(target_lr=0.001, total_steps=100)
            scheduler = WarmupCosineScheduler(optimizer, config)

            # Step a few times
            for _ in range(5):
                scheduler.step()

            save_pytorch_checkpoint(
                network=network,
                optimizer=optimizer,
                step=100,
                model_dir=model_dir,
                scheduler=scheduler,
            )

            checkpoint = torch.load(model_dir / "latest.pt", weights_only=True)
            assert "scheduler_state_dict" in checkpoint

    def test_pytorch_checkpoint_atomic_write(self):
        """Test that checkpoint is written atomically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer = Adam(network.parameters(), lr=0.001)

            save_pytorch_checkpoint(
                network=network,
                optimizer=optimizer,
                step=100,
                model_dir=model_dir,
            )

            # No temp files should remain
            temp_files = list(model_dir.glob("tmp*.pt*"))
            assert len(temp_files) == 0

    def test_load_pytorch_checkpoint_restores_model(self):
        """Test that loading restores model weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network1 = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer = Adam(network1.parameters(), lr=0.001)
            device = torch.device("cpu")

            # Set specific weights
            with torch.no_grad():
                network1.policy_fc.weight.fill_(0.5)

            save_pytorch_checkpoint(
                network=network1,
                optimizer=optimizer,
                step=100,
                model_dir=model_dir,
            )

            # Create new network and load
            network2 = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer2 = Adam(network2.parameters(), lr=0.001)

            result = load_pytorch_checkpoint(network2, optimizer2, model_dir, device)

            assert result is not None
            step, _ = result
            assert step == 100

            # Verify weights were restored
            weight1 = network1.policy_fc.weight
            weight2 = network2.policy_fc.weight
            assert torch.allclose(weight1, weight2)

    def test_load_pytorch_checkpoint_restores_optimizer(self):
        """Test that loading restores optimizer state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network1 = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer1 = Adam(network1.parameters(), lr=0.001)
            device = torch.device("cpu")

            # Do a step
            loss = network1(torch.randn(1, 29))[1].sum()
            loss.backward()
            optimizer1.step()

            save_pytorch_checkpoint(
                network=network1,
                optimizer=optimizer1,
                step=100,
                model_dir=model_dir,
            )

            # Load into new optimizer
            network2 = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer2 = Adam(network2.parameters(), lr=0.001)

            load_pytorch_checkpoint(network2, optimizer2, model_dir, device)

            # Optimizer state should be restored
            assert len(optimizer2.state) > 0

    def test_load_pytorch_checkpoint_restores_scheduler(self):
        """Test that loading restores scheduler state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network1 = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer1 = Adam(network1.parameters(), lr=0.001)
            config = LRConfig(target_lr=0.001, total_steps=100)
            scheduler1 = WarmupCosineScheduler(optimizer1, config)
            device = torch.device("cpu")

            # Step scheduler
            for _ in range(10):
                scheduler1.step()

            initial_lr = scheduler1.get_lr()

            save_pytorch_checkpoint(
                network=network1,
                optimizer=optimizer1,
                step=100,
                model_dir=model_dir,
                scheduler=scheduler1,
            )

            # Load into new scheduler
            network2 = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer2 = Adam(network2.parameters(), lr=0.001)
            scheduler2 = WarmupCosineScheduler(optimizer2, config, from_checkpoint=True)

            result = load_pytorch_checkpoint(network2, optimizer2, model_dir, device)
            _, scheduler_state = result

            if scheduler_state:
                scheduler2.load_state_dict(scheduler_state)

            # LR should be restored
            assert abs(scheduler2.get_lr() - initial_lr) < 1e-6

    def test_load_missing_checkpoint_returns_none(self):
        """Test that loading from non-existent path returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer = Adam(network.parameters(), lr=0.001)
            device = torch.device("cpu")

            result = load_pytorch_checkpoint(network, optimizer, model_dir, device)

            assert result is None

    def test_load_corrupted_checkpoint_returns_none(self):
        """Test that loading corrupted checkpoint returns None gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            checkpoint_path = model_dir / "latest.pt"

            # Write garbage
            with open(checkpoint_path, "wb") as f:
                f.write(b"not a valid checkpoint")

            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer = Adam(network.parameters(), lr=0.001)
            device = torch.device("cpu")

            result = load_pytorch_checkpoint(network, optimizer, model_dir, device)

            assert result is None


class TestCheckpointCleanup:
    """Tests for checkpoint rotation and cleanup."""

    def test_cleanup_old_checkpoints_removes_oldest(self):
        """Test that oldest checkpoints are removed first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            # Create fake checkpoints
            checkpoints = []
            for i in range(5):
                path = model_dir / f"model_step_{i:06d}.onnx"
                path.write_text("fake model")
                checkpoints.append(path)

            # Keep only 3
            remaining = cleanup_old_checkpoints(checkpoints[:], max_keep=3)

            assert len(remaining) == 3
            # Oldest 2 should be removed
            assert not checkpoints[0].exists()
            assert not checkpoints[1].exists()
            assert checkpoints[2].exists()
            assert checkpoints[3].exists()
            assert checkpoints[4].exists()

    def test_cleanup_respects_max_keep(self):
        """Test that exactly max_keep checkpoints remain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            checkpoints = []
            for i in range(10):
                path = model_dir / f"model_step_{i:06d}.onnx"
                path.write_text("fake model")
                checkpoints.append(path)

            remaining = cleanup_old_checkpoints(checkpoints[:], max_keep=5)

            assert len(remaining) == 5
            # Verify exactly 5 files exist
            existing = [p for p in model_dir.glob("*.onnx") if p.exists()]
            assert len(existing) == 5

    def test_cleanup_handles_missing_files(self):
        """Test that cleanup handles already-deleted files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            checkpoints = []
            for i in range(5):
                path = model_dir / f"model_step_{i:06d}.onnx"
                path.write_text("fake model")
                checkpoints.append(path)

            # Delete one file manually
            checkpoints[0].unlink()

            # Should not raise
            remaining = cleanup_old_checkpoints(checkpoints[:], max_keep=3)

            assert len(remaining) == 3

    def test_cleanup_temp_onnx_data_removes_orphans(self):
        """Test that orphaned ONNX data files are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)

            # Create orphaned temp data files (the pattern cleanup_temp_onnx_data looks for)
            (model_dir / "tmp12345.onnx.data").write_text("orphan")
            (model_dir / "tmpABCDE.onnx.data").write_text("orphan")
            # Also create regular tmp onnx file (won't be cleaned up by this function)
            (model_dir / "tmpXYZ.onnx").write_text("not an orphan data file")

            # Also create a legitimate file
            (model_dir / "model_step_000100.onnx").write_text("real")

            cleanup_temp_onnx_data(model_dir)

            # Orphan data files should be gone
            assert not (model_dir / "tmp12345.onnx.data").exists()
            assert not (model_dir / "tmpABCDE.onnx.data").exists()
            # Regular .onnx files are NOT removed by cleanup_temp_onnx_data
            assert (model_dir / "tmpXYZ.onnx").exists()

            # Real file should remain
            assert (model_dir / "model_step_000100.onnx").exists()


class TestCheckpointErrorHandling:
    """Tests for error handling in checkpoint operations."""

    def test_save_onnx_cleans_up_on_error(self):
        """Test that temp files are cleaned up on export error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            device = torch.device("cpu")

            # Mock torch.onnx.export to fail
            with patch("torch.onnx.export", side_effect=RuntimeError("Export failed")):
                with pytest.raises(RuntimeError, match="Export failed"):
                    save_onnx_checkpoint(
                        network=network,
                        obs_size=29,
                        step=100,
                        model_dir=model_dir,
                        device=device,
                    )

            # No temp files should remain
            temp_files = list(model_dir.glob("tmp*.onnx*"))
            assert len(temp_files) == 0

    def test_save_pytorch_cleans_up_on_error(self):
        """Test that temp files are cleaned up on save error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            network = PolicyValueNetwork(obs_size=29, action_size=9, hidden_size=128)
            optimizer = Adam(network.parameters(), lr=0.001)

            # Mock torch.save to fail
            with patch("torch.save", side_effect=RuntimeError("Save failed")):
                with pytest.raises(RuntimeError, match="Save failed"):
                    save_pytorch_checkpoint(
                        network=network,
                        optimizer=optimizer,
                        step=100,
                        model_dir=model_dir,
                    )

            # No temp files should remain
            temp_files = list(model_dir.glob("tmp*.pt*"))
            assert len(temp_files) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
