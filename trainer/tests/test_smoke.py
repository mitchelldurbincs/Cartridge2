"""Smoke tests for the trainer.

Run with: pytest tests/test_smoke.py -v

Note: Tests requiring the replay buffer need PostgreSQL running.
Set CARTRIDGE_STORAGE_POSTGRES_URL environment variable to run integration tests.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from trainer.network import AlphaZeroLoss, PolicyValueNetwork, create_network
from trainer.trainer import Trainer, TrainerConfig


class TestNetwork:
    """Tests for the neural network."""

    def test_network_creation(self):
        net = create_network("tictactoe")
        assert net.obs_size == 29
        assert net.action_size == 9

    def test_network_forward(self):
        net = create_network("tictactoe")
        batch = torch.randn(8, 29)

        policy_logits, value = net(batch)

        assert policy_logits.shape == (8, 9)
        assert value.shape == (8, 1)
        # Value should be in [-1, 1] due to tanh
        assert (value >= -1).all() and (value <= 1).all()

    def test_network_predict_with_mask(self):
        net = create_network("tictactoe")
        batch = torch.randn(4, 29)
        # Mask out positions 0, 1, 2 as illegal
        legal_mask = torch.ones(4, 9)
        legal_mask[:, :3] = 0

        policy_probs, value = net.predict(batch, legal_mask)

        assert policy_probs.shape == (4, 9)
        # Illegal moves should have zero probability
        assert (policy_probs[:, :3] == 0).all()
        # Probabilities should sum to 1
        assert torch.allclose(policy_probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_create_network(self):
        net = create_network("tictactoe")
        assert isinstance(net, PolicyValueNetwork)

        with pytest.raises(ValueError):
            create_network("unknown_game")


class TestAlphaZeroLoss:
    """Tests for the AlphaZero loss function."""

    def test_loss_computation(self):
        loss_fn = AlphaZeroLoss()
        batch_size = 16
        num_actions = 9

        policy_logits = torch.randn(batch_size, num_actions)
        values = torch.randn(batch_size, 1)
        policy_targets = torch.softmax(torch.randn(batch_size, num_actions), dim=1)
        value_targets = torch.rand(batch_size) * 2 - 1  # [-1, 1]
        legal_mask = torch.ones(batch_size, num_actions)

        total, metrics = loss_fn(
            policy_logits, values, policy_targets, value_targets, legal_mask
        )

        assert total.shape == ()
        assert total > 0
        assert "loss/total" in metrics
        assert "loss/value" in metrics
        assert "loss/policy" in metrics

    def test_loss_with_illegal_moves(self):
        loss_fn = AlphaZeroLoss()
        batch_size = 8
        num_actions = 9

        policy_logits = torch.randn(batch_size, num_actions)
        values = torch.randn(batch_size, 1)
        policy_targets = torch.softmax(torch.randn(batch_size, num_actions), dim=1)
        value_targets = torch.rand(batch_size) * 2 - 1

        # Mask out first 3 actions
        legal_mask = torch.ones(batch_size, num_actions)
        legal_mask[:, :3] = 0

        total, metrics = loss_fn(
            policy_logits, values, policy_targets, value_targets, legal_mask
        )

        # Loss should still compute
        assert total > 0
        assert "loss/total" in metrics


class TestTrainerConfig:
    """Tests for trainer configuration."""

    def test_config_defaults(self):
        config = TrainerConfig()
        assert config.model_dir == "./data/models"
        assert config.batch_size == 64
        assert config.total_steps == 1000

    def test_config_custom(self):
        config = TrainerConfig(
            model_dir="/custom/models",
            batch_size=128,
            total_steps=500,
        )
        assert config.model_dir == "/custom/models"
        assert config.batch_size == 128
        assert config.total_steps == 500


class TestTrainer:
    """Tests for the trainer.

    Note: Integration tests requiring PostgreSQL are skipped by default.
    Set CARTRIDGE_STORAGE_POSTGRES_URL to run them.
    """

    def test_trainer_creation_with_mock_replay(self):
        """Test trainer creation with mocked replay buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            config = TrainerConfig(
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=10,
            )

            # Mock the replay buffer creation to avoid needing PostgreSQL
            with patch("trainer.trainer.create_replay_buffer") as mock_factory:
                mock_replay = MagicMock()
                mock_replay.get_metadata.return_value = None
                mock_replay.count.return_value = 0
                mock_factory.return_value = mock_replay

                trainer = Trainer(config)
                assert trainer.network is not None
                assert model_dir.exists()

    @pytest.mark.skipif(
        not os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL"),
        reason="PostgreSQL not configured (set CARTRIDGE_STORAGE_POSTGRES_URL)",
    )
    def test_trainer_with_postgres(self):
        """Integration test with real PostgreSQL connection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            config = TrainerConfig(
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=10,
                max_wait=5.0,  # Short timeout for testing
            )

            trainer = Trainer(config)
            assert trainer.network is not None


class TestStorageFactory:
    """Tests for the storage factory."""

    def test_factory_requires_postgres_url(self):
        """Test that factory raises error without PostgreSQL URL."""
        from trainer.storage import create_replay_buffer

        # Clear any existing env vars and prevent central config fallback
        # (config.toml may provide a postgres_url, so we must block that path too)
        with patch.dict(os.environ, {}, clear=True), patch(
            "trainer.central_config.get_config", side_effect=Exception("no config")
        ):
            with pytest.raises(
                ValueError, match="PostgreSQL connection string required"
            ):
                create_replay_buffer()

    @pytest.mark.skipif(
        not os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL"),
        reason="PostgreSQL not configured",
    )
    def test_factory_with_postgres(self):
        """Test factory creates PostgresReplayBuffer with valid URL."""
        from trainer.storage import PostgresReplayBuffer, create_replay_buffer

        buffer = create_replay_buffer()
        assert isinstance(buffer, PostgresReplayBuffer)
        buffer.close()
