"""Smoke tests for the AlphaZero learner.

Run with: pytest tests/test_smoke.py -v

Note: Tests requiring the replay buffer need PostgreSQL running.
Set CARTRIDGE_STORAGE_POSTGRES_URL environment variable to run integration tests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from trainer.algorithms.alphazero_board_v1 import ALGORITHM_ID, DESCRIPTOR
from trainer.algorithms.alphazero_config import AlphaZeroLearnerConfig
from trainer.network import AlphaZeroLoss, PolicyValueNetwork, create_network
from trainer.storage import ReplayProfile, ReplaySelection
from trainer.trainer import AlphaZeroLearner


def alphazero_profile(env_id: str = "tictactoe") -> ReplayProfile:
    return ReplayProfile(
        env_id=env_id,
        env_contract_version=2,
        algorithm_id=ALGORITHM_ID,
        experience_schema=DESCRIPTOR.components.experience_schema,
    )


def alphazero_selection(env_id: str = "tictactoe") -> ReplaySelection:
    return ReplaySelection(alphazero_profile(env_id), "a" * 64, None)


class TestNetwork:
    """Tests for the neural network."""

    def test_network_creation(self):
        net = create_network("tictactoe")
        assert net.obs_size == 18
        assert net.action_size == 9

    def test_network_forward(self):
        net = create_network("tictactoe")
        batch = torch.randn(8, 18)

        policy_logits, value = net(batch)

        assert policy_logits.shape == (8, 9)
        assert value.shape == (8, 1)
        # Value should be in [-1, 1] due to tanh
        assert (value >= -1).all() and (value <= 1).all()

    def test_network_predict_with_mask(self):
        net = create_network("tictactoe")
        batch = torch.randn(4, 18)
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

        total, metrics = loss_fn(policy_logits, values, policy_targets, value_targets, legal_mask)

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

        total, metrics = loss_fn(policy_logits, values, policy_targets, value_targets, legal_mask)

        # Loss should still compute
        assert total > 0
        assert "loss/total" in metrics


class TestAlphaZeroLearnerConfig:
    """Tests for trainer configuration."""

    def test_config_defaults(self):
        config = AlphaZeroLearnerConfig()
        assert config.model_dir == "./data/models"
        assert config.batch_size == 64
        assert config.total_steps == 1000

    def test_config_custom(self):
        config = AlphaZeroLearnerConfig(
            model_dir="/custom/models",
            batch_size=128,
            total_steps=500,
        )
        assert config.model_dir == "/custom/models"
        assert config.batch_size == 128
        assert config.total_steps == 500


class TestAlphaZeroLearner:
    """Tests for the trainer.

    Note: Integration tests requiring PostgreSQL are skipped by default.
    Set CARTRIDGE_STORAGE_POSTGRES_URL to run them.
    """

    def test_learner_opens_its_explicit_experience_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            config = AlphaZeroLearnerConfig(
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=10,
                replay_selection=alphazero_selection(),
            )

            with patch("trainer.trainer.create_replay_store") as mock_factory:
                mock_replay = MagicMock()
                mock_factory.return_value = mock_replay

                trainer = AlphaZeroLearner(config)
                assert trainer.network is not None
                assert model_dir.exists()
                assert trainer.replay_profile == alphazero_profile()
                assert trainer._create_replay_store() is mock_replay
                mock_factory.assert_called_once_with(alphazero_selection())

    @pytest.mark.skipif(
        not os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL"),
        reason="PostgreSQL not configured (set CARTRIDGE_STORAGE_POSTGRES_URL)",
    )
    def test_trainer_with_postgres(self):
        """Integration test with real PostgreSQL connection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "models"
            stats_path = Path(tmpdir) / "stats.json"

            config = AlphaZeroLearnerConfig(
                model_dir=str(model_dir),
                stats_path=str(stats_path),
                total_steps=10,
                max_wait=5.0,  # Short timeout for testing
                replay_selection=alphazero_selection(),
            )

            trainer = AlphaZeroLearner(config)
            assert trainer.network is not None


class TestStorageFactory:
    """Tests for the storage factory."""

    def test_factory_requires_postgres_url(self):
        """Test that factory raises error without PostgreSQL URL."""
        from trainer.storage import create_replay_store

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="PostgreSQL connection string required"):
                create_replay_store(alphazero_selection())

    @pytest.mark.skipif(
        not os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL"),
        reason="PostgreSQL not configured",
    )
    def test_factory_with_postgres(self):
        """Test factory creates PostgresReplayStore with valid URL."""
        from trainer.storage import PostgresReplayStore, create_replay_store

        store = create_replay_store(alphazero_selection())
        assert isinstance(store, PostgresReplayStore)
        assert store.selection == alphazero_selection()
        store.close()
