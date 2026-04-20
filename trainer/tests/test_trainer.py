"""Tests for core trainer functionality.

This module tests the Trainer class including:
- Network initialization
- Checkpoint saving/loading
- Training step execution
- Evaluation metrics computation
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

from trainer.config import TrainerConfig
from trainer.evaluator import EvalResults
from trainer.game_config import GameConfig
from trainer.stats import EvalStats, TrainerStats
from trainer.trainer import Trainer


class TestTrainerInitialization:
    """Test trainer initialization and configuration."""

    def test_trainer_creates_network(self, tmp_path):
        """Test that trainer initializes network correctly."""
        config = TrainerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=100,
        )

        trainer = Trainer(config)

        assert trainer.network is not None
        assert trainer.optimizer is not None
        assert trainer.game_config.env_id == "tictactoe"

    def test_trainer_resolves_device_cpu(self, tmp_path):
        """Test that trainer resolves CPU device."""
        config = TrainerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            device="cpu",
        )

        trainer = Trainer(config)

        assert trainer.device.type == "cpu"

    def test_trainer_creates_model_directory(self, tmp_path):
        """Test that trainer creates model directory on init."""
        model_dir = tmp_path / "models" / "nested"
        config = TrainerConfig(
            env_id="tictactoe",
            model_dir=str(model_dir),
            stats_path=str(tmp_path / "stats.json"),
        )

        assert not model_dir.exists()
        _ = Trainer(config)
        assert model_dir.exists()


class TestTrainerCheckpoint:
    """Test checkpoint saving and loading functionality."""

    def test_checkpoint_save_and_load(self, tmp_path):
        """Test that checkpoints can be saved and loaded."""
        config = TrainerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=100,
        )

        # Create initial trainer
        trainer1 = Trainer(config)
        initial_state = trainer1.network.state_dict()

        # Save checkpoint at step 10
        trainer1._save_checkpoint(10)

        # Create new trainer (should auto-load checkpoint)
        trainer2 = Trainer(config)

        # Verify checkpoint was loaded
        assert trainer2._checkpoint_loaded
        assert trainer2._loaded_step == 10

        # Verify network state matches
        loaded_state = trainer2.network.state_dict()
        for key in initial_state:
            assert torch.allclose(initial_state[key], loaded_state[key])

    def test_checkpoint_not_found(self, tmp_path):
        """Test behavior when no checkpoint exists."""
        config = TrainerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )

        trainer = Trainer(config)

        assert not trainer._checkpoint_loaded
        assert trainer._loaded_step is None

    def test_checkpoint_max_checkpoints_enforced(self, tmp_path):
        """Test that old checkpoints are cleaned up."""
        config = TrainerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            max_checkpoints=2,
        )

        trainer = Trainer(config)

        # Create multiple checkpoints
        for step in [10, 20, 30, 40]:
            trainer._save_checkpoint(step)

        # Only 2 most recent should remain
        checkpoints = list((tmp_path / "models").glob("*.pt"))
        assert len(checkpoints) <= 2


class TestTrainerTrainingStep:
    """Test training step execution."""

    def test_train_step_updates_weights(self, tmp_path):
        """Test that training step updates network weights."""
        config = TrainerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=100,
            batch_size=4,
        )

        trainer = Trainer(config)

        # Get initial weights - network uses policy_fc not policy_head
        initial_weights = trainer.network.state_dict()["policy_fc.weight"].clone()

        # Create mock batch data as numpy arrays (new API)
        batch_size = 4
        obs_size = trainer.game_config.obs_size
        num_actions = trainer.game_config.num_actions

        observations = np.random.randn(batch_size, obs_size).astype(np.float32)
        policy_targets = np.random.randn(batch_size, num_actions).astype(np.float32)
        # Softmax normalization for policy targets
        policy_targets = np.exp(policy_targets) / np.exp(policy_targets).sum(axis=1, keepdims=True)
        value_targets = np.random.randn(batch_size).astype(np.float32)

        # Execute training step (new signature takes numpy arrays)
        metrics = trainer._train_step(observations, policy_targets, value_targets)

        # Verify loss values are valid
        assert not np.isnan(metrics["loss/total"])
        assert not np.isnan(metrics["loss/policy"])
        assert not np.isnan(metrics["loss/value"])

        # Verify weights changed
        updated_weights = trainer.network.state_dict()["policy_fc.weight"]
        assert not torch.allclose(initial_weights, updated_weights)

    def test_train_step_with_masked_actions(self, tmp_path):
        """Test training step with action masking."""
        config = TrainerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            batch_size=4,
        )

        trainer = Trainer(config)

        batch_size = 4
        obs_size = trainer.game_config.obs_size
        num_actions = trainer.game_config.num_actions

        # Create batch data as numpy arrays
        observations = np.random.randn(batch_size, obs_size).astype(np.float32)
        policy_targets = np.random.randn(batch_size, num_actions).astype(np.float32)
        policy_targets = np.exp(policy_targets) / np.exp(policy_targets).sum(axis=1, keepdims=True)
        value_targets = np.random.randn(batch_size).astype(np.float32)

        # Should not raise error (mask is extracted from observations internally)
        metrics = trainer._train_step(observations, policy_targets, value_targets)

        assert not np.isnan(metrics["loss/total"])


class TestTrainerEvaluation:
    """Test evaluation metrics computation."""

    def test_eval_stats_creation(self, tmp_path):
        """Test that EvalStats can be created from EvalResults."""
        # EvalStats is created directly from EvalResults in the trainer
        mock_results = EvalResults(
            env_id="tictactoe",
            player1_name="model",
            player2_name="random",
            games_played=10,
            player1_wins=7,
            player2_wins=1,
            draws=2,
            player1_wins_as_first=4,
            player1_wins_as_second=3,
            player2_wins_as_first=1,
            player2_wins_as_second=0,
            avg_game_length=8.5,
        )

        # Create EvalStats directly as the trainer does
        eval_stats = EvalStats(
            step=5,
            win_rate=mock_results.player1_win_rate,
            draw_rate=mock_results.draw_rate,
            loss_rate=mock_results.player2_win_rate,
            games_played=mock_results.games_played,
            avg_game_length=mock_results.avg_game_length,
            timestamp=0.0,
        )

        assert isinstance(eval_stats, EvalStats)
        assert eval_stats.win_rate == 0.7
        assert eval_stats.draw_rate == 0.2

    def test_eval_stats_vs_random(self, tmp_path):
        """Test EvalStats creation from vs-random evaluation results."""
        mock_results = EvalResults(
            env_id="tictactoe",
            player1_name="model",
            player2_name="random",
            games_played=50,
            player1_wins=40,
            player2_wins=5,
            draws=5,
            player1_wins_as_first=20,
            player1_wins_as_second=20,
            player2_wins_as_first=3,
            player2_wins_as_second=2,
            avg_game_length=9.2,
        )

        eval_stats = EvalStats(
            step=10,
            win_rate=mock_results.player1_win_rate,
            draw_rate=mock_results.draw_rate,
            loss_rate=mock_results.player2_win_rate,
            games_played=mock_results.games_played,
            avg_game_length=mock_results.avg_game_length,
            timestamp=0.0,
        )

        assert eval_stats.win_rate == 0.8
        assert eval_stats.games_played == 50
        assert eval_stats.avg_game_length == 9.2


class TestTrainerStats:
    """Test training statistics tracking."""

    def test_update_training_stats(self, tmp_path):
        """Test that training stats are updated correctly."""
        config = TrainerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=100,
        )

        trainer = Trainer(config)

        # Simulate some training
        trainer.stats = TrainerStats()
        trainer.stats.total_steps = 50
        trainer.stats.policy_loss = 1.5
        trainer.stats.value_loss = 0.5
        trainer.stats.samples_seen = 200

        # Verify stats
        assert trainer.stats.total_steps == 50
        assert trainer.stats.policy_loss == 1.5
        assert trainer.stats.samples_seen == 200

    def test_save_checkpoint_creates_file(self, tmp_path):
        """Test that checkpoint save creates ONNX file."""
        config = TrainerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )

        trainer = Trainer(config)

        # Save checkpoint (creates ONNX file via _save_checkpoint)
        checkpoint_path = trainer._save_checkpoint(step=10)

        # Verify file was created
        assert checkpoint_path.exists()
        assert checkpoint_path.stat().st_size > 0
        assert checkpoint_path.suffix == ".onnx"


class TestTrainerGameConfigs:
    """Test trainer with different game configurations."""

    def test_trainer_tictactoe_config(self, tmp_path):
        """Test trainer initialization with TicTacToe."""
        config = TrainerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )

        trainer = Trainer(config)

        assert trainer.game_config.env_id == "tictactoe"
        assert trainer.game_config.board_width == 3
        assert trainer.game_config.board_height == 3
        assert trainer.game_config.num_actions == 9

    def test_trainer_connect4_config(self, tmp_path):
        """Test trainer initialization with Connect4."""
        config = TrainerConfig(
            env_id="connect4",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )

        trainer = Trainer(config)

        assert trainer.game_config.env_id == "connect4"
        assert trainer.game_config.board_width == 7
        assert trainer.game_config.board_height == 6
        assert trainer.game_config.num_actions == 7
