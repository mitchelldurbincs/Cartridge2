"""Tests for the AlphaZero learner implementation.

This module tests the AlphaZero learner including:
- Network initialization
- Checkpoint saving/loading
- Training step execution
- Evaluation metrics computation
"""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from trainer import step_metrics as step_metrics_module
from trainer.algorithms.alphazero_board_v1 import DESCRIPTOR
from trainer.config import AlphaZeroLearnerConfig
from trainer.environment_catalog import get_environment
from trainer.evaluator import EvalResults
from trainer.stats import EvalStats, TrainerStats
from trainer.storage.base import ReplayProfile, ReplaySelection
from trainer.storage.publisher import ArtifactValidationError, RunHeadV2
from trainer.trainer import AlphaZeroLearner as _AlphaZeroLearner


def make_learner(config: AlphaZeroLearnerConfig) -> _AlphaZeroLearner:
    """Supply the exact direct-train collection fence used by each test call."""
    head_path = Path(config.model_dir) / "channels" / "current.json"
    source_checkpoint_id = None
    if head_path.exists():
        source_checkpoint_id = RunHeadV2.from_bytes(
            head_path.read_bytes()
        ).checkpoint_id
    environment = get_environment(config.env_id)
    config.replay_selection = ReplaySelection(
        profile=ReplayProfile(
            env_id=config.env_id,
            env_contract_version=environment.contract_version,
            algorithm_id=DESCRIPTOR.id,
            experience_schema=DESCRIPTOR.components.experience_schema,
        ),
        collection_scope_id="a" * 64,
        source_checkpoint_id=source_checkpoint_id,
    )
    return _AlphaZeroLearner(config)


class TestAlphaZeroLearnerInitialization:
    """Test trainer initialization and configuration."""

    def test_direct_training_requires_an_explicit_replay_selection(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )

        with pytest.raises(ValueError, match="explicit ReplaySelection"):
            _AlphaZeroLearner(config)

    def test_trainer_creates_network(self, tmp_path):
        """Test that trainer initializes network correctly."""
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=100,
        )

        trainer = make_learner(config)

        assert trainer.network is not None
        assert trainer.optimizer is not None
        assert trainer.game_config.env_id == "tictactoe"

    def test_trainer_resolves_device_cpu(self, tmp_path):
        """Test that trainer resolves CPU device."""
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            device="cpu",
        )

        trainer = make_learner(config)

        assert trainer.device.type == "cpu"

    def test_trainer_creates_model_directory(self, tmp_path):
        """Test that trainer creates model directory on init."""
        model_dir = tmp_path / "models" / "nested"
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(model_dir),
            stats_path=str(tmp_path / "stats.json"),
        )

        assert not model_dir.exists()
        _ = make_learner(config)
        assert model_dir.exists()

    def test_fresh_run_replaces_stale_projection_when_run_head_is_absent(
        self, tmp_path
    ):
        stats_path = tmp_path / "stats.json"
        stats_path.write_text('{"step":999}', encoding="utf-8")
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(stats_path),
            total_steps=25,
        )

        make_learner(config)

        projection = json.loads(stats_path.read_bytes())
        assert projection["step"] == 0
        assert projection["total_steps"] == 25
        assert projection["env_id"] == "tictactoe"


class TestAlphaZeroLearnerCheckpoint:
    """Test checkpoint saving and loading functionality."""

    def test_checkpoint_save_and_load(self, tmp_path):
        """Test that checkpoints can be saved and loaded."""
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=100,
        )

        # Create initial trainer
        trainer1 = make_learner(config)
        initial_state = trainer1.network.state_dict()

        # Publish checkpoint and stats together at step 10.
        trainer1.stats.step = 10
        trainer1._publish_run_state(trainer1._save_checkpoint(10))

        # Create new trainer (should auto-load checkpoint)
        trainer2 = make_learner(config)

        # Verify checkpoint was loaded
        assert trainer2._checkpoint_loaded
        assert trainer2._loaded_step == 10
        assert trainer2.start_step == 10

        # Verify network state matches
        loaded_state = trainer2.network.state_dict()
        for key in initial_state:
            assert torch.allclose(initial_state[key], loaded_state[key])

    def test_replay_source_must_match_authoritative_head(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )
        source = make_learner(config)
        source.stats.step = 1
        source._publish_run_state(source._save_checkpoint(1))
        assert config.replay_selection is not None
        config.replay_selection = replace(
            config.replay_selection,
            source_checkpoint_id="f" * 64,
        )

        with pytest.raises(ArtifactValidationError, match="source checkpoint"):
            _AlphaZeroLearner(config)

    def test_checkpoint_records_learner_artifact_identity(self, tmp_path):
        """Learner checkpoints declare their exact algorithm, model, and env."""
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )
        trainer = make_learner(config)

        checkpoint_ref = trainer._save_checkpoint(10)

        checkpoint = torch.load(checkpoint_ref.learner_state_path, weights_only=True)
        assert checkpoint["schema_version"] == 1
        assert checkpoint["profile"] == {
            "algorithm_id": "alphazero_board_v1",
            "env_id": "tictactoe",
            "env_contract_version": 1,
            "model_artifact_schema_version": 1,
            "model_contract": "onnx_policy_value_v1",
        }
        assert checkpoint_ref.manifest.config_sha256 == trainer.config_sha256

    def test_explicit_start_step_must_match_checkpoint(self, tmp_path):
        model_dir = tmp_path / "models"
        source = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(model_dir),
            stats_path=str(tmp_path / "stats.json"),
        )
        source_trainer = make_learner(source)
        source_trainer.stats.step = 10
        source_trainer._publish_run_state(source_trainer._save_checkpoint(10))

        mismatched = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(model_dir),
            stats_path=str(tmp_path / "stats.json"),
            start_step=5,
        )
        with pytest.raises(ValueError, match="start_step.*checkpoint"):
            make_learner(mismatched)

    def test_mismatched_checkpoint_aborts_learner_initialization(self, tmp_path):
        """A learner cannot silently resume an artifact from another env."""
        model_dir = tmp_path / "models"
        source = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(model_dir),
            stats_path=str(tmp_path / "stats.json"),
        )
        source_trainer = make_learner(source)
        source_trainer.stats.step = 10
        source_trainer._publish_run_state(source_trainer._save_checkpoint(10))

        target = AlphaZeroLearnerConfig(
            env_id="connect4",
            model_dir=str(model_dir),
            stats_path=str(tmp_path / "stats.json"),
        )
        with pytest.raises(ValueError, match="profile mismatch"):
            make_learner(target)

    def test_corrupt_checkpoint_aborts_learner_initialization(self, tmp_path):
        """A corrupt resume artifact must not be treated as no checkpoint."""
        model_dir = tmp_path / "models"
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(model_dir),
            stats_path=str(tmp_path / "stats.json"),
        )
        source = make_learner(config)
        source.stats.step = 10
        source._publish_run_state(source._save_checkpoint(10))
        checkpoint = source.last_checkpoint_ref
        assert checkpoint is not None
        checkpoint.learner_state_path.write_bytes(b"not a valid checkpoint")

        with pytest.raises(ArtifactValidationError, match="size mismatch"):
            make_learner(config)

    def test_checkpoint_not_found(self, tmp_path):
        """Test behavior when no checkpoint exists."""
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )

        trainer = make_learner(config)

        assert not trainer._checkpoint_loaded
        assert trainer._loaded_step is None

    def test_checkpoints_are_immutable_and_never_rotated(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )

        trainer = make_learner(config)

        # Create and select multiple immutable checkpoints.
        for step in [10, 20, 30, 40]:
            trainer.stats.step = step
            checkpoint = trainer._save_checkpoint(step)
            trainer._publish_run_state(checkpoint)

        manifests = list((tmp_path / "models" / "manifests" / "sha256").glob("*.json"))
        assert len(manifests) == 4
        head = RunHeadV2.from_bytes(
            (tmp_path / "models" / "channels" / "current.json").read_bytes()
        )
        assert head.checkpoint_id == trainer.parent_checkpoint_id

    def test_same_training_step_reuses_published_checkpoint(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )
        trainer = make_learner(config)

        first = trainer._save_checkpoint(10)
        second = trainer._save_checkpoint(10)

        assert second == first
        assert (
            len(list((tmp_path / "models" / "manifests" / "sha256").glob("*.json")))
            == 1
        )

    def test_identical_checkpoint_stats_publish_is_run_commit_idempotent(
        self, tmp_path
    ):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )
        trainer = make_learner(config)
        trainer.stats.step = 10
        checkpoint = trainer._save_checkpoint(10)

        first = trainer._publish_run_state(checkpoint)
        second = trainer._publish_run_state(checkpoint)

        assert second.run_commit_id == first.run_commit_id
        assert len(list((tmp_path / "models/run-commits/sha256").glob("*.json"))) == 1

    def test_resume_restores_cumulative_counters(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=100,
        )
        source = make_learner(config)
        source.stats.step = 10
        source.stats.samples_seen = 123
        source.stats.replay_record_count = 456
        source._publish_run_state(source._save_checkpoint(10))

        resumed = make_learner(config)

        assert resumed.start_step == 10
        assert resumed.samples_seen == 123
        assert resumed.stats.samples_seen == 123
        assert resumed._replay_record_count_cache == 456
        assert resumed.stats.replay_record_count == 456
        assert resumed.stats.total_steps == 110

    def test_resume_publishes_a_child_of_the_immutable_parent_commit(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=100,
        )
        source = make_learner(config)
        source.stats.step = 10
        first = source._publish_run_state(source._save_checkpoint(10))

        resumed = make_learner(config)
        original_parent_bytes = first.to_bytes()
        resumed.stats.step = 20
        resumed.stats.samples_seen += 64
        second = resumed._publish_run_state(resumed._save_checkpoint(20))

        assert first.to_bytes() == original_parent_bytes
        assert second.parent_run_commit_id == first.run_commit_id
        head = resumed.checkpoint_publisher.resolve_run_head()
        assert head is not None
        assert head.run_commit_id == second.run_commit_id
        assert [
            reference.run_commit_id
            for reference in resumed.run_commit_repository.resolve_chain(
                head.run_commit_id
            )
        ] == [
            first.run_commit_id,
            second.run_commit_id,
        ]

    def test_resume_rejects_changed_learner_config(self, tmp_path):
        model_dir = tmp_path / "models"
        stats_path = tmp_path / "stats.json"
        source = make_learner(
            AlphaZeroLearnerConfig(
                env_id="tictactoe",
                model_dir=str(model_dir),
                stats_path=str(stats_path),
            )
        )
        source.stats.step = 10
        source._publish_run_state(source._save_checkpoint(10))

        with pytest.raises(ArtifactValidationError, match="config mismatch"):
            make_learner(
                AlphaZeroLearnerConfig(
                    env_id="tictactoe",
                    model_dir=str(model_dir),
                    stats_path=str(stats_path),
                    learning_rate=0.02,
                )
            )

    @pytest.mark.parametrize("missing_half", ["run_commit", "checkpoint"])
    def test_resume_rejects_missing_authoritative_half(self, tmp_path, missing_half):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )
        source = make_learner(config)
        source.stats.step = 10
        source._publish_run_state(source._save_checkpoint(10))
        head = source.checkpoint_publisher.resolve_run_head()
        assert head is not None
        if missing_half == "run_commit":
            target = tmp_path / f"models/run-commits/sha256/{head.run_commit_id}.json"
        else:
            target = tmp_path / f"models/manifests/sha256/{head.checkpoint_id}.json"
        target.unlink()

        with pytest.raises(ArtifactValidationError, match="does not exist"):
            make_learner(config)


class TestAlphaZeroLearnerTrainingStep:
    """Test training step execution."""

    def test_zero_steps_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="total_steps"):
            AlphaZeroLearnerConfig(
                env_id="tictactoe",
                model_dir=str(tmp_path / "models"),
                stats_path=str(tmp_path / "stats.json"),
                total_steps=0,
            )

    def test_train_requests_the_manifest_observation_width(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=1,
            batch_size=2,
        )
        learner = make_learner(config)
        replay = MagicMock()
        observation = np.zeros(learner.game_config.obs_size, dtype="<f4")
        policy = np.zeros(learner.game_config.num_actions, dtype="<f4")
        policy[0] = 1.0
        payload = (
            np.concatenate((observation, policy, np.asarray([0.0], dtype="<f4")))
            .astype("<f4")
            .tobytes()
        )
        replay.sample.return_value = [
            learner.replay_selection.record(
                id=f"record-{index}",
                episode_id="episode",
                step_number=index,
                payload=payload,
            )
            for index in range(2)
        ]
        learner._create_replay_store = MagicMock(return_value=replay)
        learner._setup_replay = MagicMock()
        learner._train_step = MagicMock(return_value={})
        learner._record_step_metrics = MagicMock()
        learner._handle_replay_cleanup = MagicMock()
        learner._handle_checkpoint = MagicMock()
        learner._save_checkpoint = MagicMock()
        learner._publish_run_state = MagicMock()

        learner.train()

        replay.sample.assert_called_once_with(2)
        observations, policy_targets, value_targets = learner._train_step.call_args.args
        assert observations.shape == (2, learner.game_config.obs_size)
        assert policy_targets.shape == (2, learner.game_config.num_actions)
        assert value_targets.shape == (2,)
        replay.close.assert_called_once_with()

    def test_train_fails_on_a_short_replay_batch_without_retrying(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=1,
            batch_size=2,
        )
        learner = make_learner(config)
        replay = MagicMock()
        replay.sample.return_value = []
        learner._create_replay_store = MagicMock(return_value=replay)
        learner._setup_replay = MagicMock()

        with pytest.raises(RuntimeError, match="ReplayStore.sample contract violation"):
            learner.train()

        replay.sample.assert_called_once_with(2)
        replay.close.assert_called_once_with()

    def test_final_stats_are_bound_to_the_final_published_checkpoint(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=1,
            batch_size=1,
            stats_interval=10,
            checkpoint_interval=100,
        )
        learner = make_learner(config)
        observation = np.zeros(learner.game_config.obs_size, dtype="<f4")
        policy = np.zeros(learner.game_config.num_actions, dtype="<f4")
        policy[0] = 1.0
        payload = (
            np.concatenate((observation, policy, np.asarray([0.0], dtype="<f4")))
            .astype("<f4")
            .tobytes()
        )
        replay = MagicMock()
        replay.sample.return_value = [
            learner.replay_selection.record(
                id="record-0",
                episode_id="episode",
                step_number=0,
                payload=payload,
            )
        ]
        learner._create_replay_store = MagicMock(return_value=replay)
        learner._setup_replay = MagicMock()
        learner._train_step = MagicMock(
            return_value={
                "loss/total": 1.5,
                "loss/value": 0.5,
                "loss/policy": 1.0,
            }
        )

        result = learner.train()

        head = learner.checkpoint_publisher.resolve_run_head()
        assert head is not None
        assert learner.current_run_commit is not None
        assert learner.last_prepared_stats is not None
        assert head.checkpoint_id == result.last_checkpoint
        assert head.run_commit_id == learner.current_run_commit.run_commit_id
        resolved = learner.run_commit_repository.resolve(head.run_commit_id).commit
        assert resolved.stats_snapshot.binding.checkpoint_id == head.checkpoint_id
        assert resolved.stats_snapshot.binding.step == 1
        assert resolved.stats_snapshot.stats.step == 1
        assert resolved.stats_snapshot.stats.samples_seen == 1
        replay.close.assert_called_once_with()

    def test_deferred_loop_stages_only_one_final_checkpoint(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            checkpoint_interval=1,
            defer_run_commit=True,
        )
        learner = make_learner(config)

        learner.stats.step = 1
        learner._handle_checkpoint(step=1, global_step=1)
        learner.stats.step = 2
        learner._handle_checkpoint(step=2, global_step=2)

        assert not (tmp_path / "models/manifests").exists()
        final_checkpoint = learner._save_checkpoint(2)
        learner.stats.last_checkpoint = final_checkpoint.checkpoint_id
        prepared = learner._prepare_run_state(final_checkpoint)
        assert final_checkpoint.manifest.parent_checkpoint_id is None
        assert prepared.binding.checkpoint_id == final_checkpoint.checkpoint_id
        assert len(list((tmp_path / "models/manifests/sha256").glob("*.json"))) == 1
        assert not (tmp_path / "models/channels/current.json").exists()

    def test_train_step_updates_weights(self, tmp_path):
        """Test that training step updates network weights."""
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=100,
            batch_size=4,
        )

        trainer = make_learner(config)

        # Get initial weights - network uses policy_fc not policy_head
        initial_weights = trainer.network.state_dict()["policy_fc.weight"].clone()

        # Create mock batch data as numpy arrays (new API)
        batch_size = 4
        obs_size = trainer.game_config.obs_size
        num_actions = trainer.game_config.num_actions

        observations = np.random.randn(batch_size, obs_size).astype(np.float32)
        policy_targets = np.random.randn(batch_size, num_actions).astype(np.float32)
        # Softmax normalization for policy targets
        policy_targets = np.exp(policy_targets) / np.exp(policy_targets).sum(
            axis=1, keepdims=True
        )
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
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            batch_size=4,
        )

        trainer = make_learner(config)

        batch_size = 4
        obs_size = trainer.game_config.obs_size
        num_actions = trainer.game_config.num_actions

        # Create batch data as numpy arrays
        observations = np.random.randn(batch_size, obs_size).astype(np.float32)
        policy_targets = np.random.randn(batch_size, num_actions).astype(np.float32)
        policy_targets = np.exp(policy_targets) / np.exp(policy_targets).sum(
            axis=1, keepdims=True
        )
        value_targets = np.random.randn(batch_size).astype(np.float32)

        # Should not raise error (mask is extracted from observations internally)
        metrics = trainer._train_step(observations, policy_targets, value_targets)

        assert not np.isnan(metrics["loss/total"])


class TestAlphaZeroLearnerEvaluation:
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


class TestAlphaZeroLearnerStats:
    """Test training statistics tracking."""

    def test_update_training_stats(self, tmp_path):
        """Test that training stats are updated correctly."""
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            total_steps=100,
        )

        trainer = make_learner(config)

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
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )

        trainer = make_learner(config)
        checkpoint = trainer._save_checkpoint(step=10)

        assert checkpoint.onnx_path.is_file()
        assert checkpoint.onnx_path.stat().st_size > 0
        assert checkpoint.onnx_path.name == f"{checkpoint.manifest.onnx.sha256}.onnx"
        assert trainer.parent_checkpoint_id is None
        assert not (tmp_path / "models" / "channels" / "current.json").exists()

    def test_checkpoint_publication_failure_is_fatal(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )
        trainer = make_learner(config)
        previous_parent = trainer.parent_checkpoint_id
        trainer.checkpoint_publisher = MagicMock()
        trainer.checkpoint_publisher.stage_checkpoint.side_effect = RuntimeError(
            "upload failed"
        )

        with pytest.raises(RuntimeError, match="upload failed"):
            trainer._save_checkpoint(step=10)
        assert trainer.parent_checkpoint_id == previous_parent

    def test_stats_tick_updates_projection_without_staging_checkpoint(self, tmp_path):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            stats_interval=10,
            checkpoint_interval=100,
        )
        trainer = make_learner(config)
        trainer._save_checkpoint = MagicMock()

        trainer._record_step_metrics(
            step=10,
            global_step=10,
            metrics={
                "loss/total": 1.5,
                "loss/value": 0.5,
                "loss/policy": 1.0,
            },
            step_duration=0.01,
            batch_size=8,
            replay=MagicMock(),
            env_id="tictactoe",
        )

        trainer._save_checkpoint.assert_not_called()
        assert (tmp_path / "stats.json").is_file()
        assert not (tmp_path / "models/manifests").exists()
        assert not (tmp_path / "models/channels/current.json").exists()

    def test_step_metrics_survive_wall_clock_rollback(self, tmp_path, monkeypatch):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            stats_interval=100,
        )
        trainer = make_learner(config)
        trainer.stats.timestamp = 100.0
        monkeypatch.setattr(step_metrics_module.time, "time", lambda: 50.0)

        trainer._record_step_metrics(
            step=1,
            global_step=1,
            metrics={
                "loss/total": 1.5,
                "loss/value": 0.5,
                "loss/policy": 1.0,
            },
            step_duration=0.01,
            batch_size=8,
            replay=MagicMock(),
            env_id="tictactoe",
        )

        assert trainer.stats.timestamp == 100.0

    def test_overlapping_stats_and_checkpoint_ticks_create_one_run_commit(
        self, tmp_path
    ):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            stats_interval=10,
            checkpoint_interval=10,
        )
        trainer = make_learner(config)
        metrics = {
            "loss/total": 1.5,
            "loss/value": 0.5,
            "loss/policy": 1.0,
        }

        trainer._record_step_metrics(
            step=10,
            global_step=10,
            metrics=metrics,
            step_duration=0.01,
            batch_size=8,
            replay=MagicMock(),
            env_id="tictactoe",
        )
        trainer._handle_checkpoint(step=10, global_step=10)
        assert trainer.last_checkpoint_ref is not None
        trainer._publish_run_state(trainer.last_checkpoint_ref)

        assert len(list((tmp_path / "models/run-commits/sha256").glob("*.json"))) == 1
        assert len(list((tmp_path / "models/manifests/sha256").glob("*.json"))) == 1


class TestAlphaZeroLearnerGameConfigs:
    """Test trainer with different game configurations."""

    def test_trainer_tictactoe_config(self, tmp_path):
        """Test trainer initialization with TicTacToe."""
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )

        trainer = make_learner(config)

        assert trainer.game_config.env_id == "tictactoe"
        assert trainer.game_config.board_width == 3
        assert trainer.game_config.board_height == 3
        assert trainer.game_config.num_actions == 9

    def test_trainer_connect4_config(self, tmp_path):
        """Test trainer initialization with Connect4."""
        config = AlphaZeroLearnerConfig(
            env_id="connect4",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
        )

        trainer = make_learner(config)

        assert trainer.game_config.env_id == "connect4"
        assert trainer.game_config.board_width == 7
        assert trainer.game_config.board_height == 6
        assert trainer.game_config.num_actions == 7


class TestMetricsHook:
    """Test the per-step metrics_hook callback."""

    def _make_learner(self, tmp_path, hook):
        config = AlphaZeroLearnerConfig(
            env_id="tictactoe",
            model_dir=str(tmp_path / "models"),
            stats_path=str(tmp_path / "stats.json"),
            stats_interval=10,
            metrics_hook=hook,
        )
        return make_learner(config)

    def _step_metrics(self):
        return {"loss/total": 1.5, "loss/value": 0.5, "loss/policy": 1.0}

    def test_hook_called_at_stats_interval(self, tmp_path):
        calls = []
        trainer = self._make_learner(
            tmp_path, lambda payload, step: calls.append((payload, step))
        )

        trainer._record_step_metrics(
            step=10,
            global_step=410,
            metrics=self._step_metrics(),
            step_duration=0.01,
            batch_size=8,
            replay=MagicMock(),
            env_id="tictactoe",
        )

        assert len(calls) == 1
        payload, step = calls[0]
        assert step == 410
        assert payload["total_loss"] == 1.5
        assert payload["value_loss"] == 0.5
        assert payload["policy_loss"] == 1.0
        assert "learning_rate" in payload
        assert "samples_seen" in payload

    def test_hook_not_called_off_interval(self, tmp_path):
        calls = []
        trainer = self._make_learner(tmp_path, lambda payload, step: calls.append(step))

        trainer._record_step_metrics(
            step=7,  # not a multiple of stats_interval=10
            global_step=407,
            metrics=self._step_metrics(),
            step_duration=0.01,
            batch_size=8,
            replay=MagicMock(),
            env_id="tictactoe",
        )

        assert calls == []

    def test_raising_hook_does_not_propagate(self, tmp_path):
        def bad_hook(payload, step):
            raise RuntimeError("hook exploded")

        trainer = self._make_learner(tmp_path, bad_hook)

        # Must not raise
        trainer._record_step_metrics(
            step=10,
            global_step=410,
            metrics=self._step_metrics(),
            step_duration=0.01,
            batch_size=8,
            replay=MagicMock(),
            env_id="tictactoe",
        )
