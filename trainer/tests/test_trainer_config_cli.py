import argparse

import pytest

from trainer.config import AlphaZeroLearnerConfig


def test_configure_parser_adds_expected_arguments():
    parser = argparse.ArgumentParser()
    AlphaZeroLearnerConfig.configure_parser(parser)

    args = parser.parse_args([])
    config = AlphaZeroLearnerConfig.from_args(args)

    assert config.model_dir == "./data/models"
    assert config.stats_path == "./data/stats.json"
    assert config.batch_size == 64
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.grad_clip_norm == 1.0
    assert config.use_lr_scheduler is True
    assert config.total_steps == 1000
    assert config.device == "auto"


def test_from_args_overrides_defaults_and_actions():
    parser = argparse.ArgumentParser()
    AlphaZeroLearnerConfig.configure_parser(parser)

    args = parser.parse_args(
        [
            "--model-dir",
            "./checkpoints",
            "--stats",
            "./stats/out.json",
            "--batch-size",
            "128",
            "--lr",
            "0.01",
            "--weight-decay",
            "0.001",
            "--grad-clip",
            "2.5",
            "--no-lr-schedule",
            "--steps",
            "42",
            "--env-id",
            "connect4",
            "--device",
            "cuda",
        ]
    )

    config = AlphaZeroLearnerConfig.from_args(args)

    assert config.model_dir == "./checkpoints"
    assert config.stats_path == "./stats/out.json"
    assert config.batch_size == 128
    assert config.learning_rate == 0.01
    assert config.weight_decay == 0.001
    assert config.grad_clip_norm == 2.5
    assert config.use_lr_scheduler is False
    assert config.total_steps == 42
    assert config.env_id == "connect4"
    assert config.device == "cuda"


def test_removed_short_batch_retry_flag_is_rejected():
    parser = argparse.ArgumentParser()
    AlphaZeroLearnerConfig.configure_parser(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--max-empty-batches", "10"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": 1.0}, "batch_size"),
        ({"total_steps": 0}, "total_steps"),
        ({"checkpoint_interval": 0}, "checkpoint_interval"),
        ({"stats_interval": 0}, "stats_interval"),
        ({"log_interval": 0}, "log_interval"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"value_loss_weight": float("inf")}, "value_loss_weight"),
        ({"grad_clip_norm": -1.0}, "grad_clip_norm"),
        ({"lr_min_ratio": 1.01}, "lr_min_ratio"),
        ({"lr_warmup_start_ratio": -0.01}, "lr_warmup_start_ratio"),
        ({"wait_interval": 0.0}, "wait_interval"),
        ({"max_wait": -1.0}, "max_wait"),
        (
            {"replay_window": 0, "replay_cleanup_interval": 10},
            "replay_cleanup_interval",
        ),
        ({"use_lr_scheduler": 1}, "use_lr_scheduler"),
        ({"device": "tpu"}, "device"),
    ],
)
def test_invalid_learner_configuration_is_rejected_immediately(overrides, message):
    with pytest.raises(ValueError, match=message):
        AlphaZeroLearnerConfig(**overrides)


def test_both_loss_weights_cannot_be_disabled():
    with pytest.raises(ValueError, match="cannot both be zero"):
        AlphaZeroLearnerConfig(value_loss_weight=0.0, policy_loss_weight=0.0)


def test_replay_cleanup_cadence_is_valid_when_window_is_enabled():
    config = AlphaZeroLearnerConfig(replay_window=100, replay_cleanup_interval=0)

    assert config.learner_recipe()["replay_cleanup_cadence"] == config.stats_interval
