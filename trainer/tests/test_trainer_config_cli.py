import argparse

from trainer.trainer import TrainerConfig


def test_configure_parser_adds_expected_arguments():
    parser = argparse.ArgumentParser()
    TrainerConfig.configure_parser(parser)

    args = parser.parse_args([])
    config = TrainerConfig.from_args(args)

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
    TrainerConfig.configure_parser(parser)

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

    config = TrainerConfig.from_args(args)

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
