"""Trainer configuration and CLI argument helpers.

This module provides:
- TrainerConfig dataclass with all training parameters
- CLI field metadata for automatic argparse integration
- Methods to build and parse CLI arguments
"""

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Callable

from .backoff import DEFAULT_MAX_WAIT, DEFAULT_WAIT_INTERVAL

__all__ = ["TrainerConfig", "cli_field"]


def cli_field(
    default: Any,
    *,
    cli: str | None = None,
    help: str = "",
    choices: list[Any] | None = None,
    action: str | None = None,
) -> Any:
    """Create a dataclass field with CLI metadata.

    Args:
        default: Default value for the field.
        cli: CLI flag (e.g., "--batch-size"). If None, field is not exposed to CLI.
        help: Help text for the CLI argument.
        choices: Valid choices for the argument.
        action: argparse action (e.g., "store_true", "store_false").

    Returns:
        A dataclass field with CLI metadata attached.
    """
    metadata: dict[str, object] = {}
    if cli is not None:
        metadata["cli"] = cli
        metadata["help"] = help
        if choices:
            metadata["choices"] = choices
        if action:
            metadata["action"] = action

    return field(default=default, metadata=metadata)


@dataclass
class TrainerConfig:
    """Configuration for the trainer.

    Note: Replay buffer connection is configured via CARTRIDGE_STORAGE_POSTGRES_URL
    environment variable, not a config field.
    """

    model_dir: str = cli_field(
        "./data/models", cli="--model-dir", help="Directory for ONNX model checkpoints"
    )
    stats_path: str = cli_field(
        "./data/stats.json",
        cli="--stats",
        help="Path to write stats.json for web polling",
    )

    # Training hyperparameters
    batch_size: int = cli_field(64, cli="--batch-size", help="Batch size for training")
    learning_rate: float = cli_field(1e-3, cli="--lr", help="Learning rate")
    weight_decay: float = cli_field(1e-4, cli="--weight-decay", help="Weight decay")
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0

    # Gradient clipping (0 = disabled)
    grad_clip_norm: float = cli_field(
        1.0, cli="--grad-clip", help="Gradient clipping max norm (0 to disable)"
    )

    # Learning rate schedule
    use_lr_scheduler: bool = cli_field(
        True,
        cli="--no-lr-schedule",
        action="store_false",
        help="Disable cosine annealing LR scheduler",
    )
    lr_min_ratio: float = cli_field(
        0.1, cli="--lr-min-ratio", help="Final LR as ratio of initial LR"
    )
    lr_warmup_steps: int = cli_field(
        100,
        cli="--lr-warmup-steps",
        help="Number of warmup steps at start of training (0 to disable)",
    )
    lr_warmup_start_ratio: float = cli_field(
        0.1,
        cli="--lr-warmup-start-ratio",
        help="Starting LR as ratio of target LR during warmup",
    )
    lr_total_steps: int = cli_field(
        0,
        cli="--lr-total-steps",
        help="Total steps for LR schedule (0 = use total_steps, for continuous decay)",
    )

    # Training schedule
    total_steps: int = cli_field(1000, cli="--steps", help="Total training steps")
    checkpoint_interval: int = cli_field(
        100, cli="--checkpoint-interval", help="Steps between checkpoint saves"
    )
    stats_interval: int = cli_field(
        10, cli="--stats-interval", help="Steps between stats updates"
    )
    log_interval: int = cli_field(
        10, cli="--log-interval", help="Steps between log messages"
    )

    # Checkpoint management
    max_checkpoints: int = cli_field(
        10, cli="--max-checkpoints", help="Maximum number of checkpoints to keep"
    )

    # Wait/backoff settings
    wait_interval: float = cli_field(
        DEFAULT_WAIT_INTERVAL,
        cli="--wait-interval",
        help="Seconds between checks when waiting for data",
    )
    max_wait: float = cli_field(
        DEFAULT_MAX_WAIT,
        cli="--max-wait",
        help="Max seconds to wait for DB/data (0 = wait forever)",
    )

    # Replay buffer management
    clear_replay_on_start: bool = cli_field(
        False,
        cli="--clear-replay",
        action="store_true",
        help="Delete all transitions before training (synchronized AlphaZero)",
    )
    replay_window: int = cli_field(
        0,
        cli="--replay-window",
        help="Keep only the most recent N transitions (0 = disable cleanup)",
    )
    replay_cleanup_interval: int = cli_field(
        0,
        cli="--replay-cleanup-interval",
        help=(
            "Steps between replay cleanup when replay-window is set ("
            "0 = align with stats-interval)"
        ),
    )

    # Step offset for continuous training (checkpoint naming)
    start_step: int = cli_field(
        0, cli="--start-step", help="Starting step number for checkpoint naming"
    )

    # Evaluation settings
    eval_interval: int = cli_field(
        100, cli="--eval-interval", help="Steps between evaluations (0 to disable)"
    )
    eval_games: int = cli_field(
        50, cli="--eval-games", help="Number of games per evaluation"
    )

    # Stats history settings
    # Should be large enough to hold recent entries (1 per stats_interval steps)
    # plus downsampled entries from older iterations (~10 per 1000 steps)
    max_history_length: int = 1000

    # Environment
    env_id: str = cli_field("tictactoe", cli="--env-id", help="Environment ID")
    device: str = cli_field(
        "auto",
        cli="--device",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to train on (auto = detect best available: cuda > mps > cpu)",
    )

    # Shutdown callback (not exposed to CLI, set programmatically)
    # Returns True if shutdown was requested
    shutdown_check: Callable[[], bool] | None = field(default=None, repr=False)

    def resolve_device(self) -> str:
        """Resolve 'auto' device to the best available: cuda > mps > cpu."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @classmethod
    def configure_parser(
        cls, parser: Any, overrides: dict[str, Any] | None = None
    ) -> None:
        """Add CLI arguments to parser based on field metadata.

        Args:
            parser: argparse.ArgumentParser instance to configure.
            overrides: Optional dict mapping field names to override default values.
                       Use this to inject defaults from central config.
        """
        overrides = overrides or {}

        for f in fields(cls):
            cli_flag = f.metadata.get("cli")
            if not cli_flag:
                continue

            kwargs: dict[str, object] = {
                "help": f.metadata.get("help", ""),
            }

            # Use override if provided, otherwise use field default
            if f.name in overrides:
                kwargs["default"] = overrides[f.name]
            elif f.default is not dataclasses.MISSING:
                kwargs["default"] = f.default

            action = f.metadata.get("action")
            if action:
                kwargs["action"] = action
                kwargs.pop("default", None)
            else:
                if f.type in (int, float, str):
                    kwargs["type"] = f.type

            if f.metadata.get("choices"):
                kwargs["choices"] = f.metadata["choices"]

            parser.add_argument(cli_flag, **kwargs)

    @classmethod
    def from_args(cls, args: Any) -> "TrainerConfig":
        """Construct a TrainerConfig from parsed argparse args.

        Args:
            args: Parsed argparse namespace.

        Returns:
            TrainerConfig with values from CLI arguments.
        """
        config_kwargs: dict[str, object] = {}
        for f in fields(cls):
            cli_flag = f.metadata.get("cli")
            if not cli_flag:
                continue

            arg_name = cli_flag.lstrip("-").replace("-", "_")
            if hasattr(args, arg_name):
                config_kwargs[f.name] = getattr(args, arg_name)

        return cls(**config_kwargs)
