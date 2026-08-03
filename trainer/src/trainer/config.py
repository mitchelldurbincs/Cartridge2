"""AlphaZero learner configuration and CLI argument helpers.

This module provides:
- AlphaZeroLearnerConfig dataclass with AlphaZero training parameters
- CLI field metadata for automatic argparse integration
- Methods to build and parse CLI arguments
"""

import dataclasses
import math
from dataclasses import dataclass, field, fields
from typing import Any, Callable

from crucible.backoff import DEFAULT_MAX_WAIT, DEFAULT_WAIT_INTERVAL

from .storage.base import ReplaySelection

__all__ = ["AlphaZeroLearnerConfig", "cli_field"]


def _strict_integer(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _finite_number(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    minimum_ok = normalized >= minimum if minimum_inclusive else normalized > minimum
    if (
        not math.isfinite(normalized)
        or not minimum_ok
        or (maximum is not None and normalized > maximum)
    ):
        if maximum is not None:
            constraint = f"in [{minimum}, {maximum}]"
        elif minimum_inclusive:
            constraint = f">= {minimum}"
        else:
            constraint = f"> {minimum}"
        raise ValueError(f"{name} must be a finite number {constraint}")
    return 0.0 if normalized == 0.0 else normalized


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
class AlphaZeroLearnerConfig:
    """Configuration for the AlphaZero learner.

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

    # Replay-store management
    clear_replay_on_start: bool = cli_field(
        False,
        cli="--clear-replay",
        action="store_true",
        help="Delete records in the exact replay selection before standalone training",
    )
    replay_window: int = cli_field(
        0,
        cli="--replay-window",
        help="Keep only the most recent N replay records (0 disables cleanup)",
    )
    replay_cleanup_interval: int = cli_field(
        0,
        cli="--replay-cleanup-interval",
        help=(
            "Steps between replay cleanup when replay-window is set ("
            "0 = align with stats-interval)"
        ),
    )
    # Operational fencing is intentionally excluded from the learner recipe.
    # A learner cannot start until its caller supplies one exact collection.
    replay_selection: ReplaySelection | None = field(default=None, repr=False)

    # Internal step offset derived from RunHead or supplied by loop composition.
    start_step: int = field(default=0)

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

    # Per-step metrics callback (not exposed to CLI, set programmatically).
    # Called as metrics_hook(payload, global_step) at stats_interval cadence;
    # the orchestrator uses it to forward training metrics to W&B.
    metrics_hook: Callable[[dict, int], None] | None = field(default=None, repr=False)

    # Synchronized orchestration stages one final checkpoint/stats pair and
    # commits it only after parent-owned evaluation. Standalone learners commit
    # their own RunCommitV1/RunHeadV2 updates.
    defer_run_commit: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Reject malformed or operationally ineffective learner settings."""
        for name, minimum in (
            ("batch_size", 1),
            ("lr_warmup_steps", 0),
            ("lr_total_steps", 0),
            ("total_steps", 1),
            ("checkpoint_interval", 1),
            ("stats_interval", 1),
            ("log_interval", 1),
            ("replay_window", 0),
            ("replay_cleanup_interval", 0),
            ("start_step", 0),
        ):
            _strict_integer(getattr(self, name), name=name, minimum=minimum)

        for name, minimum, minimum_inclusive, maximum in (
            ("learning_rate", 0.0, False, None),
            ("weight_decay", 0.0, True, None),
            ("value_loss_weight", 0.0, True, None),
            ("policy_loss_weight", 0.0, True, None),
            ("grad_clip_norm", 0.0, True, None),
            ("lr_min_ratio", 0.0, True, 1.0),
            ("lr_warmup_start_ratio", 0.0, True, 1.0),
            ("wait_interval", 0.0, False, None),
            ("max_wait", 0.0, True, None),
        ):
            setattr(
                self,
                name,
                _finite_number(
                    getattr(self, name),
                    name=name,
                    minimum=minimum,
                    maximum=maximum,
                    minimum_inclusive=minimum_inclusive,
                ),
            )

        if self.value_loss_weight == 0.0 and self.policy_loss_weight == 0.0:
            raise ValueError(
                "value_loss_weight and policy_loss_weight cannot both be zero"
            )
        if self.lr_total_steps and self.lr_total_steps < self.total_steps:
            raise ValueError("lr_total_steps cannot be less than total_steps")
        if self.replay_window == 0 and self.replay_cleanup_interval != 0:
            raise ValueError("replay_cleanup_interval requires a nonzero replay_window")

        for name in ("use_lr_scheduler", "clear_replay_on_start", "defer_run_commit"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("model_dir", "stats_path", "env_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.device, str) or self.device not in {
            "auto",
            "cpu",
            "cuda",
            "mps",
        }:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if self.replay_selection is not None and not isinstance(
            self.replay_selection, ReplaySelection
        ):
            raise ValueError("replay_selection must be a ReplaySelection or None")
        for name in ("shutdown_check", "metrics_hook"):
            callback = getattr(self, name)
            if callback is not None and not callable(callback):
                raise ValueError(f"{name} must be callable or None")

    def learner_recipe(self) -> dict[str, object]:
        """Return only settings that change AlphaZero learning semantics."""
        from .algorithms.alphazero_board_v1 import get_game_config

        game = get_game_config(self.env_id)
        lr_horizon = self.lr_total_steps or self.total_steps
        cleanup_cadence = (
            (self.replay_cleanup_interval or self.stats_interval)
            if self.replay_window > 0
            else 0
        )
        return {
            "schema_version": 1,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "value_loss_weight": self.value_loss_weight,
            "policy_loss_weight": self.policy_loss_weight,
            "grad_clip_norm": self.grad_clip_norm,
            "use_lr_scheduler": self.use_lr_scheduler,
            "lr_min_ratio": self.lr_min_ratio,
            "lr_warmup_steps": self.lr_warmup_steps,
            "lr_warmup_start_ratio": self.lr_warmup_start_ratio,
            "lr_horizon_steps": lr_horizon,
            "training_steps": self.total_steps,
            "clear_replay_on_start": self.clear_replay_on_start,
            "replay_window": self.replay_window,
            "replay_cleanup_cadence": cleanup_cadence,
            "model_architecture": {
                "schema_version": 1,
                "implementation": "alphazero_policy_value_network_v1",
                "network_type": game.network_type,
                "observation_elements": game.obs_size,
                "action_count": game.num_actions,
                "hidden_size": game.hidden_size,
                "board_width": game.board_width,
                "board_height": game.board_height,
                "observation_spatial_channels": game.obs_channels,
                "legal_actions_offset": game.legal_mask_offset,
                "player_relative_observation": game.player_relative_obs,
                "residual_blocks": game.num_res_blocks,
                "residual_filters": game.num_filters,
            },
        }

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
    def from_args(cls, args: Any) -> "AlphaZeroLearnerConfig":
        """Construct an AlphaZeroLearnerConfig from parsed argparse args.

        Args:
            args: Parsed argparse namespace.

        Returns:
            AlphaZeroLearnerConfig with values from CLI arguments.
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
