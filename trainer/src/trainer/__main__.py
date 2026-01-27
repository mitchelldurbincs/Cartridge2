"""CLI entrypoint for the Cartridge2 trainer.

Provides subcommands for different operations:
    python -m trainer train     - Run training loop on replay buffer
    python -m trainer evaluate  - Evaluate model against random baseline
    python -m trainer loop      - Run synchronized AlphaZero training

For backwards compatibility, running without a subcommand defaults to 'train':
    python -m trainer --steps 1000

Entry points after pip install:
    trainer           - Same as 'python -m trainer train'
    trainer-loop      - Same as 'python -m trainer loop'
    trainer-evaluate  - Same as 'python -m trainer evaluate'

Note: Replay buffer connection is configured via CARTRIDGE_STORAGE_POSTGRES_URL
environment variable.
"""

import argparse
import logging
import sys

from .structured_logging import setup_logging


def cmd_train(args: argparse.Namespace) -> int:
    """Run the training loop."""
    from . import metrics as prom_metrics
    from .backoff import WaitTimeout
    from .trainer import Trainer, TrainerConfig

    logger = logging.getLogger(__name__)
    logger.info("Cartridge2 Trainer starting...")
    logger.info(f"Config: model_dir={args.model_dir}, steps={args.steps}")

    # Start Prometheus metrics server
    metrics_port = getattr(args, "metrics_port", 9090)
    prom_metrics.start_metrics_server(port=metrics_port)

    config = TrainerConfig.from_args(args)

    try:
        trainer = Trainer(config)
        stats = trainer.train()
        logger.info(f"Training complete! Final loss: {stats.total_loss:.4f}")
        logger.info(f"Last checkpoint: {stats.last_checkpoint}")
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except WaitTimeout as e:
        logger.error(f"Timeout: {e}")
        return 2
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run model evaluation."""
    from .evaluator import run_evaluation

    return run_evaluation(args)


def cmd_loop(args: argparse.Namespace) -> int:
    """Run synchronized AlphaZero training loop."""
    from .orchestrator import main as orchestrator_main

    # orchestrator has its own arg parsing, so we need to strip the 'loop' subcommand
    # from sys.argv before calling it
    if len(sys.argv) >= 2 and sys.argv[1] == "loop":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
    return orchestrator_main()


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Add train command arguments to a parser.

    This is shared between the 'train' subcommand and backwards-compatible
    direct invocation mode.
    """
    from .central_config import get_config as get_central_config
    from .trainer import TrainerConfig

    # Load central config for defaults
    cfg = get_central_config()

    # Configure parser with central config overrides for key settings
    TrainerConfig.configure_parser(
        parser,
        overrides={
            "checkpoint_interval": cfg.training.checkpoint_interval,
            "max_checkpoints": cfg.training.max_checkpoints,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "grad_clip_norm": cfg.training.grad_clip_norm,
            "device": cfg.training.device,
            "env_id": cfg.common.env_id,
            "model_dir": str(cfg.models_dir),
            "stats_path": str(cfg.stats_path),
        },
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=cfg.common.log_level.upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9090,
        help="Port for Prometheus metrics server",
    )


def setup_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the train subcommand parser."""
    parser = subparsers.add_parser(
        "train",
        help="Train on replay buffer data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_train_arguments(parser)
    parser.set_defaults(func=cmd_train)


def setup_evaluate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the evaluate subcommand parser."""
    from .evaluator import add_evaluate_arguments

    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model against random baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_evaluate_arguments(parser)
    parser.set_defaults(func=cmd_evaluate)


def setup_loop_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the loop subcommand parser."""
    parser = subparsers.add_parser(
        "loop",
        help="Run synchronized AlphaZero training (actor + trainer + eval)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Synchronized AlphaZero training loop that coordinates:
1. Self-play episode generation (actor)
2. Neural network training
3. Model evaluation against random baseline

Each iteration clears the replay buffer to ensure training data
comes only from the current model version.
        """,
    )
    # The loop command re-parses sys.argv via orchestrator.main()
    # so we don't need to add arguments here
    parser.set_defaults(func=cmd_loop)


def main() -> int:
    """Main entry point with subcommand support."""
    # Check if we're being called with a subcommand
    # For backwards compatibility, default to 'train' if no subcommand given
    if len(sys.argv) >= 2 and sys.argv[1] in (
        "train",
        "evaluate",
        "loop",
        "-h",
        "--help",
    ):
        # Subcommand mode
        parser = argparse.ArgumentParser(
            description="Cartridge2 AlphaZero Trainer",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        subparsers = parser.add_subparsers(
            title="commands",
            description="Available commands",
            dest="command",
        )

        setup_train_parser(subparsers)
        setup_evaluate_parser(subparsers)
        setup_loop_parser(subparsers)

        args = parser.parse_args()

        if args.command is None:
            parser.print_help()
            return 0

        # Configure structured logging (supports JSON for cloud deployments)
        log_level = getattr(args, "log_level", "INFO")
        component = "trainer" if args.command == "train" else args.command
        setup_logging(level=log_level, component=component)

        return args.func(args)

    else:
        # Backwards compatibility: treat as 'train' command
        parser = argparse.ArgumentParser(
            description="Cartridge2 AlphaZero-style Trainer",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        _add_train_arguments(parser)
        args = parser.parse_args()

        # Configure structured logging (supports JSON for cloud deployments)
        setup_logging(level=args.log_level, component="trainer")

        return cmd_train(args)


if __name__ == "__main__":
    sys.exit(main())
