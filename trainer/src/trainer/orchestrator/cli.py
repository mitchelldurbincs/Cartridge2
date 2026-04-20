"""Command-line interface for the orchestrator.

This module provides argument parsing and the main entry point for the
synchronized AlphaZero training loop.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from ..central_config import get_config as get_central_config
from ..structured_logging import setup_logging
from .config import LoopConfig

logger = logging.getLogger(__name__)


def parse_args() -> LoopConfig:
    """Parse command line arguments with config.toml defaults.

    Priority (highest to lowest):
    1. CLI arguments
    2. Environment variables (CARTRIDGE_* and legacy ALPHAZERO_*)
    3. config.toml values
    4. Built-in defaults
    """
    # Load central config (includes env var overrides)
    cfg = get_central_config()

    parser = argparse.ArgumentParser(
        description="Synchronized AlphaZero Training Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration:
    Settings are loaded from config.toml with the following override priority:
    1. CLI arguments (highest)
    2. Environment variables (CARTRIDGE_* or legacy ALPHAZERO_*)
    3. config.toml values
    4. Built-in defaults (lowest)

Examples:
    # Basic training with evaluation every iteration
    python -m trainer.orchestrator --iterations 50 --episodes 200 --steps 500

    # Connect4 with GPU
    python -m trainer.orchestrator --env-id connect4 --device cuda --iterations 100

    # Disable evaluation for faster training
    python -m trainer.orchestrator --eval-interval 0

    # Via Docker (uses config.toml mounted to /app/config.toml)
    docker compose up alphazero
        """,
    )

    # Iteration settings (defaults from central config)
    parser.add_argument(
        "--iterations",
        type=int,
        default=cfg.training.iterations,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--start-iteration",
        type=int,
        default=cfg.training.start_iteration,
        help="Starting iteration number",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=cfg.training.episodes_per_iteration,
        help="Episodes per iteration",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=cfg.training.steps_per_iteration,
        help="Training steps per iteration",
    )

    # Environment
    parser.add_argument(
        "--env-id",
        type=str,
        default=cfg.common.env_id,
        help="Environment ID: tictactoe, connect4",
    )

    # Paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default=cfg.common.data_dir,
        help="Data directory",
    )
    parser.add_argument(
        "--actor-binary",
        type=str,
        default=os.environ.get("ACTOR_BINARY"),
        help="Path to actor binary (env: ACTOR_BINARY)",
    )

    # Actor settings
    parser.add_argument(
        "--actor-log-interval",
        type=int,
        default=cfg.actor.log_interval,
        help="Actor log interval in episodes",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=cfg.training.num_actors,
        help="Number of parallel actor processes for self-play",
    )

    # MCTS simulation ramping settings
    parser.add_argument(
        "--mcts-start-sims",
        type=int,
        default=cfg.mcts.start_sims,
        help="MCTS simulations for first iteration",
    )
    parser.add_argument(
        "--mcts-max-sims",
        type=int,
        default=cfg.mcts.max_sims,
        help="Maximum MCTS simulations after ramping",
    )
    parser.add_argument(
        "--mcts-sim-ramp-rate",
        type=int,
        default=cfg.mcts.sim_ramp_rate,
        help="MCTS simulations to add per iteration",
    )

    # Temperature schedule
    parser.add_argument(
        "--temp-threshold",
        type=int,
        default=cfg.mcts.temp_threshold,
        help="Move number to reduce temperature (0=disabled). "
        "Recommended: tictactoe=5, connect4=15, othello=30",
    )

    # Trainer settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.training.batch_size,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=cfg.training.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=cfg.training.checkpoint_interval,
        help="Steps between checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.training.device,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device (auto = detect best available)",
    )

    # Evaluation settings
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=cfg.evaluation.interval,
        help="Evaluate every N iterations, 0 to disable",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=cfg.evaluation.games,
        help="Games per evaluation",
    )
    parser.add_argument(
        "--eval-win-threshold",
        type=float,
        default=cfg.evaluation.win_threshold,
        help="Win rate threshold to become new best model (0.55 = 55%%)",
    )
    parser.add_argument(
        "--eval-vs-random",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=cfg.evaluation.eval_vs_random,
        help="Also evaluate against random baseline (true/false)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default=cfg.common.log_level.upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    args = parser.parse_args()

    return LoopConfig(
        iterations=args.iterations,
        start_iteration=args.start_iteration,
        episodes_per_iteration=args.episodes,
        steps_per_iteration=args.steps,
        env_id=args.env_id,
        data_dir=Path(args.data_dir),
        actor_binary=Path(args.actor_binary) if args.actor_binary else None,
        actor_log_interval=args.actor_log_interval,
        num_actors=args.num_actors,
        mcts_start_sims=args.mcts_start_sims,
        mcts_max_sims=args.mcts_max_sims,
        mcts_sim_ramp_rate=args.mcts_sim_ramp_rate,
        temp_threshold=args.temp_threshold,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        eval_win_threshold=args.eval_win_threshold,
        eval_vs_random=args.eval_vs_random,
        device=args.device,
        log_level=args.log_level,
    )


def main() -> int:
    """Main entry point for the orchestrator."""
    from .orchestrator import Orchestrator

    config = parse_args()

    # Set up structured logging (supports JSON for cloud deployments)
    setup_logging(level=config.log_level, component="orchestrator")

    try:
        orchestrator = Orchestrator(config)
        orchestrator.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
