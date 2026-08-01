"""CLI for the player registry and tournaments.

Two commands:

    trainer register-players --env-id connect4
    trainer tournament --env-id connect4 --games 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .central_config import get_config
from .registry import DEFAULT_PLAY_TEMPERATURE, PlayerRegistry, register_checkpoints
from .tournament import run_tournament

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "./data/players.json"
DEFAULT_RESULTS_PATH = "./data/tournament.json"


def add_register_players_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-id", type=str, default="connect4", help="Game the players play"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory of ONNX checkpoints (default: the configured models dir)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=DEFAULT_REGISTRY_PATH,
        help="Player registry file",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=0,
        help="MCTS simulations these players search with (0 = policy head only). "
        "Registered as separate players from the same weights at another budget",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_PLAY_TEMPERATURE,
        help="Sampling temperature. 0 makes a model deterministic, so two such "
        "players replay one identical game however many are scheduled",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="alphazero",
        help="Label recording how these players were trained",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite entries whose id already exists instead of skipping them",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )


def add_tournament_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-id", type=str, default="connect4", help="Game to play")
    parser.add_argument(
        "--registry",
        type=str,
        default=DEFAULT_REGISTRY_PATH,
        help="Player registry file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_RESULTS_PATH,
        help="Where to write the results",
    )
    parser.add_argument("--games", type=int, default=20, help="Games per pairing")
    parser.add_argument(
        "--seed", type=int, default=42, help="Base RNG seed, shared by every pair"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )


def run_register_players(args: argparse.Namespace) -> int:
    """Add every checkpoint in a models directory to the registry."""
    models_dir = Path(args.models_dir) if args.models_dir else get_config().models_dir
    if not models_dir.exists():
        logger.error(f"Models directory not found: {models_dir}")
        return 1

    registry_path = Path(args.registry)
    registry = PlayerRegistry.load(registry_path)

    try:
        added = register_checkpoints(
            registry,
            env_id=args.env_id,
            models_dir=models_dir,
            algorithm=args.algorithm,
            simulations=args.simulations,
            temperature=args.temperature,
            replace=args.replace,
        )
    except ValueError as e:
        logger.error(str(e))
        return 1

    registry.save(registry_path)

    logger.info(
        f"Registered {len(added)} new player(s) from {models_dir}; "
        f"{len(registry.for_env(args.env_id))} total for {args.env_id}"
    )
    for record in added:
        logger.info(f"  + {record.id}")
    if not added:
        logger.info("  (nothing new — pass --replace to re-register)")
    return 0


def run_tournament_command(args: argparse.Namespace) -> int:
    """Play a round robin over the registered players and rate them."""
    registry_path = Path(args.registry)
    if not registry_path.exists():
        logger.error(
            f"No player registry at {registry_path}. "
            f"Create one with `trainer register-players --env-id {args.env_id}`."
        )
        return 1

    registry = PlayerRegistry.load(registry_path)

    try:
        results = run_tournament(
            registry,
            env_id=args.env_id,
            games_per_pair=args.games,
            seed=args.seed,
        )
    except ValueError as e:
        logger.error(str(e))
        return 1
    except (RuntimeError, OSError) as e:
        logger.error(f"Tournament failed: {e}")
        return 1

    output_path = Path(args.output)
    results.save(output_path)

    print(results.table())
    print(f"\nRatings are Elo above `{results.anchor}`.")
    print(
        f"{len(results.matches)} pairings in {results.wall_time_seconds:.1f}s -> {output_path}"
    )
    return 0
