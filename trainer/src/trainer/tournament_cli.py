"""CLI for the player registry and tournaments.

Two commands:

    trainer register-players --env-id connect4
    trainer tournament --env-id connect4 --games 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .environment_catalog import get_environment
from .registry import DEFAULT_PLAY_TEMPERATURE, PlayerRegistry, register_checkpoints
from .tournament import run_tournament

logger = logging.getLogger(__name__)

_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1


def _positive_u32_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive u32 integer") from exc
    if not 1 <= parsed <= _MAX_U32:
        raise argparse.ArgumentTypeError("expected a positive u32 integer")
    return parsed


def _u64_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a nonnegative u64 integer") from exc
    if not 0 <= parsed <= _MAX_U64:
        raise argparse.ArgumentTypeError("expected a nonnegative u64 integer")
    return parsed


def add_register_players_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-id", type=str, default="connect4", help="Game the players play"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=argparse.SUPPRESS,
        help="Checkpoint repository root (default: selected runtime profile)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=argparse.SUPPRESS,
        help="Player registry file (default: selected runtime profile)",
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
        default=argparse.SUPPRESS,
        help="Player registry file (default: selected runtime profile)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=argparse.SUPPRESS,
        help="Where to write the results (default: selected runtime profile)",
    )
    parser.add_argument(
        "--games",
        type=_positive_u32_argument,
        default=20,
        help="Games per pairing (positive u32)",
    )
    parser.add_argument(
        "--seed",
        type=_u64_argument,
        default=42,
        help="Base RNG seed, shared by every pair (nonnegative u64)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )


def run_register_players(args: argparse.Namespace) -> int:
    """Add every immutable checkpoint in a repository to the registry."""
    model_root = Path(args.models_dir)
    registry_path = Path(args.registry)
    try:
        environment = get_environment(args.env_id)
        registry = PlayerRegistry.load(registry_path)
        added = register_checkpoints(
            registry,
            env_id=args.env_id,
            model_root=model_root,
            algorithm_id=args.algorithm,
            simulations=args.simulations,
            temperature=args.temperature,
            replace=args.replace,
        )
    except ValueError as e:
        logger.error(str(e))
        return 1

    registry.save(registry_path)
    profile_count = len(
        registry.for_profile(
            env_id=args.env_id,
            env_contract_version=environment.contract_version,
            algorithm_id=args.algorithm,
        )
    )

    logger.info(
        f"Registered {len(added)} new player(s) from {model_root}; "
        f"{profile_count} "
        f"total for {args.algorithm}/{args.env_id}/v{environment.contract_version}"
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

    try:
        registry = PlayerRegistry.load(registry_path)
        results = run_tournament(
            registry,
            env_id=args.env_id,
            algorithm_id=args.algorithm,
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
