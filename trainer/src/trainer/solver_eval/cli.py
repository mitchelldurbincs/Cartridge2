"""Diagnostic command-line entry point for standalone solver evaluation."""

import argparse
import logging
import math
import struct
from collections.abc import Iterable
from pathlib import Path

from ..players import ModelPlayer, RandomPlayer
from ..registry import artifact_contract_for, discover_checkpoints
from ..storage.publisher import create_checkpoint_publisher
from .results import SolverEvalResults
from .scorer import SolverScorer, solver_evaluate

logger = logging.getLogger(__name__)

_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_MAX_F32 = float.fromhex("0x1.fffffep+127")


def _positive_u32_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive u32 integer") from exc
    if parsed <= 0 or parsed > _MAX_U32:
        raise argparse.ArgumentTypeError("expected a positive u32 integer")
    return parsed


def _u64_argument(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a nonnegative u64 integer") from exc
    if parsed < 0 or parsed > _MAX_U64:
        raise argparse.ArgumentTypeError("expected a nonnegative u64 integer")
    return parsed


def _nonnegative_f32_argument(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a finite nonnegative f32") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > _MAX_F32:
        raise argparse.ArgumentTypeError("expected a finite nonnegative f32")
    narrowed = float(struct.unpack("!f", struct.pack("!f", parsed))[0])
    return 0.0 if narrowed == 0.0 else narrowed


def _validate_solver_eval_request(args: argparse.Namespace) -> None:
    """Validate programmatic callers as strictly as the public parser."""
    if args.all_checkpoints and hasattr(args, "model"):
        raise ValueError("--model and --all-checkpoints are mutually exclusive")
    if (
        isinstance(args.games, bool)
        or not isinstance(args.games, int)
        or not 1 <= args.games <= _MAX_U32
    ):
        raise ValueError("--games must be a positive u32 integer")
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or not 0 <= args.seed <= _MAX_U64
    ):
        raise ValueError("--seed must be a nonnegative u64 integer")
    if args.seed > _MAX_U64 - (args.games - 1):
        raise ValueError("--seed plus the game index exceeds u64")
    if isinstance(args.temperature, bool) or not isinstance(
        args.temperature, (int, float)
    ):
        raise ValueError("--temperature must be a finite nonnegative f32")
    temperature = float(args.temperature)
    if not math.isfinite(temperature) or temperature < 0.0 or temperature > _MAX_F32:
        raise ValueError("--temperature must be a finite nonnegative f32")
    narrowed = float(struct.unpack("!f", struct.pack("!f", temperature))[0])
    args.temperature = 0.0 if narrowed == 0.0 else narrowed


def format_progression_table(results: Iterable[SolverEvalResults]) -> str:
    """Compact per-checkpoint progression table, sorted by training step."""
    ordered = sorted(results, key=lambda r: (r.step is None, r.step or 0))
    lines = [
        f"{'step':>8}  {'model':<28} {'games':>5} {'value-opt':>10} "
        f"{'exact-best':>10} {'blunder':>8}  {'W/L/D':>11}",
        "-" * 88,
    ]
    for r in ordered:
        step_str = str(r.step) if r.step is not None else "-"
        wld = f"{r.model_wins}/{r.model_losses}/{r.draws}"
        lines.append(
            f"{step_str:>8}  {r.model_name:<28} {r.games:>5} "
            f"{r.overall.value_optimal_rate:>10.1%} {r.overall.exact_best_rate:>10.1%} "
            f"{r.overall.blunder_rate:>8.1%}  {wld:>11}"
        )
    return "\n".join(lines)


def add_solver_eval_arguments(parser: argparse.ArgumentParser) -> None:
    """Add solver-eval arguments to a parser."""
    model_selection = parser.add_mutually_exclusive_group()
    model_selection.add_argument(
        "--model",
        type=str,
        default=argparse.SUPPRESS,
        help="One-off ONNX file. If omitted, resolve the latest checkpoint "
        "selected by RunHead",
    )
    model_selection.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Evaluate every immutable manifest in the checkpoint repository",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=argparse.SUPPRESS,
        help="Checkpoint repository root (default: selected runtime profile)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="connect4",
        choices=["connect4"],
        help="Game environment (only connect4 has a solver)",
    )
    parser.add_argument(
        "--games",
        type=_positive_u32_argument,
        default=50,
        help="Number of games to play per model",
    )
    parser.add_argument(
        "--seed",
        type=_u64_argument,
        default=42,
        help="Base RNG seed (per-game seed is seed + game index)",
    )
    parser.add_argument(
        "--temperature",
        type=_nonnegative_f32_argument,
        default=0.0,
        help="Model sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print individual game moves",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )


def run_solver_evaluation(args: argparse.Namespace) -> int:
    """Run solver evaluation with the given arguments."""
    if args.env_id != "connect4":
        logger.error(
            f"Solver evaluation requires a perfect solver and is only available "
            f"for connect4 (got '{args.env_id}'). bitbully solves standard 7x6 "
            f"Connect4 only."
        )
        return 1

    try:
        _validate_solver_eval_request(args)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    try:
        if args.all_checkpoints or not hasattr(args, "model"):
            model_root = Path(args.models_dir)
            contract = artifact_contract_for(args.env_id, args.algorithm)
            repository = create_checkpoint_publisher(contract, model_root)
            if args.all_checkpoints:
                checkpoints = discover_checkpoints(repository)
                if not checkpoints:
                    logger.error(f"No checkpoints found in {model_root}")
                    return 1
                model_jobs = [
                    (
                        checkpoint.onnx_path,
                        checkpoint.checkpoint_id,
                        checkpoint.manifest.step,
                    )
                    for checkpoint in checkpoints
                ]
                logger.info(
                    f"Evaluating {len(model_jobs)} checkpoints from {model_root}"
                )
            else:
                checkpoint = repository.resolve_head()
                if checkpoint is None:
                    logger.error(f"Checkpoint repository has no RunHead: {model_root}")
                    return 1
                model_jobs = [
                    (
                        checkpoint.onnx_path,
                        checkpoint.checkpoint_id,
                        checkpoint.manifest.step,
                    )
                ]
        else:
            model_path = Path(args.model)
            if not model_path.is_file():
                logger.error(f"Model not found: {model_path}")
                return 1
            model_jobs = [(model_path, None, None)]
    except (ImportError, OSError, ValueError) as exc:
        logger.error(f"Could not resolve checkpoints: {exc}")
        return 1

    try:
        scorer = SolverScorer()
    except (ImportError, RuntimeError) as e:
        logger.error(str(e))
        return 1

    try:
        from importlib.metadata import version

        bitbully_version = version("bitbully")
    except Exception:
        bitbully_version = None

    all_results = []
    for model_path, checkpoint_id, checkpoint_step in model_jobs:
        model = ModelPlayer(str(model_path), temperature=args.temperature)

        logger.info(f"Evaluating {model.name} over {args.games} games vs random")
        results = solver_evaluate(
            model=model,
            opponent=RandomPlayer(),
            scorer=scorer,
            algorithm_id=args.algorithm,
            env_id=args.env_id,
            num_games=args.games,
            seed=args.seed,
            verbose=args.verbose,
            checkpoint_id=checkpoint_id,
            checkpoint_step=checkpoint_step,
        )
        results.bitbully_version = bitbully_version
        all_results.append(results)

        print(results.summary())

    if len(all_results) > 1:
        print("\nProgression across checkpoints:")
        print(format_progression_table(all_results))

    return 0
