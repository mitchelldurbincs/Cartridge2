"""Command-line entry point and stats I/O for solver evaluation.

Handles checkpoint discovery, appending results to data/solver_stats.json,
progression tables across checkpoints, argument parsing, and the run driver.
"""

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

from ..evaluator import get_game_metadata_or_config
from ..logging_utils import silence_noisy_loggers
from ..policies import OnnxPolicy, RandomPolicy
from .results import CHECKPOINT_PATTERN, SolverEvalResults, infer_step_from_filename
from .scorer import SolverScorer, solver_evaluate

logger = logging.getLogger(__name__)


def discover_checkpoints(models_dir: Path) -> list[Path]:
    """All step checkpoints (numerically sorted) plus latest/best if present."""
    checkpoints = sorted(
        (
            p
            for p in models_dir.glob("model_step_*.onnx")
            if CHECKPOINT_PATTERN.fullmatch(p.name)
        ),
        key=lambda p: infer_step_from_filename(p) or 0,
    )
    for name in ("latest.onnx", "best.onnx"):
        candidate = models_dir / name
        if candidate.exists():
            checkpoints.append(candidate)
    return checkpoints


def append_solver_stats(entry: dict, output_path: Path) -> None:
    """Append one evaluation entry to the solver stats JSON file."""
    stats = {"solver_evaluations": []}
    if output_path.exists():
        try:
            with open(output_path) as f:
                loaded = json.load(f)
            if isinstance(loaded.get("solver_evaluations"), list):
                stats = loaded
            else:
                logger.warning(f"Unexpected structure in {output_path}, starting fresh")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read {output_path} ({e}), starting fresh")

    stats["solver_evaluations"].append(entry)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Solver stats appended to {output_path}")


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
    parser.add_argument(
        "--model",
        type=str,
        default="./data/models/latest.onnx",
        help="Path to ONNX model file (ignored with --all-checkpoints)",
    )
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Evaluate every model_step_*.onnx plus latest/best in --models-dir",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./data/models",
        help="Directory scanned by --all-checkpoints",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="connect4",
        choices=["tictactoe", "connect4"],
        help="Game environment (only connect4 has a solver)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of games to play per model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed (per-game seed is seed + game index)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/solver_stats.json",
        help="JSON file to append results to",
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

    if args.all_checkpoints:
        models_dir = Path(args.models_dir)
        model_paths = discover_checkpoints(models_dir)
        if not model_paths:
            logger.error(f"No checkpoints found in {models_dir}")
            return 1
        logger.info(f"Evaluating {len(model_paths)} checkpoints from {models_dir}")
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return 1
        model_paths = [model_path]

    config = get_game_metadata_or_config(args.env_id)

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
    for model_path in model_paths:
        try:
            model = OnnxPolicy(str(model_path), temperature=args.temperature)
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return 1

        logger.info(f"Evaluating {model.name} over {args.games} games vs random")
        results = solver_evaluate(
            model=model,
            opponent=RandomPolicy(),
            scorer=scorer,
            env_id=args.env_id,
            config=config,
            num_games=args.games,
            seed=args.seed,
            verbose=args.verbose,
        )
        results.bitbully_version = bitbully_version
        all_results.append(results)

        print(results.summary())
        append_solver_stats(results.to_dict(), Path(args.output))

    if len(all_results) > 1:
        print("\nProgression across checkpoints:")
        print(format_progression_table(all_results))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score Connect4 model moves against a perfect solver (bitbully)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_solver_eval_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    silence_noisy_loggers()

    return run_solver_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())
