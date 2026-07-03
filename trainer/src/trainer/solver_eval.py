"""Solver-based move-quality evaluation for Connect4.

Connect4 is a solved game, so every model decision can be scored against
ground truth. This module plays evaluation games (model vs random, both
seats, deterministic seeding) and, for each model decision, queries the
bitbully perfect solver to classify the chosen move:

- value-optimal: the move preserves the game-theoretic value class
  (win/draw/loss from the mover's perspective)
- blunder: the move drops the value class (win->draw, win->loss, draw->loss)
- exact-best: the move is in the argmax set of solver scores
  (fastest win / slowest loss)

Results are aggregated overall, by ply bucket, and by seat, then appended
to data/solver_stats.json. Unlike win-rate-vs-random, these metrics have a
fixed, objective yardstick, so they are comparable across checkpoints.

Usage (defaults assume running from the repo root):
    python -m trainer solver-eval --model ./data/models/latest.onnx --games 100
    python -m trainer solver-eval --all-checkpoints --games 100
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .evaluator import get_game_metadata_or_config, play_game
from .games import Player
from .logging_utils import silence_noisy_loggers
from .policies import OnnxPolicy, Policy, RandomPolicy

if TYPE_CHECKING:
    from .game_config import GameConfig
    from .games import GameState
    from .storage import GameMetadata

logger = logging.getLogger(__name__)

CLASS_WIN = "win"
CLASS_DRAW = "draw"
CLASS_LOSS = "loss"

BLUNDER_WIN_TO_DRAW = "win_to_draw"
BLUNDER_WIN_TO_LOSS = "win_to_loss"
BLUNDER_DRAW_TO_LOSS = "draw_to_loss"

PLY_BUCKETS = ("ply_1_8", "ply_9_20", "ply_21_plus")
SEATS = ("first", "second")

CHECKPOINT_PATTERN = re.compile(r"model_step_(\d+)\.onnx")


def classify_score(score: int) -> str:
    """Classify a solver score (mover's perspective) into win/draw/loss."""
    if score > 0:
        return CLASS_WIN
    if score < 0:
        return CLASS_LOSS
    return CLASS_DRAW


def ply_bucket(move_number: int) -> str:
    """Bucket a 1-based move number into opening/middle/late game."""
    if move_number <= 8:
        return PLY_BUCKETS[0]
    if move_number <= 20:
        return PLY_BUCKETS[1]
    return PLY_BUCKETS[2]


def infer_step_from_filename(path: str | Path) -> int | None:
    """Extract the training step from a checkpoint filename, if present.

    model_step_016000.onnx -> 16000; latest.onnx / best.onnx -> None.
    """
    match = CHECKPOINT_PATTERN.fullmatch(Path(path).name)
    return int(match.group(1)) if match else None


@dataclass
class MoveJudgment:
    """Solver verdict for a single model decision."""

    chosen_class: str
    best_class: str
    value_optimal: bool
    exact_best: bool
    blunder: str | None
    forced: bool


def judge_move(legal_scores: dict[int, int], chosen: int) -> MoveJudgment:
    """Judge a chosen move against solver scores for all legal moves.

    Args:
        legal_scores: Solver score per legal column (mover's perspective).
        chosen: The column the policy selected.

    Raises:
        ValueError: If chosen is not among the legal scored moves — this
            indicates a bug upstream, so fail loud rather than skip.
    """
    if chosen not in legal_scores:
        raise ValueError(
            f"Chosen move {chosen} not in scored legal moves {sorted(legal_scores)}"
        )

    best_score = max(legal_scores.values())
    chosen_class = classify_score(legal_scores[chosen])
    best_class = classify_score(best_score)
    value_optimal = chosen_class == best_class

    blunder = None
    if not value_optimal:
        blunder = {
            (CLASS_WIN, CLASS_DRAW): BLUNDER_WIN_TO_DRAW,
            (CLASS_WIN, CLASS_LOSS): BLUNDER_WIN_TO_LOSS,
            (CLASS_DRAW, CLASS_LOSS): BLUNDER_DRAW_TO_LOSS,
        }[(best_class, chosen_class)]

    return MoveJudgment(
        chosen_class=chosen_class,
        best_class=best_class,
        value_optimal=value_optimal,
        exact_best=legal_scores[chosen] == best_score,
        blunder=blunder,
        forced=len(legal_scores) == 1,
    )


@dataclass
class BucketStats:
    """Aggregated judgments for one slice (overall, a ply bucket, a seat)."""

    positions: int = 0
    value_optimal: int = 0
    exact_best: int = 0
    blunders_win_to_draw: int = 0
    blunders_win_to_loss: int = 0
    blunders_draw_to_loss: int = 0
    forced: int = 0

    def add(self, judgment: MoveJudgment) -> None:
        self.positions += 1
        if judgment.value_optimal:
            self.value_optimal += 1
        if judgment.exact_best:
            self.exact_best += 1
        if judgment.blunder == BLUNDER_WIN_TO_DRAW:
            self.blunders_win_to_draw += 1
        elif judgment.blunder == BLUNDER_WIN_TO_LOSS:
            self.blunders_win_to_loss += 1
        elif judgment.blunder == BLUNDER_DRAW_TO_LOSS:
            self.blunders_draw_to_loss += 1
        if judgment.forced:
            self.forced += 1

    def _rate(self, count: int) -> float:
        return count / self.positions if self.positions else 0.0

    @property
    def value_optimal_rate(self) -> float:
        return self._rate(self.value_optimal)

    @property
    def exact_best_rate(self) -> float:
        return self._rate(self.exact_best)

    @property
    def blunder_rate(self) -> float:
        blunders = (
            self.blunders_win_to_draw
            + self.blunders_win_to_loss
            + self.blunders_draw_to_loss
        )
        return self._rate(blunders)

    @property
    def forced_rate(self) -> float:
        return self._rate(self.forced)

    def to_dict(self) -> dict:
        return {
            "positions": self.positions,
            "value_optimal": self.value_optimal,
            "exact_best": self.exact_best,
            "blunders_win_to_draw": self.blunders_win_to_draw,
            "blunders_win_to_loss": self.blunders_win_to_loss,
            "blunders_draw_to_loss": self.blunders_draw_to_loss,
            "forced": self.forced,
            "value_optimal_rate": self.value_optimal_rate,
            "exact_best_rate": self.exact_best_rate,
            "blunder_rate": self.blunder_rate,
            "forced_rate": self.forced_rate,
        }


@dataclass
class SolverEvalResults:
    """Full results of a solver evaluation run for one model."""

    env_id: str
    model_name: str
    model_path: str
    step: int | None
    opponent_name: str
    games: int
    seed: int
    temperature: float
    model_wins: int = 0
    model_losses: int = 0
    draws: int = 0
    avg_game_length: float = 0.0
    overall: BucketStats = field(default_factory=BucketStats)
    by_ply: dict[str, BucketStats] = field(
        default_factory=lambda: {b: BucketStats() for b in PLY_BUCKETS}
    )
    by_seat: dict[str, BucketStats] = field(
        default_factory=lambda: {s: BucketStats() for s in SEATS}
    )
    solver_queries: int = 0
    solver_cache_hits: int = 0
    solver_time_seconds: float = 0.0
    wall_time_seconds: float = 0.0
    bitbully_version: str | None = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        cache_hit_rate = (
            self.solver_cache_hits / self.solver_queries if self.solver_queries else 0.0
        )
        return {
            "model": self.model_name,
            "model_path": self.model_path,
            "step": self.step,
            "env_id": self.env_id,
            "opponent": self.opponent_name,
            "games": self.games,
            "seed": self.seed,
            "temperature": self.temperature,
            "model_wins": self.model_wins,
            "model_losses": self.model_losses,
            "draws": self.draws,
            "avg_game_length": self.avg_game_length,
            "positions_scored": self.overall.positions,
            "forced_moves": self.overall.forced,
            "forced_move_rate": self.overall.forced_rate,
            "value_optimal_rate": self.overall.value_optimal_rate,
            "exact_best_rate": self.overall.exact_best_rate,
            "blunder_rate": self.overall.blunder_rate,
            "blunders_win_to_draw": self.overall.blunders_win_to_draw,
            "blunders_win_to_loss": self.overall.blunders_win_to_loss,
            "blunders_draw_to_loss": self.overall.blunders_draw_to_loss,
            "by_ply": {name: stats.to_dict() for name, stats in self.by_ply.items()},
            "by_seat": {name: stats.to_dict() for name, stats in self.by_seat.items()},
            "solver_queries": self.solver_queries,
            "solver_cache_hits": self.solver_cache_hits,
            "solver_cache_hit_rate": cache_hit_rate,
            "solver_time_seconds": self.solver_time_seconds,
            "wall_time_seconds": self.wall_time_seconds,
            "bitbully_version": self.bitbully_version,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        def row(label: str, stats: BucketStats) -> str:
            blunders = (
                f"{stats.blunders_win_to_draw}/{stats.blunders_win_to_loss}"
                f"/{stats.blunders_draw_to_loss}"
            )
            return (
                f"  {label:<12} {stats.positions:>6} "
                f"{stats.value_optimal_rate:>10.1%} {stats.exact_best_rate:>10.1%} "
                f"{blunders:>12} {stats.forced:>7}"
            )

        step_str = str(self.step) if self.step is not None else "-"
        lines = [
            f"{'=' * 72}",
            f"Solver Evaluation: {self.model_name} (step {step_str}) vs {self.opponent_name}",
            f"Game: {self.env_id} | games: {self.games} | seed: {self.seed}",
            f"{'=' * 72}",
            f"Results: {self.model_wins}W / {self.model_losses}L / {self.draws}D "
            f"(avg length {self.avg_game_length:.1f})",
            "",
            f"  {'slice':<12} {'moves':>6} {'value-opt':>10} {'exact-best':>10} "
            f"{'WD/WL/DL':>12} {'forced':>7}",
            row("overall", self.overall),
        ]
        lines += [row(name, stats) for name, stats in self.by_ply.items()]
        lines += [row(name, stats) for name, stats in self.by_seat.items()]
        cache_hit_rate = (
            self.solver_cache_hits / self.solver_queries if self.solver_queries else 0.0
        )
        lines += [
            "",
            f"Solver: {self.solver_queries} queries, {cache_hit_rate:.1%} cache hits, "
            f"{self.solver_time_seconds:.1f}s solving, {self.wall_time_seconds:.1f}s total",
            f"{'=' * 72}",
        ]
        return "\n".join(lines)


class SolverScorer:
    """Wraps the bitbully perfect solver behind a mirrored board.

    The scorer maintains its own bitbully Board; the caller mirrors every
    move of the game into it (model and opponent alike) and calls reset()
    at the start of each game. Every query cross-checks the full mirrored
    board against the game state cell by cell — any desync makes the
    metrics meaningless, so it raises instead of continuing.

    Solved positions are memoized across games and models: openings repeat
    heavily, so an --all-checkpoints sweep gets progressively cheaper.
    """

    def __init__(self, run_calibration: bool = True):
        try:
            import bitbully
        except ImportError as e:
            raise ImportError(
                "bitbully is required for solver evaluation. Install it with "
                "'pip install bitbully' (or reinstall the trainer: pip install -e .)"
            ) from e

        self._bitbully = bitbully
        self._agent = bitbully.BitBully()
        self._board = bitbully.Board()
        self._cache: dict[tuple, dict[int, int]] = {}
        self.queries = 0
        self.cache_hits = 0
        self.solve_time_seconds = 0.0

        if run_calibration:
            self.calibrate()

    def reset(self) -> None:
        """Start mirroring a fresh game."""
        self._board = self._bitbully.Board()

    def mirror_move(self, col: int) -> None:
        """Apply a move to the mirrored board."""
        if not self._board.play(col):
            raise RuntimeError(
                f"Solver board rejected move {col} — mirrored board has "
                "desynced from the game state"
            )

    def scores_for(self, state: "GameState") -> dict[int, int]:
        """Solver scores for every legal move in the given state.

        Scores are from the perspective of the side to move: positive is
        winning, zero drawing, negative losing.
        """
        # bitbully's to_array() is [col][row] with row 0 at the bottom and
        # cells 1/2 for first/second player — the exact layout and values of
        # Connect4State.board, so the mirrored board can be compared cell
        # for cell.
        mirrored = [int(cell) for column in self._board.to_array() for cell in column]
        if mirrored != list(state.board):
            raise RuntimeError(
                f"Solver board desynced from game state: mirrored board "
                f"{mirrored} != state board {list(state.board)}"
            )
        state_legal = sorted(state.legal_moves())

        self.queries += 1
        key = (tuple(state.board), int(state.current_player))
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        start = time.perf_counter()
        raw = self._agent.score_all_moves(self._board)
        self.solve_time_seconds += time.perf_counter() - start

        try:
            scores = {move: raw[move] for move in state_legal}
        except KeyError as e:
            raise RuntimeError(
                f"Solver did not score legal move {e} (returned {sorted(raw)})"
            ) from e
        self._cache[key] = scores
        return scores

    def calibrate(self) -> None:
        """Verify solver conventions on the empty board (known theory).

        Connect4 from the empty board: the center column is the unique
        winning first move, adjacent-to-center columns draw, the rest lose,
        and scores are left-right symmetric. This trips on any indexing,
        sign, or perspective mismatch before a single game is scored.
        """
        board = self._bitbully.Board()
        scores = self._agent.score_all_moves(board)

        failures = []
        if classify_score(scores[3]) != CLASS_WIN:
            failures.append(f"center (col 3) should be winning, got {scores[3]}")
        for col in (2, 4):
            if classify_score(scores[col]) != CLASS_DRAW:
                failures.append(f"col {col} should be drawing, got {scores[col]}")
        for col in (0, 1, 5, 6):
            if classify_score(scores[col]) != CLASS_LOSS:
                failures.append(f"col {col} should be losing, got {scores[col]}")
        for col in range(3):
            if scores[col] != scores[6 - col]:
                failures.append(
                    f"scores not symmetric: col {col}={scores[col]} "
                    f"vs col {6 - col}={scores[6 - col]}"
                )
        if self._agent.best_move(board) != 3:
            failures.append(
                f"best_move should be 3, got {self._agent.best_move(board)}"
            )

        if failures:
            raise RuntimeError(
                "bitbully calibration failed — solver conventions do not match "
                "expectations: " + "; ".join(failures)
            )
        logger.debug("bitbully calibration passed (empty-board theory verified)")


def solver_evaluate(
    model: Policy,
    opponent: Policy,
    scorer,
    env_id: str,
    config: "GameConfig | GameMetadata",
    num_games: int,
    seed: int,
    verbose: bool = False,
) -> SolverEvalResults:
    """Play games and score every model decision with the solver.

    The model plays the first num_games // 2 games as the first player and
    the rest as the second player (same convention as evaluate()). Each
    game reseeds the global RNGs with seed + game index, so runs are
    reproducible and every checkpoint faces the same conditions.
    """
    results = SolverEvalResults(
        env_id=env_id,
        model_name=model.name,
        model_path=getattr(model, "model_path", ""),
        step=infer_step_from_filename(getattr(model, "model_path", "")),
        opponent_name=opponent.name,
        games=num_games,
        seed=seed,
        temperature=getattr(model, "temperature", 0.0),
    )

    run_start = time.perf_counter()
    queries_before = scorer.queries
    hits_before = scorer.cache_hits
    solve_time_before = scorer.solve_time_seconds
    total_moves = 0

    for game_num in range(num_games):
        random.seed(seed + game_num)
        np.random.seed((seed + game_num) % 2**32)

        model_as = Player.FIRST if game_num < num_games // 2 else Player.SECOND
        seat_key = "first" if model_as == Player.FIRST else "second"
        scorer.reset()
        moves_so_far = 0

        def on_move(state: "GameState", action: int, policy: Policy) -> None:
            nonlocal moves_so_far
            if policy is model:
                judgment = judge_move(scorer.scores_for(state), action)
                results.overall.add(judgment)
                results.by_ply[ply_bucket(moves_so_far + 1)].add(judgment)
                results.by_seat[seat_key].add(judgment)
            scorer.mirror_move(action)
            moves_so_far += 1

        match = play_game(
            player1=model,
            player2=opponent,
            player1_as=model_as,
            env_id=env_id,
            config=config,
            verbose=verbose,
            on_move=on_move,
        )

        total_moves += match.moves
        if match.winner == 1:
            results.model_wins += 1
        elif match.winner == 2:
            results.model_losses += 1
        else:
            results.draws += 1

        if not verbose and (game_num + 1) % 10 == 0:
            logger.info(
                f"Progress: {game_num + 1}/{num_games} games, "
                f"{results.overall.positions} positions scored, "
                f"value-optimal {results.overall.value_optimal_rate:.1%}"
            )

    results.avg_game_length = total_moves / max(1, num_games)
    results.solver_queries = scorer.queries - queries_before
    results.solver_cache_hits = scorer.cache_hits - hits_before
    results.solver_time_seconds = scorer.solve_time_seconds - solve_time_before
    results.wall_time_seconds = time.perf_counter() - run_start
    results.timestamp = datetime.now().isoformat()
    return results


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
