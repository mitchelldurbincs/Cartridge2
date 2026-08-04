"""bitbully solver wrapper and the move-scoring driver.

The engine plays; Python judges. ``cartridge-eval`` plays the games and dumps
every decision it made, and ``solver_evaluate`` replays that dump through the
bitbully perfect solver, scoring each of the model's moves.

The split matters for correctness, not just tidiness. This used to play the
games itself against a Python reimplementation of Connect 4 and cross-check the
solver against *that* board — so if the Python rules had drifted from the
engine's, solver eval would have confidently scored a game nobody was training
on. The desync check below now compares the solver's mirrored board against the
engine's own view of the position.
"""

import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path

from ..algorithms import get_algorithm
from ..environment_catalog import get_environment
from ..evaluator import build_eval_command, run_eval_binary
from ..players import ModelPlayer, RandomPlayer
from .judgment import (
    CLASS_DRAW,
    CLASS_LOSS,
    CLASS_WIN,
    classify_score,
    judge_move,
    ply_bucket,
)
from .results import SolverEvalResults

logger = logging.getLogger(__name__)

# Connect 4 board dimensions, needed to reindex between bitbully's
# column-major array and the engine's row-major board view.
_CONNECT4_WIDTH = 7
_CONNECT4_HEIGHT = 6


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

    def _mirrored_board_in_engine_order(self) -> list[int]:
        """The mirrored bitbully board, reindexed the way the engine reports it.

        bitbully's ``to_array()`` is ``[col][row]`` with row 0 at the bottom;
        the engine's Connect 4 board view is row-major, ``row * WIDTH + col``,
        also with row 0 at the bottom. Same cell values (1/2 for first/second
        player), different flattening — so the two are only comparable after
        this reindex.

        This conversion is why the desync check is worth having. The Python
        Connect 4 mirror this driver used to play against was itself
        column-major, so the old check compared bitbully to a copy of the rules
        that disagreed with the engine about board layout; nothing compared
        either of them to the engine.
        """
        columns = self._board.to_array()
        return [
            int(columns[col][row])
            for row in range(_CONNECT4_HEIGHT)
            for col in range(_CONNECT4_WIDTH)
        ]

    def scores_for(self, board: list[int], current_player: int, legal: list[int]) -> dict[int, int]:
        """Solver scores for every legal move in the given position.

        Args:
            board: Cell owners as the engine reports them, row-major.
            current_player: Side to move (1 or 2).
            legal: Legal columns in this position.

        Scores are from the perspective of the side to move: positive is
        winning, zero drawing, negative losing.
        """
        mirrored = self._mirrored_board_in_engine_order()
        if mirrored != list(board):
            raise RuntimeError(
                f"Solver board desynced from the engine position: mirrored board "
                f"{mirrored} != engine board {list(board)}"
            )
        state_legal = sorted(legal)

        self.queries += 1
        key = (tuple(board), int(current_player))
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
            failures.append(f"best_move should be 3, got {self._agent.best_move(board)}")

        if failures:
            raise RuntimeError(
                "bitbully calibration failed — solver conventions do not match "
                "expectations: " + "; ".join(failures)
            )
        logger.debug("bitbully calibration passed (empty-board theory verified)")


def solver_evaluate(
    model: ModelPlayer,
    opponent: ModelPlayer | RandomPlayer,
    scorer,
    algorithm_id: str,
    env_id: str,
    num_games: int = 100,
    seed: int = 42,
    verbose: bool = False,
    checkpoint_id: str | None = None,
    checkpoint_step: int | None = None,
) -> SolverEvalResults:
    """Play games through the engine and score every model decision.

    The model takes seat 1 on even game indices and seat 2 on odd indices —
    the binary's deterministic alternating schedule, also used by
    ``evaluate()``. Game N uses ``seed + N``, so runs are reproducible and
    every checkpoint faces identical conditions.

    Args:
        model: The model whose moves are judged; it plays as player 1.
        opponent: Its opponent, normally the random baseline.
        scorer: A ``SolverScorer``.
        algorithm_id: Algorithm cartridge ID.
        env_id: Environment ID (connect4 — the only solved game here).
        num_games: Games to play.
        seed: Base RNG seed.
        verbose: Log every move played.
        checkpoint_id: Immutable manifest ID, when the model came from a
            checkpoint repository. Direct one-off ONNX paths leave this null.
        checkpoint_step: Manifest training step. Never inferred from a filename.
    """
    algorithm = get_algorithm(algorithm_id)
    algorithm.compatibility(get_environment(env_id)).require_compatible()

    results = SolverEvalResults(
        env_id=env_id,
        model_name=model.name,
        model_path=model.model_path,
        checkpoint_id=checkpoint_id,
        step=checkpoint_step,
        opponent_name=opponent.name,
        games=num_games,
        seed=seed,
        temperature=model.temperature,
    )

    run_start = time.perf_counter()
    queries_before = scorer.queries
    hits_before = scorer.cache_hits
    solve_time_before = scorer.solve_time_seconds

    summary, positions = _play_and_dump(model, opponent, algorithm_id, env_id, num_games, seed)

    current_game: int | None = None
    for record in positions:
        # Records arrive in play order, so a new game index means a new game.
        if record["game"] != current_game:
            current_game = record["game"]
            scorer.reset()

        if record["by"] == "p1":
            scores = scorer.scores_for(record["board"], record["player"], record["legal"])
            judgment = judge_move(scores, record["action"])
            results.overall.add(judgment)
            results.by_ply[ply_bucket(record["ply"] + 1)].add(judgment)
            # The model is player 1, so the seat it holds this game is simply
            # the seat to move on its own turns.
            results.by_seat["first" if record["player"] == 1 else "second"].add(judgment)

        scorer.mirror_move(record["action"])

        if verbose:
            logger.info(
                f"game {record['game']} ply {record['ply']}: "
                f"{record['by']} played {record['action']}"
            )

    results.model_wins = summary["player1_wins"]
    results.model_losses = summary["player2_wins"]
    results.draws = summary["draws"]
    results.avg_game_length = summary["avg_game_length"]
    results.solver_queries = scorer.queries - queries_before
    results.solver_cache_hits = scorer.cache_hits - hits_before
    results.solver_time_seconds = scorer.solve_time_seconds - solve_time_before
    results.wall_time_seconds = time.perf_counter() - run_start
    results.timestamp = datetime.now().isoformat()

    logger.info(
        f"Scored {results.overall.positions} positions over {num_games} games, "
        f"value-optimal {results.overall.value_optimal_rate:.1%}"
    )
    return results


def _play_and_dump(
    model: ModelPlayer,
    opponent: ModelPlayer | RandomPlayer,
    algorithm_id: str,
    env_id: str,
    num_games: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    """Run the games and return (summary, every decision made)."""
    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "eval.json"
        dump_path = Path(tmp) / "positions.jsonl"
        run_eval_binary(
            build_eval_command(
                model,
                opponent,
                algorithm_id,
                env_id,
                num_games,
                seed,
                output_path,
                dump_positions=dump_path,
            )
        )
        summary = json.loads(output_path.read_text())
        positions = [json.loads(line) for line in dump_path.read_text().splitlines() if line]
    return summary, positions
