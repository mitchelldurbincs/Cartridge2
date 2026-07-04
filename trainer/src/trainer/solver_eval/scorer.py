"""bitbully solver wrapper and the game-playing evaluation driver.

SolverScorer mirrors the game into a bitbully board and memoizes solved
positions; solver_evaluate plays evaluation games (model vs random, both
seats, deterministic seeding) and judges every model decision.
"""

import logging
import random
import time
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ..evaluator import play_game
from ..games import Player
from ..policies import Policy
from .judgment import (
    CLASS_DRAW,
    CLASS_LOSS,
    CLASS_WIN,
    classify_score,
    judge_move,
    ply_bucket,
)
from .results import SolverEvalResults, infer_step_from_filename

if TYPE_CHECKING:
    from ..game_config import GameConfig
    from ..games import GameState
    from ..storage import GameMetadata

logger = logging.getLogger(__name__)


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
