"""Solver-based move-quality evaluation for Connect4.

Connect4 is a solved game, so every model decision can be scored against
ground truth. This package plays evaluation games (model vs random, both
seats, deterministic seeding) and, for each model decision, queries the
bitbully perfect solver to classify the chosen move:

- value-optimal: the move preserves the game-theoretic value class
  (win/draw/loss from the mover's perspective)
- blunder: the move drops the value class (win->draw, win->loss, draw->loss)
- exact-best: the move is in the argmax set of solver scores
  (fastest win / slowest loss)

Results are aggregated overall, by ply bucket, and by seat. The standalone
command is diagnostic and writes no mutable authority; synchronized solver
evidence is persisted only inside immutable evaluation artifacts.

Usage (defaults assume running from the repo root):
    python -m trainer solver-eval --model ./candidate.onnx --games 100
    python -m trainer solver-eval --all-checkpoints --games 100

The package exports its command, scoring, judgment, and result contracts from
this canonical surface.
"""

from .cli import (
    add_solver_eval_arguments,
    format_progression_table,
    run_solver_evaluation,
)
from .judgment import (
    BLUNDER_DRAW_TO_LOSS,
    BLUNDER_WIN_TO_DRAW,
    BLUNDER_WIN_TO_LOSS,
    CLASS_DRAW,
    CLASS_LOSS,
    CLASS_WIN,
    PLY_BUCKETS,
    SEATS,
    MoveJudgment,
    classify_score,
    judge_move,
    ply_bucket,
)
from .results import BucketStats, SolverEvalResults
from .scorer import SolverScorer, solver_evaluate

__all__ = [
    "BLUNDER_DRAW_TO_LOSS",
    "BLUNDER_WIN_TO_DRAW",
    "BLUNDER_WIN_TO_LOSS",
    "CLASS_DRAW",
    "CLASS_LOSS",
    "CLASS_WIN",
    "PLY_BUCKETS",
    "SEATS",
    "BucketStats",
    "MoveJudgment",
    "SolverEvalResults",
    "SolverScorer",
    "add_solver_eval_arguments",
    "classify_score",
    "format_progression_table",
    "judge_move",
    "ply_bucket",
    "run_solver_evaluation",
    "solver_evaluate",
]
