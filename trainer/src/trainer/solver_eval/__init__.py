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

Results are aggregated overall, by ply bucket, and by seat, then appended
to data/solver_stats.json. Unlike win-rate-vs-random, these metrics have a
fixed, objective yardstick, so they are comparable across checkpoints.

Usage (defaults assume running from the repo root):
    python -m trainer solver-eval --model ./data/models/latest.onnx --games 100
    python -m trainer solver-eval --all-checkpoints --games 100

This module was split into a package; every name that was importable from
``trainer.solver_eval`` is re-exported here to keep existing imports working.
"""

from .cli import (
    add_solver_eval_arguments,
    append_solver_stats,
    discover_checkpoints,
    format_progression_table,
    main,
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
from .results import (
    CHECKPOINT_PATTERN,
    BucketStats,
    SolverEvalResults,
    infer_step_from_filename,
)
from .scorer import SolverScorer, solver_evaluate

__all__ = [
    "BLUNDER_DRAW_TO_LOSS",
    "BLUNDER_WIN_TO_DRAW",
    "BLUNDER_WIN_TO_LOSS",
    "CHECKPOINT_PATTERN",
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
    "append_solver_stats",
    "classify_score",
    "discover_checkpoints",
    "format_progression_table",
    "infer_step_from_filename",
    "judge_move",
    "main",
    "ply_bucket",
    "run_solver_evaluation",
    "solver_evaluate",
]
