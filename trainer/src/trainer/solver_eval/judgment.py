"""Per-move solver judgment: classify a chosen Connect4 move against ground truth.

Connect4 is a solved game, so every model decision can be scored against the
bitbully perfect solver. This module holds the pure classification logic:
turning a solver score into a win/draw/loss class, bucketing a move by ply,
and judging a chosen move against the scores of all legal moves.
"""

from dataclasses import dataclass

CLASS_WIN = "win"
CLASS_DRAW = "draw"
CLASS_LOSS = "loss"

BLUNDER_WIN_TO_DRAW = "win_to_draw"
BLUNDER_WIN_TO_LOSS = "win_to_loss"
BLUNDER_DRAW_TO_LOSS = "draw_to_loss"

PLY_BUCKETS = ("ply_1_8", "ply_9_20", "ply_21_plus")
SEATS = ("first", "second")


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
