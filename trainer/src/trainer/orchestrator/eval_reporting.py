"""Eval-record building and logging for the orchestrator.

``EvalReportingMixin`` is mixed into ``EvalRunner`` and relies on attributes
defined there (``self.config``, ``self.best_model_iteration``,
``self.wandb_logger``).
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from ..solver_eval import append_solver_stats

if TYPE_CHECKING:
    from ..solver_eval import SolverEvalResults

logger = logging.getLogger(__name__)


class EvalReportingMixin:
    """Eval-history record assembly, solver history, and W&B logging."""

    def _build_eval_record(
        self,
        iteration: int,
        *,
        vs_best_win_rate: float,
        vs_best_draw_rate: float,
        became_new_best: bool,
        vs_random_win_rate: float | None,
        vs_random_draw_rate: float | None,
        solver_results: "SolverEvalResults | None",
        promotion_metric: str,
    ) -> dict:
        """Assemble one eval-history record.

        The key set is a frozen schema: eval_stats.json is read by the web
        frontend (via StatsManager) — do not rename or drop keys.
        """
        return {
            "iteration": iteration,
            "step": iteration * self.config.steps_per_iteration,
            # Model vs Best results
            "vs_best_win_rate": vs_best_win_rate,
            "vs_best_draw_rate": vs_best_draw_rate,
            "vs_best_opponent_iteration": self.best_model_iteration,
            "became_new_best": became_new_best,
            # Model vs Random results (if enabled)
            "vs_random_win_rate": vs_random_win_rate,
            "vs_random_draw_rate": vs_random_draw_rate,
            # Perfect-solver move scoring (None when skipped/unavailable)
            "solver_value_optimal_rate": (
                solver_results.overall.value_optimal_rate if solver_results else None
            ),
            "solver_exact_best_rate": (
                solver_results.overall.exact_best_rate if solver_results else None
            ),
            "solver_blunder_rate": (
                solver_results.overall.blunder_rate if solver_results else None
            ),
            "solver_positions": (
                solver_results.overall.positions if solver_results else None
            ),
            "promotion_metric": promotion_metric,
            # Metadata
            "games": self.config.eval_games,
            "timestamp": datetime.now().isoformat(),
        }

    def _append_solver_history(
        self, solver_results: "SolverEvalResults | None", iteration: int
    ) -> None:
        """Append this iteration's solver results to solver_stats.json."""
        if solver_results is None:
            return

        solver_entry = solver_results.to_dict()
        solver_entry.update(
            {
                "iteration": iteration,
                "global_step": iteration * self.config.steps_per_iteration,
                "context": "loop",
            }
        )
        append_solver_stats(solver_entry, self.config.solver_stats_path)

    def _log_eval_to_wandb(
        self, eval_record: dict, solver_results: "SolverEvalResults | None"
    ) -> None:
        """Mirror the eval record (plus solver detail) to W&B, dropping Nones."""
        if self.wandb_logger is None:
            return

        wandb_metrics = {
            "eval/vs_best_win_rate": eval_record["vs_best_win_rate"],
            "eval/vs_best_draw_rate": eval_record["vs_best_draw_rate"],
            "eval/became_new_best": int(eval_record["became_new_best"]),
            "eval/best_model_iteration": self.best_model_iteration,
            "eval/vs_random_win_rate": eval_record["vs_random_win_rate"],
            "eval/vs_random_draw_rate": eval_record["vs_random_draw_rate"],
        }
        if solver_results is not None:
            overall = solver_results.overall
            wandb_metrics.update(
                {
                    "solver/value_optimal_rate": overall.value_optimal_rate,
                    "solver/exact_best_rate": overall.exact_best_rate,
                    "solver/blunder_rate": overall.blunder_rate,
                    "solver/blunders_win_to_draw": overall.blunders_win_to_draw,
                    "solver/blunders_win_to_loss": overall.blunders_win_to_loss,
                    "solver/blunders_draw_to_loss": overall.blunders_draw_to_loss,
                    "solver/positions_scored": overall.positions,
                    "solver/model_win_rate_vs_random": (
                        solver_results.model_wins / solver_results.games
                        if solver_results.games
                        else 0.0
                    ),
                    "solver/eval_seconds": solver_results.wall_time_seconds,
                }
            )
        self.wandb_logger.log(
            {k: v for k, v in wandb_metrics.items() if v is not None},
            step=eval_record["step"],
        )
