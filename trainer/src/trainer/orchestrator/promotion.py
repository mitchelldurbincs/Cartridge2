"""Promotion decisions and best-model (gatekeeper) bookkeeping.

The :func:`should_promote` pure decision function plus ``PromotionMixin``,
which is mixed into ``EvalRunner`` and relies on attributes/methods defined
there (``self.config``, ``self.best_solver_rate``, ``self._solver_enabled``,
``self._run_solver_eval``, ...).
"""

import json
import logging
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


def should_promote(
    *,
    promotion_metric: str,
    vs_best_win_rate: float | None,
    win_threshold: float,
    candidate_solver_rate: float | None,
    best_solver_rate: float | None,
    margin: float,
) -> tuple[bool, str]:
    """Decide whether the candidate model becomes the new best.

    Pure function: returns (promote, human-readable reason). The
    "solver_optimal" metric falls back to the win_rate rule when either
    solver rate is unavailable.
    """

    def win_rate_rule(prefix: str = "") -> tuple[bool, str]:
        if vs_best_win_rate is None:
            return False, prefix + "win_rate: no vs-best result, not promoting"
        if vs_best_win_rate > win_threshold:
            return True, (
                prefix
                + f"win_rate: {vs_best_win_rate:.1%} > threshold {win_threshold:.1%}"
            )
        return False, (
            prefix
            + f"win_rate: {vs_best_win_rate:.1%} <= threshold {win_threshold:.1%}"
        )

    if promotion_metric == "solver_optimal":
        if candidate_solver_rate is None or best_solver_rate is None:
            missing = "candidate" if candidate_solver_rate is None else "best"
            return win_rate_rule(
                f"solver_optimal unavailable ({missing} rate missing), falling back to "
            )
        if candidate_solver_rate > best_solver_rate + margin:
            return True, (
                f"solver_optimal: candidate {candidate_solver_rate:.1%} > "
                f"best {best_solver_rate:.1%} + margin {margin:.1%}"
            )
        return False, (
            f"solver_optimal: candidate {candidate_solver_rate:.1%} <= "
            f"best {best_solver_rate:.1%} + margin {margin:.1%}"
        )

    if promotion_metric != "win_rate":
        return win_rate_rule(f"unknown metric {promotion_metric!r}, falling back to ")
    return win_rate_rule()


class PromotionMixin:
    """Best-model tracking and promotion bookkeeping for ``EvalRunner``."""

    def _load_best_model_info(self) -> None:
        """Load best model info from disk if it exists.

        Legacy files (pre solver-eval) lack the solver rate — read with
        .get() so they load fine and the rate backfills lazily.
        """
        if not self.config.best_model_info_path.exists():
            return

        try:
            with open(self.config.best_model_info_path) as f:
                data = json.load(f)
            self.best_model_iteration = data.get("iteration")
            self.best_solver_rate = data.get("solver_value_optimal_rate")
            logger.info(
                f"Loaded best model info: iteration {self.best_model_iteration}"
            )
        except Exception as e:
            logger.warning(f"Failed to load best model info: {e}")

    def _save_best_model_info(
        self,
        iteration: int,
        win_rate: float,
        solver_rate: float | None = None,
        metric: str = "win_rate",
    ) -> None:
        """Save best model info to disk."""
        data = {
            "iteration": iteration,
            "win_rate_when_promoted": win_rate,
            "solver_value_optimal_rate": solver_rate,
            "promotion_metric": metric,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.config.best_model_info_path, "w") as f:
            json.dump(data, f, indent=2)

    def promote_to_best(
        self,
        iteration: int,
        win_rate: float,
        solver_rate: float | None = None,
        metric: str = "win_rate",
    ) -> None:
        """Copy current model to best.onnx and update tracking.

        Args:
            iteration: The iteration number being promoted.
            win_rate: The win rate that qualified this model for promotion.
            solver_rate: The candidate's solver value-optimal rate, when
                available (recorded regardless of the metric used, so a
                later switch to solver_optimal needs no backfill).
            metric: The promotion metric that made this decision.
        """
        current_path = self.config.latest_model_path
        if not current_path.exists():
            logger.warning("Cannot promote: latest.onnx not found")
            return

        shutil.copy(current_path, self.config.best_model_path)
        self.best_model_iteration = iteration
        self.best_solver_rate = solver_rate
        self._save_best_model_info(iteration, win_rate, solver_rate, metric)
        logger.info(
            f"New best model! Iteration {iteration} with {win_rate:.1%} win rate"
            + (
                f", solver value-optimal {solver_rate:.1%}"
                if solver_rate is not None
                else ""
            )
        )

    def _effective_promotion_metric(self) -> str:
        """Resolve the promotion metric, falling back when solver eval is off."""
        metric = self.config.promotion_metric
        if metric == "solver_optimal" and not self._solver_enabled():
            if not self._promotion_fallback_warned:
                logger.warning(
                    "promotion_metric=solver_optimal but solver eval is "
                    f"unavailable (env={self.config.env_id}, "
                    f"solver_games={self.config.solver_games}, "
                    f"disabled={self._solver_disabled}); using win_rate"
                )
                self._promotion_fallback_warned = True
            return "win_rate"
        return metric

    def _ensure_best_solver_rate(self, game_config) -> float | None:
        """Backfill the best model's solver rate for legacy best_model.json.

        Evaluates best.onnx once with the same scorer, seed, and game count
        as the candidate, so the comparison is same-conditions.
        """
        if self.best_solver_rate is not None:
            return self.best_solver_rate
        if not self.config.best_model_path.exists() or not self._solver_enabled():
            return None

        logger.info("Backfilling solver rate for existing best model...")
        best_results = self._run_solver_eval(self.config.best_model_path, game_config)
        if best_results is None:
            return None

        self.best_solver_rate = best_results.overall.value_optimal_rate
        # Preserve existing metadata fields while adding the rate.
        data = {}
        if self.config.best_model_info_path.exists():
            try:
                with open(self.config.best_model_info_path) as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data["solver_value_optimal_rate"] = self.best_solver_rate
        with open(self.config.best_model_info_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"Best model solver value-optimal rate: {self.best_solver_rate:.1%}"
        )
        return self.best_solver_rate
