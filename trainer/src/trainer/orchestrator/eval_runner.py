"""Evaluation and best model tracking for the orchestrator.

This module handles running evaluations and managing the "gatekeeper" best model.
"""

import logging
import time
from typing import TYPE_CHECKING

from ..evaluator import OnnxPolicy, RandomPolicy
from ..evaluator import evaluate as run_eval
from ..game_config import get_config as get_game_config
from ..solver_eval import SolverScorer, solver_evaluate
from .config import LoopConfig
from .eval_reporting import EvalReportingMixin
from .promotion import PromotionMixin, should_promote

if TYPE_CHECKING:
    from ..solver_eval import SolverEvalResults
    from ..wandb_logger import WandbLogger

# Re-exported so existing imports (and tests) keep working.
__all__ = ["EvalRunner", "should_promote"]

logger = logging.getLogger(__name__)

# Head-to-head evaluation uses a small sampling temperature: with temperature
# 0, two deterministic policies would replay the exact same game every time.
EVAL_TEMPERATURE = 0.2


class EvalRunner(PromotionMixin, EvalReportingMixin):
    """Manages model evaluation and best model (gatekeeper) tracking."""

    def __init__(self, config: LoopConfig, wandb_logger: "WandbLogger | None" = None):
        """Initialize the evaluation runner.

        Args:
            config: Loop configuration.
            wandb_logger: Optional W&B logger for eval metrics.
        """
        self.config = config
        self.wandb_logger = wandb_logger
        self.best_model_iteration: int | None = None
        self.best_solver_rate: float | None = None
        # Perfect-solver scoring: one scorer for the whole run (its position
        # cache persists across iterations), disabled permanently on failure.
        self._solver_scorer: SolverScorer | None = None
        self._solver_disabled: bool = False
        self._promotion_fallback_warned: bool = False
        self._load_best_model_info()

    def _solver_enabled(self) -> bool:
        """Solver eval runs only for connect4, when configured and healthy."""
        return (
            self.config.env_id == "connect4"
            and self.config.solver_games > 0
            and not self._solver_disabled
        )

    def _run_solver_eval(self, model_path, game_config) -> "SolverEvalResults | None":
        """Score the candidate model's moves against the perfect solver.

        Fail-soft: any solver problem (bitbully missing, calibration or
        desync failure) logs a warning and disables solver eval for the
        rest of the run — training must never crash over metrics.
        """
        if not self._solver_enabled():
            return None

        try:
            if self._solver_scorer is None:
                self._solver_scorer = SolverScorer()
            model = OnnxPolicy(str(model_path), temperature=0.0)
            results = solver_evaluate(
                model=model,
                opponent=RandomPolicy(),
                scorer=self._solver_scorer,
                env_id=self.config.env_id,
                config=game_config,
                num_games=self.config.solver_games,
                seed=self.config.solver_seed,
            )
            logger.info(
                f"Solver eval: value-optimal {results.overall.value_optimal_rate:.1%}, "
                f"exact-best {results.overall.exact_best_rate:.1%}, "
                f"blunders {results.overall.blunder_rate:.1%} "
                f"({results.overall.positions} positions)"
            )
            return results
        except (ImportError, RuntimeError) as e:
            logger.warning(f"Solver eval unavailable, disabling for this run: {e}")
            self._solver_disabled = True
            return None

    @property
    def best_iteration(self) -> int | None:
        """Get the iteration number of the current best model."""
        return self.best_model_iteration

    def _evaluate_vs_best(
        self,
        iteration: int,
        current_policy,
        game_config,
        candidate_solver_rate: float | None,
        promotion_metric: str,
    ) -> tuple[float, float, bool]:
        """Run the gatekeeper evaluation and apply the promotion decision.

        The very first evaluation has no best model to play against: the
        candidate is promoted unconditionally and credited with a 100% win
        rate. Otherwise the candidate plays against best.onnx and
        ``should_promote`` decides using the effective promotion metric.

        Returns:
            Tuple of (win_rate_vs_best, draw_rate_vs_best, became_new_best).
        """
        if not self.config.best_model_path.exists():
            # First iteration: current becomes best
            logger.info("No best model yet - promoting current model to best")
            self.promote_to_best(
                iteration,
                1.0,
                solver_rate=candidate_solver_rate,
                metric=promotion_metric,
            )
            return 1.0, 0.0, True

        logger.info(
            f"Evaluating vs best model (iter {self.best_model_iteration}) "
            f"- {self.config.eval_games} games..."
        )
        best_policy = OnnxPolicy(
            str(self.config.best_model_path), temperature=EVAL_TEMPERATURE
        )

        vs_best_results = run_eval(
            player1=current_policy,
            player2=best_policy,
            env_id=self.config.env_id,
            config=game_config,
            num_games=self.config.eval_games,
            verbose=False,
        )

        vs_best_win_rate = vs_best_results.player1_win_rate
        vs_best_draw_rate = vs_best_results.draw_rate

        logger.info(
            f"vs Best (iter {self.best_model_iteration}): "
            f"Win {vs_best_win_rate:.1%}, Draw {vs_best_draw_rate:.1%}, "
            f"Loss {vs_best_results.player2_win_rate:.1%}"
        )

        # Promotion decision (win_rate or solver_optimal)
        best_solver_rate = (
            self._ensure_best_solver_rate(game_config)
            if promotion_metric == "solver_optimal"
            else self.best_solver_rate
        )
        promote, reason = should_promote(
            promotion_metric=promotion_metric,
            vs_best_win_rate=vs_best_win_rate,
            win_threshold=self.config.eval_win_threshold,
            candidate_solver_rate=candidate_solver_rate,
            best_solver_rate=best_solver_rate,
            margin=self.config.promotion_margin,
        )
        logger.info(f"Promotion decision: {reason}")
        if promote:
            self.promote_to_best(
                iteration,
                vs_best_win_rate,
                solver_rate=candidate_solver_rate,
                metric=promotion_metric,
            )
        return vs_best_win_rate, vs_best_draw_rate, promote

    def _evaluate_vs_random(
        self, current_policy, game_config
    ) -> tuple[float | None, float | None]:
        """Evaluate the candidate against the random baseline, if enabled.

        Returns:
            Tuple of (win_rate, draw_rate); both None when disabled.
        """
        if not self.config.eval_vs_random:
            return None, None

        logger.info(f"Evaluating vs random - {self.config.eval_games} games...")
        vs_random_results = run_eval(
            player1=current_policy,
            player2=RandomPolicy(),
            env_id=self.config.env_id,
            config=game_config,
            num_games=self.config.eval_games,
            verbose=False,
        )

        logger.info(
            f"vs Random: Win {vs_random_results.player1_win_rate:.1%}, "
            f"Draw {vs_random_results.draw_rate:.1%}, "
            f"Loss {vs_random_results.player2_win_rate:.1%}"
        )
        return vs_random_results.player1_win_rate, vs_random_results.draw_rate

    def run(
        self, iteration: int, eval_history: list[dict]
    ) -> tuple[float | None, float | None, float]:
        """Run evaluation against best model and optionally random baseline.

        Args:
            iteration: Current training iteration.
            eval_history: List to append evaluation records to.

        Returns:
            Tuple of (win_rate_vs_best, draw_rate_vs_best, elapsed_seconds).
        """
        model_path = self.config.latest_model_path

        if not model_path.exists():
            logger.warning(f"Model not found for evaluation: {model_path}")
            return None, None, 0.0

        start_time = time.time()
        game_config = get_game_config(self.config.env_id)

        try:
            current_policy = OnnxPolicy(str(model_path), temperature=EVAL_TEMPERATURE)

            # Solver scoring (connect4 only) runs before the promotion
            # decision so the candidate's rate can drive it and gets
            # recorded even on the very first promotion.
            solver_results = self._run_solver_eval(model_path, game_config)
            candidate_solver_rate = (
                solver_results.overall.value_optimal_rate if solver_results else None
            )
            promotion_metric = self._effective_promotion_metric()

            vs_best_win_rate, vs_best_draw_rate, became_new_best = (
                self._evaluate_vs_best(
                    iteration,
                    current_policy,
                    game_config,
                    candidate_solver_rate=candidate_solver_rate,
                    promotion_metric=promotion_metric,
                )
            )

            vs_random_win_rate, vs_random_draw_rate = self._evaluate_vs_random(
                current_policy, game_config
            )

            elapsed = time.time() - start_time

            eval_record = self._build_eval_record(
                iteration,
                vs_best_win_rate=vs_best_win_rate,
                vs_best_draw_rate=vs_best_draw_rate,
                became_new_best=became_new_best,
                vs_random_win_rate=vs_random_win_rate,
                vs_random_draw_rate=vs_random_draw_rate,
                solver_results=solver_results,
                promotion_metric=promotion_metric,
            )
            eval_history.append(eval_record)

            self._append_solver_history(solver_results, iteration)
            self._log_eval_to_wandb(eval_record, solver_results)

            return vs_best_win_rate, vs_best_draw_rate, elapsed

        except Exception as e:
            logger.exception(f"Evaluation failed: {e}")
            return None, None, time.time() - start_time
