"""Evaluation and best model tracking for the orchestrator.

This module handles running evaluations and managing the "gatekeeper" best model.
"""

import json
import logging
import shutil
import time
from datetime import datetime
from typing import TYPE_CHECKING

from ..evaluator import OnnxPolicy, RandomPolicy
from ..evaluator import evaluate as run_eval
from ..game_config import get_config as get_game_config
from ..solver_eval import SolverScorer, append_solver_stats, solver_evaluate
from .config import LoopConfig

if TYPE_CHECKING:
    from ..solver_eval import SolverEvalResults
    from ..wandb_logger import WandbLogger

logger = logging.getLogger(__name__)

# Head-to-head evaluation uses a small sampling temperature: with temperature
# 0, two deterministic policies would replay the exact same game every time.
EVAL_TEMPERATURE = 0.2


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


class EvalRunner:
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

    def _load_best_model_info(self) -> None:
        """Load best model info from disk if it exists.

        best_model.json may lack the solver rate — either written before
        solver eval existed, or promoted while solver eval was disabled or
        unavailable. This is a permanent tolerance, not a transition shim:
        read with .get() so such files load fine and the rate backfills
        lazily via _ensure_best_solver_rate.
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
        """Backfill the best model's solver rate when best_model.json lacks it.

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
