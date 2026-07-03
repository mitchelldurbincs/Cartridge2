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
from .config import LoopConfig

if TYPE_CHECKING:
    from ..wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


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
        self._load_best_model_info()

    @property
    def best_iteration(self) -> int | None:
        """Get the iteration number of the current best model."""
        return self.best_model_iteration

    def _load_best_model_info(self) -> None:
        """Load best model info from disk if it exists."""
        if not self.config.best_model_info_path.exists():
            return

        try:
            with open(self.config.best_model_info_path) as f:
                data = json.load(f)
            self.best_model_iteration = data.get("iteration")
            logger.info(
                f"Loaded best model info: iteration {self.best_model_iteration}"
            )
        except Exception as e:
            logger.warning(f"Failed to load best model info: {e}")

    def _save_best_model_info(self, iteration: int, win_rate: float) -> None:
        """Save best model info to disk."""
        data = {
            "iteration": iteration,
            "win_rate_when_promoted": win_rate,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.config.best_model_info_path, "w") as f:
            json.dump(data, f, indent=2)

    def promote_to_best(self, iteration: int, win_rate: float) -> None:
        """Copy current model to best.onnx and update tracking.

        Args:
            iteration: The iteration number being promoted.
            win_rate: The win rate that qualified this model for promotion.
        """
        current_path = self.config.models_dir / "latest.onnx"
        if not current_path.exists():
            logger.warning("Cannot promote: latest.onnx not found")
            return

        shutil.copy(current_path, self.config.best_model_path)
        self.best_model_iteration = iteration
        self._save_best_model_info(iteration, win_rate)
        logger.info(
            f"New best model! Iteration {iteration} with {win_rate:.1%} win rate"
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
        model_path = self.config.models_dir / "latest.onnx"

        if not model_path.exists():
            logger.warning(f"Model not found for evaluation: {model_path}")
            return None, None, 0.0

        start_time = time.time()
        config = get_game_config(self.config.env_id)

        try:
            # Use small temperature for evaluation to avoid deterministic games
            # With temp=0, identical models play the exact same game 50 times
            eval_temperature = 0.2
            current_policy = OnnxPolicy(str(model_path), temperature=eval_temperature)

            # --- Model vs Best (Gatekeeper) Evaluation ---
            vs_best_win_rate = None
            vs_best_draw_rate = None
            became_new_best = False

            if not self.config.best_model_path.exists():
                # First iteration: current becomes best
                logger.info("No best model yet - promoting current model to best")
                self.promote_to_best(iteration, 1.0)
                vs_best_win_rate = 1.0
                vs_best_draw_rate = 0.0
                became_new_best = True
            else:
                # Evaluate against best
                logger.info(
                    f"Evaluating vs best model (iter {self.best_model_iteration}) "
                    f"- {self.config.eval_games} games..."
                )
                best_policy = OnnxPolicy(
                    str(self.config.best_model_path), temperature=eval_temperature
                )

                vs_best_results = run_eval(
                    player1=current_policy,
                    player2=best_policy,
                    env_id=self.config.env_id,
                    config=config,
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

                # Check if current model beats threshold
                if vs_best_win_rate > self.config.eval_win_threshold:
                    self.promote_to_best(iteration, vs_best_win_rate)
                    became_new_best = True
                else:
                    logger.info(
                        f"Current model did not beat threshold "
                        f"({vs_best_win_rate:.1%} <= {self.config.eval_win_threshold:.1%})"
                    )

            # --- Model vs Random Evaluation (optional) ---
            vs_random_win_rate = None
            vs_random_draw_rate = None

            if self.config.eval_vs_random:
                logger.info(f"Evaluating vs random - {self.config.eval_games} games...")
                random_policy = RandomPolicy()

                vs_random_results = run_eval(
                    player1=current_policy,
                    player2=random_policy,
                    env_id=self.config.env_id,
                    config=config,
                    num_games=self.config.eval_games,
                    verbose=False,
                )

                vs_random_win_rate = vs_random_results.player1_win_rate
                vs_random_draw_rate = vs_random_results.draw_rate

                logger.info(
                    f"vs Random: Win {vs_random_win_rate:.1%}, "
                    f"Draw {vs_random_draw_rate:.1%}, "
                    f"Loss {vs_random_results.player2_win_rate:.1%}"
                )

            elapsed = time.time() - start_time

            # Save to eval history (includes both evaluations)
            eval_record = {
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
                # Metadata
                "games": self.config.eval_games,
                "timestamp": datetime.now().isoformat(),
            }
            eval_history.append(eval_record)

            if self.wandb_logger is not None:
                wandb_metrics = {
                    "eval/vs_best_win_rate": vs_best_win_rate,
                    "eval/vs_best_draw_rate": vs_best_draw_rate,
                    "eval/became_new_best": int(became_new_best),
                    "eval/best_model_iteration": self.best_model_iteration,
                    "eval/vs_random_win_rate": vs_random_win_rate,
                    "eval/vs_random_draw_rate": vs_random_draw_rate,
                }
                self.wandb_logger.log(
                    {k: v for k, v in wandb_metrics.items() if v is not None},
                    step=iteration * self.config.steps_per_iteration,
                )

            return vs_best_win_rate, vs_best_draw_rate, elapsed

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback

            traceback.print_exc()
            return None, None, time.time() - start_time
