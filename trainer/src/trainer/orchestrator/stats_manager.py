"""Stats management for the orchestrator.

This module handles saving and loading of training statistics including:
- loop_stats.json: Iteration history
- eval_stats.json: Evaluation history
- stats.json: Frontend-readable stats
"""

import json
import logging
import time

from .config import IterationStats, LoopConfig

logger = logging.getLogger(__name__)


class StatsManager:
    """Manages saving and loading of training statistics."""

    def __init__(self, config: LoopConfig):
        self.config = config

    def save_loop_stats(self, history: list[IterationStats]) -> None:
        """Save iteration history to loop_stats.json."""
        stats = {
            "config": {
                "env_id": self.config.env_id,
                "episodes_per_iteration": self.config.episodes_per_iteration,
                "steps_per_iteration": self.config.steps_per_iteration,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "eval_interval": self.config.eval_interval,
                "eval_games": self.config.eval_games,
            },
            "iterations": [
                {
                    "iteration": s.iteration,
                    "episodes": s.episodes_generated,
                    "transitions": s.transitions_generated,
                    "steps": s.training_steps,
                    "actor_time": s.actor_time_seconds,
                    "trainer_time": s.trainer_time_seconds,
                    "eval_time": s.eval_time_seconds,
                    "total_time": s.total_time_seconds,
                    "eval_win_rate": s.eval_win_rate,
                    "eval_draw_rate": s.eval_draw_rate,
                    "timestamp": s.timestamp,
                }
                for s in history
            ],
        }

        with open(self.config.loop_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    def save_eval_stats(self, eval_history: list[dict]) -> None:
        """Save evaluation history to eval_stats.json."""
        with open(self.config.eval_stats_path, "w") as f:
            json.dump({"evaluations": eval_history}, f, indent=2)

    def _format_eval_for_frontend(self, record: dict) -> dict:
        """Convert one eval-history record to the shape the frontend expects."""
        return {
            "step": record.get(
                "step",
                record.get("iteration", 0) * self.config.steps_per_iteration,
            ),
            "current_iteration": record.get("iteration", 0),
            # Model vs Best results
            "opponent": "best",
            "opponent_iteration": record.get("vs_best_opponent_iteration"),
            "win_rate": record.get("vs_best_win_rate", 0.0),
            "draw_rate": record.get("vs_best_draw_rate", 0.0),
            "loss_rate": 1.0
            - record.get("vs_best_win_rate", 0.0)
            - record.get("vs_best_draw_rate", 0.0),
            "became_new_best": record.get("became_new_best", False),
            # Model vs Random results
            "vs_random_win_rate": record.get("vs_random_win_rate"),
            "vs_random_draw_rate": record.get("vs_random_draw_rate"),
            # Metadata
            "games_played": record.get("games", 0),
            "timestamp": time.time(),
        }

    def update_stats_with_eval(
        self, eval_history: list[dict], best_iteration: int | None
    ) -> None:
        """Update stats.json with evaluation results for frontend display.

        The frontend reads stats.json and expects `last_eval`, `eval_history`,
        and `best_model` fields to display evaluation metrics.
        """
        if not eval_history:
            return

        try:
            # Read existing stats.json
            stats_data = {}
            if self.config.stats_path.exists():
                with open(self.config.stats_path) as f:
                    stats_data = json.load(f)

            # Convert eval_history to the format expected by frontend
            formatted_history = [
                self._format_eval_for_frontend(record) for record in eval_history
            ]
            stats_data["last_eval"] = formatted_history[-1]
            stats_data["eval_history"] = formatted_history

            # Add best model info
            if best_iteration is not None:
                stats_data["best_model"] = {
                    "iteration": best_iteration,
                    "step": best_iteration * self.config.steps_per_iteration,
                }

            # Write back atomically
            temp_path = self.config.stats_path.with_suffix(".json.tmp")
            with open(temp_path, "w") as f:
                json.dump(stats_data, f, indent=2)
            temp_path.replace(self.config.stats_path)

            logger.debug(
                f"Updated stats.json with {len(formatted_history)} eval records"
            )

        except Exception as e:
            logger.warning(f"Failed to update stats.json with eval results: {e}")

    def load_previous_state(
        self,
    ) -> tuple[list[IterationStats], list[dict], int]:
        """Load previous training state from disk for auto-resume.

        Returns:
            Tuple of (iteration_history, eval_history, start_iteration).
            If no previous state exists, returns empty lists and start_iteration=1.
        """
        iteration_history: list[IterationStats] = []
        eval_history: list[dict] = []
        start_iteration = 1

        # Check if loop_stats.json exists
        if not self.config.loop_stats_path.exists():
            logger.debug("No loop_stats.json found, starting fresh")
            return iteration_history, eval_history, start_iteration

        try:
            with open(self.config.loop_stats_path) as f:
                saved_state = json.load(f)

            iterations = saved_state.get("iterations", [])
            if not iterations:
                logger.debug("loop_stats.json has no completed iterations")
                return iteration_history, eval_history, start_iteration

            # Find the last completed iteration
            last_iteration = max(it.get("iteration", 0) for it in iterations)
            if last_iteration <= 0:
                return iteration_history, eval_history, start_iteration

            # Resume from the next iteration
            start_iteration = last_iteration + 1
            logger.info(
                f"Auto-resuming from iteration {start_iteration} "
                f"(found {len(iterations)} completed iterations in loop_stats.json)"
            )

            # Restore iteration history for continuity
            for it_data in iterations:
                stats = IterationStats(
                    iteration=it_data.get("iteration", 0),
                    episodes_generated=it_data.get("episodes", 0),
                    transitions_generated=it_data.get("transitions", 0),
                    training_steps=it_data.get("steps", 0),
                    actor_time_seconds=it_data.get("actor_time", 0.0),
                    trainer_time_seconds=it_data.get("trainer_time", 0.0),
                    eval_time_seconds=it_data.get("eval_time", 0.0),
                    total_time_seconds=it_data.get("total_time", 0.0),
                    eval_win_rate=it_data.get("eval_win_rate"),
                    eval_draw_rate=it_data.get("eval_draw_rate"),
                    timestamp=it_data.get("timestamp", ""),
                )
                iteration_history.append(stats)

            # Also restore eval history if available
            if self.config.eval_stats_path.exists():
                try:
                    with open(self.config.eval_stats_path) as f:
                        eval_data = json.load(f)
                    eval_history = eval_data.get("evaluations", [])
                    logger.debug(f"Restored {len(eval_history)} evaluation records")
                except Exception as e:
                    logger.warning(f"Failed to restore eval history: {e}")

        except Exception as e:
            logger.warning(f"Failed to load loop_stats.json for auto-resume: {e}")

        return iteration_history, eval_history, start_iteration
