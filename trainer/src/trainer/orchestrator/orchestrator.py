"""Synchronized AlphaZero Training Loop Orchestrator.

This module implements the classic AlphaZero iteration pattern:
1. Clear replay buffer (start fresh with current model)
2. Run actor for N episodes (self-play data generation)
3. Train for M steps on the generated data
4. Run evaluation against random baseline
5. Export new model, repeat

This ensures each training iteration only uses data from the current model,
avoiding the noise that comes from mixing data from many model generations.
"""

import logging
import signal
import time
from typing import Any

from ..storage import create_replay_buffer
from ..structured_logging import (
    generate_span_id,
    generate_trace_id,
    set_trace_context,
)
from .actor_runner import ActorRunner
from .config import IterationStats, LoopConfig
from .eval_runner import EvalRunner
from .stats_manager import StatsManager

logger = logging.getLogger(__name__)


class Orchestrator:
    """Synchronized AlphaZero training loop orchestrator.

    Uses composition with specialized components for:
    - Actor management (ActorRunner)
    - Evaluation and best model tracking (EvalRunner)
    - Stats persistence (StatsManager)
    """

    def __init__(self, config: LoopConfig):
        """Initialize the orchestrator.

        Args:
            config: Loop configuration.
        """
        self.config = config
        self._shutdown_requested = False

        # Initialize shared replay buffer connection (reused across iterations)
        self._replay_buffer = create_replay_buffer()

        # Initialize components
        self.stats_manager = StatsManager(config)
        self.eval_runner = EvalRunner(config)
        self.actor_runner = ActorRunner(
            config, shutdown_check=lambda: self._shutdown_requested
        )

        # Initialize state
        self.iteration_history: list[IterationStats] = []
        self.eval_history: list[dict] = []

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Auto-resume from previous state if start_iteration is default (1)
        self._auto_resume_if_needed()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        logger.warning(f"Received signal {signum}, requesting shutdown...")
        self._shutdown_requested = True

    def _auto_resume_if_needed(self) -> None:
        """Auto-resume from previous state if start_iteration is default (1)."""
        # Only auto-resume if start_iteration is the default value (1)
        if self.config.start_iteration != 1:
            logger.debug(
                f"start_iteration={self.config.start_iteration} (not default), "
                "skipping auto-resume"
            )
            return

        history, eval_history, start_iteration = (
            self.stats_manager.load_previous_state()
        )

        if start_iteration > 1:
            self.config.start_iteration = start_iteration
            self.iteration_history = history
            self.eval_history = eval_history

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

    def _clear_replay_buffer(self) -> int:
        """Clear all transitions from the replay buffer.

        Returns the number of deleted transitions.
        """
        deleted = self._replay_buffer.clear_transitions()
        self._replay_buffer.vacuum()
        logger.info(f"Cleared {deleted} transitions from replay buffer (vacuumed)")
        return deleted

    def _get_transition_count(self) -> int:
        """Get the current number of transitions in the replay buffer."""
        return self._replay_buffer.count(env_id=self.config.env_id)

    def _run_trainer(self, num_steps: int, start_step: int) -> tuple[bool, float]:
        """Run the trainer for a specified number of steps.

        Returns (success, elapsed_seconds).
        """
        # Import locally to avoid circular imports
        from ..trainer import Trainer, TrainerConfig

        # Calculate total training steps across all iterations for continuous LR decay.
        lr_total_steps = self.config.iterations * self.config.steps_per_iteration

        config = TrainerConfig(
            model_dir=str(self.config.models_dir),
            stats_path=str(self.config.stats_path),
            env_id=self.config.env_id,
            total_steps=num_steps,
            start_step=start_step,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            checkpoint_interval=self.config.checkpoint_interval,
            device=self.config.resolve_device(),
            max_wait=60.0,  # Short timeout since we know data exists
            eval_interval=0,  # Disable trainer's built-in eval, we do it ourselves
            lr_total_steps=lr_total_steps,  # Continuous LR decay across iterations
            shutdown_check=lambda: self._shutdown_requested,  # Pass shutdown check
        )

        logger.info(f"Starting trainer for {num_steps} steps (start_step={start_step})")
        start_time = time.time()

        try:
            trainer = Trainer(config)
            stats = trainer.train()
            elapsed = time.time() - start_time

            logger.info(
                f"Trainer completed in {elapsed:.1f}s, final loss: {stats.total_loss:.4f}"
            )
            return True, elapsed

        except Exception as e:
            logger.error(f"Trainer failed: {e}")
            return False, time.time() - start_time

    def run_iteration(self, iteration: int) -> IterationStats | None:
        """Run a single training iteration.

        Returns iteration stats, or None if shutdown was requested.
        """
        iter_start = time.time()
        num_simulations = self.config.get_num_simulations(iteration)

        # Generate trace ID for this iteration (for distributed tracing)
        trace_id = generate_trace_id()
        set_trace_context(trace_id=trace_id, span_id=generate_span_id())

        logger.info("=" * 60)
        logger.info(
            f"ITERATION {iteration} | MCTS simulations: {num_simulations}",
            extra={
                "event": "iteration_start",
                "iteration": iteration,
                "trace_id": trace_id,
            },
        )
        logger.info("=" * 60)

        # Step 1: Clear replay buffer
        logger.info("Step 1: Clearing replay buffer...")
        self._clear_replay_buffer()

        if self._shutdown_requested:
            return None

        # Step 2: Run actor for N episodes
        logger.info(
            f"Step 2: Running actor for {self.config.episodes_per_iteration} episodes..."
        )
        actor_success, actor_time = self.actor_runner.run(
            self.config.episodes_per_iteration, iteration, trace_id=trace_id
        )

        if not actor_success:
            if self._shutdown_requested:
                return None
            logger.error("Actor failed, aborting iteration")
            return None

        transitions_count = self._get_transition_count()
        logger.info(f"Actor generated {transitions_count} transitions")

        if self._shutdown_requested:
            return None

        # Step 3: Train for M steps
        start_step = (iteration - 1) * self.config.steps_per_iteration
        logger.info(f"Step 3: Training for {self.config.steps_per_iteration} steps...")
        trainer_success, trainer_time = self._run_trainer(
            self.config.steps_per_iteration, start_step
        )

        if not trainer_success:
            if self._shutdown_requested:
                return None
            logger.error("Trainer failed, aborting iteration")
            return None

        if self._shutdown_requested:
            return None

        # Step 4: Run evaluation (if enabled)
        eval_time = 0.0
        win_rate = None
        draw_rate = None

        should_eval = (
            self.config.eval_interval > 0 and iteration % self.config.eval_interval == 0
        )

        if should_eval:
            logger.info("Step 4: Running evaluation...")
            win_rate, draw_rate, eval_time = self.eval_runner.run(
                iteration, self.eval_history
            )
            # Save eval stats and update frontend stats
            self.stats_manager.save_eval_stats(self.eval_history)
            self.stats_manager.update_stats_with_eval(
                self.eval_history, self.eval_runner.best_iteration
            )
        else:
            if self.config.eval_interval > 0:
                remaining = (
                    self.config.eval_interval - iteration % self.config.eval_interval
                )
                next_eval = iteration + remaining
                logger.info(
                    f"Step 4: Skipping evaluation (next at iteration {next_eval})"
                )
            else:
                logger.info("Step 4: Evaluation disabled (ALPHAZERO_EVAL_INTERVAL=0)")

        total_time = time.time() - iter_start

        stats = IterationStats(
            iteration=iteration,
            episodes_generated=self.config.episodes_per_iteration,
            transitions_generated=transitions_count,
            training_steps=self.config.steps_per_iteration,
            actor_time_seconds=actor_time,
            trainer_time_seconds=trainer_time,
            eval_time_seconds=eval_time,
            total_time_seconds=total_time,
            eval_win_rate=win_rate,
            eval_draw_rate=draw_rate,
        )

        logger.info(f"\nIteration {iteration} complete:")
        logger.info(f"  Episodes: {stats.episodes_generated}")
        logger.info(f"  Transitions: {stats.transitions_generated}")
        logger.info(f"  Actor time: {stats.actor_time_seconds:.1f}s")
        logger.info(f"  Trainer time: {stats.trainer_time_seconds:.1f}s")
        if win_rate is not None:
            logger.info(f"  Eval win rate: {win_rate:.1%}")
        logger.info(f"  Total time: {stats.total_time_seconds:.1f}s")

        return stats

    def run(self) -> None:
        """Run the full training loop."""
        self._ensure_directories()

        # Log configuration with evaluation status prominently
        logger.info("=" * 60)
        logger.info("Synchronized AlphaZero Training")
        logger.info("=" * 60)
        logger.info(f"Environment: {self.config.env_id}")

        # Show resume status
        if self.iteration_history:
            logger.info(
                f"RESUMING from iteration {self.config.start_iteration} "
                f"({len(self.iteration_history)} previous iterations loaded)"
            )
        else:
            logger.info(f"Starting from iteration {self.config.start_iteration}")

        logger.info(f"Target iterations: {self.config.iterations}")
        logger.info(f"Episodes per iteration: {self.config.episodes_per_iteration}")
        logger.info(f"Steps per iteration: {self.config.steps_per_iteration}")
        logger.info(f"Parallel actors: {self.config.num_actors}")
        logger.info(f"Data directory: {self.config.data_dir}")

        # Show MCTS simulation ramping settings
        start_sims = self.config.mcts_start_sims
        max_sims = self.config.mcts_max_sims
        ramp_rate = self.config.mcts_sim_ramp_rate
        iters_to_max = (
            max(1, (max_sims - start_sims) // ramp_rate) if ramp_rate > 0 else 1
        )
        logger.info(
            f"MCTS simulations: {start_sims} -> {max_sims} "
            f"(+{ramp_rate}/iter, reaches max at iter {iters_to_max})"
        )

        # Show temperature schedule
        if self.config.temp_threshold > 0:
            logger.info(
                f"Temperature schedule: temp=1.0 for moves 0-{self.config.temp_threshold - 1}, "
                f"temp=0.1 after move {self.config.temp_threshold}"
            )
        else:
            logger.info("Temperature schedule: DISABLED (temp=1.0 for all moves)")

        # Prominently show evaluation settings
        if self.config.eval_interval > 0:
            interval = self.config.eval_interval
            games = self.config.eval_games
            threshold = self.config.eval_win_threshold
            logger.info(
                f"Evaluation: ENABLED (every {interval} iteration(s), {games} games)"
            )
            logger.info(f"  Gatekeeper: win rate > {threshold:.0%} to become new best")
            if self.config.eval_vs_random:
                logger.info("  Also evaluating vs random baseline")
            if self.eval_runner.best_iteration:
                logger.info(
                    f"  Current best model: iteration {self.eval_runner.best_iteration}"
                )
        else:
            logger.info(
                "Evaluation: DISABLED (set ALPHAZERO_EVAL_INTERVAL > 0 to enable)"
            )

        logger.info("=" * 60)

        loop_start = time.time()

        for iteration in range(
            self.config.start_iteration,
            self.config.start_iteration + self.config.iterations,
        ):
            if self._shutdown_requested:
                logger.warning("Shutdown requested, stopping loop")
                break

            stats = self.run_iteration(iteration)
            if stats:
                self.iteration_history.append(stats)
                self.stats_manager.save_loop_stats(self.iteration_history)

        total_time = time.time() - loop_start
        total_in_history = len(self.iteration_history)
        # Count only iterations completed in this session
        new_completed = total_in_history - (self.config.start_iteration - 1)

        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        if self.config.start_iteration > 1:
            logger.info(
                f"Completed iterations this session: {new_completed} "
                f"(total in history: {total_in_history})"
            )
        else:
            logger.info(f"Completed iterations: {total_in_history}")
        logger.info(f"Session time: {total_time:.1f}s ({total_time/3600:.2f}h)")

        if self.iteration_history:
            total_episodes = sum(s.episodes_generated for s in self.iteration_history)
            total_transitions = sum(
                s.transitions_generated for s in self.iteration_history
            )
            total_steps = sum(s.training_steps for s in self.iteration_history)
            logger.info(f"Total episodes: {total_episodes}")
            logger.info(f"Total transitions: {total_transitions}")
            logger.info(f"Total training steps: {total_steps}")

            # Report final evaluation if available
            final_with_eval = [
                s for s in self.iteration_history if s.eval_win_rate is not None
            ]
            if final_with_eval:
                final = final_with_eval[-1]
                logger.info(
                    f"Final evaluation: {final.eval_win_rate:.1%} win rate vs random"
                )

        # Clean up shared resources
        self._replay_buffer.close()
