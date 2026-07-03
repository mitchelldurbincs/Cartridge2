"""Actor process management for the orchestrator.

This module handles spawning and managing actor processes for self-play
episode generation.
"""

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, NamedTuple

from .config import LoopConfig

logger = logging.getLogger(__name__)


class _ActorProcess(NamedTuple):
    """A spawned actor subprocess plus its episode quota and log label."""

    process: subprocess.Popen
    episodes: int
    actor_id: str


class ActorRunner:
    """Manages actor processes for self-play episode generation."""

    def __init__(
        self,
        config: LoopConfig,
        shutdown_check: Callable[[], bool] | None = None,
    ):
        """Initialize the actor runner.

        Args:
            config: Loop configuration.
            shutdown_check: Callback that returns True if shutdown was requested.
        """
        self.config = config
        self._shutdown_check = shutdown_check or (lambda: False)
        self._actor_binary: Path | None = None

    def _auto_detect_candidates(self) -> list[Path]:
        """Standard locations to search for the actor binary."""
        # Use the module's location to find the project root:
        # trainer/src/trainer/orchestrator/actor_runner.py -> project root
        project_root = Path(__file__).parents[4]

        return [
            # Docker location
            Path("/app/actor"),
            # Workspace-level target
            project_root / "target" / "release" / "actor",
            project_root / "target" / "debug" / "actor",
            # Actor-specific target
            project_root / "actor" / "target" / "release" / "actor",
            project_root / "actor" / "target" / "debug" / "actor",
        ]

    def find_binary(self) -> Path:
        """Find the actor binary, preferring release build.

        Checks in order:
        1. config.actor_binary (if explicitly set)
        2. ACTOR_BINARY environment variable
        3. Standard cargo build locations

        Returns:
            Path to the actor binary.

        Raises:
            FileNotFoundError: If no actor binary is found.
        """
        # Return cached binary if already found
        if self._actor_binary is not None:
            return self._actor_binary

        # Check explicit config
        if self.config.actor_binary:
            if self.config.actor_binary.exists():
                self._actor_binary = self.config.actor_binary
                return self._actor_binary
            raise FileNotFoundError(
                f"Configured actor binary not found: {self.config.actor_binary}"
            )

        # Check environment variable
        env_path = os.environ.get("ACTOR_BINARY")
        if env_path:
            path = Path(env_path)
            if path.exists():
                logger.info(f"Using actor binary from ACTOR_BINARY: {path}")
                self._actor_binary = path
                return self._actor_binary
            raise FileNotFoundError(f"ACTOR_BINARY not found: {env_path}")

        # Auto-detect from standard locations
        candidates = self._auto_detect_candidates()
        for candidate in candidates:
            if candidate.exists():
                logger.info(f"Found actor binary: {candidate}")
                self._actor_binary = candidate
                return self._actor_binary

        raise FileNotFoundError(
            f"Actor binary not found. Searched: {[str(c) for c in candidates]}. "
            "Set ACTOR_BINARY environment variable or run 'cd actor && cargo build --release'."
        )

    def _build_command(
        self,
        actor_binary: Path,
        actor_id: str,
        num_episodes: int,
        num_simulations: int,
    ) -> list[str]:
        """Build the CLI invocation for one actor process."""
        return [
            str(actor_binary),
            "--env-id",
            self.config.env_id,
            "--max-episodes",
            str(num_episodes),
            "--data-dir",
            str(self.config.data_dir),
            "--log-interval",
            str(self.config.actor_log_interval),
            "--log-level",
            "info",
            "--num-simulations",
            str(num_simulations),
            "--temp-threshold",
            str(self.config.temp_threshold),
            "--actor-id",
            actor_id,
            "--no-watch",  # Orchestrator pre-loads model; no hot-reload needed
        ]

    @staticmethod
    def _stream_output(proc: subprocess.Popen, actor_id: str) -> None:
        """Relay one actor's combined stdout/stderr to our stdout, line-prefixed."""
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(f"[{actor_id}] {line.rstrip()}", flush=True)

    def run(
        self, num_episodes: int, iteration: int, trace_id: str | None = None
    ) -> tuple[bool, float]:
        """Run actor(s) for a specified number of episodes.

        Args:
            num_episodes: Total number of self-play episodes to run.
            iteration: Current training iteration (for simulation ramping).
            trace_id: Optional trace ID for distributed tracing correlation.

        Returns:
            Tuple of (success, elapsed_seconds).
        """
        actor_binary = self.find_binary()
        num_actors = self.config.num_actors

        # Calculate MCTS simulations based on iteration (ramping schedule)
        num_simulations = self.config.get_num_simulations(iteration)

        # Split episodes across actors
        base_episodes = num_episodes // num_actors
        remainder = num_episodes % num_actors

        logger.info(
            f"Starting {num_actors} actor(s): {num_simulations} sims (iter {iteration}), "
            f"temp_threshold={self.config.temp_threshold}"
        )

        start_time = time.time()
        processes: list[_ActorProcess] = []

        try:
            # Spawn all actors
            for i in range(num_actors):
                actor_id = f"actor-{i + 1}"
                # First actor gets remainder episodes
                episodes_for_actor = base_episodes + (remainder if i == 0 else 0)

                if episodes_for_actor == 0:
                    continue

                cmd = self._build_command(
                    actor_binary, actor_id, episodes_for_actor, num_simulations
                )

                if num_actors == 1:
                    logger.info(f"Command: {' '.join(cmd)}")

                # Build environment with trace context for distributed tracing
                env = os.environ.copy()
                if trace_id:
                    env["CARTRIDGE_TRACE_ID"] = trace_id
                    env["CARTRIDGE_TRACE_PARENT"] = f"iteration_{iteration}"

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                )
                processes.append(_ActorProcess(process, episodes_for_actor, actor_id))
                logger.info(f"Started {actor_id} for {episodes_for_actor} episodes")

            # Stream output from all actors using threads
            threads = []
            for actor in processes:
                t = threading.Thread(
                    target=self._stream_output,
                    args=(actor.process, actor.actor_id),
                    daemon=True,
                )
                t.start()
                threads.append(t)

            # Wait for all processes, checking for shutdown
            all_success = True
            while processes:
                if self._shutdown_check():
                    logger.warning("Shutdown requested, terminating actors...")
                    for actor in processes:
                        actor.process.terminate()
                    for actor in processes:
                        actor.process.wait(timeout=5)
                    return False, time.time() - start_time

                # Check which processes are done
                still_running = []
                for actor in processes:
                    ret = actor.process.poll()
                    if ret is None:
                        still_running.append(actor)
                    elif ret != 0:
                        logger.error(f"{actor.actor_id} exited with code {ret}")
                        all_success = False
                    else:
                        logger.info(
                            f"{actor.actor_id} completed ({actor.episodes} episodes)"
                        )
                processes = still_running

                if processes:
                    time.sleep(0.1)

            # Wait for output threads to finish
            for t in threads:
                t.join(timeout=1.0)

            elapsed = time.time() - start_time

            if all_success:
                logger.info(f"All {num_actors} actor(s) completed in {elapsed:.1f}s")
            return all_success, elapsed

        except Exception as e:
            logger.error(f"Actor(s) failed: {e}")
            # Clean up any running processes
            for actor in processes:
                try:
                    actor.process.terminate()
                    actor.process.wait(timeout=2)
                except Exception:
                    pass
            return False, time.time() - start_time
