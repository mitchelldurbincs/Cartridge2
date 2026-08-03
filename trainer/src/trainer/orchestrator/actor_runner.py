"""Subprocess runner for Cartridge2 collector implementations."""

import json
from pathlib import Path
from typing import Any, Callable

from crucible.orchestrator.actor_runner import ActorRunner as _CoreActorRunner

from ..storage import ReplaySelection
from .config import LoopConfig

# trainer/src/trainer/orchestrator/actor_runner.py -> repository root
_PROJECT_ROOT = Path(__file__).parents[4]

# Docker location, workspace target, then actor-specific target.
_BINARY_CANDIDATES = [
    Path("/app/actor"),
    _PROJECT_ROOT / "target" / "release" / "actor",
    _PROJECT_ROOT / "target" / "debug" / "actor",
    _PROJECT_ROOT / "actor" / "target" / "release" / "actor",
    _PROJECT_ROOT / "actor" / "target" / "debug" / "actor",
]


class ActorRunner(_CoreActorRunner):
    """Cartridge2 ActorRunner: auto-detects this repo's Rust actor binary.

    Binary discovery checks the configured path, ``ACTOR_BINARY``, then the
    candidate paths above.
    """

    def __init__(
        self,
        config: LoopConfig,
        collector_config_builder: Callable[[Any, int], dict],
        shutdown_check: Callable[[], bool] | None = None,
    ):
        super().__init__(
            config,
            shutdown_check=shutdown_check,
            binary_candidates=_BINARY_CANDIDATES,
        )
        self._collector_config_builder = collector_config_builder
        self._replay_selection: ReplaySelection | None = None

    def select_replay(self, selection: ReplaySelection) -> None:
        """Fence every subsequently spawned collector to one exact attempt."""
        if not isinstance(selection, ReplaySelection):
            raise TypeError("selection must be ReplaySelection")
        if (
            selection.profile.env_id != self.config.env_id
            or selection.profile.algorithm_id != self.config.algorithm_id
        ):
            raise ValueError("Collector replay selection does not match loop profile")
        self._replay_selection = selection

    def _build_command(
        self,
        actor_binary: Path,
        actor_id: str,
        num_episodes: int,
        num_simulations: int,
    ) -> list[str]:
        selection = self._replay_selection
        if selection is None:
            raise RuntimeError("Collector replay selection has not been established")
        collector_config = self._collector_config_builder(self.config, num_simulations)
        if not isinstance(collector_config, dict):
            raise TypeError("collector cartridge configuration must be a dict")
        collector_config_json = json.dumps(
            collector_config, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        command = [
            str(actor_binary),
            "--algorithm",
            self.config.algorithm_id,
            "--env-id",
            self.config.env_id,
            "--max-episodes",
            str(num_episodes),
            "--data-dir",
            str(self.config.data_dir),
            "--log-interval",
            str(self.config.actor_log_interval),
            "--log-level",
            self.config.log_level.lower(),
            "--collector-config",
            collector_config_json,
            "--actor-id",
            actor_id,
            "--episode-timeout-secs",
            str(self.config.actor_episode_timeout_seconds),
            "--collection-scope-id",
            selection.collection_scope_id,
        ]
        if selection.source_checkpoint_id is not None:
            command.extend(["--source-checkpoint-id", selection.source_checkpoint_id])
        return command


__all__ = ["ActorRunner"]
