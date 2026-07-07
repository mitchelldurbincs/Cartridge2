"""Shim: implementation moved to the crucible package."""

from pathlib import Path
from typing import Callable

from crucible.orchestrator.actor_runner import *  # noqa: F401,F403
from crucible.orchestrator.actor_runner import ActorRunner as _CoreActorRunner

from .config import LoopConfig

# Use this module's location to find the project root, exactly as the
# pre-move code did: trainer/src/trainer/orchestrator/actor_runner.py -> root
_PROJECT_ROOT = Path(__file__).parents[4]

# The pre-move auto-detect list, verbatim: Docker location, workspace-level
# target, then actor-specific target (release before debug).
_BINARY_CANDIDATES = [
    Path("/app/actor"),
    _PROJECT_ROOT / "target" / "release" / "actor",
    _PROJECT_ROOT / "target" / "debug" / "actor",
    _PROJECT_ROOT / "actor" / "target" / "release" / "actor",
    _PROJECT_ROOT / "actor" / "target" / "debug" / "actor",
]


class ActorRunner(_CoreActorRunner):
    """Cartridge2 ActorRunner: auto-detects this repo's Rust actor binary.

    crucible's ActorRunner searches nowhere by default; injecting the
    candidate list above keeps binary discovery (config.actor_binary, then
    ACTOR_BINARY, then these paths) byte-equivalent to before the move.
    """

    def __init__(
        self,
        config: LoopConfig,
        shutdown_check: Callable[[], bool] | None = None,
    ):
        super().__init__(
            config,
            shutdown_check=shutdown_check,
            binary_candidates=_BINARY_CANDIDATES,
        )


__all__ = ["ActorRunner"]
