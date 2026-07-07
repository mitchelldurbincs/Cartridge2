"""Shim: implementation moved to the crucible package."""

from crucible.orchestrator.stats_manager import *  # noqa: F401,F403
from crucible.orchestrator.stats_manager import StatsManager

__all__ = ["StatsManager"]
