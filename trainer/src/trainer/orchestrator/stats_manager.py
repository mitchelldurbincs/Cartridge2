"""Shim: implementation moved to the training-core package."""

from training_core.orchestrator.stats_manager import *  # noqa: F401,F403
from training_core.orchestrator.stats_manager import StatsManager

__all__ = ["StatsManager"]
