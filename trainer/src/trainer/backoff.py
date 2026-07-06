"""Shim: implementation moved to the training-core package."""

from training_core.backoff import *  # noqa: F401,F403
from training_core.backoff import (
    DEFAULT_MAX_WAIT,
    DEFAULT_WAIT_INTERVAL,
    LOG_EVERY_N_WAITS,
    WaitTimeout,
    wait_with_backoff,
)

__all__ = [
    "DEFAULT_MAX_WAIT",
    "DEFAULT_WAIT_INTERVAL",
    "LOG_EVERY_N_WAITS",
    "WaitTimeout",
    "wait_with_backoff",
]
