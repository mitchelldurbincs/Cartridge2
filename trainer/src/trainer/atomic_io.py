"""Shim: implementation moved to the training-core package."""

from training_core.atomic_io import *  # noqa: F401,F403
from training_core.atomic_io import atomic_copy, atomic_write

__all__ = ["atomic_copy", "atomic_write"]
