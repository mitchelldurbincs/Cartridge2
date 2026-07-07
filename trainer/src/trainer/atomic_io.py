"""Shim: implementation moved to the crucible package."""

from crucible.atomic_io import *  # noqa: F401,F403
from crucible.atomic_io import atomic_copy, atomic_write

__all__ = ["atomic_copy", "atomic_write"]
