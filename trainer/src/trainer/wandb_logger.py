"""Shim: implementation moved to the crucible package."""

from crucible.wandb_logger import *  # noqa: F401,F403
from crucible.wandb_logger import WandbLogger, make_logger

__all__ = ["make_logger", "WandbLogger"]
