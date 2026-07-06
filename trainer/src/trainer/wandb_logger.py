"""Shim: implementation moved to the training-core package."""

from training_core.wandb_logger import *  # noqa: F401,F403
from training_core.wandb_logger import WandbLogger, make_logger

__all__ = ["make_logger", "WandbLogger"]
