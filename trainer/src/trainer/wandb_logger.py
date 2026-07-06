"""Shim: implementation moved to the training-core package."""
from training_core.wandb_logger import *  # noqa: F401,F403
from training_core.wandb_logger import make_logger, WandbLogger  # explicit re-exports
