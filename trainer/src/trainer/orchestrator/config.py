"""Shim: implementation moved to the training-core package."""

from dataclasses import dataclass, field

from training_core.orchestrator.config import *  # noqa: F401,F403
from training_core.orchestrator.config import (
    IterationStats,
)
from training_core.orchestrator.config import (
    LoopConfig as _CoreLoopConfig,
)

from ..central_config import WandbConfig


@dataclass
class LoopConfig(_CoreLoopConfig):
    """Cartridge2 LoopConfig: restores this repo's WandbConfig default.

    training-core's LoopConfig defaults ``wandb`` to None; Cartridge2 callers
    keep getting a ready-to-use ``WandbConfig()`` exactly as before the move.
    """

    wandb: WandbConfig = field(default_factory=WandbConfig)


__all__ = ["IterationStats", "LoopConfig"]
