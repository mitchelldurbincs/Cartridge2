"""Shim: implementation moved to the crucible package."""

from dataclasses import dataclass, field

from crucible.orchestrator.config import *  # noqa: F401,F403
from crucible.orchestrator.config import (
    IterationStats,
)
from crucible.orchestrator.config import (
    LoopConfig as _CoreLoopConfig,
)

from ..central_config import WandbConfig


@dataclass
class LoopConfig(_CoreLoopConfig):
    """Cartridge2 LoopConfig: restores this repo's WandbConfig default.

    crucible's LoopConfig defaults ``wandb`` to None; Cartridge2 callers
    keep getting a ready-to-use ``WandbConfig()`` exactly as before the move.
    Dataclass-subclass gotcha: the generated ``__eq__`` compares
    ``other.__class__ is self.__class__``, so an instance of this subclass
    never compares equal to a core ``LoopConfig`` instance, even with
    identical field values.
    """

    wandb: WandbConfig = field(default_factory=WandbConfig)


__all__ = ["IterationStats", "LoopConfig"]
