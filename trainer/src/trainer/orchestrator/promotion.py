"""Shim: implementation moved to the training-core package."""

from training_core.orchestrator.promotion import *  # noqa: F401,F403
from training_core.orchestrator.promotion import PromotionMixin, should_promote

__all__ = ["PromotionMixin", "should_promote"]
