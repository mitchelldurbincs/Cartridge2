"""Learning rate scheduling with warmup and cosine annealing.

This module provides a clean abstraction for LR scheduling that handles:
- Linear warmup from a low LR to the target LR
- Cosine annealing from target LR down to a minimum LR
- Checkpoint save/restore with proper state management
- Automatic warmup disabling when resuming from checkpoint

LR Schedule Visualization:

    LR
    ^
    |        target_lr
    |       /----------.
    |      /            '.
    |     /               '..
    |    /                   ''..._____ min_lr
    |   /
    +--+----------------------------->  step
       0   warmup    T_max
           steps

Usage:
    >>> from trainer.lr_scheduler import WarmupCosineScheduler, LRConfig
    >>>
    >>> config = LRConfig(
    ...     target_lr=1e-3,
    ...     warmup_steps=100,
    ...     total_steps=10000,
    ... )
    >>> scheduler = WarmupCosineScheduler(optimizer, config)
    >>>
    >>> for step in range(1, 10001):
    ...     train_step()
    ...     scheduler.step()
    ...     print(f"Step {step}: LR = {scheduler.get_lr():.2e}")
"""

import logging
import math
from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

__all__ = ["LRConfig", "WarmupCosineScheduler"]

logger = logging.getLogger(__name__)


@dataclass
class LRConfig:
    """Configuration for learning rate scheduling.

    Attributes:
        target_lr: The peak learning rate (reached after warmup).
        warmup_steps: Number of steps to linearly increase LR from start to target.
                     Set to 0 to disable warmup.
        warmup_start_ratio: Starting LR as a fraction of target_lr (default: 0.1).
        min_ratio: Final LR as a fraction of target_lr (default: 0.1).
        total_steps: Total training steps (used for cosine annealing T_max).
        enabled: Whether to use LR scheduling at all. If False, LR stays constant.
    """

    target_lr: float
    warmup_steps: int = 100
    warmup_start_ratio: float = 0.1
    min_ratio: float = 0.1
    total_steps: int = 1000
    enabled: bool = True

    @property
    def warmup_start_lr(self) -> float:
        """Starting LR during warmup phase."""
        return self.target_lr * self.warmup_start_ratio

    @property
    def min_lr(self) -> float:
        """Minimum LR at end of cosine annealing."""
        return self.target_lr * self.min_ratio

    @property
    def cosine_steps(self) -> int:
        """Number of steps for cosine annealing (excludes warmup)."""
        return max(1, self.total_steps - self.warmup_steps)


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine annealing.

    This scheduler implements a two-phase LR schedule:

    1. **Warmup Phase** (steps 1 to warmup_steps):
       LR increases linearly from warmup_start_lr to target_lr.

    2. **Cosine Annealing Phase** (steps warmup_steps+1 to total_steps):
       LR decreases following a cosine curve from target_lr to min_lr.

    Key behaviors:
    - The optimizer is initialized with target_lr (not warmup_start_lr) so that
      PyTorch's CosineAnnealingLR has the correct base_lrs.
    - During warmup, we manually override the optimizer's LR.
    - When resuming from checkpoint, warmup is automatically disabled to avoid
      loss spikes from suddenly dropping the LR.

    State Management:
    - state_dict() returns the cosine scheduler state plus warmup metadata.
    - load_state_dict() restores state and handles edge cases like completed schedules.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: LRConfig,
        *,
        from_checkpoint: bool = False,
    ):
        """Initialize the scheduler.

        Args:
            optimizer: The optimizer to schedule. Should be initialized with
                      config.target_lr as its learning rate.
            config: LR scheduling configuration.
            from_checkpoint: If True, disables warmup (for resumed training).
        """
        self.optimizer = optimizer
        self.config = config
        self._current_step = 0

        # Warmup is disabled when resuming from checkpoint
        self._warmup_steps = 0 if from_checkpoint else config.warmup_steps
        self._warmup_disabled_reason: str | None = None

        if from_checkpoint and config.warmup_steps > 0:
            self._warmup_disabled_reason = "resumed from checkpoint"
            logger.info("LR warmup disabled (resumed from checkpoint)")

        # Initialize cosine scheduler (only if scheduling is enabled)
        self._cosine_scheduler: CosineAnnealingLR | None = None
        if config.enabled:
            self._cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.cosine_steps,
                eta_min=config.min_lr,
            )

        # Set initial LR
        if self._warmup_steps > 0:
            # Start at warmup LR
            self._set_lr(config.warmup_start_lr)
            logger.info(
                f"LR warmup: {config.warmup_start_lr:.2e} -> "
                f"{config.target_lr:.2e} over {self._warmup_steps} steps"
            )
        elif config.enabled:
            logger.info(
                f"LR schedule: cosine annealing {config.target_lr:.2e} -> "
                f"{config.min_lr:.2e} over {config.cosine_steps} steps"
            )

    # -------------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------------

    def step(self) -> float:
        """Advance the scheduler by one step and return the new LR.

        Call this once per training step, after optimizer.step().

        Returns:
            The new learning rate.
        """
        self._current_step += 1

        if not self.config.enabled:
            return self.config.target_lr

        if self._in_warmup_phase():
            lr = self._compute_warmup_lr()
            self._set_lr(lr)
            return lr
        elif self._cosine_scheduler is not None:
            self._cosine_scheduler.step()
            return self.get_lr()

        return self.get_lr()

    def get_lr(self) -> float:
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    # -------------------------------------------------------------------------
    # State Management (for checkpointing)
    # -------------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing.

        The state includes both the cosine scheduler state and warmup metadata,
        allowing proper restoration even mid-warmup.
        """
        state = {
            "current_step": self._current_step,
            "warmup_steps": self._warmup_steps,
            "config_warmup_steps": self.config.warmup_steps,  # Original config value
        }

        if self._cosine_scheduler is not None:
            state["cosine_scheduler"] = self._cosine_scheduler.state_dict()

        return state

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from checkpoint.

        This method handles several edge cases:
        - Warmup is always disabled when loading (to avoid loss spikes)
        - If the cosine schedule completed, LR is set to minimum
        - base_lrs is reset to target_lr (may have been saved during warmup)

        Args:
            state: State dictionary from a previous state_dict() call,
                  or a raw CosineAnnealingLR state dict for backwards compatibility.
        """
        # Handle legacy format (just the cosine scheduler state)
        if "cosine_scheduler" not in state and "last_epoch" in state:
            cosine_state = state
            self._current_step = state.get("last_epoch", 0)
        else:
            cosine_state = state.get("cosine_scheduler")
            self._current_step = state.get("current_step", 0)

        # Always disable warmup when loading from checkpoint
        if self._warmup_steps > 0:
            self._warmup_steps = 0
            self._warmup_disabled_reason = "loaded from checkpoint"

        # Restore cosine scheduler state
        if self._cosine_scheduler is not None and cosine_state is not None:
            self._restore_cosine_state(cosine_state)

    def _restore_cosine_state(self, cosine_state: dict) -> None:
        """Restore the cosine scheduler from saved state.

        The horizon (``T_max``) is re-anchored to the *current* config rather than
        the value saved in the checkpoint. Honoring a stale/shorter saved horizon is
        what let a completed old schedule pin the LR at ``eta_min`` forever across
        resumes: once a short schedule finished, every later resume saw
        ``last_epoch >= saved_T_max`` and clamped to the floor, even when the new run
        intended a much longer schedule.
        """
        last_epoch = cosine_state.get("last_epoch", 0)
        t_max = self.config.cosine_steps

        # Check completion against the CURRENT horizon (not the saved one).
        if last_epoch >= t_max:
            self._set_lr(self.config.min_lr)
            logger.info(
                f"LR schedule complete (epoch {last_epoch} >= T_max {t_max}), "
                f"using min LR={self.config.min_lr:.2e}"
            )
            return

        # Restore the scheduler counters, then re-anchor horizon + base LR.
        self._cosine_scheduler.load_state_dict(cosine_state)
        self._cosine_scheduler.T_max = t_max
        # Reset base_lrs to target_lr: the saved base_lrs might be warmup_start_lr
        # (checkpoint saved during warmup) and cosine should anneal from target_lr.
        self._cosine_scheduler.base_lrs = [self.config.target_lr]

        # Re-seed the LR *only* when the checkpoint was pinned at eta_min. PyTorch's
        # CosineAnnealingLR.step() is recurrent -- it scales the *previous* LR by
        # (lr - eta_min). A checkpoint saved at eta_min (a short schedule that ran to
        # completion) makes that factor zero, so the LR would stay pinned at the floor
        # forever regardless of base_lrs, even though last_epoch now sits within a
        # longer horizon. Seeding the optimizer LR (and cached _last_lr) from the
        # cosine closed form restores headroom so the schedule resumes correctly.
        # When not floored we leave the restored LR untouched: the live LR is carried
        # by the optimizer checkpoint, and stepping continues the cosine normally.
        saved_last_lr = cosine_state.get("_last_lr") or []
        floored = abs(self.get_lr() - self.config.min_lr) < 1e-12 or (
            len(saved_last_lr) > 0
            and abs(saved_last_lr[0] - self.config.min_lr) < 1e-12
        )
        if floored:
            closed_form_lr = (
                self.config.min_lr
                + (self.config.target_lr - self.config.min_lr)
                * (1 + math.cos(math.pi * last_epoch / t_max))
                / 2
            )
            self._set_lr(closed_form_lr)
            self._cosine_scheduler._last_lr = [closed_form_lr]

        logger.info(
            f"Restored LR scheduler (epoch={last_epoch}, T_max={t_max}, "
            f"LR={self.get_lr():.2e})"
        )

    # -------------------------------------------------------------------------
    # Warmup Logic
    # -------------------------------------------------------------------------

    def _in_warmup_phase(self) -> bool:
        """Check if we're currently in the warmup phase."""
        return self._warmup_steps > 0 and self._current_step <= self._warmup_steps

    def _compute_warmup_lr(self) -> float:
        """Compute LR during warmup using linear interpolation."""
        progress = self._current_step / self._warmup_steps
        return self.config.warmup_start_lr + progress * (
            self.config.target_lr - self.config.warmup_start_lr
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _set_lr(self, lr: float) -> None:
        """Set the learning rate on all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    # -------------------------------------------------------------------------
    # Introspection (for logging/debugging)
    # -------------------------------------------------------------------------

    @property
    def warmup_active(self) -> bool:
        """Whether warmup is currently active (not disabled and not complete)."""
        return self._warmup_steps > 0 and self._current_step <= self._warmup_steps

    @property
    def warmup_complete(self) -> bool:
        """Whether warmup phase has completed."""
        return self._warmup_steps > 0 and self._current_step > self._warmup_steps

    @property
    def warmup_disabled(self) -> bool:
        """Whether warmup was disabled (e.g., due to checkpoint loading)."""
        return self._warmup_disabled_reason is not None

    @property
    def schedule_complete(self) -> bool:
        """Whether the entire LR schedule has completed."""
        if not self.config.enabled:
            return False
        if self._cosine_scheduler is None:
            return False
        return self._cosine_scheduler.last_epoch >= self.config.cosine_steps

    def get_phase(self) -> str:
        """Get a human-readable description of the current phase."""
        if not self.config.enabled:
            return "disabled"
        if self.warmup_active:
            return f"warmup ({self._current_step}/{self._warmup_steps})"
        if self.schedule_complete:
            return "complete"
        if self._cosine_scheduler:
            epoch = self._cosine_scheduler.last_epoch
            return f"cosine ({epoch}/{self.config.cosine_steps})"
        return "unknown"

    def __repr__(self) -> str:
        return (
            f"WarmupCosineScheduler("
            f"lr={self.get_lr():.2e}, "
            f"phase={self.get_phase()}, "
            f"step={self._current_step})"
        )
