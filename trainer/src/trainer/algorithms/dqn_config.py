"""Strict learner configuration owned by ``dqn_v1``."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..storage.base import ReplaySelection


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite(value: object, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number >= {minimum}")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be a finite number >= {minimum}")
    return result


@dataclass(frozen=True)
class DqnLearnerOptions:
    """User-facing learner defaults shared by the train and loop commands."""

    total_steps: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gamma: float = 0.99
    target_sync_interval: int = 50
    hidden_size: int = 128
    grad_clip_norm: float = 10.0
    device: str = "cpu"


LEARNER_DEFAULTS = DqnLearnerOptions()


@dataclass
class DqnLearnerConfig:
    """Complete internal learner configuration with an explicit replay fence."""

    env_id: str
    model_dir: str
    stats_path: str
    replay_selection: ReplaySelection
    total_steps: int = LEARNER_DEFAULTS.total_steps
    batch_size: int = LEARNER_DEFAULTS.batch_size
    learning_rate: float = LEARNER_DEFAULTS.learning_rate
    weight_decay: float = LEARNER_DEFAULTS.weight_decay
    gamma: float = LEARNER_DEFAULTS.gamma
    target_sync_interval: int = LEARNER_DEFAULTS.target_sync_interval
    hidden_size: int = LEARNER_DEFAULTS.hidden_size
    grad_clip_norm: float = LEARNER_DEFAULTS.grad_clip_norm
    device: str = LEARNER_DEFAULTS.device

    def __post_init__(self) -> None:
        for field in (
            "total_steps",
            "batch_size",
            "target_sync_interval",
            "hidden_size",
        ):
            setattr(self, field, _positive_integer(getattr(self, field), field))
        self.learning_rate = _finite(
            self.learning_rate, "learning_rate", minimum=float.fromhex("0x1p-149")
        )
        self.weight_decay = _finite(self.weight_decay, "weight_decay")
        self.grad_clip_norm = _finite(self.grad_clip_norm, "grad_clip_norm")
        self.gamma = _finite(self.gamma, "gamma")
        if self.gamma > 1.0:
            raise ValueError("gamma must be <= 1")
        if self.device not in {"cpu", "cuda", "mps", "auto"}:
            raise ValueError("device must be cpu, cuda, mps, or auto")
        if not isinstance(self.env_id, str) or not self.env_id.strip():
            raise ValueError("env_id must be non-empty")
        for field in ("model_dir", "stats_path"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"{field} must be non-empty")
        if not isinstance(self.replay_selection, ReplaySelection):
            raise TypeError("replay_selection must be an explicit ReplaySelection")

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def learner_recipe(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "implementation": "dqn_q_learning_v1",
            "total_steps": self.total_steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gamma": self.gamma,
            "target_sync_interval": self.target_sync_interval,
            "hidden_size": self.hidden_size,
            "grad_clip_norm": self.grad_clip_norm,
        }
