"""Strict learner configuration owned by ``dqn_v1``."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

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


@dataclass
class DqnLearnerConfig:
    env_id: str = "counter"
    model_dir: str = "./data/models"
    stats_path: str = "./data/stats.json"
    total_steps: int = 500
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gamma: float = 0.99
    target_sync_interval: int = 50
    hidden_size: int = 128
    grad_clip_norm: float = 10.0
    device: str = "cpu"
    replay_selection: ReplaySelection | None = None

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

    @classmethod
    def configure_parser(
        cls, parser: argparse.ArgumentParser, *, defaults: dict[str, Any] | None = None
    ) -> None:
        values = cls()
        overrides = defaults or {}
        parser.add_argument("--env-id", default=overrides.get("env_id", values.env_id))
        parser.add_argument("--model-dir", default=overrides.get("model_dir"))
        parser.add_argument("--stats-path", default=overrides.get("stats_path"))
        parser.add_argument("--steps", type=int, default=values.total_steps)
        parser.add_argument("--batch-size", type=int, default=values.batch_size)
        parser.add_argument("--learning-rate", type=float, default=values.learning_rate)
        parser.add_argument("--weight-decay", type=float, default=values.weight_decay)
        parser.add_argument("--gamma", type=float, default=values.gamma)
        parser.add_argument("--target-sync-interval", type=int, default=values.target_sync_interval)
        parser.add_argument("--hidden-size", type=int, default=values.hidden_size)
        parser.add_argument("--grad-clip", type=float, default=values.grad_clip_norm)
        parser.add_argument("--device", default=values.device)
        parser.add_argument("--collection-scope-id", required=True)
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--source-checkpoint-id")
        source.add_argument("--source-root", action="store_true")
        parser.add_argument("--log-level", default="INFO")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DqnLearnerConfig":
        return cls(
            env_id=args.env_id,
            model_dir=args.model_dir,
            stats_path=args.stats_path,
            total_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            gamma=args.gamma,
            target_sync_interval=args.target_sync_interval,
            hidden_size=args.hidden_size,
            grad_clip_norm=args.grad_clip,
            device=args.device,
        )
