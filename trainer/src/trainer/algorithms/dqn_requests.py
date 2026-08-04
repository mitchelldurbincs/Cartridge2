"""Typed DQN application requests and the CLI defaults used to build them."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from ..storage.base import ReplaySelection


@dataclass(frozen=True)
class DqnCollectCliDefaults:
    env_id: str = "counter"
    epsilon: float = 1.0
    seed: int = 0
    onnx_intra_threads: int = 1
    actor_id: str = "dqn-collector"
    episode_timeout_secs: int = 30
    log_level: str = "INFO"


@dataclass(frozen=True)
class DqnEvaluateCliDefaults:
    env_id: str = "counter"
    episodes: int = 100
    seed: int = 42
    onnx_intra_threads: int = 1


@dataclass(frozen=True)
class DqnLoopCliDefaults:
    env_id: str = "counter"
    episodes_per_iteration: int = 100
    epsilon_start: float = 0.25
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.95
    seed: int = 0
    onnx_intra_threads: int = 1
    evaluation_episodes: int = 100
    episode_timeout_secs: int = 30
    log_level: str = "INFO"


COLLECT_DEFAULTS = DqnCollectCliDefaults()
EVALUATE_DEFAULTS = DqnEvaluateCliDefaults()
LOOP_DEFAULTS = DqnLoopCliDefaults()


@dataclass(frozen=True)
class DqnCollectRequest:
    env_id: str
    episodes: int
    collection_scope_id: str
    source_checkpoint_id: str | None
    epsilon: float
    seed: int
    onnx_intra_threads: int
    actor_id: str
    episode_timeout_secs: int
    actor_binary: Path | None
    data_root: Path
    log_level: str

    def __post_init__(self) -> None:
        _require_identity(self.env_id, "env_id")
        _require_identity(self.collection_scope_id, "collection_scope_id")
        _require_optional_identity(self.source_checkpoint_id, "source_checkpoint_id")
        _positive_int(self.episodes, "episodes")
        _positive_int(self.onnx_intra_threads, "onnx_intra_threads")
        _positive_int(self.episode_timeout_secs, "episode_timeout_secs")
        _u64(self.seed, "seed")
        if not math.isfinite(self.epsilon) or not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        _require_identity(self.actor_id, "actor_id")
        _require_identity(self.log_level, "log_level")
        _require_path(self.data_root, "data_root")
        _require_optional_path(self.actor_binary, "actor_binary")


@dataclass(frozen=True)
class DqnTrainRequest:
    env_id: str
    replay_selection: ReplaySelection
    model_dir: Path
    stats_path: Path
    total_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gamma: float
    target_sync_interval: int
    hidden_size: int
    grad_clip_norm: float
    device: str
    log_level: str

    def __post_init__(self) -> None:
        _require_identity(self.env_id, "env_id")
        if not isinstance(self.replay_selection, ReplaySelection):
            raise TypeError("replay_selection must be an explicit ReplaySelection")
        _require_path(self.model_dir, "model_dir")
        _require_path(self.stats_path, "stats_path")
        _require_identity(self.log_level, "log_level")


@dataclass(frozen=True)
class DqnEvaluateRequest:
    env_id: str
    episodes: int
    seed: int
    checkpoint_id: str | None
    model_dir: Path
    eval_binary: Path | None
    onnx_intra_threads: int

    def __post_init__(self) -> None:
        _require_identity(self.env_id, "env_id")
        _require_optional_identity(self.checkpoint_id, "checkpoint_id")
        if isinstance(self.episodes, bool) or not 0 < self.episodes < 1 << 32:
            raise ValueError("episodes must be a positive u32")
        _u64(self.seed, "seed")
        if self.seed > (1 << 64) - self.episodes:
            raise ValueError("seed plus episode index exceeds u64")
        _positive_int(self.onnx_intra_threads, "onnx_intra_threads")
        _require_path(self.model_dir, "model_dir")
        _require_optional_path(self.eval_binary, "eval_binary")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _u64(value: int, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < 1 << 64
    ):
        raise ValueError(f"{name} must be a u64")


def _require_identity(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_optional_identity(value: str | None, name: str) -> None:
    if value is not None:
        _require_identity(value, name)


def _require_path(value: Path, name: str) -> None:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")


def _require_optional_path(value: Path | None, name: str) -> None:
    if value is not None:
        _require_path(value, name)
