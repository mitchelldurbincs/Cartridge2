"""Typed configuration sections backed by the canonical TOML defaults."""

import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_MAX_F32 = float.fromhex("0x1.fffffep+127")

PROJECT_ROOT = Path(__file__).parents[3]
PACKAGED_DEFAULTS = Path(__file__).with_name("config.defaults.toml")
CANONICAL_DEFAULTS_PATHS = [PROJECT_ROOT / "config.defaults.toml", PACKAGED_DEFAULTS]


def _load_canonical_defaults() -> dict[str, dict[str, Any]]:
    existing = [path.resolve() for path in CANONICAL_DEFAULTS_PATHS if path.is_file()]
    if not existing:
        raise FileNotFoundError("canonical config.defaults.toml was not found")
    canonical_bytes = existing[0].read_bytes()
    for mirror in existing[1:]:
        if mirror.read_bytes() != canonical_bytes:
            raise RuntimeError(
                f"Canonical config.defaults.toml copies diverge: {existing[0]} != {mirror}"
            )
    return tomllib.loads(canonical_bytes.decode("utf-8"))


CANONICAL_DEFAULTS_DATA = _load_canonical_defaults()


def _default(section: str, key: str) -> Any:
    return CANONICAL_DEFAULTS_DATA[section][key]


def _default_list(section: str, key: str) -> list[str]:
    return list(_default(section, key))


def _nonnegative_u32(value: object, *, field_name: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > _MAX_U32:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name} must be a {qualifier} u32 integer")
    return value


def _nonnegative_u64(value: object, *, field_name: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > _MAX_U64:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name} must be a {qualifier} u64 integer")
    return value


def _nonnegative_f32(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite nonnegative f32")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= _MAX_F32:
        raise ValueError(f"{field_name} must be a finite nonnegative f32")
    narrowed = float(struct.unpack("!f", struct.pack("!f", normalized))[0])
    return 0.0 if narrowed == 0.0 else narrowed


def validate_simulation_schedule(*, iterations: int, start: int, maximum: int, ramp: int) -> None:
    if start == maximum:
        if ramp != 0:
            raise ValueError("mcts.sim_ramp_rate must be zero when start_sims equals max_sims")
        return
    delta = maximum - start
    if ramp == 0 or ramp > delta:
        raise ValueError("ramped MCTS requires mcts.sim_ramp_rate in [1, max_sims - start_sims]")
    steps_to_cap = (delta + ramp - 1) // ramp
    if iterations - 1 < steps_to_cap:
        raise ValueError(
            "MCTS simulation schedule must reach mcts.max_sims within training.iterations"
        )


@dataclass
class CommonConfig:
    """Common settings shared across all components."""

    data_dir: str = _default("common", "data_dir")
    env_id: str = _default("common", "env_id")
    log_level: str = _default("common", "log_level")


@dataclass
class AlgorithmConfig:
    """Algorithm cartridge selected across collection, learning, and evaluation."""

    id: str = _default("algorithm", "id")


@dataclass
class TrainingConfig:
    """Training loop settings."""

    iterations: int = _default("training", "iterations")
    episodes_per_iteration: int = _default("training", "episodes_per_iteration")
    steps_per_iteration: int = _default("training", "steps_per_iteration")
    batch_size: int = _default("training", "batch_size")
    learning_rate: float = _default("training", "learning_rate")
    weight_decay: float = _default("training", "weight_decay")
    grad_clip_norm: float = _default("training", "grad_clip_norm")
    device: str = _default("training", "device")
    checkpoint_interval: int = _default("training", "checkpoint_interval")
    num_actors: int = _default("training", "num_actors")

    def __post_init__(self) -> None:
        _nonnegative_u64(self.iterations, field_name="training.iterations", positive=True)
        episodes = _nonnegative_u32(
            self.episodes_per_iteration,
            field_name="training.episodes_per_iteration",
            positive=True,
        )
        _nonnegative_u64(
            self.steps_per_iteration,
            field_name="training.steps_per_iteration",
            positive=True,
        )
        _nonnegative_u64(self.batch_size, field_name="training.batch_size", positive=True)
        _nonnegative_u64(
            self.checkpoint_interval,
            field_name="training.checkpoint_interval",
            positive=True,
        )
        actors = _nonnegative_u32(self.num_actors, field_name="training.num_actors", positive=True)
        if actors > episodes:
            raise ValueError("training.num_actors cannot exceed training.episodes_per_iteration")


@dataclass
class EvaluationConfig:
    """Evaluation settings."""

    interval: int = _default("evaluation", "interval")
    games: int = _default("evaluation", "games")
    win_threshold: float = _default("evaluation", "win_threshold")
    eval_vs_random: bool = _default("evaluation", "eval_vs_random")
    simulations: int = _default("evaluation", "simulations")
    temperature: float = _default("evaluation", "temperature")
    solver_games: int = _default("evaluation", "solver_games")
    evaluation_seed: int = _default("evaluation", "evaluation_seed")
    promotion_metric: str = _default("evaluation", "promotion_metric")
    promotion_margin: float = _default("evaluation", "promotion_margin")

    def __post_init__(self) -> None:
        interval = _nonnegative_u64(self.interval, field_name="evaluation.interval")
        games = _nonnegative_u32(self.games, field_name="evaluation.games", positive=True)
        if not isinstance(self.eval_vs_random, bool):
            raise ValueError("evaluation.eval_vs_random must be boolean")
        _nonnegative_u32(self.simulations, field_name="evaluation.simulations")
        solver_games = _nonnegative_u32(self.solver_games, field_name="evaluation.solver_games")
        seed = _nonnegative_u64(self.evaluation_seed, field_name="evaluation.evaluation_seed")
        largest_run = max(games, solver_games)
        if seed > _MAX_U64 - (largest_run - 1):
            raise ValueError("evaluation seed schedule exceeds u64")
        self.temperature = _nonnegative_f32(self.temperature, field_name="evaluation.temperature")
        for field_name in ("win_threshold", "promotion_margin"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"evaluation.{field_name} must be a rate in [0, 1]")
        if self.promotion_metric == "win_rate":
            if self.promotion_margin != 0.0:
                raise ValueError(
                    "evaluation.promotion_margin must be zero when promotion_metric is win_rate"
                )
        elif self.promotion_metric == "solver_optimal":
            if self.win_threshold != 0.0:
                raise ValueError(
                    "evaluation.win_threshold must be zero when promotion_metric is solver_optimal"
                )
        else:
            raise ValueError("evaluation.promotion_metric must be win_rate or solver_optimal")
        if interval > 0 and not self.eval_vs_random and solver_games == 0:
            raise ValueError(
                "scheduled evaluation requires first-candidate evidence: enable "
                "evaluation.eval_vs_random or set evaluation.solver_games > 0 "
                "for Connect4"
            )


@dataclass
class ActorConfig:
    """Actor (self-play) settings."""

    actor_id: str = _default("actor", "actor_id")
    episode_timeout_secs: int = _default("actor", "episode_timeout_secs")
    log_interval: int = _default("actor", "log_interval")

    def __post_init__(self) -> None:
        _nonnegative_u64(
            self.episode_timeout_secs,
            field_name="actor.episode_timeout_secs",
            positive=True,
        )
        _nonnegative_u32(self.log_interval, field_name="actor.log_interval")


@dataclass
class WebConfig:
    """Web server settings."""

    host: str = _default("web", "host")
    port: int = _default("web", "port")
    allowed_origins: list[str] = field(
        default_factory=lambda: _default_list("web", "allowed_origins")
    )


@dataclass
class MctsConfig:
    """MCTS (Monte Carlo Tree Search) settings."""

    c_puct: float = _default("mcts", "c_puct")
    temperature: float = _default("mcts", "temperature")
    late_temperature: float = _default("mcts", "late_temperature")
    temp_threshold: int = _default("mcts", "temp_threshold")
    dirichlet_alpha: float = _default("mcts", "dirichlet_alpha")
    dirichlet_weight: float = _default("mcts", "dirichlet_weight")
    start_sims: int = _default("mcts", "start_sims")
    max_sims: int = _default("mcts", "max_sims")
    sim_ramp_rate: int = _default("mcts", "sim_ramp_rate")
    eval_batch_size: int = _default("mcts", "eval_batch_size")
    onnx_intra_threads: int = _default("mcts", "onnx_intra_threads")

    def __post_init__(self) -> None:
        self.c_puct = _nonnegative_f32(self.c_puct, field_name="mcts.c_puct")
        self.temperature = _nonnegative_f32(self.temperature, field_name="mcts.temperature")
        self.late_temperature = _nonnegative_f32(
            self.late_temperature, field_name="mcts.late_temperature"
        )
        self.dirichlet_alpha = _nonnegative_f32(
            self.dirichlet_alpha, field_name="mcts.dirichlet_alpha"
        )
        self.dirichlet_weight = _nonnegative_f32(
            self.dirichlet_weight, field_name="mcts.dirichlet_weight"
        )
        if self.dirichlet_weight > 1.0:
            raise ValueError("mcts.dirichlet_weight must be a rate in [0, 1]")
        if (self.dirichlet_alpha == 0.0) != (self.dirichlet_weight == 0.0):
            raise ValueError(
                "mcts.dirichlet_alpha and mcts.dirichlet_weight must both be zero to disable noise"
            )
        if self.temp_threshold == 0:
            if self.late_temperature != self.temperature:
                raise ValueError(
                    "mcts.late_temperature must equal mcts.temperature when "
                    "mcts.temp_threshold is zero"
                )
        elif self.late_temperature == self.temperature:
            raise ValueError(
                "mcts.late_temperature must differ from mcts.temperature when "
                "the schedule is enabled"
            )
        _nonnegative_u32(self.temp_threshold, field_name="mcts.temp_threshold")
        start = _nonnegative_u32(self.start_sims, field_name="mcts.start_sims", positive=True)
        maximum = _nonnegative_u32(self.max_sims, field_name="mcts.max_sims", positive=True)
        _nonnegative_u32(self.sim_ramp_rate, field_name="mcts.sim_ramp_rate")
        if start > maximum:
            raise ValueError("mcts.start_sims cannot exceed mcts.max_sims")
        if start == maximum and self.sim_ramp_rate != 0:
            raise ValueError("mcts.sim_ramp_rate must be zero when start_sims equals max_sims")
        if start < maximum and (self.sim_ramp_rate == 0 or self.sim_ramp_rate > maximum - start):
            raise ValueError(
                "ramped MCTS requires mcts.sim_ramp_rate in [1, max_sims - start_sims]"
            )
        _nonnegative_u32(self.eval_batch_size, field_name="mcts.eval_batch_size", positive=True)
        _nonnegative_u32(
            self.onnx_intra_threads,
            field_name="mcts.onnx_intra_threads",
            positive=True,
        )


@dataclass
class StorageConfig:
    """Storage backend settings."""

    model_backend: str = _default("storage", "model_backend")
    postgres_url: str = _default("storage", "postgres_url")
    s3_bucket: str | None = None
    s3_endpoint: str | None = None
    pool_max_size: int = _default("storage", "pool_max_size")
    pool_connect_timeout: int = _default("storage", "pool_connect_timeout")
    pool_idle_timeout: int = _default("storage", "pool_idle_timeout")
    replay_retained_scopes: int = _default("storage", "replay_retained_scopes")
    learner_state_retained_checkpoints: int = _default(
        "storage", "learner_state_retained_checkpoints"
    )


@dataclass
class LoggingConfig:
    """Logging format settings."""

    format: str = _default("logging", "format")
    include_timestamps: bool = _default("logging", "include_timestamps")
    include_target: bool = _default("logging", "include_target")


@dataclass
class WandbConfig:
    """Weights & Biases logging settings."""

    enabled: bool = _default("wandb", "enabled")
    required: bool = _default("wandb", "required")
    project: str = _default("wandb", "project")
    entity: str = _default("wandb", "entity")
    group: str = _default("wandb", "group")
    tags: list[str] = field(default_factory=lambda: _default_list("wandb", "tags"))
    init_timeout_seconds: float = _default("wandb", "init_timeout_seconds")


@dataclass
class Config:
    """Root configuration container."""

    common: CommonConfig = field(default_factory=CommonConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    web: WebConfig = field(default_factory=WebConfig)
    mcts: MctsConfig = field(default_factory=MctsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @property
    def data_root(self) -> Path:
        return Path(self.common.data_dir)

    @property
    def runtime_profile(self):
        from .runtime_profile import resolve_runtime_profile

        return resolve_runtime_profile(self.algorithm.id, self.common.env_id)

    @property
    def data_dir(self) -> Path:
        return self.runtime_profile.data_dir(self.data_root)

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def stats_path(self) -> Path:
        return self.data_dir / "stats.json"

    @property
    def loop_stats_path(self) -> Path:
        return self.data_dir / "loop_stats.json"

    @property
    def eval_stats_path(self) -> Path:
        return self.data_dir / "eval_stats.json"
