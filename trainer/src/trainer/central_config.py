"""Centralized configuration loading from config.defaults.toml and config.toml.

This module loads default configuration values from config.defaults.toml, which is
the single source of truth for defaults shared between Rust and Python components.
User configuration from config.toml is then overlaid on top of these defaults.

Configuration Priority (highest to lowest):
    1. Environment variables (CARTRIDGE_<SECTION>_<KEY>)
    2. User configuration (config.toml)
    3. Default configuration (config.defaults.toml)

Environment Variable Override Pattern:
    CARTRIDGE_<SECTION>_<KEY>=value

    Examples:
        CARTRIDGE_COMMON_ENV_ID=connect4
        CARTRIDGE_TRAINING_ITERATIONS=50
        CARTRIDGE_EVALUATION_GAMES=100

Usage:
    from trainer.central_config import get_config, Config

    config = get_config()
    print(config.common.env_id)
    print(config.training.iterations)
"""

import logging
import math
import os
import struct
import sys
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# Use tomllib for Python 3.11+, tomli for 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)

_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_MAX_F32 = float.fromhex("0x1.fffffep+127")


def _nonnegative_u32(value: object, *, field_name: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > _MAX_U32
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name} must be a {qualifier} u32 integer")
    return value


def _nonnegative_u64(value: object, *, field_name: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > _MAX_U64
    ):
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


def _validate_simulation_schedule(
    *, iterations: int, start: int, maximum: int, ramp: int
) -> None:
    if start == maximum:
        if ramp != 0:
            raise ValueError(
                "mcts.sim_ramp_rate must be zero when start_sims equals max_sims"
            )
        return
    delta = maximum - start
    if ramp == 0 or ramp > delta:
        raise ValueError(
            "ramped MCTS requires mcts.sim_ramp_rate in " "[1, max_sims - start_sims]"
        )
    steps_to_cap = (delta + ramp - 1) // ramp
    if iterations - 1 < steps_to_cap:
        raise ValueError(
            "MCTS simulation schedule must reach mcts.max_sims within "
            "training.iterations"
        )


# Thread-safe lock for config cache access
_config_lock = threading.Lock()

# Project root and the wheel-owned byte-identical defaults resource.
_PROJECT_ROOT = Path(__file__).parents[3]
_PACKAGED_DEFAULTS = Path(__file__).with_name("config.defaults.toml")

# Default config file locations (searched in order)
CONFIG_SEARCH_PATHS = [
    Path("config.toml"),  # Current directory
    Path("/app/config.toml"),  # Docker container
    _PROJECT_ROOT / "config.toml",  # Project root
]

# Canonical defaults locations. The checkout copy wins during development; an
# installed wheel uses its packaged mirror. Co-present copies must be identical.
DEFAULTS_SEARCH_PATHS = [
    _PROJECT_ROOT / "config.defaults.toml",
    _PACKAGED_DEFAULTS,
]


@dataclass
class CommonConfig:
    """Common settings shared across all components."""

    data_dir: str = "./data"
    env_id: str = "tictactoe"
    log_level: str = "info"


@dataclass
class AlgorithmConfig:
    """Algorithm cartridge selected across collection, learning, and evaluation."""

    id: str = "alphazero_board_v1"


@dataclass
class TrainingConfig:
    """Training loop settings."""

    iterations: int = 100
    episodes_per_iteration: int = 500
    steps_per_iteration: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    checkpoint_interval: int = 100
    num_actors: int = 1  # Number of parallel actor processes for self-play

    def __post_init__(self) -> None:
        _nonnegative_u64(
            self.iterations, field_name="training.iterations", positive=True
        )
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
        _nonnegative_u64(
            self.batch_size, field_name="training.batch_size", positive=True
        )
        _nonnegative_u64(
            self.checkpoint_interval,
            field_name="training.checkpoint_interval",
            positive=True,
        )
        actors = _nonnegative_u32(
            self.num_actors, field_name="training.num_actors", positive=True
        )
        if actors > episodes:
            raise ValueError(
                "training.num_actors cannot exceed training.episodes_per_iteration"
            )


@dataclass
class EvaluationConfig:
    """Evaluation settings."""

    interval: int = 1
    games: int = 50
    win_threshold: float = 0.55  # Win rate needed to become champion
    eval_vs_random: bool = True  # Also evaluate against random baseline
    # MCTS simulations per move during evaluation. 0 plays the policy head
    # directly, which is what evaluation did before it moved into the engine;
    # above 0 the models play with search, which is how they actually play and
    # a fairer measure of strength, at proportionally more eval wall-time.
    simulations: int = 0
    temperature: float = 0.2
    # Perfect-solver move scoring during loop evaluation (connect4 only)
    solver_games: int = 0  # Games per solver eval (connect4 only; 0 = disable)
    evaluation_seed: int = 42  # Fixed seed for every evaluation game family
    promotion_metric: str = "win_rate"  # "win_rate" or "solver_optimal"
    promotion_margin: float = 0.0  # Set explicitly with solver_optimal

    def __post_init__(self) -> None:
        interval = _nonnegative_u64(self.interval, field_name="evaluation.interval")
        games = _nonnegative_u32(
            self.games, field_name="evaluation.games", positive=True
        )
        if not isinstance(self.eval_vs_random, bool):
            raise ValueError("evaluation.eval_vs_random must be boolean")
        _nonnegative_u32(self.simulations, field_name="evaluation.simulations")
        solver_games = _nonnegative_u32(
            self.solver_games, field_name="evaluation.solver_games"
        )
        seed = _nonnegative_u64(
            self.evaluation_seed, field_name="evaluation.evaluation_seed"
        )
        largest_run = max(games, solver_games)
        if seed > _MAX_U64 - (largest_run - 1):
            raise ValueError("evaluation seed schedule exceeds u64")
        self.temperature = _nonnegative_f32(
            self.temperature, field_name="evaluation.temperature"
        )
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
                    "evaluation.promotion_margin must be zero when "
                    "promotion_metric is win_rate"
                )
        elif self.promotion_metric == "solver_optimal":
            if self.win_threshold != 0.0:
                raise ValueError(
                    "evaluation.win_threshold must be zero when promotion_metric "
                    "is solver_optimal"
                )
        else:
            raise ValueError(
                "evaluation.promotion_metric must be win_rate or solver_optimal"
            )
        if interval > 0 and not self.eval_vs_random and solver_games == 0:
            raise ValueError(
                "scheduled evaluation requires first-candidate evidence: enable "
                "evaluation.eval_vs_random or set evaluation.solver_games > 0 "
                "for Connect4"
            )


@dataclass
class ActorConfig:
    """Actor (self-play) settings."""

    actor_id: str = "actor-1"
    episode_timeout_secs: int = 30
    log_interval: int = 50

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

    host: str = "0.0.0.0"
    port: int = 8080
    allowed_origins: list[str] = field(default_factory=list)


@dataclass
class MctsConfig:
    """MCTS (Monte Carlo Tree Search) settings."""

    c_puct: float = 1.4
    temperature: float = 1.0
    late_temperature: float = 1.0
    temp_threshold: int = (
        0  # Move number after which to reduce temperature (0 = disabled)
    )
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    # Simulation ramping: start_sims + (iteration-1) * sim_ramp_rate, capped at max_sims
    start_sims: int = 50  # Simulations for first iteration
    max_sims: int = 400  # Maximum simulations
    sim_ramp_rate: int = 20  # Simulations to add per iteration
    # Batch size for neural network evaluation during MCTS (1 = disabled)
    eval_batch_size: int = 32
    onnx_intra_threads: int = 1

    def __post_init__(self) -> None:
        self.c_puct = _nonnegative_f32(self.c_puct, field_name="mcts.c_puct")
        self.temperature = _nonnegative_f32(
            self.temperature, field_name="mcts.temperature"
        )
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
                "mcts.dirichlet_alpha and mcts.dirichlet_weight must both be zero "
                "to disable noise"
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
        start = _nonnegative_u32(
            self.start_sims, field_name="mcts.start_sims", positive=True
        )
        maximum = _nonnegative_u32(
            self.max_sims, field_name="mcts.max_sims", positive=True
        )
        _nonnegative_u32(self.sim_ramp_rate, field_name="mcts.sim_ramp_rate")
        if start > maximum:
            raise ValueError("mcts.start_sims cannot exceed mcts.max_sims")
        if start == maximum and self.sim_ramp_rate != 0:
            raise ValueError(
                "mcts.sim_ramp_rate must be zero when start_sims equals max_sims"
            )
        if start < maximum and (
            self.sim_ramp_rate == 0 or self.sim_ramp_rate > maximum - start
        ):
            raise ValueError(
                "ramped MCTS requires mcts.sim_ramp_rate in "
                "[1, max_sims - start_sims]"
            )
        _nonnegative_u32(
            self.eval_batch_size,
            field_name="mcts.eval_batch_size",
            positive=True,
        )
        _nonnegative_u32(
            self.onnx_intra_threads,
            field_name="mcts.onnx_intra_threads",
            positive=True,
        )


@dataclass
class StorageConfig:
    """Storage backend settings."""

    model_backend: str = "filesystem"
    postgres_url: str = "postgresql://cartridge:cartridge@localhost:5432/cartridge"
    s3_bucket: str | None = None
    s3_endpoint: str | None = None
    pool_max_size: int = 16
    pool_connect_timeout: int = 30
    pool_idle_timeout: int = 300


@dataclass
class LoggingConfig:
    """Logging format settings."""

    format: str = "text"  # "text" or "json"
    include_timestamps: bool = True
    include_target: bool = True


@dataclass
class WandbConfig:
    """Weights & Biases logging settings (used by the trainer loop)."""

    enabled: bool = False
    required: bool = False  # True: fail loudly instead of null-logger fallback
    project: str = "cartridge2"
    entity: str = ""  # Empty = the logged-in default entity
    group: str = ""  # Groups related runs in the W&B UI
    tags: list[str] = field(default_factory=list)
    init_timeout_seconds: float = 30.0


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

    # Convenience properties for commonly accessed paths
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


_CONFIG_SECTION_TYPES: dict[str, type] = {
    "common": CommonConfig,
    "algorithm": AlgorithmConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
    "actor": ActorConfig,
    "web": WebConfig,
    "mcts": MctsConfig,
    "storage": StorageConfig,
    "logging": LoggingConfig,
    "wandb": WandbConfig,
}
_CONFIG_SECTIONS: dict[str, set[str]] = {
    section: {item.name for item in fields(section_type)}
    for section, section_type in _CONFIG_SECTION_TYPES.items()
}


def _validate_canonical_defaults(data: object) -> None:
    """Require the defaults document to define the complete typed schema."""
    if not isinstance(data, dict):
        raise ValueError("Canonical config.defaults.toml must be a TOML table")
    actual_sections = set(data)
    expected_sections = set(_CONFIG_SECTIONS)
    missing_sections = sorted(expected_sections - actual_sections)
    unknown_sections = sorted(actual_sections - expected_sections)
    if missing_sections or unknown_sections:
        details = []
        if missing_sections:
            details.append("missing sections: " + ", ".join(missing_sections))
        if unknown_sections:
            details.append("unknown sections: " + ", ".join(unknown_sections))
        raise ValueError(
            "Canonical config.defaults.toml schema mismatch ("
            + "; ".join(details)
            + ")"
        )

    for section, expected_keys in _CONFIG_SECTIONS.items():
        values = data[section]
        if not isinstance(values, dict):
            raise ValueError(
                f"Canonical config.defaults.toml [{section}] must be a TOML table"
            )
        actual_keys = set(values)
        optional_none_keys = {
            item.name
            for item in fields(_CONFIG_SECTION_TYPES[section])
            if item.default is None
        }
        missing_keys = sorted(expected_keys - optional_none_keys - actual_keys)
        unknown_keys = sorted(actual_keys - expected_keys)
        if missing_keys or unknown_keys:
            details = []
            if missing_keys:
                details.append("missing keys: " + ", ".join(missing_keys))
            if unknown_keys:
                details.append("unknown keys: " + ", ".join(unknown_keys))
            raise ValueError(
                f"Canonical config.defaults.toml [{section}] schema mismatch ("
                + "; ".join(details)
                + ")"
            )


def _find_defaults_file() -> Path | None:
    """Resolve one canonical defaults copy and reject mirror divergence."""
    existing: list[Path] = []
    for candidate in DEFAULTS_SEARCH_PATHS:
        resolved = candidate.resolve()
        if resolved.is_file() and resolved not in existing:
            existing.append(resolved)
    if not existing:
        return None

    canonical_bytes = existing[0].read_bytes()
    for mirror in existing[1:]:
        if mirror.read_bytes() != canonical_bytes:
            raise RuntimeError(
                "Canonical config.defaults.toml copies diverge: "
                f"{existing[0]} != {mirror}"
            )
    return existing[0]


def _find_config_file() -> Path | None:
    """Find the config.toml file in standard locations."""
    # Check environment variable first
    if "CARTRIDGE_CONFIG" in os.environ:
        env_path = os.environ["CARTRIDGE_CONFIG"]
        path = Path(env_path)
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(
            f"CARTRIDGE_CONFIG points to missing file: {env_path!r}"
        )

    # Search default locations
    project_config = _PROJECT_ROOT / "config.toml"
    cwd_is_project = Path.cwd().resolve().is_relative_to(_PROJECT_ROOT.resolve())
    for path in CONFIG_SEARCH_PATHS:
        # The source-checkout fallback is useful when commands run from
        # Cartridge2/trainer, but must not leak a developer's config.toml into
        # an installed package or an isolated caller in another directory.
        if path == project_config and not cwd_is_project:
            continue
        if path.exists():
            return path.resolve()

    return None


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Deep merge overlay dict into base dict."""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to config data.

    Environment variables follow the pattern: CARTRIDGE_<SECTION>_<KEY>
    For example: CARTRIDGE_TRAINING_ITERATIONS=50

    """
    # Apply CARTRIDGE_* overrides.
    prefix = "CARTRIDGE_"
    for env_var, value in os.environ.items():
        if not env_var.startswith(prefix):
            continue

        # Parse CARTRIDGE_SECTION_KEY format
        parts = env_var[len(prefix) :].lower().split("_", 1)
        if len(parts) != 2:
            continue

        section, key = parts
        # CARTRIDGE also owns operational variables such as
        # CARTRIDGE_EVAL_BINARY. Only section names in the typed config are
        # configuration overrides.
        if section not in _CONFIG_SECTIONS:
            continue
        if key not in _CONFIG_SECTIONS[section]:
            raise ValueError(f"Unknown configuration override: {env_var}")
        if section not in data:
            data[section] = {}

        data[section][key] = _convert_value(value, section, key, data)
        logger.debug(f"Applied override {env_var}={value}")

    return data


def _convert_value(value: str, section: str, key: str, data: dict) -> Any:
    """Convert string value to appropriate type based on existing config."""
    # Try to infer type from existing value
    existing = data.get(section, {}).get(key)

    if existing is not None:
        if isinstance(existing, bool):
            normalized = value.lower()
            if normalized in ("true", "1", "yes"):
                return True
            if normalized in ("false", "0", "no"):
                return False
            raise ValueError(
                f"Invalid boolean value {value!r} for {section}.{key}; "
                "expected true/false"
            )
        elif isinstance(existing, int):
            return int(value)
        elif isinstance(existing, float):
            return float(value)
        elif isinstance(existing, list):
            if not value:
                return []
            items = [item.strip() for item in value.split(",")]
            if any(not item for item in items):
                raise ValueError(
                    f"Invalid list value {value!r} for {section}.{key}; "
                    "expected comma-separated non-empty strings"
                )
            return items

    # Default type inference
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _dict_to_config(data: dict[str, Any]) -> Config:
    """Convert a dictionary to a Config object."""

    unknown_sections = sorted(set(data) - set(_CONFIG_SECTIONS))
    if unknown_sections:
        raise ValueError(
            "Unknown configuration sections: " + ", ".join(unknown_sections)
        )

    def build_section(cls: type, section_name: str) -> Any:
        section_data = data.get(section_name, {})
        valid_fields = {f.name for f in fields(cls)}
        unknown = sorted(k for k in section_data if k not in valid_fields)
        if unknown:
            raise ValueError(
                f"Unknown configuration keys in [{section_name}]: " + ", ".join(unknown)
            )
        return cls(**section_data)

    config = Config(
        common=build_section(CommonConfig, "common"),
        algorithm=build_section(AlgorithmConfig, "algorithm"),
        training=build_section(TrainingConfig, "training"),
        evaluation=build_section(EvaluationConfig, "evaluation"),
        actor=build_section(ActorConfig, "actor"),
        web=build_section(WebConfig, "web"),
        mcts=build_section(MctsConfig, "mcts"),
        storage=build_section(StorageConfig, "storage"),
        logging=build_section(LoggingConfig, "logging"),
        wandb=build_section(WandbConfig, "wandb"),
    )

    if config.training.iterations > _MAX_U64 // config.training.steps_per_iteration:
        raise ValueError(
            "training.iterations * training.steps_per_iteration exceeds u64"
        )
    _validate_simulation_schedule(
        iterations=config.training.iterations,
        start=config.mcts.start_sims,
        maximum=config.mcts.max_sims,
        ramp=config.mcts.sim_ramp_rate,
    )

    def require_nonempty_string(value: Any, path: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} must be a non-empty string")

    require_nonempty_string(config.algorithm.id, "algorithm.id")
    require_nonempty_string(config.common.env_id, "common.env_id")
    require_nonempty_string(config.common.data_dir, "common.data_dir")
    require_nonempty_string(config.common.log_level, "common.log_level")
    require_nonempty_string(config.web.host, "web.host")
    require_nonempty_string(config.storage.postgres_url, "storage.postgres_url")
    require_nonempty_string(config.storage.model_backend, "storage.model_backend")
    from .environment_catalog import get_environment
    from .runtime_profile import resolve_runtime_profile

    resolve_runtime_profile(config.algorithm.id, config.common.env_id)
    max_horizon = get_environment(config.common.env_id).capabilities.max_horizon
    if max_horizon is None:
        raise ValueError("selected synchronized environment must declare max_horizon")
    if config.mcts.temp_threshold != 0 and config.mcts.temp_threshold >= max_horizon:
        raise ValueError(
            "mcts.temp_threshold must be less than the selected environment max_horizon"
        )
    if not isinstance(config.web.allowed_origins, list) or not all(
        isinstance(origin, str) and origin for origin in config.web.allowed_origins
    ):
        raise ValueError("web.allowed_origins must be an array of non-empty strings")
    if not isinstance(config.wandb.tags, list) or not all(
        isinstance(tag, str) and tag for tag in config.wandb.tags
    ):
        raise ValueError("wandb.tags must be an array of non-empty strings")
    if config.storage.model_backend not in {"filesystem", "s3"}:
        raise ValueError(
            "storage.model_backend must be 'filesystem' or 's3', got "
            f"{config.storage.model_backend!r}"
        )
    if config.storage.model_backend == "s3" and (
        not isinstance(config.storage.s3_bucket, str)
        or not config.storage.s3_bucket.strip()
    ):
        raise ValueError(
            "storage.s3_bucket is required when storage.model_backend is 's3'"
        )
    if config.storage.s3_endpoint is not None and (
        not isinstance(config.storage.s3_endpoint, str)
        or not config.storage.s3_endpoint.strip()
    ):
        raise ValueError("storage.s3_endpoint must be a non-empty string when set")
    for value, path in (
        (config.storage.pool_max_size, "storage.pool_max_size"),
        (config.storage.pool_connect_timeout, "storage.pool_connect_timeout"),
        (config.storage.pool_idle_timeout, "storage.pool_idle_timeout"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{path} must be a positive integer")
    if config.logging.format not in {"text", "json"}:
        raise ValueError("logging.format must be 'text' or 'json'")
    if config.evaluation.promotion_metric not in {"win_rate", "solver_optimal"}:
        raise ValueError(
            "evaluation.promotion_metric must be 'win_rate' or 'solver_optimal'"
        )
    if config.evaluation.solver_games > 0 and config.common.env_id != "connect4":
        raise ValueError(
            "evaluation.solver_games may be nonzero only when common.env_id is "
            "'connect4'"
        )
    if config.evaluation.promotion_metric == "solver_optimal" and (
        config.common.env_id != "connect4" or config.evaluation.solver_games == 0
    ):
        raise ValueError(
            "evaluation.promotion_metric='solver_optimal' requires connect4 with "
            "evaluation.solver_games > 0"
        )
    if (
        isinstance(config.wandb.init_timeout_seconds, bool)
        or not isinstance(config.wandb.init_timeout_seconds, (int, float))
        or not math.isfinite(float(config.wandb.init_timeout_seconds))
        or config.wandb.init_timeout_seconds <= 0
    ):
        raise ValueError("wandb.init_timeout_seconds must be greater than zero")
    return config


# Cached config instance
_cached_config: Config | None = None


def get_config(reload: bool = False) -> Config:
    """Get the configuration, loading from file if needed.

    Configuration is loaded with the following priority (highest to lowest):
        1. Environment variables (CARTRIDGE_<SECTION>_<KEY>)
        2. User configuration (config.toml)
        3. Default configuration (config.defaults.toml)

    This function is thread-safe and caches the configuration after first load.

    Args:
        reload: Force reload from file even if cached.

    Returns:
        The Config object with all settings.
    """
    global _cached_config

    # Fast path: check cache without locking
    if _cached_config is not None and not reload:
        return _cached_config

    # Slow path: load config with lock
    with _config_lock:
        # Double-check after acquiring lock
        if _cached_config is not None and not reload:
            return _cached_config

        # Step 1: Load defaults from config.defaults.toml
        defaults_path = _find_defaults_file()
        if defaults_path is None:
            raise FileNotFoundError(
                "canonical config.defaults.toml was not found in any configured location"
            )
        logger.debug(f"Loading defaults from {defaults_path}")
        with open(defaults_path, "rb") as f:
            data = tomllib.load(f)
        _validate_canonical_defaults(data)

        # Step 2: Overlay user configuration from config.toml
        config_path = _find_config_file()
        if config_path is not None:
            logger.info(f"Loading user configuration from {config_path}")
            with open(config_path, "rb") as f:
                user_data = tomllib.load(f)
            data = _deep_merge(data, user_data)

        # Step 3: Apply environment variable overrides
        data = _apply_env_overrides(data)

        _cached_config = _dict_to_config(data)
        return _cached_config


def reset_config() -> None:
    """Reset the cached config (mainly for testing).

    This function is thread-safe.
    """
    global _cached_config
    with _config_lock:
        _cached_config = None
