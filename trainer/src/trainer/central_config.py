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

import json
import logging
import math
import os
import sys
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_args, get_origin

# Use tomllib for Python 3.11+, tomli for 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)

# Thread-safe lock for config cache access
_config_lock = threading.Lock()

# Project root (where config.defaults.toml lives)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent

# Default config file locations (searched in order)
CONFIG_SEARCH_PATHS = [
    Path("config.toml"),  # Current directory
    Path("/app/config.toml"),  # Docker container
    _PROJECT_ROOT / "config.toml",  # Project root
]

# Defaults file locations (searched in order)
DEFAULTS_SEARCH_PATHS = [
    Path("config.defaults.toml"),  # Current directory
    Path("/app/config.defaults.toml"),  # Docker container
    _PROJECT_ROOT / "config.defaults.toml",  # Project root
]


@dataclass
class CommonConfig:
    """Common settings shared across all components."""

    data_dir: str = "./data"
    env_id: str = "tictactoe"
    log_level: str = "info"


@dataclass
class TrainingConfig:
    """Training loop settings."""

    iterations: int = 100
    start_iteration: int = 1
    episodes_per_iteration: int = 500
    steps_per_iteration: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip_norm: float = 1.0
    device: str = "auto"
    checkpoint_interval: int = 100
    max_checkpoints: int = 10
    num_actors: int = 1  # Number of parallel actor processes for self-play


@dataclass
class EvaluationConfig:
    """Evaluation settings."""

    interval: int = 1
    games: int = 50
    win_threshold: float = 0.55  # Win rate needed to become new best model
    eval_vs_random: bool = True  # Also evaluate against random baseline
    # MCTS simulations per move during evaluation. 0 plays the policy head
    # directly, which is what evaluation did before it moved into the engine;
    # above 0 the models play with search, which is how they actually play and
    # a fairer measure of strength, at proportionally more eval wall-time.
    simulations: int = 0
    # Perfect-solver move scoring during loop evaluation (connect4 only)
    solver_games: int = 100  # Games per solver eval (0 = disable)
    solver_seed: int = 42  # Fixed seed so rates are comparable across iterations
    promotion_metric: str = "win_rate"  # "win_rate" or "solver_optimal"
    promotion_margin: float = 0.01  # solver_optimal: candidate must exceed best by this


@dataclass
class ActorConfig:
    """Actor (self-play) settings."""

    actor_id: str = "actor-1"
    max_episodes: int = -1
    episode_timeout_secs: int = 30
    flush_interval_secs: int = 5
    log_interval: int = 50
    health_port: int = 8081


@dataclass
class WebConfig:
    """Web server settings."""

    host: str = "0.0.0.0"
    port: int = 8080
    allowed_origins: list[str] = field(default_factory=list)


@dataclass
class MctsConfig:
    """MCTS (Monte Carlo Tree Search) settings."""

    num_simulations: int = 800
    c_puct: float = 1.4
    temperature: float = 1.0
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
    eval_batch_size: int = 1
    onnx_intra_threads: int = 1


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
    def data_dir(self) -> Path:
        return Path(self.common.data_dir)

    @property
    def replay_db_path(self) -> Path:
        return self.data_dir / "replay.db"

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

    def validate(self) -> None:
        """Reject values with unambiguous invalid or unsafe semantics."""

        _require_non_empty_string("common.data_dir", self.common.data_dir)
        _require_non_empty_string("common.env_id", self.common.env_id)
        _require_choice(
            "common.log_level",
            self.common.log_level,
            # Match the choices accepted by the Python entry points. Rust also
            # supports trace and uses warn rather than warning.
            {"debug", "info", "warning", "error"},
        )

        for name, value in (
            ("training.iterations", self.training.iterations),
            ("training.start_iteration", self.training.start_iteration),
            (
                "training.episodes_per_iteration",
                self.training.episodes_per_iteration,
            ),
            ("training.steps_per_iteration", self.training.steps_per_iteration),
            ("training.batch_size", self.training.batch_size),
            ("training.checkpoint_interval", self.training.checkpoint_interval),
            ("training.num_actors", self.training.num_actors),
        ):
            _require_positive_int(name, value)
        _require_non_negative_int(
            "training.max_checkpoints", self.training.max_checkpoints
        )
        _require_positive_number("training.learning_rate", self.training.learning_rate)
        _require_non_negative_number(
            "training.weight_decay", self.training.weight_decay
        )
        _require_non_negative_number(
            "training.grad_clip_norm", self.training.grad_clip_norm
        )
        _require_choice(
            "training.device",
            self.training.device,
            {"auto", "cpu", "cuda", "mps"},
        )

        _require_non_negative_int("evaluation.interval", self.evaluation.interval)
        _require_positive_int("evaluation.games", self.evaluation.games)
        _require_unit_interval(
            "evaluation.win_threshold", self.evaluation.win_threshold
        )
        _require_bool("evaluation.eval_vs_random", self.evaluation.eval_vs_random)
        _require_non_negative_int("evaluation.simulations", self.evaluation.simulations)
        _require_non_negative_int(
            "evaluation.solver_games", self.evaluation.solver_games
        )
        _require_int("evaluation.solver_seed", self.evaluation.solver_seed)
        _require_choice(
            "evaluation.promotion_metric",
            self.evaluation.promotion_metric,
            {"win_rate", "solver_optimal"},
        )
        _require_non_negative_number(
            "evaluation.promotion_margin", self.evaluation.promotion_margin
        )

        _require_non_empty_string("actor.actor_id", self.actor.actor_id)
        _require_int("actor.max_episodes", self.actor.max_episodes)
        if self.actor.max_episodes != -1 and self.actor.max_episodes <= 0:
            raise ValueError(
                "actor.max_episodes must be -1 (unlimited) or greater than zero"
            )
        _require_positive_int(
            "actor.episode_timeout_secs", self.actor.episode_timeout_secs
        )
        _require_positive_int(
            "actor.flush_interval_secs", self.actor.flush_interval_secs
        )
        _require_non_negative_int("actor.log_interval", self.actor.log_interval)
        _require_port("actor.health_port", self.actor.health_port)

        _require_non_empty_string("web.host", self.web.host)
        _require_port("web.port", self.web.port)
        if not isinstance(self.web.allowed_origins, list) or not all(
            isinstance(origin, str) and origin.strip()
            for origin in self.web.allowed_origins
        ):
            raise ValueError("web.allowed_origins must be a list of non-empty strings")

        _require_positive_int("mcts.num_simulations", self.mcts.num_simulations)
        _require_non_negative_number("mcts.c_puct", self.mcts.c_puct)
        _require_non_negative_number("mcts.temperature", self.mcts.temperature)
        _require_non_negative_int("mcts.temp_threshold", self.mcts.temp_threshold)
        _require_non_negative_number("mcts.dirichlet_alpha", self.mcts.dirichlet_alpha)
        _require_unit_interval("mcts.dirichlet_weight", self.mcts.dirichlet_weight)
        _require_positive_int("mcts.start_sims", self.mcts.start_sims)
        _require_positive_int("mcts.max_sims", self.mcts.max_sims)
        if self.mcts.start_sims > self.mcts.max_sims:
            raise ValueError("mcts.start_sims must not exceed mcts.max_sims")
        _require_non_negative_int("mcts.sim_ramp_rate", self.mcts.sim_ramp_rate)
        _require_positive_int("mcts.eval_batch_size", self.mcts.eval_batch_size)
        # Zero is documented as ONNX Runtime auto-detection.
        _require_non_negative_int(
            "mcts.onnx_intra_threads", self.mcts.onnx_intra_threads
        )

        _require_choice(
            "storage.model_backend",
            self.storage.model_backend,
            {"filesystem", "s3"},
        )
        _require_non_empty_string("storage.postgres_url", self.storage.postgres_url)
        _require_optional_string("storage.s3_bucket", self.storage.s3_bucket)
        _require_optional_string("storage.s3_endpoint", self.storage.s3_endpoint)
        _require_positive_int("storage.pool_max_size", self.storage.pool_max_size)
        _require_positive_int(
            "storage.pool_connect_timeout", self.storage.pool_connect_timeout
        )
        _require_positive_int(
            "storage.pool_idle_timeout", self.storage.pool_idle_timeout
        )
        _require_choice("logging.format", self.logging.format, {"text", "json"})
        _require_bool("logging.include_timestamps", self.logging.include_timestamps)
        _require_bool("logging.include_target", self.logging.include_target)
        _require_bool("wandb.enabled", self.wandb.enabled)
        _require_bool("wandb.required", self.wandb.required)
        _require_non_empty_string("wandb.project", self.wandb.project)
        _require_string("wandb.entity", self.wandb.entity)
        _require_string("wandb.group", self.wandb.group)
        _require_positive_number(
            "wandb.init_timeout_seconds", self.wandb.init_timeout_seconds
        )
        if not isinstance(self.wandb.tags, list) or not all(
            isinstance(tag, str) for tag in self.wandb.tags
        ):
            raise ValueError("wandb.tags must be a list of strings")


def _require_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _require_positive_int(name: str, value: Any) -> None:
    _require_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _require_non_negative_int(name: str, value: Any) -> None:
    _require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be zero or greater")


def _require_finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _require_positive_number(name: str, value: Any) -> None:
    if _require_finite_number(name, value) <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _require_non_negative_number(name: str, value: Any) -> None:
    if _require_finite_number(name, value) < 0:
        raise ValueError(f"{name} must be zero or greater")


def _require_unit_interval(name: str, value: Any) -> None:
    numeric = _require_finite_number(name, value)
    if not 0 <= numeric <= 1:
        raise ValueError(f"{name} must be between zero and one")


def _require_non_empty_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_string(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")


def _require_optional_string(name: str, value: Any) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a string when set")


def _require_choice(name: str, value: Any, choices: set[str]) -> None:
    _require_non_empty_string(name, value)
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {expected}")


def _require_port(name: str, value: Any) -> None:
    _require_int(name, value)
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")


_CONFIG_SECTION_TYPES = {
    "common": CommonConfig,
    "training": TrainingConfig,
    "evaluation": EvaluationConfig,
    "actor": ActorConfig,
    "web": WebConfig,
    "mcts": MctsConfig,
    "storage": StorageConfig,
    "logging": LoggingConfig,
    "wandb": WandbConfig,
}

# Operational settings consumed outside the central configuration model. They
# share a recognized section prefix, so list them explicitly to avoid reporting
# valid deployment variables as typos.
_NON_CENTRAL_CARTRIDGE_ENV_VARS = {
    "CARTRIDGE_STORAGE_GCS_BUCKET",
    "CARTRIDGE_STORAGE_REPLAY_BACKEND",
}


def _find_defaults_file() -> Path | None:
    """Find the config.defaults.toml file in standard locations."""
    for path in DEFAULTS_SEARCH_PATHS:
        if path.exists():
            return path
    return None


def _find_config_file() -> Path | None:
    """Find the config.toml file in standard locations."""
    # Check environment variable first
    env_path = os.environ.get("CARTRIDGE_CONFIG")
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise FileNotFoundError(f"CARTRIDGE_CONFIG points to missing file: {path}")
        return path

    # Search default locations
    for path in CONFIG_SEARCH_PATHS:
        if path.exists():
            return path

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


def _warn_unknown_config_keys(data: dict[str, Any], source: Path) -> None:
    """Warn about unknown TOML keys without rejecting cross-version configs."""
    for section_name, section_data in data.items():
        section_type = _CONFIG_SECTION_TYPES.get(section_name)
        if section_type is None:
            logger.warning(
                "Unknown configuration section [%s] in %s; ignoring it",
                section_name,
                source,
            )
            continue
        if not isinstance(section_data, dict):
            raise ValueError(
                f"Configuration section [{section_name}] in {source} must be a table"
            )

        valid_fields = {config_field.name for config_field in fields(section_type)}
        for key in sorted(set(section_data) - valid_fields):
            logger.warning(
                "Unknown configuration key %s.%s in %s; ignoring it",
                section_name,
                key,
                source,
            )


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to config data.

    Environment variables follow the pattern: CARTRIDGE_<SECTION>_<KEY>
    For example: CARTRIDGE_TRAINING_ITERATIONS=50

    Also supports legacy ALPHAZERO_* variables for backward compatibility.
    """
    # Legacy mapping: ALPHAZERO_* -> new config paths
    legacy_mapping = {
        "ALPHAZERO_ENV_ID": ("common", "env_id"),
        "ALPHAZERO_ITERATIONS": ("training", "iterations"),
        "ALPHAZERO_START_ITERATION": ("training", "start_iteration"),
        "ALPHAZERO_EPISODES": ("training", "episodes_per_iteration"),
        "ALPHAZERO_STEPS": ("training", "steps_per_iteration"),
        "ALPHAZERO_BATCH_SIZE": ("training", "batch_size"),
        "ALPHAZERO_LR": ("training", "learning_rate"),
        "ALPHAZERO_DEVICE": ("training", "device"),
        "ALPHAZERO_CHECKPOINT_INTERVAL": ("training", "checkpoint_interval"),
        "ALPHAZERO_EVAL_INTERVAL": ("evaluation", "interval"),
        "ALPHAZERO_EVAL_GAMES": ("evaluation", "games"),
        "DATA_DIR": ("common", "data_dir"),
    }

    # Apply legacy overrides
    for env_var, (section, key) in legacy_mapping.items():
        value = os.environ.get(env_var)
        if value is not None and value != "":
            if section not in data:
                data[section] = {}
            # Convert to appropriate type
            data[section][key] = _convert_value(value, section, key, env_var=env_var)
            logger.debug(f"Applied legacy override {env_var}={value}")

    # Apply CARTRIDGE_* overrides (higher priority)
    prefix = "CARTRIDGE_"
    for env_var, value in os.environ.items():
        if not env_var.startswith(prefix):
            continue

        if env_var in _NON_CENTRAL_CARTRIDGE_ENV_VARS:
            continue

        # Skip empty values
        if value == "":
            continue

        # Parse CARTRIDGE_SECTION_KEY format
        parts = env_var[len(prefix) :].lower().split("_", 1)
        if len(parts) != 2:
            continue

        section, key = parts
        section_type = _CONFIG_SECTION_TYPES.get(section)
        if section_type is None:
            # Other Cartridge components use operational variables such as
            # CARTRIDGE_TRACE_ID and CARTRIDGE_EVAL_BINARY. They are not
            # central-config overrides.
            continue
        valid_fields = {config_field.name for config_field in fields(section_type)}
        if key not in valid_fields:
            logger.warning(
                "Unknown central configuration environment variable %s; ignoring it",
                env_var,
            )
            continue

        if section not in data or not isinstance(data[section], dict):
            data[section] = {}

        data[section][key] = _convert_value(value, section, key, env_var=env_var)
        logger.debug(f"Applied override {env_var}={value}")

    return data


def _convert_value(
    value: str,
    section: str,
    key: str,
    *,
    env_var: str,
) -> Any:
    """Convert an environment value using the known central-config schema."""
    # Always derive the type from the dataclass schema. A malformed lower-
    # priority TOML value must not change how a valid env override is parsed.
    section_type = _CONFIG_SECTION_TYPES[section]
    config_field = next(
        config_field
        for config_field in fields(section_type)
        if config_field.name == key
    )
    annotation = config_field.type
    if annotation in (bool, int, float, str):
        expected_type: type[Any] | None = annotation
    elif get_origin(annotation) is list:
        expected_type = list
    else:
        expected_type = next(
            (
                candidate
                for candidate in get_args(annotation)
                if candidate in (bool, int, float, str, list)
            ),
            str,
        )

    try:
        if expected_type is bool:
            normalized = value.lower()
            if normalized in ("true", "1", "yes", "on"):
                return True
            if normalized in ("false", "0", "no", "off"):
                return False
            raise ValueError("expected true/false, 1/0, yes/no, or on/off")
        if expected_type is int:
            return int(value)
        if expected_type is float:
            return float(value)
        if expected_type is list:
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("expected a JSON array")
            return parsed
        return value
    except (TypeError, ValueError) as error:
        type_name = expected_type.__name__ if expected_type is not None else "value"
        raise ValueError(
            f"Invalid value {value!r} for environment variable {env_var}; "
            f"expected {type_name}"
        ) from error


def _dict_to_config(data: dict[str, Any]) -> Config:
    """Convert a dictionary to a Config object."""

    def build_section(cls: type, section_name: str) -> Any:
        section_data = data.get(section_name, {})
        if not isinstance(section_data, dict):
            raise ValueError(f"Configuration section [{section_name}] must be a table")
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in section_data.items() if k in valid_fields}
        # Source-specific warnings were emitted before defaults/user data were
        # merged. Filtering here must stay silent to avoid duplicate warnings.
        return cls(**filtered)

    config = Config(
        common=build_section(CommonConfig, "common"),
        training=build_section(TrainingConfig, "training"),
        evaluation=build_section(EvaluationConfig, "evaluation"),
        actor=build_section(ActorConfig, "actor"),
        web=build_section(WebConfig, "web"),
        mcts=build_section(MctsConfig, "mcts"),
        storage=build_section(StorageConfig, "storage"),
        logging=build_section(LoggingConfig, "logging"),
        wandb=build_section(WandbConfig, "wandb"),
    )
    config.validate()
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
        if defaults_path is not None:
            logger.debug(f"Loading defaults from {defaults_path}")
            with open(defaults_path, "rb") as f:
                data = tomllib.load(f)
            _warn_unknown_config_keys(data, defaults_path)
        else:
            logger.warning("No config.defaults.toml found, using hardcoded defaults")
            data = {}

        # Step 2: Overlay user configuration from config.toml
        config_path = _find_config_file()
        if config_path is not None:
            logger.info(f"Loading user configuration from {config_path}")
            with open(config_path, "rb") as f:
                user_data = tomllib.load(f)
            _warn_unknown_config_keys(user_data, config_path)
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
