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
import os
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


@dataclass
class ActorConfig:
    """Actor (self-play) settings."""

    actor_id: str = "actor-1"
    max_episodes: int = -1
    episode_timeout_secs: int = 30
    flush_interval_secs: int = 5
    log_interval: int = 50


@dataclass
class WebConfig:
    """Web server settings."""

    host: str = "0.0.0.0"
    port: int = 8080


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
        if path.exists():
            return path
        logger.warning(f"CARTRIDGE_CONFIG={env_path} not found, searching defaults")

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
            data[section][key] = _convert_value(value, section, key, data)
            logger.debug(f"Applied legacy override {env_var}={value}")

    # Apply CARTRIDGE_* overrides (higher priority)
    prefix = "CARTRIDGE_"
    for env_var, value in os.environ.items():
        if not env_var.startswith(prefix):
            continue

        # Skip empty values
        if value == "":
            continue

        # Parse CARTRIDGE_SECTION_KEY format
        parts = env_var[len(prefix) :].lower().split("_", 1)
        if len(parts) != 2:
            continue

        section, key = parts
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
            return value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            return int(value)
        elif isinstance(existing, float):
            return float(value)

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

    def build_section(cls: type, section_name: str) -> Any:
        section_data = data.get(section_name, {})
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in section_data.items() if k in valid_fields}
        ignored = sorted(k for k in section_data if k not in valid_fields)
        if ignored:
            logger.debug(
                "Ignoring unsupported config keys for %s: %s",
                section_name,
                ", ".join(ignored),
            )
        return cls(**filtered)

    return Config(
        common=build_section(CommonConfig, "common"),
        training=build_section(TrainingConfig, "training"),
        evaluation=build_section(EvaluationConfig, "evaluation"),
        actor=build_section(ActorConfig, "actor"),
        web=build_section(WebConfig, "web"),
        mcts=build_section(MctsConfig, "mcts"),
        storage=build_section(StorageConfig, "storage"),
    )


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
        else:
            logger.warning("No config.defaults.toml found, using hardcoded defaults")
            data = {}

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
