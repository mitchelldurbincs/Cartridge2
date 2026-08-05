"""Load typed configuration from canonical defaults, user TOML, and the environment."""

import copy
import logging
import math
import os
import sys
import threading
from dataclasses import fields
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from .config_sections import (
    _MAX_U64,
    CANONICAL_DEFAULTS_DATA,
    CANONICAL_DEFAULTS_PATHS,
    PACKAGED_DEFAULTS,
    PROJECT_ROOT,
    ActorConfig,
    AlgorithmConfig,
    CommonConfig,
    Config,
    EvaluationConfig,
    LoggingConfig,
    MctsConfig,
    StorageConfig,
    TrainingConfig,
    WandbConfig,
    WebConfig,
    validate_simulation_schedule,
)

logger = logging.getLogger(__name__)
_config_lock = threading.Lock()

_PROJECT_ROOT = PROJECT_ROOT
_PACKAGED_DEFAULTS = PACKAGED_DEFAULTS
CONFIG_SEARCH_PATHS = [
    Path("config.toml"),
    Path("/app/config.toml"),
    _PROJECT_ROOT / "config.toml",
]
DEFAULTS_SEARCH_PATHS = CANONICAL_DEFAULTS_PATHS

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
            "Canonical config.defaults.toml schema mismatch (" + "; ".join(details) + ")"
        )

    for section, expected_keys in _CONFIG_SECTIONS.items():
        values = data[section]
        if not isinstance(values, dict):
            raise ValueError(f"Canonical config.defaults.toml [{section}] must be a TOML table")
        actual_keys = set(values)
        optional_none_keys = {
            item.name for item in fields(_CONFIG_SECTION_TYPES[section]) if item.default is None
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
                f"Canonical config.defaults.toml copies diverge: {existing[0]} != {mirror}"
            )
    return existing[0]


def _load_defaults() -> dict[str, Any]:
    defaults_path = _find_defaults_file()
    if defaults_path is None:
        raise FileNotFoundError(
            "canonical config.defaults.toml was not found in any configured location"
        )
    logger.debug("Loading defaults from %s", defaults_path)
    if DEFAULTS_SEARCH_PATHS is CANONICAL_DEFAULTS_PATHS:
        data = copy.deepcopy(CANONICAL_DEFAULTS_DATA)
    else:
        with open(defaults_path, "rb") as defaults_file:
            data = tomllib.load(defaults_file)
    _validate_canonical_defaults(data)
    return data


def _find_config_file() -> Path | None:
    """Find the user config in the explicit or standard locations."""
    if "CARTRIDGE_CONFIG" in os.environ:
        env_path = os.environ["CARTRIDGE_CONFIG"]
        path = Path(env_path)
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"CARTRIDGE_CONFIG points to missing file: {env_path!r}")

    project_config = _PROJECT_ROOT / "config.toml"
    cwd_is_project = Path.cwd().resolve().is_relative_to(_PROJECT_ROOT.resolve())
    for path in CONFIG_SEARCH_PATHS:
        if path == project_config and not cwd_is_project:
            continue
        if path.exists():
            return path.resolve()
    return None


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    prefix = "CARTRIDGE_"
    for env_var, value in os.environ.items():
        if not env_var.startswith(prefix):
            continue
        parts = env_var[len(prefix) :].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, key = parts
        if section not in _CONFIG_SECTIONS:
            continue
        if key not in _CONFIG_SECTIONS[section]:
            raise ValueError(f"Unknown configuration override: {env_var}")
        data.setdefault(section, {})[key] = _convert_value(value, section, key, data)
        logger.debug("Applied override %s=%s", env_var, value)
    return data


def _convert_value(value: str, section: str, key: str, data: dict) -> Any:
    existing = data.get(section, {}).get(key)
    if isinstance(existing, bool):
        normalized = value.lower()
        if normalized in ("true", "1", "yes"):
            return True
        if normalized in ("false", "0", "no"):
            return False
        raise ValueError(
            f"Invalid boolean value {value!r} for {section}.{key}; expected true/false"
        )
    if isinstance(existing, int):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    if isinstance(existing, list):
        if not value:
            return []
        items = [item.strip() for item in value.split(",")]
        if any(not item for item in items):
            raise ValueError(
                f"Invalid list value {value!r} for {section}.{key}; "
                "expected comma-separated non-empty strings"
            )
        return items
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _build_section(data: dict[str, Any], section_name: str) -> Any:
    cls = _CONFIG_SECTION_TYPES[section_name]
    section_data = data.get(section_name, {})
    unknown = sorted(set(section_data) - _CONFIG_SECTIONS[section_name])
    if unknown:
        raise ValueError(f"Unknown configuration keys in [{section_name}]: " + ", ".join(unknown))
    return cls(**section_data)


def _dict_to_config(data: dict[str, Any]) -> Config:
    unknown_sections = sorted(set(data) - set(_CONFIG_SECTIONS))
    if unknown_sections:
        raise ValueError("Unknown configuration sections: " + ", ".join(unknown_sections))
    config = Config(**{section: _build_section(data, section) for section in _CONFIG_SECTION_TYPES})
    _validate_cross_section_config(config)
    return config


def _validate_cross_section_config(config: Config) -> None:
    if config.training.iterations > _MAX_U64 // config.training.steps_per_iteration:
        raise ValueError("training.iterations * training.steps_per_iteration exceeds u64")
    validate_simulation_schedule(
        iterations=config.training.iterations,
        start=config.mcts.start_sims,
        maximum=config.mcts.max_sims,
        ramp=config.mcts.sim_ramp_rate,
    )
    _validate_identity_and_environment(config)
    _validate_storage(config)
    _validate_evaluation_and_wandb(config)


def _validate_identity_and_environment(config: Config) -> None:
    for value, path in (
        (config.algorithm.id, "algorithm.id"),
        (config.common.env_id, "common.env_id"),
        (config.common.data_dir, "common.data_dir"),
        (config.common.log_level, "common.log_level"),
        (config.web.host, "web.host"),
        (config.storage.postgres_url, "storage.postgres_url"),
        (config.storage.model_backend, "storage.model_backend"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} must be a non-empty string")
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
    for values, path in (
        (config.web.allowed_origins, "web.allowed_origins"),
        (config.wandb.tags, "wandb.tags"),
    ):
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError(f"{path} must be an array of non-empty strings")


def _validate_storage(config: Config) -> None:
    if config.storage.model_backend not in {"filesystem", "s3"}:
        raise ValueError(
            "storage.model_backend must be 'filesystem' or 's3', got "
            f"{config.storage.model_backend!r}"
        )
    if config.storage.model_backend == "s3" and (
        not isinstance(config.storage.s3_bucket, str) or not config.storage.s3_bucket.strip()
    ):
        raise ValueError("storage.s3_bucket is required when storage.model_backend is 's3'")
    if config.storage.s3_endpoint is not None and (
        not isinstance(config.storage.s3_endpoint, str) or not config.storage.s3_endpoint.strip()
    ):
        raise ValueError("storage.s3_endpoint must be a non-empty string when set")
    for value, path in (
        (config.storage.pool_max_size, "storage.pool_max_size"),
        (config.storage.pool_connect_timeout, "storage.pool_connect_timeout"),
        (config.storage.pool_idle_timeout, "storage.pool_idle_timeout"),
        (config.storage.replay_retained_scopes, "storage.replay_retained_scopes"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{path} must be a positive integer")
    if config.logging.format not in {"text", "json"}:
        raise ValueError("logging.format must be 'text' or 'json'")


def _validate_evaluation_and_wandb(config: Config) -> None:
    if config.evaluation.solver_games > 0 and config.common.env_id != "connect4":
        raise ValueError(
            "evaluation.solver_games may be nonzero only when common.env_id is 'connect4'"
        )
    if config.evaluation.promotion_metric == "solver_optimal" and (
        config.common.env_id != "connect4" or config.evaluation.solver_games == 0
    ):
        raise ValueError(
            "evaluation.promotion_metric='solver_optimal' requires connect4 with "
            "evaluation.solver_games > 0"
        )
    timeout = config.wandb.init_timeout_seconds
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("wandb.init_timeout_seconds must be greater than zero")


_cached_config: Config | None = None


def get_config(reload: bool = False) -> Config:
    """Load, validate, and cache the effective configuration."""
    global _cached_config
    if _cached_config is not None and not reload:
        return _cached_config
    with _config_lock:
        if _cached_config is not None and not reload:
            return _cached_config
        data = _load_defaults()
        config_path = _find_config_file()
        if config_path is not None:
            logger.info("Loading user configuration from %s", config_path)
            with open(config_path, "rb") as config_file:
                data = _deep_merge(data, tomllib.load(config_file))
        _cached_config = _dict_to_config(_apply_env_overrides(data))
        return _cached_config


def reset_config() -> None:
    """Reset the cached configuration, primarily for tests."""
    global _cached_config
    with _config_lock:
        _cached_config = None
