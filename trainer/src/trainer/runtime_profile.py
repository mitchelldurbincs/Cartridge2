"""Canonical runtime namespace shared by every Python component."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROFILE_NAMESPACE_DIR = "profiles"


def _validate_segment(field: str, value: str) -> None:
    if not value or any(
        not (character.isascii() and (character.islower() or character.isdigit()))
        and character not in {"_", "-"}
        for character in value
    ):
        raise ValueError(
            f"Invalid runtime profile {field} {value!r}; IDs must contain only "
            "lowercase ASCII letters, digits, '_' or '-'"
        )


@dataclass(frozen=True)
class RuntimeProfile:
    algorithm_id: str
    env_id: str
    env_contract_version: int

    def __post_init__(self) -> None:
        _validate_segment("algorithm_id", self.algorithm_id)
        _validate_segment("env_id", self.env_id)
        if (
            isinstance(self.env_contract_version, bool)
            or not isinstance(self.env_contract_version, int)
            or self.env_contract_version <= 0
        ):
            raise ValueError("Runtime profile env_contract_version must be positive")

    @property
    def storage_prefix(self) -> str:
        return (
            f"{PROFILE_NAMESPACE_DIR}/{self.algorithm_id}/{self.env_id}/"
            f"v{self.env_contract_version}"
        )

    def data_dir(self, data_root: str | Path) -> Path:
        return Path(data_root) / self.storage_prefix

    def models_dir(self, data_root: str | Path) -> Path:
        return self.data_dir(data_root) / "models"

    @property
    def model_prefix(self) -> str:
        return f"{self.storage_prefix}/models"


def resolve_runtime_profile(algorithm_id: str, env_id: str) -> RuntimeProfile:
    """Resolve a selected catalog pair to its immutable runtime namespace."""
    from .environment_catalog import get_algorithm_descriptor, get_environment

    algorithm = get_algorithm_descriptor(algorithm_id)
    environment = get_environment(env_id)
    environment.compatibility(algorithm.id).require_compatible()
    return RuntimeProfile(
        algorithm_id=algorithm.id,
        env_id=environment.env_id,
        env_contract_version=environment.contract_version,
    )


__all__ = ["PROFILE_NAMESPACE_DIR", "RuntimeProfile", "resolve_runtime_profile"]
