"""Algorithm-neutral replay contracts fenced to one exact selection."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_U32 = (1 << 32) - 1


class EmptyReplaySelectionError(RuntimeError):
    """Raised when sampling is requested from an empty exact selection."""


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")
    return value


def _integer(
    value: object,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise ValueError(
            f"{field} must be an integer in the inclusive range [{minimum}, {maximum}]"
        )
    return value


@dataclass(frozen=True)
class ReplayProfile:
    """Exact namespace shared by one collector and learner cartridge."""

    env_id: str
    env_contract_version: int
    algorithm_id: str
    experience_schema: str

    def __post_init__(self) -> None:
        for field_name in ("env_id", "algorithm_id", "experience_schema"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ReplayProfile.{field_name} cannot be empty")
        _integer(
            self.env_contract_version,
            field="ReplayProfile.env_contract_version",
            minimum=1,
            maximum=_MAX_U32,
        )

    def matches(self, record: "ReplayRecord") -> bool:
        """Return whether ``record`` belongs to this exact namespace."""
        return (
            record.env_id == self.env_id
            and record.env_contract_version == self.env_contract_version
            and record.algorithm_id == self.algorithm_id
            and record.experience_schema == self.experience_schema
        )


@dataclass(frozen=True)
class ReplaySelection:
    """One exact, attempt-scoped replay collection visible to a learner."""

    profile: ReplayProfile
    collection_scope_id: str
    source_checkpoint_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ReplayProfile):
            raise TypeError("ReplaySelection.profile must be ReplayProfile")
        _digest(
            self.collection_scope_id,
            field="ReplaySelection.collection_scope_id",
        )
        if self.source_checkpoint_id is not None:
            _digest(
                self.source_checkpoint_id,
                field="ReplaySelection.source_checkpoint_id",
            )

    def matches(self, record: "ReplayRecord") -> bool:
        return (
            self.profile.matches(record)
            and record.collection_scope_id == self.collection_scope_id
            and record.source_checkpoint_id == self.source_checkpoint_id
        )

    def record(
        self,
        *,
        id: str,
        episode_id: str,
        step_number: int,
        payload: bytes,
    ) -> "ReplayRecord":
        """Wrap algorithm-owned bytes in this exact collection selection."""
        return ReplayRecord(
            id=id,
            env_id=self.profile.env_id,
            env_contract_version=self.profile.env_contract_version,
            algorithm_id=self.profile.algorithm_id,
            experience_schema=self.profile.experience_schema,
            collection_scope_id=self.collection_scope_id,
            source_checkpoint_id=self.source_checkpoint_id,
            episode_id=episode_id,
            step_number=step_number,
            payload=payload,
        )


@dataclass(frozen=True)
class ReplayRecord:
    """Immutable replay envelope with an algorithm-owned opaque payload."""

    id: str
    env_id: str
    env_contract_version: int
    algorithm_id: str
    experience_schema: str
    collection_scope_id: str
    source_checkpoint_id: str | None
    episode_id: str
    step_number: int
    payload: bytes

    def __post_init__(self) -> None:
        for field_name in (
            "id",
            "env_id",
            "algorithm_id",
            "experience_schema",
            "episode_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ReplayRecord.{field_name} cannot be empty")
        _integer(
            self.env_contract_version,
            field="ReplayRecord.env_contract_version",
            minimum=1,
            maximum=_MAX_U32,
        )
        _digest(
            self.collection_scope_id,
            field="ReplayRecord.collection_scope_id",
        )
        if self.source_checkpoint_id is not None:
            _digest(
                self.source_checkpoint_id,
                field="ReplayRecord.source_checkpoint_id",
            )
        _integer(
            self.step_number,
            field="ReplayRecord.step_number",
            minimum=0,
            maximum=_MAX_U32,
        )
        if not isinstance(self.payload, bytes):
            raise TypeError("ReplayRecord.payload must be bytes")


class ReplayStore(ABC):
    """Algorithm-neutral persistence for one exact replay selection."""

    @property
    @abstractmethod
    def selection(self) -> ReplaySelection:
        """Exact profile, collection attempt, and source checkpoint fence."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the store and release backend resources."""
        raise NotImplementedError

    def __enter__(self) -> "ReplayStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @abstractmethod
    def count(self) -> int:
        """Count records in this store's exact replay selection."""
        raise NotImplementedError

    @abstractmethod
    def count_episodes(self) -> int:
        """Count distinct episode identities in this exact replay selection."""
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int) -> list[ReplayRecord]:
        """Sample exactly ``batch_size`` records, using replacement as needed.

        A positive request against a non-empty exact selection returns exactly
        the requested number of records even when the selection is smaller
        than the minibatch. Implementations raise
        :class:`EmptyReplaySelectionError` when that selection is empty and
        must never widen the selection fence to fill a batch.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> int:
        """Delete every record in this store's exact replay selection."""
        raise NotImplementedError

    @abstractmethod
    def cleanup(self, window_size: int) -> int:
        """Keep only the newest records in this exact replay selection."""
        raise NotImplementedError

    @abstractmethod
    def vacuum(self) -> None:
        """Ask the backend to reclaim storage after deletions."""
        raise NotImplementedError

    @abstractmethod
    def store(self, record: ReplayRecord) -> None:
        """Insert one immutable replay record."""
        raise NotImplementedError

    @abstractmethod
    def store_batch(self, records: list[ReplayRecord]) -> None:
        """Insert immutable replay records as one transaction."""
        raise NotImplementedError
