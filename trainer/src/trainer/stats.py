"""Exact canonical learner statistics for RunCommit publication.

This module prepares and verifies immutable content-addressed stats snapshots.
It deliberately owns no mutable authority: ``RunHeadV2`` selects a
``RunCommitV1``, and that commit embeds the exact snapshot and its content ID.
``stats.json`` is only a rebuildable web projection of the selected run head.
"""

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .storage.publisher import (
    ArtifactValidationError,
    CheckpointProfileV1,
    CheckpointRef,
    canonical_json_bytes,
    sha256_bytes,
    validate_sha256_digest,
)

logger = logging.getLogger(__name__)

# Default history bounds
DEFAULT_MAX_HISTORY = 10000  # Max training history entries
DEFAULT_MAX_EVAL_HISTORY = 50

# Tiered history retention thresholds
# Recent data is kept at full resolution, older data is downsampled
RECENT_STEPS_THRESHOLD = 1000  # Keep all entries within last 1000 steps
MEDIUM_STEPS_THRESHOLD = 10000  # Downsample to every 100 steps for 1000-10000 range
RECENT_RESOLUTION = 1  # Keep every entry in recent range
MEDIUM_RESOLUTION = 100  # Keep every 100th step in medium range
OLD_RESOLUTION = 500  # Keep every 500th step for older data

STATS_ARTIFACT_SCHEMA_VERSION = 3
_MAX_U64 = (1 << 64) - 1
_STATS_FIELDS = frozenset(
    {
        "step",
        "total_steps",
        "metrics",
        "learning_rate",
        "samples_seen",
        "replay_record_count",
        "last_checkpoint",
        "timestamp",
        "history",
        "env_id",
        "last_evaluation",
        "evaluation_history",
    }
)
_BINDING_FIELDS = frozenset({"profile", "config_sha256", "checkpoint_id", "step"})
_SNAPSHOT_FIELDS = frozenset(
    {
        "schema_version",
        "profile",
        "config_sha256",
        "checkpoint_id",
        "step",
        "stats",
    }
)
_EVAL_FIELDS = frozenset(
    {
        "step",
        "metrics",
        "episodes",
        "mean_episode_length",
        "timestamp",
    }
)
_HISTORY_FIELDS = frozenset(
    {
        "step",
        "metrics",
        "learning_rate",
        "grad_norm",
    }
)


class StatsArtifactError(ValueError):
    """A persisted training-stats artifact violates its exact contract."""


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise StatsArtifactError(f"Stats artifact contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise StatsArtifactError(f"Stats artifact contains non-finite number {value}")


def _decode_canonical_json(data: bytes, *, context: str) -> object:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StatsArtifactError(f"{context} is not valid UTF-8 JSON") from exc
    if canonical_json_bytes(value) != data:
        raise StatsArtifactError(f"{context} is not canonical JSON")
    return value


def _exact_object(value: object, fields: frozenset[str], *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise StatsArtifactError(f"{context} must be a JSON object")
    actual = frozenset(value)
    if actual != fields:
        raise StatsArtifactError(
            f"{context} fields must be exact "
            f"(missing={sorted(fields - actual)}, extra={sorted(actual - fields)})"
        )
    return value


def _require_nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > _MAX_U64:
        raise StatsArtifactError(f"{field_name} must be a nonnegative integer within u64")
    return value


def _require_finite_number(value: object, *, field_name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StatsArtifactError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite nonnegative" if nonnegative else "finite"
        raise StatsArtifactError(f"{field_name} must be a {qualifier} number")
    # JSON distinguishes -0.0 textually even though it has no semantic meaning
    # for these metrics. Collapse both signed zeros before content hashing.
    return 0.0 if result == 0.0 else result


def _require_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise StatsArtifactError(f"{field_name} must be a string")
    return value


def _require_nonempty_string(value: object, *, field_name: str) -> str:
    result = _require_string(value, field_name=field_name)
    if not result.strip():
        raise StatsArtifactError(f"{field_name} must be a non-empty string")
    return result


def _require_digest(value: object, *, field_name: str) -> str:
    try:
        return validate_sha256_digest(value, field=field_name)
    except ValueError as exc:
        raise StatsArtifactError(str(exc)) from exc


def _normalize_metrics(value: object, *, context: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise StatsArtifactError(f"{context} must be a JSON object")
    result: dict[str, float] = {}
    for name, metric in value.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise StatsArtifactError(f"{context} names must be nonempty trimmed strings")
        result[name] = _require_finite_number(metric, field_name=f"{context}.{name}")
    return result


def _normalize_history_entry(value: object, *, context: str) -> dict[str, object]:
    fields = _exact_object(value, _HISTORY_FIELDS, context=context)
    grad_norm = fields["grad_norm"]
    if grad_norm is not None:
        grad_norm = _require_finite_number(
            grad_norm, field_name=f"{context}.grad_norm", nonnegative=True
        )
    return {
        "step": _require_nonnegative_integer(fields["step"], field_name=f"{context}.step"),
        "metrics": _normalize_metrics(fields["metrics"], context=f"{context}.metrics"),
        "learning_rate": _require_finite_number(
            fields["learning_rate"],
            field_name=f"{context}.learning_rate",
            nonnegative=True,
        ),
        "grad_norm": grad_norm,
    }


@dataclass(frozen=True)
class StatsBindingV1:
    """Identity shared by one checkpoint and its exact stats snapshot."""

    profile: CheckpointProfileV1
    config_sha256: str
    checkpoint_id: str
    step: int

    def __post_init__(self) -> None:
        if not isinstance(self.profile, CheckpointProfileV1):
            raise StatsArtifactError("stats binding profile is invalid")
        _require_digest(self.config_sha256, field_name="stats binding config_sha256")
        _require_digest(self.checkpoint_id, field_name="stats binding checkpoint_id")
        _require_nonnegative_integer(self.step, field_name="stats binding step")

    @classmethod
    def from_checkpoint(cls, checkpoint: CheckpointRef) -> "StatsBindingV1":
        return cls(
            profile=checkpoint.manifest.profile,
            config_sha256=checkpoint.manifest.config_sha256,
            checkpoint_id=checkpoint.checkpoint_id,
            step=checkpoint.manifest.step,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile.to_dict(),
            "config_sha256": self.config_sha256,
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
        }

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> "StatsBindingV1":
        try:
            profile = CheckpointProfileV1.from_dict(fields["profile"])
        except (ArtifactValidationError, ValueError) as exc:
            raise StatsArtifactError(f"Invalid stats binding profile: {exc}") from exc
        return cls(
            profile=profile,
            config_sha256=_require_digest(
                fields["config_sha256"], field_name="stats binding config_sha256"
            ),
            checkpoint_id=_require_digest(
                fields["checkpoint_id"], field_name="stats binding checkpoint_id"
            ),
            step=_require_nonnegative_integer(fields["step"], field_name="stats binding step"),
        )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=".pointer-", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def retain_training_history(history: list[dict], current_step: int) -> list[dict]:
    """Downsample history using tiered retention strategy.

    Args:
        history: List of history entries, each with a "step" key.
        current_step: The current training step (used to determine age).

    Returns:
        Downsampled history list preserving recent data at full resolution
        and older data at reduced resolution.
    """
    if not history:
        return history

    result = []
    for entry in history:
        step = entry.get("step", 0)
        age = current_step - step

        if age <= RECENT_STEPS_THRESHOLD:
            # Recent: keep all entries
            result.append(entry)
        elif age <= MEDIUM_STEPS_THRESHOLD:
            # Medium age: keep entries at MEDIUM_RESOLUTION intervals
            if step % MEDIUM_RESOLUTION == 0:
                result.append(entry)
        else:
            # Old: keep entries at OLD_RESOLUTION intervals
            if step % OLD_RESOLUTION == 0:
                result.append(entry)

    return result[-DEFAULT_MAX_HISTORY:]


@dataclass
class EvaluationStats:
    """Algorithm-neutral metrics from one checkpoint evaluation."""

    step: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    episodes: int = 0
    mean_episode_length: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.step = _require_nonnegative_integer(self.step, field_name="evaluation.step")
        self.metrics = _normalize_metrics(self.metrics, context="evaluation.metrics")
        self.episodes = _require_nonnegative_integer(
            self.episodes, field_name="evaluation.episodes"
        )
        self.mean_episode_length = _require_finite_number(
            self.mean_episode_length,
            field_name="evaluation.mean_episode_length",
            nonnegative=True,
        )
        self.timestamp = _require_finite_number(
            self.timestamp, field_name="evaluation.timestamp", nonnegative=True
        )
        if self.episodes == 0:
            if self.metrics or self.mean_episode_length != 0.0:
                raise StatsArtifactError(
                    "zero-episode evaluation stats must have no metrics and zero mean length"
                )
        elif self.mean_episode_length == 0.0:
            raise StatsArtifactError(
                "non-empty evaluation stats must have a positive mean episode length"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "metrics": self.metrics,
            "episodes": self.episodes,
            "mean_episode_length": self.mean_episode_length,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationStats":
        fields = _exact_object(data, _EVAL_FIELDS, context="evaluation stats")
        return cls(
            step=fields["step"],
            metrics=fields["metrics"],
            episodes=fields["episodes"],
            mean_episode_length=fields["mean_episode_length"],
            timestamp=fields["timestamp"],
        )


@dataclass
class TrainerStats:
    """Algorithm-neutral training statistics for immutable publication."""

    step: int = 0
    total_steps: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    samples_seen: int = 0
    replay_record_count: int = 0
    last_checkpoint: str = ""
    timestamp: float = field(default_factory=time.time)
    history: list[dict] = field(default_factory=list)

    env_id: str = ""
    last_evaluation: EvaluationStats | None = None
    evaluation_history: list[dict] = field(default_factory=list)
    _max_evaluation_history: int = DEFAULT_MAX_EVAL_HISTORY
    _binding: StatsBindingV1 | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for field_name in (
            "step",
            "total_steps",
            "samples_seen",
            "replay_record_count",
        ):
            normalized = _require_nonnegative_integer(
                getattr(self, field_name), field_name=f"stats.{field_name}"
            )
            setattr(self, field_name, normalized)
        self.metrics = _normalize_metrics(self.metrics, context="stats.metrics")
        self.learning_rate = _require_finite_number(
            self.learning_rate,
            field_name="stats.learning_rate",
            nonnegative=True,
        )
        self.timestamp = _require_finite_number(
            self.timestamp, field_name="stats.timestamp", nonnegative=True
        )
        self.last_checkpoint = _require_string(
            self.last_checkpoint, field_name="stats.last_checkpoint"
        )
        self.env_id = _require_string(self.env_id, field_name="stats.env_id")
        if not isinstance(self.history, list):
            raise StatsArtifactError("stats.history must be an array")
        if not isinstance(self.evaluation_history, list):
            raise StatsArtifactError("stats.evaluation_history must be an array")
        if self.last_evaluation is not None and not isinstance(
            self.last_evaluation, EvaluationStats
        ):
            raise StatsArtifactError("stats.last_evaluation must be evaluation stats or null")
        if self._binding is not None and not isinstance(self._binding, StatsBindingV1):
            raise StatsArtifactError("stats binding is invalid")
        self.history = [
            _normalize_history_entry(entry, context=f"stats.history[{index}]")
            for index, entry in enumerate(self.history)
        ]
        self.evaluation_history = [
            EvaluationStats.from_dict(entry).to_dict() for entry in self.evaluation_history
        ]
        self._validate_semantics()

    def _validate_semantics(self) -> None:
        if self.total_steps < self.step:
            raise StatsArtifactError("stats.total_steps cannot be less than stats.step")
        history_steps = [entry["step"] for entry in self.history]
        if any(current <= previous for previous, current in zip(history_steps, history_steps[1:])):
            raise StatsArtifactError("stats.history steps must be strictly increasing")
        if history_steps and history_steps[-1] > self.step:
            raise StatsArtifactError("stats.history cannot extend beyond stats.step")

        evaluations = [EvaluationStats.from_dict(entry) for entry in self.evaluation_history]
        eval_steps = [entry.step for entry in evaluations]
        if any(current <= previous for previous, current in zip(eval_steps, eval_steps[1:])):
            raise StatsArtifactError("stats.evaluation_history steps must be strictly increasing")
        eval_timestamps = [entry.timestamp for entry in evaluations]
        if any(
            current < previous for previous, current in zip(eval_timestamps, eval_timestamps[1:])
        ):
            raise StatsArtifactError("stats.evaluation_history timestamps must be nondecreasing")
        if eval_steps and eval_steps[-1] > self.step:
            raise StatsArtifactError("stats.evaluation_history cannot extend beyond stats.step")
        if not evaluations:
            if self.last_evaluation is not None:
                raise StatsArtifactError(
                    "stats.last_evaluation must be null when evaluation_history is empty"
                )
        elif (
            self.last_evaluation is None
            or self.last_evaluation.to_dict() != evaluations[-1].to_dict()
        ):
            raise StatsArtifactError(
                "stats.last_evaluation must equal the final evaluation_history record"
            )

        if self._binding is not None:
            if self.step != self._binding.step:
                raise StatsArtifactError("stats.step does not match its bound checkpoint step")
            if self.env_id != self._binding.profile.env_id:
                raise StatsArtifactError("stats.env_id does not match its bound checkpoint profile")
            if self.last_checkpoint != self._binding.checkpoint_id:
                raise StatsArtifactError(
                    "stats.last_checkpoint does not match its bound checkpoint"
                )

    def append_history(self, entry: dict) -> None:
        """Append to history with tiered retention and max size bound.

        Uses a tiered downsampling strategy to keep recent data at full
        resolution while preserving coarse historical data:
        - Last 1000 steps: full resolution (every logged step)
        - 1000-10000 steps ago: every 100th step
        - 10000+ steps ago: every 500th step

        The schema-owned absolute bound is applied after downsampling.
        """
        normalized = _normalize_history_entry(entry, context="stats.history append")
        if self.history and normalized["step"] <= self.history[-1]["step"]:
            raise StatsArtifactError("stats.history steps must be strictly increasing")
        if normalized["step"] > self.step:
            raise StatsArtifactError("stats.history cannot extend beyond stats.step")
        self.history.append(normalized)
        current_step = normalized["step"]
        self.history = retain_training_history(self.history, current_step)

    def append_evaluation(self, evaluation: EvaluationStats) -> None:
        """Append evaluation result to history."""
        if not isinstance(evaluation, EvaluationStats):
            raise StatsArtifactError("evaluation append requires EvaluationStats")
        if evaluation.step > self.step:
            raise StatsArtifactError("stats.evaluation_history cannot extend beyond stats.step")
        if self.evaluation_history and evaluation.step <= self.evaluation_history[-1]["step"]:
            raise StatsArtifactError("stats.evaluation_history steps must be strictly increasing")
        self.last_evaluation = evaluation
        self.evaluation_history.append(evaluation.to_dict())
        if len(self.evaluation_history) > self._max_evaluation_history:
            self.evaluation_history = self.evaluation_history[-self._max_evaluation_history :]

    def _payload_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "metrics": self.metrics,
            "learning_rate": self.learning_rate,
            "samples_seen": self.samples_seen,
            "replay_record_count": self.replay_record_count,
            "last_checkpoint": self.last_checkpoint,
            "timestamp": self.timestamp,
            "history": self.history,  # Already bounded on append
            "env_id": self.env_id,
            "last_evaluation": (
                self.last_evaluation.to_dict() if self.last_evaluation else None
            ),
            "evaluation_history": self.evaluation_history,
        }

    def to_dict(self) -> dict[str, object]:
        self._validate_semantics()
        return self._payload_dict()

    @classmethod
    def from_dict(cls, data: dict) -> "TrainerStats":
        """Create TrainerStats from its exact serialized contract."""
        fields = _exact_object(data, _STATS_FIELDS, context="training stats")
        history = fields["history"]
        evaluation_history = fields["evaluation_history"]
        if not isinstance(history, list) or not all(isinstance(entry, dict) for entry in history):
            raise StatsArtifactError("stats.history must be an array of objects")
        if not isinstance(evaluation_history, list):
            raise StatsArtifactError("stats.evaluation_history must be an array")
        parsed_evaluations = [
            EvaluationStats.from_dict(entry) for entry in evaluation_history
        ]
        last_evaluation = fields["last_evaluation"]
        if last_evaluation is not None:
            last_evaluation = EvaluationStats.from_dict(last_evaluation)
        stats = cls(
            step=fields["step"],
            total_steps=fields["total_steps"],
            metrics=fields["metrics"],
            learning_rate=fields["learning_rate"],
            samples_seen=fields["samples_seen"],
            replay_record_count=fields["replay_record_count"],
            last_checkpoint=fields["last_checkpoint"],
            timestamp=fields["timestamp"],
            history=history,
            env_id=fields["env_id"],
            last_evaluation=last_evaluation,
            evaluation_history=[entry.to_dict() for entry in parsed_evaluations],
        )
        return stats


@dataclass(frozen=True)
class PreparedStatsSnapshotV3:
    """Canonical stats bytes ready to embed in one immutable RunCommitV1."""

    stats_id: str
    binding: StatsBindingV1
    data: bytes

    def __post_init__(self) -> None:
        _require_digest(self.stats_id, field_name="prepared stats_id")
        if not isinstance(self.binding, StatsBindingV1):
            raise StatsArtifactError("prepared stats binding is invalid")
        if not isinstance(self.data, bytes):
            raise StatsArtifactError("prepared stats data must be bytes")
        actual_id = sha256_bytes(self.data)
        if actual_id != self.stats_id:
            raise StatsArtifactError(
                f"Prepared stats SHA-256 mismatch: got {actual_id}, expected {self.stats_id}"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a fresh, fully verified object for RunCommit embedding."""
        decode_stats_snapshot(
            self.data,
            expected_stats_id=self.stats_id,
            expected_binding=self.binding,
        )
        value = _decode_canonical_json(self.data, context="prepared stats snapshot")
        fields = _exact_object(value, _SNAPSHOT_FIELDS, context="stats snapshot")
        return dict(fields)


@dataclass(frozen=True)
class LoadedStatsSnapshotV3:
    """A verified canonical stats snapshot reconstructed from a RunCommitV1."""

    stats_id: str
    binding: StatsBindingV1
    stats: TrainerStats
    data: bytes


def _validate_rebinding(stats: TrainerStats, binding: StatsBindingV1) -> None:
    if stats._binding is not None and (
        stats._binding.profile != binding.profile
        or stats._binding.config_sha256 != binding.config_sha256
    ):
        raise StatsArtifactError(
            "Stats cannot be rebound to a different checkpoint profile or config"
        )
    if stats._binding is not None and binding.step < stats._binding.step:
        raise StatsArtifactError("Stats cannot be rebound to an older checkpoint step")
    if (
        stats._binding is not None
        and binding.step == stats._binding.step
        and binding.checkpoint_id != stats._binding.checkpoint_id
    ):
        raise StatsArtifactError(
            "Stats cannot be rebound to a different checkpoint at the same step"
        )
    if stats.step != binding.step:
        raise StatsArtifactError("stats.step does not match its checkpoint step")
    if stats.env_id != binding.profile.env_id:
        raise StatsArtifactError("stats.env_id does not match its checkpoint profile")
    if stats.last_checkpoint != binding.checkpoint_id:
        raise StatsArtifactError("stats.last_checkpoint does not match its bound checkpoint")


def prepare_stats_snapshot(
    stats: TrainerStats,
    checkpoint: CheckpointRef,
) -> PreparedStatsSnapshotV3:
    """Normalize and bind mutable learner stats to one exact checkpoint.

    The returned bytes have no visibility by themselves. A ``RunCommitV1``
    embeds the snapshot object and ``stats_id`` before the sole ``RunHeadV2``
    authority can advance.
    """
    if not isinstance(stats, TrainerStats):
        raise TypeError("stats must be TrainerStats")
    if not isinstance(checkpoint, CheckpointRef):
        raise TypeError("checkpoint must be CheckpointRef")
    binding = StatsBindingV1.from_checkpoint(checkpoint)
    _validate_rebinding(stats, binding)
    previous_binding = stats._binding
    stats._binding = binding
    try:
        validated = TrainerStats.from_dict(stats._payload_dict())
        validated._binding = binding
        validated._validate_semantics()
    except Exception:
        stats._binding = previous_binding
        raise
    payload = {
        "schema_version": STATS_ARTIFACT_SCHEMA_VERSION,
        **binding.to_dict(),
        "stats": validated.to_dict(),
    }
    snapshot_bytes = canonical_json_bytes(payload)
    stats_id = sha256_bytes(snapshot_bytes)
    return PreparedStatsSnapshotV3(
        stats_id=stats_id,
        binding=binding,
        data=snapshot_bytes,
    )


def decode_stats_snapshot(
    data: bytes,
    *,
    expected_stats_id: str | None = None,
    expected_binding: StatsBindingV1 | None = None,
) -> LoadedStatsSnapshotV3:
    """Strictly reconstruct the stats snapshot embedded by a RunCommitV1."""
    if not isinstance(data, bytes):
        raise StatsArtifactError("stats snapshot data must be bytes")
    stats_id = sha256_bytes(data)
    if expected_stats_id is not None:
        expected_stats_id = _require_digest(expected_stats_id, field_name="expected stats_id")
        if stats_id != expected_stats_id:
            raise StatsArtifactError(
                f"Stats snapshot SHA-256 mismatch: got {stats_id}, expected {expected_stats_id}"
            )
    if expected_binding is not None and not isinstance(expected_binding, StatsBindingV1):
        raise TypeError("expected_binding must be StatsBindingV1 or None")

    raw = _decode_canonical_json(data, context="stats snapshot")
    fields = _exact_object(raw, _SNAPSHOT_FIELDS, context="stats snapshot")
    schema_version = fields["schema_version"]
    if isinstance(schema_version, bool) or schema_version != STATS_ARTIFACT_SCHEMA_VERSION:
        raise StatsArtifactError(
            f"stats snapshot schema_version must be exactly {STATS_ARTIFACT_SCHEMA_VERSION}"
        )
    binding = StatsBindingV1.from_fields(fields)
    if expected_binding is not None and binding != expected_binding:
        raise StatsArtifactError("Stats snapshot binding does not match its RunCommit binding")
    stats_fields = _exact_object(fields["stats"], _STATS_FIELDS, context="stats snapshot stats")
    stats = TrainerStats.from_dict(dict(stats_fields))
    stats._binding = binding
    stats._validate_semantics()
    normalized_bytes = canonical_json_bytes(
        {
            "schema_version": STATS_ARTIFACT_SCHEMA_VERSION,
            **binding.to_dict(),
            "stats": stats.to_dict(),
        }
    )
    if normalized_bytes != data:
        raise StatsArtifactError(
            "Stats snapshot numbers are not in their canonical normalized form"
        )
    logger.info(
        "Decoded immutable stats %s: %s evaluation records, %s training records",
        stats_id,
        len(stats.evaluation_history),
        len(stats.history),
    )
    return LoadedStatsSnapshotV3(stats_id, binding, stats, data)


def write_stats_projection(
    snapshot: PreparedStatsSnapshotV3 | LoadedStatsSnapshotV3,
    path: str | Path,
) -> None:
    """Atomically rebuild non-authoritative ``stats.json`` from a run snapshot."""
    if isinstance(snapshot, PreparedStatsSnapshotV3):
        loaded = decode_stats_snapshot(
            snapshot.data,
            expected_stats_id=snapshot.stats_id,
            expected_binding=snapshot.binding,
        )
    elif isinstance(snapshot, LoadedStatsSnapshotV3):
        loaded = decode_stats_snapshot(
            snapshot.data,
            expected_stats_id=snapshot.stats_id,
            expected_binding=snapshot.binding,
        )
    else:
        raise TypeError("snapshot must be PreparedStatsSnapshotV3 or LoadedStatsSnapshotV3")
    _atomic_replace(Path(path), canonical_json_bytes(loaded.stats.to_dict()))


def write_ephemeral_stats_projection(stats: TrainerStats, path: str | Path) -> None:
    """Write live in-process metrics without creating any resume authority.

    This projection may be newer than ``RunHeadV2`` between checkpoint commits.
    A restart always rebuilds it from the selected RunCommit snapshot.
    """
    if not isinstance(stats, TrainerStats):
        raise TypeError("stats must be TrainerStats")
    validated = TrainerStats.from_dict(stats._payload_dict())
    _atomic_replace(Path(path), canonical_json_bytes(validated.to_dict()))
