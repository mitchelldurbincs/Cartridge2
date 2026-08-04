"""Immutable RunCommit and orchestration records."""

from __future__ import annotations

from dataclasses import dataclass

from ..stats import (
    LoadedStatsSnapshotV3,
    PreparedStatsSnapshotV3,
    StatsBindingV1,
    decode_stats_snapshot,
)
from .artifact_codec import (
    ArtifactValidationError,
    canonical_json_bytes,
    sha256_bytes,
    validate_sha256_digest,
)
from .checkpoint_types import CheckpointProfileV1
from .evaluation_recipe import ChampionReferenceV1
from .run_commit_codec import (
    _ORCHESTRATION_FIELDS,
    _PROFILE_FIELDS,
    _RUN_COMMIT_FIELDS,
    _decode_canonical,
    _exact,
    _integer,
    _number,
    _optional_rate,
    _timestamp,
    _u32,
)
from .run_recipe import RunRecipeV1


@dataclass(frozen=True)
class OrchestrationCommitV1:
    iteration: int
    episodes_generated: int
    transitions_generated: int
    training_steps: int
    actor_time_seconds: float
    trainer_time_seconds: float
    eval_time_seconds: float
    total_time_seconds: float
    eval_win_rate: float | None
    eval_draw_rate: float | None
    timestamp: str
    evaluation_id: str | None
    collector_simulations: int
    collector_seed: int | None
    evaluation_seed: int | None
    collection_scope_id: str
    source_checkpoint_id: str | None

    def __post_init__(self) -> None:
        _integer(self.iteration, field="orchestration.iteration", positive=True)
        for field in (
            "episodes_generated",
            "transitions_generated",
            "training_steps",
        ):
            _integer(getattr(self, field), field=f"orchestration.{field}")
        _u32(
            self.collector_simulations,
            field="orchestration.collector_simulations",
        )
        if self.collector_seed is not None:
            _integer(self.collector_seed, field="orchestration.collector_seed")
        if self.evaluation_seed is not None:
            _integer(self.evaluation_seed, field="orchestration.evaluation_seed")
        validate_sha256_digest(
            self.collection_scope_id,
            field="orchestration.collection_scope_id",
        )
        if self.source_checkpoint_id is not None:
            validate_sha256_digest(
                self.source_checkpoint_id,
                field="orchestration.source_checkpoint_id",
            )
        phase_total = 0.0
        for field in (
            "actor_time_seconds",
            "trainer_time_seconds",
            "eval_time_seconds",
        ):
            normalized = _number(getattr(self, field), field=f"orchestration.{field}")
            object.__setattr__(self, field, normalized)
            phase_total += normalized
        total = _number(self.total_time_seconds, field="orchestration.total_time_seconds")
        object.__setattr__(self, "total_time_seconds", total)
        if total + 1e-12 < phase_total:
            raise ArtifactValidationError(
                "orchestration.total_time_seconds is shorter than its phases"
            )
        win_rate = _optional_rate(self.eval_win_rate, field="orchestration.eval_win_rate")
        draw_rate = _optional_rate(self.eval_draw_rate, field="orchestration.eval_draw_rate")
        object.__setattr__(self, "eval_win_rate", win_rate)
        object.__setattr__(self, "eval_draw_rate", draw_rate)
        if (win_rate is None) != (draw_rate is None):
            raise ArtifactValidationError("Orchestration evaluation rates are incomplete")
        if win_rate is not None and win_rate + draw_rate > 1.0:
            raise ArtifactValidationError("Orchestration evaluation rates exceed one")
        _timestamp(self.timestamp, field="orchestration.timestamp")
        if self.evaluation_id is None:
            if (
                self.eval_time_seconds != 0.0
                or win_rate is not None
                or self.evaluation_seed is not None
            ):
                raise ArtifactValidationError(
                    "A non-evaluation commit cannot contain evaluation metrics"
                )
        else:
            validate_sha256_digest(self.evaluation_id, field="orchestration.evaluation_id")
            if self.evaluation_seed is None:
                raise ArtifactValidationError(
                    "An evaluation commit requires evaluation_seed provenance"
                )

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in _ORCHESTRATION_FIELDS}

    @classmethod
    def from_dict(cls, value: object) -> "OrchestrationCommitV1":
        fields = _exact(value, _ORCHESTRATION_FIELDS, context="orchestration commit")
        return cls(**dict(fields))


StatsSnapshotV2 = PreparedStatsSnapshotV3 | LoadedStatsSnapshotV3


@dataclass(frozen=True)
class RunCommitV1:
    profile: CheckpointProfileV1
    config_sha256: str
    parent_run_commit_id: str | None
    checkpoint_id: str
    stats_snapshot: StatsSnapshotV2
    champion: ChampionReferenceV1 | None
    evaluation_head_id: str | None
    orchestration: OrchestrationCommitV1 | None
    run_recipe: RunRecipeV1 | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ArtifactValidationError("run_commit.schema_version must be exactly 1")
        if not isinstance(self.profile, CheckpointProfileV1):
            raise ArtifactValidationError("run_commit.profile is invalid")
        validate_sha256_digest(self.config_sha256, field="run_commit.config_sha256")
        if self.run_recipe is not None:
            if not isinstance(self.run_recipe, RunRecipeV1):
                raise ArtifactValidationError("run_commit.run_recipe is invalid")
            if self.run_recipe.learner_config_sha256 != self.config_sha256:
                raise ArtifactValidationError(
                    "RunCommit recipe learner digest disagrees with config_sha256"
                )
        if self.parent_run_commit_id is not None:
            validate_sha256_digest(
                self.parent_run_commit_id,
                field="run_commit.parent_run_commit_id",
            )
        validate_sha256_digest(self.checkpoint_id, field="run_commit.checkpoint_id")
        if not isinstance(self.stats_snapshot, (PreparedStatsSnapshotV3, LoadedStatsSnapshotV3)):
            raise ArtifactValidationError("run_commit.stats_snapshot is invalid")
        binding = self.stats_snapshot.binding
        if (
            binding.profile != self.profile
            or binding.config_sha256 != self.config_sha256
            or binding.checkpoint_id != self.checkpoint_id
        ):
            raise ArtifactValidationError(
                "RunCommit fields do not match the embedded stats binding"
            )
        if self.evaluation_head_id is not None:
            validate_sha256_digest(self.evaluation_head_id, field="run_commit.evaluation_head_id")
        if self.champion is not None and self.evaluation_head_id is None:
            raise ArtifactValidationError("A RunCommit champion requires an evaluation head")
        if self.orchestration is not None and not isinstance(
            self.orchestration, OrchestrationCommitV1
        ):
            raise ArtifactValidationError("run_commit.orchestration is invalid")

    @property
    def stats_id(self) -> str:
        return self.stats_snapshot.stats_id

    @property
    def run_recipe_id(self) -> str | None:
        return self.run_recipe.run_recipe_id if self.run_recipe is not None else None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile.to_dict(),
            "config_sha256": self.config_sha256,
            "run_recipe_id": self.run_recipe_id,
            "run_recipe": (self.run_recipe.to_dict() if self.run_recipe is not None else None),
            "parent_run_commit_id": self.parent_run_commit_id,
            "checkpoint_id": self.checkpoint_id,
            "stats_id": self.stats_id,
            "stats_snapshot": dict(
                _decode_canonical(
                    self.stats_snapshot.data,
                    context="run commit stats snapshot",
                )
            ),
            "champion": self.champion.to_dict() if self.champion is not None else None,
            "evaluation_head_id": self.evaluation_head_id,
            "orchestration": (
                self.orchestration.to_dict() if self.orchestration is not None else None
            ),
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def run_commit_id(self) -> str:
        return sha256_bytes(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "RunCommitV1":
        raw = _decode_canonical(data, context="run commit")
        fields = _exact(raw, _RUN_COMMIT_FIELDS, context="run commit")
        if fields["schema_version"] != 1 or isinstance(fields["schema_version"], bool):
            raise ArtifactValidationError("run_commit.schema_version must be exactly 1")
        profile_fields = _exact(fields["profile"], _PROFILE_FIELDS, context="run commit profile")
        profile = CheckpointProfileV1.from_dict(dict(profile_fields))
        config_sha256 = validate_sha256_digest(
            fields["config_sha256"], field="run_commit.config_sha256"
        )
        run_recipe_value = fields["run_recipe"]
        run_recipe = (
            RunRecipeV1.from_dict(run_recipe_value) if run_recipe_value is not None else None
        )
        run_recipe_id = fields["run_recipe_id"]
        if (run_recipe is None) != (run_recipe_id is None):
            raise ArtifactValidationError(
                "run_commit.run_recipe and run_recipe_id must both be null or present"
            )
        if run_recipe is not None:
            run_recipe_id = validate_sha256_digest(run_recipe_id, field="run_commit.run_recipe_id")
            if run_recipe_id != run_recipe.run_recipe_id:
                raise ArtifactValidationError("RunCommit recipe SHA-256 mismatch")
        checkpoint_id = validate_sha256_digest(
            fields["checkpoint_id"], field="run_commit.checkpoint_id"
        )
        snapshot_bytes = canonical_json_bytes(fields["stats_snapshot"])
        snapshot_binding = StatsBindingV1.from_fields(fields["stats_snapshot"])
        expected_binding = StatsBindingV1(
            profile=profile,
            config_sha256=config_sha256,
            checkpoint_id=checkpoint_id,
            step=snapshot_binding.step,
        )
        stats = decode_stats_snapshot(
            snapshot_bytes,
            expected_stats_id=validate_sha256_digest(
                fields["stats_id"], field="run_commit.stats_id"
            ),
            expected_binding=expected_binding,
        )
        parent = fields["parent_run_commit_id"]
        if parent is not None:
            parent = validate_sha256_digest(parent, field="run_commit.parent_run_commit_id")
        evaluation_head = fields["evaluation_head_id"]
        if evaluation_head is not None:
            evaluation_head = validate_sha256_digest(
                evaluation_head, field="run_commit.evaluation_head_id"
            )
        commit = cls(
            schema_version=1,
            profile=profile,
            config_sha256=config_sha256,
            run_recipe=run_recipe,
            parent_run_commit_id=parent,
            checkpoint_id=checkpoint_id,
            stats_snapshot=stats,
            champion=(
                ChampionReferenceV1.from_dict(fields["champion"])
                if fields["champion"] is not None
                else None
            ),
            evaluation_head_id=evaluation_head,
            orchestration=(
                OrchestrationCommitV1.from_dict(fields["orchestration"])
                if fields["orchestration"] is not None
                else None
            ),
        )
        if commit.to_bytes() != data:
            raise ArtifactValidationError("RunCommit is not normalized canonical JSON")
        return commit


@dataclass(frozen=True)
class RunCommitRef:
    run_commit_id: str
    commit: RunCommitV1

    def __post_init__(self) -> None:
        validate_sha256_digest(self.run_commit_id, field="run_commit_id")
        if self.run_commit_id != self.commit.run_commit_id:
            raise ArtifactValidationError("RunCommitRef ID does not match its bytes")
