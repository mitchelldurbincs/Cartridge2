"""Exact learner-stats schema, binding, and snapshot tests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from trainer.stats import (
    DEFAULT_MAX_HISTORY,
    MEDIUM_RESOLUTION,
    MEDIUM_STEPS_THRESHOLD,
    OLD_RESOLUTION,
    RECENT_STEPS_THRESHOLD,
    EvalStats,
    LoadedStatsSnapshotV2,
    PreparedStatsSnapshotV2,
    StatsArtifactError,
    StatsBindingV1,
    TrainerStats,
    decode_stats_snapshot,
    prepare_stats_snapshot,
    retain_training_history,
    write_ephemeral_stats_projection,
    write_stats_projection,
)
from trainer.storage.publisher import (
    BlobDescriptorV1,
    CheckpointManifestV1,
    CheckpointProfileV1,
    CheckpointRef,
    canonical_json_bytes,
    sha256_bytes,
)

PROFILE = CheckpointProfileV1(
    algorithm_id="alphazero_board_v1",
    env_id="tictactoe",
    env_contract_version=1,
    model_artifact_schema_version=1,
    model_contract="onnx_policy_value_v1",
)
CONFIG_SHA256 = "c" * 64


def checkpoint_ref(
    step: int,
    *,
    profile: CheckpointProfileV1 = PROFILE,
    config_sha256: str = CONFIG_SHA256,
    parent_checkpoint_id: str | None = None,
) -> CheckpointRef:
    manifest = CheckpointManifestV1(
        profile=profile,
        step=step,
        parent_checkpoint_id=parent_checkpoint_id,
        config_sha256=config_sha256,
        onnx=BlobDescriptorV1("a" * 64, 1),
        learner_state=BlobDescriptorV1("b" * 64, 1),
    )
    return CheckpointRef(
        checkpoint_id=manifest.checkpoint_id,
        manifest=manifest,
        onnx_path=Path("unused.onnx"),
        learner_state_path=Path("unused.pt"),
    )


def history_entry(step: int, loss: float | int = 0.0) -> dict[str, object]:
    return {
        "step": step,
        "total_loss": loss,
        "value_loss": float(loss) / 3.0,
        "policy_loss": float(loss) * 2.0 / 3.0,
        "learning_rate": 0.001,
        "grad_norm": None,
    }


def eval_stats(step: int, win_rate: float, *, timestamp: float = 1.0) -> EvalStats:
    return EvalStats(
        step=step,
        win_rate=win_rate,
        draw_rate=0.0,
        loss_rate=1.0 - win_rate,
        games_played=10,
        avg_game_length=1.0,
        timestamp=timestamp,
    )


def bound_stats(
    step: int,
    *,
    checkpoint: CheckpointRef | None = None,
    timestamp: float = 1.0,
    total_loss: float | int = 0.5,
) -> tuple[TrainerStats, CheckpointRef]:
    checkpoint = checkpoint or checkpoint_ref(step)
    stats = TrainerStats(
        step=step,
        total_steps=step + 10,
        total_loss=total_loss,
        value_loss=0.2,
        policy_loss=0.3,
        learning_rate=0.001,
        samples_seen=step * 4,
        replay_record_count=step * 8,
        last_checkpoint=checkpoint.checkpoint_id,
        timestamp=timestamp,
        history=[history_entry(step, total_loss)],
        env_id=checkpoint.manifest.profile.env_id,
    )
    return stats, checkpoint


class TestEvalStatsContract:
    def test_default_is_the_only_valid_zero_game_record(self):
        stats = EvalStats(timestamp=0)

        assert stats.to_dict() == {
            "step": 0,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "games_played": 0,
            "avg_game_length": 0.0,
            "timestamp": 0.0,
        }

    def test_exact_schema_round_trip(self):
        original = eval_stats(5, 0.7, timestamp=2.0)

        assert EvalStats.from_dict(original.to_dict()) == original
        with pytest.raises(StatsArtifactError, match="fields must be exact"):
            EvalStats.from_dict({**original.to_dict(), "extra": 1})
        missing = original.to_dict()
        del missing["loss_rate"]
        with pytest.raises(StatsArtifactError, match="fields must be exact"):
            EvalStats.from_dict(missing)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"games_played": 0, "win_rate": 1.0}, "zero-game"),
            (
                {
                    "games_played": 2,
                    "win_rate": 0.4,
                    "draw_rate": 0.2,
                    "loss_rate": 0.3,
                    "avg_game_length": 2.0,
                },
                "sum to one",
            ),
            (
                {
                    "games_played": 2,
                    "win_rate": 1.0,
                    "avg_game_length": 0.0,
                },
                "positive average length",
            ),
            ({"win_rate": 1.1}, r"\[0, 1\]"),
            ({"games_played": -1}, "nonnegative integer"),
            ({"timestamp": -1.0}, "nonnegative"),
        ],
    )
    def test_semantic_invariants(self, overrides, message):
        values = {
            "step": 1,
            "win_rate": 0.0,
            "draw_rate": 0.0,
            "loss_rate": 0.0,
            "games_played": 0,
            "avg_game_length": 0.0,
            "timestamp": 1.0,
        }
        values.update(overrides)

        with pytest.raises(StatsArtifactError, match=message):
            EvalStats(**values)

    def test_numeric_normalization_collapses_signed_zero(self):
        stats = EvalStats(
            win_rate=-0.0,
            draw_rate=0,
            loss_rate=0,
            games_played=0,
            avg_game_length=-0.0,
            timestamp=0,
        )

        assert canonical_json_bytes(stats.to_dict()).count(b"-0.0") == 0
        assert all(
            isinstance(stats.to_dict()[field], float)
            for field in (
                "win_rate",
                "draw_rate",
                "loss_rate",
                "avg_game_length",
                "timestamp",
            )
        )

    def test_rate_roundoff_at_probability_boundaries_is_normalized(self):
        stats = EvalStats(
            step=1,
            win_rate=0.8,
            draw_rate=0.2,
            loss_rate=1.0 - 0.8 - 0.2,
            games_played=10,
            avg_game_length=1.0,
            timestamp=1.0,
        )

        assert stats.loss_rate == 0.0


class TestTrainerStatsContract:
    def test_exact_schema_round_trip(self):
        stats, _ = bound_stats(10)
        stats.append_eval(eval_stats(10, 0.7))

        restored = TrainerStats.from_dict(stats.to_dict())

        assert restored.to_dict() == stats.to_dict()
        extra = stats.to_dict()
        extra["extra"] = 1
        with pytest.raises(StatsArtifactError, match="fields must be exact"):
            TrainerStats.from_dict(extra)
        with pytest.raises(StatsArtifactError, match="fields must be exact"):
            TrainerStats.from_dict({})

    @pytest.mark.parametrize("remove", [True, False])
    def test_history_entries_have_one_exact_schema(self, remove):
        entry = history_entry(1, 0.3)
        if remove:
            del entry["grad_norm"]
        else:
            entry["extra"] = 1
        stats = TrainerStats(step=1, total_steps=1)

        with pytest.raises(StatsArtifactError, match="fields must be exact"):
            stats.append_history(entry)

    @pytest.mark.parametrize(
        ("field_name", "value", "message"),
        [
            ("step", True, "nonnegative integer"),
            ("total_loss", -0.1, "nonnegative"),
            ("value_loss", float("nan"), "finite nonnegative"),
            ("policy_loss", float("inf"), "finite nonnegative"),
            ("learning_rate", -0.1, "nonnegative"),
            ("grad_norm", -0.1, "nonnegative"),
        ],
    )
    def test_history_numeric_contract(self, field_name, value, message):
        entry = history_entry(1, 0.3)
        entry[field_name] = value
        stats = TrainerStats(step=1, total_steps=1)

        with pytest.raises(StatsArtifactError, match=message):
            stats.append_history(entry)

    def test_history_is_strictly_ordered_and_not_in_the_future(self):
        with pytest.raises(StatsArtifactError, match="strictly increasing"):
            TrainerStats(
                step=2,
                total_steps=2,
                history=[history_entry(2), history_entry(1)],
            )
        with pytest.raises(StatsArtifactError, match="beyond stats.step"):
            TrainerStats(step=1, total_steps=2, history=[history_entry(2)])

    def test_eval_history_order_time_and_last_record_are_exact(self):
        first = eval_stats(1, 0.5, timestamp=2.0)
        second = eval_stats(2, 0.6, timestamp=1.0)
        with pytest.raises(StatsArtifactError, match="timestamps.*nondecreasing"):
            TrainerStats(
                step=2,
                total_steps=2,
                last_eval=second,
                eval_history=[first.to_dict(), second.to_dict()],
            )
        with pytest.raises(StatsArtifactError, match="last_eval must equal"):
            TrainerStats(
                step=2,
                total_steps=2,
                last_eval=first,
                eval_history=[
                    first.to_dict(),
                    eval_stats(2, 0.6, timestamp=3.0).to_dict(),
                ],
            )
        with pytest.raises(StatsArtifactError, match="last_eval must be null"):
            TrainerStats(step=1, total_steps=1, last_eval=first)

    def test_top_level_counters_and_losses_are_bounded(self):
        with pytest.raises(StatsArtifactError, match="total_steps"):
            TrainerStats(step=2, total_steps=1)
        with pytest.raises(StatsArtifactError, match="nonnegative"):
            TrainerStats(total_loss=-0.1)
        with pytest.raises(StatsArtifactError, match="within u64"):
            TrainerStats(samples_seen=2**64)

    def test_append_eval_and_history_require_current_stats_step(self):
        stats = TrainerStats(step=1, total_steps=2)
        stats.append_history(history_entry(1, 0.5))
        stats.append_eval(eval_stats(1, 0.5))

        with pytest.raises(StatsArtifactError, match="beyond stats.step"):
            stats.append_history(history_entry(2, 0.4))
        with pytest.raises(StatsArtifactError, match="beyond stats.step"):
            stats.append_eval(eval_stats(2, 0.6, timestamp=2.0))

    def test_numeric_normalization_is_recursive(self):
        stats = TrainerStats(
            total_loss=-0.0,
            value_loss=0,
            policy_loss=0,
            learning_rate=-0.0,
            timestamp=0,
            history=[
                {
                    **history_entry(0),
                    "total_loss": -0.0,
                    "learning_rate": -0.0,
                    "grad_norm": -0.0,
                }
            ],
        )

        encoded = canonical_json_bytes(stats.to_dict())
        assert b"-0.0" not in encoded
        assert stats.value_loss == 0.0
        assert isinstance(stats.value_loss, float)
        assert stats.history[0]["grad_norm"] == 0.0


class TestHistoryRetention:
    def test_empty_history_is_unchanged(self):
        assert retain_training_history([], 1000) == []

    def test_tiered_retention_preserves_expected_resolutions(self):
        current_step = 15000
        history = [{"step": step} for step in range(0, current_step + 1, 10)]

        result = retain_training_history(history, current_step)

        for entry in result:
            age = current_step - entry["step"]
            if age <= RECENT_STEPS_THRESHOLD:
                continue
            if age <= MEDIUM_STEPS_THRESHOLD:
                assert entry["step"] % MEDIUM_RESOLUTION == 0
            else:
                assert entry["step"] % OLD_RESOLUTION == 0

    def test_append_enforces_absolute_bound(self):
        last = DEFAULT_MAX_HISTORY + 4
        history = [history_entry(last, float(index)) for index in range(last + 1)]

        retained = retain_training_history(history, last)

        assert len(retained) == DEFAULT_MAX_HISTORY
        assert retained[-1]["step"] == last

    def test_eval_history_enforces_absolute_bound(self):
        stats = TrainerStats(step=9, total_steps=9)
        stats._max_eval_history = 3
        for step in range(10):
            stats.append_eval(eval_stats(step, step / 10, timestamp=float(step)))

        assert [entry["step"] for entry in stats.eval_history] == [7, 8, 9]
        assert stats.last_eval is not None
        assert stats.last_eval.step == 9


class TestPreparedStatsSnapshot:
    def test_prepare_and_decode_bind_every_checkpoint_identity_field(self):
        stats, checkpoint = bound_stats(10)

        prepared = prepare_stats_snapshot(stats, checkpoint)
        loaded = decode_stats_snapshot(
            prepared.data,
            expected_stats_id=prepared.stats_id,
            expected_binding=StatsBindingV1.from_checkpoint(checkpoint),
        )

        assert isinstance(prepared, PreparedStatsSnapshotV2)
        assert isinstance(loaded, LoadedStatsSnapshotV2)
        assert prepared.stats_id == sha256_bytes(prepared.data)
        assert loaded.binding.profile == checkpoint.manifest.profile
        assert loaded.binding.config_sha256 == checkpoint.manifest.config_sha256
        assert loaded.binding.checkpoint_id == checkpoint.checkpoint_id
        assert loaded.binding.step == checkpoint.manifest.step
        assert loaded.stats.to_dict() == stats.to_dict()
        assert prepared.to_dict() == json.loads(prepared.data)

    def test_snapshot_has_one_exact_nested_schema(self):
        stats, checkpoint = bound_stats(10)
        prepared = prepare_stats_snapshot(stats, checkpoint)
        raw = prepared.to_dict()

        assert set(raw) == {
            "schema_version",
            "profile",
            "config_sha256",
            "checkpoint_id",
            "step",
            "stats",
        }
        raw["extra"] = 1
        data = canonical_json_bytes(raw)
        with pytest.raises(StatsArtifactError, match="fields must be exact"):
            decode_stats_snapshot(data, expected_stats_id=sha256_bytes(data))

        raw = prepared.to_dict()
        del raw["stats"]["history"]
        data = canonical_json_bytes(raw)
        with pytest.raises(StatsArtifactError, match="fields must be exact"):
            decode_stats_snapshot(data, expected_stats_id=sha256_bytes(data))

    def test_decode_rejects_id_and_run_commit_binding_mismatch(self):
        stats, checkpoint = bound_stats(10)
        prepared = prepare_stats_snapshot(stats, checkpoint)

        with pytest.raises(StatsArtifactError, match="SHA-256 mismatch"):
            decode_stats_snapshot(prepared.data, expected_stats_id="f" * 64)
        other = checkpoint_ref(10, config_sha256="d" * 64)
        with pytest.raises(StatsArtifactError, match="RunCommit binding"):
            decode_stats_snapshot(
                prepared.data,
                expected_stats_id=prepared.stats_id,
                expected_binding=StatsBindingV1.from_checkpoint(other),
            )

    @pytest.mark.parametrize(
        "mutation",
        [
            lambda raw: raw["stats"].__setitem__("total_loss", 1),
            lambda raw: raw["stats"].__setitem__("total_loss", -0.0),
            lambda raw: raw["stats"]["history"][0].__setitem__("grad_norm", -0.0),
        ],
    )
    def test_decode_rejects_alternate_numeric_spellings(self, mutation):
        stats, checkpoint = bound_stats(10, total_loss=1.0)
        prepared = prepare_stats_snapshot(stats, checkpoint)
        raw = prepared.to_dict()
        mutation(raw)
        data = canonical_json_bytes(raw)

        with pytest.raises(StatsArtifactError, match="normalized form"):
            decode_stats_snapshot(data, expected_stats_id=sha256_bytes(data))

    def test_integer_and_float_inputs_produce_one_content_id(self):
        integer_stats, checkpoint = bound_stats(10, total_loss=1)
        float_stats, _ = bound_stats(10, checkpoint=checkpoint, total_loss=1.0)

        integer_snapshot = prepare_stats_snapshot(integer_stats, checkpoint)
        float_snapshot = prepare_stats_snapshot(float_stats, checkpoint)

        assert integer_snapshot.data == float_snapshot.data
        assert integer_snapshot.stats_id == float_snapshot.stats_id

    def test_prepare_rejects_cross_profile_and_config_relabeling(self):
        stats, first = bound_stats(10)
        prepare_stats_snapshot(stats, first)
        other_profile = CheckpointProfileV1(
            algorithm_id=PROFILE.algorithm_id,
            env_id="connect4",
            env_contract_version=1,
            model_artifact_schema_version=1,
            model_contract=PROFILE.model_contract,
        )

        for checkpoint in (
            checkpoint_ref(11, profile=other_profile),
            checkpoint_ref(11, config_sha256="d" * 64),
        ):
            stats.step = 11
            stats.total_steps = 20
            stats.env_id = checkpoint.manifest.profile.env_id
            stats.last_checkpoint = checkpoint.checkpoint_id
            with pytest.raises(
                StatsArtifactError, match="different.*profile or config"
            ):
                prepare_stats_snapshot(stats, checkpoint)

    def test_prepare_rejects_checkpoint_step_regression(self):
        first = checkpoint_ref(10)
        stats, _ = bound_stats(10, checkpoint=first)
        prepare_stats_snapshot(stats, first)
        older = checkpoint_ref(9)
        stats.step = 9
        stats.total_steps = 20
        stats.last_checkpoint = older.checkpoint_id

        with pytest.raises(StatsArtifactError, match="older checkpoint step"):
            prepare_stats_snapshot(stats, older)

    def test_prepare_rejects_same_step_checkpoint_relabeling(self):
        first = checkpoint_ref(10)
        stats, _ = bound_stats(10, checkpoint=first)
        prepare_stats_snapshot(stats, first)
        replacement = checkpoint_ref(10, parent_checkpoint_id="d" * 64)
        stats.last_checkpoint = replacement.checkpoint_id

        with pytest.raises(StatsArtifactError, match="different checkpoint"):
            prepare_stats_snapshot(stats, replacement)

    @pytest.mark.parametrize(
        ("field_name", "value", "message"),
        [
            ("step", 9, "checkpoint step"),
            ("env_id", "connect4", "checkpoint profile"),
            ("last_checkpoint", "", "bound checkpoint"),
        ],
    )
    def test_prepare_rejects_stats_checkpoint_disagreement(
        self, field_name, value, message
    ):
        stats, checkpoint = bound_stats(10)
        setattr(stats, field_name, value)

        with pytest.raises(StatsArtifactError, match=message):
            prepare_stats_snapshot(stats, checkpoint)
        assert stats._binding is None


class TestStatsProjection:
    def test_projection_contains_only_rebuildable_stats(self, tmp_path):
        stats, checkpoint = bound_stats(10)
        prepared = prepare_stats_snapshot(stats, checkpoint)
        path = tmp_path / "nested" / "stats.json"

        write_stats_projection(prepared, path)

        assert json.loads(path.read_bytes()) == stats.to_dict()
        assert not (tmp_path / "nested" / "stats" / "channels").exists()
        assert not (tmp_path / "nested" / "stats" / "manifests").exists()

    def test_projection_accepts_verified_loaded_snapshot(self, tmp_path):
        stats, checkpoint = bound_stats(10)
        prepared = prepare_stats_snapshot(stats, checkpoint)
        loaded = decode_stats_snapshot(
            prepared.data, expected_stats_id=prepared.stats_id
        )

        write_stats_projection(loaded, tmp_path / "stats.json")

        assert json.loads((tmp_path / "stats.json").read_bytes()) == stats.to_dict()

    def test_ephemeral_projection_can_advance_beyond_selected_binding(self, tmp_path):
        stats, checkpoint = bound_stats(10)
        prepare_stats_snapshot(stats, checkpoint)
        stats.step = 11
        stats.total_steps = 20
        stats.samples_seen += 4
        stats.append_history(history_entry(11, 0.4))

        write_ephemeral_stats_projection(stats, tmp_path / "stats.json")

        projection = json.loads((tmp_path / "stats.json").read_bytes())
        assert projection["step"] == 11
        assert projection["last_checkpoint"] == checkpoint.checkpoint_id

    def test_concurrent_projection_reads_never_observe_partial_json(self, tmp_path):
        path = tmp_path / "stats.json"
        initial_stats, initial_checkpoint = bound_stats(0)
        write_stats_projection(
            prepare_stats_snapshot(initial_stats, initial_checkpoint), path
        )
        errors: list[str] = []
        observed: list[int] = []

        def reader() -> None:
            for _ in range(50):
                try:
                    observed.append(json.loads(path.read_bytes())["step"])
                except json.JSONDecodeError as exc:
                    errors.append(str(exc))
                time.sleep(0.001)

        def writer() -> None:
            for step in range(1, 31):
                stats, checkpoint = bound_stats(step)
                write_stats_projection(prepare_stats_snapshot(stats, checkpoint), path)
                time.sleep(0.001)

        threads = [threading.Thread(target=reader), threading.Thread(target=writer)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert all(isinstance(step, int) and 0 <= step <= 30 for step in observed)
