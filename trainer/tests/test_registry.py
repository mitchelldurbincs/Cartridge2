"""Tests for content-addressed player registration."""

import json
import math
import struct
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainer.algorithms.alphazero_board_v1 import ALGORITHM_ID
from trainer.players import ModelPlayer, RandomPlayer
from trainer.registry import (
    PlayerRecord,
    PlayerRegistry,
    artifact_contract_for,
    checkpoint_player_id,
    discover_checkpoints,
    random_player_id,
    register_checkpoints,
)
from trainer.storage.publisher import (
    ArtifactValidationError,
    BlobDescriptorV1,
    CheckpointManifestV1,
    CheckpointRef,
)

REGISTERED_AT = "2026-08-02T12:00:00"
CONFIG_SHA256 = "c" * 64
_REFERENCES: dict[str, CheckpointRef] = {}


def contract(env_id: str = "connect4"):
    return artifact_contract_for(env_id, ALGORITHM_ID)


def checkpoint_ref(
    root: Path,
    step: int,
    *,
    env_id: str = "connect4",
    salt: str = "",
) -> CheckpointRef:
    onnx_data = f"onnx:{env_id}:{step}:{salt}".encode()
    learner_data = f"learner:{env_id}:{step}:{salt}".encode()
    manifest = CheckpointManifestV1(
        profile=contract(env_id).profile,
        step=step,
        parent_checkpoint_id=None,
        config_sha256=CONFIG_SHA256,
        onnx=BlobDescriptorV1.from_bytes(onnx_data),
        learner_state=BlobDescriptorV1.from_bytes(learner_data),
    )
    reference = CheckpointRef(
        checkpoint_id=manifest.checkpoint_id,
        manifest=manifest,
        onnx_path=root / "blobs" / "sha256" / f"{manifest.onnx.sha256}.onnx",
        learner_state_path=(root / "blobs" / "sha256" / f"{manifest.learner_state.sha256}.pt"),
    )
    _REFERENCES[reference.checkpoint_id] = reference
    return reference


def random_record(env_id: str = "connect4", **changes) -> PlayerRecord:
    profile = contract(env_id)
    values = {
        "id": random_player_id(
            env_id=env_id,
            env_contract_version=profile.env_contract_version,
            algorithm_id=ALGORITHM_ID,
        ),
        "env_id": env_id,
        "env_contract_version": profile.env_contract_version,
        "algorithm_id": ALGORITHM_ID,
        "model_contract": profile.model_contract,
        "model_artifact_schema_version": profile.model_artifact_schema_version,
        "kind": "random",
        "registered_at": REGISTERED_AT,
    }
    values.update(changes)
    return PlayerRecord(**values)


def model_record(
    reference: CheckpointRef,
    *,
    env_id: str = "connect4",
    simulations: int = 0,
    temperature: float = 0.2,
    **changes,
) -> PlayerRecord:
    profile = contract(env_id)
    values = {
        "id": checkpoint_player_id(
            env_id=env_id,
            env_contract_version=profile.env_contract_version,
            algorithm_id=ALGORITHM_ID,
            checkpoint_id=reference.checkpoint_id,
            simulations=simulations,
            temperature=temperature,
        ),
        "env_id": env_id,
        "env_contract_version": profile.env_contract_version,
        "algorithm_id": ALGORITHM_ID,
        "model_contract": profile.model_contract,
        "model_artifact_schema_version": profile.model_artifact_schema_version,
        "kind": "model",
        "registered_at": REGISTERED_AT,
        "checkpoint_id": reference.checkpoint_id,
        "onnx_path": str(reference.onnx_path),
        "simulations": simulations,
        "temperature": temperature,
        "step": reference.manifest.step,
    }
    values.update(changes)
    return PlayerRecord(**values)


@pytest.fixture(autouse=True)
def artifact_resolver(monkeypatch):
    _REFERENCES.clear()

    def resolve(*, checkpoint_id, onnx_path, contract):
        del contract
        try:
            reference = _REFERENCES[checkpoint_id]
        except KeyError as exc:
            raise ArtifactValidationError(
                f"Checkpoint manifest does not exist: {checkpoint_id}"
            ) from exc
        if Path(onnx_path) != reference.onnx_path:
            raise ArtifactValidationError("registered ONNX path mismatch")
        return reference

    monkeypatch.setattr("trainer.registry._resolve_registered_checkpoint", resolve)


class TestPlayerIdentity:
    def test_checkpoint_and_adapter_settings_are_all_part_of_id(self, tmp_path):
        first = checkpoint_ref(tmp_path, 100, salt="first")
        second = checkpoint_ref(tmp_path, 100, salt="second")
        profile = contract()

        def identity(reference, simulations=0, temperature=0.2):
            return checkpoint_player_id(
                env_id="connect4",
                env_contract_version=profile.env_contract_version,
                algorithm_id=ALGORITHM_ID,
                checkpoint_id=reference.checkpoint_id,
                simulations=simulations,
                temperature=temperature,
            )

        ids = {
            identity(first),
            identity(second),
            identity(first, simulations=100),
            identity(first, temperature=0.3),
        }
        assert len(ids) == 4
        assert first.checkpoint_id in identity(first)
        assert second.checkpoint_id in identity(second)

    @pytest.mark.parametrize(
        ("simulations", "temperature"),
        [
            (-1, 0.2),
            (True, 0.2),
            (1 << 32, 0.2),
            (0, float("nan")),
            (0, float("inf")),
            (0, -0.1),
            (0, float.fromhex("0x1p+128")),
        ],
    )
    def test_noncanonical_adapter_settings_are_rejected(self, simulations, temperature):
        with pytest.raises(ValueError):
            checkpoint_player_id(
                env_id="connect4",
                env_contract_version=1,
                algorithm_id=ALGORITHM_ID,
                checkpoint_id="a" * 64,
                simulations=simulations,
                temperature=temperature,
            )


class TestPlayerRecord:
    def test_random_record_builds_random_player(self):
        assert isinstance(random_record().to_player(), RandomPlayer)

    def test_model_record_resolves_immutable_blob(self, tmp_path):
        reference = checkpoint_ref(tmp_path, 100)
        record = model_record(reference, simulations=100, temperature=0.3)

        player = record.to_player()

        assert isinstance(player, ModelPlayer)
        assert player.model_path == str(reference.onnx_path)
        assert player.simulations == 100
        expected_temperature = float(struct.unpack("!f", struct.pack("!f", 0.3))[0])
        assert record.temperature == expected_temperature
        assert player.temperature == record.temperature

    def test_record_normalizes_negative_zero_temperature(self, tmp_path):
        record = model_record(checkpoint_ref(tmp_path, 100), temperature=-0.0)

        assert record.temperature == 0.0
        assert math.copysign(1.0, record.temperature) == 1.0

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"checkpoint_id": None}, "checkpoint_id"),
            ({"onnx_path": None}, "ONNX path"),
            ({"step": None}, "training step"),
            ({"kind": "unknown"}, "unknown kind"),
            ({"env_contract_version": 0}, "env_contract_version"),
            ({"env_contract_version": 1 << 32}, "env_contract_version.*u32"),
            ({"model_artifact_schema_version": True}, "model_artifact_schema_version"),
            (
                {"model_artifact_schema_version": 1 << 32},
                "model_artifact_schema_version.*u32",
            ),
            ({"registered_at": "not-a-time"}, "registered_at"),
            ({"simulations": -1}, "simulations"),
            ({"temperature": float("nan")}, "temperature"),
            ({"step": 1 << 64}, "step.*u64"),
        ],
    )
    def test_malformed_fields_are_rejected(self, tmp_path, changes, message):
        with pytest.raises(ValueError, match=message):
            model_record(checkpoint_ref(tmp_path, 100), **changes)

    def test_record_accepts_rust_integer_wire_boundaries(self, tmp_path):
        record = model_record(
            checkpoint_ref(tmp_path, 100),
            env_contract_version=(1 << 32) - 1,
            model_artifact_schema_version=(1 << 32) - 1,
            simulations=(1 << 32) - 1,
            step=(1 << 64) - 1,
        )

        assert record.env_contract_version == (1 << 32) - 1
        assert record.model_artifact_schema_version == (1 << 32) - 1
        assert record.simulations == (1 << 32) - 1
        assert record.step == (1 << 64) - 1

    def test_missing_immutable_artifact_is_fatal(self, tmp_path):
        reference = checkpoint_ref(tmp_path, 100)
        record = model_record(reference)
        _REFERENCES.clear()

        with pytest.raises(ArtifactValidationError, match="does not exist"):
            record.validate_artifact()

    def test_manifest_step_must_match_record(self, tmp_path):
        record = model_record(checkpoint_ref(tmp_path, 100), step=101)

        with pytest.raises(ArtifactValidationError, match="does not match"):
            record.validate_artifact()

    def test_profile_lineage_must_match_engine_manifest(self, tmp_path):
        record = replace(
            model_record(checkpoint_ref(tmp_path, 100)),
            env_contract_version=contract().env_contract_version + 1,
        )

        with pytest.raises(ValueError, match="does not match the engine profile"):
            record.validate_artifact()

    def test_ids_must_be_canonical(self):
        with pytest.raises(ValueError, match="not canonical"):
            replace(random_record(), id="random").validate_artifact()


class TestPlayerRegistry:
    def test_duplicate_ids_are_rejected_unless_replacing(self):
        registry = PlayerRegistry([random_record()])
        with pytest.raises(ValueError, match="already registered"):
            registry.add(random_record())
        replacement = replace(random_record(), registered_at="2026-08-02T13:00:00")
        registry.add(replacement, replace=True)
        assert registry.get(replacement.id) == replacement

    def test_for_profile_filters_and_orders_by_manifest_step(self, tmp_path):
        c4_200 = model_record(checkpoint_ref(tmp_path, 200))
        c4_100 = model_record(checkpoint_ref(tmp_path, 100))
        ttt = random_record("tictactoe")
        registry = PlayerRegistry([c4_200, c4_100, random_record(), ttt])

        players = registry.for_profile(
            env_id="connect4",
            env_contract_version=contract().env_contract_version,
            algorithm_id=ALGORITHM_ID,
        )
        assert [player.step for player in players] == [None, 100, 200]

    def test_round_trips_exact_v5_records(self, tmp_path):
        path = tmp_path / "players.json"
        reference = checkpoint_ref(tmp_path / "models", 100)
        original = PlayerRegistry([random_record(), model_record(reference, simulations=50)])

        original.save(path)
        loaded = PlayerRegistry.load(path)

        assert len(loaded) == 2
        assert loaded.get(model_record(reference, simulations=50).id) == model_record(
            reference, simulations=50
        )
        assert json.loads(path.read_text())["schema_version"] == 5

    @pytest.mark.parametrize(
        "payload",
        [
            {"schema_version": 4, "players": []},
            {"schema_version": True, "players": []},
            {"schema_version": 5.0, "players": []},
            {"schema_version": 5, "players": [], "future": True},
            {"schema_version": 5, "players": {}},
            [],
        ],
    )
    def test_registry_document_shape_fails_closed(self, tmp_path, payload):
        path = tmp_path / "players.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            PlayerRegistry.load(path)

    def test_duplicate_json_keys_are_rejected(self, tmp_path):
        path = tmp_path / "players.json"
        path.write_text('{"schema_version":5,"schema_version":5,"players":[]}')
        with pytest.raises(ValueError, match="duplicate key"):
            PlayerRegistry.load(path)

    def test_entry_fields_are_exact(self, tmp_path):
        entry = asdict(random_record())
        entry["unknown_field"] = "unexpected"
        path = tmp_path / "players.json"
        path.write_text(json.dumps({"schema_version": 5, "players": [entry]}))
        with pytest.raises(ValueError, match="invalid fields"):
            PlayerRegistry.load(path)


class TestCheckpointDiscovery:
    def test_repository_order_is_preserved_when_canonical(self, tmp_path):
        first = checkpoint_ref(tmp_path, 100)
        second = checkpoint_ref(tmp_path, 200)
        repository = SimpleNamespace(list_checkpoints=lambda: [first, second])
        assert discover_checkpoints(repository) == [first, second]

    def test_noncanonical_repository_order_is_rejected(self, tmp_path):
        first = checkpoint_ref(tmp_path, 100)
        second = checkpoint_ref(tmp_path, 200)
        repository = SimpleNamespace(list_checkpoints=lambda: [second, first])
        with pytest.raises(ArtifactValidationError, match="order"):
            discover_checkpoints(repository)

    def test_duplicate_manifest_id_is_rejected(self, tmp_path):
        reference = checkpoint_ref(tmp_path, 100)
        repository = SimpleNamespace(list_checkpoints=lambda: [reference, reference])
        with pytest.raises(ArtifactValidationError, match="duplicate"):
            discover_checkpoints(repository)


class TestRegisterCheckpoints:
    def test_registers_manifest_refs_plus_versioned_baseline(self, tmp_path):
        first = checkpoint_ref(tmp_path, 100)
        second = checkpoint_ref(tmp_path, 200)
        repository = SimpleNamespace(list_checkpoints=lambda: [first, second])
        registry = PlayerRegistry()

        added = register_checkpoints(
            registry,
            "connect4",
            tmp_path,
            algorithm_id=ALGORITHM_ID,
            repository=repository,
        )

        assert [record.step for record in added] == [None, 100, 200]
        assert {record.checkpoint_id for record in added if record.kind == "model"} == {
            first.checkpoint_id,
            second.checkpoint_id,
        }
        assert all(
            record.kind == "random" or f"checkpoint:{record.checkpoint_id}:adapter:" in record.id
            for record in added
        )

    def test_rerun_only_adds_new_checkpoint_and_adapter_changes_id(self, tmp_path):
        first = checkpoint_ref(tmp_path, 100)
        references = [first]
        repository = SimpleNamespace(list_checkpoints=lambda: list(references))
        registry = PlayerRegistry()
        register_checkpoints(
            registry,
            "connect4",
            tmp_path,
            algorithm_id=ALGORITHM_ID,
            repository=repository,
        )
        second = checkpoint_ref(tmp_path, 200)
        references.append(second)

        added = register_checkpoints(
            registry,
            "connect4",
            tmp_path,
            algorithm_id=ALGORITHM_ID,
            repository=repository,
        )
        searched = register_checkpoints(
            registry,
            "connect4",
            tmp_path,
            algorithm_id=ALGORITHM_ID,
            simulations=100,
            repository=repository,
        )

        assert [record.checkpoint_id for record in added] == [second.checkpoint_id]
        assert len(searched) == 2
        assert {record.checkpoint_id for record in searched} == {
            first.checkpoint_id,
            second.checkpoint_id,
        }

    def test_mismatched_manifest_profile_leaves_registry_unmodified(self, tmp_path):
        reference = checkpoint_ref(tmp_path, 100, env_id="tictactoe")
        repository = SimpleNamespace(list_checkpoints=lambda: [reference])
        registry = PlayerRegistry()

        with pytest.raises(ArtifactValidationError, match="profile"):
            register_checkpoints(
                registry,
                "connect4",
                tmp_path,
                algorithm_id=ALGORITHM_ID,
                repository=repository,
            )
        assert len(registry) == 0
