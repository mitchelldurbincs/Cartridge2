"""Tests for the player registry."""

import json

import pytest

from trainer.players import ModelPlayer, RandomPlayer
from trainer.registry import (
    RANDOM_PLAYER_ID,
    PlayerRecord,
    PlayerRegistry,
    discover_checkpoints,
    infer_step,
    register_checkpoints,
)


def checkpoint(models_dir, name: str):
    path = models_dir / name
    path.write_bytes(b"not really onnx")
    return path


@pytest.fixture
def models_dir(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()
    return directory


class TestInferStep:
    def test_step_checkpoints_parse(self):
        assert infer_step("model_step_016000.onnx") == 16000
        assert infer_step("/a/b/model_step_000100.onnx") == 100

    def test_aliases_have_no_step(self):
        assert infer_step("latest.onnx") is None
        assert infer_step("best.onnx") is None


class TestPlayerRecord:
    def test_random_record_builds_a_random_player(self):
        record = PlayerRecord(id="random", env_id="connect4", kind="random")
        assert isinstance(record.to_player(), RandomPlayer)

    def test_model_record_carries_search_budget_into_the_player(self):
        record = PlayerRecord(
            id="m",
            env_id="connect4",
            kind="model",
            checkpoint="/models/a.onnx",
            simulations=100,
            temperature=0.2,
        )
        player = record.to_player()
        assert isinstance(player, ModelPlayer)
        assert player.model_path == "/models/a.onnx"
        assert player.simulations == 100
        assert player.temperature == 0.2

    def test_model_without_a_checkpoint_is_an_error(self):
        record = PlayerRecord(id="m", env_id="connect4", kind="model")
        with pytest.raises(ValueError, match="no checkpoint"):
            record.to_player()

    def test_unknown_kind_is_an_error(self):
        record = PlayerRecord(id="m", env_id="connect4", kind="wat")
        with pytest.raises(ValueError, match="unknown kind"):
            record.to_player()

    def test_playability_tracks_the_checkpoint_on_disk(self, models_dir):
        path = checkpoint(models_dir, "model_step_000100.onnx")
        present = PlayerRecord(id="a", env_id="c", kind="model", checkpoint=str(path))
        missing = PlayerRecord(
            id="b", env_id="c", kind="model", checkpoint=str(models_dir / "x")
        )

        assert present.is_playable()
        assert not missing.is_playable()
        assert PlayerRecord(id="r", env_id="c", kind="random").is_playable()


class TestPlayerRegistry:
    def test_duplicate_ids_are_rejected_unless_replacing(self):
        registry = PlayerRegistry()
        registry.add(PlayerRecord(id="a", env_id="connect4", kind="random"))

        with pytest.raises(ValueError, match="already registered"):
            registry.add(PlayerRecord(id="a", env_id="connect4", kind="random"))

        registry.add(
            PlayerRecord(id="a", env_id="othello", kind="random"), replace=True
        )
        assert registry.get("a").env_id == "othello"

    def test_unknown_player_error_lists_what_is_registered(self):
        registry = PlayerRegistry([PlayerRecord(id="a", env_id="c", kind="random")])
        with pytest.raises(KeyError, match="a"):
            registry.get("nope")

    def test_for_env_filters_and_orders_by_step(self):
        registry = PlayerRegistry(
            [
                PlayerRecord(id="s200", env_id="connect4", kind="model", step=200),
                PlayerRecord(id="s100", env_id="connect4", kind="model", step=100),
                PlayerRecord(id="random", env_id="connect4", kind="random"),
                PlayerRecord(id="other", env_id="othello", kind="random"),
            ]
        )

        # Stepless players first, then training order — so a rating table reads
        # as a curve rather than a lexical sort.
        assert [p.id for p in registry.for_env("connect4")] == [
            "random",
            "s100",
            "s200",
        ]
        assert [p.id for p in registry.for_env("othello")] == ["other"]

    def test_round_trips_through_disk(self, tmp_path):
        path = tmp_path / "players.json"
        original = PlayerRegistry(
            [
                PlayerRecord(id="random", env_id="connect4", kind="random"),
                PlayerRecord(
                    id="s100",
                    env_id="connect4",
                    kind="model",
                    checkpoint="/m/a.onnx",
                    simulations=50,
                    step=100,
                    algorithm="alphazero",
                ),
            ]
        )
        original.save(path)
        loaded = PlayerRegistry.load(path)

        assert len(loaded) == 2
        assert loaded.get("s100") == original.get("s100")

    def test_missing_file_loads_as_empty(self, tmp_path):
        assert len(PlayerRegistry.load(tmp_path / "nothing.json")) == 0

    def test_saved_file_is_sorted_and_versioned(self, tmp_path):
        path = tmp_path / "players.json"
        PlayerRegistry(
            [
                PlayerRecord(id="zeta", env_id="c", kind="random"),
                PlayerRecord(id="alpha", env_id="c", kind="random"),
            ]
        ).save(path)

        payload = json.loads(path.read_text())
        assert payload["schema_version"] == 1
        assert [p["id"] for p in payload["players"]] == ["alpha", "zeta"]

    def test_unknown_fields_in_the_file_are_ignored(self, tmp_path):
        # A registry written by a newer version must not crash an older one.
        path = tmp_path / "players.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 99,
                    "players": [
                        {
                            "id": "a",
                            "env_id": "c",
                            "kind": "random",
                            "future_field": True,
                        }
                    ],
                }
            )
        )
        assert PlayerRegistry.load(path).get("a").id == "a"


class TestDiscoverCheckpoints:
    def test_step_checkpoints_sort_numerically_not_lexically(self, models_dir):
        for name in (
            "model_step_000100.onnx",
            "model_step_020000.onnx",
            "model_step_003000.onnx",
        ):
            checkpoint(models_dir, name)

        found = [p.name for p in discover_checkpoints(models_dir)]
        assert found == [
            "model_step_000100.onnx",
            "model_step_003000.onnx",
            "model_step_020000.onnx",
        ]

    def test_aliases_come_last_and_only_when_present(self, models_dir):
        checkpoint(models_dir, "model_step_000100.onnx")
        checkpoint(models_dir, "best.onnx")

        found = [p.name for p in discover_checkpoints(models_dir)]
        assert found == ["model_step_000100.onnx", "best.onnx"]


class TestRegisterCheckpoints:
    def test_registers_every_checkpoint_plus_the_baseline(self, models_dir):
        checkpoint(models_dir, "model_step_000100.onnx")
        checkpoint(models_dir, "model_step_000200.onnx")
        registry = PlayerRegistry()

        added = register_checkpoints(registry, "connect4", models_dir)

        assert {r.id for r in added} == {
            RANDOM_PLAYER_ID,
            "model_step_000100",
            "model_step_000200",
        }
        assert registry.get("model_step_000200").step == 200
        assert registry.get(RANDOM_PLAYER_ID).kind == "random"

    def test_rerunning_only_adds_what_is_new(self, models_dir):
        checkpoint(models_dir, "model_step_000100.onnx")
        registry = PlayerRegistry()
        register_checkpoints(registry, "connect4", models_dir)

        checkpoint(models_dir, "model_step_000200.onnx")
        added = register_checkpoints(registry, "connect4", models_dir)

        assert [r.id for r in added] == ["model_step_000200"]
        assert len(registry) == 3

    def test_search_budget_makes_a_distinct_player_id(self, models_dir):
        # The same weights at 0 and 100 simulations are different players; if
        # their ids collided one would silently shadow the other in a pool.
        checkpoint(models_dir, "model_step_000100.onnx")
        registry = PlayerRegistry()

        register_checkpoints(registry, "connect4", models_dir, simulations=0)
        register_checkpoints(registry, "connect4", models_dir, simulations=100)

        assert "model_step_000100" in registry
        assert "model_step_000100-s100" in registry
        assert registry.get("model_step_000100-s100").simulations == 100

    def test_algorithm_label_is_recorded(self, models_dir):
        checkpoint(models_dir, "model_step_000100.onnx")
        registry = PlayerRegistry()

        register_checkpoints(registry, "connect4", models_dir, algorithm="ppo")

        assert registry.get("model_step_000100").algorithm == "ppo"
