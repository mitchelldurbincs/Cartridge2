"""Tests for algorithm-scoped registry and tournament CLI wiring."""

import argparse
from types import SimpleNamespace

import pytest

from trainer import tournament_cli
from trainer.algorithms.alphazero_board_v1 import ALGORITHM_ID
from trainer.environment_catalog import get_environment
from trainer.registry import (
    PlayerRegistry,
    artifact_contract_for,
    checkpoint_player_id,
    random_player_id,
)
from trainer.storage.publisher import (
    BlobDescriptorV1,
    CheckpointManifestV1,
    CheckpointRef,
)

ENVIRONMENT = get_environment("connect4")


def test_register_parser_leaves_algorithm_to_the_global_cli():
    parser = argparse.ArgumentParser()

    tournament_cli.add_register_players_arguments(parser)

    assert not hasattr(parser.parse_args([]), "algorithm")


def test_tournament_parser_leaves_algorithm_to_the_global_cli():
    parser = argparse.ArgumentParser()

    tournament_cli.add_tournament_arguments(parser)

    assert not hasattr(parser.parse_args([]), "algorithm")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--games", "0"],
        ["--games", str(1 << 32)],
        ["--seed", "-1"],
        ["--seed", str(1 << 64)],
    ],
)
def test_tournament_parser_rejects_values_outside_rust_wire_types(arguments):
    parser = argparse.ArgumentParser()
    tournament_cli.add_tournament_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(arguments)


def test_tournament_parser_accepts_rust_wire_boundaries():
    parser = argparse.ArgumentParser()
    tournament_cli.add_tournament_arguments(parser)

    max_games = parser.parse_args(["--games", str((1 << 32) - 1), "--seed", "0"])
    max_seed = parser.parse_args(["--games", "1", "--seed", str((1 << 64) - 1)])

    assert max_games.games == (1 << 32) - 1
    assert max_seed.seed == (1 << 64) - 1


def test_register_command_persists_the_selected_profile(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    contract = artifact_contract_for("connect4", ALGORITHM_ID)
    onnx = BlobDescriptorV1.from_bytes(b"onnx")
    learner = BlobDescriptorV1.from_bytes(b"learner")
    manifest = CheckpointManifestV1(
        profile=contract.profile,
        step=100,
        parent_checkpoint_id=None,
        config_sha256="c" * 64,
        onnx=onnx,
        learner_state=learner,
    )
    reference = CheckpointRef(
        checkpoint_id=manifest.checkpoint_id,
        manifest=manifest,
        onnx_path=models_dir / "blobs" / "sha256" / f"{onnx.sha256}.onnx",
        learner_state_path=models_dir / "blobs" / "sha256" / f"{learner.sha256}.pt",
    )
    repository = SimpleNamespace(list_checkpoints=lambda: [reference])
    registry_path = tmp_path / "players.json"
    args = SimpleNamespace(
        models_dir=str(models_dir),
        registry=str(registry_path),
        env_id="connect4",
        algorithm=ALGORITHM_ID,
        simulations=0,
        temperature=0.2,
        replace=False,
    )
    monkeypatch.setattr(
        "trainer.registry.create_checkpoint_publisher",
        lambda *_args, **_kwargs: repository,
    )
    monkeypatch.setattr(
        "trainer.registry._resolve_registered_checkpoint",
        lambda **_kwargs: reference,
    )

    assert tournament_cli.run_register_players(args) == 0

    registry = PlayerRegistry.load(registry_path)
    ids = [
        p.id
        for p in registry.for_profile(
            env_id="connect4",
            env_contract_version=ENVIRONMENT.contract_version,
            algorithm_id=ALGORITHM_ID,
        )
    ]
    assert ids == [
        random_player_id(
            env_id="connect4",
            env_contract_version=ENVIRONMENT.contract_version,
            algorithm_id=ALGORITHM_ID,
        ),
        checkpoint_player_id(
            env_id="connect4",
            env_contract_version=ENVIRONMENT.contract_version,
            algorithm_id=ALGORITHM_ID,
            checkpoint_id=reference.checkpoint_id,
            temperature=0.2,
        ),
    ]


def test_tournament_command_passes_the_selected_algorithm(tmp_path, monkeypatch, capsys):
    registry_path = tmp_path / "players.json"
    PlayerRegistry().save(registry_path)
    output_path = tmp_path / "tournament.json"
    calls = []

    class FakeResults:
        anchor = random_player_id(
            env_id="connect4",
            env_contract_version=ENVIRONMENT.contract_version,
            algorithm_id=ALGORITHM_ID,
        )
        matches = [object()]
        wall_time_seconds = 0.25

        @staticmethod
        def table() -> str:
            return "ratings"

        @staticmethod
        def save(path) -> None:
            path.write_text("saved")

    def fake_run_tournament(registry, **kwargs):
        calls.append((registry, kwargs))
        return FakeResults()

    monkeypatch.setattr(tournament_cli, "run_tournament", fake_run_tournament)
    args = SimpleNamespace(
        registry=str(registry_path),
        output=str(output_path),
        env_id="connect4",
        algorithm=ALGORITHM_ID,
        games=12,
        seed=99,
    )

    assert tournament_cli.run_tournament_command(args) == 0

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs == {
        "env_id": "connect4",
        "algorithm_id": ALGORITHM_ID,
        "games_per_pair": 12,
        "seed": 99,
    }
    assert output_path.read_text() == "saved"
    assert "ratings" in capsys.readouterr().out
