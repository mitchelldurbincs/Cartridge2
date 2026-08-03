from pathlib import Path

import pytest

from trainer.runtime_profile import RuntimeProfile, resolve_runtime_profile


def test_runtime_profile_matches_language_neutral_namespace():
    profile = RuntimeProfile("alphazero_board_v1", "connect4", 3)

    assert profile.storage_prefix == "profiles/alphazero_board_v1/connect4/v3"
    assert profile.data_dir("/data") == Path("/data/profiles/alphazero_board_v1/connect4/v3")
    assert profile.models_dir("/data") == Path(
        "/data/profiles/alphazero_board_v1/connect4/v3/models"
    )
    assert profile.model_prefix == "profiles/alphazero_board_v1/connect4/v3/models"


@pytest.mark.parametrize("env_id", ["", "Connect4", "../connect4", "connect/4", "connect.4"])
def test_runtime_profile_rejects_unsafe_ids(env_id):
    with pytest.raises(ValueError, match="Invalid runtime profile"):
        RuntimeProfile("alphazero_board_v1", env_id, 1)


@pytest.mark.parametrize("version", [0, -1, True, "1"])
def test_runtime_profile_requires_positive_integer_version(version):
    with pytest.raises(ValueError, match="env_contract_version"):
        RuntimeProfile("alphazero_board_v1", "connect4", version)


def test_catalog_profile_resolution_is_compatible_and_versioned():
    profile = resolve_runtime_profile("alphazero_board_v1", "connect4")

    assert profile == RuntimeProfile("alphazero_board_v1", "connect4", 2)
