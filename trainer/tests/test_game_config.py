"""Tests for the AlphaZero recipe and engine-owned environment catalog.

Tests cover:
- AlphaZeroGameConfig initialization and properties
- Environment and algorithm registry functionality
- Configuration validation
- Specific game configurations (TicTacToe, Connect4, Othello)
"""

import pytest

from trainer.algorithms.alphazero_board_v1 import (
    ALGORITHM_ID,
    AlphaZeroGameConfig,
    get_game_config,
    list_compatible_environments,
)
from trainer.environment_catalog import ENVIRONMENTS, get_environment, list_environments


class TestAlphaZeroGameConfig:
    """Tests for the algorithm-specific learner configuration."""

    def test_game_config_defaults(self):
        """Test AlphaZeroGameConfig default values."""
        config = AlphaZeroGameConfig(
            env_id="test",
            display_name="Test Game",
            board_width=5,
            board_height=5,
            num_actions=25,
            obs_size=50,
            obs_channels=2,
        )

        assert config.env_id == "test"
        assert config.display_name == "Test Game"
        assert config.board_width == 5
        assert config.board_height == 5
        assert config.num_actions == 25
        assert config.obs_size == 50
        # Check defaults
        assert config.hidden_size == 128
        assert config.network_type == "mlp"
        assert config.num_res_blocks == 4
        assert config.num_filters == 128
        assert config.obs_channels == 2

    def test_game_config_custom(self):
        """Test AlphaZeroGameConfig with custom values."""
        config = AlphaZeroGameConfig(
            env_id="test",
            display_name="Test Game",
            board_width=8,
            board_height=8,
            num_actions=64,
            obs_size=192,
            hidden_size=512,
            network_type="resnet",
            num_res_blocks=6,
            num_filters=256,
            obs_channels=3,
        )

        assert config.hidden_size == 512
        assert config.network_type == "resnet"
        assert config.num_res_blocks == 6
        assert config.num_filters == 256
        assert config.obs_channels == 3

    def test_board_size_property(self):
        """Test board_size property calculation."""
        config = AlphaZeroGameConfig(
            env_id="test",
            display_name="Test Game",
            board_width=3,
            board_height=3,
            num_actions=9,
            obs_size=9,
            obs_channels=1,
        )

        assert config.board_size == 9

        config2 = AlphaZeroGameConfig(
            env_id="test2",
            display_name="Test Game 2",
            board_width=7,
            board_height=6,
            num_actions=7,
            obs_size=42,
            obs_channels=1,
        )

        assert config2.board_size == 42


class TestEnvironmentCatalog:
    """Tests for engine-owned environment discovery."""

    def test_list_environments_returns_all_registered_ids(self):
        environments = list_environments()

        assert isinstance(environments, list)
        assert "tictactoe" in environments
        assert "connect4" in environments
        assert "othello" in environments
        assert len(environments) >= 3

    def test_get_config_tictactoe(self):
        """Test that tictactoe config is returned correctly."""
        config = get_game_config("tictactoe")

        assert isinstance(config, AlphaZeroGameConfig)
        assert config.env_id == "tictactoe"
        assert config.display_name == "Tic-Tac-Toe"

    def test_get_config_connect4(self):
        """Test that connect4 config is returned correctly."""
        config = get_game_config("connect4")

        assert isinstance(config, AlphaZeroGameConfig)
        assert config.env_id == "connect4"
        assert config.display_name == "Connect 4"

    def test_get_config_othello(self):
        """Test that othello config is returned correctly."""
        config = get_game_config("othello")

        assert isinstance(config, AlphaZeroGameConfig)
        assert config.env_id == "othello"
        assert config.display_name == "Othello"

    def test_get_config_unknown_raises(self):
        """Test that unknown game raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_game_config("unknown_game")

        assert "unknown_game" in str(exc_info.value)
        assert "Available" in str(exc_info.value)

    def test_get_config_case_sensitive(self):
        """Test that game IDs are case sensitive."""
        # These should all raise ValueError
        with pytest.raises(ValueError):
            get_game_config("TicTacToe")

        with pytest.raises(ValueError):
            get_game_config("TICTACTOE")

        with pytest.raises(ValueError):
            get_game_config("Connect4")


class TestTicTacToeConfig:
    """Tests specific to TicTacToe configuration."""

    def test_tictactoe_dimensions(self):
        """Test TicTacToe board dimensions."""
        config = get_game_config("tictactoe")

        assert config.board_width == 3
        assert config.board_height == 3
        assert config.board_size == 9

    def test_tictactoe_actions(self):
        """Test TicTacToe action space."""
        config = get_game_config("tictactoe")

        assert config.num_actions == 9

    def test_tictactoe_observation(self):
        """Test TicTacToe observation structure."""
        config = get_game_config("tictactoe")

        assert config.obs_size == 18
        assert config.obs_channels == 2

    def test_tictactoe_network_type(self):
        """Test TicTacToe uses MLP network."""
        config = get_game_config("tictactoe")

        assert config.network_type == "mlp"
        assert config.hidden_size == 128


class TestConnect4Config:
    """Tests specific to Connect4 configuration."""

    def test_connect4_dimensions(self):
        """Test Connect4 board dimensions."""
        config = get_game_config("connect4")

        assert config.board_width == 7
        assert config.board_height == 6
        assert config.board_size == 42

    def test_connect4_actions(self):
        """Test Connect4 action space."""
        config = get_game_config("connect4")

        assert config.num_actions == 7

    def test_connect4_observation(self):
        """Test Connect4 observation structure."""
        config = get_game_config("connect4")

        assert config.obs_size == 84

    def test_connect4_network_type(self):
        """Test Connect4 uses ResNet."""
        config = get_game_config("connect4")

        assert config.network_type == "resnet"
        assert config.num_res_blocks == 4
        assert config.num_filters == 128
        assert config.obs_channels == 2
        # hidden_size is not carried for resnet games: the ResNet value head is
        # a fixed 256-unit layer, so the field only affects the MLP path.
        assert config.hidden_size == 128  # dataclass default, unused here


class TestOthelloConfig:
    """Tests specific to Othello configuration."""

    def test_othello_dimensions(self):
        """Test Othello board dimensions."""
        config = get_game_config("othello")

        assert config.board_width == 8
        assert config.board_height == 8
        assert config.board_size == 64

    def test_othello_actions(self):
        """Test Othello action space."""
        config = get_game_config("othello")

        assert config.num_actions == 65  # 64 positions + 1 pass

    def test_othello_observation(self):
        """Test Othello observation structure."""
        config = get_game_config("othello")

        assert config.obs_size == 128

    def test_othello_network_type(self):
        """Test Othello uses ResNet."""
        config = get_game_config("othello")

        assert config.network_type == "resnet"
        assert config.num_res_blocks == 6
        assert config.num_filters == 256
        assert config.obs_channels == 2
        # See the Connect4 equivalent: hidden_size is unused on the resnet path.
        assert config.hidden_size == 128  # dataclass default, unused here


class TestAlphaZeroRecipeRegistry:
    """Tests for the algorithm-owned environment recipes."""

    def test_registry_has_expected_games(self):
        """Test that all expected games are in the registry."""
        expected_games = ["tictactoe", "connect4", "othello"]

        compatible = list_compatible_environments()
        for game in expected_games:
            assert game in compatible
            assert isinstance(get_game_config(game), AlphaZeroGameConfig)

    def test_recipes_build_algorithm_configs(self):
        for env_id in list_compatible_environments():
            config = get_game_config(env_id)
            assert isinstance(config, AlphaZeroGameConfig)
            assert config.env_id == env_id

    def test_every_compatible_environment_has_a_safe_network_recipe(self):
        for env_id in list_compatible_environments():
            config = get_game_config(env_id)
            assert config.network_type in {"mlp", "resnet"}
            assert config.hidden_size > 0
            assert config.num_res_blocks > 0
            assert config.num_filters > 0


class TestGeneralsConfig:
    """Generals 8x8 — the only game with a non-trivial observation encoding."""

    def test_generals_dimensions(self):
        config = get_game_config("generals_8x8")

        assert config.board_width == 8
        assert config.board_height == 8
        assert config.num_actions == 257  # 64 tiles * 4 directions + wait
        assert config.obs_size == 640

    def test_generals_uses_player_relative_resnet(self):
        config = get_game_config("generals_8x8")

        assert config.network_type == "resnet"
        assert config.num_res_blocks == 6
        assert config.num_filters == 128
        # 10 generals_obs:v2 planes, encoded own/enemy relative to the player to
        # act and including the exact cap countdown.
        assert config.obs_channels == 10


class TestManifestIntegrity:
    """The manifest is the engine's word on game facts; check we honour it.

    These tests are what make the single-source scheme real on the Python side.
    The engine has matching assertions (engine-games), so a layout change that
    slips past one has to get past the other too.
    """

    def test_manifest_is_loadable(self):
        """Fail here, attributably, rather than at some other module's import.

        A missing or malformed manifest should fail at catalog import rather
        than surface later as a confusing learner error.
        """
        assert ENVIRONMENTS, "manifest produced no environments"

    def test_every_compatible_catalog_entry_builds_an_algorithm_config(self):
        for env_id in list_compatible_environments():
            config = get_game_config(env_id)
            assert config.env_id == env_id

    def test_observation_layout_invariants(self):
        """Mirrors the engine-side assertions in engine-games."""
        for env_id in list_compatible_environments():
            config = get_game_config(env_id)
            assert config.obs_channels > 0, f"{env_id}: obs_channels unset"
            assert config.obs_size == config.obs_channels * config.board_size, (
                f"{env_id}: observation must be exactly the spatial tensor"
            )

    def test_facts_are_not_restated_in_python(self):
        """The registry must be built from the manifest, not hardcoded.

        Guards against someone "fixing" a drift failure by pasting environment
        literals into the learner recipe, which would reintroduce the
        two-sources-of-truth problem the manifest removed.
        """
        import json
        from importlib.resources import files

        manifest = json.loads(
            files("trainer").joinpath("environment_manifest.json").read_text(encoding="utf-8")
        )
        by_id = {
            environment["metadata"]["id"]: environment for environment in manifest["environments"]
        }

        assert set(by_id) == set(ENVIRONMENTS)
        for env_id, environment in ENVIRONMENTS.items():
            source = by_id[env_id]
            board = source["metadata"]["board"]
            if board is None:
                assert environment.board is None
                continue
            parsed = environment.require_board()
            assert parsed.width == board["width"]
            assert parsed.height == board["height"]
            tensor = environment.capabilities.encoding.observation_tensor
            assert tensor is not None
            assert tensor.fixed_elements == get_game_config(env_id).obs_size

    def test_every_environment_has_the_alpha_zero_compatibility_report(self):
        for env_id in list_environments():
            report = get_environment(env_id).compatibility(ALGORITHM_ID)
            assert report.algorithm_id == ALGORITHM_ID
            assert report.env_id == env_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
