"""Tests for game_config.py - Game configuration registry.

Tests cover:
- GameConfig dataclass initialization and properties
- Game registry functionality
- Configuration validation
- Specific game configurations (TicTacToe, Connect4, Othello)
"""

import pytest

from trainer.game_config import GAME_CONFIGS, GameConfig, get_config, list_games


class TestGameConfig:
    """Tests for the GameConfig dataclass."""

    def test_game_config_defaults(self):
        """Test GameConfig default values."""
        config = GameConfig(
            env_id="test",
            display_name="Test Game",
            board_width=5,
            board_height=5,
            num_actions=25,
            obs_size=50,
            legal_mask_offset=25,
        )

        assert config.env_id == "test"
        assert config.display_name == "Test Game"
        assert config.board_width == 5
        assert config.board_height == 5
        assert config.num_actions == 25
        assert config.obs_size == 50
        assert config.legal_mask_offset == 25
        # Check defaults
        assert config.hidden_size == 128
        assert config.network_type == "mlp"
        assert config.num_res_blocks == 4
        assert config.num_filters == 128
        assert config.input_channels == 2

    def test_game_config_custom(self):
        """Test GameConfig with custom values."""
        config = GameConfig(
            env_id="test",
            display_name="Test Game",
            board_width=8,
            board_height=8,
            num_actions=64,
            obs_size=200,
            legal_mask_offset=128,
            hidden_size=512,
            network_type="resnet",
            num_res_blocks=6,
            num_filters=256,
            input_channels=3,
        )

        assert config.hidden_size == 512
        assert config.network_type == "resnet"
        assert config.num_res_blocks == 6
        assert config.num_filters == 256
        assert config.input_channels == 3

    def test_board_size_property(self):
        """Test board_size property calculation."""
        config = GameConfig(
            env_id="test",
            display_name="Test Game",
            board_width=3,
            board_height=3,
            num_actions=9,
            obs_size=20,
            legal_mask_offset=10,
        )

        assert config.board_size == 9

        config2 = GameConfig(
            env_id="test2",
            display_name="Test Game 2",
            board_width=7,
            board_height=6,
            num_actions=7,
            obs_size=100,
            legal_mask_offset=50,
        )

        assert config2.board_size == 42

    def test_legal_mask_bits_property(self):
        """Test legal_mask_bits property calculation."""
        config = GameConfig(
            env_id="test",
            display_name="Test Game",
            board_width=3,
            board_height=3,
            num_actions=9,
            obs_size=20,
            legal_mask_offset=10,
        )

        # (1 << 9) - 1 = 0b111111111 = 511
        assert config.legal_mask_bits == 511

        config2 = GameConfig(
            env_id="test2",
            display_name="Test Game 2",
            board_width=7,
            board_height=6,
            num_actions=7,
            obs_size=100,
            legal_mask_offset=50,
        )

        # (1 << 7) - 1 = 0b1111111 = 127
        assert config2.legal_mask_bits == 127

    def test_legal_mask_end_property(self):
        """Test legal_mask_end property calculation."""
        config = GameConfig(
            env_id="test",
            display_name="Test Game",
            board_width=3,
            board_height=3,
            num_actions=9,
            obs_size=20,
            legal_mask_offset=10,
        )

        # 10 + 9 = 19
        assert config.legal_mask_end == 19


class TestGameRegistry:
    """Tests for the game configuration registry."""

    def test_list_games_returns_all_games(self):
        """Test that list_games returns all registered game IDs."""
        games = list_games()

        assert isinstance(games, list)
        assert "tictactoe" in games
        assert "connect4" in games
        assert "othello" in games
        assert len(games) >= 3

    def test_get_config_tictactoe(self):
        """Test that tictactoe config is returned correctly."""
        config = get_config("tictactoe")

        assert isinstance(config, GameConfig)
        assert config.env_id == "tictactoe"
        assert config.display_name == "Tic-Tac-Toe"

    def test_get_config_connect4(self):
        """Test that connect4 config is returned correctly."""
        config = get_config("connect4")

        assert isinstance(config, GameConfig)
        assert config.env_id == "connect4"
        assert config.display_name == "Connect 4"

    def test_get_config_othello(self):
        """Test that othello config is returned correctly."""
        config = get_config("othello")

        assert isinstance(config, GameConfig)
        assert config.env_id == "othello"
        assert config.display_name == "Othello"

    def test_get_config_unknown_raises(self):
        """Test that unknown game raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_config("unknown_game")

        assert "unknown_game" in str(exc_info.value)
        assert "Available games" in str(exc_info.value)

    def test_get_config_case_sensitive(self):
        """Test that game IDs are case sensitive."""
        # These should all raise ValueError
        with pytest.raises(ValueError):
            get_config("TicTacToe")

        with pytest.raises(ValueError):
            get_config("TICTACTOE")

        with pytest.raises(ValueError):
            get_config("Connect4")


class TestTicTacToeConfig:
    """Tests specific to TicTacToe configuration."""

    def test_tictactoe_dimensions(self):
        """Test TicTacToe board dimensions."""
        config = get_config("tictactoe")

        assert config.board_width == 3
        assert config.board_height == 3
        assert config.board_size == 9

    def test_tictactoe_actions(self):
        """Test TicTacToe action space."""
        config = get_config("tictactoe")

        assert config.num_actions == 9
        assert config.legal_mask_bits == 511  # 0b111111111

    def test_tictactoe_observation(self):
        """Test TicTacToe observation structure."""
        config = get_config("tictactoe")

        # obs_size = 18 (board) + 9 (legal mask) + 2 (player) = 29
        assert config.obs_size == 29
        assert config.legal_mask_offset == 18
        assert config.legal_mask_end == 27  # 18 + 9

    def test_tictactoe_network_type(self):
        """Test TicTacToe uses MLP network."""
        config = get_config("tictactoe")

        assert config.network_type == "mlp"
        assert config.hidden_size == 128


class TestConnect4Config:
    """Tests specific to Connect4 configuration."""

    def test_connect4_dimensions(self):
        """Test Connect4 board dimensions."""
        config = get_config("connect4")

        assert config.board_width == 7
        assert config.board_height == 6
        assert config.board_size == 42

    def test_connect4_actions(self):
        """Test Connect4 action space."""
        config = get_config("connect4")

        assert config.num_actions == 7
        assert config.legal_mask_bits == 127  # 0b1111111

    def test_connect4_observation(self):
        """Test Connect4 observation structure."""
        config = get_config("connect4")

        # obs_size = 42 (Red) + 42 (Yellow) + 7 (legal) + 2 (player) = 93
        assert config.obs_size == 93
        assert config.legal_mask_offset == 84  # After both board views
        assert config.legal_mask_end == 91  # 84 + 7

    def test_connect4_network_type(self):
        """Test Connect4 uses ResNet."""
        config = get_config("connect4")

        assert config.network_type == "resnet"
        assert config.hidden_size == 512
        assert config.num_res_blocks == 4
        assert config.num_filters == 128
        assert config.input_channels == 2


class TestOthelloConfig:
    """Tests specific to Othello configuration."""

    def test_othello_dimensions(self):
        """Test Othello board dimensions."""
        config = get_config("othello")

        assert config.board_width == 8
        assert config.board_height == 8
        assert config.board_size == 64

    def test_othello_actions(self):
        """Test Othello action space."""
        config = get_config("othello")

        assert config.num_actions == 65  # 64 positions + 1 pass
        assert config.legal_mask_bits == (1 << 65) - 1  # Very large number

    def test_othello_observation(self):
        """Test Othello observation structure."""
        config = get_config("othello")

        # obs_size = 128 (board: 64*2) + 65 (legal) + 2 (player) = 195
        assert config.obs_size == 195
        assert config.legal_mask_offset == 128  # After board encoding
        assert config.legal_mask_end == 193  # 128 + 65

    def test_othello_network_type(self):
        """Test Othello uses ResNet."""
        config = get_config("othello")

        assert config.network_type == "resnet"
        assert config.hidden_size == 512
        assert config.num_res_blocks == 6
        assert config.num_filters == 256
        assert config.input_channels == 2


class TestGameConfigsRegistry:
    """Tests for the GAME_CONFIGS dictionary directly."""

    def test_registry_has_expected_games(self):
        """Test that all expected games are in the registry."""
        expected_games = ["tictactoe", "connect4", "othello"]

        for game in expected_games:
            assert game in GAME_CONFIGS
            assert isinstance(GAME_CONFIGS[game], GameConfig)

    def test_registry_values_are_game_configs(self):
        """Test that all registry values are GameConfig instances."""
        for env_id, config in GAME_CONFIGS.items():
            assert isinstance(config, GameConfig)
            assert config.env_id == env_id

    def test_registry_is_dict(self):
        """Test that GAME_CONFIGS is a dictionary."""
        assert isinstance(GAME_CONFIGS, dict)
        assert len(GAME_CONFIGS) >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
