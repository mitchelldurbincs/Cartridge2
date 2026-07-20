"""Tests for resnet.py - Convolutional ResNet with configurable channels.

Tests cover:
- Forward pass shapes for the registered 2-channel games (Connect4, Othello)
- Arbitrary spatial channel counts (e.g. 9 channels for a future game)
- Observation reshape logic (board planes + derived current-player plane)
- ONNX export signature is preserved regardless of channel count
"""

import onnx
import pytest
import torch

from trainer.checkpoint import save_onnx_checkpoint
from trainer.game_config import GameConfig, get_config
from trainer.resnet import ConvPolicyValueNetwork


def make_config(input_channels: int, width: int = 5, height: int = 4) -> GameConfig:
    """Build a synthetic resnet GameConfig with N spatial channels.

    Observation layout: input_channels board planes, then legal mask
    (num_actions), then the 2-element player one-hot.
    """
    board_size = width * height
    num_actions = board_size
    return GameConfig(
        env_id="testgame",
        display_name="Test Game",
        board_width=width,
        board_height=height,
        num_actions=num_actions,
        obs_size=input_channels * board_size + num_actions + 2,
        legal_mask_offset=input_channels * board_size,
        network_type="resnet",
        num_res_blocks=1,
        num_filters=16,
        input_channels=input_channels,
    )


def make_obs(config: GameConfig, batch_size: int) -> torch.Tensor:
    """Random observation batch with a valid player one-hot."""
    obs = torch.rand(batch_size, config.obs_size)
    offset = config.player_indicator_offset
    obs[:, offset : offset + 2] = 0.0
    obs[: batch_size // 2, offset] = 1.0  # first player to move
    obs[batch_size // 2 :, offset + 1] = 1.0  # second player to move
    return obs


class TestRegisteredGames:
    """The 2-channel registered games keep their exact shapes."""

    @pytest.mark.parametrize("env_id", ["connect4", "othello"])
    def test_forward_shapes(self, env_id):
        config = get_config(env_id)
        network = ConvPolicyValueNetwork(config)
        obs = make_obs(config, batch_size=3)

        policy_logits, value = network(obs)

        assert policy_logits.shape == (3, config.num_actions)
        assert value.shape == (3, 1)
        assert torch.all(value >= -1.0) and torch.all(value <= 1.0)

    @pytest.mark.parametrize("env_id", ["connect4", "othello"])
    def test_channel_counts(self, env_id):
        config = get_config(env_id)
        network = ConvPolicyValueNetwork(config)

        assert network.board_planes == 2
        assert network.input_channels == 3  # +1 derived player plane
        assert network.initial_conv.in_channels == 3


class TestArbitraryChannels:
    """The network must work for any spatial channel count from the config."""

    @pytest.mark.parametrize("channels", [1, 2, 3, 9])
    def test_forward_shapes(self, channels):
        config = make_config(channels)
        network = ConvPolicyValueNetwork(config)
        obs = make_obs(config, batch_size=4)

        policy_logits, value = network(obs)

        assert network.board_planes == channels
        assert network.input_channels == channels + 1
        assert network.initial_conv.in_channels == channels + 1
        assert policy_logits.shape == (4, config.num_actions)
        assert value.shape == (4, 1)

    @pytest.mark.parametrize("channels", [2, 9])
    def test_reshape_observation_layout(self, channels):
        config = make_config(channels)
        network = ConvPolicyValueNetwork(config)
        obs = make_obs(config, batch_size=4)

        spatial = network._reshape_observation(obs)

        board_size = config.board_size
        assert spatial.shape == (
            4,
            channels + 1,
            config.board_height,
            config.board_width,
        )
        # Each board plane is the corresponding flat slice, reshaped
        for i in range(channels):
            expected = obs[:, i * board_size : (i + 1) * board_size].reshape(
                4, config.board_height, config.board_width
            )
            assert torch.equal(spatial[:, i], expected)
        # Derived player plane: +1 for first player, -1 for second
        assert torch.all(spatial[:2, channels] == 1.0)
        assert torch.all(spatial[2:, channels] == -1.0)

    @pytest.mark.parametrize("channels", [2, 9])
    def test_onnx_export_signature(self, channels, tmp_path):
        config = make_config(channels)
        network = ConvPolicyValueNetwork(config)

        checkpoint_path = save_onnx_checkpoint(
            network=network,
            obs_size=config.obs_size,
            step=1,
            model_dir=tmp_path,
            device=torch.device("cpu"),
        )

        model = onnx.load(str(checkpoint_path))
        graph_inputs = [i.name for i in model.graph.input]
        graph_outputs = [o.name for o in model.graph.output]
        assert graph_inputs == ["observation"]
        assert graph_outputs == ["policy_logits", "value"]
        # Input is the flat observation: (batch, obs_size)
        obs_dims = model.graph.input[0].type.tensor_type.shape.dim
        assert obs_dims[1].dim_value == config.obs_size
