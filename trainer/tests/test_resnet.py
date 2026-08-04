"""Tests for resnet.py - Convolutional ResNet with configurable channels.

Tests cover:
- Forward pass shapes for the registered 2-channel games (Connect4, Othello)
- Arbitrary spatial channel counts (e.g. 9 channels for a future game)
- Observation reshape logic for exact environment-declared board planes
- ONNX export signature is preserved regardless of channel count
"""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from trainer.algorithms.alphazero_board_v1 import (
    AlphaZeroGameConfig,
    get_game_config,
    policy_value_artifact_contract,
)
from trainer.checkpoint import export_onnx_artifact
from trainer.environment_catalog import get_environment
from trainer.network import create_network
from trainer.resnet import ConvPolicyValueNetwork


def make_config(obs_channels: int, width: int = 5, height: int = 4) -> AlphaZeroGameConfig:
    """Build a synthetic AlphaZero game config with N spatial channels.

    Observation layout: exactly ``obs_channels`` board planes.
    """
    board_size = width * height
    num_actions = board_size
    return AlphaZeroGameConfig(
        env_id="testgame",
        display_name="Test Game",
        board_width=width,
        board_height=height,
        num_actions=num_actions,
        obs_size=obs_channels * board_size,
        network_type="resnet",
        num_res_blocks=1,
        num_filters=16,
        obs_channels=obs_channels,
    )


def make_obs(config: AlphaZeroGameConfig, batch_size: int) -> torch.Tensor:
    """Random observation batch with the declared tensor shape."""
    return torch.rand(batch_size, config.obs_size)


class TestRegisteredGames:
    """The 2-channel registered games keep their exact shapes."""

    @pytest.mark.parametrize("env_id", ["connect4", "othello"])
    def test_forward_shapes(self, env_id):
        config = get_game_config(env_id)
        network = ConvPolicyValueNetwork(config)
        obs = make_obs(config, batch_size=3)

        policy_logits, value = network(obs)

        assert policy_logits.shape == (3, config.num_actions)
        assert value.shape == (3, 1)
        assert torch.all(value >= -1.0) and torch.all(value <= 1.0)

    @pytest.mark.parametrize("env_id", ["connect4", "othello"])
    def test_channel_counts(self, env_id):
        config = get_game_config(env_id)
        network = ConvPolicyValueNetwork(config)

        assert network.board_planes == 2
        assert network.input_channels == 2
        assert network.initial_conv.in_channels == 2


class TestArbitraryChannels:
    """The network must work for any spatial channel count from the config."""

    @pytest.mark.parametrize("channels", [1, 2, 3, 9])
    def test_forward_shapes(self, channels):
        config = make_config(channels)
        network = ConvPolicyValueNetwork(config)
        obs = make_obs(config, batch_size=4)

        policy_logits, value = network(obs)

        assert network.board_planes == channels
        assert network.input_channels == channels
        assert network.initial_conv.in_channels == channels
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
            channels,
            config.board_height,
            config.board_width,
        )
        # Each board plane is the corresponding flat slice, reshaped
        for i in range(channels):
            expected = obs[:, i * board_size : (i + 1) * board_size].reshape(
                4, config.board_height, config.board_width
            )
            assert torch.equal(spatial[:, i], expected)

    @pytest.mark.parametrize("channels", [2, 9])
    def test_onnx_export_signature(self, channels, tmp_path):
        config = make_config(channels)
        network = ConvPolicyValueNetwork(config)
        network.eval()

        checkpoint_path = export_onnx_artifact(
            network=network,
            output_path=tmp_path / "model.onnx",
            device=torch.device("cpu"),
            artifact_contract=policy_value_artifact_contract(
                algorithm_id="alphazero_board_v1",
                env_id=config.env_id,
                env_contract_version=1,
                model_artifact_schema_version=1,
                model_contract="onnx_policy_value_v1",
                obs_size=config.obs_size,
                num_actions=config.num_actions,
            ),
        )

        model = onnx.load(str(checkpoint_path))
        graph_inputs = [i.name for i in model.graph.input]
        graph_outputs = [o.name for o in model.graph.output]
        assert graph_inputs == ["observation"]
        assert graph_outputs == ["policy_logits", "value"]
        tensors = [*model.graph.input, *model.graph.output]
        assert [
            [dim.dim_param or dim.dim_value for dim in item.type.tensor_type.shape.dim]
            for item in tensors
        ] == [
            ["batch_size", config.obs_size],
            ["batch_size", config.num_actions],
            ["batch_size", 1],
        ]

        observations = make_obs(config, batch_size=3)
        with torch.no_grad():
            expected_policy, expected_value = network(observations)
        session = ort.InferenceSession(str(checkpoint_path), providers=["CPUExecutionProvider"])
        actual_policy, actual_value = session.run(
            ["policy_logits", "value"],
            {"observation": observations.numpy()},
        )
        np.testing.assert_allclose(actual_policy, expected_policy.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(actual_value, expected_value.numpy(), rtol=1e-4, atol=1e-5)


def test_generals_v3_deep_resnet_export_passes_runtime_equivalence(tmp_path):
    environment = get_environment("generals_8x8")
    config = get_game_config(environment.env_id)
    assert environment.contract_version == 3
    assert config.obs_size == 640
    assert config.num_actions == 257
    assert config.num_res_blocks == 6
    assert config.num_filters == 128
    with torch.random.fork_rng():
        torch.manual_seed(1)
        network = create_network(environment.env_id, config)

    checkpoint_path = export_onnx_artifact(
        network=network,
        output_path=tmp_path / "generals-v3.onnx",
        device=torch.device("cpu"),
        artifact_contract=policy_value_artifact_contract(
            algorithm_id="alphazero_board_v1",
            env_id=environment.env_id,
            env_contract_version=environment.contract_version,
            model_artifact_schema_version=1,
            model_contract="onnx_policy_value_v1",
            obs_size=config.obs_size,
            num_actions=config.num_actions,
        ),
    )

    model = onnx.load(checkpoint_path, load_external_data=False)
    tensors = [*model.graph.input, *model.graph.output]
    assert [
        [dim.dim_param or dim.dim_value for dim in item.type.tensor_type.shape.dim]
        for item in tensors
    ] == [
        ["batch_size", 640],
        ["batch_size", 257],
        ["batch_size", 1],
    ]
