"""ONNX neural network policy implementation."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    from ..game_config import GameConfig
    from ..games import GameState
    from ..storage import GameMetadata


class OnnxPolicy:
    """Policy using an ONNX neural network model.

    Uses game configuration (from database or fallback) to correctly parse
    observations and policy outputs for different games.
    """

    def __init__(self, model_path: str, temperature: float = 0.0):
        """
        Args:
            model_path: Path to ONNX model file.
            temperature: Sampling temperature. 0 = greedy (argmax).
        """
        self.model_path = model_path
        self.temperature = temperature

        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.policy_output = self.session.get_outputs()[0].name
        self.value_output = self.session.get_outputs()[1].name

    @property
    def name(self) -> str:
        return f"ONNX({Path(self.model_path).name})"

    def select_action(
        self, state: "GameState", config: "GameConfig | GameMetadata"
    ) -> int:
        """Select an action using the neural network policy.

        Args:
            state: Current game state.
            config: Game configuration (from database metadata or fallback).

        Returns:
            Selected action index.
        """
        legal = state.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")

        # Get observation using config for correct encoding
        obs = state.to_observation(config)
        obs_batch = obs.reshape(1, -1)

        # Run inference
        outputs = self.session.run(
            [self.policy_output, self.value_output],
            {self.input_name: obs_batch},
        )
        policy_logits = outputs[0][0]  # Shape: (num_actions,)

        # Mask illegal moves
        legal_mask = state.legal_moves_mask()
        masked_logits = np.where(
            np.array(legal_mask) == 1.0,
            policy_logits,
            -np.inf,
        )

        if self.temperature == 0.0:
            # Greedy selection
            return int(np.argmax(masked_logits))
        else:
            # Sample with temperature
            logits = masked_logits / self.temperature
            logits = logits - np.max(logits)  # Numerical stability
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
            return int(np.random.choice(config.num_actions, p=probs))
