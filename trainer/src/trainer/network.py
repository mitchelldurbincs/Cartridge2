"""AlphaZero-style neural network for game agents.

This module provides a simple policy-value network suitable for board games.
The architecture follows AlphaZero conventions:
- Shared representation layers
- Policy head (action probabilities)
- Value head (expected outcome)

Input:
    - Observation vector (size depends on game)

Output:
    - policy: logits (one per possible action)
    - value: scalar in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game_config import get_config


class BasePolicyValueNetwork(nn.Module):
    """Base class for policy-value networks.

    Provides the shared `predict()` method that applies legal move masking
    and softmax to convert logits to probabilities.

    Subclasses must implement `forward()` returning (policy_logits, value).
    """

    def predict(
        self, x: torch.Tensor, legal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy probabilities and value.

        Args:
            x: Observation tensor of shape (batch, obs_size)
            legal_mask: Optional mask of shape (batch, action_size)
                        where 1.0 = legal, 0.0 = illegal

        Returns:
            Tuple of (policy_probs, value):
                - policy_probs: Shape (batch, action_size) - probabilities summing to 1
                - value: Shape (batch, 1) - value in [-1, 1]
        """
        policy_logits, value = self.forward(x)

        # Apply legal move mask if provided
        if legal_mask is not None:
            # Set illegal moves to very negative value before softmax
            policy_logits = policy_logits.masked_fill(legal_mask == 0, float("-inf"))

        policy_probs = F.softmax(policy_logits, dim=-1)

        return policy_probs, value


class PolicyValueNetwork(BasePolicyValueNetwork):
    """Policy-value network for board games.

    Architecture:
        Input (obs_size) -> FC(hidden) -> ReLU -> FC(hidden) -> ReLU -> FC(hidden/2) -> ReLU
            -> Policy head: FC(num_actions)
            -> Value head: FC(hidden/4) -> ReLU -> FC(1) -> Tanh
    """

    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        self.obs_size = obs_size
        self.action_size = action_size

        # Shared layers (3 layers for more representational capacity)
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)

        # Policy head
        self.policy_fc = nn.Linear(hidden_size // 2, action_size)

        # Value head
        self.value_fc1 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.value_fc2 = nn.Linear(hidden_size // 4, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Observation tensor of shape (batch, obs_size)

        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: Shape (batch, action_size) - raw logits
                - value: Shape (batch, 1) - value in [-1, 1]
        """
        # Shared representation
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        # Policy head (raw logits, softmax applied during loss)
        policy_logits = self.policy_fc(h)

        # Value head
        v = F.relu(self.value_fc1(h))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


class AlphaZeroLoss:
    """Combined loss function for AlphaZero training.

    Loss = (z - v)^2 - pi^T * log(p)

    Where:
        - z: Target value (game outcome or MCTS value estimate)
        - v: Predicted value
        - pi: Target policy (MCTS visit count distribution)
        - p: Predicted policy (log softmax of logits)
    """

    def __init__(self, value_weight: float = 1.0, policy_weight: float = 1.0):
        self.value_weight = value_weight
        self.policy_weight = policy_weight

    def __call__(
        self,
        policy_logits: torch.Tensor,
        value: torch.Tensor,
        target_policy: torch.Tensor,
        target_values: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute AlphaZero loss.

        Args:
            policy_logits: Predicted policy logits (batch, action_size)
            value: Predicted value (batch, 1)
            target_policy: Target policy distribution from MCTS (batch, action_size)
            target_values: Target values (batch,)
            legal_mask: Optional legal move mask (batch, action_size)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Value loss (MSE)
        value_loss = F.mse_loss(value.squeeze(-1), target_values)

        # Policy loss: cross-entropy with soft targets
        # L_policy = -sum(pi * log(p)) where pi is MCTS distribution
        if legal_mask is not None:
            # Detect samples with no legal moves (all-zero mask).
            # This shouldn't happen in normal training data (observations are from
            # BEFORE the action was taken), but we handle it defensively to prevent
            # NaN from log_softmax(all -inf) = log(0/0) = NaN.
            has_legal_moves = legal_mask.sum(dim=-1, keepdim=True) > 0  # (batch, 1)

            # Only mask where we have legal moves; otherwise keep original logits
            # This prevents all-inf inputs to softmax which cause NaN
            masked_logits = policy_logits.masked_fill(legal_mask == 0, float("-inf"))
            policy_logits = torch.where(has_legal_moves, masked_logits, policy_logits)

        log_probs = F.log_softmax(policy_logits, dim=-1)

        # Cross-entropy with soft targets: -sum(target * log_pred)
        # Clamp log_probs to avoid NaN from 0 * -inf when target is 0 on masked positions
        log_probs_clamped = torch.clamp(log_probs, min=-100.0)
        policy_loss = -torch.sum(target_policy * log_probs_clamped, dim=-1).mean()

        # Combined loss
        total_loss = self.value_weight * value_loss + self.policy_weight * policy_loss

        metrics = {
            "loss/total": total_loss.item(),
            "loss/value": value_loss.item(),
            "loss/policy": policy_loss.item(),
        }

        return total_loss, metrics


def create_network(env_id: str = "tictactoe") -> nn.Module:
    """Factory function to create a network for the specified environment.

    Automatically selects the appropriate architecture based on the game's
    network_type configuration:
    - "mlp": Simple feedforward network (default, good for small games)
    - "resnet": Convolutional ResNet (for spatially-structured games)

    Args:
        env_id: Environment identifier (e.g., "tictactoe", "connect4")

    Returns:
        Neural network configured for the specified game.

    Raises:
        ValueError: If the game is not registered or network_type is invalid.
    """
    config = get_config(env_id)

    if config.network_type == "resnet":
        from .resnet import ConvPolicyValueNetwork

        return ConvPolicyValueNetwork(config)
    elif config.network_type == "mlp":
        return PolicyValueNetwork(
            obs_size=config.obs_size,
            action_size=config.num_actions,
            hidden_size=config.hidden_size,
        )
    else:
        raise ValueError(
            f"Unknown network_type '{config.network_type}' for {env_id}. "
            f"Valid options: 'mlp', 'resnet'"
        )
