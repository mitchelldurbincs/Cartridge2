"""Abstract base classes for storage backends.

These interfaces define the contract for storage backends, allowing
the trainer to work with SQLite locally or PostgreSQL/S3 in K8s.

Also home to the serialization/naming conventions shared by all ModelStore
backends: the checkpoint filename format and the best-model marker payload.
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Checkpoint naming convention shared by all ModelStore backends (the
# direct-filesystem path in trainer/checkpoint.py writes the same format).
CHECKPOINT_STEP_RE = re.compile(r"model_step_(\d+)\.onnx$")


def checkpoint_filename(step: int) -> str:
    """Canonical ONNX checkpoint filename for a training step."""
    return f"model_step_{step:06d}.onnx"


def parse_checkpoint_step(name: str) -> int | None:
    """Extract the training step from a checkpoint filename/key, or None."""
    match = CHECKPOINT_STEP_RE.search(name)
    return int(match.group(1)) if match else None


def encode_best_model_metadata(step: int) -> str:
    """Serialize the best-model marker payload shared by ModelStore backends."""
    return json.dumps({"step": step, "timestamp": time.time()})


def decode_best_model_metadata(data: str | bytes) -> int | None:
    """Parse a payload written by encode_best_model_metadata (None if malformed)."""
    try:
        return json.loads(data).get("step")
    except (ValueError, AttributeError, TypeError):
        return None


@dataclass
class GameMetadata:
    """Game metadata stored in the replay database by the actor.

    This makes the database self-describing, allowing the trainer to
    dynamically configure itself based on the game being trained.
    """

    env_id: str
    display_name: str
    board_width: int
    board_height: int
    num_actions: int
    obs_size: int
    legal_mask_offset: int
    player_count: int

    @property
    def board_size(self) -> int:
        """Total number of board cells."""
        return self.board_width * self.board_height

    @property
    def legal_mask_end(self) -> int:
        """End index of legal mask in observation."""
        return self.legal_mask_offset + self.num_actions


@dataclass
class Transition:
    """A single transition from the replay buffer."""

    id: str
    env_id: str
    episode_id: str
    step_number: int
    state: bytes
    action: bytes
    next_state: bytes
    observation: bytes
    next_observation: bytes
    reward: float
    done: bool
    timestamp: int
    policy_probs: bytes | None  # f32[num_actions] MCTS visit distribution
    mcts_value: float  # MCTS value estimate at this position
    game_outcome: float | None  # Final game outcome from this player's perspective


class ReplayBufferBase(ABC):
    """Abstract interface for replay buffer storage.

    Implementations must be thread-safe for concurrent reads.
    Write operations may have backend-specific concurrency guarantees.
    """

    @abstractmethod
    def close(self) -> None:
        """Close the connection and release resources."""
        pass

    def __enter__(self) -> "ReplayBufferBase":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @abstractmethod
    def count(self, env_id: str | None = None) -> int:
        """Get total number of transitions in the buffer.

        Args:
            env_id: Optional environment ID to filter by.
        """
        pass

    @abstractmethod
    def get_metadata(self, env_id: str | None = None) -> GameMetadata | None:
        """Get game metadata from the database.

        Args:
            env_id: Specific game to get metadata for. If None, returns
                    metadata for the first game found.
        """
        pass

    @abstractmethod
    def list_metadata(self) -> list[GameMetadata]:
        """List all game metadata in the database."""
        pass

    @abstractmethod
    def sample(self, batch_size: int, env_id: str | None = None) -> list[Transition]:
        """Sample random transitions for training.

        Args:
            batch_size: Number of transitions to sample.
            env_id: Optional environment ID to filter by.
        """
        pass

    @abstractmethod
    def clear_transitions(self) -> int:
        """Delete all transitions from the buffer.

        Preserves game_metadata. Used for synchronized AlphaZero training.

        Returns:
            Number of deleted transitions.
        """
        pass

    @abstractmethod
    def cleanup(self, window_size: int) -> int:
        """Delete old transitions to maintain a sliding window.

        Args:
            window_size: Maximum number of transitions to keep.

        Returns:
            Number of deleted transitions.
        """
        pass

    @abstractmethod
    def vacuum(self) -> None:
        """Reclaim storage space after deletions.

        Implementation-specific optimization (e.g., PostgreSQL VACUUM).
        May be a no-op for some backends.
        """
        pass

    def sample_batch_tensors(
        self, batch_size: int, num_actions: int, env_id: str | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Sample transitions and return as numpy arrays ready for training.

        Default implementation uses sample() and converts to tensors.
        Backends may override for efficiency.

        Args:
            batch_size: Number of transitions to sample.
            num_actions: Number of actions for the game.
            env_id: Optional environment ID to filter by.

        Returns:
            Tuple of (observations, policy_targets, value_targets) as numpy arrays,
            or None if not enough data.
        """
        transitions = self.sample(batch_size, env_id=env_id)
        if len(transitions) < batch_size:
            return None

        # Determine observation size from first transition
        first_obs = np.frombuffer(transitions[0].observation, dtype=np.float32)
        obs_size = len(first_obs)

        # Pre-allocate arrays
        observations = np.empty((batch_size, obs_size), dtype=np.float32)
        policy_targets = np.empty((batch_size, num_actions), dtype=np.float32)
        value_targets = np.empty(batch_size, dtype=np.float32)

        for i, t in enumerate(transitions):
            # Parse observation
            observations[i] = np.frombuffer(t.observation, dtype=np.float32)

            # Use MCTS policy distribution as target if available
            if t.policy_probs is not None and len(t.policy_probs) > 0:
                policy = np.frombuffer(t.policy_probs, dtype=np.float32)
                if len(policy) == num_actions:
                    policy_targets[i] = policy
                else:
                    # Fallback to one-hot if shape mismatch
                    logger.warning(
                        "Policy shape mismatch: got %d, expected %d. "
                        "Falling back to one-hot for transition %s (episode %s, step %d). "
                        "This degrades training quality - check actor/trainer config.",
                        len(policy),
                        num_actions,
                        t.id,
                        t.episode_id,
                        t.step_number,
                    )
                    policy_targets[i] = 0.0
                    action_idx = int.from_bytes(t.action, byteorder="little")
                    if action_idx >= num_actions:
                        raise ValueError(
                            f"Action index {action_idx} out of bounds for "
                            f"{num_actions} actions in transition {t.id}"
                        )
                    policy_targets[i, action_idx] = 1.0
            else:
                # Fallback to one-hot action if no MCTS data
                policy_targets[i] = 0.0
                action_idx = int.from_bytes(t.action, byteorder="little")
                if action_idx >= num_actions:
                    raise ValueError(
                        f"Action index {action_idx} out of bounds for "
                        f"{num_actions} actions in transition {t.id}"
                    )
                policy_targets[i, action_idx] = 1.0

            # Use game_outcome as value target (required for proper AlphaZero)
            if t.game_outcome is not None:
                value_targets[i] = t.game_outcome
            else:
                logger.warning(
                    "Missing game_outcome for transition %s (episode %s, step %d). "
                    "Using mcts_value (%.3f) as fallback - this degrades training quality.",
                    t.id,
                    t.episode_id,
                    t.step_number,
                    t.mcts_value,
                )
                value_targets[i] = t.mcts_value

        return observations, policy_targets, value_targets


@dataclass
class ModelInfo:
    """Metadata about a stored model."""

    path: str  # Backend-specific path/key (file path or S3 key)
    step: int
    timestamp: float
    is_latest: bool = False
    is_best: bool = False


class ModelStore(ABC):
    """Abstract interface for model storage.

    Handles ONNX model checkpoints and PyTorch training state.
    Supports atomic writes and change detection for hot-reload.
    """

    @abstractmethod
    def save_onnx(
        self,
        model_bytes: bytes,
        step: int,
        is_latest: bool = True,
    ) -> ModelInfo:
        """Save an ONNX model checkpoint.

        Args:
            model_bytes: Serialized ONNX model.
            step: Training step number.
            is_latest: Whether to also update the "latest" pointer.

        Returns:
            ModelInfo with the saved location.
        """
        pass

    @abstractmethod
    def save_pytorch(
        self,
        state_dict: dict,
        step: int,
    ) -> str:
        """Save PyTorch training state (model + optimizer + scheduler).

        Args:
            state_dict: Dictionary with model_state_dict, optimizer_state_dict, etc.
            step: Training step number.

        Returns:
            Path/key where the checkpoint was saved.
        """
        pass

    @abstractmethod
    def load_pytorch(self) -> tuple[dict, int] | None:
        """Load the latest PyTorch training state.

        Returns:
            Tuple of (state_dict, step) or None if no checkpoint exists.
        """
        pass

    @abstractmethod
    def get_latest_info(self) -> ModelInfo | None:
        """Get info about the latest model.

        Returns:
            ModelInfo for the latest model, or None if no model exists.
        """
        pass

    @abstractmethod
    def get_latest_version(self) -> int | None:
        """Get the version/step of the latest model.

        Used for change detection without downloading the full model.

        Returns:
            Step number of latest model, or None if no model exists.
        """
        pass

    @abstractmethod
    def load_latest_onnx(self) -> bytes | None:
        """Load the latest ONNX model bytes.

        Returns:
            Model bytes or None if no model exists.
        """
        pass

    @abstractmethod
    def list_checkpoints(self) -> list[ModelInfo]:
        """List all available model checkpoints.

        Returns:
            List of ModelInfo, sorted by step (oldest first).
        """
        pass

    def cleanup_old_checkpoints(self, max_keep: int) -> int:
        """Remove old checkpoints to save storage.

        Deletes oldest-first via _delete_checkpoint, never deleting the
        best model, until at most max_keep checkpoints remain.

        Args:
            max_keep: Maximum number of checkpoints to retain.

        Returns:
            Number of checkpoints deleted.
        """
        checkpoints = self.list_checkpoints()
        deleted = 0

        while len(checkpoints) > max_keep:
            old_checkpoint = checkpoints.pop(0)
            # Don't delete the best model
            if old_checkpoint.is_best:
                continue
            try:
                self._delete_checkpoint(old_checkpoint)
                logger.debug(f"Removed old checkpoint: {old_checkpoint.path}")
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to remove {old_checkpoint.path}: {e}")

        return deleted

    @abstractmethod
    def _delete_checkpoint(self, checkpoint: ModelInfo) -> None:
        """Delete a single checkpoint artifact (backend-specific)."""
        pass

    @abstractmethod
    def mark_as_best(self, step: int) -> None:
        """Mark a specific checkpoint as the "best" model.

        Used by gatekeeper evaluation to track the best performing model.

        Args:
            step: Step number of the checkpoint to mark as best.
        """
        pass

    @abstractmethod
    def get_best_info(self) -> ModelInfo | None:
        """Get info about the best model.

        Returns:
            ModelInfo for the best model, or None if not set.
        """
        pass
