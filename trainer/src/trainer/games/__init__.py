"""Game state implementations for evaluation.

This module provides pure Python implementations of game states used
for model evaluation. These mirror the Rust implementations in the
engine crates but are needed for Python-only evaluation.
"""

from enum import IntEnum
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from ..game_config import GameConfig
    from ..storage import GameMetadata


class Player(IntEnum):
    """Player identifiers matching game implementations."""

    FIRST = 1  # First player (X in TicTacToe, Red in Connect4)
    SECOND = 2  # Second player (O in TicTacToe, Yellow in Connect4)


class Cell(IntEnum):
    """Cell states."""

    EMPTY = 0
    FIRST = 1  # First player's piece
    SECOND = 2  # Second player's piece


class GameState(Protocol):
    """Protocol for game state implementations."""

    @property
    def done(self) -> bool:
        """Return True if game is over."""
        ...

    @property
    def winner(self) -> int | None:
        """Return winner (1=first player, 2=second player, None=draw/ongoing)."""
        ...

    @property
    def current_player(self) -> Player:
        """Return current player to move."""
        ...

    def legal_moves(self) -> list[int]:
        """Return list of legal move indices."""
        ...

    def legal_moves_mask(self) -> list[float]:
        """Return mask where 1.0 = legal, 0.0 = illegal."""
        ...

    def make_move(self, pos: int) -> None:
        """Make a move at the given position."""
        ...

    def to_observation(self, config: "GameConfig | GameMetadata") -> np.ndarray:
        """Convert to neural network observation format."""
        ...

    def display(self) -> str:
        """Return a string representation for debugging."""
        ...

    def copy(self) -> "GameState":
        """Return a copy of the state."""
        ...


def create_game_state(env_id: str) -> GameState:
    """Create a new game state for the given environment.

    Args:
        env_id: Environment ID (e.g., "tictactoe", "connect4").

    Returns:
        New game state instance.

    Raises:
        ValueError: If env_id is not supported.
    """
    # Import here to avoid circular imports
    from .connect4 import Connect4State
    from .generals import GeneralsState
    from .tictactoe import TicTacToeState

    if env_id == "tictactoe":
        return TicTacToeState.new()
    elif env_id == "connect4":
        return Connect4State.new()
    elif env_id == "generals_8x8":
        return GeneralsState.new()
    else:
        raise ValueError(f"Unsupported game: {env_id}")


__all__ = [
    "Cell",
    "GameState",
    "Player",
    "create_game_state",
]
