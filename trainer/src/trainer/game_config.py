"""Game configuration registry for the trainer.

Game *facts* — board dimensions, action count, observation layout — come from
``game_metadata.json``, which is generated from the Rust engine by
``make game-manifest``. The engine is the single source of truth for them; this
module must never restate them, or the two drift and the trainer silently
mistrains against an observation layout the actor is not producing.
(``cargo test`` fails if the committed manifest falls out of date.)

Training *hyperparameters* — which network architecture to build for a game —
are a trainer-side choice with no engine counterpart, and live in
``_TRAINING_OVERRIDES`` below.

Keeping those two categories apart is the point. They used to be mixed in one
hand-maintained table, and a DB-metadata path that rebuilt only the fact half
silently reset the rest to defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from importlib.resources import files
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass
class GameConfig:
    """Game configuration for the trainer."""

    # Game identity
    env_id: str
    display_name: str

    # Board dimensions
    board_width: int
    board_height: int

    # Neural network dimensions
    num_actions: int
    obs_size: int
    legal_mask_offset: int

    # Network architecture hints
    hidden_size: int = 128

    # Network architecture selection
    network_type: str = "mlp"  # "mlp" or "resnet"

    # CNN-specific settings (used when network_type="resnet")
    num_res_blocks: int = 4  # Number of residual blocks
    num_filters: int = 128  # Filters per conv layer

    # --- Observation-layout facts (engine-owned, from the manifest) ---
    # Number of spatial board planes at the front of the observation
    # (2 for one plane per player's pieces, 9 for the Generals encoding).
    obs_channels: int = 2
    # True when the obs board planes are already encoded from the
    # current player's perspective (own/enemy), in which case the network
    # must NOT receive the current-player indicator: with a seat-relative
    # board it is a pure side channel, and a systematically advantaged seat
    # lets the value head collapse into a seat detector instead of learning
    # positions (observed with generals: P2 wins ~all alternating-turn
    # self-play games at adjudication).
    player_relative_obs: bool = False

    @property
    def board_size(self) -> int:
        """Total number of board cells."""
        return self.board_width * self.board_height

    @property
    def legal_mask_bits(self) -> int:
        """Bitmask for extracting legal moves from info bits."""
        return (1 << self.num_actions) - 1

    @property
    def legal_mask_end(self) -> int:
        """End index of legal mask in observation."""
        return self.legal_mask_offset + self.num_actions

    @property
    def player_indicator_offset(self) -> int:
        """Start index of the 2-element player one-hot in observation."""
        return self.legal_mask_end

    def extract_legal_mask(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract the legal move mask from a batch of observations.

        Args:
            obs: Observation tensor of shape (batch, obs_size).

        Returns:
            Legal mask tensor of shape (batch, num_actions).
        """
        return obs[:, self.legal_mask_offset : self.legal_mask_end]  # type: ignore[index]

    def extract_player_indicator(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract the 2-element player one-hot from a batch of observations.

        Args:
            obs: Observation tensor of shape (batch, obs_size).

        Returns:
            Player indicator tensor of shape (batch, 2).
        """
        offset = self.player_indicator_offset
        return obs[:, offset : offset + 2]  # type: ignore[index]


MANIFEST_FILENAME = "game_metadata.json"
REGENERATE_HINT = (
    f"Run `make game-manifest` to regenerate {MANIFEST_FILENAME} from the Rust engine."
)

# Fields the engine owns. Deliberately an explicit whitelist rather than
# `set(manifest) & {f.name for f in fields(GameConfig)}`: with an intersection,
# adding a field to the Rust GameMetadata that happens to share a name with a
# trainer hyperparameter would silently let the engine start dictating it.
_ENGINE_FACT_FIELDS = (
    "env_id",
    "display_name",
    "board_width",
    "board_height",
    "num_actions",
    "obs_size",
    "legal_mask_offset",
    "obs_channels",
    "player_relative_obs",
)

# Trainer-side network architecture, keyed by env_id. No engine counterpart:
# which network to train is our choice, not a property of the game.
#
# `hidden_size` only affects the MLP path — the ResNet value head is a fixed
# 256-unit layer (see resnet.py), so it is omitted for resnet games rather than
# carrying a number that does nothing.
_TRAINING_OVERRIDES: dict[str, dict[str, Any]] = {
    "tictactoe": {"network_type": "mlp", "hidden_size": 128},
    "connect4": {"network_type": "resnet", "num_res_blocks": 4, "num_filters": 128},
    "othello": {"network_type": "resnet", "num_res_blocks": 6, "num_filters": 256},
    "generals_8x8": {"network_type": "resnet", "num_res_blocks": 6, "num_filters": 128},
}


def _load_manifest() -> list[dict[str, Any]]:
    """Read the engine-generated game manifest shipped inside this package."""
    resource = files("trainer").joinpath(MANIFEST_FILENAME)
    try:
        raw = resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            f"Game metadata manifest is missing ({resource}). {REGENERATE_HINT}"
        ) from exc

    try:
        return json.loads(raw)["games"]
    except (ValueError, KeyError) as exc:
        raise RuntimeError(
            f"Game metadata manifest at {resource} is malformed. {REGENERATE_HINT}"
        ) from exc


def _build_registry() -> dict[str, GameConfig]:
    """Combine engine-owned facts with trainer-owned network settings."""
    valid_fields = {f.name for f in fields(GameConfig)}
    registry: dict[str, GameConfig] = {}

    for game in _load_manifest():
        env_id = game["env_id"]
        try:
            overrides = _TRAINING_OVERRIDES[env_id]
        except KeyError as exc:
            # Failing loudly here is the point. Defaulting instead would give a
            # newly-added spatial game `network_type="mlp"` and train a dense
            # net on a board — a regression that produces no error, only a
            # model that never gets good.
            raise RuntimeError(
                f"Game '{env_id}' is in the engine manifest but has no entry in "
                f"_TRAINING_OVERRIDES. Add one in game_config.py choosing its "
                f"network architecture."
            ) from exc

        unknown = set(overrides) - valid_fields
        if unknown:
            raise RuntimeError(
                f"_TRAINING_OVERRIDES['{env_id}'] has unknown field(s): {sorted(unknown)}"
            )

        facts = {k: game[k] for k in _ENGINE_FACT_FIELDS}
        registry[env_id] = GameConfig(**facts, **overrides)

    return registry


# Game configuration registry: engine facts from the manifest, network
# architecture from _TRAINING_OVERRIDES.
GAME_CONFIGS: dict[str, GameConfig] = _build_registry()


def get_config(env_id: str) -> GameConfig:
    """Get the game configuration for a given environment ID.

    Args:
        env_id: Environment identifier (e.g., "tictactoe", "connect4")

    Returns:
        GameConfig for the specified game.

    Raises:
        ValueError: If the game is not registered.
    """
    if env_id not in GAME_CONFIGS:
        available = ", ".join(GAME_CONFIGS.keys())
        raise ValueError(f"Unknown game: {env_id}. Available games: {available}")
    return GAME_CONFIGS[env_id]


def list_games() -> list[str]:
    """List all registered game IDs."""
    return list(GAME_CONFIGS.keys())


if __name__ == "__main__":
    # Simple test
    for env_id in list_games():
        config = get_config(env_id)
        print(f"{config.display_name}:")
        print(f"  Board: {config.board_width}x{config.board_height}")
        print(f"  Actions: {config.num_actions}")
        print(f"  Obs size: {config.obs_size}")
        print(f"  Legal mask offset: {config.legal_mask_offset}")
        print()
