"""Installed Python implementations for engine-declared algorithms."""

from __future__ import annotations

from ..environment_catalog import ALGORITHMS
from .alphazero_board_v1 import ALGORITHM_ID, AlphaZeroBoardV1
from .base import Algorithm
from .dqn_v1 import ALGORITHM_ID as DQN_ALGORITHM_ID
from .dqn_v1 import DqnV1

_ALGORITHMS: dict[str, Algorithm] = {
    ALGORITHM_ID: AlphaZeroBoardV1(),
    DQN_ALGORITHM_ID: DqnV1(),
}

_declared = set(ALGORITHMS)
_implemented = set(_ALGORITHMS)
if _declared != _implemented:
    missing = ", ".join(sorted(_declared - _implemented)) or "(none)"
    extra = ", ".join(sorted(_implemented - _declared)) or "(none)"
    raise RuntimeError(
        "Python algorithm registry does not match the engine catalog: "
        f"missing implementations: {missing}; undeclared implementations: {extra}"
    )


def get_algorithm(algorithm_id: str) -> Algorithm:
    try:
        return _ALGORITHMS[algorithm_id]
    except KeyError as exc:
        available = ", ".join(sorted(_ALGORITHMS))
        raise ValueError(f"Unknown algorithm '{algorithm_id}'. Available: {available}") from exc


def list_algorithms() -> list[str]:
    return sorted(_ALGORITHMS)
