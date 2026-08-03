"""Who occupies a seat in an evaluation game.

These are *specifications*, not policies. Evaluation games are played by the
Rust ``cartridge-eval`` binary against the real engine, so a player here is
just the set of command-line arguments that tells the binary what to put in a
seat — this module holds no model session and selects no actions.

That is the point. The previous ``OnnxPolicy``/``RandomPolicy`` played games in
Python against a hand-maintained Python copy of each game's rules, which meant
the promotion gate scored a game that nothing kept in sync with the one being
trained, and could only cover games somebody had reimplemented (Othello never
was). See ``evaluator.evaluate``.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

_MAX_U32 = (1 << 32) - 1
_MAX_F32 = float.fromhex("0x1.fffffep+127")


@dataclass(frozen=True)
class RandomPlayer:
    """The uniform-random baseline."""

    @property
    def name(self) -> str:
        return "Random"

    def cli_args(self, slot: str) -> list[str]:
        """Arguments placing this player in ``slot`` ("p1" or "p2")."""
        return [f"--{slot}", "random"]


@dataclass(frozen=True)
class ModelPlayer:
    """A trained model, optionally playing with search.

    ``simulations`` is the MCTS budget per move. Zero means no search: the
    binary samples the policy head directly. Above zero the model plays with
    search, which is how an AlphaZero system actually plays and the honest
    measure of its strength — the policy head alone understates it.
    """

    model_path: str
    temperature: float = 0.0
    simulations: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("model_path must be a nonempty string")
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, (int, float)
        ):
            raise ValueError("temperature must be a finite nonnegative f32")
        temperature = float(self.temperature)
        if (
            not math.isfinite(temperature)
            or temperature < 0.0
            or temperature > _MAX_F32
        ):
            raise ValueError("temperature must be a finite nonnegative f32")
        narrowed = float(struct.unpack("!f", struct.pack("!f", temperature))[0])
        object.__setattr__(self, "temperature", 0.0 if narrowed == 0.0 else narrowed)
        if (
            isinstance(self.simulations, bool)
            or not isinstance(self.simulations, int)
            or not 0 <= self.simulations <= _MAX_U32
        ):
            raise ValueError("simulations must be a nonnegative u32 integer")

    @property
    def name(self) -> str:
        return f"ONNX({Path(self.model_path).name})"

    def cli_args(self, slot: str) -> list[str]:
        """Arguments placing this player in ``slot`` ("p1" or "p2")."""
        return [
            f"--{slot}",
            str(self.model_path),
            f"--{slot}-temperature",
            str(self.temperature),
            f"--{slot}-sims",
            str(self.simulations),
        ]


__all__ = ["ModelPlayer", "RandomPlayer"]
