"""The player registry: which players exist and how to instantiate them.

A *player* is anything that can occupy a seat in a game — the random baseline,
an AlphaZero checkpoint, the same checkpoint given a search budget, later a PPO
checkpoint. Training produces players; evaluation, tournaments and the web UI
consume them.

Until now players were implicit, identified by filename convention
(``latest.onnx``, ``best.onnx``). That is enough to ask "is the candidate better
than the champion" and not enough to ask "how do all of these compare", which is
what a tournament needs. This module makes them explicit and durable.

The registry is a JSON file written atomically, like ``stats.json`` and
``best_model.json`` — not a database table. There is no concurrent writer yet,
and a file is inspectable and diffable.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from .atomic_io import atomic_write
from .players import ModelPlayer, RandomPlayer

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: The id reserved for the uniform-random baseline. Tournaments anchor their
#: rating scale to it, so every rating reads as "Elo above random".
RANDOM_PLAYER_ID = "random"

#: Default sampling temperature for a registered player.
#:
#: Deliberately not 0. Two greedy models on a deterministic opening replay the
#: same game every time, so a 20-game match is one game counted 20 times — the
#: rating fit then reads 20 correlated samples as 20 independent ones and
#: reports a confident, meaningless spread. Measured on the Connect 4
#: checkpoints: every model-vs-model pairing came out 0-20, 10-10 or 20-0.
#: crucible's EVAL_TEMPERATURE exists for the same reason.
DEFAULT_PLAY_TEMPERATURE = 0.2

# Checkpoints are named model_step_016000.onnx. Deliberately a local copy of
# this pattern rather than an import: solver_eval owns a Connect 4-specific one,
# and checkpoint.py — the other plausible home — pulls in torch and onnx, which
# this module has no reason to load.
_CHECKPOINT_STEP = re.compile(r"model_step_(\d+)\.onnx")


def infer_step(path: str | Path) -> int | None:
    """Training step from a checkpoint filename, if it encodes one."""
    match = _CHECKPOINT_STEP.fullmatch(Path(path).name)
    return int(match.group(1)) if match else None


@dataclass(frozen=True)
class PlayerRecord:
    """One registered player.

    ``kind`` is ``"model"`` or ``"random"``. ``algorithm`` is free-form and
    exists so a tournament table can say where a player came from — it is a
    label, not a dispatch key; nothing branches on it.
    """

    id: str
    env_id: str
    kind: str
    checkpoint: str | None = None
    simulations: int = 0
    temperature: float = 0.0
    algorithm: str = "unknown"
    step: int | None = None
    registered_at: str = ""

    def to_player(self) -> ModelPlayer | RandomPlayer:
        """Build the spec that ``cartridge-eval`` consumes."""
        if self.kind == "random":
            return RandomPlayer()
        if self.kind == "model":
            if not self.checkpoint:
                raise ValueError(f"Player '{self.id}' is a model with no checkpoint")
            return ModelPlayer(
                model_path=self.checkpoint,
                temperature=self.temperature,
                simulations=self.simulations,
            )
        raise ValueError(f"Player '{self.id}' has unknown kind '{self.kind}'")

    def is_playable(self) -> bool:
        """Whether this player can actually be instantiated right now.

        A registry entry outlives the file it points at — checkpoint rotation
        deletes old models. Callers skip unplayable entries rather than failing
        a whole tournament over one deleted checkpoint.
        """
        if self.kind == "random":
            return True
        return bool(self.checkpoint) and Path(self.checkpoint).exists()


class PlayerRegistry:
    """A collection of players, persisted as JSON."""

    def __init__(self, players: list[PlayerRecord] | None = None):
        self._players: dict[str, PlayerRecord] = {}
        for player in players or []:
            self._players[player.id] = player

    def __len__(self) -> int:
        return len(self._players)

    def __contains__(self, player_id: object) -> bool:
        return player_id in self._players

    def __iter__(self):
        return iter(self._players.values())

    def get(self, player_id: str) -> PlayerRecord:
        try:
            return self._players[player_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._players)) or "(registry is empty)"
            raise KeyError(f"No player '{player_id}'. Registered: {known}") from exc

    def add(self, player: PlayerRecord, replace: bool = False) -> None:
        """Register a player.

        Raises on a duplicate id unless ``replace`` — silently overwriting would
        make a tournament's ratings refer to a player that no longer means what
        the results say it did.
        """
        if player.id in self._players and not replace:
            raise ValueError(f"Player '{player.id}' is already registered")
        self._players[player.id] = player

    def for_env(self, env_id: str) -> list[PlayerRecord]:
        """Registered players for one game, ordered by step then id.

        Step order is what makes a rating table read as a training curve.
        Players without a step (``random``, ``best``) sort first.
        """
        return sorted(
            (p for p in self._players.values() if p.env_id == env_id),
            key=lambda p: (0 if p.step is None else 1, p.step or 0, p.id),
        )

    # --- persistence ---

    @classmethod
    def load(cls, path: Path) -> PlayerRegistry:
        """Read a registry, or return an empty one if the file does not exist."""
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        known = {f.name for f in fields(PlayerRecord)}
        players = [
            PlayerRecord(**{k: v for k, v in entry.items() if k in known})
            for entry in raw.get("players", [])
        ]
        return cls(players)

    def save(self, path: Path) -> None:
        """Write the registry atomically, sorted for stable diffs."""
        payload = {
            "schema_version": SCHEMA_VERSION,
            "players": [
                asdict(p) for p in sorted(self._players.values(), key=lambda p: p.id)
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            path, lambda tmp: Path(tmp).write_text(json.dumps(payload, indent=2) + "\n")
        )


def make_random_player(env_id: str) -> PlayerRecord:
    """The baseline every rating scale is anchored to."""
    return PlayerRecord(
        id=RANDOM_PLAYER_ID,
        env_id=env_id,
        kind="random",
        algorithm="baseline",
        registered_at=datetime.now().isoformat(),
    )


def discover_checkpoints(models_dir: Path) -> list[Path]:
    """Every ONNX checkpoint in a models directory, in training order.

    Step checkpoints first (numerically, not lexically), then ``best.onnx`` and
    ``latest.onnx`` if present — those are aliases of some step, so they are
    registered under their own ids and compared like anything else.
    """
    stepped = sorted(
        (p for p in models_dir.glob("model_step_*.onnx")),
        key=lambda p: infer_step(p) or 0,
    )
    aliases = [models_dir / name for name in ("best.onnx", "latest.onnx")]
    return stepped + [p for p in aliases if p.exists()]


def register_checkpoints(
    registry: PlayerRegistry,
    env_id: str,
    models_dir: Path,
    *,
    algorithm: str = "alphazero",
    simulations: int = 0,
    temperature: float = DEFAULT_PLAY_TEMPERATURE,
    replace: bool = False,
) -> list[PlayerRecord]:
    """Register every checkpoint in ``models_dir``, plus the random baseline.

    Ids are the checkpoint stem, suffixed with the search budget when there is
    one — the same weights at 0 and 100 simulations are genuinely different
    players and must not collide.

    Returns the records that were newly added; already-registered players are
    skipped rather than replaced, so re-running this after more training only
    adds the new checkpoints.
    """
    added: list[PlayerRecord] = []
    now = datetime.now().isoformat()

    if RANDOM_PLAYER_ID not in registry:
        baseline = make_random_player(env_id)
        registry.add(baseline)
        added.append(baseline)

    for checkpoint in discover_checkpoints(models_dir):
        suffix = f"-s{simulations}" if simulations else ""
        player_id = f"{checkpoint.stem}{suffix}"
        if player_id in registry and not replace:
            continue
        record = PlayerRecord(
            id=player_id,
            env_id=env_id,
            kind="model",
            checkpoint=str(checkpoint),
            simulations=simulations,
            temperature=temperature,
            algorithm=algorithm,
            step=infer_step(checkpoint),
            registered_at=now,
        )
        registry.add(record, replace=replace)
        added.append(record)

    return added
