"""Strict registry of immutable players available to evaluation workflows.

A model player is the combination of one content-addressed checkpoint and one
versioned gameplay-adapter configuration.  Registry IDs therefore never depend
on filenames and never change meaning when a serving channel advances.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from crucible.atomic_io import atomic_write

from .players import ModelPlayer, RandomPlayer
from .storage.publisher import (
    ArtifactValidationError,
    CheckpointPublisher,
    CheckpointRef,
    OnnxArtifactContract,
    canonical_json_bytes,
    create_checkpoint_publisher,
    sha256_bytes,
    validate_sha256_digest,
)

SCHEMA_VERSION = 5
PLAYER_ADAPTER_SCHEMA_VERSION = 1

# A little exploration is important in a repeated match between deterministic
# policies; otherwise every scheduled game per seat is the same sample.
DEFAULT_PLAY_TEMPERATURE = 0.2
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_MAX_F32 = float.fromhex("0x1.fffffep+127")


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON keys instead of accepting the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Player registry contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Player registry contains non-finite number {value}")


def _validate_adapter_settings(simulations: int, temperature: float) -> float:
    if (
        isinstance(simulations, bool)
        or not isinstance(simulations, int)
        or not 0 <= simulations <= _MAX_U32
    ):
        raise ValueError("simulations must be a nonnegative u32 integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0.0 <= temperature <= _MAX_F32
    ):
        raise ValueError("temperature must be a finite nonnegative f32")
    narrowed = float(struct.unpack("!f", struct.pack("!f", float(temperature)))[0])
    return 0.0 if narrowed == 0.0 else narrowed


def artifact_contract_for(env_id: str, algorithm_id: str) -> OnnxArtifactContract:
    """Return the exact ONNX contract authorized by the engine manifest."""
    from .algorithms import get_algorithm
    from .environment_catalog import get_environment

    algorithm = get_algorithm(algorithm_id)
    environment = get_environment(env_id)
    algorithm.compatibility(environment).require_compatible()
    board = environment.require_board()
    descriptor = algorithm.descriptor
    return OnnxArtifactContract(
        algorithm_id=descriptor.id,
        env_id=environment.env_id,
        env_contract_version=environment.contract_version,
        model_artifact_schema_version=descriptor.model_artifact_schema_version,
        model_contract=descriptor.components.model_contract,
        obs_size=board.observation.elements,
        num_actions=board.action_count,
    )


def random_player_id(
    *, env_id: str, env_contract_version: int, algorithm_id: str
) -> str:
    """Globally unique identity for one profile's random baseline."""
    return f"{algorithm_id}:{env_id}:v{env_contract_version}:random"


def checkpoint_player_id(
    *,
    env_id: str,
    env_contract_version: int,
    algorithm_id: str,
    checkpoint_id: str,
    simulations: int = 0,
    temperature: float = DEFAULT_PLAY_TEMPERATURE,
) -> str:
    """Canonical identity for checkpoint weights plus gameplay adapter config."""
    validate_sha256_digest(checkpoint_id, field="checkpoint_id")
    canonical_temperature = _validate_adapter_settings(simulations, temperature)
    adapter = {
        "schema_version": PLAYER_ADAPTER_SCHEMA_VERSION,
        "simulations": simulations,
        "temperature": canonical_temperature,
    }
    adapter_id = sha256_bytes(canonical_json_bytes(adapter))
    return (
        f"{algorithm_id}:{env_id}:v{env_contract_version}:"
        f"checkpoint:{checkpoint_id}:adapter:{adapter_id}"
    )


def _model_root_from_blob(path: Path) -> Path:
    """Recover and validate the repository root encoded by an immutable blob path."""
    if (
        path.suffix != ".onnx"
        or path.parent.name != "sha256"
        or path.parent.parent.name != "blobs"
    ):
        raise ArtifactValidationError(
            "Registered ONNX path must use " "<model-root>/blobs/sha256/<digest>.onnx"
        )
    validate_sha256_digest(path.stem, field="onnx_path digest")
    return path.parents[2]


def _resolve_registered_checkpoint(
    *,
    checkpoint_id: str,
    onnx_path: str,
    contract: OnnxArtifactContract,
) -> CheckpointRef:
    """Resolve a registry entry through the strict local artifact repository."""
    path = Path(onnx_path)
    model_root = _model_root_from_blob(path)
    repository = create_checkpoint_publisher(contract, model_root)
    checkpoint = repository.resolve_checkpoint(checkpoint_id)
    if checkpoint.onnx_path != path:
        raise ArtifactValidationError(
            f"Registered ONNX path {path} does not match checkpoint "
            f"{checkpoint_id}: {checkpoint.onnx_path}"
        )
    return checkpoint


@dataclass(frozen=True)
class PlayerRecord:
    """One random baseline or immutable model plus its play configuration."""

    id: str
    env_id: str
    env_contract_version: int
    algorithm_id: str
    model_contract: str
    model_artifact_schema_version: int
    kind: str
    registered_at: str
    checkpoint_id: str | None = None
    onnx_path: str | None = None
    simulations: int = 0
    temperature: float = 0.0
    step: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "id",
            "env_id",
            "algorithm_id",
            "model_contract",
            "registered_at",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"PlayerRecord.{field_name} must be a non-empty string"
                )
        for field_name in (
            "env_contract_version",
            "model_artifact_schema_version",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= _MAX_U32
            ):
                raise ValueError(
                    f"PlayerRecord.{field_name} must be a positive u32 integer"
                )
        if self.kind not in {"model", "random"}:
            raise ValueError(f"Player '{self.id}' has unknown kind '{self.kind}'")
        canonical_temperature = _validate_adapter_settings(
            self.simulations, self.temperature
        )
        object.__setattr__(self, "temperature", canonical_temperature)
        if self.step is not None and (
            isinstance(self.step, bool)
            or not isinstance(self.step, int)
            or not 0 <= self.step <= _MAX_U64
        ):
            raise ValueError(
                "PlayerRecord.step must be a nonnegative u64 integer or null"
            )
        try:
            datetime.fromisoformat(self.registered_at)
        except ValueError as exc:
            raise ValueError(
                "PlayerRecord.registered_at must be an ISO-8601 timestamp"
            ) from exc

        if self.kind == "random":
            if self.checkpoint_id is not None or self.onnx_path is not None:
                raise ValueError(
                    "Random players cannot declare a checkpoint or ONNX blob"
                )
            if self.step is not None:
                raise ValueError("Random players cannot declare a training step")
            if self.simulations != 0 or self.temperature != 0.0:
                raise ValueError(
                    "Random players cannot declare search or sampling settings"
                )
        else:
            validate_sha256_digest(self.checkpoint_id, field="checkpoint_id")
            if not isinstance(self.onnx_path, str) or not self.onnx_path:
                raise ValueError(f"Player '{self.id}' is a model with no ONNX path")
            if self.step is None:
                raise ValueError(f"Player '{self.id}' is a model with no training step")

    def artifact_contract(self) -> OnnxArtifactContract:
        """Validate profile lineage and canonical player identity."""
        contract = artifact_contract_for(self.env_id, self.algorithm_id)
        expected = {
            "env_contract_version": contract.env_contract_version,
            "model_contract": contract.model_contract,
            "model_artifact_schema_version": contract.model_artifact_schema_version,
        }
        mismatches = [
            f"{field_name}={getattr(self, field_name)!r} (expected {value!r})"
            for field_name, value in expected.items()
            if getattr(self, field_name) != value
        ]
        if mismatches:
            raise ValueError(
                f"Player '{self.id}' does not match the engine profile: "
                + "; ".join(mismatches)
            )

        expected_id = (
            random_player_id(
                env_id=self.env_id,
                env_contract_version=self.env_contract_version,
                algorithm_id=self.algorithm_id,
            )
            if self.kind == "random"
            else checkpoint_player_id(
                env_id=self.env_id,
                env_contract_version=self.env_contract_version,
                algorithm_id=self.algorithm_id,
                checkpoint_id=self.checkpoint_id or "",
                simulations=self.simulations,
                temperature=self.temperature,
            )
        )
        if self.id != expected_id:
            raise ValueError(
                f"Player id {self.id!r} is not canonical for its profile; "
                f"expected {expected_id!r}"
            )
        return contract

    def validate_artifact(self) -> CheckpointRef | None:
        """Require the referenced immutable checkpoint to be complete and valid."""
        contract = self.artifact_contract()
        if self.kind == "random":
            return None
        checkpoint = _resolve_registered_checkpoint(
            checkpoint_id=self.checkpoint_id or "",
            onnx_path=self.onnx_path or "",
            contract=contract,
        )
        if checkpoint.manifest.step != self.step:
            raise ArtifactValidationError(
                f"Player '{self.id}' step {self.step} does not match checkpoint "
                f"step {checkpoint.manifest.step}"
            )
        return checkpoint

    def to_player(self) -> ModelPlayer | RandomPlayer:
        """Build the seat specification consumed by ``cartridge-eval``."""
        checkpoint = self.validate_artifact()
        if self.kind == "random":
            return RandomPlayer()
        assert checkpoint is not None
        return ModelPlayer(
            model_path=str(checkpoint.onnx_path),
            temperature=float(self.temperature),
            simulations=self.simulations,
        )


class PlayerRegistry:
    """A strict collection of players persisted as one versioned JSON file."""

    def __init__(self, players: list[PlayerRecord] | None = None):
        self._players: dict[str, PlayerRecord] = {}
        for player in players or []:
            self.add(player)

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
        """Add a validated player without silently changing an existing ID."""
        player.validate_artifact()
        if player.id in self._players and not replace:
            raise ValueError(f"Player '{player.id}' is already registered")
        self._players[player.id] = player

    def for_profile(
        self,
        *,
        env_id: str,
        env_contract_version: int,
        algorithm_id: str,
    ) -> list[PlayerRecord]:
        """Return one profile's players in training order."""
        contract = artifact_contract_for(env_id, algorithm_id)
        if env_contract_version != contract.env_contract_version:
            raise ValueError(
                f"Environment '{env_id}' contract version is "
                f"{contract.env_contract_version}, not {env_contract_version}"
            )
        return sorted(
            (
                player
                for player in self._players.values()
                if player.env_id == env_id
                and player.env_contract_version == env_contract_version
                and player.algorithm_id == algorithm_id
                and player.model_contract == contract.model_contract
                and player.model_artifact_schema_version
                == contract.model_artifact_schema_version
            ),
            key=lambda player: (
                0 if player.step is None else 1,
                player.step or 0,
                player.id,
            ),
        )

    @classmethod
    def load(cls, path: Path) -> PlayerRegistry:
        """Read a registry; a missing path denotes an empty registry."""
        if not path.exists():
            return cls()
        if not path.is_file():
            raise ValueError(f"Player registry path is not a regular file: {path}")
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "players"}:
            raise ValueError(
                "Player registry must contain exactly 'schema_version' and 'players'"
            )
        schema_version = raw["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != SCHEMA_VERSION
        ):
            raise ValueError(
                f"Unsupported player registry schema {schema_version!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        entries = raw["players"]
        if not isinstance(entries, list):
            raise ValueError("Player registry 'players' must be an array")
        fields = set(PlayerRecord.__dataclass_fields__)
        players = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not all(
                isinstance(key, str) for key in entry
            ):
                raise ValueError(f"Player registry entry {index} must be an object")
            if set(entry) != fields:
                missing = sorted(fields - set(entry))
                extra = sorted(set(entry) - fields)
                details = []
                if missing:
                    details.append("missing " + ", ".join(missing))
                if extra:
                    details.append("unknown " + ", ".join(extra))
                raise ValueError(
                    f"Player registry entry {index} has invalid fields: "
                    + "; ".join(details)
                )
            players.append(PlayerRecord(**entry))
        return cls(players)

    def save(self, path: Path) -> None:
        """Write the validated registry atomically with stable ordering."""
        for player in self._players.values():
            player.validate_artifact()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "players": [
                asdict(player)
                for player in sorted(self._players.values(), key=lambda item: item.id)
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            path,
            lambda temporary: Path(temporary).write_text(
                json.dumps(payload, indent=2) + "\n"
            ),
        )


def make_random_player(env_id: str, algorithm_id: str) -> PlayerRecord:
    """Build the baseline that anchors this profile's rating scale."""
    contract = artifact_contract_for(env_id, algorithm_id)
    return PlayerRecord(
        id=random_player_id(
            env_id=env_id,
            env_contract_version=contract.env_contract_version,
            algorithm_id=algorithm_id,
        ),
        env_id=env_id,
        env_contract_version=contract.env_contract_version,
        algorithm_id=algorithm_id,
        model_contract=contract.model_contract,
        model_artifact_schema_version=contract.model_artifact_schema_version,
        kind="random",
        registered_at=datetime.now().isoformat(),
    )


def discover_checkpoints(repository: CheckpointPublisher) -> list[CheckpointRef]:
    """Return all fully verified immutable checkpoints in manifest step order."""
    checkpoints = repository.list_checkpoints()
    checkpoint_ids = [checkpoint.checkpoint_id for checkpoint in checkpoints]
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ArtifactValidationError(
            "Checkpoint repository returned a duplicate manifest ID"
        )
    ordered = sorted(
        checkpoints,
        key=lambda checkpoint: (checkpoint.manifest.step, checkpoint.checkpoint_id),
    )
    if checkpoints != ordered:
        raise ArtifactValidationError(
            "Checkpoint repository returned a non-canonical discovery order"
        )
    return checkpoints


def register_checkpoints(
    registry: PlayerRegistry,
    env_id: str,
    model_root: Path,
    *,
    algorithm_id: str,
    simulations: int = 0,
    temperature: float = DEFAULT_PLAY_TEMPERATURE,
    replace: bool = False,
    repository: CheckpointPublisher | None = None,
) -> list[PlayerRecord]:
    """Register every immutable checkpoint plus the profile's random baseline."""
    canonical_temperature = _validate_adapter_settings(simulations, temperature)
    contract = artifact_contract_for(env_id, algorithm_id)
    if repository is None:
        repository = create_checkpoint_publisher(contract, model_root)

    added: list[PlayerRecord] = []
    now = datetime.now().isoformat()
    candidates = []
    for checkpoint in discover_checkpoints(repository):
        if checkpoint.manifest.profile != contract.profile:
            raise ArtifactValidationError(
                f"Checkpoint {checkpoint.checkpoint_id} profile does not match "
                f"{algorithm_id}/{env_id}/v{contract.env_contract_version}"
            )
        record = PlayerRecord(
            id=checkpoint_player_id(
                env_id=env_id,
                env_contract_version=contract.env_contract_version,
                algorithm_id=algorithm_id,
                checkpoint_id=checkpoint.checkpoint_id,
                simulations=simulations,
                temperature=canonical_temperature,
            ),
            env_id=env_id,
            env_contract_version=contract.env_contract_version,
            algorithm_id=algorithm_id,
            model_contract=contract.model_contract,
            model_artifact_schema_version=contract.model_artifact_schema_version,
            kind="model",
            registered_at=now,
            checkpoint_id=checkpoint.checkpoint_id,
            onnx_path=str(checkpoint.onnx_path),
            simulations=simulations,
            temperature=canonical_temperature,
            step=checkpoint.manifest.step,
        )
        record.validate_artifact()
        candidates.append(record)

    baseline = make_random_player(env_id, algorithm_id)
    if baseline.id not in registry:
        registry.add(baseline)
        added.append(baseline)

    for record in candidates:
        if record.id in registry and not replace:
            continue
        registry.add(record, replace=replace)
        added.append(record)
    return added
