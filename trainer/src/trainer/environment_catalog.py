"""Strict consumer for the engine-generated environment/algorithm manifest."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Callable, TypeVar

MANIFEST_FILENAME = "environment_manifest.json"
MANIFEST_SCHEMA_VERSION = 4
REGENERATE_HINT = f"Run `make environment-manifest` to regenerate {MANIFEST_FILENAME}."
U32_MAX = 2**32 - 1
USIZE_MAX = 2 ** (8 * struct.calcsize("P")) - 1
F32_MAX = float.fromhex("0x1.fffffep+127")


@dataclass(frozen=True)
class AlgorithmComponents:
    collector: str
    learner: str
    orchestration: str
    experience_schema: str
    model_contract: str
    evaluation_suite: str
    serving: str


@dataclass(frozen=True)
class AlgorithmDescriptor:
    id: str
    version: int
    model_artifact_schema_version: int
    display_name: str
    components: AlgorithmComponents
    requirements: tuple[str, ...]


@dataclass(frozen=True)
class CompatibilityIssue:
    code: str
    message: str


@dataclass(frozen=True)
class CompatibilityReport:
    algorithm_id: str
    env_id: str
    compatible: bool
    issues: tuple[CompatibilityIssue, ...]
    unverified_assumptions: tuple[str, ...]

    def require_compatible(self) -> None:
        if self.compatible:
            return
        reasons = "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)
        raise ValueError(
            f"Algorithm '{self.algorithm_id}' is incompatible with environment "
            f"'{self.env_id}': {reasons or 'no compatible profile declaration'}"
        )


@dataclass(frozen=True)
class EngineIdentity:
    env_id: str
    build_id: str


@dataclass(frozen=True)
class WireEncoding:
    schema_version: int
    state: str
    action_kind: str
    action_custom_id: str | None
    observation_kind: str
    observation_elements: int | None
    observation_custom_id: str | None


@dataclass(frozen=True)
class EnvironmentSemantics:
    turn_kind: str
    turn_order: str | None
    information_model: str
    planning_state_model: str
    transition_dynamics: str
    chance_model: str
    reward_model: str


@dataclass(frozen=True)
class ActionSpaceDescriptor:
    kind: str
    discrete_size: int | None = None
    dimensions: tuple[int, ...] = ()
    low: tuple[float, ...] = ()
    high: tuple[float, ...] = ()
    shape: tuple[int, ...] = ()


@dataclass(frozen=True)
class AgentDescriptor:
    id: int
    action_space: ActionSpaceDescriptor


@dataclass(frozen=True)
class AgentModelDescriptor:
    kind: str
    agents: tuple[AgentDescriptor, ...] = ()
    shared_action_space: ActionSpaceDescriptor | None = None


@dataclass(frozen=True)
class EnvironmentCapabilities:
    identity: EngineIdentity
    contract_version: int
    encoding: WireEncoding
    semantics: EnvironmentSemantics
    max_horizon: int | None
    agents: AgentModelDescriptor
    preferred_batch: int


@dataclass(frozen=True)
class BoardObservationDescriptor:
    elements: int
    spatial_channels: int
    legal_actions_offset: int
    player_relative: bool


@dataclass(frozen=True)
class BoardPlayerDescriptor:
    name: str
    symbol: str


@dataclass(frozen=True)
class BoardDescriptor:
    width: int
    height: int
    action_count: int
    observation: BoardObservationDescriptor
    players: tuple[BoardPlayerDescriptor, ...]
    renderer: str

    @property
    def size(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class EnvironmentDescriptor:
    env_id: str
    display_name: str
    description: str
    board: BoardDescriptor | None
    capabilities: EnvironmentCapabilities
    algorithm_profiles: dict[str, CompatibilityReport]

    @property
    def contract_version(self) -> int:
        return self.capabilities.contract_version

    @property
    def compatible_algorithms(self) -> tuple[str, ...]:
        return tuple(
            algorithm_id
            for algorithm_id, report in self.algorithm_profiles.items()
            if report.compatible
        )

    def require_board(self) -> BoardDescriptor:
        if self.board is None:
            raise ValueError(
                f"Environment '{self.env_id}' does not expose the board-game profile"
            )
        return self.board

    def compatibility(self, algorithm_id: str) -> CompatibilityReport:
        try:
            return self.algorithm_profiles[algorithm_id]
        except KeyError as exc:
            known = ", ".join(sorted(self.algorithm_profiles)) or "(none)"
            raise ValueError(
                f"Environment '{self.env_id}' has no compatibility report for "
                f"algorithm '{algorithm_id}'. Known profiles: {known}. {REGENERATE_HINT}"
            ) from exc


def _error(path: str, message: str) -> RuntimeError:
    return RuntimeError(
        f"Invalid environment catalog field {path}: {message}. {REGENERATE_HINT}"
    )


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _error(path, "expected an object with string keys")
    return value


def _keys(raw: dict[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected - set(raw))
    extra = sorted(set(raw) - expected)
    if not missing and not extra:
        return
    details = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if extra:
        details.append("unknown " + ", ".join(extra))
    raise _error(path, "; ".join(details))


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise _error(path, "expected a string")
    return value


def _string(value: Any, path: str) -> str:
    value = _text(value, path)
    if not value:
        raise _error(path, "expected a non-empty string")
    return value


def _nonblank_string(value: Any, path: str) -> str:
    value = _string(value, path)
    if not value.strip():
        raise _error(path, "expected a non-blank string")
    return value


def _runtime_segment(value: Any, path: str) -> str:
    value = _string(value, path)
    if not all(
        character.isascii()
        and (character.islower() or character.isdigit() or character in "_-")
        for character in value
    ):
        raise _error(
            path,
            "expected only lowercase ASCII letters, digits, '_' or '-'",
        )
    return value


def _bounded_integer(
    value: Any,
    path: str,
    *,
    maximum: int,
    rust_type: str,
    minimum: int = 1,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise _error(
            path,
            f"expected a {rust_type} integer between {minimum} and {maximum}",
        )
    return value


def _u32(value: Any, path: str, *, minimum: int = 1) -> int:
    return _bounded_integer(
        value,
        path,
        maximum=U32_MAX,
        rust_type="u32",
        minimum=minimum,
    )


def _usize(value: Any, path: str, *, minimum: int = 1) -> int:
    return _bounded_integer(
        value,
        path,
        maximum=USIZE_MAX,
        rust_type="usize",
        minimum=minimum,
    )


def _checked_usize_product(values: tuple[int, ...], path: str) -> int:
    product = 1
    for value in values:
        if product > USIZE_MAX // value:
            raise _error(path, "product overflows usize")
        product *= value
    return product


def _checked_usize_sum(values: tuple[int, ...], path: str) -> int:
    total = 0
    for value in values:
        if total > USIZE_MAX - value:
            raise _error(path, "sum overflows usize")
        total += value
    return total


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _error(path, "expected a boolean")
    return value


def _enum(value: Any, path: str, allowed: set[str]) -> str:
    value = _string(value, path)
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise _error(path, f"unsupported value {value!r}; expected one of {choices}")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(path, "expected an array")
    return value


def _strings(value: Any, path: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_list(value, path))
    )


def _f32_numbers(value: Any, path: str) -> tuple[float, ...]:
    result = []
    for index, item in enumerate(_list(value, path)):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _error(f"{path}[{index}]", "expected a finite f32 number")
        try:
            number = float(item)
        except (OverflowError, ValueError):
            raise _error(f"{path}[{index}]", "expected a finite f32 number") from None
        if not math.isfinite(number) or abs(number) > F32_MAX:
            raise _error(f"{path}[{index}]", "expected a finite f32 number")
        result.append(number)
    return tuple(result)


def _read_manifest() -> dict[str, Any]:
    resource = files("trainer").joinpath(MANIFEST_FILENAME)
    try:
        return _mapping(json.loads(resource.read_text(encoding="utf-8")), "$")
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Environment catalog is missing or malformed ({resource}). {REGENERATE_HINT}"
        ) from exc


def _parse_algorithm(value: Any, path: str) -> AlgorithmDescriptor:
    raw = _mapping(value, path)
    _keys(
        raw,
        {
            "id",
            "version",
            "model_artifact_schema_version",
            "display_name",
            "components",
            "requirements",
        },
        path,
    )
    component_path = f"{path}.components"
    components = _mapping(raw["components"], component_path)
    fields = {
        "collector",
        "learner",
        "orchestration",
        "experience_schema",
        "model_contract",
        "evaluation_suite",
        "serving",
    }
    _keys(components, fields, component_path)
    return AlgorithmDescriptor(
        id=_runtime_segment(raw["id"], f"{path}.id"),
        version=_u32(raw["version"], f"{path}.version"),
        model_artifact_schema_version=_u32(
            raw["model_artifact_schema_version"],
            f"{path}.model_artifact_schema_version",
        ),
        display_name=_string(raw["display_name"], f"{path}.display_name"),
        components=AlgorithmComponents(
            **{
                field: _string(components[field], f"{component_path}.{field}")
                for field in fields
            }
        ),
        requirements=_strings(raw["requirements"], f"{path}.requirements"),
    )


def _parse_report(value: Any, path: str) -> CompatibilityReport:
    raw = _mapping(value, path)
    _keys(
        raw,
        {"algorithm_id", "env_id", "compatible", "issues", "unverified_assumptions"},
        path,
    )
    issues = []
    for index, value in enumerate(_list(raw["issues"], f"{path}.issues")):
        issue_path = f"{path}.issues[{index}]"
        issue = _mapping(value, issue_path)
        _keys(issue, {"code", "message"}, issue_path)
        issues.append(
            CompatibilityIssue(
                code=_string(issue["code"], f"{issue_path}.code"),
                message=_string(issue["message"], f"{issue_path}.message"),
            )
        )
    return CompatibilityReport(
        algorithm_id=_runtime_segment(raw["algorithm_id"], f"{path}.algorithm_id"),
        env_id=_runtime_segment(raw["env_id"], f"{path}.env_id"),
        compatible=_boolean(raw["compatible"], f"{path}.compatible"),
        issues=tuple(issues),
        unverified_assumptions=_strings(
            raw["unverified_assumptions"], f"{path}.unverified_assumptions"
        ),
    )


def _parse_encoding(value: Any, path: str) -> WireEncoding:
    raw = _mapping(value, path)
    _keys(raw, {"state", "action", "observation", "schema_version"}, path)
    action_path = f"{path}.action"
    action = _mapping(raw["action"], action_path)
    action_kind = _string(action.get("kind"), f"{action_path}.kind")
    if action_kind in {
        "discrete_u32_little_endian",
        "multi_discrete_u32_little_endian",
        "continuous_f32_little_endian",
    }:
        _keys(action, {"kind"}, action_path)
        action_custom_id = None
    elif action_kind == "custom":
        _keys(action, {"kind", "id"}, action_path)
        action_custom_id = _nonblank_string(action["id"], f"{action_path}.id")
    else:
        raise _error(f"{action_path}.kind", f"unsupported encoding {action_kind!r}")

    observation_path = f"{path}.observation"
    observation = _mapping(raw["observation"], observation_path)
    observation_kind = _string(observation.get("kind"), f"{observation_path}.kind")
    if observation_kind == "f32_little_endian":
        _keys(observation, {"kind", "elements"}, observation_path)
        observation_elements = _usize(
            observation["elements"], f"{observation_path}.elements"
        )
        observation_custom_id = None
    elif observation_kind == "custom":
        _keys(observation, {"kind", "id"}, observation_path)
        observation_elements = None
        observation_custom_id = _nonblank_string(
            observation["id"], f"{observation_path}.id"
        )
    else:
        raise _error(
            f"{observation_path}.kind", f"unsupported encoding {observation_kind!r}"
        )
    return WireEncoding(
        schema_version=_u32(raw["schema_version"], f"{path}.schema_version"),
        state=_nonblank_string(raw["state"], f"{path}.state"),
        action_kind=action_kind,
        action_custom_id=action_custom_id,
        observation_kind=observation_kind,
        observation_elements=observation_elements,
        observation_custom_id=observation_custom_id,
    )


def _parse_semantics(value: Any, path: str) -> EnvironmentSemantics:
    raw = _mapping(value, path)
    _keys(
        raw,
        {
            "turn_model",
            "information_model",
            "planning_state_model",
            "transition_dynamics",
            "chance_model",
            "reward_model",
        },
        path,
    )
    turn_path = f"{path}.turn_model"
    turn = _mapping(raw["turn_model"], turn_path)
    turn_kind = _string(turn.get("kind"), f"{turn_path}.kind")
    if turn_kind == "sequential":
        _keys(turn, {"kind", "order"}, turn_path)
        turn_order = _enum(
            turn["order"],
            f"{turn_path}.order",
            {"alternating", "environment_defined"},
        )
    elif turn_kind in {"single_agent", "simultaneous"}:
        _keys(turn, {"kind"}, turn_path)
        turn_order = None
    else:
        raise _error(f"{turn_path}.kind", f"unsupported turn model {turn_kind!r}")
    reward_model = _enum(
        raw["reward_model"], f"{path}.reward_model", {"terminal_zero_sum", "general"}
    )
    chance_model = _enum(
        raw["chance_model"],
        f"{path}.chance_model",
        {"none", "explicit", "environment_sampled"},
    )
    information_model = _enum(
        raw["information_model"],
        f"{path}.information_model",
        {"perfect_information_markov", "partially_observed"},
    )
    planning_state_model = _enum(
        raw["planning_state_model"],
        f"{path}.planning_state_model",
        {"complete_snapshot", "external_state"},
    )
    transition_dynamics = _enum(
        raw["transition_dynamics"],
        f"{path}.transition_dynamics",
        {"deterministic", "stochastic"},
    )
    if (transition_dynamics == "deterministic") != (chance_model == "none"):
        raise _error(
            path,
            "transition_dynamics and chance_model describe inconsistent stochasticity",
        )
    if (
        chance_model == "environment_sampled"
        and planning_state_model == "complete_snapshot"
    ):
        raise _error(
            path,
            "environment-sampled chance depends on runtime RNG state and requires external_state",
        )
    return EnvironmentSemantics(
        turn_kind=turn_kind,
        turn_order=turn_order,
        information_model=information_model,
        planning_state_model=planning_state_model,
        transition_dynamics=transition_dynamics,
        chance_model=chance_model,
        reward_model=reward_model,
    )


def _parse_action_space(value: Any, path: str) -> ActionSpaceDescriptor:
    raw = _mapping(value, path)
    kind = _string(raw.get("kind"), f"{path}.kind")
    if kind == "discrete":
        _keys(raw, {"kind", "size"}, path)
        return ActionSpaceDescriptor(
            kind=kind, discrete_size=_u32(raw["size"], f"{path}.size")
        )
    if kind == "multi_discrete":
        _keys(raw, {"kind", "dimensions"}, path)
        dimensions = tuple(
            _u32(item, f"{path}.dimensions[{index}]")
            for index, item in enumerate(_list(raw["dimensions"], f"{path}.dimensions"))
        )
        if not dimensions:
            raise _error(f"{path}.dimensions", "must not be empty")
        return ActionSpaceDescriptor(kind=kind, dimensions=dimensions)
    if kind == "continuous":
        _keys(raw, {"kind", "low", "high", "shape"}, path)
        low = _f32_numbers(raw["low"], f"{path}.low")
        high = _f32_numbers(raw["high"], f"{path}.high")
        shape = tuple(
            _u32(item, f"{path}.shape[{index}]")
            for index, item in enumerate(_list(raw["shape"], f"{path}.shape"))
        )
        elements = _checked_usize_product(shape, f"{path}.shape") if shape else None
        if not low or len(low) != len(high) or not shape or elements != len(low):
            raise _error(
                path,
                "low/high lengths must equal the product of the non-empty shape",
            )
        if any(lower >= upper for lower, upper in zip(low, high)):
            raise _error(
                path, "every continuous lower bound must be below its upper bound"
            )
        return ActionSpaceDescriptor(kind=kind, low=low, high=high, shape=shape)
    raise _error(f"{path}.kind", f"unsupported action space {kind!r}")


def _parse_agent_model(value: Any, path: str) -> AgentModelDescriptor:
    raw = _mapping(value, path)
    kind = _string(raw.get("kind"), f"{path}.kind")
    if kind == "fixed":
        _keys(raw, {"kind", "agents"}, path)
        agents = []
        for index, value in enumerate(_list(raw["agents"], f"{path}.agents")):
            agent_path = f"{path}.agents[{index}]"
            agent = _mapping(value, agent_path)
            _keys(agent, {"id", "action_space"}, agent_path)
            agents.append(
                AgentDescriptor(
                    id=_u32(agent["id"], f"{agent_path}.id", minimum=0),
                    action_space=_parse_action_space(
                        agent["action_space"], f"{agent_path}.action_space"
                    ),
                )
            )
        if not agents or len({agent.id for agent in agents}) != len(agents):
            raise _error(f"{path}.agents", "must contain unique fixed agent IDs")
        return AgentModelDescriptor(kind=kind, agents=tuple(agents))
    if kind == "dynamic":
        _keys(raw, {"kind", "action_space"}, path)
        return AgentModelDescriptor(
            kind=kind,
            shared_action_space=_parse_action_space(
                raw["action_space"], f"{path}.action_space"
            ),
        )
    raise _error(f"{path}.kind", f"unsupported agent model {kind!r}")


def _parse_capabilities(value: Any, path: str) -> EnvironmentCapabilities:
    raw = _mapping(value, path)
    _keys(
        raw,
        {
            "id",
            "contract_version",
            "encoding",
            "semantics",
            "max_horizon",
            "agents",
            "preferred_batch",
        },
        path,
    )
    identity_path = f"{path}.id"
    identity = _mapping(raw["id"], identity_path)
    _keys(identity, {"env_id", "build_id"}, identity_path)
    max_horizon = raw["max_horizon"]
    if max_horizon is not None:
        max_horizon = _u32(max_horizon, f"{path}.max_horizon")
    capabilities = EnvironmentCapabilities(
        identity=EngineIdentity(
            env_id=_runtime_segment(identity["env_id"], f"{identity_path}.env_id"),
            build_id=_nonblank_string(
                identity["build_id"], f"{identity_path}.build_id"
            ),
        ),
        contract_version=_u32(raw["contract_version"], f"{path}.contract_version"),
        encoding=_parse_encoding(raw["encoding"], f"{path}.encoding"),
        semantics=_parse_semantics(raw["semantics"], f"{path}.semantics"),
        max_horizon=max_horizon,
        agents=_parse_agent_model(raw["agents"], f"{path}.agents"),
        preferred_batch=_u32(raw["preferred_batch"], f"{path}.preferred_batch"),
    )
    if (
        capabilities.semantics.turn_kind == "single_agent"
        and capabilities.agents.kind == "fixed"
        and len(capabilities.agents.agents) != 1
    ):
        raise _error(
            f"{path}.agents",
            "single_agent semantics require exactly one fixed agent",
        )
    spaces = (
        tuple(agent.action_space for agent in capabilities.agents.agents)
        if capabilities.agents.kind == "fixed"
        else (capabilities.agents.shared_action_space,)
    )
    standard_action_spaces = {
        "discrete_u32_little_endian": "discrete",
        "multi_discrete_u32_little_endian": "multi_discrete",
        "continuous_f32_little_endian": "continuous",
    }
    expected_space = standard_action_spaces.get(capabilities.encoding.action_kind)
    if expected_space is not None and any(
        space is None or space.kind != expected_space for space in spaces
    ):
        raise _error(
            f"{path}.encoding.action",
            f"requires {expected_space} action spaces for every agent",
        )
    if (
        capabilities.semantics.turn_kind == "simultaneous"
        and capabilities.encoding.action_kind != "custom"
    ):
        raise _error(
            f"{path}.encoding.action",
            "simultaneous decisions require a custom joint-action codec until the "
            "wire ABI defines a standard decision-action envelope",
        )
    if (
        capabilities.semantics.chance_model == "explicit"
        and capabilities.encoding.action_kind != "custom"
    ):
        raise _error(
            f"{path}.encoding.action",
            "explicit chance requires a custom chance/agent action codec until the "
            "wire ABI defines a standard decision-action envelope",
        )
    return capabilities


def _parse_board(value: Any, path: str) -> BoardDescriptor | None:
    if value is None:
        return None
    raw = _mapping(value, path)
    _keys(
        raw,
        {"width", "height", "action_count", "observation", "players", "renderer"},
        path,
    )
    observation_path = f"{path}.observation"
    observation = _mapping(raw["observation"], observation_path)
    _keys(
        observation,
        {"elements", "spatial_channels", "legal_actions_offset", "player_relative"},
        observation_path,
    )
    players = []
    for index, value in enumerate(_list(raw["players"], f"{path}.players")):
        player_path = f"{path}.players[{index}]"
        player = _mapping(value, player_path)
        _keys(player, {"name", "symbol"}, player_path)
        players.append(
            BoardPlayerDescriptor(
                name=_nonblank_string(player["name"], f"{player_path}.name"),
                symbol=_nonblank_string(player["symbol"], f"{player_path}.symbol"),
            )
        )
    if not players:
        raise _error(f"{path}.players", "must not be empty")
    return BoardDescriptor(
        width=_usize(raw["width"], f"{path}.width"),
        height=_usize(raw["height"], f"{path}.height"),
        action_count=_usize(raw["action_count"], f"{path}.action_count"),
        observation=BoardObservationDescriptor(
            elements=_usize(observation["elements"], f"{observation_path}.elements"),
            spatial_channels=_usize(
                observation["spatial_channels"], f"{observation_path}.spatial_channels"
            ),
            legal_actions_offset=_usize(
                observation["legal_actions_offset"],
                f"{observation_path}.legal_actions_offset",
                minimum=0,
            ),
            player_relative=_boolean(
                observation["player_relative"], f"{observation_path}.player_relative"
            ),
        ),
        players=tuple(players),
        renderer=_enum(
            raw["renderer"],
            f"{path}.renderer",
            {"grid", "drop_column", "generals"},
        ),
    )


def _parse_environment(
    value: Any, path: str, algorithm_ids: set[str]
) -> EnvironmentDescriptor:
    raw = _mapping(value, path)
    _keys(raw, {"metadata", "capabilities", "algorithm_profiles"}, path)
    metadata_path = f"{path}.metadata"
    metadata = _mapping(raw["metadata"], metadata_path)
    _keys(metadata, {"id", "display_name", "description", "board"}, metadata_path)
    env_id = _runtime_segment(metadata["id"], f"{metadata_path}.id")
    board = _parse_board(metadata["board"], f"{metadata_path}.board")
    capabilities = _parse_capabilities(raw["capabilities"], f"{path}.capabilities")
    profiles_raw = _mapping(raw["algorithm_profiles"], f"{path}.algorithm_profiles")
    if set(profiles_raw) != algorithm_ids:
        raise _error(
            f"{path}.algorithm_profiles",
            "profile IDs must exactly match the algorithm catalog",
        )
    profiles = {
        algorithm_id: _parse_report(report, f"{path}.algorithm_profiles.{algorithm_id}")
        for algorithm_id, report in profiles_raw.items()
    }
    descriptor = EnvironmentDescriptor(
        env_id=env_id,
        display_name=_nonblank_string(
            metadata["display_name"], f"{metadata_path}.display_name"
        ),
        description=_text(metadata["description"], f"{metadata_path}.description"),
        board=board,
        capabilities=capabilities,
        algorithm_profiles=profiles,
    )
    if capabilities.identity.env_id != env_id:
        raise _error(f"{path}.capabilities.id.env_id", "does not match metadata.id")
    if board is not None:
        board_path = f"{metadata_path}.board"
        if len(board.players) != 2:
            raise _error(f"{board_path}.players", "board profile requires two players")
        board_size = _checked_usize_product(
            (board.width, board.height), f"{board_path}.width/height"
        )
        expected_offset = _checked_usize_product(
            (board.observation.spatial_channels, board_size),
            f"{board_path}.observation.spatial_channels",
        )
        if board.observation.legal_actions_offset != expected_offset:
            raise _error(
                f"{board_path}.observation.legal_actions_offset",
                f"board profile requires spatial_channels * board size ({expected_offset})",
            )
        expected_elements = _checked_usize_sum(
            (expected_offset, board.action_count, 2),
            f"{board_path}.observation.elements",
        )
        if board.observation.elements != expected_elements:
            raise _error(
                f"{board_path}.observation.elements",
                "board profile requires planes + legal mask + two player "
                f"indicators ({expected_elements})",
            )
        if board.action_count > U32_MAX:
            raise _error(
                f"{board_path}.action_count",
                "board profile action count must fit the canonical u32 encoding",
            )
        if capabilities.encoding.action_kind != "discrete_u32_little_endian":
            raise _error(
                f"{path}.capabilities.encoding.action",
                "board profile requires discrete_u32_little_endian",
            )
        elements = capabilities.encoding.observation_elements
        if capabilities.encoding.observation_kind != "f32_little_endian":
            raise _error(
                f"{path}.capabilities.encoding.observation",
                "board profile requires f32_little_endian",
            )
        if elements != board.observation.elements:
            raise _error(
                f"{path}.capabilities.encoding.observation",
                "does not match board observation elements",
            )
        agents = capabilities.agents
        if agents.kind != "fixed" or [agent.id for agent in agents.agents] != [1, 2]:
            raise _error(
                f"{path}.capabilities.agents",
                "board profile requires fixed agents [1, 2] in seat order",
            )
        if any(
            agent.action_space.kind != "discrete"
            or agent.action_space.discrete_size != board.action_count
            for agent in agents.agents
        ):
            raise _error(
                f"{path}.capabilities.agents",
                "board profile action spaces must match board action_count",
            )
        semantics = capabilities.semantics
        expected_semantics = (
            "sequential",
            "alternating",
            "perfect_information_markov",
            "complete_snapshot",
            "deterministic",
            "none",
            "terminal_zero_sum",
        )
        actual_semantics = (
            semantics.turn_kind,
            semantics.turn_order,
            semantics.information_model,
            semantics.planning_state_model,
            semantics.transition_dynamics,
            semantics.chance_model,
            semantics.reward_model,
        )
        if actual_semantics != expected_semantics:
            raise _error(
                f"{path}.capabilities.semantics",
                "board profile requires deterministic alternating perfect-information "
                "complete-snapshot terminal-zero-sum semantics",
            )
    for algorithm_id, report in profiles.items():
        if report.algorithm_id != algorithm_id or report.env_id != env_id:
            raise _error(
                f"{path}.algorithm_profiles.{algorithm_id}",
                "profile identity does not match its key/environment",
            )
    return descriptor


T = TypeVar("T")


def _index_unique(items: list[T], key: Callable[[T], str], kind: str) -> dict[str, T]:
    index: dict[str, T] = {}
    for item in items:
        item_key = key(item)
        if item_key in index:
            raise RuntimeError(f"Duplicate {kind} ID '{item_key}'. {REGENERATE_HINT}")
        index[item_key] = item
    return index


def _load_catalog(
    document: dict[str, Any],
) -> tuple[dict[str, AlgorithmDescriptor], dict[str, EnvironmentDescriptor]]:
    _keys(
        document, {"schema_version", "generated_by", "algorithms", "environments"}, "$"
    )
    schema_version = _u32(document["schema_version"], "$.schema_version", minimum=0)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported environment catalog schema {schema_version!r}; expected "
            f"{MANIFEST_SCHEMA_VERSION}. {REGENERATE_HINT}"
        )
    _string(document["generated_by"], "$.generated_by")
    algorithms = _index_unique(
        [
            _parse_algorithm(value, f"$.algorithms[{index}]")
            for index, value in enumerate(_list(document["algorithms"], "$.algorithms"))
        ],
        lambda descriptor: descriptor.id,
        "algorithm",
    )
    if not algorithms:
        raise _error("$.algorithms", "must not be empty")
    environments = _index_unique(
        [
            _parse_environment(value, f"$.environments[{index}]", set(algorithms))
            for index, value in enumerate(
                _list(document["environments"], "$.environments")
            )
        ],
        lambda descriptor: descriptor.env_id,
        "environment",
    )
    return algorithms, environments


ALGORITHMS, ENVIRONMENTS = _load_catalog(_read_manifest())


def get_environment(env_id: str) -> EnvironmentDescriptor:
    try:
        return ENVIRONMENTS[env_id]
    except KeyError as exc:
        available = ", ".join(sorted(ENVIRONMENTS))
        raise ValueError(
            f"Unknown environment '{env_id}'. Available: {available}"
        ) from exc


def get_algorithm_descriptor(algorithm_id: str) -> AlgorithmDescriptor:
    try:
        return ALGORITHMS[algorithm_id]
    except KeyError as exc:
        available = ", ".join(sorted(ALGORITHMS))
        raise ValueError(
            f"Unknown algorithm '{algorithm_id}'. Available: {available}"
        ) from exc


def list_environments() -> list[str]:
    return sorted(ENVIRONMENTS)


def list_algorithm_ids() -> list[str]:
    return sorted(ALGORITHMS)
