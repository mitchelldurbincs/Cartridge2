"""Protocol tests for the engine-generated manifest consumer."""

from copy import deepcopy

import pytest

from trainer import environment_catalog as catalog


def manifest():
    return deepcopy(catalog._read_manifest())


def environment(document, env_id):
    return next(item for item in document["environments"] if item["metadata"]["id"] == env_id)


def test_catalog_exposes_exact_environment_and_artifact_versions():
    environment = catalog.get_environment("connect4")
    algorithm = catalog.get_algorithm_descriptor("alphazero_board_v1")

    assert environment.contract_version == 2
    assert environment.capabilities.encoding.action_kind == "discrete_u32_little_endian"
    assert environment.capabilities.encoding.observation_kind == "tensor"
    board = environment.require_board()
    tensor = environment.capabilities.encoding.observation_tensor
    assert tensor is not None
    assert tensor.dtype == "f32_little_endian"
    assert tensor.fixed_elements == 84
    assert board.size == 42
    assert environment.capabilities.semantics.turn_kind == "sequential"
    assert environment.capabilities.semantics.turn_order == "alternating"
    assert environment.capabilities.semantics.reward_model == "terminal_zero_sum"
    assert environment.capabilities.semantics.chance_model == "none"
    assert environment.capabilities.agents.kind == "fixed"
    assert [agent.id for agent in environment.capabilities.agents.agents] == [1, 2]
    assert all(
        agent.action_availability_kind == "discrete_mask"
        for agent in environment.capabilities.agents.agents
    )
    assert algorithm.model_artifact_schema_version == 1
    assert algorithm.components.serving == "alphazero_mcts_web_v1"


def test_catalog_carries_a_non_board_environment_without_alpha_zero_projection():
    counter = catalog.get_environment("counter")

    assert counter.board is None
    assert counter.capabilities.semantics.turn_kind == "single_agent"
    assert counter.capabilities.semantics.reward_model == "general"
    assert not counter.compatibility("alphazero_board_v1").compatible


def test_catalog_rejects_continuous_bounds_that_do_not_match_shape():
    document = manifest()
    counter = environment(document, "counter")
    counter["capabilities"]["encoding"]["action"] = {"kind": "continuous_f32_little_endian"}
    counter["capabilities"]["agents"]["agents"][0]["action_space"] = {
        "kind": "continuous",
        "low": [-1.0, -1.0],
        "high": [1.0, 1.0],
        "shape": [3],
    }

    with pytest.raises(RuntimeError, match="product"):
        catalog._load_catalog(document)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document["algorithms"][0].update(version=catalog.U32_MAX + 1),
        lambda document: environment(document, "counter")["capabilities"].update(
            contract_version=catalog.U32_MAX + 1
        ),
        lambda document: environment(document, "counter")["capabilities"]["agents"]["agents"][
            0
        ].update(id=catalog.U32_MAX + 1),
        lambda document: environment(document, "counter")["capabilities"]["agents"]["agents"][0][
            "action_space"
        ].update(size=catalog.U32_MAX + 1),
    ],
)
def test_catalog_rejects_values_outside_rust_u32_fields(mutate):
    document = manifest()
    mutate(document)

    with pytest.raises(RuntimeError, match="u32 integer"):
        catalog._load_catalog(document)


def test_catalog_rejects_values_outside_rust_usize_fields():
    document = manifest()
    environment(document, "connect4")["metadata"]["board"]["width"] = catalog.USIZE_MAX + 1

    with pytest.raises(RuntimeError, match="usize integer"):
        catalog._load_catalog(document)


def test_catalog_rejects_board_layout_arithmetic_that_overflows_usize():
    document = manifest()
    board = environment(document, "connect4")["metadata"]["board"]
    board["width"] = catalog.USIZE_MAX
    board["height"] = 2

    with pytest.raises(RuntimeError, match="product overflows usize"):
        catalog._load_catalog(document)


def test_catalog_rejects_continuous_shape_product_that_overflows_usize():
    document = manifest()
    counter = environment(document, "counter")
    counter["capabilities"]["encoding"]["action"] = {"kind": "continuous_f32_little_endian"}
    counter["capabilities"]["agents"]["agents"][0]["action_space"] = {
        "kind": "continuous",
        "low": [-1.0],
        "high": [1.0],
        "shape": [catalog.U32_MAX, catalog.U32_MAX, 2],
    }

    with pytest.raises(RuntimeError, match="product overflows usize"):
        catalog._load_catalog(document)


@pytest.mark.parametrize("invalid_bound", [catalog.F32_MAX * 2, 10**1000])
def test_catalog_rejects_continuous_bounds_outside_finite_f32(invalid_bound):
    document = manifest()
    counter = environment(document, "counter")
    counter["capabilities"]["encoding"]["action"] = {"kind": "continuous_f32_little_endian"}
    counter["capabilities"]["agents"]["agents"][0]["action_space"] = {
        "kind": "continuous",
        "low": [0.0],
        "high": [invalid_bound],
        "shape": [1],
    }

    with pytest.raises(RuntimeError, match="finite f32"):
        catalog._load_catalog(document)


def test_catalog_rejects_float_manifest_schema_even_when_numerically_equal():
    document = manifest()
    document["schema_version"] = float(catalog.MANIFEST_SCHEMA_VERSION)

    with pytest.raises(RuntimeError, match="u32 integer"):
        catalog._load_catalog(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("information_model", "omniscient-ish"),
        ("planning_state_model", "maybe_snapshot"),
        ("transition_dynamics", "chaotic"),
    ],
)
def test_catalog_rejects_unknown_semantic_enums(field, value):
    document = manifest()
    environment(document, "counter")["capabilities"]["semantics"][field] = value

    with pytest.raises(RuntimeError, match=field):
        catalog._load_catalog(document)


def test_catalog_rejects_inconsistent_chance_and_dynamics():
    document = manifest()
    environment(document, "counter")["capabilities"]["semantics"]["chance_model"] = (
        "environment_sampled"
    )

    with pytest.raises(RuntimeError, match="inconsistent stochasticity"):
        catalog._load_catalog(document)


def test_catalog_rejects_environment_sampled_chance_as_complete_snapshot():
    document = manifest()
    semantics = environment(document, "counter")["capabilities"]["semantics"]
    semantics["transition_dynamics"] = "stochastic"
    semantics["chance_model"] = "environment_sampled"

    with pytest.raises(RuntimeError, match="runtime RNG state"):
        catalog._load_catalog(document)

    semantics["planning_state_model"] = "external_state"
    catalog._load_catalog(document)


def test_catalog_requires_custom_joint_actions_for_simultaneous_decisions():
    document = manifest()
    capabilities = environment(document, "counter")["capabilities"]
    capabilities["semantics"]["turn_model"] = {"kind": "simultaneous"}

    with pytest.raises(RuntimeError, match="custom joint-action codec"):
        catalog._load_catalog(document)

    capabilities["encoding"]["action"] = {
        "kind": "custom",
        "id": "joint-action:v1",
    }
    catalog._load_catalog(document)


def test_catalog_requires_custom_actions_for_explicit_chance():
    document = manifest()
    capabilities = environment(document, "counter")["capabilities"]
    capabilities["semantics"]["transition_dynamics"] = "stochastic"
    capabilities["semantics"]["chance_model"] = "explicit"

    with pytest.raises(RuntimeError, match="custom chance/agent action codec"):
        catalog._load_catalog(document)

    capabilities["encoding"]["action"] = {
        "kind": "custom",
        "id": "chance-or-agent:v1",
    }
    catalog._load_catalog(document)


def test_catalog_rejects_string_boolean_instead_of_coercing_it():
    document = manifest()
    document["environments"][0]["algorithm_profiles"]["alphazero_board_v1"]["compatible"] = "false"

    with pytest.raises(RuntimeError, match="expected a boolean"):
        catalog._load_catalog(document)


def test_catalog_rejects_missing_contract_fields():
    document = manifest()
    document["environments"][0]["capabilities"].pop("contract_version")

    with pytest.raises(RuntimeError, match="contract_version"):
        catalog._load_catalog(document)


def test_catalog_rejects_zero_sized_tensor_dimension():
    document = manifest()
    observation = document["environments"][0]["capabilities"]["encoding"]["observation"]
    observation["spec"]["dimensions"][0]["size"] = 0

    with pytest.raises(RuntimeError, match="u32 integer between 1"):
        catalog._load_catalog(document)


def test_catalog_rejects_single_agent_semantics_with_multiple_fixed_agents():
    document = manifest()
    agents = environment(document, "counter")["capabilities"]["agents"]["agents"]
    extra = deepcopy(agents[0])
    extra["id"] = 8
    agents.append(extra)

    with pytest.raises(RuntimeError, match="exactly one fixed agent"):
        catalog._load_catalog(document)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda item: item["capabilities"]["agents"]["agents"][0].pop("action_availability"),
            "action_availability",
        ),
        (
            lambda item: item["capabilities"]["agents"]["agents"][0].update(
                action_availability={"kind": "sometimes"}
            ),
            "unsupported action availability",
        ),
        (
            lambda item: item["capabilities"]["encoding"]["observation"]["spec"][
                "dimensions"
            ].append(
                dict(item["capabilities"]["encoding"]["observation"]["spec"]["dimensions"][0])
            ),
            "unique named dimensions",
        ),
    ],
)
def test_catalog_rejects_malformed_board_profiles(mutate, message):
    document = manifest()
    mutate(environment(document, "connect4"))

    with pytest.raises(RuntimeError, match=message):
        catalog._load_catalog(document)


def test_catalog_requires_one_report_for_every_declared_algorithm():
    document = manifest()
    document["environments"][0]["algorithm_profiles"] = {}

    with pytest.raises(RuntimeError, match="exactly match the algorithm catalog"):
        catalog._load_catalog(document)


def test_catalog_rejects_unknown_fields_without_a_schema_bump():
    document = manifest()
    document["algorithms"][0]["components"]["legacy_loader"] = "guess_v0"

    with pytest.raises(RuntimeError, match="legacy_loader"):
        catalog._load_catalog(document)


def test_catalog_rejects_unsafe_environment_namespace_segments():
    document = manifest()
    counter = environment(document, "counter")
    counter["metadata"]["id"] = "../Counter"
    counter["capabilities"]["id"]["env_id"] = "../Counter"
    counter["algorithm_profiles"]["alphazero_board_v1"]["env_id"] = "../Counter"

    with pytest.raises(RuntimeError, match="lowercase ASCII"):
        catalog._load_catalog(document)


def test_catalog_rejects_whitespace_only_rust_contract_strings():
    document = manifest()
    connect4 = environment(document, "connect4")
    connect4["metadata"]["board"]["players"][0]["name"] = "   "

    with pytest.raises(RuntimeError, match="non-blank string"):
        catalog._load_catalog(document)
