use super::*;
use engine_core::board_profile::{BoardGameMetadata, BoardPlayerMetadata};
use engine_core::typed::{
    ActionEncoding, AgentId, AgentModel, Capabilities, ChanceModel, Encoding, EngineId,
    EnvironmentSemantics, InformationModel, ObservationEncoding, PlanningStateModel, RewardModel,
    TensorSpec, TransitionDynamics, TurnModel, WIRE_ENCODING_SCHEMA_VERSION,
};
use engine_core::{ActionSpace, EnvironmentMetadata};
use std::path::PathBuf;

use crate::compatibility::{ALPHAZERO_UNVERIFIED_ASSUMPTIONS, DQN_UNVERIFIED_ASSUMPTIONS};

fn compatible_contract() -> (Capabilities, EnvironmentMetadata) {
    let metadata = EnvironmentMetadata::new("test", "Test").with_board(
        BoardGameMetadata::new(3, 3).with_players(vec![
            BoardPlayerMetadata::new("A", "A"),
            BoardPlayerMetadata::new("B", "B"),
        ]),
    );
    let capabilities = Capabilities {
        id: EngineId {
            env_id: "test".into(),
            build_id: "test".into(),
        },
        contract_version: 1,
        encoding: Encoding::discrete_u32_le(
            "test_state:v1",
            TensorSpec::f32_fixed([("channel", 2), ("row", 3), ("column", 3)]),
        ),
        semantics:
            EnvironmentSemantics::deterministic_alternating_perfect_information_terminal_zero_sum(),
        max_horizon: Some(9),
        agents: AgentModel::fixed_homogeneous_masked(
            [AgentId(1), AgentId(2)],
            ActionSpace::discrete(9),
        ),
        preferred_batch: 1,
    };
    (capabilities, metadata)
}

fn dqn_compatible_contract() -> (Capabilities, EnvironmentMetadata) {
    let metadata = EnvironmentMetadata::new("counter", "Counter");
    let capabilities = Capabilities {
        id: EngineId {
            env_id: "counter".into(),
            build_id: "test".into(),
        },
        contract_version: 2,
        encoding: Encoding::discrete_u32_le(
            "counter_state:v1",
            TensorSpec::f32_fixed([("feature", 2)]),
        ),
        semantics: EnvironmentSemantics::deterministic_single_agent_general_reward(),
        max_horizon: Some(8),
        agents: AgentModel::fixed_homogeneous([AgentId(0)], ActionSpace::discrete(2)),
        preferred_batch: 32,
    };
    (capabilities, metadata)
}

#[test]
fn canonical_algorithm_resolves_and_describes_every_component() {
    let algorithm = resolve_algorithm(ALPHAZERO_BOARD_V1_ID).unwrap();
    let descriptor = algorithm.descriptor();

    assert_eq!(descriptor.id, ALPHAZERO_BOARD_V1_ID);
    assert!(!descriptor.components.collector.is_empty());
    assert!(!descriptor.components.learner.is_empty());
    assert_eq!(
        descriptor.components.orchestration,
        "synchronized_alphazero_v1"
    );
    assert!(!descriptor.components.experience_schema.is_empty());
    assert!(!descriptor.components.model_contract.is_empty());
    assert!(!descriptor.components.evaluation_suite.is_empty());
    assert!(!descriptor.components.serving.is_empty());
    assert_eq!(
        descriptor.model_artifact_schema_version,
        MODEL_ARTIFACT_SCHEMA_VERSION
    );
}

#[test]
fn dqn_descriptor_and_single_agent_contract_are_explicit() {
    let algorithm = resolve_algorithm(DQN_V1_ID).unwrap();
    let descriptor = algorithm.descriptor();
    assert_eq!(descriptor.id, DQN_V1_ID);
    assert_eq!(descriptor.components.model_contract, "onnx_q_values_v1");
    assert_eq!(descriptor.components.experience_schema, "dqn_transition_v1");

    let (capabilities, metadata) = dqn_compatible_contract();
    let report = algorithm.compatibility_for(&capabilities, &metadata);
    assert!(report.compatible, "{:?}", report.issues);
    assert_eq!(report.unverified_assumptions, DQN_UNVERIFIED_ASSUMPTIONS);

    let alpha = BuiltinAlgorithm::AlphaZeroBoardV1.compatibility_for(&capabilities, &metadata);
    assert!(!alpha.compatible);
}

#[test]
fn dqn_rejects_multi_agent_continuous_and_explicit_chance_contracts() {
    let (mut capabilities, metadata) = dqn_compatible_contract();
    capabilities.agents = AgentModel::fixed_homogeneous(
        [AgentId(0), AgentId(1)],
        ActionSpace::Continuous {
            low: vec![-1.0],
            high: vec![1.0],
            shape: vec![1],
        },
    );
    capabilities.semantics.chance_model = ChanceModel::Explicit;

    let report = BuiltinAlgorithm::DqnV1.compatibility_for(&capabilities, &metadata);
    let codes = report
        .issues
        .iter()
        .map(|issue| issue.code)
        .collect::<Vec<_>>();
    assert!(codes.contains(&"agents.fixed_count"));
    assert!(codes.contains(&"semantics.explicit_chance"));
}

#[test]
fn dqn_compatibility_issue_order_and_messages_are_stable() {
    let (mut capabilities, metadata) = dqn_compatible_contract();
    capabilities.contract_version = 0;
    capabilities.encoding.schema_version = WIRE_ENCODING_SCHEMA_VERSION + 1;
    capabilities.semantics.turn_model = TurnModel::Simultaneous;
    capabilities.max_horizon = None;

    let report = BuiltinAlgorithm::DqnV1.compatibility_for(&capabilities, &metadata);
    let actual = report
        .issues
        .iter()
        .map(|issue| (issue.code, issue.message.as_str()))
        .collect::<Vec<_>>();

    assert_eq!(
        actual,
        vec![
            (
                "identity.contract_version",
                "requires a non-zero immutable environment contract version",
            ),
            (
                "encoding.schema_version",
                "requires wire encoding schema version 1, got 2",
            ),
            (
                "semantics.turn_model",
                "requires single-agent turns, got Simultaneous",
            ),
            (
                "episode.max_horizon",
                "requires a finite non-zero maximum horizon",
            ),
        ]
    );
}

#[test]
fn model_artifact_contract_is_derived_from_the_algorithm_descriptor() {
    let descriptor = resolve_algorithm(ALPHAZERO_BOARD_V1_ID)
        .unwrap()
        .descriptor();
    let identity = descriptor.model_artifact_contract("connect4", 7);

    assert_eq!(identity.schema_version, MODEL_ARTIFACT_SCHEMA_VERSION);
    assert_eq!(identity.algorithm_id, ALPHAZERO_BOARD_V1_ID);
    assert_eq!(identity.model_contract, "onnx_policy_value_v1");
    assert_eq!(identity.env_id, "connect4");
    assert_eq!(identity.env_contract_version, 7);
    assert_eq!(
        identity.required_metadata(),
        [
            (MODEL_METADATA_SCHEMA_VERSION, "1".to_string()),
            (
                MODEL_METADATA_ALGORITHM_ID,
                ALPHAZERO_BOARD_V1_ID.to_string()
            ),
            (MODEL_METADATA_CONTRACT, "onnx_policy_value_v1".to_string()),
            (MODEL_METADATA_ENV_ID, "connect4".to_string()),
            (MODEL_METADATA_ENV_CONTRACT_VERSION, "7".to_string()),
        ]
    );
}

#[test]
fn runtime_profile_owns_one_canonical_namespace() {
    let profile = RuntimeProfile::new(ALPHAZERO_BOARD_V1_ID, "connect4", 3).unwrap();

    assert_eq!(
        profile.storage_prefix(),
        "profiles/alphazero_board_v1/connect4/v3"
    );
    assert_eq!(
        profile.model_dir("/data"),
        PathBuf::from("/data/profiles/alphazero_board_v1/connect4/v3/models")
    );
    assert_eq!(
        profile.model_prefix(),
        "profiles/alphazero_board_v1/connect4/v3/models"
    );
}

#[test]
fn runtime_profile_rejects_unsafe_or_ambiguous_segments() {
    for env_id in ["", "Connect4", "../connect4", "connect/4", "connect.4"] {
        assert!(RuntimeProfile::new(ALPHAZERO_BOARD_V1_ID, env_id, 1).is_err());
    }
    assert!(RuntimeProfile::new(ALPHAZERO_BOARD_V1_ID, "connect4", 0).is_err());
}

#[test]
fn board_profile_is_descriptor_compatible_with_behavioral_assumptions_exposed() {
    let (capabilities, metadata) = compatible_contract();
    let report = BuiltinAlgorithm::AlphaZeroBoardV1.compatibility_for(&capabilities, &metadata);

    assert!(report.compatible, "{:?}", report.issues);
    assert!(report.issues.is_empty());
    assert_eq!(
        report.unverified_assumptions,
        ALPHAZERO_UNVERIFIED_ASSUMPTIONS
    );
    report.require_compatible().unwrap();
}

#[test]
fn semantic_contract_rejects_every_non_alphazero_declaration() {
    let (mut capabilities, metadata) = compatible_contract();
    capabilities.semantics = EnvironmentSemantics {
        turn_model: TurnModel::Simultaneous,
        information_model: InformationModel::PartiallyObserved,
        planning_state_model: PlanningStateModel::ExternalState,
        transition_dynamics: TransitionDynamics::Stochastic,
        chance_model: ChanceModel::Explicit,
        reward_model: RewardModel::General,
    };

    let report = BuiltinAlgorithm::AlphaZeroBoardV1.compatibility_for(&capabilities, &metadata);
    let codes: Vec<_> = report.issues.iter().map(|issue| issue.code).collect();

    assert!(codes.contains(&"semantics.turn_model"));
    assert!(codes.contains(&"semantics.information_model"));
    assert!(codes.contains(&"semantics.planning_state_model"));
    assert!(codes.contains(&"semantics.transition_dynamics"));
    assert!(codes.contains(&"semantics.chance_model"));
    assert!(codes.contains(&"semantics.reward_model"));
}

#[test]
fn wire_contract_rejects_noncanonical_codecs_counts_and_versions() {
    let (mut capabilities, metadata) = compatible_contract();
    capabilities.contract_version = 0;
    capabilities.encoding.schema_version = WIRE_ENCODING_SCHEMA_VERSION + 1;
    capabilities.encoding.action = ActionEncoding::Custom {
        id: "discrete_u32_little_endian_extended".into(),
    };
    capabilities.encoding.observation = ObservationEncoding::Tensor {
        spec: TensorSpec::f32_fixed([("feature", 28)]),
    };

    let report = BuiltinAlgorithm::AlphaZeroBoardV1.compatibility_for(&capabilities, &metadata);
    let codes: Vec<_> = report.issues.iter().map(|issue| issue.code).collect();

    assert!(codes.contains(&"identity.contract_version"));
    assert!(codes.contains(&"encoding.schema_version"));
    assert!(codes.contains(&"action.encoding"));
    assert!(codes.contains(&"observation.encoding"));

    capabilities.contract_version = 1;
    capabilities.encoding.schema_version = WIRE_ENCODING_SCHEMA_VERSION;
    capabilities.encoding.action = ActionEncoding::DiscreteU32LittleEndian;
    capabilities.encoding.observation = ObservationEncoding::Custom {
        id: "f32_little_endian_extended".into(),
    };
    let report = BuiltinAlgorithm::AlphaZeroBoardV1.compatibility_for(&capabilities, &metadata);
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.code == "observation.encoding"));
}

#[test]
fn incompatibility_report_collects_all_action_agent_and_layout_failures() {
    let (mut capabilities, mut metadata) = compatible_contract();
    capabilities.agents = AgentModel::fixed_homogeneous_masked(
        [AgentId(1), AgentId(2)],
        ActionSpace::Continuous {
            low: vec![-1.0],
            high: vec![1.0],
            shape: vec![1],
        },
    );
    let board = metadata.board.as_mut().unwrap();
    board.players.pop();
    capabilities.encoding.observation = ObservationEncoding::Tensor {
        spec: TensorSpec::f32_fixed([("feature", 7)]),
    };

    let report = BuiltinAlgorithm::AlphaZeroBoardV1.compatibility_for(&capabilities, &metadata);
    let codes: Vec<_> = report.issues.iter().map(|issue| issue.code).collect();

    assert!(!report.compatible);
    assert!(codes.contains(&"action_space.profile_mismatch"));
    assert!(codes.contains(&"agents.player_count"));
    assert!(codes.contains(&"observation.encoding"));
    assert!(report.require_compatible().is_err());
}

#[test]
fn unknown_algorithm_error_lists_the_installed_profile() {
    let error = resolve_algorithm("ppo").unwrap_err().to_string();
    assert!(error.contains("ppo"));
    assert!(error.contains(ALPHAZERO_BOARD_V1_ID));
    assert!(error.contains(DQN_V1_ID));
}
