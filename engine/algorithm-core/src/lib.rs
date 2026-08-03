//! Algorithm identity, component contracts, and environment compatibility.
//!
//! An environment being registered does not imply that every learner can use
//! it. This crate is the small shared catalog that lets the actor, trainer
//! manifest, and evaluator agree on which algorithm is being requested and
//! why an environment is (or is not) compatible with it.

use engine_core::typed::{
    ActionEncoding, AgentId, AgentModel, Capabilities, ChanceModel, InformationModel,
    ObservationEncoding, PlanningStateModel, RewardModel, SequentialTurnOrder, TransitionDynamics,
    TurnModel, WIRE_ENCODING_SCHEMA_VERSION,
};
use engine_core::{ActionSpace, EngineContext, EnvironmentMetadata};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Canonical identifier for the board-game AlphaZero cartridge.
pub const ALPHAZERO_BOARD_V1_ID: &str = "alphazero_board_v1";
pub const PROFILE_NAMESPACE_DIR: &str = "profiles";

/// Version of the required model-artifact identity metadata schema.
pub const MODEL_ARTIFACT_SCHEMA_VERSION: u32 = 1;

/// ONNX custom-metadata keys used to identify a model artifact.
pub const MODEL_METADATA_SCHEMA_VERSION: &str = "cartridge.schema_version";
pub const MODEL_METADATA_ALGORITHM_ID: &str = "cartridge.algorithm_id";
pub const MODEL_METADATA_CONTRACT: &str = "cartridge.model_contract";
pub const MODEL_METADATA_ENV_ID: &str = "cartridge.env_id";
pub const MODEL_METADATA_ENV_CONTRACT_VERSION: &str = "cartridge.env_contract_version";

const ALPHAZERO_REQUIREMENTS: &[&str] = &[
    "two fixed players",
    "one active player at a time with alternating turns",
    "canonical little-endian u32 discrete actions",
    "fixed-size little-endian f32 spatial observations with an embedded legal mask and two-element player indicator",
    "complete deterministic planning snapshots after reset",
    "perfect-information Markov observations",
    "terminal-only zero-sum rewards emitted for every agent",
];

const ALPHAZERO_UNVERIFIED_ASSUMPTIONS: &[&str] = &[
    "state encoding is a complete round-trippable planning snapshot",
    "observations are Markov and reveal all strategically relevant state",
    "running transitions alternate seats 1 and 2",
    "the embedded legal mask exactly matches actions accepted by the environment",
];

/// Stable names for the independently replaceable parts of an algorithm.
///
/// These are language-neutral contract identifiers, not Rust type names. The
/// Rust actor and evaluator and the Python trainer each bind their local
/// implementation to the same descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AlgorithmComponents {
    pub collector: &'static str,
    pub learner: &'static str,
    pub orchestration: &'static str,
    pub experience_schema: &'static str,
    pub model_contract: &'static str,
    pub evaluation_suite: &'static str,
    pub serving: &'static str,
}

/// Public description of an installed algorithm cartridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AlgorithmDescriptor {
    pub id: &'static str,
    pub version: u32,
    pub model_artifact_schema_version: u32,
    pub display_name: &'static str,
    pub components: AlgorithmComponents,
    pub requirements: &'static [&'static str],
}

impl AlgorithmDescriptor {
    /// Build the exact compatibility contract every model consumer must
    /// require for an artifact intended for `env_id`.
    pub fn model_artifact_contract(
        &self,
        env_id: impl Into<String>,
        env_contract_version: u32,
    ) -> ModelArtifactContract {
        assert!(
            env_contract_version > 0,
            "environment contract version must be greater than zero"
        );
        ModelArtifactContract {
            schema_version: self.model_artifact_schema_version,
            algorithm_id: self.id.to_string(),
            model_contract: self.components.model_contract.to_string(),
            env_id: env_id.into(),
            env_contract_version,
        }
    }
}

/// Exact compatibility contract required from a serialized model artifact.
///
/// The fields are deliberately non-optional: model loading is never allowed
/// to infer an algorithm, model contract, or environment from tensor shapes or
/// from the process that happened to open the file. The content identity of a
/// specific set of weights is the checkpoint-manifest SHA-256, not this value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelArtifactContract {
    pub schema_version: u32,
    pub algorithm_id: String,
    pub model_contract: String,
    pub env_id: String,
    pub env_contract_version: u32,
}

impl ModelArtifactContract {
    /// Required ONNX custom metadata in stable key order.
    pub fn required_metadata(&self) -> [(&'static str, String); 5] {
        [
            (
                MODEL_METADATA_SCHEMA_VERSION,
                self.schema_version.to_string(),
            ),
            (MODEL_METADATA_ALGORITHM_ID, self.algorithm_id.clone()),
            (MODEL_METADATA_CONTRACT, self.model_contract.clone()),
            (MODEL_METADATA_ENV_ID, self.env_id.clone()),
            (
                MODEL_METADATA_ENV_CONTRACT_VERSION,
                self.env_contract_version.to_string(),
            ),
        ]
    }
}

/// Canonical namespace for every mutable/runtime artifact belonging to one
/// algorithm/environment contract revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RuntimeProfile {
    pub algorithm_id: String,
    pub env_id: String,
    pub env_contract_version: u32,
}

impl RuntimeProfile {
    pub fn new(
        algorithm_id: impl Into<String>,
        env_id: impl Into<String>,
        env_contract_version: u32,
    ) -> Result<Self, AlgorithmError> {
        let profile = Self {
            algorithm_id: algorithm_id.into(),
            env_id: env_id.into(),
            env_contract_version,
        };
        validate_profile_segment("algorithm_id", &profile.algorithm_id)?;
        validate_profile_segment("env_id", &profile.env_id)?;
        if profile.env_contract_version == 0 {
            return Err(AlgorithmError::InvalidRuntimeProfile {
                field: "env_contract_version",
                value: "0".to_string(),
            });
        }
        Ok(profile)
    }

    /// Language-neutral slash-separated namespace used by filesystems and
    /// object stores alike.
    pub fn storage_prefix(&self) -> String {
        format!(
            "{PROFILE_NAMESPACE_DIR}/{}/{}/v{}",
            self.algorithm_id, self.env_id, self.env_contract_version
        )
    }

    pub fn data_dir(&self, data_root: impl AsRef<Path>) -> PathBuf {
        data_root.as_ref().join(self.storage_prefix())
    }

    pub fn model_dir(&self, data_root: impl AsRef<Path>) -> PathBuf {
        self.data_dir(data_root).join("models")
    }

    pub fn model_prefix(&self) -> String {
        format!("{}/models", self.storage_prefix())
    }
}

fn validate_profile_segment(field: &'static str, value: &str) -> Result<(), AlgorithmError> {
    if value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
        })
    {
        return Err(AlgorithmError::InvalidRuntimeProfile {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

/// A machine-checkable incompatibility between an algorithm and environment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompatibilityIssue {
    pub code: &'static str,
    pub message: String,
}

/// Full compatibility result, including any remaining explicitly identified
/// assumptions that the environment contract cannot prove mechanically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompatibilityReport {
    pub algorithm_id: &'static str,
    pub env_id: String,
    pub compatible: bool,
    pub issues: Vec<CompatibilityIssue>,
    pub unverified_assumptions: &'static [&'static str],
}

impl CompatibilityReport {
    /// Turn an explanatory report into a startup guard.
    pub fn require_compatible(&self) -> Result<(), AlgorithmError> {
        if self.compatible {
            return Ok(());
        }

        let reasons = self
            .issues
            .iter()
            .map(|issue| format!("{}: {}", issue.code, issue.message))
            .collect::<Vec<_>>()
            .join("; ");
        Err(AlgorithmError::Incompatible {
            algorithm_id: self.algorithm_id.to_string(),
            env_id: self.env_id.clone(),
            reasons,
        })
    }
}

/// Built-in algorithms available for runtime dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinAlgorithm {
    AlphaZeroBoardV1,
}

const ALPHAZERO_DESCRIPTOR: AlgorithmDescriptor = AlgorithmDescriptor {
    id: ALPHAZERO_BOARD_V1_ID,
    version: 1,
    model_artifact_schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
    display_name: "AlphaZero board-game profile",
    components: AlgorithmComponents {
        collector: "alphazero_mcts_self_play_v1",
        learner: "alphazero_policy_value_v1",
        orchestration: "synchronized_alphazero_v1",
        experience_schema: "alphazero_transition_v1",
        model_contract: "onnx_policy_value_v1",
        evaluation_suite: "two_player_seat_balanced_v1",
        serving: "alphazero_mcts_web_v1",
    },
    requirements: ALPHAZERO_REQUIREMENTS,
};

const BUILTIN_ALGORITHMS: &[BuiltinAlgorithm] = &[BuiltinAlgorithm::AlphaZeroBoardV1];

impl BuiltinAlgorithm {
    pub const fn descriptor(self) -> &'static AlgorithmDescriptor {
        match self {
            Self::AlphaZeroBoardV1 => &ALPHAZERO_DESCRIPTOR,
        }
    }

    /// Check every structural and semantic requirement represented by the
    /// engine contract.
    pub fn compatibility(self, context: &EngineContext) -> CompatibilityReport {
        self.compatibility_for(&context.capabilities(), &context.metadata())
    }

    /// Compatibility check over descriptors, kept separate for manifest
    /// generation and focused contract tests.
    pub fn compatibility_for(
        self,
        capabilities: &Capabilities,
        metadata: &EnvironmentMetadata,
    ) -> CompatibilityReport {
        match self {
            Self::AlphaZeroBoardV1 => alphazero_compatibility(capabilities, metadata),
        }
    }
}

impl FromStr for BuiltinAlgorithm {
    type Err = AlgorithmError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        BUILTIN_ALGORITHMS
            .iter()
            .copied()
            .find(|algorithm| algorithm.descriptor().id == value)
            .ok_or_else(|| AlgorithmError::UnknownAlgorithm {
                requested: value.to_string(),
                available: available_algorithm_ids().join(", "),
            })
    }
}

/// Resolve an algorithm ID to its runtime dispatch key.
pub fn resolve_algorithm(id: &str) -> Result<BuiltinAlgorithm, AlgorithmError> {
    id.parse()
}

/// Installed descriptors in deterministic order.
pub fn algorithm_descriptors() -> Vec<&'static AlgorithmDescriptor> {
    BUILTIN_ALGORITHMS
        .iter()
        .map(|algorithm| algorithm.descriptor())
        .collect()
}

/// Installed algorithm IDs in deterministic order.
pub fn available_algorithm_ids() -> Vec<&'static str> {
    BUILTIN_ALGORITHMS
        .iter()
        .map(|algorithm| algorithm.descriptor().id)
        .collect()
}

/// Algorithms whose machine-checkable requirements pass for an environment.
pub fn compatible_algorithm_ids(context: &EngineContext) -> Vec<&'static str> {
    BUILTIN_ALGORITHMS
        .iter()
        .copied()
        .filter(|algorithm| algorithm.compatibility(context).compatible)
        .map(|algorithm| algorithm.descriptor().id)
        .collect()
}

/// Compatibility reports for every installed algorithm, in catalog order.
pub fn compatibility_reports(context: &EngineContext) -> Vec<CompatibilityReport> {
    BUILTIN_ALGORITHMS
        .iter()
        .copied()
        .map(|algorithm| algorithm.compatibility(context))
        .collect()
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AlgorithmError {
    #[error("unknown algorithm '{requested}'; available algorithms: {available}")]
    UnknownAlgorithm {
        requested: String,
        available: String,
    },
    #[error("algorithm '{algorithm_id}' is incompatible with environment '{env_id}': {reasons}")]
    Incompatible {
        algorithm_id: String,
        env_id: String,
        reasons: String,
    },
    #[error(
        "invalid runtime profile {field} {value:?}; IDs must contain only lowercase ASCII letters, digits, '_' or '-', and contract versions must be positive"
    )]
    InvalidRuntimeProfile { field: &'static str, value: String },
}

fn alphazero_compatibility(
    capabilities: &Capabilities,
    metadata: &EnvironmentMetadata,
) -> CompatibilityReport {
    let mut issues = Vec::new();
    let mut add_issue = |code, message| issues.push(CompatibilityIssue { code, message });

    if capabilities.id.env_id != metadata.id {
        add_issue(
            "identity.env_id_mismatch",
            format!(
                "capabilities use '{}' but metadata uses '{}'",
                capabilities.id.env_id, metadata.id
            ),
        );
    }
    if capabilities.contract_version == 0 {
        add_issue(
            "identity.contract_version",
            "requires a non-zero immutable environment contract version".to_string(),
        );
    }

    if capabilities.encoding.schema_version != WIRE_ENCODING_SCHEMA_VERSION {
        add_issue(
            "encoding.schema_version",
            format!(
                "requires wire encoding schema version {}, got {}",
                WIRE_ENCODING_SCHEMA_VERSION, capabilities.encoding.schema_version
            ),
        );
    }
    if capabilities.encoding.action != ActionEncoding::DiscreteU32LittleEndian {
        add_issue(
            "action.encoding",
            format!(
                "requires DiscreteU32LittleEndian actions, got {:?}",
                capabilities.encoding.action
            ),
        );
    }

    let fixed_agents = match &capabilities.agents {
        AgentModel::Fixed { agents } => {
            let ids = agents.iter().map(|agent| agent.id).collect::<Vec<_>>();
            if ids != [AgentId(1), AgentId(2)] {
                add_issue(
                    "agents.fixed_seats",
                    format!("requires fixed seats [1, 2], got {ids:?}"),
                );
            }
            Some(agents.as_slice())
        }
        AgentModel::Dynamic { .. } => {
            add_issue(
                "agents.dynamic",
                "requires exactly two fixed agents".to_string(),
            );
            None
        }
    };

    let board = metadata.board.as_ref();
    if board.is_none() {
        add_issue(
            "profile.board_missing",
            "environment does not expose the board-game profile".to_string(),
        );
    }
    if let Some(board) = board {
        if board.players.len() != 2 {
            add_issue(
                "agents.player_count",
                format!("requires exactly 2 players, got {}", board.players.len()),
            );
        }
        if board.width == 0 || board.height == 0 {
            add_issue(
                "observation.board_shape",
                format!(
                    "requires a non-empty board, got {}x{}",
                    board.width, board.height
                ),
            );
        }
        if board.action_count == 0 {
            add_issue(
                "action_space.empty",
                "requires at least one discrete action".to_string(),
            );
        }
        if board.observation.spatial_channels == 0 {
            add_issue(
                "observation.spatial_channels",
                "requires at least one spatial observation channel".to_string(),
            );
        }
        if let Some(agents) = fixed_agents {
            for agent in agents {
                match &agent.action_space {
                    ActionSpace::Discrete { size } if *size as usize == board.action_count => {}
                    other => add_issue(
                        "action_space.profile_mismatch",
                        format!(
                            "agent {} requires Discrete({}), got {other:?}",
                            agent.id.0, board.action_count
                        ),
                    ),
                }
            }
        }

        match board.board_size() {
            Ok(board_size) => match board.observation.spatial_channels.checked_mul(board_size) {
                Some(expected_mask_offset)
                    if board.observation.legal_actions_offset == expected_mask_offset => {}
                Some(expected_mask_offset) => add_issue(
                    "observation.legal_mask_offset",
                    format!(
                        "expected legal mask at {}, got {}",
                        expected_mask_offset, board.observation.legal_actions_offset
                    ),
                ),
                None => add_issue(
                    "observation.layout_overflow",
                    "spatial observation layout overflows the platform size".to_string(),
                ),
            },
            Err(error) => add_issue("observation.board_shape", error.to_string()),
        }
        match board
            .observation
            .legal_actions_offset
            .checked_add(board.action_count)
            .and_then(|elements| elements.checked_add(2))
        {
            Some(expected_elements) if board.observation.elements == expected_elements => {}
            Some(expected_elements) => add_issue(
                "observation.layout_size",
                format!(
                    "expected {expected_elements} f32 values (planes + legal mask + player one-hot), got {}",
                    board.observation.elements
                ),
            ),
            None => add_issue(
                "observation.layout_overflow",
                "observation element count overflows the platform size".to_string(),
            ),
        }
        match &capabilities.encoding.observation {
            ObservationEncoding::F32LittleEndian { elements }
                if *elements == board.observation.elements => {}
            other => add_issue(
                "observation.encoding",
                format!(
                    "requires F32LittleEndian({}), got {other:?}",
                    board.observation.elements
                ),
            ),
        }
    }

    if capabilities.semantics.turn_model
        != (TurnModel::Sequential {
            order: SequentialTurnOrder::Alternating,
        })
    {
        add_issue(
            "semantics.turn_model",
            format!(
                "requires alternating sequential turns, got {:?}",
                capabilities.semantics.turn_model
            ),
        );
    }
    if capabilities.semantics.information_model != InformationModel::PerfectInformationMarkov {
        add_issue(
            "semantics.information_model",
            format!(
                "requires perfect-information Markov observations, got {:?}",
                capabilities.semantics.information_model
            ),
        );
    }
    if capabilities.semantics.planning_state_model != PlanningStateModel::CompleteSnapshot {
        add_issue(
            "semantics.planning_state_model",
            format!(
                "requires complete planning snapshots, got {:?}",
                capabilities.semantics.planning_state_model
            ),
        );
    }
    if capabilities.semantics.transition_dynamics != TransitionDynamics::Deterministic {
        add_issue(
            "semantics.transition_dynamics",
            format!(
                "requires deterministic transitions, got {:?}",
                capabilities.semantics.transition_dynamics
            ),
        );
    }
    if capabilities.semantics.chance_model != ChanceModel::None {
        add_issue(
            "semantics.chance_model",
            format!(
                "requires no chance decisions, got {:?}",
                capabilities.semantics.chance_model
            ),
        );
    }
    if capabilities.semantics.reward_model != RewardModel::TerminalZeroSum {
        add_issue(
            "semantics.reward_model",
            format!(
                "requires terminal zero-sum per-agent rewards, got {:?}",
                capabilities.semantics.reward_model
            ),
        );
    }
    if capabilities.max_horizon.is_none() || capabilities.max_horizon == Some(0) {
        add_issue(
            "episode.max_horizon",
            "requires a finite non-zero maximum horizon".to_string(),
        );
    }

    CompatibilityReport {
        algorithm_id: ALPHAZERO_BOARD_V1_ID,
        env_id: metadata.id.clone(),
        compatible: issues.is_empty(),
        issues,
        unverified_assumptions: ALPHAZERO_UNVERIFIED_ASSUMPTIONS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::board_profile::{BoardGameMetadata, BoardPlayerMetadata};
    use engine_core::typed::{AgentModel, Encoding, EngineId, EnvironmentSemantics};
    use engine_core::EnvironmentMetadata;

    fn compatible_contract() -> (Capabilities, EnvironmentMetadata) {
        let metadata = EnvironmentMetadata::new("test", "Test").with_board(
            BoardGameMetadata::new(3, 3, 9)
                .with_observation(29, 2, 18, false)
                .with_players(vec![
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
            encoding: Encoding::discrete_u32_le_f32_le("test_state:v1", 29),
            semantics:
                EnvironmentSemantics::deterministic_alternating_perfect_information_terminal_zero_sum(),
            max_horizon: Some(9),
            agents: AgentModel::fixed_homogeneous(
                [AgentId(1), AgentId(2)],
                ActionSpace::discrete(9),
            ),
            preferred_batch: 1,
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
        capabilities.encoding.observation = ObservationEncoding::F32LittleEndian { elements: 28 };

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
        capabilities.agents = AgentModel::fixed_homogeneous(
            [AgentId(1), AgentId(2)],
            ActionSpace::Continuous {
                low: vec![-1.0],
                high: vec![1.0],
                shape: vec![1],
            },
        );
        let board = metadata.board.as_mut().unwrap();
        board.players.pop();
        board.observation.elements = 7;

        let report = BuiltinAlgorithm::AlphaZeroBoardV1.compatibility_for(&capabilities, &metadata);
        let codes: Vec<_> = report.issues.iter().map(|issue| issue.code).collect();

        assert!(!report.compatible);
        assert!(codes.contains(&"action_space.profile_mismatch"));
        assert!(codes.contains(&"agents.player_count"));
        assert!(codes.contains(&"observation.layout_size"));
        assert!(report.require_compatible().is_err());
    }

    #[test]
    fn unknown_algorithm_error_lists_the_installed_profile() {
        let error = resolve_algorithm("ppo").unwrap_err().to_string();
        assert!(error.contains("ppo"));
        assert!(error.contains(ALPHAZERO_BOARD_V1_ID));
    }
}
