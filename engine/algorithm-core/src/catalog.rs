use engine_core::typed::Capabilities;
use engine_core::{EngineContext, EnvironmentMetadata};
use serde::Serialize;
use std::str::FromStr;

use crate::compatibility::{alphazero_compatibility, dqn_compatibility};
use crate::{AlgorithmError, CompatibilityReport, MODEL_ARTIFACT_SCHEMA_VERSION};

/// Canonical identifier for the board-game AlphaZero cartridge.
pub const ALPHAZERO_BOARD_V1_ID: &str = "alphazero_board_v1";
/// Canonical identifier for the single-agent discrete DQN cartridge.
pub const DQN_V1_ID: &str = "dqn_v1";

const ALPHAZERO_REQUIREMENTS: &[&str] = &[
    "two fixed players",
    "one active player at a time with alternating turns",
    "canonical little-endian u32 discrete actions",
    "fixed-shape little-endian f32 spatial observations with first-class discrete legal-action masks",
    "complete deterministic planning snapshots after reset",
    "perfect-information Markov observations",
    "terminal-only zero-sum rewards emitted for every agent",
];

const DQN_REQUIREMENTS: &[&str] = &[
    "one fixed agent acting alone",
    "canonical little-endian u32 discrete actions",
    "fixed-shape little-endian f32 observations",
    "perfect-information Markov observations",
    "finite episode horizon with general per-agent rewards",
    "no explicit chance decisions",
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

/// Built-in algorithms available for runtime dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinAlgorithm {
    AlphaZeroBoardV1,
    DqnV1,
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

const DQN_DESCRIPTOR: AlgorithmDescriptor = AlgorithmDescriptor {
    id: DQN_V1_ID,
    version: 1,
    model_artifact_schema_version: MODEL_ARTIFACT_SCHEMA_VERSION,
    display_name: "DQN single-agent discrete profile",
    components: AlgorithmComponents {
        collector: "dqn_epsilon_greedy_v1",
        learner: "dqn_q_learning_v1",
        orchestration: "off_policy_dqn_v1",
        experience_schema: "dqn_transition_v1",
        model_contract: "onnx_q_values_v1",
        evaluation_suite: "single_agent_return_v1",
        serving: "dqn_greedy_v1",
    },
    requirements: DQN_REQUIREMENTS,
};

const BUILTIN_ALGORITHMS: &[BuiltinAlgorithm] =
    &[BuiltinAlgorithm::AlphaZeroBoardV1, BuiltinAlgorithm::DqnV1];

impl BuiltinAlgorithm {
    pub const fn descriptor(self) -> &'static AlgorithmDescriptor {
        match self {
            Self::AlphaZeroBoardV1 => &ALPHAZERO_DESCRIPTOR,
            Self::DqnV1 => &DQN_DESCRIPTOR,
        }
    }

    pub fn compatibility(self, context: &EngineContext) -> CompatibilityReport {
        self.compatibility_for(&context.capabilities(), &context.metadata())
    }

    pub fn compatibility_for(
        self,
        capabilities: &Capabilities,
        metadata: &EnvironmentMetadata,
    ) -> CompatibilityReport {
        match self {
            Self::AlphaZeroBoardV1 => alphazero_compatibility(capabilities, metadata),
            Self::DqnV1 => dqn_compatibility(capabilities, metadata),
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

pub fn resolve_algorithm(id: &str) -> Result<BuiltinAlgorithm, AlgorithmError> {
    id.parse()
}

pub fn algorithm_descriptors() -> Vec<&'static AlgorithmDescriptor> {
    BUILTIN_ALGORITHMS
        .iter()
        .map(|algorithm| algorithm.descriptor())
        .collect()
}

pub fn available_algorithm_ids() -> Vec<&'static str> {
    BUILTIN_ALGORITHMS
        .iter()
        .map(|algorithm| algorithm.descriptor().id)
        .collect()
}

pub fn compatible_algorithm_ids(context: &EngineContext) -> Vec<&'static str> {
    BUILTIN_ALGORITHMS
        .iter()
        .copied()
        .filter(|algorithm| algorithm.compatibility(context).compatible)
        .map(|algorithm| algorithm.descriptor().id)
        .collect()
}

pub fn compatibility_reports(context: &EngineContext) -> Vec<CompatibilityReport> {
    BUILTIN_ALGORITHMS
        .iter()
        .copied()
        .map(|algorithm| algorithm.compatibility(context))
        .collect()
}
