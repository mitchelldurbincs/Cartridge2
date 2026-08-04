use engine_core::typed::{
    ActionAvailabilityContract, AgentModel, Capabilities, ChanceModel, InformationModel,
    ObservationEncoding, RewardModel, TensorDType, TurnModel,
};
use engine_core::{ActionSpace, EnvironmentMetadata};

use super::{
    add_horizon_issue, add_identity_and_wire_issues, add_issue, CompatibilityIssue,
    CompatibilityReport,
};
use crate::DQN_V1_ID;

pub(crate) const DQN_UNVERIFIED_ASSUMPTIONS: &[&str] = &[
    "observations contain enough information for a feed-forward Q-network",
    "the declared action availability exactly matches accepted actions",
];

pub(crate) fn dqn_compatibility(
    capabilities: &Capabilities,
    metadata: &EnvironmentMetadata,
) -> CompatibilityReport {
    let mut issues = Vec::new();
    add_identity_and_wire_issues(capabilities, metadata, &mut issues);
    add_observation_issue(capabilities, &mut issues);
    add_agent_issues(capabilities, &mut issues);
    add_semantic_issues(capabilities, &mut issues);
    add_horizon_issue(capabilities, &mut issues);
    CompatibilityReport {
        algorithm_id: DQN_V1_ID,
        env_id: metadata.id.clone(),
        compatible: issues.is_empty(),
        issues,
        unverified_assumptions: DQN_UNVERIFIED_ASSUMPTIONS,
    }
}

fn add_observation_issue(capabilities: &Capabilities, issues: &mut Vec<CompatibilityIssue>) {
    let valid = matches!(
        &capabilities.encoding.observation,
        ObservationEncoding::Tensor { spec }
            if spec.dtype == TensorDType::F32LittleEndian
                && !spec.dimensions.is_empty()
                && spec
                    .dimensions
                    .iter()
                    .all(|dimension| dimension.size.is_some_and(|size| size > 0))
    );
    if !valid {
        add_issue(
            issues,
            "observation.encoding",
            format!(
                "requires a fixed non-empty f32 tensor, got {:?}",
                capabilities.encoding.observation
            ),
        );
    }
}

fn add_agent_issues(capabilities: &Capabilities, issues: &mut Vec<CompatibilityIssue>) {
    match &capabilities.agents {
        AgentModel::Fixed { agents } if agents.len() == 1 => {
            let agent = &agents[0];
            if !matches!(&agent.action_space, ActionSpace::Discrete { size } if *size > 0) {
                add_issue(
                    issues,
                    "action_space.discrete",
                    format!(
                        "requires one non-empty discrete action space, got {:?}",
                        agent.action_space
                    ),
                );
            }
            if !matches!(
                &agent.action_availability,
                ActionAvailabilityContract::All | ActionAvailabilityContract::DiscreteMask
            ) {
                add_issue(
                    issues,
                    "action_availability.unsupported",
                    format!(
                        "requires all-actions or discrete-mask availability, got {:?}",
                        agent.action_availability
                    ),
                );
            }
        }
        AgentModel::Fixed { agents } => add_issue(
            issues,
            "agents.fixed_count",
            format!("requires exactly one fixed agent, got {}", agents.len()),
        ),
        AgentModel::Dynamic { .. } => add_issue(
            issues,
            "agents.dynamic",
            "requires exactly one fixed agent".to_string(),
        ),
    }
}

fn add_semantic_issues(capabilities: &Capabilities, issues: &mut Vec<CompatibilityIssue>) {
    let semantics = &capabilities.semantics;
    if semantics.turn_model != TurnModel::SingleAgent {
        add_issue(
            issues,
            "semantics.turn_model",
            format!(
                "requires single-agent turns, got {:?}",
                semantics.turn_model
            ),
        );
    }
    if semantics.information_model != InformationModel::PerfectInformationMarkov {
        add_issue(
            issues,
            "semantics.information_model",
            format!(
                "requires Markov observations, got {:?}",
                semantics.information_model
            ),
        );
    }
    if semantics.chance_model == ChanceModel::Explicit {
        add_issue(
            issues,
            "semantics.explicit_chance",
            "explicit chance decisions are not supported by the DQN collector".to_string(),
        );
    }
    if semantics.reward_model != RewardModel::General {
        add_issue(
            issues,
            "semantics.reward_model",
            format!(
                "requires general per-agent rewards, got {:?}",
                semantics.reward_model
            ),
        );
    }
}
