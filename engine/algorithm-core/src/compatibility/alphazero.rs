use engine_core::typed::{
    ActionAvailabilityContract, AgentId, AgentModel, Capabilities, ChanceModel, InformationModel,
    ObservationEncoding, PlanningStateModel, RewardModel, SequentialTurnOrder, TensorDType,
    TransitionDynamics, TurnModel,
};
use engine_core::{ActionSpace, EnvironmentMetadata};

use super::{
    add_horizon_issue, add_identity_and_wire_issues, add_issue, CompatibilityIssue,
    CompatibilityReport,
};
use crate::ALPHAZERO_BOARD_V1_ID;

pub(crate) const ALPHAZERO_UNVERIFIED_ASSUMPTIONS: &[&str] = &[
    "state encoding is a complete round-trippable planning snapshot",
    "observations are Markov and reveal all strategically relevant state",
    "running transitions alternate seats 1 and 2",
    "the decision legal mask exactly matches actions accepted by the environment",
];

pub(crate) fn alphazero_compatibility(
    capabilities: &Capabilities,
    metadata: &EnvironmentMetadata,
) -> CompatibilityReport {
    let mut issues = Vec::new();
    add_identity_and_wire_issues(capabilities, metadata, &mut issues);
    add_agent_issues(capabilities, &mut issues);
    add_board_issues(capabilities, metadata, &mut issues);
    add_semantic_issues(capabilities, &mut issues);
    add_horizon_issue(capabilities, &mut issues);
    CompatibilityReport {
        algorithm_id: ALPHAZERO_BOARD_V1_ID,
        env_id: metadata.id.clone(),
        compatible: issues.is_empty(),
        issues,
        unverified_assumptions: ALPHAZERO_UNVERIFIED_ASSUMPTIONS,
    }
}

fn add_agent_issues(capabilities: &Capabilities, issues: &mut Vec<CompatibilityIssue>) {
    let fixed_agents = match &capabilities.agents {
        AgentModel::Fixed { agents } => {
            let ids = agents.iter().map(|agent| agent.id).collect::<Vec<_>>();
            if ids != [AgentId(1), AgentId(2)] {
                add_issue(
                    issues,
                    "agents.fixed_seats",
                    format!("requires fixed seats [1, 2], got {ids:?}"),
                );
            }
            Some(agents.as_slice())
        }
        AgentModel::Dynamic { .. } => {
            add_issue(
                issues,
                "agents.dynamic",
                "requires exactly two fixed agents".to_string(),
            );
            None
        }
    };
    let action_count = fixed_agents.and_then(|agents| match &agents.first()?.action_space {
        ActionSpace::Discrete { size } => Some(*size),
        _ => None,
    });
    for agent in fixed_agents.into_iter().flatten() {
        if !matches!(&agent.action_space, ActionSpace::Discrete { size } if Some(*size) == action_count)
        {
            add_issue(
                issues,
                "action_space.profile_mismatch",
                format!(
                    "agent {} must share one non-empty discrete action space, got {:?}",
                    agent.id.0, agent.action_space
                ),
            );
        }
        if agent.action_availability != ActionAvailabilityContract::DiscreteMask {
            add_issue(
                issues,
                "action_availability.discrete_mask",
                format!(
                    "agent {} requires first-class discrete-mask availability, got {:?}",
                    agent.id.0, agent.action_availability
                ),
            );
        }
    }
}

fn add_board_issues(
    capabilities: &Capabilities,
    metadata: &EnvironmentMetadata,
    issues: &mut Vec<CompatibilityIssue>,
) {
    let Some(board) = metadata.board.as_ref() else {
        add_issue(
            issues,
            "profile.board_missing",
            "environment does not expose the board-game profile".to_string(),
        );
        return;
    };
    if board.players.len() != 2 {
        add_issue(
            issues,
            "agents.player_count",
            format!("requires exactly 2 players, got {}", board.players.len()),
        );
    }
    if board.width == 0 || board.height == 0 {
        add_issue(
            issues,
            "observation.board_shape",
            format!(
                "requires a non-empty board, got {}x{}",
                board.width, board.height
            ),
        );
    }
    let valid = matches!(
        &capabilities.encoding.observation,
        ObservationEncoding::Tensor { spec }
            if spec.dtype == TensorDType::F32LittleEndian
                && matches!(
                    spec.dimensions.as_slice(),
                    [channel, row, column]
                        if channel.name == "channel"
                            && channel.size.is_some_and(|size| size > 0)
                            && row.name == "row"
                            && row.size == u32::try_from(board.height).ok()
                            && column.name == "column"
                            && column.size == u32::try_from(board.width).ok()
                )
    );
    if !valid {
        add_issue(
            issues,
            "observation.encoding",
            format!(
                "requires a fixed f32 [channel,row,column] tensor matching the {}x{} board, got {:?}",
                board.width, board.height, capabilities.encoding.observation
            ),
        );
    }
}

fn add_semantic_issues(capabilities: &Capabilities, issues: &mut Vec<CompatibilityIssue>) {
    let semantics = &capabilities.semantics;
    if semantics.turn_model
        != (TurnModel::Sequential {
            order: SequentialTurnOrder::Alternating,
        })
    {
        add_issue(
            issues,
            "semantics.turn_model",
            format!(
                "requires alternating sequential turns, got {:?}",
                semantics.turn_model
            ),
        );
    }
    for (invalid, code, requirement, actual) in [
        (
            semantics.information_model != InformationModel::PerfectInformationMarkov,
            "semantics.information_model",
            "perfect-information Markov observations",
            format!("{:?}", semantics.information_model),
        ),
        (
            semantics.planning_state_model != PlanningStateModel::CompleteSnapshot,
            "semantics.planning_state_model",
            "complete planning snapshots",
            format!("{:?}", semantics.planning_state_model),
        ),
        (
            semantics.transition_dynamics != TransitionDynamics::Deterministic,
            "semantics.transition_dynamics",
            "deterministic transitions",
            format!("{:?}", semantics.transition_dynamics),
        ),
        (
            semantics.chance_model != ChanceModel::None,
            "semantics.chance_model",
            "no chance decisions",
            format!("{:?}", semantics.chance_model),
        ),
        (
            semantics.reward_model != RewardModel::TerminalZeroSum,
            "semantics.reward_model",
            "terminal zero-sum per-agent rewards",
            format!("{:?}", semantics.reward_model),
        ),
    ] {
        if invalid {
            add_issue(
                issues,
                code,
                format!("requires {requirement}, got {actual}"),
            );
        }
    }
}
