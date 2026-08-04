use std::collections::BTreeSet;

use crate::erased::{ErasedEnvironmentError, ErasedTimestep};
#[cfg(test)]
use crate::typed::Timestep;
use crate::typed::{
    ActionAvailability, ActionAvailabilityContract, ActionSpace, AgentId, AgentModel, AgentOutcome,
    Capabilities, ChanceModel, Decision, EpisodeStatus, ObservationEncoding, RewardModel,
    TensorDType, TransitionSource, TurnModel,
};

fn known_agent(capabilities: &Capabilities, roster: &BTreeSet<AgentId>, id: AgentId) -> bool {
    match &capabilities.agents {
        AgentModel::Fixed { agents } => agents.iter().any(|agent| agent.id == id),
        AgentModel::Dynamic { .. } => roster.contains(&id),
    }
}

fn validate_action_availability(
    capabilities: &Capabilities,
    agent_id: AgentId,
    availability: &ActionAvailability,
) -> Result<(), ErasedEnvironmentError> {
    let action_space = capabilities.action_space(agent_id).ok_or_else(|| {
        ErasedEnvironmentError::ContractViolation(format!(
            "decision agent {} has no declared action space",
            agent_id.0
        ))
    })?;
    let contract = capabilities
        .agents
        .action_availability(agent_id)
        .ok_or_else(|| {
            ErasedEnvironmentError::ContractViolation(format!(
                "decision agent {} has no availability contract",
                agent_id.0
            ))
        })?;
    match (contract, availability) {
        (ActionAvailabilityContract::All, ActionAvailability::All) => Ok(()),
        (ActionAvailabilityContract::DiscreteMask, ActionAvailability::DiscreteMask { mask }) => {
            let ActionSpace::Discrete { size } = action_space else {
                unreachable!("descriptor validation binds discrete masks to discrete spaces")
            };
            if mask.num_actions() != *size as usize {
                return Err(ErasedEnvironmentError::ContractViolation(format!(
                    "decision agent {} legal mask has width {}, expected {}",
                    agent_id.0,
                    mask.num_actions(),
                    size
                )));
            }
            if mask.is_empty() {
                return Err(ErasedEnvironmentError::ContractViolation(format!(
                    "running decision agent {} has no available action",
                    agent_id.0
                )));
            }
            Ok(())
        }
        (
            ActionAvailabilityContract::Custom { id },
            ActionAvailability::Custom { contract, .. },
        ) if id == contract => Ok(()),
        _ => Err(ErasedEnvironmentError::ContractViolation(format!(
            "decision agent {} availability does not match its declared contract",
            agent_id.0
        ))),
    }
}

fn validate_timestep_fields(
    capabilities: &Capabilities,
    agents: &[AgentId],
    observation_agent_ids: impl IntoIterator<Item = AgentId>,
    outcomes: &[AgentOutcome],
    decision: &Decision,
    episode: EpisodeStatus,
    source: &TransitionSource,
) -> Result<(), ErasedEnvironmentError> {
    let roster = validate_timestep_roster(capabilities, agents)?;
    let observation_ids =
        validate_observation_agents(capabilities, &roster, observation_agent_ids)?;
    let outcome_ids = validate_outcomes(capabilities, &roster, outcomes)?;
    validate_episode_outcomes(capabilities, outcomes, episode)?;
    validate_decision(
        capabilities,
        &roster,
        &observation_ids,
        outcomes,
        decision,
        episode,
    )?;
    validate_transition_source(capabilities, &roster, &outcome_ids, source)
}

fn validate_timestep_roster(
    capabilities: &Capabilities,
    agents: &[AgentId],
) -> Result<BTreeSet<AgentId>, ErasedEnvironmentError> {
    let roster = agents.iter().copied().collect::<BTreeSet<_>>();
    if roster.len() != agents.len() {
        return Err(ErasedEnvironmentError::ContractViolation(
            "timestep agent roster contains duplicate IDs".to_string(),
        ));
    }
    if capabilities.semantics.turn_model == TurnModel::SingleAgent && roster.len() > 1 {
        return Err(ErasedEnvironmentError::ContractViolation(format!(
            "single-agent timestep may contain at most one agent, got {}",
            roster.len()
        )));
    }
    if let AgentModel::Fixed { agents: declared } = &capabilities.agents {
        let expected = declared
            .iter()
            .map(|agent| agent.id)
            .collect::<BTreeSet<_>>();
        if roster != expected {
            return Err(ErasedEnvironmentError::ContractViolation(
                "fixed environment timestep roster must equal its declared agents".to_string(),
            ));
        }
    }
    Ok(roster)
}

fn validate_observation_agents(
    capabilities: &Capabilities,
    roster: &BTreeSet<AgentId>,
    observation_agent_ids: impl IntoIterator<Item = AgentId>,
) -> Result<BTreeSet<AgentId>, ErasedEnvironmentError> {
    let mut observation_ids = BTreeSet::new();
    for agent_id in observation_agent_ids {
        if !known_agent(capabilities, roster, agent_id) {
            return Err(ErasedEnvironmentError::ContractViolation(format!(
                "observation targets unknown agent {}",
                agent_id.0
            )));
        }
        if !observation_ids.insert(agent_id) {
            return Err(ErasedEnvironmentError::ContractViolation(format!(
                "duplicate observation for agent {}",
                agent_id.0
            )));
        }
    }
    Ok(observation_ids)
}

fn validate_outcomes(
    capabilities: &Capabilities,
    roster: &BTreeSet<AgentId>,
    outcomes: &[AgentOutcome],
) -> Result<BTreeSet<AgentId>, ErasedEnvironmentError> {
    let mut outcome_ids = BTreeSet::new();
    for outcome in outcomes {
        if !known_agent(capabilities, roster, outcome.agent_id) {
            return Err(ErasedEnvironmentError::ContractViolation(format!(
                "outcome targets unknown agent {}",
                outcome.agent_id.0
            )));
        }
        if !outcome_ids.insert(outcome.agent_id) {
            return Err(ErasedEnvironmentError::ContractViolation(format!(
                "duplicate outcome for agent {}",
                outcome.agent_id.0
            )));
        }
        if !outcome.reward.is_finite() || (outcome.terminated && outcome.truncated) {
            return Err(ErasedEnvironmentError::ContractViolation(format!(
                "invalid outcome for agent {}",
                outcome.agent_id.0
            )));
        }
    }
    if outcome_ids != *roster {
        return Err(ErasedEnvironmentError::ContractViolation(
            "every timestep agent must have exactly one outcome".to_string(),
        ));
    }
    Ok(outcome_ids)
}

fn validate_episode_outcomes(
    capabilities: &Capabilities,
    outcomes: &[AgentOutcome],
    episode: EpisodeStatus,
) -> Result<(), ErasedEnvironmentError> {
    match episode {
        EpisodeStatus::Terminated if outcomes.iter().any(|outcome| !outcome.terminated) => {
            return Err(ErasedEnvironmentError::ContractViolation(
                "terminated episode requires every current agent outcome to be terminated"
                    .to_string(),
            ));
        }
        EpisodeStatus::Truncated if outcomes.iter().any(|outcome| !outcome.truncated) => {
            return Err(ErasedEnvironmentError::ContractViolation(
                "truncated episode requires every current agent outcome to be truncated"
                    .to_string(),
            ));
        }
        _ => {}
    }
    if capabilities.semantics.reward_model != RewardModel::TerminalZeroSum {
        return Ok(());
    }
    if episode != EpisodeStatus::Terminated && outcomes.iter().any(|outcome| outcome.reward != 0.0)
    {
        return Err(ErasedEnvironmentError::ContractViolation(
            "terminal-zero-sum environments must emit zero reward before termination".to_string(),
        ));
    }
    let reward_sum = outcomes
        .iter()
        .map(|outcome| f64::from(outcome.reward))
        .sum::<f64>();
    if episode == EpisodeStatus::Terminated && reward_sum.abs() > 1e-6 {
        return Err(ErasedEnvironmentError::ContractViolation(format!(
            "terminal-zero-sum rewards must sum to zero, got {reward_sum}"
        )));
    }
    Ok(())
}

fn validate_decision(
    capabilities: &Capabilities,
    roster: &BTreeSet<AgentId>,
    observation_ids: &BTreeSet<AgentId>,
    outcomes: &[AgentOutcome],
    decision: &Decision,
    episode: EpisodeStatus,
) -> Result<(), ErasedEnvironmentError> {
    match (episode, decision) {
        (EpisodeStatus::Running, Decision::None) => {
            return Err(ErasedEnvironmentError::ContractViolation(
                "running timestep cannot have decision=none".to_string(),
            ));
        }
        (EpisodeStatus::Terminated | EpisodeStatus::Truncated, value)
            if *value != Decision::None =>
        {
            return Err(ErasedEnvironmentError::ContractViolation(
                "completed timestep must have decision=none".to_string(),
            ));
        }
        (_, Decision::Chance) if capabilities.semantics.chance_model != ChanceModel::Explicit => {
            return Err(ErasedEnvironmentError::ContractViolation(
                "decision=chance requires chance_model=explicit".to_string(),
            ));
        }
        (_, Decision::Agents { decisions }) => {
            validate_agent_decisions(capabilities, roster, observation_ids, outcomes, decisions)?;
        }
        _ => {}
    }
    Ok(())
}

fn validate_agent_decisions(
    capabilities: &Capabilities,
    roster: &BTreeSet<AgentId>,
    observation_ids: &BTreeSet<AgentId>,
    outcomes: &[AgentOutcome],
    decisions: &[crate::typed::AgentDecision],
) -> Result<(), ErasedEnvironmentError> {
    let unique = decisions
        .iter()
        .map(|decision| decision.agent_id)
        .collect::<BTreeSet<_>>();
    if decisions.is_empty() || unique.len() != decisions.len() {
        return Err(ErasedEnvironmentError::ContractViolation(
            "agent decision must contain unique agent IDs".to_string(),
        ));
    }
    if decisions
        .iter()
        .any(|decision| !known_agent(capabilities, roster, decision.agent_id))
    {
        return Err(ErasedEnvironmentError::ContractViolation(
            "decision contains an unknown agent".to_string(),
        ));
    }
    for decision in decisions {
        let agent_id = decision.agent_id;
        if !observation_ids.contains(&agent_id) {
            return Err(ErasedEnvironmentError::ContractViolation(format!(
                "decision agent {} has no observation",
                agent_id.0
            )));
        }
        let outcome = outcomes
            .iter()
            .find(|outcome| outcome.agent_id == agent_id)
            .ok_or_else(|| {
                ErasedEnvironmentError::ContractViolation(format!(
                    "decision agent {} has no outcome",
                    agent_id.0
                ))
            })?;
        if outcome.terminated || outcome.truncated {
            return Err(ErasedEnvironmentError::ContractViolation(format!(
                "decision agent {} is already complete",
                agent_id.0
            )));
        }
        validate_action_availability(capabilities, agent_id, &decision.availability)?;
    }
    if !matches!(capabilities.semantics.turn_model, TurnModel::Simultaneous) && decisions.len() != 1
    {
        return Err(ErasedEnvironmentError::ContractViolation(
            "non-simultaneous environment must request exactly one agent".to_string(),
        ));
    }
    Ok(())
}

fn validate_transition_source(
    capabilities: &Capabilities,
    roster: &BTreeSet<AgentId>,
    outcome_ids: &BTreeSet<AgentId>,
    source: &TransitionSource,
) -> Result<(), ErasedEnvironmentError> {
    let source_ids = match source {
        TransitionSource::Reset => None,
        TransitionSource::Chance => {
            if capabilities.semantics.chance_model != ChanceModel::Explicit {
                return Err(ErasedEnvironmentError::ContractViolation(
                    "chance transition requires chance_model=explicit".to_string(),
                ));
            }
            None
        }
        TransitionSource::Agents { agent_ids } => Some(agent_ids),
    };
    let Some(agent_ids) = source_ids else {
        return Ok(());
    };
    let unique = agent_ids.iter().copied().collect::<BTreeSet<_>>();
    if agent_ids.is_empty()
        || unique.len() != agent_ids.len()
        || agent_ids
            .iter()
            .any(|agent_id| !known_agent(capabilities, roster, *agent_id))
    {
        return Err(ErasedEnvironmentError::ContractViolation(
            "transition source must contain unique agent IDs from the transition roster"
                .to_string(),
        ));
    }
    if !matches!(capabilities.semantics.turn_model, TurnModel::Simultaneous) && agent_ids.len() != 1
    {
        return Err(ErasedEnvironmentError::ContractViolation(
            "non-simultaneous transition must have exactly one source agent".to_string(),
        ));
    }
    if let Some(agent_id) = agent_ids
        .iter()
        .find(|agent_id| !outcome_ids.contains(agent_id))
    {
        return Err(ErasedEnvironmentError::ContractViolation(format!(
            "transition source agent {} has no outcome",
            agent_id.0
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn validate_typed_timestep<O>(
    capabilities: &Capabilities,
    timestep: &Timestep<O>,
) -> Result<(), ErasedEnvironmentError> {
    validate_timestep_fields(
        capabilities,
        &timestep.agents,
        timestep
            .observations
            .iter()
            .map(|observation| observation.agent_id),
        &timestep.outcomes,
        &timestep.decision,
        timestep.episode,
        &timestep.source,
    )
}

pub(crate) fn validate_encoded_observation(
    capabilities: &Capabilities,
    agent_id: AgentId,
    data: &[u8],
) -> Result<(), ErasedEnvironmentError> {
    let ObservationEncoding::Tensor { spec } = &capabilities.encoding.observation else {
        return Ok(());
    };
    if let Some(expected) = spec.fixed_bytes() {
        if data.len() != expected {
            return Err(ErasedEnvironmentError::ContractViolation(format!(
                "observation for agent {} encoded {} bytes, expected {}",
                agent_id.0,
                data.len(),
                expected
            )));
        }
    } else if !data.len().is_multiple_of(spec.dtype.element_size()) {
        return Err(ErasedEnvironmentError::ContractViolation(format!(
            "dynamic observation for agent {} has a partial tensor element",
            agent_id.0
        )));
    }

    match spec.dtype {
        TensorDType::F32LittleEndian => {
            for chunk in data.chunks_exact(std::mem::size_of::<f32>()) {
                let value = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
                if !value.is_finite() {
                    return Err(ErasedEnvironmentError::ContractViolation(format!(
                        "observation for agent {} contains a non-finite f32",
                        agent_id.0
                    )));
                }
            }
        }
        TensorDType::Bool if data.iter().any(|value| !matches!(value, 0 | 1)) => {
            return Err(ErasedEnvironmentError::ContractViolation(format!(
                "observation for agent {} contains a non-boolean byte",
                agent_id.0
            )));
        }
        TensorDType::U8 | TensorDType::I64LittleEndian | TensorDType::Bool => {}
    }
    Ok(())
}

pub(crate) fn validate_erased_timestep(
    capabilities: &Capabilities,
    timestep: &ErasedTimestep,
) -> Result<(), ErasedEnvironmentError> {
    for observation in &timestep.observations {
        validate_encoded_observation(capabilities, observation.agent_id, &observation.data)?;
    }
    validate_timestep_fields(
        capabilities,
        &timestep.agents,
        timestep
            .observations
            .iter()
            .map(|observation| observation.agent_id),
        &timestep.outcomes,
        &timestep.decision,
        timestep.episode,
        &timestep.source,
    )
}
