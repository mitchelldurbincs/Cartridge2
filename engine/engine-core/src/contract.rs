//! Runtime validation shared by typed adapters and raw erased environments.

use std::collections::BTreeSet;

use crate::erased::{ErasedEnvironmentError, ErasedTimestep};
use crate::metadata::{BoardGameMetadata, EnvironmentMetadata};
use crate::typed::{
    ActionAvailability, ActionAvailabilityContract, ActionEncoding, ActionSpace, AgentId,
    AgentModel, AgentOutcome, Capabilities, ChanceModel, Decision, EngineId, EpisodeStatus,
    ObservationEncoding, RewardModel, TensorDType, Timestep, TransitionDynamics, TransitionSource,
    TurnModel, WIRE_ENCODING_SCHEMA_VERSION,
};

fn valid_runtime_segment(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
        })
}

pub(crate) fn validate_action_space(space: &ActionSpace) -> Result<(), ErasedEnvironmentError> {
    let invalid = match space {
        ActionSpace::Discrete { size } => *size == 0,
        ActionSpace::MultiDiscrete { dimensions } => {
            dimensions.is_empty() || dimensions.contains(&0)
        }
        ActionSpace::Continuous { low, high, shape } => {
            let elements = shape.iter().try_fold(1usize, |elements, size| {
                elements.checked_mul(usize::try_from(*size).ok()?)
            });
            shape.is_empty()
                || shape.contains(&0)
                || low.is_empty()
                || low.len() != high.len()
                || elements != Some(low.len())
                || low
                    .iter()
                    .zip(high)
                    .any(|(low, high)| !low.is_finite() || !high.is_finite() || low >= high)
        }
    };
    if invalid {
        return Err(ErasedEnvironmentError::ContractViolation(format!(
            "invalid action space {space:?}"
        )));
    }
    Ok(())
}

fn validate_availability_contract(
    space: &ActionSpace,
    contract: &ActionAvailabilityContract,
) -> Result<(), ErasedEnvironmentError> {
    match contract {
        ActionAvailabilityContract::All => Ok(()),
        ActionAvailabilityContract::DiscreteMask
            if matches!(space, ActionSpace::Discrete { .. }) =>
        {
            Ok(())
        }
        ActionAvailabilityContract::DiscreteMask => Err(ErasedEnvironmentError::ContractViolation(
            "discrete-mask availability requires a discrete action space".to_string(),
        )),
        ActionAvailabilityContract::Custom { id } if id.trim().is_empty() => {
            Err(ErasedEnvironmentError::ContractViolation(
                "custom action-availability contract ID must be non-empty".to_string(),
            ))
        }
        ActionAvailabilityContract::Custom { .. } => Ok(()),
    }
}

pub(crate) fn validate_encoding(capabilities: &Capabilities) -> Result<(), ErasedEnvironmentError> {
    let encoding = &capabilities.encoding;
    if encoding.state.trim().is_empty() {
        return Err(ErasedEnvironmentError::ContractViolation(
            "state encoding ID must be non-empty".to_string(),
        ));
    }
    let action_spaces = match &capabilities.agents {
        AgentModel::Fixed { agents } => agents
            .iter()
            .map(|agent| &agent.action_space)
            .collect::<Vec<_>>(),
        AgentModel::Dynamic { action_space, .. } => vec![action_space],
    };
    match &encoding.action {
        ActionEncoding::DiscreteU32LittleEndian
            if action_spaces
                .iter()
                .any(|space| !matches!(space, ActionSpace::Discrete { .. })) =>
        {
            return Err(ErasedEnvironmentError::ContractViolation(
                "discrete_u32_little_endian actions require discrete action spaces".to_string(),
            ));
        }
        ActionEncoding::MultiDiscreteU32LittleEndian
            if action_spaces
                .iter()
                .any(|space| !matches!(space, ActionSpace::MultiDiscrete { .. })) =>
        {
            return Err(ErasedEnvironmentError::ContractViolation(
                "multi_discrete_u32_little_endian actions require multi-discrete action spaces"
                    .to_string(),
            ));
        }
        ActionEncoding::ContinuousF32LittleEndian
            if action_spaces
                .iter()
                .any(|space| !matches!(space, ActionSpace::Continuous { .. })) =>
        {
            return Err(ErasedEnvironmentError::ContractViolation(
                "continuous_f32_little_endian actions require continuous action spaces".to_string(),
            ));
        }
        ActionEncoding::Custom { id } if id.trim().is_empty() => {
            return Err(ErasedEnvironmentError::ContractViolation(
                "custom action encoding ID must be non-empty".to_string(),
            ));
        }
        _ => {}
    }
    match &encoding.observation {
        ObservationEncoding::Tensor { spec } => {
            if spec.dimensions.is_empty() {
                return Err(ErasedEnvironmentError::ContractViolation(
                    "tensor observation must declare at least one dimension".to_string(),
                ));
            }
            let mut names = BTreeSet::new();
            for dimension in &spec.dimensions {
                if !valid_runtime_segment(&dimension.name) || !names.insert(&dimension.name) {
                    return Err(ErasedEnvironmentError::ContractViolation(
                        "tensor observation dimensions require unique runtime-segment names"
                            .to_string(),
                    ));
                }
                if dimension.size == Some(0) {
                    return Err(ErasedEnvironmentError::ContractViolation(
                        "fixed tensor observation dimensions must be positive".to_string(),
                    ));
                }
            }
            if spec.fixed_elements() == Some(0) {
                return Err(ErasedEnvironmentError::ContractViolation(
                    "tensor observation must contain at least one element".to_string(),
                ));
            }
            Ok(())
        }
        ObservationEncoding::Custom { id } if id.trim().is_empty() => {
            Err(ErasedEnvironmentError::ContractViolation(
                "custom observation encoding ID must be non-empty".to_string(),
            ))
        }
        _ => Ok(()),
    }
}

fn validate_board_profile(board: &BoardGameMetadata) -> Result<(), ErasedEnvironmentError> {
    if board.width == 0 || board.height == 0 {
        return Err(ErasedEnvironmentError::ContractViolation(
            "board profile dimensions must be positive".to_string(),
        ));
    }
    board.board_size().map_err(|error| {
        ErasedEnvironmentError::ContractViolation(format!("invalid board profile: {error}"))
    })?;
    if board.players.len() != 2
        || board
            .players
            .iter()
            .any(|player| player.name.trim().is_empty() || player.symbol.trim().is_empty())
    {
        return Err(ErasedEnvironmentError::ContractViolation(
            "board profile requires exactly two players with non-empty names and symbols"
                .to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_descriptors(
    id: &EngineId,
    capabilities: &Capabilities,
    metadata: &EnvironmentMetadata,
) -> Result<(), ErasedEnvironmentError> {
    if !valid_runtime_segment(&id.env_id) {
        return Err(ErasedEnvironmentError::ContractViolation(
            "environment ID must contain only lowercase ASCII letters, digits, '_' or '-'"
                .to_string(),
        ));
    }
    if id.build_id.trim().is_empty() {
        return Err(ErasedEnvironmentError::ContractViolation(
            "engine build ID must be non-empty".to_string(),
        ));
    }
    if id != &capabilities.id {
        return Err(ErasedEnvironmentError::ContractViolation(
            "engine_id() does not equal capabilities.id".to_string(),
        ));
    }
    if metadata.id != id.env_id {
        return Err(ErasedEnvironmentError::ContractViolation(format!(
            "metadata id '{}' does not equal environment id '{}'",
            metadata.id, id.env_id
        )));
    }
    if metadata.display_name.trim().is_empty() {
        return Err(ErasedEnvironmentError::ContractViolation(
            "environment display name must be non-empty".to_string(),
        ));
    }
    if capabilities.contract_version == 0 {
        return Err(ErasedEnvironmentError::ContractViolation(
            "contract_version must be positive".to_string(),
        ));
    }
    if capabilities.encoding.schema_version != WIRE_ENCODING_SCHEMA_VERSION {
        return Err(ErasedEnvironmentError::ContractViolation(format!(
            "unsupported wire encoding schema {}",
            capabilities.encoding.schema_version
        )));
    }
    validate_encoding(capabilities)?;
    if capabilities.semantics.turn_model == TurnModel::Simultaneous
        && !matches!(&capabilities.encoding.action, ActionEncoding::Custom { .. })
    {
        return Err(ErasedEnvironmentError::ContractViolation(
            "simultaneous decisions require a custom joint-action codec until the wire ABI defines a standard decision-action envelope"
                .to_string(),
        ));
    }
    if capabilities.semantics.chance_model == ChanceModel::Explicit
        && !matches!(&capabilities.encoding.action, ActionEncoding::Custom { .. })
    {
        return Err(ErasedEnvironmentError::ContractViolation(
            "explicit chance requires a custom chance/agent action codec until the wire ABI defines a standard decision-action envelope"
                .to_string(),
        ));
    }
    if capabilities.max_horizon == Some(0) || capabilities.preferred_batch == 0 {
        return Err(ErasedEnvironmentError::ContractViolation(
            "declared horizons and batch sizes must be positive".to_string(),
        ));
    }
    match (
        capabilities.semantics.transition_dynamics,
        capabilities.semantics.chance_model,
    ) {
        (TransitionDynamics::Deterministic, ChanceModel::None)
        | (TransitionDynamics::Stochastic, ChanceModel::Explicit)
        | (TransitionDynamics::Stochastic, ChanceModel::EnvironmentSampled) => {}
        (dynamics, chance_model) => {
            return Err(ErasedEnvironmentError::ContractViolation(format!(
                "transition dynamics {dynamics:?} are inconsistent with chance model {chance_model:?}"
            )));
        }
    }
    if capabilities.semantics.chance_model == ChanceModel::EnvironmentSampled
        && capabilities.semantics.planning_state_model
            == crate::typed::PlanningStateModel::CompleteSnapshot
    {
        return Err(ErasedEnvironmentError::ContractViolation(
            "environment-sampled chance uses runtime RNG state and therefore requires planning_state_model=external_state"
                .to_string(),
        ));
    }
    match &capabilities.agents {
        AgentModel::Fixed { agents } => {
            if agents.is_empty() {
                return Err(ErasedEnvironmentError::ContractViolation(
                    "fixed agent model must contain at least one agent".to_string(),
                ));
            }
            let mut ids = BTreeSet::new();
            for agent in agents {
                if !ids.insert(agent.id) {
                    return Err(ErasedEnvironmentError::ContractViolation(format!(
                        "duplicate fixed agent id {}",
                        agent.id.0
                    )));
                }
                validate_action_space(&agent.action_space)?;
                validate_availability_contract(&agent.action_space, &agent.action_availability)?;
            }
            if capabilities.semantics.turn_model == TurnModel::SingleAgent && agents.len() != 1 {
                return Err(ErasedEnvironmentError::ContractViolation(format!(
                    "single-agent semantics require exactly one fixed agent, got {}",
                    agents.len()
                )));
            }
        }
        AgentModel::Dynamic {
            action_space,
            action_availability,
        } => {
            validate_action_space(action_space)?;
            validate_availability_contract(action_space, action_availability)?;
        }
    }
    if let Some(board) = &metadata.board {
        validate_board_profile(board)?;
    }
    Ok(())
}

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

    let mut observation_ids = BTreeSet::new();
    for agent_id in observation_agent_ids {
        if !known_agent(capabilities, &roster, agent_id) {
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

    let mut outcome_ids = BTreeSet::new();
    for outcome in outcomes {
        if !known_agent(capabilities, &roster, outcome.agent_id) {
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
    if outcome_ids != roster {
        return Err(ErasedEnvironmentError::ContractViolation(
            "every timestep agent must have exactly one outcome".to_string(),
        ));
    }

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
    if capabilities.semantics.reward_model == RewardModel::TerminalZeroSum {
        if episode != EpisodeStatus::Terminated
            && outcomes.iter().any(|outcome| outcome.reward != 0.0)
        {
            return Err(ErasedEnvironmentError::ContractViolation(
                "terminal-zero-sum environments must emit zero reward before termination"
                    .to_string(),
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
    }

    match (episode, decision) {
        (EpisodeStatus::Running, Decision::None) => {
            return Err(ErasedEnvironmentError::ContractViolation(
                "running timestep cannot have decision=none".to_string(),
            ));
        }
        (EpisodeStatus::Terminated | EpisodeStatus::Truncated, decision)
            if *decision != Decision::None =>
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
                .any(|decision| !known_agent(capabilities, &roster, decision.agent_id))
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
            if !matches!(capabilities.semantics.turn_model, TurnModel::Simultaneous)
                && decisions.len() != 1
            {
                return Err(ErasedEnvironmentError::ContractViolation(
                    "non-simultaneous environment must request exactly one agent".to_string(),
                ));
            }
        }
        _ => {}
    }

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
    if let Some(agent_ids) = source_ids {
        let unique = agent_ids.iter().copied().collect::<BTreeSet<_>>();
        if agent_ids.is_empty()
            || unique.len() != agent_ids.len()
            || agent_ids
                .iter()
                .any(|agent_id| !known_agent(capabilities, &roster, *agent_id))
        {
            return Err(ErasedEnvironmentError::ContractViolation(
                "transition source must contain unique agent IDs from the transition roster"
                    .to_string(),
            ));
        }
        if !matches!(capabilities.semantics.turn_model, TurnModel::Simultaneous)
            && agent_ids.len() != 1
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
    }
    Ok(())
}

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
