use std::collections::BTreeSet;

use crate::erased::ErasedEnvironmentError;
use crate::metadata::{BoardGameMetadata, EnvironmentMetadata};
use crate::typed::{
    ActionAvailabilityContract, ActionEncoding, ActionSpace, AgentModel, Capabilities, ChanceModel,
    EngineId, ObservationEncoding, TransitionDynamics, TurnModel, WIRE_ENCODING_SCHEMA_VERSION,
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
    validate_descriptor_identity(id, capabilities, metadata)?;
    validate_descriptor_semantics(capabilities)?;
    validate_agent_model(capabilities)?;
    if let Some(board) = &metadata.board {
        validate_board_profile(board)?;
    }
    Ok(())
}

fn validate_descriptor_identity(
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
    validate_encoding(capabilities)
}

fn validate_descriptor_semantics(
    capabilities: &Capabilities,
) -> Result<(), ErasedEnvironmentError> {
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
    Ok(())
}

fn validate_agent_model(capabilities: &Capabilities) -> Result<(), ErasedEnvironmentError> {
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
    Ok(())
}
