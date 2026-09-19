//! High-level validated context for one erased environment instance.

use crate::adapter::EnvironmentAdapter;
use crate::board_view::Presentation;
use crate::contract;
use crate::erased::{ErasedEnvironment, ErasedEnvironmentError, ErasedTimestep};
use crate::metadata::EnvironmentMetadata;
use crate::registry::{create_environment, RegistryError};
use crate::typed::{ActionSpace, AgentId, Capabilities, EngineId, Environment, TransitionSource};

#[derive(Debug, thiserror::Error)]
pub enum EngineContextError {
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error(transparent)]
    Contract(#[from] ErasedEnvironmentError),
}

#[derive(Debug)]
pub struct EngineContext {
    environment: Box<dyn ErasedEnvironment>,
    id: EngineId,
    capabilities: Capabilities,
    metadata: EnvironmentMetadata,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepResult {
    pub state: Vec<u8>,
    pub timestep: ErasedTimestep,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResetResult {
    pub state: Vec<u8>,
    pub timestep: ErasedTimestep,
}

impl EngineContext {
    pub fn new(env_id: &str) -> Result<Self, EngineContextError> {
        Ok(Self::from_erased(create_environment(env_id)?)?)
    }

    /// Construct an isolated context directly from a typed environment.
    pub fn from_environment<E: Environment>(
        environment: E,
    ) -> Result<Self, ErasedEnvironmentError> {
        Self::from_erased(Box::new(EnvironmentAdapter::new(environment)))
    }

    fn from_erased(
        environment: Box<dyn ErasedEnvironment>,
    ) -> Result<Self, ErasedEnvironmentError> {
        let id = environment.engine_id();
        let capabilities = environment.capabilities();
        let metadata = environment.metadata();
        contract::validate_descriptors(&id, &capabilities, &metadata)?;
        Ok(Self {
            environment,
            id,
            capabilities,
            metadata,
        })
    }

    fn validate_descriptors_unchanged(&self) -> Result<(), ErasedEnvironmentError> {
        let id = self.environment.engine_id();
        let capabilities = self.environment.capabilities();
        let metadata = self.environment.metadata();
        if id != self.id || capabilities != self.capabilities || metadata != self.metadata {
            // Unchanged descriptors retain their construction-time validation.
            // Validate drift before rejecting it to preserve specific contract errors.
            contract::validate_descriptors(&id, &capabilities, &metadata)?;
            return Err(ErasedEnvironmentError::ContractViolation(
                "environment descriptors changed after context construction".to_string(),
            ));
        }
        Ok(())
    }

    pub fn engine_id(&self) -> EngineId {
        self.id.clone()
    }

    pub fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }

    pub fn action_space(&self, agent_id: AgentId) -> Option<ActionSpace> {
        self.capabilities.action_space(agent_id).cloned()
    }

    pub fn metadata(&self) -> EnvironmentMetadata {
        self.metadata.clone()
    }

    pub fn describe_discrete_action(
        &self,
        agent: crate::AgentId,
        action: u32,
    ) -> Option<crate::ActionPresentation> {
        match self.action_space(agent) {
            Some(crate::ActionSpace::Discrete { size }) if action < size => {
                self.environment.describe_discrete_action(agent, action)
            }
            _ => None,
        }
    }

    pub fn presentation(
        &self,
        state: &[u8],
    ) -> Result<Option<Presentation>, ErasedEnvironmentError> {
        self.validate_descriptors_unchanged()?;
        self.environment.presentation(state)
    }

    /// Reset with owned outputs, using the same validation as [`Self::reset_into`].
    pub fn reset(&mut self, seed: u64, hint: &[u8]) -> Result<ResetResult, ErasedEnvironmentError> {
        let mut state = Vec::with_capacity(256);
        let mut timestep = ErasedTimestep::default();
        self.reset_into(seed, hint, &mut state, &mut timestep)?;
        Ok(ResetResult { state, timestep })
    }

    /// Step with owned outputs, using the same validation as [`Self::step_into`].
    pub fn step(
        &mut self,
        state: &[u8],
        action: &[u8],
    ) -> Result<StepResult, ErasedEnvironmentError> {
        let mut state_out = Vec::with_capacity(256);
        let mut timestep = ErasedTimestep::default();
        self.step_into(state, action, &mut state_out, &mut timestep)?;
        Ok(StepResult {
            state: state_out,
            timestep,
        })
    }

    /// Reset into caller-owned buffers and validate the resulting transition.
    ///
    /// Outputs replace any previous contents on success. On error they may be
    /// partially written and must not be used as a valid transition.
    pub fn reset_into(
        &mut self,
        seed: u64,
        hint: &[u8],
        state: &mut Vec<u8>,
        timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        self.validate_descriptors_unchanged()?;
        state.clear();
        self.environment.reset(seed, hint, state, timestep)?;
        if timestep.source != TransitionSource::Reset {
            return Err(ErasedEnvironmentError::ContractViolation(
                "reset must produce transition source=reset".to_string(),
            ));
        }
        contract::validate_erased_timestep(&self.capabilities, timestep)?;
        Ok(())
    }

    /// Step into caller-owned buffers and validate the resulting transition.
    ///
    /// Outputs replace any previous contents on success. On error they may be
    /// partially written and must not be used as a valid transition.
    pub fn step_into(
        &mut self,
        state: &[u8],
        action: &[u8],
        state_out: &mut Vec<u8>,
        timestep_out: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        self.validate_descriptors_unchanged()?;
        state_out.clear();
        self.environment
            .step(state, action, state_out, timestep_out)?;
        if timestep_out.source == TransitionSource::Reset {
            return Err(ErasedEnvironmentError::ContractViolation(
                "step cannot produce transition source=reset".to_string(),
            ));
        }
        contract::validate_erased_timestep(&self.capabilities, timestep_out)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "context_tests.rs"]
mod tests;
