//! Bytes-only environment ABI used by the registry and runtime contexts.

use crate::board_view::Presentation;
use crate::metadata::EnvironmentMetadata;
use crate::typed::{
    AgentId, AgentOutcome, Capabilities, Decision, EngineId, EpisodeStatus, TransitionSource,
};

#[derive(Debug, thiserror::Error)]
pub enum ErasedEnvironmentError {
    #[error("encoding error: {0}")]
    Encoding(String),
    #[error("decoding error: {0}")]
    Decoding(String),
    #[error("environment error: {0}")]
    Environment(String),
    #[error("environment contract violation: {0}")]
    ContractViolation(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedObservation {
    pub agent_id: AgentId,
    pub data: Vec<u8>,
}

/// Algorithm-neutral transition envelope at the erased boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct ErasedTimestep {
    /// Transition roster; dynamic departures remain for their final outcome
    /// and disappear from the following timestep.
    pub agents: Vec<AgentId>,
    pub observations: Vec<EncodedObservation>,
    pub outcomes: Vec<AgentOutcome>,
    pub decision: Decision,
    pub episode: EpisodeStatus,
    pub source: TransitionSource,
    pub info: Vec<u8>,
}

impl Default for ErasedTimestep {
    fn default() -> Self {
        Self {
            agents: Vec::new(),
            observations: Vec::new(),
            outcomes: Vec::new(),
            decision: Decision::None,
            episode: EpisodeStatus::Running,
            source: TransitionSource::Reset,
            info: Vec::new(),
        }
    }
}

impl ErasedTimestep {
    pub fn reward_for(&self, agent_id: AgentId) -> Option<f32> {
        self.outcomes
            .iter()
            .find(|outcome| outcome.agent_id == agent_id)
            .map(|outcome| outcome.reward)
    }

    pub fn observation_for(&self, agent_id: AgentId) -> Option<&[u8]> {
        self.observations
            .iter()
            .find(|observation| observation.agent_id == agent_id)
            .map(|observation| observation.data.as_slice())
    }

    pub fn sole_observation(&self) -> Result<&EncodedObservation, ErasedEnvironmentError> {
        match self.observations.as_slice() {
            [observation] => Ok(observation),
            observations => Err(ErasedEnvironmentError::ContractViolation(format!(
                "expected exactly one observation, got {}",
                observations.len()
            ))),
        }
    }
}

pub(crate) trait ErasedEnvironment: Send + Sync + std::fmt::Debug + 'static {
    fn engine_id(&self) -> EngineId;
    fn capabilities(&self) -> Capabilities;
    fn metadata(&self) -> EnvironmentMetadata;
    fn describe_discrete_action(
        &self,
        _agent: AgentId,
        _action: u32,
    ) -> Option<crate::ActionPresentation> {
        None
    }

    fn reset(
        &mut self,
        seed: u64,
        hint: &[u8],
        out_state: &mut Vec<u8>,
        out_timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError>;

    fn step(
        &mut self,
        state: &[u8],
        action: &[u8],
        out_state: &mut Vec<u8>,
        out_timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError>;

    fn presentation(&self, state: &[u8]) -> Result<Option<Presentation>, ErasedEnvironmentError>;
}
