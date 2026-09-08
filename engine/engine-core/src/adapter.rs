//! Conversion adapter from the typed [`Environment`] API to the erased runtime ABI.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::board_view::Presentation;
use crate::erased::{
    EncodedObservation, ErasedEnvironment, ErasedEnvironmentError, ErasedTimestep,
};
use crate::metadata::EnvironmentMetadata;
use crate::typed::{Capabilities, EngineId, Environment};

#[derive(Debug)]
pub(crate) struct EnvironmentAdapter<E: Environment> {
    environment: E,
    rng: ChaCha20Rng,
}

impl<E: Environment> EnvironmentAdapter<E> {
    pub(crate) fn new(environment: E) -> Self {
        Self {
            environment,
            rng: ChaCha20Rng::seed_from_u64(0),
        }
    }

    fn encode_timestep(
        &self,
        timestep: crate::typed::Timestep<E::Observation>,
        out: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        out.agents = timestep.agents;
        out.observations.clear();
        for observation in timestep.observations {
            let mut data = Vec::new();
            E::encode_observation(&observation.observation, &mut data)
                .map_err(|error| ErasedEnvironmentError::Encoding(error.to_string()))?;
            out.observations.push(EncodedObservation {
                agent_id: observation.agent_id,
                data,
            });
        }
        out.outcomes = timestep.outcomes;
        out.decision = timestep.decision;
        out.episode = timestep.episode;
        out.source = timestep.source;
        out.info = timestep.info;
        Ok(())
    }
}

impl<E: Environment> ErasedEnvironment for EnvironmentAdapter<E> {
    fn engine_id(&self) -> EngineId {
        self.environment.engine_id()
    }

    fn capabilities(&self) -> Capabilities {
        self.environment.capabilities()
    }

    fn metadata(&self) -> EnvironmentMetadata {
        self.environment.metadata()
    }

    fn describe_discrete_action(
        &self,
        agent: crate::AgentId,
        action: u32,
    ) -> Option<crate::ActionPresentation> {
        self.environment.describe_discrete_action(agent, action)
    }

    fn reset(
        &mut self,
        seed: u64,
        hint: &[u8],
        out_state: &mut Vec<u8>,
        out_timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        self.rng = ChaCha20Rng::seed_from_u64(seed);
        let (state, timestep) = self
            .environment
            .reset(&mut self.rng, hint)
            .map_err(|error| ErasedEnvironmentError::Environment(error.to_string()))?;
        out_state.clear();
        E::encode_state(&state, out_state)
            .map_err(|error| ErasedEnvironmentError::Encoding(error.to_string()))?;
        self.encode_timestep(timestep, out_timestep)
    }

    fn step(
        &mut self,
        state: &[u8],
        action: &[u8],
        out_state: &mut Vec<u8>,
        out_timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        let mut state = E::decode_state(state)
            .map_err(|error| ErasedEnvironmentError::Decoding(error.to_string()))?;
        let action = E::decode_action(action)
            .map_err(|error| ErasedEnvironmentError::Decoding(error.to_string()))?;
        let timestep = self
            .environment
            .step(&mut state, action, &mut self.rng)
            .map_err(|error| ErasedEnvironmentError::Environment(error.to_string()))?;
        out_state.clear();
        E::encode_state(&state, out_state)
            .map_err(|error| ErasedEnvironmentError::Encoding(error.to_string()))?;
        self.encode_timestep(timestep, out_timestep)
    }

    fn presentation(&self, state: &[u8]) -> Result<Option<Presentation>, ErasedEnvironmentError> {
        let state = E::decode_state(state)
            .map_err(|error| ErasedEnvironmentError::Decoding(error.to_string()))?;
        Ok(E::presentation(&state))
    }
}

#[cfg(test)]
#[path = "adapter_tests.rs"]
mod tests;
