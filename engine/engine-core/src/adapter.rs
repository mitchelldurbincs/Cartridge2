//! Validated adapter from the typed [`Environment`] API to the erased runtime ABI.

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::board_view::Presentation;
use crate::contract;
use crate::erased::{
    EncodedObservation, ErasedEnvironment, ErasedEnvironmentError, ErasedTimestep,
};
use crate::metadata::EnvironmentMetadata;
use crate::typed::{Capabilities, EngineId, Environment, TransitionSource};

#[derive(Debug)]
pub(crate) struct EnvironmentAdapter<E: Environment> {
    environment: E,
    rng: ChaCha20Rng,
    id: EngineId,
    capabilities: Capabilities,
    metadata: EnvironmentMetadata,
}

impl<E: Environment> EnvironmentAdapter<E> {
    pub(crate) fn try_new(environment: E) -> Result<Self, ErasedEnvironmentError> {
        let id = environment.engine_id();
        let capabilities = environment.capabilities();
        let metadata = environment.metadata();
        contract::validate_descriptors(&id, &capabilities, &metadata)?;
        Ok(Self {
            environment,
            rng: ChaCha20Rng::seed_from_u64(0),
            id,
            capabilities,
            metadata,
        })
    }

    fn validate_descriptors_unchanged(&self) -> Result<(), ErasedEnvironmentError> {
        let id = self.environment.engine_id();
        let capabilities = self.environment.capabilities();
        let metadata = self.environment.metadata();
        contract::validate_descriptors(&id, &capabilities, &metadata)?;
        if id != self.id || capabilities != self.capabilities || metadata != self.metadata {
            return Err(ErasedEnvironmentError::ContractViolation(
                "environment descriptors changed after construction".to_string(),
            ));
        }
        Ok(())
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
            contract::validate_encoded_observation(
                &self.capabilities,
                observation.agent_id,
                &data,
            )?;
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
        self.id.clone()
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities.clone()
    }

    fn metadata(&self) -> EnvironmentMetadata {
        self.metadata.clone()
    }

    fn reset(
        &mut self,
        seed: u64,
        hint: &[u8],
        out_state: &mut Vec<u8>,
        out_timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        self.validate_descriptors_unchanged()?;
        self.rng = ChaCha20Rng::seed_from_u64(seed);
        let (state, timestep) = self
            .environment
            .reset(&mut self.rng, hint)
            .map_err(|error| ErasedEnvironmentError::Environment(error.to_string()))?;
        if timestep.source != TransitionSource::Reset {
            return Err(ErasedEnvironmentError::ContractViolation(
                "reset must produce transition source=reset".to_string(),
            ));
        }
        contract::validate_typed_timestep(&self.capabilities, &timestep)?;
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
        self.validate_descriptors_unchanged()?;
        let mut state = E::decode_state(state)
            .map_err(|error| ErasedEnvironmentError::Decoding(error.to_string()))?;
        let action = E::decode_action(action)
            .map_err(|error| ErasedEnvironmentError::Decoding(error.to_string()))?;
        let timestep = self
            .environment
            .step(&mut state, action, &mut self.rng)
            .map_err(|error| ErasedEnvironmentError::Environment(error.to_string()))?;
        if timestep.source == TransitionSource::Reset {
            return Err(ErasedEnvironmentError::ContractViolation(
                "step cannot produce transition source=reset".to_string(),
            ));
        }
        contract::validate_typed_timestep(&self.capabilities, &timestep)?;
        out_state.clear();
        E::encode_state(&state, out_state)
            .map_err(|error| ErasedEnvironmentError::Encoding(error.to_string()))?;
        self.encode_timestep(timestep, out_timestep)
    }

    fn presentation(&self, state: &[u8]) -> Result<Option<Presentation>, ErasedEnvironmentError> {
        self.validate_descriptors_unchanged()?;
        let state = E::decode_state(state)
            .map_err(|error| ErasedEnvironmentError::Decoding(error.to_string()))?;
        Ok(E::presentation(&state))
    }
}

#[cfg(test)]
#[path = "adapter_tests.rs"]
mod tests;
