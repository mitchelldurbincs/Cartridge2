//! Small non-board reference environment for the generic Cartridge contract.
//!
//! One agent moves an integer counter left or right. Reaching the target
//! terminates with a positive reward; exhausting the horizon truncates. It is
//! intentionally incompatible with the AlphaZero board cartridge and exists
//! as an executable canary for generic catalog and runtime paths.

use engine_core::{
    register_environment, ActionAvailability, ActionSpace, AgentId, AgentModel, AgentObservation,
    AgentOutcome, Capabilities, Decision, DecodeError, EncodeError, Encoding, EngineId,
    Environment, EnvironmentError, EnvironmentMetadata, EnvironmentSemantics, EpisodeStatus,
    TensorSpec, Timestep, TransitionSource,
};
use rand_chacha::ChaCha20Rng;

pub const ENV_CONTRACT_VERSION: u32 = 2;
pub const AGENT: AgentId = AgentId(0);
pub const TARGET: i32 = 3;
pub const MAX_STEPS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CounterState {
    pub position: i32,
    pub steps: u32,
}

#[derive(Debug, Default)]
pub struct CounterEnvironment;

impl CounterEnvironment {
    fn observation(state: &CounterState) -> [f32; 2] {
        [
            state.position as f32 / TARGET as f32,
            (MAX_STEPS - state.steps) as f32 / MAX_STEPS as f32,
        ]
    }

    fn timestep(state: &CounterState, reward: f32, source: TransitionSource) -> Timestep<[f32; 2]> {
        let terminated = state.position == TARGET;
        let truncated = !terminated && state.steps >= MAX_STEPS;
        let episode = if terminated {
            EpisodeStatus::Terminated
        } else if truncated {
            EpisodeStatus::Truncated
        } else {
            EpisodeStatus::Running
        };
        Timestep {
            agents: vec![AGENT],
            observations: vec![AgentObservation {
                agent_id: AGENT,
                observation: Self::observation(state),
            }],
            outcomes: vec![AgentOutcome {
                agent_id: AGENT,
                reward,
                terminated,
                truncated,
            }],
            decision: if episode.is_done() {
                Decision::None
            } else {
                Decision::single(AGENT, ActionAvailability::All)
            },
            episode,
            source,
            info: Vec::new(),
        }
    }
}

impl Environment for CounterEnvironment {
    type State = CounterState;
    type Action = u32;
    type Observation = [f32; 2];

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "counter".to_string(),
            build_id: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: ENV_CONTRACT_VERSION,
            encoding: Encoding::discrete_u32_le(
                "counter-state:v1",
                TensorSpec::f32_fixed([("feature", 2)]),
            ),
            semantics: EnvironmentSemantics::deterministic_single_agent_general_reward(),
            max_horizon: Some(MAX_STEPS),
            agents: AgentModel::fixed_homogeneous([AGENT], ActionSpace::discrete(2)),
            preferred_batch: 32,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("counter", "Counter")
            .with_description("Move a scalar counter to the target before the horizon")
    }

    fn describe_discrete_action(
        &self,
        agent: AgentId,
        action: u32,
    ) -> Option<engine_core::ActionPresentation> {
        if agent != AGENT {
            return None;
        }
        match action {
            0 => Some(engine_core::ActionPresentation::named(action, "Left")),
            1 => Some(engine_core::ActionPresentation::named(action, "Right")),
            _ => None,
        }
    }

    fn reset(
        &mut self,
        _rng: &mut ChaCha20Rng,
        hint: &[u8],
    ) -> Result<(Self::State, Timestep<Self::Observation>), EnvironmentError> {
        if !hint.is_empty() {
            return Err(EnvironmentError::InvalidHint(
                "counter does not accept reset hints".to_string(),
            ));
        }
        let state = CounterState {
            position: 0,
            steps: 0,
        };
        let timestep = Self::timestep(&state, 0.0, TransitionSource::Reset);
        Ok((state, timestep))
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> Result<Timestep<Self::Observation>, EnvironmentError> {
        if state.position == TARGET || state.steps >= MAX_STEPS {
            return Err(EnvironmentError::InvalidState(
                "cannot step a completed counter episode".to_string(),
            ));
        }
        match action {
            0 => state.position -= 1,
            1 => state.position += 1,
            _ => {
                return Err(EnvironmentError::InvalidAction(format!(
                    "counter action {action} is outside [0, 2)"
                )));
            }
        }
        state.steps += 1;
        let reward = if state.position == TARGET { 1.0 } else { -0.01 };
        Ok(Self::timestep(
            state,
            reward,
            TransitionSource::Agents {
                agent_ids: vec![AGENT],
            },
        ))
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.extend_from_slice(&state.position.to_le_bytes());
        out.extend_from_slice(&state.steps.to_le_bytes());
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        if buf.len() != 8 {
            return Err(DecodeError::InvalidLength {
                expected: 8,
                actual: buf.len(),
            });
        }
        let position = i32::from_le_bytes(buf[0..4].try_into().expect("four-byte position"));
        let steps = u32::from_le_bytes(buf[4..8].try_into().expect("four-byte step count"));
        if steps > MAX_STEPS {
            return Err(DecodeError::CorruptedData(format!(
                "counter step count {steps} exceeds horizon {MAX_STEPS}"
            )));
        }
        Ok(CounterState { position, steps })
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        if *action >= 2 {
            return Err(EncodeError::InvalidData(format!(
                "counter action {action} is outside [0, 2)"
            )));
        }
        out.extend_from_slice(&action.to_le_bytes());
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        if buf.len() != 4 {
            return Err(DecodeError::InvalidLength {
                expected: 4,
                actual: buf.len(),
            });
        }
        let action = u32::from_le_bytes(buf.try_into().expect("four-byte action"));
        if action >= 2 {
            return Err(DecodeError::CorruptedData(format!(
                "counter action {action} is outside [0, 2)"
            )));
        }
        Ok(action)
    }

    fn encode_observation(
        observation: &Self::Observation,
        out: &mut Vec<u8>,
    ) -> Result<(), EncodeError> {
        for value in observation {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }
}

pub fn register_counter() {
    register_environment::<CounterEnvironment>()
        .expect("counter environment must only be registered once");
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::EngineContext;

    #[test]
    fn direct_environment_runs_without_a_board_profile() {
        let mut context = EngineContext::from_environment(CounterEnvironment).unwrap();
        assert!(context.metadata().board.is_none());

        let reset = context.reset(7, &[]).unwrap();
        assert_eq!(reset.timestep.agents, vec![AGENT]);
        assert_eq!(reset.timestep.episode, EpisodeStatus::Running);

        let first = context.step(&reset.state, &1u32.to_le_bytes()).unwrap();
        let second = context.step(&first.state, &1u32.to_le_bytes()).unwrap();
        let third = context.step(&second.state, &1u32.to_le_bytes()).unwrap();
        assert_eq!(third.timestep.episode, EpisodeStatus::Terminated);
        assert_eq!(third.timestep.reward_for(AGENT), Some(1.0));
    }

    #[test]
    fn horizon_is_reported_as_truncation_not_termination() {
        let mut context = EngineContext::from_environment(CounterEnvironment).unwrap();
        let mut state = context.reset(11, &[]).unwrap().state;
        let mut timestep = None;
        for action in [0u32, 1, 0, 1, 0, 1, 0, 1] {
            let result = context.step(&state, &action.to_le_bytes()).unwrap();
            state = result.state;
            timestep = Some(result.timestep);
        }
        let timestep = timestep.unwrap();
        assert_eq!(timestep.episode, EpisodeStatus::Truncated);
        let outcome = &timestep.outcomes[0];
        assert!(outcome.truncated);
        assert!(!outcome.terminated);
    }
}
