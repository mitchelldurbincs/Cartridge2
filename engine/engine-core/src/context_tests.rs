use crate::erased::{
    EncodedObservation, ErasedEnvironment, ErasedEnvironmentError, ErasedTimestep,
};
use crate::metadata::EnvironmentMetadata;
use crate::typed::{
    ActionSpace, AgentId, AgentModel, AgentOutcome, Capabilities, Decision, Encoding, EngineId,
    EnvironmentSemantics, EpisodeStatus, TransitionSource,
};
use crate::EngineContext;

#[derive(Debug)]
struct MockEnvironment;

impl MockEnvironment {
    fn timestep(value: u32, source: TransitionSource) -> ErasedTimestep {
        ErasedTimestep {
            agents: vec![AgentId(3)],
            observations: vec![EncodedObservation {
                agent_id: AgentId(3),
                data: value.to_le_bytes().to_vec(),
            }],
            outcomes: vec![AgentOutcome {
                agent_id: AgentId(3),
                reward: value as f32,
                terminated: value >= 1,
                truncated: false,
            }],
            decision: if value >= 1 {
                Decision::None
            } else {
                Decision::Agents {
                    agent_ids: vec![AgentId(3)],
                }
            },
            episode: if value >= 1 {
                EpisodeStatus::Terminated
            } else {
                EpisodeStatus::Running
            },
            source,
            info: vec![],
        }
    }
}

impl ErasedEnvironment for MockEnvironment {
    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "mock".into(),
            build_id: "test".into(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: 1,
            encoding: Encoding::custom("state:v1", "action:v1", "obs:v1"),
            semantics: EnvironmentSemantics::deterministic_single_agent_general_reward(),
            max_horizon: Some(1),
            agents: AgentModel::fixed_homogeneous([AgentId(3)], ActionSpace::discrete(4)),
            preferred_batch: 1,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("mock", "Mock")
    }

    fn reset(
        &mut self,
        _seed: u64,
        _hint: &[u8],
        out_state: &mut Vec<u8>,
        out_timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        out_state.clear();
        out_state.extend_from_slice(&0u32.to_le_bytes());
        *out_timestep = Self::timestep(0, TransitionSource::Reset);
        Ok(())
    }

    fn step(
        &mut self,
        _state: &[u8],
        _action: &[u8],
        out_state: &mut Vec<u8>,
        out_timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        out_state.clear();
        out_state.extend_from_slice(&1u32.to_le_bytes());
        *out_timestep = Self::timestep(
            1,
            TransitionSource::Agents {
                agent_ids: vec![AgentId(3)],
            },
        );
        Ok(())
    }

    fn presentation(
        &self,
        _state: &[u8],
    ) -> Result<Option<crate::Presentation>, ErasedEnvironmentError> {
        Ok(None)
    }
}

#[test]
fn context_returns_full_timesteps_without_scalar_shortcuts() {
    let mut context = EngineContext::from_erased(Box::new(MockEnvironment)).unwrap();
    let reset = context.reset(42, &[]).unwrap();
    assert_eq!(
        reset.timestep.decision,
        Decision::Agents {
            agent_ids: vec![AgentId(3)]
        }
    );
    assert_eq!(
        reset.timestep.observation_for(AgentId(3)),
        Some(&0u32.to_le_bytes()[..])
    );

    let step = context.step(&reset.state, &0u32.to_le_bytes()).unwrap();
    assert_eq!(step.timestep.reward_for(AgentId(3)), Some(1.0));
    assert_eq!(step.timestep.episode, EpisodeStatus::Terminated);
}
