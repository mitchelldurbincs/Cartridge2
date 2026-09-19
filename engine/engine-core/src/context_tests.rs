use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::erased::{
    EncodedObservation, ErasedEnvironment, ErasedEnvironmentError, ErasedTimestep,
};
use crate::metadata::EnvironmentMetadata;
use crate::typed::{
    ActionSpace, AgentId, AgentModel, AgentOutcome, Capabilities, Decision, Encoding, EngineId,
    EnvironmentSemantics, EpisodeStatus, ObservationEncoding, TensorSpec, TransitionSource,
};
use crate::EngineContext;

#[derive(Debug, Clone, Copy, Default)]
enum Fault {
    #[default]
    None,
    Source,
    Reward,
    Observation,
    Descriptors,
    InvalidDescriptors,
    Environment,
    Encoding,
    Decoding,
}

#[derive(Debug, Default)]
struct MockEnvironment {
    fault: Fault,
    calls: Arc<AtomicUsize>,
}

impl MockEnvironment {
    fn timestep(value: u32, source: TransitionSource) -> ErasedTimestep {
        ErasedTimestep {
            agents: vec![AgentId(3)],
            observations: vec![EncodedObservation {
                agent_id: AgentId(3),
                data: (value as f32).to_le_bytes().to_vec(),
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
                Decision::agents([AgentId(3)])
            },
            episode: if value >= 1 {
                EpisodeStatus::Terminated
            } else {
                EpisodeStatus::Running
            },
            source,
            info: if value == 0 { vec![42] } else { vec![] },
        }
    }

    fn emit(&self, timestep: &mut ErasedTimestep) -> Result<(), ErasedEnvironmentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.fault {
            Fault::Source => {
                // Also corrupt the reward to verify source errors take precedence.
                timestep.outcomes[0].reward = f32::NAN;
                timestep.source = match timestep.source {
                    TransitionSource::Reset => TransitionSource::Agents {
                        agent_ids: vec![AgentId(3)],
                    },
                    _ => TransitionSource::Reset,
                };
            }
            Fault::Reward => timestep.outcomes[0].reward = f32::NAN,
            Fault::Observation => timestep.observations[0].data.truncate(3),
            Fault::Environment => {
                return Err(ErasedEnvironmentError::Environment("fixture".into()))
            }
            Fault::Encoding => return Err(ErasedEnvironmentError::Encoding("fixture".into())),
            Fault::Decoding => return Err(ErasedEnvironmentError::Decoding("fixture".into())),
            Fault::None | Fault::Descriptors | Fault::InvalidDescriptors => {}
        }
        Ok(())
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
            contract_version: match (self.fault, self.calls.load(Ordering::SeqCst) > 0) {
                (Fault::Descriptors, true) => 2,
                (Fault::InvalidDescriptors, true) => 0,
                _ => 1,
            },
            encoding: Encoding {
                observation: ObservationEncoding::Tensor {
                    spec: TensorSpec::f32_fixed([("value", 1)]),
                },
                ..Encoding::custom("state:v1", "action:v1", "obs:v1")
            },
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
        // The context must clear caller buffers before invoking the environment.
        assert!(out_state.is_empty());
        out_state.extend_from_slice(&0u32.to_le_bytes());
        *out_timestep = Self::timestep(0, TransitionSource::Reset);
        self.emit(out_timestep)
    }

    fn step(
        &mut self,
        _state: &[u8],
        _action: &[u8],
        out_state: &mut Vec<u8>,
        out_timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        assert!(out_state.is_empty());
        out_state.extend_from_slice(&1u32.to_le_bytes());
        *out_timestep = Self::timestep(
            1,
            TransitionSource::Agents {
                agent_ids: vec![AgentId(3)],
            },
        );
        self.emit(out_timestep)
    }

    fn presentation(
        &self,
        _state: &[u8],
    ) -> Result<Option<crate::Presentation>, ErasedEnvironmentError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    }
}

#[test]
fn context_returns_full_timesteps_without_scalar_shortcuts() {
    let mut context = EngineContext::from_erased(Box::<MockEnvironment>::default()).unwrap();
    let reset = context.reset(42, &[]).unwrap();
    assert_eq!(reset.timestep.decision, Decision::agents([AgentId(3)]));
    assert_eq!(
        reset.timestep.observation_for(AgentId(3)),
        Some(&0u32.to_le_bytes()[..])
    );

    let step = context.step(&reset.state, &0u32.to_le_bytes()).unwrap();
    assert_eq!(step.timestep.reward_for(AgentId(3)), Some(1.0));
    assert_eq!(step.timestep.episode, EpisodeStatus::Terminated);
}

#[test]
fn owned_and_buffer_apis_match_when_outputs_are_reused() {
    let mut owned = EngineContext::from_erased(Box::<MockEnvironment>::default()).unwrap();
    let mut buffered = EngineContext::from_erased(Box::<MockEnvironment>::default()).unwrap();
    let mut state = vec![255; 16];
    let mut timestep = ErasedTimestep::default();

    for seed in [42, 7, 42] {
        let reset = owned.reset(seed, &[]).unwrap();
        buffered
            .reset_into(seed, &[], &mut state, &mut timestep)
            .unwrap();
        assert_eq!(state, reset.state);
        assert_eq!(timestep, reset.timestep);

        let step = owned.step(&reset.state, &0u32.to_le_bytes()).unwrap();
        let previous_state = state.clone();
        buffered
            .step_into(
                &previous_state,
                &0u32.to_le_bytes(),
                &mut state,
                &mut timestep,
            )
            .unwrap();
        assert_eq!(state, step.state);
        assert_eq!(timestep, step.timestep);
        assert!(timestep.info.is_empty());
        assert_eq!(reset.timestep.info, vec![42]);
    }
}

#[derive(Debug, Clone, Copy)]
enum TransitionApi {
    Reset,
    ResetInto,
    Step,
    StepInto,
}

impl TransitionApi {
    const ALL: [Self; 4] = [Self::Reset, Self::ResetInto, Self::Step, Self::StepInto];

    fn invoke(
        self,
        context: &mut EngineContext,
        state: &mut Vec<u8>,
        timestep: &mut ErasedTimestep,
    ) -> Result<(), ErasedEnvironmentError> {
        match self {
            Self::Reset => {
                let result = context.reset(42, &[])?;
                *state = result.state;
                *timestep = result.timestep;
                Ok(())
            }
            Self::ResetInto => context.reset_into(42, &[], state, timestep),
            Self::Step => {
                let result = context.step(&0u32.to_le_bytes(), &0u32.to_le_bytes())?;
                *state = result.state;
                *timestep = result.timestep;
                Ok(())
            }
            Self::StepInto => {
                context.step_into(&0u32.to_le_bytes(), &0u32.to_le_bytes(), state, timestep)
            }
        }
    }
}

#[test]
fn all_transition_apis_preserve_validation_and_error_categories() {
    for api in TransitionApi::ALL {
        for fault in [
            Fault::Source,
            Fault::Reward,
            Fault::Observation,
            Fault::Environment,
            Fault::Encoding,
            Fault::Decoding,
        ] {
            let mut context = EngineContext::from_erased(Box::new(MockEnvironment {
                fault,
                ..Default::default()
            }))
            .unwrap();
            let error = api
                .invoke(
                    &mut context,
                    &mut vec![255; 16],
                    &mut ErasedTimestep::default(),
                )
                .unwrap_err();
            let expected = match fault {
                Fault::Source => ErasedEnvironmentError::ContractViolation(match api {
                    TransitionApi::Reset | TransitionApi::ResetInto => {
                        "reset must produce transition source=reset".into()
                    }
                    TransitionApi::Step | TransitionApi::StepInto => {
                        "step cannot produce transition source=reset".into()
                    }
                }),
                Fault::Reward => {
                    ErasedEnvironmentError::ContractViolation("invalid outcome for agent 3".into())
                }
                Fault::Observation => ErasedEnvironmentError::ContractViolation(
                    "observation for agent 3 encoded 3 bytes, expected 4".into(),
                ),
                Fault::Environment => ErasedEnvironmentError::Environment("fixture".into()),
                Fault::Encoding => ErasedEnvironmentError::Encoding("fixture".into()),
                Fault::Decoding => ErasedEnvironmentError::Decoding("fixture".into()),
                _ => unreachable!(),
            };
            assert_eq!(
                std::mem::discriminant(&error),
                std::mem::discriminant(&expected),
                "{api:?} with {fault:?}",
            );
            assert_eq!(
                error.to_string(),
                expected.to_string(),
                "{api:?} with {fault:?}"
            );
        }
    }
}

#[test]
fn all_transition_apis_reject_descriptor_drift_before_invocation() {
    for (fault, expected) in descriptor_faults() {
        for api in TransitionApi::ALL {
            let calls = Arc::new(AtomicUsize::new(0));
            let mut context = EngineContext::from_erased(Box::new(MockEnvironment {
                fault,
                calls: calls.clone(),
            }))
            .unwrap();
            // The first invocation changes the descriptor; the next must fail preflight.
            context.reset(42, &[]).unwrap();
            let original_state = vec![255; 16];
            let original_timestep = MockEnvironment::timestep(0, TransitionSource::Reset);
            let mut state = original_state.clone();
            let mut timestep = original_timestep.clone();
            let error = api
                .invoke(&mut context, &mut state, &mut timestep)
                .unwrap_err();
            assert!(
                matches!(error, ErasedEnvironmentError::ContractViolation(ref message)
            if message == expected),
                "{api:?} with {fault:?}: {error}"
            );
            assert_eq!(calls.load(Ordering::SeqCst), 1, "{api:?}");
            assert_eq!(state, original_state, "{api:?}");
            assert_eq!(timestep, original_timestep, "{api:?}");
        }
    }
}

fn descriptor_faults() -> [(Fault, &'static str); 2] {
    [
        (
            Fault::Descriptors,
            "environment descriptors changed after context construction",
        ),
        (
            Fault::InvalidDescriptors,
            "contract_version must be positive",
        ),
    ]
}

#[test]
fn presentation_rejects_valid_and_invalid_descriptor_drift_before_invocation() {
    for (fault, expected) in descriptor_faults() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut context = EngineContext::from_erased(Box::new(MockEnvironment {
            fault,
            calls: calls.clone(),
        }))
        .unwrap();
        let reset = context.reset(42, &[]).unwrap();
        let error = context.presentation(&reset.state).unwrap_err();
        assert!(
            matches!(error, ErasedEnvironmentError::ContractViolation(ref message)
            if message == expected)
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1, "{fault:?}");
    }
}

#[test]
fn context_rejects_invalid_descriptors_at_construction() {
    let error = EngineContext::from_erased(Box::new(MockEnvironment {
        fault: Fault::InvalidDescriptors,
        calls: Arc::new(AtomicUsize::new(1)),
    }))
    .unwrap_err();
    assert!(
        matches!(error, ErasedEnvironmentError::ContractViolation(ref message)
        if message == "contract_version must be positive")
    );
}
