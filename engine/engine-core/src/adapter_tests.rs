use rand_chacha::ChaCha20Rng;

use super::*;
use crate::erased::ErasedEnvironment;
use crate::metadata::{BoardGameMetadata, BoardPlayerMetadata, EnvironmentMetadata};
use crate::typed::{
    ActionEncoding, ActionSpace, AgentId, AgentModel, AgentObservation, AgentOutcome, Capabilities,
    ChanceModel, Decision, DecodeError, EncodeError, Encoding, EngineId, Environment,
    EnvironmentError, EnvironmentSemantics, EpisodeStatus, ObservationEncoding, PlanningStateModel,
    RewardModel, TensorSpec, Timestep, TransitionDynamics, TransitionSource, TurnModel,
};

#[derive(Debug)]
struct CounterEnvironment {
    invalid_reward: bool,
}

impl CounterEnvironment {
    fn timestep(value: u32, reward: f32, source: TransitionSource) -> Timestep<f32> {
        let done = value >= 2;
        Timestep {
            agents: vec![AgentId(7)],
            observations: vec![AgentObservation {
                agent_id: AgentId(7),
                observation: value as f32,
            }],
            outcomes: vec![AgentOutcome {
                agent_id: AgentId(7),
                reward,
                terminated: done,
                truncated: false,
            }],
            decision: if done {
                Decision::None
            } else {
                Decision::agents([AgentId(7)])
            },
            episode: if done {
                EpisodeStatus::Terminated
            } else {
                EpisodeStatus::Running
            },
            source,
            info: value.to_le_bytes().to_vec(),
        }
    }
}

impl Environment for CounterEnvironment {
    type State = u32;
    type Action = u32;
    type Observation = f32;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "counter".into(),
            build_id: "test".into(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: 1,
            encoding: Encoding::custom("counter-state:v1", "increment:v1", "scalar:v1"),
            semantics: EnvironmentSemantics::deterministic_single_agent_general_reward(),
            max_horizon: Some(2),
            agents: AgentModel::fixed_homogeneous([AgentId(7)], ActionSpace::discrete(2)),
            preferred_batch: 1,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("counter", "Counter")
    }

    fn reset(
        &mut self,
        _rng: &mut ChaCha20Rng,
        _hint: &[u8],
    ) -> Result<(Self::State, Timestep<Self::Observation>), EnvironmentError> {
        let reward = if self.invalid_reward { f32::NAN } else { 0.0 };
        Ok((0, Self::timestep(0, reward, TransitionSource::Reset)))
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> Result<Timestep<Self::Observation>, EnvironmentError> {
        *state += action;
        Ok(Self::timestep(
            *state,
            action as f32,
            TransitionSource::Agents {
                agent_ids: vec![AgentId(7)],
            },
        ))
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.extend_from_slice(&state.to_le_bytes());
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        let bytes: [u8; 4] = buf.try_into().map_err(|_| DecodeError::InvalidLength {
            expected: 4,
            actual: buf.len(),
        })?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.extend_from_slice(&action.to_le_bytes());
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        Self::decode_state(buf)
    }

    fn encode_observation(
        observation: &Self::Observation,
        out: &mut Vec<u8>,
    ) -> Result<(), EncodeError> {
        out.extend_from_slice(&observation.to_le_bytes());
        Ok(())
    }
}

#[test]
fn adapter_preserves_generic_timestep_and_has_no_board_projection() {
    let mut adapter = EnvironmentAdapter::new(CounterEnvironment {
        invalid_reward: false,
    });
    let mut state = Vec::new();
    let mut timestep = crate::ErasedTimestep::default();
    adapter.reset(41, &[], &mut state, &mut timestep).unwrap();

    assert_eq!(u32::from_le_bytes(state.clone().try_into().unwrap()), 0);
    assert_eq!(
        timestep.observation_for(AgentId(7)),
        Some(&0.0f32.to_le_bytes()[..])
    );
    assert_eq!(timestep.source, TransitionSource::Reset);
    assert!(adapter.presentation(&state).unwrap().is_none());
    assert!(adapter.metadata().board.is_none());

    let mut next_state = Vec::new();
    adapter
        .step(&state, &2u32.to_le_bytes(), &mut next_state, &mut timestep)
        .unwrap();
    assert_eq!(timestep.reward_for(AgentId(7)), Some(2.0));
    assert_eq!(timestep.episode, EpisodeStatus::Terminated);
    assert_eq!(timestep.decision, Decision::None);
}

#[test]
fn engine_context_rejects_non_finite_per_agent_rewards() {
    let mut context = crate::EngineContext::from_environment(CounterEnvironment {
        invalid_reward: true,
    })
    .unwrap();
    let error = context.reset(0, &[]).unwrap_err();
    assert!(matches!(
        error,
        crate::ErasedEnvironmentError::ContractViolation(_)
    ));
}

#[test]
fn adapter_rejects_malformed_encoded_inputs() {
    let mut adapter = EnvironmentAdapter::new(CounterEnvironment {
        invalid_reward: false,
    });
    let error = adapter
        .step(
            &[1, 2],
            &1u32.to_le_bytes(),
            &mut Vec::new(),
            &mut crate::ErasedTimestep::default(),
        )
        .unwrap_err();
    assert!(matches!(error, crate::ErasedEnvironmentError::Decoding(_)));
}

fn capabilities_with_action_space(action_space: ActionSpace) -> Capabilities {
    let environment = CounterEnvironment {
        invalid_reward: false,
    };
    let mut capabilities = environment.capabilities();
    capabilities.agents = AgentModel::fixed_homogeneous([AgentId(7)], action_space);
    capabilities
}

#[test]
fn adapter_rejects_continuous_bounds_that_do_not_match_shape() {
    let malformed = ActionSpace::Continuous {
        low: vec![0.0, 0.0],
        high: vec![1.0, 1.0],
        shape: vec![3],
    };

    let error = crate::contract::validate_action_space(&malformed).unwrap_err();
    assert!(matches!(
        error,
        crate::ErasedEnvironmentError::ContractViolation(_)
    ));
}

#[test]
fn adapter_rejects_discrete_encoding_for_non_discrete_actions() {
    let mut capabilities = capabilities_with_action_space(ActionSpace::MultiDiscrete {
        dimensions: vec![2, 3],
    });
    capabilities.encoding.action = ActionEncoding::DiscreteU32LittleEndian;

    let error = crate::contract::validate_encoding(&capabilities).unwrap_err();
    assert!(matches!(
        error,
        crate::ErasedEnvironmentError::ContractViolation(_)
    ));
}

#[test]
fn adapter_rejects_empty_custom_codec_ids() {
    let mut capabilities = capabilities_with_action_space(ActionSpace::discrete(2));
    capabilities.encoding.action = ActionEncoding::Custom { id: " ".into() };
    capabilities.encoding.observation = ObservationEncoding::Custom {
        id: "observation:v1".into(),
    };
    assert!(crate::contract::validate_encoding(&capabilities).is_err());

    capabilities.encoding.action = ActionEncoding::Custom {
        id: "action:v1".into(),
    };
    capabilities.encoding.observation = ObservationEncoding::Custom { id: "".into() };
    assert!(crate::contract::validate_encoding(&capabilities).is_err());
}

#[test]
fn adapter_rejects_environment_sampled_rng_as_a_complete_snapshot() {
    let environment = CounterEnvironment {
        invalid_reward: false,
    };
    let id = environment.engine_id();
    let metadata = environment.metadata();
    let mut capabilities = environment.capabilities();
    capabilities.semantics.transition_dynamics = TransitionDynamics::Stochastic;
    capabilities.semantics.chance_model = ChanceModel::EnvironmentSampled;
    capabilities.semantics.planning_state_model = PlanningStateModel::CompleteSnapshot;

    let error = crate::contract::validate_descriptors(&id, &capabilities, &metadata).unwrap_err();
    assert!(error.to_string().contains("runtime RNG state"));

    capabilities.semantics.planning_state_model = PlanningStateModel::ExternalState;
    crate::contract::validate_descriptors(&id, &capabilities, &metadata).unwrap();
}

#[test]
fn descriptors_require_custom_joint_actions_for_simultaneous_decisions() {
    let environment = CounterEnvironment {
        invalid_reward: false,
    };
    let id = environment.engine_id();
    let metadata = environment.metadata();
    let mut capabilities = environment.capabilities();
    capabilities.semantics.turn_model = TurnModel::Simultaneous;
    capabilities.encoding.action = ActionEncoding::DiscreteU32LittleEndian;

    let error = crate::contract::validate_descriptors(&id, &capabilities, &metadata).unwrap_err();
    assert!(error.to_string().contains("custom joint-action codec"));

    capabilities.encoding.action = ActionEncoding::Custom {
        id: "joint-action:v1".into(),
    };
    crate::contract::validate_descriptors(&id, &capabilities, &metadata).unwrap();
}

#[test]
fn descriptors_require_custom_actions_for_explicit_chance() {
    let environment = CounterEnvironment {
        invalid_reward: false,
    };
    let id = environment.engine_id();
    let metadata = environment.metadata();
    let mut capabilities = environment.capabilities();
    capabilities.semantics.transition_dynamics = TransitionDynamics::Stochastic;
    capabilities.semantics.chance_model = ChanceModel::Explicit;
    capabilities.encoding.action = ActionEncoding::DiscreteU32LittleEndian;

    let error = crate::contract::validate_descriptors(&id, &capabilities, &metadata).unwrap_err();
    assert!(error
        .to_string()
        .contains("custom chance/agent action codec"));

    capabilities.encoding.action = ActionEncoding::Custom {
        id: "chance-or-agent:v1".into(),
    };
    crate::contract::validate_descriptors(&id, &capabilities, &metadata).unwrap();
}

#[test]
fn descriptors_reject_single_agent_semantics_with_multiple_fixed_agents() {
    let environment = CounterEnvironment {
        invalid_reward: false,
    };
    let id = environment.engine_id();
    let metadata = environment.metadata();
    let mut capabilities = environment.capabilities();
    capabilities.agents =
        AgentModel::fixed_homogeneous([AgentId(7), AgentId(8)], ActionSpace::discrete(2));

    let error = crate::contract::validate_descriptors(&id, &capabilities, &metadata).unwrap_err();
    assert!(error.to_string().contains("exactly one fixed agent"));
}

#[test]
fn descriptors_reject_environment_ids_that_are_not_safe_runtime_segments() {
    let environment = CounterEnvironment {
        invalid_reward: false,
    };
    let mut id = environment.engine_id();
    let mut capabilities = environment.capabilities();
    let mut metadata = environment.metadata();
    id.env_id = "../Counter".into();
    capabilities.id = id.clone();
    metadata.id = id.env_id.clone();

    let error = crate::contract::validate_descriptors(&id, &capabilities, &metadata).unwrap_err();
    assert!(error.to_string().contains("lowercase ASCII"));
}

#[test]
fn timestep_rejects_multiple_agents_under_single_agent_semantics() {
    let (mut capabilities, timestep) = simultaneous_contract();
    capabilities.agents = AgentModel::Dynamic {
        action_space: ActionSpace::discrete(2),
        action_availability: crate::ActionAvailabilityContract::All,
    };
    capabilities.semantics.turn_model = TurnModel::SingleAgent;

    let error = crate::contract::validate_typed_timestep(&capabilities, &timestep).unwrap_err();
    assert!(error.to_string().contains("at most one agent"));
}

fn valid_board_descriptors() -> (EngineId, Capabilities, EnvironmentMetadata) {
    let id = EngineId {
        env_id: "board-test".into(),
        build_id: "test".into(),
    };
    let capabilities = Capabilities {
        id: id.clone(),
        contract_version: 1,
        encoding: Encoding::discrete_u32_le(
            "board-state:v1",
            TensorSpec::f32_fixed([("channel", 2), ("row", 3), ("column", 3)]),
        ),
        semantics:
            EnvironmentSemantics::deterministic_alternating_perfect_information_terminal_zero_sum(),
        max_horizon: Some(9),
        agents: AgentModel::fixed_homogeneous([AgentId(1), AgentId(2)], ActionSpace::discrete(9)),
        preferred_batch: 1,
    };
    let metadata = EnvironmentMetadata::new("board-test", "Board Test").with_board(
        BoardGameMetadata::new(3, 3).with_players(vec![
            BoardPlayerMetadata::new("One", "1"),
            BoardPlayerMetadata::new("Two", "2"),
        ]),
    );
    (id, capabilities, metadata)
}

#[test]
fn descriptors_validate_the_entire_board_profile() {
    let (id, capabilities, metadata) = valid_board_descriptors();
    crate::contract::validate_descriptors(&id, &capabilities, &metadata).unwrap();

    let mut malformed = metadata.clone();
    malformed.board.as_mut().unwrap().players.pop();
    let error = crate::contract::validate_descriptors(&id, &capabilities, &malformed).unwrap_err();
    assert!(error.to_string().contains("exactly two players"));

    let mut overflow = metadata;
    let board = overflow.board.as_mut().unwrap();
    board.width = usize::MAX;
    board.height = 2;
    let error = crate::contract::validate_descriptors(&id, &capabilities, &overflow).unwrap_err();
    assert!(error.to_string().contains("overflow"));
}

#[test]
fn adapter_validates_declared_f32_observation_payloads() {
    let mut capabilities = capabilities_with_action_space(ActionSpace::discrete(2));
    capabilities.encoding.observation = ObservationEncoding::Tensor {
        spec: TensorSpec::f32_fixed([("feature", 2)]),
    };

    assert!(crate::contract::validate_encoded_observation(
        &capabilities,
        AgentId(7),
        &1.0f32.to_le_bytes(),
    )
    .is_err());

    let mut non_finite = 1.0f32.to_le_bytes().to_vec();
    non_finite.extend_from_slice(&f32::NAN.to_le_bytes());
    assert!(
        crate::contract::validate_encoded_observation(&capabilities, AgentId(7), &non_finite,)
            .is_err()
    );

    let mut valid = 1.0f32.to_le_bytes().to_vec();
    valid.extend_from_slice(&2.0f32.to_le_bytes());
    crate::contract::validate_encoded_observation(&capabilities, AgentId(7), &valid).unwrap();
}

fn simultaneous_contract() -> (Capabilities, Timestep<f32>) {
    let mut capabilities = capabilities_with_action_space(ActionSpace::discrete(2));
    capabilities.agents =
        AgentModel::fixed_homogeneous([AgentId(7), AgentId(8)], ActionSpace::discrete(2));
    capabilities.semantics.turn_model = TurnModel::Simultaneous;
    let timestep = Timestep {
        agents: vec![AgentId(7), AgentId(8)],
        observations: vec![
            AgentObservation {
                agent_id: AgentId(7),
                observation: 1.0,
            },
            AgentObservation {
                agent_id: AgentId(8),
                observation: 2.0,
            },
        ],
        outcomes: vec![
            AgentOutcome {
                agent_id: AgentId(7),
                reward: 0.0,
                terminated: false,
                truncated: false,
            },
            AgentOutcome {
                agent_id: AgentId(8),
                reward: 0.0,
                terminated: false,
                truncated: false,
            },
        ],
        decision: Decision::agents([AgentId(7), AgentId(8)]),
        episode: EpisodeStatus::Running,
        source: TransitionSource::Chance,
        info: Vec::new(),
    };
    (capabilities, timestep)
}

#[test]
fn adapter_requires_an_observation_for_every_decision_agent() {
    let (capabilities, mut timestep) = simultaneous_contract();
    timestep
        .observations
        .retain(|observation| observation.agent_id != AgentId(8));

    let error = crate::contract::validate_typed_timestep(&capabilities, &timestep).unwrap_err();
    assert!(error
        .to_string()
        .contains("decision agent 8 has no observation"));
}

#[test]
fn adapter_never_requests_an_action_from_a_completed_agent() {
    let (capabilities, mut timestep) = simultaneous_contract();
    timestep.outcomes[1].terminated = true;

    let error = crate::contract::validate_typed_timestep(&capabilities, &timestep).unwrap_err();
    assert!(error
        .to_string()
        .contains("decision agent 8 is already complete"));
}

#[test]
fn timestep_requires_global_episode_status_to_match_every_current_agent() {
    let (capabilities, mut timestep) = simultaneous_contract();
    timestep.episode = EpisodeStatus::Terminated;
    timestep.decision = Decision::None;
    timestep.outcomes[0].terminated = true;

    let error = crate::contract::validate_typed_timestep(&capabilities, &timestep).unwrap_err();
    assert!(error.to_string().contains("every current agent"));
}

#[test]
fn timestep_enforces_declared_terminal_zero_sum_rewards() {
    let (mut capabilities, mut timestep) = simultaneous_contract();
    capabilities.semantics.reward_model = RewardModel::TerminalZeroSum;
    timestep.source = TransitionSource::Reset;
    timestep.outcomes[0].reward = 0.25;
    let error = crate::contract::validate_typed_timestep(&capabilities, &timestep).unwrap_err();
    assert!(error.to_string().contains("zero reward before termination"));

    timestep.episode = EpisodeStatus::Terminated;
    timestep.decision = Decision::None;
    for outcome in &mut timestep.outcomes {
        outcome.terminated = true;
        outcome.reward = 1.0;
    }
    let error = crate::contract::validate_typed_timestep(&capabilities, &timestep).unwrap_err();
    assert!(error.to_string().contains("sum to zero"));

    timestep.outcomes[1].reward = -1.0;
    crate::contract::validate_typed_timestep(&capabilities, &timestep).unwrap();
}

#[test]
fn dynamic_departure_stays_in_the_transition_roster_for_its_final_outcome() {
    let (mut capabilities, mut timestep) = simultaneous_contract();
    capabilities.agents = AgentModel::Dynamic {
        action_space: ActionSpace::discrete(2),
        action_availability: crate::ActionAvailabilityContract::All,
    };
    capabilities.semantics.turn_model = TurnModel::Sequential {
        order: crate::typed::SequentialTurnOrder::EnvironmentDefined,
    };
    timestep
        .observations
        .retain(|item| item.agent_id == AgentId(8));
    timestep.outcomes[0].terminated = true;
    timestep.decision = Decision::agents([AgentId(8)]);
    timestep.source = TransitionSource::Agents {
        agent_ids: vec![AgentId(7)],
    };

    crate::contract::validate_typed_timestep(&capabilities, &timestep).unwrap();

    timestep.agents.retain(|agent_id| *agent_id != AgentId(7));
    timestep
        .outcomes
        .retain(|outcome| outcome.agent_id != AgentId(7));
    let error = crate::contract::validate_typed_timestep(&capabilities, &timestep).unwrap_err();
    assert!(error.to_string().contains("transition roster"));
}
