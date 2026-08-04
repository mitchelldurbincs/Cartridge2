use engine_core::{
    ActionAvailabilityContract, ActionEncoding, ActionSpace, AgentId, AgentModel, AgentObservation,
    AgentOutcome, Capabilities, ChanceModel, Decision, DecodeError, EncodeError, Encoding,
    EngineContext, EngineId, Environment, EnvironmentError, EnvironmentMetadata,
    EnvironmentSemantics, EpisodeStatus, InformationModel, PlanningStateModel, RewardModel,
    SequentialTurnOrder, TensorSpec, Timestep, TransitionDynamics, TransitionSource, TurnModel,
};
use rand_chacha::ChaCha20Rng;

fn decode_u8(buf: &[u8]) -> Result<u8, DecodeError> {
    match buf {
        [value] => Ok(*value),
        _ => Err(DecodeError::InvalidLength {
            expected: 1,
            actual: buf.len(),
        }),
    }
}

fn encode_f32(value: &f32, out: &mut Vec<u8>) -> Result<(), EncodeError> {
    out.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

#[derive(Debug, Default)]
struct SimultaneousEnvironment;

impl Environment for SimultaneousEnvironment {
    type State = u8;
    type Action = (u32, u32);
    type Observation = f32;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "simultaneous_fixture".into(),
            build_id: "test".into(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: 1,
            encoding: Encoding::custom("sim-state:v1", "joint-pair:v1", "private-f32:v1"),
            semantics: EnvironmentSemantics {
                turn_model: TurnModel::Simultaneous,
                information_model: InformationModel::PartiallyObserved,
                planning_state_model: PlanningStateModel::CompleteSnapshot,
                transition_dynamics: TransitionDynamics::Deterministic,
                chance_model: ChanceModel::None,
                reward_model: RewardModel::General,
            },
            max_horizon: Some(1),
            agents: AgentModel::fixed_homogeneous(
                [AgentId(10), AgentId(20)],
                ActionSpace::discrete(2),
            ),
            preferred_batch: 2,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("simultaneous_fixture", "Simultaneous fixture")
    }

    fn reset(
        &mut self,
        _rng: &mut ChaCha20Rng,
        _hint: &[u8],
    ) -> Result<(Self::State, Timestep<Self::Observation>), EnvironmentError> {
        Ok((
            0,
            Timestep {
                agents: vec![AgentId(10), AgentId(20)],
                observations: vec![
                    AgentObservation {
                        agent_id: AgentId(10),
                        observation: 10.0,
                    },
                    AgentObservation {
                        agent_id: AgentId(20),
                        observation: 20.0,
                    },
                ],
                outcomes: vec![
                    AgentOutcome {
                        agent_id: AgentId(10),
                        reward: 0.0,
                        terminated: false,
                        truncated: false,
                    },
                    AgentOutcome {
                        agent_id: AgentId(20),
                        reward: 0.0,
                        terminated: false,
                        truncated: false,
                    },
                ],
                decision: Decision::agents([AgentId(10), AgentId(20)]),
                episode: EpisodeStatus::Running,
                source: TransitionSource::Reset,
                info: Vec::new(),
            },
        ))
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> Result<Timestep<Self::Observation>, EnvironmentError> {
        if *state != 0 || action.0 >= 2 || action.1 >= 2 {
            return Err(EnvironmentError::InvalidAction(
                "invalid joint action".into(),
            ));
        }
        *state = 1;
        let reward = action.0 as f32 - action.1 as f32;
        Ok(Timestep {
            agents: vec![AgentId(10), AgentId(20)],
            observations: vec![],
            outcomes: vec![
                AgentOutcome {
                    agent_id: AgentId(10),
                    reward,
                    terminated: true,
                    truncated: false,
                },
                AgentOutcome {
                    agent_id: AgentId(20),
                    reward: -reward,
                    terminated: true,
                    truncated: false,
                },
            ],
            decision: Decision::None,
            episode: EpisodeStatus::Terminated,
            source: TransitionSource::Agents {
                agent_ids: vec![AgentId(10), AgentId(20)],
            },
            info: Vec::new(),
        })
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.push(*state);
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        decode_u8(buf)
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.extend_from_slice(&action.0.to_le_bytes());
        out.extend_from_slice(&action.1.to_le_bytes());
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        if buf.len() != 8 {
            return Err(DecodeError::InvalidLength {
                expected: 8,
                actual: buf.len(),
            });
        }
        Ok((
            u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            u32::from_le_bytes(buf[4..8].try_into().unwrap()),
        ))
    }

    fn encode_observation(
        observation: &Self::Observation,
        out: &mut Vec<u8>,
    ) -> Result<(), EncodeError> {
        encode_f32(observation, out)
    }
}

#[derive(Debug, Clone, Copy)]
enum ChanceAction {
    Resolve(u8),
    Agent(u32),
}

#[derive(Debug, Default)]
struct ExplicitChanceEnvironment;

impl Environment for ExplicitChanceEnvironment {
    type State = u8;
    type Action = ChanceAction;
    type Observation = f32;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "chance_fixture".into(),
            build_id: "test".into(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: 1,
            encoding: Encoding {
                state: "chance-state:v1".into(),
                action: ActionEncoding::Custom {
                    id: "chance-or-agent:v1".into(),
                },
                observation: engine_core::ObservationEncoding::Tensor {
                    spec: TensorSpec::f32_fixed([("feature", 1)]),
                },
                schema_version: engine_core::WIRE_ENCODING_SCHEMA_VERSION,
            },
            semantics: EnvironmentSemantics {
                turn_model: TurnModel::SingleAgent,
                information_model: InformationModel::PerfectInformationMarkov,
                planning_state_model: PlanningStateModel::CompleteSnapshot,
                transition_dynamics: TransitionDynamics::Stochastic,
                chance_model: ChanceModel::Explicit,
                reward_model: RewardModel::General,
            },
            max_horizon: Some(2),
            agents: AgentModel::fixed_homogeneous([AgentId(5)], ActionSpace::discrete(2)),
            preferred_batch: 1,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("chance_fixture", "Explicit chance fixture")
    }

    fn reset(
        &mut self,
        _rng: &mut ChaCha20Rng,
        _hint: &[u8],
    ) -> Result<(Self::State, Timestep<Self::Observation>), EnvironmentError> {
        Ok((
            0,
            Timestep {
                agents: vec![AgentId(5)],
                observations: vec![],
                outcomes: vec![AgentOutcome {
                    agent_id: AgentId(5),
                    reward: 0.0,
                    terminated: false,
                    truncated: false,
                }],
                decision: Decision::Chance,
                episode: EpisodeStatus::Running,
                source: TransitionSource::Reset,
                info: Vec::new(),
            },
        ))
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> Result<Timestep<Self::Observation>, EnvironmentError> {
        match (*state, action) {
            (0, ChanceAction::Resolve(outcome)) => {
                *state = 1;
                Ok(Timestep {
                    agents: vec![AgentId(5)],
                    observations: vec![AgentObservation {
                        agent_id: AgentId(5),
                        observation: f32::from(outcome),
                    }],
                    outcomes: vec![AgentOutcome {
                        agent_id: AgentId(5),
                        reward: 0.0,
                        terminated: false,
                        truncated: false,
                    }],
                    decision: Decision::agents([AgentId(5)]),
                    episode: EpisodeStatus::Running,
                    source: TransitionSource::Chance,
                    info: Vec::new(),
                })
            }
            (1, ChanceAction::Agent(action)) if action < 2 => {
                *state = 2;
                Ok(Timestep {
                    agents: vec![AgentId(5)],
                    observations: vec![],
                    outcomes: vec![AgentOutcome {
                        agent_id: AgentId(5),
                        reward: action as f32,
                        terminated: true,
                        truncated: false,
                    }],
                    decision: Decision::None,
                    episode: EpisodeStatus::Terminated,
                    source: TransitionSource::Agents {
                        agent_ids: vec![AgentId(5)],
                    },
                    info: Vec::new(),
                })
            }
            _ => Err(EnvironmentError::InvalidAction(
                "action does not resolve the current decision".into(),
            )),
        }
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.push(*state);
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        decode_u8(buf)
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        match action {
            ChanceAction::Resolve(value) => out.extend_from_slice(&[0, *value]),
            ChanceAction::Agent(value) => {
                out.push(1);
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        match buf {
            [0, value] => Ok(ChanceAction::Resolve(*value)),
            [1, value @ ..] if value.len() == 4 => Ok(ChanceAction::Agent(u32::from_le_bytes(
                value.try_into().unwrap(),
            ))),
            _ => Err(DecodeError::CorruptedData(
                "invalid chance/agent action envelope".into(),
            )),
        }
    }

    fn encode_observation(
        observation: &Self::Observation,
        out: &mut Vec<u8>,
    ) -> Result<(), EncodeError> {
        encode_f32(observation, out)
    }
}

#[derive(Debug, Default)]
struct DynamicRosterEnvironment;

impl Environment for DynamicRosterEnvironment {
    type State = u8;
    type Action = u32;
    type Observation = f32;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "dynamic_fixture".into(),
            build_id: "test".into(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: 1,
            encoding: Encoding::discrete_u32_le(
                "dynamic-state:v1",
                TensorSpec::f32_fixed([("feature", 1)]),
            ),
            semantics: EnvironmentSemantics {
                turn_model: TurnModel::Sequential {
                    order: SequentialTurnOrder::EnvironmentDefined,
                },
                information_model: InformationModel::PerfectInformationMarkov,
                planning_state_model: PlanningStateModel::CompleteSnapshot,
                transition_dynamics: TransitionDynamics::Deterministic,
                chance_model: ChanceModel::None,
                reward_model: RewardModel::General,
            },
            max_horizon: Some(2),
            agents: AgentModel::Dynamic {
                action_space: ActionSpace::discrete(1),
                action_availability: ActionAvailabilityContract::All,
            },
            preferred_batch: 1,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("dynamic_fixture", "Dynamic roster fixture")
    }

    fn reset(
        &mut self,
        _rng: &mut ChaCha20Rng,
        _hint: &[u8],
    ) -> Result<(Self::State, Timestep<Self::Observation>), EnvironmentError> {
        Ok((
            0,
            Timestep {
                agents: vec![AgentId(1)],
                observations: vec![AgentObservation {
                    agent_id: AgentId(1),
                    observation: 1.0,
                }],
                outcomes: vec![AgentOutcome {
                    agent_id: AgentId(1),
                    reward: 0.0,
                    terminated: false,
                    truncated: false,
                }],
                decision: Decision::agents([AgentId(1)]),
                episode: EpisodeStatus::Running,
                source: TransitionSource::Reset,
                info: Vec::new(),
            },
        ))
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> Result<Timestep<Self::Observation>, EnvironmentError> {
        if action != 0 {
            return Err(EnvironmentError::InvalidAction(
                "expected action zero".into(),
            ));
        }
        match *state {
            0 => {
                *state = 1;
                Ok(Timestep {
                    agents: vec![AgentId(1), AgentId(2)],
                    observations: vec![AgentObservation {
                        agent_id: AgentId(2),
                        observation: 2.0,
                    }],
                    outcomes: vec![
                        AgentOutcome {
                            agent_id: AgentId(1),
                            reward: 0.0,
                            terminated: true,
                            truncated: false,
                        },
                        AgentOutcome {
                            agent_id: AgentId(2),
                            reward: 0.0,
                            terminated: false,
                            truncated: false,
                        },
                    ],
                    decision: Decision::agents([AgentId(2)]),
                    episode: EpisodeStatus::Running,
                    source: TransitionSource::Agents {
                        agent_ids: vec![AgentId(1)],
                    },
                    info: Vec::new(),
                })
            }
            1 => {
                *state = 2;
                Ok(Timestep {
                    agents: vec![AgentId(2)],
                    observations: vec![],
                    outcomes: vec![AgentOutcome {
                        agent_id: AgentId(2),
                        reward: 1.0,
                        terminated: true,
                        truncated: false,
                    }],
                    decision: Decision::None,
                    episode: EpisodeStatus::Terminated,
                    source: TransitionSource::Agents {
                        agent_ids: vec![AgentId(2)],
                    },
                    info: Vec::new(),
                })
            }
            _ => Err(EnvironmentError::InvalidState("episode is complete".into())),
        }
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.push(*state);
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        decode_u8(buf)
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.extend_from_slice(&action.to_le_bytes());
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        let bytes: [u8; 4] = buf.try_into().map_err(|_| DecodeError::InvalidLength {
            expected: 4,
            actual: buf.len(),
        })?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn encode_observation(
        observation: &Self::Observation,
        out: &mut Vec<u8>,
    ) -> Result<(), EncodeError> {
        encode_f32(observation, out)
    }
}

#[derive(Debug, Default)]
struct MultiDiscreteEnvironment;

impl Environment for MultiDiscreteEnvironment {
    type State = u8;
    type Action = [u32; 2];
    type Observation = f32;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "multi_discrete_fixture".into(),
            build_id: "test".into(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: 1,
            encoding: Encoding::multi_discrete_u32_le(
                "multi-state:v1",
                TensorSpec::f32_fixed([("feature", 1)]),
            ),
            semantics: EnvironmentSemantics::deterministic_single_agent_general_reward(),
            max_horizon: Some(1),
            agents: AgentModel::fixed_homogeneous(
                [AgentId(0)],
                ActionSpace::MultiDiscrete {
                    dimensions: vec![2, 3],
                },
            ),
            preferred_batch: 1,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("multi_discrete_fixture", "MultiDiscrete fixture")
    }

    fn reset(
        &mut self,
        _rng: &mut ChaCha20Rng,
        _hint: &[u8],
    ) -> Result<(Self::State, Timestep<Self::Observation>), EnvironmentError> {
        Ok((0, one_agent_running(0.0)))
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> Result<Timestep<Self::Observation>, EnvironmentError> {
        if action[0] >= 2 || action[1] >= 3 {
            return Err(EnvironmentError::InvalidAction("out of bounds".into()));
        }
        *state = 1;
        Ok(one_agent_terminal(action[0] as f32 + action[1] as f32))
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.push(*state);
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        decode_u8(buf)
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        for value in action {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        if buf.len() != 8 {
            return Err(DecodeError::InvalidLength {
                expected: 8,
                actual: buf.len(),
            });
        }
        Ok([
            u32::from_le_bytes(buf[0..4].try_into().unwrap()),
            u32::from_le_bytes(buf[4..8].try_into().unwrap()),
        ])
    }

    fn encode_observation(
        observation: &Self::Observation,
        out: &mut Vec<u8>,
    ) -> Result<(), EncodeError> {
        encode_f32(observation, out)
    }
}

#[derive(Debug, Default)]
struct ContinuousEnvironment;

impl Environment for ContinuousEnvironment {
    type State = u8;
    type Action = [f32; 2];
    type Observation = f32;

    fn engine_id(&self) -> EngineId {
        EngineId {
            env_id: "continuous_fixture".into(),
            build_id: "test".into(),
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            id: self.engine_id(),
            contract_version: 1,
            encoding: Encoding::continuous_f32_le(
                "continuous-state:v1",
                TensorSpec::f32_fixed([("feature", 1)]),
            ),
            semantics: EnvironmentSemantics::deterministic_single_agent_general_reward(),
            max_horizon: Some(1),
            agents: AgentModel::fixed_homogeneous(
                [AgentId(0)],
                ActionSpace::Continuous {
                    low: vec![-1.0, -2.0],
                    high: vec![1.0, 2.0],
                    shape: vec![2],
                },
            ),
            preferred_batch: 1,
        }
    }

    fn metadata(&self) -> EnvironmentMetadata {
        EnvironmentMetadata::new("continuous_fixture", "Continuous fixture")
    }

    fn reset(
        &mut self,
        _rng: &mut ChaCha20Rng,
        _hint: &[u8],
    ) -> Result<(Self::State, Timestep<Self::Observation>), EnvironmentError> {
        Ok((0, one_agent_running(0.0)))
    }

    fn step(
        &mut self,
        state: &mut Self::State,
        action: Self::Action,
        _rng: &mut ChaCha20Rng,
    ) -> Result<Timestep<Self::Observation>, EnvironmentError> {
        if !(-1.0..=1.0).contains(&action[0]) || !(-2.0..=2.0).contains(&action[1]) {
            return Err(EnvironmentError::InvalidAction("out of bounds".into()));
        }
        *state = 1;
        Ok(one_agent_terminal(action[0] + action[1]))
    }

    fn encode_state(state: &Self::State, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        out.push(*state);
        Ok(())
    }

    fn decode_state(buf: &[u8]) -> Result<Self::State, DecodeError> {
        decode_u8(buf)
    }

    fn encode_action(action: &Self::Action, out: &mut Vec<u8>) -> Result<(), EncodeError> {
        for value in action {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Ok(())
    }

    fn decode_action(buf: &[u8]) -> Result<Self::Action, DecodeError> {
        if buf.len() != 8 {
            return Err(DecodeError::InvalidLength {
                expected: 8,
                actual: buf.len(),
            });
        }
        let action = [
            f32::from_le_bytes(buf[0..4].try_into().unwrap()),
            f32::from_le_bytes(buf[4..8].try_into().unwrap()),
        ];
        if action.iter().any(|value| !value.is_finite()) {
            return Err(DecodeError::CorruptedData("non-finite action".into()));
        }
        Ok(action)
    }

    fn encode_observation(
        observation: &Self::Observation,
        out: &mut Vec<u8>,
    ) -> Result<(), EncodeError> {
        encode_f32(observation, out)
    }
}

fn one_agent_running(observation: f32) -> Timestep<f32> {
    Timestep {
        agents: vec![AgentId(0)],
        observations: vec![AgentObservation {
            agent_id: AgentId(0),
            observation,
        }],
        outcomes: vec![AgentOutcome {
            agent_id: AgentId(0),
            reward: 0.0,
            terminated: false,
            truncated: false,
        }],
        decision: Decision::agents([AgentId(0)]),
        episode: EpisodeStatus::Running,
        source: TransitionSource::Reset,
        info: Vec::new(),
    }
}

fn one_agent_terminal(reward: f32) -> Timestep<f32> {
    Timestep {
        agents: vec![AgentId(0)],
        observations: vec![],
        outcomes: vec![AgentOutcome {
            agent_id: AgentId(0),
            reward,
            terminated: true,
            truncated: false,
        }],
        decision: Decision::None,
        episode: EpisodeStatus::Terminated,
        source: TransitionSource::Agents {
            agent_ids: vec![AgentId(0)],
        },
        info: Vec::new(),
    }
}

#[test]
fn simultaneous_private_observations_and_joint_custom_action_run_end_to_end() {
    let mut context = EngineContext::from_environment(SimultaneousEnvironment).unwrap();
    let reset = context.reset(1, &[]).unwrap();
    assert_eq!(
        reset.timestep.decision,
        Decision::agents([AgentId(10), AgentId(20)])
    );
    assert_ne!(
        reset.timestep.observation_for(AgentId(10)),
        reset.timestep.observation_for(AgentId(20))
    );
    let mut joint = 1u32.to_le_bytes().to_vec();
    joint.extend_from_slice(&0u32.to_le_bytes());
    let terminal = context.step(&reset.state, &joint).unwrap();
    assert_eq!(terminal.timestep.episode, EpisodeStatus::Terminated);
    assert_eq!(terminal.timestep.reward_for(AgentId(10)), Some(1.0));
}

#[test]
fn explicit_chance_then_agent_decision_runs_end_to_end() {
    let mut context = EngineContext::from_environment(ExplicitChanceEnvironment).unwrap();
    let reset = context.reset(2, &[]).unwrap();
    assert_eq!(reset.timestep.decision, Decision::Chance);

    let chance = context.step(&reset.state, &[0, 7]).unwrap();
    assert_eq!(chance.timestep.source, TransitionSource::Chance);
    assert_eq!(chance.timestep.decision, Decision::agents([AgentId(5)]));

    let mut agent_action = vec![1];
    agent_action.extend_from_slice(&1u32.to_le_bytes());
    let terminal = context.step(&chance.state, &agent_action).unwrap();
    assert_eq!(terminal.timestep.episode, EpisodeStatus::Terminated);
}

#[test]
fn dynamic_roster_spawns_and_retires_agents_with_a_final_transition() {
    let mut context = EngineContext::from_environment(DynamicRosterEnvironment).unwrap();
    let reset = context.reset(3, &[]).unwrap();
    assert_eq!(reset.timestep.agents, vec![AgentId(1)]);

    let spawned = context.step(&reset.state, &0u32.to_le_bytes()).unwrap();
    assert_eq!(spawned.timestep.agents, vec![AgentId(1), AgentId(2)]);
    assert!(spawned.timestep.outcomes[0].terminated);
    assert_eq!(
        spawned.timestep.source,
        TransitionSource::Agents {
            agent_ids: vec![AgentId(1)]
        }
    );
    assert_eq!(spawned.timestep.decision, Decision::agents([AgentId(2)]));

    let removed = context.step(&spawned.state, &0u32.to_le_bytes()).unwrap();
    assert_eq!(removed.timestep.agents, vec![AgentId(2)]);
    assert_eq!(removed.timestep.episode, EpisodeStatus::Terminated);
}

#[test]
fn standard_multi_discrete_codec_runs_end_to_end() {
    let mut context = EngineContext::from_environment(MultiDiscreteEnvironment).unwrap();
    let reset = context.reset(4, &[]).unwrap();
    let mut action = 1u32.to_le_bytes().to_vec();
    action.extend_from_slice(&2u32.to_le_bytes());
    let terminal = context.step(&reset.state, &action).unwrap();
    assert_eq!(terminal.timestep.reward_for(AgentId(0)), Some(3.0));
}

#[test]
fn standard_continuous_codec_runs_end_to_end() {
    let mut context = EngineContext::from_environment(ContinuousEnvironment).unwrap();
    let reset = context.reset(5, &[]).unwrap();
    let mut action = 0.5f32.to_le_bytes().to_vec();
    action.extend_from_slice(&(-1.25f32).to_le_bytes());
    let terminal = context.step(&reset.state, &action).unwrap();
    assert_eq!(terminal.timestep.reward_for(AgentId(0)), Some(-0.75));
}
