use super::*;

#[test]
fn action_spaces_have_a_stable_tagged_wire_shape() {
    assert_eq!(
        serde_json::to_value(ActionSpace::discrete(9)).unwrap(),
        serde_json::json!({"kind": "discrete", "size": 9})
    );
    assert_eq!(
        serde_json::to_value(ActionSpace::MultiDiscrete {
            dimensions: vec![2, 3]
        })
        .unwrap(),
        serde_json::json!({"kind": "multi_discrete", "dimensions": [2, 3]})
    );
}

#[test]
fn timestep_represents_simultaneous_agents_and_individual_outcomes() {
    let timestep = Timestep {
        agents: vec![AgentId(10), AgentId(20)],
        observations: vec![
            AgentObservation {
                agent_id: AgentId(10),
                observation: "private-a",
            },
            AgentObservation {
                agent_id: AgentId(20),
                observation: "private-b",
            },
        ],
        outcomes: vec![
            AgentOutcome {
                agent_id: AgentId(10),
                reward: 1.5,
                terminated: false,
                truncated: false,
            },
            AgentOutcome {
                agent_id: AgentId(20),
                reward: -0.25,
                terminated: false,
                truncated: false,
            },
        ],
        decision: Decision::Agents {
            agent_ids: vec![AgentId(10), AgentId(20)],
        },
        episode: EpisodeStatus::Running,
        source: TransitionSource::Chance,
        info: vec![1, 2, 3],
    };

    assert_eq!(timestep.observation_for(AgentId(20)), Some(&"private-b"));
    assert_eq!(timestep.reward_for(AgentId(10)), Some(1.5));
    assert!(matches!(
        timestep.sole_observation(),
        Err(TimestepAccessError::ExpectedOneObservation { actual: 2 })
    ));
}

#[test]
fn continuing_and_chance_environments_are_expressible() {
    let capabilities = Capabilities {
        id: EngineId {
            env_id: "market".into(),
            build_id: "test".into(),
        },
        contract_version: 1,
        encoding: Encoding::custom("market-state:v1", "joint-orders:v1", "book:v1"),
        semantics: EnvironmentSemantics {
            turn_model: TurnModel::Simultaneous,
            information_model: InformationModel::PartiallyObserved,
            planning_state_model: PlanningStateModel::CompleteSnapshot,
            transition_dynamics: TransitionDynamics::Stochastic,
            chance_model: ChanceModel::Explicit,
            reward_model: RewardModel::General,
        },
        max_horizon: None,
        agents: AgentModel::Dynamic {
            action_space: ActionSpace::Continuous {
                low: vec![-1.0],
                high: vec![1.0],
                shape: vec![1],
            },
        },
        preferred_batch: 32,
    };

    assert_eq!(capabilities.max_horizon, None);
    assert_eq!(capabilities.semantics.chance_model, ChanceModel::Explicit);
    assert!(matches!(capabilities.agents, AgentModel::Dynamic { .. }));
}
