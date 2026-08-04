use super::*;
use engine_core::{AgentId, Decision, EngineContext, EpisodeStatus};
use model_watcher::ModelInfo;
use std::time::Duration;
use tracing::debug;

fn loaded_model(checkpoint_id: &str) -> ModelInfo {
    ModelInfo {
        loaded: true,
        checkpoint_id: Some(checkpoint_id.to_string()),
        model_sha256: Some("c".repeat(64)),
        path: Some("model.onnx".into()),
        loaded_at: Some(1),
        training_step: Some(1),
    }
}

#[test]
fn root_collection_requires_an_absent_run_head() {
    assert!(require_source_checkpoint(None, false, &ModelInfo::default()).is_ok());
    let error = require_source_checkpoint(None, true, &loaded_model(&"a".repeat(64)))
        .unwrap_err()
        .to_string();
    assert!(error.contains("root collection requires an absent RunHead"));
}

#[test]
fn descendant_collection_requires_the_exact_loaded_source() {
    let expected = "a".repeat(64);
    assert!(require_source_checkpoint(Some(&expected), true, &loaded_model(&expected)).is_ok());
    let absent = require_source_checkpoint(Some(&expected), false, &ModelInfo::default())
        .unwrap_err()
        .to_string();
    assert!(absent.contains("no RunHead model was loaded"));
    let mismatch = require_source_checkpoint(Some(&expected), true, &loaded_model(&"b".repeat(64)))
        .unwrap_err()
        .to_string();
    assert!(mismatch.contains("does not match required source checkpoint"));
}

#[test]
fn temperature_threshold_must_be_reachable_within_environment_horizon() {
    assert!(require_reachable_temperature_threshold(0, 9).is_ok());
    assert!(require_reachable_temperature_threshold(8, 9).is_ok());
    for threshold in [9, 10] {
        let error = require_reachable_temperature_threshold(threshold, 9)
            .unwrap_err()
            .to_string();
        assert!(error.contains("late_temperature is unreachable"), "{error}");
    }
}

#[test]
fn episode_timeout_is_exact_for_every_horizon() {
    assert_eq!(
        EpisodeContext::new("a", 0, 180, 402).timeout,
        Duration::from_secs(180)
    );
    assert_eq!(
        EpisodeContext::new("a", 0, 180, 42).timeout,
        Duration::from_secs(180)
    );
}

#[test]
fn episode_limits_report_the_exact_reason() {
    let context = EpisodeContext::new("a", 0, 300, 42);
    assert_eq!(context.limit_exceeded(0), None);
    assert_eq!(context.limit_exceeded(context.max_steps - 1), None);
    assert_eq!(
        context.limit_exceeded(context.max_steps),
        Some(AbandonReason::MaxSteps)
    );
    let timed_out = EpisodeContext::new("a", 0, 0, 1);
    assert_eq!(timed_out.timeout, Duration::ZERO);
    assert_eq!(timed_out.limit_exceeded(0), Some(AbandonReason::Timeout));
}

#[test]
fn episode_ids_bind_scope_and_process_token() {
    let scope = "f".repeat(64);
    let prefix_one = episode_id_prefix("actor-1", &scope, 0x0123456789abcdef);
    assert_eq!(prefix_one, "actor-1-ffffffff-0123456789abcdef");
    let prefix_two = episode_id_prefix("actor-1", &scope, 0xfedcba9876543210);
    let id_one = EpisodeContext::new(&prefix_one, 3, 30, 9).id;
    let id_two = EpisodeContext::new(&prefix_two, 3, 30, 9).id;
    assert_ne!(id_one, id_two);
    assert!(id_one.ends_with("-ep-3"));
    assert_eq!(episode_id_prefix("a", "abc", 1), "a-abc-0000000000000001");
}

#[test]
fn abandon_reason_strings_are_stable() {
    assert_eq!(AbandonReason::Timeout.as_str(), "timeout");
    assert_eq!(AbandonReason::MaxSteps.as_str(), "max_steps");
    assert_eq!(
        AbandonReason::EnvironmentTruncated.as_str(),
        "environment_truncated"
    );
}

#[test]
fn alphazero_reset_requires_one_matching_decision_and_observation() {
    engine_games::register_all_environments();
    let mut engine = EngineContext::new("tictactoe").unwrap();
    let reset = engine.reset(42, &[]).unwrap();
    let (agent, observation) = require_reset_timestep(&reset.timestep).unwrap();
    assert_eq!(agent, AgentId(1));
    assert_eq!(observation.len(), 18 * std::mem::size_of::<f32>());

    let mut multiple_decisions = reset.timestep.clone();
    multiple_decisions.decision = Decision::agents([AgentId(1), AgentId(2)]);
    assert!(require_reset_timestep(&multiple_decisions)
        .unwrap_err()
        .to_string()
        .contains("exactly one acting agent"));

    let mut mismatched_observation = reset.timestep;
    mismatched_observation.observations[0].agent_id = AgentId(2);
    assert!(require_reset_timestep(&mismatched_observation)
        .unwrap_err()
        .to_string()
        .contains("observation belongs to agent 2"));
}

#[test]
fn alphazero_step_maps_reward_from_transition_source_agent() {
    engine_games::register_all_environments();
    let mut engine = EngineContext::new("tictactoe").unwrap();
    let reset = engine.reset(42, &[]).unwrap();
    let mut state = reset.state;
    let mut timestep = reset.timestep;
    for (index, action) in [0u32, 3, 1, 4, 2].into_iter().enumerate() {
        let (actor, _) = require_active_position(&timestep).unwrap();
        let step = engine.step(&state, &action.to_le_bytes()).unwrap();
        let actor_reward = require_step_timestep(&step.timestep, actor).unwrap();
        if index == 4 {
            assert_eq!(step.timestep.episode, EpisodeStatus::Terminated);
            assert_eq!(actor, AgentId(1));
            assert_eq!(actor_reward, 1.0);
            assert_eq!(step.timestep.reward_for(AgentId(1)), Some(1.0));
            assert_eq!(step.timestep.reward_for(AgentId(2)), Some(-1.0));
        } else {
            assert_eq!(step.timestep.episode, EpisodeStatus::Running);
            assert_eq!(actor_reward, 0.0);
        }
        state = step.state;
        timestep = step.timestep;
    }
}

#[test]
fn alphazero_step_rejects_wrong_transition_source_actor() {
    engine_games::register_all_environments();
    let mut engine = EngineContext::new("tictactoe").unwrap();
    let reset = engine.reset(42, &[]).unwrap();
    let step = engine.step(&reset.state, &0u32.to_le_bytes()).unwrap();
    let error = require_step_timestep(&step.timestep, AgentId(2))
        .unwrap_err()
        .to_string();
    assert!(error.contains("reported acting agent 1, expected 2"));
}

fn test_config() -> Config {
    Config {
        actor_id: "test-actor".into(),
        env_id: "tictactoe".into(),
        algorithm_id: algorithm_core::ALPHAZERO_BOARD_V1_ID.into(),
        max_episodes: 1,
        collection_scope_id: "a".repeat(64),
        source_checkpoint_id: None,
        collector_config: serde_json::to_string(&alpha_config()).unwrap(),
        episode_timeout_secs: 30,
        log_level: "info".into(),
        log_interval: 10,
        data_dir: "./data".into(),
        postgres_url: std::env::var("CARTRIDGE_STORAGE_POSTGRES_URL")
            .unwrap_or_else(|_| "postgresql://cartridge:cartridge@localhost:5432/cartridge".into()),
    }
}

fn alpha_config() -> AlphaZeroCollectorConfig {
    AlphaZeroCollectorConfig {
        schema_version: crate::algorithms::COLLECTOR_CONFIG_SCHEMA_VERSION,
        num_simulations: 50,
        c_puct: 1.4,
        temperature: 1.0,
        late_temperature: 1.0,
        temp_threshold: 0,
        dirichlet_alpha: 0.3,
        dirichlet_weight: 0.25,
        eval_batch_size: 32,
        onnx_intra_threads: 1,
    }
}

#[tokio::test]
#[ignore]
async fn actor_creation() {
    assert!(AlphaZeroCollector::new(test_config(), alpha_config())
        .await
        .is_ok());
}

#[tokio::test]
#[ignore]
async fn actor_runs_single_episode() {
    let actor = AlphaZeroCollector::new(test_config(), alpha_config())
        .await
        .unwrap();
    match actor.run_episode().await.unwrap() {
        EpisodeOutcome::Completed {
            steps,
            player_one_outcome,
            ..
        } => {
            assert!(steps > 0);
            debug!(steps, player_one_outcome, "Episode completed");
        }
        EpisodeOutcome::Abandoned { reason, steps, .. } => {
            panic!("TicTacToe episode abandoned ({reason:?}) after {steps} steps");
        }
    }
}

#[tokio::test]
#[ignore]
async fn nonexistent_game_is_rejected() {
    let mut config = test_config();
    config.env_id = "nonexistent_game".into();
    let error = AlphaZeroCollector::new(config, alpha_config())
        .await
        .err()
        .expect("unknown environment must fail")
        .to_string();
    assert!(error.contains("not registered"));
}

#[tokio::test]
#[ignore]
async fn actor_stores_replay_records() {
    let actor = AlphaZeroCollector::new(test_config(), alpha_config())
        .await
        .unwrap();
    actor.run_episode().await.unwrap();
    assert!(actor.replay.count().await.unwrap() > 0);
}
