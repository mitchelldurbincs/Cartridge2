//! Bounded single-agent DQN experience collector.

use algorithm_core::{resolve_algorithm, RuntimeProfile};
use anyhow::{anyhow, bail, Result};
use dqn_runtime::{availability_bits, available_actions, DqnQPolicy};
use engine_core::{
    ActionSpace, AgentId, Decision, EngineContext, EpisodeStatus, ObservationEncoding,
    TransitionSource,
};
use model_watcher::{resolve_current_filesystem_model, ModelSelection};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use tracing::{debug, info};

use crate::algorithms::{encode_dqn_transition, DqnCollectorConfig, DqnTransition};
use crate::config::Config;
use crate::stats::ActorStats;
use crate::storage::{
    create_replay_store, ReplayProfile, ReplaySelection, ReplayStore, StorageConfig,
};

pub struct DqnCollector {
    config: Config,
    cartridge_config: DqnCollectorConfig,
    replay_selection: ReplaySelection,
    agent_id: AgentId,
    obs_size: usize,
    num_actions: usize,
    max_horizon: u32,
    q_policy: Option<DqnQPolicy>,
    engine: Mutex<EngineContext>,
    replay: Arc<dyn ReplayStore>,
    shutdown_signal: AtomicBool,
    stats: ActorStats,
    episode_prefix: String,
}

impl DqnCollector {
    pub async fn new(config: Config, cartridge_config: DqnCollectorConfig) -> Result<Self> {
        config.validate()?;
        cartridge_config.validate()?;
        engine_games::register_all_environments();
        let engine = EngineContext::new(&config.env_id)
            .map_err(|error| anyhow!("Environment '{}' is unavailable: {error}", config.env_id))?;
        let algorithm = resolve_algorithm(&config.algorithm_id)?;
        let compatibility = algorithm.compatibility(&engine);
        compatibility.require_compatible()?;
        let descriptor = algorithm.descriptor();
        let capabilities = engine.capabilities();
        let max_horizon = capabilities
            .max_horizon
            .filter(|value| *value > 0)
            .ok_or_else(|| anyhow!("DQN requires a finite non-zero max_horizon"))?;
        let agents = capabilities
            .agents
            .fixed_agents()
            .ok_or_else(|| anyhow!("DQN requires one fixed agent"))?;
        let [agent] = agents else {
            bail!("DQN requires one fixed agent, got {}", agents.len());
        };
        let num_actions = match &agent.action_space {
            ActionSpace::Discrete { size } => usize::try_from(*size)?,
            other => bail!("DQN requires discrete actions, got {other:?}"),
        };
        let obs_size = match &capabilities.encoding.observation {
            ObservationEncoding::Tensor { spec } => spec
                .fixed_elements()
                .ok_or_else(|| anyhow!("DQN requires a fixed observation tensor"))?,
            other => bail!("DQN requires tensor observations, got {other:?}"),
        };

        let runtime_profile = RuntimeProfile::new(
            descriptor.id,
            config.env_id.clone(),
            capabilities.contract_version,
        )?;
        let model_dir = runtime_profile.model_dir(&config.data_dir);
        std::fs::create_dir_all(&model_dir)?;
        let model_contract = descriptor
            .model_artifact_contract(config.env_id.clone(), capabilities.contract_version);
        let resolved = resolve_current_filesystem_model(
            &model_dir,
            &model_contract,
            max_horizon,
            ModelSelection::Latest,
        )?;
        let q_policy = match (config.source_checkpoint_id.as_deref(), resolved) {
            (None, None) => {
                if cartridge_config.epsilon != 1.0 {
                    bail!("root DQN collection requires epsilon=1 without a Q-model");
                }
                None
            }
            (None, Some(model)) => bail!(
                "root DQN collection requires an absent RunHead, found checkpoint '{}'",
                model.checkpoint_id
            ),
            (Some(expected), None) => bail!(
                "DQN collection requires source checkpoint '{expected}', but RunHead is absent"
            ),
            (Some(expected), Some(model)) if model.checkpoint_id != expected => bail!(
                "DQN RunHead checkpoint '{}' does not match required source '{expected}'",
                model.checkpoint_id
            ),
            (Some(_), Some(model)) => Some(DqnQPolicy::load(
                &model.path,
                obs_size,
                num_actions,
                usize::try_from(cartridge_config.onnx_intra_threads)?,
                &model_contract,
            )?),
        };
        let replay_selection = ReplaySelection {
            profile: ReplayProfile {
                env_id: config.env_id.clone(),
                env_contract_version: capabilities.contract_version,
                algorithm_id: descriptor.id.to_string(),
                experience_schema: descriptor.components.experience_schema.to_string(),
            },
            collection_scope_id: config.collection_scope_id.clone(),
            source_checkpoint_id: config.source_checkpoint_id.clone(),
        };
        let storage_config = StorageConfig {
            postgres_url: config.postgres_url.clone(),
            pool_config: config.pool_config(),
            selection: replay_selection.clone(),
        };
        let replay = create_replay_store(&storage_config).await?;
        info!(
            algorithm = descriptor.id,
            env_id = %config.env_id,
            experience_schema = descriptor.components.experience_schema,
            obs_size,
            num_actions,
            "DQN collector initialized"
        );
        Ok(Self {
            stats: ActorStats::new(&config.env_id),
            episode_prefix: crate::actor::episode_id_prefix(
                &config.actor_id,
                &config.collection_scope_id,
                rand::random::<u64>(),
            ),
            config,
            cartridge_config,
            replay_selection,
            agent_id: agent.id,
            obs_size,
            num_actions,
            max_horizon,
            q_policy,
            engine: Mutex::new(engine),
            replay: Arc::from(replay),
            shutdown_signal: AtomicBool::new(false),
        })
    }

    pub fn shutdown(&self) {
        self.shutdown_signal.store(true, Ordering::Relaxed);
    }

    pub async fn run(&self) -> Result<()> {
        for episode in 0..self.config.max_episodes {
            if self.shutdown_signal.load(Ordering::Relaxed) {
                bail!("shutdown interrupted DQN collection after {episode} episodes");
            }
            let (steps, return_value) = self.run_episode(episode).await?;
            self.stats.record_episode(steps, return_value);
        }
        let stats = self.stats.snapshot();
        info!(
            episodes = stats.episodes_completed,
            transitions = stats.total_steps,
            "DQN collection complete"
        );
        Ok(())
    }

    async fn run_episode(&self, episode: u32) -> Result<(u32, f32)> {
        let seed = self.cartridge_config.seed.wrapping_add(u64::from(episode));
        let reset = self
            .engine
            .lock()
            .map_err(|error| anyhow!("failed to lock DQN environment: {error}"))?
            .reset(seed, &[])?;
        if reset.timestep.source != TransitionSource::Reset {
            bail!("DQN environment reset returned a non-reset transition");
        }
        // The prefix carries a random per-process token (see
        // `actor::episode_id_prefix`), so a restarted collector reusing this
        // scope can never collide with episode IDs it wrote before dying.
        let episode_id = format!("{}-ep-{episode}", self.episode_prefix);
        let mut state = reset.state;
        let mut timestep = reset.timestep;
        let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xd151_5eed);
        let mut records = Vec::with_capacity(self.max_horizon as usize);
        let mut episode_return = 0.0f32;
        let started = Instant::now();

        for step in 0..self.max_horizon {
            if started.elapsed() > Duration::from_secs(self.config.episode_timeout_secs) {
                bail!("DQN episode {episode_id} exceeded its timeout");
            }
            let decision = timestep
                .decision
                .sole_agent()
                .ok_or_else(|| anyhow!("DQN running timestep requires one agent decision"))?;
            if decision.agent_id != self.agent_id {
                bail!("DQN decision agent does not match the fixed agent");
            }
            let observation = timestep.sole_observation()?;
            if observation.agent_id != self.agent_id {
                bail!("DQN observation agent does not match the fixed agent");
            }
            let actions = available_actions(&decision.availability, self.num_actions)?;
            let action = match &self.q_policy {
                None => actions[rng.gen_range(0..actions.len())],
                Some(_) if rng.gen::<f32>() < self.cartridge_config.epsilon => {
                    actions[rng.gen_range(0..actions.len())]
                }
                Some(policy) => policy.select_greedy(&observation.data, &decision.availability)?,
            };
            let result = self
                .engine
                .lock()
                .map_err(|error| anyhow!("failed to lock DQN environment: {error}"))?
                .step(&state, &action.to_le_bytes())?;
            let reward = result
                .timestep
                .reward_for(self.agent_id)
                .ok_or_else(|| anyhow!("DQN step has no outcome for its agent"))?;
            let next_observation = result.timestep.sole_observation()?;
            let next_availability = match result.timestep.episode {
                EpisodeStatus::Running => {
                    let next =
                        result.timestep.decision.sole_agent().ok_or_else(|| {
                            anyhow!("DQN running step requires one next decision")
                        })?;
                    availability_bits(&next.availability, self.num_actions)?
                }
                EpisodeStatus::Terminated | EpisodeStatus::Truncated => {
                    if result.timestep.decision != Decision::None {
                        bail!("completed DQN timestep must not request another action");
                    }
                    vec![false; self.num_actions]
                }
            };
            let payload = encode_dqn_transition(
                DqnTransition {
                    observation: &observation.data,
                    action,
                    reward,
                    next_observation: &next_observation.data,
                    terminated: result.timestep.episode == EpisodeStatus::Terminated,
                    truncated: result.timestep.episode == EpisodeStatus::Truncated,
                    next_availability: &next_availability,
                },
                self.obs_size,
                self.num_actions,
            )?;
            records.push(self.replay_selection.record(
                format!("{episode_id}-step-{step}"),
                &episode_id,
                step,
                payload,
            ));
            episode_return += reward;
            debug!(episode = %episode_id, step, action, reward, "DQN transition collected");
            if result.timestep.episode.is_done() {
                self.replay.store_batch(&records).await?;
                return Ok((step + 1, episode_return));
            }
            state = result.state;
            timestep = result.timestep;
        }
        bail!("DQN environment exceeded its declared max_horizon")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::mock::MockReplayStore;

    #[tokio::test]
    async fn counter_episode_writes_complete_dqn_transition_records() {
        engine_games::register_all_environments();
        let replay_selection = ReplaySelection {
            profile: ReplayProfile {
                env_id: "counter".into(),
                env_contract_version: 2,
                algorithm_id: algorithm_core::DQN_V1_ID.into(),
                experience_schema: "dqn_transition_v1".into(),
            },
            collection_scope_id: "a".repeat(64),
            source_checkpoint_id: None,
        };
        let replay = Arc::new(MockReplayStore::new(replay_selection.clone()));
        let collector = DqnCollector {
            config: Config {
                actor_id: "dqn-test".into(),
                env_id: "counter".into(),
                algorithm_id: algorithm_core::DQN_V1_ID.into(),
                max_episodes: 1,
                collection_scope_id: "a".repeat(64),
                source_checkpoint_id: None,
                collector_config: "{}".into(),
                episode_timeout_secs: 5,
                log_level: "info".into(),
                log_interval: 1,
                data_dir: "./data".into(),
                postgres_url: "unused".into(),
            },
            cartridge_config: DqnCollectorConfig {
                schema_version: 1,
                epsilon: 1.0,
                seed: 9,
                onnx_intra_threads: 1,
            },
            replay_selection,
            agent_id: AgentId(0),
            obs_size: 2,
            num_actions: 2,
            max_horizon: 8,
            q_policy: None,
            engine: Mutex::new(EngineContext::new("counter").unwrap()),
            replay: replay.clone(),
            shutdown_signal: AtomicBool::new(false),
            stats: ActorStats::new("counter"),
            episode_prefix: crate::actor::episode_id_prefix(
                "dqn-test",
                &"a".repeat(64),
                0x0123456789abcdef,
            ),
        };

        let (steps, episode_return) = collector.run_episode(0).await.unwrap();
        let records = replay.get_records();
        assert_eq!(records.len(), steps as usize);
        assert!(episode_return.is_finite());
        assert!(records.iter().all(|record| record.payload.len() == 28));
        let last = &records.last().unwrap().payload;
        assert!(last[24] == 1 || last[25] == 1);
        assert_eq!(&last[26..], &[0, 0]);
    }
}
