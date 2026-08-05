//! AlphaZero self-play collector using engine-core directly.

mod episode;
mod runner;
mod setup;

pub(crate) use episode::episode_id_prefix;

use anyhow::{anyhow, Result};
use engine_core::EngineContext;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex, MutexGuard,
};
use tracing::info;

use crate::algorithms::AlphaZeroCollectorConfig;
use crate::config::Config;
use crate::mcts_policy::MctsPolicy;
use crate::stats::ActorStats;
use crate::storage::{ReplaySelection, ReplayStore};

#[cfg(test)]
use episode::{
    require_active_position, require_reset_timestep, require_step_timestep, AbandonReason,
    EpisodeContext, EpisodeOutcome,
};
#[cfg(test)]
use setup::{require_reachable_temperature_threshold, require_source_checkpoint};

pub struct AlphaZeroCollector {
    config: Config,
    replay_selection: ReplaySelection,
    obs_size: usize,
    num_actions: usize,
    engine: Mutex<EngineContext>,
    mcts_policy: Mutex<MctsPolicy>,
    replay: Arc<dyn ReplayStore>,
    episode_count: AtomicU32,
    shutdown_signal: AtomicBool,
    stats: ActorStats,
    episode_prefix: String,
}

impl AlphaZeroCollector {
    pub async fn new(config: Config, cartridge_config: AlphaZeroCollectorConfig) -> Result<Self> {
        let dependencies = setup::build(&config, &cartridge_config).await?;
        Ok(Self {
            stats: ActorStats::new(&config.env_id),
            episode_prefix: episode::episode_id_prefix(
                &config.actor_id,
                &config.collection_scope_id,
                rand::random::<u64>(),
            ),
            config,
            replay_selection: dependencies.replay_selection,
            obs_size: dependencies.obs_size,
            num_actions: dependencies.num_actions,
            engine: Mutex::new(dependencies.engine),
            mcts_policy: Mutex::new(dependencies.mcts_policy),
            replay: dependencies.replay,
            episode_count: AtomicU32::new(0),
            shutdown_signal: AtomicBool::new(false),
        })
    }

    pub fn shutdown(&self) {
        self.shutdown_signal.store(true, Ordering::Relaxed);
        info!("Shutdown signal set");
    }

    pub async fn run(&self) -> Result<()> {
        runner::run(self).await
    }

    fn lock_engine(&self) -> Result<MutexGuard<'_, EngineContext>> {
        self.engine
            .lock()
            .map_err(|error| anyhow!("Engine lock poisoned: {error}"))
    }

    fn lock_mcts_policy(&self) -> Result<MutexGuard<'_, MctsPolicy>> {
        self.mcts_policy
            .lock()
            .map_err(|error| anyhow!("MCTS policy lock poisoned: {error}"))
    }
}

#[cfg(test)]
mod tests;
