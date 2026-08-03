//! AlphaZero self-play collector using engine-core directly.

use algorithm_core::{resolve_algorithm, RuntimeProfile};
use anyhow::{anyhow, Result};
use engine_core::board_profile::BoardGameMetadata;
use engine_core::{
    AgentId, Capabilities, Decision, EngineContext, EpisodeStatus, ErasedTimestep, TransitionSource,
};
use indicatif::{ProgressBar, ProgressStyle};
use mcts::{MctsConfig, SearchStats};
#[cfg(feature = "s3")]
use model_watcher::s3::{S3Config, S3ModelWatcher};
use model_watcher::{ModelInfo, ModelLoadSpec, ModelSelection, ModelWatcher};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex, MutexGuard,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};

use crate::algorithms::encode_experience;
use crate::config::Config;
use crate::mcts_policy::MctsPolicy;
use crate::resources::rss_mb;
use crate::stats::ActorStats;
use crate::storage::{
    create_replay_store, ReplayProfile, ReplaySelection, ReplayStore, StorageConfig,
};

/// Context for a single episode, containing metadata and timing information.
struct EpisodeContext {
    id: String,
    start_time: Instant,
    timeout: Duration,
    max_steps: u32,
}

/// Aggregated MCTS stats for an episode.
#[derive(Debug, Default)]
pub(crate) struct EpisodeStats {
    /// Number of MCTS searches performed
    pub search_count: u32,
    /// Total wall-clock time across all searches (microseconds)
    pub total_time_us: u64,
    /// Total time spent in tree selection (microseconds)
    pub selection_time_us: u64,
    /// Total time spent in neural network inference (microseconds)
    pub inference_time_us: u64,
    /// Total time spent expanding nodes (microseconds)
    pub expansion_time_us: u64,
    /// Total time spent in backpropagation (microseconds)
    pub backprop_time_us: u64,
    /// Total number of NN batch calls
    pub num_batches: u32,
    /// Total number of NN evaluations
    pub total_evals: u32,
    /// Total game step() calls during expansion
    pub game_steps: u32,
    /// Total terminal nodes hit
    pub terminal_hits: u32,
}

impl EpisodeStats {
    /// Add stats from a single MCTS search.
    fn add(&mut self, stats: &SearchStats) {
        self.search_count += 1;
        self.total_time_us += stats.total_time_us;
        self.selection_time_us += stats.selection_time_us;
        self.inference_time_us += stats.inference_time_us;
        self.expansion_time_us += stats.expansion_time_us;
        self.backprop_time_us += stats.backprop_time_us;
        self.num_batches += stats.num_batches;
        self.total_evals += stats.total_evals;
        self.game_steps += stats.game_steps;
        self.terminal_hits += stats.terminal_hits;
    }

    /// Log a summary of the episode stats.
    fn log_summary(&self, episode_num: u32) {
        if self.search_count == 0 || self.total_time_us == 0 {
            return;
        }

        let total_ms = self.total_time_us as f64 / 1000.0;
        let inference_pct = (self.inference_time_us as f64 / self.total_time_us as f64) * 100.0;
        let expansion_pct = (self.expansion_time_us as f64 / self.total_time_us as f64) * 100.0;
        let selection_pct = (self.selection_time_us as f64 / self.total_time_us as f64) * 100.0;
        let backprop_pct = (self.backprop_time_us as f64 / self.total_time_us as f64) * 100.0;
        let avg_batch_size = if self.num_batches > 0 {
            self.total_evals as f64 / self.num_batches as f64
        } else {
            0.0
        };

        info!(
            episode = episode_num,
            searches = self.search_count,
            total_ms = format!("{:.1}", total_ms),
            inference_pct = format!("{:.1}%", inference_pct),
            expansion_pct = format!("{:.1}%", expansion_pct),
            selection_pct = format!("{:.1}%", selection_pct),
            backprop_pct = format!("{:.1}%", backprop_pct),
            nn_batches = self.num_batches,
            avg_batch_size = format!("{:.1}", avg_batch_size),
            game_steps = self.game_steps,
            terminal_hits = self.terminal_hits,
            "MCTS episode stats"
        );
    }
}

impl EpisodeContext {
    /// Create a new episode context with generated ID and timing.
    fn new(
        actor_id: &str,
        episode_count: u32,
        timeout_secs: u64,
        max_horizon: u32,
    ) -> Result<Self> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let id = format!("{}-ep-{}-{}", actor_id, episode_count, now.as_secs());
        // This value is part of the authenticated RunRecipe and is therefore
        // the exact terminal bound, never an input to a hidden derived floor.
        let timeout = Duration::from_secs(timeout_secs);
        // Use 10x max_horizon as generous upper bound to protect against infinite loops
        let max_steps = max_horizon.saturating_mul(10).max(1000);

        Ok(Self {
            id,
            start_time: Instant::now(),
            timeout,
            max_steps,
        })
    }

    /// Check if the episode has exceeded its timeout.
    fn is_timed_out(&self) -> bool {
        self.start_time.elapsed() > self.timeout
    }

    /// Whether this episode must be abandoned, and why.
    fn limit_exceeded(&self, steps_taken: u32) -> Option<AbandonReason> {
        if self.is_timed_out() {
            Some(AbandonReason::Timeout)
        } else if steps_taken >= self.max_steps {
            Some(AbandonReason::MaxSteps)
        } else {
            None
        }
    }
}

/// Why an episode ended without reaching a terminal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AbandonReason {
    /// The wall-clock budget ran out.
    Timeout,
    /// The step guard tripped before the environment terminated.
    MaxSteps,
    /// The environment ended without a terminal outcome, so AlphaZero value
    /// targets cannot be constructed.
    EnvironmentTruncated,
}

impl AbandonReason {
    fn as_str(self) -> &'static str {
        match self {
            AbandonReason::Timeout => "timeout",
            AbandonReason::MaxSteps => "max_steps",
            AbandonReason::EnvironmentTruncated => "environment_truncated",
        }
    }

    fn guidance(self) -> &'static str {
        match self {
            AbandonReason::Timeout => {
                "increase actor.episode_timeout_secs if timeouts persist"
            }
            AbandonReason::MaxSteps => {
                "fix the environment termination contract or increase its declared horizon"
            }
            AbandonReason::EnvironmentTruncated => {
                "AlphaZero requires terminal outcomes; use a cartridge that supports truncation or change the environment contract"
            }
        }
    }
}

/// Result of one self-play episode attempt.
#[derive(Debug)]
pub(crate) enum EpisodeOutcome {
    /// Reached a terminal state; its training experiences were stored.
    Completed {
        steps: u32,
        player_one_outcome: f32,
        stats: EpisodeStats,
    },
    /// Ended early, so its pending experiences were **discarded**.
    ///
    /// Without a terminal state there is no game outcome to backfill, and
    /// value targets are the game outcome. The AlphaZero v1 payload requires a
    /// terminal target, so no valid record exists for this episode. The caller
    /// must account for the loss so it can never pass unnoticed.
    Abandoned {
        reason: AbandonReason,
        steps: u32,
        discarded: usize,
        /// The effective budget that was exceeded — the configured timeout
        /// after the horizon floor is applied, not the raw config value.
        timeout_secs: u64,
    },
}

pub struct AlphaZeroCollector {
    config: Config,
    replay_selection: ReplaySelection,
    board_metadata: BoardGameMetadata,
    engine: Mutex<EngineContext>,
    mcts_policy: Mutex<MctsPolicy>,
    replay: Arc<dyn ReplayStore>,
    episode_count: AtomicU32,
    shutdown_signal: AtomicBool,
    stats: ActorStats,
}

/// One transition retained in memory until a terminal outcome is available.
///
/// The acting agent is part of the AlphaZero cartridge's target semantics,
/// not the storage envelope, so it is kept alongside the row only while the
/// episode is being assembled.
struct PendingExperience {
    actor: AgentId,
    step_number: u32,
    observation: Vec<u8>,
    policy_target: Vec<f32>,
}

fn require_max_horizon(capabilities: &Capabilities) -> Result<u32> {
    capabilities
        .max_horizon
        .filter(|horizon| *horizon > 0)
        .ok_or_else(|| {
            anyhow!(
                "AlphaZero environment '{}' must declare a finite non-zero max_horizon",
                capabilities.id.env_id
            )
        })
}

fn require_reachable_temperature_threshold(temp_threshold: u32, max_horizon: u32) -> Result<()> {
    if temp_threshold > 0 && temp_threshold >= max_horizon {
        return Err(anyhow!(
            "temp_threshold {temp_threshold} must be zero or less than environment max_horizon {max_horizon}; otherwise late_temperature is unreachable"
        ));
    }
    Ok(())
}

fn require_two_player_outcomes(timestep: &ErasedTimestep) -> Result<()> {
    if timestep.outcomes.len() != 2 {
        return Err(anyhow!(
            "AlphaZero timestep must contain exactly two agent outcomes, got {}",
            timestep.outcomes.len()
        ));
    }

    for agent_id in [AgentId(1), AgentId(2)] {
        let matching = timestep
            .outcomes
            .iter()
            .filter(|outcome| outcome.agent_id == agent_id)
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(anyhow!(
                "AlphaZero timestep must contain exactly one outcome for agent {}, got {}",
                agent_id.0,
                matching.len()
            ));
        }
        let outcome = matching[0];
        if !outcome.reward.is_finite() {
            return Err(anyhow!(
                "AlphaZero timestep contains non-finite reward for agent {}",
                agent_id.0
            ));
        }
        let expected_flags = match timestep.episode {
            EpisodeStatus::Running => !outcome.terminated && !outcome.truncated,
            EpisodeStatus::Terminated => outcome.terminated && !outcome.truncated,
            EpisodeStatus::Truncated => !outcome.terminated && outcome.truncated,
        };
        if !expected_flags {
            return Err(anyhow!(
                "outcome flags for agent {} disagree with episode status {:?}",
                agent_id.0,
                timestep.episode
            ));
        }
    }

    let player_one = timestep
        .reward_for(AgentId(1))
        .expect("validated outcome for agent 1");
    let player_two = timestep
        .reward_for(AgentId(2))
        .expect("validated outcome for agent 2");
    match timestep.episode {
        EpisodeStatus::Running | EpisodeStatus::Truncated
            if player_one != 0.0 || player_two != 0.0 =>
        {
            Err(anyhow!(
                "AlphaZero terminal-only reward contract emitted rewards ({player_one}, {player_two}) for {:?} episode",
                timestep.episode
            ))
        }
        EpisodeStatus::Terminated if (player_one + player_two).abs() > 1e-6 => Err(anyhow!(
            "AlphaZero terminal rewards must be zero-sum, got ({player_one}, {player_two})"
        )),
        _ => Ok(()),
    }
}

fn require_active_position(timestep: &ErasedTimestep) -> Result<(AgentId, &[u8])> {
    if timestep.episode != EpisodeStatus::Running {
        return Err(anyhow!(
            "AlphaZero action selection requires a running episode, got {:?}",
            timestep.episode
        ));
    }
    let active_agent = match &timestep.decision {
        Decision::Agents { agent_ids } if agent_ids.len() == 1 => agent_ids[0],
        decision => {
            return Err(anyhow!(
                "AlphaZero requires exactly one acting agent, got {decision:?}"
            ))
        }
    };
    if !matches!(active_agent, AgentId(1) | AgentId(2)) {
        return Err(anyhow!(
            "AlphaZero active agent must be seat 1 or 2, got {}",
            active_agent.0
        ));
    }
    let observation = timestep.sole_observation()?;
    if observation.agent_id != active_agent {
        return Err(anyhow!(
            "AlphaZero observation belongs to agent {}, but decision belongs to agent {}",
            observation.agent_id.0,
            active_agent.0
        ));
    }
    Ok((active_agent, observation.data.as_slice()))
}

fn require_reset_timestep(timestep: &ErasedTimestep) -> Result<(AgentId, &[u8])> {
    if timestep.source != TransitionSource::Reset {
        return Err(anyhow!(
            "environment reset returned non-reset transition source {:?}",
            timestep.source
        ));
    }
    require_two_player_outcomes(timestep)?;
    require_active_position(timestep)
}

fn require_step_timestep(timestep: &ErasedTimestep, expected_actor: AgentId) -> Result<f32> {
    let actor = match &timestep.source {
        TransitionSource::Agents { agent_ids } if agent_ids.len() == 1 => agent_ids[0],
        source => {
            return Err(anyhow!(
                "AlphaZero step must report exactly one acting agent, got {source:?}"
            ))
        }
    };
    if actor != expected_actor {
        return Err(anyhow!(
            "environment reported acting agent {}, expected {}",
            actor.0,
            expected_actor.0
        ));
    }

    require_two_player_outcomes(timestep)?;
    let actor_reward = timestep
        .reward_for(actor)
        .ok_or_else(|| anyhow!("missing reward for acting agent {}", actor.0))?;
    let observation = timestep.sole_observation()?;

    match timestep.episode {
        EpisodeStatus::Running => {
            let (next_actor, _) = require_active_position(timestep)?;
            if next_actor == actor {
                return Err(anyhow!(
                    "AlphaZero alternating-turn contract kept agent {} active after its step",
                    actor.0
                ));
            }
        }
        EpisodeStatus::Terminated | EpisodeStatus::Truncated => {
            if timestep.decision != Decision::None {
                return Err(anyhow!(
                    "completed AlphaZero timestep must have no next decision, got {:?}",
                    timestep.decision
                ));
            }
            if !matches!(observation.agent_id, AgentId(1) | AgentId(2)) {
                return Err(anyhow!(
                    "terminal AlphaZero observation has unknown agent {}",
                    observation.agent_id.0
                ));
            }
        }
    }

    Ok(actor_reward)
}

enum RuntimeModelWatcher {
    Filesystem(ModelWatcher),
    #[cfg(feature = "s3")]
    S3(S3ModelWatcher),
}

impl RuntimeModelWatcher {
    async fn try_load_existing(&self) -> Result<bool> {
        match self {
            Self::Filesystem(watcher) => watcher.try_load_existing(),
            #[cfg(feature = "s3")]
            Self::S3(watcher) => watcher.try_load_existing().await,
        }
    }

    fn model_info(&self) -> Arc<std::sync::RwLock<ModelInfo>> {
        match self {
            Self::Filesystem(watcher) => watcher.model_info(),
            #[cfg(feature = "s3")]
            Self::S3(watcher) => watcher.model_info(),
        }
    }
}

fn require_source_checkpoint(
    expected_source_checkpoint_id: Option<&str>,
    loaded: bool,
    model_info: &ModelInfo,
) -> Result<()> {
    match expected_source_checkpoint_id {
        None if loaded || model_info.loaded || model_info.checkpoint_id.is_some() => Err(anyhow!(
            "root collection requires an absent RunHead, but checkpoint '{}' was loaded",
            model_info.checkpoint_id.as_deref().unwrap_or("unknown")
        )),
        None => Ok(()),
        Some(expected) if !loaded || !model_info.loaded => Err(anyhow!(
            "collection requires source checkpoint '{expected}', but no RunHead model was loaded"
        )),
        Some(expected) if model_info.checkpoint_id.as_deref() != Some(expected) => Err(anyhow!(
            "loaded RunHead checkpoint '{}' does not match required source checkpoint '{expected}'",
            model_info.checkpoint_id.as_deref().unwrap_or("unknown")
        )),
        Some(_) => Ok(()),
    }
}

impl AlphaZeroCollector {
    pub async fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let eval_batch_size = usize::try_from(config.eval_batch_size)
            .map_err(|_| anyhow!("eval_batch_size does not fit this platform's usize"))?;
        let onnx_intra_threads = usize::try_from(config.onnx_intra_threads)
            .map_err(|_| anyhow!("onnx_intra_threads does not fit this platform's usize"))?;

        // Register all games
        engine_games::register_all_environments();

        // Create engine context for the specified game
        let engine = EngineContext::new(&config.env_id)
            .map_err(|error| anyhow!("Environment '{}' is unavailable: {error}", config.env_id))?;

        let algorithm = resolve_algorithm(&config.algorithm_id)?;
        let compatibility = algorithm.compatibility(&engine);
        compatibility.require_compatible()?;
        let algorithm = algorithm.descriptor();
        info!(
            algorithm = algorithm.id,
            env_id = %config.env_id,
            model_contract = algorithm.components.model_contract,
            experience_schema = algorithm.components.experience_schema,
            "Algorithm compatibility validated"
        );
        debug!(
            algorithm = algorithm.id,
            assumptions = ?compatibility.unverified_assumptions,
            "Environment semantics not yet machine-verifiable"
        );

        let caps = engine.capabilities();
        let max_horizon = require_max_horizon(&caps)?;
        require_reachable_temperature_threshold(config.temp_threshold, max_horizon)?;
        let environment_metadata = engine.metadata();
        let board_metadata = environment_metadata.require_board()?.clone();
        info!(
            "Actor {} initialized for environment {}",
            config.actor_id, caps.id.env_id
        );
        info!(
            "Game capabilities: max_horizon={}, preferred_batch={}",
            max_horizon, caps.preferred_batch
        );

        let num_actions = board_metadata.action_count;
        let obs_size = board_metadata.observation.elements;

        // Virtual loss remains owned by the versioned MCTS component. Every
        // run-varying collector search setting is supplied by the authenticated
        // run recipe and reaches this exact configuration.
        let mut mcts_config = MctsConfig::for_training()
            .with_simulations(config.num_simulations)
            .with_eval_batch_size(eval_batch_size)
            .with_c_puct(config.c_puct)
            .with_temperature(config.temperature);
        mcts_config.dirichlet_alpha = config.dirichlet_alpha;
        mcts_config.dirichlet_epsilon = config.dirichlet_weight;
        mcts_config
            .validate()
            .map_err(|error| anyhow!("invalid authenticated MCTS configuration: {error}"))?;
        let virtual_loss = mcts_config.virtual_loss;

        let mcts_policy = MctsPolicy::new(config.env_id.clone(), num_actions, obs_size)
            .with_config(mcts_config)
            .with_temp_schedule(config.temp_threshold, config.late_temperature);

        info!(
            num_simulations = config.num_simulations,
            c_puct = config.c_puct,
            temperature = config.temperature,
            late_temperature = config.late_temperature,
            temp_threshold = config.temp_threshold,
            dirichlet_alpha = config.dirichlet_alpha,
            dirichlet_weight = config.dirichlet_weight,
            eval_batch_size = config.eval_batch_size,
            virtual_loss,
            "Authenticated collector MCTS configuration"
        );

        let runtime_profile =
            RuntimeProfile::new(algorithm.id, config.env_id.clone(), caps.contract_version)?;

        // Create the exact profile-bound model watcher and try to load once.
        let model_contract =
            algorithm.model_artifact_contract(config.env_id.clone(), caps.contract_version);
        let model_spec = ModelLoadSpec::new(
            obs_size,
            num_actions,
            onnx_intra_threads,
            max_horizon,
            model_contract,
        )?;
        let watcher = match crate::config::central_config()
            .storage
            .model_backend
            .as_str()
        {
            "filesystem" => {
                let model_dir = runtime_profile.model_dir(&config.data_dir);
                std::fs::create_dir_all(&model_dir)?;
                RuntimeModelWatcher::Filesystem(ModelWatcher::new(
                    model_dir,
                    model_spec,
                    ModelSelection::Latest,
                    mcts_policy.evaluator_ref(),
                ))
            }
            "s3" => {
                #[cfg(feature = "s3")]
                {
                    let storage = &crate::config::central_config().storage;
                    let profile_data_dir = runtime_profile.data_dir(&config.data_dir);
                    let bucket = storage.s3_bucket.clone().ok_or_else(|| {
                        anyhow!("storage.s3_bucket is required for S3 model watching")
                    })?;
                    RuntimeModelWatcher::S3(
                        S3ModelWatcher::new(
                            S3Config {
                                bucket,
                                prefix: runtime_profile.model_prefix(),
                                endpoint_url: storage.s3_endpoint.clone(),
                                region: None,
                                cache_dir: profile_data_dir.join("model-cache"),
                            },
                            model_spec,
                            ModelSelection::Latest,
                            mcts_policy.evaluator_ref(),
                        )
                        .await?,
                    )
                }
                #[cfg(not(feature = "s3"))]
                {
                    return Err(anyhow!(
                        "storage.model_backend is 's3' but the actor binary was built without the s3 feature"
                    ));
                }
            }
            backend => return Err(anyhow!("unsupported model storage backend '{backend}'")),
        };
        let loaded = watcher.try_load_existing().await?;
        let model_info = watcher.model_info();
        let model_info = model_info
            .read()
            .map_err(|error| anyhow!("failed to read loaded model identity: {error}"))?
            .clone();
        require_source_checkpoint(config.source_checkpoint_id.as_deref(), loaded, &model_info)?;
        if loaded {
            info!(
                checkpoint_id = model_info.checkpoint_id.as_deref().unwrap_or("unknown"),
                "Pinned required source model for this one-shot collection"
            );
        } else {
            info!("Confirmed root collection has no RunHead; using uniform evaluator");
        }
        // Dropping the watcher freezes the evaluator selected above. The
        // process never subscribes to RunHead updates mid-collection.
        drop(watcher);

        // Open replay only after the exact model generation has been pinned.
        let replay_profile = ReplayProfile {
            env_id: config.env_id.clone(),
            env_contract_version: caps.contract_version,
            algorithm_id: algorithm.id.to_string(),
            experience_schema: algorithm.components.experience_schema.to_string(),
        };
        let replay_selection = ReplaySelection {
            profile: replay_profile,
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
            selection = ?replay_selection,
            "Opaque replay record store initialized (PostgreSQL)"
        );

        // Initialize stats tracking
        let stats = ActorStats::new(&config.env_id);

        Ok(Self {
            config,
            replay_selection,
            board_metadata,
            engine: Mutex::new(engine),
            mcts_policy: Mutex::new(mcts_policy),
            replay: Arc::from(replay),
            episode_count: AtomicU32::new(0),
            shutdown_signal: AtomicBool::new(false),
            stats,
        })
    }

    pub fn shutdown(&self) {
        self.shutdown_signal.store(true, Ordering::Relaxed);
        info!("Shutdown signal set");
    }

    /// Run exactly the configured episode quota or fail the scoped attempt.
    pub async fn run(&self) -> Result<()> {
        let initial_rss = rss_mb().unwrap_or(0.0);
        info!(
            actor_id = %self.config.actor_id,
            max_episodes = self.config.max_episodes,
            collection_scope_id = %self.config.collection_scope_id,
            source_checkpoint_id = self.config.source_checkpoint_id.as_deref().unwrap_or("root"),
            initial_rss_mb = format!("{:.1}", initial_rss),
            "Actor starting bounded one-shot collection"
        );

        // Create progress bar when stderr is a TTY.
        let progress = if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
            let pb = ProgressBar::new(self.config.max_episodes as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} episodes ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };

        info!("Entering one-shot episode loop");

        loop {
            let current_episode_count = self.episode_count.load(Ordering::Relaxed);
            if current_episode_count >= self.config.max_episodes {
                info!(
                    "Reached maximum episodes ({}), stopping",
                    self.config.max_episodes
                );
                break;
            }
            if self.shutdown_signal.load(Ordering::Relaxed) {
                return Err(anyhow!(
                    "shutdown interrupted bounded collection after {current_episode_count}/{} completed episodes",
                    self.config.max_episodes
                ));
            }

            // Run an episode
            let episode_start = Instant::now();
            match self.run_episode().await {
                Ok(EpisodeOutcome::Abandoned {
                    reason,
                    steps,
                    discarded,
                    timeout_secs,
                }) => {
                    let abandoned = self.stats.record_abandoned_episode(discarded);
                    // Report the running rate, not just this one episode: a
                    // stray drop is noise, a steady stream means the replay
                    // buffer is quietly losing its longest games.
                    let completed = self.episode_count.load(Ordering::Relaxed);
                    let attempted = completed + abandoned;
                    warn!(
                        reason = reason.as_str(),
                        steps,
                        discarded,
                        timeout_secs,
                        abandoned_total = abandoned,
                        attempted,
                        guidance = reason.guidance(),
                        abandoned_pct =
                            format!("{:.1}", 100.0 * abandoned as f64 / attempted.max(1) as f64),
                        "Episode abandoned before terminal state; its replay records were discarded"
                    );
                    return Err(anyhow!(
                        "bounded collection abandoned episode after {steps} steps ({})",
                        reason.as_str()
                    ));
                }
                Ok(EpisodeOutcome::Completed {
                    steps,
                    player_one_outcome,
                    stats: episode_stats,
                }) => {
                    let new_count = self.episode_count.fetch_add(1, Ordering::Relaxed) + 1;
                    let duration = episode_start.elapsed().as_secs_f64();
                    debug!(
                        episode = new_count,
                        steps, player_one_outcome, duration, "Episode completed"
                    );

                    // Record episode in stats tracker
                    self.stats.record_episode(steps, player_one_outcome);
                    self.stats.record_mcts_stats(
                        episode_stats.search_count,
                        episode_stats.inference_time_us,
                    );

                    // Update progress bar
                    if let Some(ref pb) = progress {
                        pb.inc(1);
                    }

                    if self.config.log_interval > 0
                        && new_count.is_multiple_of(self.config.log_interval)
                    {
                        // Include memory diagnostics in periodic logging
                        let rss_info = rss_mb()
                            .map(|mb| format!(", RSS: {:.1} MB", mb))
                            .unwrap_or_default();

                        let log_progress = || {
                            info!(
                                "Completed {} episodes (last: {:.2}s{})",
                                new_count, duration, rss_info
                            );
                            // Log MCTS performance breakdown
                            episode_stats.log_summary(new_count);
                        };

                        // Suspend progress bar while logging to avoid visual glitches
                        match &progress {
                            Some(pb) => pb.suspend(log_progress),
                            None => log_progress(),
                        }
                    }
                }
                Err(e) => {
                    let count = self.episode_count.load(Ordering::Relaxed);
                    return Err(anyhow!(
                        "bounded collection episode {} failed: {e}",
                        count + 1
                    ));
                }
            }
        }

        // Finish progress bar
        if let Some(pb) = progress {
            pb.finish_with_message("done");
        }

        // Report final memory usage
        let final_rss = rss_mb().unwrap_or(0.0);
        let rss_growth = final_rss - initial_rss;
        let final_stats = self.stats.snapshot();
        info!(
            env_id = %final_stats.env_id,
            episodes_completed = final_stats.episodes_completed,
            total_steps = final_stats.total_steps,
            player1_wins = final_stats.player1_wins,
            player2_wins = final_stats.player2_wins,
            draws = final_stats.draws,
            episodes_abandoned = final_stats.episodes_abandoned,
            replay_records_discarded = final_stats.replay_records_discarded,
            avg_episode_length = final_stats.avg_episode_length,
            episodes_per_second = final_stats.episodes_per_second,
            runtime_seconds = final_stats.runtime_seconds,
            mcts_avg_inference_us = final_stats.mcts_avg_inference_us,
            snapshot_timestamp = final_stats.timestamp,
            final_rss_mb = format!("{:.1}", final_rss),
            rss_growth_mb = format!("{:.1}", rss_growth),
            "Actor completed bounded collection"
        );
        Ok(())
    }

    /// Acquire engine lock with consistent error handling
    fn lock_engine(&self) -> Result<MutexGuard<'_, EngineContext>> {
        self.engine
            .lock()
            .map_err(|e| anyhow!("Engine lock poisoned: {}", e))
    }

    /// Acquire MCTS policy lock with consistent error handling
    fn lock_mcts_policy(&self) -> Result<MutexGuard<'_, MctsPolicy>> {
        self.mcts_policy
            .lock()
            .map_err(|e| anyhow!("MCTS policy lock poisoned: {}", e))
    }

    /// Encode terminal value targets and store immutable replay records.
    async fn finalize_episode(
        &self,
        pending_experiences: Vec<PendingExperience>,
        terminal_timestep: &ErasedTimestep,
        episode_id: &str,
    ) -> Result<f32> {
        if terminal_timestep.episode != EpisodeStatus::Terminated {
            return Err(anyhow!(
                "cannot finalize AlphaZero replay from {:?} episode",
                terminal_timestep.episode
            ));
        }
        require_two_player_outcomes(terminal_timestep)?;

        let replay_records = pending_experiences
            .into_iter()
            .map(|pending| {
                let value_target =
                    terminal_timestep.reward_for(pending.actor).ok_or_else(|| {
                        anyhow!(
                            "terminal timestep has no outcome for experience actor {}",
                            pending.actor.0
                        )
                    })?;
                let payload = encode_experience(
                    &pending.observation,
                    self.board_metadata.observation.elements,
                    &pending.policy_target,
                    self.board_metadata.action_count,
                    value_target,
                )?;
                Ok(self.replay_selection.record(
                    format!("{episode_id}-step-{}", pending.step_number),
                    episode_id,
                    pending.step_number,
                    payload,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let player_one_outcome = terminal_timestep
            .reward_for(AgentId(1))
            .expect("validated terminal outcome for player one");

        self.replay
            .store_batch(&replay_records)
            .await
            .map_err(|e| {
                error!(
                    "Failed to store replay records for episode {}: {}",
                    episode_id, e
                );
                e
            })?;

        debug!(
            "Stored {} replay records with player_one_outcome={} for episode {}",
            replay_records.len(),
            player_one_outcome,
            episode_id
        );

        Ok(player_one_outcome)
    }

    async fn run_episode(&self) -> Result<EpisodeOutcome> {
        let episode_count = self.episode_count.load(Ordering::Relaxed);

        // Get max_horizon and reset the game
        let (reset_result, max_horizon) = {
            let mut engine = self.lock_engine()?;
            let caps = engine.capabilities();
            let seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;
            let reset = engine.reset(seed, &[])?;
            (reset, require_max_horizon(&caps)?)
        };

        // Create episode context with timing and limits
        let ctx = EpisodeContext::new(
            &self.config.actor_id,
            episode_count,
            self.config.episode_timeout_secs,
            max_horizon,
        )?;

        debug!(
            episode = episode_count + 1,
            env_id = %self.config.env_id,
            timeout_secs = ctx.timeout.as_secs(),
            max_steps = ctx.max_steps,
            "Starting episode {}",
            ctx.id
        );

        // Episode state
        let mut current_state = reset_result.state;
        let mut current_timestep = reset_result.timestep;
        let (mut current_agent, reset_observation) = require_reset_timestep(&current_timestep)?;
        let mut current_obs = reset_observation.to_vec();
        let mut current_legal_mask = self.board_metadata.legal_mask_from_obs(&current_obs)?;
        let mut step_number = 0u32;
        let mut steps_taken = 0u32;
        let mut pending_experiences: Vec<PendingExperience> = Vec::with_capacity(12);
        let mut episode_stats = EpisodeStats::default();

        loop {
            if let Some(reason) = ctx.limit_exceeded(steps_taken) {
                return Ok(EpisodeOutcome::Abandoned {
                    reason,
                    steps: steps_taken,
                    discarded: pending_experiences.len(),
                    timeout_secs: ctx.timeout.as_secs(),
                });
            }

            // Select action using MCTS policy
            let policy_result = {
                let mut policy = self.lock_mcts_policy()?;
                policy.select_action(
                    &current_state,
                    &current_timestep,
                    &current_legal_mask,
                    step_number,
                )?
            };

            // Accumulate MCTS performance stats
            episode_stats.add(&policy_result.stats);

            // Take step in environment
            let step_result = {
                let mut engine = self.lock_engine()?;
                engine.step(&current_state, &policy_result.action)?
            };

            require_step_timestep(&step_result.timestep, current_agent)?;
            steps_taken += 1;

            pending_experiences.push(PendingExperience {
                actor: current_agent,
                step_number,
                observation: std::mem::take(&mut current_obs),
                policy_target: policy_result.policy,
            });

            match step_result.timestep.episode {
                EpisodeStatus::Terminated => {
                    let player_one_outcome = self
                        .finalize_episode(pending_experiences, &step_result.timestep, &ctx.id)
                        .await?;
                    debug!(
                        "Episode {} completed in {} steps, player-one outcome: {:.2}",
                        ctx.id,
                        step_number + 1,
                        player_one_outcome
                    );
                    return Ok(EpisodeOutcome::Completed {
                        steps: steps_taken,
                        player_one_outcome,
                        stats: episode_stats,
                    });
                }
                EpisodeStatus::Truncated => {
                    return Ok(EpisodeOutcome::Abandoned {
                        reason: AbandonReason::EnvironmentTruncated,
                        steps: steps_taken,
                        discarded: pending_experiences.len(),
                        timeout_secs: ctx.timeout.as_secs(),
                    });
                }
                EpisodeStatus::Running => {}
            }

            // Update state for next step
            current_state = step_result.state;
            current_timestep = step_result.timestep;
            let (next_agent, observation) = require_active_position(&current_timestep)?;
            current_agent = next_agent;
            current_obs = observation.to_vec();
            // Read the next legal-action mask from its authoritative observation.
            current_legal_mask = self.board_metadata.legal_mask_from_obs(&current_obs)?;
            step_number += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let mismatch =
            require_source_checkpoint(Some(&expected), true, &loaded_model(&"b".repeat(64)))
                .unwrap_err()
                .to_string();
        assert!(mismatch.contains("does not match required source checkpoint"));
    }

    // ========================================
    // Episode limit / abandonment tests
    // ========================================

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
    fn test_episode_timeout_is_exact_for_long_horizon() {
        let ctx = EpisodeContext::new("a", 0, 180, 402).unwrap();
        assert_eq!(ctx.timeout, Duration::from_secs(180));
    }

    #[test]
    fn test_episode_timeout_is_exact_for_short_horizon() {
        let ctx = EpisodeContext::new("a", 0, 180, 42).unwrap();
        assert_eq!(ctx.timeout, Duration::from_secs(180));
    }

    #[test]
    fn test_limit_exceeded_reports_no_reason_within_budget() {
        let ctx = EpisodeContext::new("a", 0, 300, 42).unwrap();
        assert_eq!(ctx.limit_exceeded(0), None);
        assert_eq!(ctx.limit_exceeded(ctx.max_steps - 1), None);
    }

    #[test]
    fn test_limit_exceeded_reports_max_steps() {
        let ctx = EpisodeContext::new("a", 0, 300, 42).unwrap();
        assert_eq!(
            ctx.limit_exceeded(ctx.max_steps),
            Some(AbandonReason::MaxSteps)
        );
    }

    #[test]
    fn test_limit_exceeded_reports_timeout() {
        // Config validation rejects zero, but the internal context still obeys
        // the exact value and reports timeout before the step guard.
        let ctx = EpisodeContext::new("a", 0, 0, 1).unwrap();
        assert_eq!(ctx.timeout, Duration::ZERO);
        assert_eq!(ctx.limit_exceeded(0), Some(AbandonReason::Timeout));
    }

    #[test]
    fn test_abandon_reason_strings_are_stable() {
        // These values are part of the structured-log schema used by queries.
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
        assert_eq!(observation.len(), 29 * std::mem::size_of::<f32>());

        let mut multiple_decisions = reset.timestep.clone();
        multiple_decisions.decision = Decision::Agents {
            agent_ids: vec![AgentId(1), AgentId(2)],
        };
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

        // Player one wins across the top row. Each reward is read using the
        // acting AgentId reported by TransitionSource, never by ply parity.
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
        // These tests require a running PostgreSQL instance
        // Run: docker compose up postgres
        Config {
            actor_id: "test-actor".into(),
            env_id: "tictactoe".into(),
            algorithm_id: algorithm_core::ALPHAZERO_BOARD_V1_ID.into(),
            max_episodes: 1,
            collection_scope_id: "a".repeat(64),
            source_checkpoint_id: None,
            episode_timeout_secs: 30,
            log_level: "info".into(),
            log_interval: 10,
            data_dir: "./data".into(),
            num_simulations: 50, // Fewer for tests
            c_puct: 1.4,
            temperature: 1.0,
            late_temperature: 1.0,
            temp_threshold: 0, // Disabled for tests
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
            eval_batch_size: 32,
            onnx_intra_threads: 1,
            postgres_url: std::env::var("CARTRIDGE_STORAGE_POSTGRES_URL").unwrap_or_else(|_| {
                "postgresql://cartridge:cartridge@localhost:5432/cartridge".into()
            }),
        }
    }

    #[tokio::test]
    #[ignore] // Requires running PostgreSQL: docker compose up postgres
    async fn test_actor_creation() {
        let config = test_config();

        let actor = AlphaZeroCollector::new(config).await;
        assert!(actor.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires running PostgreSQL: docker compose up postgres
    async fn test_actor_run_single_episode() {
        let config = test_config();

        let actor = AlphaZeroCollector::new(config).await.unwrap();

        // Run a single episode
        let result = actor.run_episode().await;
        assert!(result.is_ok());

        match result.unwrap() {
            EpisodeOutcome::Completed {
                steps,
                player_one_outcome,
                ..
            } => {
                assert!(steps > 0, "Episode should have at least one step");
                // TicTacToe gives reward at end of game
                debug!(steps, player_one_outcome, "Episode completed");
            }
            EpisodeOutcome::Abandoned { reason, steps, .. } => {
                panic!("TicTacToe episode abandoned ({reason:?}) after {steps} steps");
            }
        }
    }

    #[tokio::test]
    #[ignore] // Requires running PostgreSQL: docker compose up postgres
    async fn test_actor_nonexistent_game() {
        let mut config = test_config();
        config.env_id = "nonexistent_game".into();

        let result = AlphaZeroCollector::new(config).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        let err_msg = err.to_string();
        assert!(err_msg.contains("not registered"));
    }

    #[tokio::test]
    #[ignore] // Requires running PostgreSQL: docker compose up postgres
    async fn test_actor_stores_replay_records() {
        let config = test_config();

        let actor = AlphaZeroCollector::new(config).await.unwrap();

        // Run an episode
        actor.run_episode().await.unwrap();

        // Check that replay records were stored
        let count = actor.replay.count().await.unwrap();
        assert!(count > 0, "Should have stored some replay records");
    }
}
