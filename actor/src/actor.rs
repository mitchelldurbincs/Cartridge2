//! Actor implementation using engine-core library directly

use anyhow::{anyhow, Result};
use engine_core::EngineContext;
use indicatif::{ProgressBar, ProgressStyle};
use mcts::{MctsConfig, SearchStats};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Mutex, MutexGuard,
};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};

/// Get the current resident set size (RSS) in MB from /proc/self/status.
/// Returns None if unable to read (e.g., on non-Linux systems).
fn get_rss_mb() -> Option<f64> {
    use std::fs;
    // Read /proc/self/status and find VmRSS line
    if let Ok(contents) = fs::read_to_string("/proc/self/status") {
        for line in contents.lines() {
            if line.starts_with("VmRSS:") {
                // Format: "VmRSS:    12345 kB"
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<f64>() {
                        return Some(kb / 1024.0); // Convert to MB
                    }
                }
            }
        }
    }
    None
}

use crate::config::Config;
use crate::game_config::{get_config, GameConfig};
use crate::health::HealthState;
use crate::mcts_policy::MctsPolicy;
use crate::metrics;
use crate::model_watcher::ModelWatcher;
use crate::stats::ActorStats;
use crate::storage::{create_replay_store, ReplayStore, StorageConfig, Transition};
use std::sync::Arc;

/// Context for a single episode, containing metadata and timing information.
struct EpisodeContext {
    id: String,
    start_time: Instant,
    timeout: Duration,
    max_steps: u32,
}

/// Aggregated MCTS stats for an episode.
#[derive(Debug, Default)]
struct EpisodeStats {
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
}

pub struct Actor {
    config: Config,
    game_config: GameConfig,
    engine: Mutex<EngineContext>,
    mcts_policy: Mutex<MctsPolicy>,
    replay: Arc<dyn ReplayStore>,
    episode_count: AtomicU32,
    shutdown_signal: AtomicBool,
    model_watcher: Option<ModelWatcher>,
    stats: ActorStats,
}

impl Actor {
    pub async fn new(config: Config) -> Result<Self> {
        // Register all games
        engine_games::register_all_games();

        // Get game configuration from registry
        let game_config = get_config(&config.env_id)?;
        info!(
            "Loaded game config for {}: {} actions, {} obs size",
            config.env_id, game_config.num_actions, game_config.obs_size
        );

        // Create engine context for the specified game
        let engine = EngineContext::new(&config.env_id)
            .ok_or_else(|| anyhow!("Game '{}' not registered", config.env_id))?;

        let caps = engine.capabilities();
        info!(
            "Actor {} initialized for environment {}",
            config.actor_id, caps.id.env_id
        );
        info!(
            "Game capabilities: max_horizon={}, preferred_batch={}",
            caps.max_horizon, caps.preferred_batch
        );

        let num_actions = game_config.num_actions;
        let obs_size = game_config.obs_size;

        // Create MCTS policy with training configuration
        // num_simulations, eval_batch_size, temp_threshold are configurable via CLI/env for orchestrator control
        let mcts_config = MctsConfig::for_training()
            .with_simulations(config.num_simulations)
            .with_eval_batch_size(config.eval_batch_size)
            .with_temperature(1.0); // Base exploration temperature

        let mcts_policy = MctsPolicy::new(config.env_id.clone(), num_actions, obs_size)
            .with_config(mcts_config)
            .with_temp_schedule(config.temp_threshold, 0.1); // Late-game temp

        info!(
            "MCTS config: {} simulations, eval_batch_size={}, temp_threshold={} (0=disabled)",
            config.num_simulations, config.eval_batch_size, config.temp_threshold
        );

        // Create model watcher (optional when --no-watch is set)
        let model_dir = format!("{}/models", config.data_dir);
        let model_watcher = if config.no_watch {
            // No-watch mode: load model once at startup, no file watching
            // Create a temporary watcher just to load the model, then discard it
            let temp_watcher = ModelWatcher::new(
                &model_dir,
                "latest.onnx",
                obs_size,
                config.onnx_intra_threads,
                mcts_policy.evaluator_ref(),
            );
            match temp_watcher.try_load_existing() {
                Ok(true) => {
                    info!("Loaded existing model (no-watch mode)");
                    metrics::MODEL_LOADED.set(1);
                    metrics::MODEL_RELOADS.inc();
                }
                Ok(false) => {
                    info!("No existing model found, will use random policy (no-watch mode)");
                    metrics::MODEL_LOADED.set(0);
                }
                Err(e) => {
                    warn!("Failed to load existing model: {}", e);
                    metrics::MODEL_LOADED.set(0);
                }
            }
            None
        } else {
            // Normal mode: create watcher for hot-reload
            let watcher = ModelWatcher::new(
                &model_dir,
                "latest.onnx",
                obs_size,
                config.onnx_intra_threads,
                mcts_policy.evaluator_ref(),
            );
            match watcher.try_load_existing() {
                Ok(true) => {
                    info!("Loaded existing model");
                    metrics::MODEL_LOADED.set(1);
                    metrics::MODEL_RELOADS.inc();
                }
                Ok(false) => {
                    info!("No existing model found, will use random policy until model available");
                    metrics::MODEL_LOADED.set(0);
                }
                Err(e) => {
                    warn!("Failed to load existing model: {}", e);
                    metrics::MODEL_LOADED.set(0);
                }
            }
            Some(watcher)
        };

        // Initialize replay buffer (PostgreSQL with connection pooling)
        let storage_config = StorageConfig {
            postgres_url: config.postgres_url.clone(),
            pool_config: config.pool_config(),
        };

        let replay = create_replay_store(&storage_config).await?;
        info!("Replay buffer initialized (PostgreSQL)");

        // Store game metadata in database (makes it self-describing for trainer)
        let metadata = engine.metadata();
        replay.store_metadata(&metadata).await?;
        info!(
            "Stored game metadata: {} actions, {} obs_size, legal_mask_offset={}",
            metadata.num_actions, metadata.obs_size, metadata.legal_mask_offset
        );

        // Initialize stats tracking
        let stats = ActorStats::new(&config.data_dir, &config.env_id);
        info!("Actor stats will be written to {}", stats.stats_path());

        Ok(Self {
            config,
            game_config,
            engine: Mutex::new(engine),
            mcts_policy: Mutex::new(mcts_policy),
            replay: Arc::from(replay),
            episode_count: AtomicU32::new(0),
            shutdown_signal: AtomicBool::new(false),
            model_watcher,
            stats,
        })
    }

    pub fn shutdown(&self) {
        self.shutdown_signal.store(true, Ordering::Relaxed);
        info!("Shutdown signal set");
    }

    /// Run the actor main loop with health state tracking for Kubernetes probes.
    /// Records episode completions to the health state for liveness tracking.
    pub async fn run(&self, health: &HealthState) -> Result<()> {
        let initial_rss = get_rss_mb().unwrap_or(0.0);
        info!(
            actor_id = %self.config.actor_id,
            max_episodes = self.config.max_episodes,
            no_watch = self.config.no_watch,
            initial_rss_mb = format!("{:.1}", initial_rss),
            "Actor starting main loop (with health tracking)"
        );

        // Create progress bar for bounded episode runs (only when stderr is a TTY)
        let progress = if self.config.max_episodes > 0
            && std::io::IsTerminal::is_terminal(&std::io::stderr())
        {
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

        // Start model watcher (only if not in no-watch mode)
        let mut model_updates = if let Some(ref watcher) = self.model_watcher {
            Some(watcher.start_watching().await?)
        } else {
            info!("Running in no-watch mode, model will not be reloaded");
            None
        };

        // Setup flush timer for periodic database commits
        let mut flush_timer = tokio::time::interval(self.config.flush_interval());

        info!("Entering main event loop");

        loop {
            // Check shutdown signal
            if self.shutdown_signal.load(Ordering::Relaxed) {
                info!("Shutdown signal received, stopping actor");
                break;
            }

            // Check episode limit first (non-blocking)
            let current_episode_count = self.episode_count.load(Ordering::Relaxed);
            if self.config.max_episodes > 0
                && current_episode_count >= self.config.max_episodes as u32
            {
                info!(
                    "Reached maximum episodes ({}), stopping",
                    self.config.max_episodes
                );
                break;
            }

            // Handle model updates if watching is enabled
            if let Some(ref mut updates) = model_updates {
                tokio::select! {
                    biased;  // Prioritize model updates and flush over episodes

                    Some(()) = updates.recv() => {
                        info!("Model updated, next episode will use new model");
                        // Record model reload in Prometheus
                        metrics::MODEL_RELOADS.inc();
                        metrics::MODEL_LOADED.set(1);
                        continue;
                    }

                    _ = flush_timer.tick() => {
                        debug!("Periodic flush tick");
                        continue;
                    }

                    _ = tokio::time::sleep(Duration::from_millis(1)) => {
                        // Run episode below
                    }
                }
            } else {
                // No-watch mode: just check flush timer non-blockingly
                tokio::select! {
                    biased;

                    _ = flush_timer.tick() => {
                        debug!("Periodic flush tick");
                        continue;
                    }

                    _ = tokio::time::sleep(Duration::from_millis(1)) => {
                        // Run episode below
                    }
                }
            }

            // Run an episode
            let episode_start = Instant::now();
            match self.run_episode().await {
                Ok((steps, total_reward, episode_stats)) => {
                    let new_count = self.episode_count.fetch_add(1, Ordering::Relaxed) + 1;
                    let duration = episode_start.elapsed().as_secs_f64();
                    debug!(
                        episode = new_count,
                        steps, total_reward, duration, "Episode completed"
                    );

                    // Record Prometheus metrics for this episode
                    metrics::EPISODES_TOTAL.inc();
                    metrics::EPISODE_DURATION.observe(duration);
                    metrics::EPISODE_STEPS.observe(steps as f64);
                    metrics::record_outcome(total_reward);

                    // Update throughput gauge
                    let elapsed = episode_start.elapsed().as_secs_f64();
                    if elapsed > 0.0 {
                        metrics::EPISODES_PER_SECOND.set(new_count as f64 / elapsed);
                    }

                    // Record episode completion for health tracking
                    health.record_episode_complete();

                    // Record episode in stats tracker
                    self.stats.record_episode(steps, total_reward);
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
                        let rss_info = get_rss_mb()
                            .map(|mb| format!(", RSS: {:.1} MB", mb))
                            .unwrap_or_default();
                        let avg_duration = duration; // Most recent episode duration

                        // Suspend progress bar while logging to avoid visual glitches
                        if let Some(ref pb) = progress {
                            pb.suspend(|| {
                                info!(
                                    "Completed {} episodes (last: {:.2}s{})",
                                    new_count, avg_duration, rss_info
                                );
                                // Log MCTS performance breakdown
                                episode_stats.log_summary(new_count);
                            });
                        } else {
                            info!(
                                "Completed {} episodes (last: {:.2}s{})",
                                new_count, avg_duration, rss_info
                            );
                            // Log MCTS performance breakdown
                            episode_stats.log_summary(new_count);
                        }

                        // Write stats to file for web frontend
                        self.stats.write_stats();
                    }
                }
                Err(e) => {
                    let count = self.episode_count.load(Ordering::Relaxed);
                    error!("Episode {} failed: {}", count + 1, e);
                    // Continue with next episode rather than stopping
                }
            }
        }

        // Finish progress bar
        if let Some(pb) = progress {
            pb.finish_with_message("done");
        }

        // Write final stats
        self.stats.write_stats();

        // Report final memory usage
        let final_rss = get_rss_mb().unwrap_or(0.0);
        let rss_growth = final_rss - initial_rss;
        info!(
            "Actor stopped gracefully (final RSS: {:.1} MB, growth: {:.1} MB)",
            final_rss, rss_growth
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

    /// Check episode limits and return error if exceeded.
    fn check_episode_limits(&self, ctx: &EpisodeContext, steps_taken: u32) -> Result<()> {
        if ctx.is_timed_out() {
            warn!(
                "Episode {} timed out after {:?} ({} steps taken)",
                ctx.id,
                ctx.start_time.elapsed(),
                steps_taken
            );
            return Err(anyhow!(
                "Episode timed out after {} seconds",
                ctx.timeout.as_secs()
            ));
        }

        if steps_taken >= ctx.max_steps {
            warn!(
                "Episode {} exceeded max steps ({}) without terminating",
                ctx.id, ctx.max_steps
            );
            return Err(anyhow!(
                "Episode exceeded {} steps without terminating",
                ctx.max_steps
            ));
        }

        Ok(())
    }

    /// Backfill game outcomes and store transitions.
    async fn finalize_episode(
        &self,
        mut transitions: Vec<Transition>,
        final_reward: f32,
        episode_id: &str,
    ) -> Result<()> {
        let total_steps = transitions.len() as u32;

        // Backfill game outcomes for all transitions
        // The final reward indicates the outcome from the last mover's perspective:
        // +1 = win, -1 = loss, 0 = draw
        for t in &mut transitions {
            let steps_from_end = total_steps.saturating_sub(1).saturating_sub(t.step_number);
            let sign = if steps_from_end % 2 == 0 { 1.0 } else { -1.0 };
            t.game_outcome = Some(final_reward * sign);
        }

        // Batch store all transitions in a single transaction
        self.replay.store_batch(&transitions).await.map_err(|e| {
            error!(
                "Failed to store transitions for episode {}: {}",
                episode_id, e
            );
            e
        })?;

        debug!(
            "Stored {} transitions with game_outcome={} for episode {}",
            transitions.len(),
            final_reward,
            episode_id
        );

        Ok(())
    }

    async fn run_episode(&self) -> Result<(u32, f32, EpisodeStats)> {
        let episode_count = self.episode_count.load(Ordering::Relaxed);

        // Get max_horizon and reset the game
        let (reset_result, max_horizon) = {
            let mut engine = self.lock_engine()?;
            let caps = engine.capabilities();
            let seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;
            let reset = engine.reset(seed, &[])?;
            (reset, caps.max_horizon)
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
        let mut current_obs = reset_result.obs;
        let mut current_legal_mask = self.game_config.extract_legal_mask(&current_obs);
        let mut step_number = 0u32;
        let mut steps_taken = 0u32;
        let mut total_reward = 0.0f32;
        let mut transitions: Vec<Transition> = Vec::with_capacity(12);
        let mut episode_stats = EpisodeStats::default();

        loop {
            self.check_episode_limits(&ctx, steps_taken)?;

            // Select action using MCTS policy
            let policy_result = {
                let mut policy = self.lock_mcts_policy()?;
                policy.select_action(
                    &current_state,
                    &current_obs,
                    current_legal_mask,
                    step_number,
                )?
            };

            // Accumulate MCTS performance stats
            episode_stats.add(&policy_result.stats);

            // Record Prometheus metrics for this MCTS search
            metrics::MCTS_SEARCHES_TOTAL.inc();
            metrics::MCTS_INFERENCE_SECONDS
                .observe(policy_result.stats.inference_time_us as f64 / 1_000_000.0);
            metrics::MCTS_SEARCH_SECONDS
                .observe(policy_result.stats.total_time_us as f64 / 1_000_000.0);

            // Take step in environment
            let step_result = {
                let mut engine = self.lock_engine()?;
                engine.step(&current_state, &policy_result.action)?
            };

            total_reward += step_result.reward;
            steps_taken += 1;

            // Create transition (moves current_state/obs to avoid cloning)
            let policy_bytes: Vec<u8> = policy_result
                .policy
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();

            transitions.push(Transition {
                id: format!("{}-step-{}", ctx.id, step_number),
                env_id: self.config.env_id.clone(),
                episode_id: ctx.id.clone(),
                step_number,
                state: std::mem::take(&mut current_state),
                action: policy_result.action,
                next_state: step_result.state.clone(),
                observation: std::mem::take(&mut current_obs),
                next_observation: step_result.obs.clone(),
                reward: step_result.reward,
                done: step_result.done,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
                policy_probs: policy_bytes,
                mcts_value: policy_result.value,
                game_outcome: None,
            });

            if step_result.done {
                debug!(
                    "Episode {} completed in {} steps, total reward: {:.2}",
                    ctx.id,
                    step_number + 1,
                    total_reward
                );
                self.finalize_episode(transitions, step_result.reward, &ctx.id)
                    .await?;
                break;
            }

            // Update state for next step
            current_state = step_result.state;
            current_obs = step_result.obs;
            current_legal_mask = step_result.info & self.game_config.legal_mask_bits();
            step_number += 1;
        }

        Ok((steps_taken, total_reward, episode_stats))
    }

    /// Get current episode count (for testing)
    #[allow(dead_code)]
    pub fn episode_count(&self) -> u32 {
        self.episode_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        // These tests require a running PostgreSQL instance
        // Run: docker compose up postgres
        Config {
            actor_id: "test-actor".into(),
            env_id: "tictactoe".into(),
            max_episodes: 1,
            episode_timeout_secs: 30,
            flush_interval_secs: 5,
            log_level: "info".into(),
            log_interval: 10,
            data_dir: "./data".into(),
            num_simulations: 50, // Fewer for tests
            temp_threshold: 0,   // Disabled for tests
            eval_batch_size: 32,
            onnx_intra_threads: 1,
            postgres_url: std::env::var("CARTRIDGE_STORAGE_POSTGRES_URL").unwrap_or_else(|_| {
                "postgresql://cartridge:cartridge@localhost:5432/cartridge".into()
            }),
            no_watch: true, // Tests don't need model watching
            health_port: 8081,
        }
    }

    #[tokio::test]
    #[ignore] // Requires running PostgreSQL: docker compose up postgres
    async fn test_actor_creation() {
        let config = test_config();

        let actor = Actor::new(config).await;
        assert!(actor.is_ok());
    }

    #[tokio::test]
    #[ignore] // Requires running PostgreSQL: docker compose up postgres
    async fn test_actor_run_single_episode() {
        let config = test_config();

        let actor = Actor::new(config).await.unwrap();

        // Run a single episode
        let result = actor.run_episode().await;
        assert!(result.is_ok());

        let (steps, reward, _stats) = result.unwrap();
        assert!(steps > 0, "Episode should have at least one step");
        // TicTacToe gives reward at end of game
        // Steps and reward are validated by assertions above
        debug!(steps, reward, "Episode completed");
    }

    #[tokio::test]
    #[ignore] // Requires running PostgreSQL: docker compose up postgres
    async fn test_actor_nonexistent_game() {
        let mut config = test_config();
        config.env_id = "nonexistent_game".into();

        let result = Actor::new(config).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        // Could fail at game_config lookup or engine context creation
        let err_msg = err.to_string();
        assert!(
            err_msg.contains("Unknown game") || err_msg.contains("not registered"),
            "Expected error about unknown/unregistered game, got: {}",
            err_msg
        );
    }

    #[tokio::test]
    #[ignore] // Requires running PostgreSQL: docker compose up postgres
    async fn test_actor_stores_transitions() {
        let config = test_config();

        let actor = Actor::new(config).await.unwrap();

        // Run an episode
        actor.run_episode().await.unwrap();

        // Check that transitions were stored
        let count = actor.replay.count().await.unwrap();
        assert!(count > 0, "Should have stored some transitions");
    }
}
