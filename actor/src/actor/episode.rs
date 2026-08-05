use anyhow::{anyhow, Result};
use engine_core::{AgentId, Decision, EpisodeStatus, ErasedTimestep, TransitionSource};
use mcts::SearchStats;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info};

use super::AlphaZeroCollector;
use crate::algorithms::encode_experience;

pub(super) struct EpisodeContext {
    pub id: String,
    start_time: Instant,
    pub timeout: Duration,
    pub max_steps: u32,
}

impl EpisodeContext {
    pub fn new(
        episode_prefix: &str,
        episode_count: u32,
        timeout_secs: u64,
        max_horizon: u32,
    ) -> Self {
        Self {
            id: format!("{episode_prefix}-ep-{episode_count}"),
            start_time: Instant::now(),
            timeout: Duration::from_secs(timeout_secs),
            max_steps: max_horizon.saturating_mul(10).max(1000),
        }
    }

    pub fn limit_exceeded(&self, steps_taken: u32) -> Option<AbandonReason> {
        if self.start_time.elapsed() > self.timeout {
            Some(AbandonReason::Timeout)
        } else if steps_taken >= self.max_steps {
            Some(AbandonReason::MaxSteps)
        } else {
            None
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct EpisodeStats {
    pub search_count: u32,
    pub total_time_us: u64,
    pub selection_time_us: u64,
    pub inference_time_us: u64,
    pub expansion_time_us: u64,
    pub backprop_time_us: u64,
    pub num_batches: u32,
    pub total_evals: u32,
    pub game_steps: u32,
    pub terminal_hits: u32,
}

impl EpisodeStats {
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

    pub fn log_summary(&self, episode_num: u32) {
        if self.search_count == 0 || self.total_time_us == 0 {
            return;
        }
        let percentage = |value| value as f64 / self.total_time_us as f64 * 100.0;
        let avg_batch_size = if self.num_batches > 0 {
            self.total_evals as f64 / self.num_batches as f64
        } else {
            0.0
        };
        info!(
            episode = episode_num,
            searches = self.search_count,
            total_ms = format!("{:.1}", self.total_time_us as f64 / 1000.0),
            inference_pct = format!("{:.1}%", percentage(self.inference_time_us)),
            expansion_pct = format!("{:.1}%", percentage(self.expansion_time_us)),
            selection_pct = format!("{:.1}%", percentage(self.selection_time_us)),
            backprop_pct = format!("{:.1}%", percentage(self.backprop_time_us)),
            nn_batches = self.num_batches,
            avg_batch_size = format!("{avg_batch_size:.1}"),
            game_steps = self.game_steps,
            terminal_hits = self.terminal_hits,
            "MCTS episode stats"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AbandonReason {
    Timeout,
    MaxSteps,
    EnvironmentTruncated,
}

impl AbandonReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::MaxSteps => "max_steps",
            Self::EnvironmentTruncated => "environment_truncated",
        }
    }

    pub fn guidance(self) -> &'static str {
        match self {
            Self::Timeout => "increase actor.episode_timeout_secs if timeouts persist",
            Self::MaxSteps => {
                "fix the environment termination contract or increase its declared horizon"
            }
            Self::EnvironmentTruncated => {
                "AlphaZero requires terminal outcomes; use a cartridge that supports truncation or change the environment contract"
            }
        }
    }
}

#[derive(Debug)]
pub(super) enum EpisodeOutcome {
    Completed {
        steps: u32,
        player_one_outcome: f32,
        stats: EpisodeStats,
    },
    Abandoned {
        reason: AbandonReason,
        steps: u32,
        discarded: usize,
        timeout_secs: u64,
    },
}

struct PendingExperience {
    actor: AgentId,
    step_number: u32,
    observation: Vec<u8>,
    policy_target: Vec<f32>,
}

struct EpisodeState {
    state: Vec<u8>,
    timestep: ErasedTimestep,
    agent: AgentId,
    observation: Vec<u8>,
    step_number: u32,
    steps_taken: u32,
    pending: Vec<PendingExperience>,
    stats: EpisodeStats,
}

/// Shared by every collector (AlphaZero and DQN alike): the random
/// `process_token` makes episode IDs collide-free across process restarts
/// within the same collection scope.
pub(crate) fn episode_id_prefix(
    actor_id: &str,
    collection_scope_id: &str,
    process_token: u64,
) -> String {
    let scope = collection_scope_id.get(..8).unwrap_or(collection_scope_id);
    format!("{actor_id}-{scope}-{process_token:016x}")
}

pub(super) fn require_active_position(timestep: &ErasedTimestep) -> Result<(AgentId, &[u8])> {
    if timestep.episode != EpisodeStatus::Running {
        return Err(anyhow!(
            "AlphaZero action selection requires a running episode, got {:?}",
            timestep.episode
        ));
    }
    let active_agent = timestep
        .decision
        .sole_agent()
        .ok_or_else(|| {
            anyhow!(
                "AlphaZero requires exactly one acting agent, got {:?}",
                timestep.decision
            )
        })?
        .agent_id;
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

pub(super) fn require_reset_timestep(timestep: &ErasedTimestep) -> Result<(AgentId, &[u8])> {
    require_active_position(timestep)
}

pub(super) fn require_step_timestep(
    timestep: &ErasedTimestep,
    expected_actor: AgentId,
) -> Result<f32> {
    let actor = match &timestep.source {
        TransitionSource::Agents { agent_ids } if agent_ids.len() == 1 => agent_ids[0],
        source => {
            return Err(anyhow!(
                "AlphaZero step must report exactly one acting agent, got {source:?}"
            ));
        }
    };
    if actor != expected_actor {
        return Err(anyhow!(
            "environment reported acting agent {}, expected {}",
            actor.0,
            expected_actor.0
        ));
    }
    let actor_reward = timestep
        .reward_for(actor)
        .ok_or_else(|| anyhow!("missing reward for acting agent {}", actor.0))?;
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
            let observation = timestep.sole_observation()?;
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

impl AlphaZeroCollector {
    pub(super) async fn run_episode(&self) -> Result<EpisodeOutcome> {
        let (context, mut state) = self.start_episode()?;
        loop {
            if let Some(reason) = context.limit_exceeded(state.steps_taken) {
                return Ok(abandoned(&context, &state, reason));
            }
            let policy_result = self.lock_mcts_policy()?.select_action(
                &state.state,
                &state.timestep,
                state.step_number,
            )?;
            state.stats.add(&policy_result.stats);
            let step = self
                .lock_engine()?
                .step(&state.state, &policy_result.action)?;
            require_step_timestep(&step.timestep, state.agent)?;
            state.steps_taken += 1;
            state.pending.push(PendingExperience {
                actor: state.agent,
                step_number: state.step_number,
                observation: std::mem::take(&mut state.observation),
                policy_target: policy_result.policy,
            });
            match step.timestep.episode {
                EpisodeStatus::Terminated => {
                    let outcome = self
                        .finalize_episode(state.pending, &step.timestep, &context.id)
                        .await?;
                    return Ok(EpisodeOutcome::Completed {
                        steps: state.steps_taken,
                        player_one_outcome: outcome,
                        stats: state.stats,
                    });
                }
                EpisodeStatus::Truncated => {
                    return Ok(abandoned(
                        &context,
                        &state,
                        AbandonReason::EnvironmentTruncated,
                    ));
                }
                EpisodeStatus::Running => update_running_state(&mut state, step)?,
            }
        }
    }

    fn start_episode(&self) -> Result<(EpisodeContext, EpisodeState)> {
        let episode_count = self.episode_count.load(Ordering::Relaxed);
        let (reset, max_horizon) = {
            let mut engine = self.lock_engine()?;
            let max_horizon = engine
                .capabilities()
                .max_horizon
                .filter(|value| *value > 0)
                .ok_or_else(|| anyhow!("AlphaZero requires a finite non-zero max_horizon"))?;
            let seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;
            (engine.reset(seed, &[])?, max_horizon)
        };
        let context = EpisodeContext::new(
            &self.episode_prefix,
            episode_count,
            self.config.episode_timeout_secs,
            max_horizon,
        );
        let (agent, observation) = require_reset_timestep(&reset.timestep)?;
        let observation = observation.to_vec();
        debug!(
            episode = episode_count + 1,
            env_id = %self.config.env_id,
            timeout_secs = context.timeout.as_secs(),
            max_steps = context.max_steps,
            "Starting episode {}",
            context.id
        );
        Ok((
            context,
            EpisodeState {
                state: reset.state,
                timestep: reset.timestep,
                agent,
                observation,
                step_number: 0,
                steps_taken: 0,
                pending: Vec::with_capacity(12),
                stats: EpisodeStats::default(),
            },
        ))
    }

    async fn finalize_episode(
        &self,
        pending: Vec<PendingExperience>,
        terminal: &ErasedTimestep,
        episode_id: &str,
    ) -> Result<f32> {
        if terminal.episode != EpisodeStatus::Terminated {
            return Err(anyhow!(
                "cannot finalize AlphaZero replay from {:?} episode",
                terminal.episode
            ));
        }
        let records = pending
            .into_iter()
            .map(|item| {
                let value_target = terminal.reward_for(item.actor).ok_or_else(|| {
                    anyhow!(
                        "terminal timestep has no outcome for experience actor {}",
                        item.actor.0
                    )
                })?;
                let payload = encode_experience(
                    &item.observation,
                    self.obs_size,
                    &item.policy_target,
                    self.num_actions,
                    value_target,
                )?;
                Ok(self.replay_selection.record(
                    format!("{episode_id}-step-{}", item.step_number),
                    episode_id,
                    item.step_number,
                    payload,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let player_one_outcome = terminal
            .reward_for(AgentId(1))
            .ok_or_else(|| anyhow!("terminal timestep has no outcome for player one"))?;
        self.replay.store_batch(&records).await.map_err(|error| {
            error!(episode_id, %error, "Failed to store replay records");
            error
        })?;
        debug!(
            records = records.len(),
            player_one_outcome, episode_id, "Stored terminal AlphaZero replay records"
        );
        Ok(player_one_outcome)
    }
}

fn abandoned(
    context: &EpisodeContext,
    state: &EpisodeState,
    reason: AbandonReason,
) -> EpisodeOutcome {
    EpisodeOutcome::Abandoned {
        reason,
        steps: state.steps_taken,
        discarded: state.pending.len(),
        timeout_secs: context.timeout.as_secs(),
    }
}

fn update_running_state(state: &mut EpisodeState, step: engine_core::StepResult) -> Result<()> {
    let (agent, observation) = require_active_position(&step.timestep)?;
    state.state = step.state;
    state.observation = observation.to_vec();
    state.timestep = step.timestep;
    state.agent = agent;
    state.step_number += 1;
    Ok(())
}
