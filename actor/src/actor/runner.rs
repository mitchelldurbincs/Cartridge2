use anyhow::{anyhow, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::IsTerminal;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tracing::{debug, info, warn};

use super::episode::{EpisodeOutcome, EpisodeStats};
use super::AlphaZeroCollector;
use crate::resources::rss_mb;

pub(super) async fn run(collector: &AlphaZeroCollector) -> Result<()> {
    let initial_rss = rss_mb().unwrap_or(0.0);
    info!(
        actor_id = %collector.config.actor_id,
        max_episodes = collector.config.max_episodes,
        collection_scope_id = %collector.config.collection_scope_id,
        source_checkpoint_id = collector.config.source_checkpoint_id.as_deref().unwrap_or("root"),
        initial_rss_mb = format!("{initial_rss:.1}"),
        "Actor starting bounded one-shot collection"
    );
    let progress = progress_bar(collector.config.max_episodes);
    collect_quota(collector, progress.as_ref()).await?;
    if let Some(progress) = progress {
        progress.finish_with_message("done");
    }
    report_completion(collector, initial_rss);
    Ok(())
}

async fn collect_quota(
    collector: &AlphaZeroCollector,
    progress: Option<&ProgressBar>,
) -> Result<()> {
    loop {
        let completed = collector.episode_count.load(Ordering::Relaxed);
        if completed >= collector.config.max_episodes {
            info!(
                max_episodes = collector.config.max_episodes,
                "Reached episode quota"
            );
            return Ok(());
        }
        if collector.shutdown_signal.load(Ordering::Relaxed) {
            return Err(anyhow!(
                "shutdown interrupted bounded collection after {completed}/{} completed episodes",
                collector.config.max_episodes
            ));
        }
        let episode_start = Instant::now();
        match collector.run_episode().await {
            Ok(EpisodeOutcome::Abandoned {
                reason,
                steps,
                discarded,
                timeout_secs,
            }) => {
                collector.stats.record_abandoned_episode(discarded);
                warn!(
                    reason = reason.as_str(),
                    steps,
                    discarded,
                    timeout_secs,
                    completed_before_abandonment = completed,
                    guidance = reason.guidance(),
                    "Episode abandoned; failing bounded collection"
                );
                return Err(anyhow!(
                    "bounded collection abandoned episode after {steps} steps ({})",
                    reason.as_str()
                ));
            }
            Ok(EpisodeOutcome::Completed {
                steps,
                player_one_outcome,
                stats,
            }) => record_completed(
                collector,
                progress,
                episode_start,
                steps,
                player_one_outcome,
                stats,
            ),
            Err(error) => {
                return Err(anyhow!(
                    "bounded collection episode {} failed: {error}",
                    completed + 1
                ));
            }
        }
    }
}

fn record_completed(
    collector: &AlphaZeroCollector,
    progress: Option<&ProgressBar>,
    episode_start: Instant,
    steps: u32,
    player_one_outcome: f32,
    episode_stats: EpisodeStats,
) {
    let count = collector.episode_count.fetch_add(1, Ordering::Relaxed) + 1;
    let duration = episode_start.elapsed().as_secs_f64();
    debug!(
        episode = count,
        steps, player_one_outcome, duration, "Episode completed"
    );
    collector.stats.record_episode(steps, player_one_outcome);
    collector
        .stats
        .record_mcts_stats(episode_stats.search_count, episode_stats.inference_time_us);
    if let Some(progress) = progress {
        progress.inc(1);
    }
    if collector.config.log_interval > 0 && count.is_multiple_of(collector.config.log_interval) {
        let log_progress = || {
            let rss = rss_mb()
                .map(|value| format!(", RSS: {value:.1} MB"))
                .unwrap_or_default();
            info!("Completed {count} episodes (last: {duration:.2}s{rss})");
            episode_stats.log_summary(count);
        };
        match progress {
            Some(progress) => progress.suspend(log_progress),
            None => log_progress(),
        }
    }
}

fn progress_bar(max_episodes: u32) -> Option<ProgressBar> {
    if !std::io::stderr().is_terminal() {
        return None;
    }
    let progress = ProgressBar::new(u64::from(max_episodes));
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} episodes ({eta})")
            .expect("static progress template is valid")
            .progress_chars("#>-"),
    );
    Some(progress)
}

fn report_completion(collector: &AlphaZeroCollector, initial_rss: f64) {
    let final_rss = rss_mb().unwrap_or(0.0);
    let stats = collector.stats.snapshot();
    info!(
        env_id = %stats.env_id,
        episodes_completed = stats.episodes_completed,
        total_steps = stats.total_steps,
        player1_wins = stats.player1_wins,
        player2_wins = stats.player2_wins,
        draws = stats.draws,
        episodes_abandoned = stats.episodes_abandoned,
        replay_records_discarded = stats.replay_records_discarded,
        avg_episode_length = stats.avg_episode_length,
        episodes_per_second = stats.episodes_per_second,
        runtime_seconds = stats.runtime_seconds,
        mcts_avg_inference_us = stats.mcts_avg_inference_us,
        snapshot_timestamp = stats.timestamp,
        final_rss_mb = format!("{final_rss:.1}"),
        rss_growth_mb = format!("{:.1}", final_rss - initial_rss),
        "Actor completed bounded collection"
    );
}
