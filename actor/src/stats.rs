//! Actor statistics tracking and persistence.
//!
//! This module provides statistics tracking for the actor, including:
//! - Episode counts and outcomes
//! - MCTS performance metrics
//! - Episode timing information
//!
//! Stats are written to a JSON file for the web frontend to display.

use engine_core::GameOutcome;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, warn};

/// Aggregated actor statistics, designed for lock-free updates.
#[derive(Debug)]
pub struct ActorStats {
    /// Number of episodes completed
    episodes_completed: AtomicU32,
    /// Total game steps across all episodes
    total_steps: AtomicU64,
    /// Episodes won by the seat that moves first
    player1_wins: AtomicU32,
    /// Episodes won by the seat that moves second
    player2_wins: AtomicU32,
    /// Episodes that ended without a winner
    draws: AtomicU32,
    /// Episodes abandoned before reaching a terminal state
    episodes_abandoned: AtomicU32,
    /// Transitions thrown away with those abandoned episodes
    transitions_discarded: AtomicU64,
    /// Start time for rate calculations
    start_time: Instant,
    /// Path to write stats file
    stats_path: String,
    /// Environment ID
    env_id: String,
    /// MCTS stats: total inference time (microseconds)
    mcts_inference_us: AtomicU64,
    /// MCTS stats: total searches performed
    mcts_searches: AtomicU64,
}

/// Serializable stats for JSON output.
#[derive(Debug, Serialize, Deserialize)]
pub struct ActorStatsSnapshot {
    pub env_id: String,
    pub episodes_completed: u32,
    pub total_steps: u64,
    pub player1_wins: u32,
    pub player2_wins: u32,
    pub draws: u32,
    /// Episodes abandoned before a terminal state (timeout or step guard).
    /// Their transitions never reached the replay buffer, so a non-zero
    /// value here means self-play data is being lost — and lost with a bias,
    /// since the episodes that run out of wall clock are the long ones.
    pub episodes_abandoned: u32,
    /// Transitions discarded along with those episodes.
    pub transitions_discarded: u64,
    pub avg_episode_length: f64,
    pub episodes_per_second: f64,
    pub runtime_seconds: f64,
    pub mcts_avg_inference_us: f64,
    pub timestamp: u64,
}

impl ActorStats {
    /// Create new stats tracker.
    pub fn new(data_dir: &str, env_id: &str) -> Self {
        let stats_path = format!("{}/actor_stats.json", data_dir);

        // Ensure data directory exists
        if let Err(e) = fs::create_dir_all(data_dir) {
            warn!("Failed to create data directory: {}", e);
        }

        Self {
            episodes_completed: AtomicU32::new(0),
            total_steps: AtomicU64::new(0),
            player1_wins: AtomicU32::new(0),
            player2_wins: AtomicU32::new(0),
            draws: AtomicU32::new(0),
            episodes_abandoned: AtomicU32::new(0),
            transitions_discarded: AtomicU64::new(0),
            start_time: Instant::now(),
            stats_path,
            env_id: env_id.to_string(),
            mcts_inference_us: AtomicU64::new(0),
            mcts_searches: AtomicU64::new(0),
        }
    }

    /// Record a completed episode.
    ///
    /// Takes a decoded [`GameOutcome`] rather than a reward on purpose. This
    /// used to accept the episode's summed reward and read its sign as a seat,
    /// which cannot work: rewards are relative to the player who just moved
    /// and the winning move is made by the winner, so every decisive game
    /// looked like a player-1 win and `player2_wins` never left zero.
    pub fn record_episode(&self, steps: u32, outcome: GameOutcome) {
        self.episodes_completed.fetch_add(1, Ordering::Relaxed);
        self.total_steps.fetch_add(steps as u64, Ordering::Relaxed);

        let counter = match outcome {
            GameOutcome::Player1Win => &self.player1_wins,
            GameOutcome::Player2Win => &self.player2_wins,
            GameOutcome::Draw => &self.draws,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an episode that was abandoned before reaching a terminal
    /// state, along with the transitions discarded with it.
    ///
    /// Returns the new abandoned-episode total so the caller can report the
    /// running rate — a single dropped episode is noise, a steady stream is
    /// a silently shrinking (and length-biased) replay buffer.
    pub fn record_abandoned_episode(&self, discarded: usize) -> u32 {
        self.transitions_discarded
            .fetch_add(discarded as u64, Ordering::Relaxed);
        self.episodes_abandoned.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Record MCTS performance for an episode.
    pub fn record_mcts_stats(&self, searches: u32, inference_us: u64) {
        self.mcts_searches
            .fetch_add(searches as u64, Ordering::Relaxed);
        self.mcts_inference_us
            .fetch_add(inference_us, Ordering::Relaxed);
    }

    /// Get a snapshot of current stats.
    pub fn snapshot(&self) -> ActorStatsSnapshot {
        let episodes = self.episodes_completed.load(Ordering::Relaxed);
        let total_steps = self.total_steps.load(Ordering::Relaxed);
        let runtime = self.start_time.elapsed().as_secs_f64();
        let searches = self.mcts_searches.load(Ordering::Relaxed);
        let inference_us = self.mcts_inference_us.load(Ordering::Relaxed);

        let avg_episode_length = if episodes > 0 {
            total_steps as f64 / episodes as f64
        } else {
            0.0
        };

        let episodes_per_second = if runtime > 0.0 {
            episodes as f64 / runtime
        } else {
            0.0
        };

        let mcts_avg_inference_us = if searches > 0 {
            inference_us as f64 / searches as f64
        } else {
            0.0
        };

        ActorStatsSnapshot {
            env_id: self.env_id.clone(),
            episodes_completed: episodes,
            total_steps,
            player1_wins: self.player1_wins.load(Ordering::Relaxed),
            player2_wins: self.player2_wins.load(Ordering::Relaxed),
            draws: self.draws.load(Ordering::Relaxed),
            episodes_abandoned: self.episodes_abandoned.load(Ordering::Relaxed),
            transitions_discarded: self.transitions_discarded.load(Ordering::Relaxed),
            avg_episode_length,
            episodes_per_second,
            runtime_seconds: runtime,
            mcts_avg_inference_us,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Write stats to JSON file (atomic write-then-rename).
    pub fn write_stats(&self) {
        let snapshot = self.snapshot();

        // Serialize to JSON
        let json = match serde_json::to_string_pretty(&snapshot) {
            Ok(j) => j,
            Err(e) => {
                warn!("Failed to serialize actor stats: {}", e);
                return;
            }
        };

        // Write to temp file then rename (atomic on most filesystems)
        let temp_path = format!("{}.tmp", self.stats_path);
        match fs::File::create(&temp_path) {
            Ok(mut file) => {
                if let Err(e) = file.write_all(json.as_bytes()) {
                    warn!("Failed to write actor stats: {}", e);
                    return;
                }
            }
            Err(e) => {
                warn!("Failed to create temp stats file: {}", e);
                return;
            }
        }

        if let Err(e) = fs::rename(&temp_path, &self.stats_path) {
            warn!("Failed to rename stats file: {}", e);
            // Try to clean up temp file
            let _ = fs::remove_file(&temp_path);
            return;
        }

        debug!("Wrote actor stats to {}", self.stats_path);
    }

    /// Path of the JSON stats file this tracker writes to.
    pub fn stats_path(&self) -> &str {
        &self.stats_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn test_record_abandoned_episode_accumulates() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "generals_8x8");

        assert_eq!(stats.record_abandoned_episode(120), 1);
        assert_eq!(stats.record_abandoned_episode(87), 2);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.episodes_abandoned, 2);
        assert_eq!(snapshot.transitions_discarded, 207);

        // Abandoned episodes are not completed episodes: they must not
        // inflate throughput or outcome counts.
        assert_eq!(snapshot.episodes_completed, 0);
        assert_eq!(snapshot.total_steps, 0);
        assert_eq!(snapshot.player1_wins, 0);
        assert_eq!(snapshot.player2_wins, 0);
        assert_eq!(snapshot.draws, 0);
    }

    #[test]
    fn test_record_episode() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        // Record some episodes
        stats.record_episode(9, GameOutcome::Player1Win);
        stats.record_episode(8, GameOutcome::Player2Win);
        stats.record_episode(9, GameOutcome::Draw);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.episodes_completed, 3);
        assert_eq!(snapshot.player1_wins, 1);
        assert_eq!(snapshot.player2_wins, 1);
        assert_eq!(snapshot.draws, 1);
        assert_eq!(snapshot.total_steps, 26);
    }

    #[test]
    fn test_write_stats() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        stats.record_episode(9, GameOutcome::Player1Win);
        stats.write_stats();

        // Verify file exists and is valid JSON
        let path = Path::new(stats.stats_path());
        assert!(path.exists());

        let content = fs::read_to_string(path).unwrap();
        let parsed: ActorStatsSnapshot = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.episodes_completed, 1);
    }

    // ========================================
    // Edge case tests
    // ========================================

    #[test]
    fn test_average_with_zero_episodes() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        // Don't record any episodes
        let snapshot = stats.snapshot();

        // Averages should be 0.0, not NaN or panic
        assert_eq!(snapshot.episodes_completed, 0);
        assert_eq!(snapshot.avg_episode_length, 0.0);
        assert!(!snapshot.avg_episode_length.is_nan());
    }

    #[test]
    fn test_mcts_average_with_zero_searches() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        // Record episode but no MCTS stats
        stats.record_episode(5, GameOutcome::Player1Win);

        let snapshot = stats.snapshot();

        // MCTS average should be 0.0 when no searches recorded
        assert_eq!(snapshot.mcts_avg_inference_us, 0.0);
        assert!(!snapshot.mcts_avg_inference_us.is_nan());
    }

    /// Each outcome increments exactly one counter, and only that one.
    ///
    /// Replaces three tests that asserted the sign of a reward selected the
    /// seat ("any positive reward is a P1 win"). That mapping was the bug:
    /// the terminal reward is relative to the mover, so it is `+1.0` for
    /// either seat's win and `player2_wins` could never be non-zero.
    #[test]
    fn test_each_outcome_increments_only_its_own_counter() {
        let cases = [
            (GameOutcome::Player1Win, (1, 0, 0)),
            (GameOutcome::Player2Win, (0, 1, 0)),
            (GameOutcome::Draw, (0, 0, 1)),
        ];

        for (outcome, (p1, p2, draws)) in cases {
            let dir = tempdir().unwrap();
            let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

            stats.record_episode(5, outcome);

            let snapshot = stats.snapshot();
            assert_eq!(snapshot.player1_wins, p1, "p1 count for {outcome:?}");
            assert_eq!(snapshot.player2_wins, p2, "p2 count for {outcome:?}");
            assert_eq!(snapshot.draws, draws, "draw count for {outcome:?}");
            assert_eq!(snapshot.episodes_completed, 1);
        }
    }

    /// Regression test for the reason this fix exists.
    ///
    /// A run in which player 2 wins every game must report exactly that. The
    /// old reward-sign mapping reported it as 100% player-1 wins, which is
    /// also what the frontend's outcome bar rendered — so a seat-imbalanced
    /// run, the failure this metric exists to catch, was indistinguishable
    /// from a healthy one.
    #[test]
    fn test_a_run_of_player2_wins_is_not_reported_as_player1() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        for _ in 0..10 {
            stats.record_episode(9, GameOutcome::Player2Win);
        }

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.player2_wins, 10);
        assert_eq!(snapshot.player1_wins, 0);
        assert_eq!(snapshot.draws, 0);
    }

    #[test]
    fn test_mcts_stats_accumulation() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        // Record multiple MCTS stats
        stats.record_mcts_stats(10, 1000); // 10 searches, 1000us
        stats.record_mcts_stats(20, 3000); // 20 searches, 3000us

        let snapshot = stats.snapshot();

        // Total: 30 searches, 4000us -> avg = 4000/30 ≈ 133.33
        let expected_avg = 4000.0 / 30.0;
        assert!((snapshot.mcts_avg_inference_us - expected_avg).abs() < 0.1);
    }

    #[test]
    fn test_stats_path_format() {
        let dir = tempdir().unwrap();
        let dir_path = dir.path().to_str().unwrap();
        let stats = ActorStats::new(dir_path, "tictactoe");

        let expected = format!("{}/actor_stats.json", dir_path);
        assert_eq!(stats.stats_path(), expected);
    }

    #[test]
    fn test_env_id_preserved() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "connect4");

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.env_id, "connect4");
    }

    #[test]
    fn test_episodes_per_second_calculation() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        // Record episodes (note: there will be some real elapsed time)
        stats.record_episode(9, GameOutcome::Player1Win);
        stats.record_episode(9, GameOutcome::Player1Win);

        let snapshot = stats.snapshot();

        // Should have positive runtime and rate
        assert!(snapshot.runtime_seconds > 0.0);
        assert!(snapshot.episodes_per_second > 0.0);

        // Rate should be episodes / runtime
        let expected_rate = 2.0 / snapshot.runtime_seconds;
        assert!((snapshot.episodes_per_second - expected_rate).abs() < 0.1);
    }

    #[test]
    fn test_total_steps_accumulation() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        stats.record_episode(5, GameOutcome::Player1Win);
        stats.record_episode(9, GameOutcome::Player2Win);
        stats.record_episode(7, GameOutcome::Draw);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_steps, 5 + 9 + 7);
    }

    #[test]
    fn test_avg_episode_length_calculation() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        stats.record_episode(6, GameOutcome::Player1Win);
        stats.record_episode(10, GameOutcome::Player2Win);
        stats.record_episode(8, GameOutcome::Draw);

        let snapshot = stats.snapshot();
        // Average: (6 + 10 + 8) / 3 = 8.0
        assert!((snapshot.avg_episode_length - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_timestamp_is_recent() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");
        stats.record_episode(5, GameOutcome::Player1Win);

        let snapshot = stats.snapshot();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Timestamp should be within 1 second of now
        assert!(snapshot.timestamp >= now - 1);
        assert!(snapshot.timestamp <= now + 1);
    }

    #[test]
    fn test_snapshot_serialization_roundtrip() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        stats.record_episode(9, GameOutcome::Player1Win);
        stats.record_mcts_stats(100, 50000);

        let snapshot = stats.snapshot();

        // Serialize and deserialize
        let json = serde_json::to_string(&snapshot).unwrap();
        let parsed: ActorStatsSnapshot = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.env_id, snapshot.env_id);
        assert_eq!(parsed.episodes_completed, snapshot.episodes_completed);
        assert_eq!(parsed.player1_wins, snapshot.player1_wins);
        assert!((parsed.mcts_avg_inference_us - snapshot.mcts_avg_inference_us).abs() < 0.1);
    }

    #[test]
    fn test_write_stats_atomic() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        // Write initial stats
        stats.record_episode(5, GameOutcome::Player1Win);
        stats.write_stats();

        // Read and verify
        let path = Path::new(stats.stats_path());
        let content1 = fs::read_to_string(path).unwrap();
        let parsed1: ActorStatsSnapshot = serde_json::from_str(&content1).unwrap();
        assert_eq!(parsed1.episodes_completed, 1);

        // Update and write again
        stats.record_episode(7, GameOutcome::Player2Win);
        stats.write_stats();

        // Should see updated value
        let content2 = fs::read_to_string(path).unwrap();
        let parsed2: ActorStatsSnapshot = serde_json::from_str(&content2).unwrap();
        assert_eq!(parsed2.episodes_completed, 2);
    }

    #[test]
    fn test_concurrent_updates() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempdir().unwrap();
        let stats = Arc::new(ActorStats::new(dir.path().to_str().unwrap(), "tictactoe"));

        // Spawn multiple threads recording episodes concurrently
        let mut handles = vec![];
        for _ in 0..10 {
            let stats_clone = Arc::clone(&stats);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    stats_clone.record_episode(5, GameOutcome::Player1Win);
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        let snapshot = stats.snapshot();
        // Should have 10 threads * 100 episodes = 1000 episodes
        assert_eq!(snapshot.episodes_completed, 1000);
    }
}
