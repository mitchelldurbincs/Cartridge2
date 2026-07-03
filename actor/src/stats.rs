//! Actor statistics tracking and persistence.
//!
//! This module provides statistics tracking for the actor, including:
//! - Episode counts and outcomes
//! - MCTS performance metrics
//! - Episode timing information
//!
//! Stats are written to a JSON file for the web frontend to display.

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
    /// Episodes that ended in player 1 win (reward > 0)
    player1_wins: AtomicU32,
    /// Episodes that ended in player 2 win (reward < 0)
    player2_wins: AtomicU32,
    /// Episodes that ended in draw (reward == 0)
    draws: AtomicU32,
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
            start_time: Instant::now(),
            stats_path,
            env_id: env_id.to_string(),
            mcts_inference_us: AtomicU64::new(0),
            mcts_searches: AtomicU64::new(0),
        }
    }

    /// Record a completed episode.
    pub fn record_episode(&self, steps: u32, final_reward: f32) {
        self.episodes_completed.fetch_add(1, Ordering::Relaxed);
        self.total_steps.fetch_add(steps as u64, Ordering::Relaxed);

        // Categorize outcome based on final reward
        // Positive = player 1 wins, negative = player 2 wins, zero = draw
        if final_reward > 0.0 {
            self.player1_wins.fetch_add(1, Ordering::Relaxed);
        } else if final_reward < 0.0 {
            self.player2_wins.fetch_add(1, Ordering::Relaxed);
        } else {
            self.draws.fetch_add(1, Ordering::Relaxed);
        }
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
    fn test_record_episode() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        // Record some episodes
        stats.record_episode(9, 1.0); // P1 win
        stats.record_episode(8, -1.0); // P2 win
        stats.record_episode(9, 0.0); // Draw

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

        stats.record_episode(9, 1.0);
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
        stats.record_episode(5, 1.0);

        let snapshot = stats.snapshot();

        // MCTS average should be 0.0 when no searches recorded
        assert_eq!(snapshot.mcts_avg_inference_us, 0.0);
        assert!(!snapshot.mcts_avg_inference_us.is_nan());
    }

    #[test]
    fn test_outcome_categorization_positive_reward() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        // Any positive reward is a P1 win
        stats.record_episode(5, 0.001); // Small positive
        stats.record_episode(5, 1.0); // Normal win
        stats.record_episode(5, 100.0); // Large positive

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.player1_wins, 3);
        assert_eq!(snapshot.player2_wins, 0);
        assert_eq!(snapshot.draws, 0);
    }

    #[test]
    fn test_outcome_categorization_negative_reward() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        // Any negative reward is a P2 win
        stats.record_episode(5, -0.001); // Small negative
        stats.record_episode(5, -1.0); // Normal loss
        stats.record_episode(5, -100.0); // Large negative

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.player1_wins, 0);
        assert_eq!(snapshot.player2_wins, 3);
        assert_eq!(snapshot.draws, 0);
    }

    #[test]
    fn test_outcome_categorization_zero_reward() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        // Exactly zero is a draw
        stats.record_episode(5, 0.0);
        stats.record_episode(7, 0.0);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.player1_wins, 0);
        assert_eq!(snapshot.player2_wins, 0);
        assert_eq!(snapshot.draws, 2);
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
        stats.record_episode(9, 1.0);
        stats.record_episode(9, 1.0);

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

        stats.record_episode(5, 1.0);
        stats.record_episode(9, -1.0);
        stats.record_episode(7, 0.0);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_steps, 5 + 9 + 7);
    }

    #[test]
    fn test_avg_episode_length_calculation() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");

        stats.record_episode(6, 1.0);
        stats.record_episode(10, -1.0);
        stats.record_episode(8, 0.0);

        let snapshot = stats.snapshot();
        // Average: (6 + 10 + 8) / 3 = 8.0
        assert!((snapshot.avg_episode_length - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_timestamp_is_recent() {
        let dir = tempdir().unwrap();
        let stats = ActorStats::new(dir.path().to_str().unwrap(), "tictactoe");
        stats.record_episode(5, 1.0);

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

        stats.record_episode(9, 1.0);
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
        stats.record_episode(5, 1.0);
        stats.write_stats();

        // Read and verify
        let path = Path::new(stats.stats_path());
        let content1 = fs::read_to_string(path).unwrap();
        let parsed1: ActorStatsSnapshot = serde_json::from_str(&content1).unwrap();
        assert_eq!(parsed1.episodes_completed, 1);

        // Update and write again
        stats.record_episode(7, -1.0);
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
                    stats_clone.record_episode(5, 1.0);
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
