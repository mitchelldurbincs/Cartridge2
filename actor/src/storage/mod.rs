//! Storage backend for replay buffer (PostgreSQL).
//!
//! This module provides the PostgreSQL storage backend for the replay buffer.
//!
//! # Usage
//!
//! ```rust,ignore
//! use actor::storage::{create_replay_store, ReplayStore};
//!
//! let store = create_replay_store(&config).await?;
//!
//! // Store transitions
//! store.store_batch(&transitions).await?;
//! ```

mod postgres;

pub use postgres::{PoolConfig, PostgresReplayStore};

use anyhow::Result;
use async_trait::async_trait;
use engine_core::GameMetadata;

/// A single transition from one game state to the next
#[derive(Debug, Clone)]
pub struct Transition {
    pub id: String,
    pub env_id: String,
    pub episode_id: String,
    pub step_number: u32,
    pub state: Vec<u8>,
    pub action: Vec<u8>,
    pub next_state: Vec<u8>,
    pub observation: Vec<u8>,
    pub next_observation: Vec<u8>,
    pub reward: f32,
    pub done: bool,
    pub timestamp: u64,
    /// MCTS policy probabilities for training (stored as f32 bytes)
    pub policy_probs: Vec<u8>,
    /// MCTS value estimate from root
    pub mcts_value: f32,
    /// Final game outcome from this player's perspective (+1 win, -1 loss, 0 draw)
    /// This is backfilled after the episode ends
    pub game_outcome: Option<f32>,
}

/// Abstract interface for replay buffer storage.
///
/// Implementations must be thread-safe and support concurrent writes
/// from multiple actor instances.
#[async_trait]
#[allow(dead_code)]
pub trait ReplayStore: Send + Sync {
    /// Store a single transition in the replay buffer
    async fn store(&self, transition: &Transition) -> Result<()>;

    /// Store multiple transitions in a batch (more efficient)
    async fn store_batch(&self, transitions: &[Transition]) -> Result<()>;

    /// Get the total number of transitions in the buffer
    async fn count(&self) -> Result<usize>;

    /// Store or update game metadata (upsert)
    async fn store_metadata(&self, metadata: &GameMetadata) -> Result<()>;

    /// Clear all transitions (preserves metadata)
    async fn clear(&self) -> Result<()>;
}

/// Configuration for creating a replay store.
///
/// Built from [`crate::config::Config`], which resolves the connection URL
/// and pool settings from CLI/env/config.toml.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// PostgreSQL connection string
    pub postgres_url: String,
    /// Connection pool configuration
    pub pool_config: PoolConfig,
}

/// Create a replay store based on configuration
pub async fn create_replay_store(config: &StorageConfig) -> Result<Box<dyn ReplayStore>> {
    let store =
        PostgresReplayStore::with_pool_config(&config.postgres_url, config.pool_config.clone())
            .await?;
    Ok(Box::new(store))
}

/// Mock replay store for testing without database dependencies.
///
/// Stores transitions in memory and tracks all operations for verification.
#[cfg(test)]
pub mod mock {
    use super::*;
    use std::sync::Mutex;

    /// In-memory mock implementation of ReplayStore for testing.
    ///
    /// This mock stores all transitions in memory and provides helper methods
    /// for test assertions. It can also be configured to fail on specific
    /// operations to test error handling.
    #[derive(Debug, Default)]
    pub struct MockReplayStore {
        /// Stored transitions (in order of insertion)
        transitions: Mutex<Vec<Transition>>,
        /// Stored game metadata
        metadata: Mutex<Option<GameMetadata>>,
        /// Flag to simulate store failures
        fail_store: Mutex<bool>,
        /// Flag to simulate count failures
        fail_count: Mutex<bool>,
    }

    impl MockReplayStore {
        /// Create a new empty mock store.
        pub fn new() -> Self {
            Self::default()
        }

        /// Create a mock store that fails on store operations.
        pub fn failing_store() -> Self {
            Self {
                fail_store: Mutex::new(true),
                ..Default::default()
            }
        }

        /// Get all stored transitions.
        pub fn get_transitions(&self) -> Vec<Transition> {
            self.transitions.lock().unwrap().clone()
        }

        /// Get the stored metadata.
        pub fn get_metadata(&self) -> Option<GameMetadata> {
            self.metadata.lock().unwrap().clone()
        }

        /// Set whether store operations should fail.
        #[allow(dead_code)]
        pub fn set_fail_store(&self, fail: bool) {
            *self.fail_store.lock().unwrap() = fail;
        }

        /// Set whether count operations should fail.
        #[allow(dead_code)]
        pub fn set_fail_count(&self, fail: bool) {
            *self.fail_count.lock().unwrap() = fail;
        }

        /// Get transitions for a specific episode.
        pub fn get_episode_transitions(&self, episode_id: &str) -> Vec<Transition> {
            self.transitions
                .lock()
                .unwrap()
                .iter()
                .filter(|t| t.episode_id == episode_id)
                .cloned()
                .collect()
        }

        /// Verify all transitions have game outcomes set.
        #[allow(dead_code)]
        pub fn all_have_outcomes(&self) -> bool {
            self.transitions
                .lock()
                .unwrap()
                .iter()
                .all(|t| t.game_outcome.is_some())
        }

        /// Verify game outcomes are correctly backfilled (alternating signs).
        pub fn verify_outcome_backfill(&self, episode_id: &str) -> bool {
            let transitions = self.get_episode_transitions(episode_id);
            if transitions.is_empty() {
                return true;
            }

            // Get the final outcome (from the perspective of the last mover)
            let last = transitions.last().unwrap();
            let final_outcome = match last.game_outcome {
                Some(o) => o,
                None => return false,
            };

            // Verify each transition has correctly signed outcome
            let total_steps = transitions.len() as u32;
            for t in &transitions {
                let steps_from_end = total_steps.saturating_sub(1).saturating_sub(t.step_number);
                let expected_sign = if steps_from_end % 2 == 0 { 1.0 } else { -1.0 };
                let expected = final_outcome * expected_sign;

                match t.game_outcome {
                    Some(actual) if (actual - expected).abs() < 1e-6 => {}
                    _ => return false,
                }
            }
            true
        }
    }

    #[async_trait]
    impl ReplayStore for MockReplayStore {
        async fn store(&self, transition: &Transition) -> Result<()> {
            if *self.fail_store.lock().unwrap() {
                return Err(anyhow::anyhow!("Mock store failure"));
            }
            self.transitions.lock().unwrap().push(transition.clone());
            Ok(())
        }

        async fn store_batch(&self, transitions: &[Transition]) -> Result<()> {
            if *self.fail_store.lock().unwrap() {
                return Err(anyhow::anyhow!("Mock batch store failure"));
            }
            self.transitions
                .lock()
                .unwrap()
                .extend(transitions.iter().cloned());
            Ok(())
        }

        async fn count(&self) -> Result<usize> {
            if *self.fail_count.lock().unwrap() {
                return Err(anyhow::anyhow!("Mock count failure"));
            }
            Ok(self.transitions.lock().unwrap().len())
        }

        async fn store_metadata(&self, metadata: &GameMetadata) -> Result<()> {
            if *self.fail_store.lock().unwrap() {
                return Err(anyhow::anyhow!("Mock metadata store failure"));
            }
            *self.metadata.lock().unwrap() = Some(metadata.clone());
            Ok(())
        }

        async fn clear(&self) -> Result<()> {
            self.transitions.lock().unwrap().clear();
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn sample_transition(id: &str, episode_id: &str, step: u32) -> Transition {
            Transition {
                id: id.to_string(),
                env_id: "tictactoe".to_string(),
                episode_id: episode_id.to_string(),
                step_number: step,
                state: vec![0u8; 10],
                action: vec![4, 0, 0, 0], // Action 4
                next_state: vec![1u8; 10],
                observation: vec![0u8; 116],
                next_observation: vec![1u8; 116],
                reward: 0.0,
                done: false,
                timestamp: 1234567890,
                policy_probs: vec![0u8; 36], // 9 f32s
                mcts_value: 0.5,
                game_outcome: None,
            }
        }

        #[tokio::test]
        async fn test_mock_store_single() {
            let store = MockReplayStore::new();
            let t = sample_transition("t1", "ep1", 0);

            store.store(&t).await.unwrap();

            assert_eq!(store.count().await.unwrap(), 1);
            let stored = store.get_transitions();
            assert_eq!(stored[0].id, "t1");
        }

        #[tokio::test]
        async fn test_mock_store_batch() {
            let store = MockReplayStore::new();
            let transitions = vec![
                sample_transition("t1", "ep1", 0),
                sample_transition("t2", "ep1", 1),
                sample_transition("t3", "ep1", 2),
            ];

            store.store_batch(&transitions).await.unwrap();

            assert_eq!(store.count().await.unwrap(), 3);
        }

        #[tokio::test]
        async fn test_mock_clear() {
            let store = MockReplayStore::new();
            store
                .store_batch(&[sample_transition("t1", "ep1", 0)])
                .await
                .unwrap();

            store.clear().await.unwrap();

            assert_eq!(store.count().await.unwrap(), 0);
        }

        #[tokio::test]
        async fn test_mock_metadata() {
            let store = MockReplayStore::new();
            let metadata = GameMetadata::new("tictactoe", "Tic-Tac-Toe")
                .with_board(3, 3)
                .with_actions(9)
                .with_observation(29, 18)
                .with_players(2, vec!["X".to_string(), "O".to_string()], vec!['X', 'O']);

            store.store_metadata(&metadata).await.unwrap();

            let stored = store.get_metadata().unwrap();
            assert_eq!(stored.env_id, "tictactoe");
            assert_eq!(stored.num_actions, 9);
        }

        #[tokio::test]
        async fn test_mock_failure_mode() {
            let store = MockReplayStore::failing_store();

            let result = store.store(&sample_transition("t1", "ep1", 0)).await;
            assert!(result.is_err());

            let result = store
                .store_batch(&[sample_transition("t2", "ep1", 1)])
                .await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_get_episode_transitions() {
            let store = MockReplayStore::new();
            store
                .store_batch(&[
                    sample_transition("t1", "ep1", 0),
                    sample_transition("t2", "ep2", 0),
                    sample_transition("t3", "ep1", 1),
                ])
                .await
                .unwrap();

            let ep1 = store.get_episode_transitions("ep1");
            assert_eq!(ep1.len(), 2);

            let ep2 = store.get_episode_transitions("ep2");
            assert_eq!(ep2.len(), 1);
        }

        #[tokio::test]
        async fn test_outcome_backfill_verification() {
            let store = MockReplayStore::new();

            // Create transitions with properly backfilled outcomes
            // 3-step game ending in P1 win (+1)
            // Step 0: P1 moves, outcome should be +1 (0 steps from end, even)
            // Step 1: P2 moves, outcome should be -1 (1 step from end, odd)
            // Step 2: P1 moves, outcome should be +1 (2 steps from end, even) - WINNER
            let mut t0 = sample_transition("t0", "ep1", 0);
            t0.game_outcome = Some(1.0);
            let mut t1 = sample_transition("t1", "ep1", 1);
            t1.game_outcome = Some(-1.0);
            let mut t2 = sample_transition("t2", "ep1", 2);
            t2.game_outcome = Some(1.0);

            store.store_batch(&[t0, t1, t2]).await.unwrap();

            assert!(store.verify_outcome_backfill("ep1"));
        }
    }
}
