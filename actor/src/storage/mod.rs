//! Selection-bound, algorithm-neutral replay storage.
//!
//! PostgreSQL stores an immutable envelope plus opaque experience bytes. The
//! selected algorithm cartridge owns the payload codec named by
//! [`ReplayProfile::experience_schema`]. Storage never interprets observations,
//! actions, rewards, policies, board layouts, or any other algorithm detail.
//! Every operation is fenced to one exact [`ReplaySelection`], so records from
//! stale or concurrent collection attempts are never visible to its learner.

mod postgres;

pub use postgres::{PoolConfig, PostgresReplayStore};

use anyhow::{bail, Result};
use async_trait::async_trait;

const SHA256_HEX_LENGTH: usize = 64;

pub(crate) fn validate_replay_digest(value: &str, field: &str) -> Result<()> {
    if value.len() != SHA256_HEX_LENGTH
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{field} must be exactly 64 lowercase hexadecimal characters");
    }
    Ok(())
}

/// Exact namespace owned by one collector/learner pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayProfile {
    pub env_id: String,
    /// Immutable revision of the environment state/action/observation contract.
    pub env_contract_version: u32,
    pub algorithm_id: String,
    /// Language-neutral payload codec owned by the algorithm cartridge.
    pub experience_schema: String,
}

impl ReplayProfile {
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("env_id", self.env_id.as_str()),
            ("algorithm_id", self.algorithm_id.as_str()),
            ("experience_schema", self.experience_schema.as_str()),
        ] {
            if value.trim().is_empty() {
                bail!("replay profile {name} cannot be empty");
            }
        }
        if self.env_contract_version == 0 {
            bail!("replay profile env_contract_version must be positive");
        }
        Ok(())
    }

    fn matches(&self, record: &ReplayRecord) -> bool {
        record.env_id == self.env_id
            && record.env_contract_version == self.env_contract_version
            && record.algorithm_id == self.algorithm_id
            && record.experience_schema == self.experience_schema
    }
}

/// Exact replay collection selected by one synchronized iteration attempt.
///
/// `collection_scope_id` is a new random SHA-256-shaped identity for every
/// attempt. `source_checkpoint_id` binds the collected experience to the model
/// generation that produced it; it is null only for a run's initial root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplaySelection {
    pub profile: ReplayProfile,
    pub collection_scope_id: String,
    pub source_checkpoint_id: Option<String>,
}

impl ReplaySelection {
    pub fn validate(&self) -> Result<()> {
        self.profile.validate()?;
        validate_replay_digest(&self.collection_scope_id, "collection_scope_id")?;
        if let Some(source_checkpoint_id) = &self.source_checkpoint_id {
            validate_replay_digest(source_checkpoint_id, "source_checkpoint_id")?;
        }
        Ok(())
    }

    fn matches(&self, record: &ReplayRecord) -> bool {
        self.profile.matches(record)
            && record.collection_scope_id == self.collection_scope_id
            && record.source_checkpoint_id == self.source_checkpoint_id
    }

    /// Wrap an algorithm-owned payload in this exact replay selection.
    pub fn record(
        &self,
        id: impl Into<String>,
        episode_id: impl Into<String>,
        step_number: u32,
        payload: Vec<u8>,
    ) -> ReplayRecord {
        ReplayRecord {
            id: id.into(),
            env_id: self.profile.env_id.clone(),
            env_contract_version: self.profile.env_contract_version,
            algorithm_id: self.profile.algorithm_id.clone(),
            experience_schema: self.profile.experience_schema.clone(),
            collection_scope_id: self.collection_scope_id.clone(),
            source_checkpoint_id: self.source_checkpoint_id.clone(),
            episode_id: episode_id.into(),
            step_number,
            payload,
        }
    }
}

/// Immutable replay envelope.
///
/// `payload` is opaque to this module. Its exact meaning is selected by the
/// `(algorithm_id, experience_schema)` pair in the profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayRecord {
    pub id: String,
    pub env_id: String,
    pub env_contract_version: u32,
    pub algorithm_id: String,
    pub experience_schema: String,
    pub collection_scope_id: String,
    pub source_checkpoint_id: Option<String>,
    pub episode_id: String,
    pub step_number: u32,
    pub payload: Vec<u8>,
}

/// Algorithm-neutral replay persistence.
#[async_trait]
#[allow(dead_code)]
pub trait ReplayStore: Send + Sync {
    async fn store(&self, record: &ReplayRecord) -> Result<()>;
    async fn store_batch(&self, records: &[ReplayRecord]) -> Result<()>;
    async fn count(&self) -> Result<usize>;
    async fn count_episodes(&self) -> Result<usize>;

    /// Clear only this store's exact selection; every other scope is kept.
    async fn clear(&self) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub postgres_url: String,
    pub pool_config: PoolConfig,
    pub selection: ReplaySelection,
}

pub async fn create_replay_store(config: &StorageConfig) -> Result<Box<dyn ReplayStore>> {
    config.selection.validate()?;
    let store = PostgresReplayStore::with_pool_config(
        &config.postgres_url,
        config.pool_config.clone(),
        config.selection.clone(),
    )
    .await?;
    Ok(Box::new(store))
}

#[cfg(test)]
pub mod mock {
    use super::*;
    use std::sync::Mutex;

    #[derive(Debug)]
    pub struct MockReplayStore {
        selection: ReplaySelection,
        records: Mutex<Vec<ReplayRecord>>,
        fail_store: Mutex<bool>,
        fail_count: Mutex<bool>,
    }

    impl MockReplayStore {
        pub fn new(selection: ReplaySelection) -> Self {
            selection.validate().expect("valid mock replay selection");
            Self {
                selection,
                records: Mutex::new(Vec::new()),
                fail_store: Mutex::new(false),
                fail_count: Mutex::new(false),
            }
        }

        pub fn failing_store(selection: ReplaySelection) -> Self {
            let store = Self::new(selection);
            *store.fail_store.lock().unwrap() = true;
            store
        }

        pub fn get_records(&self) -> Vec<ReplayRecord> {
            self.records.lock().unwrap().clone()
        }

        #[allow(dead_code)]
        pub fn set_fail_store(&self, fail: bool) {
            *self.fail_store.lock().unwrap() = fail;
        }

        #[allow(dead_code)]
        pub fn set_fail_count(&self, fail: bool) {
            *self.fail_count.lock().unwrap() = fail;
        }

        pub fn get_episode_records(&self, episode_id: &str) -> Vec<ReplayRecord> {
            self.records
                .lock()
                .unwrap()
                .iter()
                .filter(|record| record.episode_id == episode_id)
                .cloned()
                .collect()
        }

        fn require_selection(&self, record: &ReplayRecord) -> Result<()> {
            if !self.selection.matches(record) {
                bail!(
                    "replay record '{}' does not match replay selection {:?}",
                    record.id,
                    self.selection
                );
            }
            Ok(())
        }
    }

    #[async_trait]
    impl ReplayStore for MockReplayStore {
        async fn store(&self, record: &ReplayRecord) -> Result<()> {
            if *self.fail_store.lock().unwrap() {
                bail!("Mock store failure");
            }
            self.require_selection(record)?;
            self.records.lock().unwrap().push(record.clone());
            Ok(())
        }

        async fn store_batch(&self, records: &[ReplayRecord]) -> Result<()> {
            if *self.fail_store.lock().unwrap() {
                bail!("Mock batch store failure");
            }
            for record in records {
                self.require_selection(record)?;
            }
            self.records.lock().unwrap().extend(records.iter().cloned());
            Ok(())
        }

        async fn count(&self) -> Result<usize> {
            if *self.fail_count.lock().unwrap() {
                bail!("Mock count failure");
            }
            Ok(self.records.lock().unwrap().len())
        }

        async fn count_episodes(&self) -> Result<usize> {
            if *self.fail_count.lock().unwrap() {
                bail!("Mock count failure");
            }
            Ok(self
                .records
                .lock()
                .unwrap()
                .iter()
                .map(|record| record.episode_id.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len())
        }

        async fn clear(&self) -> Result<()> {
            self.records.lock().unwrap().clear();
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn profile() -> ReplayProfile {
            ReplayProfile {
                env_id: "tictactoe".into(),
                env_contract_version: 1,
                algorithm_id: algorithm_core::ALPHAZERO_BOARD_V1_ID.into(),
                experience_schema: "alphazero_transition_v1".into(),
            }
        }

        fn selection() -> ReplaySelection {
            ReplaySelection {
                profile: profile(),
                collection_scope_id: "a".repeat(64),
                source_checkpoint_id: Some("b".repeat(64)),
            }
        }

        fn record(id: &str, episode_id: &str, step: u32) -> ReplayRecord {
            selection().record(id, episode_id, step, vec![step as u8])
        }

        #[test]
        fn profile_rejects_empty_or_unversioned_namespaces() {
            let mut value = profile();
            value.algorithm_id.clear();
            assert!(value
                .validate()
                .unwrap_err()
                .to_string()
                .contains("algorithm_id"));

            let mut value = profile();
            value.env_contract_version = 0;
            assert!(value
                .validate()
                .unwrap_err()
                .to_string()
                .contains("must be positive"));
        }

        #[test]
        fn selection_requires_exact_lowercase_digests() {
            assert!(selection().validate().is_ok());

            let mut value = selection();
            value.collection_scope_id = "A".repeat(64);
            assert!(value
                .validate()
                .unwrap_err()
                .to_string()
                .contains("collection_scope_id"));

            let mut value = selection();
            value.source_checkpoint_id = Some("f".repeat(63));
            assert!(value
                .validate()
                .unwrap_err()
                .to_string()
                .contains("source_checkpoint_id"));

            let mut root = selection();
            root.source_checkpoint_id = None;
            assert!(root.validate().is_ok());
        }

        #[tokio::test]
        async fn mock_stores_and_clears_profile_records() {
            let store = MockReplayStore::new(selection());
            store
                .store_batch(&[record("one", "episode", 0), record("two", "episode", 1)])
                .await
                .unwrap();

            assert_eq!(store.count().await.unwrap(), 2);
            assert_eq!(store.count_episodes().await.unwrap(), 1);
            assert_eq!(store.get_episode_records("episode").len(), 2);
            assert_eq!(store.get_records()[1].payload, vec![1]);

            store.clear().await.unwrap();
            assert_eq!(store.count().await.unwrap(), 0);
        }

        #[tokio::test]
        async fn mock_rejects_cross_selection_records() {
            let store = MockReplayStore::new(selection());
            let mut wrong = record("one", "episode", 0);
            wrong.collection_scope_id = "c".repeat(64);

            let error = store.store(&wrong).await.unwrap_err().to_string();
            assert!(error.contains("does not match replay selection"));

            let mut wrong_source = record("two", "episode", 0);
            wrong_source.source_checkpoint_id = None;
            let error = store.store(&wrong_source).await.unwrap_err().to_string();
            assert!(error.contains("does not match replay selection"));
        }

        #[tokio::test]
        async fn mock_failure_is_propagated() {
            let store = MockReplayStore::failing_store(selection());
            assert!(store.store(&record("one", "episode", 0)).await.is_err());
        }
    }
}
