//! PostgreSQL implementation of the algorithm-neutral replay envelope.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use deadpool_postgres::{Config, Object, Pool, Runtime};
use std::time::Duration;
use tokio_postgres::types::ToSql;
use tokio_postgres::NoTls;

use super::{ReplayRecord, ReplaySelection, ReplayStore};

mod schema;

const COLS_PER_RECORD: usize = 10;
/// PostgreSQL's extended protocol carries the bind-parameter count as a u16,
/// so one statement can never carry more than 65535 parameters. Long episodes
/// (e.g. Generals) can exceed that as one giant multi-row INSERT, so batches
/// are chunked below the ceiling and committed in a single transaction.
const MAX_PARAMETERS_PER_STATEMENT: usize = u16::MAX as usize;
const MAX_RECORDS_PER_INSERT: usize = MAX_PARAMETERS_PER_STATEMENT / COLS_PER_RECORD;
const SELECTION_PREDICATE: &str = "env_id = $1 AND env_contract_version = $2
     AND algorithm_id = $3 AND experience_schema = $4
     AND collection_scope_id = $5
     AND source_checkpoint_id IS NOT DISTINCT FROM $6";

fn build_batch_insert_sql(batch_size: usize) -> String {
    let mut sql = String::with_capacity(256 + batch_size * 48);
    sql.push_str(
        "INSERT INTO replay_records
         (id, env_id, env_contract_version, algorithm_id, experience_schema,
          collection_scope_id, source_checkpoint_id, episode_id, step_number, payload)
         VALUES ",
    );

    for row in 0..batch_size {
        if row > 0 {
            sql.push_str(", ");
        }
        sql.push('(');
        for column in 0..COLS_PER_RECORD {
            if column > 0 {
                sql.push_str(", ");
            }
            sql.push('$');
            sql.push_str(&(row * COLS_PER_RECORD + column + 1).to_string());
        }
        sql.push(')');
    }
    sql
}

fn selection_query(prefix: &str) -> String {
    format!("{prefix}\nWHERE {SELECTION_PREDICATE}")
}

#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub max_size: usize,
    pub connect_timeout_secs: u64,
    pub idle_timeout_secs: Option<u64>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_size: 16,
            connect_timeout_secs: 30,
            idle_timeout_secs: Some(300),
        }
    }
}

pub struct PostgresReplayStore {
    pool: Pool,
    selection: ReplaySelection,
}

impl PostgresReplayStore {
    #[allow(dead_code)]
    pub async fn new(connection_string: &str, selection: ReplaySelection) -> Result<Self> {
        Self::with_pool_config(connection_string, PoolConfig::default(), selection).await
    }

    pub async fn with_pool_config(
        connection_string: &str,
        pool_config: PoolConfig,
        selection: ReplaySelection,
    ) -> Result<Self> {
        selection.validate()?;
        let pg_config: tokio_postgres::Config = connection_string.parse()?;
        let mut config = Config::new();

        if let Some(host) = pg_config.get_hosts().first() {
            config.host = Some(match host {
                tokio_postgres::config::Host::Tcp(value) => value.clone(),
                #[cfg(unix)]
                tokio_postgres::config::Host::Unix(path) => path.to_string_lossy().into_owned(),
            });
        }
        config.port = pg_config.get_ports().first().copied();
        config.user = pg_config.get_user().map(str::to_string);
        config.password = pg_config
            .get_password()
            .map(|password| String::from_utf8_lossy(password).into_owned());
        config.dbname = pg_config.get_dbname().map(str::to_string);
        config.pool = Some(deadpool_postgres::PoolConfig {
            max_size: pool_config.max_size,
            timeouts: deadpool_postgres::Timeouts {
                wait: Some(Duration::from_secs(pool_config.connect_timeout_secs)),
                create: Some(Duration::from_secs(pool_config.connect_timeout_secs)),
                recycle: pool_config.idle_timeout_secs.map(Duration::from_secs),
            },
            ..Default::default()
        });

        let pool = config.create_pool(Some(Runtime::Tokio1), NoTls)?;
        let store = Self { pool, selection };
        {
            let mut client = store.client().await?;
            schema::ensure_schema(&mut client).await?;
        }
        tracing::info!(
            max_size = pool_config.max_size,
            "PostgreSQL replay pool initialized"
        );
        Ok(store)
    }

    async fn client(&self) -> Result<Object> {
        self.pool
            .get()
            .await
            .context("failed to get PostgreSQL connection from replay pool")
    }
}

#[async_trait]
impl ReplayStore for PostgresReplayStore {
    async fn store(&self, record: &ReplayRecord) -> Result<()> {
        self.store_batch(std::slice::from_ref(record)).await
    }

    async fn store_batch(&self, records: &[ReplayRecord]) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        if let Some(record) = records
            .iter()
            .find(|record| !self.selection.matches(record))
        {
            bail!(
                "replay record '{}' does not match replay selection {:?}",
                record.id,
                self.selection
            );
        }
        if let Some(record) = records
            .iter()
            .find(|record| record.id.is_empty() || record.episode_id.is_empty())
        {
            bail!(
                "replay record id and episode_id must be non-empty (record {:?})",
                record.id
            );
        }

        let step_numbers = records
            .iter()
            .map(|record| i64::from(record.step_number))
            .collect::<Vec<_>>();
        let contract_versions = records
            .iter()
            .map(|record| i64::from(record.env_contract_version))
            .collect::<Vec<_>>();

        // All chunks commit atomically: either the whole batch is stored or
        // none of it is, so a failure can never leave a partial episode.
        let mut client = self.client().await?;
        let transaction = client
            .transaction()
            .await
            .context("failed to start replay insert transaction")?;
        for (chunk_index, chunk) in records.chunks(MAX_RECORDS_PER_INSERT).enumerate() {
            let offset = chunk_index * MAX_RECORDS_PER_INSERT;
            let sql = build_batch_insert_sql(chunk.len());
            let mut parameters: Vec<&(dyn ToSql + Sync)> =
                Vec::with_capacity(chunk.len() * COLS_PER_RECORD);
            for (index, record) in chunk.iter().enumerate() {
                parameters.push(&record.id);
                parameters.push(&record.env_id);
                parameters.push(&contract_versions[offset + index]);
                parameters.push(&record.algorithm_id);
                parameters.push(&record.experience_schema);
                parameters.push(&record.collection_scope_id);
                parameters.push(&record.source_checkpoint_id);
                parameters.push(&record.episode_id);
                parameters.push(&step_numbers[offset + index]);
                parameters.push(&record.payload);
            }
            transaction
                .execute(&sql, &parameters)
                .await
                .with_context(|| {
                    format!(
                        "failed to insert replay records {}..{} of {}",
                        offset,
                        offset + chunk.len(),
                        records.len()
                    )
                })?;
        }
        transaction
            .commit()
            .await
            .with_context(|| format!("failed to commit {} replay records", records.len()))?;
        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let client = self.client().await?;
        let sql = selection_query("SELECT COUNT(*) FROM replay_records");
        let row = client
            .query_one(
                &sql,
                &[
                    &self.selection.profile.env_id,
                    &i64::from(self.selection.profile.env_contract_version),
                    &self.selection.profile.algorithm_id,
                    &self.selection.profile.experience_schema,
                    &self.selection.collection_scope_id,
                    &self.selection.source_checkpoint_id,
                ],
            )
            .await
            .context("failed to count replay records")?;
        let count: i64 = row.get(0);
        usize::try_from(count).context("PostgreSQL returned a negative replay record count")
    }

    async fn count_episodes(&self) -> Result<usize> {
        let client = self.client().await?;
        let sql = selection_query("SELECT COUNT(DISTINCT episode_id) FROM replay_records");
        let row = client
            .query_one(
                &sql,
                &[
                    &self.selection.profile.env_id,
                    &i64::from(self.selection.profile.env_contract_version),
                    &self.selection.profile.algorithm_id,
                    &self.selection.profile.experience_schema,
                    &self.selection.collection_scope_id,
                    &self.selection.source_checkpoint_id,
                ],
            )
            .await
            .context("failed to count replay episodes")?;
        let count: i64 = row.get(0);
        usize::try_from(count).context("PostgreSQL returned a negative replay episode count")
    }

    async fn clear(&self) -> Result<()> {
        let client = self.client().await?;
        let sql = selection_query("DELETE FROM replay_records");
        client
            .execute(
                &sql,
                &[
                    &self.selection.profile.env_id,
                    &i64::from(self.selection.profile.env_contract_version),
                    &self.selection.profile.algorithm_id,
                    &self.selection.profile.experience_schema,
                    &self.selection.collection_scope_id,
                    &self.selection.source_checkpoint_id,
                ],
            )
            .await
            .context("failed to clear replay profile")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_insert_is_immutable_and_uses_ten_columns_per_record() {
        let sql = build_batch_insert_sql(2);
        assert!(sql.contains("INSERT INTO replay_records"));
        assert!(sql.contains("payload"));
        assert!(sql.contains("$1"));
        assert!(sql.contains("collection_scope_id"));
        assert!(sql.contains("source_checkpoint_id"));
        assert!(sql.contains("$20"));
        assert!(!sql.contains("ON CONFLICT"));
        assert!(!sql.contains("UPDATE"));
    }

    #[test]
    fn insert_chunks_stay_below_the_protocol_parameter_ceiling() {
        assert_eq!(MAX_RECORDS_PER_INSERT, 6553);
        assert_eq!(MAX_RECORDS_PER_INSERT * COLS_PER_RECORD, 65530);

        // The largest permitted chunk must end on its exact final placeholder
        // and never reach the u16 bind-parameter limit.
        let sql = build_batch_insert_sql(MAX_RECORDS_PER_INSERT);
        assert!(sql.ends_with(&format!("${})", MAX_RECORDS_PER_INSERT * COLS_PER_RECORD)));
        assert!(!sql.contains("$65536"));
    }

    #[test]
    fn every_read_and_delete_query_uses_the_exact_nullable_selection() {
        for prefix in [
            "SELECT COUNT(*) FROM replay_records",
            "SELECT COUNT(DISTINCT episode_id) FROM replay_records",
            "DELETE FROM replay_records",
        ] {
            let sql = selection_query(prefix);
            assert!(sql.contains("collection_scope_id = $5"));
            assert!(sql.contains("source_checkpoint_id IS NOT DISTINCT FROM $6"));
            assert!(!sql.contains("source_checkpoint_id = $6"));
        }
    }
}
