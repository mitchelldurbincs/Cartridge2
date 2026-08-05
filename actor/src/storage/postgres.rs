//! PostgreSQL implementation of the algorithm-neutral replay envelope.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use deadpool_postgres::{Manager, ManagerConfig, Object, Pool, RecyclingMethod, Runtime};
use std::time::Duration;
use tokio_postgres::config::SslMode;
use tokio_postgres::types::ToSql;
use tokio_postgres::{NoTls, Transaction};

use super::{ReplayRecord, ReplaySelection, ReplayStore};

const SCHEMA_SQL: &str = include_str!("../../../sql/schema.sql");
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

const REPLAY_SCHEMA_VERSION: i32 = 3;
const REPLAY_SCHEMA_TABLES: &[&str] = &["cartridge_schema_versions", "replay_records"];
const EXPECTED_SCHEMA_MARKER_COLUMNS: &[(&str, &str, &str)] = &[
    ("component", "text", "NO"),
    ("schema_version", "integer", "NO"),
];
const EXPECTED_SCHEMA_MARKER_PRIMARY_KEY: &[&str] = &["component"];
const EXPECTED_RECORD_COLUMNS: &[(&str, &str, &str)] = &[
    ("id", "text", "NO"),
    ("env_id", "text", "NO"),
    ("env_contract_version", "bigint", "NO"),
    ("algorithm_id", "text", "NO"),
    ("experience_schema", "text", "NO"),
    ("collection_scope_id", "text", "NO"),
    ("source_checkpoint_id", "text", "YES"),
    ("episode_id", "text", "NO"),
    ("step_number", "bigint", "NO"),
    ("payload", "bytea", "NO"),
    ("created_at", "timestamp without time zone", "NO"),
];
const EXPECTED_RECORD_PRIMARY_KEY: &[&str] = &[
    "env_id",
    "env_contract_version",
    "algorithm_id",
    "experience_schema",
    "collection_scope_id",
    "id",
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct ColumnSpec {
    name: String,
    data_type: String,
    is_nullable: String,
}

impl ColumnSpec {
    fn from_row(row: tokio_postgres::Row) -> Self {
        Self {
            name: row.get(0),
            data_type: row.get(1),
            is_nullable: row.get(2),
        }
    }

    #[cfg(test)]
    fn new(name: &str, data_type: &str, is_nullable: &str) -> Self {
        Self {
            name: name.into(),
            data_type: data_type.into(),
            is_nullable: is_nullable.into(),
        }
    }
}

fn split_sql_statements(sql: &str) -> Vec<String> {
    sql.lines()
        .filter(|line| !line.trim_start().starts_with("--"))
        .collect::<Vec<_>>()
        .join("\n")
        .split(';')
        .map(str::trim)
        .filter(|statement| !statement.is_empty())
        .map(str::to_owned)
        .collect()
}

fn format_columns(columns: &[ColumnSpec]) -> String {
    columns
        .iter()
        .map(|column| {
            format!(
                "{} {} {}",
                column.name,
                column.data_type,
                if column.is_nullable == "YES" {
                    "NULL"
                } else {
                    "NOT NULL"
                }
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn expected_columns(columns: &[(&str, &str, &str)]) -> Vec<ColumnSpec> {
    columns
        .iter()
        .map(|(name, data_type, is_nullable)| ColumnSpec {
            name: (*name).into(),
            data_type: (*data_type).into(),
            is_nullable: (*is_nullable).into(),
        })
        .collect()
}

fn validate_schema_tables(tables: &[String]) -> Result<()> {
    let mut actual = tables.to_vec();
    actual.sort();
    let mut expected = REPLAY_SCHEMA_TABLES
        .iter()
        .map(|table| (*table).to_string())
        .collect::<Vec<_>>();
    expected.sort();
    if actual != expected {
        bail!(
            "unsupported replay schema tables {actual:?}; expected {expected:?}; recreate the database from sql/schema.sql"
        );
    }
    Ok(())
}

fn validate_table_schema(
    table: &str,
    columns: &[ColumnSpec],
    primary_key: &[String],
    expected_column_specs: &[(&str, &str, &str)],
    expected_primary_key: &[&str],
) -> Result<()> {
    let expected = expected_columns(expected_column_specs);
    if columns != expected {
        bail!(
            "unsupported replay schema: {table} columns are ({}), expected ({}); recreate the database from sql/schema.sql",
            format_columns(columns),
            format_columns(&expected)
        );
    }

    let expected_primary_key = expected_primary_key
        .iter()
        .map(|column| (*column).to_string())
        .collect::<Vec<_>>();
    if primary_key != expected_primary_key {
        bail!(
            "unsupported replay schema: {table} primary key is ({}) but must be ({}); recreate the database from sql/schema.sql",
            primary_key.join(", "),
            expected_primary_key.join(", ")
        );
    }
    Ok(())
}

fn validate_schema_marker(
    columns: &[ColumnSpec],
    primary_key: &[String],
    rows: &[(String, i32)],
) -> Result<()> {
    validate_table_schema(
        "cartridge_schema_versions",
        columns,
        primary_key,
        EXPECTED_SCHEMA_MARKER_COLUMNS,
        EXPECTED_SCHEMA_MARKER_PRIMARY_KEY,
    )?;
    match rows {
        [(component, REPLAY_SCHEMA_VERSION)] if component == "replay" => Ok(()),
        _ => bail!(
            "unsupported replay schema marker rows {rows:?}; expected [(\"replay\", {REPLAY_SCHEMA_VERSION})]; recreate the database from sql/schema.sql"
        ),
    }
}

fn validate_record_schema(columns: &[ColumnSpec], primary_key: &[String]) -> Result<()> {
    validate_table_schema(
        "replay_records",
        columns,
        primary_key,
        EXPECTED_RECORD_COLUMNS,
        EXPECTED_RECORD_PRIMARY_KEY,
    )
}

async fn table_columns(transaction: &Transaction<'_>, table: &str) -> Result<Vec<ColumnSpec>> {
    Ok(transaction
        .query(
            "SELECT column_name, data_type, is_nullable
             FROM information_schema.columns
             WHERE table_schema = current_schema() AND table_name = $1
             ORDER BY ordinal_position",
            &[&table],
        )
        .await
        .with_context(|| format!("failed to inspect {table} columns"))?
        .into_iter()
        .map(ColumnSpec::from_row)
        .collect())
}

async fn table_primary_key(transaction: &Transaction<'_>, table: &str) -> Result<Vec<String>> {
    Ok(transaction
        .query(
            "SELECT kcu.column_name
             FROM information_schema.table_constraints AS tc
             JOIN information_schema.key_column_usage AS kcu
               ON tc.constraint_name = kcu.constraint_name
              AND tc.constraint_schema = kcu.constraint_schema
             WHERE tc.table_schema = current_schema()
               AND tc.table_name = $1
               AND tc.constraint_type = 'PRIMARY KEY'
             ORDER BY kcu.ordinal_position",
            &[&table],
        )
        .await
        .with_context(|| format!("failed to inspect {table} primary key"))?
        .into_iter()
        .map(|row| row.get(0))
        .collect())
}

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
    // Records are immutable and keyed by content, so re-inserting the same
    // rows must be a no-op: store_batch retries after ambiguous outcomes
    // (e.g. a connection lost mid-commit), and a restarted worker may replay
    // an already-persisted episode. DO NOTHING can never overwrite a record.
    sql.push_str(
        " ON CONFLICT (env_id, env_contract_version, algorithm_id, experience_schema, \
         collection_scope_id, id) DO NOTHING",
    );
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

/// Transient failures worth retrying: the connection died, the server is
/// restarting/failing over, or the transaction lost a concurrency race.
/// Validation and constraint errors are never transient and fail immediately.
fn is_transient_storage_error(error: &anyhow::Error) -> bool {
    for cause in error.chain() {
        if cause.downcast_ref::<std::io::Error>().is_some() {
            return true;
        }
        if let Some(pool_error) = cause.downcast_ref::<deadpool_postgres::PoolError>() {
            if matches!(pool_error, deadpool_postgres::PoolError::Timeout(_)) {
                return true;
            }
        }
        if let Some(pg_error) = cause.downcast_ref::<tokio_postgres::Error>() {
            if pg_error.is_closed() {
                return true;
            }
            if let Some(state) = pg_error.code() {
                let code = state.code();
                // 08xxx connection exceptions; 57P01-57P03 shutdown/failover;
                // 40001/40P01 serialization failure and deadlock.
                if code.starts_with("08")
                    || matches!(code, "57P01" | "57P02" | "57P03" | "40001" | "40P01")
                {
                    return true;
                }
            }
        }
    }
    false
}

const STORE_RETRY_ATTEMPTS: u32 = 4;

fn store_retry_delay(attempt: u32) -> Duration {
    let base_ms = 100u64.saturating_mul(1 << attempt);
    let jitter_ms = rand::random::<u64>() % (base_ms / 2 + 1);
    Duration::from_millis(base_ms + jitter_ms)
}

/// Build a TLS connector that trusts the host's native root certificates.
/// Hostname verification stays on for every TLS mode (stricter than libpq's
/// `require`), so a connection that negotiates TLS is always authenticated.
fn rustls_connector() -> Result<tokio_postgres_rustls::MakeRustlsConnect> {
    let loaded = rustls_native_certs::load_native_certs();
    for error in &loaded.errors {
        tracing::warn!(%error, "failed to load a native root certificate");
    }
    if loaded.certs.is_empty() {
        bail!("no native root certificates available for PostgreSQL TLS");
    }
    let mut roots = rustls::RootCertStore::empty();
    for cert in loaded.certs {
        roots
            .add(cert)
            .context("failed to add a native root certificate")?;
    }
    let config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(tokio_postgres_rustls::MakeRustlsConnect::new(config))
}

pub struct PostgresReplayStore {
    pool: Pool,
    selection: ReplaySelection,
    idle_timeout: Option<Duration>,
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
        // The fully parsed configuration flows into the pool manager, so every
        // DSN parameter — sslmode, application_name, options, additional
        // hosts — takes effect instead of being silently dropped.
        let mut pg_config: tokio_postgres::Config = connection_string.parse()?;
        if pg_config.get_connect_timeout().is_none() {
            pg_config.connect_timeout(Duration::from_secs(pool_config.connect_timeout_secs));
        }
        let manager_config = ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        };
        let manager = match pg_config.get_ssl_mode() {
            SslMode::Disable => Manager::from_config(pg_config, NoTls, manager_config),
            _ => Manager::from_config(pg_config, rustls_connector()?, manager_config),
        };
        let connect_timeout = Duration::from_secs(pool_config.connect_timeout_secs);
        let pool = Pool::builder(manager)
            .max_size(pool_config.max_size)
            .wait_timeout(Some(connect_timeout))
            .create_timeout(Some(connect_timeout))
            .runtime(Runtime::Tokio1)
            .build()
            .context("failed to build PostgreSQL replay pool")?;
        let store = Self {
            pool,
            selection,
            idle_timeout: pool_config.idle_timeout_secs.map(Duration::from_secs),
        };
        store.ensure_schema().await?;
        tracing::info!(
            max_size = pool_config.max_size,
            "PostgreSQL replay pool initialized"
        );
        Ok(store)
    }

    async fn client(&self) -> Result<Object> {
        if let Some(idle) = self.idle_timeout {
            // deadpool never retires idle connections on its own (its
            // `recycle` timeout bounds the recycle *operation*, not idle
            // lifetime), so approximate the configured idle timeout by
            // dropping stale connections whenever the pool is next used.
            self.pool.retain(|_, metrics| metrics.last_used() < idle);
        }
        self.pool
            .get()
            .await
            .context("failed to get PostgreSQL connection from replay pool")
    }

    async fn ensure_schema(&self) -> Result<()> {
        let mut client = self.client().await?;
        let transaction = client
            .transaction()
            .await
            .context("failed to start replay schema validation transaction")?;
        transaction
            .query_one("SELECT pg_advisory_xact_lock(745472510202)", &[])
            .await
            .context("failed to lock replay schema validation")?;

        let existing_tables = transaction
            .query(
                "SELECT table_name
                 FROM information_schema.tables
                 WHERE table_schema = current_schema() AND table_type = 'BASE TABLE'",
                &[],
            )
            .await
            .context("failed to inspect replay schema tables")?
            .into_iter()
            .map(|row| row.get::<_, String>(0))
            .collect::<Vec<_>>();

        if existing_tables.is_empty() {
            for statement in split_sql_statements(SCHEMA_SQL) {
                transaction
                    .execute(&statement as &str, &[])
                    .await
                    .with_context(|| {
                        format!(
                            "failed to execute schema statement starting with {:?}",
                            statement.lines().next().unwrap_or("")
                        )
                    })?;
            }
        } else {
            validate_schema_tables(&existing_tables)?;
        }

        let marker_columns = table_columns(&transaction, "cartridge_schema_versions").await?;
        let marker_primary_key =
            table_primary_key(&transaction, "cartridge_schema_versions").await?;
        let marker_rows = transaction
            .query(
                "SELECT component, schema_version
                 FROM cartridge_schema_versions ORDER BY component",
                &[],
            )
            .await
            .context("failed to inspect replay schema marker rows")?
            .into_iter()
            .map(|row| (row.get(0), row.get(1)))
            .collect::<Vec<_>>();
        validate_schema_marker(&marker_columns, &marker_primary_key, &marker_rows)?;

        let record_columns = table_columns(&transaction, "replay_records").await?;
        let record_primary_key = table_primary_key(&transaction, "replay_records").await?;
        validate_record_schema(&record_columns, &record_primary_key)?;

        transaction
            .commit()
            .await
            .context("failed to commit replay schema validation")?;
        tracing::info!(
            schema_version = REPLAY_SCHEMA_VERSION,
            "PostgreSQL replay schema validated"
        );
        Ok(())
    }

    /// One attempt at storing a batch. All chunks commit atomically: either
    /// the whole batch is stored or none of it is, so a failure can never
    /// leave a partial episode.
    async fn try_store_batch(
        &self,
        records: &[ReplayRecord],
        contract_versions: &[i64],
        step_numbers: &[i64],
    ) -> Result<()> {
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

        // Transient failures are retried with bounded backoff. That is safe
        // because the insert is idempotent (`ON CONFLICT ... DO NOTHING` over
        // immutable content-keyed rows), so retrying after an ambiguous
        // outcome can never duplicate or overwrite a record.
        let mut attempt = 0;
        loop {
            match self
                .try_store_batch(records, &contract_versions, &step_numbers)
                .await
            {
                Ok(()) => return Ok(()),
                Err(error)
                    if attempt + 1 < STORE_RETRY_ATTEMPTS && is_transient_storage_error(&error) =>
                {
                    let delay = store_retry_delay(attempt);
                    tracing::warn!(
                        attempt = attempt + 1,
                        delay_ms = delay.as_millis() as u64,
                        error = %error,
                        "transient replay store failure; retrying"
                    );
                    tokio::time::sleep(delay).await;
                    attempt += 1;
                }
                Err(error) => return Err(error),
            }
        }
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

    const CONTAINER_SCHEMA_SQL: &str = include_str!("../../../scripts/init-postgres.sql");
    const PYTHON_SCHEMA_SQL: &str = include_str!("../../../trainer/src/trainer/storage/schema.sql");
    const K8S_SCHEMA_CONFIG: &str = include_str!("../../../k8s/base/postgres/init-configmap.yaml");

    fn k8s_schema_sql() -> String {
        let (_, body) = K8S_SCHEMA_CONFIG
            .split_once("  01-schema.sql: |\n")
            .expect("Kubernetes schema block");
        let mut schema = body
            .lines()
            .map(|line| line.strip_prefix("    ").unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n");
        schema.push('\n');
        schema
    }

    fn column_specs(columns: &[(&str, &str, &str)]) -> Vec<ColumnSpec> {
        columns
            .iter()
            .map(|(name, data_type, nullable)| ColumnSpec::new(name, data_type, nullable))
            .collect()
    }

    #[test]
    fn sql_splitter_strips_comments_and_empty_statements() {
        let statements = split_sql_statements(
            "-- comment\nCREATE TABLE a (x INT); ;\n-- second\nCREATE INDEX i ON a(x);",
        );
        assert_eq!(statements.len(), 2);
        assert!(statements[0].starts_with("CREATE TABLE a"));
        assert!(statements[1].starts_with("CREATE INDEX i"));
    }

    #[test]
    fn all_runtime_schema_copies_are_identical() {
        assert_eq!(CONTAINER_SCHEMA_SQL, SCHEMA_SQL);
        assert_eq!(PYTHON_SCHEMA_SQL, SCHEMA_SQL);
        assert_eq!(k8s_schema_sql(), SCHEMA_SQL);
    }

    #[test]
    fn schema_is_v3_selection_fenced_opaque_record_contract() {
        assert!(SCHEMA_SQL.contains("VALUES ('replay', 3)"));
        assert!(SCHEMA_SQL.contains("CREATE TABLE IF NOT EXISTS replay_records"));
        assert!(SCHEMA_SQL.contains("env_contract_version BETWEEN 1 AND 4294967295"));
        assert!(SCHEMA_SQL.contains("step_number BIGINT NOT NULL"));
        assert!(SCHEMA_SQL.contains("step_number BETWEEN 0 AND 4294967295"));
        assert!(SCHEMA_SQL.contains("collection_scope_id TEXT NOT NULL"));
        assert!(SCHEMA_SQL.contains("source_checkpoint_id TEXT"));
        assert!(SCHEMA_SQL.contains("payload BYTEA NOT NULL"));
        for leaked in [
            "game_metadata",
            "policy_probs",
            "mcts_value",
            "game_outcome",
            "legal_mask_offset",
            "board_width",
        ] {
            assert!(!SCHEMA_SQL.contains(leaked), "schema leaked {leaked}");
        }
    }

    #[test]
    fn exact_schema_validation_rejects_extra_tables_or_columns() {
        let tables = vec![
            "cartridge_schema_versions".into(),
            "replay_records".into(),
            "legacy".into(),
        ];
        assert!(validate_schema_tables(&tables).is_err());

        let mut columns = column_specs(EXPECTED_RECORD_COLUMNS);
        columns.push(ColumnSpec::new("reward", "real", "NO"));
        let primary_key = EXPECTED_RECORD_PRIMARY_KEY
            .iter()
            .map(|column| (*column).to_string())
            .collect::<Vec<_>>();
        assert!(validate_record_schema(&columns, &primary_key).is_err());
    }

    #[test]
    fn schema_marker_requires_exact_v3_row() {
        let columns = column_specs(EXPECTED_SCHEMA_MARKER_COLUMNS);
        let primary_key = vec!["component".to_string()];
        assert!(validate_schema_marker(
            &columns,
            &primary_key,
            &[("replay".into(), REPLAY_SCHEMA_VERSION)]
        )
        .is_ok());
        assert!(validate_schema_marker(&columns, &primary_key, &[("replay".into(), 1)]).is_err());
    }

    #[test]
    fn batch_insert_is_idempotent_and_never_overwrites() {
        // `ON CONFLICT ... DO NOTHING` is required, not forbidden: store_batch
        // retries after ambiguous commit outcomes, and re-inserting immutable
        // content-keyed rows must be a no-op. What stays forbidden is any
        // update path — a conflict may never overwrite an existing record.
        let sql = build_batch_insert_sql(2);
        assert!(sql.contains("INSERT INTO replay_records"));
        assert!(sql.contains("payload"));
        assert!(sql.contains("$1"));
        assert!(sql.contains("collection_scope_id"));
        assert!(sql.contains("source_checkpoint_id"));
        assert!(sql.contains("$20"));
        assert!(sql.contains(
            "ON CONFLICT (env_id, env_contract_version, algorithm_id, experience_schema, \
             collection_scope_id, id) DO NOTHING"
        ));
        assert!(!sql.contains("DO UPDATE"));
        assert!(!sql.contains("SET "));
    }

    #[test]
    fn insert_chunks_stay_below_the_protocol_parameter_ceiling() {
        assert_eq!(MAX_RECORDS_PER_INSERT, 6553);
        assert_eq!(MAX_RECORDS_PER_INSERT * COLS_PER_RECORD, 65530);

        // The largest permitted chunk must reach its exact final placeholder
        // and never the u16 bind-parameter limit.
        let sql = build_batch_insert_sql(MAX_RECORDS_PER_INSERT);
        assert!(sql.contains(&format!("${})", MAX_RECORDS_PER_INSERT * COLS_PER_RECORD)));
        assert!(!sql.contains("$65536"));
    }

    #[test]
    fn transient_classification_retries_io_but_not_validation_errors() {
        let io_error = anyhow::Error::new(std::io::Error::new(
            std::io::ErrorKind::ConnectionReset,
            "connection reset by peer",
        ))
        .context("failed to commit 3 replay records");
        assert!(is_transient_storage_error(&io_error));

        let validation_error = anyhow::anyhow!("replay record 'x' does not match selection");
        assert!(!is_transient_storage_error(&validation_error));
    }

    #[test]
    fn store_retry_delays_are_bounded_exponential() {
        for attempt in 0..STORE_RETRY_ATTEMPTS {
            let base = 100u64 << attempt;
            let delay = store_retry_delay(attempt).as_millis() as u64;
            assert!(delay >= base, "attempt {attempt}: {delay} < {base}");
            assert!(
                delay <= base + base / 2,
                "attempt {attempt}: {delay} too large"
            );
        }
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
