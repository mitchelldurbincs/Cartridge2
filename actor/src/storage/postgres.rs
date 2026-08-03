//! PostgreSQL implementation of the algorithm-neutral replay envelope.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use deadpool_postgres::{Config, Object, Pool, Runtime};
use std::time::Duration;
use tokio_postgres::types::ToSql;
use tokio_postgres::{NoTls, Transaction};

use super::{ReplayRecord, ReplaySelection, ReplayStore};

const SCHEMA_SQL: &str = include_str!("../../../sql/schema.sql");
const COLS_PER_RECORD: usize = 10;
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
        store.ensure_schema().await?;
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

        let sql = build_batch_insert_sql(records.len());
        let client = self.client().await?;

        let step_numbers = records
            .iter()
            .map(|record| i64::from(record.step_number))
            .collect::<Vec<_>>();
        let contract_versions = records
            .iter()
            .map(|record| i64::from(record.env_contract_version))
            .collect::<Vec<_>>();
        let mut parameters: Vec<&(dyn ToSql + Sync)> =
            Vec::with_capacity(records.len() * COLS_PER_RECORD);

        for (index, record) in records.iter().enumerate() {
            parameters.push(&record.id);
            parameters.push(&record.env_id);
            parameters.push(&contract_versions[index]);
            parameters.push(&record.algorithm_id);
            parameters.push(&record.experience_schema);
            parameters.push(&record.collection_scope_id);
            parameters.push(&record.source_checkpoint_id);
            parameters.push(&record.episode_id);
            parameters.push(&step_numbers[index]);
            parameters.push(&record.payload);
        }

        client
            .execute(&sql, &parameters)
            .await
            .with_context(|| format!("failed to insert {} replay records", records.len()))?;
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
