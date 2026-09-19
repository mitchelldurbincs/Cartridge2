//! Replay schema bootstrap and exact compatibility validation.
//!
//! The advisory lock, optional bootstrap, and validation share one transaction.
//! Existing schemas are checked as-is; only an empty schema is initialized.

use anyhow::{bail, Context, Result};
use deadpool_postgres::Object;
use tokio_postgres::Transaction;

const SCHEMA_SQL: &str = include_str!("../../../../sql/schema.sql");

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

/// Commit only after the complete replay schema has passed validation.
pub(super) async fn ensure_schema(client: &mut Object) -> Result<()> {
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
    let marker_primary_key = table_primary_key(&transaction, "cartridge_schema_versions").await?;
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

#[cfg(test)]
mod tests {
    use super::*;

    const CONTAINER_SCHEMA_SQL: &str = include_str!("../../../../scripts/init-postgres.sql");
    const PYTHON_SCHEMA_SQL: &str =
        include_str!("../../../../trainer/src/trainer/storage/schema.sql");
    const K8S_SCHEMA_CONFIG: &str =
        include_str!("../../../../k8s/base/postgres/init-configmap.yaml");

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
    fn exact_schema_accepts_valid_tables_in_either_order() {
        let mut tables = vec!["replay_records".into(), "cartridge_schema_versions".into()];
        assert!(validate_schema_tables(&tables).is_ok());
        tables.reverse();
        assert!(validate_schema_tables(&tables).is_ok());

        let columns = column_specs(EXPECTED_RECORD_COLUMNS);
        let primary_key = EXPECTED_RECORD_PRIMARY_KEY
            .iter()
            .map(|column| (*column).to_string())
            .collect::<Vec<_>>();
        assert!(validate_record_schema(&columns, &primary_key).is_ok());
    }

    #[test]
    fn existing_schema_requires_both_tables() {
        for tables in [
            vec![],
            vec!["replay_records".into()],
            vec!["cartridge_schema_versions".into()],
        ] {
            let error = validate_schema_tables(&tables).unwrap_err().to_string();
            assert!(error.starts_with("unsupported replay schema tables"));
            assert!(error.ends_with("recreate the database from sql/schema.sql"));
        }
    }

    #[test]
    fn record_schema_requires_exact_column_order_types_and_nullability() {
        let columns = column_specs(EXPECTED_RECORD_COLUMNS);
        let primary_key = EXPECTED_RECORD_PRIMARY_KEY
            .iter()
            .map(|column| (*column).to_string())
            .collect::<Vec<_>>();
        let mut missing = columns.clone();
        missing.pop();
        let mut reordered = columns.clone();
        reordered.swap(0, 1);
        let mut renamed = columns.clone();
        renamed[0].name = "record_id".into();
        let mut wrong_type = columns.clone();
        wrong_type[2].data_type = "integer".into();
        let mut required_source = columns.clone();
        required_source[6].is_nullable = "NO".into();
        let mut nullable_payload = columns.clone();
        nullable_payload[9].is_nullable = "YES".into();

        for invalid in [
            missing,
            reordered,
            renamed,
            wrong_type,
            required_source,
            nullable_payload,
        ] {
            let error = validate_record_schema(&invalid, &primary_key)
                .unwrap_err()
                .to_string();
            assert!(error.starts_with("unsupported replay schema: replay_records columns"));
        }
    }

    #[test]
    fn record_schema_requires_exact_primary_key_order_and_members() {
        let columns = column_specs(EXPECTED_RECORD_COLUMNS);
        let primary_key = EXPECTED_RECORD_PRIMARY_KEY
            .iter()
            .map(|column| (*column).to_string())
            .collect::<Vec<_>>();
        let mut reordered = primary_key.clone();
        reordered.swap(0, 1);
        let mut missing_scope = primary_key.clone();
        missing_scope.remove(4);
        let mut extra_source = primary_key;
        extra_source.push("source_checkpoint_id".into());

        for invalid in [vec![], reordered, missing_scope, extra_source] {
            let error = validate_record_schema(&columns, &invalid)
                .unwrap_err()
                .to_string();
            assert!(error.starts_with("unsupported replay schema: replay_records primary key"));
        }
    }

    #[test]
    fn schema_marker_rejects_missing_extra_or_wrong_identity_rows() {
        let columns = column_specs(EXPECTED_SCHEMA_MARKER_COLUMNS);
        let primary_key = vec!["component".to_string()];
        for rows in [
            vec![],
            vec![("other".into(), 3)],
            vec![("replay".into(), 4)],
            vec![("replay".into(), 3), ("other".into(), 3)],
        ] {
            let error = validate_schema_marker(&columns, &primary_key, &rows)
                .unwrap_err()
                .to_string();
            assert!(error.starts_with("unsupported replay schema marker rows"));
        }
    }

    #[test]
    fn schema_marker_validates_table_shape_before_rows() {
        let columns = column_specs(EXPECTED_SCHEMA_MARKER_COLUMNS);
        let primary_key = vec!["component".to_string()];
        let mut wrong_type = columns.clone();
        wrong_type[1].data_type = "bigint".into();
        let error = validate_schema_marker(&wrong_type, &primary_key, &[])
            .unwrap_err()
            .to_string();
        assert!(error.starts_with("unsupported replay schema: cartridge_schema_versions columns"));

        let error = validate_schema_marker(&columns, &["schema_version".into()], &[])
            .unwrap_err()
            .to_string();
        assert!(
            error.starts_with("unsupported replay schema: cartridge_schema_versions primary key")
        );
    }
}
