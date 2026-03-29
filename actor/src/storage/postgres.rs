//! PostgreSQL backend for replay buffer storage.
//!
//! This is the only storage backend for the actor.
//! Supports concurrent writes from multiple actor instances.
//! Uses connection pooling via deadpool-postgres for improved throughput.

use anyhow::Result;
use async_trait::async_trait;
use deadpool_postgres::{Config, Pool, Runtime};
use engine_core::GameMetadata;
use std::time::{Duration, Instant};
use tokio_postgres::types::ToSql;
use tokio_postgres::NoTls;

use super::{ReplayStore, Transition};
use crate::metrics;

/// SQL schema embedded at compile time from the shared schema file.
const SCHEMA_SQL: &str = include_str!("../../../sql/schema.sql");

/// Number of columns in the transitions table INSERT.
const COLS_PER_TRANSITION: usize = 15;

/// Build a multi-row INSERT statement for batch inserts.
///
/// Generates SQL like:
/// ```sql
/// INSERT INTO transitions (...) VALUES ($1, ..., $15), ($16, ..., $30), ...
/// ON CONFLICT (id) DO UPDATE SET ...
/// ```
fn build_batch_insert_sql(batch_size: usize) -> String {
    let mut sql = String::with_capacity(512 + batch_size * 64);
    sql.push_str(
        "INSERT INTO transitions
         (id, env_id, episode_id, step_number, state, action, next_state,
          observation, next_observation, reward, done, timestamp,
          policy_probs, mcts_value, game_outcome)
         VALUES ",
    );

    for i in 0..batch_size {
        if i > 0 {
            sql.push_str(", ");
        }
        sql.push('(');
        for j in 0..COLS_PER_TRANSITION {
            if j > 0 {
                sql.push_str(", ");
            }
            sql.push('$');
            // Parameter indices are 1-based
            let param_idx = i * COLS_PER_TRANSITION + j + 1;
            sql.push_str(&param_idx.to_string());
        }
        sql.push(')');
    }

    sql.push_str(
        " ON CONFLICT (id) DO UPDATE SET
             game_outcome = EXCLUDED.game_outcome,
             mcts_value = EXCLUDED.mcts_value",
    );

    sql
}

/// Configuration for PostgreSQL connection pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of connections in the pool.
    pub max_size: usize,
    /// Timeout in seconds to wait for a connection from the pool.
    pub connect_timeout_secs: u64,
    /// Idle timeout for connections in seconds (None = no timeout).
    pub idle_timeout_secs: Option<u64>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_size: 16,
            connect_timeout_secs: 30,
            idle_timeout_secs: Some(300), // 5 minutes
        }
    }
}

/// PostgreSQL-backed replay buffer implementation with connection pooling.
///
/// This backend supports:
/// - Concurrent writes from multiple actors
/// - Connection pooling for improved throughput
/// - Automatic connection recovery
/// - Efficient batch inserts with ON CONFLICT handling
pub struct PostgresReplayStore {
    pool: Pool,
}

impl PostgresReplayStore {
    /// Create a new PostgreSQL replay store with default pool configuration.
    ///
    /// # Arguments
    /// * `connection_string` - PostgreSQL connection URL
    ///   Format: `postgresql://user:password@host:port/database`
    #[allow(dead_code)]
    pub async fn new(connection_string: &str) -> Result<Self> {
        Self::with_pool_config(connection_string, PoolConfig::default()).await
    }

    /// Create a new PostgreSQL replay store with custom pool configuration.
    ///
    /// # Arguments
    /// * `connection_string` - PostgreSQL connection URL
    /// * `pool_config` - Connection pool configuration
    pub async fn with_pool_config(
        connection_string: &str,
        pool_config: PoolConfig,
    ) -> Result<Self> {
        // Parse connection string to extract components
        let pg_config: tokio_postgres::Config = connection_string.parse()?;

        let mut cfg = Config::new();

        // Extract host
        if let Some(host) = pg_config.get_hosts().first() {
            cfg.host = Some(match host {
                tokio_postgres::config::Host::Tcp(s) => s.clone(),
                #[cfg(unix)]
                tokio_postgres::config::Host::Unix(path) => path.to_string_lossy().into_owned(),
            });
        }

        // Extract other connection parameters
        cfg.port = pg_config.get_ports().first().copied();
        cfg.user = pg_config.get_user().map(|s| s.to_string());
        cfg.password = pg_config
            .get_password()
            .map(|p| String::from_utf8_lossy(p).to_string());
        cfg.dbname = pg_config.get_dbname().map(|s| s.to_string());

        // Configure pool settings
        cfg.pool = Some(deadpool_postgres::PoolConfig {
            max_size: pool_config.max_size,
            timeouts: deadpool_postgres::Timeouts {
                wait: Some(Duration::from_secs(pool_config.connect_timeout_secs)),
                create: Some(Duration::from_secs(pool_config.connect_timeout_secs)),
                recycle: pool_config.idle_timeout_secs.map(Duration::from_secs),
            },
            ..Default::default()
        });

        let pool = cfg.create_pool(Some(Runtime::Tokio1), NoTls)?;

        let store = Self { pool };

        // Ensure schema exists
        store.ensure_schema().await?;

        tracing::info!(
            max_size = pool_config.max_size,
            "PostgreSQL connection pool initialized"
        );

        Ok(store)
    }

    async fn ensure_schema(&self) -> Result<()> {
        let client = self.pool.get().await?;

        // Execute each statement from the shared schema file
        // Split on semicolons and execute non-empty statements
        for statement in SCHEMA_SQL.split(';') {
            let stmt = statement.trim();
            // Skip empty statements and comments
            if stmt.is_empty() || stmt.starts_with("--") {
                continue;
            }
            client.execute(stmt, &[]).await?;
        }

        tracing::info!("PostgreSQL schema validated/created");
        Ok(())
    }
}

#[async_trait]
impl ReplayStore for PostgresReplayStore {
    async fn store(&self, transition: &Transition) -> Result<()> {
        let client = self.pool.get().await?;
        client
            .execute(
                "INSERT INTO transitions
                 (id, env_id, episode_id, step_number, state, action, next_state,
                  observation, next_observation, reward, done, timestamp,
                  policy_probs, mcts_value, game_outcome)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                 ON CONFLICT (id) DO UPDATE SET
                     game_outcome = EXCLUDED.game_outcome,
                     mcts_value = EXCLUDED.mcts_value",
                &[
                    &transition.id,
                    &transition.env_id,
                    &transition.episode_id,
                    &(transition.step_number as i32),
                    &transition.state,
                    &transition.action,
                    &transition.next_state,
                    &transition.observation,
                    &transition.next_observation,
                    &transition.reward,
                    &transition.done,
                    &(transition.timestamp as i64),
                    &transition.policy_probs,
                    &transition.mcts_value,
                    &transition.game_outcome,
                ],
            )
            .await?;
        Ok(())
    }

    async fn store_batch(&self, transitions: &[Transition]) -> Result<()> {
        if transitions.is_empty() {
            return Ok(());
        }

        let start = Instant::now();

        // Build single multi-row INSERT statement (reduces N round-trips to 1)
        let sql = build_batch_insert_sql(transitions.len());
        let client = self.pool.get().await?;

        // Update pool metrics
        let pool_status = self.pool.status();
        metrics::DB_POOL_SIZE.set(pool_status.size as i64);
        metrics::DB_POOL_AVAILABLE.set(pool_status.available as i64);
        metrics::DB_POOL_WAITING.set(pool_status.waiting as i64);

        // We need to store converted values (i32, i64) so references remain valid
        let step_numbers: Vec<i32> = transitions.iter().map(|t| t.step_number as i32).collect();
        let timestamps: Vec<i64> = transitions.iter().map(|t| t.timestamp as i64).collect();

        // Build flattened parameter list
        let mut params: Vec<&(dyn ToSql + Sync)> =
            Vec::with_capacity(transitions.len() * COLS_PER_TRANSITION);

        for (i, t) in transitions.iter().enumerate() {
            params.push(&t.id);
            params.push(&t.env_id);
            params.push(&t.episode_id);
            params.push(&step_numbers[i]);
            params.push(&t.state);
            params.push(&t.action);
            params.push(&t.next_state);
            params.push(&t.observation);
            params.push(&t.next_observation);
            params.push(&t.reward);
            params.push(&t.done);
            params.push(&timestamps[i]);
            params.push(&t.policy_probs);
            params.push(&t.mcts_value);
            params.push(&t.game_outcome);
        }

        client.execute(&sql as &str, &params).await?;

        // Record metrics
        let duration = start.elapsed().as_secs_f64();
        metrics::DB_WRITE_SECONDS.observe(duration);
        metrics::TRANSITIONS_STORED.inc_by(transitions.len() as u64);

        Ok(())
    }

    async fn count(&self) -> Result<usize> {
        let client = self.pool.get().await?;
        let row = client
            .query_one("SELECT COUNT(*) FROM transitions", &[])
            .await?;
        let count: i64 = row.get(0);
        Ok(count as usize)
    }

    async fn store_metadata(&self, metadata: &GameMetadata) -> Result<()> {
        let client = self.pool.get().await?;
        client
            .execute(
                "INSERT INTO game_metadata
                 (env_id, display_name, board_width, board_height, num_actions,
                  obs_size, legal_mask_offset, player_count, updated_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP)
                 ON CONFLICT (env_id) DO UPDATE SET
                     display_name = EXCLUDED.display_name,
                     board_width = EXCLUDED.board_width,
                     board_height = EXCLUDED.board_height,
                     num_actions = EXCLUDED.num_actions,
                     obs_size = EXCLUDED.obs_size,
                     legal_mask_offset = EXCLUDED.legal_mask_offset,
                     player_count = EXCLUDED.player_count,
                     updated_at = CURRENT_TIMESTAMP",
                &[
                    &metadata.env_id,
                    &metadata.display_name,
                    &(metadata.board_width as i32),
                    &(metadata.board_height as i32),
                    &(metadata.num_actions as i32),
                    &(metadata.obs_size as i32),
                    &(metadata.legal_mask_offset as i32),
                    &(metadata.player_count as i32),
                ],
            )
            .await?;
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let client = self.pool.get().await?;
        client.execute("DELETE FROM transitions", &[]).await?;
        Ok(())
    }
}
