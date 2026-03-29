"""PostgreSQL backend for replay buffer storage.

This is the primary backend for all deployments (local and cloud).
Supports concurrent access from multiple actors and trainers.

Requires: psycopg2

Environment variables:
    CARTRIDGE_STORAGE_POSTGRES_URL: PostgreSQL connection string
        Format: postgresql://user:password@host:port/database
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generator

from trainer.storage.base import GameMetadata, ReplayBufferBase, Transition

# Path to shared SQL schema file (relative to project root)
# postgres.py -> storage -> trainer -> src -> trainer -> cartridge2/
_SCHEMA_PATH = Path(__file__).parent.parent.parent.parent.parent / "sql" / "schema.sql"

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)


def _load_schema() -> str:
    """Load the shared SQL schema file."""
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Schema file not found: {_SCHEMA_PATH}. "
            "Ensure you're running from the project root."
        )
    return _SCHEMA_PATH.read_text()


class PostgresReplayBuffer(ReplayBufferBase):
    """PostgreSQL-backed replay buffer implementation.

    This backend supports:
    - Concurrent writes from multiple actors
    - Concurrent reads from multiple trainers
    - Efficient random sampling using TABLESAMPLE or ORDER BY RANDOM()
    - Connection pooling for high throughput
    """

    def __init__(
        self,
        connection_string: str,
        validate_schema: bool = True,
        pool_size: int = 5,
    ):
        """Initialize PostgreSQL replay buffer.

        Args:
            connection_string: PostgreSQL connection URL.
            validate_schema: Whether to validate/create schema on connect.
            pool_size: Connection pool size (for future pooling support).

        Raises:
            ImportError: If psycopg2 is not installed.
            ConnectionError: If database connection fails.
        """
        try:
            import psycopg2
            from psycopg2 import pool
        except ImportError as e:
            raise ImportError(
                "PostgreSQL backend requires psycopg2. "
                "Install with: pip install psycopg2-binary"
            ) from e

        self.connection_string = connection_string
        self._pool_size = pool_size

        # Create connection pool
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=pool_size,
                dsn=connection_string,
            )
        except psycopg2.Error as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

        if validate_schema:
            self._ensure_schema()

    def _get_conn(self) -> "PgConnection":
        """Get a connection from the pool."""
        return self._pool.getconn()

    def _put_conn(self, conn: "PgConnection") -> None:
        """Return a connection to the pool."""
        self._pool.putconn(conn)

    @contextmanager
    def _connection(self) -> Generator["PgConnection", None, None]:
        """Context manager for connection pool access."""
        conn = self._get_conn()
        try:
            yield conn
        finally:
            self._put_conn(conn)

    def close(self) -> None:
        """Close all connections in the pool."""
        self._pool.closeall()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist using the shared schema file."""
        schema_sql = _load_schema()
        with self._connection() as conn:
            with conn.cursor() as cur:
                # Execute each statement from the shared schema file
                for statement in schema_sql.split(";"):
                    stmt = statement.strip()
                    # Skip empty statements and comment-only lines
                    if not stmt or stmt.startswith("--"):
                        continue
                    cur.execute(stmt)

                conn.commit()
                logger.info("PostgreSQL schema validated/created")

    def count(self, env_id: str | None = None) -> int:
        """Get total number of transitions in the buffer."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                if env_id is not None:
                    cur.execute(
                        "SELECT COUNT(*) FROM transitions WHERE env_id = %s", (env_id,)
                    )
                else:
                    cur.execute("SELECT COUNT(*) FROM transitions")
                return cur.fetchone()[0]

    def get_metadata(self, env_id: str | None = None) -> GameMetadata | None:
        """Get game metadata from the database."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                if env_id:
                    cur.execute(
                        """SELECT env_id, display_name, board_width, board_height,
                                  num_actions, obs_size, legal_mask_offset, player_count
                           FROM game_metadata WHERE env_id = %s""",
                        (env_id,),
                    )
                else:
                    cur.execute(
                        """SELECT env_id, display_name, board_width, board_height,
                                  num_actions, obs_size, legal_mask_offset, player_count
                           FROM game_metadata LIMIT 1"""
                    )

                row = cur.fetchone()
                if row is None:
                    return None

                return GameMetadata(
                    env_id=row[0],
                    display_name=row[1],
                    board_width=row[2],
                    board_height=row[3],
                    num_actions=row[4],
                    obs_size=row[5],
                    legal_mask_offset=row[6],
                    player_count=row[7],
                )

    def list_metadata(self) -> list[GameMetadata]:
        """List all game metadata in the database."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""SELECT env_id, display_name, board_width, board_height,
                              num_actions, obs_size, legal_mask_offset, player_count
                       FROM game_metadata""")

                return [
                    GameMetadata(
                        env_id=row[0],
                        display_name=row[1],
                        board_width=row[2],
                        board_height=row[3],
                        num_actions=row[4],
                        obs_size=row[5],
                        legal_mask_offset=row[6],
                        player_count=row[7],
                    )
                    for row in cur.fetchall()
                ]

    def sample(self, batch_size: int, env_id: str | None = None) -> list[Transition]:
        """Sample random transitions for training.

        Uses PostgreSQL's TABLESAMPLE for efficient random sampling on large tables,
        falling back to ORDER BY RANDOM() for smaller tables or when TABLESAMPLE
        doesn't return enough rows.
        """
        with self._connection() as conn:
            with conn.cursor() as cur:
                # Try TABLESAMPLE first (efficient for large tables)
                # SYSTEM samples pages, so we oversample and limit
                sample_pct = min(
                    100.0, (batch_size * 10.0) / max(1, self.count(env_id))
                )

                if env_id is not None:
                    cur.execute(
                        """
                        SELECT id, env_id, episode_id, step_number, state, action,
                               next_state, observation, next_observation, reward,
                               done, timestamp, policy_probs, mcts_value, game_outcome
                        FROM transitions TABLESAMPLE SYSTEM(%s)
                        WHERE env_id = %s
                        LIMIT %s
                        """,
                        (sample_pct, env_id, batch_size),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, env_id, episode_id, step_number, state, action,
                               next_state, observation, next_observation, reward,
                               done, timestamp, policy_probs, mcts_value, game_outcome
                        FROM transitions TABLESAMPLE SYSTEM(%s)
                        LIMIT %s
                        """,
                        (sample_pct, batch_size),
                    )

                rows = cur.fetchall()

                # If TABLESAMPLE didn't return enough, fall back to ORDER BY RANDOM()
                if len(rows) < batch_size:
                    if env_id is not None:
                        cur.execute(
                            """
                            SELECT id, env_id, episode_id, step_number, state, action,
                                   next_state, observation, next_observation, reward,
                                   done, timestamp, policy_probs, mcts_value, game_outcome
                            FROM transitions
                            WHERE env_id = %s
                            ORDER BY RANDOM()
                            LIMIT %s
                            """,
                            (env_id, batch_size),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, env_id, episode_id, step_number, state, action,
                                   next_state, observation, next_observation, reward,
                                   done, timestamp, policy_probs, mcts_value, game_outcome
                            FROM transitions
                            ORDER BY RANDOM()
                            LIMIT %s
                            """,
                            (batch_size,),
                        )
                    rows = cur.fetchall()

                return self._rows_to_transitions(rows)

    def _rows_to_transitions(self, rows: list) -> list[Transition]:
        """Convert database rows to Transition objects."""
        return [
            Transition(
                id=row[0],
                env_id=row[1],
                episode_id=row[2],
                step_number=row[3],
                state=bytes(row[4]) if row[4] else b"",
                action=bytes(row[5]) if row[5] else b"",
                next_state=bytes(row[6]) if row[6] else b"",
                observation=bytes(row[7]) if row[7] else b"",
                next_observation=bytes(row[8]) if row[8] else b"",
                reward=row[9],
                done=bool(row[10]),
                timestamp=row[11],
                policy_probs=bytes(row[12]) if row[12] else None,
                mcts_value=row[13] or 0.0,
                game_outcome=row[14],
            )
            for row in rows
        ]

    def clear_transitions(self) -> int:
        """Delete all transitions from the buffer."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM transitions")
                count = cur.rowcount
                conn.commit()
                return count

    def cleanup(self, window_size: int) -> int:
        """Delete old transitions to maintain a sliding window.

        Returns the number of deleted transitions.
        """
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM transitions
                    WHERE id NOT IN (
                        SELECT id FROM transitions
                        ORDER BY created_at DESC
                        LIMIT %s
                    )
                    """,
                    (window_size,),
                )
                count = cur.rowcount
                conn.commit()
                return count

    def vacuum(self) -> None:
        """Run VACUUM to reclaim storage space.

        Note: PostgreSQL VACUUM cannot run inside a transaction,
        so we need autocommit mode.
        """
        with self._connection() as conn:
            old_autocommit = conn.autocommit
            conn.autocommit = True
            try:
                with conn.cursor() as cur:
                    cur.execute("VACUUM transitions")
            finally:
                conn.autocommit = old_autocommit

    def store_batch(self, transitions: list[Transition]) -> None:
        """Store multiple transitions in a batch.

        This method is provided for actors that need to write to the buffer.
        Uses executemany for efficient batch inserts.
        """
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO transitions
                    (id, env_id, episode_id, step_number, state, action, next_state,
                     observation, next_observation, reward, done, timestamp,
                     policy_probs, mcts_value, game_outcome)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        game_outcome = EXCLUDED.game_outcome,
                        mcts_value = EXCLUDED.mcts_value
                    """,
                    [
                        (
                            t.id,
                            t.env_id,
                            t.episode_id,
                            t.step_number,
                            t.state,
                            t.action,
                            t.next_state,
                            t.observation,
                            t.next_observation,
                            t.reward,
                            t.done,
                            t.timestamp,
                            t.policy_probs,
                            t.mcts_value,
                            t.game_outcome,
                        )
                        for t in transitions
                    ],
                )
                conn.commit()

    def store_metadata(self, metadata: GameMetadata) -> None:
        """Store or update game metadata (upsert)."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO game_metadata
                    (env_id, display_name, board_width, board_height, num_actions,
                     obs_size, legal_mask_offset, player_count, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (env_id) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        board_width = EXCLUDED.board_width,
                        board_height = EXCLUDED.board_height,
                        num_actions = EXCLUDED.num_actions,
                        obs_size = EXCLUDED.obs_size,
                        legal_mask_offset = EXCLUDED.legal_mask_offset,
                        player_count = EXCLUDED.player_count,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        metadata.env_id,
                        metadata.display_name,
                        metadata.board_width,
                        metadata.board_height,
                        metadata.num_actions,
                        metadata.obs_size,
                        metadata.legal_mask_offset,
                        metadata.player_count,
                    ),
                )
                conn.commit()
