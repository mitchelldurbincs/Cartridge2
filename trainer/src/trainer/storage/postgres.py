"""PostgreSQL persistence for opaque, exact-selection-bound replay records."""

import logging
import random
from contextlib import contextmanager
from importlib.resources import files
from typing import TYPE_CHECKING, Generator

from trainer.storage.base import (
    EmptyReplaySelectionError,
    ReplayRecord,
    ReplaySelection,
    ReplayStore,
)

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)

REPLAY_SCHEMA_VERSION = 3
_REPLAY_SCHEMA_TABLES = {"cartridge_schema_versions", "replay_records"}
_EXPECTED_SCHEMA_MARKER_COLUMNS = (
    ("component", "text", "NO"),
    ("schema_version", "integer", "NO"),
)
_EXPECTED_SCHEMA_MARKER_PRIMARY_KEY = ("component",)
_EXPECTED_SCHEMA_MARKER_ROWS = (("replay", REPLAY_SCHEMA_VERSION),)
_EXPECTED_RECORD_COLUMNS = (
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
)
_EXPECTED_RECORD_PRIMARY_KEY = (
    "env_id",
    "env_contract_version",
    "algorithm_id",
    "experience_schema",
    "collection_scope_id",
    "id",
)
_SELECTION_WHERE = """
    env_id = %s AND env_contract_version = %s
    AND algorithm_id = %s AND experience_schema = %s
    AND collection_scope_id = %s
    AND source_checkpoint_id IS NOT DISTINCT FROM %s
"""


def _format_columns(columns: tuple[tuple[str, str, str], ...]) -> str:
    return ", ".join(
        f"{name} {data_type} {'NULL' if nullable == 'YES' else 'NOT NULL'}"
        for name, data_type, nullable in columns
    )


def _validate_schema_tables(tables: set[str]) -> None:
    if tables == _REPLAY_SCHEMA_TABLES:
        return
    missing = sorted(_REPLAY_SCHEMA_TABLES - tables)
    extra = sorted(tables - _REPLAY_SCHEMA_TABLES)
    details = []
    if missing:
        details.append(f"missing tables: {', '.join(missing)}")
    if extra:
        details.append(f"unexpected tables: {', '.join(extra)}")
    raise RuntimeError(
        "Replay database has an unsupported schema; "
        f"{'; '.join(details)}. Recreate the database from sql/schema.sql."
    )


def _validate_table_schema(
    table: str,
    columns: tuple[tuple[str, str, str], ...],
    primary_key: tuple[str, ...],
    expected_columns: tuple[tuple[str, str, str], ...],
    expected_primary_key: tuple[str, ...],
) -> None:
    if columns != expected_columns:
        raise RuntimeError(
            f"Replay database uses an unsupported {table} schema; columns are "
            f"({_format_columns(columns)}), expected "
            f"({_format_columns(expected_columns)}). Recreate the database "
            "from sql/schema.sql."
        )
    if primary_key != expected_primary_key:
        actual = ", ".join(primary_key) or "none"
        expected = ", ".join(expected_primary_key)
        raise RuntimeError(
            f"Replay database uses an unsupported {table} schema; "
            f"primary key is ({actual}), expected ({expected}). Recreate the "
            "database from sql/schema.sql."
        )


def _validate_record_schema(
    columns: tuple[tuple[str, str, str], ...], primary_key: tuple[str, ...]
) -> None:
    _validate_table_schema(
        "replay_records",
        columns,
        primary_key,
        _EXPECTED_RECORD_COLUMNS,
        _EXPECTED_RECORD_PRIMARY_KEY,
    )


def _validate_schema_marker(
    columns: tuple[tuple[str, str, str], ...],
    primary_key: tuple[str, ...],
    rows: tuple[tuple[str, int], ...],
) -> None:
    _validate_table_schema(
        "cartridge_schema_versions",
        columns,
        primary_key,
        _EXPECTED_SCHEMA_MARKER_COLUMNS,
        _EXPECTED_SCHEMA_MARKER_PRIMARY_KEY,
    )
    if not rows:
        raise RuntimeError(
            "Replay database is missing its schema version marker. Recreate the "
            "database from sql/schema.sql."
        )
    if rows != _EXPECTED_SCHEMA_MARKER_ROWS:
        raise RuntimeError(
            f"Replay database schema marker rows are {rows!r}, expected "
            f"{_EXPECTED_SCHEMA_MARKER_ROWS!r}. Recreate the database from "
            "sql/schema.sql."
        )


def _load_schema() -> str:
    """Load the replay DDL bundled with the installed trainer package."""
    return files("trainer.storage").joinpath("schema.sql").read_text(encoding="utf-8")


class PostgresReplayStore(ReplayStore):
    """Concurrent PostgreSQL store bound to one exact replay selection."""

    def __init__(
        self,
        connection_string: str,
        selection: ReplaySelection,
        validate_schema: bool = True,
        pool_size: int = 5,
    ):
        try:
            import psycopg2
            from psycopg2 import pool
        except ImportError as exc:
            raise ImportError(
                "PostgreSQL replay requires psycopg2; install psycopg2-binary"
            ) from exc

        self.connection_string = connection_string
        if not isinstance(selection, ReplaySelection):
            raise TypeError("selection must be ReplaySelection")
        self._selection = selection
        self._pool_size = pool_size
        try:
            self._pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=pool_size,
                dsn=connection_string,
            )
        except psycopg2.Error as exc:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {exc}") from exc

        if validate_schema:
            self._ensure_schema()

    @property
    def selection(self) -> ReplaySelection:
        return self._selection

    @property
    def _selection_params(self) -> tuple[object, ...]:
        return (
            self.selection.profile.env_id,
            self.selection.profile.env_contract_version,
            self.selection.profile.algorithm_id,
            self.selection.profile.experience_schema,
            self.selection.collection_scope_id,
            self.selection.source_checkpoint_id,
        )

    def _get_conn(self) -> "PgConnection":
        return self._pool.getconn()

    def _put_conn(self, conn: "PgConnection") -> None:
        self._pool.putconn(conn)

    @contextmanager
    def _connection(self) -> Generator["PgConnection", None, None]:
        conn = self._get_conn()
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def close(self) -> None:
        self._pool.closeall()

    @staticmethod
    def _table_contract(cur, table: str):
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table,),
        )
        columns = tuple((row[0], row[1], row[2]) for row in cur.fetchall())
        cur.execute(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.constraint_schema = kcu.constraint_schema
            WHERE tc.table_schema = current_schema()
              AND tc.table_name = %s
              AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY kcu.ordinal_position
            """,
            (table,),
        )
        return columns, tuple(row[0] for row in cur.fetchall())

    def _ensure_schema(self) -> None:
        """Create only an empty schema, then require the exact v3 protocol."""
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(745472510202)")
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = current_schema()
                      AND table_type = 'BASE TABLE'
                    """)
                existing_tables = {row[0] for row in cur.fetchall()}
                if not existing_tables:
                    executable_sql = "\n".join(
                        line
                        for line in _load_schema().splitlines()
                        if not line.lstrip().startswith("--")
                    )
                    for statement in executable_sql.split(";"):
                        if statement.strip():
                            cur.execute(statement.strip())
                else:
                    _validate_schema_tables(existing_tables)

                marker_columns, marker_primary_key = self._table_contract(
                    cur, "cartridge_schema_versions"
                )
                cur.execute("""
                    SELECT component, schema_version
                    FROM cartridge_schema_versions
                    ORDER BY component
                    """)
                marker_rows = tuple((row[0], row[1]) for row in cur.fetchall())
                _validate_schema_marker(marker_columns, marker_primary_key, marker_rows)

                record_columns, record_primary_key = self._table_contract(cur, "replay_records")
                _validate_record_schema(record_columns, record_primary_key)
                conn.commit()
                logger.info("PostgreSQL replay schema v3 validated/created")

    def _count_selection(self, cur) -> int:
        cur.execute(
            f"SELECT COUNT(*) FROM replay_records WHERE {_SELECTION_WHERE}",
            self._selection_params,
        )
        return cur.fetchone()[0]

    def count(self) -> int:
        with self._connection() as conn:
            with conn.cursor() as cur:
                return self._count_selection(cur)

    def count_episodes(self) -> int:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(DISTINCT episode_id) FROM replay_records "
                    f"WHERE {_SELECTION_WHERE}",
                    self._selection_params,
                )
                return cur.fetchone()[0]

    def sample(self, batch_size: int) -> list[ReplayRecord]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        with self._connection() as conn:
            with conn.cursor() as cur:
                selection_count = self._count_selection(cur)
                if selection_count == 0:
                    raise EmptyReplaySelectionError(
                        "cannot sample from an empty exact replay selection"
                    )
                # Keep database work bounded by the minibatch. If the exact
                # selection is smaller, fill the remainder with replacement
                # after decoding the selected immutable rows.
                sample_size = min(batch_size, selection_count)
                cur.execute(
                    f"""
                    SELECT id, env_id, env_contract_version, algorithm_id,
                           experience_schema, collection_scope_id,
                           source_checkpoint_id, episode_id, step_number, payload
                    FROM replay_records
                    WHERE {_SELECTION_WHERE}
                    ORDER BY RANDOM()
                    LIMIT %s
                    """,
                    (*self._selection_params, sample_size),
                )
                rows = cur.fetchall()
                records = self._rows_to_records(rows)
        if not records:
            raise EmptyReplaySelectionError("cannot sample from an empty exact replay selection")
        if len(records) < batch_size:
            records.extend(random.choices(records, k=batch_size - len(records)))
        return records

    @staticmethod
    def _rows_to_records(rows: list) -> list[ReplayRecord]:
        return [
            ReplayRecord(
                id=row[0],
                env_id=row[1],
                env_contract_version=row[2],
                algorithm_id=row[3],
                experience_schema=row[4],
                collection_scope_id=row[5],
                source_checkpoint_id=row[6],
                episode_id=row[7],
                step_number=row[8],
                payload=bytes(row[9]),
            )
            for row in rows
        ]

    def clear(self) -> int:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM replay_records WHERE {_SELECTION_WHERE}",
                    self._selection_params,
                )
                count = cur.rowcount
                conn.commit()
                return count

    def cleanup(self, window_size: int) -> int:
        if window_size < 0:
            raise ValueError("window_size cannot be negative")
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM replay_records
                    WHERE {_SELECTION_WHERE}
                      AND id NOT IN (
                        SELECT id FROM replay_records
                        WHERE {_SELECTION_WHERE}
                        ORDER BY created_at DESC, id DESC
                        LIMIT %s
                      )
                    """,
                    (
                        *self._selection_params,
                        *self._selection_params,
                        window_size,
                    ),
                )
                count = cur.rowcount
                conn.commit()
                return count

    def vacuum(self) -> None:
        with self._connection() as conn:
            old_autocommit = conn.autocommit
            conn.autocommit = True
            try:
                with conn.cursor() as cur:
                    cur.execute("VACUUM replay_records")
            finally:
                conn.autocommit = old_autocommit

    def _require_selection(self, records: list[ReplayRecord]) -> None:
        if not isinstance(records, list):
            raise TypeError("Replay record batch must be a list")
        if any(not isinstance(record, ReplayRecord) for record in records):
            raise TypeError("Replay record batch must contain only ReplayRecord values")
        mismatched = [record.id for record in records if not self.selection.matches(record)]
        if mismatched:
            raise ValueError(
                f"Replay records do not match replay selection {self.selection}: "
                + ", ".join(mismatched)
            )

    def store(self, record: ReplayRecord) -> None:
        self.store_batch([record])

    def store_batch(self, records: list[ReplayRecord]) -> None:
        self._require_selection(records)
        if not records:
            return
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO replay_records
                    (id, env_id, env_contract_version, algorithm_id,
                     experience_schema, collection_scope_id, source_checkpoint_id,
                     episode_id, step_number, payload)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            record.id,
                            record.env_id,
                            record.env_contract_version,
                            record.algorithm_id,
                            record.experience_schema,
                            record.collection_scope_id,
                            record.source_checkpoint_id,
                            record.episode_id,
                            record.step_number,
                            record.payload,
                        )
                        for record in records
                    ],
                )
                conn.commit()
