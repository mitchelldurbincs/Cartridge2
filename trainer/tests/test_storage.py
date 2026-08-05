"""Tests for the opaque, exact-selection replay v4 storage contract."""

import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

import trainer.storage as storage
from trainer.storage.base import (
    EmptyReplaySelectionError,
    ReplayProfile,
    ReplayRecord,
    ReplaySelection,
)
from trainer.storage.factory import create_replay_store
from trainer.storage.postgres import (
    _EXPECTED_RECORD_COLUMNS,
    _EXPECTED_RECORD_PRIMARY_KEY,
    _EXPECTED_SCHEMA_MARKER_COLUMNS,
    _EXPECTED_SCHEMA_MARKER_PRIMARY_KEY,
    _ID_CACHE_REFRESH_INTERVAL,
    REPLAY_SCHEMA_VERSION,
    PostgresReplayStore,
    _load_schema,
    _validate_record_schema,
    _validate_schema_marker,
    _validate_schema_tables,
)

TEST_PROFILE = ReplayProfile(
    env_id="testgame",
    env_contract_version=1,
    algorithm_id="alphazero_board_v1",
    experience_schema="alphazero_transition_v1",
)
TEST_SELECTION = ReplaySelection(TEST_PROFILE, "a" * 64, None)

postgres_available = bool(os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL"))
requires_postgres = pytest.mark.skipif(
    not postgres_available,
    reason="PostgreSQL not configured (set CARTRIDGE_STORAGE_POSTGRES_URL)",
)


def make_record(
    selection: ReplaySelection = TEST_SELECTION,
    *,
    record_id: str = "test-001",
    step_number: int = 0,
    payload: bytes | None = None,
) -> ReplayRecord:
    return selection.record(
        id=record_id,
        episode_id="ep-001",
        step_number=step_number,
        payload=payload if payload is not None else f"payload-{step_number}".encode(),
    )


def _record_row(record: ReplayRecord) -> tuple:
    """One database row for `record`, as the sampler's fetch query returns it."""
    return (
        record.id,
        record.env_id,
        record.env_contract_version,
        record.algorithm_id,
        record.experience_schema,
        record.collection_scope_id,
        record.source_checkpoint_id,
        record.episode_id,
        record.step_number,
        memoryview(record.payload),
    )


@pytest.fixture
def replay_store():
    url = os.environ["CARTRIDGE_STORAGE_POSTGRES_URL"]
    store = create_replay_store(TEST_SELECTION, connection_string=url)
    store.clear()
    try:
        yield store
    finally:
        store.clear()
        store.close()


class _RecordingCursor:
    def __init__(self, *, fetchone=(0,), fetchall=None, rowcount=0, fetchall_script=None):
        self.calls: list[tuple[str, tuple | list | None]] = []
        self._fetchone = fetchone
        self._fetchall = [] if fetchall is None else fetchall
        # When provided, each fetchall() consumes the next scripted result,
        # letting one test drive multi-query flows (id snapshot, then rows).
        self._fetchall_script = fetchall_script
        self.rowcount = rowcount

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))

    def executemany(self, sql, params):
        self.calls.append((sql, list(params)))

    def fetchone(self):
        return self._fetchone

    def fetchall(self):
        if self._fetchall_script is not None:
            return self._fetchall_script.pop(0) if self._fetchall_script else []
        return self._fetchall


class _RecordingConnection:
    def __init__(self, cursor: _RecordingCursor):
        self._cursor = cursor
        self.commits = 0
        self.checkouts = 0
        self.checked_out = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1


def store_with_recording_cursor(
    *, fetchone=(0,), fetchall=None, rowcount=0, fetchall_script=None
) -> tuple[PostgresReplayStore, _RecordingCursor, _RecordingConnection]:
    cursor = _RecordingCursor(
        fetchone=fetchone,
        fetchall=fetchall,
        rowcount=rowcount,
        fetchall_script=fetchall_script,
    )
    connection = _RecordingConnection(cursor)
    store = PostgresReplayStore.__new__(PostgresReplayStore)
    store._selection = TEST_SELECTION
    # No transient classes registered: retry becomes a pass-through, so these
    # unit tests observe exactly one attempt per operation.
    store._transient_errors = ()
    store._id_cache = None
    store._samples_since_refresh = 0

    @contextmanager
    def connection_scope():
        if connection.checked_out:
            raise AssertionError("nested replay connection checkout")
        connection.checked_out = True
        connection.checkouts += 1
        try:
            yield connection
        finally:
            connection.checked_out = False

    store._connection = connection_scope
    return store, cursor, connection


class TestReplayV4Contract:
    @pytest.mark.parametrize(
        "legacy_name",
        [
            "ExperienceProfile",
            "GameMetadata",
            "Transition",
            "ReplayBufferBase",
            "PostgresReplayBuffer",
            "create_replay_buffer",
        ],
    )
    def test_legacy_storage_names_are_not_exported(self, legacy_name):
        assert not hasattr(storage, legacy_name)

    def test_replay_store_has_no_algorithm_specific_tensor_or_metadata_api(self):
        for method in (
            "sample_batch_tensors",
            "get_metadata",
            "list_metadata",
            "store_metadata",
            "clear_transitions",
        ):
            assert not hasattr(PostgresReplayStore, method)

    def test_all_deployment_schemas_match_shared_schema_byte_for_byte(self):
        shared_schema = Path(__file__).parents[2] / "sql" / "schema.sql"
        expected = shared_schema.read_text()
        assert _load_schema() == expected
        assert (Path(__file__).parents[2] / "scripts" / "init-postgres.sql").read_text() == expected

        configmap = (
            Path(__file__).parents[2] / "k8s" / "base" / "postgres" / "init-configmap.yaml"
        ).read_text()
        marker = "  01-schema.sql: |\n"
        yaml_body = configmap.split(marker, maxsplit=1)[1]
        embedded = "\n".join(
            line[4:] if line.startswith("    ") else line for line in yaml_body.splitlines()
        )
        assert f"{embedded}\n" == expected

    def test_schema_validation_requires_only_v4_tables(self):
        exact = {"cartridge_schema_versions", "collection_scopes", "replay_records"}
        _validate_schema_tables(exact)
        with pytest.raises(RuntimeError, match="missing tables: replay_records"):
            _validate_schema_tables({"cartridge_schema_versions", "collection_scopes"})
        with pytest.raises(RuntimeError, match="missing tables: collection_scopes"):
            _validate_schema_tables({"cartridge_schema_versions", "replay_records"})
        with pytest.raises(RuntimeError, match="unexpected tables: game_metadata, transitions"):
            _validate_schema_tables(exact | {"transitions", "game_metadata"})

    def test_record_schema_accepts_only_exact_columns_and_profile_primary_key(self):
        _validate_record_schema(
            _EXPECTED_RECORD_COLUMNS,
            _EXPECTED_RECORD_PRIMARY_KEY,
        )
        with pytest.raises(RuntimeError, match="replay_records.*primary key"):
            _validate_record_schema(_EXPECTED_RECORD_COLUMNS, ("id",))
        for columns in (
            _EXPECTED_RECORD_COLUMNS[:-1],
            _EXPECTED_RECORD_COLUMNS + (("observation", "bytea", "YES"),),
            tuple(
                (
                    name,
                    "integer" if name == "env_contract_version" else sql_type,
                    nullable,
                )
                for name, sql_type, nullable in _EXPECTED_RECORD_COLUMNS
            ),
            tuple(
                (name, sql_type, "YES" if name == "payload" else nullable)
                for name, sql_type, nullable in _EXPECTED_RECORD_COLUMNS
            ),
        ):
            with pytest.raises(RuntimeError, match="replay_records.*columns"):
                _validate_record_schema(columns, _EXPECTED_RECORD_PRIMARY_KEY)

    def test_schema_marker_requires_exact_v4_shape_and_single_row(self):
        with pytest.raises(RuntimeError, match="missing.*version marker"):
            _validate_schema_marker(
                _EXPECTED_SCHEMA_MARKER_COLUMNS,
                _EXPECTED_SCHEMA_MARKER_PRIMARY_KEY,
                (),
            )
        for rows in (
            (("replay", REPLAY_SCHEMA_VERSION - 1),),
            (("replay", REPLAY_SCHEMA_VERSION), ("legacy", 1)),
        ):
            with pytest.raises(RuntimeError, match="marker rows.*expected"):
                _validate_schema_marker(
                    _EXPECTED_SCHEMA_MARKER_COLUMNS,
                    _EXPECTED_SCHEMA_MARKER_PRIMARY_KEY,
                    rows,
                )
        _validate_schema_marker(
            _EXPECTED_SCHEMA_MARKER_COLUMNS,
            _EXPECTED_SCHEMA_MARKER_PRIMARY_KEY,
            (("replay", REPLAY_SCHEMA_VERSION),),
        )

    @pytest.mark.parametrize("field", ["env_id", "algorithm_id", "experience_schema"])
    def test_profile_rejects_blank_identifiers(self, field):
        with pytest.raises(ValueError, match=field):
            replace(TEST_PROFILE, **{field: "  "})

    @pytest.mark.parametrize("value", [True, 1.5, 0, -1, 1 << 32])
    def test_profile_requires_positive_u32_contract_version(self, value):
        with pytest.raises(ValueError, match="env_contract_version.*inclusive range"):
            replace(TEST_PROFILE, env_contract_version=value)

    def test_selection_wraps_and_matches_opaque_bytes(self):
        record = make_record(payload=b"\x00\xff")
        assert TEST_SELECTION.matches(record)
        assert record.payload == b"\x00\xff"
        assert not TEST_SELECTION.matches(replace(record, algorithm_id="other_v1"))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("collection_scope_id", "A" * 64),
            ("collection_scope_id", "a" * 63),
            ("source_checkpoint_id", "not-a-digest"),
        ],
    )
    def test_selection_rejects_noncanonical_fences(self, field, value):
        with pytest.raises(ValueError, match="64-character SHA-256"):
            replace(TEST_SELECTION, **{field: value})

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("id", "", "id"),
            ("episode_id", " ", "episode_id"),
            ("env_contract_version", 0, "inclusive range"),
            ("step_number", -1, "inclusive range"),
            ("payload", bytearray(b"x"), "must be bytes"),
            ("collection_scope_id", "bad", "64-character SHA-256"),
            ("source_checkpoint_id", "bad", "64-character SHA-256"),
        ],
    )
    def test_record_rejects_invalid_envelope(self, field, value, message):
        error = TypeError if field == "payload" else ValueError
        with pytest.raises(error, match=message):
            replace(make_record(), **{field: value})

    @pytest.mark.parametrize("value", [True, 1.5, 1 << 32])
    def test_record_requires_positive_u32_contract_version(self, value):
        with pytest.raises(ValueError, match="env_contract_version.*inclusive range"):
            replace(make_record(), env_contract_version=value)

    @pytest.mark.parametrize("value", [True, 1.5, -1, 1 << 32])
    def test_record_requires_u32_step_number(self, value):
        with pytest.raises(ValueError, match="step_number.*inclusive range"):
            replace(make_record(), step_number=value)

    def test_record_accepts_u32_wire_boundaries(self):
        assert replace(TEST_PROFILE, env_contract_version=(1 << 32) - 1)
        assert replace(make_record(), step_number=(1 << 32) - 1)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("env_id", "othergame"),
            ("env_contract_version", 2),
            ("algorithm_id", "other_algorithm_v1"),
            ("experience_schema", "other_transition_v1"),
            ("collection_scope_id", "b" * 64),
            ("source_checkpoint_id", "c" * 64),
        ],
    )
    def test_store_rejects_every_profile_mismatch(self, field, value):
        store = PostgresReplayStore.__new__(PostgresReplayStore)
        store._selection = TEST_SELECTION
        record = replace(make_record(), **{field: value})
        with pytest.raises(ValueError, match="do not match replay selection"):
            store.store_batch([record])

    @pytest.mark.parametrize("records", [(), iter(())])
    def test_store_batch_rejects_non_list_batches(self, records):
        store = PostgresReplayStore.__new__(PostgresReplayStore)
        store._selection = TEST_SELECTION
        with pytest.raises(TypeError, match="batch must be a list"):
            store.store_batch(records)

    def test_store_batch_rejects_non_record_members(self):
        store = PostgresReplayStore.__new__(PostgresReplayStore)
        store._selection = TEST_SELECTION
        with pytest.raises(TypeError, match="only ReplayRecord"):
            store.store_batch([object()])

    def test_factory_forwards_exact_selection(self, monkeypatch):
        captured = {}

        class StubPostgresReplayStore:
            def __init__(self, connection_string, selection, **kwargs):
                captured.update(
                    connection_string=connection_string,
                    selection=selection,
                    kwargs=kwargs,
                )

        monkeypatch.setattr(
            "trainer.storage.postgres.PostgresReplayStore",
            StubPostgresReplayStore,
        )
        result = create_replay_store(
            TEST_SELECTION,
            connection_string="postgresql://example/test",
            validate_schema=False,
            pool_size=2,
        )
        assert isinstance(result, StubPostgresReplayStore)
        assert captured == {
            "connection_string": "postgresql://example/test",
            "selection": TEST_SELECTION,
            "kwargs": {"validate_schema": False, "pool_size": 2},
        }


class TestSelectionScopedSql:
    def test_count_filters_exact_selection(self):
        store, cursor, _ = store_with_recording_cursor(fetchone=(7,))
        assert store.count() == 7
        sql, params = cursor.calls[-1]
        assert "SELECT COUNT(*) FROM replay_records" in sql
        assert params == (
            TEST_PROFILE.env_id,
            TEST_PROFILE.env_contract_version,
            TEST_PROFILE.algorithm_id,
            TEST_PROFILE.experience_schema,
            TEST_SELECTION.collection_scope_id,
            TEST_SELECTION.source_checkpoint_id,
        )

    def test_count_episodes_filters_exact_selection(self):
        store, cursor, _ = store_with_recording_cursor(fetchone=(3,))
        assert store.count_episodes() == 3
        sql, params = cursor.calls[-1]
        assert "COUNT(DISTINCT episode_id)" in sql
        assert "source_checkpoint_id IS NOT DISTINCT FROM %s" in sql
        assert params == store._selection_params

    def test_clear_filters_exact_selection(self):
        store, cursor, connection = store_with_recording_cursor(rowcount=4)
        assert store.clear() == 4
        sql, params = cursor.calls[-1]
        assert sql.lstrip().startswith("DELETE FROM replay_records")
        assert params == (
            TEST_PROFILE.env_id,
            TEST_PROFILE.env_contract_version,
            TEST_PROFILE.algorithm_id,
            TEST_PROFILE.experience_schema,
            TEST_SELECTION.collection_scope_id,
            TEST_SELECTION.source_checkpoint_id,
        )
        assert connection.commits == 1

    def test_sampling_snapshots_ids_and_fetches_by_id_exactly_fenced(self):
        record = make_record()
        row = _record_row(record)
        store, cursor, connection = store_with_recording_cursor(
            fetchall_script=[[(record.id,)], [row]]
        )

        assert store.sample(3) == [record, record, record]

        ids_call, fetch_call = cursor.calls
        selection_key = (
            TEST_PROFILE.env_id,
            TEST_PROFILE.env_contract_version,
            TEST_PROFILE.algorithm_id,
            TEST_PROFILE.experience_schema,
            TEST_SELECTION.collection_scope_id,
            TEST_SELECTION.source_checkpoint_id,
        )
        assert ids_call[0].lstrip().startswith("SELECT id FROM replay_records")
        assert ids_call[1] == selection_key
        assert "id = ANY(%s)" in fetch_call[0]
        assert fetch_call[1] == (*selection_key, [record.id])
        for sql, _ in cursor.calls:
            assert "TABLESAMPLE" not in sql
            assert "RANDOM()" not in sql
            assert "COUNT(" not in sql
        assert connection.checkouts == 1

    def test_sampling_reuses_the_id_snapshot_until_mutation(self):
        record = make_record()
        row = _record_row(record)
        store, cursor, _ = store_with_recording_cursor(
            fetchall_script=[[(record.id,)], [row], [row], [(record.id,)], [row]]
        )

        store.sample(2)
        store.sample(2)
        # Two samples share one id snapshot: 1 id query + 2 fetches.
        assert len(cursor.calls) == 3

        store.clear()
        store.sample(2)
        # The mutation invalidated the snapshot, forcing a fresh id query.
        id_queries = [sql for sql, _ in cursor.calls if sql.lstrip().startswith("SELECT id FROM")]
        assert len(id_queries) == 2

    def test_sampling_refreshes_the_id_snapshot_periodically(self):
        record = make_record()
        row = _record_row(record)
        store, cursor, _ = store_with_recording_cursor(
            fetchall_script=[[(record.id,)], [row], [(record.id,)], [row]]
        )

        store.sample(1)
        store._samples_since_refresh = _ID_CACHE_REFRESH_INTERVAL
        store.sample(1)

        id_queries = [sql for sql, _ in cursor.calls if sql.lstrip().startswith("SELECT id FROM")]
        assert len(id_queries) == 2
        assert store._samples_since_refresh == 1

    def test_stale_snapshot_ids_trigger_one_refresh_and_redraw(self):
        stale = make_record(record_id="stale-001")
        fresh = make_record(record_id="fresh-001", step_number=1)
        store, cursor, _ = store_with_recording_cursor(
            fetchall_script=[
                [(stale.id,)],  # initial snapshot
                [],  # stale.id vanished (concurrent delete)
                [(fresh.id,)],  # refreshed snapshot
                [_record_row(fresh)],  # redraw fetch succeeds
            ]
        )

        assert store.sample(2) == [fresh, fresh]

        # A second disappearance in the same sample is a hard error.
        store_two, _, _ = store_with_recording_cursor(
            fetchall_script=[[(stale.id,)], [], [(stale.id,)], []]
        )
        with pytest.raises(RuntimeError, match="disappeared twice"):
            store_two.sample(1)

    def test_empty_selection_sample_fails_loudly(self):
        store, _, connection = store_with_recording_cursor(fetchall=[])

        with pytest.raises(EmptyReplaySelectionError, match="empty exact"):
            store.sample(3)

        assert connection.checkouts == 1

    def test_cleanup_scopes_delete_and_window_to_exact_selection(self):
        store, cursor, connection = store_with_recording_cursor(rowcount=3)
        assert store.cleanup(window_size=100) == 3
        sql, params = cursor.calls[-1]
        for column in (
            "env_id = %s",
            "env_contract_version = %s",
            "algorithm_id = %s",
            "experience_schema = %s",
            "collection_scope_id = %s",
            "source_checkpoint_id IS NOT DISTINCT FROM %s",
        ):
            assert sql.count(column) == 2
        selection_key = (
            TEST_PROFILE.env_id,
            TEST_PROFILE.env_contract_version,
            TEST_PROFILE.algorithm_id,
            TEST_PROFILE.experience_schema,
            TEST_SELECTION.collection_scope_id,
            TEST_SELECTION.source_checkpoint_id,
        )
        assert params == (*selection_key, *selection_key, 100)
        assert connection.commits == 1

    def test_store_is_a_plain_immutable_insert(self):
        store, cursor, connection = store_with_recording_cursor()
        records = [make_record(record_id="one"), make_record(record_id="two")]
        store.store_batch(records)
        sql, params = cursor.calls[-1]
        assert "INSERT INTO replay_records" in sql
        assert "ON CONFLICT" not in sql
        assert "UPDATE" not in sql
        assert params == [
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
        ]
        assert connection.commits == 1

    def test_store_one_and_empty_batch_have_clean_semantics(self):
        store, cursor, connection = store_with_recording_cursor()
        store.store(make_record())
        assert len(cursor.calls) == 1
        assert connection.commits == 1

        store, cursor, connection = store_with_recording_cursor()
        store.store_batch([])
        assert cursor.calls == []
        assert connection.checkouts == 0
        assert connection.commits == 0

    def test_database_rows_become_opaque_records(self):
        row = (
            "id",
            *(
                TEST_PROFILE.env_id,
                TEST_PROFILE.env_contract_version,
                TEST_PROFILE.algorithm_id,
                TEST_PROFILE.experience_schema,
            ),
            TEST_SELECTION.collection_scope_id,
            TEST_SELECTION.source_checkpoint_id,
            "ep-001",
            4,
            memoryview(b"opaque"),
        )
        assert PostgresReplayStore._rows_to_records([row]) == [
            make_record(record_id="id", step_number=4, payload=b"opaque")
        ]

    def test_invalid_sample_and_cleanup_sizes_fail_before_sql(self):
        store, _, connection = store_with_recording_cursor()
        for batch_size in (0, -1, True, 1.5, "1"):
            with pytest.raises(ValueError, match="batch_size"):
                store.sample(batch_size)
        with pytest.raises(ValueError, match="window_size"):
            store.cleanup(-1)
        assert connection.checkouts == 0


@requires_postgres
class TestPostgresReplayStore:
    def test_connection_and_schema(self, replay_store):
        assert replay_store.selection == TEST_SELECTION
        assert isinstance(replay_store.count(), int)

    def test_multiple_selection_bound_connections(self):
        url = os.environ["CARTRIDGE_STORAGE_POSTGRES_URL"]
        stores = [create_replay_store(TEST_SELECTION, connection_string=url) for _ in range(3)]
        try:
            assert all(isinstance(store.count(), int) for store in stores)
        finally:
            for store in stores:
                store.close()

    def test_store_count_and_sample_round_trip(self, replay_store):
        records = [
            make_record(record_id=f"round-trip-{index}", step_number=index) for index in range(10)
        ]
        replay_store.store_batch(records)
        assert replay_store.count() == 10
        batch = replay_store.sample(5)
        assert len(batch) == 5
        assert all(isinstance(item, ReplayRecord) for item in batch)
        assert all(TEST_SELECTION.matches(item) for item in batch)

    def test_one_record_fills_a_minibatch_with_replacement(self, replay_store):
        record = make_record(record_id="only-record")
        replay_store.store(record)

        assert replay_store.sample(8) == [record] * 8

    def test_empty_selection_sample_fails_loudly(self, replay_store):
        with pytest.raises(EmptyReplaySelectionError, match="empty exact"):
            replay_store.sample(1)

    def test_duplicate_identity_is_rejected_not_updated(self, replay_store):
        original = make_record(record_id="immutable", payload=b"first")
        replay_store.store(original)
        with pytest.raises(Exception):
            replay_store.store(replace(original, payload=b"second"))
        assert replay_store.count() == 1
        [stored] = replay_store.sample(1)
        assert stored.payload == b"first"

    def test_exact_selection_isolation_for_count_and_sample(self):
        url = os.environ["CARTRIDGE_STORAGE_POSTGRES_URL"]
        profiles = [
            TEST_PROFILE,
            replace(TEST_PROFILE, env_id="othergame"),
            replace(TEST_PROFILE, env_contract_version=2),
            replace(TEST_PROFILE, algorithm_id="other_algorithm_v1"),
            replace(TEST_PROFILE, experience_schema="other_transition_v1"),
        ]
        selections = [
            ReplaySelection(profile, f"{index + 1:064x}", None)
            for index, profile in enumerate(profiles)
        ]
        stores = [create_replay_store(selection, connection_string=url) for selection in selections]
        try:
            for store in stores:
                store.clear()
            for index, (selection, store) in enumerate(zip(selections, stores)):
                store.store(
                    make_record(
                        selection,
                        record_id="shared-id",
                        step_number=index,
                    )
                )
            for selection, store in zip(selections, stores):
                assert store.count() == 1
                [record] = store.sample(1)
                assert selection.matches(record)
        finally:
            for store in stores:
                store.clear()
                store.close()

    def test_clear_and_cleanup_preserve_other_selections(self, replay_store):
        url = os.environ["CARTRIDGE_STORAGE_POSTGRES_URL"]
        other_selection = replace(TEST_SELECTION, collection_scope_id="b" * 64)
        other = create_replay_store(other_selection, connection_string=url)
        try:
            other.clear()
            replay_store.store_batch(
                [make_record(record_id=f"primary-{index}", step_number=index) for index in range(5)]
            )
            other.store_batch(
                [
                    make_record(
                        other_selection,
                        record_id=f"other-{index}",
                        step_number=index,
                    )
                    for index in range(3)
                ]
            )
            assert replay_store.cleanup(2) == 3
            assert replay_store.count() == 2
            assert other.count() == 3
            assert replay_store.clear() == 2
            assert replay_store.count() == 0
            assert other.count() == 3
        finally:
            other.clear()
            other.close()

    def test_vacuum_keeps_store_usable(self, replay_store):
        replay_store.vacuum()
        assert replay_store.count() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestScopeRegistry:
    def test_register_or_verify_accepts_the_exact_registered_binding(self):
        expected_row = (
            TEST_PROFILE.env_id,
            TEST_PROFILE.env_contract_version,
            TEST_PROFILE.algorithm_id,
            TEST_PROFILE.experience_schema,
            TEST_SELECTION.source_checkpoint_id,
        )
        store, cursor, _ = store_with_recording_cursor(fetchone=expected_row)
        with store._connection() as conn:
            store._register_or_verify_scope(conn.cursor())
        insert_sql, insert_params = cursor.calls[0]
        assert "INSERT INTO collection_scopes" in insert_sql
        assert "ON CONFLICT (scope_id) DO NOTHING" in insert_sql
        assert insert_params[0] == TEST_SELECTION.collection_scope_id

    def test_register_or_verify_rejects_a_conflicting_source_checkpoint(self):
        conflicting = (
            TEST_PROFILE.env_id,
            TEST_PROFILE.env_contract_version,
            TEST_PROFILE.algorithm_id,
            TEST_PROFILE.experience_schema,
            "f" * 64,  # scope was registered against a different checkpoint
        )
        store, _, _ = store_with_recording_cursor(fetchone=conflicting)
        with store._connection() as conn:
            with pytest.raises(RuntimeError, match="one scope binds exactly one source"):
                store._register_or_verify_scope(conn.cursor())


class _FakeReaperCursor(_RecordingCursor):
    def __init__(self, *, victims):
        super().__init__(rowcount=7)
        self._victims = victims

    def fetchall(self):
        return [(victim,) for victim in self._victims]


class _FakeReaperConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.autocommit = False
        self.closed = False
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.commits += 1
        return False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


class TestScopeReaper:
    @pytest.mark.parametrize("retained", [0, -1, True, "2", 1.5])
    def test_reaper_rejects_invalid_retained_scopes(self, retained):
        from trainer.storage.scope_reaper import reap_profile_scopes

        with pytest.raises(ValueError, match="positive integer"):
            reap_profile_scopes(
                profile=TEST_PROFILE,
                retained_scopes=retained,
                connection_string="postgresql://unused",
            )

    def test_reaper_deletes_profile_scopes_beyond_the_window(self, monkeypatch):
        import trainer.storage.scope_reaper as scope_reaper

        victims = ["b" * 64, "c" * 64]
        cursor = _FakeReaperCursor(victims=victims)
        connection = _FakeReaperConnection(cursor)
        import psycopg2

        monkeypatch.setattr(psycopg2, "connect", lambda dsn, connect_timeout: connection)

        deleted = scope_reaper.reap_profile_scopes(
            profile=TEST_PROFILE,
            retained_scopes=2,
            connection_string="postgresql://unused",
        )

        assert deleted == 7
        select_sql, select_params = cursor.calls[0]
        assert "FROM collection_scopes" in select_sql
        assert "OFFSET %s" in select_sql
        assert select_params[-1] == 2
        delete_records_sql, delete_records_params = cursor.calls[1]
        assert "DELETE FROM replay_records" in delete_records_sql
        assert "collection_scope_id = ANY(%s)" in delete_records_sql
        assert delete_records_params[-1] == victims
        delete_scopes_sql, delete_scopes_params = cursor.calls[2]
        assert "DELETE FROM collection_scopes" in delete_scopes_sql
        assert delete_scopes_params == (victims,)
        vacuum_sql, _ = cursor.calls[3]
        assert vacuum_sql.startswith("VACUUM")
        assert connection.autocommit is True
        assert connection.closed is True

    def test_reaper_is_a_noop_when_the_window_covers_every_scope(self, monkeypatch):
        import trainer.storage.scope_reaper as scope_reaper

        cursor = _FakeReaperCursor(victims=[])
        connection = _FakeReaperConnection(cursor)
        import psycopg2

        monkeypatch.setattr(psycopg2, "connect", lambda dsn, connect_timeout: connection)

        assert (
            scope_reaper.reap_profile_scopes(
                profile=TEST_PROFILE,
                retained_scopes=5,
                connection_string="postgresql://unused",
            )
            == 0
        )
        assert len(cursor.calls) == 1
        assert connection.closed is True
