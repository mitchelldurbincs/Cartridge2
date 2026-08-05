"""Tests for the opaque, exact-selection replay v3 storage contract."""

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
    def __init__(self, *, fetchone=(0,), fetchall=None, rowcount=0):
        self.calls: list[tuple[str, tuple | list | None]] = []
        self._fetchone = fetchone
        self._fetchall = [] if fetchall is None else fetchall
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
    *, fetchone=(0,), fetchall=None, rowcount=0
) -> tuple[PostgresReplayStore, _RecordingCursor, _RecordingConnection]:
    cursor = _RecordingCursor(
        fetchone=fetchone,
        fetchall=fetchall,
        rowcount=rowcount,
    )
    connection = _RecordingConnection(cursor)
    store = PostgresReplayStore.__new__(PostgresReplayStore)
    store._selection = TEST_SELECTION
    # No transient classes registered: retry becomes a pass-through, so these
    # unit tests observe exactly one attempt per operation.
    store._transient_errors = ()

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


class TestReplayV3Contract:
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

    def test_schema_validation_requires_only_v3_tables(self):
        exact = {"cartridge_schema_versions", "replay_records"}
        _validate_schema_tables(exact)
        with pytest.raises(RuntimeError, match="missing tables: replay_records"):
            _validate_schema_tables({"cartridge_schema_versions"})
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

    def test_schema_marker_requires_exact_v3_shape_and_single_row(self):
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

    def test_large_selection_sampling_is_bounded_and_exactly_fenced(self):
        record = make_record()
        row = (
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
        store, cursor, connection = store_with_recording_cursor(fetchone=(1000,), fetchall=[row])

        assert store.sample(3) == [record, record, record]

        count_call, sampled_call, fallback_call = cursor.calls
        selection_key = (
            TEST_PROFILE.env_id,
            TEST_PROFILE.env_contract_version,
            TEST_PROFILE.algorithm_id,
            TEST_PROFILE.experience_schema,
            TEST_SELECTION.collection_scope_id,
            TEST_SELECTION.source_checkpoint_id,
        )
        assert "replay_records" in count_call[0]
        assert count_call[1] == selection_key
        assert "FROM replay_records TABLESAMPLE" in sampled_call[0]
        assert sampled_call[1] == (3.0, *selection_key, 3)
        assert "FROM replay_records" in fallback_call[0]
        assert fallback_call[1] == (*selection_key, 3)
        assert all("ARRAY_AGG" not in sql and "MATERIALIZED" not in sql for sql, _ in cursor.calls)
        assert connection.checkouts == 1

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
