"""Unit tests for PostgreSQL storage safety that do not require a database."""

import logging

import pytest

from trainer.storage.postgres import PostgresReplayBuffer, _load_schema


class RecordingCursor:
    def __init__(self, rowcount: int = 0):
        self.calls = []
        self.rowcount = rowcount

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, params=None):
        self.calls.append((query, params))


class FakeConnection:
    def __init__(self, *, rowcount: int = 0, rollback_error: Exception | None = None):
        self.closed = 0
        self.cursor_instance = RecordingCursor(rowcount=rowcount)
        self.rollback_error = rollback_error
        self.rollback_calls = 0
        self.commit_calls = 0

    def cursor(self):
        return self.cursor_instance

    def rollback(self):
        self.rollback_calls += 1
        if self.rollback_error is not None:
            raise self.rollback_error

    def commit(self):
        self.commit_calls += 1


class FakePool:
    def __init__(self, connection):
        self.connection = connection
        self.put_calls = []

    def getconn(self):
        return self.connection

    def putconn(self, connection, close=False):
        self.put_calls.append((connection, close))


def make_buffer(connection: FakeConnection) -> tuple[PostgresReplayBuffer, FakePool]:
    pool = FakePool(connection)
    buffer = object.__new__(PostgresReplayBuffer)
    buffer._pool = pool
    return buffer, pool


def test_schema_is_executed_as_one_complete_script():
    connection = FakeConnection()
    buffer, _ = make_buffer(connection)

    buffer._ensure_schema()

    assert connection.cursor_instance.calls == [(_load_schema(), None)]
    assert connection.commit_calls == 1


def test_connection_rolls_back_before_reuse_after_error():
    connection = FakeConnection()
    buffer, pool = make_buffer(connection)

    with pytest.raises(ValueError, match="statement failed"):
        with buffer._connection():
            raise ValueError("statement failed")

    assert connection.rollback_calls == 1
    assert pool.put_calls == [(connection, False)]


def test_connection_rolls_back_successful_read_before_reuse():
    connection = FakeConnection()
    buffer, pool = make_buffer(connection)

    with buffer._connection() as borrowed:
        assert borrowed is connection

    assert connection.rollback_calls == 1
    assert pool.put_calls == [(connection, False)]


def test_connection_discards_connection_when_rollback_fails(caplog):
    connection = FakeConnection(rollback_error=RuntimeError("connection lost"))
    buffer, pool = make_buffer(connection)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError, match="original failure"):
            with buffer._connection():
                raise ValueError("original failure")

    assert connection.rollback_calls == 1
    assert pool.put_calls == [(connection, True)]
    assert "discarding it" in caplog.text


def test_connection_discards_already_closed_connection_after_error():
    connection = FakeConnection()
    connection.closed = 1
    buffer, pool = make_buffer(connection)

    with pytest.raises(ValueError):
        with buffer._connection():
            raise ValueError("statement failed")

    assert connection.rollback_calls == 0
    assert pool.put_calls == [(connection, True)]


def test_clear_transitions_filters_by_environment():
    connection = FakeConnection(rowcount=4)
    buffer, _ = make_buffer(connection)

    deleted = buffer.clear_transitions("connect4")

    assert deleted == 4
    assert connection.cursor_instance.calls == [
        ("DELETE FROM transitions WHERE env_id = %s", ("connect4",))
    ]
    assert connection.commit_calls == 1


def test_cleanup_filters_outer_delete_and_retention_window_by_environment():
    connection = FakeConnection(rowcount=7)
    buffer, _ = make_buffer(connection)

    deleted = buffer.cleanup(100, env_id="othello")

    assert deleted == 7
    query, params = connection.cursor_instance.calls[0]
    assert query.count("env_id = %s") == 2
    assert params == ("othello", "othello", 100)
    assert connection.commit_calls == 1
