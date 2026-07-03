"""Tests for backoff.py - wait/retry utilities.

Uses a fake clock patched over time.time/time.sleep so the tests are
deterministic and run instantly.
"""

import pytest

from trainer import backoff
from trainer.backoff import MAX_BACKOFF_INTERVAL, WaitTimeout, wait_with_backoff


class FakeClock:
    """Deterministic stand-in for time.time/time.sleep."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(backoff.time, "time", fake.time)
    monkeypatch.setattr(backoff.time, "sleep", fake.sleep)
    return fake


class TestWaitWithBackoff:
    def test_returns_immediately_when_condition_true(self, clock):
        wait_with_backoff(lambda: True, "already satisfied")

        assert clock.sleeps == []

    def test_waits_until_condition_becomes_true(self, clock):
        results = iter([False, False, True])

        wait_with_backoff(
            lambda: next(results), "condition", interval=1.0, max_wait=100.0
        )

        # Sleep interval doubles on each wait
        assert clock.sleeps == [1.0, 2.0]

    def test_raises_wait_timeout_with_description(self, clock):
        with pytest.raises(WaitTimeout, match="replay data"):
            wait_with_backoff(lambda: False, "replay data", interval=3.0, max_wait=10.0)

        # Sleeps are clamped so total waiting never exceeds max_wait
        assert clock.sleeps == [3.0, 6.0, 1.0]
        assert clock.now == 10.0

    def test_zero_max_wait_waits_forever_with_capped_backoff(self, clock):
        checks = iter([False, False, False, True])

        wait_with_backoff(lambda: next(checks), "condition", interval=40.0, max_wait=0)

        # Backoff doubles but is capped at MAX_BACKOFF_INTERVAL, and the
        # loop keeps waiting well past what any positive max_wait allows.
        assert clock.sleeps == [40.0, MAX_BACKOFF_INTERVAL, MAX_BACKOFF_INTERVAL]

    def test_logs_waiting_message(self, clock, caplog):
        results = iter([False, True])
        test_logger = backoff.logging.getLogger("test_backoff")

        with caplog.at_level("INFO", logger="test_backoff"):
            wait_with_backoff(
                lambda: next(results),
                "buffered episodes",
                interval=1.0,
                max_wait=100.0,
                logger=test_logger,
            )

        assert any("buffered episodes" in message for message in caplog.messages)
