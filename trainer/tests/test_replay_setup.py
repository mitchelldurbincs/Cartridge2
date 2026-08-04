"""Tests for profile-bound replay setup and cleanup."""

from types import SimpleNamespace

from trainer.algorithms.alphazero_board_v1 import ALGORITHM_ID, DESCRIPTOR
from trainer.replay_setup import handle_replay_cleanup, setup_replay
from trainer.storage import ReplayProfile


class ProfileBoundReplay:
    """Replay fake that records attempts to widen its namespace."""

    def __init__(self, *, count: int, count_schedule: tuple[int, ...] = ()):
        self.profile = ReplayProfile(
            env_id="connect4",
            env_contract_version=1,
            algorithm_id=ALGORITHM_ID,
            experience_schema=DESCRIPTOR.components.experience_schema,
        )
        self._count = count
        self._count_schedule = list(count_schedule)
        self.count_calls: list[tuple[tuple, dict]] = []
        self.clear_calls = 0
        self.cleanup_calls: list[int] = []

    def count(self, *args, **kwargs):
        self.count_calls.append((args, kwargs))
        if self._count_schedule:
            self._count = self._count_schedule.pop(0)
        return self._count

    def clear(self):
        self.clear_calls += 1
        self._count = 0
        return 4

    def cleanup(self, window_size):
        self.cleanup_calls.append(window_size)
        self._count = min(self._count, window_size)
        return 3


def learner_for_replay(*, clear=False, batch_size=4, replay_window=8):
    learner = SimpleNamespace()
    learner.config = SimpleNamespace(
        clear_replay_on_start=clear,
        batch_size=batch_size,
        wait_interval=0.01,
        max_wait=0.1,
        replay_window=replay_window,
    )
    learner.stats = SimpleNamespace(replay_record_count=-1)
    learner._replay_record_count_cache = -1
    learner._replay_cleanup_every = 5
    return learner


def test_setup_accepts_one_record_for_any_minibatch_size():
    learner = learner_for_replay(batch_size=64)
    replay = ProfileBoundReplay(count=1)

    setup_replay(learner, replay, "connect4")

    assert replay.count_calls == [((), {})]
    assert learner._replay_record_count_cache == 1
    assert learner.stats.replay_record_count == 1


def test_setup_waits_only_for_the_first_usable_record():
    learner = learner_for_replay(batch_size=64)
    replay = ProfileBoundReplay(count=0, count_schedule=(0, 1))

    setup_replay(learner, replay, "connect4")

    assert replay.count_calls == [((), {}), ((), {}), ((), {})]
    assert learner._replay_record_count_cache == 1
    assert learner.stats.replay_record_count == 1


def test_setup_clear_is_profile_scoped_by_store():
    learner = learner_for_replay(clear=True, batch_size=64)
    replay = ProfileBoundReplay(count=4, count_schedule=(0, 1))

    setup_replay(learner, replay, "connect4")

    assert replay.clear_calls == 1
    assert replay.count_calls == [((), {}), ((), {}), ((), {})]
    assert learner.stats.replay_record_count == 1


def test_cleanup_uses_profile_bound_window_and_refreshes_count():
    learner = learner_for_replay(replay_window=3)
    replay = ProfileBoundReplay(count=9)

    handle_replay_cleanup(learner, global_step=10, replay=replay, env_id="connect4")

    assert replay.cleanup_calls == [3]
    assert replay.count_calls == [((), {})]
    assert learner._replay_record_count_cache == 3
    assert learner.stats.replay_record_count == 3
