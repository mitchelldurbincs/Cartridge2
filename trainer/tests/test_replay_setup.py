"""Tests for the replay-buffer metadata cross-check.

Game facts come from the engine-generated manifest, not the database. The
database row is written by the actor on every startup, so comparing the two
detects an actor and a trainer built from different commits — a state in which
the buffer's observations do not match the network's input layout.
"""

import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest

from trainer.game_config import get_config
from trainer.replay_setup import (
    MetadataMismatch,
    check_metadata_agrees,
    handle_replay_cleanup,
    setup_replay,
)


class FakeReplay:
    """Minimal replay stand-in: only get_metadata is exercised."""

    def __init__(self, metadata):
        self._metadata = metadata

    def get_metadata(self, env_id):
        return self._metadata


class FakeTrainer:
    def __init__(self, config):
        self.game_config = config


class LifecycleReplay(FakeReplay):
    """Replay stand-in that records destructive-operation scopes."""

    def __init__(self, metadata, count=8):
        super().__init__(metadata)
        self.transition_count = count
        self.clear_calls = []
        self.cleanup_calls = []

    def clear_transitions(self, env_id):
        self.clear_calls.append(env_id)
        return 3

    def cleanup(self, window_size, *, env_id):
        self.cleanup_calls.append((window_size, env_id))
        return 2

    def count(self, env_id):
        return self.transition_count


def lifecycle_trainer(env_id="connect4"):
    return SimpleNamespace(
        config=SimpleNamespace(
            clear_replay_on_start=True,
            batch_size=4,
            replay_window=100,
        ),
        game_config=get_config(env_id),
        stats=SimpleNamespace(replay_buffer_size=0),
        _buffer_size_cache=0,
        _replay_cleanup_every=10,
    )


def trainer_for(env_id="connect4"):
    return FakeTrainer(get_config(env_id))


def test_matching_metadata_passes_and_leaves_config_untouched():
    config = get_config("connect4")
    trainer = FakeTrainer(config)

    check_metadata_agrees(trainer, FakeReplay(config), "connect4")

    # The manifest is authoritative; the DB must never overwrite it.
    assert trainer.game_config is config


def test_missing_metadata_warns_but_proceeds(caplog):
    trainer = trainer_for()

    with caplog.at_level(logging.WARNING):
        check_metadata_agrees(trainer, FakeReplay(None), "connect4")

    assert "No metadata in database" in caplog.text


@pytest.mark.parametrize(
    "field",
    ["obs_size", "legal_mask_offset", "num_actions", "board_width", "board_height"],
)
def test_layout_mismatch_raises(field):
    """A layout disagreement must stop training, not be logged and ignored.

    Training on transitions whose observation layout differs from the network's
    produces a model from misread inputs — silently, for as long as the run
    lasts.
    """
    config = get_config("connect4")
    trainer = FakeTrainer(config)
    stale = replace(config, **{field: getattr(config, field) + 1})

    with pytest.raises(MetadataMismatch) as excinfo:
        check_metadata_agrees(trainer, FakeReplay(stale), "connect4")

    assert field in str(excinfo.value)


def test_channel_count_change_at_identical_obs_size_raises():
    """The dimension-aliasing case: same obs_size, different plane count.

    obs_channels is not a database column, but it is determined by ones that
    are. A game that kept its obs_size while redistributing it over a different
    number of planes would otherwise pass every stored-field comparison, and
    the network would reshape the observation wrongly with no error.
    """
    config = get_config("generals_8x8")  # 9 planes x 64 = 576
    trainer = FakeTrainer(config)

    # Same obs_size and num_actions, same total plane floats, but 4 planes of a
    # 12x12 board instead of 9 planes of an 8x8 one.
    aliased = replace(config, board_width=12, board_height=12)

    with pytest.raises(MetadataMismatch) as excinfo:
        check_metadata_agrees(trainer, FakeReplay(aliased), "generals_8x8")

    message = str(excinfo.value)
    assert "board_width" in message or "obs_channels" in message


def test_cosmetic_mismatch_only_warns(caplog):
    """Descriptive differences cannot corrupt learning, so they must not raise."""
    config = get_config("connect4")
    trainer = FakeTrainer(config)
    renamed = replace(config, display_name="Connect Four (old build)")

    with caplog.at_level(logging.WARNING):
        check_metadata_agrees(trainer, FakeReplay(renamed), "connect4")

    assert "display_name" in caplog.text
    assert trainer.game_config is config


def test_setup_replay_clears_only_the_requested_environment():
    trainer = lifecycle_trainer()
    replay = LifecycleReplay(trainer.game_config)

    setup_replay(trainer, replay, "connect4")

    assert replay.clear_calls == ["connect4"]
    assert trainer._buffer_size_cache == 8


def test_periodic_cleanup_trims_only_the_requested_environment():
    trainer = lifecycle_trainer()
    replay = LifecycleReplay(trainer.game_config)

    handle_replay_cleanup(trainer, global_step=20, replay=replay, env_id="connect4")

    assert replay.cleanup_calls == [(100, "connect4")]
    assert trainer._buffer_size_cache == 8
