"""Byte-level contract tests for the DQN cartridge's replay codec."""

import struct
from dataclasses import replace

import numpy as np
import pytest

from trainer.algorithms.dqn_replay import DqnReplayBatch, decode_replay_batch
from trainer.storage import ReplayProfile, ReplaySelection

PROFILE = ReplayProfile("counter", 1, "dqn_v1", "dqn_transition_v1")
SELECTION = ReplaySelection(PROFILE, "a" * 64, "b" * 64)
# obs=[0, 1], action=1, reward=-0.5, next_obs=[0.25, 0.75],
# terminated=false, truncated=false, next_availability=[false, true].
PAYLOAD = bytes.fromhex("00000000 0000803f 01000000 000000bf 0000803e 0000403f 0000 0001")


def record(encoded=PAYLOAD):
    return SELECTION.record(id="transition", episode_id="episode", step_number=0, payload=encoded)


def decode(records):
    return decode_replay_batch(records, selection=SELECTION, obs_size=2, num_actions=2)


def test_decoder_preserves_little_endian_values_shapes_and_dtypes():
    batch = decode([record()])
    assert isinstance(batch, DqnReplayBatch)
    for array, expected, dtype in (
        (batch.observations, [[0.0, 1.0]], np.float32),
        (batch.actions, [1], np.int64),
        (batch.rewards, [-0.5], np.float32),
        (batch.next_observations, [[0.25, 0.75]], np.float32),
        (batch.terminated, [False], np.bool_),
        (batch.truncated, [False], np.bool_),
        (batch.next_availability, [[False, True]], np.bool_),
    ):
        np.testing.assert_array_equal(array, expected)
        assert array.dtype == dtype


def test_decoder_handles_empty_batches():
    batch = decode([])
    assert batch.observations.shape == (0, 2)
    assert batch.next_observations.shape == (0, 2)
    assert batch.next_availability.shape == (0, 2)
    for array in (batch.actions, batch.rewards, batch.terminated, batch.truncated):
        assert array.shape == (0,)


def test_decoder_preserves_distinct_completion_flags_in_one_batch():
    batch = decode(
        [
            record(),
            record(PAYLOAD[:24] + b"\x01\x00\x00\x00"),
            record(PAYLOAD[:24] + b"\x00\x01\x00\x00"),
        ]
    )
    np.testing.assert_array_equal(batch.terminated, [False, True, False])
    np.testing.assert_array_equal(batch.truncated, [False, False, True])
    np.testing.assert_array_equal(
        batch.next_availability, [[False, True], [False, False], [False, False]]
    )


@pytest.mark.parametrize("size", [0, 27, 29, 32])
def test_decoder_requires_exact_payload_length(size):
    with pytest.raises(ValueError, match=rf"has {size} bytes, expected 28"):
        decode([record(b"\x00" * size)])


@pytest.mark.parametrize(
    ("offset", "replacement", "message"),
    [
        (0, struct.pack("<f", float("nan")), "non-finite observation"),
        (4, struct.pack("<f", float("inf")), "non-finite observation"),
        (12, struct.pack("<f", float("-inf")), "non-finite reward"),
        (16, struct.pack("<f", float("inf")), "non-finite observation"),
        (20, struct.pack("<f", float("nan")), "non-finite observation"),
        (8, struct.pack("<I", 2), "invalid action 2"),
        (24, b"\x02\x00", "invalid completion flags"),
        (24, b"\x00\x02", "invalid completion flags"),
        (24, b"\x01\x01", "invalid completion flags"),
        (26, b"\x02", "invalid availability"),
        (26, b"\x00\x00", "requires non-empty next availability"),
        (24, b"\x01\x00", "requires empty next availability"),
        (24, b"\x00\x01", "requires empty next availability"),
    ],
)
def test_decoder_rejects_invalid_transition_fields(offset, replacement, message):
    encoded = PAYLOAD[:offset] + replacement + PAYLOAD[offset + len(replacement) :]
    with pytest.raises(ValueError, match=message):
        decode([record(encoded)])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("env_id", "other"),
        ("env_contract_version", 2),
        ("algorithm_id", "other_algorithm_v1"),
        ("experience_schema", "other_payload_v1"),
        ("collection_scope_id", "c" * 64),
        ("source_checkpoint_id", "d" * 64),
        ("source_checkpoint_id", None),
    ],
)
def test_decoder_rejects_every_selection_mismatch(field, value):
    with pytest.raises(ValueError, match="crosses its selection fence"):
        decode([replace(record(), **{field: value})])


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (replace(PROFILE, algorithm_id="other_algorithm_v1"), "wrong algorithm"),
        (replace(PROFILE, experience_schema="dqn_transition_v2"), "wrong experience schema"),
    ],
)
def test_decoder_rejects_wrong_cartridge_even_for_empty_batches(profile, message):
    with pytest.raises(ValueError, match=message):
        decode_replay_batch(
            [], selection=replace(SELECTION, profile=profile), obs_size=2, num_actions=2
        )


def test_cartridge_preserves_replay_imports():
    from trainer.algorithms import dqn_v1

    assert dqn_v1.DqnReplayBatch is DqnReplayBatch
    assert dqn_v1.decode_replay_batch is decode_replay_batch
