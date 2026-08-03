"""Tests for the AlphaZero cartridge's owned replay payload codec."""

from dataclasses import replace

import numpy as np
import pytest

from trainer.algorithms.alphazero_board_v1 import (
    ALGORITHM_ID,
    DESCRIPTOR,
    decode_replay_batch,
)
from trainer.storage import ReplayProfile, ReplaySelection

PROFILE = ReplayProfile(
    env_id="tictactoe",
    env_contract_version=1,
    algorithm_id=ALGORITHM_ID,
    experience_schema=DESCRIPTOR.components.experience_schema,
)
SELECTION = ReplaySelection(PROFILE, "a" * 64, "b" * 64)


def payload(
    observation=(1.0, 2.0, 3.0),
    policy=(0.25, 0.75),
    value=1.0,
) -> bytes:
    return np.asarray((*observation, *policy, value), dtype="<f4").tobytes()


def record(*, record_id="record", encoded=None):
    return SELECTION.record(
        id=record_id,
        episode_id="episode",
        step_number=0,
        payload=payload() if encoded is None else encoded,
    )


def decode(records):
    return decode_replay_batch(
        records,
        selection=SELECTION,
        obs_size=3,
        num_actions=2,
    )


def test_decoder_returns_exact_training_tensors():
    observations, policies, values = decode(
        [
            record(record_id="one"),
            record(
                record_id="two",
                encoded=payload(
                    observation=(-3.0, 0.0, 4.5),
                    policy=(1.0, 0.0),
                    value=-1.0,
                ),
            ),
        ]
    )
    assert observations.dtype == np.float32
    assert policies.dtype == np.float32
    assert values.dtype == np.float32
    np.testing.assert_array_equal(observations, [[1.0, 2.0, 3.0], [-3.0, 0.0, 4.5]])
    np.testing.assert_array_equal(policies, [[0.25, 0.75], [1.0, 0.0]])
    np.testing.assert_array_equal(values, [1.0, -1.0])


def test_decoder_handles_an_empty_batch_without_storage_semantics():
    observations, policies, values = decode([])
    assert observations.shape == (0, 3)
    assert policies.shape == (0, 2)
    assert values.shape == (0,)


@pytest.mark.parametrize("size", [19, 20, 25])
def test_decoder_requires_exact_payload_length(size):
    with pytest.raises(ValueError, match=rf"payload is {size} bytes.*exactly 24 bytes"):
        decode([record(encoded=b"\x00" * size)])


@pytest.mark.parametrize(
    ("values", "index"),
    [
        ((np.nan, 0.0, 0.0, 0.5, 0.5, 0.0), 0),
        ((0.0, 0.0, 0.0, np.inf, 0.0, 0.0), 3),
        ((0.0, 0.0, 0.0, 0.5, 0.5, -np.inf), 5),
    ],
)
def test_decoder_rejects_nonfinite_values_anywhere(values, index):
    encoded = np.asarray(values, dtype="<f4").tobytes()
    with pytest.raises(ValueError, match=rf"non-finite f32.*index {index}"):
        decode([record(encoded=encoded)])


@pytest.mark.parametrize(
    "policy",
    [(-0.01, 1.01), (1.01, -0.01)],
)
def test_decoder_rejects_policy_values_outside_probability_range(policy):
    with pytest.raises(ValueError, match=r"policy\[\d\].*probability in \[0, 1\]"):
        decode([record(encoded=payload(policy=policy))])


@pytest.mark.parametrize("policy", [(0.2, 0.2), (0.6, 0.6)])
def test_decoder_rejects_policy_that_does_not_sum_to_one(policy):
    with pytest.raises(ValueError, match="policy sums to .*expected 1"):
        decode([record(encoded=payload(policy=policy))])


@pytest.mark.parametrize("value", [-1.001, 1.001])
def test_decoder_rejects_value_outside_terminal_range(value):
    with pytest.raises(ValueError, match=r"value target.*expected \[-1, 1\]"):
        decode([record(encoded=payload(value=value))])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("env_id", "other"),
        ("env_contract_version", 2),
        ("algorithm_id", "other_algorithm_v1"),
        ("experience_schema", "other_payload_v1"),
        ("collection_scope_id", "c" * 64),
        ("source_checkpoint_id", "d" * 64),
    ],
)
def test_decoder_rejects_records_from_every_other_profile(field, value):
    wrong = replace(record(), **{field: value})
    with pytest.raises(ValueError, match="belongs to selection .*expected"):
        decode([wrong])


@pytest.mark.parametrize(
    "profile",
    [
        replace(PROFILE, algorithm_id="other_algorithm_v1"),
        replace(PROFILE, experience_schema="alphazero_transition_v2"),
    ],
)
def test_decoder_itself_requires_the_declared_cartridge_profile(profile):
    with pytest.raises(ValueError, match="AlphaZero replay decoder requires profile"):
        decode_replay_batch(
            [],
            selection=replace(SELECTION, profile=profile),
            obs_size=3,
            num_actions=2,
        )


@pytest.mark.parametrize(("obs_size", "num_actions"), [(0, 2), (3, 0)])
def test_decoder_rejects_nonpositive_contract_dimensions(obs_size, num_actions):
    with pytest.raises(ValueError, match="must be positive"):
        decode_replay_batch(
            [],
            selection=SELECTION,
            obs_size=obs_size,
            num_actions=num_actions,
        )
