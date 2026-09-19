"""Strict DQN replay decoding, separate from cartridge composition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..storage.base import ReplayRecord, ReplaySelection
from .dqn_contract import ALGORITHM_ID, DESCRIPTOR


@dataclass(frozen=True)
class DqnReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    next_availability: np.ndarray


def decode_replay_batch(
    records: list[ReplayRecord],
    *,
    selection: ReplaySelection,
    obs_size: int,
    num_actions: int,
) -> DqnReplayBatch:
    """Decode exact ``dqn_transition_v1`` opaque replay records."""
    if selection.profile.algorithm_id != ALGORITHM_ID:
        raise ValueError("DQN replay selection uses the wrong algorithm")
    if selection.profile.experience_schema != DESCRIPTOR.components.experience_schema:
        raise ValueError("DQN replay selection uses the wrong experience schema")
    expected_bytes = obs_size * 8 + 10 + num_actions
    observations = np.empty((len(records), obs_size), dtype=np.float32)
    actions = np.empty(len(records), dtype=np.int64)
    rewards = np.empty(len(records), dtype=np.float32)
    next_observations = np.empty((len(records), obs_size), dtype=np.float32)
    terminated = np.empty(len(records), dtype=np.bool_)
    truncated = np.empty(len(records), dtype=np.bool_)
    next_availability = np.empty((len(records), num_actions), dtype=np.bool_)
    observation_bytes = obs_size * 4

    for index, record in enumerate(records):
        if not selection.matches(record):
            raise ValueError(f"DQN replay record {record.id!r} crosses its selection fence")
        payload = record.payload
        if len(payload) != expected_bytes:
            raise ValueError(
                f"DQN replay record {record.id!r} has {len(payload)} bytes, "
                f"expected {expected_bytes}"
            )
        cursor = 0
        observations[index] = np.frombuffer(payload, dtype="<f4", count=obs_size, offset=cursor)
        cursor += observation_bytes
        action = int.from_bytes(payload[cursor : cursor + 4], "little")
        cursor += 4
        if action >= num_actions:
            raise ValueError(f"DQN replay record {record.id!r} has invalid action {action}")
        actions[index] = action
        reward = np.frombuffer(payload, dtype="<f4", count=1, offset=cursor)[0]
        cursor += 4
        if not np.isfinite(reward):
            raise ValueError(f"DQN replay record {record.id!r} has non-finite reward")
        rewards[index] = reward
        next_observations[index] = np.frombuffer(
            payload, dtype="<f4", count=obs_size, offset=cursor
        )
        cursor += observation_bytes
        flags = payload[cursor : cursor + 2]
        cursor += 2
        if any(flag not in (0, 1) for flag in flags) or flags == b"\x01\x01":
            raise ValueError(f"DQN replay record {record.id!r} has invalid completion flags")
        terminated[index] = bool(flags[0])
        truncated[index] = bool(flags[1])
        availability = np.frombuffer(payload, dtype=np.uint8, offset=cursor)
        if np.any(availability > 1):
            raise ValueError(f"DQN replay record {record.id!r} has invalid availability")
        next_availability[index] = availability.astype(np.bool_)
        done = bool(flags[0] or flags[1])
        if done == bool(availability.any()):
            expectation = "empty" if done else "non-empty"
            raise ValueError(
                f"DQN replay record {record.id!r} requires {expectation} next availability"
            )
        if (
            not np.isfinite(observations[index]).all()
            or not np.isfinite(next_observations[index]).all()
        ):
            raise ValueError(f"DQN replay record {record.id!r} has non-finite observation")

    return DqnReplayBatch(
        observations,
        actions,
        rewards,
        next_observations,
        terminated,
        truncated,
        next_availability,
    )
