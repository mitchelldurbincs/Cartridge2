"""Strict AlphaZero replay decoding, separate from cartridge composition."""

from __future__ import annotations

import numpy as np

from ..storage.base import ReplayRecord, ReplaySelection
from .alphazero_contract import ALGORITHM_ID, DESCRIPTOR

F32_BYTES = 4
POLICY_SUM_TOLERANCE = 1e-3


def decode_replay_batch(
    records: list[ReplayRecord],
    *,
    selection: ReplaySelection,
    obs_size: int,
    num_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the exact ``alphazero_transition_v1`` experience payload.

    Storage deliberately treats ``payload`` as bytes. This cartridge owns the
    language-neutral layout: little-endian f32 observation values, followed by
    a policy distribution and one terminal value target.
    """
    expected_schema = DESCRIPTOR.components.experience_schema
    if (
        selection.profile.algorithm_id != ALGORITHM_ID
        or selection.profile.experience_schema != expected_schema
    ):
        raise ValueError(
            "AlphaZero replay decoder requires profile "
            f"({ALGORITHM_ID!r}, {expected_schema!r}), got "
            f"({selection.profile.algorithm_id!r}, "
            f"{selection.profile.experience_schema!r})"
        )
    if obs_size <= 0:
        raise ValueError("obs_size must be positive")
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")

    expected_values = obs_size + num_actions + 1
    expected_bytes = expected_values * F32_BYTES
    observations = np.empty((len(records), obs_size), dtype=np.float32)
    policy_targets = np.empty((len(records), num_actions), dtype=np.float32)
    value_targets = np.empty(len(records), dtype=np.float32)

    for index, record in enumerate(records):
        if not selection.matches(record):
            actual = (
                record.env_id,
                record.env_contract_version,
                record.algorithm_id,
                record.experience_schema,
            )
            expected = (
                selection.profile.env_id,
                selection.profile.env_contract_version,
                selection.profile.algorithm_id,
                selection.profile.experience_schema,
                selection.collection_scope_id,
                selection.source_checkpoint_id,
            )
            actual += (record.collection_scope_id, record.source_checkpoint_id)
            raise ValueError(
                f"Replay record {record.id!r} belongs to selection {actual}, expected {expected}"
            )
        if len(record.payload) != expected_bytes:
            raise ValueError(
                f"Replay record {record.id!r} payload is {len(record.payload)} bytes; "
                f"{expected_schema} requires exactly {expected_bytes} bytes "
                f"({expected_values} little-endian f32 values)"
            )

        values = np.frombuffer(record.payload, dtype="<f4")
        finite = np.isfinite(values)
        if not finite.all():
            value_index = int(np.flatnonzero(~finite)[0])
            raise ValueError(
                f"Replay record {record.id!r} contains non-finite f32 at "
                f"payload index {value_index}: {values[value_index]}"
            )

        observation = values[:obs_size]
        policy = values[obs_size : obs_size + num_actions]
        value = float(values[-1])
        invalid_policy = (policy < 0.0) | (policy > 1.0)
        if invalid_policy.any():
            policy_index = int(np.flatnonzero(invalid_policy)[0])
            raise ValueError(
                f"Replay record {record.id!r} has policy[{policy_index}]="
                f"{float(policy[policy_index])}, expected a probability in [0, 1]"
            )
        policy_sum = float(policy.sum(dtype=np.float32))
        if abs(policy_sum - 1.0) > POLICY_SUM_TOLERANCE:
            raise ValueError(f"Replay record {record.id!r} policy sums to {policy_sum}, expected 1")
        if not -1.0 <= value <= 1.0:
            raise ValueError(
                f"Replay record {record.id!r} value target is {value}, expected [-1, 1]"
            )

        observations[index] = observation
        policy_targets[index] = policy
        value_targets[index] = value

    return observations, policy_targets, value_targets
