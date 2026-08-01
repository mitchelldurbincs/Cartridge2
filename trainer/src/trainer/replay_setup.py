"""Replay-buffer setup and maintenance helpers.

Extracted from ``trainer.py`` as pure code motion. Functions take the owning
``Trainer`` instance as their first argument and operate on its state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .backoff import wait_with_backoff as _wait_with_backoff

if TYPE_CHECKING:
    from .trainer import Trainer

logger = logging.getLogger(__name__)


# Layout fields: a disagreement here means the observations in the buffer are
# not the shape the network expects, so training on them is meaningless.
_FATAL_FIELDS = ("obs_size", "legal_mask_offset", "num_actions")

# Descriptive fields: worth reporting, but they cannot corrupt learning.
_COSMETIC_FIELDS = ("display_name", "board_width", "board_height")


class MetadataMismatch(RuntimeError):
    """The replay buffer was written by an engine that disagrees with ours."""


def check_metadata_agrees(trainer: "Trainer", replay, env_id: str) -> None:
    """Cross-check the engine's game metadata against the replay buffer's.

    Game facts come from the engine-generated manifest (see ``game_config``),
    not from the database — this only verifies the two agree.

    The actor upserts its metadata row on every startup, so the row reflects
    the actor binary that ran most recently. A layout disagreement therefore
    means the running actor and the installed trainer were built from different
    commits, and the transitions in the buffer do not match the network being
    trained. Raising is the right outcome: the alternative is burning hours of
    compute producing a model from mismatched observations.
    """
    db_metadata = replay.get_metadata(env_id)
    if not db_metadata:
        logger.warning(
            f"No metadata in database for {env_id}; cannot cross-check the replay "
            f"buffer's observation layout against the engine."
        )
        return

    config = trainer.game_config

    mismatches = [
        f"{field}: engine={getattr(config, field)!r} database={getattr(db_metadata, field)!r}"
        for field in _FATAL_FIELDS
        if getattr(config, field) != getattr(db_metadata, field)
    ]
    if mismatches:
        raise MetadataMismatch(
            f"Replay buffer for '{env_id}' was written with a different observation "
            f"layout than this trainer expects: " + "; ".join(mismatches) + ". "
            "The actor and trainer are almost certainly built from different commits. "
            "Rebuild both from the same revision, or clear the buffer."
        )

    cosmetic = [
        f"{field}: engine={getattr(config, field)!r} database={getattr(db_metadata, field)!r}"
        for field in _COSMETIC_FIELDS
        if getattr(config, field) != getattr(db_metadata, field)
    ]
    if cosmetic:
        logger.warning(
            f"Game metadata for {env_id} differs from the database in descriptive "
            f"fields (training is unaffected): " + "; ".join(cosmetic)
        )
    else:
        logger.info(f"Game metadata for {env_id} matches the replay buffer")


def wait_with_backoff(
    trainer: "Trainer",
    condition_fn,
    description: str,
    check_interval: float | None = None,
) -> None:
    """Wait for a condition with periodic checks and timeout.

    Args:
        trainer: The owning Trainer instance.
        condition_fn: Callable returning True when condition is met.
        description: Human-readable description for logging.
        check_interval: Override default wait interval.

    Raises:
        WaitTimeout: If max_wait is exceeded (and max_wait > 0).
    """
    interval = check_interval or trainer.config.wait_interval
    _wait_with_backoff(
        condition_fn=condition_fn,
        description=description,
        interval=interval,
        max_wait=trainer.config.max_wait,
        logger=logger,
    )


def setup_replay(trainer: "Trainer", replay, env_id: str) -> None:
    """Set up the replay buffer: clear if needed, load metadata, wait for data.

    Args:
        trainer: The owning Trainer instance.
        replay: The replay buffer instance.
        env_id: Environment identifier for filtering.

    Raises:
        WaitTimeout: If max_wait is exceeded waiting for data.
    """
    if trainer.config.clear_replay_on_start:
        deleted = replay.clear_transitions()
        logger.info(f"Cleared {deleted} transitions from replay buffer before training")

    check_metadata_agrees(trainer, replay, env_id)

    buffer_size = replay.count(env_id=env_id)
    logger.info(f"Replay buffer contains {buffer_size} transitions for {env_id}")

    # Wait for enough data with proper backoff
    if buffer_size < trainer.config.batch_size:
        wait_with_backoff(
            trainer,
            lambda: replay.count(env_id=env_id) >= trainer.config.batch_size,
            f"sufficient data ({trainer.config.batch_size} samples for {env_id})",
        )
        buffer_size = replay.count(env_id=env_id)
        logger.info(f"Replay buffer now has {buffer_size} transitions for {env_id}")

    trainer._buffer_size_cache = buffer_size
    trainer.stats.replay_buffer_size = buffer_size


def handle_replay_cleanup(
    trainer: "Trainer", global_step: int, replay, env_id: str
) -> None:
    """Clean up old replay transitions if configured.

    Args:
        trainer: The owning Trainer instance.
        global_step: Current global step.
        replay: Replay buffer instance.
        env_id: Environment identifier.
    """
    if (
        trainer.config.replay_window > 0
        and global_step % trainer._replay_cleanup_every == 0
    ):
        deleted = replay.cleanup(trainer.config.replay_window)
        if deleted > 0:
            logger.info(
                f"Replay cleanup removed {deleted} old transitions "
                f"(window={trainer.config.replay_window})"
            )
        trainer._buffer_size_cache = replay.count(env_id=env_id)
        trainer.stats.replay_buffer_size = trainer._buffer_size_cache
