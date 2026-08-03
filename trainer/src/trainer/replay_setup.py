"""Replay-store setup and maintenance helpers.

Functions take the owning ``AlphaZeroLearner`` as their first argument and
operate on its state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from crucible.backoff import wait_with_backoff as _wait_with_backoff

if TYPE_CHECKING:
    from .trainer import AlphaZeroLearner

logger = logging.getLogger(__name__)


def wait_with_backoff(
    trainer: "AlphaZeroLearner",
    condition_fn,
    description: str,
    check_interval: float | None = None,
) -> None:
    """Wait for a condition with periodic checks and timeout.

    Args:
        trainer: The owning AlphaZero learner.
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


def setup_replay(trainer: "AlphaZeroLearner", replay, env_id: str) -> None:
    """Optionally clear the exact selection, then wait for one record.

    Args:
        trainer: The owning AlphaZero learner.
        replay: Store already bound to the learner's exact replay selection.
        env_id: Environment identifier used in status messages.

    Raises:
        WaitTimeout: If max_wait is exceeded waiting for data.
    """
    if trainer.config.clear_replay_on_start:
        deleted = replay.clear()
        logger.info(f"Cleared {deleted} records from the exact replay selection")

    record_count = replay.count()
    logger.info(f"Replay contains {record_count} records for {env_id}")

    if record_count == 0:
        wait_with_backoff(
            trainer,
            lambda: replay.count() > 0,
            f"at least one usable replay record for {env_id}",
        )
        record_count = replay.count()
        logger.info(f"Replay now has {record_count} records for {env_id}")

    trainer._replay_record_count_cache = record_count
    trainer.stats.replay_record_count = record_count


def handle_replay_cleanup(
    trainer: "AlphaZeroLearner", global_step: int, replay, env_id: str
) -> None:
    """Clean up old replay records if configured.

    Args:
        trainer: The owning AlphaZero learner.
        global_step: Current global step.
        replay: Exact-selection-bound replay store.
        env_id: Environment identifier.
    """
    if (
        trainer.config.replay_window > 0
        and global_step % trainer._replay_cleanup_every == 0
    ):
        deleted = replay.cleanup(trainer.config.replay_window)
        if deleted > 0:
            logger.info(
                f"Replay cleanup removed {deleted} old records "
                f"(window={trainer.config.replay_window})"
            )
        trainer._replay_record_count_cache = replay.count()
        trainer.stats.replay_record_count = trainer._replay_record_count_cache
