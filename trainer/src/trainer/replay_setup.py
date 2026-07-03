"""Replay-buffer setup and maintenance helpers.

Extracted from ``trainer.py`` as pure code motion. Functions take the owning
``Trainer`` instance as their first argument and operate on its state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .backoff import wait_with_backoff as _wait_with_backoff
from .game_config import GameConfig

if TYPE_CHECKING:
    from .trainer import Trainer

logger = logging.getLogger(__name__)


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

    # Try to get game metadata from database (preferred, self-describing)
    db_metadata = replay.get_metadata(env_id)
    if db_metadata:
        logger.info(f"Using game metadata from database for {env_id}")
        trainer.game_config = GameConfig(
            env_id=db_metadata.env_id,
            display_name=db_metadata.display_name,
            board_width=db_metadata.board_width,
            board_height=db_metadata.board_height,
            num_actions=db_metadata.num_actions,
            obs_size=db_metadata.obs_size,
            legal_mask_offset=db_metadata.legal_mask_offset,
        )
    else:
        logger.warning(f"No metadata in database for {env_id}, using fallback config")

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
