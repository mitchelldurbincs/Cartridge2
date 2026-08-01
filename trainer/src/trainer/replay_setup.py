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
#
# board_width/board_height belong here, not among the descriptive fields: the
# ResNet reshapes the leading slice using board dimensions, and they are what
# obs_channels is derived from below. Different dimensions with the same
# obs_size reinterpret every plane.
_FATAL_FIELDS = (
    "obs_size",
    "legal_mask_offset",
    "num_actions",
    "board_width",
    "board_height",
)

# Descriptive fields: worth reporting, but they cannot corrupt learning.
_COSMETIC_FIELDS = ("display_name",)


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

    **What this does not do: validate replay lineage.** The metadata row is
    mutable and describes the *current* actor, while individual transitions
    carry no record of the schema that produced them. So this cannot detect
    transitions written by an earlier actor whose encoding differed, whether
    they sit alone in the buffer or mixed in with current ones — a new actor
    simply overwrites the row and the check then passes.

    Nor can it see changes that leave every stored dimension identical:
    ``player_relative_obs`` is not a database column, and a same-width change
    in what the planes *mean* is invisible here. (The realistic version of that
    change also alters the channel count, which the derived check below does
    catch.)

    Closing those gaps needs an immutable per-transition schema identifier —
    covering observation encoding, algorithm, payload schema and target
    semantics — with replay sampling filtered by it. That belongs with the
    general transition envelope, not here; a second mutable column on this row
    would narrow one hole while implying a lineage guarantee that does not
    exist. Until then, the practical mitigation is the loop's
    ``clear_replay_on_start``, which drops the buffer between iterations.
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
    # obs_channels is not a database column, but it is fully determined by
    # columns that are: the board planes occupy [0, obs_channels * board_size).
    # Deriving it here covers the case where a game keeps its obs_size while
    # redistributing it over a different number of planes.
    db_board_size = db_metadata.board_width * db_metadata.board_height
    if db_board_size and db_metadata.legal_mask_offset % db_board_size == 0:
        db_channels = db_metadata.legal_mask_offset // db_board_size
        if db_channels != config.obs_channels:
            mismatches.append(
                f"obs_channels: engine={config.obs_channels!r} "
                f"database={db_channels!r} (derived from legal_mask_offset/board_size)"
            )

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
