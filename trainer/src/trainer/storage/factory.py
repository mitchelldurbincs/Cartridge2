"""Factory for the profile-bound PostgreSQL replay store."""

import logging
import os

from trainer.storage.base import ReplaySelection, ReplayStore

logger = logging.getLogger(__name__)


def create_replay_store(
    selection: ReplaySelection,
    connection_string: str | None = None,
    validate_schema: bool = True,
    **kwargs,
) -> ReplayStore:
    """Create a PostgreSQL replay store.

    Args:
        selection: Exact profile, collection scope, and source checkpoint to expose.
        connection_string: PostgreSQL connection string. If None, reads from
                          environment variable.
        validate_schema: Whether to validate/create schema on connect.
        **kwargs: Additional backend-specific options (e.g., pool_size).

    Returns:
        A PostgresReplayStore instance.

    Raises:
        ValueError: If connection string is not provided.
        ConnectionError: If PostgreSQL connection fails.

    Environment variable:
        CARTRIDGE_STORAGE_POSTGRES_URL
    """
    # Get connection string from argument or environment
    if connection_string is None:
        connection_string = os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL")

    if connection_string is None:
        raise ValueError(
            "PostgreSQL connection string required. "
            "Set CARTRIDGE_STORAGE_POSTGRES_URL environment variable "
            "or pass connection_string parameter.\n"
            "Example: postgresql://user:password@localhost:5432/cartridge"
        )

    from trainer.storage.postgres import PostgresReplayStore

    logger.info("Connecting to PostgreSQL replay store")
    return PostgresReplayStore(
        connection_string,
        selection=selection,
        validate_schema=validate_schema,
        **kwargs,
    )
