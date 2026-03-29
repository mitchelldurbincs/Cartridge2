"""Factory functions for creating storage backends.

PostgreSQL is the only replay buffer backend.
Model storage supports filesystem (local) and S3 (cloud/MinIO).
"""

import logging
import os
from pathlib import Path

from trainer.storage.base import ModelStore, ReplayBufferBase

logger = logging.getLogger(__name__)


def create_replay_buffer(
    connection_string: str | None = None,
    validate_schema: bool = True,
    **kwargs,
) -> ReplayBufferBase:
    """Create a PostgreSQL replay buffer.

    Args:
        connection_string: PostgreSQL connection string. If None, reads from
                          environment variable.
        validate_schema: Whether to validate/create schema on connect.
        **kwargs: Additional backend-specific options (e.g., pool_size).

    Returns:
        A PostgresReplayBuffer instance.

    Raises:
        ValueError: If connection string is not provided.
        ConnectionError: If PostgreSQL connection fails.

    Environment variables (checked in order):
        CARTRIDGE_STORAGE_POSTGRES_URL
        CARTRIDGE_POSTGRES_URL
    """
    # Get connection string from argument, env, or central config
    if connection_string is None:
        connection_string = os.environ.get(
            "CARTRIDGE_STORAGE_POSTGRES_URL",
            os.environ.get("CARTRIDGE_POSTGRES_URL"),
        )

    if connection_string is None:
        try:
            from trainer.central_config import get_config as get_central_config

            connection_string = get_central_config().storage.postgres_url
        except Exception:
            pass

    if connection_string is None:
        raise ValueError(
            "PostgreSQL connection string required. "
            "Set CARTRIDGE_STORAGE_POSTGRES_URL environment variable "
            "or pass connection_string parameter.\n"
            "Example: postgresql://user:password@localhost:5432/cartridge"
        )

    from trainer.storage.postgres import PostgresReplayBuffer

    logger.info("Connecting to PostgreSQL replay buffer")
    return PostgresReplayBuffer(
        connection_string, validate_schema=validate_schema, **kwargs
    )


def create_model_store(
    backend: str | None = None,
    path: str | Path | None = None,
    bucket: str | None = None,
    endpoint: str | None = None,
    **kwargs,
) -> ModelStore:
    """Create a model store with the specified backend.

    Args:
        backend: Backend type ("filesystem" or "s3"). If None, auto-detects
                 from environment or defaults to "filesystem".
        path: Path to model directory (for filesystem backend).
        bucket: S3 bucket name (for s3 backend).
        endpoint: S3-compatible endpoint URL (for s3 backend, e.g., MinIO).
        **kwargs: Additional backend-specific options.

    Returns:
        A ModelStore implementation.

    Raises:
        ValueError: If backend is unknown or required parameters are missing.

    Environment variables (checked in order):
        CARTRIDGE_STORAGE_MODEL_BACKEND / CARTRIDGE_MODEL_BACKEND: Backend type
        CARTRIDGE_MODEL_DIR: Filesystem path
        CARTRIDGE_STORAGE_S3_BUCKET / CARTRIDGE_S3_BUCKET: S3 bucket
        CARTRIDGE_STORAGE_S3_ENDPOINT / CARTRIDGE_S3_ENDPOINT: S3 endpoint (for MinIO)
    """
    # Auto-detect backend from environment (check both naming conventions)
    if backend is None:
        backend = os.environ.get(
            "CARTRIDGE_STORAGE_MODEL_BACKEND",
            os.environ.get("CARTRIDGE_MODEL_BACKEND", "filesystem"),
        ).lower()

    if backend == "filesystem":
        # Get path from argument, env, or default
        if path is None:
            path = os.environ.get("CARTRIDGE_MODEL_DIR", "./data/models")

        from trainer.storage.filesystem import FilesystemModelStore

        logger.info(f"Using filesystem model store: {path}")
        return FilesystemModelStore(path)

    elif backend == "s3":
        # Get bucket from argument or env (check both naming conventions)
        if bucket is None:
            bucket = os.environ.get(
                "CARTRIDGE_STORAGE_S3_BUCKET",
                os.environ.get("CARTRIDGE_S3_BUCKET"),
            )

        if bucket is None:
            raise ValueError(
                "S3 bucket required. "
                "Set CARTRIDGE_STORAGE_S3_BUCKET or pass bucket parameter."
            )

        # Get endpoint from argument or env (check both naming conventions)
        if endpoint is None:
            endpoint = os.environ.get(
                "CARTRIDGE_STORAGE_S3_ENDPOINT",
                os.environ.get("CARTRIDGE_S3_ENDPOINT"),
            )

        from trainer.storage.s3 import S3ModelStore

        logger.info(f"Using S3 model store: {bucket}")
        return S3ModelStore(bucket, endpoint=endpoint, **kwargs)

    else:
        raise ValueError(
            f"Unknown model store backend: {backend}. "
            "Supported backends: filesystem, s3"
        )
