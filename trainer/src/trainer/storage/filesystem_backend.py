"""Small durable filesystem operations used by artifact repositories."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .artifact_codec import ArtifactValidationError


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_or_verify(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ArtifactValidationError(
                f"Immutable checkpoint object exists with different bytes: {path}"
            )
        return
    file_descriptor, temp_name = tempfile.mkstemp(prefix=".staging-", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != data:
                raise ArtifactValidationError(
                    f"Immutable checkpoint object raced with different bytes: {path}"
                )
        else:
            fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(prefix=".pointer-", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


@contextmanager
def directory_lock(path: Path):
    """Hold an advisory lock on a repository directory without extra objects."""
    import fcntl

    path.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
