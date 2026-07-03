"""Atomic file-write helpers.

Shared by checkpoint.py (the trainer's direct checkpoint path) and
storage/filesystem.py (the ModelStore backend), which both rely on
write-to-temp-then-rename so readers never observe partial files.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable


def atomic_write(dest: Path, write: Callable[[str], None]) -> None:
    """Write to a temp file in dest's directory, then atomically rename onto dest.

    The ``write`` callback receives the temp file path to write into. On any
    failure the temp file is removed and the exception re-raised, leaving any
    existing ``dest`` untouched.
    """
    temp_fd, temp_path = tempfile.mkstemp(suffix=dest.suffix, dir=dest.parent)
    os.close(temp_fd)
    try:
        write(temp_path)
        os.replace(temp_path, dest)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def atomic_copy(src: Path, dest: Path) -> None:
    """Copy ``src`` over ``dest`` atomically (temp copy in dest's directory)."""
    atomic_write(dest, lambda temp_path: shutil.copy2(src, temp_path))
