"""Canonical artifact encoding and primitive validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

DIGEST_LENGTH = 64
DIGEST_CHARACTERS = frozenset("0123456789abcdef")
MAX_U32 = (1 << 32) - 1
MAX_U64 = (1 << 64) - 1


class ArtifactValidationError(ValueError):
    """An artifact or checkpoint metadata object violates its contract."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"Value is not canonical JSON: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_exact_fields(
    value: object, expected: frozenset[str], *, context: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{context} must be a JSON object")
    actual = frozenset(value)
    if actual != expected:
        raise ArtifactValidationError(
            f"{context} fields must be exact "
            f"(missing={sorted(expected - actual)}, extra={sorted(actual - expected)})"
        )
    return value


def require_positive_integer(value: object, *, field: str, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or (maximum is not None and value > maximum)
    ):
        raise ArtifactValidationError(f"{field} must be a positive integer")
    return value


def require_nonnegative_integer(value: object, *, field: str, maximum: int | None = None) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or (maximum is not None and value > maximum)
    ):
        raise ArtifactValidationError(f"{field} must be a nonnegative integer")
    return value


def require_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{field} must be a non-empty string")
    return value


def require_digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != DIGEST_LENGTH
        or any(character not in DIGEST_CHARACTERS for character in value)
    ):
        raise ArtifactValidationError(f"{field} must be a lowercase 64-character SHA-256 digest")
    return value


def validate_sha256_digest(value: object, *, field: str) -> str:
    return require_digest(value, field=field)


def decode_canonical_json(data: bytes, *, context: str) -> object:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"{context} is not valid UTF-8 JSON") from exc
    if canonical_json_bytes(value) != data:
        raise ArtifactValidationError(f"{context} is not canonical JSON")
    return value
