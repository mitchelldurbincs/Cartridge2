"""Delete replay scopes that no live training selection will ever read again.

Under `scoped_fresh_iteration_v1` every iteration collects into a brand-new
collection scope and a failed attempt gets a new scope rather than reusing the
old one, so scopes older than the newest few are dead by construction:
training never re-reads them, and the per-selection `clear()`/`cleanup()`
operations cannot reach them. Without this reaper the replay table grows
without bound and every prior iteration's rows stay behind forever.

The reaper runs after each successful RunCommit (fail-closed: an error
propagates to the orchestrator rather than being swallowed), keeps the newest
`retained_scopes` scopes of the profile, deletes everything older — registry
row and replay rows together — and vacuums the table so the reclaimed space
is actually reusable and the planner statistics stay honest.
"""

from __future__ import annotations

import logging
import os

from trainer.storage.base import ReplayProfile

logger = logging.getLogger(__name__)

_PROFILE_WHERE = """
    env_id = %s AND env_contract_version = %s
    AND algorithm_id = %s AND experience_schema = %s
"""


def reap_profile_scopes(
    *,
    profile: ReplayProfile,
    retained_scopes: int,
    connection_string: str | None = None,
) -> int:
    """Delete all but the newest ``retained_scopes`` scopes of ``profile``.

    Returns the number of replay records deleted. Deletion order matters:
    replay rows first, then their registry rows, in one transaction — the
    foreign key from replay_records means a registry row can never disappear
    while its records remain.
    """
    if not isinstance(profile, ReplayProfile):
        raise TypeError("profile must be ReplayProfile")
    if isinstance(retained_scopes, bool) or not isinstance(retained_scopes, int):
        raise ValueError("retained_scopes must be a positive integer")
    if retained_scopes < 1:
        raise ValueError("retained_scopes must be a positive integer")

    import psycopg2

    dsn = connection_string or os.environ.get("CARTRIDGE_STORAGE_POSTGRES_URL")
    if not dsn:
        raise ValueError(
            "PostgreSQL connection string required. Set "
            "CARTRIDGE_STORAGE_POSTGRES_URL or pass connection_string."
        )
    profile_params = (
        profile.env_id,
        profile.env_contract_version,
        profile.algorithm_id,
        profile.experience_schema,
    )
    conn = psycopg2.connect(dsn, connect_timeout=10)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT scope_id FROM collection_scopes
                    WHERE {_PROFILE_WHERE}
                    ORDER BY created_at DESC, scope_id DESC
                    OFFSET %s
                    """,
                    (*profile_params, retained_scopes),
                )
                victims = [row[0] for row in cur.fetchall()]
                if not victims:
                    return 0
                cur.execute(
                    f"""
                    DELETE FROM replay_records
                    WHERE {_PROFILE_WHERE}
                      AND collection_scope_id = ANY(%s)
                    """,
                    (*profile_params, victims),
                )
                deleted_records = cur.rowcount
                cur.execute(
                    "DELETE FROM collection_scopes WHERE scope_id = ANY(%s)",
                    (victims,),
                )
        # VACUUM cannot run inside a transaction block; the bulk delete above
        # is exactly the workload that leaves the table bloated otherwise.
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("VACUUM (ANALYZE) replay_records")
        logger.info(
            "Reaped %d replay scope(s) (%d records) for %s/%s v%d",
            len(victims),
            deleted_records,
            profile.algorithm_id,
            profile.env_id,
            profile.env_contract_version,
        )
        return deleted_records
    finally:
        conn.close()
