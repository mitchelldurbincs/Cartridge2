"""Thin wandb wrapper that no-ops when disabled, missing, or failing.

Training code calls :func:`make_logger` once, then :meth:`WandbLogger.log`
repeatedly, then :meth:`WandbLogger.finish` in a finally block. If wandb is
unavailable or disabled, a null logger is returned and training continues
unchanged.

Disable wandb via ``WANDB_MODE=disabled`` or ``enabled = false`` in the
``[wandb]`` section of config.toml. Override the default project/entity via
``WANDB_PROJECT`` / ``WANDB_ENTITY`` env vars or the matching config fields.
Set ``required = true`` to fail loudly instead of falling back silently.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .central_config import WandbConfig

logger = logging.getLogger(__name__)

DEFAULT_PROJECT = "cartridge2"


class WandbLogger(Protocol):
    enabled: bool

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None: ...
    def finish(self) -> None: ...


class _NullLogger:
    enabled = False

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        return

    def finish(self) -> None:
        return


class _ActiveLogger:
    enabled = True

    def __init__(self, run: Any) -> None:
        self._run = run

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        import wandb

        try:
            wandb.log(dict(metrics), step=step)
        except Exception as exc:
            logger.warning(f"wandb.log failed ({exc}); continuing")

    def finish(self) -> None:
        import wandb

        try:
            wandb.finish()
        except Exception as exc:
            logger.warning(f"wandb.finish failed ({exc})")


def _git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return None


def _normalize_tags(tags: Any) -> list[str]:
    """Accept a list of tags or a comma-separated string (env override case)."""
    if not tags:
        return []
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return [str(t) for t in tags]


def make_logger(
    *,
    wandb_config: "WandbConfig | None",
    run_name: str,
    run_config: Mapping[str, Any],
    tags: Sequence[str] = (),
) -> WandbLogger:
    """Construct an active wandb logger or a no-op fallback.

    Falls back to a null logger when wandb is disabled (config or
    ``WANDB_MODE=disabled``), when the ``wandb`` package is missing, or when
    ``wandb.init`` raises (e.g. no API key, network unreachable). With
    ``required = true`` those fallbacks raise RuntimeError instead — except
    ``enabled = false``, which always wins.
    """
    if wandb_config is None or not wandb_config.enabled:
        logger.info("W&B startup: logger=null (disabled by config)")
        return _NullLogger()

    project = os.environ.get("WANDB_PROJECT") or wandb_config.project or DEFAULT_PROJECT
    entity = os.environ.get("WANDB_ENTITY") or wandb_config.entity or None
    group = wandb_config.group or None
    init_timeout = float(wandb_config.init_timeout_seconds)
    required = bool(wandb_config.required)

    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        if required:
            raise RuntimeError("wandb required=true but WANDB_MODE=disabled")
        logger.info(f"W&B startup: project={project!r} entity={entity!r} logger=null")
        return _NullLogger()

    try:
        import wandb
    except ImportError:
        if required:
            raise RuntimeError(
                "wandb required=true but the wandb package is not installed"
            )
        logger.info(f"W&B startup: project={project!r} entity={entity!r} logger=null")
        logger.warning("wandb not installed; metrics will not be logged")
        return _NullLogger()

    all_tags = _normalize_tags(tags) + _normalize_tags(wandb_config.tags)

    enriched_config = dict(run_config)
    commit = _git_commit_short()
    if commit:
        enriched_config.setdefault("git_commit", commit)

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=all_tags or None,
            config=enriched_config,
            settings=wandb.Settings(
                init_timeout=init_timeout,
                login_timeout=init_timeout,
                x_service_wait=min(init_timeout, 30.0),
            ),
            reinit="finish_previous",
        )
    except Exception as exc:
        if required:
            raise RuntimeError(
                f"wandb required=true but wandb.init failed: {exc}"
            ) from exc
        logger.info(f"W&B startup: project={project!r} entity={entity!r} logger=null")
        logger.warning(f"wandb.init failed ({exc}); metrics will not be logged")
        return _NullLogger()

    print(f"[wandb] run_url={getattr(run, 'url', None)}", flush=True)
    logger.info(f"W&B startup: project={project!r} entity={entity!r} logger=active")
    return _ActiveLogger(run)
