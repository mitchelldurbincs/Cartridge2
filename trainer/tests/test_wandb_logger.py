"""Tests for the wandb logger wrapper.

The wrapper must never break training: every wandb failure mode falls back
to a null logger (unless required=true, which escalates). The active-logger
path is tested against a fake in-memory wandb module — no network, no
account, no real wandb import needed.
"""

import subprocess
import types
from types import SimpleNamespace

import pytest

from trainer.wandb_logger import (
    _ActiveLogger,
    _git_commit_short,
    _normalize_tags,
    _NullLogger,
    make_logger,
)


def wandb_config(**overrides) -> SimpleNamespace:
    """WandbConfig stand-in (the wrapper duck-types the dataclass)."""
    defaults = {
        "enabled": False,
        "required": False,
        "project": "cartridge2",
        "entity": "",
        "group": "",
        "tags": [],
        "init_timeout_seconds": 30.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_fake_wandb(init_exception: Exception | None = None) -> types.SimpleNamespace:
    """In-memory wandb stand-in recording init/log/finish calls."""
    calls = {"init": [], "log": [], "finish": 0}

    def init(**kwargs):
        if init_exception is not None:
            raise init_exception
        calls["init"].append(kwargs)
        return SimpleNamespace(url="https://wandb.test/run/abc123")

    def log(metrics, step=None):
        calls["log"].append((metrics, step))

    def finish():
        calls["finish"] += 1

    fake = types.SimpleNamespace(
        init=init,
        log=log,
        finish=finish,
        Settings=lambda **kwargs: kwargs,
        calls=calls,
    )
    return fake


class TestNullFallbacks:
    """Every disabled/unavailable path returns a working null logger."""

    def test_none_config_gives_null(self):
        logger = make_logger(wandb_config=None, run_name="r", run_config={})
        assert isinstance(logger, _NullLogger)
        assert not logger.enabled
        logger.log({"a": 1}, step=1)  # no-ops must not raise
        logger.finish()

    def test_disabled_config_gives_null(self):
        logger = make_logger(
            wandb_config=wandb_config(enabled=False), run_name="r", run_config={}
        )
        assert isinstance(logger, _NullLogger)

    def test_disabled_wins_over_required(self):
        # enabled=false always wins, even with required=true.
        logger = make_logger(
            wandb_config=wandb_config(enabled=False, required=True),
            run_name="r",
            run_config={},
        )
        assert isinstance(logger, _NullLogger)

    def test_wandb_mode_disabled_gives_null(self, monkeypatch):
        monkeypatch.setenv("WANDB_MODE", "disabled")
        logger = make_logger(
            wandb_config=wandb_config(enabled=True), run_name="r", run_config={}
        )
        assert isinstance(logger, _NullLogger)

    def test_wandb_mode_disabled_with_required_raises(self, monkeypatch):
        monkeypatch.setenv("WANDB_MODE", "disabled")
        with pytest.raises(RuntimeError):
            make_logger(
                wandb_config=wandb_config(enabled=True, required=True),
                run_name="r",
                run_config={},
            )

    def test_missing_package_gives_null(self, monkeypatch):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        monkeypatch.setitem(__import__("sys").modules, "wandb", None)
        logger = make_logger(
            wandb_config=wandb_config(enabled=True), run_name="r", run_config={}
        )
        assert isinstance(logger, _NullLogger)

    def test_missing_package_with_required_raises(self, monkeypatch):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        monkeypatch.setitem(__import__("sys").modules, "wandb", None)
        with pytest.raises(RuntimeError):
            make_logger(
                wandb_config=wandb_config(enabled=True, required=True),
                run_name="r",
                run_config={},
            )

    def test_init_failure_gives_null(self, monkeypatch):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        fake = make_fake_wandb(init_exception=ConnectionError("no network"))
        monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
        logger = make_logger(
            wandb_config=wandb_config(enabled=True), run_name="r", run_config={}
        )
        assert isinstance(logger, _NullLogger)

    def test_init_failure_with_required_raises(self, monkeypatch):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        fake = make_fake_wandb(init_exception=ConnectionError("no network"))
        monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
        with pytest.raises(RuntimeError):
            make_logger(
                wandb_config=wandb_config(enabled=True, required=True),
                run_name="r",
                run_config={},
            )


class TestActivePath:
    """Active-logger construction against a fake wandb module."""

    @pytest.fixture
    def fake(self, monkeypatch):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        monkeypatch.delenv("WANDB_ENTITY", raising=False)
        fake = make_fake_wandb()
        monkeypatch.setitem(__import__("sys").modules, "wandb", fake)
        return fake

    def test_active_logger_created_with_init_kwargs(self, fake):
        logger = make_logger(
            wandb_config=wandb_config(enabled=True, group="exp1", tags=["from_config"]),
            run_name="connect4_loop_it001",
            run_config={"env_id": "connect4", "seed": 42},
            tags=("connect4", "alphazero"),
        )
        assert isinstance(logger, _ActiveLogger)
        assert logger.enabled

        assert len(fake.calls["init"]) == 1
        kwargs = fake.calls["init"][0]
        assert kwargs["project"] == "cartridge2"
        assert kwargs["entity"] is None  # empty string -> None
        assert kwargs["name"] == "connect4_loop_it001"
        assert kwargs["group"] == "exp1"
        assert kwargs["tags"] == ["connect4", "alphazero", "from_config"]
        assert kwargs["config"]["env_id"] == "connect4"
        assert kwargs["config"]["seed"] == 42
        assert kwargs["reinit"] == "finish_previous"

    def test_git_commit_injected_into_config(self, fake):
        make_logger(
            wandb_config=wandb_config(enabled=True), run_name="r", run_config={}
        )
        kwargs = fake.calls["init"][0]
        # This repo is a git checkout, so a commit hash must be captured.
        assert kwargs["config"].get("git_commit")

    def test_env_vars_override_config(self, fake, monkeypatch):
        monkeypatch.setenv("WANDB_PROJECT", "env-project")
        monkeypatch.setenv("WANDB_ENTITY", "env-entity")
        make_logger(
            wandb_config=wandb_config(
                enabled=True, project="toml-project", entity="toml-entity"
            ),
            run_name="r",
            run_config={},
        )
        kwargs = fake.calls["init"][0]
        assert kwargs["project"] == "env-project"
        assert kwargs["entity"] == "env-entity"

    def test_log_and_finish_delegate(self, fake):
        logger = make_logger(
            wandb_config=wandb_config(enabled=True), run_name="r", run_config={}
        )
        logger.log({"train/loss": 1.5}, step=10)
        logger.finish()
        assert fake.calls["log"] == [({"train/loss": 1.5}, 10)]
        assert fake.calls["finish"] == 1

    def test_log_and_finish_swallow_exceptions(self, monkeypatch, caplog):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        fake = make_fake_wandb()

        def raising_log(metrics, step=None):
            raise RuntimeError("wandb service crashed")

        def raising_finish():
            raise RuntimeError("wandb service crashed")

        fake.log = raising_log
        fake.finish = raising_finish
        monkeypatch.setitem(__import__("sys").modules, "wandb", fake)

        logger = make_logger(
            wandb_config=wandb_config(enabled=True), run_name="r", run_config={}
        )
        logger.log({"a": 1}, step=1)  # must not raise
        logger.finish()  # must not raise
        assert any("wandb.log failed" in r.message for r in caplog.records)
        assert any("wandb.finish failed" in r.message for r in caplog.records)


class TestHelpers:
    def test_git_commit_short_in_repo(self):
        commit = _git_commit_short()
        assert commit is not None
        assert len(commit) >= 7

    def test_git_commit_short_failure_returns_none(self, monkeypatch):
        def raising(*args, **kwargs):
            raise subprocess.CalledProcessError(128, "git")

        monkeypatch.setattr(subprocess, "check_output", raising)
        assert _git_commit_short() is None

    def test_normalize_tags(self):
        assert _normalize_tags(None) == []
        assert _normalize_tags([]) == []
        assert _normalize_tags(["a", "b"]) == ["a", "b"]
        # Env override case: CARTRIDGE_WANDB_TAGS arrives as one string.
        assert _normalize_tags("a, b ,c") == ["a", "b", "c"]
