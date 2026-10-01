"""Pilot configuration, device failure and bounded subprocess fixtures (no RL)."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from trainer.__main__ import build_parser
from trainer.algorithms import get_algorithm
from trainer.device import resolve_device
from trainer.orchestrator.cli import loop_config_from_args, run_loop
from trainer.orchestrator.config import LoopConfig
from trainer.orchestrator.orchestrator import _run_recipe
from trainer.pilot import smoke_arguments
from trainer.supervise import positive_seconds, supervise


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_explicit_unavailable_device_fails(requested, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        resolve_device(requested)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_cuda_failure_precedes_storage_metrics_and_actors(monkeypatch):
    from trainer import metrics
    from trainer.orchestrator import orchestrator

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        metrics, "start_metrics_server", lambda **kw: pytest.fail("metrics started")
    )
    monkeypatch.setattr(orchestrator, "Orchestrator", lambda *a: pytest.fail("storage opened"))
    assert run_loop(LoopConfig(device="cuda")) == 1


def test_same_frozen_smoke_recipe_for_all_devices(tmp_path, monkeypatch):
    # Configuration portability only; this does not exercise CUDA or MPS.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    algorithm = get_algorithm("alphazero_board_v1")
    recipes = []
    for device in ("cpu", "mps", "cuda"):
        args = build_parser(algorithm).parse_args(smoke_arguments(device, tmp_path / "smoke"))
        config = loop_config_from_args(args)
        assert config.device == device
        assert config.iterations * config.steps_per_iteration == 8
        recipes.append(_run_recipe(config, algorithm))
    assert recipes[0] == recipes[1] == recipes[2]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_deadline_must_be_finite_and_positive(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_seconds(value)


def _alive(pid):
    # Orphans can briefly be zombies until the host/container init reaps them.
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group contract")
@pytest.mark.parametrize("exit_code", [None, 0, 7])
def test_supervisor_kills_stubborn_actor_and_evaluator_after_timeout_or_parent_exit(
    tmp_path, exit_code
):
    """Real processes, including an exited group leader, with no DB/GPU/training."""
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"root = Path({str(tmp_path)!r})\n"
        "for name in ['actor', 'evaluator']:\n"
        "    subprocess.Popen([sys.executable, str(root/'worker.py'), str(root/name)])\n"
        "while not all((root/name).exists() for name in ['actor', 'evaluator']):\n"
        "    time.sleep(.01)\n"
        + ("time.sleep(60)\n" if exit_code is None else f"sys.exit({exit_code})\n")
    )
    started = time.monotonic()
    status = supervise([sys.executable, str(driver)], timeout=1.5, grace=0.15)
    assert status == (124 if exit_code is None else exit_code)
    assert time.monotonic() - started < 5
    for name in ("actor", "evaluator"):
        assert not _alive(int((tmp_path / name).read_text()))


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
def test_supervisor_forwards_shutdown_and_keeps_durable_log(tmp_path):
    ready = tmp_path / "ready"
    log = tmp_path / "pilot.log"
    command = [
        sys.executable,
        "-m",
        "trainer.supervise",
        "--timeout-seconds",
        "20",
        "--grace-seconds",
        ".15",
        "--log-file",
        str(log),
        "--",
        sys.executable,
        "-c",
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text(str(os.getpid())); "
        "print('fixture output', flush=True); time.sleep(60)",
    ]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    parent = subprocess.Popen(command, env=env)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=5) == 143
        assert not _alive(int(ready.read_text()))
        assert "fixture output" in log.read_text()
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


def test_failed_child_spawn_restores_signal_handlers():
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(FileNotFoundError):
        supervise(["/nonexistent/cartridge-pilot-test"], timeout=1, grace=0.1)
    assert signal.getsignal(signal.SIGTERM) is previous


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group contract")
@pytest.mark.parametrize("role", ["actor", "evaluator"])
def test_real_runner_remains_inside_supervised_group(tmp_path, role, monkeypatch):
    """Use production launchers with a stubborn dummy binary, not a Rust/RL run."""
    worker = tmp_path / "worker.py"
    marker = tmp_path / "pid"
    worker.write_text(
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    driver = tmp_path / "driver.py"
    command = repr([sys.executable, str(worker)])
    if role == "actor":
        driver.write_text(
            "from pathlib import Path\n"
            "from trainer.orchestrator.actor_runner import ActorRunner\n"
            "from trainer.orchestrator.config import LoopConfig\n"
            "r = ActorRunner(LoopConfig(num_actors=1), lambda *a: {})\n"
            f"r.find_binary = lambda: Path({sys.executable!r})\n"
            f"r._build_command = lambda *a: {command}\n"
            "r.run(1, 1)\n"
        )
    else:
        driver.write_text(
            f"from trainer.evaluator import run_eval_binary\nrun_eval_binary({command})\n"
        )
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(sys.path))
    assert supervise([sys.executable, str(driver)], timeout=4, grace=0.2) == 124
    assert marker.exists(), "production launcher must have actually started the fixture"
    assert not _alive(int(marker.read_text()))


def test_cpu_preflight_exports_model_before_rust_consumer(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import onnx

    from trainer import evaluator
    from trainer.pilot import preflight

    seen = []

    def consume(player, opponent, **kwargs):
        model = onnx.load(player.model_path)
        onnx.checker.check_model(model)
        assert kwargs["env_id"] == "connect4"
        seen.append(player.model_path)
        return SimpleNamespace(games_played=2)

    monkeypatch.setattr(evaluator, "evaluate", consume)
    preflight("cpu")
    assert len(seen) == 1
