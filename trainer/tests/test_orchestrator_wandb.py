"""Orchestrator W&B lifecycle tests (no Postgres, no wandb)."""

from unittest.mock import MagicMock

from trainer.orchestrator import orchestrator as orchestrator_module
from trainer.orchestrator.config import LoopConfig
from trainer.orchestrator.orchestrator import Orchestrator


class RecordingLogger:
    """Implements the WandbLogger protocol, recording all calls."""

    enabled = True

    def __init__(self):
        self.logged = []
        self.finish_calls = 0

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))

    def finish(self):
        self.finish_calls += 1


def make_orchestrator(tmp_path, monkeypatch, iterations=0):
    recorder = RecordingLogger()
    monkeypatch.setattr(orchestrator_module, "create_replay_buffer", MagicMock())
    monkeypatch.setattr(orchestrator_module, "make_logger", lambda **kwargs: recorder)

    config = LoopConfig(
        iterations=iterations,
        data_dir=tmp_path,
        env_id="tictactoe",
    )
    return Orchestrator(config), recorder


class TestWandbLifecycle:
    def test_finish_called_once_on_normal_completion(self, tmp_path, monkeypatch):
        orchestrator, recorder = make_orchestrator(tmp_path, monkeypatch, iterations=0)

        orchestrator.run()

        assert recorder.finish_calls == 1

    def test_finish_called_when_iteration_raises(self, tmp_path, monkeypatch):
        orchestrator, recorder = make_orchestrator(tmp_path, monkeypatch, iterations=1)

        def boom(iteration):
            raise RuntimeError("iteration exploded")

        monkeypatch.setattr(orchestrator, "run_iteration", boom)

        try:
            orchestrator.run()
        except RuntimeError:
            pass

        assert recorder.finish_calls == 1

    def test_train_metrics_hook_maps_payload(self, tmp_path, monkeypatch):
        orchestrator, recorder = make_orchestrator(tmp_path, monkeypatch)

        orchestrator._train_metrics_hook(
            {
                "step": 410,
                "total_loss": 1.5,
                "value_loss": 0.5,
                "policy_loss": 1.0,
                "learning_rate": 0.001,
                "grad_norm": 2.5,
                "samples_seen": 1024,
            },
            410,
        )

        assert len(recorder.logged) == 1
        metrics, step = recorder.logged[0]
        assert step == 410
        assert metrics["train/total_loss"] == 1.5
        assert metrics["train/lr"] == 0.001
        assert metrics["train/grad_norm"] == 2.5
        assert metrics["train/samples_seen"] == 1024
