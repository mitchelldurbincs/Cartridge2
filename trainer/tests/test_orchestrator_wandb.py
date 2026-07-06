"""Orchestrator component tests (no Postgres, no wandb).

Covers the W&B logger lifecycle and auto-resume. LoopConfig, StatsManager,
and ActorRunner tests moved to the training-core repo with their
implementations.
"""

import json
from unittest.mock import MagicMock

from trainer.orchestrator import orchestrator as orchestrator_module
from trainer.orchestrator.config import IterationStats, LoopConfig
from trainer.orchestrator.orchestrator import Orchestrator
from trainer.orchestrator.stats_manager import StatsManager


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


def make_orchestrator(tmp_path, monkeypatch, iterations=0, **config_overrides):
    recorder = RecordingLogger()
    monkeypatch.setattr(orchestrator_module, "create_replay_buffer", MagicMock())
    monkeypatch.setattr(orchestrator_module, "make_logger", lambda **kwargs: recorder)

    config = LoopConfig(
        iterations=iterations,
        data_dir=tmp_path,
        env_id="tictactoe",
        **config_overrides,
    )
    return Orchestrator(config), recorder


def make_iteration_stats(iteration):
    return IterationStats(
        iteration=iteration,
        episodes_generated=10,
        transitions_generated=100,
        training_steps=50,
        actor_time_seconds=1.0,
        trainer_time_seconds=2.0,
        eval_time_seconds=0.5,
        total_time_seconds=3.5,
        eval_win_rate=0.6,
        eval_draw_rate=0.1,
    )


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

    def test_run_config_dict_is_json_friendly(self, tmp_path, monkeypatch):
        orchestrator, _ = make_orchestrator(tmp_path, monkeypatch)

        run_config = orchestrator._run_config_dict()

        assert run_config["data_dir"] == str(tmp_path)
        assert run_config["env_id"] == "tictactoe"
        json.dumps(run_config)  # W&B run config must be JSON-serializable


class TestAutoResume:
    def test_auto_resume_picks_up_previous_iterations(self, tmp_path, monkeypatch):
        StatsManager(LoopConfig(data_dir=tmp_path)).save_loop_stats(
            [make_iteration_stats(1), make_iteration_stats(2)]
        )

        orchestrator, _ = make_orchestrator(tmp_path, monkeypatch)

        assert orchestrator.config.start_iteration == 3
        assert [s.iteration for s in orchestrator.iteration_history] == [1, 2]

    def test_auto_resume_restores_eval_history(self, tmp_path, monkeypatch):
        manager = StatsManager(LoopConfig(data_dir=tmp_path))
        manager.save_loop_stats([make_iteration_stats(1)])
        manager.save_eval_stats([{"iteration": 1, "vs_best_win_rate": 0.6}])

        orchestrator, _ = make_orchestrator(tmp_path, monkeypatch)

        assert orchestrator.eval_history == [{"iteration": 1, "vs_best_win_rate": 0.6}]

    def test_explicit_start_iteration_skips_auto_resume(self, tmp_path, monkeypatch):
        StatsManager(LoopConfig(data_dir=tmp_path)).save_loop_stats(
            [make_iteration_stats(1)]
        )

        orchestrator, _ = make_orchestrator(tmp_path, monkeypatch, start_iteration=5)

        assert orchestrator.config.start_iteration == 5
        assert orchestrator.iteration_history == []
