"""Orchestrator component tests (no Postgres, no wandb, no real actor binary).

Covers the W&B logger lifecycle, auto-resume, LoopConfig helpers, ActorRunner
subprocess handling (via a fake actor executable), and StatsManager
persistence.
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from trainer.orchestrator import orchestrator as orchestrator_module
from trainer.orchestrator.actor_runner import ActorRunner
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


class TestLoopConfig:
    def test_model_paths_derive_from_data_dir(self, tmp_path):
        config = LoopConfig(data_dir=tmp_path)

        assert config.latest_model_path == tmp_path / "models" / "latest.onnx"
        assert config.best_model_path == tmp_path / "models" / "best.onnx"

    def test_explicit_device_passes_through(self):
        assert LoopConfig(device="cpu").resolve_device() == "cpu"

    def test_num_simulations_ramp_and_cap(self):
        config = LoopConfig(
            mcts_start_sims=50, mcts_max_sims=400, mcts_sim_ramp_rate=20
        )

        assert config.get_num_simulations(1) == 50
        assert config.get_num_simulations(3) == 90
        assert config.get_num_simulations(1000) == 400


# --- ActorRunner ---


def make_fake_actor(tmp_path, exit_code=0, sleep_secs=0.0):
    """A stand-in actor executable that records its argv and env, then exits.

    Writes one file per --actor-id under <tmp_path>/runs: args_<id> holds the
    argv (one arg per line) and env_<id> holds CARTRIDGE_TRACE_ID.
    """
    record_dir = tmp_path / "runs"
    record_dir.mkdir(exist_ok=True)
    script = tmp_path / "fake_actor.sh"
    ending = f"exec sleep {sleep_secs}" if sleep_secs else f"exit {exit_code}"
    script.write_text(
        "#!/bin/sh\n"
        "actor_id=unknown\n"
        'prev=""\n'
        'for a in "$@"; do\n'
        '  if [ "$prev" = "--actor-id" ]; then actor_id="$a"; fi\n'
        '  prev="$a"\n'
        "done\n"
        f'printf \'%s\\n\' "$@" > "{record_dir}/args_$actor_id"\n'
        f'printf \'%s\\n\' "$CARTRIDGE_TRACE_ID" > "{record_dir}/env_$actor_id"\n'
        'echo "fake actor $actor_id running"\n'
        f"{ending}\n"
    )
    script.chmod(0o755)
    return script, record_dir


def make_actor_runner(tmp_path, num_actors=1, shutdown_check=None, **fake_kwargs):
    script, record_dir = make_fake_actor(tmp_path, **fake_kwargs)
    config = LoopConfig(data_dir=tmp_path, actor_binary=script, num_actors=num_actors)
    return ActorRunner(config, shutdown_check=shutdown_check), record_dir


def read_args(record_dir, actor_id):
    return (record_dir / f"args_{actor_id}").read_text().splitlines()


def arg_value(args, flag):
    return args[args.index(flag) + 1]


class TestActorRunnerFindBinary:
    def test_explicit_config_path_used_and_cached(self, tmp_path):
        script, _ = make_fake_actor(tmp_path)
        runner = ActorRunner(LoopConfig(data_dir=tmp_path, actor_binary=script))

        assert runner.find_binary() == script

        # Cached: a later call must not re-validate the config path.
        runner.config.actor_binary = tmp_path / "now-missing"
        assert runner.find_binary() == script

    def test_explicit_config_path_missing_raises(self, tmp_path):
        runner = ActorRunner(
            LoopConfig(data_dir=tmp_path, actor_binary=tmp_path / "missing")
        )

        with pytest.raises(FileNotFoundError, match="Configured actor binary"):
            runner.find_binary()

    def test_env_var_used_when_config_unset(self, tmp_path, monkeypatch):
        script, _ = make_fake_actor(tmp_path)
        monkeypatch.setenv("ACTOR_BINARY", str(script))

        runner = ActorRunner(LoopConfig(data_dir=tmp_path))

        assert runner.find_binary() == script

    def test_env_var_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ACTOR_BINARY", str(tmp_path / "nope"))

        with pytest.raises(FileNotFoundError, match="ACTOR_BINARY not found"):
            ActorRunner(LoopConfig(data_dir=tmp_path)).find_binary()

    def test_auto_detect_first_existing_candidate(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ACTOR_BINARY", raising=False)
        script, _ = make_fake_actor(tmp_path)
        monkeypatch.setattr(
            ActorRunner,
            "_auto_detect_candidates",
            lambda self: [tmp_path / "missing", script],
        )

        runner = ActorRunner(LoopConfig(data_dir=tmp_path))

        assert runner.find_binary() == script

    def test_no_binary_anywhere_raises_with_searched_paths(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ACTOR_BINARY", raising=False)
        monkeypatch.setattr(
            ActorRunner,
            "_auto_detect_candidates",
            lambda self: [tmp_path / "a", tmp_path / "b"],
        )

        with pytest.raises(FileNotFoundError, match="Actor binary not found"):
            ActorRunner(LoopConfig(data_dir=tmp_path)).find_binary()


class TestActorRunnerRun:
    def test_single_actor_success(self, tmp_path, capfd):
        runner, record_dir = make_actor_runner(tmp_path)

        success, elapsed = runner.run(num_episodes=5, iteration=1, trace_id="trace-abc")

        assert success is True
        assert elapsed >= 0.0

        args = read_args(record_dir, "actor-1")
        assert arg_value(args, "--max-episodes") == "5"
        assert arg_value(args, "--env-id") == "tictactoe"
        assert arg_value(args, "--actor-id") == "actor-1"
        assert "--no-watch" in args

        # Trace context is forwarded through the subprocess environment.
        assert (record_dir / "env_actor-1").read_text().strip() == "trace-abc"

        # Actor output is streamed to stdout with the actor-id prefix.
        out, _ = capfd.readouterr()
        assert "[actor-1] fake actor actor-1 running" in out

    def test_episodes_split_with_remainder_to_first_actor(self, tmp_path):
        runner, record_dir = make_actor_runner(tmp_path, num_actors=3)

        success, _ = runner.run(num_episodes=7, iteration=1)

        assert success is True
        episodes = {
            actor_id: int(arg_value(read_args(record_dir, actor_id), "--max-episodes"))
            for actor_id in ("actor-1", "actor-2", "actor-3")
        }
        assert episodes == {"actor-1": 3, "actor-2": 2, "actor-3": 2}

    def test_zero_episode_actors_not_spawned(self, tmp_path):
        runner, record_dir = make_actor_runner(tmp_path, num_actors=3)

        success, _ = runner.run(num_episodes=1, iteration=1)

        assert success is True
        assert (record_dir / "args_actor-1").exists()
        assert not (record_dir / "args_actor-2").exists()
        assert not (record_dir / "args_actor-3").exists()

    def test_mcts_simulations_follow_ramp_schedule(self, tmp_path):
        runner, record_dir = make_actor_runner(tmp_path)

        runner.run(num_episodes=1, iteration=3)

        args = read_args(record_dir, "actor-1")
        expected = runner.config.get_num_simulations(3)
        assert int(arg_value(args, "--num-simulations")) == expected

    def test_no_trace_id_leaves_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CARTRIDGE_TRACE_ID", raising=False)
        runner, record_dir = make_actor_runner(tmp_path)

        runner.run(num_episodes=1, iteration=1)

        assert (record_dir / "env_actor-1").read_text().strip() == ""

    def test_nonzero_exit_reports_failure(self, tmp_path):
        runner, _ = make_actor_runner(tmp_path, exit_code=3)

        success, _ = runner.run(num_episodes=2, iteration=1)

        assert success is False

    def test_one_failing_actor_fails_the_batch(self, tmp_path):
        # Both actors run the same failing script; the batch must report failure.
        runner, _ = make_actor_runner(tmp_path, num_actors=2, exit_code=1)

        success, _ = runner.run(num_episodes=4, iteration=1)

        assert success is False

    def test_shutdown_terminates_running_actors(self, tmp_path):
        runner, _ = make_actor_runner(
            tmp_path, sleep_secs=30, shutdown_check=lambda: True
        )

        start = time.time()
        success, _ = runner.run(num_episodes=1, iteration=1)

        assert success is False
        assert time.time() - start < 10  # terminated, not waited out


# --- StatsManager ---


class TestStatsManager:
    def _manager(self, tmp_path, **overrides):
        return StatsManager(LoopConfig(data_dir=tmp_path, **overrides))

    def test_save_loop_stats_schema(self, tmp_path):
        manager = self._manager(tmp_path, env_id="connect4")

        manager.save_loop_stats([make_iteration_stats(1)])

        data = json.loads(manager.config.loop_stats_path.read_text())
        assert data["config"]["env_id"] == "connect4"
        entry = data["iterations"][0]
        assert entry["iteration"] == 1
        assert entry["episodes"] == 10
        assert entry["transitions"] == 100
        assert entry["eval_win_rate"] == 0.6

    def test_save_and_load_round_trip(self, tmp_path):
        manager = self._manager(tmp_path)
        history = [make_iteration_stats(1), make_iteration_stats(2)]
        manager.save_loop_stats(history)
        manager.save_eval_stats([{"iteration": 1, "vs_best_win_rate": 0.7}])

        loaded_history, eval_history, start_iteration = manager.load_previous_state()

        assert start_iteration == 3
        assert loaded_history == history
        assert eval_history == [{"iteration": 1, "vs_best_win_rate": 0.7}]

    def test_load_previous_state_fresh_dir(self, tmp_path):
        assert self._manager(tmp_path).load_previous_state() == ([], [], 1)

    def test_load_previous_state_no_completed_iterations(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.save_loop_stats([])

        assert manager.load_previous_state() == ([], [], 1)

    def test_corrupt_loop_stats_starts_fresh(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.config.loop_stats_path.write_text("{not json")

        assert manager.load_previous_state() == ([], [], 1)

    def test_corrupt_eval_stats_keeps_iteration_history(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.save_loop_stats([make_iteration_stats(4)])
        manager.config.eval_stats_path.write_text("{not json")

        history, eval_history, start_iteration = manager.load_previous_state()

        assert start_iteration == 5
        assert [s.iteration for s in history] == [4]
        assert eval_history == []

    def test_update_stats_with_eval_writes_frontend_fields(self, tmp_path):
        manager = self._manager(tmp_path)
        manager.config.stats_path.write_text(json.dumps({"policy_loss": 1.2}))
        eval_history = [
            {
                "iteration": 2,
                "step": 2000,
                "vs_best_win_rate": 0.6,
                "vs_best_draw_rate": 0.2,
                "vs_best_opponent_iteration": 1,
                "became_new_best": True,
                "vs_random_win_rate": 0.9,
                "vs_random_draw_rate": 0.05,
                "games": 50,
            }
        ]

        manager.update_stats_with_eval(eval_history, best_iteration=2)

        data = json.loads(manager.config.stats_path.read_text())
        assert data["policy_loss"] == 1.2  # pre-existing keys preserved
        assert len(data["eval_history"]) == 1
        last = data["last_eval"]
        assert last["step"] == 2000
        assert last["opponent"] == "best"
        assert last["opponent_iteration"] == 1
        assert last["win_rate"] == 0.6
        assert last["loss_rate"] == pytest.approx(0.2)
        assert last["became_new_best"] is True
        assert last["games_played"] == 50
        assert data["best_model"] == {
            "iteration": 2,
            "step": 2 * manager.config.steps_per_iteration,
        }

    def test_update_stats_with_eval_step_falls_back_to_iteration(self, tmp_path):
        manager = self._manager(tmp_path)

        manager.update_stats_with_eval([{"iteration": 3}], best_iteration=None)

        data = json.loads(manager.config.stats_path.read_text())
        assert data["last_eval"]["step"] == 3 * manager.config.steps_per_iteration
        assert data["last_eval"]["win_rate"] == 0.0
        assert "best_model" not in data

    def test_update_stats_with_eval_empty_history_is_noop(self, tmp_path):
        manager = self._manager(tmp_path)

        manager.update_stats_with_eval([], best_iteration=1)

        assert not manager.config.stats_path.exists()
