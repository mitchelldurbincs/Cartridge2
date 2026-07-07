"""Composition tests: Cartridge2's orchestrator wiring into crucible.

The orchestrator loop and its eval/config/actor modules live in
crucible with injected seams; trainer.orchestrator is this repo's
composition root. Each test here constructs through THIS repo's production
wiring and pins behavior that crucible deliberately does not provide.
They would stay green against crucible alone only if the composition
root silently lost its wiring -- which is exactly the regression they exist
to catch.
"""

import json
from types import SimpleNamespace

from trainer import trainer as trainer_mod
from trainer.central_config import WandbConfig
from trainer.config import TrainerConfig
from trainer.orchestrator import eval_runner as eval_runner_shim
from trainer.orchestrator import orchestrator as orchestrator_module
from trainer.orchestrator.actor_runner import _PROJECT_ROOT
from trainer.orchestrator.actor_runner import ActorRunner as ShimActorRunner
from trainer.orchestrator.config import LoopConfig
from trainer.orchestrator.orchestrator import Orchestrator, TrainSpec
from trainer.solver_eval import SolverEvalResults, judge_move
from trainer.structured_logging import get_trace_context


class StubPolicy:
    def __init__(self, model_path, temperature=0.0):
        self.model_path = str(model_path)
        self.temperature = temperature

    @property
    def name(self):
        return f"Stub({self.model_path})"

    def select_action(self, state, config):
        return state.legal_moves()[0]


class FakeReplayBuffer:
    """ReplayBuffer stand-in for the composition root's factory seam."""

    def __init__(self, transitions: int = 0):
        self.transitions = transitions
        self.clear_calls = 0
        self.vacuum_calls = 0
        self.close_calls = 0

    def clear_transitions(self) -> int:
        self.clear_calls += 1
        return 0

    def vacuum(self) -> None:
        self.vacuum_calls += 1

    def count(self, env_id: str) -> int:
        return self.transitions

    def close(self) -> None:
        self.close_calls += 1


def make_stub_trainer(monkeypatch):
    """Replace trainer.trainer.Trainer (the outermost trainer seam).

    The composition root's trainer factory resolves Trainer at call time,
    so patching the source module stubs every trainer the loop builds.
    Returns the list the stubs are recorded into.
    """
    built = []

    class StubTrainer:
        def __init__(self, config):
            self.config = config
            built.append(self)

        def train(self):
            return SimpleNamespace(total_loss=0.25)

    monkeypatch.setattr(trainer_mod, "Trainer", StubTrainer)
    return built


def make_solver_results(
    value_optimal_rate: float, positions: int = 10
) -> SolverEvalResults:
    """A real SolverEvalResults with a chosen overall value-optimal rate."""
    results = SolverEvalResults(
        env_id="connect4",
        model_name="Stub(latest.onnx)",
        model_path="latest.onnx",
        step=None,
        opponent_name="Random",
        games=6,
        seed=42,
        temperature=0.0,
    )
    optimal = round(positions * value_optimal_rate)
    for i in range(positions):
        if i < optimal:
            results.overall.add(judge_move({3: 5, 0: -1}, chosen=3))  # optimal
        else:
            results.overall.add(judge_move({3: 5, 0: -1}, chosen=0))  # blunder
    results.model_wins = 4
    results.model_losses = 2
    results.timestamp = "2026-07-03T00:00:00"
    return results


class TestEvalRunnerComposition:
    def test_eval_temperature_literal_value(self):
        # Byte-equivalence insurance until Gate B: head-to-head eval games
        # must keep sampling at exactly this temperature; a drift in
        # crucible would silently change Cartridge2's eval behavior.
        assert eval_runner_shim.EVAL_TEMPERATURE == 0.2

    def test_solver_stats_file_written(self, tmp_path, monkeypatch):
        """solver_stats.json is written through the ORCHESTRATOR'S eval runner.

        crucible's default solver-stats appender is a silent no-op, so
        both suites would stay green if the composition root ever stopped
        wiring the shim EvalRunner (which re-pins the real
        ``append_solver_stats``) into the loop. The runner under test is
        the wired Orchestrator's own eval_runner -- the production path --
        with only the model/eval/solver backends and the replay buffer
        stubbed at their module-global seams.
        """
        config = LoopConfig(data_dir=tmp_path, env_id="connect4", eval_games=4)
        config.models_dir.mkdir(parents=True, exist_ok=True)
        (config.models_dir / "latest.onnx").touch()
        (config.models_dir / "best.onnx").touch()

        # The shims read these module globals when constructing/scoring, so
        # patching them before construction stubs every external backend.
        monkeypatch.setattr(eval_runner_shim, "OnnxPolicy", StubPolicy)
        monkeypatch.setattr(
            eval_runner_shim,
            "run_eval",
            lambda **kwargs: SimpleNamespace(
                player1_win_rate=0.4, player2_win_rate=0.6, draw_rate=0.0
            ),
        )
        monkeypatch.setattr(eval_runner_shim, "SolverScorer", lambda: object())
        monkeypatch.setattr(
            eval_runner_shim,
            "solver_evaluate",
            lambda **kwargs: make_solver_results(0.6),
        )
        monkeypatch.setattr(
            orchestrator_module, "create_replay_buffer", FakeReplayBuffer
        )

        runner = Orchestrator(config).eval_runner
        assert type(runner) is eval_runner_shim.EvalRunner

        runner.run(iteration=3, eval_history=[])

        stats_path = config.solver_stats_path
        assert stats_path.exists()
        with open(stats_path) as f:
            stats = json.load(f)
        entry = stats["solver_evaluations"][0]
        assert entry["iteration"] == 3
        assert entry["context"] == "loop"
        assert entry["global_step"] == 3 * config.steps_per_iteration
        assert entry["value_optimal_rate"] == 0.6


class TestTrainSpecConversion:
    def test_make_trainer_maps_spec_field_for_field(self, monkeypatch):
        """The composition root converts a core TrainSpec into a real
        TrainerConfig without renaming, dropping, or defaulting any field."""
        built = make_stub_trainer(monkeypatch)

        def hook(payload, step):
            return None

        def check():
            return False

        spec = TrainSpec(
            model_dir="m",
            stats_path="s",
            env_id="connect4",
            total_steps=7,
            start_step=3,
            batch_size=16,
            learning_rate=0.01,
            checkpoint_interval=9,
            device="cpu",
            max_wait=12.5,
            eval_interval=0,
            lr_total_steps=70,
            shutdown_check=check,
            metrics_hook=hook,
        )

        trainer = orchestrator_module._make_trainer(spec)

        assert built == [trainer]
        tc = trainer.config
        assert isinstance(tc, TrainerConfig)
        assert (tc.model_dir, tc.stats_path, tc.env_id) == ("m", "s", "connect4")
        assert (tc.total_steps, tc.start_step) == (7, 3)
        assert (tc.batch_size, tc.learning_rate) == (16, 0.01)
        assert (tc.checkpoint_interval, tc.device) == (9, "cpu")
        assert (tc.max_wait, tc.eval_interval, tc.lr_total_steps) == (12.5, 0, 70)
        assert tc.shutdown_check is check
        assert tc.metrics_hook is hook


class TestTraceComposition:
    def test_trace_starter_sets_context_and_returns_id(self):
        """The trace seam preserves pre-move behavior: a fresh 32-hex trace
        id is returned AND installed (with a 16-hex span) in this repo's
        structured-logging context."""
        trace_id = orchestrator_module._start_iteration_trace()

        assert len(trace_id) == 32
        ctx = get_trace_context()
        assert ctx["trace_id"] == trace_id
        assert len(ctx["span_id"]) == 16


class TestOrchestratorComposition:
    def test_wired_orchestrator_runs_one_iteration(self, tmp_path, monkeypatch):
        """The full production object graph assembles and one iteration runs.

        All factories are the composition root's real ones; stubs sit only
        at the outermost seams: the replay buffer (module global), Trainer
        (patched at trainer.trainer), a touched-but-never-executed actor
        binary with zero episodes, and W&B on its enabled=False null
        logger. No actor binaries run, no PostgreSQL, no network.
        """
        buffer = FakeReplayBuffer(transitions=5)
        monkeypatch.setattr(orchestrator_module, "create_replay_buffer", lambda: buffer)
        built = make_stub_trainer(monkeypatch)

        fake_binary = tmp_path / "actor-stub.exe"
        fake_binary.touch()
        config = LoopConfig(
            iterations=1,
            episodes_per_iteration=0,  # zero episodes: no process ever spawns
            steps_per_iteration=2,
            data_dir=tmp_path / "data",
            env_id="tictactoe",
            actor_binary=fake_binary,  # must exist; never executed
            eval_interval=1,  # eval runs; latest.onnx absent -> real early return
        )

        orchestrator = Orchestrator(config)

        # Production graph: the shim subclasses, not fakes or core classes.
        assert type(orchestrator.actor_runner) is ShimActorRunner
        assert type(orchestrator.eval_runner) is eval_runner_shim.EvalRunner

        orchestrator.run()

        # One iteration completed end to end and was persisted.
        assert [s.iteration for s in orchestrator.iteration_history] == [1]
        stats = orchestrator.iteration_history[0]
        assert stats.transitions_generated == 5
        assert stats.eval_win_rate is None  # eval ran without a model
        assert buffer.clear_calls == 1
        assert buffer.vacuum_calls == 1
        assert buffer.close_calls == 1
        with open(config.loop_stats_path) as f:
            saved = json.load(f)
        assert [it["iteration"] for it in saved["iterations"]] == [1]

        # TrainSpec -> TrainerConfig conversion carried the loop's values.
        (trainer,) = built
        tc = trainer.config
        assert isinstance(tc, TrainerConfig)
        assert tc.model_dir == str(config.models_dir)
        assert tc.stats_path == str(config.stats_path)
        assert tc.env_id == "tictactoe"
        assert tc.total_steps == 2
        assert tc.start_step == 0
        assert tc.max_wait == 60.0
        assert tc.eval_interval == 0  # trainer's built-in eval stays off
        assert tc.lr_total_steps == 2  # iterations * steps_per_iteration
        assert tc.shutdown_check() is False
        assert tc.metrics_hook is not None


class TestLoopConfigComposition:
    def test_wandb_default_is_real_wandb_config(self):
        """The config shim must keep restoring this repo's WandbConfig default.

        crucible's LoopConfig defaults ``wandb`` to None; every
        Cartridge2 caller relies on getting a ready-to-use WandbConfig.
        """
        assert isinstance(LoopConfig().wandb, WandbConfig)


class TestActorRunnerComposition:
    def test_project_root_resolves_to_repo_root(self):
        """Shape-pin the shim's Path(__file__).parents[4] arithmetic.

        The injected cargo-target candidates are computed relative to
        _PROJECT_ROOT; if the shim module ever moves in the tree, this
        catches the stale parents[] math before binary discovery breaks.
        """
        assert (_PROJECT_ROOT / "trainer").is_dir()
