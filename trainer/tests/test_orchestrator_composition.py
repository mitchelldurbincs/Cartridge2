"""Composition tests: Cartridge2's orchestrator shims wired into training-core.

The orchestrator's eval/config/actor modules now live in training-core with
injected seams; each test here constructs through THIS repo's shims and pins
a default that training-core deliberately does not provide. They would stay
green against training-core alone only if a shim silently lost its wiring --
which is exactly the regression they exist to catch.
"""

import json
from types import SimpleNamespace

from trainer.central_config import WandbConfig
from trainer.orchestrator import eval_runner as eval_runner_shim
from trainer.orchestrator.actor_runner import _PROJECT_ROOT
from trainer.orchestrator.config import LoopConfig
from trainer.solver_eval import SolverEvalResults, judge_move


class StubPolicy:
    def __init__(self, model_path, temperature=0.0):
        self.model_path = str(model_path)
        self.temperature = temperature

    @property
    def name(self):
        return f"Stub({self.model_path})"

    def select_action(self, state, config):
        return state.legal_moves()[0]


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
    def test_solver_stats_file_written(self, tmp_path, monkeypatch):
        """solver_stats.json is written end-to-end through the shim EvalRunner.

        training-core's default solver-stats appender is a silent no-op, so
        both suites would stay green if the shim ever stopped re-pinning the
        real ``append_solver_stats`` (via the eval_reporting shim mixin).
        Only the model/eval/solver backends are stubbed here -- the reporting
        path down to the real file is fully real.
        """
        config = LoopConfig(data_dir=tmp_path, env_id="connect4", eval_games=4)
        config.models_dir.mkdir(parents=True, exist_ok=True)
        (config.models_dir / "latest.onnx").touch()
        (config.models_dir / "best.onnx").touch()

        # The shim reads these module globals when constructing/scoring, so
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
        runner = eval_runner_shim.EvalRunner(config, wandb_logger=None)

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


class TestLoopConfigComposition:
    def test_wandb_default_is_real_wandb_config(self):
        """The config shim must keep restoring this repo's WandbConfig default.

        training-core's LoopConfig defaults ``wandb`` to None; every
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
