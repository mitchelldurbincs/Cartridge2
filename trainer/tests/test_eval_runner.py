"""Tests for the orchestrator's EvalRunner (solver metrics + promotion).

All tests are Postgres-free and bitbully-free: game evaluation and solver
scoring are monkeypatched at the eval_runner module level.
"""

import json
from types import SimpleNamespace

from trainer.orchestrator import eval_runner as eval_runner_module
from trainer.orchestrator.config import LoopConfig
from trainer.orchestrator.eval_runner import EvalRunner
from trainer.solver_eval import SolverEvalResults, judge_move


class RecordingLogger:
    enabled = True

    def __init__(self):
        self.logged = []

    def log(self, metrics, step=None):
        self.logged.append((dict(metrics), step))

    def finish(self):
        pass


class StubPolicy:
    def __init__(self, model_path, temperature=0.0):
        self.model_path = str(model_path)
        self.temperature = temperature

    @property
    def name(self):
        return f"Stub({self.model_path})"

    def select_action(self, state, config):
        return state.legal_moves()[0]


def stub_eval_results(win_rate: float):
    return SimpleNamespace(
        player1_win_rate=win_rate,
        player2_win_rate=1.0 - win_rate,
        draw_rate=0.0,
    )


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


def make_runner(
    tmp_path,
    monkeypatch,
    env_id="connect4",
    vs_best_win_rate=0.4,
    solver_rate=0.6,
    with_best=True,
    logger=None,
    **config_overrides,
):
    """EvalRunner with all external calls stubbed; latest.onnx (and best.onnx) touched."""
    config = LoopConfig(
        data_dir=tmp_path, env_id=env_id, eval_games=4, **config_overrides
    )
    config.models_dir.mkdir(parents=True, exist_ok=True)
    (config.models_dir / "latest.onnx").touch()
    if with_best:
        (config.models_dir / "best.onnx").touch()

    monkeypatch.setattr(eval_runner_module, "OnnxPolicy", StubPolicy)
    monkeypatch.setattr(
        eval_runner_module,
        "run_eval",
        lambda **kwargs: stub_eval_results(vs_best_win_rate),
    )
    if solver_rate is not None:
        monkeypatch.setattr(
            eval_runner_module,
            "solver_evaluate",
            lambda **kwargs: make_solver_results(solver_rate),
        )
        monkeypatch.setattr(eval_runner_module, "SolverScorer", lambda: object())

    return EvalRunner(config, wandb_logger=logger)


class TestSolverMetricsInEval:
    def test_eval_record_carries_solver_metrics(self, tmp_path, monkeypatch):
        runner = make_runner(tmp_path, monkeypatch, solver_rate=0.6)
        history = []

        runner.run(iteration=3, eval_history=history)

        assert len(history) == 1
        record = history[0]
        assert record["solver_value_optimal_rate"] == 0.6
        assert record["solver_blunder_rate"] == 0.4
        assert record["solver_positions"] == 10

    def test_solver_stats_file_written(self, tmp_path, monkeypatch):
        runner = make_runner(tmp_path, monkeypatch, solver_rate=0.6)

        runner.run(iteration=3, eval_history=[])

        stats_path = runner.config.solver_stats_path
        assert stats_path.exists()
        with open(stats_path) as f:
            stats = json.load(f)
        entry = stats["solver_evaluations"][0]
        assert entry["iteration"] == 3
        assert entry["context"] == "loop"
        assert entry["global_step"] == 3 * runner.config.steps_per_iteration

    def test_wandb_gets_eval_and_solver_metrics(self, tmp_path, monkeypatch):
        logger = RecordingLogger()
        runner = make_runner(tmp_path, monkeypatch, solver_rate=0.6, logger=logger)

        runner.run(iteration=3, eval_history=[])

        assert len(logger.logged) == 1
        metrics, step = logger.logged[0]
        assert step == 3 * runner.config.steps_per_iteration
        assert metrics["eval/vs_best_win_rate"] == 0.4
        assert metrics["solver/value_optimal_rate"] == 0.6
        assert metrics["solver/positions_scored"] == 10
        assert "solver/model_win_rate_vs_random" in metrics

    def test_non_connect4_skips_solver(self, tmp_path, monkeypatch):
        runner = make_runner(
            tmp_path, monkeypatch, env_id="tictactoe", solver_rate=None
        )
        history = []

        runner.run(iteration=1, eval_history=history)

        assert history[0]["solver_value_optimal_rate"] is None
        assert not runner.config.solver_stats_path.exists()

    def test_solver_games_zero_disables(self, tmp_path, monkeypatch):
        runner = make_runner(tmp_path, monkeypatch, solver_rate=0.6, solver_games=0)
        history = []

        runner.run(iteration=1, eval_history=history)

        assert history[0]["solver_value_optimal_rate"] is None

    def test_solver_failure_disables_permanently(self, tmp_path, monkeypatch):
        runner = make_runner(tmp_path, monkeypatch, solver_rate=0.6)

        def raising_scorer():
            raise ImportError("bitbully is required")

        monkeypatch.setattr(eval_runner_module, "SolverScorer", raising_scorer)
        history = []

        runner.run(iteration=1, eval_history=history)
        assert history[0]["solver_value_optimal_rate"] is None
        assert runner._solver_disabled

        # Second run must not try to construct the scorer again.
        calls = []
        monkeypatch.setattr(
            eval_runner_module, "SolverScorer", lambda: calls.append(1) or object()
        )
        runner.run(iteration=2, eval_history=history)
        assert calls == []

    def test_first_eval_promotes_and_still_scores_solver(self, tmp_path, monkeypatch):
        runner = make_runner(tmp_path, monkeypatch, solver_rate=0.6, with_best=False)
        history = []

        runner.run(iteration=1, eval_history=history)

        assert history[0]["became_new_best"] is True
        assert history[0]["solver_value_optimal_rate"] == 0.6
        assert runner.config.best_model_path.exists()


class TestSolverOptimalPromotion:
    def _path_aware_solver(self, monkeypatch, rates: dict):
        """solver_evaluate stub returning a rate keyed by model filename."""
        from pathlib import Path

        def stub(**kwargs):
            name = Path(kwargs["model"].model_path).name
            return make_solver_results(rates[name])

        monkeypatch.setattr(eval_runner_module, "solver_evaluate", stub)

    def test_promotes_when_candidate_beats_best_by_margin(self, tmp_path, monkeypatch):
        runner = make_runner(
            tmp_path,
            monkeypatch,
            promotion_metric="solver_optimal",
            vs_best_win_rate=0.4,
        )
        self._path_aware_solver(monkeypatch, {"latest.onnx": 0.7, "best.onnx": 0.6})
        history = []

        runner.run(iteration=2, eval_history=history)

        # Win rate 0.4 would NOT promote; solver_optimal does.
        assert history[0]["became_new_best"] is True
        assert history[0]["promotion_metric"] == "solver_optimal"
        assert runner.best_solver_rate == 0.7

    def test_rejects_within_margin(self, tmp_path, monkeypatch):
        runner = make_runner(
            tmp_path,
            monkeypatch,
            promotion_metric="solver_optimal",
            vs_best_win_rate=0.9,  # would promote under win_rate
        )
        self._path_aware_solver(monkeypatch, {"latest.onnx": 0.605, "best.onnx": 0.6})
        history = []

        runner.run(iteration=2, eval_history=history)

        assert history[0]["became_new_best"] is False

    def test_backfills_legacy_best_rate_once(self, tmp_path, monkeypatch):
        runner = make_runner(
            tmp_path,
            monkeypatch,
            promotion_metric="solver_optimal",
            vs_best_win_rate=0.4,
        )
        runner.config.best_model_info_path.write_text(
            json.dumps({"iteration": 5, "win_rate_when_promoted": 1.0})
        )
        runner.best_solver_rate = None
        self._path_aware_solver(monkeypatch, {"latest.onnx": 0.7, "best.onnx": 0.6})

        runner.run(iteration=2, eval_history=[])

        # Backfill wrote the best rate before promotion replaced it.
        assert runner.best_solver_rate == 0.7  # promoted candidate is the new best
        with open(runner.config.best_model_info_path) as f:
            data = json.load(f)
        assert data["solver_value_optimal_rate"] == 0.7

    def test_tictactoe_falls_back_to_win_rate(self, tmp_path, monkeypatch):
        runner = make_runner(
            tmp_path,
            monkeypatch,
            env_id="tictactoe",
            solver_rate=None,
            promotion_metric="solver_optimal",
            vs_best_win_rate=0.7,
        )
        history = []

        runner.run(iteration=2, eval_history=history)

        # Solver unavailable for tictactoe -> effective metric is win_rate,
        # and 0.7 > 0.55 promotes.
        assert history[0]["promotion_metric"] == "win_rate"
        assert history[0]["became_new_best"] is True


class TestEvalRecordSchema:
    def test_eval_record_key_set_is_frozen(self, tmp_path, monkeypatch):
        """The record schema is consumed by the frontend — guard the key set."""
        runner = make_runner(tmp_path, monkeypatch)
        history = []

        runner.run(iteration=2, eval_history=history)

        assert set(history[0]) == {
            "iteration",
            "step",
            "vs_best_win_rate",
            "vs_best_draw_rate",
            "vs_best_opponent_iteration",
            "became_new_best",
            "vs_random_win_rate",
            "vs_random_draw_rate",
            "solver_value_optimal_rate",
            "solver_exact_best_rate",
            "solver_blunder_rate",
            "solver_positions",
            "promotion_metric",
            "games",
            "timestamp",
        }

    def test_vs_random_disabled_leaves_fields_none(self, tmp_path, monkeypatch):
        runner = make_runner(tmp_path, monkeypatch, eval_vs_random=False)
        history = []

        runner.run(iteration=1, eval_history=history)

        assert history[0]["vs_random_win_rate"] is None
        assert history[0]["vs_random_draw_rate"] is None

    def test_wandb_drops_none_valued_metrics(self, tmp_path, monkeypatch):
        logger = RecordingLogger()
        runner = make_runner(
            tmp_path,
            monkeypatch,
            env_id="tictactoe",
            solver_rate=None,
            logger=logger,
            eval_vs_random=False,
        )

        runner.run(iteration=1, eval_history=[])

        metrics, _ = logger.logged[0]
        assert "eval/vs_random_win_rate" not in metrics
        assert not any(key.startswith("solver/") for key in metrics)
        assert metrics["eval/vs_best_win_rate"] == 0.4


class TestEvalFailurePaths:
    def test_missing_model_skips_eval(self, tmp_path, monkeypatch):
        runner = make_runner(tmp_path, monkeypatch)
        runner.config.latest_model_path.unlink()
        history = []

        result = runner.run(iteration=1, eval_history=history)

        assert result == (None, None, 0.0)
        assert history == []

    def test_eval_exception_yields_none_rates_and_no_record(
        self, tmp_path, monkeypatch
    ):
        runner = make_runner(tmp_path, monkeypatch)

        def boom(**kwargs):
            raise RuntimeError("eval exploded")

        monkeypatch.setattr(eval_runner_module, "run_eval", boom)
        history = []

        win_rate, draw_rate, elapsed = runner.run(iteration=1, eval_history=history)

        assert win_rate is None
        assert draw_rate is None
        assert elapsed >= 0.0
        assert history == []

    def test_promote_without_latest_model_keeps_state(self, tmp_path, monkeypatch):
        runner = make_runner(tmp_path, monkeypatch)
        runner.best_model_iteration = 3
        runner.config.latest_model_path.unlink()

        runner.promote_to_best(9, 0.9)

        assert runner.best_model_iteration == 3
        assert not runner.config.best_model_info_path.exists()
