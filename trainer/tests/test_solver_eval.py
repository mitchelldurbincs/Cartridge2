"""Tests for the solver_eval module.

Groups:
- Pure classification/judgment logic (no bitbully required)
- Aggregation dataclasses and invariants
- End-to-end driver with a MockScorer (no bitbully required)
- bitbully integration (skipped when bitbully is not installed)
"""

import argparse
import json
import random

import pytest

from trainer.game_config import GameConfig
from trainer.games import create_game_state
from trainer.policies import Policy, RandomPolicy
from trainer.solver_eval import (
    BucketStats,
    SolverEvalResults,
    append_solver_stats,
    classify_score,
    discover_checkpoints,
    format_progression_table,
    infer_step_from_filename,
    judge_move,
    ply_bucket,
    run_solver_evaluation,
    solver_evaluate,
)


class MockPolicy(Policy):
    """Mock policy playing a predetermined sequence, falling back to first legal."""

    def __init__(self, name: str, actions: list[int] | None = None):
        self._name = name
        self._actions = actions or []
        self._action_idx = 0

    @property
    def name(self) -> str:
        return self._name

    def select_action(self, state, config) -> int:
        legal = state.legal_moves()
        if self._action_idx < len(self._actions):
            action = self._actions[self._action_idx]
            self._action_idx += 1
            if action in legal:
                return action
        return legal[0] if legal else 0


class MockScorer:
    """Duck-typed stand-in for SolverScorer: first legal move wins, rest lose."""

    def __init__(self):
        self.queries = 0
        self.cache_hits = 0
        self.solve_time_seconds = 0.0
        self.resets = 0
        self.mirrored_moves = []
        self.scored_states = []

    def reset(self) -> None:
        self.resets += 1

    def mirror_move(self, col: int) -> None:
        self.mirrored_moves.append(col)

    def scores_for(self, state) -> dict[int, int]:
        self.queries += 1
        self.scored_states.append(len(state.legal_moves()))
        legal = state.legal_moves()
        return {move: (1 if i == 0 else -1) for i, move in enumerate(legal)}


def create_test_config(env_id: str = "connect4") -> GameConfig:
    """Create a test GameConfig with all required fields."""
    configs = {
        "tictactoe": GameConfig(
            env_id="tictactoe",
            display_name="Tic Tac Toe",
            board_width=3,
            board_height=3,
            num_actions=9,
            obs_size=27,
            legal_mask_offset=27,
        ),
        "connect4": GameConfig(
            env_id="connect4",
            display_name="Connect 4",
            board_width=7,
            board_height=6,
            num_actions=7,
            obs_size=84,
            legal_mask_offset=84,
        ),
    }
    return configs.get(env_id, configs["connect4"])


class TestClassifyScore:
    """Test score sign classification."""

    def test_classify_score_signs(self):
        assert classify_score(5) == "win"
        assert classify_score(1) == "win"
        assert classify_score(0) == "draw"
        assert classify_score(-1) == "loss"
        assert classify_score(-3) == "loss"


class TestJudgeMove:
    """Test single-move judgment logic."""

    def test_exact_best_win(self):
        judgment = judge_move({3: 5, 2: 0, 0: -2}, chosen=3)
        assert judgment.value_optimal
        assert judgment.exact_best
        assert judgment.blunder is None
        assert not judgment.forced

    def test_value_optimal_not_exact(self):
        judgment = judge_move({3: 5, 2: 3, 0: -2}, chosen=2)
        assert judgment.value_optimal
        assert not judgment.exact_best
        assert judgment.blunder is None

    def test_blunder_win_to_draw(self):
        judgment = judge_move({3: 5, 2: 0}, chosen=2)
        assert not judgment.value_optimal
        assert judgment.blunder == "win_to_draw"

    def test_blunder_win_to_loss(self):
        judgment = judge_move({3: 5, 0: -1}, chosen=0)
        assert not judgment.value_optimal
        assert judgment.blunder == "win_to_loss"

    def test_blunder_draw_to_loss(self):
        judgment = judge_move({3: 0, 0: -1}, chosen=0)
        assert not judgment.value_optimal
        assert judgment.blunder == "draw_to_loss"

    def test_all_moves_lose(self):
        # Best class is loss, so any move is value-optimal; argmax is the
        # slowest loss.
        judgment = judge_move({0: -1, 1: -5}, chosen=1)
        assert judgment.value_optimal
        assert judgment.blunder is None
        assert not judgment.exact_best

        judgment = judge_move({0: -1, 1: -5}, chosen=0)
        assert judgment.exact_best

    def test_exact_best_tie(self):
        judgment = judge_move({2: 4, 4: 4, 3: 1}, chosen=4)
        assert judgment.exact_best
        assert judgment.value_optimal

    def test_forced_move(self):
        judgment = judge_move({5: -2}, chosen=5)
        assert judgment.forced
        assert judgment.value_optimal
        assert judgment.exact_best

    def test_rejects_illegal_chosen(self):
        with pytest.raises(ValueError):
            judge_move({3: 5, 2: 0}, chosen=6)


class TestPlyBucket:
    """Test ply bucket boundaries."""

    def test_ply_bucket_boundaries(self):
        assert ply_bucket(1) == "ply_1_8"
        assert ply_bucket(8) == "ply_1_8"
        assert ply_bucket(9) == "ply_9_20"
        assert ply_bucket(20) == "ply_9_20"
        assert ply_bucket(21) == "ply_21_plus"
        assert ply_bucket(42) == "ply_21_plus"


class TestInferStep:
    """Test checkpoint filename parsing."""

    def test_infer_step_from_filename(self):
        assert infer_step_from_filename("model_step_016000.onnx") == 16000
        assert infer_step_from_filename("model_step_015450.onnx") == 15450
        assert infer_step_from_filename("data/models/model_step_000100.onnx") == 100
        assert infer_step_from_filename("latest.onnx") is None
        assert infer_step_from_filename("best.onnx") is None
        assert infer_step_from_filename("model_step_16000.pt") is None


class TestBucketStats:
    """Test aggregation and metric invariants."""

    def test_zero_positions_rates(self):
        stats = BucketStats()
        assert stats.value_optimal_rate == 0.0
        assert stats.exact_best_rate == 0.0
        assert stats.blunder_rate == 0.0
        assert stats.forced_rate == 0.0

    def test_invariants(self):
        stats = BucketStats()
        judgments = [
            judge_move({3: 5, 2: 0, 0: -2}, chosen=3),  # exact best
            judge_move({3: 5, 2: 3, 0: -2}, chosen=2),  # optimal, not exact
            judge_move({3: 5, 2: 0}, chosen=2),  # win_to_draw
            judge_move({3: 5, 0: -1}, chosen=0),  # win_to_loss
            judge_move({3: 0, 0: -1}, chosen=0),  # draw_to_loss
            judge_move({5: -2}, chosen=5),  # forced
        ]
        for judgment in judgments:
            stats.add(judgment)

        assert stats.positions == 6
        # Not-value-optimal and blunder are complements.
        assert stats.blunder_rate == pytest.approx(1.0 - stats.value_optimal_rate)
        # An argmax move always has the best class.
        assert stats.exact_best_rate <= stats.value_optimal_rate

    def test_to_dict_shape(self):
        stats = BucketStats()
        stats.add(judge_move({3: 5}, chosen=3))
        d = stats.to_dict()
        for key in (
            "positions",
            "value_optimal",
            "exact_best",
            "blunders_win_to_draw",
            "blunders_win_to_loss",
            "blunders_draw_to_loss",
            "forced",
            "value_optimal_rate",
            "exact_best_rate",
            "blunder_rate",
            "forced_rate",
        ):
            assert key in d


class TestSolverEvalResults:
    """Test results serialization and summary."""

    def _results(self) -> SolverEvalResults:
        results = SolverEvalResults(
            env_id="connect4",
            model_name="ONNX(latest.onnx)",
            model_path="data/models/latest.onnx",
            step=None,
            opponent_name="Random",
            games=10,
            seed=42,
            temperature=0.0,
        )
        results.overall.add(judge_move({3: 5, 0: -1}, chosen=3))
        return results

    def test_to_dict_shape(self):
        d = self._results().to_dict()
        for key in (
            "model",
            "step",
            "games",
            "seed",
            "value_optimal_rate",
            "exact_best_rate",
            "blunder_rate",
            "positions_scored",
            "forced_move_rate",
            "solver_cache_hit_rate",
            "by_ply",
            "by_seat",
            "timestamp",
        ):
            assert key in d
        assert d["step"] is None
        assert set(d["by_ply"]) == {"ply_1_8", "ply_9_20", "ply_21_plus"}
        assert set(d["by_seat"]) == {"first", "second"}

    def test_summary_contains_metrics(self):
        summary = self._results().summary()
        assert "ONNX(latest.onnx)" in summary
        assert "Random" in summary
        assert "overall" in summary
        assert "%" in summary


class TestSolverEvaluateDriver:
    """End-to-end driver tests with MockScorer + MockPolicy (no bitbully)."""

    def test_scores_only_model_moves(self):
        scorer = MockScorer()
        model = MockPolicy("model")
        opponent = MockPolicy("opponent")

        results = solver_evaluate(
            model=model,
            opponent=opponent,
            scorer=scorer,
            env_id="connect4",
            config=create_test_config("connect4"),
            num_games=2,
            seed=42,
        )

        total_moves = len(scorer.mirrored_moves)
        assert results.overall.positions == scorer.queries
        assert 0 < scorer.queries < total_moves  # model's share only
        assert scorer.resets == 2
        # avg_game_length reflects mirrored move count across the 2 games
        assert results.avg_game_length == pytest.approx(total_moves / 2)

    def test_seat_split_and_ply_buckets(self):
        scorer = MockScorer()
        results = solver_evaluate(
            model=MockPolicy("model"),
            opponent=MockPolicy("opponent"),
            scorer=scorer,
            env_id="connect4",
            config=create_test_config("connect4"),
            num_games=4,
            seed=42,
        )

        first = results.by_seat["first"]
        second = results.by_seat["second"]
        assert first.positions > 0
        assert second.positions > 0
        assert first.positions + second.positions == results.overall.positions
        assert results.by_ply["ply_1_8"].positions > 0
        by_ply_total = sum(s.positions for s in results.by_ply.values())
        assert by_ply_total == results.overall.positions

    def test_seed_reproducibility(self):
        volatile_keys = {"timestamp", "wall_time_seconds", "solver_time_seconds"}

        def run(seed: int) -> dict:
            results = solver_evaluate(
                model=MockPolicy("model"),
                opponent=RandomPolicy(),
                scorer=MockScorer(),
                env_id="connect4",
                config=create_test_config("connect4"),
                num_games=4,
                seed=seed,
            )
            return {
                k: v for k, v in results.to_dict().items() if k not in volatile_keys
            }

        assert run(42) == run(42)

    def test_run_rejects_non_connect4(self):
        args = argparse.Namespace(env_id="tictactoe")
        # Guard fires before models or bitbully are touched, so the sparse
        # namespace is sufficient.
        assert run_solver_evaluation(args) == 1


class TestStatsFile:
    """Test solver_stats.json append behavior."""

    def test_append_creates_and_appends(self, tmp_path):
        output = tmp_path / "solver_stats.json"

        append_solver_stats({"model": "a"}, output)
        with open(output) as f:
            stats = json.load(f)
        assert stats["solver_evaluations"] == [{"model": "a"}]

        append_solver_stats({"model": "b"}, output)
        with open(output) as f:
            stats = json.load(f)
        assert len(stats["solver_evaluations"]) == 2

    def test_append_recovers_from_corrupt_file(self, tmp_path):
        output = tmp_path / "solver_stats.json"
        output.write_text("{not json")

        append_solver_stats({"model": "a"}, output)
        with open(output) as f:
            stats = json.load(f)
        assert len(stats["solver_evaluations"]) == 1


class TestCheckpointDiscovery:
    """Test checkpoint discovery and progression table."""

    def test_discover_checkpoints_ordering(self, tmp_path):
        for name in (
            "model_step_016000.onnx",
            "model_step_009000.onnx",
            "model_step_015450.onnx",
            "latest.onnx",
            "best.onnx",
            "unrelated.pt",
        ):
            (tmp_path / name).touch()

        found = [p.name for p in discover_checkpoints(tmp_path)]
        assert found == [
            "model_step_009000.onnx",
            "model_step_015450.onnx",
            "model_step_016000.onnx",
            "latest.onnx",
            "best.onnx",
        ]

    def test_discover_checkpoints_missing_latest_best(self, tmp_path):
        (tmp_path / "model_step_000100.onnx").touch()
        found = [p.name for p in discover_checkpoints(tmp_path)]
        assert found == ["model_step_000100.onnx"]

    def test_format_progression_table(self):
        def make(step, name):
            return SolverEvalResults(
                env_id="connect4",
                model_name=name,
                model_path=name,
                step=step,
                opponent_name="Random",
                games=10,
                seed=42,
                temperature=0.0,
            )

        table = format_progression_table(
            [make(None, "latest.onnx"), make(16000, "b.onnx"), make(15450, "a.onnx")]
        )
        lines = table.splitlines()
        assert "value-opt" in lines[0]
        # Sorted by step, None last as "-"
        assert lines[2].strip().startswith("15450")
        assert lines[3].strip().startswith("16000")
        assert lines[4].strip().startswith("-")


bitbully = pytest.importorskip("bitbully", reason="bitbully not installed")


class TestSolverScorerIntegration:
    """Integration tests against the real bitbully solver."""

    @pytest.fixture(scope="class")
    def scorer(self):
        from trainer.solver_eval import SolverScorer

        # Calibration runs inside the constructor.
        return SolverScorer()

    def test_calibration_and_empty_board(self, scorer):
        state = create_game_state("connect4")
        scorer.reset()
        scores = scorer.scores_for(state)

        assert set(scores) == set(range(7))
        assert classify_score(scores[3]) == "win"
        assert classify_score(scores[2]) == "draw"
        assert classify_score(scores[0]) == "loss"
        assert all(scores[i] == scores[6 - i] for i in range(3))

    def test_mirror_random_playout_agrees_with_connect4state(self, scorer):
        for seed in (1, 2, 3):
            rng = random.Random(seed)
            state = create_game_state("connect4")
            scorer.reset()

            while not state.done:
                # Cross-check legality before every move (raises on desync).
                scores = scorer.scores_for(state)
                assert sorted(scores) == sorted(state.legal_moves())
                move = rng.choice(state.legal_moves())
                scorer.mirror_move(move)
                state.make_move(move)

            assert scorer._board.is_game_over() == state.done

    def test_cache_hit_counting(self, scorer):
        state = create_game_state("connect4")
        scorer.reset()

        queries_before = scorer.queries
        hits_before = scorer.cache_hits
        first = scorer.scores_for(state)
        second = scorer.scores_for(state)

        assert scorer.queries == queries_before + 2
        assert scorer.cache_hits >= hits_before + 1
        assert first == second

    def test_full_column_filtered(self, scorer):
        state = create_game_state("connect4")
        scorer.reset()
        # Alternating colors stack column 3 full without a win.
        for _ in range(6):
            scorer.mirror_move(3)
            state.make_move(3)

        scores = scorer.scores_for(state)
        assert 3 not in scores
        assert sorted(scores) == sorted(state.legal_moves())

    def test_detects_desync(self, scorer):
        state = create_game_state("connect4")
        scorer.reset()
        scorer.mirror_move(3)  # mirrored but not applied to state

        with pytest.raises(RuntimeError):
            scorer.scores_for(state)
        scorer.reset()

    def test_immediate_win_classified(self, scorer):
        # FIRST has three stacked in column 3 and is to move: 3 wins now.
        moves = [3, 0, 3, 1, 3, 2]
        state = create_game_state("connect4")
        scorer.reset()
        for move in moves:
            scorer.mirror_move(move)
            state.make_move(move)

        scores = scorer.scores_for(state)
        assert classify_score(scores[3]) == "win"
        assert scores[3] == max(scores.values())

    def test_solver_evaluate_end_to_end_random_model(self, scorer):
        """Full driver run against the real solver with a random 'model'."""
        results = solver_evaluate(
            model=MockPolicy("model"),
            opponent=MockPolicy("opponent"),
            scorer=scorer,
            env_id="connect4",
            config=create_test_config("connect4"),
            num_games=2,
            seed=7,
        )
        assert results.overall.positions > 0
        assert 0.0 <= results.overall.value_optimal_rate <= 1.0
        assert results.overall.exact_best_rate <= results.overall.value_optimal_rate
