"""Tests for the solver_eval module.

Groups:
- Pure classification/judgment logic (no bitbully required)
- Aggregation dataclasses and invariants
- End-to-end driver with a MockScorer (no bitbully required)
- bitbully integration (skipped when bitbully is not installed)
"""

import argparse
import json
import shutil
from pathlib import Path

import pytest

from trainer.players import ModelPlayer, RandomPlayer
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

WIDTH, HEIGHT = 7, 6


class MockScorer:
    """Duck-typed stand-in for SolverScorer: first legal move wins, rest lose."""

    def __init__(self):
        self.queries = 0
        self.cache_hits = 0
        self.solve_time_seconds = 0.0
        self.resets = 0
        self.mirrored_moves = []
        self.scored_positions = []

    def reset(self) -> None:
        self.resets += 1

    def mirror_move(self, col: int) -> None:
        self.mirrored_moves.append(col)

    def scores_for(self, board, current_player, legal) -> dict[int, int]:
        self.queries += 1
        self.scored_positions.append((tuple(board), current_player))
        return {move: (1 if i == 0 else -1) for i, move in enumerate(sorted(legal))}


def engine_board(moves: list[int]) -> tuple[list[int], list[int], int]:
    """Board, legal columns and side to move after playing ``moves``.

    A fixture builder, not a rules implementation: it drops pieces and tracks
    column heights, with no win detection. The layout is the engine's —
    row-major, ``row * WIDTH + col``, row 0 at the bottom — which is what
    ``scores_for`` is given.
    """
    board = [0] * (WIDTH * HEIGHT)
    heights = [0] * WIDTH
    player = 1
    for col in moves:
        board[heights[col] * WIDTH + col] = player
        heights[col] += 1
        player = 3 - player
    legal = [col for col in range(WIDTH) if heights[col] < HEIGHT]
    return board, legal, player


def fake_dump(num_games: int, plies_per_game: int = 6) -> list[dict]:
    """Synthetic position records shaped like the ones cartridge-eval writes.

    The model is player 1 and, like the binary, holds seat 1 for the first half
    of the games and seat 2 for the rest.
    """
    records = []
    for game in range(num_games):
        p1_first = game < num_games // 2
        moves: list[int] = []
        for ply in range(plies_per_game):
            seat = 1 + ply % 2
            board, legal, _ = engine_board(moves)
            records.append(
                {
                    "game": game,
                    "ply": ply,
                    "player": seat,
                    "by": "p1" if (seat == 1) == p1_first else "p2",
                    "action": legal[ply % len(legal)],
                    "board": board,
                    "legal": legal,
                }
            )
            moves.append(records[-1]["action"])
    return records


def fake_summary(num_games: int, plies_per_game: int = 6) -> dict:
    return {
        "env_id": "connect4",
        "player1_name": "ONNX(latest.onnx)",
        "player2_name": "Random",
        "games_played": num_games,
        "player1_wins": num_games,
        "player2_wins": 0,
        "draws": 0,
        "player1_wins_as_first": num_games // 2,
        "player1_wins_as_second": num_games - num_games // 2,
        "player2_wins_as_first": 0,
        "player2_wins_as_second": 0,
        "avg_game_length": float(plies_per_game),
    }


@pytest.fixture
def stub_eval_binary(monkeypatch, tmp_path):
    """Answer cartridge-eval invocations with canned output.

    The Python CI job has no Rust toolchain, so the driver is exercised against
    the binary's *contract* — its argument vector and the two files it writes —
    rather than the binary itself, which engine/evaluator tests cover.
    """
    from trainer import evaluator
    from trainer.solver_eval import scorer as scorer_module

    binary = tmp_path / "cartridge-eval"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv(evaluator.EVAL_BINARY_ENV, str(binary))

    calls = []

    def fake_run(command):
        calls.append(command)
        games = int(command[command.index("--games") + 1])
        out = Path(command[command.index("--output") + 1])
        dump = Path(command[command.index("--dump-positions") + 1])
        out.write_text(json.dumps(fake_summary(games)))
        dump.write_text(
            "\n".join(json.dumps(record) for record in fake_dump(games)) + "\n"
        )

    monkeypatch.setattr(scorer_module, "run_eval_binary", fake_run)
    return calls


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
    """Driver tests with a stubbed binary and MockScorer (no bitbully, no Rust)."""

    def test_scores_only_model_moves(self, stub_eval_binary):
        scorer = MockScorer()

        results = solver_evaluate(
            model=ModelPlayer("/models/latest.onnx"),
            opponent=RandomPlayer(),
            scorer=scorer,
            env_id="connect4",
            num_games=2,
            seed=42,
        )

        total_moves = len(scorer.mirrored_moves)
        assert results.overall.positions == scorer.queries
        assert 0 < scorer.queries < total_moves  # the model's share only
        assert scorer.resets == 2  # one fresh mirrored board per game

    def test_every_move_is_mirrored_even_when_not_scored(self, stub_eval_binary):
        # The solver board has to follow the whole game, not just the model's
        # half, or it desyncs on the very next query.
        scorer = MockScorer()
        solver_evaluate(
            model=ModelPlayer("/models/latest.onnx"),
            opponent=RandomPlayer(),
            scorer=scorer,
            env_id="connect4",
            num_games=2,
            seed=42,
        )

        assert len(scorer.mirrored_moves) == len(fake_dump(2))

    def test_outcome_counts_come_from_the_binary_summary(self, stub_eval_binary):
        results = solver_evaluate(
            model=ModelPlayer("/models/latest.onnx"),
            opponent=RandomPlayer(),
            scorer=MockScorer(),
            env_id="connect4",
            num_games=4,
            seed=42,
        )

        expected = fake_summary(4)
        assert results.model_wins == expected["player1_wins"]
        assert results.model_losses == expected["player2_wins"]
        assert results.draws == expected["draws"]
        assert results.avg_game_length == expected["avg_game_length"]

    def test_seat_split_and_ply_buckets(self, stub_eval_binary):
        scorer = MockScorer()
        results = solver_evaluate(
            model=ModelPlayer("/models/latest.onnx"),
            opponent=RandomPlayer(),
            scorer=scorer,
            env_id="connect4",
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

    def test_seed_is_passed_through_to_the_binary(self, stub_eval_binary):
        solver_evaluate(
            model=ModelPlayer("/models/latest.onnx"),
            opponent=RandomPlayer(),
            scorer=MockScorer(),
            env_id="connect4",
            num_games=2,
            seed=1234,
        )

        command = stub_eval_binary[0]
        assert command[command.index("--seed") + 1] == "1234"

    def test_seed_reproducibility(self, stub_eval_binary):
        volatile_keys = {"timestamp", "wall_time_seconds", "solver_time_seconds"}

        def run(seed: int) -> dict:
            results = solver_evaluate(
                model=ModelPlayer("/models/latest.onnx"),
                opponent=RandomPlayer(),
                scorer=MockScorer(),
                env_id="connect4",
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
        board, legal, player = engine_board([])
        scorer.reset()
        scores = scorer.scores_for(board, player, legal)

        assert set(scores) == set(range(WIDTH))
        assert classify_score(scores[3]) == "win"
        assert classify_score(scores[2]) == "draw"
        assert classify_score(scores[0]) == "loss"
        assert all(scores[i] == scores[6 - i] for i in range(3))

    def test_mirrored_board_matches_the_engines_row_major_layout(self, scorer):
        # bitbully's array is column-major and the engine's board view is
        # row-major; if the reindex between them were wrong, scores_for would
        # raise on the first non-symmetric position rather than agree.
        moves = [3, 3, 4, 0, 4, 1]
        scorer.reset()
        for i, col in enumerate(moves):
            board, legal, player = engine_board(moves[:i])
            scores = scorer.scores_for(board, player, legal)
            assert sorted(scores) == sorted(legal)
            scorer.mirror_move(col)

    def test_cache_hit_counting(self, scorer):
        board, legal, player = engine_board([])
        scorer.reset()

        queries_before = scorer.queries
        hits_before = scorer.cache_hits
        first = scorer.scores_for(board, player, legal)
        second = scorer.scores_for(board, player, legal)

        assert scorer.queries == queries_before + 2
        assert scorer.cache_hits >= hits_before + 1
        assert first == second

    def test_full_column_filtered(self, scorer):
        # Alternating colors stack column 3 full without a win.
        moves = [3] * HEIGHT
        scorer.reset()
        for col in moves:
            scorer.mirror_move(col)

        board, legal, player = engine_board(moves)
        scores = scorer.scores_for(board, player, legal)
        assert 3 not in scores
        assert sorted(scores) == sorted(legal)

    def test_detects_desync(self, scorer):
        scorer.reset()
        scorer.mirror_move(3)  # mirrored, but the position says nothing played

        board, legal, player = engine_board([])
        with pytest.raises(RuntimeError, match="desynced"):
            scorer.scores_for(board, player, legal)
        scorer.reset()

    def test_immediate_win_classified(self, scorer):
        # Player 1 has three stacked in column 3 and is to move: 3 wins now.
        moves = [3, 0, 3, 1, 3, 2]
        scorer.reset()
        for col in moves:
            scorer.mirror_move(col)

        board, legal, player = engine_board(moves)
        scores = scorer.scores_for(board, player, legal)
        assert classify_score(scores[3]) == "win"
        assert scores[3] == max(scores.values())

    def test_solver_evaluate_end_to_end(self, scorer, stub_eval_binary):
        results = solver_evaluate(
            model=ModelPlayer("/models/latest.onnx"),
            opponent=RandomPlayer(),
            scorer=scorer,
            env_id="connect4",
            num_games=2,
            seed=7,
        )
        assert results.overall.positions > 0
        assert 0.0 <= results.overall.value_optimal_rate <= 1.0
        assert results.overall.exact_best_rate <= results.overall.value_optimal_rate


def _eval_binary() -> Path | None:
    """The built cartridge-eval binary, if this checkout has one."""
    from trainer.evaluator import EvalBinaryNotFound, find_eval_binary

    try:
        return find_eval_binary()
    except EvalBinaryNotFound:
        found = shutil.which("cartridge-eval")
        return Path(found) if found else None


@pytest.mark.skipif(
    _eval_binary() is None,
    reason="cartridge-eval not built (Rust toolchain absent in the Python CI job)",
)
class TestSolverAgainstRealEngineGames:
    """The mirrored solver board must agree with the engine's own positions.

    This is the check the old driver could not make: it played against a Python
    reimplementation of Connect 4 and compared bitbully to *that*. Here the
    engine plays, dumps every position it saw, and the solver is replayed
    through them — so a disagreement between solver and engine is a failure
    rather than something nobody was looking at.
    """

    def test_replaying_an_engine_dump_never_desyncs(self, tmp_path):
        from trainer.evaluator import build_eval_command, run_eval_binary
        from trainer.solver_eval import SolverScorer

        output = tmp_path / "eval.json"
        dump = tmp_path / "positions.jsonl"
        run_eval_binary(
            build_eval_command(
                RandomPlayer(),
                RandomPlayer(),
                "connect4",
                4,
                11,
                output,
                dump_positions=dump,
            )
        )
        records = [json.loads(line) for line in dump.read_text().splitlines() if line]
        assert records

        scorer = SolverScorer()
        game = None
        for record in records:
            if record["game"] != game:
                game = record["game"]
                scorer.reset()
            # Raises on any disagreement with the engine's board.
            scores = scorer.scores_for(
                record["board"], record["player"], record["legal"]
            )
            assert sorted(scores) == sorted(record["legal"])
            scorer.mirror_move(record["action"])
