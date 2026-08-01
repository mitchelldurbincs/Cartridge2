"""Tests for the evaluation driver.

Games themselves are played by the Rust ``cartridge-eval`` binary and tested in
``engine/evaluator``; what belongs here is the Python side of that boundary —
locating the binary, building its argument vector, and parsing what it writes
back. The binary is stubbed at ``run_eval_binary`` so these run in the Python
CI job, which has no Rust toolchain.
"""

import json
from pathlib import Path

import pytest

from trainer import evaluator
from trainer.evaluator import (
    EvalBinaryNotFound,
    EvalResults,
    build_eval_command,
    evaluate,
    find_eval_binary,
)
from trainer.players import ModelPlayer, RandomPlayer

SUMMARY_FIELDS = {
    "env_id": "connect4",
    "player1_name": "ONNX(latest.onnx)",
    "player2_name": "Random",
    "games_played": 10,
    "player1_wins": 6,
    "player2_wins": 3,
    "draws": 1,
    "player1_wins_as_first": 4,
    "player1_wins_as_second": 2,
    "player2_wins_as_first": 1,
    "player2_wins_as_second": 2,
    "avg_game_length": 21.5,
}


def arg_value(command: list[str], flag: str) -> str:
    """The value following ``flag`` in an argument vector."""
    return command[command.index(flag) + 1]


@pytest.fixture
def stub_binary(monkeypatch, tmp_path):
    """Make the binary resolvable without building it."""
    binary = tmp_path / "cartridge-eval"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv(evaluator.EVAL_BINARY_ENV, str(binary))
    return binary


class TestFindEvalBinary:
    def test_env_var_wins(self, stub_binary):
        assert find_eval_binary() == stub_binary

    def test_missing_env_var_target_is_an_error_not_a_silent_fallback(
        self, monkeypatch, tmp_path
    ):
        # Falling back to a stale build here would evaluate with the wrong
        # binary and report the result as if nothing were wrong.
        monkeypatch.setenv(evaluator.EVAL_BINARY_ENV, str(tmp_path / "nope"))
        with pytest.raises(EvalBinaryNotFound, match="does not exist"):
            find_eval_binary()

    def test_error_names_the_build_command(self, monkeypatch):
        monkeypatch.delenv(evaluator.EVAL_BINARY_ENV, raising=False)
        monkeypatch.setattr(evaluator, "_BINARY_CANDIDATES", ())
        with pytest.raises(EvalBinaryNotFound, match="cargo build"):
            find_eval_binary()


class TestBuildEvalCommand:
    def test_random_player_needs_no_model_arguments(self, stub_binary):
        command = build_eval_command(
            RandomPlayer(), RandomPlayer(), "othello", 8, 3, Path("/tmp/out.json")
        )

        assert arg_value(command, "--p1") == "random"
        assert arg_value(command, "--p2") == "random"
        assert arg_value(command, "--env-id") == "othello"
        assert arg_value(command, "--games") == "8"
        assert arg_value(command, "--seed") == "3"
        assert "--p1-temperature" not in command

    def test_model_player_carries_its_temperature_and_search_budget(self, stub_binary):
        command = build_eval_command(
            ModelPlayer("/models/latest.onnx", temperature=0.2, simulations=100),
            RandomPlayer(),
            "connect4",
            10,
            42,
            Path("/tmp/out.json"),
        )

        assert arg_value(command, "--p1") == "/models/latest.onnx"
        assert arg_value(command, "--p1-temperature") == "0.2"
        assert arg_value(command, "--p1-sims") == "100"

    def test_positions_are_dumped_only_when_asked(self, stub_binary):
        without = build_eval_command(
            RandomPlayer(), RandomPlayer(), "connect4", 2, 1, Path("/tmp/out.json")
        )
        assert "--dump-positions" not in without

        with_dump = build_eval_command(
            RandomPlayer(),
            RandomPlayer(),
            "connect4",
            2,
            1,
            Path("/tmp/out.json"),
            dump_positions=Path("/tmp/pos.jsonl"),
        )
        assert arg_value(with_dump, "--dump-positions") == "/tmp/pos.jsonl"

    def test_both_seats_are_configured_independently(self, stub_binary):
        command = build_eval_command(
            ModelPlayer("/models/a.onnx", temperature=0.2, simulations=50),
            ModelPlayer("/models/b.onnx", temperature=0.0, simulations=25),
            "connect4",
            4,
            1,
            Path("/tmp/out.json"),
        )

        assert arg_value(command, "--p1") == "/models/a.onnx"
        assert arg_value(command, "--p2") == "/models/b.onnx"
        assert arg_value(command, "--p1-sims") == "50"
        assert arg_value(command, "--p2-sims") == "25"


class TestEvaluate:
    def test_parses_the_binary_output(self, stub_binary, monkeypatch):
        seen = {}

        def fake_run(command):
            seen["command"] = command
            Path(arg_value(command, "--output")).write_text(json.dumps(SUMMARY_FIELDS))

        monkeypatch.setattr(evaluator, "run_eval_binary", fake_run)

        results = evaluate(
            player1=ModelPlayer("/models/latest.onnx"),
            player2=RandomPlayer(),
            env_id="connect4",
            num_games=10,
        )

        assert results == EvalResults(**SUMMARY_FIELDS)
        assert arg_value(seen["command"], "--games") == "10"

    def test_config_keyword_is_accepted_and_ignored(self, stub_binary, monkeypatch):
        # crucible's HeadToHeadEvalLike seam always passes config; the engine
        # already knows the game, so it must not change anything here.
        monkeypatch.setattr(
            evaluator,
            "run_eval_binary",
            lambda command: Path(arg_value(command, "--output")).write_text(
                json.dumps(SUMMARY_FIELDS)
            ),
        )

        with_config = evaluate(
            player1=RandomPlayer(),
            player2=RandomPlayer(),
            env_id="connect4",
            config=object(),
            num_games=10,
            verbose=False,
        )
        assert with_config == EvalResults(**SUMMARY_FIELDS)

    def test_binary_failure_propagates(self, stub_binary, monkeypatch):
        def fail(command):
            raise RuntimeError("cartridge-eval exited with 1: boom")

        monkeypatch.setattr(evaluator, "run_eval_binary", fail)

        with pytest.raises(RuntimeError, match="boom"):
            evaluate(
                player1=RandomPlayer(),
                player2=RandomPlayer(),
                env_id="connect4",
                num_games=2,
            )


class TestEvalResults:
    def test_rates_are_shares_of_games_played(self):
        results = EvalResults(**SUMMARY_FIELDS)

        assert results.player1_win_rate == pytest.approx(0.6)
        assert results.player2_win_rate == pytest.approx(0.3)
        assert results.draw_rate == pytest.approx(0.1)

    def test_zero_games_yields_zero_rates_not_a_division_error(self):
        results = EvalResults(**{**SUMMARY_FIELDS, "games_played": 0})

        assert results.player1_win_rate == 0.0
        assert results.player2_win_rate == 0.0
        assert results.draw_rate == 0.0

    def test_summary_reports_both_players_and_the_outcome(self):
        summary = EvalResults(**SUMMARY_FIELDS).summary()

        assert "connect4" in summary
        assert "ONNX(latest.onnx)" in summary
        assert "Random" in summary
        assert "60.0%" in summary
        assert "21.5" in summary


class TestPlayers:
    def test_random_player_is_named_for_reporting(self):
        assert RandomPlayer().name == "Random"

    def test_model_player_is_named_after_its_file(self):
        assert ModelPlayer("/models/model_step_016000.onnx").name == (
            "ONNX(model_step_016000.onnx)"
        )

    def test_model_player_defaults_to_greedy_play_without_search(self):
        # The pre-migration Python evaluator behaved this way; keeping it the
        # default is what makes eval numbers comparable across the move.
        player = ModelPlayer("/models/latest.onnx")
        assert player.temperature == 0.0
        assert player.simulations == 0
