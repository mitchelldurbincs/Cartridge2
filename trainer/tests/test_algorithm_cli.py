"""Tests for the canonical algorithm-owned trainer command surface."""

from __future__ import annotations

import argparse
import importlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from trainer import __main__ as cli
from trainer.algorithms import get_algorithm
from trainer.algorithms.alphazero_board_v1 import (
    ALGORITHM_ID,
    DESCRIPTOR,
    AlphaZeroBoardV1,
)
from trainer.algorithms.base import AlgorithmCommand
from trainer.central_config import get_config
from trainer.orchestrator.cli import loop_config_from_args
from trainer.runtime_profile import resolve_runtime_profile


def test_selected_cartridge_owns_the_complete_command_set():
    algorithm = get_algorithm(ALGORITHM_ID)

    assert [command.name for command in algorithm.commands()] == [
        "train",
        "evaluate",
        "loop",
        "solver-eval",
        "register-players",
        "tournament",
    ]


def test_algorithm_is_a_global_option_before_the_command():
    parser = cli.build_parser(get_algorithm(ALGORITHM_ID))

    args = parser.parse_args(
        [
            "--algorithm",
            ALGORITHM_ID,
            "train",
            "--steps",
            "7",
            "--collection-scope-id",
            "a" * 64,
            "--source-root",
        ]
    )
    assert args.algorithm == ALGORITHM_ID
    assert args.command == "train"
    assert args.steps == 7

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--algorithm", ALGORITHM_ID])


def test_internal_start_step_is_not_a_public_train_option():
    parser = cli.build_parser(get_algorithm(ALGORITHM_ID))

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--algorithm",
                ALGORITHM_ID,
                "train",
                "--collection-scope-id",
                "a" * 64,
                "--source-root",
                "--start-step",
                "1",
            ]
        )


def test_standalone_solver_eval_has_no_output_option():
    parser = cli.build_parser(get_algorithm(ALGORITHM_ID))

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--algorithm",
                ALGORITHM_ID,
                "solver-eval",
                "--env-id",
                "connect4",
                "--output",
                "solver.json",
            ]
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--games", "0"],
        ["--games", str(1 << 32)],
        ["--simulations", "-1"],
        ["--simulations", str(1 << 32)],
        ["--seed", "-1"],
        ["--seed", str(1 << 64)],
        ["--temperature", "-0.1"],
        ["--temperature", "nan"],
        ["--temperature", "inf"],
    ],
)
def test_standalone_evaluate_rejects_invalid_numeric_arguments(arguments):
    parser = cli.build_parser(get_algorithm(ALGORITHM_ID))

    with pytest.raises(SystemExit):
        parser.parse_args(["--algorithm", ALGORITHM_ID, "evaluate", *arguments])


def test_standalone_evaluate_rejects_seed_schedule_overflow():
    from trainer.evaluator import run_evaluation

    args = SimpleNamespace(
        games=2,
        seed=(1 << 64) - 1,
        simulations=0,
        temperature=0.0,
    )

    assert run_evaluation(args) == 1


def test_standalone_solver_eval_rejects_conflicting_model_selection():
    parser = cli.build_parser(get_algorithm(ALGORITHM_ID))

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--algorithm",
                ALGORITHM_ID,
                "solver-eval",
                "--model",
                "candidate.onnx",
                "--all-checkpoints",
            ]
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--games", "0"],
        ["--games", str(1 << 32)],
        ["--seed", "-1"],
        ["--seed", str(1 << 64)],
        ["--temperature", "-0.1"],
        ["--temperature", "nan"],
        ["--temperature", "inf"],
    ],
)
def test_standalone_solver_eval_rejects_invalid_numeric_arguments(arguments):
    parser = cli.build_parser(get_algorithm(ALGORITHM_ID))

    with pytest.raises(SystemExit):
        parser.parse_args(["--algorithm", ALGORITHM_ID, "solver-eval", *arguments])


def test_standalone_solver_eval_rejects_seed_schedule_overflow():
    from trainer.solver_eval.cli import run_solver_evaluation

    args = SimpleNamespace(
        env_id="connect4",
        all_checkpoints=False,
        games=2,
        seed=(1 << 64) - 1,
        temperature=0.0,
    )

    assert run_solver_evaluation(args) == 1


def test_standalone_solver_eval_programmatic_call_rejects_conflicting_selection():
    from trainer.solver_eval.cli import run_solver_evaluation

    args = SimpleNamespace(
        env_id="connect4",
        all_checkpoints=True,
        model="candidate.onnx",
        games=1,
        seed=0,
        temperature=0.0,
    )

    assert run_solver_evaluation(args) == 1


def test_loop_cli_carries_all_optimizer_search_and_evaluation_settings():
    parser = cli.build_parser(get_algorithm(ALGORITHM_ID))
    args = parser.parse_args(
        [
            "--algorithm",
            ALGORITHM_ID,
            "loop",
            "--weight-decay",
            "0.03",
            "--grad-clip",
            "2.25",
            "--c-puct",
            "1.75",
            "--temperature",
            "0.9",
            "--late-temperature",
            "0.15",
            "--dirichlet-alpha",
            "0.45",
            "--dirichlet-weight",
            "0.3",
            "--eval-simulations",
            "33",
            "--eval-temperature",
            "0.05",
        ]
    )

    config = loop_config_from_args(args)

    assert config.weight_decay == 0.03
    assert config.grad_clip_norm == 2.25
    assert config.c_puct == pytest.approx(1.75)
    assert config.temperature == pytest.approx(0.9)
    assert config.late_temperature == pytest.approx(0.15)
    assert config.dirichlet_alpha == pytest.approx(0.45)
    assert config.dirichlet_weight == pytest.approx(0.3)
    assert config.eval_simulations == 33
    assert config.eval_temperature == pytest.approx(0.05)


def test_loop_cli_rejects_removed_checkpoint_interval():
    parser = cli.build_parser(get_algorithm(ALGORITHM_ID))

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--algorithm",
                ALGORITHM_ID,
                "loop",
                "--checkpoint-interval",
                "10",
            ]
        )


def test_train_paths_follow_the_cli_selected_environment(monkeypatch):
    from trainer import metrics

    algorithm = get_algorithm(ALGORITHM_ID)
    parser = cli.build_parser(algorithm)
    args = parser.parse_args(
        [
            "--algorithm",
            ALGORITHM_ID,
            "train",
            "--env-id",
            "othello",
            "--collection-scope-id",
            "a" * 64,
            "--source-root",
        ]
    )
    assert not hasattr(args, "model_dir")
    assert not hasattr(args, "stats_path")

    captured = []
    monkeypatch.setattr(metrics, "start_metrics_server", lambda **kwargs: None)
    monkeypatch.setattr(
        algorithm,
        "build_learner",
        lambda config: SimpleNamespace(
            train=lambda: SimpleNamespace(
                metrics={"loss/total": 0.5}, last_checkpoint="checkpoint"
            ),
            config=captured.append(config),
        ),
    )

    assert args._command_runner(args) == 0

    profile_dir = resolve_runtime_profile(ALGORITHM_ID, "othello").data_dir(get_config().data_root)
    assert Path(args.model_dir) == profile_dir / "models"
    assert Path(args.stats_path) == profile_dir / "stats.json"


@pytest.mark.parametrize(
    ("command", "module_name", "runner_name", "expected_paths"),
    [
        (
            "solver-eval",
            "trainer.solver_eval",
            "run_solver_evaluation",
            {
                "models_dir": "models",
            },
        ),
        (
            "register-players",
            "trainer.tournament_cli",
            "run_register_players",
            {"models_dir": "models", "registry": "players.json"},
        ),
        (
            "tournament",
            "trainer.tournament_cli",
            "run_tournament_command",
            {"registry": "players.json", "output": "tournament.json"},
        ),
    ],
)
def test_artifact_commands_resolve_profile_defaults_after_env_parsing(
    monkeypatch, command, module_name, runner_name, expected_paths
):
    algorithm = get_algorithm(ALGORITHM_ID)
    parser = cli.build_parser(algorithm)
    args = parser.parse_args(["--algorithm", ALGORITHM_ID, command, "--env-id", "connect4"])
    assert all(not hasattr(args, attribute) for attribute in expected_paths)

    calls = []
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, runner_name, lambda parsed: calls.append(parsed) or 0)

    assert args._command_runner(args) == 0
    assert calls == [args]

    profile_dir = resolve_runtime_profile(ALGORITHM_ID, "connect4").data_dir(get_config().data_root)
    for attribute, relative_path in expected_paths.items():
        assert Path(getattr(args, attribute)) == profile_dir / relative_path


def test_evaluate_default_resolves_verified_current_checkpoint(monkeypatch):
    algorithm = get_algorithm(ALGORITHM_ID)
    parser = cli.build_parser(algorithm)
    args = parser.parse_args(["--algorithm", ALGORITHM_ID, "evaluate", "--env-id", "connect4"])
    assert not hasattr(args, "model")
    profile_dir = resolve_runtime_profile(ALGORITHM_ID, "connect4").data_dir(get_config().data_root)
    immutable = profile_dir / "models" / "blobs" / "sha256" / f"{'a' * 64}.onnx"
    from trainer import evaluator
    from trainer.storage import publisher

    monkeypatch.setattr(
        publisher,
        "create_checkpoint_publisher",
        lambda *_args, **_kwargs: SimpleNamespace(
            resolve_head=lambda: SimpleNamespace(onnx_path=immutable)
        ),
    )
    calls = []
    monkeypatch.setattr(evaluator, "run_evaluation", lambda parsed: calls.append(parsed) or 0)

    assert args._command_runner(args) == 0
    assert calls == [args]
    assert Path(args.model) == immutable


def test_shared_entrypoint_dispatches_through_the_selected_binding(monkeypatch):
    calls = []

    def configure(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--value", type=int, required=True)

    def run(args: argparse.Namespace) -> int:
        calls.append((args.algorithm, args.value))
        return 17

    fake = SimpleNamespace(
        descriptor=SimpleNamespace(id="test_algorithm"),
        commands=lambda: (
            AlgorithmCommand(
                name="execute",
                help="execute the fake algorithm",
                configure_parser=configure,
                run=run,
            ),
        ),
    )
    monkeypatch.setattr(cli, "_select_algorithm", lambda argv: fake)
    monkeypatch.setattr(cli, "list_algorithms", lambda: ["test_algorithm"])
    monkeypatch.setattr(cli, "setup_logging", lambda **kwargs: None)

    assert cli.main(["--algorithm", "test_algorithm", "execute", "--value", "9"]) == 17
    assert calls == [("test_algorithm", 9)]


def test_descriptor_component_drift_rejects_the_python_binding(monkeypatch):
    bad_components = replace(DESCRIPTOR.components, learner="different_learner")
    monkeypatch.setattr(
        AlphaZeroBoardV1,
        "descriptor",
        replace(DESCRIPTOR, components=bad_components),
    )

    with pytest.raises(RuntimeError, match="learner='different_learner'"):
        AlphaZeroBoardV1()
