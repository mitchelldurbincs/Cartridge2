"""Characterize the schema of disposable solver-history projections."""

import json
import re
from copy import deepcopy

import pytest

from trainer.orchestrator.config import LoopConfig
from trainer.orchestrator.solver_reporting_validation import validate_solver_history
from trainer.orchestrator.stats_manager import StatsManager


@pytest.fixture
def solver_entry():
    """A report with independent expected counts, including an empty ply bucket."""
    bucket = {
        "positions": 1,
        "value_optimal": 1,
        "exact_best": 1,
        "blunders_win_to_draw": 0,
        "blunders_win_to_loss": 0,
        "blunders_draw_to_loss": 0,
        "forced": 0,
        "value_optimal_rate": 1.0,
        "exact_best_rate": 1.0,
        "blunder_rate": 0.0,
        "forced_rate": 0.0,
    }
    empty = {name: 0 for name in bucket}
    return {
        "model": "checkpoint:" + "a" * 64,
        "model_path": "checkpoint:" + "a" * 64,
        "checkpoint_id": "a" * 64,
        "step": 7,
        "env_id": "connect4",
        "opponent": "random_v1",
        "games": 2,
        "seed": 42,
        "temperature": 0.0,
        "model_wins": 1,
        "model_losses": 0,
        "draws": 1,
        "avg_game_length": 9.5,
        "positions_scored": 2,
        "forced_moves": 0,
        "forced_move_rate": 0.0,
        "value_optimal_rate": 1.0,
        "exact_best_rate": 1.0,
        "blunder_rate": 0.0,
        "blunders_win_to_draw": 0,
        "blunders_win_to_loss": 0,
        "blunders_draw_to_loss": 0,
        "by_ply": {
            "ply_1_8": dict(bucket),
            "ply_9_20": dict(bucket),
            "ply_21_plus": dict(empty),
        },
        "by_seat": {"first": dict(bucket), "second": dict(bucket)},
        "solver_queries": 2,
        "solver_cache_hits": 1,
        "solver_cache_hit_rate": 0.5,
        "solver_time_seconds": 0.5,
        "wall_time_seconds": 1.5,
        "bitbully_version": "test-solver-v1",
        "timestamp": "2026-08-02T10:00:01.000000Z",
        "iteration": 1,
        "global_step": 7,
        "context": "loop",
        "evaluation_id": "b" * 64,
    }


def test_valid_history_preserves_records_and_allows_gaps(solver_entry):
    later = deepcopy(solver_entry)
    later.update(iteration=3, evaluation_id="c" * 64, step=21, global_step=21)
    history = [solver_entry, later]
    original = deepcopy(history)

    assert validate_solver_history(history) == original
    assert history == original
    assert validate_solver_history([]) == []


@pytest.mark.parametrize("value", [None, {}, ()])
def test_history_requires_a_list(value):
    with pytest.raises(ValueError, match="solver_evaluations must be a list"):
        validate_solver_history(value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("games", True, "games must be a positive integer"),
        ("positions_scored", -1, "positions_scored must be a nonnegative integer"),
        ("value_optimal_rate", 0.5, "value_optimal_rate does not match its source counts"),
        ("exact_best_rate", float("nan"), "exact_best_rate is outside its valid range"),
        ("blunder_rate", 1.1, "blunder_rate is outside its valid range"),
        ("model_wins", 2, "game outcomes must partition games"),
        ("global_step", 8, "step and global_step must match"),
        ("checkpoint_id", "invalid", "checkpoint_id"),
        ("evaluation_id", "A" * 64, "evaluation_id"),
        ("context", "standalone", "context must be exactly 'loop'"),
        ("seed", 2**64 - 1, "seed plus game index exceeds u64"),
        ("positions_scored", 3, "positions_scored does not match solver slices"),
        ("solver_queries", 3, "solver_queries must equal positions_scored"),
        ("solver_cache_hits", 3, "solver_cache_hits cannot exceed solver_queries"),
        ("solver_cache_hit_rate", 0.0, "solver_cache_hit_rate does not match its source counts"),
        ("solver_time_seconds", float("inf"), "solver_time_seconds is outside its valid range"),
        ("timestamp", "not-a-time", "timestamp is invalid"),
        ("bitbully_version", "", "bitbully_version must be null or nonempty"),
    ],
)
def test_invalid_entry_is_rejected_with_field_context(solver_entry, field, value, message):
    solver_entry[field] = value

    with pytest.raises(ValueError, match=re.escape(message)) as error:
        validate_solver_history([solver_entry])

    assert "solver_evaluation[0]" in str(error.value)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_entry_schema_is_exact(solver_entry, change):
    if change == "missing":
        del solver_entry["context"]
    else:
        solver_entry["unexpected"] = 0

    with pytest.raises(ValueError, match=r"solver_evaluation\[0\] has an invalid schema"):
        validate_solver_history([solver_entry])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("positions", True, "positions must be a nonnegative integer"),
        ("forced", 2, "count exceeds positions"),
        ("value_optimal", 0, "exact_best cannot exceed value_optimal"),
        ("blunders_win_to_loss", 1, "blunders do not partition non-optimal positions"),
        ("value_optimal_rate", 0.0, "value_optimal_rate does not match its source counts"),
    ],
)
def test_invalid_bucket_is_rejected_with_slice_context(solver_entry, field, value, message):
    solver_entry["by_ply"]["ply_1_8"][field] = value

    with pytest.raises(ValueError, match=re.escape(message)) as error:
        validate_solver_history([solver_entry])

    assert "solver_evaluation[0].by_ply.ply_1_8" in str(error.value)


def test_individually_valid_slices_must_agree(solver_entry):
    solver_entry["by_seat"]["second"] = dict(solver_entry["by_ply"]["ply_21_plus"])

    with pytest.raises(ValueError, match="solver slices disagree on positions"):
        validate_solver_history([solver_entry])


@pytest.mark.parametrize("iteration", [1, 0])
def test_history_rejects_nonincreasing_iterations(solver_entry, iteration):
    solver_entry["iteration"] = 2
    later = deepcopy(solver_entry)
    later.update(iteration=iteration + 1, evaluation_id="c" * 64)

    with pytest.raises(ValueError, match="iterations must be strictly increasing"):
        validate_solver_history([solver_entry, later])


def test_history_rejects_reused_evaluation_identity(solver_entry):
    later = deepcopy(solver_entry)
    later["iteration"] = 2

    with pytest.raises(ValueError, match="identifiers must be unique"):
        validate_solver_history([solver_entry, later])


def test_save_validates_before_replacing_solver_projection(tmp_path, solver_entry):
    config = LoopConfig(data_dir=tmp_path)
    manager = StatsManager(config)
    manager.save_solver_stats([solver_entry])
    original = config.solver_stats_path.read_bytes()
    assert json.loads(original) == {"solver_evaluations": [solver_entry]}

    invalid = deepcopy(solver_entry)
    invalid["value_optimal_rate"] = 0.0
    with pytest.raises(ValueError, match="value_optimal_rate does not match its source counts"):
        manager.save_solver_stats([invalid])

    assert config.solver_stats_path.read_bytes() == original
