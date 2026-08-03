"""Disposable projection rebuilding from authoritative RunCommit inputs."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from crucible.orchestrator.config import IterationStats

from trainer.orchestrator import durable_json as durable_json_module
from trainer.orchestrator.config import LoopConfig
from trainer.orchestrator.stats_manager import StatsManager
from trainer.solver_eval import SolverEvalResults, judge_move
from trainer.stats import TrainerStats, prepare_stats_snapshot
from trainer.storage.publisher import (
    BlobDescriptorV1,
    CheckpointManifestV1,
    CheckpointProfileV1,
    CheckpointRef,
)

PROFILE = CheckpointProfileV1(
    algorithm_id="alphazero_board_v1",
    env_id="tictactoe",
    env_contract_version=1,
    model_artifact_schema_version=1,
    model_contract="onnx_policy_value_v1",
)


def loop_record(
    iteration: int,
    *,
    win_rate: float | None = None,
    draw_rate: float | None = None,
) -> IterationStats:
    return IterationStats(
        iteration=iteration,
        episodes_generated=2,
        transitions_generated=10,
        training_steps=3,
        actor_time_seconds=1.0,
        trainer_time_seconds=2.0,
        eval_time_seconds=0.5 if win_rate is not None else 0.0,
        total_time_seconds=3.5,
        eval_win_rate=win_rate,
        eval_draw_rate=draw_rate,
        timestamp=f"2026-08-02T10:00:0{iteration}",
    )


def eval_record(
    iteration: int,
    *,
    evaluation_id: str | None = None,
    champion: dict | None = None,
    promoted: bool = True,
) -> dict:
    return {
        "iteration": iteration,
        "step": iteration * 3,
        "candidate_checkpoint_id": f"{iteration:064x}",
        "evaluation_id": evaluation_id or f"{iteration + 100:064x}",
        "vs_champion_checkpoint_id": (
            champion["candidate_checkpoint_id"] if champion is not None else None
        ),
        "vs_champion_evaluation_id": (
            champion["evaluation_id"] if champion is not None else None
        ),
        "vs_champion_win_rate": 0.5 if champion is not None else None,
        "vs_champion_draw_rate": 0.25 if champion is not None else None,
        "vs_champion_average_game_length": 5.0 if champion is not None else None,
        "vs_champion_iteration": (
            champion["iteration"] if champion is not None else None
        ),
        "promoted": promoted,
        "promotion_reason": "promotion decision",
        "vs_random_win_rate": None,
        "vs_random_draw_rate": None,
        "vs_random_average_game_length": None,
        "solver_value_optimal_rate": None,
        "solver_exact_best_rate": None,
        "solver_blunder_rate": None,
        "solver_positions": None,
        "promotion_metric": "win_rate",
        "requested_vs_champion_games": 4 if champion is not None else 0,
        "requested_vs_random_games": 0,
        "timestamp": f"2026-08-02T10:00:0{iteration}.000000Z",
    }


def solver_entry(iteration: int, *, evaluation_id: str | None = None) -> dict:
    results = SolverEvalResults(
        env_id="connect4",
        model_name="candidate",
        model_path=f"checkpoint:{iteration:064x}",
        checkpoint_id=f"{iteration:064x}",
        step=iteration * 3,
        opponent_name="Random",
        games=2,
        seed=42,
        temperature=0.0,
    )
    for index, judgment in enumerate(
        (
            judge_move({3: 5, 0: -1}, chosen=3),
            judge_move({3: 5, 0: -1}, chosen=0),
        )
    ):
        results.overall.add(judgment)
        results.by_ply["ply_1_8"].add(judgment)
        results.by_seat["first" if index == 0 else "second"].add(judgment)
    results.model_wins = 1
    results.draws = 1
    results.avg_game_length = 5.0
    results.solver_queries = 2
    results.solver_cache_hits = 1
    results.solver_time_seconds = 0.25
    results.wall_time_seconds = 0.5
    results.timestamp = f"2026-08-02T10:00:0{iteration}"
    entry = results.to_dict()
    entry.update(
        {
            "iteration": iteration,
            "global_step": iteration * 3,
            "context": "loop",
            "evaluation_id": evaluation_id or f"{iteration + 200:064x}",
        }
    )
    return entry


def stats_snapshot(step: int = 3):
    manifest = CheckpointManifestV1(
        profile=PROFILE,
        step=step,
        parent_checkpoint_id=None,
        config_sha256="c" * 64,
        onnx=BlobDescriptorV1("a" * 64, 1),
        learner_state=BlobDescriptorV1("b" * 64, 1),
    )
    checkpoint = CheckpointRef(
        checkpoint_id=manifest.checkpoint_id,
        manifest=manifest,
        onnx_path=Path("unused.onnx"),
        learner_state_path=Path("unused.pt"),
    )
    stats = TrainerStats(
        step=step,
        total_steps=step,
        samples_seen=step * 2,
        last_checkpoint=checkpoint.checkpoint_id,
        env_id="tictactoe",
    )
    return prepare_stats_snapshot(stats, checkpoint)


def manager(tmp_path, **overrides) -> StatsManager:
    return StatsManager(LoopConfig(data_dir=tmp_path, env_id="tictactoe", **overrides))


def test_rebuild_writes_every_projection_from_supplied_authority(tmp_path):
    stats_manager = manager(tmp_path)
    first = eval_record(1)
    second = eval_record(3, champion=first)
    loops = [
        loop_record(1),
        loop_record(2),
        loop_record(3, win_rate=0.5, draw_rate=0.25),
    ]

    stats_manager.rebuild_projections(
        history=loops,
        eval_history=[first, second],
        solver_history=[solver_entry(1), solver_entry(3)],
        stats_snapshot=stats_snapshot(),
    )

    assert (
        json.loads(stats_manager.config.loop_stats_path.read_bytes())["iterations"][-1][
            "iteration"
        ]
        == 3
    )
    assert json.loads(stats_manager.config.eval_stats_path.read_bytes())[
        "evaluations"
    ] == [
        first,
        second,
    ]
    assert (
        len(
            json.loads(stats_manager.config.solver_stats_path.read_bytes())[
                "solver_evaluations"
            ]
        )
        == 2
    )
    assert json.loads(stats_manager.config.stats_path.read_bytes())["step"] == 3


def test_projection_rebuild_uses_actual_chain_not_configured_modulo(tmp_path):
    stats_manager = manager(tmp_path, eval_interval=1)
    first = eval_record(1)
    third = eval_record(3, champion=first)

    stats_manager.rebuild_projections(
        history=[
            loop_record(1),
            loop_record(2),
            loop_record(3, win_rate=0.5, draw_rate=0.25),
        ],
        eval_history=[first, third],
        solver_history=[],
        stats_snapshot=stats_snapshot(),
    )

    assert stats_manager.config.eval_stats_path.is_file()


def test_rebuild_rejects_evaluation_without_loop_iteration(tmp_path):
    with pytest.raises(ValueError, match="no completed loop iteration"):
        manager(tmp_path).rebuild_projections(
            history=[loop_record(1)],
            eval_history=[eval_record(2)],
            solver_history=[],
            stats_snapshot=stats_snapshot(),
        )


def test_rebuild_rejects_loop_and_evaluation_rate_disagreement(tmp_path):
    first = eval_record(1)
    second = eval_record(2, champion=first)
    with pytest.raises(ValueError, match="disagrees with its loop iteration"):
        manager(tmp_path).rebuild_projections(
            history=[loop_record(1), loop_record(2, win_rate=0.4, draw_rate=0.25)],
            eval_history=[first, second],
            solver_history=[],
            stats_snapshot=stats_snapshot(),
        )


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                eval_record(1, evaluation_id="f" * 64),
                eval_record(
                    3,
                    evaluation_id="f" * 64,
                    champion=eval_record(1, evaluation_id="f" * 64),
                ),
            ],
            "identifiers must be unique",
        ),
        ([eval_record(3), eval_record(1)], "iterations must be strictly increasing"),
    ],
)
def test_eval_projection_rejects_duplicate_and_out_of_order_records(
    tmp_path, records, message
):
    with pytest.raises(ValueError, match=message):
        manager(tmp_path).save_eval_stats(records)


def test_eval_projection_rejects_stale_champion_and_bad_timestamp(tmp_path):
    stats_manager = manager(tmp_path)
    first = eval_record(1)
    second = eval_record(2, champion=first)
    with pytest.raises(ValueError, match="latest champion"):
        stats_manager.save_eval_stats([first, second, eval_record(3, champion=first)])
    first["timestamp"] = "2026-08-02 10:00:01.000000Z"
    with pytest.raises(ValueError, match="UTC timestamp format"):
        stats_manager.save_eval_stats([first])


@pytest.mark.parametrize(
    ("positions", "optimal", "exact", "blunder", "message"),
    [
        (10, 0.6, 0.7, 0.4, "exact-best rate exceeds"),
        (10, 0.6, 0.5, 0.3, "does not complement"),
        (0, 0.1, 0.0, 0.0, "zero-position solver rates must be zero"),
    ],
)
def test_eval_projection_rejects_inconsistent_solver_rates(
    tmp_path, positions, optimal, exact, blunder, message
):
    record = eval_record(1)
    record.update(
        {
            "solver_positions": positions,
            "solver_value_optimal_rate": optimal,
            "solver_exact_best_rate": exact,
            "solver_blunder_rate": blunder,
        }
    )
    with pytest.raises(ValueError, match=message):
        manager(tmp_path).save_eval_stats([record])


def test_solver_projection_rejects_invalid_duplicate_and_out_of_order_records(tmp_path):
    stats_manager = manager(tmp_path)
    invalid = solver_entry(1)
    invalid["solver_queries"] = 1
    with pytest.raises(ValueError, match="must equal positions_scored"):
        stats_manager.save_solver_stats([invalid])
    duplicate = "e" * 64
    with pytest.raises(ValueError, match="identifiers must be unique"):
        stats_manager.save_solver_stats(
            [
                solver_entry(1, evaluation_id=duplicate),
                solver_entry(2, evaluation_id=duplicate),
            ]
        )
    with pytest.raises(ValueError, match="iterations must be strictly increasing"):
        stats_manager.save_solver_stats([solver_entry(2), solver_entry(1)])


def test_projection_writes_fsync_file_and_parent_directory(tmp_path, monkeypatch):
    sync_kinds: list[str] = []
    real_fsync = durable_json_module.os.fsync

    def recording_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        sync_kinds.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(durable_json_module.os, "fsync", recording_fsync)
    stats_manager = manager(tmp_path)
    for write in (
        lambda: stats_manager.save_loop_stats([loop_record(1)]),
        lambda: stats_manager.save_eval_stats([eval_record(1)]),
        lambda: stats_manager.save_solver_stats([solver_entry(1)]),
    ):
        sync_kinds.clear()
        write()
        assert "file" in sync_kinds
        assert "directory" in sync_kinds
