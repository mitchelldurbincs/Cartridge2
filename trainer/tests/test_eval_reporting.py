"""Solver-evidence projection tests for the orchestrator reporting mixin.

Regression scope: `_build_solver_record` previously read the checkpoint
manifest as `checkpoint.manifest.step`, which raised AttributeError the first
time a solver-enabled run projected its history — after the RunHead had
already committed.
"""

from __future__ import annotations

import pytest
import torch
from torch.optim import Adam

from trainer.checkpoint import (
    LearnerStateContract,
    export_onnx_artifact,
    write_learner_state_artifact,
)
from trainer.network import PolicyValueNetwork
from trainer.orchestrator.eval_reporting import EvalReportingMixin
from trainer.storage.evaluation import (
    EvaluationArtifactV2,
    EvaluationRecipeV1,
    FilesystemEvaluationRepository,
    HeadToHeadResultV1,
    ObservedResultsV1,
    PromotionDecisionV1,
    RequestedGamesV1,
    SolverBucketV1,
    SolverResultV1,
)
from trainer.storage.publisher import (
    FilesystemCheckpointPublisher,
    OnnxArtifactContract,
)

CONTRACT = OnnxArtifactContract(
    algorithm_id="alphazero_board_v1",
    env_id="tictactoe",
    env_contract_version=1,
    model_artifact_schema_version=1,
    model_contract="onnx_policy_value_v1",
    obs_size=29,
    num_actions=9,
)
CONFIG_SHA256 = "a" * 64
CHECKPOINT_STEP = 7


@pytest.fixture(scope="module")
def staged_blobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval-reporting-checkpoint")
    network = PolicyValueNetwork(obs_size=29, action_size=9)
    optimizer = Adam(network.parameters(), lr=0.001)
    return (
        export_onnx_artifact(
            network,
            29,
            root / "model.onnx",
            torch.device("cpu"),
            CONTRACT,
        ),
        write_learner_state_artifact(
            network,
            optimizer,
            CHECKPOINT_STEP,
            root / "learner.pt",
            CONTRACT,
            CONFIG_SHA256,
        ),
        LearnerStateContract(network, optimizer),
    )


def solver_result() -> SolverResultV1:
    def bucket(counts):
        return SolverBucketV1(*counts)

    return SolverResultV1(
        games_played=2,
        candidate_wins=1,
        opponent_wins=1,
        draws=0,
        average_game_length=9.5,
        # (positions, value_optimal, exact_best, w->d, w->l, d->l, forced)
        overall=bucket((10, 8, 6, 1, 1, 0, 2)),
        by_ply={
            "ply_1_8": bucket((6, 5, 4, 1, 0, 0, 2)),
            "ply_9_20": bucket((4, 3, 2, 0, 1, 0, 0)),
            "ply_21_plus": bucket((0, 0, 0, 0, 0, 0, 0)),
        },
        by_seat={
            "first": bucket((5, 4, 3, 1, 0, 0, 1)),
            "second": bucket((5, 4, 3, 0, 1, 0, 1)),
        },
        solver_queries=10,
        solver_cache_hits=3,
        solver_time_seconds=0.5,
        wall_time_seconds=1.5,
        solver_version="bitbully-1.2.3",
    )


def head_to_head(*, wins=1, losses=0, draws=1) -> HeadToHeadResultV1:
    games = wins + losses + draws
    return HeadToHeadResultV1(
        games_played=games,
        candidate_wins=wins,
        opponent_wins=losses,
        draws=draws,
        candidate_wins_as_first=min(wins, (games + 1) // 2),
        candidate_wins_as_second=max(0, wins - (games + 1) // 2),
        opponent_wins_while_candidate_first=0,
        opponent_wins_while_candidate_second=losses,
        average_game_length=7.0,
    )


def solver_artifact(candidate_id, *, solver=None) -> EvaluationArtifactV2:
    vs_random = head_to_head()
    return EvaluationArtifactV2(
        profile=CONTRACT.profile,
        iteration=1,
        candidate_checkpoint_id=candidate_id,
        previous_evaluation_id=None,
        champion_before=None,
        recipe=EvaluationRecipeV1(
            simulations=0,
            temperature=0.2,
            promotion_metric="win_rate",
            promotion_margin=0.0,
            win_threshold=0.55,
            seed=42,
            requested_games=RequestedGamesV1(
                vs_champion=0,
                vs_random=vs_random.games_played,
                candidate_solver=solver.games_played if solver else 0,
                champion_solver=0,
            ),
        ),
        results=ObservedResultsV1(
            vs_champion=None,
            vs_random=vs_random,
            candidate_solver=solver,
            champion_solver=None,
        ),
        decision=PromotionDecisionV1(promoted=True, reason="strict decision"),
        started_at="2026-08-02T10:00:00.000000Z",
        completed_at="2026-08-02T10:00:01.000000Z",
    )


class _Reporter(EvalReportingMixin):
    """Minimal host exposing exactly what the mixin reads."""

    def __init__(self, checkpoints, evaluations):
        self.checkpoints = checkpoints
        self.evaluations = evaluations
        self.wandb_logger = None


@pytest.fixture()
def published(tmp_path, staged_blobs):
    onnx_path, learner_path, learner_contract = staged_blobs
    checkpoints = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    evaluations = FilesystemEvaluationRepository(checkpoints)
    candidate = checkpoints.stage_checkpoint(
        onnx_path,
        learner_path,
        step=CHECKPOINT_STEP,
        parent_checkpoint_id=None,
        config_sha256=CONFIG_SHA256,
        learner_state_contract=learner_contract,
    )
    reference = evaluations.publish_evidence(
        solver_artifact(candidate.checkpoint_id, solver=solver_result())
    )
    return _Reporter(checkpoints, evaluations), reference


def test_solver_history_projects_checkpoint_step_and_valid_entries(published):
    reporter, reference = published

    history = reporter._build_solver_history([reference])

    assert len(history) == 1
    entry = history[0]
    # The regression: global_step must come from the validated checkpoint
    # manifest's step field, exactly like step.
    assert entry["step"] == CHECKPOINT_STEP
    assert entry["global_step"] == CHECKPOINT_STEP
    assert entry["checkpoint_id"] == reference.artifact.candidate_checkpoint_id
    assert entry["evaluation_id"] == reference.evaluation_id
    assert entry["value_optimal_rate"] == pytest.approx(0.8)
    assert entry["exact_best_rate"] == pytest.approx(0.6)
    assert entry["blunder_rate"] == pytest.approx(0.2)
    assert entry["solver_cache_hit_rate"] == pytest.approx(0.3)
    assert entry["context"] == "loop"


def test_solver_record_is_absent_without_solver_evidence(tmp_path, staged_blobs):
    onnx_path, learner_path, learner_contract = staged_blobs
    checkpoints = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    evaluations = FilesystemEvaluationRepository(checkpoints)
    candidate = checkpoints.stage_checkpoint(
        onnx_path,
        learner_path,
        step=CHECKPOINT_STEP,
        parent_checkpoint_id=None,
        config_sha256=CONFIG_SHA256,
        learner_state_contract=learner_contract,
    )
    reference = evaluations.publish_evidence(
        solver_artifact(candidate.checkpoint_id, solver=None)
    )
    reporter = _Reporter(checkpoints, evaluations)

    assert reporter._build_solver_record(reference) is None
    assert reporter._build_solver_history([reference]) == []


def test_eval_record_projects_solver_rates_from_the_artifact(published):
    reporter, reference = published

    record = reporter._build_eval_record(reference)

    assert record["step"] == CHECKPOINT_STEP
    assert record["solver_value_optimal_rate"] == pytest.approx(0.8)
    assert record["solver_exact_best_rate"] == pytest.approx(0.6)
    assert record["solver_blunder_rate"] == pytest.approx(0.2)
    assert record["solver_positions"] == 10
    assert record["promoted"] is True
