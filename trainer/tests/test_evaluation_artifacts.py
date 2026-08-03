"""Evaluation V2 immutable-chain integrity tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from torch.optim import Adam

from trainer.checkpoint import (
    LearnerStateContract,
    export_onnx_artifact,
    write_learner_state_artifact,
)
from trainer.network import PolicyValueNetwork
from trainer.storage.evaluation import (
    ChampionReferenceV1,
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
    ArtifactValidationError,
    FilesystemCheckpointPublisher,
    OnnxArtifactContract,
    canonical_json_bytes,
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


@pytest.fixture(scope="module")
def staged_blobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluation-v2-checkpoint")
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
            1,
            root / "learner.pt",
            CONTRACT,
            CONFIG_SHA256,
        ),
        LearnerStateContract(network, optimizer),
    )


def stage_checkpoint(repository, staged_blobs, step, parent=None):
    onnx_path, learner_path, learner_contract = staged_blobs
    if step != 1:
        staged = learner_path.with_name(f"learner-{step}.pt")
        state = torch.load(learner_path, map_location="cpu", weights_only=True)
        state["step"] = step
        torch.save(state, staged)
        learner_path = staged
    return repository.stage_checkpoint(
        onnx_path,
        learner_path,
        step=step,
        parent_checkpoint_id=parent,
        config_sha256=CONFIG_SHA256,
        learner_state_contract=learner_contract,
    )


def result(*, wins=1, losses=0, draws=1):
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


def artifact(
    candidate_id,
    *,
    iteration=1,
    previous=None,
    champion=None,
    promoted=True,
    started="2026-08-02T10:00:00.000000Z",
    completed="2026-08-02T10:00:01.000000Z",
    vs_champion=None,
    vs_random=None,
):
    return EvaluationArtifactV2(
        profile=CONTRACT.profile,
        iteration=iteration,
        candidate_checkpoint_id=candidate_id,
        previous_evaluation_id=previous,
        champion_before=champion,
        recipe=EvaluationRecipeV1(
            simulations=0,
            temperature=0.2,
            promotion_metric="win_rate",
            promotion_margin=0.0,
            win_threshold=0.55,
            seed=42,
            requested_games=RequestedGamesV1(
                vs_champion=vs_champion.games_played if vs_champion else 0,
                vs_random=vs_random.games_played if vs_random else 0,
                candidate_solver=0,
                champion_solver=0,
            ),
        ),
        results=ObservedResultsV1(
            vs_champion=vs_champion,
            vs_random=vs_random,
            candidate_solver=None,
            champion_solver=None,
        ),
        decision=PromotionDecisionV1(promoted=promoted, reason="strict decision"),
        started_at=started,
        completed_at=completed,
    )


def repositories(tmp_path):
    checkpoints = FilesystemCheckpointPublisher(model_root=tmp_path, contract=CONTRACT)
    return checkpoints, FilesystemEvaluationRepository(checkpoints)


def test_v2_evidence_is_canonical_immutable_and_has_no_champion_channel(
    tmp_path, staged_blobs
):
    checkpoints, evaluations = repositories(tmp_path)
    candidate = stage_checkpoint(checkpoints, staged_blobs, 1)
    evidence = artifact(candidate.checkpoint_id, vs_random=result())

    reference = evaluations.publish_evidence(evidence)

    assert reference.path.read_bytes() == evidence.to_bytes()
    assert reference.evaluation_id == evidence.evaluation_id
    assert evaluations.publish_evidence(evidence) == reference
    assert evaluations.list_evaluations(reference.evaluation_id) == [reference]
    assert not (tmp_path / "channels" / "champion.json").exists()
    assert json.loads(reference.path.read_text())["schema_version"] == 2


def test_rejected_evidence_advances_chain_without_changing_champion_lineage(
    tmp_path, staged_blobs
):
    checkpoints, evaluations = repositories(tmp_path)
    first = stage_checkpoint(checkpoints, staged_blobs, 1)
    first_ref = evaluations.publish_evidence(
        artifact(first.checkpoint_id, vs_random=result())
    )
    champion = ChampionReferenceV1(first.checkpoint_id, first_ref.evaluation_id)
    second = stage_checkpoint(checkpoints, staged_blobs, 2, parent=first.checkpoint_id)
    rejected = artifact(
        second.checkpoint_id,
        iteration=2,
        previous=first_ref.evaluation_id,
        champion=champion,
        promoted=False,
        started="2026-08-02T10:00:02.000000Z",
        completed="2026-08-02T10:00:03.000000Z",
        vs_champion=result(wins=0, losses=1, draws=1),
    )

    rejected_ref = evaluations.publish_evidence(rejected)

    assert [
        item.evaluation_id
        for item in evaluations.list_evaluations(rejected_ref.evaluation_id)
    ] == [first_ref.evaluation_id, rejected_ref.evaluation_id]
    assert rejected_ref.artifact.champion_before == champion


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"iteration": 1}, "iterations must be strictly increasing"),
        (
            {
                "started_at": "2026-08-02T10:00:01.000000Z",
                "completed_at": "2026-08-02T10:00:02.000000Z",
            },
            "timestamps must be strictly increasing",
        ),
    ],
)
def test_publication_rejects_nonincreasing_iteration_or_time_before_write(
    tmp_path, staged_blobs, change, message
):
    checkpoints, evaluations = repositories(tmp_path)
    first = stage_checkpoint(checkpoints, staged_blobs, 1)
    first_ref = evaluations.publish_evidence(
        artifact(first.checkpoint_id, vs_random=result())
    )
    second = stage_checkpoint(checkpoints, staged_blobs, 2, parent=first.checkpoint_id)
    values = {
        "iteration": 2,
        "previous": first_ref.evaluation_id,
        "champion": ChampionReferenceV1(first.checkpoint_id, first_ref.evaluation_id),
        "promoted": False,
        "started": "2026-08-02T10:00:02.000000Z",
        "completed": "2026-08-02T10:00:03.000000Z",
        "vs_champion": result(wins=0, losses=1, draws=1),
    }
    if "started_at" in change:
        values["started"] = change["started_at"]
        values["completed"] = change["completed_at"]
    else:
        values.update(change)
    invalid = artifact(second.checkpoint_id, **values)

    with pytest.raises(ArtifactValidationError, match=message):
        evaluations.publish_evidence(invalid)

    assert not evaluations._evaluation_path(invalid.evaluation_id).exists()


def test_recipe_rejects_evidence_free_candidate():
    with pytest.raises(ArtifactValidationError, match="candidate evidence"):
        EvaluationRecipeV1(
            simulations=0,
            temperature=0.2,
            promotion_metric="win_rate",
            promotion_margin=0.0,
            win_threshold=0.55,
            seed=0,
            requested_games=RequestedGamesV1(0, 0, 0, 0),
        )


@pytest.mark.parametrize(
    ("games", "average"),
    [(0, 1.0), (1, 0.0)],
)
def test_head_to_head_average_length_is_zero_iff_no_games(games, average):
    with pytest.raises(ArtifactValidationError, match="average game length"):
        HeadToHeadResultV1(
            games_played=games,
            candidate_wins=games,
            opponent_wins=0,
            draws=0,
            candidate_wins_as_first=games,
            candidate_wins_as_second=0,
            opponent_wins_while_candidate_first=0,
            opponent_wins_while_candidate_second=0,
            average_game_length=average,
        )


def test_solver_evidence_requires_concrete_version():
    empty = SolverBucketV1(0, 0, 0, 0, 0, 0, 0)
    with pytest.raises(ArtifactValidationError, match="solver_version"):
        SolverResultV1(
            games_played=0,
            candidate_wins=0,
            opponent_wins=0,
            draws=0,
            average_game_length=0.0,
            overall=empty,
            by_ply={"ply_1_8": empty, "ply_9_20": empty, "ply_21_plus": empty},
            by_seat={"first": empty, "second": empty},
            solver_queries=0,
            solver_cache_hits=0,
            solver_time_seconds=0.0,
            wall_time_seconds=0.0,
            solver_version=None,
        )


def test_v1_or_noncanonical_artifact_is_rejected(tmp_path, staged_blobs):
    checkpoints, _ = repositories(tmp_path)
    candidate = stage_checkpoint(checkpoints, staged_blobs, 1)
    evidence = artifact(candidate.checkpoint_id, vs_random=result())
    raw = evidence.to_dict()
    raw["schema_version"] = 1

    with pytest.raises(ArtifactValidationError, match="schema_version"):
        EvaluationArtifactV2.from_bytes(canonical_json_bytes(raw))

    with pytest.raises(ArtifactValidationError, match="canonical"):
        EvaluationArtifactV2.from_bytes(
            json.dumps(evidence.to_dict(), indent=2).encode()
        )


def test_promotion_margin_is_a_rate():
    with pytest.raises(ArtifactValidationError, match="valid range"):
        replace(
            EvaluationRecipeV1(
                simulations=0,
                temperature=0.2,
                promotion_metric="win_rate",
                promotion_margin=0.0,
                win_threshold=0.55,
                seed=0,
                requested_games=RequestedGamesV1(0, 1, 0, 0),
            ),
            promotion_margin=1.1,
        )


@pytest.mark.parametrize(
    "recipe",
    [
        lambda: EvaluationRecipeV1(
            simulations=0,
            temperature=0.2,
            promotion_metric="win_rate",
            promotion_margin=0.1,
            win_threshold=0.55,
            seed=0,
            requested_games=RequestedGamesV1(0, 1, 0, 0),
        ),
        lambda: EvaluationRecipeV1(
            simulations=0,
            temperature=0.2,
            promotion_metric="solver_optimal",
            promotion_margin=0.1,
            win_threshold=0.55,
            seed=0,
            requested_games=RequestedGamesV1(0, 0, 1, 0),
        ),
    ],
)
def test_inactive_promotion_parameters_are_rejected(recipe):
    with pytest.raises(ArtifactValidationError, match="must be zero"):
        recipe()
