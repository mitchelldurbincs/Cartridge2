"""EvalRunner preparation and immutable commit flow tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pytest
import torch
from torch.optim import Adam

from trainer.algorithms import get_algorithm
from trainer.checkpoint import (
    LearnerStateContract,
    export_onnx_artifact,
    write_learner_state_artifact,
)
from trainer.evaluator import EvalResults
from trainer.network import PolicyValueNetwork
from trainer.orchestrator.config import LoopConfig
from trainer.orchestrator.eval_runner import EvalRunner, PreparedEvaluation
from trainer.orchestrator.orchestrator import _run_recipe
from trainer.stats import EvalStats, TrainerStats, prepare_stats_snapshot
from trainer.storage.evaluation import (
    ChampionReferenceV1,
    FilesystemEvaluationRepository,
    RequestedGamesV1,
    SolverBucketV1,
    SolverResultV1,
)
from trainer.storage.publisher import (
    ArtifactValidationError,
    BlobDescriptorV1,
    CheckpointManifestV1,
    CheckpointRef,
    FilesystemCheckpointPublisher,
    OnnxArtifactContract,
)
from trainer.storage.run_commit import (
    OrchestrationCommitV1,
    RunCommitRepository,
    RunCommitV1,
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
CONNECT4_CONTRACT = OnnxArtifactContract(
    algorithm_id="alphazero_board_v1",
    env_id="connect4",
    env_contract_version=1,
    model_artifact_schema_version=1,
    model_contract="onnx_policy_value_v1",
    obs_size=93,
    num_actions=7,
)
BASE_CONFIG = LoopConfig(
    iterations=3,
    episodes_per_iteration=1,
    steps_per_iteration=1,
    batch_size=1,
    mcts_max_sims=50,
    mcts_sim_ramp_rate=0,
    eval_games=2,
    solver_games=0,
)
RUN_RECIPE = _run_recipe(BASE_CONFIG, get_algorithm(BASE_CONFIG.algorithm_id))
CONFIG_SHA256 = RUN_RECIPE.learner_config_sha256


@pytest.fixture(scope="module")
def staged_blobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluation-flow-v2")
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
        staged = learner_path.with_name(f"flow-learner-{step}.pt")
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


@dataclass(frozen=True)
class StubPolicy:
    model_path: str
    temperature: float


@dataclass(frozen=True)
class EmptyRunHeadRepository:
    contract: OnnxArtifactContract

    def resolve_run_head(self):
        return None


def synthetic_checkpoint(tmp_path, contract: OnnxArtifactContract) -> CheckpointRef:
    blob = BlobDescriptorV1(sha256="a" * 64, size_bytes=1)
    manifest = CheckpointManifestV1(
        profile=contract.profile,
        step=1,
        parent_checkpoint_id=None,
        config_sha256="b" * 64,
        onnx=blob,
        learner_state=blob,
    )
    return CheckpointRef(
        checkpoint_id=manifest.checkpoint_id,
        manifest=manifest,
        onnx_path=tmp_path / "candidate.onnx",
        learner_state_path=tmp_path / "candidate.pt",
    )


def one_game_solver_result() -> SolverResultV1:
    empty = SolverBucketV1(0, 0, 0, 0, 0, 0, 0)
    scored = SolverBucketV1(1, 1, 1, 0, 0, 0, 0)
    return SolverResultV1(
        games_played=1,
        candidate_wins=0,
        opponent_wins=0,
        draws=1,
        average_game_length=42.0,
        overall=scored,
        by_ply={
            "ply_1_8": scored,
            "ply_9_20": empty,
            "ply_21_plus": empty,
        },
        by_seat={"first": scored, "second": empty},
        solver_queries=1,
        solver_cache_hits=0,
        solver_time_seconds=0.0,
        wall_time_seconds=0.0,
        solver_version="test-solver-v1",
    )


def make_runner(tmp_path, checkpoints, evaluations, **overrides):
    values = {
        "data_dir": tmp_path / "runtime",
        "env_id": "tictactoe",
        "eval_games": 2,
        "eval_vs_random": True,
        "eval_simulations": 0,
        "eval_temperature": 0.2,
        "eval_win_threshold": 0.55,
        "solver_games": 0,
        "iterations": 3,
        "episodes_per_iteration": 1,
        "steps_per_iteration": 1,
        "batch_size": 1,
        "mcts_max_sims": 50,
        "mcts_sim_ramp_rate": 0,
    }
    values.update(overrides)
    runner = EvalRunner(
        LoopConfig(**values),
        checkpoint_repository=checkpoints,
        evaluation_repository=evaluations,
    )
    runner._policy_loader = lambda path, temperature: StubPolicy(path, temperature)
    runner._baseline_policy_factory = lambda: StubPolicy("random", 0.0)
    return runner


def outcome(*, candidate_wins, opponent_wins, draws):
    return EvalResults(
        env_id="tictactoe",
        player1_name="candidate",
        player2_name="opponent",
        games_played=2,
        player1_wins=candidate_wins,
        player2_wins=opponent_wins,
        draws=draws,
        player1_wins_as_first=min(candidate_wins, 1),
        player1_wins_as_second=max(0, candidate_wins - 1),
        player2_wins_as_first=opponent_wins,
        player2_wins_as_second=0,
        avg_game_length=7.0,
    )


def repositories(tmp_path):
    checkpoints = FilesystemCheckpointPublisher(
        model_root=tmp_path / "models", contract=CONTRACT
    )
    evaluations = FilesystemEvaluationRepository(checkpoints)
    return checkpoints, evaluations


def establish_first_run(checkpoints, evaluations, runner, candidate):
    runner._run_eval = lambda **_: outcome(candidate_wins=1, opponent_wins=0, draws=1)
    prepared = runner.prepare(1, candidate, None)
    runner.commit(prepared)
    stats = TrainerStats(
        step=1,
        total_steps=1,
        samples_seen=1,
        last_checkpoint=candidate.checkpoint_id,
        env_id="tictactoe",
    )
    vs_random = prepared.artifact.results.vs_random
    assert vs_random is not None
    completed = datetime.fromisoformat(
        prepared.artifact.completed_at.replace("Z", "+00:00")
    )
    stats.append_eval(
        EvalStats(
            step=1,
            win_rate=vs_random.candidate_win_rate,
            draw_rate=vs_random.draw_rate,
            loss_rate=(1.0 - vs_random.candidate_win_rate - vs_random.draw_rate),
            games_played=vs_random.games_played,
            avg_game_length=vs_random.average_game_length,
            timestamp=completed.timestamp(),
        )
    )
    commit = RunCommitV1(
        profile=candidate.manifest.profile,
        config_sha256=candidate.manifest.config_sha256,
        parent_run_commit_id=None,
        checkpoint_id=candidate.checkpoint_id,
        stats_snapshot=prepare_stats_snapshot(stats, candidate),
        champion=ChampionReferenceV1(candidate.checkpoint_id, prepared.evaluation_id),
        evaluation_head_id=prepared.evaluation_id,
        orchestration=OrchestrationCommitV1(
            iteration=1,
            episodes_generated=1,
            transitions_generated=1,
            training_steps=1,
            actor_time_seconds=0.1,
            trainer_time_seconds=0.1,
            eval_time_seconds=prepared.elapsed_seconds,
            total_time_seconds=1.0 + prepared.elapsed_seconds,
            eval_win_rate=None,
            eval_draw_rate=None,
            timestamp=prepared.artifact.completed_at,
            evaluation_id=prepared.evaluation_id,
            collector_simulations=RUN_RECIPE.simulations_for(1),
            collector_seed=None,
            evaluation_seed=RUN_RECIPE.evaluation_seed,
            collection_scope_id="a" * 64,
            source_checkpoint_id=None,
        ),
        run_recipe=RUN_RECIPE,
    )
    RunCommitRepository(checkpoints, evaluations).publish(commit)
    checkpoints.commit_run_head(
        checkpoint_id=candidate.checkpoint_id,
        run_commit_id=commit.run_commit_id,
        expected_run_commit_id=None,
    )
    return commit, prepared


def test_prepare_runs_games_without_writing_evidence_or_mutating_authority(
    tmp_path, staged_blobs
):
    checkpoints, evaluations = repositories(tmp_path)
    candidate = stage_checkpoint(checkpoints, staged_blobs, 1)
    runner = make_runner(tmp_path, checkpoints, evaluations)
    runner._run_eval = lambda **_: outcome(candidate_wins=1, opponent_wins=0, draws=1)

    prepared = runner.prepare(1, candidate, None)

    assert isinstance(prepared, PreparedEvaluation)
    assert prepared.artifact.decision.promoted is True
    assert prepared.win_rate is None
    assert prepared.draw_rate is None
    assert not (checkpoints.model_root / "evaluations").exists()
    assert checkpoints.resolve_run_head() is None
    assert not (checkpoints.model_root / "channels" / "champion.json").exists()


def test_commit_materializes_evidence_but_never_advances_run_head(
    tmp_path, staged_blobs
):
    checkpoints, evaluations = repositories(tmp_path)
    candidate = stage_checkpoint(checkpoints, staged_blobs, 1)
    runner = make_runner(tmp_path, checkpoints, evaluations)
    runner._run_eval = lambda **_: outcome(candidate_wins=1, opponent_wins=0, draws=1)
    prepared = runner.prepare(1, candidate, None)

    reference = runner.commit(prepared)

    assert reference.evaluation_id == prepared.evaluation_id
    assert runner.commit(prepared) == reference
    assert checkpoints.resolve_run_head() is None


def test_backend_failure_leaves_no_evidence(tmp_path, staged_blobs):
    checkpoints, evaluations = repositories(tmp_path)
    candidate = stage_checkpoint(checkpoints, staged_blobs, 1)
    runner = make_runner(tmp_path, checkpoints, evaluations)

    def fail(**_):
        raise RuntimeError("backend failed")

    runner._run_eval = fail
    with pytest.raises(RuntimeError, match="backend failed"):
        runner.prepare(1, candidate, None)

    assert not (checkpoints.model_root / "evaluations").exists()


def test_candidate_lineage_is_preflighted_before_expensive_games(
    tmp_path, staged_blobs
):
    checkpoints, evaluations = repositories(tmp_path)
    first = stage_checkpoint(checkpoints, staged_blobs, 1)
    runner = make_runner(tmp_path, checkpoints, evaluations)
    parent, _ = establish_first_run(checkpoints, evaluations, runner, first)
    invalid_root = stage_checkpoint(checkpoints, staged_blobs, 2, parent=None)
    runner._run_eval = lambda **_: pytest.fail("games must not run")

    with pytest.raises(ArtifactValidationError, match="direct child"):
        runner.prepare(2, invalid_root, parent)


def test_second_candidate_uses_runcommit_champion_and_rejection_is_evidence_only(
    tmp_path, staged_blobs
):
    checkpoints, evaluations = repositories(tmp_path)
    first = stage_checkpoint(checkpoints, staged_blobs, 1)
    runner = make_runner(tmp_path, checkpoints, evaluations)
    parent, first_prepared = establish_first_run(
        checkpoints, evaluations, runner, first
    )
    second = stage_checkpoint(checkpoints, staged_blobs, 2, parent=first.checkpoint_id)
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        return outcome(candidate_wins=0, opponent_wins=1, draws=1)

    runner._run_eval = evaluate
    prepared = runner.prepare(2, second, parent)

    assert len(calls) == 2
    assert calls[0]["player1"].model_path == str(second.onnx_path)
    assert calls[0]["player2"].model_path == str(first.onnx_path)
    assert prepared.artifact.champion_before == ChampionReferenceV1(
        first.checkpoint_id, first_prepared.evaluation_id
    )
    assert prepared.artifact.decision.promoted is False
    original_head = checkpoints.resolve_run_head()
    runner.commit(prepared)
    assert checkpoints.resolve_run_head() == original_head


def test_configuration_requires_first_candidate_evidence(tmp_path):
    checkpoints, evaluations = repositories(tmp_path)
    with pytest.raises(ValueError, match="first-candidate evidence"):
        make_runner(
            tmp_path,
            checkpoints,
            evaluations,
            eval_vs_random=False,
            solver_games=0,
        )


def test_connect4_first_candidate_uses_solver_as_sole_evidence(tmp_path):
    config = LoopConfig(
        data_dir=tmp_path / "runtime",
        env_id="connect4",
        iterations=3,
        episodes_per_iteration=1,
        steps_per_iteration=1,
        batch_size=1,
        mcts_max_sims=50,
        mcts_sim_ramp_rate=0,
        eval_games=2,
        eval_vs_random=False,
        solver_games=1,
    )
    checkpoints = EmptyRunHeadRepository(CONNECT4_CONTRACT)
    runner = EvalRunner(
        config,
        checkpoint_repository=checkpoints,
        evaluation_repository=object(),
    )
    runner._policy_loader = lambda path, temperature: StubPolicy(path, temperature)
    runner._run_eval = lambda **_: pytest.fail("head-to-head games must not run")
    solver_result = one_game_solver_result()
    runner._run_solver_eval = lambda _: solver_result

    prepared = runner.prepare(
        1,
        synthetic_checkpoint(tmp_path, CONNECT4_CONTRACT),
        None,
    )

    assert prepared.artifact.recipe.requested_games == RequestedGamesV1(
        vs_champion=0,
        vs_random=0,
        candidate_solver=1,
        champion_solver=0,
    )
    assert prepared.artifact.results.candidate_solver == solver_result
    assert prepared.artifact.results.vs_random is None
    assert prepared.artifact.results.vs_champion is None
    assert prepared.artifact.decision.promoted is True
