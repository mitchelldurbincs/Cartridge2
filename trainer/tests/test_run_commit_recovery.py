"""RunCommit authority, poison prevention, and crash recovery tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest
import torch
from torch.optim import Adam

from trainer.algorithms import get_algorithm
from trainer.algorithms.alphazero_board_v1 import policy_value_artifact_contract
from trainer.checkpoint import (
    LearnerStateContract,
    export_onnx_artifact,
    write_learner_state_artifact,
)
from trainer.network import PolicyValueNetwork
from trainer.orchestrator import eval_runner as eval_runner_module
from trainer.orchestrator import orchestrator as orchestrator_module
from trainer.orchestrator.config import LoopConfig
from trainer.orchestrator.eval_runner import PreparedEvaluation
from trainer.orchestrator.orchestrator import Orchestrator, _run_recipe
from trainer.orchestrator.run_journal import PreparedRunV1, RunJournal
from trainer.stats import (
    EvaluationStats,
    PreparedStatsSnapshotV3,
    StatsBindingV1,
    TrainerStats,
    decode_stats_snapshot,
    prepare_stats_snapshot,
)
from trainer.storage.evaluation import (
    ChampionReferenceV1,
    EvaluationArtifactV2,
    EvaluationRecipeV1,
    FilesystemEvaluationRepository,
    HeadToHeadResultV1,
    ObservedResultsV1,
    PromotionDecisionV1,
    RequestedGamesV1,
)
from trainer.storage.publisher import (
    ArtifactValidationError,
    FilesystemCheckpointPublisher,
)
from trainer.storage.run_commit import (
    LearnerRecipeV1,
    OrchestrationCommitV1,
    RunCommitRepository,
    RunCommitV1,
)

CONTRACT = policy_value_artifact_contract(
    algorithm_id="alphazero_board_v1",
    env_id="tictactoe",
    env_contract_version=2,
    model_artifact_schema_version=1,
    model_contract="onnx_policy_value_v1",
    obs_size=18,
    num_actions=9,
)


def loop_config(data_dir, **overrides) -> LoopConfig:
    values = {
        "data_dir": data_dir,
        "env_id": "tictactoe",
        "iterations": 4,
        "episodes_per_iteration": 1,
        "steps_per_iteration": 1,
        "batch_size": 2,
        "mcts_start_sims": 1,
        "mcts_max_sims": 4,
        "mcts_sim_ramp_rate": 1,
        "eval_interval": 1,
        "eval_games": 2,
        "solver_games": 0,
    }
    values.update(overrides)
    return LoopConfig(**values)


def recipe_for(config: LoopConfig):
    return _run_recipe(config, get_algorithm(config.algorithm_id))


class FakeReplayStore:
    def clear(self):
        return 0

    def vacuum(self):
        return None

    def count(self):
        return 0

    def close(self):
        return None


@pytest.fixture(scope="module")
def staged_blobs(tmp_path_factory):
    root = tmp_path_factory.mktemp("run-commit-recovery-v2")
    network = PolicyValueNetwork(obs_size=18, action_size=9)
    optimizer = Adam(network.parameters(), lr=0.001)
    return (
        export_onnx_artifact(
            network,
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
            "0" * 64,
        ),
        LearnerStateContract(network, optimizer),
    )


def repositories(config: LoopConfig):
    checkpoints = FilesystemCheckpointPublisher(
        model_root=config.models_dir,
        contract=CONTRACT,
    )
    evaluations = FilesystemEvaluationRepository(checkpoints)
    return checkpoints, evaluations, RunCommitRepository(checkpoints, evaluations)


def stage_checkpoint(
    repository,
    staged_blobs,
    *,
    step: int,
    config_sha256: str,
    parent: str | None = None,
):
    onnx_path, source_learner, learner_contract = staged_blobs
    repository.model_root.mkdir(parents=True, exist_ok=True)
    learner_path = repository.model_root / f".learner-{step}-{parent or 'root'}.pt"
    state = torch.load(source_learner, map_location="cpu", weights_only=True)
    state["step"] = step
    state["config_sha256"] = config_sha256
    torch.save(state, learner_path)
    return repository.stage_checkpoint(
        onnx_path,
        learner_path,
        step=step,
        parent_checkpoint_id=parent,
        config_sha256=config_sha256,
        learner_state_contract=learner_contract,
    )


def head_to_head(*, promoted: bool) -> HeadToHeadResultV1:
    return HeadToHeadResultV1(
        games_played=2,
        candidate_wins=2 if promoted else 0,
        opponent_wins=0 if promoted else 1,
        draws=0 if promoted else 1,
        candidate_wins_as_first=1 if promoted else 0,
        candidate_wins_as_second=1 if promoted else 0,
        opponent_wins_while_candidate_first=0,
        opponent_wins_while_candidate_second=0 if promoted else 1,
        average_game_length=5.0,
    )


def evaluation_for(
    config: LoopConfig,
    candidate,
    parent: RunCommitV1 | None,
    *,
    iteration: int,
    promoted: bool,
) -> PreparedEvaluation:
    run_recipe = recipe_for(config)
    result = head_to_head(promoted=promoted)
    champion = parent.champion if parent is not None else None
    started_second = iteration * 4
    artifact = EvaluationArtifactV2(
        profile=CONTRACT.profile,
        iteration=iteration,
        candidate_checkpoint_id=candidate.checkpoint_id,
        previous_evaluation_id=(parent.evaluation_head_id if parent is not None else None),
        champion_before=champion,
        recipe=EvaluationRecipeV1(
            simulations=run_recipe.evaluation_simulations,
            temperature=run_recipe.evaluation_temperature,
            promotion_metric=run_recipe.promotion_metric,
            promotion_margin=run_recipe.promotion_margin,
            win_threshold=run_recipe.evaluation_win_threshold,
            seed=run_recipe.evaluation_seed,
            requested_games=RequestedGamesV1(
                run_recipe.evaluation_games if champion is not None else 0,
                run_recipe.evaluation_games,
                0,
                0,
            ),
        ),
        results=ObservedResultsV1(
            result if champion is not None else None,
            result,
            None,
            None,
        ),
        decision=PromotionDecisionV1(promoted, "test decision"),
        started_at=f"2026-08-02T10:00:{started_second:02d}.000000Z",
        completed_at=f"2026-08-02T10:00:{started_second + 1:02d}.000000Z",
    )
    return PreparedEvaluation(
        artifact=artifact,
        evaluation_id=artifact.evaluation_id,
        win_rate=result.candidate_win_rate if champion is not None else None,
        draw_rate=result.draw_rate if champion is not None else None,
        elapsed_seconds=0.25,
    )


def commit_for(
    config: LoopConfig,
    candidate,
    parent: RunCommitV1 | None,
    *,
    iteration: int,
    evaluation: PreparedEvaluation | None,
    run_recipe_override=None,
) -> RunCommitV1:
    run_recipe = run_recipe_override or recipe_for(config)
    if parent is None:
        stats = TrainerStats(env_id="tictactoe")
        inherited_champion = None
        inherited_evaluation = None
    else:
        previous = decode_stats_snapshot(
            parent.stats_snapshot.data,
            expected_stats_id=parent.stats_id,
            expected_binding=parent.stats_snapshot.binding,
        ).stats
        stats = TrainerStats.from_dict(previous.to_dict())
        inherited_champion = parent.champion
        inherited_evaluation = parent.evaluation_head_id
    stats.step = candidate.manifest.step
    stats.total_steps = candidate.manifest.step
    stats.samples_seen += config.steps_per_iteration * config.batch_size
    stats.last_checkpoint = candidate.checkpoint_id
    stats.timestamp = float(candidate.manifest.step)

    champion = inherited_champion
    evaluation_head = inherited_evaluation
    if evaluation is not None:
        evaluation_head = evaluation.evaluation_id
        if evaluation.artifact.decision.promoted:
            champion = ChampionReferenceV1(
                candidate.checkpoint_id,
                evaluation.evaluation_id,
            )
        random_result = evaluation.artifact.results.vs_random
        if random_result is not None:
            completed = datetime.fromisoformat(
                evaluation.artifact.completed_at.replace("Z", "+00:00")
            )
            stats.append_evaluation(
                EvaluationStats(
                    step=candidate.manifest.step,
                    metrics={
                        "outcome/win_rate": random_result.candidate_win_rate,
                        "outcome/draw_rate": random_result.draw_rate,
                        "outcome/loss_rate": (
                            1.0 - random_result.candidate_win_rate - random_result.draw_rate
                        ),
                    },
                    episodes=random_result.games_played,
                    mean_episode_length=random_result.average_game_length,
                    timestamp=completed.timestamp(),
                )
            )
    result = evaluation.artifact.results.vs_champion if evaluation is not None else None
    timestamp_second = iteration * 4 + 2
    orchestration = OrchestrationCommitV1(
        iteration=iteration,
        episodes_generated=config.episodes_per_iteration,
        transitions_generated=2,
        training_steps=config.steps_per_iteration,
        actor_time_seconds=0.1,
        trainer_time_seconds=0.2,
        eval_time_seconds=(evaluation.elapsed_seconds if evaluation is not None else 0.0),
        total_time_seconds=1.0,
        eval_win_rate=result.candidate_win_rate if result is not None else None,
        eval_draw_rate=result.draw_rate if result is not None else None,
        timestamp=f"2026-08-02T10:00:{timestamp_second:02d}.000000Z",
        evaluation_id=(evaluation.evaluation_id if evaluation is not None else None),
        collector_simulations=run_recipe.simulations_for(iteration),
        collector_seed=None,
        evaluation_seed=(run_recipe.evaluation_seed if evaluation is not None else None),
        collection_scope_id=f"{iteration + 1000:064x}",
        source_checkpoint_id=(parent.checkpoint_id if parent is not None else None),
    )
    return RunCommitV1(
        profile=CONTRACT.profile,
        config_sha256=run_recipe.learner_config_sha256,
        parent_run_commit_id=(parent.run_commit_id if parent is not None else None),
        checkpoint_id=candidate.checkpoint_id,
        stats_snapshot=prepare_stats_snapshot(stats, candidate),
        champion=champion,
        evaluation_head_id=evaluation_head,
        orchestration=orchestration,
        run_recipe=run_recipe,
    )


def build_root(config: LoopConfig, staged_blobs, *, with_evaluation: bool = True):
    checkpoints, evaluations, commits = repositories(config)
    run_recipe = recipe_for(config)
    candidate = stage_checkpoint(
        checkpoints,
        staged_blobs,
        step=1,
        config_sha256=run_recipe.learner_config_sha256,
    )
    evaluation = (
        evaluation_for(config, candidate, None, iteration=1, promoted=True)
        if with_evaluation
        else None
    )
    commit = commit_for(
        config,
        candidate,
        None,
        iteration=1,
        evaluation=evaluation,
    )
    return (
        checkpoints,
        evaluations,
        commits,
        candidate,
        PreparedRunV1(commit, evaluation),
    )


@pytest.mark.parametrize("boundary", ["prepared", "evaluation", "commit", "head"])
def test_startup_recovers_every_commit_boundary_without_rerunning_games(
    tmp_path, staged_blobs, monkeypatch, boundary
):
    config = loop_config(tmp_path)
    checkpoints, evaluations, commits, _, prepared = build_root(config, staged_blobs)
    journal = RunJournal(checkpoints, commits)
    journal.publish(prepared)
    if boundary in {"evaluation", "commit", "head"}:
        evaluations.publish_evidence(prepared.evaluation.artifact)
    if boundary in {"commit", "head"}:
        commits.publish(prepared.run_commit)
    if boundary == "head":
        checkpoints.commit_run_head(
            checkpoint_id=prepared.run_commit.checkpoint_id,
            run_commit_id=prepared.run_commit_id,
            expected_run_commit_id=None,
        )

    monkeypatch.setattr(
        orchestrator_module,
        "create_replay_store",
        lambda profile: FakeReplayStore(),
    )
    monkeypatch.setattr(
        eval_runner_module,
        "run_eval",
        lambda **kwargs: pytest.fail("recovery must not rerun games"),
    )

    recovered = Orchestrator(config)

    head = recovered.eval_runner.checkpoints.resolve_run_head()
    assert head is not None and head.run_commit_id == prepared.run_commit_id
    assert [item.iteration for item in recovered.iteration_history] == [1]
    assert config.start_iteration == 1
    assert recovered.config.start_iteration == 2
    assert config.loop_stats_path.exists()
    assert config.eval_stats_path.exists()
    assert config.stats_path.exists()


def test_orchestration_numbers_normalize_and_enforce_u64():
    normalized = OrchestrationCommitV1(
        iteration=1,
        episodes_generated=1,
        transitions_generated=0,
        training_steps=1,
        actor_time_seconds=-0.0,
        trainer_time_seconds=0,
        eval_time_seconds=0,
        total_time_seconds=-0.0,
        eval_win_rate=None,
        eval_draw_rate=None,
        timestamp="2026-08-02T10:00:00.000000Z",
        evaluation_id=None,
        collector_simulations=1,
        collector_seed=None,
        evaluation_seed=None,
        collection_scope_id="a" * 64,
        source_checkpoint_id=None,
    )
    assert normalized.actor_time_seconds == 0.0
    assert isinstance(normalized.trainer_time_seconds, float)
    with pytest.raises(ArtifactValidationError, match="nonnegative integer"):
        replace(normalized, transitions_generated=2**64)


@pytest.mark.parametrize(
    "override",
    [
        {"episodes_per_iteration": 2**32},
        {"num_actors": 2**32},
    ],
)
def test_collector_quota_fields_fit_the_actor_u32_wire(tmp_path, override):
    with pytest.raises(ValueError, match="u32"):
        loop_config(tmp_path, **override)


def test_run_recipe_rejects_inactive_promotion_parameters(tmp_path):
    recipe = recipe_for(loop_config(tmp_path))

    with pytest.raises(ArtifactValidationError, match="promotion_margin"):
        replace(recipe, promotion_margin=0.1)
    with pytest.raises(ArtifactValidationError, match="evaluation_win_threshold"):
        replace(
            recipe,
            promotion_metric="solver_optimal",
            solver_games=1,
        )


def test_run_recipe_rejects_noncanonical_mcts_schedules(tmp_path):
    recipe = recipe_for(loop_config(tmp_path))

    with pytest.raises(ArtifactValidationError, match="does not reach"):
        replace(recipe, total_iterations=3)
    with pytest.raises(ArtifactValidationError, match="zero ramp"):
        replace(
            recipe,
            mcts_max_simulations=recipe.mcts_start_simulations,
        )


def test_run_commit_rejects_unreachable_temperature_schedule(tmp_path, staged_blobs):
    config = loop_config(tmp_path)
    _, _, commits, _, prepared = build_root(config, staged_blobs)
    assert prepared.run_commit.run_recipe is not None
    invalid_recipe = replace(
        prepared.run_commit.run_recipe,
        collector_late_temperature=0.1,
        temperature_move_threshold=9,
    )
    invalid_commit = replace(prepared.run_commit, run_recipe=invalid_recipe)

    with pytest.raises(ArtifactValidationError, match="unreachable"):
        commits.publish(invalid_commit)


def test_missing_evaluation_fails_before_runcommit_or_head_materialization(tmp_path, staged_blobs):
    config = loop_config(tmp_path)
    checkpoints, _, commits, _, prepared = build_root(config, staged_blobs)

    with pytest.raises(ArtifactValidationError, match="does not exist"):
        commits.publish(prepared.run_commit)

    assert checkpoints.resolve_run_head() is None
    assert not (config.models_dir / "run-commits" / "sha256").exists()


def test_invalid_preparation_cannot_poison_immutable_parent_journal(tmp_path, staged_blobs):
    config = loop_config(tmp_path)
    checkpoints, _, commits, _, prepared = build_root(config, staged_blobs)
    invalid = PreparedRunV1(
        replace(
            prepared.run_commit,
            orchestration=replace(prepared.run_commit.orchestration, iteration=2),
        ),
        prepared.evaluation,
    )

    with pytest.raises(ArtifactValidationError):
        RunJournal(checkpoints, commits).publish(invalid)

    assert not (config.models_dir / "run-preparations" / "by-parent").exists()


def test_mismatched_snapshot_wrapper_is_rejected_before_journal_write(tmp_path, staged_blobs):
    config = loop_config(tmp_path)
    checkpoints, _, commits, _, prepared = build_root(config, staged_blobs)
    snapshot = prepared.run_commit.stats_snapshot
    bad_binding = StatsBindingV1(
        profile=snapshot.binding.profile,
        config_sha256=snapshot.binding.config_sha256,
        checkpoint_id=snapshot.binding.checkpoint_id,
        step=snapshot.binding.step + 1,
    )
    bad_snapshot = PreparedStatsSnapshotV3(
        stats_id=snapshot.stats_id,
        binding=bad_binding,
        data=snapshot.data,
    )
    invalid_commit = replace(
        prepared.run_commit,
        stats_snapshot=bad_snapshot,
    )

    with pytest.raises(Exception, match="snapshot"):
        RunJournal(checkpoints, commits).publish(PreparedRunV1(invalid_commit, prepared.evaluation))

    assert not (config.models_dir / "run-preparations" / "by-parent").exists()


def test_model_contract_mismatch_is_rejected_before_journal_write(tmp_path, staged_blobs):
    config = loop_config(tmp_path, eval_interval=0)
    checkpoints, _, commits = repositories(config)
    original = recipe_for(config)
    learner_data = original.learner_recipe.to_dict()
    model = dict(learner_data["model_architecture"])
    model["action_count"] = CONTRACT.output("policy_logits").shape[1] + 1
    learner_data["model_architecture"] = model
    learner = LearnerRecipeV1(learner_data)
    recipe = replace(
        original,
        learner_recipe=learner,
        learner_config_sha256=learner.config_sha256,
    )
    candidate = stage_checkpoint(
        checkpoints,
        staged_blobs,
        step=1,
        config_sha256=recipe.learner_config_sha256,
    )
    commit = commit_for(
        config,
        candidate,
        None,
        iteration=1,
        evaluation=None,
        run_recipe_override=recipe,
    )

    with pytest.raises(ArtifactValidationError, match="model dimensions"):
        RunJournal(checkpoints, commits).publish(PreparedRunV1(commit, None))

    assert not (config.models_dir / "run-preparations" / "by-parent").exists()


def test_stats_regression_is_rejected_before_materialization(tmp_path, staged_blobs):
    config = loop_config(tmp_path, eval_interval=0)
    checkpoints, evaluations, commits, root_candidate, prepared = build_root(
        config, staged_blobs, with_evaluation=False
    )
    root = commits.publish(prepared.run_commit).commit
    checkpoints.commit_run_head(
        checkpoint_id=root.checkpoint_id,
        run_commit_id=root.run_commit_id,
        expected_run_commit_id=None,
    )
    child = stage_checkpoint(
        checkpoints,
        staged_blobs,
        step=2,
        parent=root_candidate.checkpoint_id,
        config_sha256=root.config_sha256,
    )
    valid_child = commit_for(config, child, root, iteration=2, evaluation=None)
    loaded = decode_stats_snapshot(
        valid_child.stats_snapshot.data,
        expected_stats_id=valid_child.stats_id,
        expected_binding=valid_child.stats_snapshot.binding,
    ).stats
    loaded.samples_seen = 0
    loaded.timestamp = 0.0
    regressed = replace(
        valid_child,
        stats_snapshot=prepare_stats_snapshot(loaded, child),
    )

    with pytest.raises(ArtifactValidationError, match="samples_seen|timestamp"):
        commits.validate_prepared(regressed)

    assert len(list((config.models_dir / "run-commits" / "sha256").glob("*.json"))) == 1


@pytest.mark.parametrize("violation", ["source", "duplicate_scope", "episodes"])
def test_replay_provenance_violation_is_rejected_before_child_materialization(
    tmp_path, staged_blobs, violation
):
    config = loop_config(tmp_path, eval_interval=0)
    checkpoints, _, commits, _, prepared = build_root(config, staged_blobs, with_evaluation=False)
    parent = commits.publish(prepared.run_commit).commit
    candidate = stage_checkpoint(
        checkpoints,
        staged_blobs,
        step=2,
        parent=parent.checkpoint_id,
        config_sha256=parent.config_sha256,
    )
    child = commit_for(
        config,
        candidate,
        parent,
        iteration=2,
        evaluation=None,
    )
    assert child.orchestration is not None
    if violation == "source":
        orchestration = replace(child.orchestration, source_checkpoint_id="f" * 64)
        message = "replay source"
    elif violation == "duplicate_scope":
        assert parent.orchestration is not None
        orchestration = replace(
            child.orchestration,
            collection_scope_id=parent.orchestration.collection_scope_id,
        )
        message = "collection scopes must be unique"
    else:
        orchestration = replace(
            child.orchestration,
            episodes_generated=config.episodes_per_iteration + 1,
        )
        message = "immutable run recipe"
    invalid = replace(child, orchestration=orchestration)

    with pytest.raises(ArtifactValidationError, match=message):
        RunJournal(checkpoints, commits).publish(PreparedRunV1(invalid, None))

    assert not (config.models_dir / "run-preparations" / "by-parent").exists()
    assert not (
        config.models_dir / "run-commits" / "sha256" / f"{invalid.run_commit_id}.json"
    ).exists()


def test_run_modes_are_disjoint_and_same_checkpoint_children_are_rejected(tmp_path, staged_blobs):
    config = loop_config(tmp_path, eval_interval=0)
    checkpoints, evaluations, commits = repositories(config)
    run_recipe = recipe_for(config)
    standalone_checkpoint = stage_checkpoint(
        checkpoints,
        staged_blobs,
        step=1,
        config_sha256=run_recipe.learner_config_sha256,
    )
    standalone_stats = TrainerStats(
        step=1,
        total_steps=1,
        samples_seen=0,
        last_checkpoint=standalone_checkpoint.checkpoint_id,
        env_id="tictactoe",
        timestamp=1.0,
    )
    standalone = RunCommitV1(
        profile=CONTRACT.profile,
        config_sha256=run_recipe.learner_config_sha256,
        parent_run_commit_id=None,
        checkpoint_id=standalone_checkpoint.checkpoint_id,
        stats_snapshot=prepare_stats_snapshot(standalone_stats, standalone_checkpoint),
        champion=None,
        evaluation_head_id=None,
        orchestration=None,
        run_recipe=None,
    )
    commits.publish(standalone)
    child_checkpoint = stage_checkpoint(
        checkpoints,
        staged_blobs,
        step=2,
        parent=standalone.checkpoint_id,
        config_sha256=run_recipe.learner_config_sha256,
    )
    synchronized_child = commit_for(
        config,
        child_checkpoint,
        standalone,
        iteration=1,
        evaluation=None,
    )
    with pytest.raises(ArtifactValidationError, match="cannot be mixed"):
        commits.validate_prepared(synchronized_child)

    no_op = replace(
        standalone,
        parent_run_commit_id=standalone.run_commit_id,
    )
    with pytest.raises(ArtifactValidationError, match="new direct-child checkpoint"):
        commits.validate_prepared(no_op)


def test_changed_recipe_fails_before_replay_is_opened(tmp_path, staged_blobs, monkeypatch):
    config = loop_config(tmp_path, eval_interval=0)
    checkpoints, _, commits, _, prepared = build_root(config, staged_blobs, with_evaluation=False)
    commits.publish(prepared.run_commit)
    checkpoints.commit_run_head(
        checkpoint_id=prepared.run_commit.checkpoint_id,
        run_commit_id=prepared.run_commit_id,
        expected_run_commit_id=None,
    )
    changed = loop_config(tmp_path, eval_interval=0, learning_rate=0.02)
    opened = False

    def unexpected_replay(profile):
        nonlocal opened
        opened = True
        raise AssertionError("replay must not open")

    monkeypatch.setattr(orchestrator_module, "create_replay_store", unexpected_replay)
    with pytest.raises(ArtifactValidationError, match="learner config"):
        Orchestrator(changed)
    assert opened is False


def test_resolve_chain_memoizes_validated_prefixes_on_the_publisher(tmp_path, staged_blobs):
    config = loop_config(tmp_path / "lineage-cache")
    checkpoints, evaluations, commits, root_candidate, prepared = build_root(config, staged_blobs)
    evaluations.publish_evidence(prepared.evaluation.artifact)
    root = commits.publish(prepared.run_commit)

    child_candidate = stage_checkpoint(
        checkpoints,
        staged_blobs,
        step=2,
        config_sha256=recipe_for(config).learner_config_sha256,
        parent=root_candidate.checkpoint_id,
    )
    child_evaluation = evaluation_for(
        config, child_candidate, root.commit, iteration=2, promoted=False
    )
    evaluations.publish_evidence(child_evaluation.artifact)
    head = commits.publish(
        commit_for(config, child_candidate, root.commit, iteration=2, evaluation=child_evaluation)
    )

    reads: list[str] = []
    original_read = checkpoints.read_run_commit_bytes
    checkpoints.read_run_commit_bytes = lambda run_commit_id: (
        reads.append(run_commit_id) or original_read(run_commit_id)
    )

    # publish() already validated and cached the root chain, so resolving the
    # new head reads exactly one commit: the head itself.
    chain = commits.resolve_chain(head.run_commit_id)
    assert [ref.run_commit_id for ref in chain] == [root.run_commit_id, head.run_commit_id]
    assert reads == [head.run_commit_id]

    # A repeated resolution is a pure memo hit with zero storage reads, and a
    # fresh repository over the same publisher shares the cache.
    commits.resolve_chain(head.run_commit_id)
    FilesystemEvaluationRepository(checkpoints)
    fresh = RunCommitRepository(checkpoints, FilesystemEvaluationRepository(checkpoints))
    fresh.resolve_chain(head.run_commit_id)
    assert reads == [head.run_commit_id]

    # Settled history is not re-read: deleting the root's bytes does not
    # disturb this process, while a publisher with a cold cache fails closed.
    checkpoints._run_commit_path(root.run_commit_id).unlink()
    fresh.resolve_chain(head.run_commit_id)
    cold_publisher = FilesystemCheckpointPublisher(model_root=config.models_dir, contract=CONTRACT)
    cold = RunCommitRepository(cold_publisher, FilesystemEvaluationRepository(cold_publisher))
    with pytest.raises(ArtifactValidationError, match="does not exist"):
        cold.resolve_chain(head.run_commit_id)
