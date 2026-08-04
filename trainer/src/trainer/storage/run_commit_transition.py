"""RunCommit lineage and transition validation."""

from __future__ import annotations

from datetime import datetime

from ..stats import (
    DEFAULT_MAX_EVAL_HISTORY,
    EvaluationStats,
    decode_stats_snapshot,
    retain_training_history,
)
from .artifact_codec import ArtifactValidationError
from .checkpoint_types import CheckpointPublisher
from .evaluation_artifact import EvaluationArtifactV2
from .evaluation_recipe import ChampionReferenceV1
from .evaluation_repository import EvaluationRepository
from .run_commit_codec import _MAX_U64
from .run_commit_types import RunCommitV1
from .run_recipe import RunRecipeV1


def _loaded_stats(commit: RunCommitV1):
    return decode_stats_snapshot(
        commit.stats_snapshot.data,
        expected_stats_id=commit.stats_id,
        expected_binding=commit.stats_snapshot.binding,
    ).stats


def _validate_stats_continuity(
    commit: RunCommitV1,
    parent: RunCommitV1 | None,
) -> tuple[object, object | None]:
    current = _loaded_stats(commit)
    if parent is None:
        return current, None
    previous = _loaded_stats(parent)
    if commit.checkpoint_id == parent.checkpoint_id:
        if commit.stats_id != parent.stats_id:
            raise ArtifactValidationError(
                "A same-checkpoint RunCommit must carry the exact stats snapshot"
            )
        return current, previous
    if current.samples_seen < previous.samples_seen:
        raise ArtifactValidationError("RunCommit stats samples_seen regressed")
    if current.total_steps < previous.total_steps:
        raise ArtifactValidationError("RunCommit stats total_steps regressed")
    if current.timestamp < previous.timestamp:
        raise ArtifactValidationError("RunCommit stats timestamp regressed")

    new_history = [entry for entry in current.history if entry["step"] > previous.step]
    expected_history = retain_training_history(
        [*previous.history, *new_history],
        current.step,
    )
    if current.history != expected_history:
        raise ArtifactValidationError(
            "RunCommit stats training history is not the exact retained continuation"
        )
    return current, previous


def _validate_eval_stats_continuity(current, previous, artifact) -> None:
    inherited = previous.evaluation_history if previous is not None else []
    result = artifact.results.vs_random if artifact is not None else None
    if result is None:
        if current.evaluation_history != inherited:
            raise ArtifactValidationError(
                "RunCommit stats evaluation history changed without random evidence"
            )
        return
    completed = datetime.fromisoformat(artifact.completed_at.replace("Z", "+00:00"))
    appended = EvaluationStats(
        step=current.step,
        metrics={
            "outcome/win_rate": result.candidate_win_rate,
            "outcome/draw_rate": result.draw_rate,
            "outcome/loss_rate": 1.0 - result.candidate_win_rate - result.draw_rate,
        },
        episodes=result.games_played,
        mean_episode_length=result.average_game_length,
        timestamp=completed.timestamp(),
    ).to_dict()
    expected = [*inherited, appended][-DEFAULT_MAX_EVAL_HISTORY:]
    if current.evaluation_history != expected or current.last_evaluation is None:
        raise ArtifactValidationError(
            "RunCommit stats do not contain the exact random-evaluation projection"
        )


def _validate_evaluation_recipe(
    artifact: EvaluationArtifactV2,
    recipe: RunRecipeV1,
) -> None:
    requested = artifact.recipe.requested_games
    has_champion = artifact.champion_before is not None
    solver_enabled = artifact.profile.env_id == "connect4" and recipe.solver_games > 0
    expected_requested = (
        recipe.evaluation_games if has_champion else 0,
        recipe.evaluation_games if recipe.evaluation_vs_random else 0,
        recipe.solver_games if solver_enabled else 0,
        recipe.solver_games if solver_enabled and has_champion else 0,
    )
    actual_requested = (
        requested.vs_champion,
        requested.vs_random,
        requested.candidate_solver,
        requested.champion_solver,
    )
    if (
        artifact.recipe.simulations != recipe.evaluation_simulations
        or artifact.recipe.temperature != recipe.evaluation_temperature
        or artifact.recipe.promotion_metric != recipe.promotion_metric
        or artifact.recipe.promotion_margin != recipe.promotion_margin
        or artifact.recipe.win_threshold != recipe.evaluation_win_threshold
        or artifact.recipe.seed != recipe.evaluation_seed
        or actual_requested != expected_requested
    ):
        raise ArtifactValidationError("Evaluation evidence disagrees with the immutable run recipe")


def _validate_mode(commit: RunCommitV1, parent: RunCommitV1 | None) -> None:
    if not isinstance(commit, RunCommitV1):
        raise TypeError("commit must be RunCommitV1")
    if parent is not None and not isinstance(parent, RunCommitV1):
        raise TypeError("parent must be RunCommitV1 or None")
    expected_parent_id = parent.run_commit_id if parent is not None else None
    if commit.parent_run_commit_id != expected_parent_id:
        raise ArtifactValidationError("RunCommit parent identity is inconsistent")
    if parent is None:
        if commit.orchestration is None and commit.run_recipe is not None:
            raise ArtifactValidationError(
                "A standalone prefix cannot introduce a synchronized run recipe"
            )
    elif parent.run_recipe is None:
        if commit.orchestration is not None or commit.run_recipe is not None:
            raise ArtifactValidationError("Standalone and synchronized run modes cannot be mixed")
    else:
        if commit.run_recipe != parent.run_recipe:
            raise ArtifactValidationError("RunCommit changed the immutable run recipe")
        if commit.orchestration is None:
            raise ArtifactValidationError("A recipe-owned run cannot contain standalone commits")


def _validate_checkpoint_edge(
    commit: RunCommitV1,
    parent: RunCommitV1 | None,
    checkpoints: CheckpointPublisher,
):
    checkpoint = checkpoints.read_checkpoint_manifest_exact(commit.checkpoint_id)
    if (
        checkpoint.profile != commit.profile
        or checkpoint.config_sha256 != commit.config_sha256
        or checkpoint.step != commit.stats_snapshot.binding.step
    ):
        raise ArtifactValidationError(
            "RunCommit checkpoint does not match its profile/config/stats binding"
        )
    current_stats, previous_stats = _validate_stats_continuity(commit, parent)
    if parent is None:
        if checkpoint.parent_checkpoint_id is not None:
            raise ArtifactValidationError(
                "The first RunCommit checkpoint must start a checkpoint lineage"
            )
    else:
        if commit.profile != parent.profile or commit.config_sha256 != parent.config_sha256:
            raise ArtifactValidationError("RunCommit profile/config changed within a run")
        parent_checkpoint = checkpoints.read_checkpoint_manifest_exact(parent.checkpoint_id)
        if commit.checkpoint_id == parent.checkpoint_id:
            raise ArtifactValidationError("A RunCommit must select a new direct-child checkpoint")
        if (
            checkpoint.parent_checkpoint_id != parent.checkpoint_id
            or checkpoint.step <= parent_checkpoint.step
        ):
            raise ArtifactValidationError(
                "RunCommit checkpoint must be a strictly newer direct child"
            )
    return checkpoint, current_stats, previous_stats


def _validate_recipe_contract(
    commit: RunCommitV1,
    checkpoints: CheckpointPublisher,
) -> RunRecipeV1:
    recipe = commit.run_recipe
    if recipe is None:
        raise ArtifactValidationError("An orchestration RunCommit requires an immutable run recipe")
    if commit.profile.env_id != "connect4" and (
        recipe.solver_games > 0 or recipe.promotion_metric == "solver_optimal"
    ):
        raise ArtifactValidationError(
            "RunCommit solver evaluation settings require the connect4 profile"
        )
    from ..environment_catalog import get_environment

    max_horizon = get_environment(commit.profile.env_id).capabilities.max_horizon
    if max_horizon is None:
        raise ArtifactValidationError(
            "Synchronized RunCommit environment must declare a finite max_horizon"
        )
    if recipe.temperature_move_threshold != 0 and recipe.temperature_move_threshold >= max_horizon:
        raise ArtifactValidationError(
            "RunCommit temperature threshold is unreachable for its environment"
        )
    model_architecture = recipe.learner_recipe.to_dict()["model_architecture"]
    if not isinstance(model_architecture, dict):
        raise ArtifactValidationError("RunCommit learner model architecture is invalid")
    if checkpoints.contract.input("observation").shape != (
        "batch_size",
        model_architecture["observation_elements"],
    ) or checkpoints.contract.output("policy_logits").shape != (
        "batch_size",
        model_architecture["action_count"],
    ):
        raise ArtifactValidationError(
            "RunCommit learner model dimensions disagree with the checkpoint contract"
        )
    return recipe


def _validate_orchestration_facts(
    commit,
    parent,
    checkpoint,
    checkpoints,
    current_stats,
    previous_stats,
    recipe,
) -> None:
    orchestration = commit.orchestration
    assert orchestration is not None
    expected_source = parent.checkpoint_id if parent is not None else None
    if orchestration.source_checkpoint_id != expected_source:
        raise ArtifactValidationError(
            "Orchestration replay source does not match its parent checkpoint"
        )
    if orchestration.iteration > recipe.total_iterations:
        raise ArtifactValidationError("Orchestration iteration exceeds the run's total target")
    if (
        orchestration.episodes_generated != recipe.episodes_per_iteration
        or orchestration.training_steps != recipe.training_steps_per_iteration
        or orchestration.collector_simulations != recipe.simulations_for(orchestration.iteration)
        or orchestration.collector_seed is not None
    ):
        raise ArtifactValidationError("Orchestration facts disagree with the immutable run recipe")
    batch_size = recipe.learner_recipe.to_dict().get("batch_size")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or batch_size > _MAX_U64
    ):
        raise ArtifactValidationError(
            "run_recipe learner batch_size must be a positive u64 integer"
        )
    if orchestration.training_steps > _MAX_U64 // batch_size:
        raise ArtifactValidationError("orchestration training_steps times batch_size exceeds u64")
    previous_samples = previous_stats.samples_seen if previous_stats is not None else 0
    if current_stats.samples_seen - previous_samples != orchestration.training_steps * batch_size:
        raise ArtifactValidationError(
            "RunCommit samples_seen delta disagrees with training_steps times batch_size"
        )
    if current_stats.total_steps != current_stats.step:
        raise ArtifactValidationError(
            "Orchestration stats total_steps must equal the selected checkpoint step"
        )
    if recipe.evaluation_scheduled(orchestration.iteration) != (
        orchestration.evaluation_id is not None
    ):
        raise ArtifactValidationError(
            "Orchestration evaluation presence disagrees with the run schedule"
        )
    parent_step = (
        checkpoints.read_checkpoint_manifest_exact(parent.checkpoint_id).step
        if parent is not None
        else 0
    )
    if orchestration.training_steps == 0:
        raise ArtifactValidationError(
            "An orchestration RunCommit must record positive training_steps"
        )
    if checkpoint.step - parent_step != orchestration.training_steps:
        raise ArtifactValidationError(
            "RunCommit checkpoint step delta must equal orchestration training_steps"
        )


def _resolve_evaluation(
    orchestration,
    evaluations: EvaluationRepository,
    prepared_evaluation: EvaluationArtifactV2 | None,
):
    if prepared_evaluation is not None:
        if prepared_evaluation.evaluation_id != orchestration.evaluation_id:
            raise ArtifactValidationError("Prepared evaluation ID disagrees with its RunCommit")
        evaluations.validate_evidence(prepared_evaluation)
        return prepared_evaluation, prepared_evaluation.evaluation_id
    evaluation = evaluations.resolve_evaluation(orchestration.evaluation_id)
    return evaluation.artifact, evaluation.evaluation_id


def _validate_evaluation_binding(
    commit,
    inherited_champion,
    inherited_evaluation,
    artifact,
    evaluation_id,
) -> None:
    orchestration = commit.orchestration
    assert orchestration is not None
    if (
        artifact.profile != commit.profile
        or artifact.iteration != orchestration.iteration
        or artifact.candidate_checkpoint_id != commit.checkpoint_id
        or artifact.previous_evaluation_id != inherited_evaluation
        or artifact.champion_before != inherited_champion
        or commit.evaluation_head_id != evaluation_id
        or orchestration.evaluation_seed != artifact.recipe.seed
    ):
        raise ArtifactValidationError(
            "RunCommit evaluation does not match its authoritative transition"
        )
    if artifact.completed_at > orchestration.timestamp:
        raise ArtifactValidationError("RunCommit timestamp precedes evaluation completion")
    observed_win = (
        artifact.results.vs_champion.candidate_win_rate
        if artifact.results.vs_champion is not None
        else None
    )
    observed_draw = (
        artifact.results.vs_champion.draw_rate if artifact.results.vs_champion is not None else None
    )
    if orchestration.eval_win_rate != observed_win or orchestration.eval_draw_rate != observed_draw:
        raise ArtifactValidationError("RunCommit evaluation rates disagree with immutable evidence")
    expected_champion = (
        ChampionReferenceV1(
            checkpoint_id=commit.checkpoint_id,
            evaluation_id=evaluation_id,
        )
        if artifact.decision.promoted
        else inherited_champion
    )
    if commit.champion != expected_champion:
        raise ArtifactValidationError("RunCommit champion disagrees with the evaluation decision")


def _validate_inherited_evaluation(
    commit,
    inherited_champion,
    inherited_evaluation,
    current_stats,
    previous_stats,
    *,
    standalone: bool,
) -> None:
    if commit.champion != inherited_champion or commit.evaluation_head_id != inherited_evaluation:
        message = (
            "A standalone RunCommit must carry evaluation state unchanged"
            if standalone
            else "A non-evaluation iteration must carry evaluation state unchanged"
        )
        raise ArtifactValidationError(message)
    _validate_eval_stats_continuity(current_stats, previous_stats, None)


def validate_transition(
    commit: RunCommitV1,
    parent: RunCommitV1 | None,
    *,
    checkpoints: CheckpointPublisher,
    evaluations: EvaluationRepository,
    prepared_evaluation: EvaluationArtifactV2 | None = None,
) -> None:
    """Validate one full RunCommit edge and every referenced immutable object."""
    _validate_mode(commit, parent)
    checkpoint, current_stats, previous_stats = _validate_checkpoint_edge(
        commit, parent, checkpoints
    )
    inherited_champion = parent.champion if parent is not None else None
    inherited_evaluation = parent.evaluation_head_id if parent is not None else None
    orchestration = commit.orchestration
    if orchestration is None:
        _validate_inherited_evaluation(
            commit,
            inherited_champion,
            inherited_evaluation,
            current_stats,
            previous_stats,
            standalone=True,
        )
        return
    recipe = _validate_recipe_contract(commit, checkpoints)
    _validate_orchestration_facts(
        commit,
        parent,
        checkpoint,
        checkpoints,
        current_stats,
        previous_stats,
        recipe,
    )
    if orchestration.evaluation_id is None:
        _validate_inherited_evaluation(
            commit,
            inherited_champion,
            inherited_evaluation,
            current_stats,
            previous_stats,
            standalone=False,
        )
        return
    artifact, evaluation_id = _resolve_evaluation(orchestration, evaluations, prepared_evaluation)
    _validate_evaluation_recipe(artifact, recipe)
    _validate_evaluation_binding(
        commit,
        inherited_champion,
        inherited_evaluation,
        artifact,
        evaluation_id,
    )
    if prepared_evaluation is None:
        lineage = evaluations.list_evaluations(commit.evaluation_head_id)
        if not lineage or lineage[-1].evaluation_id != evaluation_id:
            raise ArtifactValidationError("RunCommit evaluation lineage is incomplete")
    _validate_eval_stats_continuity(current_stats, previous_stats, artifact)
