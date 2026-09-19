"""Crash-consistent Cartridge2 synchronized training composition root."""

from __future__ import annotations

import copy
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from crucible.orchestrator.config import IterationStats
from crucible.orchestrator.orchestrator import (
    Orchestrator as _CoreOrchestrator,
)
from crucible.orchestrator.orchestrator import (
    TrainSpec,
)

from ..algorithms import get_algorithm
from ..environment_catalog import get_environment
from ..stats import (
    EvaluationStats,
    PreparedStatsSnapshotV3,
    decode_stats_snapshot,
    prepare_stats_snapshot,
)
from ..storage import ReplayProfile, ReplaySelection, create_replay_store
from ..storage.evaluation import (
    ChampionReferenceV1,
    create_evaluation_repository,
)
from ..storage.publisher import (
    ArtifactValidationError,
    CheckpointRef,
    create_checkpoint_publisher,
)
from ..storage.run_commit import (
    LearnerRecipeV1,
    OrchestrationCommitV1,
    RunCommitRef,
    RunCommitRepository,
    RunCommitV1,
    RunRecipeV1,
)
from ..structured_logging import (
    generate_span_id,
    generate_trace_id,
    set_trace_context,
)
from .config import LoopConfig
from .eval_runner import EvalRunner, PreparedEvaluation
from .run_journal import PreparedRunV1, RunJournal
from .stats_manager import StatsManager

SUPPORTED_ORCHESTRATION = "synchronized_alphazero_v1"
logger = logging.getLogger(__name__)


@dataclass
class _LoopTrainSpec(TrainSpec):
    replay_selection: ReplaySelection | None = None


def _start_iteration_trace() -> str:
    trace_id = generate_trace_id()
    set_trace_context(trace_id=trace_id, span_id=generate_span_id())
    return trace_id


def _utc_after(*timestamps: str | None) -> str:
    now = datetime.now(timezone.utc)
    for value in timestamps:
        if value is None:
            continue
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if now <= parsed:
            now = parsed + timedelta(microseconds=1)
    return now.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _loop_train_spec(
    config: LoopConfig,
    *,
    num_steps: int,
    start_step: int,
    shutdown_check=None,
    metrics_hook=None,
    replay_selection: ReplaySelection | None = None,
) -> _LoopTrainSpec:
    """Build the exact TrainSpec used by preflight hashing and execution."""
    return _LoopTrainSpec(
        model_dir=str(config.models_dir),
        stats_path=str(config.stats_path),
        env_id=config.env_id,
        total_steps=num_steps,
        start_step=start_step,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        # Deferred loop learners never publish intermediate checkpoints.  This
        # required host-interface value is therefore operationally inert.
        checkpoint_interval=1,
        device=config.resolve_device(),
        lr_total_steps=config.iterations * config.steps_per_iteration,
        shutdown_check=shutdown_check,
        metrics_hook=metrics_hook,
        replay_selection=replay_selection,
    )


def _run_recipe(config: LoopConfig, algorithm) -> RunRecipeV1:
    spec = _loop_train_spec(
        config,
        num_steps=config.steps_per_iteration,
        start_step=0,
    )
    learner_recipe = LearnerRecipeV1(algorithm.loop_learner_recipe(spec, config))
    learner_digest = algorithm.loop_learner_config_sha256(spec, config)
    return RunRecipeV1(
        learner_recipe=learner_recipe,
        learner_config_sha256=learner_digest,
        total_iterations=config.iterations,
        episodes_per_iteration=config.episodes_per_iteration,
        training_steps_per_iteration=config.steps_per_iteration,
        num_actors=config.num_actors,
        collector_episode_timeout_seconds=config.actor_episode_timeout_seconds,
        collector_eval_batch_size=config.actor_eval_batch_size,
        collector_onnx_intra_threads=config.actor_onnx_intra_threads,
        mcts_start_simulations=config.mcts_start_sims,
        mcts_max_simulations=config.mcts_max_sims,
        mcts_simulation_ramp=config.mcts_sim_ramp_rate,
        collector_c_puct=config.c_puct,
        collector_temperature=config.temperature,
        collector_late_temperature=config.late_temperature,
        temperature_move_threshold=config.temp_threshold,
        collector_dirichlet_alpha=config.dirichlet_alpha,
        collector_dirichlet_weight=config.dirichlet_weight,
        collector_seed_strategy="system_entropy_v1",
        replay_policy="scoped_fresh_iteration_v1",
        evaluation_interval=config.eval_interval,
        evaluation_games=config.eval_games,
        evaluation_simulations=config.eval_simulations,
        evaluation_temperature=config.eval_temperature,
        evaluation_win_threshold=config.eval_win_threshold,
        evaluation_vs_random=config.eval_vs_random,
        solver_games=config.solver_games,
        evaluation_seed=config.evaluation_seed,
        promotion_metric=config.promotion_metric,
        promotion_margin=config.promotion_margin,
    )


class Orchestrator(_CoreOrchestrator):
    """Own the transactional loop around immutable Cartridge2 artifacts."""

    def __init__(self, config: LoopConfig):
        # Own a private execution snapshot. Callers may keep and mutate their
        # request object, but actor/evaluator/trainer execution is fenced to
        # the exact snapshot authenticated below.
        config = copy.deepcopy(config)
        algorithm = get_algorithm(config.algorithm_id)
        environment = get_environment(config.env_id)
        algorithm.compatibility(environment).require_compatible()
        orchestration = algorithm.descriptor.components.orchestration
        if orchestration != SUPPORTED_ORCHESTRATION:
            raise ValueError(
                f"Algorithm '{config.algorithm_id}' requires unsupported loop recipe "
                f"'{orchestration}'"
            )
        self.algorithm = algorithm
        self.run_recipe = _run_recipe(config, algorithm)
        checkpoint_repository = create_checkpoint_publisher(
            algorithm.artifact_contract(environment),
            config.models_dir,
        )
        evaluation_repository = create_evaluation_repository(checkpoint_repository)
        preflight_commits = RunCommitRepository(
            checkpoint_repository,
            evaluation_repository,
        )
        preflight_head = checkpoint_repository.resolve_run_head()
        preflight_chain = preflight_commits.resolve_chain(
            preflight_head.run_commit_id if preflight_head is not None else None
        )
        preflight_parent_id = preflight_chain[-1].run_commit_id if preflight_chain else None
        if preflight_chain:
            selected = preflight_chain[-1].commit
            if selected.config_sha256 != self.run_recipe.learner_config_sha256:
                raise ArtifactValidationError(
                    "RunHead learner config does not match the requested run recipe"
                )
            if selected.run_recipe != self.run_recipe:
                raise ArtifactValidationError(
                    "RunHead is not part of the requested synchronized run recipe"
                )
        preflight_journal = RunJournal(checkpoint_repository, preflight_commits)
        pending = preflight_journal.resolve(preflight_parent_id)
        recovered_before_logger: RunCommitV1 | None = None
        if pending is not None:
            if (
                pending.run_commit.config_sha256 != self.run_recipe.learner_config_sha256
                or pending.run_commit.run_recipe != self.run_recipe
            ):
                raise ArtifactValidationError(
                    "Prepared run does not match the requested run recipe"
                )
            recovery_evaluator = EvalRunner(
                config,
                checkpoint_repository=checkpoint_repository,
                evaluation_repository=evaluation_repository,
            )
            if pending.evaluation is not None:
                recovery_evaluator.commit(pending.evaluation)
            preflight_commits.publish(pending.run_commit)
            recovered_head = checkpoint_repository.commit_run_head(
                checkpoint_id=pending.run_commit.checkpoint_id,
                run_commit_id=pending.run_commit_id,
                expected_run_commit_id=preflight_parent_id,
            )
            if recovered_head.run_commit_id != pending.run_commit_id:
                raise ArtifactValidationError(
                    "RunHead advanced beyond the recovered preparation; stop and "
                    "restart from authoritative state"
                )
            recovered_before_logger = pending.run_commit
            preflight_head = checkpoint_repository.resolve_run_head()
            preflight_chain = preflight_commits.resolve_chain(preflight_head.run_commit_id)
        previous_iterations = [
            reference.commit.orchestration.iteration
            for reference in preflight_chain
            if reference.commit.orchestration is not None
        ]
        config._set_start_iteration(previous_iterations[-1] + 1 if previous_iterations else 1)
        replay_profile = ReplayProfile(
            env_id=config.env_id,
            env_contract_version=environment.contract_version,
            algorithm_id=algorithm.descriptor.id,
            experience_schema=algorithm.descriptor.components.experience_schema,
        )
        initial_source = preflight_chain[-1].commit.checkpoint_id if preflight_chain else None
        self.replay_profile = replay_profile
        self._active_replay_selection = ReplaySelection(
            profile=replay_profile,
            collection_scope_id=secrets.token_hex(32),
            source_checkpoint_id=initial_source,
        )
        super().__init__(
            config,
            replay_buffer_factory=lambda: create_replay_store(self._active_replay_selection),
            trainer_factory=lambda spec: algorithm.build_loop_learner(spec, config),
            actor_runner_factory=algorithm.build_collector_runner,
            eval_runner_factory=lambda loop_config, wandb_logger=None: EvalRunner(
                loop_config,
                wandb_logger=wandb_logger,
                checkpoint_repository=checkpoint_repository,
                evaluation_repository=evaluation_repository,
            ),
            trace_starter=_start_iteration_trace,
        )
        self.run_commits = RunCommitRepository(
            self.eval_runner.checkpoints,
            self.eval_runner.evaluations,
        )
        self.run_journal = RunJournal(
            self.eval_runner.checkpoints,
            self.run_commits,
        )
        self._recover_authoritative_state()
        if recovered_before_logger is not None:
            self._report_committed(recovered_before_logger)

    def _auto_resume_if_needed(self) -> None:
        """Install projection writer; RunHead recovery occurs after EvalRunner exists."""
        self.stats_manager = StatsManager(self.config)

    def _require_execution_matches_recipe(self) -> None:
        """Fence every execution phase against authenticated-recipe drift."""
        if _run_recipe(self.config, self.algorithm) != self.run_recipe:
            raise ArtifactValidationError(
                "Live loop configuration no longer matches the authenticated run recipe"
            )

    def _begin_replay_attempt(self, parent: RunCommitV1 | None) -> ReplaySelection:
        """Open a cryptographically fresh collection fence for one attempt."""
        selection = ReplaySelection(
            profile=self.replay_profile,
            collection_scope_id=secrets.token_hex(32),
            source_checkpoint_id=(parent.checkpoint_id if parent is not None else None),
        )
        replay_buffer = create_replay_store(selection)
        try:
            self.actor_runner.select_replay(selection)
        except Exception:
            replay_buffer.close()
            raise
        previous = self._replay_buffer
        self._replay_buffer = replay_buffer
        self._active_replay_selection = selection
        previous.close()
        if self._replay_buffer.count() != 0:
            raise ArtifactValidationError("Fresh replay collection scope is unexpectedly nonempty")
        return selection

    def _get_transition_count(self) -> int:
        return self._replay_buffer.count()

    def _head_chain(self) -> list[RunCommitRef]:
        head = self.eval_runner.checkpoints.resolve_run_head()
        return self.run_commits.resolve_chain(head.run_commit_id if head is not None else None)

    @staticmethod
    def _iteration_stats(commit: RunCommitV1) -> IterationStats | None:
        value = commit.orchestration
        if value is None:
            return None
        return IterationStats(
            iteration=value.iteration,
            episodes_generated=value.episodes_generated,
            transitions_generated=value.transitions_generated,
            training_steps=value.training_steps,
            actor_time_seconds=value.actor_time_seconds,
            trainer_time_seconds=value.trainer_time_seconds,
            eval_time_seconds=value.eval_time_seconds,
            total_time_seconds=value.total_time_seconds,
            eval_win_rate=value.eval_win_rate,
            eval_draw_rate=value.eval_draw_rate,
            timestamp=value.timestamp,
        )

    def _finish_prepared(self, prepared: PreparedRunV1) -> RunCommitRef:
        self.run_commits.validate_prepared(
            prepared.run_commit,
            prepared.evaluation.artifact if prepared.evaluation is not None else None,
        )
        head = self.eval_runner.checkpoints.resolve_run_head()
        actual_parent = head.run_commit_id if head is not None else None
        if actual_parent != prepared.parent_run_commit_id:
            if head is not None and head.run_commit_id == prepared.run_commit_id:
                return self.run_commits.resolve(prepared.run_commit_id)
            raise ArtifactValidationError("RunHead no longer matches the prepared run parent")
        if prepared.evaluation is not None:
            self.eval_runner.commit(prepared.evaluation)
        reference = self.run_commits.publish(prepared.run_commit)
        committed_head = self.eval_runner.checkpoints.commit_run_head(
            checkpoint_id=prepared.run_commit.checkpoint_id,
            run_commit_id=reference.run_commit_id,
            expected_run_commit_id=prepared.parent_run_commit_id,
        )
        if committed_head.run_commit_id != reference.run_commit_id:
            raise ArtifactValidationError(
                "RunHead advanced beyond this prepared iteration; stop and restart "
                "from authoritative state"
            )
        return reference

    def _rebuild_projections(self, chain: list[RunCommitRef]) -> None:
        self.iteration_history = [
            stats
            for reference in chain
            if (stats := self._iteration_stats(reference.commit)) is not None
        ]
        evaluations = [
            self.eval_runner.evaluations.resolve_evaluation(
                reference.commit.orchestration.evaluation_id
            )
            for reference in chain
            if reference.commit.orchestration is not None
            and reference.commit.orchestration.evaluation_id is not None
        ]
        self.eval_history = [
            self.eval_runner._build_eval_record(evaluation) for evaluation in evaluations
        ]
        solver_history = self.eval_runner._build_solver_history(evaluations)
        latest = chain[-1].commit if chain else None
        self.eval_runner.set_authoritative_state(latest)
        if latest is not None:
            self.stats_manager.rebuild_projections(
                history=self.iteration_history,
                eval_history=self.eval_history,
                solver_history=solver_history,
                stats_snapshot=latest.stats_snapshot,
            )

    def _recover_authoritative_state(self) -> None:
        chain = self._head_chain()
        parent_id = chain[-1].run_commit_id if chain else None
        prepared = self.run_journal.resolve(parent_id)
        recovered: PreparedRunV1 | None = None
        if prepared is not None:
            self._finish_prepared(prepared)
            recovered = prepared
            chain = self._head_chain()
        if not chain:
            for path in (
                self.config.stats_path,
                self.config.loop_stats_path,
                self.config.eval_stats_path,
                self.config.solver_stats_path,
            ):
                if path.exists():
                    logger.warning(
                        "Discarding stale mutable projection without RunHead: %s",
                        path,
                    )
                    path.unlink()
            return
        self._rebuild_projections(chain)
        next_iteration = (
            self.iteration_history[-1].iteration + 1
            if self.iteration_history
            else self.config.start_iteration
        )
        self.config._set_start_iteration(next_iteration)
        if recovered is not None:
            self._report_committed(recovered.run_commit)

    def _run_trainer_deferred(
        self,
        num_steps: int,
        start_step: int,
        replay_selection: ReplaySelection,
    ) -> tuple[bool, float, CheckpointRef | None, PreparedStatsSnapshotV3 | None]:
        spec = _loop_train_spec(
            self.config,
            num_steps=num_steps,
            start_step=start_step,
            shutdown_check=lambda: self._shutdown_requested,
            metrics_hook=self._train_metrics_hook,
            replay_selection=replay_selection,
        )
        started = time.perf_counter()
        try:
            learner = self._trainer_factory(spec)
            learner.train()
            checkpoint = getattr(learner, "last_checkpoint_ref", None)
            snapshot = getattr(learner, "last_prepared_stats", None)
            if not isinstance(checkpoint, CheckpointRef) or not isinstance(
                snapshot, PreparedStatsSnapshotV3
            ):
                raise ArtifactValidationError(
                    "Deferred learner did not return its exact checkpoint/stats handoff"
                )
            return True, time.perf_counter() - started, checkpoint, snapshot
        except Exception:
            logger.exception("Trainer failed")
            return False, time.perf_counter() - started, None, None

    @staticmethod
    def _snapshot_with_evaluation(
        snapshot: PreparedStatsSnapshotV3,
        candidate: CheckpointRef,
        evaluation: PreparedEvaluation | None,
    ) -> PreparedStatsSnapshotV3:
        if evaluation is None or evaluation.artifact.results.vs_random is None:
            return snapshot
        loaded = decode_stats_snapshot(
            snapshot.data,
            expected_stats_id=snapshot.stats_id,
            expected_binding=snapshot.binding,
        )
        result = evaluation.artifact.results.vs_random
        completed = datetime.fromisoformat(evaluation.artifact.completed_at.replace("Z", "+00:00"))
        loaded.stats.append_evaluation(
            EvaluationStats(
                step=candidate.manifest.step,
                metrics={
                    "outcome/win_rate": result.candidate_win_rate,
                    "outcome/draw_rate": result.draw_rate,
                    "outcome/loss_rate": (1.0 - result.candidate_win_rate - result.draw_rate),
                },
                episodes=result.games_played,
                mean_episode_length=result.average_game_length,
                timestamp=completed.timestamp(),
            )
        )
        return prepare_stats_snapshot(loaded.stats, candidate)

    def _build_run_commit(
        self,
        *,
        parent: RunCommitV1 | None,
        iteration: int,
        replay_selection: ReplaySelection,
        episodes: int,
        candidate: CheckpointRef,
        snapshot: PreparedStatsSnapshotV3,
        actor_time: float,
        trainer_time: float,
        transitions: int,
        total_time: float,
        evaluation: PreparedEvaluation | None,
    ) -> RunCommitV1:
        previous_timestamp = None
        for reference in reversed(self._head_chain()):
            if reference.commit.orchestration is not None:
                previous_timestamp = reference.commit.orchestration.timestamp
                break
        evaluation_completed = evaluation.artifact.completed_at if evaluation is not None else None
        timestamp = _utc_after(previous_timestamp, evaluation_completed)
        orchestration = OrchestrationCommitV1(
            iteration=iteration,
            episodes_generated=episodes,
            transitions_generated=transitions,
            training_steps=self.config.steps_per_iteration,
            actor_time_seconds=actor_time,
            trainer_time_seconds=trainer_time,
            eval_time_seconds=(evaluation.elapsed_seconds if evaluation is not None else 0.0),
            total_time_seconds=total_time,
            eval_win_rate=evaluation.win_rate if evaluation is not None else None,
            eval_draw_rate=evaluation.draw_rate if evaluation is not None else None,
            timestamp=timestamp,
            evaluation_id=(evaluation.evaluation_id if evaluation is not None else None),
            collector_simulations=self.run_recipe.simulations_for(iteration),
            collector_seed=None,
            evaluation_seed=(evaluation.artifact.recipe.seed if evaluation is not None else None),
            collection_scope_id=replay_selection.collection_scope_id,
            source_checkpoint_id=replay_selection.source_checkpoint_id,
        )
        inherited_champion = parent.champion if parent is not None else None
        inherited_evaluation = parent.evaluation_head_id if parent is not None else None
        champion = inherited_champion
        evaluation_head = inherited_evaluation
        if evaluation is not None:
            evaluation_head = evaluation.evaluation_id
            if evaluation.artifact.decision.promoted:
                champion = ChampionReferenceV1(
                    checkpoint_id=candidate.checkpoint_id,
                    evaluation_id=evaluation.evaluation_id,
                )
        return RunCommitV1(
            profile=candidate.manifest.profile,
            config_sha256=candidate.manifest.config_sha256,
            parent_run_commit_id=parent.run_commit_id if parent is not None else None,
            checkpoint_id=candidate.checkpoint_id,
            stats_snapshot=snapshot,
            champion=champion,
            evaluation_head_id=evaluation_head,
            orchestration=orchestration,
            run_recipe=self.run_recipe,
        )

    def _report_committed(self, commit: RunCommitV1) -> None:
        orchestration = commit.orchestration
        if orchestration is None:
            return
        metrics = {
            "loop/iteration": orchestration.iteration,
            "loop/actor_seconds": orchestration.actor_time_seconds,
            "loop/trainer_seconds": orchestration.trainer_time_seconds,
            "loop/eval_seconds": orchestration.eval_time_seconds,
            "loop/total_seconds": orchestration.total_time_seconds,
            "loop/episodes": orchestration.episodes_generated,
            "loop/transitions": orchestration.transitions_generated,
            "loop/mcts_simulations": self.run_recipe.simulations_for(orchestration.iteration),
        }
        self.wandb_logger.log(
            metrics,
            step=orchestration.iteration * self.config.steps_per_iteration,
        )
        if orchestration.evaluation_id is not None:
            evaluation = self.eval_runner.evaluations.resolve_evaluation(
                orchestration.evaluation_id
            )
            self.eval_runner._log_eval_to_wandb(self.eval_runner._build_eval_record(evaluation))

    def run_iteration(self, iteration: int) -> IterationStats | None:
        self._require_execution_matches_recipe()
        iter_started = time.perf_counter()
        trace_id = self._trace_starter()
        chain = self._head_chain()
        parent = chain[-1].commit if chain else None
        replay_selection = self._begin_replay_attempt(parent)
        logger.info("ITERATION %s", iteration)
        if self._shutdown_requested:
            return None
        actor_success, actor_time = self.actor_runner.run(
            self.config.episodes_per_iteration,
            iteration,
            trace_id=trace_id,
        )
        if not actor_success:
            if not self._shutdown_requested:
                logger.error("Actor failed, aborting iteration")
            return None
        episodes = self._replay_buffer.count_episodes()
        if episodes != self.config.episodes_per_iteration:
            raise ArtifactValidationError(
                "Collector completed without the exact configured episode count: "
                f"expected {self.config.episodes_per_iteration}, got {episodes}"
            )
        transitions = self._get_transition_count()
        if self._shutdown_requested:
            return None
        start_step = parent.stats_snapshot.binding.step if parent is not None else 0
        trainer_success, trainer_time, candidate, snapshot = self._run_trainer_deferred(
            self.config.steps_per_iteration,
            start_step,
            replay_selection,
        )
        if not trainer_success or candidate is None or snapshot is None:
            return None
        if self._shutdown_requested:
            return None
        should_eval = self.run_recipe.evaluation_scheduled(iteration)
        self._require_execution_matches_recipe()
        evaluation = self.eval_runner.prepare(iteration, candidate, parent) if should_eval else None
        snapshot = self._snapshot_with_evaluation(snapshot, candidate, evaluation)
        total_time = time.perf_counter() - iter_started
        commit = self._build_run_commit(
            parent=parent,
            iteration=iteration,
            replay_selection=replay_selection,
            episodes=episodes,
            candidate=candidate,
            snapshot=snapshot,
            actor_time=actor_time,
            trainer_time=trainer_time,
            transitions=transitions,
            total_time=total_time,
            evaluation=evaluation,
        )
        prepared = PreparedRunV1(run_commit=commit, evaluation=evaluation)
        self.run_journal.publish(prepared)
        self._finish_prepared(prepared)
        committed_chain = self._head_chain()
        self._rebuild_projections(committed_chain)
        self._report_committed(commit)
        return self._iteration_stats(commit)

    def _log_startup_summary(self) -> None:
        logger.info(
            "Synchronized training: env=%s start=%s iterations=%s champion=%s",
            self.config.env_id,
            self.config.start_iteration,
            self.config.iterations,
            self.eval_runner.champion_iteration,
        )

    def run(self) -> None:
        self._ensure_directories()
        self._log_startup_summary()
        loop_started = time.perf_counter()
        try:
            for iteration in range(
                self.config.start_iteration,
                self.config.iterations + 1,
            ):
                if self._shutdown_requested:
                    break
                if self.run_iteration(iteration) is None:
                    if self._shutdown_requested:
                        break
                    raise RuntimeError(f"iteration {iteration} failed")
        finally:
            self.wandb_logger.finish()
            self._replay_buffer.close()
        self._log_completion_summary(time.perf_counter() - loop_started)


__all__ = ["Orchestrator"]
