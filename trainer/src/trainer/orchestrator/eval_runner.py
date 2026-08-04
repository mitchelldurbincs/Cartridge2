"""Content-addressed evaluation and champion selection for Cartridge2."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from ..algorithms import get_algorithm
from ..environment_catalog import get_environment
from ..evaluator import evaluate as run_eval
from ..players import ModelPlayer, RandomPlayer
from ..solver_eval import SolverScorer, solver_evaluate
from ..storage.evaluation import (
    ChampionReferenceV1,
    EvaluationArtifactV2,
    EvaluationRecipeV1,
    EvaluationRepository,
    HeadToHeadResultV1,
    ObservedResultsV1,
    PromotionDecisionV1,
    RequestedGamesV1,
    SolverResultV1,
    create_evaluation_repository,
)
from ..storage.publisher import (
    ArtifactValidationError,
    CheckpointPublisher,
    CheckpointRef,
    create_checkpoint_publisher,
)
from ..storage.run_commit import RunCommitV1
from .config import LoopConfig
from .eval_reporting import EvalReportingMixin

if TYPE_CHECKING:
    from crucible.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_MAX_F32 = float.fromhex("0x1.fffffep+127")


def _should_promote(
    *,
    promotion_metric: str,
    vs_champion_win_rate: float,
    win_threshold: float,
    candidate_solver_rate: float | None,
    champion_solver_rate: float | None,
    margin: float,
) -> tuple[bool, str]:
    """Make the promotion decision using only champion terminology."""
    if promotion_metric == "solver_optimal":
        if candidate_solver_rate is None or champion_solver_rate is None:
            raise RuntimeError(
                "solver_optimal promotion requires complete candidate and champion solver results"
            )
        if candidate_solver_rate > champion_solver_rate + margin:
            return True, (
                f"solver_optimal: candidate {candidate_solver_rate:.1%} > "
                f"champion {champion_solver_rate:.1%} + margin {margin:.1%}"
            )
        return False, (
            f"solver_optimal: candidate {candidate_solver_rate:.1%} <= "
            f"champion {champion_solver_rate:.1%} + margin {margin:.1%}"
        )

    if vs_champion_win_rate > win_threshold:
        return True, (f"win_rate: {vs_champion_win_rate:.1%} > threshold {win_threshold:.1%}")
    return False, (f"win_rate: {vs_champion_win_rate:.1%} <= threshold {win_threshold:.1%}")


def _model_player(model_path: str, temperature: float, simulations: int) -> ModelPlayer:
    return ModelPlayer(
        model_path=model_path,
        temperature=temperature,
        simulations=simulations,
    )


def _utc_strictly_after(previous: str | None) -> str:
    now = datetime.now(timezone.utc)
    if previous is not None:
        prior = datetime.fromisoformat(previous.replace("Z", "+00:00"))
        if now <= prior:
            now = prior + timedelta(microseconds=1)
    return now.isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class PreparedEvaluation:
    """Fully evaluated, canonical evidence that has not mutated any authority."""

    artifact: EvaluationArtifactV2
    evaluation_id: str
    win_rate: float | None
    draw_rate: float | None
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, EvaluationArtifactV2):
            raise TypeError("artifact must be EvaluationArtifactV2")
        if self.evaluation_id != self.artifact.evaluation_id:
            raise ArtifactValidationError(
                "Prepared evaluation ID does not match its canonical artifact"
            )
        observed = self.artifact.results.vs_champion
        expected_win = observed.candidate_win_rate if observed is not None else None
        expected_draw = observed.draw_rate if observed is not None else None
        if (self.win_rate, self.draw_rate) != (expected_win, expected_draw):
            raise ArtifactValidationError(
                "Prepared evaluation rates disagree with immutable evidence"
            )
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        normalized = float(self.elapsed_seconds)
        object.__setattr__(self, "elapsed_seconds", 0.0 if normalized == 0.0 else normalized)


class EvalRunner(EvalReportingMixin):
    """Evaluate the current immutable checkpoint and publish promotion evidence."""

    def __init__(
        self,
        config: LoopConfig,
        wandb_logger: "WandbLogger | None" = None,
        *,
        checkpoint_repository: CheckpointPublisher | None = None,
        evaluation_repository: EvaluationRepository | None = None,
    ):
        self.config = config
        self.wandb_logger = wandb_logger
        self._validate_config()
        algorithm = get_algorithm(config.algorithm_id)
        environment = get_environment(config.env_id)
        algorithm.compatibility(environment).require_compatible()
        contract = algorithm.artifact_contract(environment)
        self.checkpoints = checkpoint_repository or create_checkpoint_publisher(
            contract, config.models_dir
        )
        self.evaluations = evaluation_repository or create_evaluation_repository(self.checkpoints)
        self._policy_loader = lambda path, temperature: _model_player(
            path, temperature, config.eval_simulations
        )
        self._baseline_policy_factory = RandomPlayer
        self._run_eval = lambda **kwargs: run_eval(algorithm_id=config.algorithm_id, **kwargs)
        self._solver_scorer_factory = SolverScorer
        self._run_solver = lambda **kwargs: solver_evaluate(
            algorithm_id=config.algorithm_id, **kwargs
        )
        self._solver_scorer: object | None = None
        self.champion_iteration: int | None = None

    def _validate_config(self) -> None:
        if not isinstance(self.config.eval_vs_random, bool):
            raise ValueError("eval_vs_random must be boolean")
        for field in (
            "eval_games",
            "solver_games",
            "eval_simulations",
            "evaluation_seed",
        ):
            value = getattr(self.config, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if self.config.eval_games <= 0:
            raise ValueError("eval_games must be positive")
        if self.config.eval_games > _MAX_U32:
            raise ValueError("eval_games exceeds u32")
        if self.config.solver_games > _MAX_U32:
            raise ValueError("solver_games exceeds u32")
        if self.config.eval_simulations > _MAX_U32:
            raise ValueError("eval_simulations exceeds u32")
        if self.config.evaluation_seed > _MAX_U64:
            raise ValueError("evaluation_seed exceeds u64")
        for field in ("eval_temperature", "eval_win_threshold", "promotion_margin"):
            value = getattr(self.config, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{field} must be finite")
        if self.config.eval_temperature < 0:
            raise ValueError("eval_temperature must be nonnegative")
        if self.config.eval_temperature > _MAX_F32:
            raise ValueError("eval_temperature exceeds f32")
        largest_run = max(self.config.eval_games, self.config.solver_games)
        if self.config.evaluation_seed > _MAX_U64 - (largest_run - 1):
            raise ValueError("evaluation_seed plus the game index exceeds u64")
        if not 0.0 <= self.config.eval_win_threshold <= 1.0:
            raise ValueError("eval_win_threshold must be between zero and one")
        if not 0.0 <= self.config.promotion_margin <= 1.0:
            raise ValueError("promotion_margin must be between zero and one")
        if self.config.promotion_metric not in {"win_rate", "solver_optimal"}:
            raise ValueError("promotion_metric must be 'win_rate' or 'solver_optimal'")
        if self.config.promotion_metric == "solver_optimal" and (
            self.config.env_id != "connect4" or self.config.solver_games <= 0
        ):
            raise ValueError("solver_optimal promotion requires connect4 with solver_games > 0")
        if (
            self.config.eval_interval > 0
            and not self.config.eval_vs_random
            and not (self.config.env_id == "connect4" and self.config.solver_games > 0)
        ):
            raise ValueError(
                "Scheduled evaluation requires first-candidate evidence: enable "
                "eval_vs_random or use connect4 with solver_games > 0"
            )

    def _solver_enabled(self) -> bool:
        return self.config.env_id == "connect4" and self.config.solver_games > 0

    def _run_head_to_head(self, candidate, opponent):
        raw = self._run_eval(
            player1=candidate,
            player2=opponent,
            env_id=self.config.env_id,
            num_games=self.config.eval_games,
            verbose=False,
            seed=self.config.evaluation_seed,
        )
        return raw, HeadToHeadResultV1.from_results(raw)

    def _run_solver_eval(self, checkpoint):
        if not self._solver_enabled():
            return None
        if self._solver_scorer is None:
            self._solver_scorer = self._solver_scorer_factory()
        raw = self._run_solver(
            model=self._policy_loader(str(checkpoint.onnx_path), 0.0),
            opponent=self._baseline_policy_factory(),
            scorer=self._solver_scorer,
            env_id=self.config.env_id,
            num_games=self.config.solver_games,
            seed=self.config.evaluation_seed,
            checkpoint_id=checkpoint.checkpoint_id,
            checkpoint_step=checkpoint.manifest.step,
        )
        return SolverResultV1.from_results(raw)

    def _resolve_champion(
        self, parent: RunCommitV1 | None
    ) -> tuple[ChampionReferenceV1 | None, CheckpointRef | None]:
        reference = parent.champion if parent is not None else None
        if reference is None:
            return None, None
        evaluation = self.evaluations.resolve_evaluation(reference.evaluation_id)
        if (
            not evaluation.artifact.decision.promoted
            or evaluation.artifact.candidate_checkpoint_id != reference.checkpoint_id
        ):
            raise ArtifactValidationError(
                "RunCommit champion is not supported by promoted evaluation evidence"
            )
        checkpoint = self.checkpoints.resolve_checkpoint(reference.checkpoint_id)
        self.champion_iteration = evaluation.artifact.iteration
        return reference, checkpoint

    def set_authoritative_state(self, commit: RunCommitV1 | None) -> None:
        """Refresh reporting state from the sole committed RunHead lineage."""
        self.champion_iteration = None
        self._resolve_champion(commit)

    def prepare(
        self,
        iteration: int,
        candidate: CheckpointRef,
        parent: RunCommitV1 | None,
    ) -> PreparedEvaluation:
        """Run games and return canonical evidence without writing any artifact."""
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= 0:
            raise ValueError("iteration must be a positive integer")
        if not isinstance(candidate, CheckpointRef):
            raise TypeError("candidate must be CheckpointRef")
        if candidate.manifest.profile != self.checkpoints.contract.profile:
            raise ArtifactValidationError("Evaluation candidate profile mismatch")
        if parent is not None and not isinstance(parent, RunCommitV1):
            raise TypeError("parent must be RunCommitV1 or None")

        expected_parent_id = parent.run_commit_id if parent is not None else None
        expected_checkpoint_parent = parent.checkpoint_id if parent is not None else None
        if candidate.manifest.parent_checkpoint_id != expected_checkpoint_parent:
            raise ArtifactValidationError(
                "Evaluation candidate must be a direct child of the RunHead checkpoint"
            )
        parent_step = (
            self.checkpoints.read_checkpoint_manifest_exact(parent.checkpoint_id).step
            if parent is not None
            else 0
        )
        if candidate.manifest.step <= parent_step:
            raise ArtifactValidationError(
                "Evaluation candidate checkpoint step must strictly increase"
            )
        head = self.checkpoints.resolve_run_head()
        actual_parent_id = head.run_commit_id if head is not None else None
        if actual_parent_id != expected_parent_id:
            raise ArtifactValidationError("RunHead changed before evaluation preparation")
        previous_evaluation_id = parent.evaluation_head_id if parent is not None else None
        previous_evaluation = (
            self.evaluations.resolve_evaluation(previous_evaluation_id)
            if previous_evaluation_id is not None
            else None
        )
        champion_reference, champion_checkpoint = self._resolve_champion(parent)
        started_clock = time.perf_counter()
        started_at = _utc_strictly_after(
            previous_evaluation.artifact.completed_at if previous_evaluation is not None else None
        )
        candidate_policy = self._policy_loader(
            str(candidate.onnx_path), self.config.eval_temperature
        )

        candidate_solver = self._run_solver_eval(candidate)
        champion_solver = None
        champion_solver_games = 0
        vs_champion = None
        if champion_reference is None:
            promote = True
            reason = "no champion exists; first valid candidate promoted"
            win_rate = None
            draw_rate = None
        else:
            if champion_checkpoint is None:
                raise ArtifactValidationError("RunCommit champion checkpoint is absent")
            _, vs_champion = self._run_head_to_head(
                candidate_policy,
                self._policy_loader(
                    str(champion_checkpoint.onnx_path),
                    self.config.eval_temperature,
                ),
            )
            win_rate = vs_champion.candidate_win_rate
            draw_rate = vs_champion.draw_rate
            champion_solver_rate = None
            if self.config.promotion_metric == "solver_optimal":
                champion_solver = self._run_solver_eval(champion_checkpoint)
                champion_solver_games = self.config.solver_games
                champion_solver_rate = (
                    champion_solver.overall.value_optimal_rate
                    if champion_solver is not None
                    else None
                )
            candidate_solver_rate = (
                candidate_solver.overall.value_optimal_rate
                if candidate_solver is not None
                else None
            )
            promote, reason = _should_promote(
                promotion_metric=self.config.promotion_metric,
                vs_champion_win_rate=win_rate,
                win_threshold=self.config.eval_win_threshold,
                candidate_solver_rate=candidate_solver_rate,
                champion_solver_rate=champion_solver_rate,
                margin=self.config.promotion_margin,
            )

        vs_random = None
        if self.config.eval_vs_random:
            _, vs_random = self._run_head_to_head(
                candidate_policy,
                self._baseline_policy_factory(),
            )
        latest_head = self.checkpoints.resolve_run_head()
        latest_parent_id = latest_head.run_commit_id if latest_head is not None else None
        if latest_parent_id != expected_parent_id:
            raise ArtifactValidationError("RunHead changed during evaluation")

        completed_at = _utc_strictly_after(started_at)
        artifact = EvaluationArtifactV2(
            profile=candidate.manifest.profile,
            iteration=iteration,
            candidate_checkpoint_id=candidate.checkpoint_id,
            previous_evaluation_id=previous_evaluation_id,
            champion_before=champion_reference,
            recipe=EvaluationRecipeV1(
                simulations=self.config.eval_simulations,
                temperature=self.config.eval_temperature,
                promotion_metric=self.config.promotion_metric,
                promotion_margin=self.config.promotion_margin,
                win_threshold=self.config.eval_win_threshold,
                seed=self.config.evaluation_seed,
                requested_games=RequestedGamesV1(
                    vs_champion=(self.config.eval_games if champion_reference is not None else 0),
                    vs_random=(self.config.eval_games if self.config.eval_vs_random else 0),
                    candidate_solver=(self.config.solver_games if self._solver_enabled() else 0),
                    champion_solver=champion_solver_games,
                ),
            ),
            results=ObservedResultsV1(
                vs_champion=vs_champion,
                vs_random=vs_random,
                candidate_solver=candidate_solver,
                champion_solver=champion_solver,
            ),
            decision=PromotionDecisionV1(promoted=promote, reason=reason),
            started_at=started_at,
            completed_at=completed_at,
        )
        return PreparedEvaluation(
            artifact=artifact,
            evaluation_id=artifact.evaluation_id,
            win_rate=win_rate,
            draw_rate=draw_rate,
            elapsed_seconds=time.perf_counter() - started_clock,
        )

    def commit(self, prepared: PreparedEvaluation):
        """Idempotently materialize exact evidence; RunHead chooses its effects."""
        if not isinstance(prepared, PreparedEvaluation):
            raise TypeError("prepared must be PreparedEvaluation")
        reference = self.evaluations.publish_evidence(prepared.artifact)
        if reference.evaluation_id != prepared.evaluation_id:
            raise ArtifactValidationError("Committed evaluation identity changed")
        return reference


__all__ = ["EvalRunner", "PreparedEvaluation"]
