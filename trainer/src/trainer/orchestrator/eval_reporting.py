"""Reporting projections derived from immutable evaluation artifacts."""

from __future__ import annotations

from ..storage.evaluation import EvaluationRef, SolverBucketV1
from ..storage.publisher import ArtifactValidationError
from .solver_reporting_validation import (
    SOLVER_PLY_BUCKETS,
    SOLVER_SEAT_BUCKETS,
    validate_solver_entry,
    validate_solver_history,
)


class EvalReportingMixin:
    """Project durable evidence into UI history, solver history, and W&B."""

    def _build_eval_record(self, evaluation: EvaluationRef) -> dict:
        artifact = evaluation.artifact
        vs_champion = artifact.results.vs_champion
        vs_random = artifact.results.vs_random
        solver = artifact.results.candidate_solver
        candidate_checkpoint = self.checkpoints.read_checkpoint_manifest_exact(
            artifact.candidate_checkpoint_id
        )
        champion_reference = artifact.champion_before
        champion_evaluation = None
        if champion_reference is not None:
            champion_evaluation = self.evaluations.resolve_evaluation(
                champion_reference.evaluation_id
            )
            if (
                champion_evaluation.artifact.candidate_checkpoint_id
                != champion_reference.checkpoint_id
            ):
                raise ArtifactValidationError(
                    "Evaluation record champion reference is inconsistent"
                )
        return {
            "iteration": artifact.iteration,
            "step": candidate_checkpoint.step,
            "candidate_checkpoint_id": artifact.candidate_checkpoint_id,
            "evaluation_id": evaluation.evaluation_id,
            "vs_champion_checkpoint_id": (
                champion_reference.checkpoint_id if champion_reference is not None else None
            ),
            "vs_champion_evaluation_id": (
                champion_reference.evaluation_id if champion_reference is not None else None
            ),
            "vs_champion_win_rate": (
                vs_champion.candidate_win_rate if vs_champion is not None else None
            ),
            "vs_champion_draw_rate": (vs_champion.draw_rate if vs_champion is not None else None),
            "vs_champion_average_game_length": (
                vs_champion.average_game_length if vs_champion is not None else None
            ),
            "vs_champion_iteration": (
                champion_evaluation.artifact.iteration if champion_evaluation is not None else None
            ),
            "promoted": artifact.decision.promoted,
            "promotion_reason": artifact.decision.reason,
            "vs_random_win_rate": (vs_random.candidate_win_rate if vs_random is not None else None),
            "vs_random_draw_rate": (vs_random.draw_rate if vs_random is not None else None),
            "vs_random_average_game_length": (
                vs_random.average_game_length if vs_random is not None else None
            ),
            "solver_value_optimal_rate": (
                solver.overall.value_optimal_rate if solver is not None else None
            ),
            "solver_exact_best_rate": (
                solver.overall.exact_best / solver.overall.positions
                if solver is not None and solver.overall.positions
                else (0.0 if solver is not None else None)
            ),
            "solver_blunder_rate": (
                (
                    solver.overall.blunders_win_to_draw
                    + solver.overall.blunders_win_to_loss
                    + solver.overall.blunders_draw_to_loss
                )
                / solver.overall.positions
                if solver is not None and solver.overall.positions
                else (0.0 if solver is not None else None)
            ),
            "solver_positions": (solver.overall.positions if solver is not None else None),
            "promotion_metric": artifact.recipe.promotion_metric,
            "requested_vs_champion_games": (artifact.recipe.requested_games.vs_champion),
            "requested_vs_random_games": artifact.recipe.requested_games.vs_random,
            "timestamp": artifact.completed_at,
        }

    @staticmethod
    def _solver_bucket_projection(bucket: SolverBucketV1) -> dict[str, int | float]:
        positions = bucket.positions
        blunders = (
            bucket.blunders_win_to_draw + bucket.blunders_win_to_loss + bucket.blunders_draw_to_loss
        )

        def rate(count: int) -> float:
            return count / positions if positions else 0.0

        return {
            **bucket.to_dict(),
            "value_optimal_rate": rate(bucket.value_optimal),
            "exact_best_rate": rate(bucket.exact_best),
            "blunder_rate": rate(blunders),
            "forced_rate": rate(bucket.forced),
        }

    def _build_solver_record(self, evaluation: EvaluationRef) -> dict | None:
        artifact = evaluation.artifact
        solver = artifact.results.candidate_solver
        if solver is None:
            return None
        checkpoint = self.checkpoints.read_checkpoint_manifest_exact(
            artifact.candidate_checkpoint_id
        )
        overall = solver.overall
        blunders = (
            overall.blunders_win_to_draw
            + overall.blunders_win_to_loss
            + overall.blunders_draw_to_loss
        )

        def rate(count: int) -> float:
            return count / overall.positions if overall.positions else 0.0

        entry = {
            "model": f"checkpoint:{artifact.candidate_checkpoint_id}",
            "model_path": f"checkpoint:{artifact.candidate_checkpoint_id}",
            "checkpoint_id": artifact.candidate_checkpoint_id,
            "step": checkpoint.step,
            "env_id": artifact.profile.env_id,
            "opponent": "random_v1",
            "games": solver.games_played,
            "seed": artifact.recipe.seed,
            "temperature": 0.0,
            "model_wins": solver.candidate_wins,
            "model_losses": solver.opponent_wins,
            "draws": solver.draws,
            "avg_game_length": solver.average_game_length,
            "positions_scored": overall.positions,
            "forced_moves": overall.forced,
            "forced_move_rate": rate(overall.forced),
            "value_optimal_rate": rate(overall.value_optimal),
            "exact_best_rate": rate(overall.exact_best),
            "blunder_rate": rate(blunders),
            "blunders_win_to_draw": overall.blunders_win_to_draw,
            "blunders_win_to_loss": overall.blunders_win_to_loss,
            "blunders_draw_to_loss": overall.blunders_draw_to_loss,
            "by_ply": {
                name: self._solver_bucket_projection(solver.by_ply[name])
                for name in SOLVER_PLY_BUCKETS
            },
            "by_seat": {
                name: self._solver_bucket_projection(solver.by_seat[name])
                for name in SOLVER_SEAT_BUCKETS
            },
            "solver_queries": solver.solver_queries,
            "solver_cache_hits": solver.solver_cache_hits,
            "solver_cache_hit_rate": (
                solver.solver_cache_hits / solver.solver_queries if solver.solver_queries else 0.0
            ),
            "solver_time_seconds": solver.solver_time_seconds,
            "wall_time_seconds": solver.wall_time_seconds,
            "bitbully_version": solver.solver_version,
            "timestamp": artifact.completed_at,
            "iteration": artifact.iteration,
            "global_step": checkpoint.step,
            "context": "loop",
            "evaluation_id": evaluation.evaluation_id,
        }
        validate_solver_entry(entry, "projection")
        return entry

    def _build_solver_history(self, evaluations: list[EvaluationRef]) -> list[dict]:
        records = [
            record
            for evaluation in evaluations
            if (record := self._build_solver_record(evaluation)) is not None
        ]
        return validate_solver_history(records)

    def _log_eval_to_wandb(self, eval_record: dict) -> None:
        if self.wandb_logger is None:
            return
        metrics = {
            "eval/vs_champion_win_rate": eval_record["vs_champion_win_rate"],
            "eval/vs_champion_draw_rate": eval_record["vs_champion_draw_rate"],
            "eval/promoted": int(eval_record["promoted"]),
            "eval/champion_iteration": self.champion_iteration,
            "eval/vs_random_win_rate": eval_record["vs_random_win_rate"],
            "eval/vs_random_draw_rate": eval_record["vs_random_draw_rate"],
            "solver/value_optimal_rate": eval_record["solver_value_optimal_rate"],
            "solver/exact_best_rate": eval_record["solver_exact_best_rate"],
            "solver/blunder_rate": eval_record["solver_blunder_rate"],
            "solver/positions_scored": eval_record["solver_positions"],
        }
        self.wandb_logger.log(
            {key: value for key, value in metrics.items() if value is not None},
            step=eval_record["step"],
        )


__all__ = ["EvalReportingMixin"]
