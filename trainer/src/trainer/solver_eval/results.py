"""Aggregated solver-eval results and their JSON/summary serialization.

Judgments are aggregated overall, by ply bucket, and by seat via BucketStats,
then rolled up into a SolverEvalResults for one model. Unlike win-rate-vs-random,
these metrics have a fixed, objective yardstick, so they are comparable across
checkpoints.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from .judgment import (
    BLUNDER_DRAW_TO_LOSS,
    BLUNDER_WIN_TO_DRAW,
    BLUNDER_WIN_TO_LOSS,
    PLY_BUCKETS,
    SEATS,
    MoveJudgment,
)

CHECKPOINT_PATTERN = re.compile(r"model_step_(\d+)\.onnx")


def infer_step_from_filename(path: str | Path) -> int | None:
    """Extract the training step from a checkpoint filename, if present.

    model_step_016000.onnx -> 16000; latest.onnx / best.onnx -> None.
    """
    match = CHECKPOINT_PATTERN.fullmatch(Path(path).name)
    return int(match.group(1)) if match else None


@dataclass
class BucketStats:
    """Aggregated judgments for one slice (overall, a ply bucket, a seat)."""

    positions: int = 0
    value_optimal: int = 0
    exact_best: int = 0
    blunders_win_to_draw: int = 0
    blunders_win_to_loss: int = 0
    blunders_draw_to_loss: int = 0
    forced: int = 0

    def add(self, judgment: MoveJudgment) -> None:
        self.positions += 1
        if judgment.value_optimal:
            self.value_optimal += 1
        if judgment.exact_best:
            self.exact_best += 1
        if judgment.blunder == BLUNDER_WIN_TO_DRAW:
            self.blunders_win_to_draw += 1
        elif judgment.blunder == BLUNDER_WIN_TO_LOSS:
            self.blunders_win_to_loss += 1
        elif judgment.blunder == BLUNDER_DRAW_TO_LOSS:
            self.blunders_draw_to_loss += 1
        if judgment.forced:
            self.forced += 1

    def _rate(self, count: int) -> float:
        return count / self.positions if self.positions else 0.0

    @property
    def value_optimal_rate(self) -> float:
        return self._rate(self.value_optimal)

    @property
    def exact_best_rate(self) -> float:
        return self._rate(self.exact_best)

    @property
    def blunder_rate(self) -> float:
        blunders = (
            self.blunders_win_to_draw
            + self.blunders_win_to_loss
            + self.blunders_draw_to_loss
        )
        return self._rate(blunders)

    @property
    def forced_rate(self) -> float:
        return self._rate(self.forced)

    def to_dict(self) -> dict:
        return {
            "positions": self.positions,
            "value_optimal": self.value_optimal,
            "exact_best": self.exact_best,
            "blunders_win_to_draw": self.blunders_win_to_draw,
            "blunders_win_to_loss": self.blunders_win_to_loss,
            "blunders_draw_to_loss": self.blunders_draw_to_loss,
            "forced": self.forced,
            "value_optimal_rate": self.value_optimal_rate,
            "exact_best_rate": self.exact_best_rate,
            "blunder_rate": self.blunder_rate,
            "forced_rate": self.forced_rate,
        }


@dataclass
class SolverEvalResults:
    """Full results of a solver evaluation run for one model."""

    env_id: str
    model_name: str
    model_path: str
    step: int | None
    opponent_name: str
    games: int
    seed: int
    temperature: float
    model_wins: int = 0
    model_losses: int = 0
    draws: int = 0
    avg_game_length: float = 0.0
    overall: BucketStats = field(default_factory=BucketStats)
    by_ply: dict[str, BucketStats] = field(
        default_factory=lambda: {b: BucketStats() for b in PLY_BUCKETS}
    )
    by_seat: dict[str, BucketStats] = field(
        default_factory=lambda: {s: BucketStats() for s in SEATS}
    )
    solver_queries: int = 0
    solver_cache_hits: int = 0
    solver_time_seconds: float = 0.0
    wall_time_seconds: float = 0.0
    bitbully_version: str | None = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        cache_hit_rate = (
            self.solver_cache_hits / self.solver_queries if self.solver_queries else 0.0
        )
        return {
            "model": self.model_name,
            "model_path": self.model_path,
            "step": self.step,
            "env_id": self.env_id,
            "opponent": self.opponent_name,
            "games": self.games,
            "seed": self.seed,
            "temperature": self.temperature,
            "model_wins": self.model_wins,
            "model_losses": self.model_losses,
            "draws": self.draws,
            "avg_game_length": self.avg_game_length,
            "positions_scored": self.overall.positions,
            "forced_moves": self.overall.forced,
            "forced_move_rate": self.overall.forced_rate,
            "value_optimal_rate": self.overall.value_optimal_rate,
            "exact_best_rate": self.overall.exact_best_rate,
            "blunder_rate": self.overall.blunder_rate,
            "blunders_win_to_draw": self.overall.blunders_win_to_draw,
            "blunders_win_to_loss": self.overall.blunders_win_to_loss,
            "blunders_draw_to_loss": self.overall.blunders_draw_to_loss,
            "by_ply": {name: stats.to_dict() for name, stats in self.by_ply.items()},
            "by_seat": {name: stats.to_dict() for name, stats in self.by_seat.items()},
            "solver_queries": self.solver_queries,
            "solver_cache_hits": self.solver_cache_hits,
            "solver_cache_hit_rate": cache_hit_rate,
            "solver_time_seconds": self.solver_time_seconds,
            "wall_time_seconds": self.wall_time_seconds,
            "bitbully_version": self.bitbully_version,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        def row(label: str, stats: BucketStats) -> str:
            blunders = (
                f"{stats.blunders_win_to_draw}/{stats.blunders_win_to_loss}"
                f"/{stats.blunders_draw_to_loss}"
            )
            return (
                f"  {label:<12} {stats.positions:>6} "
                f"{stats.value_optimal_rate:>10.1%} {stats.exact_best_rate:>10.1%} "
                f"{blunders:>12} {stats.forced:>7}"
            )

        step_str = str(self.step) if self.step is not None else "-"
        lines = [
            f"{'=' * 72}",
            f"Solver Evaluation: {self.model_name} (step {step_str}) vs {self.opponent_name}",
            f"Game: {self.env_id} | games: {self.games} | seed: {self.seed}",
            f"{'=' * 72}",
            f"Results: {self.model_wins}W / {self.model_losses}L / {self.draws}D "
            f"(avg length {self.avg_game_length:.1f})",
            "",
            f"  {'slice':<12} {'moves':>6} {'value-opt':>10} {'exact-best':>10} "
            f"{'WD/WL/DL':>12} {'forced':>7}",
            row("overall", self.overall),
        ]
        lines += [row(name, stats) for name, stats in self.by_ply.items()]
        lines += [row(name, stats) for name, stats in self.by_seat.items()]
        cache_hit_rate = (
            self.solver_cache_hits / self.solver_queries if self.solver_queries else 0.0
        )
        lines += [
            "",
            f"Solver: {self.solver_queries} queries, {cache_hit_rate:.1%} cache hits, "
            f"{self.solver_time_seconds:.1f}s solving, {self.wall_time_seconds:.1f}s total",
            f"{'=' * 72}",
        ]
        return "\n".join(lines)
