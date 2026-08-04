"""Configuration for Cartridge2's synchronized AlphaZero run recipe."""

from __future__ import annotations

import math
import struct
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path

from ..central_config import WandbConfig
from ..environment_catalog import get_environment
from ..runtime_profile import resolve_runtime_profile

_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_MAX_F32 = float.fromhex("0x1.fffffep+127")


def _u32(value: object, *, field_name: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > _MAX_U32:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name} must be a {qualifier} u32 integer")
    return value


def _u64(value: object, *, field_name: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > _MAX_U64:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name} must be a {qualifier} u64 integer")
    return value


def _nonnegative_f32(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite nonnegative f32")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= _MAX_F32:
        raise ValueError(f"{field_name} must be a finite nonnegative f32")
    narrowed = float(struct.unpack("!f", struct.pack("!f", normalized))[0])
    return 0.0 if narrowed == 0.0 else narrowed


def _finite_number(
    value: object,
    *,
    field_name: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name} must be finite and {qualifier}")
    normalized = float(value)
    valid = normalized > 0.0 if positive else normalized >= 0.0
    if not math.isfinite(normalized) or not valid:
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{field_name} must be finite and {qualifier}")
    return 0.0 if normalized == 0.0 else normalized


def _validate_simulation_schedule(*, iterations: int, start: int, maximum: int, ramp: int) -> None:
    if start == maximum:
        if ramp != 0:
            raise ValueError(
                "mcts_sim_ramp_rate must be zero when mcts_start_sims equals mcts_max_sims"
            )
        return
    delta = maximum - start
    if ramp == 0 or ramp > delta:
        raise ValueError(
            "a ramped MCTS schedule requires mcts_sim_ramp_rate in "
            "[1, mcts_max_sims - mcts_start_sims]"
        )
    steps_to_cap = (delta + ramp - 1) // ramp
    if iterations - 1 < steps_to_cap:
        raise ValueError("MCTS simulation schedule must reach mcts_max_sims within iterations")


@dataclass
class LoopConfig:
    """One immutable run recipe plus a process-local execution request.

    ``iterations`` is the run's global target, not a number of additional
    iterations to execute.  The next iteration is derived exclusively from
    the authoritative RunHead; callers cannot select it.
    """

    iterations: int = 100
    episodes_per_iteration: int = 500
    steps_per_iteration: int = 1000

    env_id: str = "tictactoe"
    algorithm_id: str = "alphazero_board_v1"

    data_dir: Path = field(default_factory=lambda: Path("./data"))
    actor_binary: Path | None = None

    actor_log_interval: int = 200
    actor_episode_timeout_seconds: int = 30
    actor_eval_batch_size: int = 32
    actor_onnx_intra_threads: int = 1
    num_actors: int = 1
    mcts_start_sims: int = 50
    mcts_max_sims: int = 400
    mcts_sim_ramp_rate: int = 20
    c_puct: float = 1.4
    temperature: float = 1.0
    late_temperature: float = 1.0
    temp_threshold: int = 0
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25

    batch_size: int = 64
    learning_rate: float = 1e-3
    device: str = "cpu"
    weight_decay: float = 0.0001
    grad_clip_norm: float = 1.0

    eval_interval: int = 1
    eval_games: int = 50
    eval_win_threshold: float = 0.55
    eval_vs_random: bool = True
    eval_temperature: float = 0.2
    eval_simulations: int = 0
    solver_games: int = 0
    evaluation_seed: int = 42
    promotion_metric: str = "win_rate"
    promotion_margin: float = 0.0

    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_level: str = "INFO"
    metrics_port: int = 9090

    # Crucible's generic loop currently reads this property.  Cartridge2 owns
    # its value and never accepts it as input.
    _next_iteration: int = field(default=1, init=False, repr=False)
    _sealed: bool = field(default=False, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False) and name != "_next_iteration":
            raise FrozenInstanceError(f"cannot mutate authenticated LoopConfig field {name!r}")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        iterations = _u64(self.iterations, field_name="iterations", positive=True)
        episodes = _u32(
            self.episodes_per_iteration,
            field_name="episodes_per_iteration",
            positive=True,
        )
        steps = _u64(
            self.steps_per_iteration,
            field_name="steps_per_iteration",
            positive=True,
        )
        if iterations > _MAX_U64 // steps:
            raise ValueError("iterations * steps_per_iteration exceeds u64")
        actors = _u32(self.num_actors, field_name="num_actors", positive=True)
        if actors > episodes:
            raise ValueError("num_actors cannot exceed episodes_per_iteration")
        if not isinstance(self.env_id, str) or not self.env_id.strip():
            raise ValueError("env_id must be a nonempty string")
        if not isinstance(self.algorithm_id, str) or not self.algorithm_id.strip():
            raise ValueError("algorithm_id must be a nonempty string")
        resolve_runtime_profile(self.algorithm_id, self.env_id)
        max_horizon = get_environment(self.env_id).capabilities.max_horizon
        if max_horizon is None:
            raise ValueError("synchronized AlphaZero requires a finite max_horizon")
        if not isinstance(self.data_dir, Path):
            raise TypeError("data_dir must be a pathlib.Path")
        if self.actor_binary is not None and not isinstance(self.actor_binary, Path):
            raise TypeError("actor_binary must be a pathlib.Path or None")
        _u32(self.actor_log_interval, field_name="actor_log_interval")
        _u64(
            self.actor_episode_timeout_seconds,
            field_name="actor_episode_timeout_seconds",
            positive=True,
        )
        _u32(
            self.actor_eval_batch_size,
            field_name="actor_eval_batch_size",
            positive=True,
        )
        _u32(
            self.actor_onnx_intra_threads,
            field_name="actor_onnx_intra_threads",
            positive=True,
        )
        start = _u32(self.mcts_start_sims, field_name="mcts_start_sims", positive=True)
        maximum = _u32(self.mcts_max_sims, field_name="mcts_max_sims", positive=True)
        _u32(self.mcts_sim_ramp_rate, field_name="mcts_sim_ramp_rate")
        _u32(self.temp_threshold, field_name="temp_threshold")
        if start > maximum:
            raise ValueError("mcts_start_sims cannot exceed mcts_max_sims")
        _validate_simulation_schedule(
            iterations=iterations,
            start=start,
            maximum=maximum,
            ramp=self.mcts_sim_ramp_rate,
        )
        for field_name in (
            "c_puct",
            "temperature",
            "late_temperature",
            "dirichlet_alpha",
            "dirichlet_weight",
            "eval_temperature",
        ):
            setattr(
                self,
                field_name,
                _nonnegative_f32(getattr(self, field_name), field_name=field_name),
            )
        if self.dirichlet_weight > 1.0:
            raise ValueError("dirichlet_weight must be a rate in [0, 1]")
        if (self.dirichlet_alpha == 0.0) != (self.dirichlet_weight == 0.0):
            raise ValueError(
                "dirichlet_alpha and dirichlet_weight must both be zero to disable noise"
            )
        if self.temp_threshold == 0:
            if self.late_temperature != self.temperature:
                raise ValueError(
                    "late_temperature must equal temperature when temp_threshold is zero"
                )
        else:
            if self.late_temperature == self.temperature:
                raise ValueError(
                    "late_temperature must differ from temperature when the schedule is enabled"
                )
            if self.temp_threshold >= max_horizon:
                raise ValueError(
                    "temp_threshold must be less than the selected environment max_horizon"
                )
        _u64(self.batch_size, field_name="batch_size", positive=True)
        for field_name, positive in (
            ("learning_rate", True),
            ("weight_decay", False),
            ("grad_clip_norm", False),
        ):
            setattr(
                self,
                field_name,
                _finite_number(
                    getattr(self, field_name),
                    field_name=field_name,
                    positive=positive,
                ),
            )
        eval_interval = _u64(self.eval_interval, field_name="eval_interval")
        games = _u32(self.eval_games, field_name="eval_games", positive=True)
        if not isinstance(self.eval_vs_random, bool):
            raise ValueError("eval_vs_random must be boolean")
        _u32(self.eval_simulations, field_name="eval_simulations")
        solver_games = _u32(self.solver_games, field_name="solver_games")
        seed = _u64(self.evaluation_seed, field_name="evaluation_seed")
        largest_run = max(games, solver_games)
        if seed > _MAX_U64 - (largest_run - 1):
            raise ValueError("evaluation seed schedule exceeds u64")
        for field_name in ("eval_win_threshold", "promotion_margin"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{field_name} must be a finite rate in [0, 1]")
        if self.promotion_metric == "win_rate":
            if self.promotion_margin != 0.0:
                raise ValueError("promotion_margin must be zero when promotion_metric is win_rate")
        elif self.promotion_metric == "solver_optimal":
            if self.eval_win_threshold != 0.0:
                raise ValueError(
                    "eval_win_threshold must be zero when promotion_metric is solver_optimal"
                )
        else:
            raise ValueError("promotion_metric must be win_rate or solver_optimal")
        if self.solver_games > 0 and self.env_id != "connect4":
            raise ValueError("solver_games may be nonzero only for connect4")
        if self.promotion_metric == "solver_optimal" and (
            self.env_id != "connect4" or self.solver_games == 0
        ):
            raise ValueError("solver_optimal promotion requires connect4 with solver_games > 0")
        solver_evidence_enabled = self.env_id == "connect4" and solver_games > 0
        if eval_interval > 0 and not self.eval_vs_random and not solver_evidence_enabled:
            raise ValueError(
                "scheduled evaluation requires first-candidate evidence: enable "
                "eval_vs_random or use connect4 with solver_games > 0"
            )
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level must be DEBUG, INFO, WARNING, or ERROR")
        if not isinstance(self.wandb, WandbConfig):
            raise TypeError("wandb must be a WandbConfig")
        if not isinstance(self.wandb.enabled, bool) or not isinstance(self.wandb.required, bool):
            raise ValueError("wandb enabled/required flags must be boolean")
        _finite_number(
            self.wandb.init_timeout_seconds,
            field_name="wandb.init_timeout_seconds",
            positive=True,
        )
        if (
            isinstance(self.metrics_port, bool)
            or not isinstance(self.metrics_port, int)
            or not 1 <= self.metrics_port <= 65535
        ):
            raise ValueError("metrics_port must be an integer in [1, 65535]")
        object.__setattr__(self, "_sealed", True)

    @property
    def start_iteration(self) -> int:
        return self._next_iteration

    def _set_start_iteration(self, value: int) -> None:
        """Advance the process-local cursor without changing run semantics."""
        _u64(value, field_name="next iteration", positive=True)
        object.__setattr__(self, "_next_iteration", value)

    @property
    def runtime_profile(self):
        return resolve_runtime_profile(self.algorithm_id, self.env_id)

    @property
    def profile_dir(self) -> Path:
        return self.runtime_profile.data_dir(self.data_dir)

    @property
    def models_dir(self) -> Path:
        return self.profile_dir / "models"

    @property
    def stats_path(self) -> Path:
        return self.profile_dir / "stats.json"

    @property
    def loop_stats_path(self) -> Path:
        return self.profile_dir / "loop_stats.json"

    @property
    def eval_stats_path(self) -> Path:
        return self.profile_dir / "eval_stats.json"

    @property
    def solver_stats_path(self) -> Path:
        return self.profile_dir / "solver_stats.json"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def get_num_simulations(self, iteration: int) -> int:
        _u64(iteration, field_name="iteration", positive=True)
        if self.mcts_sim_ramp_rate == 0:
            return self.mcts_start_sims
        remaining = self.mcts_max_sims - self.mcts_start_sims
        steps_to_cap = (remaining + self.mcts_sim_ramp_rate - 1) // self.mcts_sim_ramp_rate
        if iteration - 1 >= steps_to_cap:
            return self.mcts_max_sims
        return self.mcts_start_sims + (iteration - 1) * self.mcts_sim_ramp_rate


__all__ = ["LoopConfig"]
