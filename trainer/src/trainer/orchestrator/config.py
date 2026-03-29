"""Configuration dataclasses for the orchestrator.

This module contains the configuration dataclasses used by the orchestrator
and its components. These are intentionally in a separate module to avoid
circular import issues.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class IterationStats:
    """Statistics for a single training iteration."""

    iteration: int
    episodes_generated: int
    transitions_generated: int
    training_steps: int
    actor_time_seconds: float
    trainer_time_seconds: float
    eval_time_seconds: float
    total_time_seconds: float
    eval_win_rate: float | None = None
    eval_draw_rate: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LoopConfig:
    """Configuration for the synchronized training loop."""

    # Iteration settings
    iterations: int = 100
    start_iteration: int = 1
    episodes_per_iteration: int = 500
    steps_per_iteration: int = 1000

    # Environment
    env_id: str = "tictactoe"

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    actor_binary: Path | None = None  # Auto-detect or use ACTOR_BINARY env var

    # Actor settings
    actor_log_interval: int = 200
    num_actors: int = 1  # Number of parallel actor processes

    # MCTS simulation ramping: start_sims + (iteration-1) * sim_ramp_rate, capped at max_sims
    mcts_start_sims: int = 50  # Simulations for first iteration
    mcts_max_sims: int = 400  # Maximum simulations (reached after ramping)
    mcts_sim_ramp_rate: int = 20  # Simulations to add per iteration

    # Temperature schedule: after temp_threshold moves, reduce temperature for exploitation
    # Set to 0 to disable (always use temp=1.0)
    # Recommended: ~60% of typical game length (tictactoe=5, connect4=20, othello=30)
    temp_threshold: int = 0

    # Trainer settings
    batch_size: int = 64
    learning_rate: float = 1e-3
    checkpoint_interval: int = 100
    device: str = "cpu"

    # Evaluation settings - enabled by default
    eval_interval: int = 1  # Run evaluation every N iterations (0 to disable)
    eval_games: int = 50  # Games per evaluation
    eval_win_threshold: float = 0.55  # Win rate needed to become new best
    eval_vs_random: bool = True  # Also evaluate against random baseline

    # Logging
    log_level: str = "INFO"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def stats_path(self) -> Path:
        return self.data_dir / "stats.json"

    @property
    def loop_stats_path(self) -> Path:
        return self.data_dir / "loop_stats.json"

    @property
    def eval_stats_path(self) -> Path:
        return self.data_dir / "eval_stats.json"

    @property
    def best_model_path(self) -> Path:
        return self.models_dir / "best.onnx"

    @property
    def best_model_info_path(self) -> Path:
        return self.data_dir / "best_model.json"

    def resolve_device(self) -> str:
        """Resolve 'auto' device to the best available: cuda > mps > cpu."""
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
        """Calculate MCTS simulations for given iteration (ramping schedule).

        Starts at mcts_start_sims and increases by mcts_sim_ramp_rate per iteration,
        capped at mcts_max_sims.
        """
        sims = self.mcts_start_sims + (iteration - 1) * self.mcts_sim_ramp_rate
        return min(sims, self.mcts_max_sims)
