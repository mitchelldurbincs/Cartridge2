"""Synchronized AlphaZero Training Loop Orchestrator.

This package implements the classic AlphaZero iteration pattern:
1. Clear replay buffer (start fresh with current model)
2. Run actor for N episodes (self-play data generation)
3. Train for M steps on the generated data
4. Evaluate the new model (gatekeeper vs best, optional random baseline)
5. Export new model, repeat

Usage:
    # As a module
    python -m trainer.orchestrator --iterations 100 --episodes 500 --steps 1000

    # Via console entry point
    trainer-loop --iterations 100 --episodes 500 --steps 1000

    # Programmatic usage
    from trainer.orchestrator import Orchestrator, LoopConfig

    config = LoopConfig(iterations=10, episodes_per_iteration=100)
    orchestrator = Orchestrator(config)
    orchestrator.run()
"""

# Re-export components for advanced usage
from .actor_runner import ActorRunner
from .cli import main, parse_args
from .config import IterationStats, LoopConfig
from .eval_runner import EvalRunner
from .orchestrator import Orchestrator
from .stats_manager import StatsManager

__all__ = [
    # Main API
    "Orchestrator",
    "LoopConfig",
    "IterationStats",
    "main",
    "parse_args",
    # Components (for advanced usage)
    "ActorRunner",
    "EvalRunner",
    "StatsManager",
]
