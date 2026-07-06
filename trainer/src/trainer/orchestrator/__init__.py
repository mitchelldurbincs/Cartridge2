"""Synchronized AlphaZero Training Loop Orchestrator (composition root).

The loop coordinator and its components live in the training-core package
and take every backend through injected seams; this package composes them
with Cartridge2's concrete pieces:

- orchestrator.py binds the factories (storage.create_replay_buffer,
  Trainer/TrainerConfig via TrainSpec, the shim ActorRunner/EvalRunner
  subclasses, structured_logging tracing) into an Orchestrator subclass
  whose ``Orchestrator(config)`` signature keeps cli.py and the console
  entry points untouched.
- config.py, actor_runner.py, eval_runner.py, and eval_reporting.py
  restore this repo's defaults on top of the core classes; promotion.py
  and stats_manager.py re-export unchanged.
- cli.py stays host-side: its argument defaults come from this repo's
  central config, and it calls setup_logging() process-wide before
  constructing the Orchestrator.

The loop implements the classic AlphaZero iteration pattern:
1. Clear replay buffer (start fresh with current model)
2. Run actor for N episodes (self-play data generation)
3. Train for M steps on the generated data
4. Evaluate the new model (gatekeeper vs best, optional random baseline)
5. Export new model, repeat

Usage:
    # As a subcommand
    python -m trainer loop --iterations 100 --episodes 500 --steps 1000

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
