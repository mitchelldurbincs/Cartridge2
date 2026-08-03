"""Synchronized training-loop composition root.

Crucible supplies the loop coordinator. The selected algorithm cartridge
supplies collector, learner, and evaluator factories after compatibility is
validated against the selected environment.

The currently installed ``alphazero_board_v1`` recipe runs this iteration:
1. Allocate a fresh replay collection scope bound to the current checkpoint
2. Run bounded collectors until exactly N complete episodes are sealed
3. Train for M steps from that exact scope
4. Evaluate the candidate against the champion and optional random baseline
5. Publish immutable evidence and advance the sole RunHead, then repeat

Usage:
    python -m trainer --algorithm alphazero_board_v1 loop \
        --iterations 100 --episodes 500 --steps 1000

    # Programmatic usage
    from trainer.orchestrator import Orchestrator, LoopConfig

    config = LoopConfig(iterations=10, episodes_per_iteration=100)
    orchestrator = Orchestrator(config)
    orchestrator.run()
"""

from .config import LoopConfig
from .orchestrator import Orchestrator

__all__ = [
    "Orchestrator",
    "LoopConfig",
]
