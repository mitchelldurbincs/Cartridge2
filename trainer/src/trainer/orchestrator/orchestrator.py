"""Shim: implementation moved to the crucible package.

This module is also Cartridge2's COMPOSITION ROOT for the training loop.
crucible's Orchestrator takes every backend as a required keyword-only
factory seam; the subclass below pre-binds this repo's concrete pieces so
the ``Orchestrator(config)`` construction signature -- and therefore
cli.py's ``Orchestrator(config).run()``, ``python -m trainer loop``, and
the ``trainer-loop`` console script -- stays byte-identical. A small
wrapper class was chosen over ``functools.partial`` so
``from .orchestrator import Orchestrator`` keeps importing a real class
from this module (cli.py and test imports stay untouched) and
isinstance/type checks keep working.

Factory bindings (pre-move behavior, verbatim):

- ``replay_buffer_factory``: this repo's ``storage.create_replay_buffer``,
  resolved from this module's globals at construction time (pre-move test
  seam preserved: monkeypatching that name here still takes effect).
- ``trainer_factory``: ``_make_trainer`` converts the core ``TrainSpec``
  into a real ``TrainerConfig`` field-for-field and builds a ``Trainer``;
  the import stays lazy (as pre-move) and resolves at call time, so
  monkeypatching ``trainer.trainer.Trainer`` stubs the loop's trainers.
- ``actor_runner_factory`` / ``eval_runner_factory``: the shim
  ``ActorRunner`` / ``EvalRunner`` subclasses from this package (binary
  candidates, eval backends, solver hooks, and the solver-stats appender
  pre-wired), resolved from this module's globals at construction time.
- ``trace_starter``: ``_start_iteration_trace`` reproduces the pre-move
  ``generate_trace_id`` + ``set_trace_context`` pair from this repo's
  structured_logging.
"""

from crucible.orchestrator.orchestrator import *  # noqa: F401,F403
from crucible.orchestrator.orchestrator import (
    Orchestrator as _CoreOrchestrator,
)
from crucible.orchestrator.orchestrator import (
    TrainSpec,
)

from ..storage import create_replay_buffer
from ..structured_logging import (
    generate_span_id,
    generate_trace_id,
    set_trace_context,
)
from .actor_runner import ActorRunner
from .config import LoopConfig
from .eval_runner import EvalRunner


def _make_trainer(spec: TrainSpec):
    """Build this repo's Trainer for one iteration's TrainSpec."""
    # Import locally to avoid circular imports
    from ..trainer import Trainer, TrainerConfig

    config = TrainerConfig(
        model_dir=spec.model_dir,
        stats_path=spec.stats_path,
        env_id=spec.env_id,
        total_steps=spec.total_steps,
        start_step=spec.start_step,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
        checkpoint_interval=spec.checkpoint_interval,
        device=spec.device,
        max_wait=spec.max_wait,
        eval_interval=spec.eval_interval,
        lr_total_steps=spec.lr_total_steps,
        shutdown_check=spec.shutdown_check,
        metrics_hook=spec.metrics_hook,
    )
    return Trainer(config)


def _start_iteration_trace() -> str:
    """Begin one iteration's trace context (pre-move behavior, verbatim)."""
    trace_id = generate_trace_id()
    set_trace_context(trace_id=trace_id, span_id=generate_span_id())
    return trace_id


class Orchestrator(_CoreOrchestrator):
    """Cartridge2 Orchestrator: the core loop with this repo's backends bound.

    See the module docstring for the factory bindings. The ``(config)``
    construction signature is unchanged for cli.py and other callers.
    """

    def __init__(self, config: LoopConfig):
        super().__init__(
            config,
            replay_buffer_factory=create_replay_buffer,
            trainer_factory=_make_trainer,
            actor_runner_factory=ActorRunner,
            eval_runner_factory=EvalRunner,
            trace_starter=_start_iteration_trace,
        )


__all__ = ["Orchestrator", "TrainSpec"]
