"""Shim: implementation moved to the training-core package."""

from training_core.orchestrator.eval_reporting import *  # noqa: F401,F403
from training_core.orchestrator.eval_reporting import (
    EvalReportingMixin as _CoreEvalReportingMixin,
)

from ..solver_eval import append_solver_stats


class EvalReportingMixin(_CoreEvalReportingMixin):
    """Cartridge2 mixin: solver history persists via this repo's solver_eval.

    training-core's mixin defaults its solver-stats appender to a no-op;
    pinning ``append_solver_stats`` here keeps ``EvalRunner`` behavior
    byte-equivalent to before the move.
    """

    _solver_stats_appender = staticmethod(append_solver_stats)


__all__ = ["EvalReportingMixin"]
