"""Shim: implementation moved to the training-core package."""

from typing import TYPE_CHECKING

from training_core.orchestrator.eval_runner import *  # noqa: F401,F403
from training_core.orchestrator.eval_runner import (
    # Redundant alias: EVAL_TEMPERATURE was a public module attribute here
    # before the move but (as before) is deliberately not in __all__.
    EVAL_TEMPERATURE as EVAL_TEMPERATURE,
)
from training_core.orchestrator.eval_runner import (
    EvalRunner as _CoreEvalRunner,
)
from training_core.orchestrator.eval_runner import (
    should_promote,
)

from ..evaluator import OnnxPolicy, RandomPolicy
from ..evaluator import evaluate as run_eval
from ..game_config import get_config as get_game_config
from ..solver_eval import SolverScorer, solver_evaluate
from .config import LoopConfig
from .eval_reporting import EvalReportingMixin

if TYPE_CHECKING:
    from ..wandb_logger import WandbLogger


class _Cartridge2SolverHooks:
    """training-core ``SolverHooks`` backed by this repo's solver_eval.

    Both hooks resolve ``SolverScorer``/``solver_evaluate`` from this
    module's globals at call time, preserving the pre-move test seam
    (monkeypatching those names on this module still takes effect).
    """

    @staticmethod
    def make_scorer():
        return SolverScorer()

    @staticmethod
    def evaluate(**kwargs):
        return solver_evaluate(**kwargs)


class EvalRunner(EvalReportingMixin, _CoreEvalRunner):
    """Cartridge2 EvalRunner: this repo's eval stack wired into the core runner.

    training-core's EvalRunner takes its evaluation backends by constructor
    injection and defaults its solver-stats appender to a no-op; this
    subclass restores pre-move behavior exactly:

    - policies load via ``OnnxPolicy`` / ``RandomPolicy``,
    - head-to-head games run via ``evaluator.evaluate``,
    - game configs come from ``game_config.get_config``,
    - solver scoring uses ``SolverScorer`` + ``solver_evaluate`` (keeping
      the connect4-only gate: with hooks always injected, the core's gate
      reduces to the old env_id/solver_games/disabled condition), and
    - mixing in this repo's ``EvalReportingMixin`` re-pins the real
      ``append_solver_stats`` so solver_stats.json keeps being written.

    The ``(config, wandb_logger)`` construction signature is unchanged for
    orchestrator.py and other callers.
    """

    def __init__(self, config: LoopConfig, wandb_logger: "WandbLogger | None" = None):
        super().__init__(
            config,
            wandb_logger,
            policy_loader=OnnxPolicy,
            baseline_policy_factory=RandomPolicy,
            run_eval=run_eval,
            game_config_getter=get_game_config,
            solver=_Cartridge2SolverHooks(),
        )


__all__ = ["EvalRunner", "should_promote"]
