"""Language-local implementation interface for algorithm cartridges.

The shared trainer executable knows how to select a cartridge and install its
commands.  It deliberately knows nothing about the commands an individual
cartridge exposes or the arguments those commands accept.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from ..storage.publisher import OnnxArtifactContract

from ..environment_catalog import (
    AlgorithmDescriptor,
    CompatibilityReport,
    EnvironmentDescriptor,
)


class Learner(Protocol):
    def train(self) -> Any: ...


class CollectorRunner(Protocol):
    def run(self, *args, **kwargs) -> Any: ...


class EvaluationRunner(Protocol):
    def run(self, *args, **kwargs) -> Any: ...


@dataclass(frozen=True)
class AlgorithmCommand:
    """One CLI command exported by an installed algorithm cartridge."""

    name: str
    help: str
    configure_parser: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]
    description: str | None = None
    formatter_class: type[argparse.HelpFormatter] = argparse.ArgumentDefaultsHelpFormatter


class Algorithm(Protocol):
    """The only capabilities every installed cartridge must provide."""

    descriptor: AlgorithmDescriptor

    def commands(self) -> tuple[AlgorithmCommand, ...]: ...

    def compatibility(self, environment: EnvironmentDescriptor) -> CompatibilityReport: ...

    def artifact_contract(self, environment: EnvironmentDescriptor) -> "OnnxArtifactContract": ...


class SynchronizedLoopAlgorithm(Algorithm, Protocol):
    """Composition hooks required by the synchronized AlphaZero loop recipe."""

    def build_learner(self, config: Any) -> Learner: ...

    def build_loop_learner(self, spec: Any, loop_config: Any) -> Learner: ...

    def loop_learner_config_sha256(self, spec: Any, loop_config: Any) -> str: ...

    def loop_learner_recipe(self, spec: Any, loop_config: Any) -> dict: ...

    def build_collector_runner(
        self, config: Any, shutdown_check: Callable[[], bool] | None = None
    ) -> CollectorRunner: ...

    def collector_config(self, config: Any, num_simulations: int) -> dict: ...

    def build_evaluation_runner(
        self, config: Any, wandb_logger: Any = None
    ) -> EvaluationRunner: ...
