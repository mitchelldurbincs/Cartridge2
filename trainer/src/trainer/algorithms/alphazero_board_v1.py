"""Python implementation of the AlphaZero board-game cartridge."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..environment_catalog import (
    ENVIRONMENTS,
    AlgorithmDescriptor,
    CompatibilityReport,
    EnvironmentDescriptor,
    get_algorithm_descriptor,
    get_environment,
)
from ..storage.base import ReplayRecord, ReplaySelection
from .base import AlgorithmCommand

logger = logging.getLogger(__name__)

ALGORITHM_ID = "alphazero_board_v1"
DESCRIPTOR = get_algorithm_descriptor(ALGORITHM_ID)

EXPECTED_COMPONENTS = {
    "collector": "alphazero_mcts_self_play_v1",
    "learner": "alphazero_policy_value_v1",
    "orchestration": "synchronized_alphazero_v1",
    "experience_schema": "alphazero_transition_v1",
    "model_contract": "onnx_policy_value_v1",
    "evaluation_suite": "two_player_seat_balanced_v1",
    "serving": "alphazero_mcts_web_v1",
}
MODEL_ARTIFACT_SCHEMA_VERSION = 1
F32_BYTES = 4
POLICY_SUM_TOLERANCE = 1e-3

# Network architecture is an AlphaZero learner choice, not an environment fact.
NETWORK_OVERRIDES: dict[str, dict[str, Any]] = {
    "tictactoe": {"network_type": "mlp", "hidden_size": 128},
    "connect4": {"network_type": "resnet", "num_res_blocks": 4, "num_filters": 128},
    "othello": {"network_type": "resnet", "num_res_blocks": 6, "num_filters": 256},
    "generals_8x8": {
        "network_type": "resnet",
        "num_res_blocks": 6,
        "num_filters": 128,
    },
}

_LOOP_DESCRIPTION = """\
Run the synchronized AlphaZero recipe:
1. Allocate a fresh replay scope bound to the source checkpoint
2. Generate and seal the configured number of complete self-play episodes
3. Train the policy/value model from that exact scope
4. Evaluate and optionally promote the resulting artifact

Every attempt receives a new scope. Rows from previous or abandoned attempts
remain invisible, and the learner refuses to run unless the exact episode seal
matches the configured quota.
"""


@dataclass(frozen=True)
class AlphaZeroGameConfig:
    env_id: str
    display_name: str
    board_width: int
    board_height: int
    num_actions: int
    obs_size: int
    legal_mask_offset: int
    obs_channels: int
    player_relative_obs: bool
    hidden_size: int = 128
    network_type: str = "mlp"
    num_res_blocks: int = 4
    num_filters: int = 128

    @property
    def board_size(self) -> int:
        return self.board_width * self.board_height

    @property
    def legal_mask_end(self) -> int:
        return self.legal_mask_offset + self.num_actions

    @property
    def player_indicator_offset(self) -> int:
        return self.legal_mask_end

    def extract_legal_mask(self, obs):
        return obs[:, self.legal_mask_offset : self.legal_mask_end]

    def extract_player_indicator(self, obs):
        offset = self.player_indicator_offset
        return obs[:, offset : offset + 2]


def compatibility(env_id: str) -> CompatibilityReport:
    return get_environment(env_id).compatibility(ALGORITHM_ID)


def get_game_config(env_id: str) -> AlphaZeroGameConfig:
    environment = get_environment(env_id)
    environment.compatibility(ALGORITHM_ID).require_compatible()
    board = environment.require_board()
    network = NETWORK_OVERRIDES.get(env_id, {})

    return AlphaZeroGameConfig(
        env_id=environment.env_id,
        display_name=environment.display_name,
        board_width=board.width,
        board_height=board.height,
        num_actions=board.action_count,
        obs_size=board.observation.elements,
        legal_mask_offset=board.observation.legal_actions_offset,
        obs_channels=board.observation.spatial_channels,
        player_relative_obs=board.observation.player_relative,
        **network,
    )


def list_compatible_environments() -> list[str]:
    return sorted(
        env_id
        for env_id in ENVIRONMENTS
        if get_environment(env_id).compatibility(ALGORITHM_ID).compatible
    )


def decode_replay_batch(
    records: list[ReplayRecord],
    *,
    selection: ReplaySelection,
    obs_size: int,
    num_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the exact ``alphazero_transition_v1`` experience payload.

    Storage deliberately treats ``payload`` as bytes. This cartridge owns the
    language-neutral layout: little-endian f32 observation values, followed by
    a policy distribution and one terminal value target.
    """
    expected_schema = DESCRIPTOR.components.experience_schema
    if (
        selection.profile.algorithm_id != ALGORITHM_ID
        or selection.profile.experience_schema != expected_schema
    ):
        raise ValueError(
            "AlphaZero replay decoder requires profile "
            f"({ALGORITHM_ID!r}, {expected_schema!r}), got "
            f"({selection.profile.algorithm_id!r}, "
            f"{selection.profile.experience_schema!r})"
        )
    if obs_size <= 0:
        raise ValueError("obs_size must be positive")
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")

    expected_values = obs_size + num_actions + 1
    expected_bytes = expected_values * F32_BYTES
    observations = np.empty((len(records), obs_size), dtype=np.float32)
    policy_targets = np.empty((len(records), num_actions), dtype=np.float32)
    value_targets = np.empty(len(records), dtype=np.float32)

    for index, record in enumerate(records):
        if not selection.matches(record):
            actual = (
                record.env_id,
                record.env_contract_version,
                record.algorithm_id,
                record.experience_schema,
            )
            expected = (
                selection.profile.env_id,
                selection.profile.env_contract_version,
                selection.profile.algorithm_id,
                selection.profile.experience_schema,
                selection.collection_scope_id,
                selection.source_checkpoint_id,
            )
            actual += (record.collection_scope_id, record.source_checkpoint_id)
            raise ValueError(
                f"Replay record {record.id!r} belongs to selection {actual}, "
                f"expected {expected}"
            )
        if len(record.payload) != expected_bytes:
            raise ValueError(
                f"Replay record {record.id!r} payload is {len(record.payload)} bytes; "
                f"{expected_schema} requires exactly {expected_bytes} bytes "
                f"({expected_values} little-endian f32 values)"
            )

        values = np.frombuffer(record.payload, dtype="<f4")
        invalid = np.flatnonzero(~np.isfinite(values))
        if invalid.size:
            value_index = int(invalid[0])
            raise ValueError(
                f"Replay record {record.id!r} contains non-finite f32 at "
                f"payload index {value_index}: {values[value_index]}"
            )

        observation = values[:obs_size]
        policy = values[obs_size : obs_size + num_actions]
        value = float(values[-1])
        invalid_policy = np.flatnonzero((policy < 0.0) | (policy > 1.0))
        if invalid_policy.size:
            policy_index = int(invalid_policy[0])
            raise ValueError(
                f"Replay record {record.id!r} has policy[{policy_index}]="
                f"{float(policy[policy_index])}, expected a probability in [0, 1]"
            )
        policy_sum = float(policy.sum(dtype=np.float32))
        if abs(policy_sum - 1.0) > POLICY_SUM_TOLERANCE:
            raise ValueError(
                f"Replay record {record.id!r} policy sums to {policy_sum}, expected 1"
            )
        if not -1.0 <= value <= 1.0:
            raise ValueError(
                f"Replay record {record.id!r} value target is {value}, expected [-1, 1]"
            )

        observations[index] = observation
        policy_targets[index] = policy
        value_targets[index] = value

    return observations, policy_targets, value_targets


class AlphaZeroBoardV1:
    """Every Python binding owned by ``alphazero_board_v1``."""

    descriptor: AlgorithmDescriptor = DESCRIPTOR

    def __init__(self) -> None:
        self._validate_descriptor()

    def _validate_descriptor(self) -> None:
        mismatches = []
        for component, expected in EXPECTED_COMPONENTS.items():
            actual = getattr(self.descriptor.components, component)
            if actual != expected:
                mismatches.append(f"{component}={actual!r} (expected {expected!r})")
        if (
            self.descriptor.model_artifact_schema_version
            != MODEL_ARTIFACT_SCHEMA_VERSION
        ):
            mismatches.append(
                "model_artifact_schema_version="
                f"{self.descriptor.model_artifact_schema_version!r} "
                f"(expected {MODEL_ARTIFACT_SCHEMA_VERSION})"
            )
        if mismatches:
            raise RuntimeError(
                f"Python implementation does not match algorithm '{self.descriptor.id}': "
                + "; ".join(mismatches)
            )

    def _require_component(self, name: str) -> None:
        expected = EXPECTED_COMPONENTS[name]
        actual = getattr(self.descriptor.components, name)
        if actual != expected:
            raise RuntimeError(
                f"Algorithm '{self.descriptor.id}' declares {name}={actual!r}; "
                f"this implementation provides {expected!r}"
            )

    def _require_environment(self, env_id: str) -> EnvironmentDescriptor:
        environment = get_environment(env_id)
        self.compatibility(environment).require_compatible()
        return environment

    def _apply_runtime_path_defaults(
        self, args: argparse.Namespace, paths: dict[str, str]
    ) -> None:
        """Fill omitted paths from the parsed environment's profile namespace."""
        from ..central_config import get_config
        from ..runtime_profile import resolve_runtime_profile

        config = get_config()
        profile_dir = resolve_runtime_profile(self.descriptor.id, args.env_id).data_dir(
            config.data_root
        )
        for attribute, relative_path in paths.items():
            if getattr(args, attribute, None) is None:
                setattr(args, attribute, str(profile_dir / relative_path))

    def commands(self) -> tuple[AlgorithmCommand, ...]:
        """CLI surface exported only when this cartridge is selected."""
        return (
            AlgorithmCommand(
                name="train",
                help="Train a policy/value model from replay experience",
                configure_parser=self._configure_train_parser,
                run=self._run_train,
            ),
            AlgorithmCommand(
                name="evaluate",
                help="Evaluate a model against random play",
                configure_parser=self._configure_evaluate_parser,
                run=self._run_evaluate,
            ),
            AlgorithmCommand(
                name="loop",
                help="Run synchronized collection, learning, and evaluation",
                description=_LOOP_DESCRIPTION,
                formatter_class=argparse.RawDescriptionHelpFormatter,
                configure_parser=self._configure_loop_parser,
                run=self._run_loop,
            ),
            AlgorithmCommand(
                name="solver-eval",
                help="Score Connect4 model moves against the perfect solver",
                configure_parser=self._configure_solver_eval_parser,
                run=self._run_solver_eval,
            ),
            AlgorithmCommand(
                name="register-players",
                help="Register ONNX checkpoints as evaluation players",
                configure_parser=self._configure_register_players_parser,
                run=self._run_register_players,
            ),
            AlgorithmCommand(
                name="tournament",
                help="Rate registered players in a round-robin tournament",
                configure_parser=self._configure_tournament_parser,
                run=self._run_tournament,
            ),
        )

    def compatibility(self, environment: EnvironmentDescriptor) -> CompatibilityReport:
        return environment.compatibility(self.descriptor.id)

    def build_learner(self, config: Any):
        # Lazy imports keep registry lookup free of torch and preserve a clean
        # dependency direction from composition roots into implementations.
        from ..trainer import AlphaZeroLearner

        self._require_component("learner")
        self._require_environment(config.env_id)
        return AlphaZeroLearner(config)

    def build_loop_learner(self, spec: Any, loop_config: Any):
        from ..trainer import AlphaZeroLearner

        self._require_component("learner")
        self._require_component("orchestration")
        config = self._build_loop_learner_config(spec, loop_config)
        self._require_environment(config.env_id)
        return AlphaZeroLearner(config)

    def _build_loop_learner_config(self, spec: Any, loop_config: Any):
        """Build the one exact learner config used for hashing and execution."""
        from ..config import AlphaZeroLearnerConfig

        return AlphaZeroLearnerConfig(
            model_dir=spec.model_dir,
            stats_path=spec.stats_path,
            env_id=spec.env_id,
            total_steps=spec.total_steps,
            start_step=spec.start_step,
            batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            weight_decay=loop_config.weight_decay,
            grad_clip_norm=loop_config.grad_clip_norm,
            checkpoint_interval=spec.checkpoint_interval,
            device=spec.device,
            max_wait=spec.max_wait,
            lr_total_steps=spec.lr_total_steps,
            shutdown_check=spec.shutdown_check,
            metrics_hook=spec.metrics_hook,
            defer_run_commit=True,
            # Crucible's generic TrainSpec intentionally has no cartridge-owned
            # replay field; the synchronized composition supplies _LoopTrainSpec.
            replay_selection=getattr(spec, "replay_selection", None),
        )

    def loop_learner_config_sha256(self, spec: Any, loop_config: Any) -> str:
        from ..checkpoint import learner_config_sha256

        self._require_component("learner")
        self._require_component("orchestration")
        config = self._build_loop_learner_config(spec, loop_config)
        self._require_environment(config.env_id)
        return learner_config_sha256(config)

    def loop_learner_recipe(self, spec: Any, loop_config: Any) -> dict:
        from ..checkpoint import learner_config_recipe

        self._require_component("learner")
        self._require_component("orchestration")
        config = self._build_loop_learner_config(spec, loop_config)
        self._require_environment(config.env_id)
        return learner_config_recipe(config)

    def build_collector_runner(
        self, config: Any, shutdown_check: Callable[[], bool] | None = None
    ):
        from ..orchestrator.actor_runner import ActorRunner

        self._require_component("collector")
        self._require_environment(config.env_id)
        return ActorRunner(config, shutdown_check=shutdown_check)

    def build_evaluation_runner(self, config: Any, wandb_logger: Any = None):
        from ..orchestrator.eval_runner import EvalRunner

        self._require_component("evaluation_suite")
        self._require_environment(config.env_id)
        return EvalRunner(config, wandb_logger=wandb_logger)

    def _configure_train_parser(self, parser: argparse.ArgumentParser) -> None:
        from ..central_config import get_config
        from ..config import AlphaZeroLearnerConfig

        cfg = get_config()
        AlphaZeroLearnerConfig.configure_parser(
            parser,
            overrides={
                "checkpoint_interval": cfg.training.checkpoint_interval,
                "batch_size": cfg.training.batch_size,
                "learning_rate": cfg.training.learning_rate,
                "weight_decay": cfg.training.weight_decay,
                "grad_clip_norm": cfg.training.grad_clip_norm,
                "device": cfg.training.device,
                "env_id": cfg.common.env_id,
                # These depend on the command's parsed --env-id, so the runner
                # resolves them after parsing instead of baking in config.toml's
                # environment at parser-construction time.
                "model_dir": argparse.SUPPRESS,
                "stats_path": argparse.SUPPRESS,
            },
        )
        parser.add_argument(
            "--log-level",
            default=cfg.common.log_level.upper(),
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level",
        )
        parser.add_argument(
            "--metrics-port",
            type=int,
            default=9090,
            help="Prometheus metrics server port",
        )
        parser.add_argument(
            "--collection-scope-id",
            required=True,
            help="Exact 64-hex replay collection scope to train from",
        )
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument(
            "--source-checkpoint-id",
            help="Exact 64-hex checkpoint that generated this collection",
        )
        source.add_argument(
            "--source-root",
            action="store_const",
            const=None,
            dest="source_checkpoint_id",
            help="Declare that this collection was generated without a checkpoint",
        )

    def _configure_evaluate_parser(self, parser: argparse.ArgumentParser) -> None:
        from ..evaluator import add_evaluate_arguments

        add_evaluate_arguments(parser)

    def _configure_loop_parser(self, parser: argparse.ArgumentParser) -> None:
        from ..orchestrator.cli import add_loop_arguments

        add_loop_arguments(parser)

    def _configure_solver_eval_parser(self, parser: argparse.ArgumentParser) -> None:
        from ..solver_eval import add_solver_eval_arguments

        add_solver_eval_arguments(parser)

    def _configure_register_players_parser(
        self, parser: argparse.ArgumentParser
    ) -> None:
        from ..tournament_cli import add_register_players_arguments

        add_register_players_arguments(parser)

    def _configure_tournament_parser(self, parser: argparse.ArgumentParser) -> None:
        from ..tournament_cli import add_tournament_arguments

        add_tournament_arguments(parser)

    def _run_train(self, args: argparse.Namespace) -> int:
        from crucible.backoff import WaitTimeout

        from .. import metrics as prom_metrics
        from ..config import AlphaZeroLearnerConfig

        try:
            self._require_component("learner")
            self._require_environment(args.env_id)
            self._apply_runtime_path_defaults(
                args,
                {
                    "model_dir": "models",
                    "stats_path": "stats.json",
                },
            )
            logger.info("Cartridge trainer starting")
            prom_metrics.start_metrics_server(port=args.metrics_port)
            from ..storage.base import ReplayProfile, ReplaySelection

            environment = get_environment(args.env_id)
            learner_config = AlphaZeroLearnerConfig.from_args(args)
            learner_config.replay_selection = ReplaySelection(
                profile=ReplayProfile(
                    env_id=args.env_id,
                    env_contract_version=environment.contract_version,
                    algorithm_id=self.descriptor.id,
                    experience_schema=self.descriptor.components.experience_schema,
                ),
                collection_scope_id=args.collection_scope_id,
                source_checkpoint_id=args.source_checkpoint_id,
            )
            learner = self.build_learner(learner_config)
            stats = learner.train()
            logger.info(f"Training complete; final loss: {stats.total_loss:.4f}")
            logger.info(f"Last checkpoint: {stats.last_checkpoint}")
            return 0
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return 130
        except WaitTimeout as exc:
            logger.error(f"Training timed out: {exc}")
            return 2
        except Exception as exc:
            logger.exception(f"Training failed: {exc}")
            return 1

    def _run_evaluate(self, args: argparse.Namespace) -> int:
        from ..evaluator import run_evaluation

        try:
            self._require_component("evaluation_suite")
            self._require_component("model_contract")
            environment = self._require_environment(args.env_id)
            if getattr(args, "model", None) is None:
                from ..central_config import get_config
                from ..runtime_profile import resolve_runtime_profile
                from ..storage.publisher import (
                    OnnxArtifactContract,
                    create_checkpoint_publisher,
                )

                game = get_game_config(args.env_id)
                profile_dir = resolve_runtime_profile(
                    self.descriptor.id, args.env_id
                ).data_dir(get_config().data_root)
                repository = create_checkpoint_publisher(
                    OnnxArtifactContract(
                        algorithm_id=self.descriptor.id,
                        env_id=args.env_id,
                        env_contract_version=environment.contract_version,
                        model_artifact_schema_version=(
                            self.descriptor.model_artifact_schema_version
                        ),
                        model_contract=self.descriptor.components.model_contract,
                        obs_size=game.obs_size,
                        num_actions=game.num_actions,
                    ),
                    profile_dir / "models",
                )
                current = repository.resolve_head()
                if current is None:
                    raise ValueError(
                        "RunHead is absent; pass an explicit "
                        "immutable --model path or publish a checkpoint"
                    )
                args.model = str(current.onnx_path)
        except (RuntimeError, ValueError) as exc:
            logger.error(str(exc))
            return 1
        return run_evaluation(args)

    def _run_loop(self, args: argparse.Namespace) -> int:
        from ..orchestrator.cli import loop_config_from_args, run_loop

        try:
            self._require_component("collector")
            self._require_component("learner")
            self._require_component("orchestration")
            self._require_component("evaluation_suite")
            self._require_environment(args.env_id)
        except (RuntimeError, ValueError) as exc:
            logger.error(str(exc))
            return 1
        return run_loop(loop_config_from_args(args))

    def _run_solver_eval(self, args: argparse.Namespace) -> int:
        from ..solver_eval import run_solver_evaluation

        try:
            self._require_component("evaluation_suite")
            self._require_environment(args.env_id)
            self._apply_runtime_path_defaults(
                args,
                {
                    "models_dir": "models",
                },
            )
        except (RuntimeError, ValueError) as exc:
            logger.error(str(exc))
            return 1
        return run_solver_evaluation(args)

    def _run_register_players(self, args: argparse.Namespace) -> int:
        from ..tournament_cli import run_register_players

        try:
            self._require_component("evaluation_suite")
            self._require_component("model_contract")
            self._require_environment(args.env_id)
            self._apply_runtime_path_defaults(
                args,
                {
                    "models_dir": "models",
                    "registry": "players.json",
                },
            )
        except (RuntimeError, ValueError) as exc:
            logger.error(str(exc))
            return 1
        return run_register_players(args)

    def _run_tournament(self, args: argparse.Namespace) -> int:
        from ..tournament_cli import run_tournament_command

        try:
            self._require_component("evaluation_suite")
            self._require_environment(args.env_id)
            self._apply_runtime_path_defaults(
                args,
                {
                    "registry": "players.json",
                    "output": "tournament.json",
                },
            )
        except (RuntimeError, ValueError) as exc:
            logger.error(str(exc))
            return 1
        return run_tournament_command(args)
