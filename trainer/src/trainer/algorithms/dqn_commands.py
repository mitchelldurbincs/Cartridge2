"""DQN command-line parsers and exception-to-exit-code translation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..central_config import get_config
from ..environment_catalog import get_environment
from ..runtime_profile import resolve_runtime_profile
from ..storage import ReplayProfile, ReplaySelection
from .base import AlgorithmCommand
from .dqn_application import format_evaluation_result
from .dqn_config import LEARNER_DEFAULTS
from .dqn_loop import DqnLoop, DqnLoopConfig
from .dqn_requests import (
    COLLECT_DEFAULTS,
    EVALUATE_DEFAULTS,
    LOOP_DEFAULTS,
    DqnCollectRequest,
    DqnEvaluateRequest,
    DqnTrainRequest,
)

if TYPE_CHECKING:
    from .dqn_v1 import DqnV1

logger = logging.getLogger(__name__)


def commands(cartridge: "DqnV1") -> tuple[AlgorithmCommand, ...]:
    return (
        AlgorithmCommand(
            name="collect",
            help="Collect epsilon-greedy DQN transitions",
            configure_parser=configure_collect_parser,
            run=lambda args: run_collect(cartridge, args),
        ),
        AlgorithmCommand(
            name="train",
            help="Train a Q-network from an exact DQN replay collection",
            configure_parser=configure_train_parser,
            run=lambda args: run_train(cartridge, args),
        ),
        AlgorithmCommand(
            name="evaluate",
            help="Evaluate a greedy Q-policy by single-agent episode return",
            configure_parser=configure_evaluate_parser,
            run=lambda args: run_evaluate(cartridge, args),
        ),
        AlgorithmCommand(
            name="loop",
            help="Run bounded off-policy collect/train/evaluate iterations",
            configure_parser=configure_loop_parser,
            run=lambda args: run_loop(cartridge, args),
        ),
    )


def configure_collect_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-id", default=COLLECT_DEFAULTS.env_id)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--collection-scope-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-checkpoint-id")
    source.add_argument("--source-root", action="store_true")
    parser.add_argument("--epsilon", type=float, default=COLLECT_DEFAULTS.epsilon)
    parser.add_argument("--seed", type=int, default=COLLECT_DEFAULTS.seed)
    parser.add_argument(
        "--onnx-intra-threads", type=int, default=COLLECT_DEFAULTS.onnx_intra_threads
    )
    parser.add_argument("--actor-id", default=COLLECT_DEFAULTS.actor_id)
    parser.add_argument(
        "--episode-timeout-secs",
        type=int,
        default=COLLECT_DEFAULTS.episode_timeout_secs,
    )
    parser.add_argument("--actor-binary")
    parser.add_argument("--data-dir")
    parser.add_argument("--log-level", default=COLLECT_DEFAULTS.log_level)


def configure_train_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-id", default=COLLECT_DEFAULTS.env_id)
    parser.add_argument("--model-dir")
    parser.add_argument("--stats-path")
    parser.add_argument("--steps", type=int, default=LEARNER_DEFAULTS.total_steps)
    parser.add_argument("--batch-size", type=int, default=LEARNER_DEFAULTS.batch_size)
    parser.add_argument("--learning-rate", type=float, default=LEARNER_DEFAULTS.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=LEARNER_DEFAULTS.weight_decay)
    parser.add_argument("--gamma", type=float, default=LEARNER_DEFAULTS.gamma)
    parser.add_argument(
        "--target-sync-interval",
        type=int,
        default=LEARNER_DEFAULTS.target_sync_interval,
    )
    parser.add_argument("--hidden-size", type=int, default=LEARNER_DEFAULTS.hidden_size)
    parser.add_argument("--grad-clip", type=float, default=LEARNER_DEFAULTS.grad_clip_norm)
    parser.add_argument("--device", default=LEARNER_DEFAULTS.device)
    parser.add_argument("--collection-scope-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-checkpoint-id")
    source.add_argument("--source-root", action="store_true")
    parser.add_argument("--log-level", default=COLLECT_DEFAULTS.log_level)


def configure_evaluate_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-id", default=EVALUATE_DEFAULTS.env_id)
    parser.add_argument("--episodes", type=int, default=EVALUATE_DEFAULTS.episodes)
    parser.add_argument("--seed", type=int, default=EVALUATE_DEFAULTS.seed)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-id")
    source.add_argument("--random", action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--eval-binary")
    parser.add_argument(
        "--onnx-intra-threads", type=int, default=EVALUATE_DEFAULTS.onnx_intra_threads
    )


def configure_loop_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-id", default=LOOP_DEFAULTS.env_id)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--episodes-per-iteration",
        type=int,
        default=LOOP_DEFAULTS.episodes_per_iteration,
    )
    parser.add_argument("--steps-per-iteration", type=int, default=LEARNER_DEFAULTS.total_steps)
    parser.add_argument("--batch-size", type=int, default=LEARNER_DEFAULTS.batch_size)
    parser.add_argument("--learning-rate", type=float, default=LEARNER_DEFAULTS.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=LEARNER_DEFAULTS.weight_decay)
    parser.add_argument("--gamma", type=float, default=LEARNER_DEFAULTS.gamma)
    parser.add_argument(
        "--target-sync-interval",
        type=int,
        default=LEARNER_DEFAULTS.target_sync_interval,
    )
    parser.add_argument("--hidden-size", type=int, default=LEARNER_DEFAULTS.hidden_size)
    parser.add_argument("--grad-clip", type=float, default=LEARNER_DEFAULTS.grad_clip_norm)
    parser.add_argument("--device", default=LEARNER_DEFAULTS.device)
    parser.add_argument("--epsilon-start", type=float, default=LOOP_DEFAULTS.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=LOOP_DEFAULTS.epsilon_end)
    parser.add_argument("--epsilon-decay", type=float, default=LOOP_DEFAULTS.epsilon_decay)
    parser.add_argument("--seed", type=int, default=LOOP_DEFAULTS.seed)
    parser.add_argument("--onnx-intra-threads", type=int, default=LOOP_DEFAULTS.onnx_intra_threads)
    parser.add_argument(
        "--evaluation-episodes", type=int, default=LOOP_DEFAULTS.evaluation_episodes
    )
    parser.add_argument(
        "--episode-timeout-secs", type=int, default=LOOP_DEFAULTS.episode_timeout_secs
    )
    parser.add_argument("--actor-binary")
    parser.add_argument("--eval-binary")
    parser.add_argument("--data-dir")
    parser.add_argument("--log-level", default=LOOP_DEFAULTS.log_level)


def run_collect(cartridge: "DqnV1", args: argparse.Namespace) -> int:
    return _with_exit_code(
        "DQN collection failed", lambda: cartridge.collect(_collect_request(args))
    )


def run_train(cartridge: "DqnV1", args: argparse.Namespace) -> int:
    def train() -> int:
        cartridge.train(_train_request(cartridge, args))
        return 0

    return _with_exit_code("DQN training failed", train)


def run_evaluate(cartridge: "DqnV1", args: argparse.Namespace) -> int:
    def evaluate() -> int:
        print(format_evaluation_result(cartridge.evaluate(_evaluate_request(cartridge, args))))
        return 0

    return _with_exit_code("DQN evaluation failed", evaluate)


def run_loop(cartridge: "DqnV1", args: argparse.Namespace) -> int:
    return _with_exit_code("DQN loop failed", lambda: DqnLoop(cartridge, _loop_config(args)).run())


def _collect_request(args: argparse.Namespace) -> DqnCollectRequest:
    data_root = Path(args.data_dir) if args.data_dir else get_config().data_root
    return DqnCollectRequest(
        env_id=args.env_id,
        episodes=args.episodes,
        collection_scope_id=args.collection_scope_id,
        source_checkpoint_id=args.source_checkpoint_id,
        epsilon=args.epsilon,
        seed=args.seed,
        onnx_intra_threads=args.onnx_intra_threads,
        actor_id=args.actor_id,
        episode_timeout_secs=args.episode_timeout_secs,
        actor_binary=Path(args.actor_binary) if args.actor_binary else None,
        data_root=data_root,
        log_level=args.log_level,
    )


def _train_request(cartridge: "DqnV1", args: argparse.Namespace) -> DqnTrainRequest:
    environment = get_environment(args.env_id)
    profile_dir = resolve_runtime_profile(cartridge.descriptor.id, environment.env_id).data_dir(
        get_config().data_root
    )
    selection = _replay_selection(cartridge, environment, args)
    return DqnTrainRequest(
        env_id=environment.env_id,
        replay_selection=selection,
        model_dir=Path(args.model_dir) if args.model_dir else profile_dir / "models",
        stats_path=(Path(args.stats_path) if args.stats_path else profile_dir / "stats.json"),
        total_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        target_sync_interval=args.target_sync_interval,
        hidden_size=args.hidden_size,
        grad_clip_norm=args.grad_clip,
        device=args.device,
        log_level=args.log_level,
    )


def _evaluate_request(cartridge: "DqnV1", args: argparse.Namespace) -> DqnEvaluateRequest:
    environment = get_environment(args.env_id)
    profile_dir = resolve_runtime_profile(cartridge.descriptor.id, environment.env_id).data_dir(
        get_config().data_root
    )
    return DqnEvaluateRequest(
        env_id=environment.env_id,
        episodes=args.episodes,
        seed=args.seed,
        checkpoint_id=args.checkpoint_id,
        model_dir=Path(args.model_dir) if args.model_dir else profile_dir / "models",
        eval_binary=Path(args.eval_binary) if args.eval_binary else None,
        onnx_intra_threads=args.onnx_intra_threads,
    )


def _loop_config(args: argparse.Namespace) -> DqnLoopConfig:
    return DqnLoopConfig(
        env_id=args.env_id,
        iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        steps_per_iteration=args.steps_per_iteration,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        target_sync_interval=args.target_sync_interval,
        hidden_size=args.hidden_size,
        grad_clip_norm=args.grad_clip,
        device=args.device,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        onnx_intra_threads=args.onnx_intra_threads,
        evaluation_episodes=args.evaluation_episodes,
        episode_timeout_secs=args.episode_timeout_secs,
        actor_binary=Path(args.actor_binary) if args.actor_binary else None,
        eval_binary=Path(args.eval_binary) if args.eval_binary else None,
        data_root=Path(args.data_dir) if args.data_dir else get_config().data_root,
        log_level=args.log_level,
    )


def _replay_selection(cartridge: "DqnV1", environment, args) -> ReplaySelection:
    return ReplaySelection(
        profile=ReplayProfile(
            env_id=environment.env_id,
            env_contract_version=environment.contract_version,
            algorithm_id=cartridge.descriptor.id,
            experience_schema=cartridge.descriptor.components.experience_schema,
        ),
        collection_scope_id=args.collection_scope_id,
        source_checkpoint_id=args.source_checkpoint_id,
    )


def _with_exit_code(message: str, operation: Callable[[], int]) -> int:
    try:
        return operation()
    except Exception:
        logger.exception(message)
        return 1
