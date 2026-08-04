"""Parser and runtime helpers for the synchronized loop command.

The installed algorithm cartridge chooses whether to expose this recipe and
owns the binding from the canonical trainer CLI to these helpers.  This module
does not parse process arguments or provide a standalone entrypoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from ..central_config import get_config as get_central_config
from .config import LoopConfig

logger = logging.getLogger(__name__)


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def add_loop_arguments(parser: argparse.ArgumentParser) -> None:
    """Install synchronized-loop arguments using central-config defaults."""
    cfg = get_central_config()

    parser.add_argument(
        "--iterations",
        type=int,
        default=cfg.training.iterations,
        help="Global target iteration count for this run",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=cfg.training.episodes_per_iteration,
        help="Episodes per iteration",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=cfg.training.steps_per_iteration,
        help="Training steps per iteration",
    )
    parser.add_argument(
        "--env-id",
        default=cfg.common.env_id,
        help="Environment ID",
    )
    parser.add_argument(
        "--data-dir",
        default=cfg.common.data_dir,
        help="Data directory",
    )
    parser.add_argument(
        "--actor-binary",
        default=None,
        help="Path to the collector executable",
    )
    parser.add_argument(
        "--actor-log-interval",
        type=int,
        default=cfg.actor.log_interval,
        help="Collector log interval in episodes",
    )
    parser.add_argument(
        "--actor-episode-timeout-seconds",
        type=int,
        default=cfg.actor.episode_timeout_secs,
        help="Hard timeout for one self-play episode",
    )
    parser.add_argument(
        "--actor-eval-batch-size",
        type=int,
        default=cfg.mcts.eval_batch_size,
        help="Neural-evaluation batch size used inside collector MCTS",
    )
    parser.add_argument(
        "--actor-onnx-intra-threads",
        type=int,
        default=cfg.mcts.onnx_intra_threads,
        help="ONNX intra-op threads used by each collector",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=cfg.training.num_actors,
        help="Number of parallel collector processes",
    )
    parser.add_argument(
        "--mcts-start-sims",
        type=int,
        default=cfg.mcts.start_sims,
        help="MCTS simulations for the first iteration",
    )
    parser.add_argument(
        "--mcts-max-sims",
        type=int,
        default=cfg.mcts.max_sims,
        help="Maximum MCTS simulations after ramping",
    )
    parser.add_argument(
        "--mcts-sim-ramp-rate",
        type=int,
        default=cfg.mcts.sim_ramp_rate,
        help="MCTS simulations added per iteration",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=cfg.mcts.c_puct,
        help="Collector MCTS exploration constant",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=cfg.mcts.temperature,
        help="Collector action-selection temperature",
    )
    parser.add_argument(
        "--late-temperature",
        type=float,
        default=cfg.mcts.late_temperature,
        help="Collector temperature at or after the move threshold",
    )
    parser.add_argument(
        "--temp-threshold",
        type=int,
        default=cfg.mcts.temp_threshold,
        help="Move number at which to reduce temperature (0 disables it)",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=cfg.mcts.dirichlet_alpha,
        help="Collector MCTS root-noise concentration",
    )
    parser.add_argument(
        "--dirichlet-weight",
        type=float,
        default=cfg.mcts.dirichlet_weight,
        help="Collector MCTS root-noise mixture weight",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=cfg.training.batch_size,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=cfg.training.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=cfg.training.weight_decay,
        help="Optimizer weight decay",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=cfg.training.grad_clip_norm,
        help="Gradient clipping max norm (0 disables it)",
    )
    parser.add_argument(
        "--device",
        default=cfg.training.device,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=cfg.evaluation.interval,
        help="Evaluate every N iterations (0 disables it)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=cfg.evaluation.games,
        help="Games per evaluation",
    )
    parser.add_argument(
        "--eval-win-threshold",
        type=float,
        default=cfg.evaluation.win_threshold,
        help="Win rate required to promote a model",
    )
    parser.add_argument(
        "--eval-vs-random",
        type=_parse_bool,
        default=cfg.evaluation.eval_vs_random,
        help="Also evaluate against a random baseline",
    )
    parser.add_argument(
        "--eval-simulations",
        type=int,
        default=cfg.evaluation.simulations,
        help="MCTS simulations per evaluation move (0 uses the policy head)",
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=cfg.evaluation.temperature,
        help="Action-selection temperature for evaluation games",
    )
    parser.add_argument(
        "--solver-games",
        type=int,
        default=cfg.evaluation.solver_games,
        help="Perfect-solver games per evaluation (0 disables them)",
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=cfg.evaluation.evaluation_seed,
        help="Fixed seed for head-to-head, random, and solver evaluation",
    )
    parser.add_argument(
        "--promotion-metric",
        default=cfg.evaluation.promotion_metric,
        choices=["win_rate", "solver_optimal"],
        help="Champion promotion criterion",
    )
    parser.add_argument(
        "--promotion-margin",
        type=float,
        default=cfg.evaluation.promotion_margin,
        help="Required solver-optimal improvement for promotion",
    )
    parser.add_argument(
        "--wandb-enabled",
        type=_parse_bool,
        default=cfg.wandb.enabled,
        help="Log this loop run to Weights & Biases",
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
        help="Prometheus metrics server port for the synchronized trainer",
    )


def loop_config_from_args(args: argparse.Namespace) -> LoopConfig:
    """Convert the selected cartridge's parsed command into a loop request."""
    cfg = get_central_config()
    return LoopConfig(
        iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        steps_per_iteration=args.steps,
        env_id=args.env_id,
        algorithm_id=args.algorithm,
        data_dir=Path(args.data_dir),
        actor_binary=Path(args.actor_binary) if args.actor_binary else None,
        actor_log_interval=args.actor_log_interval,
        actor_episode_timeout_seconds=args.actor_episode_timeout_seconds,
        actor_eval_batch_size=args.actor_eval_batch_size,
        actor_onnx_intra_threads=args.actor_onnx_intra_threads,
        num_actors=args.num_actors,
        mcts_start_sims=args.mcts_start_sims,
        mcts_max_sims=args.mcts_max_sims,
        mcts_sim_ramp_rate=args.mcts_sim_ramp_rate,
        c_puct=args.c_puct,
        temperature=args.temperature,
        late_temperature=args.late_temperature,
        temp_threshold=args.temp_threshold,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_weight=args.dirichlet_weight,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        eval_simulations=args.eval_simulations,
        eval_temperature=args.eval_temperature,
        eval_win_threshold=args.eval_win_threshold,
        eval_vs_random=args.eval_vs_random,
        solver_games=args.solver_games,
        evaluation_seed=args.evaluation_seed,
        promotion_metric=args.promotion_metric,
        promotion_margin=args.promotion_margin,
        wandb=dataclasses.replace(cfg.wandb, enabled=args.wandb_enabled),
        device=args.device,
        log_level=args.log_level,
        metrics_port=args.metrics_port,
    )


def run_loop(config: LoopConfig) -> int:
    """Run one already-parsed loop request."""
    from .. import metrics as prom_metrics
    from .orchestrator import Orchestrator

    try:
        prom_metrics.start_metrics_server(port=config.metrics_port)
        Orchestrator(config).run()
        return 0
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.exception(f"Training failed: {exc}")
        return 1
