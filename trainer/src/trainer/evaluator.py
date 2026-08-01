"""Model evaluation, played by the engine.

Games are played by the Rust ``cartridge-eval`` binary, the same way self-play
games are played by the actor binary: a subprocess over shared files. This
module launches it and reads back the summary.

**Why not play them here.** It used to. Doing so required a second
implementation of every game's rules in Python (``trainer/games/``) that
nothing kept in sync with the engine — so the promotion gate scored a game that
could silently drift from the one being trained — and it could only cover games
somebody had reimplemented, which is why Othello was never evaluable. Playing
through the engine also lets evaluation use MCTS; the Python evaluator could
only play the raw policy argmax, which understates a model.

Usage (defaults assume running from the trainer/ directory):
    python -m trainer.evaluator --model ../data/models/latest.onnx --games 100
    python -m trainer.evaluator --env-id connect4 --games 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .game_config import get_config
from .logging_utils import silence_noisy_loggers
from .players import ModelPlayer, RandomPlayer

logger = logging.getLogger(__name__)

# Environment variable naming the binary explicitly, mirroring ACTOR_BINARY.
EVAL_BINARY_ENV = "CARTRIDGE_EVAL_BINARY"

# Where to look when the environment does not say. Mirrors the actor's
# auto-detection: the Docker location first, then release before debug.
# trainer/src/trainer/evaluator.py -> repository root
_PROJECT_ROOT = Path(__file__).parents[3]
_BINARY_CANDIDATES = (
    Path("/app/cartridge-eval"),
    _PROJECT_ROOT / "engine" / "target" / "release" / "cartridge-eval",
    _PROJECT_ROOT / "engine" / "target" / "debug" / "cartridge-eval",
)

# Default RNG seed for evaluation runs. Fixed rather than random so a model's
# eval is reproducible and successive iterations face the same conditions.
DEFAULT_SEED = 42


class EvalBinaryNotFound(RuntimeError):
    """The cartridge-eval binary could not be located."""


def find_eval_binary() -> Path:
    """Locate the evaluation binary.

    Order: ``CARTRIDGE_EVAL_BINARY``, then the standard build locations.

    Raises:
        EvalBinaryNotFound: If no binary is found.
    """
    override = os.environ.get(EVAL_BINARY_ENV)
    if override:
        path = Path(override)
        if path.exists():
            return path
        raise EvalBinaryNotFound(
            f"{EVAL_BINARY_ENV} is set to {path}, which does not exist."
        )

    for candidate in _BINARY_CANDIDATES:
        if candidate.exists():
            return candidate

    searched = "\n  ".join(str(c) for c in _BINARY_CANDIDATES)
    raise EvalBinaryNotFound(
        "Could not find the cartridge-eval binary. Build it with\n"
        "  cargo build --release --manifest-path engine/Cargo.toml -p evaluator\n"
        f"or set {EVAL_BINARY_ENV}. Searched:\n  {searched}"
    )


@dataclass
class EvalResults:
    """Aggregated evaluation results.

    Field names match the JSON written by ``cartridge-eval`` (see
    ``engine/evaluator/src/results.rs``); the two are deserialized directly into
    each other, so they must be kept in step.
    """

    env_id: str
    player1_name: str
    player2_name: str
    games_played: int
    player1_wins: int
    player2_wins: int
    draws: int
    player1_wins_as_first: int
    player1_wins_as_second: int
    player2_wins_as_first: int
    player2_wins_as_second: int
    avg_game_length: float

    @property
    def player1_win_rate(self) -> float:
        return self.player1_wins / self.games_played if self.games_played else 0.0

    @property
    def player2_win_rate(self) -> float:
        return self.player2_wins / self.games_played if self.games_played else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games_played if self.games_played else 0.0

    def summary(self) -> str:
        """Human-readable summary block."""
        return (
            f"\n{'=' * 60}\n"
            f"Evaluation Results: {self.env_id}\n"
            f"{'=' * 60}\n"
            f"{self.player1_name} vs {self.player2_name}\n"
            f"Games played: {self.games_played}\n"
            f"  {self.player1_name}: {self.player1_wins} wins "
            f"({self.player1_win_rate:.1%})\n"
            f"  {self.player2_name}: {self.player2_wins} wins "
            f"({self.player2_win_rate:.1%})\n"
            f"  Draws: {self.draws} ({self.draw_rate:.1%})\n"
            f"Average game length: {self.avg_game_length:.1f} moves\n"
            f"{'=' * 60}"
        )


def build_eval_command(
    player1: ModelPlayer | RandomPlayer,
    player2: ModelPlayer | RandomPlayer,
    env_id: str,
    num_games: int,
    seed: int,
    output_path: Path,
    dump_positions: Path | None = None,
) -> list[str]:
    """Assemble the ``cartridge-eval`` argument vector."""
    command = [
        str(find_eval_binary()),
        "--env-id",
        env_id,
        "--games",
        str(num_games),
        "--seed",
        str(seed),
        "--output",
        str(output_path),
        *player1.cli_args("p1"),
        *player2.cli_args("p2"),
    ]
    if dump_positions is not None:
        command += ["--dump-positions", str(dump_positions)]
    return command


def run_eval_binary(command: list[str]) -> None:
    """Run the evaluation binary, raising with its stderr on failure."""
    logger.debug("Running: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"cartridge-eval exited with {result.returncode}: "
            f"{result.stderr.strip() or '(no stderr)'}"
        )


def evaluate(
    player1: ModelPlayer | RandomPlayer,
    player2: ModelPlayer | RandomPlayer,
    env_id: str,
    config: object = None,
    num_games: int = 100,
    verbose: bool = False,
    seed: int = DEFAULT_SEED,
) -> EvalResults:
    """Play ``num_games`` between two players and return the aggregate result.

    Each player takes the first seat for half the games, matching the binary's
    own split, so first-mover advantage cancels.

    Args:
        player1: Player being evaluated (typically the candidate model).
        player2: Opponent (typically the random baseline or the best model).
        env_id: Environment ID.
        config: Accepted and ignored. The engine knows the game; this keyword
            exists because crucible's ``HeadToHeadEvalLike`` seam passes it,
            and removing it there is a separate change.
        num_games: Total games to play.
        verbose: Log the resulting summary block.
        seed: Base RNG seed; game N uses ``seed + N``.

    Returns:
        EvalResults with aggregated statistics.
    """
    del config  # See the docstring: part of the injected seam, not used here.

    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "eval.json"
        run_eval_binary(
            build_eval_command(player1, player2, env_id, num_games, seed, output_path)
        )
        results = EvalResults(**json.loads(output_path.read_text()))

    if verbose:
        logger.info(results.summary())
    return results


def add_evaluate_arguments(
    parser: argparse.ArgumentParser,
    model_default: str = "./data/models/latest.onnx",
) -> None:
    """Add evaluation arguments to a parser.

    This is shared between the standalone evaluator and the 'evaluate' subcommand
    in __main__.py.

    Args:
        parser: ArgumentParser to add arguments to.
        model_default: Default path for the model file.
    """
    parser.add_argument(
        "--model",
        type=str,
        default=model_default,
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="tictactoe",
        help="Game environment to evaluate",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=0,
        help="MCTS simulations per move (0 = play the policy head directly)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base RNG seed; game N uses seed + N",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log the full results summary",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )


def run_evaluation(args: argparse.Namespace) -> int:
    """Run model evaluation with the given arguments.

    This is the core evaluation logic shared between the standalone evaluator
    and the 'evaluate' subcommand in __main__.py.

    Args:
        args: Parsed arguments with model, env_id, games, temperature, verbose.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    config = get_config(args.env_id)
    logger.info(
        f"Game config for {args.env_id}: "
        f"board={config.board_width}x{config.board_height}, "
        f"actions={config.num_actions}, obs_size={config.obs_size}"
    )

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    model = ModelPlayer(
        model_path=str(model_path),
        temperature=args.temperature,
        simulations=args.simulations,
    )
    opponent = RandomPlayer()

    logger.info(f"Running {args.games} games: {model.name} vs {opponent.name}")

    try:
        results = evaluate(
            player1=model,
            player2=opponent,
            env_id=args.env_id,
            num_games=args.games,
            verbose=args.verbose,
            seed=args.seed,
        )
    except (EvalBinaryNotFound, RuntimeError) as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

    print(results.summary())

    if results.player1_win_rate > 0.7:
        print("\nModel is significantly better than random play!")
    elif results.player1_win_rate > 0.5:
        print("\nModel is slightly better than random play.")
    elif results.player1_win_rate > 0.3:
        print("\nModel is roughly equivalent to random play.")
    else:
        print("\nModel is worse than random play!")

    if args.env_id == "tictactoe" and results.draw_rate > 0.8:
        print(
            "\nNote: High draw rate suggests defensive play, which is optimal for TicTacToe."
        )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trained model against random play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_evaluate_arguments(parser, model_default="../data/models/latest.onnx")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    silence_noisy_loggers()

    return run_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())
