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

Usage:
    python -m trainer --algorithm alphazero_board_v1 evaluate --games 100
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .algorithms import get_algorithm
from .environment_catalog import get_environment
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
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1
_MAX_F32 = float.fromhex("0x1.fffffep+127")


def _positive_u32(value: object, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_U32
    ):
        raise ValueError(f"{field} must be a positive u32 integer")
    return value


def _u32(value: object, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_U32
    ):
        raise ValueError(f"{field} must be a nonnegative u32 integer")
    return value


def _u64(value: object, *, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_U64
    ):
        raise ValueError(f"{field} must be a nonnegative u64 integer")
    return value


def _canonical_f32(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite nonnegative f32")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or normalized > _MAX_F32:
        raise ValueError(f"{field} must be a finite nonnegative f32")
    narrowed = float(struct.unpack("!f", struct.pack("!f", normalized))[0])
    return 0.0 if narrowed == 0.0 else narrowed


def _validate_evaluation_schedule(num_games: object, seed: object) -> tuple[int, int]:
    games = _positive_u32(num_games, field="num_games")
    base_seed = _u64(seed, field="seed")
    if base_seed > _MAX_U64 - (games - 1):
        raise ValueError("seed plus the game index exceeds u64")
    return games, base_seed


def _positive_u32_argument(value: str) -> int:
    try:
        return _positive_u32(int(value), field="--games")
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected a positive u32 integer") from exc


def _u32_argument(value: str) -> int:
    try:
        return _u32(int(value), field="--simulations")
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected a nonnegative u32 integer") from exc


def _u64_argument(value: str) -> int:
    try:
        return _u64(int(value), field="--seed")
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected a nonnegative u64 integer") from exc


def _f32_argument(value: str) -> float:
    try:
        return _canonical_f32(float(value), field="--temperature")
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("expected a finite nonnegative f32") from exc


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

    def __post_init__(self) -> None:
        for field in ("env_id", "player1_name", "player2_name"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a nonempty string")
        _positive_u32(self.games_played, field="games_played")
        for field in (
            "player1_wins",
            "player2_wins",
            "draws",
            "player1_wins_as_first",
            "player1_wins_as_second",
            "player2_wins_as_first",
            "player2_wins_as_second",
        ):
            _u32(getattr(self, field), field=field)
        if self.player1_wins + self.player2_wins + self.draws != self.games_played:
            raise ValueError("evaluation outcome counts must sum to games_played")
        if (
            self.player1_wins_as_first + self.player1_wins_as_second
            != self.player1_wins
        ):
            raise ValueError("player1 seat-win counts must sum to player1_wins")
        if (
            self.player2_wins_as_first + self.player2_wins_as_second
            != self.player2_wins
        ):
            raise ValueError("player2 seat-win counts must sum to player2_wins")
        if (
            isinstance(self.avg_game_length, bool)
            or not isinstance(self.avg_game_length, (int, float))
            or not math.isfinite(float(self.avg_game_length))
            or self.avg_game_length < 0.0
        ):
            raise ValueError("avg_game_length must be finite and nonnegative")
        normalized = float(self.avg_game_length)
        self.avg_game_length = 0.0 if normalized == 0.0 else normalized

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
    algorithm_id: str,
    env_id: str,
    num_games: int,
    seed: int,
    output_path: Path,
    dump_positions: Path | None = None,
) -> list[str]:
    """Assemble the ``cartridge-eval`` argument vector."""
    num_games, seed = _validate_evaluation_schedule(num_games, seed)
    command = [
        str(find_eval_binary()),
        "--algorithm",
        algorithm_id,
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
    algorithm_id: str,
    env_id: str,
    num_games: int = 100,
    verbose: bool = False,
    seed: int = DEFAULT_SEED,
) -> EvalResults:
    """Play ``num_games`` between two players and return the aggregate result.

    Player 1 takes the first seat on even game indices and the second seat on
    odd indices, matching the binary's deterministic alternating schedule.

    Args:
        player1: Player being evaluated (typically the candidate model).
        player2: Opponent (typically the random baseline or the champion).
        algorithm_id: Algorithm cartridge ID.
        env_id: Environment ID.
        num_games: Total games to play.
        verbose: Log the resulting summary block.
        seed: Base RNG seed; game N uses ``seed + N``.

    Returns:
        EvalResults with aggregated statistics.
    """
    num_games, seed = _validate_evaluation_schedule(num_games, seed)
    algorithm = get_algorithm(algorithm_id)
    algorithm.compatibility(get_environment(env_id)).require_compatible()

    with tempfile.TemporaryDirectory() as tmp:
        output_path = Path(tmp) / "eval.json"
        run_eval_binary(
            build_eval_command(
                player1,
                player2,
                algorithm_id,
                env_id,
                num_games,
                seed,
                output_path,
            )
        )
        results = EvalResults(**json.loads(output_path.read_text()))

    if verbose:
        logger.info(results.summary())
    return results


def add_evaluate_arguments(
    parser: argparse.ArgumentParser,
    model_default: str | None = None,
) -> None:
    """Add evaluation arguments to a parser.

    This is shared between the standalone evaluator and the 'evaluate' subcommand
    in __main__.py.

    Args:
        parser: ArgumentParser to add arguments to.
        model_default: Default path for the model file. The canonical
            cartridge binding leaves this unset and resolves it from the
            parsed environment's runtime profile.
    """
    from .central_config import get_config

    central_config = get_config()
    parser.add_argument(
        "--model",
        type=str,
        default=argparse.SUPPRESS if model_default is None else model_default,
        help="Path to ONNX model file (default: selected runtime profile)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=central_config.common.env_id,
        help="Game environment to evaluate",
    )
    parser.add_argument(
        "--games",
        type=_positive_u32_argument,
        default=100,
        help="Number of games to play",
    )
    parser.add_argument(
        "--temperature",
        type=_f32_argument,
        default=0.0,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--simulations",
        type=_u32_argument,
        default=0,
        help="MCTS simulations per move (0 = play the policy head directly)",
    )
    parser.add_argument(
        "--seed",
        type=_u64_argument,
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
    try:
        args.games, args.seed = _validate_evaluation_schedule(args.games, args.seed)
        args.simulations = _u32(args.simulations, field="simulations")
        args.temperature = _canonical_f32(args.temperature, field="temperature")
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    algorithm = get_algorithm(args.algorithm)
    environment = get_environment(args.env_id)
    algorithm.compatibility(environment).require_compatible()
    board = environment.require_board()
    logger.info(
        f"Environment {args.env_id} with {args.algorithm}: "
        f"board={board.width}x{board.height}, "
        f"actions={board.action_count}, obs_size={board.observation.elements}"
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
            algorithm_id=args.algorithm,
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
