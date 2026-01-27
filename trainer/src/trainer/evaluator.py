"""Evaluator for measuring trained model performance against baselines.

This module provides evaluation capabilities to measure how well a trained
model plays games compared to random play. It can load game configuration from
the PostgreSQL database (like the Rust actor does), making it self-describing
and supporting multiple games.

Usage (defaults assume running from trainer/ directory):
    python -m trainer.evaluator --model ../data/models/latest.onnx --games 100
    python -m trainer.evaluator --env-id connect4 --games 100
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from .game_config import GameConfig, get_config
from .games import Player, create_game_state
from .logging_utils import silence_noisy_loggers
from .policies import OnnxPolicy, Policy, RandomPolicy
from .storage import GameMetadata, create_replay_buffer

logger = logging.getLogger(__name__)


def get_game_metadata_or_config(env_id: str) -> GameConfig | GameMetadata:
    """Load game configuration from database or fall back to hardcoded config.

    This follows the same pattern as trainer.py, making the evaluator self-describing
    by reading metadata from the PostgreSQL database when available.

    Args:
        env_id: Environment ID (e.g., "tictactoe", "connect4").

    Returns:
        GameConfig or GameMetadata with game configuration.
    """
    try:
        replay = create_replay_buffer()
        try:
            metadata = replay.get_metadata(env_id)
            if metadata:
                logger.info(f"Loaded game metadata from database for {env_id}")
                return metadata
            logger.warning(f"No metadata in database for {env_id}, using fallback")
        finally:
            replay.close()
    except Exception as e:
        logger.warning(f"Failed to read database metadata: {e}, using fallback")

    return get_config(env_id)


@dataclass
class MatchResult:
    """Result of a single game."""

    winner: int | None  # 1=player1, 2=player2, None=draw
    moves: int
    player1_as: Player  # Which color player1 played


@dataclass
class EvalResults:
    """Aggregated evaluation results."""

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
        if self.games_played == 0:
            return 0.0
        return self.player1_wins / self.games_played

    @property
    def player2_win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.player2_wins / self.games_played

    @property
    def draw_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.draws / self.games_played

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            f"{'=' * 50}",
            f"Evaluation Results: {self.player1_name} vs {self.player2_name}",
            f"Game: {self.env_id}",
            f"{'=' * 50}",
            f"Games played: {self.games_played}",
            "",
            f"{self.player1_name}:",
            f"  Wins: {self.player1_wins} ({self.player1_win_rate:.1%})",
            f"    As first player: {self.player1_wins_as_first}",
            f"    As second player: {self.player1_wins_as_second}",
            "",
            f"{self.player2_name}:",
            f"  Wins: {self.player2_wins} ({self.player2_win_rate:.1%})",
            f"    As first player: {self.player2_wins_as_first}",
            f"    As second player: {self.player2_wins_as_second}",
            "",
            f"Draws: {self.draws} ({self.draw_rate:.1%})",
            f"Average game length: {self.avg_game_length:.1f} moves",
            f"{'=' * 50}",
        ]
        return "\n".join(lines)


def play_game(
    player1: Policy,
    player2: Policy,
    player1_as: Player,
    env_id: str,
    config: GameConfig | GameMetadata,
    verbose: bool = False,
) -> MatchResult:
    """Play a single game between two policies.

    Args:
        player1: First policy to evaluate.
        player2: Second policy (opponent).
        player1_as: Which color player1 plays (FIRST or SECOND).
        env_id: Environment ID for creating game state.
        config: Game configuration (from database or fallback).
        verbose: Print game moves.

    Returns:
        MatchResult with winner and game length.
    """
    state = create_game_state(env_id)
    moves = 0

    # Assign policies to player slots
    if player1_as == Player.FIRST:
        first_policy, second_policy = player1, player2
    else:
        first_policy, second_policy = player2, player1

    if verbose:
        role = "first" if player1_as == Player.FIRST else "second"
        logger.info(f"Game start: {player1.name} as {role}")
        logger.info(f"\n{state.display()}")

    while not state.done:
        # Select current player's policy
        policy = first_policy if state.current_player == Player.FIRST else second_policy

        # Get action (pass config for observation encoding)
        action = policy.select_action(state, config)
        state.make_move(action)
        moves += 1

        if verbose:
            logger.info(f"\n{policy.name} plays position {action}")
            logger.info(f"\n{state.display()}")

    # Determine winner from player1's perspective
    if state.winner is None:
        winner = None
    elif state.winner == player1_as:
        winner = 1
    else:
        winner = 2

    if verbose:
        if winner == 1:
            logger.info(f"{player1.name} wins!")
        elif winner == 2:
            logger.info(f"{player2.name} wins!")
        else:
            logger.info("Draw!")

    return MatchResult(winner=winner, moves=moves, player1_as=player1_as)


def evaluate(
    player1: Policy,
    player2: Policy,
    env_id: str,
    config: GameConfig | GameMetadata,
    num_games: int = 100,
    verbose: bool = False,
) -> EvalResults:
    """Run evaluation between two policies.

    Each policy plays half the games as first player and half as second
    to account for first-mover advantage.

    Args:
        player1: Policy to evaluate (typically the trained model).
        player2: Opponent policy (typically random).
        env_id: Environment ID for creating game states.
        config: Game configuration (from database or fallback).
        num_games: Total number of games to play.
        verbose: Print individual game details.

    Returns:
        EvalResults with aggregated statistics.
    """
    results = EvalResults(
        env_id=env_id,
        player1_name=player1.name,
        player2_name=player2.name,
        games_played=0,
        player1_wins=0,
        player2_wins=0,
        draws=0,
        player1_wins_as_first=0,
        player1_wins_as_second=0,
        player2_wins_as_first=0,
        player2_wins_as_second=0,
        avg_game_length=0.0,
    )

    total_moves = 0
    games_per_side = num_games // 2

    # Play half as first player, half as second
    for game_num in range(num_games):
        # Alternate sides
        player1_as = Player.FIRST if game_num < games_per_side else Player.SECOND

        result = play_game(
            player1, player2, player1_as, env_id, config, verbose=verbose
        )

        results.games_played += 1
        total_moves += result.moves

        if result.winner == 1:
            results.player1_wins += 1
            if player1_as == Player.FIRST:
                results.player1_wins_as_first += 1
            else:
                results.player1_wins_as_second += 1
        elif result.winner == 2:
            results.player2_wins += 1
            if player1_as == Player.FIRST:
                results.player2_wins_as_first += 1
            else:
                results.player2_wins_as_second += 1
        else:
            results.draws += 1

        # Progress logging
        if not verbose and (game_num + 1) % 10 == 0:
            logger.info(
                f"Progress: {game_num + 1}/{num_games} games "
                f"({results.player1_name}: {results.player1_wins}W, "
                f"{results.player2_name}: {results.player2_wins}W, "
                f"{results.draws}D)"
            )

    results.avg_game_length = total_moves / max(1, results.games_played)

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
        choices=["tictactoe", "connect4"],
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
        "--verbose",
        action="store_true",
        help="Print individual game moves",
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
    # Load game configuration from database (preferred) or fallback to hardcoded
    config = get_game_metadata_or_config(args.env_id)
    logger.info(
        f"Game config for {args.env_id}: "
        f"board={config.board_width}x{config.board_height}, "
        f"actions={config.num_actions}, obs_size={config.obs_size}"
    )

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    logger.info(f"Loading model: {model_path}")

    try:
        model_policy = OnnxPolicy(str(model_path), temperature=args.temperature)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    random_policy = RandomPolicy()

    logger.info(
        f"Running {args.games} games: {model_policy.name} vs {random_policy.name}"
    )

    # Run evaluation with game configuration
    results = evaluate(
        player1=model_policy,
        player2=random_policy,
        env_id=args.env_id,
        config=config,
        num_games=args.games,
        verbose=args.verbose,
    )

    # Print results
    print(results.summary())

    # Interpretation
    if results.player1_win_rate > 0.7:
        print("\nModel is significantly better than random play!")
    elif results.player1_win_rate > 0.5:
        print("\nModel is slightly better than random play.")
    elif results.player1_win_rate > 0.3:
        print("\nModel is roughly equivalent to random play.")
    else:
        print("\nModel is worse than random play!")

    # Game-specific analysis
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

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    silence_noisy_loggers()

    return run_evaluation(args)


if __name__ == "__main__":
    sys.exit(main())
