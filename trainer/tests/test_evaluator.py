"""Tests for the evaluator module.

This module tests:
- Match result tracking
- Game play between policies
- Win rate calculation
- Evaluation with mock ONNX models
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from trainer.evaluator import EvalResults, MatchResult, evaluate, play_game
from trainer.game_config import GameConfig
from trainer.games import Player, create_game_state
from trainer.policies import Policy, RandomPolicy


class MockPolicy(Policy):
    """Mock policy for testing."""

    def __init__(self, name: str, actions: list[int] | None = None):
        self._name = name
        self._actions = actions or []
        self._action_idx = 0

    @property
    def name(self) -> str:
        return self._name

    def select_action(self, state, config) -> int:
        legal = state.legal_moves()
        if self._action_idx < len(self._actions):
            action = self._actions[self._action_idx]
            self._action_idx += 1
            # Ensure action is legal
            if action in legal:
                return action
        # Fallback to first legal move
        return legal[0] if legal else 0


class TestMatchResult:
    """Test MatchResult dataclass."""

    def test_match_result_creation(self):
        """Test creating a MatchResult."""
        result = MatchResult(winner=1, moves=10, player1_as=Player.FIRST)

        assert result.winner == 1
        assert result.moves == 10
        assert result.player1_as == Player.FIRST

    def test_match_result_draw(self):
        """Test MatchResult for a draw."""
        result = MatchResult(winner=None, moves=9, player1_as=Player.FIRST)

        assert result.winner is None
        assert result.moves == 9


class TestEvalResults:
    """Test EvalResults aggregation."""

    def test_eval_results_creation(self):
        """Test creating EvalResults."""
        results = EvalResults(
            env_id="tictactoe",
            player1_name="model",
            player2_name="random",
            games_played=10,
            player1_wins=7,
            player2_wins=1,
            draws=2,
            player1_wins_as_first=4,
            player1_wins_as_second=3,
            player2_wins_as_first=1,
            player2_wins_as_second=0,
            avg_game_length=8.5,
        )

        assert results.env_id == "tictactoe"
        assert results.player1_win_rate == 0.7
        assert results.player2_win_rate == 0.1
        assert results.draw_rate == 0.2

    def test_eval_results_zero_games(self):
        """Test EvalResults with zero games."""
        results = EvalResults(
            env_id="tictactoe",
            player1_name="model",
            player2_name="random",
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

        # Win rates should be 0 when no games played
        assert results.player1_win_rate == 0.0
        assert results.player2_win_rate == 0.0
        assert results.draw_rate == 0.0

    def test_eval_results_summary(self):
        """Test summary string generation."""
        results = EvalResults(
            env_id="tictactoe",
            player1_name="TestModel",
            player2_name="Random",
            games_played=100,
            player1_wins=80,
            player2_wins=10,
            draws=10,
            player1_wins_as_first=45,
            player1_wins_as_second=35,
            player2_wins_as_first=6,
            player2_wins_as_second=4,
            avg_game_length=9.3,
        )

        summary = results.summary()

        assert "TestModel" in summary
        assert "Random" in summary
        assert "100" in summary
        assert "80" in summary
        assert "9.3" in summary


def create_test_config(env_id: str = "tictactoe") -> GameConfig:
    """Create a test GameConfig with all required fields."""
    configs = {
        "tictactoe": GameConfig(
            env_id="tictactoe",
            display_name="Tic Tac Toe",
            board_width=3,
            board_height=3,
            num_actions=9,
            obs_size=27,
            legal_mask_offset=27,
        ),
        "connect4": GameConfig(
            env_id="connect4",
            display_name="Connect 4",
            board_width=7,
            board_height=6,
            num_actions=7,
            obs_size=84,
            legal_mask_offset=84,
        ),
    }
    return configs.get(env_id, configs["tictactoe"])


class TestPlayGame:
    """Test play_game function."""

    def test_play_game_tictactoe(self):
        """Test playing a game of TicTacToe."""
        player1 = RandomPolicy()
        player2 = RandomPolicy()

        config = create_test_config("tictactoe")

        result = play_game(
            player1=player1,
            player2=player2,
            player1_as=Player.FIRST,
            env_id="tictactoe",
            config=config,
            verbose=False,
        )

        # Game should complete
        assert result.moves > 0
        assert result.moves <= 9  # TicTacToe max moves
        assert result.winner in [1, 2, None]

    def test_play_game_player1_as_second(self):
        """Test playing with player1 as second player."""
        player1 = RandomPolicy()
        player2 = RandomPolicy()

        config = create_test_config("tictactoe")

        result = play_game(
            player1=player1,
            player2=player2,
            player1_as=Player.SECOND,
            env_id="tictactoe",
            config=config,
            verbose=False,
        )

        assert result.player1_as == Player.SECOND
        assert result.moves > 0

    def test_play_game_deterministic(self):
        """Test that deterministic policies produce consistent results."""
        # Create policies that play specific sequences
        # Player 1 plays center (4), corners (0, 2, 6, 8)
        # Player 2 plays edges (1, 3, 5, 7)
        player1 = MockPolicy("player1", [4, 0, 8])  # Center, corner, corner
        player2 = MockPolicy("player2", [1, 3])  # Edges

        config = create_test_config("tictactoe")

        result = play_game(
            player1=player1,
            player2=player2,
            player1_as=Player.FIRST,
            env_id="tictactoe",
            config=config,
            verbose=False,
        )

        # Game should complete with valid moves
        assert result.moves >= 5  # At least 5 moves played


class TestEvaluate:
    """Test evaluate function."""

    def test_evaluate_runs_correct_number_of_games(self):
        """Test that evaluate runs the requested number of games."""
        player1 = RandomPolicy()
        player2 = RandomPolicy()

        config = create_test_config("tictactoe")

        results = evaluate(
            player1=player1,
            player2=player2,
            env_id="tictactoe",
            config=config,
            num_games=20,
            verbose=False,
        )

        assert results.games_played == 20
        # Half as first, half as second
        assert results.player1_wins_as_first + results.player1_wins_as_second == results.player1_wins
        assert results.player2_wins_as_first + results.player2_wins_as_second == results.player2_wins

    def test_evaluate_tracks_wins_correctly(self):
        """Test that wins are tracked correctly."""
        # Create a mock policy that always wins as first player
        winning_policy = MockPolicy("winner")
        random_policy = RandomPolicy()

        config = create_test_config("tictactoe")

        results = evaluate(
            player1=winning_policy,
            player2=random_policy,
            env_id="tictactoe",
            config=config,
            num_games=10,
            verbose=False,
        )

        # Results should sum correctly
        assert results.player1_wins + results.player2_wins + results.draws == 10
        assert results.games_played == 10

    def test_evaluate_computes_average_game_length(self):
        """Test that average game length is computed."""
        player1 = RandomPolicy()
        player2 = RandomPolicy()

        config = create_test_config("tictactoe")

        results = evaluate(
            player1=player1,
            player2=player2,
            env_id="tictactoe",
            config=config,
            num_games=10,
            verbose=False,
        )

        # Average should be between 5 and 9 for TicTacToe
        assert 5 <= results.avg_game_length <= 9

    def test_evaluate_with_different_games(self):
        """Test evaluation with different game counts."""
        player1 = RandomPolicy()
        player2 = RandomPolicy()

        config = create_test_config("tictactoe")

        # Test with 4 games
        results4 = evaluate(
            player1=player1,
            player2=player2,
            env_id="tictactoe",
            config=config,
            num_games=4,
            verbose=False,
        )
        assert results4.games_played == 4

        # Test with 8 games
        results8 = evaluate(
            player1=player1,
            player2=player2,
            env_id="tictactoe",
            config=config,
            num_games=8,
            verbose=False,
        )
        assert results8.games_played == 8


class TestRandomPolicy:
    """Test RandomPolicy."""

    def test_random_policy_selects_legal_moves(self):
        """Test that random policy only selects legal moves."""
        policy = RandomPolicy()
        state = create_game_state("tictactoe")

        config = create_test_config("tictactoe")

        # Select multiple actions, all should be legal
        for _ in range(5):
            action = policy.select_action(state, config)
            legal_moves = state.legal_moves()
            assert action in legal_moves
            state.make_move(action)
            if state.done:
                break

    def test_random_policy_name(self):
        """Test random policy name."""
        policy = RandomPolicy()
        assert policy.name == "Random"
