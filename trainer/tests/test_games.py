"""Tests for the pure Python game implementations in trainer.games.

test_evaluator.py exercises TicTacToe indirectly through play_game; this
module tests the game states directly. Connect4State previously had no
direct coverage in this suite.
"""

import pytest

from trainer.game_config import get_config
from trainer.games import Player, create_game_state
from trainer.games.connect4 import Connect4State
from trainer.games.tictactoe import TicTacToeState


def _c4_index(col: int, row: int) -> int:
    """Board index for (col, row) in Connect4's column-major layout."""
    return col * Connect4State.HEIGHT + row


def _c4_state(pieces: dict[tuple[int, int], Player], to_move: Player) -> Connect4State:
    """Build a Connect4 position from {(col, row): player} placements."""
    board = [0] * (Connect4State.WIDTH * Connect4State.HEIGHT)
    for (col, row), player in pieces.items():
        board[_c4_index(col, row)] = player
    return Connect4State(board=board, current_player=to_move)


class TestCreateGameState:
    def test_creates_tictactoe(self):
        state = create_game_state("tictactoe")
        assert isinstance(state, TicTacToeState)
        assert state.current_player == Player.FIRST
        assert not state.done

    def test_creates_connect4(self):
        state = create_game_state("connect4")
        assert isinstance(state, Connect4State)
        assert state.current_player == Player.FIRST
        assert not state.done

    def test_unknown_game_raises(self):
        with pytest.raises(ValueError, match="Unsupported game"):
            create_game_state("chess")


class TestConnect4Moves:
    def test_new_state_all_columns_legal(self):
        state = Connect4State.new()
        assert state.legal_moves() == list(range(7))
        assert state.legal_moves_mask() == [1.0] * 7

    def test_pieces_stack_from_bottom(self):
        state = Connect4State.new()
        state.make_move(3)
        state.make_move(3)
        assert state.board[_c4_index(3, 0)] == Player.FIRST
        assert state.board[_c4_index(3, 1)] == Player.SECOND
        assert state.current_player == Player.FIRST

    def test_full_column_becomes_illegal(self):
        state = Connect4State.new()
        # Alternating pieces never make 4-in-a-row vertically
        for _ in range(3):
            state.make_move(0)
            state.make_move(0)

        assert 0 not in state.legal_moves()
        assert state.legal_moves_mask()[0] == 0.0
        with pytest.raises(ValueError, match="full"):
            state.make_move(0)

    def test_move_after_game_over_raises(self):
        state = Connect4State.new()
        # First player wins vertically in column 0
        for _ in range(3):
            state.make_move(0)
            state.make_move(1)
        state.make_move(0)

        assert state.done
        with pytest.raises(ValueError, match="already over"):
            state.make_move(2)

    def test_no_legal_moves_when_done(self):
        state = Connect4State.new()
        for _ in range(3):
            state.make_move(0)
            state.make_move(1)
        state.make_move(0)

        assert state.legal_moves() == []
        assert state.legal_moves_mask() == [0.0] * 7

    def test_copy_is_independent(self):
        state = Connect4State.new()
        state.make_move(2)
        clone = state.copy()
        clone.make_move(4)

        assert state.board[_c4_index(4, 0)] == 0
        assert clone.board[_c4_index(4, 0)] == Player.SECOND


class TestConnect4WinDetection:
    def test_vertical_win(self):
        state = Connect4State.new()
        for _ in range(3):
            state.make_move(0)  # First
            state.make_move(1)  # Second
        state.make_move(0)  # First's 4th piece in column 0

        assert state.done
        assert state.winner == Player.FIRST

    def test_horizontal_win(self):
        state = Connect4State.new()
        for col in range(3):
            state.make_move(col)  # First: columns 0, 1, 2
            state.make_move(6)  # Second: stacks column 6
        state.make_move(3)  # First completes 0-3 on the bottom row

        assert state.done
        assert state.winner == Player.FIRST

    def test_second_player_can_win(self):
        state = Connect4State.new()
        state.make_move(6)  # First plays elsewhere
        for _ in range(2):
            state.make_move(0)  # Second
            state.make_move(6)  # First (column 6 stays below 4)
        state.make_move(0)
        state.make_move(5)
        state.make_move(0)  # Second's 4th piece in column 0

        assert state.done
        assert state.winner == Player.SECOND

    def test_diagonal_up_right_win(self):
        # First has (0,0), (1,1), (2,2); dropping in column 3 lands on
        # (3,3) and completes the diagonal.
        pieces = {
            (0, 0): Player.FIRST,
            (1, 0): Player.SECOND,
            (1, 1): Player.FIRST,
            (2, 0): Player.SECOND,
            (2, 1): Player.SECOND,
            (2, 2): Player.FIRST,
            (3, 0): Player.FIRST,
            (3, 1): Player.SECOND,
            (3, 2): Player.SECOND,
        }
        state = _c4_state(pieces, to_move=Player.FIRST)
        state.make_move(3)

        assert state.done
        assert state.winner == Player.FIRST

    def test_diagonal_up_left_win(self):
        # Mirror image: First has (6,0), (5,1), (4,2); dropping in
        # column 3 lands on (3,3) and completes the anti-diagonal.
        pieces = {
            (6, 0): Player.FIRST,
            (5, 0): Player.SECOND,
            (5, 1): Player.FIRST,
            (4, 0): Player.SECOND,
            (4, 1): Player.SECOND,
            (4, 2): Player.FIRST,
            (3, 0): Player.FIRST,
            (3, 1): Player.SECOND,
            (3, 2): Player.SECOND,
        }
        state = _c4_state(pieces, to_move=Player.FIRST)
        state.make_move(3)

        assert state.done
        assert state.winner == Player.FIRST


class TestConnect4Draw:
    def test_full_board_without_winner_is_draw(self):
        # Column pattern chosen so no four-in-a-row exists anywhere:
        # columns 0-2 alternate F,S bottom-up; columns 3-5 alternate S,F;
        # column 6 alternates F,S but is left one short.
        pieces = {}
        for col in range(7):
            height = 5 if col == 6 else 6
            for row in range(height):
                if 3 <= col <= 5:
                    player = Player.SECOND if row % 2 == 0 else Player.FIRST
                else:
                    player = Player.FIRST if row % 2 == 0 else Player.SECOND
                pieces[(col, row)] = player

        state = _c4_state(pieces, to_move=Player.SECOND)
        assert state.legal_moves() == [6]

        state.make_move(6)  # Fills the last cell without making a line

        assert state.done
        assert state.winner is None


class TestConnect4Observation:
    def test_observation_encodes_planes_mask_and_player(self):
        config = get_config("connect4")
        board_size = config.board_width * config.board_height

        state = Connect4State.new()
        state.make_move(3)  # First at (3, 0)
        obs = state.to_observation(config)

        assert obs.shape == (config.obs_size,)
        # First player's plane has exactly the dropped piece
        assert obs[_c4_index(3, 0)] == 1.0
        assert obs[:board_size].sum() == 1.0
        # Second player's plane is still empty
        assert obs[board_size : 2 * board_size].sum() == 0.0
        # All columns legal
        mask = obs[config.legal_mask_offset : config.legal_mask_offset + 7]
        assert mask.tolist() == [1.0] * 7
        # Second player to move
        player_offset = config.legal_mask_offset + config.num_actions
        assert obs[player_offset] == 0.0
        assert obs[player_offset + 1] == 1.0

    def test_observation_mask_reflects_full_column(self):
        config = get_config("connect4")

        state = Connect4State.new()
        for _ in range(3):
            state.make_move(0)
            state.make_move(0)
        obs = state.to_observation(config)

        mask = obs[config.legal_mask_offset : config.legal_mask_offset + 7]
        assert mask.tolist() == [0.0] + [1.0] * 6


class TestTicTacToeState:
    def test_row_win(self):
        state = TicTacToeState.new()
        for move in [0, 3, 1, 4, 2]:  # X takes the top row
            state.make_move(move)

        assert state.done
        assert state.winner == Player.FIRST
        assert state.legal_moves() == []

    def test_column_win(self):
        state = TicTacToeState.new()
        for move in [0, 1, 3, 2, 6]:  # X takes the left column
            state.make_move(move)

        assert state.done
        assert state.winner == Player.FIRST

    def test_second_player_diagonal_win(self):
        state = TicTacToeState.new()
        for move in [1, 0, 3, 4, 7, 8]:  # O takes the 0-4-8 diagonal
            state.make_move(move)

        assert state.done
        assert state.winner == Player.SECOND

    def test_draw(self):
        state = TicTacToeState.new()
        for move in [0, 1, 2, 4, 3, 5, 7, 6, 8]:
            state.make_move(move)

        assert state.done
        assert state.winner is None

    def test_occupied_cell_raises(self):
        state = TicTacToeState.new()
        state.make_move(4)
        with pytest.raises(ValueError, match="not empty"):
            state.make_move(4)

    def test_move_after_game_over_raises(self):
        state = TicTacToeState.new()
        for move in [0, 3, 1, 4, 2]:
            state.make_move(move)
        with pytest.raises(ValueError, match="already over"):
            state.make_move(5)

    def test_legal_moves_shrink_as_board_fills(self):
        state = TicTacToeState.new()
        assert state.legal_moves() == list(range(9))

        state.make_move(4)
        assert 4 not in state.legal_moves()
        assert len(state.legal_moves()) == 8
        assert state.legal_moves_mask()[4] == 0.0

    def test_observation_encodes_planes_mask_and_player(self):
        config = get_config("tictactoe")
        board_size = config.board_width * config.board_height

        state = TicTacToeState.new()
        state.make_move(4)  # X in the center
        obs = state.to_observation(config)

        assert obs.shape == (config.obs_size,)
        assert obs[4] == 1.0
        assert obs[:board_size].sum() == 1.0
        assert obs[board_size : 2 * board_size].sum() == 0.0
        mask = obs[config.legal_mask_offset : config.legal_mask_offset + 9]
        assert mask[4] == 0.0
        assert mask.sum() == 8.0
        player_offset = config.legal_mask_offset + config.num_actions
        assert obs[player_offset + 1] == 1.0  # O to move
