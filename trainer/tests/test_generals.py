"""Tests for the pure-Python Generals game (generals_8x8).

Mirrors the key cases of the Rust suite in
``engine/games-generals/src/tests.rs`` so the two implementations can't
silently drift: combat semantics, general capture, round clock/production,
draw adjudication, legal-move masks, mapgen invariants, and the
``generals_obs:v1`` encoding contract.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from trainer.game_config import get_config
from trainer.games import Player, create_game_state
from trainer.games.generals import (
    BOARD_SIZE,
    CITY,
    CITY_RATIO,
    CITY_START_ARMY,
    GENERAL,
    GENERAL_START_ARMY,
    MAX_ARMY_NORM,
    MAX_TURNS,
    MOUNTAIN,
    NEUTRAL,
    NORMAL,
    NUM_ACTIONS,
    WAIT_ACTION,
    GeneralsState,
    _generate_map,
    _xy,
)
from trainer.resnet import ConvPolicyValueNetwork

CONFIG = get_config("generals_8x8")


def encode_move(from_idx: int, direction: int) -> int:
    return from_idx * 4 + direction


def flat_state() -> GeneralsState:
    """Two generals in opposite corners, no terrain."""
    owner = [NEUTRAL] * BOARD_SIZE
    army = [0] * BOARD_SIZE
    kind = [NORMAL] * BOARD_SIZE
    g1, g2 = 0, BOARD_SIZE - 1
    owner[g1], army[g1], kind[g1] = 1, GENERAL_START_ARMY, GENERAL
    owner[g2], army[g2], kind[g2] = 2, GENERAL_START_ARMY, GENERAL
    return GeneralsState(owner=owner, army=army, kind=kind)


class TestCombat:
    def test_move_to_own_tile_consolidates(self):
        state = flat_state()
        state.owner[0], state.army[0] = 1, 5
        state.owner[1], state.army[1] = 1, 3
        state._apply_move(1, 0, 1)
        assert state.army[0] == 1
        assert state.army[1] == 7  # 3 + (5 - 1) moved armies
        assert state.owner[1] == 1

    def test_capture_neutral_tile(self):
        state = flat_state()
        state.army[0] = 5
        state._apply_move(1, 0, 1)
        assert state.owner[1] == 1
        assert state.army[1] == 4  # 4 attackers vs 0 defenders

    def test_capture_enemy_tile(self):
        state = flat_state()
        state.army[0] = 10
        state.owner[1], state.army[1] = 2, 4
        state._apply_move(1, 0, 1)
        assert state.owner[1] == 1
        assert state.army[1] == 5  # 9 attackers - 4 defenders

    def test_failed_attack_defender_keeps_tile(self):
        state = flat_state()
        state.army[0] = 4
        state.owner[1], state.army[1] = 2, 8
        state._apply_move(1, 0, 1)
        assert state.owner[1] == 2
        assert state.army[1] == 5  # 8 - 3 attackers
        assert state.army[0] == 1

    def test_equal_armies_attack_fails(self):
        # Attacker needs strictly MORE armies than the defender
        state = flat_state()
        state.army[0] = 5
        state.owner[1], state.army[1] = 2, 4
        state._apply_move(1, 0, 1)
        assert state.owner[1] == 2
        assert state.army[1] == 0  # wiped out but holds the tile

    def test_general_capture_transfers_all_tiles(self):
        state = flat_state()
        g2 = BOARD_SIZE - 1
        p2_extra = 8  # extra player-2 tile elsewhere
        state.owner[p2_extra], state.army[p2_extra] = 2, 7
        attacker = g2 - 1
        state.owner[attacker], state.army[attacker] = 1, 50

        state.make_move(encode_move(attacker, 1))  # right, onto g2

        assert state.done
        assert state.winner == 1
        assert not state.alive[1]
        assert state.owner[p2_extra] == 1
        assert state.owner[g2] == 1
        assert state.kind[g2] == GENERAL  # kind unchanged (Go/Rust behavior)


class TestTurnStructure:
    def test_alternating_turns_and_round_clock(self):
        state = flat_state()
        assert state.current_player == Player.FIRST
        state.make_move(WAIT_ACTION)
        assert state.current_player == Player.SECOND
        assert state.round == 0
        state.make_move(WAIT_ACTION)
        assert state.current_player == Player.FIRST
        assert state.round == 1

    def test_production_runs_at_round_end(self):
        state = flat_state()
        before = state.army[0]
        state.make_move(WAIT_ACTION)  # P1
        assert state.army[0] == before
        state.make_move(WAIT_ACTION)  # P2 -> round end
        assert state.army[0] == before + 1  # general production

    def test_normal_tile_growth_interval(self):
        state = flat_state()
        state.owner[10], state.army[10] = 1, 3
        state.round = 24  # next round end -> round 25, growth fires
        state.make_move(WAIT_ACTION)
        state.make_move(WAIT_ACTION)
        assert state.round == 25
        assert state.army[10] == 4

    def test_illegal_action_degrades_to_wait(self):
        state = flat_state()
        unowned = 30
        snapshot = (state.owner.copy(), state.army.copy())
        state.make_move(encode_move(unowned, 1))
        assert (state.owner, state.army) == snapshot
        assert state.current_player == Player.SECOND
        assert not state.done

    def test_out_of_range_action_raises(self):
        state = flat_state()
        try:
            state.make_move(NUM_ACTIONS)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    def test_max_turns_symmetric_position_is_draw(self):
        # flat_state is perfectly symmetric, so cap adjudication is a draw
        state = flat_state()
        state.round = MAX_TURNS - 1
        state.make_move(WAIT_ACTION)
        state.make_move(WAIT_ACTION)
        assert state.done
        assert state.winner is None  # draw maps to None in the protocol
        assert state._winner_code == 3

    def test_max_turns_adjudicates_by_territory(self):
        state = flat_state()
        state.owner[27], state.army[27] = 2, 1  # extra tile for player 2
        state.round = MAX_TURNS - 1
        state.make_move(WAIT_ACTION)
        state.make_move(WAIT_ACTION)
        assert state.done
        assert state.winner == 2

    def test_odd_cap_gives_player1_the_last_move(self):
        state = flat_state()
        state.cap_plies = 2 * MAX_TURNS - 1
        state.round = MAX_TURNS - 1
        state.owner[27], state.army[27] = 1, 1  # P1 ahead on tiles
        state.make_move(WAIT_ACTION)  # P1's final ply ends the game
        assert state.done
        assert state.winner == 1

    def test_new_samples_both_cap_parities(self):
        caps = {GeneralsState.new(random.Random(seed)).cap_plies for seed in range(40)}
        assert caps == {2 * MAX_TURNS, 2 * MAX_TURNS - 1}

    def test_adjudication_army_tiebreak(self):
        state = flat_state()
        state.army[0] = 10  # equal tiles, player 1 has more armies
        assert state._adjudicate_at_cap() == 1
        state.army[0] = GENERAL_START_ARMY
        assert state._adjudicate_at_cap() == 3


class TestLegalMoves:
    def test_wait_always_legal_and_corner_directions(self):
        state = flat_state()
        legal = set(state.legal_moves())
        assert WAIT_ACTION in legal
        # General at corner 0 (2 armies): right and down legal, up/left off-board
        assert encode_move(0, 1) in legal
        assert encode_move(0, 2) in legal
        assert encode_move(0, 0) not in legal
        assert encode_move(0, 3) not in legal
        # Unowned tiles offer no moves
        assert encode_move(27, 0) not in legal

    def test_mountain_blocks_moves(self):
        state = flat_state()
        state.kind[1] = MOUNTAIN  # right of the corner general
        legal = set(state.legal_moves())
        assert encode_move(0, 1) not in legal
        assert encode_move(0, 2) in legal

    def test_no_legal_moves_when_done(self):
        state = flat_state()
        state._winner_code = 1
        assert state.legal_moves() == []
        assert all(v == 0.0 for v in state.legal_moves_mask())


class TestMapgen:
    def test_deterministic_for_seed(self):
        a = _generate_map(random.Random(1234))
        b = _generate_map(random.Random(1234))
        assert a == b

    def test_invariants(self):
        for seed in range(30):
            owner, army, kind = _generate_map(random.Random(seed))

            generals = [i for i in range(BOARD_SIZE) if kind[i] == GENERAL]
            assert len(generals) == 2, f"seed {seed}"
            assert sorted(owner[g] for g in generals) == [1, 2], f"seed {seed}"
            for g in generals:
                assert army[g] == GENERAL_START_ARMY, f"seed {seed}"

            cities = [i for i in range(BOARD_SIZE) if kind[i] == CITY]
            assert len(cities) == BOARD_SIZE // CITY_RATIO, f"seed {seed}"
            for c in cities:
                assert army[c] == CITY_START_ARMY and owner[c] == NEUTRAL

            for i in range(BOARD_SIZE):
                if kind[i] == MOUNTAIN:
                    assert army[i] == 0 and owner[i] == NEUTRAL

    def test_general_spacing(self):
        for seed in range(30):
            owner, army, kind = _generate_map(random.Random(seed))
            g1, g2 = (i for i in range(BOARD_SIZE) if kind[i] == GENERAL)
            (x1, y1), (x2, y2) = _xy(g1), _xy(g2)
            assert abs(x1 - x2) + abs(y1 - y2) >= 5, f"seed {seed}"


class TestObservation:
    def test_shape_and_config_agreement(self):
        assert CONFIG.num_actions == NUM_ACTIONS == 257
        assert CONFIG.obs_size == 9 * BOARD_SIZE + NUM_ACTIONS + 2 == 835
        assert CONFIG.legal_mask_offset == 9 * BOARD_SIZE == 576
        assert CONFIG.input_channels == 9

        obs = flat_state().to_observation(CONFIG)
        assert obs.shape == (835,)
        assert obs.dtype == np.float32

    def test_mask_slice_matches_legal_moves(self):
        state = GeneralsState.new(random.Random(7))
        obs = state.to_observation(CONFIG)
        mask = obs[CONFIG.legal_mask_offset : CONFIG.legal_mask_offset + NUM_ACTIONS]
        assert set(np.nonzero(mask > 0.5)[0].tolist()) == set(state.legal_moves())

    def test_player_relative_channels(self):
        state = flat_state()
        g1, g2 = 0, BOARD_SIZE - 1
        obs_p1 = state.to_observation(CONFIG)

        assert obs_p1[g1] == 1.0  # ch0 own territory
        assert obs_p1[BOARD_SIZE + g2] == 1.0  # ch1 enemy territory
        assert obs_p1[7 * BOARD_SIZE + g1] == 1.0  # ch7 own general
        assert obs_p1[7 * BOARD_SIZE + g2] == -1.0  # ch7 enemy general

        # Player 2's view is mirrored
        state._current_player = Player.SECOND
        obs_p2 = state.to_observation(CONFIG)
        assert obs_p2[g2] == 1.0
        assert obs_p2[BOARD_SIZE + g1] == 1.0
        assert obs_p2[7 * BOARD_SIZE + g2] == 1.0
        assert obs_p2[7 * BOARD_SIZE + g1] == -1.0

        # Player one-hot flips with the perspective
        player_offset = CONFIG.legal_mask_offset + NUM_ACTIONS
        assert obs_p1[player_offset] == 1.0 and obs_p1[player_offset + 1] == 0.0
        assert obs_p2[player_offset] == 0.0 and obs_p2[player_offset + 1] == 1.0

    def test_army_log_normalization(self):
        state = flat_state()
        state.army[0] = 99
        obs = state.to_observation(CONFIG)
        expected = math.log1p(99) / math.log1p(MAX_ARMY_NORM)
        assert abs(obs[3 * BOARD_SIZE + 0] - expected) < 1e-6

    def test_turn_progress_plane(self):
        state = flat_state()
        state.round = MAX_TURNS // 2
        obs = state.to_observation(CONFIG)
        plane = obs[8 * BOARD_SIZE : 9 * BOARD_SIZE]
        assert np.allclose(plane, 0.5)


class TestIntegration:
    def test_create_game_state(self):
        state = create_game_state("generals_8x8")
        assert isinstance(state, GeneralsState)
        assert not state.done
        assert state.current_player == Player.FIRST

    def test_random_playout_terminates(self):
        rng = random.Random(0)
        for seed in range(5):
            state = GeneralsState.new(random.Random(seed))
            plies = 0
            while not state.done:
                state.make_move(rng.choice(state.legal_moves()))
                plies += 1
                assert plies <= MAX_TURNS * 2 + 2, f"seed {seed} exceeded horizon"
            assert state._winner_code in (1, 2, 3)

    def test_copy_is_independent(self):
        state = GeneralsState.new(random.Random(3))
        clone = state.copy()
        clone.make_move(WAIT_ACTION)
        assert state.current_player == Player.FIRST
        assert clone.current_player == Player.SECOND

    def test_network_is_seat_blind(self):
        """For player-relative obs the network must ignore the current-player
        one-hot: flipping the seat indicator alone cannot change the output.
        (Regression test for the value head collapsing into a seat detector
        when one seat systematically wins self-play.)"""
        assert CONFIG.player_relative_obs
        network = ConvPolicyValueNetwork(CONFIG)
        assert network.input_channels == 9  # no derived player plane
        network.eval()

        state = GeneralsState.new(random.Random(13))
        obs = state.to_observation(CONFIG).copy()
        obs_flipped = obs.copy()
        player_offset = CONFIG.legal_mask_offset + NUM_ACTIONS
        obs_flipped[player_offset], obs_flipped[player_offset + 1] = (
            obs_flipped[player_offset + 1],
            obs_flipped[player_offset],
        )

        with torch.no_grad():
            p1, v1 = network(torch.from_numpy(obs).unsqueeze(0))
            p2, v2 = network(torch.from_numpy(obs_flipped).unsqueeze(0))
        assert torch.equal(p1, p2)
        assert torch.equal(v1, v2)

    def test_resnet_forward_and_masking(self):
        """The real network consumes a real observation end to end."""
        network = ConvPolicyValueNetwork(CONFIG)
        network.eval()

        state = GeneralsState.new(random.Random(11))
        obs = torch.from_numpy(state.to_observation(CONFIG)).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = network(obs)

        assert policy_logits.shape == (1, NUM_ACTIONS)
        assert value.shape == (1, 1)
        assert -1.0 <= value.item() <= 1.0

        # Masked softmax over the obs-embedded legal mask stays on legal moves
        mask = CONFIG.extract_legal_mask(obs)
        probs = torch.softmax(
            policy_logits.masked_fill(mask < 0.5, float("-inf")), dim=-1
        )
        legal = set(state.legal_moves())
        for action in range(NUM_ACTIONS):
            if action not in legal:
                assert probs[0, action].item() == 0.0

    def test_display_smoke(self):
        state = GeneralsState.new(random.Random(1))
        text = state.display()
        assert "round=0" in text
