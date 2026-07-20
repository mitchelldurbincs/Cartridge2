"""Pure Python Generals (generals_8x8) state for evaluation.

Mirrors the Rust implementation in ``engine/games-generals`` exactly:
full-information, strictly alternating turns, no half-moves.

Rules recap (see the Rust crate's lib.rs for the authoritative version):
- 8x8 board; actions ``(y*8 + x)*4 + dir`` with dir 0=up, 1=right, 2=down,
  3=left; action 256 = wait. Illegal in-range actions degrade to wait.
- Each ply resolves immediately. After player 2's ply the round counter
  ticks, production runs (generals/cities +1; normal owned tiles +1 every
  25 rounds), and at the round cap the game is adjudicated by territory:
  more tiles wins, total armies break ties, and only a perfectly even
  position is a draw. (A pure draw cap collapsed self-play into 100%
  draws; territory adjudication keeps the game zero-sum but decisive.)
- Combat: attacker sends ``army - 1``; strictly more attackers than
  defenders captures (difference remains), otherwise the defender keeps
  the tile and loses the attacking armies.
- Capturing a general eliminates its owner and transfers all their tiles
  (armies and tile kinds unchanged).

The observation encoding matches ``generals_obs:v1`` (see the Rust
``obs.rs``): 9 player-relative channels x 64 tiles, the 257-float legal
mask at offset 576, then the current-player one-hot.

Map generation follows the same algorithm and parameters as the Rust
``mapgen.rs`` but uses Python's RNG — maps are drawn from the same
distribution, not bit-identical for a given seed. Evaluation plays on
fresh random maps, which is what we want for win-rate estimates.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import Player

if TYPE_CHECKING:
    from ..game_config import GameConfig
    from ..storage import GameMetadata

# Board / action space (must match engine/games-generals/src/params.rs)
WIDTH = 8
HEIGHT = 8
BOARD_SIZE = WIDTH * HEIGHT
NUM_MOVE_ACTIONS = BOARD_SIZE * 4
WAIT_ACTION = NUM_MOVE_ACTIONS
NUM_ACTIONS = NUM_MOVE_ACTIONS + 1

# Balance parameters
CITY_RATIO = 20
CITY_START_ARMY = 40
MIN_GENERAL_SPACING = 5
GENERAL_START_ARMY = 2
GENERAL_PRODUCTION = 1
CITY_PRODUCTION = 1
NORMAL_PRODUCTION = 1
NORMAL_GROW_INTERVAL = 25
MAX_TURNS = 200
NUM_MOUNTAIN_VEINS = BOARD_SIZE // 50
MIN_VEIN_LENGTH = 3
MAX_VEIN_LENGTH = WIDTH // 4
MAX_ARMY_NORM = 1000.0

# Tile kinds (match TileKind in board.rs)
NORMAL = 0
GENERAL = 1
CITY = 2
MOUNTAIN = 3

NEUTRAL = 0

# Direction offsets in canonical up/right/down/left order
DIR_DX = (0, 1, 0, -1)
DIR_DY = (-1, 0, 1, 0)


def _idx(x: int, y: int) -> int:
    return y * WIDTH + x


def _xy(i: int) -> tuple[int, int]:
    return i % WIDTH, i // WIDTH


def _move_target(from_idx: int, direction: int) -> int | None:
    """Target tile index for a move, or None if it leaves the board."""
    x, y = _xy(from_idx)
    nx = x + DIR_DX[direction]
    ny = y + DIR_DY[direction]
    if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
        return _idx(nx, ny)
    return None


def _generate_map(rng: random.Random) -> tuple[list[int], list[int], list[int]]:
    """Generate (owner, army, kind) arrays: mountains, cities, then generals."""
    owner = [NEUTRAL] * BOARD_SIZE
    army = [0] * BOARD_SIZE
    kind = [NORMAL] * BOARD_SIZE

    # Mountain veins
    for _ in range(NUM_MOUNTAIN_VEINS):
        start = None
        for _ in range(100):
            i = rng.randrange(BOARD_SIZE)
            if kind[i] == NORMAL and owner[i] == NEUTRAL:
                start = i
                break
        if start is None:
            continue

        current = start
        kind[current] = MOUNTAIN
        vein_length = (
            MIN_VEIN_LENGTH + rng.randint(0, MAX_VEIN_LENGTH - MIN_VEIN_LENGTH)
            if MAX_VEIN_LENGTH > MIN_VEIN_LENGTH
            else MIN_VEIN_LENGTH
        )
        for _ in range(1, vein_length):
            x, y = _xy(current)
            candidates = []
            for dx, dy in zip(DIR_DX, DIR_DY):
                nx, ny = x + dx, y + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    ni = _idx(nx, ny)
                    if kind[ni] == NORMAL and owner[ni] == NEUTRAL:
                        candidates.append(ni)
            if not candidates:
                break
            current = rng.choice(candidates)
            kind[current] = MOUNTAIN

    # Cities
    want = BOARD_SIZE // CITY_RATIO
    placed = 0
    attempts = 0
    while placed < want and attempts < want * 20:
        i = rng.randrange(BOARD_SIZE)
        if owner[i] == NEUTRAL and kind[i] == NORMAL:
            kind[i] = CITY
            army[i] = CITY_START_ARMY
            placed += 1
        attempts += 1

    # Generals
    spacing = min(MIN_GENERAL_SPACING, WIDTH // 2 + HEIGHT // 2)
    generals: list[int] = []
    for player in (1, 2):
        spot = _find_general_spot(owner, kind, generals, spacing, rng)
        if spot is None:
            spot = _find_general_spot(owner, kind, generals, 0, rng)
        assert spot is not None, "8x8 board must have a free tile for a general"
        owner[spot] = player
        army[spot] = GENERAL_START_ARMY
        kind[spot] = GENERAL
        generals.append(spot)

    return owner, army, kind


def _find_general_spot(
    owner: list[int],
    kind: list[int],
    existing: list[int],
    spacing: int,
    rng: random.Random,
) -> int | None:
    def spaced_ok(i: int) -> bool:
        x, y = _xy(i)
        for g in existing:
            gx, gy = _xy(g)
            if abs(x - gx) + abs(y - gy) < spacing:
                return False
        return True

    for _ in range(BOARD_SIZE):
        i = rng.randrange(BOARD_SIZE)
        if owner[i] == NEUTRAL and kind[i] == NORMAL and spaced_ok(i):
            return i
    for i in range(BOARD_SIZE):
        if owner[i] == NEUTRAL and kind[i] == NORMAL and spaced_ok(i):
            return i
    return None


@dataclass
class GeneralsState:
    """Pure Python Generals state for evaluation."""

    owner: list[int]  # 0=neutral, 1/2=players
    army: list[int]
    kind: list[int]  # NORMAL/GENERAL/CITY/MOUNTAIN
    round: int = 0
    _current_player: Player = Player.FIRST
    alive: list[bool] = field(default_factory=lambda: [True, True])
    # 0=ongoing, 1/2=winner, 3=draw (matches the Rust State.winner)
    _winner_code: int = 0
    # Ply count at which the game adjudicates: 2*MAX_TURNS or one less,
    # coin-flipped at reset so each seat gets the final move in half of
    # all games (mirrors the Rust State.cap_plies; see that doc comment).
    cap_plies: int = 2 * MAX_TURNS

    @property
    def done(self) -> bool:
        return self._winner_code != 0

    @property
    def winner(self) -> int | None:
        """1/2 for a decisive result, None for draw or ongoing."""
        return self._winner_code if self._winner_code in (1, 2) else None

    @property
    def current_player(self) -> Player:
        return self._current_player

    @classmethod
    def new(cls, rng: random.Random | None = None) -> "GeneralsState":
        rng = rng or random.Random()
        owner, army, kind = _generate_map(rng)
        return cls(
            owner=owner,
            army=army,
            kind=kind,
            cap_plies=2 * MAX_TURNS - rng.randint(0, 1),
        )

    def copy(self) -> "GeneralsState":
        return GeneralsState(
            owner=self.owner.copy(),
            army=self.army.copy(),
            kind=self.kind.copy(),
            round=self.round,
            _current_player=self._current_player,
            alive=self.alive.copy(),
            _winner_code=self._winner_code,
            cap_plies=self.cap_plies,
        )

    def _valid_move_target(self, player: int, from_idx: int, direction: int) -> int | None:
        """Mirror of the Rust ``valid_move_target``."""
        if self.owner[from_idx] != player or self.army[from_idx] <= 1:
            return None
        to = _move_target(from_idx, direction)
        if to is None or self.kind[to] == MOUNTAIN:
            return None
        return to

    def legal_moves(self) -> list[int]:
        """Legal action indices for the current player ([] once done)."""
        if self.done:
            return []
        player = int(self._current_player)
        moves = []
        for from_idx in range(BOARD_SIZE):
            if self.owner[from_idx] != player or self.army[from_idx] <= 1:
                continue
            for direction in range(4):
                if self._valid_move_target(player, from_idx, direction) is not None:
                    moves.append(from_idx * 4 + direction)
        moves.append(WAIT_ACTION)
        return moves

    def legal_moves_mask(self) -> list[float]:
        mask = [0.0] * NUM_ACTIONS
        for action in self.legal_moves():
            mask[action] = 1.0
        return mask

    def make_move(self, pos: int) -> None:
        """Apply one ply. In-range illegal actions degrade to wait (Rust
        semantics); out-of-range actions are a caller bug and raise."""
        if self.done:
            raise ValueError("Game is already over")
        if not 0 <= pos < NUM_ACTIONS:
            raise ValueError(f"Action {pos} out of range 0-{NUM_ACTIONS - 1}")

        previous_player = int(self._current_player)

        if pos != WAIT_ACTION:
            from_idx, direction = pos // 4, pos % 4
            to = self._valid_move_target(previous_player, from_idx, direction)
            if to is not None:
                self._apply_move(previous_player, from_idx, to)

        winner = self._check_winner()

        # End of round after player 2's ply: production
        if winner == 0 and previous_player == 2:
            self.round += 1
            self._apply_production()

        # Territory adjudication at the (parity-randomized) ply cap
        if winner == 0:
            plies_played = self.round * 2 + (1 if previous_player == 1 else 0)
            if plies_played >= self.cap_plies:
                winner = self._adjudicate_at_cap()

        self._winner_code = winner
        if winner == 0:
            self._current_player = (
                Player.SECOND if previous_player == 1 else Player.FIRST
            )

    def _apply_move(self, player: int, from_idx: int, to: int) -> None:
        armies_to_move = self.army[from_idx] - 1
        self.army[from_idx] = 1

        defender_owner = self.owner[to]
        if defender_owner == player:
            self.army[to] += armies_to_move
            return

        if armies_to_move > self.army[to]:
            # Capture
            captured_kind = self.kind[to]
            self.owner[to] = player
            self.army[to] = armies_to_move - self.army[to]
            if captured_kind == GENERAL and defender_owner != NEUTRAL:
                # Eliminate: transfer all tiles, kinds and armies unchanged
                for i in range(BOARD_SIZE):
                    if self.owner[i] == defender_owner:
                        self.owner[i] = player
                self.alive[defender_owner - 1] = False
        else:
            self.army[to] -= armies_to_move

    def _adjudicate_at_cap(self) -> int:
        """Territory adjudication at the round cap (mirrors the Rust
        ``adjudicate_at_cap``): more tiles wins, armies tiebreak, 3=draw."""
        tiles = [0, 0]
        armies = [0, 0]
        for i in range(BOARD_SIZE):
            if self.owner[i] in (1, 2):
                p = self.owner[i] - 1
                tiles[p] += 1
                armies[p] += self.army[i]
        if tiles[0] != tiles[1]:
            return 1 if tiles[0] > tiles[1] else 2
        if armies[0] != armies[1]:
            return 1 if armies[0] > armies[1] else 2
        return 3

    def _check_winner(self) -> int:
        if self.alive[0] and self.alive[1]:
            return 0
        if self.alive[0]:
            return 1
        if self.alive[1]:
            return 2
        return 3

    def _apply_production(self) -> None:
        grow_normal = self.round % NORMAL_GROW_INTERVAL == 0
        for i in range(BOARD_SIZE):
            if self.owner[i] == NEUTRAL:
                continue
            if self.kind[i] == GENERAL:
                self.army[i] += GENERAL_PRODUCTION
            elif self.kind[i] == CITY:
                self.army[i] += CITY_PRODUCTION
            elif self.kind[i] == NORMAL and grow_normal:
                self.army[i] += NORMAL_PRODUCTION

    def to_observation(self, config: "GameConfig | GameMetadata") -> np.ndarray:
        """Encode as ``generals_obs:v1`` — must match the Rust ``obs.rs``."""
        obs = np.zeros(config.obs_size, dtype=np.float32)
        player = int(self._current_player)
        norm = math.log1p(MAX_ARMY_NORM)
        turn_progress = min(self.round / MAX_TURNS, 1.0)

        for i in range(BOARD_SIZE):
            own = self.owner[i] == player
            enemy = self.owner[i] != NEUTRAL and not own

            if own:
                obs[i] = 1.0  # ch0 own territory
                obs[3 * BOARD_SIZE + i] = math.log1p(self.army[i]) / norm
            elif enemy:
                obs[BOARD_SIZE + i] = 1.0  # ch1 enemy territory
                obs[4 * BOARD_SIZE + i] = math.log1p(self.army[i]) / norm
            elif self.kind[i] != MOUNTAIN:
                obs[2 * BOARD_SIZE + i] = 1.0  # ch2 neutral passable

            if self.kind[i] == CITY:
                obs[5 * BOARD_SIZE + i] = 1.0
            elif self.kind[i] == MOUNTAIN:
                obs[6 * BOARD_SIZE + i] = 1.0
            elif self.kind[i] == GENERAL:
                obs[7 * BOARD_SIZE + i] = 1.0 if own else -1.0
            obs[8 * BOARD_SIZE + i] = turn_progress

        # Legal mask at legal_mask_offset. The Rust encoder fills this from
        # the acting player's live legality (alive check), which for a
        # non-terminal state equals legal_moves_mask().
        if not self.done and self.alive[player - 1]:
            for action in self.legal_moves():
                obs[config.legal_mask_offset + action] = 1.0

        # Current player one-hot
        player_offset = config.legal_mask_offset + config.num_actions
        obs[player_offset + player - 1] = 1.0

        return obs

    def display(self) -> str:
        """ASCII board: owner glyph + army, '###' mountains, 'c' cities."""
        rows = []
        for y in range(HEIGHT):
            cells = []
            for x in range(WIDTH):
                i = _idx(x, y)
                if self.kind[i] == MOUNTAIN:
                    cells.append("###")
                    continue
                glyph = {NEUTRAL: ".", 1: "R", 2: "B"}[self.owner[i]]
                if self.kind[i] == GENERAL:
                    glyph = glyph.lower() if self.owner[i] == NEUTRAL else f"*{glyph}"
                elif self.kind[i] == CITY:
                    glyph = f"c{glyph}"
                cells.append(f"{glyph}{self.army[i]}".ljust(3))
            rows.append(" ".join(cells))
        rows.append(
            f"round={self.round} to_move={int(self._current_player)} "
            f"winner={self._winner_code}"
        )
        return "\n".join(rows)
