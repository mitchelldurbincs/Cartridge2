"""Round-robin tournaments over registered players, rated with Bradley-Terry Elo.

Win rate against one opponent answers "is the candidate better than the
champion". It does not answer "how do all of these compare", because it depends
entirely on who the opponent was. A round-robin plus a rating model does, and
over a run's checkpoints it reads as a training curve — which is the progress
signal win-rate-vs-random cannot give, since every decent checkpoint pins at
100% and the curve goes flat exactly when you want resolution.

Nothing here knows how a player was trained. A PPO checkpoint and an AlphaZero
checkpoint enter the same pool and get comparable ratings.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .atomic_io import atomic_write
from .evaluator import DEFAULT_SEED, evaluate
from .registry import RANDOM_PLAYER_ID, PlayerRecord, PlayerRegistry

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Elo points per factor-of-10 odds. The conventional constant; changing it
#: rescales every rating, so it is not a knob.
ELO_SCALE = 400.0

#: Virtual games each player is credited with against a rating-0 opponent,
#: split evenly. Without it an undefeated player has unbounded rating and the
#: fit does not converge; with it, a player who beats everyone lands at a large
#: but finite number whose size reflects how much evidence there is.
PRIOR_GAMES = 2.0

_MAX_ITERATIONS = 1000
_TOLERANCE = 1e-9


@dataclass
class MatchResult:
    """Outcome of one pairing."""

    player1: str
    player2: str
    games: int
    player1_wins: int
    player2_wins: int
    draws: int


@dataclass
class Rating:
    """Where a player landed."""

    player_id: str
    rating: float
    games: int
    wins: int
    losses: int
    draws: int
    step: int | None = None
    algorithm: str = "unknown"

    @property
    def score_rate(self) -> float:
        """Points per game, draws counting a half."""
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games


@dataclass
class TournamentResults:
    """A whole round-robin: every pairing, and the ratings fitted from them."""

    env_id: str
    games_per_pair: int
    seed: int
    anchor: str
    schema_version: int = SCHEMA_VERSION
    timestamp: str = ""
    wall_time_seconds: float = 0.0
    matches: list[MatchResult] = field(default_factory=list)
    ratings: list[Rating] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "env_id": self.env_id,
            "games_per_pair": self.games_per_pair,
            "seed": self.seed,
            "anchor": self.anchor,
            "timestamp": self.timestamp,
            "wall_time_seconds": self.wall_time_seconds,
            "ratings": [asdict(r) for r in self.ratings],
            "matches": [asdict(m) for m in self.matches],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            path,
            lambda tmp: Path(tmp).write_text(
                json.dumps(self.to_dict(), indent=2) + "\n"
            ),
        )

    def table(self) -> str:
        """Ratings, strongest first."""
        if not self.ratings:
            return "(no ratings)"
        ranked = sorted(self.ratings, key=lambda r: r.rating, reverse=True)
        width = max(len(r.player_id) for r in ranked)
        lines = [
            f"{'player'.ljust(width)}  {'elo':>7}  {'score':>6}  {'W-L-D':>12}  step",
            f"{'-' * width}  {'-' * 7}  {'-' * 6}  {'-' * 12}  {'-' * 8}",
        ]
        for r in ranked:
            record = f"{r.wins}-{r.losses}-{r.draws}"
            step = "" if r.step is None else str(r.step)
            lines.append(
                f"{r.player_id.ljust(width)}  {r.rating:7.0f}  "
                f"{r.score_rate:6.1%}  {record:>12}  {step}"
            )
        return "\n".join(lines)


def fit_ratings(
    player_ids: list[str],
    matches: list[MatchResult],
    anchor: str | None = None,
    prior_games: float = PRIOR_GAMES,
) -> dict[str, float]:
    """Fit Bradley-Terry ratings on the Elo scale from pairwise results.

    Uses the Zermelo/minorization-maximization iteration, which converges
    without a learning rate to tune. Draws count as half a win to each side.

    Sequential Elo updates were the obvious alternative and are wrong here: they
    depend on the order games happen to be played, so the same round-robin would
    yield different ratings depending on scheduling. This fit is a property of
    the results alone.

    Ratings are shifted so ``anchor`` sits at 0, making every number read as
    "Elo above the baseline". With no anchor the mean is 0 instead.
    """
    if not player_ids:
        return {}

    index = {pid: i for i, pid in enumerate(player_ids)}
    n = len(player_ids)

    # score[i][j] = points i took off j; played[i][j] = games between them.
    score = [[0.0] * n for _ in range(n)]
    played = [[0.0] * n for _ in range(n)]
    for match in matches:
        if match.player1 not in index or match.player2 not in index:
            continue
        i, j = index[match.player1], index[match.player2]
        total = match.player1_wins + match.player2_wins + match.draws
        score[i][j] += match.player1_wins + 0.5 * match.draws
        score[j][i] += match.player2_wins + 0.5 * match.draws
        played[i][j] += total
        played[j][i] += total

    wins = [sum(score[i]) + prior_games / 2 for i in range(n)]
    gamma = [1.0] * n

    for _ in range(_MAX_ITERATIONS):
        largest_change = 0.0
        for i in range(n):
            denominator = prior_games / (gamma[i] + 1.0)
            for j in range(n):
                if i != j and played[i][j]:
                    denominator += played[i][j] / (gamma[i] + gamma[j])
            if denominator <= 0.0:
                continue
            updated = wins[i] / denominator
            largest_change = max(largest_change, abs(updated - gamma[i]))
            gamma[i] = updated
        if largest_change < _TOLERANCE:
            break

    ratings = {pid: ELO_SCALE * math.log10(gamma[i]) for pid, i in index.items()}

    offset = ratings.get(anchor) if anchor else None
    if offset is None:
        offset = sum(ratings.values()) / len(ratings)
    return {pid: rating - offset for pid, rating in ratings.items()}


def playable_field(registry: PlayerRegistry, env_id: str) -> list[PlayerRecord]:
    """Registered players for ``env_id`` whose checkpoints still exist.

    Checkpoint rotation deletes old models, so the registry outlives the files
    it points at. One missing checkpoint should cost its own entry, not the
    whole tournament.
    """
    field_ = []
    for player in registry.for_env(env_id):
        if player.is_playable():
            field_.append(player)
        else:
            logger.warning(
                "Skipping player '%s': checkpoint %s is missing",
                player.id,
                player.checkpoint,
            )
    return field_


def warn_about_deterministic_players(players: list[PlayerRecord]) -> list[str]:
    """Warn about model players that will replay one identical game.

    A greedy model with no search is deterministic, so a pairing of two of them
    plays the same game every time and a 20-game match is one game counted 20
    times. The ratings that come out look confident and mean nothing. This is
    easy to do by accident — temperature 0 is the natural default for "play its
    best move" — so say so rather than silently producing the table.

    Returns the ids warned about, for testing.
    """
    deterministic = [
        p.id
        for p in players
        if p.kind == "model" and p.temperature <= 0.0 and p.simulations == 0
    ]
    if len(deterministic) > 1:
        logger.warning(
            "%d players are deterministic (temperature 0, no search): %s. "
            "Every pairing among them replays one identical game per seat, so "
            "their ratings will be confident but meaningless. Re-register with "
            "--temperature 0.2 or give them a search budget.",
            len(deterministic),
            ", ".join(deterministic),
        )
    return deterministic


def run_tournament(
    registry: PlayerRegistry,
    env_id: str,
    games_per_pair: int = 20,
    seed: int = DEFAULT_SEED,
    run_match=evaluate,
) -> TournamentResults:
    """Play every pairing once and fit ratings from the results.

    Every pair plays ``games_per_pair`` games with the same seed, so each
    matchup faces identical conditions and re-running reproduces the table.

    ``run_match`` is the head-to-head callable, injected so tests do not need
    the binary.
    """
    players = playable_field(registry, env_id)
    if len(players) < 2:
        raise ValueError(
            f"A tournament needs at least 2 playable players for '{env_id}', found "
            f"{len(players)}. Register some with `trainer register-players`."
        )

    results = TournamentResults(
        env_id=env_id,
        games_per_pair=games_per_pair,
        seed=seed,
        anchor=RANDOM_PLAYER_ID,
    )
    warn_about_deterministic_players(players)
    by_id = {p.id: p for p in players}
    pairings = list(itertools.combinations(players, 2))
    logger.info(
        "Tournament: %d players, %d pairings, %d games each (%d games total)",
        len(players),
        len(pairings),
        games_per_pair,
        len(pairings) * games_per_pair,
    )

    start = time.perf_counter()
    for number, (first, second) in enumerate(pairings, start=1):
        outcome = run_match(
            player1=first.to_player(),
            player2=second.to_player(),
            env_id=env_id,
            num_games=games_per_pair,
            verbose=False,
            seed=seed,
        )
        results.matches.append(
            MatchResult(
                player1=first.id,
                player2=second.id,
                games=outcome.games_played,
                player1_wins=outcome.player1_wins,
                player2_wins=outcome.player2_wins,
                draws=outcome.draws,
            )
        )
        if number % 25 == 0 or number == len(pairings):
            logger.info("  %d/%d pairings played", number, len(pairings))

    ratings = fit_ratings(
        [p.id for p in players],
        results.matches,
        anchor=RANDOM_PLAYER_ID if RANDOM_PLAYER_ID in by_id else None,
    )

    tallies = {p.id: [0, 0, 0, 0] for p in players}  # games, wins, losses, draws
    for match in results.matches:
        for pid, won, lost in (
            (match.player1, match.player1_wins, match.player2_wins),
            (match.player2, match.player2_wins, match.player1_wins),
        ):
            tallies[pid][0] += match.games
            tallies[pid][1] += won
            tallies[pid][2] += lost
            tallies[pid][3] += match.draws

    results.ratings = [
        Rating(
            player_id=pid,
            rating=ratings[pid],
            games=tallies[pid][0],
            wins=tallies[pid][1],
            losses=tallies[pid][2],
            draws=tallies[pid][3],
            step=by_id[pid].step,
            algorithm=by_id[pid].algorithm,
        )
        for pid in (p.id for p in players)
    ]
    results.wall_time_seconds = time.perf_counter() - start
    results.timestamp = datetime.now().isoformat()
    return results
