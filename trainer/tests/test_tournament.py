"""Tests for round-robin tournaments and Bradley-Terry rating.

The head-to-head callable is injected, so nothing here needs the
``cartridge-eval`` binary — the Python CI job has no Rust toolchain.
"""

import json
from types import SimpleNamespace

import pytest

from trainer.registry import PlayerRecord, PlayerRegistry
from trainer.tournament import (
    MatchResult,
    fit_ratings,
    playable_field,
    run_tournament,
    warn_about_deterministic_players,
)


def match(p1: str, p2: str, p1_wins: int, p2_wins: int, draws: int = 0) -> MatchResult:
    return MatchResult(
        player1=p1,
        player2=p2,
        games=p1_wins + p2_wins + draws,
        player1_wins=p1_wins,
        player2_wins=p2_wins,
        draws=draws,
    )


def registry_of(*ids: str) -> PlayerRegistry:
    """A registry of random players — playable without any checkpoint on disk."""
    return PlayerRegistry(
        [PlayerRecord(id=pid, env_id="connect4", kind="random") for pid in ids]
    )


class TestFitRatings:
    def test_a_stronger_player_rates_higher(self):
        ratings = fit_ratings(["strong", "weak"], [match("strong", "weak", 18, 2)])
        assert ratings["strong"] > ratings["weak"]

    def test_even_results_rate_equally(self):
        ratings = fit_ratings(["a", "b"], [match("a", "b", 10, 10)])
        assert ratings["a"] == pytest.approx(ratings["b"], abs=1e-6)

    def test_draws_count_as_half_a_win_each(self):
        all_draws = fit_ratings(["a", "b"], [match("a", "b", 0, 0, draws=20)])
        split = fit_ratings(["a", "b"], [match("a", "b", 10, 10)])
        assert all_draws["a"] == pytest.approx(split["a"], abs=1e-6)

    def test_transitive_field_comes_out_in_order(self):
        matches = [
            match("strong", "middle", 15, 5),
            match("middle", "weak", 15, 5),
            match("strong", "weak", 19, 1),
        ]
        ratings = fit_ratings(["strong", "middle", "weak"], matches)
        assert ratings["strong"] > ratings["middle"] > ratings["weak"]

    def test_anchor_sits_at_zero(self):
        matches = [match("model", "random", 18, 2)]
        ratings = fit_ratings(["model", "random"], matches, anchor="random")

        assert ratings["random"] == pytest.approx(0.0, abs=1e-9)
        assert ratings["model"] > 0

    def test_without_an_anchor_the_mean_is_zero(self):
        matches = [match("a", "b", 15, 5)]
        ratings = fit_ratings(["a", "b"], matches)
        assert sum(ratings.values()) == pytest.approx(0.0, abs=1e-9)

    def test_an_undefeated_player_stays_finite(self):
        # Without the prior this diverges and the fit never converges.
        matches = [match("perfect", "hopeless", 20, 0)]
        ratings = fit_ratings(["perfect", "hopeless"], matches)

        assert all(abs(r) < 10_000 for r in ratings.values())

    def test_results_do_not_depend_on_match_order(self):
        # The reason for fitting rather than applying sequential Elo updates:
        # ratings must be a property of the results, not of the schedule.
        matches = [
            match("a", "b", 12, 8),
            match("b", "c", 15, 5),
            match("a", "c", 17, 3),
        ]
        forward = fit_ratings(["a", "b", "c"], matches)
        backward = fit_ratings(["a", "b", "c"], list(reversed(matches)))

        for pid in forward:
            assert forward[pid] == pytest.approx(backward[pid], abs=1e-6)

    def test_matches_naming_unknown_players_are_ignored(self):
        ratings = fit_ratings(["a", "b"], [match("a", "ghost", 20, 0)])
        assert set(ratings) == {"a", "b"}

    def test_empty_field_is_empty(self):
        assert fit_ratings([], []) == {}


class TestPlayableField:
    def test_missing_checkpoints_are_skipped_not_fatal(self, tmp_path):
        present = tmp_path / "here.onnx"
        present.write_bytes(b"x")
        registry = PlayerRegistry(
            [
                PlayerRecord(id="random", env_id="connect4", kind="random"),
                PlayerRecord(
                    id="here", env_id="connect4", kind="model", checkpoint=str(present)
                ),
                PlayerRecord(
                    id="gone",
                    env_id="connect4",
                    kind="model",
                    checkpoint=str(tmp_path / "gone.onnx"),
                ),
            ]
        )

        field = playable_field(registry, "connect4")

        # Stepless players sort alphabetically, so "here" precedes "random";
        # "gone" is dropped rather than failing the whole tournament.
        assert [p.id for p in field] == ["here", "random"]


class TestDeterministicWarning:
    """Greedy, searchless models replay one game — that must not pass silently.

    Observed on the real Connect 4 checkpoints: at temperature 0 every
    model-vs-model pairing came out exactly 0-20, 10-10 or 20-0, and the
    resulting ratings were confident nonsense.
    """

    def model(self, pid, temperature=0.0, simulations=0):
        return PlayerRecord(
            id=pid,
            env_id="connect4",
            kind="model",
            checkpoint=f"/m/{pid}.onnx",
            temperature=temperature,
            simulations=simulations,
        )

    def test_warns_when_two_or_more_players_are_deterministic(self, caplog):
        players = [self.model("a"), self.model("b"), self.model("c")]
        flagged = warn_about_deterministic_players(players)

        assert flagged == ["a", "b", "c"]
        assert "deterministic" in caplog.text

    def test_a_single_deterministic_player_is_fine(self, caplog):
        # It can only meet stochastic opponents, so no pairing replays.
        players = [self.model("a"), self.model("b", temperature=0.2)]
        warn_about_deterministic_players(players)

        assert "deterministic" not in caplog.text

    def test_sampling_or_search_makes_a_player_non_deterministic(self):
        players = [
            self.model("sampled", temperature=0.2),
            self.model("searching", simulations=50),
        ]
        assert warn_about_deterministic_players(players) == []

    def test_random_players_are_never_flagged(self):
        players = [
            PlayerRecord(id="r1", env_id="connect4", kind="random"),
            PlayerRecord(id="r2", env_id="connect4", kind="random"),
        ]
        assert warn_about_deterministic_players(players) == []


class TestRunTournament:
    def stub_match(self, table=None):
        """A head-to-head stub; `table` maps (p1, p2) -> p1 wins out of 20."""
        table = table or {}

        def run(*, player1, player2, env_id, num_games, verbose, seed):
            del env_id, verbose, seed
            key = (player1.name, player2.name)
            wins = table.get(key, num_games // 2)
            return SimpleNamespace(
                games_played=num_games,
                player1_wins=wins,
                player2_wins=num_games - wins,
                draws=0,
            )

        return run

    def test_plays_every_pairing_once(self):
        registry = registry_of("a", "b", "c", "d")
        results = run_tournament(
            registry, "connect4", games_per_pair=10, run_match=self.stub_match()
        )

        assert len(results.matches) == 6  # C(4,2)
        pairs = {(m.player1, m.player2) for m in results.matches}
        assert len(pairs) == 6

    def test_needs_at_least_two_players(self):
        with pytest.raises(ValueError, match="at least 2"):
            run_tournament(
                registry_of("lonely"), "connect4", run_match=self.stub_match()
            )

    def test_tallies_are_symmetric_with_the_matches(self):
        registry = registry_of("a", "b", "c")
        results = run_tournament(
            registry, "connect4", games_per_pair=10, run_match=self.stub_match()
        )

        by_id = {r.player_id: r for r in results.ratings}
        # Each player meets the other two, 10 games each.
        for rating in by_id.values():
            assert rating.games == 20
            assert rating.wins + rating.losses + rating.draws == 20

        total_wins = sum(r.wins for r in by_id.values())
        total_losses = sum(r.losses for r in by_id.values())
        assert total_wins == total_losses

    def test_ratings_reflect_the_results(self, tmp_path):
        # Model players, so each has a distinguishable name for the stub to key
        # on regardless of the order pairings come out in.
        def model(pid: str) -> PlayerRecord:
            path = tmp_path / f"{pid}.onnx"
            path.write_bytes(b"x")
            return PlayerRecord(
                id=pid, env_id="connect4", kind="model", checkpoint=str(path)
            )

        registry = PlayerRegistry(
            [
                PlayerRecord(id="random", env_id="connect4", kind="random"),
                model("weak"),
                model("strong"),
            ]
        )

        # How many of the 20 games the named player takes in each matchup.
        favourites = {
            frozenset({"Random", "ONNX(weak.onnx)"}): ("ONNX(weak.onnx)", 16),
            frozenset({"Random", "ONNX(strong.onnx)"}): ("ONNX(strong.onnx)", 19),
            frozenset({"ONNX(weak.onnx)", "ONNX(strong.onnx)"}): (
                "ONNX(strong.onnx)",
                15,
            ),
        }

        def run(*, player1, player2, env_id, num_games, verbose, seed):
            del env_id, verbose, seed
            winner, wins = favourites[frozenset({player1.name, player2.name})]
            p1_wins = wins if player1.name == winner else num_games - wins
            return SimpleNamespace(
                games_played=num_games,
                player1_wins=p1_wins,
                player2_wins=num_games - p1_wins,
                draws=0,
            )

        results = run_tournament(registry, "connect4", games_per_pair=20, run_match=run)
        by_id = {r.player_id: r.rating for r in results.ratings}

        assert by_id["random"] == pytest.approx(0.0, abs=1e-9)
        assert by_id["strong"] > by_id["weak"] > by_id["random"]

    def test_results_serialize_and_reload(self, tmp_path):
        registry = registry_of("a", "b")
        results = run_tournament(
            registry, "connect4", games_per_pair=10, run_match=self.stub_match()
        )
        path = tmp_path / "tournament.json"
        results.save(path)

        payload = json.loads(path.read_text())
        assert payload["schema_version"] == 1
        assert payload["env_id"] == "connect4"
        assert len(payload["matches"]) == 1
        assert len(payload["ratings"]) == 2

    def test_table_ranks_strongest_first(self):
        registry = registry_of("a", "b", "c")
        results = run_tournament(
            registry, "connect4", games_per_pair=10, run_match=self.stub_match()
        )
        lines = results.table().splitlines()

        assert lines[0].startswith("player")
        ranked = [line.split()[0] for line in lines[2:]]
        ratings = {r.player_id: r.rating for r in results.ratings}
        assert ranked == sorted(ranked, key=lambda pid: ratings[pid], reverse=True)

    def test_every_pair_gets_the_same_seed(self):
        seeds = []

        def run(*, player1, player2, env_id, num_games, verbose, seed):
            del player1, player2, env_id, verbose
            seeds.append(seed)
            return SimpleNamespace(
                games_played=num_games, player1_wins=5, player2_wins=5, draws=0
            )

        run_tournament(registry_of("a", "b", "c"), "connect4", seed=1234, run_match=run)

        assert seeds == [1234, 1234, 1234]
