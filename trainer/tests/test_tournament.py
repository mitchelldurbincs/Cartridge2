"""Tests for versioned round-robin tournament profiles and ratings."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from trainer.algorithms.alphazero_board_v1 import ALGORITHM_ID
from trainer.registry import (
    PlayerRecord,
    PlayerRegistry,
    artifact_contract_for,
    checkpoint_player_id,
    make_random_player,
    random_player_id,
)
from trainer.storage.publisher import (
    ArtifactValidationError,
    BlobDescriptorV1,
    CheckpointManifestV1,
    CheckpointRef,
)
from trainer.tournament import (
    MatchResult,
    fit_ratings,
    playable_field,
    run_tournament,
    warn_about_deterministic_players,
)

REGISTERED_AT = "2026-08-02T12:00:00"
CONFIG_SHA256 = "c" * 64
_REFERENCES = {}


def profile(env_id: str = "connect4"):
    return artifact_contract_for(env_id, ALGORITHM_ID)


def model_record(
    source_path,
    *,
    env_id: str = "connect4",
    temperature: float = 0.2,
    simulations: int = 0,
) -> PlayerRecord:
    contract = profile(env_id)
    source_path = str(source_path)
    onnx = BlobDescriptorV1.from_bytes(f"onnx:{source_path}".encode())
    learner = BlobDescriptorV1.from_bytes(f"learner:{source_path}".encode())
    manifest = CheckpointManifestV1(
        profile=contract.profile,
        step=100,
        parent_checkpoint_id=None,
        config_sha256=CONFIG_SHA256,
        onnx=onnx,
        learner_state=learner,
    )
    model_root = Path(source_path).parent / "models"
    reference = CheckpointRef(
        checkpoint_id=manifest.checkpoint_id,
        manifest=manifest,
        onnx_path=model_root / "blobs" / "sha256" / f"{onnx.sha256}.onnx",
        learner_state_path=model_root / "blobs" / "sha256" / f"{learner.sha256}.pt",
    )
    _REFERENCES[reference.checkpoint_id] = reference
    return PlayerRecord(
        id=checkpoint_player_id(
            env_id=env_id,
            env_contract_version=contract.env_contract_version,
            algorithm_id=ALGORITHM_ID,
            checkpoint_id=reference.checkpoint_id,
            simulations=simulations,
            temperature=temperature,
        ),
        env_id=env_id,
        env_contract_version=contract.env_contract_version,
        algorithm_id=ALGORITHM_ID,
        model_contract=contract.model_contract,
        model_artifact_schema_version=contract.model_artifact_schema_version,
        kind="model",
        registered_at=REGISTERED_AT,
        checkpoint_id=reference.checkpoint_id,
        onnx_path=str(reference.onnx_path),
        simulations=simulations,
        temperature=temperature,
        step=manifest.step,
    )


def registry_of(directory, *names: str, baseline: bool = False) -> PlayerRegistry:
    records = [make_random_player("connect4", ALGORITHM_ID)] if baseline else []
    for name in names:
        records.append(model_record(directory / f"{name}.onnx"))
    return PlayerRegistry(records)


@pytest.fixture(autouse=True)
def artifact_resolver(monkeypatch):
    _REFERENCES.clear()

    def resolve(*, checkpoint_id, onnx_path, contract):
        del contract
        try:
            reference = _REFERENCES[checkpoint_id]
        except KeyError as exc:
            raise ArtifactValidationError("checkpoint artifact is missing") from exc
        if str(reference.onnx_path) != onnx_path:
            raise ArtifactValidationError("ONNX path mismatch")
        return reference

    resolver = MagicMock(side_effect=resolve)
    monkeypatch.setattr("trainer.registry._resolve_registered_checkpoint", resolver)
    return resolver


def match(p1: str, p2: str, p1_wins: int, p2_wins: int, draws: int = 0) -> MatchResult:
    return MatchResult(
        player1=p1,
        player2=p2,
        games=p1_wins + p2_wins + draws,
        player1_wins=p1_wins,
        player2_wins=p2_wins,
        draws=draws,
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
        ratings = fit_ratings(
            ["model", "random"],
            [match("model", "random", 18, 2)],
            anchor="random",
        )
        assert ratings["random"] == pytest.approx(0.0, abs=1e-9)
        assert ratings["model"] > 0

    def test_without_anchor_mean_is_zero_and_undefeated_is_finite(self):
        ratings = fit_ratings(["a", "b"], [match("a", "b", 20, 0)])
        assert sum(ratings.values()) == pytest.approx(0.0, abs=1e-9)
        assert all(abs(value) < 10_000 for value in ratings.values())

    def test_results_do_not_depend_on_match_order(self):
        matches = [
            match("a", "b", 12, 8),
            match("b", "c", 15, 5),
            match("a", "c", 17, 3),
        ]
        forward = fit_ratings(["a", "b", "c"], matches)
        backward = fit_ratings(["a", "b", "c"], list(reversed(matches)))
        for player_id in forward:
            assert forward[player_id] == pytest.approx(backward[player_id], abs=1e-6)

    def test_unknown_match_players_are_rejected(self):
        with pytest.raises(ValueError, match="absent from player_ids.*ghost"):
            fit_ratings(["a", "b"], [match("a", "ghost", 20, 0)])

    def test_unknown_anchor_is_rejected(self):
        with pytest.raises(ValueError, match="anchor 'ghost'.*player_ids"):
            fit_ratings(["a", "b"], [match("a", "b", 10, 10)], anchor="ghost")

    def test_duplicate_player_ids_are_rejected(self):
        with pytest.raises(ValueError, match="player_ids.*duplicates"):
            fit_ratings(["a", "a"], [])

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("games", -1),
            ("player1_wins", True),
            ("player2_wins", 1.5),
            ("draws", float("nan")),
            ("draws", float("inf")),
        ],
    )
    def test_invalid_match_counts_are_rejected(self, field_name, value):
        result = match("a", "b", 10, 10)
        setattr(result, field_name, value)

        with pytest.raises(ValueError, match=f"{field_name}.*non-negative integer"):
            fit_ratings(["a", "b"], [result])

    def test_games_must_match_outcome_counts(self):
        result = match("a", "b", 10, 10)
        result.games += 1

        with pytest.raises(ValueError, match="games must equal"):
            fit_ratings(["a", "b"], [result])

    @pytest.mark.parametrize(
        "prior_games",
        [0.0, -1.0, True, float("nan"), float("inf"), float("-inf")],
    )
    def test_invalid_prior_is_rejected(self, prior_games):
        with pytest.raises(ValueError, match="prior_games.*finite positive"):
            fit_ratings(["a", "b"], [], prior_games=prior_games)

    def test_self_matches_are_rejected(self):
        with pytest.raises(ValueError, match="two distinct players"):
            fit_ratings(["a"], [match("a", "a", 1, 0)])

    def test_empty_field_is_empty(self):
        assert fit_ratings([], []) == {}


class TestPlayableField:
    def test_missing_checkpoint_is_fatal(self, tmp_path):
        present = model_record(tmp_path / "present.onnx")
        missing = model_record(tmp_path / "missing.onnx")
        registry = PlayerRegistry(
            [
                make_random_player("connect4", ALGORITHM_ID),
                present,
                missing,
                make_random_player("tictactoe", ALGORITHM_ID),
            ]
        )
        _REFERENCES.pop(missing.checkpoint_id)

        with pytest.raises(ArtifactValidationError, match="missing"):
            playable_field(
                registry,
                env_id="connect4",
                env_contract_version=profile().env_contract_version,
                algorithm_id=ALGORITHM_ID,
            )

    def test_present_invalid_checkpoint_is_fatal(self, tmp_path, artifact_resolver):
        path = tmp_path / "invalid.onnx"
        # Construct the registry first, then simulate replacement/corruption
        # before the tournament's own boundary validation.
        registry = PlayerRegistry([model_record(path)])
        artifact_resolver.side_effect = ArtifactValidationError("wrong metadata")

        with pytest.raises(ArtifactValidationError, match="wrong metadata"):
            playable_field(
                registry,
                env_id="connect4",
                env_contract_version=profile().env_contract_version,
                algorithm_id=ALGORITHM_ID,
            )


class TestDeterministicWarning:
    def model(self, path, temperature=0.0, simulations=0):
        return model_record(path, temperature=temperature, simulations=simulations)

    def test_warns_for_multiple_deterministic_models(self, tmp_path, caplog):
        players = [
            self.model(tmp_path / "a.onnx"),
            self.model(tmp_path / "b.onnx"),
            self.model(tmp_path / "c.onnx"),
        ]
        assert warn_about_deterministic_players(players) == [p.id for p in players]
        assert "deterministic" in caplog.text

    def test_sampling_or_search_prevents_warning(self, tmp_path, caplog):
        players = [
            self.model(tmp_path / "sampled.onnx", temperature=0.2),
            self.model(tmp_path / "searching.onnx", simulations=50),
        ]
        assert warn_about_deterministic_players(players) == []
        assert "deterministic" not in caplog.text

    def test_random_baseline_is_never_flagged(self):
        assert (
            warn_about_deterministic_players([make_random_player("connect4", ALGORITHM_ID)]) == []
        )


class TestRunTournament:
    def stub_match(self, table=None):
        table = table or {}

        def run(*, player1, player2, algorithm_id, env_id, num_games, verbose, seed):
            assert algorithm_id == ALGORITHM_ID
            assert env_id == "connect4"
            del verbose, seed
            key = (player1.name, player2.name)
            wins = table.get(key, num_games // 2)
            return SimpleNamespace(
                games_played=num_games,
                player1_wins=wins,
                player2_wins=num_games - wins,
                draws=0,
            )

        return run

    @pytest.mark.parametrize(
        "games_per_pair",
        [0, -1, True, 1.5, 1 << 32],
    )
    def test_rejects_games_outside_positive_u32(self, games_per_pair):
        run_match = MagicMock()

        with pytest.raises(ValueError, match="games_per_pair.*positive u32"):
            run_tournament(
                PlayerRegistry(),
                "connect4",
                ALGORITHM_ID,
                games_per_pair=games_per_pair,
                run_match=run_match,
            )

        run_match.assert_not_called()

    @pytest.mark.parametrize("seed", [-1, True, 1.5, 1 << 64])
    def test_rejects_seeds_outside_nonnegative_u64(self, seed):
        run_match = MagicMock()

        with pytest.raises(ValueError, match="seed.*nonnegative u64"):
            run_tournament(
                PlayerRegistry(),
                "connect4",
                ALGORITHM_ID,
                seed=seed,
                run_match=run_match,
            )

        run_match.assert_not_called()

    def test_rejects_seed_schedule_that_overflows_u64(self):
        run_match = MagicMock()

        with pytest.raises(ValueError, match="seed plus the game index exceeds u64"):
            run_tournament(
                PlayerRegistry(),
                "connect4",
                ALGORITHM_ID,
                games_per_pair=2,
                seed=(1 << 64) - 1,
                run_match=run_match,
            )

        run_match.assert_not_called()

    @pytest.mark.parametrize(
        ("games_per_pair", "seed"),
        [(1, (1 << 64) - 1), ((1 << 32) - 1, 0)],
    )
    def test_accepts_valid_rust_wire_boundaries(self, tmp_path, games_per_pair, seed):
        results = run_tournament(
            registry_of(tmp_path, "a", "b"),
            "connect4",
            ALGORITHM_ID,
            games_per_pair=games_per_pair,
            seed=seed,
            run_match=self.stub_match(),
        )

        assert results.games_per_pair == games_per_pair
        assert results.seed == seed
        assert results.matches[0].games == games_per_pair

    def test_plays_every_pairing_once(self, tmp_path):
        results = run_tournament(
            registry_of(tmp_path, "a", "b", "c", "d"),
            "connect4",
            ALGORITHM_ID,
            games_per_pair=10,
            run_match=self.stub_match(),
        )
        assert len(results.matches) == 6
        assert len({(m.player1, m.player2) for m in results.matches}) == 6

    def test_needs_at_least_two_players(self, tmp_path):
        with pytest.raises(ValueError, match="at least 2"):
            run_tournament(
                registry_of(tmp_path, "lonely"),
                "connect4",
                ALGORITHM_ID,
                run_match=self.stub_match(),
            )

    def test_tallies_are_symmetric(self, tmp_path):
        results = run_tournament(
            registry_of(tmp_path, "a", "b", "c"),
            "connect4",
            ALGORITHM_ID,
            games_per_pair=10,
            run_match=self.stub_match(),
        )
        for rating in results.ratings:
            assert rating.games == 20
            assert rating.wins + rating.losses + rating.draws == 20
        assert sum(r.wins for r in results.ratings) == sum(r.losses for r in results.ratings)

    def test_ratings_are_anchored_to_versioned_random_baseline(self, tmp_path):
        weak_record = model_record(tmp_path / "weak.onnx")
        strong_record = model_record(tmp_path / "strong.onnx")
        registry = PlayerRegistry(
            [
                make_random_player("connect4", ALGORITHM_ID),
                weak_record,
                strong_record,
            ]
        )
        weak_name = weak_record.to_player().name
        strong_name = strong_record.to_player().name
        favourites = {
            frozenset({"Random", weak_name}): (weak_name, 16),
            frozenset({"Random", strong_name}): (strong_name, 19),
            frozenset({weak_name, strong_name}): (strong_name, 15),
        }

        def run(*, player1, player2, algorithm_id, env_id, num_games, verbose, seed):
            assert algorithm_id == ALGORITHM_ID
            del env_id, verbose, seed
            winner, wins = favourites[frozenset({player1.name, player2.name})]
            player1_wins = wins if player1.name == winner else num_games - wins
            return SimpleNamespace(
                games_played=num_games,
                player1_wins=player1_wins,
                player2_wins=num_games - player1_wins,
                draws=0,
            )

        results = run_tournament(
            registry,
            "connect4",
            ALGORITHM_ID,
            games_per_pair=20,
            run_match=run,
        )
        ratings = {rating.player_id: rating.rating for rating in results.ratings}
        baseline = random_player_id(
            env_id="connect4",
            env_contract_version=profile().env_contract_version,
            algorithm_id=ALGORITHM_ID,
        )
        weak = weak_record.id
        strong = strong_record.id
        assert ratings[baseline] == pytest.approx(0.0, abs=1e-9)
        assert ratings[strong] > ratings[weak] > ratings[baseline]

    def test_results_serialize_complete_profile_lineage(self, tmp_path):
        results = run_tournament(
            registry_of(tmp_path, "a", "b"),
            "connect4",
            ALGORITHM_ID,
            games_per_pair=10,
            run_match=self.stub_match(),
        )
        path = tmp_path / "tournament.json"
        results.save(path)

        payload = json.loads(path.read_text())
        assert payload["schema_version"] == 5
        assert payload["env_id"] == "connect4"
        assert payload["env_contract_version"] == profile().env_contract_version
        assert payload["algorithm_id"] == ALGORITHM_ID
        assert payload["model_contract"] == profile().model_contract
        assert payload["model_artifact_schema_version"] == profile().model_artifact_schema_version
        assert len(payload["matches"]) == 1
        assert len(payload["ratings"]) == 2

    def test_table_ranks_strongest_first(self, tmp_path):
        results = run_tournament(
            registry_of(tmp_path, "a", "b", "c"),
            "connect4",
            ALGORITHM_ID,
            games_per_pair=10,
            run_match=self.stub_match(),
        )
        ranked = [line.split()[0] for line in results.table().splitlines()[2:]]
        ratings = {r.player_id: r.rating for r in results.ratings}
        assert ranked == sorted(ranked, key=lambda pid: ratings[pid], reverse=True)

    def test_every_pair_gets_same_seed_and_profile(self, tmp_path):
        calls = []

        def run(*, player1, player2, algorithm_id, env_id, num_games, verbose, seed):
            del player1, player2, verbose
            calls.append((algorithm_id, env_id, seed))
            player1_wins = num_games // 2
            return SimpleNamespace(
                games_played=num_games,
                player1_wins=player1_wins,
                player2_wins=num_games - player1_wins,
                draws=0,
            )

        run_tournament(
            registry_of(tmp_path, "a", "b", "c"),
            "connect4",
            ALGORITHM_ID,
            seed=1234,
            run_match=run,
        )
        assert calls == [(ALGORITHM_ID, "connect4", 1234)] * 3
