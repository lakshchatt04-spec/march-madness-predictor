"""Tests for ensemble entropy, concentration, and Monte Carlo standard errors.

The observables describe the shape of the simulated title distribution - how
open or settled the championship looks under the model - not how good the
forecast is (the backtest scores that).  The load-bearing checks here are that
each statistic agrees with an independent calculation from the raw counts,
that its mathematical bounds hold, and that batch-based error estimation
never changes any simulated count.
"""

from __future__ import annotations

import math
import statistics

import pytest

from src.bracket import REGIONS, TournamentBracket
from src.data_loader import ParticleTeam
from src.matchup import MatchupParams
from src.models import VolatilityParams
from src.simulate import SimulationResult, simulate_bracket, simulate_seasons

F = 64


def _team(team_id: int, elo: float) -> ParticleTeam:
    return ParticleTeam(
        team_id=team_id,
        team_name=f"T{team_id}",
        elo_rating=elo,
        p3ar=0.38,
        efficiency_variance=0.02,
    )


def _flat_bracket(season: int = 2025) -> TournamentBracket:
    seeds = {}
    for region in REGIONS:
        for n in range(1, 17):
            seeds[f"{region}{n:02d}"] = len(seeds) + 1
    slots = {}
    for region in REGIONS:
        for i in range(1, 9):
            slots[f"R1{region}{i}"] = (
                f"{region}{(i - 1) * 2 + 1:02d}",
                f"{region}{(i - 1) * 2 + 2:02d}",
            )
        for i in range(1, 5):
            slots[f"R2{region}{i}"] = (
                f"R1{region}{(i - 1) * 2 + 1}",
                f"R1{region}{(i - 1) * 2 + 2}",
            )
        slots[f"R3{region}1"] = (f"R2{region}1", f"R2{region}2")
        slots[f"R3{region}2"] = (f"R2{region}3", f"R2{region}4")
        slots[f"R4{region}1"] = (f"R3{region}1", f"R3{region}2")
    slots["R5WX"] = ("R4W1", "R4X1")
    slots["R5YZ"] = ("R4Y1", "R4Z1")
    slots["R6CH"] = ("R5WX", "R5YZ")
    return TournamentBracket(season, seeds, slots)


@pytest.fixture
def bracket() -> TournamentBracket:
    return _flat_bracket()


@pytest.fixture
def field() -> dict[int, ParticleTeam]:
    return {tid: _team(tid, 1500.0 + 60.0 * ((tid % 8) - 3.5)) for tid in range(1, 65)}


@pytest.fixture
def runaway() -> dict[int, ParticleTeam]:
    teams = {tid: _team(tid, 1500.0) for tid in range(2, 65)}
    teams[1] = _team(1, 2000.0)
    return teams


def _params(**kw) -> MatchupParams:
    return MatchupParams(
        elo_per_point=kw.get("elo_per_point", 30.0),
        home_advantage_points=0.0,
        volatility=kw.get("volatility", VolatilityParams(1.0, 0.0, 0.0)),
    )


def _uniform_titles(per_team: int, field_size: int = F) -> SimulationResult:
    """Every team wins the title exactly ``per_team`` times."""
    counts = dict.fromkeys(range(1, field_size + 1), per_team)
    return SimulationResult(
        season=2025,
        n_sims=per_team * field_size,
        champion_counts=counts,
        round_counts={},
        appearance_counts={},
        field_size=field_size,
    )


class TestChampionshipEntropy:
    def test_agrees_with_independent_calculation(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=600, seed=21)
        probs = [c / res.n_sims for c in res.champion_counts.values()]
        expected = -sum(p * math.log(p) for p in probs)
        assert res.championship_entropy() == pytest.approx(expected)
        assert res.observables()["H_champ"] == pytest.approx(expected)

    def test_certain_champion_has_zero_entropy(self) -> None:
        res = SimulationResult(
            season=2025,
            n_sims=10,
            champion_counts={1: 10},
            round_counts={},
            appearance_counts={},
            field_size=F,
        )
        assert res.championship_entropy() == 0.0
        assert res.normalized_championship_entropy() == 0.0

    def test_uniform_titles_reach_the_maximum(self) -> None:
        res = _uniform_titles(4)
        assert res.championship_entropy() == pytest.approx(math.log(F))
        assert res.normalized_championship_entropy() == pytest.approx(1.0)

    def test_zero_probability_team_is_ignored(self) -> None:
        """A zero count must not produce a log(0) term."""
        res = SimulationResult(
            season=2025,
            n_sims=10,
            champion_counts={1: 0, 2: 10},
            round_counts={},
            appearance_counts={},
            field_size=F,
        )
        assert res.championship_entropy() == 0.0
        assert res.observables()["H_champ"] == 0.0

    def test_entropy_stays_within_field_bounds(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=500, seed=31)
        assert res.field_size == F
        assert 0.0 <= res.championship_entropy() <= math.log(F) + 1e-12
        assert 0.0 <= res.normalized_championship_entropy() <= 1.0 + 1e-12

    def test_unknown_field_size_is_inferred_from_appearances(self) -> None:
        """Only four teams ever win, but all 64 play: field must be 64."""
        res = SimulationResult(
            season=2025,
            n_sims=16,
            champion_counts=dict.fromkeys(range(1, 5), 4),
            round_counts={},
            appearance_counts={1: dict.fromkeys(range(1, F + 1), 16)},
        )
        assert res.normalized_championship_entropy() == pytest.approx(
            math.log(4) / math.log(64)
        )


class TestConcentration:
    def test_single_champion_is_fully_concentrated(self) -> None:
        res = SimulationResult(
            season=2025,
            n_sims=10,
            champion_counts={1: 10},
            round_counts={},
            appearance_counts={},
            field_size=F,
        )
        assert res.championship_concentration() == {
            "P_max": 1.0,
            "P_top3": 1.0,
            "C": 1.0,
            "N_eff": 1.0,
        }

    def test_uniform_field_matches_closed_form(self) -> None:
        conc = _uniform_titles(4).championship_concentration()
        assert conc["P_max"] == pytest.approx(1.0 / F)
        assert conc["P_top3"] == pytest.approx(3.0 / F)
        assert conc["C"] == pytest.approx(1.0 / F)
        assert conc["N_eff"] == pytest.approx(F)

    def test_bounds_hold_on_a_real_run(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=500, seed=41)
        conc = res.championship_concentration()
        assert conc["N_eff"] == pytest.approx(1.0 / conc["C"])
        assert 1.0 / F <= conc["C"] <= 1.0
        assert 1.0 <= conc["N_eff"] <= F + 1e-9
        assert conc["P_max"] <= conc["P_top3"] <= min(1.0, 3.0 * conc["P_max"]) + 1e-12

    def test_dominant_team_means_one_effective_contender(
        self, bracket, runaway
    ) -> None:
        res = simulate_bracket(bracket, runaway, _params(), n_sims=400, seed=11)
        assert res.championship_concentration()["N_eff"] < 1.05

    def test_empty_result_has_no_concentration(self) -> None:
        res = SimulationResult(
            season=2025,
            n_sims=0,
            champion_counts={},
            round_counts={},
            appearance_counts={},
        )
        assert res.championship_concentration() == {}


class TestRoundEntropy:
    def test_first_round_is_uniform_and_maximal(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=300, seed=51)
        assert res.round_entropy(1) == pytest.approx(math.log(F))
        assert res.normalized_round_entropy(1) == pytest.approx(1.0)
        assert res.observables()["H_r1"] == pytest.approx(math.log(F))

    def test_agrees_with_independent_calculation(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=400, seed=53)
        counts = res.appearance_counts[3]
        total = sum(counts.values())
        expected = -sum((c / total) * math.log(c / total) for c in counts.values())
        assert res.round_entropy(3) == pytest.approx(expected)
        assert res.observables()["H_r3"] == pytest.approx(expected)

    def test_unrecorded_round_returns_zero(self) -> None:
        res = SimulationResult(
            season=2025,
            n_sims=5,
            champion_counts={1: 5},
            round_counts={},
            appearance_counts={},
            field_size=F,
        )
        assert res.round_entropy(4) == 0.0
        assert res.normalized_round_entropy(4) == 0.0

    def test_round_entropies_are_normalised_by_field(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=400, seed=57)
        obs = res.observables()
        rounds = sorted(k[3:] for k in obs if k.startswith("h_r"))
        assert rounds
        for round_no in rounds:
            assert 0.0 <= obs[f"h_r{round_no}"] <= 1.0 + 1e-12

    def test_later_rounds_are_more_certain(self, bracket, field) -> None:
        """The first round is a uniform full field; the final cannot be."""
        res = simulate_bracket(bracket, field, _params(), n_sims=600, seed=59)
        assert res.round_entropy(6) < res.round_entropy(1)


class TestMonteCarloStandardErrors:
    def test_error_matches_manual_batch_spread(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=2000, seed=61)
        errors = res.standard_errors()
        assert errors
        manual = statistics.stdev(
            [batch["H_champ"] for batch in res.batch_observables]
        ) / math.sqrt(len(res.batch_observables))
        assert errors["H_champ"] == pytest.approx(manual)
        assert set(errors) == set(res.observables())
        assert all(e >= 0.0 and math.isfinite(e) for e in errors.values())
        assert 0.0 < errors["H_champ"] < 0.5

    def test_single_batch_reports_no_error(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=200, seed=63, n_batches=1)
        assert len(res.batch_observables) == 1
        assert res.standard_errors() == {}

    def test_batching_never_changes_counts(self, bracket, field) -> None:
        one = simulate_bracket(bracket, field, _params(), n_sims=800, seed=67, n_batches=1)
        many = simulate_bracket(bracket, field, _params(), n_sims=800, seed=67, n_batches=20)
        assert one.champion_counts == many.champion_counts
        assert one.appearance_counts == many.appearance_counts
        assert one.round_counts == many.round_counts

    def test_uneven_split_records_every_sim(self, bracket, field) -> None:
        """83 sims across 20 batches leaves no batch empty and no sim lost."""
        res = simulate_bracket(bracket, field, _params(), n_sims=83, seed=69, n_batches=20)
        assert len(res.batch_observables) == 20
        assert all("H_champ" in batch for batch in res.batch_observables)
        assert sum(res.champion_counts.values()) == 83
        assert sum(res.appearance_counts[1].values()) == 83 * F

    def test_nonpositive_batches_are_rejected(self, bracket, field) -> None:
        with pytest.raises(ValueError, match="n_batches must be >= 1"):
            simulate_bracket(bracket, field, _params(), n_sims=10, n_batches=0)

    def test_zero_sims_still_rejected_first(self, bracket, field) -> None:
        with pytest.raises(ValueError, match="n_sims must be >= 1"):
            simulate_bracket(bracket, field, _params(), n_sims=0, n_batches=0)


class TestSummaryReporting:
    def test_summary_lists_the_observables(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=400, seed=71)
        text = res.summary({t: f"T{t}" for t in range(1, 65)})
        for label in (
            "Championship entropy",
            "Normalized entropy",
            "P(max)",
            "P(top 3)",
            "Herfindahl concentration",
            "Effective contenders",
            "Normalized round entropy",
        ):
            assert label in text
        assert "+/-" in text

    def test_summary_omits_errors_without_batches(self, bracket, field) -> None:
        res = simulate_bracket(bracket, field, _params(), n_sims=100, seed=73, n_batches=1)
        assert "+/-" not in res.summary({})

    def test_empty_result_summarises_without_observables(self) -> None:
        res = SimulationResult(
            season=2025,
            n_sims=0,
            champion_counts={},
            round_counts={},
            appearance_counts={},
        )
        text = res.summary({})
        assert "2025: 0 sims" in text
        assert "observables" not in text


class TestMultiSeason:
    def test_batching_passes_through(self, bracket, field) -> None:
        b2 = TournamentBracket(2024, bracket.seeds, bracket.slots)
        out = simulate_seasons(
            {2024: b2, 2025: bracket}, field, _params(), n_sims=120, seed=81, n_batches=6
        )
        for res in out.values():
            assert len(res.batch_observables) == 6
            assert res.standard_errors()
