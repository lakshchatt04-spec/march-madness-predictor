"""Tests for the ``noise_scale`` knob on :class:`MatchupParams`.

``noise_scale`` divides the deterministic margin (the Elo gap plus the site
effect) and nothing else - the latent drift and the volatility scale are
untouched, so a value ``n`` replaces ``Phi(mu / tau)`` with
``Phi(deterministic / (n * tau) + drift / tau)``.  It is the "lambda" of the
design notes: the fitted value ``1.0`` is a bit-for-bit no-op, and larger
values flatten every win probability toward 0.5.

Two properties carry the whole design:

*   ``noise_scale = 1.0`` changes exactly nothing, so a run at scale 1 is
    indistinguishable from the fitted model.
*   Larger scales flatten every win probability toward 0.5, but the random
    draws (latent field and per-game uniforms) are untouched - which is why
    runs across scales share common random numbers, and why the variance
    budget ``sigma^2 phi + sigma^2 (1 - phi)`` never changes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.bracket import REGIONS, TournamentBracket
from src.data_loader import ParticleTeam
from src.matchup import (
    MatchupParams,
    Venue,
    draw_latent_field,
    expected_margin_points,
    normal_cdf,
    win_probability,
)
from src.models import VolatilityParams
from src.simulate import simulate_bracket, simulate_seasons

N_TEAMS = 64


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
    """A field with a real spread, so results vary run to run."""
    return {tid: _team(tid, 1500.0 + 60.0 * ((tid % 8) - 3.5)) for tid in range(1, 65)}


def _params(**kw) -> MatchupParams:
    return MatchupParams(
        elo_per_point=kw.get("elo_per_point", 30.0),
        home_advantage_points=kw.get("home_advantage_points", 0.0),
        volatility=kw.get("volatility", VolatilityParams(1.0, 0.0, 0.0)),
        latent_fraction=kw.get("latent_fraction", 0.0),
        noise_scale=kw.get("noise_scale", 1.0),
    )


class TestScaleOneIsNoOp:
    def test_win_probability_unchanged_at_one(self, field) -> None:
        strong = _team(1, 1700.0)
        weak = _team(2, 1500.0)
        assert win_probability(
            strong, weak, params=_params(noise_scale=1.0)
        ) == win_probability(strong, weak, params=_params())

    def test_matchup_equality_at_one(self) -> None:
        assert MatchupParams() == MatchupParams(noise_scale=1.0)

    def test_bracket_run_is_bitwise_identical_at_one(self, bracket, field) -> None:
        baseline = simulate_bracket(
            bracket, field, _params(), n_sims=300, seed=11, n_batches=20
        )
        scaled = simulate_bracket(
            bracket,
            field,
            _params(),
            n_sims=300,
            seed=11,
            n_batches=20,
            noise_scale=1.0,
        )
        assert scaled.champion_counts == baseline.champion_counts
        assert scaled.appearance_counts == baseline.appearance_counts
        assert scaled.round_counts == baseline.round_counts
        assert scaled.observables() == baseline.observables()

    def test_latent_run_is_bitwise_identical_at_one(self, bracket, field) -> None:
        baseline = simulate_bracket(
            bracket,
            field,
            _params(latent_fraction=0.5),
            n_sims=200,
            seed=13,
            use_latent=True,
            latent_seed=13,
        )
        scaled = simulate_bracket(
            bracket,
            field,
            _params(latent_fraction=0.5),
            n_sims=200,
            seed=13,
            use_latent=True,
            latent_seed=13,
            noise_scale=1.0,
        )
        assert scaled.champion_counts == baseline.champion_counts
        assert scaled.appearance_counts == baseline.appearance_counts


class TestFlattensOdds:
    def test_larger_scale_flattens_toward_half(self, field) -> None:
        strong = _team(1, 1700.0)
        weak = _team(2, 1500.0)
        p_base = win_probability(strong, weak, params=_params())
        p_flatter = win_probability(strong, weak, params=_params(noise_scale=2.0))
        p_steeper = win_probability(strong, weak, params=_params(noise_scale=0.5))
        assert 0.5 < p_flatter < p_base
        assert p_base < p_steeper

    def test_equal_teams_stay_at_half_for_any_scale(self) -> None:
        a = _team(1, 1500.0)
        b = _team(2, 1500.0)
        for scale in (0.25, 0.5, 1.0, 2.0, 3.0):
            assert win_probability(a, b, params=_params(noise_scale=scale)) == pytest.approx(
                0.5
            )

    def test_matches_analytic_scaled_cdf(self, field) -> None:
        strong = _team(1, 1700.0)
        weak = _team(2, 1500.0)
        params = _params()
        tau = math.hypot(1.0, 1.0)
        x = 200.0 / params.elo_per_point
        for scale in (0.5, 1.0, 2.0):
            expected = normal_cdf(x / (scale * tau))
            got = win_probability(
                strong, weak, params=_params(noise_scale=scale)
            )
            assert got == pytest.approx(expected)

    def test_matches_analytic_scaled_cdf_with_home_advantage(self, field) -> None:
        strong = _team(1, 1700.0)
        weak = _team(2, 1500.0)
        params = _params(home_advantage_points=4.0)
        tau = math.hypot(1.0, 1.0)
        x = 200.0 / params.elo_per_point + 4.0
        for scale in (0.5, 1.0, 2.0):
            expected = normal_cdf(x / (scale * tau))
            got = win_probability(
                strong, weak, params=_params(home_advantage_points=4.0, noise_scale=scale),
                venue=Venue.HOME,
            )
            assert got == pytest.approx(expected)


class TestLatentUntouched:
    def test_drift_is_not_divided_by_noise_scale(self) -> None:
        a = _team(1, 1620.0)
        b = _team(2, 1500.0)
        latent = (2.0, 0.0)
        p = _params()
        det = (1620.0 - 1500.0) / p.elo_per_point
        base = expected_margin_points(a, b, params=p, latent=latent)
        scaled = expected_margin_points(a, b, params=_params(noise_scale=2.0), latent=latent)
        assert base == pytest.approx(det + 2.0)
        assert scaled == pytest.approx(det / 2.0 + 2.0)
        # The drift contribution is exactly identical in both.
        assert (scaled - det / 2.0) == pytest.approx(2.0)

    def test_conditional_win_probability_uses_unscaled_latent(self) -> None:
        """A latent edge stays a fixed +points shift at every scale."""
        a = _team(1, 1620.0)
        b = _team(2, 1500.0)
        latent = (2.0, 0.0)
        phi = 0.5
        x = (1620.0 - 1500.0) / 30.0
        # tau_cond = hypot(1, 1) * sqrt(1 - phi) = 1.0
        tau_cond = math.hypot(1.0, 1.0) * math.sqrt(1.0 - phi)
        for scale in (0.5, 1.0, 2.0):
            expected = normal_cdf((x / scale + 2.0) / tau_cond)
            got = win_probability(
                a, b, params=_params(latent_fraction=phi, noise_scale=scale), latent=latent
            )
            assert got == pytest.approx(expected)


class TestTournamentLevel:
    def test_bigger_scale_makes_field_flatter(self, bracket, field) -> None:
        base = simulate_bracket(bracket, field, _params(), n_sims=800, seed=9)
        loud = simulate_bracket(
            bracket, field, _params(noise_scale=2.0), n_sims=800, seed=9
        )
        assert loud.championship_entropy() > base.championship_entropy()
        assert loud.championship_concentration()["P_max"] < base.championship_concentration()[
            "P_max"
        ]

    def test_noise_scale_kwarg_equals_preconfigured_params(self, bracket, field) -> None:
        via_kwarg = simulate_bracket(
            bracket, field, _params(), n_sims=400, seed=7, noise_scale=2.0
        )
        via_params = simulate_bracket(
            bracket, field, _params(noise_scale=2.0), n_sims=400, seed=7
        )
        assert via_kwarg.champion_counts == via_params.champion_counts
        assert via_kwarg.appearance_counts == via_params.appearance_counts

    def test_simulate_seasons_threads_noise_scale(self, bracket, field) -> None:
        b2 = _flat_bracket(2024)
        out = simulate_seasons(
            {2024: b2, 2025: bracket},
            field,
            _params(),
            n_sims=120,
            seed=3,
            noise_scale=2.0,
        )
        expected = simulate_bracket(
            bracket,
            field,
            _params(noise_scale=2.0),
            n_sims=120,
            seed=3 + 2025,
        )
        assert out[2025].champion_counts == expected.champion_counts

    def test_latent_field_draws_are_identical_across_scales(self, field) -> None:
        """Common random numbers: the same latent field under any noise_scale."""
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        p_a = _params(latent_fraction=0.5, noise_scale=1.0)
        p_b = _params(latent_fraction=0.5, noise_scale=3.0)
        assert draw_latent_field(field, rng_a, params=p_a) == draw_latent_field(
            field, rng_b, params=p_b
        )


class TestValidation:
    @pytest.mark.parametrize("bad", [0.0, -1.0, -0.5, float("nan"), float("inf")])
    def test_invalid_noise_scale_is_rejected(self, bad) -> None:
        with pytest.raises(ValueError, match="noise_scale"):
            MatchupParams(noise_scale=bad)
