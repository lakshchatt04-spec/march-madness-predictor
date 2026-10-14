"""Tests for recency-weighted dyadic fitting (`half_life`).

The experiment this guards is deliberately narrow: an exponential, mean-one
reweighting of historical games, off by default.  Three things must hold and
are checked here:

* the weight calculation is exactly ``2 ** (-(s_max - s) / H)``, mean 1, and
  only ever depends on season *gaps* (no absolute-season leak);
* ``half_life = inf`` is bit-for-bit the previous, unweighted implementation
  (golden values captured from the pre-change code);
* a finite half-life actually down-weights old games in the fit.

Nothing here tests the *quality* of recency weighting - that is an open
empirical question, not a property.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.calibration import (
    _hetero_loglik,
    _recency_weights,
    _solve_ridge_with_home,
    fit_dyadic,
)
from src.models import ParticleTeam

# --------------------------------------------------------------------------
# Synthetic fixture (deterministic; mirrors tests/test_calibration._simulate)
# --------------------------------------------------------------------------


def _synth(seed: int = 0, n_games: int = 1500, n_teams: int = 40, n_seasons: int = 6):
    rng = np.random.default_rng(seed)
    ids = np.arange(1100, 1100 + n_teams)
    strength = rng.normal(0.0, 8.0, n_teams)
    z = rng.normal(size=n_teams)
    w = rng.normal(size=n_teams)
    sigma = 7.4 + 0.5 * z + 0.2 * w
    teams = {
        int(t): ParticleTeam(
            int(t), f"T{t}", 1500.0 + strength[i] * 30.0,
            0.35 + z[i] * 0.05, 0.03 + w[i] * 0.005,
        )
        for i, t in enumerate(ids)
    }
    wi = rng.integers(0, n_teams, n_games)
    li = rng.integers(0, n_teams, n_games)
    keep = wi != li
    wi, li = wi[keep], li[keep]
    loc = rng.choice(["H", "A", "N"], wi.size, p=[0.4, 0.4, 0.2])
    home = np.where(loc == "H", 1.0, np.where(loc == "A", -1.0, 0.0))
    season = rng.integers(0, n_seasons, wi.size)
    margin = (
        strength[wi] - strength[li] + 3.0 * home
        + rng.normal(0.0, np.sqrt(sigma[wi] ** 2 + sigma[li] ** 2))
    )
    flip = margin < 0.0
    wi, li = np.where(flip, li, wi), np.where(flip, wi, li)
    loc = np.where(flip, np.where(loc == "H", "A", np.where(loc == "A", "H", "N")), loc)
    margin = np.abs(margin)
    games = pd.DataFrame({
        "WTeamID": ids[wi], "LTeamID": ids[li],
        "WTeamScore": np.rint(72.0 + margin / 2.0).astype(int),
        "LTeamScore": np.rint(72.0 - margin / 2.0).astype(int),
        "WLoc": loc, "Season": 2000 + season,
    })
    return teams, games


# Captured from the pre-weighting implementation on `_synth()` (see the
# experiment notes).  `half_life=inf` must reproduce these exactly.
GOLDEN_STRENGTHS = {
    1100: 1.0400192068392002, 1101: -2.8606933687602747, 1102: 4.502732834349227,
    1103: 3.0217961763890044, 1104: -2.737819653019559, 1105: 3.3999854325534202,
    1106: 12.846960759965798, 1107: 8.829787980077171, 1108: -4.014334972761344,
    1109: -11.717770663347316, 1110: -4.510826504219805, 1111: -0.16507741153528524,
    1112: -19.289006145310765, 1113: -0.20910870043132634, 1114: -8.650785028782742,
    1115: -3.320452173834166, 1116: -3.627643644967078, 1117: -1.21743602134263,
    1118: 5.543840310510975, 1119: 9.960580667271538, 1120: 0.17183319183765236,
    1121: 9.246489132962232, 1122: -3.228604107635589, 1123: 2.0489392406704052,
    1124: 7.736411956012863, 1125: 2.646571514776985, 1126: -5.24246748325785,
    1127: -8.32599261927953, 1128: -4.500508308729436, 1129: 2.810244497469889,
    1130: -8.478392902730043, 1131: -0.8693422073110985, 1132: -0.7361626152596866,
    1133: 5.500887690474148, 1134: 1.7253367950101617, 1135: 1.826039878164362,
    1136: -6.56450576075529, 1137: 0.3715618958983534, 1138: 6.078025990588416,
    1139: 10.958885141449517,
}
GOLDEN_LOGLIK = -5504.829119648564
GOLDEN_HCA = 2.659913722827757
GOLDEN_VOL = (7.414236068725586, 0.5546875, 0.625)


# --------------------------------------------------------------------------
# 1. The weight calculation
# --------------------------------------------------------------------------


class TestRecencyWeights:
    def test_inf_is_unweighted(self) -> None:
        w = _recency_weights(np.array([2000, 2001, 2002]), math.inf)
        assert np.array_equal(w, np.ones(3))

    def test_mean_is_one(self) -> None:
        w = _recency_weights(np.arange(2000, 2010), 3.0)
        assert float(w.mean()) == pytest.approx(1.0)

    def test_latest_season_has_the_largest_weight(self) -> None:
        w = _recency_weights(np.array([2000, 2001, 2002, 2003, 2004]), 2.0)
        assert np.all(np.diff(w) > 0.0)
        assert float(w.argmax()) == 4.0

    def test_halves_over_a_half_life(self) -> None:
        seasons = np.arange(2000, 2011)
        for h in (1.0, 2.0, 4.0):
            w = _recency_weights(seasons, h)
            # a game h-seasons-newer counts twice as much as the older one, i.e.
            # the older game is down-weighted by half per half-life
            assert w[-1] / w[-1 - int(h)] == pytest.approx(2.0)

    def test_exact_ratio_formula(self) -> None:
        w = _recency_weights(np.array([2000, 2005]), 2.5)
        assert w[1] / w[0] == pytest.approx(2.0 ** (5.0 / 2.5))

    def test_invariant_to_absolute_season_shift(self) -> None:
        a = _recency_weights(np.array([1998, 1999, 2000, 2001]), 2.0)
        b = _recency_weights(np.array([2000, 2001, 2002, 2003]), 2.0)
        assert np.allclose(a, b)

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, -math.inf])
    def test_rejects_non_positive(self, bad: float) -> None:
        with pytest.raises(ValueError, match="half_life"):
            _recency_weights(np.array([2000, 2001]), bad)


# --------------------------------------------------------------------------
# 2. The weighted objective (math, against a reference solve)
# --------------------------------------------------------------------------


class TestWeightedObjective:
    def test_loglik_none_equals_all_ones(self) -> None:
        rng = np.random.default_rng(1)
        resid = rng.normal(size=50)
        tau = rng.uniform(2.0, 10.0, 50)
        assert _hetero_loglik(resid, tau) == _hetero_loglik(
            resid, tau, np.ones(50)
        )

    def test_weighted_loglik_is_the_weighted_sum(self) -> None:
        rng = np.random.default_rng(2)
        resid = rng.normal(size=40)
        tau = rng.uniform(2.0, 8.0, 40)
        w = rng.uniform(0.1, 1.0, 40)
        per_game = -0.5 * (np.log(2.0 * np.pi * tau**2) + resid**2 / tau**2)
        assert _hetero_loglik(resid, tau, w) == pytest.approx(float(np.sum(per_game * w)))

    def test_weighted_ridge_matches_reference_normal_equations(self) -> None:
        rng = np.random.default_rng(3)
        n_games, n_teams = 40, 6
        wi = rng.integers(0, n_teams, n_games)
        li = rng.integers(0, n_teams, n_games)
        clash = wi == li
        li[clash] = (li[clash] + 1) % n_teams
        design = np.zeros((n_games, n_teams))
        design[np.arange(n_games), wi] = 1.0
        design[np.arange(n_games), li] = -1.0
        home = rng.choice([-1.0, 0.0, 1.0], n_games)
        margin = rng.normal(0.0, 10.0, n_games)
        prior = rng.normal(0.0, 5.0, n_teams)
        ridge = 2.0
        w = rng.uniform(0.2, 1.0, n_games)

        s, hca = _solve_ridge_with_home(design, home, margin, prior, ridge, w)

        X = np.column_stack([design, home])
        root = np.sqrt(w)
        Xw, mw = X * root[:, None], margin * root
        penalty = np.diag([ridge] * n_teams + [0.0])
        ref = np.linalg.solve(
            Xw.T @ Xw + penalty, Xw.T @ mw + np.append(ridge * prior, 0.0)
        )
        assert s == pytest.approx(ref[:n_teams])
        assert hca == pytest.approx(ref[n_teams])

    def test_unweighted_solve_equals_plain_ridge(self) -> None:
        rng = np.random.default_rng(4)
        n_games, n_teams = 25, 5
        design = np.zeros((n_games, n_teams))
        wi = rng.integers(0, n_teams, n_games)
        li = (wi + 1 + rng.integers(0, n_teams - 1, n_games)) % n_teams
        design[np.arange(n_games), wi] = 1.0
        design[np.arange(n_games), li] = -1.0
        home = np.zeros(n_games)
        margin = rng.normal(0.0, 9.0, n_games)
        prior = rng.normal(0.0, 4.0, n_teams)
        s, hca = _solve_ridge_with_home(design, home, margin, prior, 1.0)
        gram = design.T @ design + 1.0 * np.eye(n_teams)
        ref = np.linalg.solve(gram, design.T @ margin + prior)
        assert s == pytest.approx(ref)
        assert hca == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# 3. Default invariance against the pre-change implementation
# --------------------------------------------------------------------------


class TestDefaultInvariance:
    def test_default_reproduces_golden_values(self) -> None:
        teams, games = _synth()
        fit = fit_dyadic(games, teams, ridge=1.0, elo_per_point=30.0, max_iter=30)
        assert fit.strengths == GOLDEN_STRENGTHS
        assert fit.log_likelihood == GOLDEN_LOGLIK
        assert fit.home_advantage_points == GOLDEN_HCA
        assert fit.volatility.sigma_0 == GOLDEN_VOL[0]
        assert fit.volatility.alpha == GOLDEN_VOL[1]
        assert fit.volatility.beta == GOLDEN_VOL[2]

    def test_explicit_inf_equals_default(self) -> None:
        teams, games = _synth()
        a = fit_dyadic(games, teams, ridge=1.0, elo_per_point=30.0, max_iter=30)
        b = fit_dyadic(
            games, teams, ridge=1.0, elo_per_point=30.0, max_iter=30, half_life=math.inf
        )
        assert a.strengths == b.strengths
        assert a.log_likelihood == b.log_likelihood
        assert a.volatility == b.volatility
        assert a.home_advantage_points == b.home_advantage_points

    def test_half_life_recorded_in_feature_summary(self) -> None:
        teams, games = _synth()
        fit = fit_dyadic(games, teams)
        assert fit.feature_summary["half_life"] == math.inf


# --------------------------------------------------------------------------
# 4. Temporal behaviour of the fit
# --------------------------------------------------------------------------


def _form_shift_frame() -> pd.DataFrame:
    """1100 loses by 8 early, wins by 8 late; all other teams are neutral."""
    others = list(range(1101, 1109))
    rows = []
    for season in range(2000, 2006):
        recent = season >= 2003
        for opp in others:
            if recent:
                rows.append((1100, opp, 76, 68, "N", season))
            else:
                rows.append((opp, 1100, 76, 68, "N", season))
    return pd.DataFrame(
        rows,
        columns=["WTeamID", "LTeamID", "WTeamScore", "LTeamScore", "WLoc", "Season"],
    )


class TestRecencyChangesTheFit:
    def test_recent_form_is_weighted_more_than_old(self) -> None:
        games = _form_shift_frame()
        teams = {
            t: ParticleTeam(t, f"T{t}", 1500.0, 0.35, 0.03)
            for t in range(1100, 1109)
        }
        flat = fit_dyadic(games, teams)
        recency = fit_dyadic(games, teams, half_life=1.0)
        # A team that is only good recently must be rated higher once the old
        # seasons are down-weighted.
        assert recency.strength_of(1100) > flat.strength_of(1100)

    def test_finite_half_life_needs_a_season_column(self) -> None:
        teams, games = _synth()
        with pytest.raises(ValueError, match="Season"):
            fit_dyadic(games.drop(columns=["Season"]), teams, half_life=2.0)

    def test_finite_half_life_changes_strengths(self) -> None:
        teams, games = _synth()
        flat = fit_dyadic(games, teams, max_iter=30)
        rec = fit_dyadic(games, teams, max_iter=30, half_life=2.0)
        assert flat.strengths != rec.strengths
        assert rec.converged
