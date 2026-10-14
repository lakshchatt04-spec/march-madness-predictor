"""Tests for the seed-only baseline.

The baseline is the floor the full model has to clear, so two things matter:
the game model really is "the better seed usually wins" and nothing more, and
the analytic per-seed rates can be scored by the *unchanged* backtest metrics
through :class:`SeedBaselineResult`.  None of this checks the baseline's
quality - that is an empirical question answered by the backtest, not a
property.
"""

from __future__ import annotations

import math
import random

import pytest

from src.backtest import (
    ActualGame,
    ActualOutcome,
    Venue,
    advancement_brier,
    reach_brier,
    title_log_loss,
)
from src.baselines import (
    SeedBaselineResult,
    SeedHistory,
    SeedWinModel,
    build_seed_baseline,
    collect_seed_history,
    fit_seed_win_model,
    seed_win_log_loss,
    team_seed_numbers,
)
from src.bracket import TournamentBracket
from src.simulate import SimulationResult


def _bracket(season: int, seeds: dict[str, int]) -> TournamentBracket:
    return TournamentBracket(season=season, seeds=seeds, slots={})


def _mini_season(base: int) -> tuple[TournamentBracket, ActualOutcome]:
    """A four-team, two-round tournament the 1-seed always wins."""
    t1, t2, t3, t4 = base + 1, base + 2, base + 3, base + 4
    bracket = _bracket(base, {"W01": t1, "W02": t2, "W03": t3, "W04": t4})
    games = (
        ActualGame(1, t1, t2, t1, Venue.NEUTRAL),
        ActualGame(1, t3, t4, t3, Venue.NEUTRAL),
        ActualGame(2, t1, t3, t1, Venue.NEUTRAL),
    )
    outcome = ActualOutcome(
        season=base,
        champion=t1,
        round_teams={1: frozenset({t1, t2, t3, t4}), 2: frozenset({t1, t3})},
        bracket_teams=frozenset({t1, t2, t3, t4}),
        games=games,
    )
    return bracket, outcome


def _synthetic_history(
    n_seasons: int = 8, games_per_season: int = 400, slope: float = 0.25, seed: int = 0
) -> tuple[dict[int, TournamentBracket], dict[int, ActualOutcome]]:
    """Tournaments whose games follow a known logistic on the seed gap."""
    rng = random.Random(seed)
    brackets: dict[int, TournamentBracket] = {}
    outcomes: dict[int, ActualOutcome] = {}
    next_id = 1000
    for season in range(2003, 2003 + n_seasons):
        roster: dict[int, int] = {}
        seeds: dict[str, int] = {}
        for number in range(1, 17):
            roster[next_id] = number
            seeds[f"W{number:02d}"] = next_id
            next_id += 1
        games = []
        for _ in range(games_per_season):
            (t1, s1), (t2, s2) = rng.sample(list(roster.items()), 2)
            better, worse = (t1, t2) if s1 <= s2 else (t2, t1)
            p_better = 1.0 / (1.0 + math.exp(-slope * abs(s1 - s2)))
            winner = better if rng.random() < p_better else worse
            games.append(ActualGame(1, t1, t2, winner, Venue.NEUTRAL))
        brackets[season] = _bracket(season, seeds)
        outcomes[season] = ActualOutcome(
            season=season,
            champion=games[0].winner,
            round_teams={1: frozenset(roster)},
            bracket_teams=frozenset(roster),
            games=tuple(games),
        )
    return brackets, outcomes


class TestTeamSeedNumbers:
    def test_play_in_seeds_collapse_to_their_number(self) -> None:
        bracket = _bracket(
            2025, {"W01": 1, "W16A": 2, "W16B": 3, "Z05": 4}
        )
        assert team_seed_numbers(bracket) == {1: 1, 2: 16, 3: 16, 4: 5}


class TestSeedWinModel:
    def test_probability_is_monotone_bounded_and_symmetric(self) -> None:
        model = SeedWinModel(intercept=0.0, slope=0.3)
        assert model.probability(1, 16) > model.probability(8, 9) > 0.5
        assert model.probability(1, 16) + model.probability(16, 1) == pytest.approx(1.0)
        assert model.probability(8, 8) == pytest.approx(0.5)
        assert 0.0 < model.probability(16, 1) < 0.5

    def test_fit_recovers_a_positive_slope(self) -> None:
        brackets, outcomes = _synthetic_history(slope=0.25)
        model = fit_seed_win_model(brackets, outcomes)
        assert model.slope == pytest.approx(0.25, abs=0.05)
        assert model.intercept == pytest.approx(0.0, abs=0.1)
        assert model.probability(1, 16) > 0.9

    def test_no_spread_collapses_to_the_base_rate(self) -> None:
        # Two games, both between seeds 1 and 2, one win each: the gap carries
        # no information, so the slope is unidentified and the fit reports the
        # base rate at a zero slope.
        bracket = _bracket(2003, {"W01": 1, "W02": 2})
        games = (
            ActualGame(1, 1, 2, 1, Venue.NEUTRAL),
            ActualGame(1, 2, 1, 2, Venue.NEUTRAL),
        )
        outcome = ActualOutcome(
            season=2003,
            champion=1,
            round_teams={1: frozenset({1, 2})},
            bracket_teams=frozenset({1, 2}),
            games=games,
        )
        model = fit_seed_win_model({2003: bracket}, {2003: outcome})
        assert model.slope == 0.0
        assert model.probability(1, 2) == pytest.approx(0.5)


class TestCollectSeedHistory:
    def test_counts_titles_presence_and_advancement(self) -> None:
        bracket_a, outcome_a = _mini_season(100)
        bracket_b, outcome_b = _mini_season(200)
        history = collect_seed_history(
            {outcome_a.season: bracket_a, outcome_b.season: bracket_b},
            {outcome_a.season: outcome_a, outcome_b.season: outcome_b},
        )
        assert history.n_seasons == 2
        assert history.title_counts == {1: 2}
        # Four teams per season, so every seed is in the field twice.
        assert history.field_counts[1] == 2
        assert history.present_counts[1][1] == 2
        assert history.present_counts[1][2] == 2
        assert history.advance_counts[1][1] == 2
        assert history.advance_counts.get(2, {}).get(1, 0) == 0
        assert history.advance_counts[1][2] == 2
        assert history.advance_counts.get(3, {}).get(2, 0) == 0

    def test_title_share_is_a_distribution_with_a_prior(self) -> None:
        bracket, outcome = _mini_season(100)
        history = collect_seed_history({outcome.season: bracket}, {outcome.season: outcome})
        shares = [history.title_share(seed) for seed in range(1, 17)]
        assert sum(shares) == pytest.approx(1.0)
        assert all(share > 0.0 for share in shares)
        assert history.title_share(1) > history.title_share(16)

    def test_missing_outcomes_are_skipped(self) -> None:
        bracket, outcome = _mini_season(100)
        history = collect_seed_history({100: bracket, 101: bracket}, {100: outcome})
        assert history.n_seasons == 1


class TestSeedWinLogLoss:
    def test_a_confident_model_scores_below_a_coin_flip(self) -> None:
        bracket, outcome = _mini_season(100)
        confident = SeedWinModel(intercept=0.0, slope=5.0)
        mean, n_games = seed_win_log_loss(outcome, bracket, confident)
        assert n_games == len(outcome.games) == 3
        assert mean < math.log(2.0)

    def test_a_flat_model_scores_the_coin_flip(self) -> None:
        bracket, outcome = _mini_season(100)
        mean, _ = seed_win_log_loss(outcome, bracket, SeedWinModel(0.0, 0.0))
        assert mean == pytest.approx(math.log(2.0))


class TestSeedBaselineResult:
    def test_scores_through_the_unchanged_metrics(self) -> None:
        brackets: dict[int, TournamentBracket] = {}
        outcomes: dict[int, ActualOutcome] = {}
        for index in range(5):
            bracket, outcome = _mini_season(100 * (index + 1))
            brackets[bracket.season] = bracket
            outcomes[outcome.season] = outcome
        eval_bracket, eval_outcome = _mini_season(9999)

        baseline = build_seed_baseline(brackets, outcomes, eval_bracket)
        assert isinstance(baseline.result, SimulationResult)
        assert isinstance(baseline.result, SeedBaselineResult)

        probs = baseline.result.title_probabilities()
        assert set(probs) == set(eval_bracket.seeds.values())
        assert sum(probs.values()) == pytest.approx(1.0)
        # Seed 1 won every historical title, so it must be the top pick.
        top = max(probs, key=lambda team: probs[team])
        assert team_seed_numbers(eval_bracket)[top] == 1

        assert title_log_loss(baseline.result, eval_outcome.champion) > 0.0
        assert 0.0 <= advancement_brier(baseline.result, eval_outcome) <= 1.0
        assert 0.0 <= reach_brier(baseline.result, eval_outcome) <= 1.0
        win, n_games = baseline.win_log_loss(eval_outcome)
        assert n_games == len(eval_outcome.games)
        assert math.isfinite(win)

    def test_rounds_absent_from_history_return_no_probabilities(self) -> None:
        history = SeedHistory(
            n_seasons=1,
            title_counts={1: 1},
            present_counts={1: {1: 1}},
            advance_counts={1: {1: 1}},
            field_counts={1: 1},
        )
        assert 4 not in history.rounds()
