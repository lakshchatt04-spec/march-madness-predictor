"""Seed-only baseline forecasts for the tournament backtest.

The full model earns its keep only if it beats the obvious thing a bracket
expert does with nothing but the seed line.  This module builds that obvious
thing, using **only** information available before the evaluated season:

* **game outcomes** - a one-feature logistic regression on the seed difference,
  fitted on earlier tournaments.  Higher seeds (larger numbers) are worse, so
  the slope should come out positive;
* **title probabilities** - the historical share of titles won by each seed,
  with Laplace smoothing so a seed that has never won still gets mass, split
  evenly across the teams carrying that seed; and
* **advancement and reach** - per-seed, per-round historical rates.

The rates live in :class:`SeedBaselineResult`, a :class:`SimulationResult`
subclass backed by analytic per-seed probabilities rather than Monte Carlo
draws, so the backtest's scoring functions consume it unchanged even though
nothing was simulated.

Why the per-seed split rather than a per-team fit
-------------------------------------------------
Seeds are the one feature every bracket shares, and four teams carry the same
seed (one per region) with genuinely different strength.  A seed baseline
cannot tell them apart, which is exactly the edge the full model is supposed
to have; measuring the gap is the point of this module.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from src.backtest import ActualOutcome, IncompleteTournamentError, actual_outcome
from src.bracket import TournamentBracket, load_bracket, load_tourney_games, seed_number
from src.data_loader import DataSchemaError
from src.simulate import SimulationResult

__all__ = [
    "SEED_COUNT",
    "SeedBaseline",
    "SeedBaselineResult",
    "SeedHistory",
    "SeedWinModel",
    "build_seed_baseline",
    "collect_seed_history",
    "fit_seed_win_model",
    "load_seed_history",
    "seed_win_log_loss",
    "team_seed_numbers",
]

#: Seeds run 1-16 in every region; used to size the title prior.
SEED_COUNT = 16

#: Add-one smoothing on the title share: one pseudo-championship per seed.
TITLE_PRIOR = 1.0

#: Additive smoothing for advancement and reach rates.  Half a pseudo-event on
#: each side keeps a seed that has never (or always) survived a round off the
#: exact 0/1 boundary, which would otherwise wreck a Brier score on a handful
#: of historical teams.
RATE_PRIOR = 0.5

#: Same guard as the backtest: a probability of exactly zero would make a log
#: loss infinite.
_MIN_PROB = 1e-12


def _logistic(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    """Inverse logistic, with the endpoints nudged inward."""
    clamped = min(max(p, _MIN_PROB), 1.0 - _MIN_PROB)
    return math.log(clamped / (1.0 - clamped))


def team_seed_numbers(bracket: TournamentBracket) -> dict[int, int]:
    """Map each TeamID in a bracket to its numeric seed (1-16).

    Play-in seeds such as ``W16a`` and ``W16b`` both count as seed 16, since
    the baseline deliberately ignores which side of a play-in a team is on.
    """
    return {team: seed_number(seed) for seed, team in bracket.seeds.items()}


def _smoothed_ratio(successes: int, trials: int, prior: float = RATE_PRIOR) -> float:
    """Laplace-style rate, kept strictly inside ``(0, 1)``."""
    if trials <= 0:
        return 0.5
    return (successes + prior) / (trials + 2.0 * prior)


@dataclass(frozen=True)
class SeedHistory:
    """Historical seed outcomes accumulated over earlier tournaments.

    Attributes:
        n_seasons: Number of seasons that contributed.
        title_counts: Seed -> number of championships.
        present_counts: Seed -> round -> teams present at the start of it.
        advance_counts: Seed -> round -> teams that won that round.
        field_counts: Seed -> team-seasons in the field (denominator for reach).
    """

    n_seasons: int
    title_counts: Mapping[int, int]
    present_counts: Mapping[int, Mapping[int, int]]
    advance_counts: Mapping[int, Mapping[int, int]]
    field_counts: Mapping[int, int]

    def title_share(self, seed: int) -> float:
        """Smoothed probability a championship belongs to ``seed``.

        The prior is add-one per seed, so the shares sum to 1 over the 16
        seeds and a seed with no prior title still receives probability.
        """
        denominator = self.n_seasons + TITLE_PRIOR * SEED_COUNT
        if denominator <= 0.0:
            return 0.0
        return (self.title_counts.get(seed, 0) + TITLE_PRIOR) / denominator

    def rounds(self) -> list[int]:
        """Rounds seen anywhere in the history, ascending."""
        seen = {r for counts in self.present_counts.values() for r in counts}
        return sorted(seen)


def collect_seed_history(
    brackets: Mapping[int, TournamentBracket],
    outcomes: Mapping[int, ActualOutcome],
) -> SeedHistory:
    """Accumulate per-seed title and advancement counts over earlier seasons.

    Seasons with no matching outcome are skipped, so a defective tournament
    log never poisons the rates.
    """
    title_counts: dict[int, int] = {}
    present: dict[int, dict[int, int]] = {}
    advance: dict[int, dict[int, int]] = {}
    field_counts: dict[int, int] = {}
    n_seasons = 0

    for season, bracket in brackets.items():
        outcome = outcomes.get(season)
        if outcome is None:
            continue
        n_seasons += 1
        seeds = team_seed_numbers(bracket)
        for team in bracket.seeds.values():
            seed = seeds[team]
            field_counts[seed] = field_counts.get(seed, 0) + 1

        champion_seed = seeds[outcome.champion]
        title_counts[champion_seed] = title_counts.get(champion_seed, 0) + 1

        for round_no, teams in outcome.round_teams.items():
            advanced = outcome.advancers(round_no)
            for team in teams:
                seed = seeds[team]
                row = present.setdefault(seed, {})
                row[round_no] = row.get(round_no, 0) + 1
                if team in advanced:
                    won = advance.setdefault(seed, {})
                    won[round_no] = won.get(round_no, 0) + 1

    return SeedHistory(
        n_seasons=n_seasons,
        title_counts=title_counts,
        present_counts=present,
        advance_counts=advance,
        field_counts=field_counts,
    )


@dataclass(frozen=True)
class SeedWinModel:
    """One-feature logistic model of "the better seed wins".

    ``probability(seed_a, seed_b)`` is ``P(a beats b)``.  A team with a smaller
    seed number is the favourite, so a positive ``slope`` is the expected fit.
    """

    intercept: float
    slope: float

    def probability(self, seed_a: int, seed_b: int) -> float:
        """Probability the team with ``seed_a`` beats the team with ``seed_b``."""
        return _logistic(self.intercept + self.slope * (seed_b - seed_a))


def _fit_logistic(xs: Sequence[float], ys: Sequence[float]) -> SeedWinModel:
    """Newton-Raphson fit of ``P(y=1) = sigmoid(b0 + b1 x)``.

    Two parameters, so the observed-information matrix inverts by hand.  When
    the covariate has no spread the slope is not identified and the fit
    collapses to the base rate at ``x``.
    """
    if not xs:
        return SeedWinModel(0.0, 0.0)
    if len(set(xs)) < 2:
        base = sum(ys) / len(ys)
        return SeedWinModel(_logit(base), 0.0)

    b0 = 0.0
    b1 = 0.0
    for _ in range(100):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys, strict=True):
            p = _logistic(b0 + b1 * x)
            residual = y - p
            weight = p * (1.0 - p)
            g0 += residual
            g1 += residual * x
            h00 += weight
            h01 += weight * x
            h11 += weight * x * x
        determinant = h00 * h11 - h01 * h01
        if abs(determinant) < 1e-18:
            break
        d0 = (h11 * g0 - h01 * g1) / determinant
        d1 = (h00 * g1 - h01 * g0) / determinant
        b0 += d0
        b1 += d1
        if abs(d0) < 1e-12 and abs(d1) < 1e-12:
            break
    return SeedWinModel(b0, b1)


def fit_seed_win_model(
    brackets: Mapping[int, TournamentBracket],
    outcomes: Mapping[int, ActualOutcome],
) -> SeedWinModel:
    """Fit the seed-difference logistic on earlier tournaments' games.

    Each game contributes the gap between the two seeds and whether the better
    seed won.  Ordering every game so the better seed is first is what gives
    the fit both classes: an upset is a ``y = 0`` at a positive gap.
    """
    xs: list[float] = []
    ys: list[float] = []
    for season, bracket in brackets.items():
        outcome = outcomes.get(season)
        if outcome is None:
            continue
        seeds = team_seed_numbers(bracket)
        for game in outcome.games:
            s_a = seeds[game.team_a]
            s_b = seeds[game.team_b]
            if s_a <= s_b:
                better, gap = game.team_a, s_b - s_a
            else:
                better, gap = game.team_b, s_a - s_b
            xs.append(float(gap))
            ys.append(1.0 if better == game.winner else 0.0)
    return _fit_logistic(xs, ys)


def seed_win_log_loss(
    outcome: ActualOutcome,
    bracket: TournamentBracket,
    model: SeedWinModel,
) -> tuple[float, int]:
    """Mean ``-log p`` the seed model gave each actual winner.

    One observation per scored game, matching :func:`src.backtest.win_log_loss`,
    but using the logistic seed forecast rather than the model's analytic
    matchup forecast.
    """
    seeds = team_seed_numbers(bracket)
    total = 0.0
    count = 0
    for game in outcome.games:
        loser = game.team_b if game.winner == game.team_a else game.team_a
        p = model.probability(seeds[game.winner], seeds[loser])
        total += -math.log(max(p, _MIN_PROB))
        count += 1
    if not count:
        return float("nan"), 0
    return total / count, count


@dataclass(frozen=True)
class SeedBaselineResult(SimulationResult):
    """A :class:`SimulationResult` backed by analytic seed rates.

    The base counts are left empty and the three probability accessors are
    overridden, so downstream scoring reads the per-seed numbers directly.
    Nothing was drawn; the subclass exists only to present the rates in the
    shape the backtest already consumes.
    """

    title_probs: Mapping[int, float] = field(default_factory=dict)
    round_probs: Mapping[int, Mapping[int, float]] = field(default_factory=dict)
    appearance_probs: Mapping[int, Mapping[int, float]] = field(default_factory=dict)

    def title_probabilities(self) -> dict[int, float]:
        return dict(self.title_probs)

    def round_win_probabilities(self, round_no: int) -> dict[int, float]:
        return dict(self.round_probs.get(round_no, {}))

    def appearance_probabilities(self, round_no: int) -> dict[int, float]:
        return dict(self.appearance_probs.get(round_no, {}))


@dataclass(frozen=True)
class SeedBaseline:
    """A fitted seed baseline: probabilities plus the game-level win model."""

    result: SeedBaselineResult
    win_model: SeedWinModel
    bracket: TournamentBracket

    def win_log_loss(self, outcome: ActualOutcome) -> tuple[float, int]:
        """Mean ``-log p`` of the actual winners under the seed win model."""
        return seed_win_log_loss(outcome, self.bracket, self.win_model)


def _title_probabilities(
    bracket: TournamentBracket, history: SeedHistory
) -> dict[int, float]:
    """Smoothed seed title shares, split evenly across teams with that seed.

    Shares are renormalised over the seeds actually present, so the result is
    a proper distribution even on a partial bracket.  A real 64/68-team field
    contains every seed 1-16, so the renormalisation is exact there.
    """
    by_seed: dict[int, list[int]] = {}
    for team, seed in team_seed_numbers(bracket).items():
        by_seed.setdefault(seed, []).append(team)
    shares = {seed: history.title_share(seed) for seed in by_seed}
    total = sum(shares.values())
    probs: dict[int, float] = {}
    for seed, teams in by_seed.items():
        share = shares[seed] / total if total > 0.0 else 0.0
        for team in teams:
            probs[team] = share / len(teams)
    return probs


def _advance_probabilities(
    bracket: TournamentBracket, history: SeedHistory
) -> dict[int, dict[int, float]]:
    """Per-seed probability of winning each round, keyed by round then team."""
    seeds = team_seed_numbers(bracket)
    out: dict[int, dict[int, float]] = {}
    for round_no in history.rounds():
        row: dict[int, float] = {}
        for team, seed in seeds.items():
            present = history.present_counts.get(seed, {}).get(round_no, 0)
            won = history.advance_counts.get(seed, {}).get(round_no, 0)
            row[team] = _smoothed_ratio(won, present)
        out[round_no] = row
    return out


def _reach_probabilities(
    bracket: TournamentBracket, history: SeedHistory
) -> dict[int, dict[int, float]]:
    """Per-seed probability of reaching each round, keyed by round then team."""
    seeds = team_seed_numbers(bracket)
    out: dict[int, dict[int, float]] = {}
    for round_no in history.rounds():
        row: dict[int, float] = {}
        for team, seed in seeds.items():
            present = history.present_counts.get(seed, {}).get(round_no, 0)
            in_field = history.field_counts.get(seed, 0)
            row[team] = _smoothed_ratio(present, in_field)
        out[round_no] = row
    return out


def build_seed_baseline(
    brackets: Mapping[int, TournamentBracket],
    outcomes: Mapping[int, ActualOutcome],
    eval_bracket: TournamentBracket,
) -> SeedBaseline:
    """Fit the seed baseline on earlier seasons and apply it to one bracket."""
    history = collect_seed_history(brackets, outcomes)
    model = fit_seed_win_model(brackets, outcomes)
    result = SeedBaselineResult(
        season=eval_bracket.season,
        n_sims=0,
        champion_counts={},
        round_counts={},
        appearance_counts={},
        field_size=eval_bracket.field_size,
        title_probs=_title_probabilities(eval_bracket, history),
        round_probs=_advance_probabilities(eval_bracket, history),
        appearance_probs=_reach_probabilities(eval_bracket, history),
    )
    return SeedBaseline(result=result, win_model=model, bracket=eval_bracket)


def load_seed_history(
    data: str | Path, seasons: Sequence[int]
) -> tuple[dict[int, TournamentBracket], dict[int, ActualOutcome]]:
    """Load brackets and played-out outcomes for a set of seasons.

    Seasons whose tournament data is missing or defective (2020 was cancelled;
    2021 lacks a game) are skipped rather than raising, so the baseline can be
    fitted on whatever history actually exists.
    """
    frame = load_tourney_games(data, seasons=tuple(seasons))
    season_column = frame["Season"]
    brackets: dict[int, TournamentBracket] = {}
    outcomes: dict[int, ActualOutcome] = {}
    for season in seasons:
        games = frame.loc[season_column == season]
        if games.empty:
            continue
        try:
            bracket = load_bracket(data, season)
            outcomes[season] = actual_outcome(games, bracket)
        except (IncompleteTournamentError, DataSchemaError):
            continue
        brackets[season] = bracket
    return brackets, outcomes
