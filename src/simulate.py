"""Monte Carlo tournament simulation over a validated bracket.

The bracket graph from :mod:`src.bracket` says who *can* meet.  This module
answers who *does*: it samples each matchup, advances winners through the
slots, and aggregates title probabilities over many runs.

Design notes
------------
Latent strength is drawn **once per team per simulation**, not per game.  A
team's hot (or cold) stretch spans its whole run, which is the point of a
particle model.  Resampling noise every game would make each team memoryless
and destroy the autocorrelation that makes upsets cluster.

Tournament games are neutral site, so ``venue_home_advantage`` is not applied.
A tied game is resolved by a coin flip at a seeded ``tie_prob`` rather than by
clipping the margin, so tuning tie handling never requires touching the
sampling code.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from math import fsum, log, sqrt
from random import Random
from typing import cast

import numpy as np

from src.bracket import TournamentBracket
from src.matchup import (
    MatchupParams,
    Venue,
    draw_latent_field,
    win_probability,
)
from src.models import ParticleTeam

__all__ = [
    "DEFAULT_N_BATCHES",
    "DEFAULT_TIE_PROB",
    "SimulationResult",
    "TournamentSimulator",
    "simulate_bracket",
    "simulate_seasons",
]

#: Probability a game ends tied.
#:
#: Zero by default, and zero is correct for the men's tournament: across all
#: 125,978 games in Kaggle's regular-season and tournament files there is not one
#: tie, because men's games go to overtime.  Ties are structurally impossible
#: rather than rare, so a non-zero default would inject noise that never occurs.
#: The knob remains for women's basketball, which does allow ties.
DEFAULT_TIE_PROB = 0.0

#: Independent batches each simulation is split into for estimating Monte
#: Carlo standard errors.  Purely diagnostic: batching only *counts* outcomes
#: and never draws random numbers, so no simulated result depends on it.
#: Twenty batches give 19 degrees of freedom for the sample standard
#: deviation while keeping each batch large enough (500 of the default 10,000
#: brackets) that per-batch plug-in entropies are stable.
DEFAULT_N_BATCHES = 20


def _entropy(counts: Iterable[int], total: int) -> float:
    """Shannon entropy in nats of a histogram with ``total`` observations.

    ``H = -sum_i p_i ln p_i`` with ``p_i = c_i / total``.  Counts that are
    zero (or negative, defensively) contribute nothing, so a team recorded
    with zero occurrences never produces a ``log(0)`` term.
    """
    if total <= 0:
        return 0.0
    return -sum((c / total) * log(c / total) for c in counts if c > 0)


def _normalize_entropy(value: float, field_size: int) -> float:
    """Scale an entropy to ``[0, 1]`` by the maximum possible, ``ln F``."""
    if field_size <= 1:
        return 0.0
    return value / log(field_size)


def _infer_field_size(
    appearance_counts: Mapping[int, Mapping[int, int]],
    champion_counts: Mapping[int, int],
) -> int:
    """Field size for results that did not record one: distinct teams seen."""
    teams = {team for counts in appearance_counts.values() for team in counts}
    if teams:
        return len(teams)
    return len(champion_counts)


def _ensemble_observables(
    champion_counts: Mapping[int, int],
    appearance_counts: Mapping[int, Mapping[int, int]],
    n_sims: int,
    field_size: int = 0,
) -> dict[str, float]:
    """Every ensemble observable, computed from marginal counts alone.

    Used for the full result and for each simulation batch, so standard
    errors are estimated from exactly the same estimators that get reported.
    Keys: ``H_champ`` (title entropy, nats), ``h_champ`` (entropy / ``ln F``),
    ``P_max``, ``P_top3``, ``C`` (Herfindahl concentration), ``N_eff`` (the
    effective number of contenders, ``1/C``), and ``H_r{N}`` / ``h_r{N}`` for
    each recorded round ``N``.  Returns an empty dict when ``n_sims <= 0``.
    """
    if n_sims <= 0:
        return {}
    field = field_size or _infer_field_size(appearance_counts, champion_counts)
    h_champ = _entropy(champion_counts.values(), n_sims)
    probs = [c / n_sims for c in champion_counts.values() if c > 0]
    concentration = fsum(p * p for p in probs)
    observables: dict[str, float] = {
        "H_champ": h_champ,
        "h_champ": _normalize_entropy(h_champ, field),
        "P_max": max(probs) if probs else 0.0,
        "P_top3": fsum(sorted(probs, reverse=True)[:3]),
        "C": concentration,
        "N_eff": 1.0 / concentration if concentration > 0.0 else 0.0,
    }
    for round_no in sorted(appearance_counts):
        counts = appearance_counts[round_no]
        h_round = _entropy(counts.values(), sum(counts.values()))
        observables[f"H_r{round_no}"] = h_round
        observables[f"h_r{round_no}"] = _normalize_entropy(h_round, field)
    return observables


@dataclass(frozen=True)
class SimulationResult:
    """Aggregated output of many bracket simulations.

    Attributes:
        season: Championship year simulated.
        n_sims: Number of complete brackets played.
        champion_counts: Titles won per TeamID.
        round_counts: Wins at each round per TeamID (a win *in* round N means
            the team advanced past round N).
        appearance_counts: Teams present at the start of each round.
        field_size: Teams in the tournament field, used to normalize entropy.
            Zero means unknown, in which case it is inferred from the counts.
        batch_observables: Per-batch observable estimates from independent
            chunks of the run, used only to estimate Monte Carlo standard
            errors.  Empty for hand-constructed results.
    """

    season: int
    n_sims: int
    champion_counts: Mapping[int, int]
    round_counts: Mapping[int, Mapping[int, int]]
    appearance_counts: Mapping[int, Mapping[int, int]]
    field_size: int = 0
    batch_observables: tuple[Mapping[str, float], ...] = ()

    def title_probabilities(self) -> dict[int, float]:
        """Share of simulations won by each team."""
        if not self.n_sims:
            return {}
        return {t: c / self.n_sims for t, c in self.champion_counts.items()}

    def round_win_probabilities(self, round_no: int) -> dict[int, float]:
        """Chance a team wins a round, i.e. advances beyond it."""
        return self._rates(self.round_counts.get(round_no, {}))

    def appearance_probabilities(self, round_no: int) -> dict[int, float]:
        """Chance a team is present at the start of a round."""
        return self._rates(self.appearance_counts.get(round_no, {}))

    def _rates(self, counts: Mapping[int, int]) -> dict[int, float]:
        if not self.n_sims:
            return {}
        return {t: c / self.n_sims for t, c in counts.items()}

    def observables(self) -> dict[str, float]:
        """Ensemble summary statistics computed from the simulation counts.

        See :func:`_ensemble_observables` for the key layout.  These describe
        the shape of the simulated title distribution - how open or settled
        the championship looks under the model - not the quality of the
        forecast, which is what the backtest scores.  Empty when there are no
        simulations to summarize.
        """
        return _ensemble_observables(
            self.champion_counts,
            self.appearance_counts,
            self.n_sims,
            self.field_size,
        )

    def standard_errors(self) -> dict[str, float]:
        """Monte Carlo standard error of each observable, from batch spread.

        Each entry is the sample standard deviation of the observable across
        the independent batches in ``batch_observables``, divided by the
        square root of the batch count: how much the estimate would move
        under a rerun of the same number of simulations.  Estimating errors
        this way consumes no random draws, so counts are identical for any
        batching.  Returns an empty dict when fewer than two batches were
        recorded, since a spread needs at least two samples.
        """
        batches = self.batch_observables
        if len(batches) < 2:
            return {}
        keys = set(batches[0])
        for batch in batches[1:]:
            keys &= set(batch)
        errors: dict[str, float] = {}
        for key in sorted(keys):
            values = [batch[key] for batch in batches]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
            errors[key] = sqrt(variance / len(values))
        return errors

    def championship_entropy(self) -> float:
        """Shannon entropy of the title distribution, in nats.

        ``H = -sum_i P_i ln P_i`` over title probabilities: zero when one
        team wins every simulation, ``ln F`` when the title is spread evenly
        over the whole field.
        """
        return self.observables().get("H_champ", 0.0)

    def normalized_championship_entropy(self) -> float:
        """``H_champ / ln F``, so 0 is one certain champion, 1 is uniform."""
        return self.observables().get("h_champ", 0.0)

    def championship_concentration(self) -> dict[str, float]:
        """How concentrated the title distribution is.

        Returns ``P_max`` (the favourite's title probability), ``P_top3``
        (the three largest summed), ``C`` (the Herfindahl index,
        ``sum_i P_i^2``, from ``1/F`` for an even field to 1 for a certain
        champion) and ``N_eff`` (``1/C``, the effective number of
        contenders: the count of teams that would share the title evenly to
        reproduce the same concentration).  Entropy and concentration are
        two units for the same picture - a settled championship has low
        entropy and high concentration, an open one the reverse.  Empty when
        there are no simulations to summarize.
        """
        obs = self.observables()
        if not obs:
            return {}
        return {key: obs[key] for key in ("P_max", "P_top3", "C", "N_eff")}

    def round_entropy(self, round_no: int) -> float:
        """Entropy of the round-appearance marginal, in nats.

        Over many simulations each team has a probability of being present
        at the start of round ``N``; normalized against the round's total
        appearances that gives a distribution ``A_i``, and
        ``H = -sum_i A_i ln A_i`` measures how uncertain the set of teams
        still alive there is.  This is a per-round marginal, not the entropy
        of whole tournament paths.  Unrecorded rounds return 0.
        """
        return self.observables().get(f"H_r{round_no}", 0.0)

    def normalized_round_entropy(self, round_no: int) -> float:
        """``round_entropy(round_no) / ln F``, bounded to ``[0, 1]``."""
        return self.observables().get(f"h_r{round_no}", 0.0)

    def summary(self, names: Mapping[int, str], top: int = 10) -> str:
        """Human-readable leaderboard plus ensemble observables.

        Used by tools/ and ad-hoc checks.  The observables block is appended
        only when there is something to summarize; standard errors appear as
        ``value +/- se`` when batch estimates were recorded.
        """
        rows = sorted(self.title_probabilities().items(), key=lambda kv: -kv[1])[:top]
        lines = [f"{self.season}: {self.n_sims} sims"]
        for team, prob in rows:
            lines.append(f"  {prob * 100:5.2f}%  {names.get(team, str(team))}")
        lines.extend(self._observables_lines())
        return "\n".join(lines)

    def _observables_lines(self) -> list[str]:
        obs = self.observables()
        if not obs:
            return []
        se = self.standard_errors()

        def fmt(key: str, digits: int = 4, with_error: bool = True) -> str:
            text = f"{obs[key]:.{digits}f}"
            err = se.get(key) if with_error else None
            return text if err is None else f"{text} +/- {err:.{digits}f}"

        header = "  ensemble observables"
        if se:
            header += " (+/- = Monte Carlo standard error)"
        lines = ["", f"{header}:"]
        lines.append(f"    {'Championship entropy:':<27}{fmt('H_champ')} nats")
        lines.append(f"    {'Normalized entropy:':<27}{fmt('h_champ')}")
        lines.append(f"    {'P(max):':<27}{fmt('P_max')}")
        lines.append(f"    {'P(top 3):':<27}{fmt('P_top3')}")
        lines.append(f"    {'Herfindahl concentration:':<27}{fmt('C')}")
        lines.append(f"    {'Effective contenders:':<27}{fmt('N_eff', 2)}")
        rounds = sorted(
            int(key[3:]) for key in obs if key.startswith("h_r") and key[3:].isdigit()
        )
        if rounds:
            value_parts: list[str] = []
            error_parts: list[str] = []
            for round_no in rounds:
                key = f"h_r{round_no}"
                value_parts.append(f"r{round_no} {fmt(key, with_error=False)}")
                error_parts.append(f"r{round_no} {fmt(key)}")
            lines.append(f"    {'Normalized round entropy:':<27}{'  '.join(value_parts)}")
            if se:
                lines.append(
                    f"    {'Round entropy SE:':<27}{'  '.join(error_parts)}"
                )
        return lines


@dataclass
class TournamentSimulator:
    """Simulator bound to one bracket and one set of team particles.

    Args:
        bracket: Validated bracket structure.
        teams: Particle state keyed by TeamID.
        matchup: Calibrated matchup parameters.
        tie_prob: Probability a game ends tied.
        rng: Seeded RNG, so results are reproducible.
    """

    bracket: TournamentBracket
    teams: Mapping[int, ParticleTeam]
    matchup: MatchupParams
    tie_prob: float = DEFAULT_TIE_PROB
    rng: Random = field(default_factory=Random)
    draw_latent: Callable[[], Mapping[int, float]] | None = None

    def __post_init__(self) -> None:
        missing = sorted(s for s, t in self.bracket.seeds.items() if t not in self.teams)
        if missing:
            preview = ", ".join(missing[:8])
            more = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
            raise ValueError(
                f"{len(missing)} seeded team(s) missing from ParticleTeams: "
                f"{preview}{more}"
            )

    def play_once(self) -> tuple[int, dict[int, set[int]]]:
        """Play one complete bracket.

        Returns:
            ``(champion_id, appearances)`` where ``appearances[round]`` is the
            set of teams that took part in that round.  Round 0 is the play-in.
        """
        occupants: dict[str, int] = dict(self.bracket.seeds)
        appearances: dict[int, set[int]] = {}

        # One latent draw per team per simulation, so a hot team stays hot for
        # its whole run.  Resampling per game would make upsets independent.
        latent: Mapping[int, float] = self.draw_latent() if self.draw_latent else {}

        for round_no, slots in self.bracket.rounds().items():
            appearing = {
                self._resolve(occupants, side)
                for slot in slots
                for side in self.bracket.slots[slot]
            }
            appearances[round_no] = appearing

            winners: dict[str, int] = {}
            for slot in slots:
                strong, weak = self.bracket.slots[slot]
                a = self._resolve(occupants, strong)
                b = self._resolve(occupants, weak)
                winners[slot] = self._play(a, b, latent)
            occupants.update(winners)

        champion_slot = self.champion_slot()
        return self._resolve(occupants, champion_slot), appearances

    def champion_slot(self) -> str:
        """The single slot nothing feeds into - i.e. the final."""
        referenced = {
            side
            for pair in self.bracket.slots.values()
            for side in pair
            if side in self.bracket.slots
        }
        terminal = set(self.bracket.slots) - referenced
        if len(terminal) != 1:
            raise ValueError(
                f"expected exactly one terminal slot, found {sorted(terminal)}"
            )
        return terminal.pop()

    def _resolve(self, occupants: Mapping[str, int], key: str) -> int:
        """Turn a seed or slot reference into a TeamID."""
        if key in occupants:
            return occupants[key]
        return self.bracket.seeds[key]

    def _play(self, a: int, b: int, latent: Mapping[int, float] | None = None) -> int:
        """Pick a winner.  Teams are interchangeable by symmetry, so reorder."""
        p_a = win_probability(
            self.teams[a],
            self.teams[b],
            params=self.matchup,
            venue=Venue.NEUTRAL,
            latent=(latent.get(a, 0.0), latent.get(b, 0.0)) if latent else None,
        )
        p_b = 1.0 - p_a
        fav, dog = (a, b) if p_a >= p_b else (b, a)
        p_fav = max(p_a, p_b)
        # Two independent draws: reusing one would make the non-tie branch
        # conditional on having already failed the tie test, which quietly
        # skews the winner distribution whenever tie_prob is non-zero.
        if self.tie_prob and self.rng.random() < self.tie_prob:
            return fav if self.rng.random() < 0.5 else dog
        return fav if self.rng.random() < p_fav else dog


def simulate_bracket(
    bracket: TournamentBracket,
    teams: Mapping[int, ParticleTeam],
    matchup: MatchupParams,
    *,
    n_sims: int = 10_000,
    seed: int | None = 0,
    tie_prob: float = DEFAULT_TIE_PROB,
    use_latent: bool = False,
    latent_seed: int | None = None,
    n_batches: int = DEFAULT_N_BATCHES,
    noise_scale: float | None = None,
) -> SimulationResult:
    """Play ``n_sims`` brackets and aggregate probabilities.

    Args:
        bracket: Validated bracket for one season.
        teams: Particle state keyed by TeamID.
        matchup: Calibrated matchup parameters.
        n_sims: Number of complete brackets to play.
        seed: RNG seed; pass ``None`` for non-reproducible output.
        tie_prob: Probability a game ends tied.
        use_latent: Draw per-team latent strength offsets.  Requires
            NumPy, since the draws use `normal()` not
            `normalvariate()`.
        latent_seed: RNG seed for the latent draws, kept independent
            of `seed` so changing one does not shift the other.
        n_batches: Split the run into this many independent batches (capped
            at ``n_sims``) to estimate Monte Carlo standard errors from
            batch-to-batch spread.  Purely diagnostic: the split only counts
            outcomes, never draws random numbers, so every simulated count is
            identical for any ``n_batches``.
        noise_scale: Divisor on the deterministic margin (the Elo gap plus the
            site effect), applied by
            ``dataclasses.replace(matchup, noise_scale=noise_scale)``.
            ``None`` keeps ``matchup``'s own ``noise_scale`` (the fitted model
            uses 1.0).  Larger values flatten every win probability toward 0.5
            without changing any random draw, so two runs that differ only in
            this value share identical latent fields and uniforms.

    Returns:
        Aggregated probabilities across all simulations.
    """
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")
    if n_batches < 1:
        raise ValueError(f"n_batches must be >= 1, got {n_batches}")
    if noise_scale is not None:
        matchup = replace(matchup, noise_scale=noise_scale)

    latent_rng = None
    if use_latent:
        base = 0 if seed is None else seed if latent_seed is None else latent_seed
        latent_rng = np.random.default_rng(base + bracket.season)

    sim = TournamentSimulator(
        bracket=bracket,
        teams=teams,
        matchup=matchup,
        tie_prob=tie_prob,
        rng=Random(seed),
        draw_latent=(
            (lambda: draw_latent_field(teams, latent_rng, params=matchup))
            if latent_rng is not None
            else None
        ),
    )

    # Sizes differ by at most one and every batch is non-empty, so the batch
    # standard deviation never sees an empty estimate.
    n_batches_eff = min(n_batches, n_sims)
    batch_base, extra = divmod(n_sims, n_batches_eff)
    batch_champions: list[dict[int, int]] = []
    batch_appearances: list[dict[int, dict[int, int]]] = []
    for batch_no in range(n_batches_eff):
        size = batch_base + (1 if batch_no < extra else 0)
        champions: dict[int, int] = {}
        appearances_per_round: dict[int, dict[int, int]] = {}
        for _ in range(size):
            champion, appearances = sim.play_once()
            champions[champion] = champions.get(champion, 0) + 1
            for round_no, teams_in_round in appearances.items():
                target = appearances_per_round.setdefault(round_no, {})
                for team in teams_in_round:
                    target[team] = target.get(team, 0) + 1
        batch_champions.append(champions)
        batch_appearances.append(appearances_per_round)

    champion_counts: dict[int, int] = {}
    appearance_counts: dict[int, dict[int, int]] = {}
    for champions in batch_champions:
        for team, count in champions.items():
            champion_counts[team] = champion_counts.get(team, 0) + count
    for appearances_per_round in batch_appearances:
        for round_no, counts in appearances_per_round.items():
            target = appearance_counts.setdefault(round_no, {})
            for team, count in counts.items():
                target[team] = target.get(team, 0) + count

    # Wins in round N are exactly appearances in round N+1: every team present
    # in round N plays there, and the ones who win are the ones who show up
    # next.  Subtracting appearances(N) would be wrong - it made every count
    # negative, and no test asserted the sign.
    #
    # The final has no following round, so its single win is a title.
    # Round 0, when it exists, is the play-in, and the same rule holds: a
    # play-in winner is simply the participant who reaches round 1.
    last_round = max(appearance_counts, default=0)
    round_counts: dict[int, dict[int, int]] = {}
    for round_no in sorted(appearance_counts):
        nxt = appearance_counts.get(round_no + 1, {})
        target = round_counts.setdefault(round_no, {})
        for team in appearance_counts[round_no]:
            won = (
                champion_counts.get(team, 0)
                if round_no == last_round
                else nxt.get(team, 0)
            )
            if won:
                target[team] = target.get(team, 0) + won

    batch_observables = tuple(
        _ensemble_observables(
            champions,
            appearances_per_round,
            sum(champions.values()),
            bracket.field_size,
        )
        for champions, appearances_per_round in zip(
            batch_champions, batch_appearances, strict=True
        )
    )

    return SimulationResult(
        season=bracket.season,
        n_sims=n_sims,
        champion_counts=champion_counts,
        round_counts=round_counts,
        appearance_counts=appearance_counts,
        field_size=bracket.field_size,
        batch_observables=batch_observables,
    )


def simulate_seasons(
    brackets: Mapping[int, TournamentBracket],
    teams: Mapping[int, object],
    matchup: Mapping[int, object] | MatchupParams,
    *,
    n_sims: int = 10_000,
    seed: int | None = 0,
    tie_prob: float = DEFAULT_TIE_PROB,
    use_latent: bool = False,
    latent_seed: int | None = None,
    n_batches: int = DEFAULT_N_BATCHES,
    noise_scale: float | None = None,
) -> dict[int, SimulationResult]:
    """Simulate several seasons, using an independent RNG stream per season.

    Args:
        brackets: Season to bracket.
        teams: Either `{season: {TeamID: ParticleTeam}}` for a true backtest,
            or one season-agnostic `{TeamID: ParticleTeam}}` map.  **Prefer
            the nested form.**  A single shared map silently feeds the
            end-of-season state of whichever season loaded last into every
            season, which leaks information backwards and makes a backtest
            meaningless.
        matchup: One `MatchupParams` shared by all seasons, or a
            `{season: MatchupParams}` mapping.
        n_sims: Brackets per season.
        seed: Base RNG seed; each season derives `seed + season`.
        tie_prob: Probability a game ends tied.
        use_latent: Draw per-team latent offsets.
        latent_seed: Base seed for the latent draws.  Defaults to `seed`, so
            the whole run stays reproducible from one number.
        n_batches: Batches per season for Monte Carlo standard errors; see
            :func:`simulate_bracket`.
        noise_scale: Divisor on the deterministic margin; passed through to
            :func:`simulate_bracket` for every season.  ``None`` keeps each
            season's own ``noise_scale``.

    Returns:
        `{season: SimulationResult}`.

    Raises:
        TypeError: If an entry has the wrong shape.
        ValueError: If a nested map is missing an entry for a requested season.
    """
    nested = _looks_nested(teams)

    out: dict[int, SimulationResult] = {}
    for season, bracket in sorted(brackets.items()):
        if nested:
            season_teams = teams.get(season)
            if season_teams is None:
                raise ValueError(f"season {season} missing from teams mapping")
        else:
            season_teams = teams
        season_matchup = (
            matchup.get(season) if isinstance(matchup, Mapping) else matchup
        )

        if not isinstance(season_teams, Mapping):
            raise TypeError(
                f"season {season} teams must be a mapping of TeamID to "
                f"ParticleTeam, got {type(season_teams).__name__}"
            )
        if not isinstance(season_matchup, MatchupParams):
            raise TypeError(
                f"season {season} matchup must be MatchupParams, got "
                f"{type(season_matchup).__name__}"
            )

        out[season] = simulate_bracket(
            bracket,
            cast("Mapping[int, ParticleTeam]", season_teams),
            season_matchup,
            n_sims=n_sims,
            seed=None if seed is None else seed + season,
            tie_prob=tie_prob,
            use_latent=use_latent,
            latent_seed=None if latent_seed is None else latent_seed + season,
            n_batches=n_batches,
            noise_scale=noise_scale,
        )
    return out


def _looks_nested(mapping: Mapping[int, object]) -> bool:
    """Heuristic: does this map go from season to per-season object?

    Distinguishes `{2025: {1100: ParticleTeam}}` from
    `{1100: ParticleTeam}` by inspecting the first value's type rather than
    the key range, because team ids and season numbers overlap.
    """
    for value in mapping.values():
        if isinstance(value, (ParticleTeam, MatchupParams)):
            return False
        if isinstance(value, Mapping):
            return True
    return False
