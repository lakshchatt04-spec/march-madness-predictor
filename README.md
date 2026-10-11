# march-madness-physics

Simulate the men's NCAA tournament and come out with *probabilities*, not just
picks.

**Status: work in progress.** The core pipeline (fit -> simulate -> backtest)
runs and is tested, but the modelling and the analysis are still being
developed.

## Goal

A bracket pool rewards picking winners; a sportsbook, a seeding study or any
serious "how good is this team" question needs `P(team X wins it all)`. The
goal here is a model whose probabilities are actually *calibrated* - when it
says 70%, the team should win about 70% of the time - and a backtest rigorous
enough to tell whether that is true.

The thermodynamics framing is the organising idea: each team is a particle with
a small vector of end-of-season state variables, and tournament outcomes are
sampled from that state rather than hand-tuned.

## How it works

1. **Ingest** (`src/data_loader.py`) - read Kaggle's regular-season results and
   build each team's state: a sequential Elo rating, three-point attempt rate
   (`p3ar`), and variance of per-game offensive efficiency.
2. **Fit** (`src/calibration.py`) - a dyadic ridge regression on game margins,
   anchored to the Elo ratings, that estimates relative team strengths, the
   Elo-to-points conversion, home-court advantage, and a per-team score
   volatility `sigma_i = sigma_0 + alpha*z_p3ar + beta*z_effvar`.
3. **Match up** (`src/matchup.py`) - a Gaussian margin model gives
   `P(i beats j) = Phi((Elo_i - Elo_j + HCA) / (elo_per_point * tau_ij))`,
   where `tau_ij` combines both teams' volatilities. A latent strength offset is
   drawn once per team per tournament, so a hot or cold run spans the whole
   bracket instead of resetting every game.
4. **Simulate** (`src/bracket.py`, `src/simulate.py`) - the bracket graph comes
   from Kaggle's seeds/slots files, then Monte Carlo plays it out many times
   and aggregates title and per-round probabilities plus the ensemble
   observables described below.
5. **Score** (`src/backtest.py`, `tools/backtest.py`) - walk-forward backtest:
   each evaluation season is predicted using only earlier seasons, then scored
   with log loss and Brier against their trivial baselines, plus reliability
   tables.

## Ensemble observables

Beyond picking winners, `SimulationResult` reports summary statistics of the
*simulated ensemble* - they describe the shape of the model's title
distribution, not how good the forecast is (that is the backtest's job, via
log loss and Brier).

- **Championship entropy** - Shannon entropy of the title distribution,
  `H = -sum_i P_i ln P_i`, in nats. It is zero when one team wins every
  simulation and `ln F` when the title is spread evenly over the whole field.
  The normalized version `h = H / ln F` sits in `[0, 1]`: low means the model
  sees a few clear contenders, high means the title looks wide open.
- **Concentration** - the same picture in different units. `P(max)` is the
  favourite's title probability, `P(top 3)` the three front-runners summed,
  `C = sum_i P_i^2` the Herfindahl index (from `1/F` for an even field to 1
  for a certain champion), and `N_eff = 1/C` the effective number of
  contenders - how many teams would have to share the title evenly to produce
  the same concentration. A settled championship has low entropy and high
  concentration; an open one the reverse.
- **Round entropy** - the entropy of the marginal distribution over which
  teams are present at the start of each round, normalized by `ln F`. The
  first round sits at 1 (everyone plays), and later rounds fall as the field
  narrows toward the finalists. It is a per-round marginal - the uncertainty
  about who is still alive - not the entropy of whole tournament paths.

Monte Carlo error: each run is split into independent batches (20 by default)
and the standard error of every observable is the batch-to-batch spread
divided by `sqrt(n_batches)`, reported as `value +/- se` at the end of
`SimulationResult.summary()`. The batching only *counts* - it never draws
random numbers - so simulated results are identical for any batch count.

`SimulationResult.observables()` returns everything as a dict
(`H_champ`, `h_champ`, `P_max`, `P_top3`, `C`, `N_eff`, `H_r{N}`, `h_r{N}`),
and `SimulationResult.standard_errors()` returns their standard errors.

## Getting started

```bash
pip install -e ".[dev]"
```

Download the data from the
[Kaggle competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
and put the `march-machine-learning-mania-2026/` folder somewhere above the
repo root (the tools search upwards for it).

```bash
# Simulate one season's bracket
python tools/simulate_season.py --season 2025 --sims 20000

# Walk-forward backtest over several seasons
python tools/backtest.py --seasons 2021 2022 2023 2024 2025
```

Checks:

```bash
pytest          # 336 tests
ruff check .
mypy
```

## Layout

| Path | What it is |
| --- | --- |
| `src/data_loader.py` | CSV ingestion and feature engineering |
| `src/models.py` | `ParticleTeam` state and volatility parameters |
| `src/calibration.py` | dyadic fit, latent-fraction estimate |
| `src/matchup.py` | win probability and latent strength draws |
| `src/bracket.py` | bracket graph from seeds/slots |
| `src/simulate.py` | Monte Carlo tournament runner |
| `src/backtest.py` | scoring metrics and ground truth |
| `tools/` | runnable CLIs: simulate, backtest, comparisons |
| `tests/` | pytest suite |

## Where it's going

- More seasons in the backtest and tighter calibration reporting
- Feature work on volatility and the latent-strength term
- A written-up results section once the metrics settle
