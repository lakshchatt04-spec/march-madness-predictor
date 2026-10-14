"""Lambda (noise_scale) sweep over the six experiment seasons.

Fits each season walk-forward on 2003..season-1, then simulates at a grid of
noise_scale values with shared seeds (common random numbers) and records the
ensemble observables.  Emits a CSV of numbers plus README-ready PNG figures.

This is the experiment harness described during the design review: CRN by
default, lambda=1 column doubles as the baseline, six 68-team seasons.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.bracket import load_bracket  # noqa: E402
from src.calibration import (  # noqa: E402
    build_particle_teams_from_fit,
    estimate_latent_fraction,
    fit_dyadic,
)
from src.data_loader import load_games, load_particle_teams  # noqa: E402
from src.simulate import simulate_bracket  # noqa: E402
from tools.backtest import DEFAULT_DATA  # noqa: E402

SEASONS = [2016, 2018, 2019, 2021, 2023, 2025]
GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def _fit_season(season: int):
    history = tuple(range(2003, season))
    games = load_games(DEFAULT_DATA, seasons=history)
    fit_teams = load_particle_teams(DEFAULT_DATA, seasons=history)
    fit = fit_dyadic(games, fit_teams)
    season_teams = load_particle_teams(DEFAULT_DATA, seasons=(season,))
    sim_teams = build_particle_teams_from_fit(fit, season_teams, unseen="prior")
    bracket = load_bracket(DEFAULT_DATA, season)
    phi = estimate_latent_fraction(games, fit_teams, fit)
    matchup = dataclasses.replace(fit.matchup_params(), latent_fraction=phi)
    return sim_teams, bracket, phi, matchup


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sims", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--season", type=int, nargs="+", default=SEASONS)
    p.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    for season in args.season:
        print(f"{season}: fitting on 2003-{season - 1}", flush=True)
        sim_teams, bracket, phi, matchup = _fit_season(season)
        for scale in GRID:
            result = simulate_bracket(
                bracket,
                sim_teams,
                matchup,
                n_sims=args.sims,
                seed=args.seed,
                use_latent=phi > 0.0,
                noise_scale=scale,
            )
            obs = result.observables()
            conc = result.championship_concentration()
            rows.append(
                {
                    "season": season,
                    "noise_scale": scale,
                    "phi": phi,
                    "h_champ": obs["h_champ"],
                    "P_max": conc["P_max"],
                    "P_top3": conc["P_top3"],
                    "C": conc["C"],
                    "N_eff": conc["N_eff"],
                }
            )
            print(
                f"  scale {scale:<4} h_champ={obs['h_champ']:.3f} "
                f"P_max={conc['P_max']:.3f} N_eff={conc['N_eff']:.1f}"
            )

    path = args.outdir / "noise_sweep.csv"
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")

    plot_sweep(rows, args.outdir)
    plot_rounds(args.outdir, args.sims, args.seed)
    return 0


def plot_sweep(rows: list[dict], outdir: Path) -> None:
    """Championship entropy and favourite/champ probability vs noise_scale."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
    for season in SEASONS:
        rows_s = [r for r in rows if r["season"] == season]
        x = [r["noise_scale"] for r in rows_s]
        axes[0].plot(
            x, [r["h_champ"] for r in rows_s], marker="o", ms=3, label=str(season)
        )
        axes[1].plot(
            x, [r["P_max"] for r in rows_s], marker="o", ms=3, label=str(season)
        )
    for ax in axes:
        ax.axvline(1.0, color="gray", ls=":", lw=1)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.3)
    axes[0].set_xlabel("noise_scale (lambda)")
    axes[0].set_ylabel("normalized championship entropy  h")
    axes[1].set_xlabel("noise_scale (lambda)")
    axes[1].set_ylabel("favourite title probability  P(max)")
    fig.suptitle("Raising noise_scale flattens the title race")
    fig.tight_layout()
    out = outdir / "noise_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_rounds(outdir: Path, sims: int, seed: int) -> None:
    """Round-entropy curve for the most recent season at lambda 1 vs 2."""
    season = max(SEASONS)
    sim_teams, bracket, phi, matchup = _fit_season(season)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for scale, color, label in (
        (1.0, "#1f77b4", "fitted model  (noise_scale = 1)"),
        (2.0, "#d62728", "doubled noise  (noise_scale = 2)"),
    ):
        result = simulate_bracket(
            bracket,
            sim_teams,
            matchup,
            n_sims=sims,
            seed=seed,
            use_latent=phi > 0.0,
            noise_scale=scale,
        )
        obs = result.observables()
        rounds = sorted(int(k[3:]) for k in obs if k.startswith("h_r"))
        ax.plot(rounds, [obs[f"h_r{r}"] for r in rounds], marker="o", ms=4,
                color=color, label=label)
    ax.set_xlabel("tournament round  (1 = first round, 6 = final)")
    ax.set_ylabel("round entropy  h_r  (normalized)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = outdir / "round_entropy.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
