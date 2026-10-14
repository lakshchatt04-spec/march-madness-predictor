"""Controlled experiment: recency-weighted dyadic fitting (`half_life`).

This is a *measurement* harness, not a production model.  The shipped default
is still the unweighted, all-history fit (``half_life = inf``); this tool exists
only to decide whether giving recent games more weight would be an improvement.

Protocol (fixed before looking at the held-out seasons):

* Selection (dev) seasons: **2015-2019**.  2015 is the earliest season whose
  fit window is at least 12 seasons long, which is what the latent-fraction
  estimator needs; 2020 was cancelled and the 2021 source file is defective, so
  neither can be used.
* Grid: ``half_life in {2, 4, 8, 16, inf}``.  ``inf`` is the current model.
  ``16`` is only run if ``8`` is a strict interior improvement over both ``4``
  and ``inf`` (otherwise the curve is flat in that region and 16 is dropped).
* Among latent-on settings, choose the half-life with the lowest mean dev
  **title log loss**; ties go to the larger half-life.
* Guardrail (a rule of thumb, **not** a significance test): reject any half-life
  whose mean dev **win log loss** exceeds the unweighted mean by more than one
  sample standard deviation (``ddof=1``) of the unweighted per-season win log
  loss.
* Held-out seasons **2022-2025** are scored once, after the choice is locked,
  as four models: A ``(inf, latent on)``, B ``(chosen, latent on)``,
  C ``(inf, latent off)``, D ``(chosen, latent off)``.

Usage::

    python tools/recency_experiment.py select
    python tools/recency_experiment.py evaluate --half-life 8
    python tools/recency_experiment.py diagnose --half-life 4 --seasons 2022 2023 2024 2025
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest import IncompleteTournamentError  # noqa: E402
from src.bracket import load_bracket, seed_number  # noqa: E402
from src.calibration import fit_dyadic  # noqa: E402
from src.data_loader import (  # noqa: E402
    load_games,
    load_particle_teams,
    load_teams,
)
from tools.backtest import (  # noqa: E402
    DEFAULT_DATA,
    MIN_HISTORY,
    evaluate_season,
)

DEV_SEASONS = (2015, 2016, 2017, 2018, 2019)
HELD_OUT_SEASONS = (2022, 2023, 2024, 2025)
GRID = (2.0, 4.0, 8.0, 16.0, math.inf)
TIE_EPS = 1e-9


def _namespace(
    half_life: float, *, latent: bool, sims: int, seed: int, first_fit: int,
    data: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        data=data,
        seasons=[],
        first_fit_season=first_fit,
        half_life=half_life,
        trailing=None,
        sims=sims,
        seed=seed,
        latent=latent,
        top=5,
    )


def _run_config(
    seasons: tuple[int, ...],
    half_life: float,
    *,
    latent: bool,
    sims: int,
    seed: int,
    first_fit: int,
    data: Path,
) -> list[dict[str, float]]:
    args = _namespace(
        half_life, latent=latent, sims=sims, seed=seed, first_fit=first_fit, data=data
    )
    rows: list[dict[str, float]] = []
    for season in seasons:
        try:
            metrics, _, _, _, _ = evaluate_season(season, args)
        except IncompleteTournamentError as exc:
            print(f"  {season}: SKIPPED ({exc})", file=sys.stderr)
            continue
        rows.append(metrics)
    return rows


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


def _label(half_life: float) -> str:
    return "inf" if math.isinf(half_life) else f"{half_life:g}"


def _fmt_half_life(half_life: float) -> str:
    return "inf" if math.isinf(half_life) else f"{half_life:>3.0f}"


# --------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------


def cmd_select(args: argparse.Namespace) -> int:
    print("Recency-weighting selection on dev seasons "
          f"{DEV_SEASONS[0]}-{DEV_SEASONS[-1]} (never the held-out seasons)")
    print(f"grid: {', '.join(_label(h) for h in GRID)}   sims={args.sims} seed={args.seed}")
    print()

    results: dict[float, list[dict[str, float]]] = {}

    def run(half_life: float) -> None:
        print(f"H={_fmt_half_life(half_life)}: fitting {DEV_SEASONS[0]}"
              f"..{DEV_SEASONS[-1]} ...", flush=True)
        results[half_life] = _run_config(
            DEV_SEASONS, half_life, latent=True, sims=args.sims, seed=args.seed,
            first_fit=args.first_fit_season, data=args.data,
        )

    # 16 is a refine step, so it is only worth running once 4, 8 and inf exist.
    for half_life in GRID:
        if half_life != 16.0:
            run(half_life)
    if args.force_refine or (
        _mean(results[8.0], "title_ll") < _mean(results[4.0], "title_ll")
        and _mean(results[8.0], "title_ll") < _mean(results[math.inf], "title_ll")
    ):
        run(16.0)
    else:
        print("H=16 dropped: dev title log loss is flat between H=8 and H=inf "
              "(no interior minimum at 8 to refine)")
    ordered = [h for h in GRID if h in results]
    print()
    print("dev-season means (latent on)")
    print(f"  {'H':>4s} {'title_ll':>9s} {'win_ll':>8s} {'adv_brier':>10s} "
          f"{'top_p':>6s} {'hit':>5s} {'phi':>6s}")
    for half_life in ordered:
        rows = results[half_life]
        print(
            f"  {_fmt_half_life(half_life):>4s} "
            f"{_mean(rows, 'title_ll'):>9.3f} "
            f"{_mean(rows, 'win_ll'):>9.3f} "
            f"{_mean(rows, 'adv_brier'):>10.4f} "
            f"{_mean(rows, 'top_p'):>6.1%} {_mean(rows, 'top_hit'):>6.1%} "
            f" phi={_label(half_life)}: {_mean(results[half_life], 'phi'):.3f}"
        )
    print()
    for half_life in ordered:
        phis = [r["phi"] for r in results[half_life]]
        print(f"  phi by season, H={_label(half_life)}: "
              + ", ".join(f"{s}:{p:.3f}" for s, p in zip(DEV_SEASONS, phis, strict=True)))

    # Guardrail: deviate from the unweighted per-season win log loss by more
    # than one sample SD rejects the setting.
    base_win = [r["win_ll"] for r in results[math.inf]]
    base_mean = sum(base_win) / len(base_win)
    base_sd = float(np.std(base_win, ddof=1)) if len(base_win) > 1 else 0.0
    threshold = base_mean + base_sd
    print()
    print(f"guardrail: mean dev win log loss must be <= {threshold:.4f} "
          f"(unweighted {base_mean:.4f} + 1 SD {base_sd:.4f})")

    eligible: list[float] = []
    for half_life, rows in results.items():
        latent_on = all(r["phi"] > 0.0 for r in rows)
        m_win = _mean(rows, "win_ll")
        reason = None
        if not latent_on:
            reason = "latent off (phi=0)"
        elif m_win > threshold:
            reason = f"guardrail (win log loss {m_win:.4f})"
        if reason is None:
            eligible.append(half_life)
        else:
            print(f"  reject H={_label(half_life)}: {reason}")

    if not eligible:
        print("  no setting passed; keeping the unweighted default (H=inf)")
        return 0

    best_ll = min(_mean(results[h], "title_ll") for h in eligible)
    winners = [h for h in eligible if _mean(results[h], "title_ll") <= best_ll + TIE_EPS]
    chosen = max(winners)
    print()
    print(f"selected half_life = {_label(chosen)}  "
          f"(dev mean title log loss {best_ll:.4f}, win log loss "
          f"{_mean(results[chosen], 'win_ll'):.4f})")
    if math.isinf(chosen):
        print("  -> recency weighting does NOT beat the all-history model on dev")
    print()
    print("locked. evaluate held-out seasons with:")
    print(f"  python tools/recency_experiment.py evaluate --half-life {_label(chosen)}")
    return 0


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    h = args.half_life
    configs = [
        ("A  (inf, latent on)", math.inf, True),
        (f"B  ({_label(h)}, latent on)", h, True),
        ("C  (inf, latent off)", math.inf, False),
        (f"D  ({_label(h)}, latent off)", h, False),
    ]
    print(f"Held-out evaluation on {HELD_OUT_SEASONS[0]}-{HELD_OUT_SEASONS[-1]}, "
          f"half_life={_label(h)}  sims={args.sims} seed={args.seed}")
    print()
    summary: list[tuple[str, list[dict[str, float]]]] = []
    for name, half_life, latent in configs:
        print(f"{name}")
        rows = _run_config(
            HELD_OUT_SEASONS, half_life, latent=latent, sims=args.sims,
            seed=args.seed, first_fit=args.first_fit_season, data=args.data,
        )
        summary.append((name, rows))
        for season, r in zip(HELD_OUT_SEASONS, rows, strict=True):
            print(f"  {season}  title {r['title_ll']:.3f} | win {r['win_ll']:.3f} | "
                  f"adv {r['adv_brier']:.4f} | reach {r['reach_brier']:.4f} | "
                  f"phi {r['phi']:.3f} | top {r['top_p']:.1%} "
                  f"{'[HIT]' if r['top_hit'] else ''}")
        print()

    print("held-out means")
    print(f"  {'model':<26s} {'title_ll':>9s} {'win_ll':>8s} {'adv':>8s} "
          f"{'reach':>8s} {'top_p':>7s} {'hit':>6s}")
    for name, rows in summary:
        print(f"  {name:<26s} {_mean(rows, 'title_ll'):>9.3f} "
              f"{_mean(rows, 'win_ll'):>8.3f} {_mean(rows, 'adv_brier'):>8.4f} "
              f"{_mean(rows, 'reach_brier'):>8.4f} {_mean(rows, 'top_p'):>7.1%} "
              f"{_mean(rows, 'top_hit'):>6.1%}")
    print()
    a = summary[0][1]
    b = summary[1][1]
    d_ll = _mean(b, "title_ll") - _mean(a, "title_ll")
    d_win = _mean(b, "win_ll") - _mean(a, "win_ll")
    print(f"recency effect (B - A): title log loss {d_ll:+.3f}, "
          f"win log loss {d_win:+.3f}  (negative = recency helps)")
    return 0


# --------------------------------------------------------------------------
# diagnose
# --------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(np.asarray(a, float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, float)).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def cmd_diagnose(args: argparse.Namespace) -> int:
    names = load_teams(args.data)
    for season in sorted(args.seasons):
        history = tuple(range(args.first_fit_season, season))
        games = load_games(args.data, seasons=history)
        fit_teams = load_particle_teams(args.data, seasons=history)
        season_teams = load_particle_teams(args.data, seasons=(season,))
        bracket = load_bracket(args.data, season)
        seed_of = {int(t): seed_number(s) for s, t in bracket.seeds.items()}
        tids = sorted(seed_of)
        seed = np.array([seed_of[t] for t in tids], float)
        elo = np.array([season_teams[t].elo_rating for t in tids])

        fits = {"inf": fit_dyadic(games, fit_teams)}
        if not math.isinf(args.half_life):
            fits[_label(args.half_life)] = fit_dyadic(
                games, fit_teams, half_life=args.half_life
            )

        print(f"===== {season}  (fit {history[0]}-{history[-1]}, "
              f"field {len(tids)}) =====")
        for label, fit in fits.items():
            s_fit = np.array([fit.strengths.get(t, 0.0) for t in tids])
            print(f"  H={label:<4s} spearman(fit, -seed) {_spearman(s_fit, -seed):+.3f}"
                  f"   spearman(fit, elo) {_spearman(s_fit, elo):+.3f}"
                  f"   spearman(elo, -seed) {_spearman(elo, -seed):+.3f}")
        print(f"  top 8 by fitted strength (H={_label(args.half_life) if not math.isinf(args.half_life) else 'inf'}):")
        ref = fits[_label(args.half_life)] if not math.isinf(args.half_life) else fits["inf"]
        s_fit = np.array([ref.strengths.get(t, 0.0) for t in tids])
        for i in np.argsort(-s_fit)[:8]:
            t = tids[i]
            print(f"    {names.get(t, t):<18} seed {int(seed[i]):>2}  "
                  f"fit {s_fit[i]:>6.2f}  elo {elo[i]:>7.1f}")
        print()
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--sims", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--first-fit-season", type=int, default=MIN_HISTORY)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("select", help="run the dev-season grid and pick a half-life")
    _common(ps)
    ps.add_argument("--force-refine", action="store_true",
                    help="always try H=16 even if the dev curve looks flat")
    ps.set_defaults(func=cmd_select)

    pe = sub.add_parser("evaluate", help="score the four models on held-out seasons")
    _common(pe)
    pe.add_argument("--half-life", type=float, required=True)
    pe.set_defaults(func=cmd_evaluate)

    pd_ = sub.add_parser("diagnose", help="strength-vs-seed diagnostics per season")
    _common(pd_)
    pd_.add_argument("--half-life", type=float, default=math.inf)
    pd_.add_argument("--seasons", type=int, nargs="+",
                     default=list(HELD_OUT_SEASONS))
    pd_.set_defaults(func=cmd_diagnose)

    args = p.parse_args(argv)
    return args.func(args)



if __name__ == "__main__":
    raise SystemExit(main())
