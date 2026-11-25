from __future__ import annotations
"""
run.py — Main entry point for the co-citation pipeline.

Usage:
    python run.py <seed_set_file> <citation_tsv> [options]

Options:
    --year-start  INT    Analysis window start year (default: 1996)
    --year-end    INT    Analysis window end year   (default: 2025)
    --c-min       INT    Co-citation threshold      (default: 5)
    --t-min       FLOAT  Min triple area threshold  (default: 0.1)
    --ngrid       INT    Grid integration resolution (default: 500)
    --bootstrap        Run year-block bootstrap for τ₂ CI (slow)
    --n-boot      INT    Bootstrap resamples        (default: 1000)
    --sensitivity      Run 3x3 c_min x Y0 sensitivity grid (slow)
    --out-dir     PATH   Directory for output CSV files (default: .)

Validation against the PN morphology paper:
    python run.py seed_set.txt filtered_citations_pn_v2.tsv

Expected outputs for the PN morphology set (fixed core set, start 2000, 14 pairs):
    β (free origin) = 0.31, CI [0.26, 0.39];  β (origin 1995) = 0.39
    τ₂ = 6.95 yr (window-dependent descriptor), CI [5.65, 9.26]
    τ₃ = 15.3 yr (parametric-bootstrap CI [8.2, 27.5])
    Tri-citation r = 0.955

The τ₂ and β bootstrap (--bootstrap) is the year-block bootstrap of the paper,
with years re-sorted; the number of unusable resamples is reported.
"""

import argparse
import itertools
import math
import sys
import json
import csv
from pathlib import Path
import numpy as np

# Allow running from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import seed_set as ss
from pipeline import counts, geometry as geo, metrics, fitting, cohesion


def parse_args():
    p = argparse.ArgumentParser(description='Co-citation pipeline')
    p.add_argument('seed_file',   help='Seed set .txt file')
    p.add_argument('citation_tsv', help='Filtered citation TSV')
    p.add_argument('--year-start', type=int, default=1996)
    p.add_argument('--year-end',   type=int, default=2025)
    p.add_argument('--c-min',      type=int, default=5)
    p.add_argument('--t-min',      type=float, default=0.1)
    p.add_argument('--ngrid',      type=int, default=500)
    p.add_argument('--bootstrap',  action='store_true')
    p.add_argument('--n-boot',     type=int, default=1000)
    p.add_argument('--sensitivity',action='store_true')
    p.add_argument('--out-dir',    default='.')
    return p.parse_args()


def fmt(v, decimals=2):
    return f'{v:.{decimals}f}' if v is not None and not math.isnan(v) else 'NaN'


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load seed set ──────────────────────────────────────────────────
    print("=" * 65)
    print(ss.summary(args.seed_file))
    seeds = ss.bibcodes(args.seed_file)
    seed_meta = ss.load(args.seed_file)

    # ── 2. Build count data ───────────────────────────────────────────────
    print(f"\nLoading citation data from: {args.citation_tsv}")
    print(f"Window: {args.year_start}–{args.year_end},  c_min={args.c_min}")
    data = counts.build(args.citation_tsv, seeds,
                        year_start=args.year_start,
                        year_end=args.year_end)
    print(f"  {data['n_edges']} citation edges loaded.")
    print()
    print(counts.coverage_report(data, c_min=args.c_min))

    # ── 3. Compute metrics ────────────────────────────────────────────────
    print("\nComputing metrics (geometry engine running — may take a minute)...")
    m = metrics.compute_all(data, c_min=args.c_min,
                            t_min=args.t_min, ngrid=args.ngrid)

    # ── 4. Fit exponential decays ─────────────────────────────────────────
    coh = cohesion.analyse(data, c_min=args.c_min, n_boot=0)
    fit2 = coh['exp']
    fit3 = fitting.fit_decay(m['years'], m['omega3'])

    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)

    print(f"\nΓ₂(t)  — pairwise cohesion, fixed core set "
          f"({coh['n_pairs']} pairs valid at start {coh['start']}; {coh['n_epochs']} epochs)")
    pf, pl = coh['pl_free'], coh['pl_fixed']
    if pf:
        print(f"  β (free origin)      = {fmt(pf['beta'], 3)}   t0 = {fmt(pf['t0'], 1)}   R² = {fmt(pf['r2'], 3)}")
    print(f"  β (origin {pl['origin']})    = {fmt(pl['beta'], 3)}   R² = {fmt(pl['r2'], 3)}")
    print(f"  τ₂ (window-dependent) = {fmt(fit2['tau'])} yr   c∞ = {fmt(fit2['c_inf'], 4)}   R² = {fmt(fit2['r2'], 3)}")
    print(f"  [Paper: β = 0.31, τ₂ = 6.95 yr for the PN morphology set]")
    print(f"  (all-valid-pairs series in the CSV is for reference; fits use the fixed core set)")

    if fit3:
        print(f"\nΩ₃(t)  — triple mean-overlap attenuation  (units: 1/area)")
        print(f"  τ₃ (point estimate)  = {fmt(fit3['tau'])} yr")
        print(f"  c∞                   = {fmt(fit3['c_inf'], 4)}")
        print(f"  R²                   = {fmt(fit3['r2'], 3)}")
        print(f"  [Paper parametric-bootstrap value: τ₃ = 15.3 yr, CI [8.2, 27.5]]")
    else:
        print("\nΩ₃(t): fit failed (insufficient data)")

    # Mean normalised quad residual (structural 2D embedding limit)
    valid_res = [r for r in m['norm_resid_quad'] if not math.isnan(r)]
    if valid_res:
        print(f"\nΓ₄(t)  — quadruple cohesion (exploratory)")
        print(f"  Mean normalised LS residual = {np.mean(valid_res):.3f} "
              f"(range {min(valid_res):.3f}–{max(valid_res):.3f})")
        print(f"  [Paper value: ~0.37, stable across epochs]")

    # Soker & Livio (1994) centrality check
    s_l94 = '1994ApJ...421..219S'
    if s_l94 in seeds:
        final_kappa = {s: m['centrality'][s][-1] for s in seeds
                       if not math.isnan(m['centrality'][s][-1])}
        if final_kappa:
            highest = max(final_kappa, key=lambda s: final_kappa[s])
            print(f"\nGeometric centrality at {m['years'][-1]}:")
            for s in sorted(final_kappa, key=lambda s: -final_kappa[s]):
                label = next((x['notes'].split(';')[0] for x in seed_meta
                              if x['bibcode'] == s), s)
                marker = ' ← highest' if s == highest else ''
                print(f"  {s}  κ={fmt(final_kappa[s], 4)}{marker}")
            if highest == s_l94:
                print("  ✓ Soker & Livio (1994) is most central — matches paper")

    # ── 5. Parametric bootstrap for τ₃ ───────────────────────────────────
    print("\nRunning parametric bootstrap for τ₃ (n=2000)...")
    bs3 = fitting.parametric_bootstrap_tau3(m['omega3'], m['years'],
                                             n_resamples=2000)
    if bs3['n_resamples'] > 0:
        print(f"  τ₃ = {fmt(fit3['tau'] if fit3 else math.nan)} yr  "
              f"95% CI [{fmt(bs3['ci_95'][0])}, {fmt(bs3['ci_95'][1])}]")
        print(f"  [Paper: τ₃ = 15.3 yr, CI [8.2, 27.5]]")

    # ── 6. Optional: year-block bootstrap for τ₂ ─────────────────────────
    if args.bootstrap:
        print(f"\nRunning year-block bootstrap for β, τ₂ (n={args.n_boot}) — slow...")
        coh = cohesion.analyse(data, c_min=args.c_min, n_boot=args.n_boot)
        b2 = coh['bootstrap']
        print(f"  usable resamples: {b2['usable']} of {b2['n']} ({b2['unusable']} unusable; CIs conditional)")
        print(f"  β (free origin) 95% CI [{fmt(b2['beta_free_ci'][0], 3)}, {fmt(b2['beta_free_ci'][1], 3)}]")
        print(f"  β (origin {coh['pl_fixed']['origin']}) 95% CI [{fmt(b2['beta_fixed_ci'][0], 3)}, {fmt(b2['beta_fixed_ci'][1], 3)}]")
        print(f"  τ₂ 95% CI [{fmt(b2['tau_ci'][0])}, {fmt(b2['tau_ci'][1])}]")
        print(f"  [Paper: β CI [0.26, 0.39]; τ₂ CI [5.65, 9.26]]")
    else:
        print("\n(Skipping year-block bootstrap for τ₂; add --bootstrap to run)")

    # ── 7. Optional: sensitivity grid ────────────────────────────────────
    if args.sensitivity:
        print("\nRunning 3×3 sensitivity grid (c_min × Y₀) — slow...")
        grid = fitting.sensitivity_grid(
            args.citation_tsv, seeds,
            c_min_values=[3, 5, 10],
            year_starts=[1996, 1998, 2000],
            year_end=args.year_end,
            ngrid=args.ngrid, t_min=args.t_min)
        print(f"\n  {'c_min':>6}  {'Y₀':>6}  {'τ₂':>8}  {'τ₃':>8}")
        print("  " + "-" * 35)
        for (c, y0), res in sorted(grid.items()):
            print(f"  {c:>6}  {y0:>6}  {fmt(res['tau2']):>8}  {fmt(res['tau3']):>8}")
        print(f"  [Paper: τ₂ range 4.1–7.8 yr; τ₃ range 4–50+ yr]")

    # ── 8. Save outputs ───────────────────────────────────────────────────
    specialty = Path(args.seed_file).stem.replace('seed_set_', '').replace('seed_set', 'output')

    # Cohesion time series
    rows = []
    for idx, yr in enumerate(m['years']):
        row = {'year': yr,
               'gamma2': m['gamma2'][idx],
               'omega3': m['omega3'][idx],
               'gamma4': m['gamma4'][idx],
               'n_valid_pairs': m['n_valid_pairs'][idx],
               'n_valid_triples': m['n_valid_triples'][idx],
               'n_valid_quads': m['n_valid_quads'][idx],
               'norm_resid_quad': m['norm_resid_quad'][idx]}
        rows.append(row)
    out_ts = out_dir / f'{specialty}_cohesion_timeseries.csv'
    with open(out_ts, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    # Centrality time series
    rows_c = []
    for idx, yr in enumerate(m['years']):
        row = {'year': yr}
        for s in seeds:
            row[s] = m['centrality'][s][idx]
        row['n_valid_pairs_total'] = m['n_valid_pairs'][idx]
        rows_c.append(row)
    out_c = out_dir / f'{specialty}_centrality.csv'
    with open(out_c, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows_c[0].keys())
        writer.writeheader(); writer.writerows(rows_c)

    # Fit parameters
    fit_summary = {
        'seed_file': args.seed_file,
        'citation_tsv': args.citation_tsv,
        'year_start': args.year_start,
        'year_end': args.year_end,
        'c_min': args.c_min,
        't_min': args.t_min,
        'ngrid': args.ngrid,
        'cohesion': {k: v for k, v in coh.items() if k not in ('years', 'gamma2')},
        'cohesion_series': {'years': coh['years'], 'gamma2': coh['gamma2']},
        'omega3_fit': fit3,
        'tau3_bootstrap': {
            'ci_95': bs3['ci_95'],
            'n_resamples': bs3['n_resamples']
        },
    }
    out_fit = out_dir / f'{specialty}_fit_parameters.json'
    with open(out_fit, 'w') as f:
        json.dump(fit_summary, f, indent=2)

    print(f"\nOutputs written to {out_dir}/")
    print(f"  {out_ts.name}")
    print(f"  {out_c.name}")
    print(f"  {out_fit.name}")
    print("=" * 65)


if __name__ == '__main__':
    main()
