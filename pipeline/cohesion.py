from __future__ import annotations
"""
cohesion.py — Pairwise cohesion Γ₂(t) on a fixed core pair set (paper Sections 2.8–2.9).

Γ₂ averaged over "all pairs valid at each epoch" changes composition as pairs
cross the c_min threshold. Here the pair set is fixed:

  * start year s = first epoch with at least floor(max_valid_pairs / 2) valid pairs
    (half-coverage rule; 2000 with 14 of 28 pairs for the PN morphology set);
  * core set S = pairs valid at s; cumulative counts never decrease, so every
    pair in S stays valid for t >= s;
  * Γ₂(t) = 1 / mean_{(i,j) in S} d_ij(t),  t >= s.

Fits: exponential (τ₂, window-dependent descriptor) and power law
Γ = A (t - t0)^(-β) with free origin t0 (headline β) and fixed origin (default 1995).
The year-block bootstrap re-sorts resampled years chronologically; resamples in
which a core pair drops below c_min are unusable and are counted, so the CIs
are conditional on the usable resamples.
"""
import numpy as np
from scipy.optimize import curve_fit
from . import geometry as geo

_EXP = lambda t, a, tau, c: a * np.exp(-t / tau) + c
_PL = lambda y, A, b, t0: A * (y - t0) ** (-b)


def start_year(data: dict, c_min: int = 5) -> int:
    n = {y: len(geo.distance_matrix(data, y, c_min=c_min)) for y in data['years']}
    half = max(n.values()) // 2
    return next(y for y in data['years'] if n[y] >= half)


def core_set(data: dict, s: int, c_min: int = 5) -> set:
    return set(geo.distance_matrix(data, s, c_min=c_min))


def core_series(data: dict, S: set, s: int, c_min: int = 5):
    """(years, Γ₂) for t >= s, or None if any core pair is invalid at some epoch."""
    ys, g = [], []
    for y in data['years']:
        if y < s:
            continue
        D = geo.distance_matrix(data, y, c_min=c_min)
        if any(p not in D for p in S):
            return None
        ys.append(y); g.append(1.0 / np.mean([D[p] for p in S]))
    return np.array(ys, float), np.array(g)


def fit_exp(y, g):
    t = y - y[0]
    p, _ = curve_fit(_EXP, t, g, p0=[g[0] - g[-1], 5., g[-1]],
                     bounds=([0, 0.5, 0], [10, 200, 1]), maxfev=20000)
    r2 = 1 - ((g - _EXP(t, *p)) ** 2).sum() / ((g - g.mean()) ** 2).sum()
    return dict(a=float(p[0]), tau=float(p[1]), c_inf=float(p[2]), r2=float(r2))


def fit_pl_fixed(y, g, origin=1995):
    b, a = np.polyfit(np.log(y - origin), np.log(g), 1)
    pred = np.exp(a) * (y - origin) ** b
    return dict(beta=float(-b), A=float(np.exp(a)), origin=origin,
                r2=float(1 - ((g - pred) ** 2).sum() / ((g - g.mean()) ** 2).sum()))


def fit_pl_free(y, g):
    best = None
    for t0 in (1990, 1980, 1970):
        try:
            p, _ = curve_fit(_PL, y, g, p0=[0.5, 0.5, t0],
                             bounds=([0, 0.05, 1900], [1e4, 5, y[0] - 0.5]), maxfev=50000)
            r = ((g - _PL(y, *p)) ** 2).sum()
            if best is None or r < best[0]:
                best = (r, p)
        except Exception:
            pass
    if best is None:
        return None
    p = best[1]
    return dict(A=float(p[0]), beta=float(p[1]), t0=float(p[2]),
                r2=float(1 - best[0] / ((g - g.mean()) ** 2).sum()))


def _resampled(data: dict, order: list) -> dict:
    """Rebuild cumulative counts from annual counts taken in the given (sorted) year order."""
    ys = data['years']
    def cum(ann):
        out, tot = {}, 0
        for y, src in zip(ys, order):
            tot += ann[src]; out[y] = tot
        return out
    return dict(years=ys, seeds=data['seeds'],
                cum_cit={s: cum(data['annual_cit'][s]) for s in data['seeds']},
                cum_cocit={k: cum(v) for k, v in data['annual_cocit'].items()})


def analyse(data: dict, c_min: int = 5, start: int | None = None,
            n_boot: int = 1000, rng_seed: int = 42, origin: int = 1995) -> dict:
    s = start if start is not None else start_year(data, c_min)
    S = core_set(data, s, c_min)
    y, g = core_series(data, S, s, c_min)
    res = dict(start=s, n_pairs=len(S), n_epochs=len(y), years=y.tolist(), gamma2=g.tolist(),
               exp=fit_exp(y, g), pl_fixed=fit_pl_fixed(y, g, origin), pl_free=fit_pl_free(y, g))
    if n_boot:
        rng = np.random.default_rng(rng_seed)
        T, B95, B, T0 = [], [], [], []
        failed = 0
        for _ in range(n_boot):
            order = sorted(rng.choice(data['years'], size=len(data['years']), replace=True).tolist())
            r = core_series(_resampled(data, order), S, s, c_min)
            if r is None:
                failed += 1; continue
            try:
                T.append(fit_exp(*r)['tau']); B95.append(fit_pl_fixed(*r, origin)['beta'])
                f = fit_pl_free(*r)
                if f:
                    B.append(f['beta']); T0.append(f['t0'])
            except Exception:
                failed += 1
        q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        res['bootstrap'] = dict(n=n_boot, usable=len(T), unusable=failed,
                                tau_ci=q(T), beta_fixed_ci=q(B95), beta_free_ci=q(B), t0_free_ci=q(T0),
                                note='CIs conditional on usable resamples')
    return res
