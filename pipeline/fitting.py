from __future__ import annotations
"""
fitting.py — Exponential decay fitting and uncertainty quantification.

Implements:
  - fit_decay()           : three-parameter exponential fit
  - (year-block bootstrap for τ₂, β: see cohesion.py)
  - parametric_bootstrap_tau3() : parametric bootstrap for τ₃ CI
  - sensitivity_grid()    : 3×3 c_min × Y₀ sensitivity grid

All procedures follow Section 2.7 (Robustness and uncertainty quantification)
of the paper exactly.
"""

import math
import itertools
import numpy as np
from scipy.optimize import curve_fit
from . import counts as cnt
from . import geometry as geo
from . import metrics as met


# ── Exponential decay model ───────────────────────────────────────────────────

def _model(t, a, tau, c_inf):
    return a * np.exp(-(t - t[0]) / tau) + c_inf


def fit_decay(years: list, values: list) -> dict | None:
    """
    Fit Γ(t) = a · exp(-(t - t₀)/τ) + c∞ by nonlinear least squares.

    Parameters
    ----------
    years  : list of epoch years (int)
    values : corresponding metric values (float or nan)

    Returns
    -------
    dict with keys: a, tau, c_inf, r2, fitted_years, fitted_values
    or None if the fit fails or insufficient data.
    """
    yrs  = np.array(years, dtype=float)
    vals = np.array(values, dtype=float)
    mask = ~np.isnan(vals)
    yf, vf = yrs[mask], vals[mask]
    if len(yf) < 5:
        return None
    try:
        p0 = [max(vf[0] - vf[-1], 0.01), 5.0, max(vf[-1], 0.0)]
        popt, _ = curve_fit(_model, yf, vf, p0=p0,
                            bounds=([0, 0.5, 0], [np.inf, 200, np.inf]),
                            maxfev=20000)
        a, tau, c_inf = popt
        residuals = vf - _model(yf, *popt)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((vf - np.mean(vf))**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            'a':            float(a),
            'tau':          float(tau),
            'c_inf':        float(c_inf),
            'r2':           float(r2),
            'fitted_years': yf.tolist(),
            'fitted_values': _model(yf, *popt).tolist(),
        }
    except Exception:
        return None


# The year-block bootstrap for τ₂ / β lives in cohesion.py (fixed core pair set).


# ── Parametric bootstrap for τ₃ ──────────────────────────────────────────────

def parametric_bootstrap_tau3(omega3: list[float],
                               years: list[int],
                               n_resamples: int = 2000,
                               rng_seed: int = 42) -> dict:
    """
    Parametric bootstrap for τ₃ by resampling residuals from the base fit.

    Parameters
    ----------
    omega3      : Ω₃(t) time series (may contain nan)
    years       : corresponding years
    n_resamples : number of parametric bootstrap resamples
    rng_seed    : numpy random seed

    Returns
    -------
    dict with keys: tau3_samples, ci_95, n_resamples
    """
    base_fit = fit_decay(years, omega3)
    if base_fit is None:
        return {'tau3_samples': [], 'ci_95': [np.nan, np.nan], 'n_resamples': 0}

    yrs  = np.array(years, dtype=float)
    vals = np.array(omega3, dtype=float)
    mask = ~np.isnan(vals)
    yf, vf = yrs[mask], vals[mask]

    fitted = np.array(base_fit['fitted_values'])
    residuals = vf - fitted

    rng = np.random.default_rng(rng_seed)
    tau3_samples = []

    for _ in range(n_resamples):
        resampled_res = rng.choice(residuals, size=len(residuals), replace=True)
        omega3_b = (fitted + resampled_res).tolist()
        fit_b = fit_decay(yf.tolist(), omega3_b)
        if fit_b is not None:
            tau3_samples.append(fit_b['tau'])

    arr = np.array(tau3_samples)
    return {
        'tau3_samples':  arr.tolist(),
        'ci_95':         [float(np.percentile(arr, 2.5)),
                          float(np.percentile(arr, 97.5))],
        'n_resamples':   len(tau3_samples),
    }


# ── Sensitivity grid ─────────────────────────────────────────────────────────

def sensitivity_grid(tsv_path: str,
                     seed_bibcodes: list[str],
                     c_min_values: list[int] = (3, 5, 10),
                     year_starts: list[int] = (1996, 1998, 2000),
                     year_end: int = 2025,
                     ngrid: int = 500,
                     t_min: float = 0.1) -> dict:
    """
    3×3 sensitivity grid for τ₂ and τ₃ over c_min × Y₀ combinations.

    Returns
    -------
    dict mapping (c_min, y0) → {'tau2': float|None, 'tau3': float|None}
    """
    results = {}
    for c_min in c_min_values:
        for y0 in year_starts:
            data = cnt.build(tsv_path, seed_bibcodes, y0, year_end)
            m = met.compute_all(data, c_min=c_min, t_min=t_min, ngrid=ngrid)
            from . import cohesion as coh_
            try:
                c2 = coh_.analyse(data, c_min=c_min, n_boot=0)
                f2 = dict(tau=c2['exp']['tau'], r2=c2['exp']['r2'], beta=c2['pl_free']['beta'] if c2['pl_free'] else None, start=c2['start'])
            except Exception:
                f2 = None
            f3 = fit_decay(m['years'], m['omega3'])
            results[(c_min, y0)] = {
                'tau2': f2['tau'] if f2 else None,
                'r2_2': f2['r2'] if f2 else None,
                'beta': f2['beta'] if f2 else None,
                'start': f2['start'] if f2 else None,
                'tau3': f3['tau'] if f3 else None,
                'r2_3': f3['r2'] if f3 else None,
            }
    return results
