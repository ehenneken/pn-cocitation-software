from __future__ import annotations
"""
metrics.py — Scalar metric computation (Algorithm B/C scalar outputs).

Computes per-epoch:
  - κᵢ(t)   geometric centrality
  - Gᵢ(t)   co-citation growth rate
  - Γ₂(t)   pairwise cluster cohesion
  - Ω₃(t)   mean triple-overlap attenuation  (units: 1/area)
  - Γ₄(t)   quadruple cohesion              (exploratory)

All metrics use the notation and definitions from Section 2.5 of the paper.
"""

import math
import itertools
import numpy as np
from . import geometry as geo


def compute_all(data: dict,
                c_min: int = 5,
                t_min: float = 0.1,
                ngrid: int = 500) -> dict:
    """
    Compute all scalar metrics for every epoch in data['years'].

    Parameters
    ----------
    data   : output of counts.build()
    c_min  : minimum co-citation threshold for valid pairs
    t_min  : minimum predicted triple area to include in Ω₃
    ngrid  : grid resolution for overlap integration

    Returns
    -------
    dict with keys:
        years                list[int]
        seeds                list[str]
        centrality           dict  {bibcode: list[float|nan]}   length=n_years
        growth_rate          dict  {bibcode: list[float]}
        gamma2               list[float|nan]   Γ₂(t)
        omega3               list[float|nan]   Ω₃(t)
        gamma4               list[float|nan]   Γ₄(t)
        n_valid_pairs        list[int]
        n_valid_triples      list[int]
        n_valid_quads        list[int]
        norm_resid_quad      list[float|nan]   mean normalised LS residual
        D_by_year            dict  {year: D_dict}   (for downstream use)
    """
    years   = data['years']
    seeds   = data['seeds']
    cum_cit = data['cum_cit']
    ann_cocit = data['annual_cocit']

    PAIRS   = list(itertools.combinations(seeds, 2))
    TRIPLES = list(itertools.combinations(seeds, 3))
    QUADS   = list(itertools.combinations(seeds, 4))

    centrality      = {s: [] for s in seeds}
    growth_rate     = {s: [] for s in seeds}
    gamma2          = []
    omega3          = []
    gamma4          = []
    n_valid_pairs   = []
    n_valid_triples = []
    n_valid_quads   = []
    norm_resid_quad = []
    D_by_year       = {}

    for yr in years:
        # ── Distance matrix ───────────────────────────────────────────────
        D = geo.distance_matrix(data, yr, c_min=c_min)
        D_by_year[yr] = D

        # ── Geometric centrality κᵢ(t) ────────────────────────────────────
        for s in seeds:
            partners = [D.get((min(s, o), max(s, o))) for o in seeds if o != s]
            valid    = [d for d in partners if d is not None]
            centrality[s].append(1.0 / np.mean(valid) if len(valid) >= 3 else math.nan)

        # ── Co-citation growth rate Gᵢ(t) ────────────────────────────────
        for s in seeds:
            g = sum(
                ann_cocit.get((min(s, o), max(s, o)), {}).get(yr, 0)
                for o in seeds if o != s
            )
            growth_rate[s].append(float(g))

        # ── Pairwise cohesion Γ₂(t) ──────────────────────────────────────
        all_d = list(D.values())
        gamma2.append(1.0 / np.mean(all_d) if len(all_d) >= 2 else math.nan)
        n_valid_pairs.append(len(all_d))

        # ── Triple attenuation Ω₃(t) ─────────────────────────────────────
        t_areas = []
        for (i, j, k) in TRIPLES:
            dij = D.get((min(i, j), max(i, j)))
            dik = D.get((min(i, k), max(i, k)))
            djk = D.get((min(j, k), max(j, k)))
            if dij is None or dik is None or djk is None:
                continue
            ri = math.sqrt(cum_cit[i][yr] / math.pi)
            rj = math.sqrt(cum_cit[j][yr] / math.pi)
            rk = math.sqrt(cum_cit[k][yr] / math.pi)
            T = geo.triple_overlap(dij, dik, djk, ri, rj, rk, ngrid=ngrid)
            if T is not None and T >= t_min:
                t_areas.append(T)
        omega3.append(1.0 / np.mean(t_areas) if len(t_areas) >= 3 else math.nan)
        n_valid_triples.append(len(t_areas))

        # ── Quadruple cohesion Γ₄(t) — exploratory ───────────────────────
        q_areas    = []
        q_residuals = []
        for quad in QUADS:
            pairs_in_quad = list(itertools.combinations(quad, 2))
            dists = [D.get((min(a, b), max(a, b))) for a, b in pairs_in_quad]
            if any(d is None for d in dists):
                continue
            n_q = len(quad)
            D4  = np.zeros((n_q, n_q))
            for idx, (a, b) in enumerate(pairs_in_quad):
                ai = quad.index(a)
                bi = quad.index(b)
                D4[ai, bi] = D4[bi, ai] = D[(min(a, b), max(a, b))]
            radii = [math.sqrt(cum_cit[s][yr] / math.pi) for s in quad]
            result = geo.place_circles(D4, n_q)
            if result is None:
                continue
            xs, ys, norm_res = result
            Q = geo.grid_overlap(xs, ys, radii, ngrid=ngrid)
            if Q > 0:
                q_areas.append(Q)
                q_residuals.append(norm_res)

        gamma4.append(1.0 / np.mean(q_areas) if len(q_areas) >= 3 else math.nan)
        n_valid_quads.append(len(q_areas))
        norm_resid_quad.append(float(np.mean(q_residuals)) if q_residuals else math.nan)

    return {
        'years':           years,
        'seeds':           seeds,
        'centrality':      centrality,
        'growth_rate':     growth_rate,
        'gamma2':          gamma2,
        'omega3':          omega3,
        'gamma4':          gamma4,
        'n_valid_pairs':   n_valid_pairs,
        'n_valid_triples': n_valid_triples,
        'n_valid_quads':   n_valid_quads,
        'norm_resid_quad': norm_resid_quad,
        'D_by_year':       D_by_year,
    }
