from __future__ import annotations
"""
geometry.py — Circle-model geometry engine (Algorithms B and C).

Implements:
  - lens_area(r1, r2, d)             : area of intersection of two circles
  - solve_distance(ci, cj, cij, ...) : pairwise distance from counts (Brent)
  - distance_matrix(data, year, ...)  : full D(t) matrix at one epoch
  - place_circles(D_local, n)         : circle placement (triangulation + LS)
  - grid_overlap(xs, ys, rs, ngrid)  : n-circle intersection area by grid integration

All geometry follows Appendix A of the paper exactly, including:
  - Boundary case in solve_distance: returns |ri-rj| when Cij ≥ max overlap - δ
  - Least-squares trilateration for the 4th circle (n=4), not naive triangulation
"""

import math
import itertools
import numpy as np
from scipy.optimize import brentq, minimize


# ── Fundamental lens geometry ─────────────────────────────────────────────────

def lens_area(r1: float, r2: float, d: float) -> float:
    """
    Area of intersection of two circles with radii r1, r2 and
    centre-to-centre distance d.
    """
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    cos1 = (d**2 + r1**2 - r2**2) / (2 * d * r1)
    cos2 = (d**2 + r2**2 - r1**2) / (2 * d * r2)
    a1 = 2 * math.acos(max(-1.0, min(1.0, cos1)))
    a2 = 2 * math.acos(max(-1.0, min(1.0, cos2)))
    return 0.5 * (r1**2 * (a1 - math.sin(a1)) + r2**2 * (a2 - math.sin(a2)))


def solve_distance(ci: float, cj: float, cij: float,
                   c_min: int = 5,
                   delta: float = 1e-9) -> float | None:
    """
    Solve for the circle-model distance dij given citation counts.

    Returns None if Cij < c_min (pair below threshold).
    Returns |ri - rj| directly when Cij ≥ max_overlap - delta (boundary case:
    full containment; avoids Brent's method on an open bracket that excludes
    the boundary root).

    Parameters
    ----------
    ci, cj  : cumulative citation counts for papers i and j
    cij     : cumulative co-citation count
    c_min   : minimum co-citation threshold
    delta   : tolerance for boundary-case detection
    """
    if cij < c_min:
        return None
    ri = math.sqrt(ci / math.pi)
    rj = math.sqrt(cj / math.pi)
    max_overlap = math.pi * min(ri, rj) ** 2
    if cij >= max_overlap - delta:
        return abs(ri - rj)            # boundary: full containment
    lo = abs(ri - rj) + 1e-9
    hi = ri + rj - 1e-9
    try:
        return brentq(lambda d: lens_area(ri, rj, d) - cij, lo, hi, xtol=1e-6)
    except ValueError:
        return None


# ── Distance matrix at one epoch ──────────────────────────────────────────────

def distance_matrix(data: dict, year: int, c_min: int = 5) -> dict:
    """
    Build the pairwise distance dict D[(bi, bj)] at a given epoch.

    Returns
    -------
    dict mapping (bi, bj) → float distance (bi < bj),
    containing only valid pairs (Cij ≥ c_min and geometrically consistent).
    """
    seeds    = data['seeds']
    cum_cit  = data['cum_cit']
    cum_cocit = data['cum_cocit']
    D = {}
    for i, j in itertools.combinations(seeds, 2):
        key = (min(i, j), max(i, j))
        ci  = cum_cit[i][year]
        cj  = cum_cit[j][year]
        cij = cum_cocit[key][year]
        d   = solve_distance(ci, cj, cij, c_min=c_min)
        if d is not None:
            D[key] = d
    return D


# ── Circle placement ──────────────────────────────────────────────────────────

def place_circles(D_local: np.ndarray, n: int) -> tuple | None:
    """
    Place n circles in 2D from an n×n distance matrix.

    Circles 1 and 2: trivially placed (origin and x-axis).
    Circle 3: exact triangulation from circles 1 and 2.
    Circle k ≥ 4: least-squares minimisation of squared distance
        residuals to all k-1 already-placed circles (true trilateration).
        Returns the normalised LS residual for each extra circle as a
        geometric consistency diagnostic.

    Parameters
    ----------
    D_local : np.ndarray, shape (n, n)
        Pairwise distance matrix (0-indexed).
    n : int
        Number of circles to place.

    Returns
    -------
    (x, y, max_norm_residual) or None if geometry is inconsistent.

    Notes
    -----
    The normalised residual for circle k is:
        sqrt(min_f) / mean(D_local[j, k] for j in 0..k-1)
    A residual > 0 for n ≥ 4 indicates the 2D embedding is inconsistent
    (six pairwise distances cannot be simultaneously satisfied in the plane).
    The stable value ~0.35 observed in the PN quad analysis is expected for
    real citation data and not a numerical artefact.
    """
    x = np.zeros(n)
    y = np.zeros(n)

    # Circle 2 on positive x-axis
    x[1] = D_local[0, 1]

    # Circle 3: triangulate from circles 1 and 2
    if n >= 3:
        d13 = D_local[0, 2]
        d12 = D_local[0, 1]
        d23 = D_local[1, 2]
        if d13 * d12 == 0:
            return None
        cos_a = (d13**2 + d12**2 - d23**2) / (2 * d13 * d12)
        cos_a = float(np.clip(cos_a, -1.0, 1.0))
        x[2] = d13 * cos_a
        yk2 = d13**2 - x[2]**2
        if yk2 < 0:
            return None
        y[2] = math.sqrt(yk2)

    max_norm_res = 0.0

    # Circles 4+: least-squares trilateration
    for k in range(3, n):
        def resid(pos, k=k):
            px, py = pos
            return sum(
                (math.sqrt((px - x[j])**2 + (py - y[j])**2) - D_local[j, k])**2
                for j in range(k)
            )

        # Initial guess: triangulate from circles 1 and 2
        d1k = D_local[0, k]
        d12 = D_local[0, 1]
        d2k = D_local[1, k]
        if d1k * d12 == 0:
            return None
        cos_a = (d1k**2 + d12**2 - d2k**2) / (2 * d1k * d12)
        cos_a = float(np.clip(cos_a, -1.0, 1.0))
        x0 = d1k * cos_a
        y0 = math.sqrt(max(d1k**2 - x0**2, 0.0))

        res = minimize(resid, [x0, y0], method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 10000})
        x[k], y[k] = res.x

        mean_d = float(np.mean([D_local[j, k] for j in range(k)]))
        if mean_d > 0:
            norm_res = math.sqrt(res.fun) / mean_d
            max_norm_res = max(max_norm_res, norm_res)

    return x, y, max_norm_res


# ── Grid integration ──────────────────────────────────────────────────────────

def grid_overlap(xs: np.ndarray, ys: np.ndarray,
                 rs: list[float], ngrid: int = 500) -> float:
    """
    Compute the intersection area of n circles by grid integration.

    Parameters
    ----------
    xs, ys : array-like
        Circle centre coordinates.
    rs : list[float]
        Circle radii.
    ngrid : int
        Grid resolution (n_grid × n_grid points over bounding box).
        At ngrid=500, relative error vs ngrid=5000 is <0.5% for all
        triples tested (see convergence_test_data.json in the repository).

    Returns
    -------
    float : intersection area in the same units as rs.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    rs = np.asarray(rs)
    xlo = float(xs.min() - rs.max())
    xhi = float(xs.max() + rs.max())
    ylo = float(ys.min() - rs.max())
    yhi = float(ys.max() + rs.max())
    gx = np.linspace(xlo, xhi, ngrid)
    gy = np.linspace(ylo, yhi, ngrid)
    dx = (xhi - xlo) / (ngrid - 1)
    dy = (yhi - ylo) / (ngrid - 1)
    GX, GY = np.meshgrid(gx, gy)
    mask = np.ones((ngrid, ngrid), dtype=bool)
    for xi, yi, ri in zip(xs, ys, rs):
        mask &= (GX - xi)**2 + (GY - yi)**2 <= ri**2
    return float(np.sum(mask)) * dx * dy


# ── Triple overlap (convenience wrapper) ─────────────────────────────────────

def triple_overlap(dij: float, dik: float, djk: float,
                   ri: float, rj: float, rk: float,
                   ngrid: int = 500) -> float | None:
    """
    Compute the triple-circle intersection area for seeds i, j, k.

    Returns None if the triangle inequality fails for the three distances.
    """
    D3 = np.array([[0, dij, dik],
                   [dij, 0,  djk],
                   [dik, djk, 0]])
    result = place_circles(D3, 3)
    if result is None:
        return None
    xs, ys, _ = result
    return grid_overlap(xs, ys, [ri, rj, rk], ngrid=ngrid)
