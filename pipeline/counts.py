from __future__ import annotations
"""
counts.py — Algorithm A: streaming filter and cumulative count construction.

Reads an ADS bulk citation TSV (cited_bibcode TAB citing_bibcode TAB year)
and builds annual + cumulative citation, co-citation, tri-citation, and
quad-citation counts for a given seed set.

The TSV is expected to have a header row:
    citing_bibcode  cited_bibcode  year

This matches the format of filtered_citations_pn_v2.tsv.

All counts are BINARY per citing paper (a paper citing seed i multiple times
contributes exactly 1 to Ci). All higher-order counts are derived from the
same set R of seeds cited by each paper in a single pass (Algorithm A,
Appendix A of the paper).
"""

import csv
import itertools
import collections
from pathlib import Path


def build(tsv_path: str,
          seed_bibcodes: list[str],
          year_start: int = 1996,
          year_end: int = 2025) -> dict:
    """
    Stream-filter a citation TSV and return cumulative count dicts.

    Parameters
    ----------
    tsv_path : str
        Path to the filtered citation TSV
        (columns: citing_bibcode, cited_bibcode, year).
    seed_bibcodes : list[str]
        Ordered list of ADS bibcodes defining the seed set.
    year_start, year_end : int
        Analysis window endpoints (inclusive).

    Returns
    -------
    dict with keys:
        years           list[int]   — [year_start, ..., year_end]
        seeds           list[str]   — seed_bibcodes as passed in
        cum_cit         dict        — {bibcode: {year: int}}
        cum_cocit       dict        — {(bi, bj): {year: int}}   bi < bj
        cum_tricit      dict        — {(bi, bj, bk): {year: int}}
        cum_quadcit     dict        — {(bi, bj, bk, bl): {year: int}}
        annual_cit      dict        — {bibcode: {year: int}}
        annual_cocit    dict        — {(bi, bj): {year: int}}
        n_citing_papers dict        — {year: int}  papers citing ≥1 seed
        n_edges         int         — total citation edges after filtering
    """
    seeds = list(seed_bibcodes)
    seed_set = set(seeds)
    seeds_sorted = sorted(seeds)   # canonical order for key construction
    years = list(range(year_start, year_end + 1))
    year_set = set(years)

    # --- initialise annual accumulators ---
    ann_cit   = {s: {y: 0 for y in years} for s in seeds}
    ann_cocit = {(min(i, j), max(i, j)): {y: 0 for y in years}
                 for i, j in itertools.combinations(seeds_sorted, 2)}
    ann_tri   = {t: {y: 0 for y in years}
                 for t in itertools.combinations(seeds_sorted, 3)}
    ann_quad  = {q: {y: 0 for y in years}
                 for q in itertools.combinations(seeds_sorted, 4)}

    # --- streaming pass ---
    # accumulate cited seeds per (year, citing_bibcode)
    paper_seeds: dict[int, dict[str, set]] = collections.defaultdict(
        lambda: collections.defaultdict(set)
    )
    n_edges = 0

    with open(tsv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cited  = row['cited_bibcode'].strip()
            citing = row['citing_bibcode'].strip()
            yr     = int(row['year'])
            if cited not in seed_set or yr not in year_set:
                continue
            paper_seeds[yr][citing].add(cited)
            n_edges += 1

    n_citing_papers = {}
    for yr in years:
        n_citing_papers[yr] = len(paper_seeds[yr])
        for citing, cited_set_yr in paper_seeds[yr].items():
            R = sorted(cited_set_yr)           # sorted for canonical key order
            for s in R:
                ann_cit[s][yr] += 1
            for i, j in itertools.combinations(R, 2):
                ann_cocit[(i, j)][yr] += 1
            for i, j, k in itertools.combinations(R, 3):
                ann_tri[(i, j, k)][yr] += 1
            for i, j, k, l in itertools.combinations(R, 4):
                ann_quad[(i, j, k, l)][yr] += 1

    # --- cumulative sums ---
    def cumulate(annual: dict) -> dict:
        cum = {}
        for key, yr_dict in annual.items():
            running = 0
            cum[key] = {}
            for yr in years:
                running += yr_dict[yr]
                cum[key][yr] = running
        return cum

    return {
        'years':            years,
        'seeds':            seeds,
        'cum_cit':          cumulate(ann_cit),
        'cum_cocit':        cumulate(ann_cocit),
        'cum_tricit':       cumulate(ann_tri),
        'cum_quadcit':      cumulate(ann_quad),
        'annual_cit':       ann_cit,
        'annual_cocit':     ann_cocit,
        'n_citing_papers':  n_citing_papers,
        'n_edges':          n_edges,
    }


def coverage_report(data: dict, c_min: int = 5) -> str:
    """Return a human-readable coverage summary for a data dict."""
    years  = data['years']
    seeds  = data['seeds']
    ccit   = data['cum_cit']
    ccocit = data['cum_cocit']
    ctri   = data['cum_tricit']
    cquad  = data['cum_quadcit']
    n = len(seeds)

    lines = [
        f"Coverage report  (c_min={c_min}, window={years[0]}–{years[-1]})",
        f"  Seed papers   : {n}",
        f"  Citation edges: {data['n_edges']}",
        f"  Citing papers : {sum(data['n_citing_papers'].values())} total "
        f"({len([y for y in years if data['n_citing_papers'][y]>0])} years with activity)",
        "",
        f"  {'Year':>6}  {'Citing':>8}  {'Pairs≥c_min':>12}  {'Triples':>8}  {'Quads':>7}",
        "  " + "-" * 52,
    ]
    n_pairs  = len(list(itertools.combinations(seeds, 2)))
    n_triples = len(list(itertools.combinations(seeds, 3)))
    n_quads  = len(list(itertools.combinations(seeds, 4)))

    for yr in years:
        vp = sum(1 for k, d in ccocit.items() if d[yr] >= c_min)
        vt = sum(1 for k, d in ctri.items()   if d[yr] >= 1)
        vq = sum(1 for k, d in cquad.items()  if d[yr] >= 1)
        lines.append(
            f"  {yr:>6}  {data['n_citing_papers'][yr]:>8}  "
            f"{vp:>5}/{n_pairs:<5}  {vt:>4}/{n_triples:<3}  {vq:>3}/{n_quads}"
        )
    return '\n'.join(lines)
