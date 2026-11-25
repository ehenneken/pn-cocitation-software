# Co-citation geometry pipeline

Code for *The Planetary Nebulae Morphology Literature: Temporal Co-citation Geometry, Cohesion Hierarchies, and
Critical Scaling* (E. A. Henneken). It computes the Small (1974) circle model at annual epochs from a set of seed
papers, extends it to tri- and quad-citation, and fits the decay of cohesion. Data: see the companion Zenodo data deposit.
Paper section numbers below refer to the manuscript.

## What it does

1. **Filter** the ADS bulk citation file to edges whose cited paper is a seed paper (`filter_citations.py`).
2. **Count** annual and cumulative citations, co-citations, tri- and quad-citations in one pass over citing papers
   (`counts.py`, Algorithm A).
3. **Geometry** (`geometry.py`): circle radii r = sqrt(C/pi); pairwise distance from the lens-area equation by Brent's
   method; triple overlap by grid integration; quad placement by exact triangulation of circles 1-3 and least-squares
   trilateration of the fourth.
4. **Metrics** (`metrics.py`): geometric centrality, pairwise cohesion over all valid pairs, triple attenuation
   Omega3 = 1/mean(T_ijk), quad cohesion Gamma4 and the normalized placement residual.
5. **Cohesion fits** (`cohesion.py`, Sections 2.8-2.9): pairwise cohesion Gamma2 on a **fixed core set of pairs**, an
   exponential fit (tau2), a power-law fit (beta, free and fixed time origin), and the year-block bootstrap.
6. **Other fits** (`fitting.py`): exponential fit for Omega3 (all valid epochs), parametric bootstrap for tau3, and the
   3x3 c_min x start-year sensitivity grid.

`run.py` runs steps 2-6 and writes the results.

## Install

Python >= 3.8, `numpy>=1.21`, `scipy>=1.7`:

    pip install -r pipeline/requirements.txt

## Usage

Filter the bulk file once (the June 2026 snapshot has about 365 million rows; one streaming pass, roughly 5-15 minutes).
Skip this step if you use the filtered edge file from the data deposit:

    python pipeline/filter_citations.py seed_set_pn_morphology.txt <ads_bulk_citations> --out filtered_citations_pn_v2.tsv

Run the analysis:

    python pipeline/run.py seed_set_pn_morphology.txt filtered_citations_pn_v2.tsv --out-dir out --bootstrap

| Option | Default | Meaning |
|---|---|---|
| `--year-start`, `--year-end` | 1996, 2025 | Analysis window (citing years) |
| `--c-min` | 5 | Minimum co-citation count for a valid pair |
| `--t-min` | 0.1 | Minimum predicted triple area |
| `--ngrid` | 500 | Grid resolution for overlap integration |
| `--bootstrap` | off | Year-block bootstrap for beta and tau2 |
| `--n-boot` | 1000 | Bootstrap resamples |
| `--sensitivity` | off | 3x3 sensitivity grid (slow) |
| `--out-dir` | `.` | Output directory |

### Inputs

* **Seed set file:** one seed per line, `bibcode | sub-theme | notes`; `#` comments and blank lines are ignored.
* **Filtered edge file:** tab-separated with header `citing_bibcode  cited_bibcode  year`. The filter keeps rows whose cited
  paper is a seed and whose citing year (characters 0-3 of the bibcode) is within the window. It does not restrict the
  type of the citing record (refereed articles, proceedings, theses and preprints are all kept).

### Outputs (in `--out-dir`)

Files are prefixed with the specialty name taken from the seed file name (`seed_set_pn_morphology.txt` gives `pn_morphology_`).

* `<specialty>_cohesion_timeseries.csv`: per year, Gamma2 (all valid pairs), Omega3, Gamma4, numbers of valid pairs, triples
  and quads, and the mean normalized quad residual. Column names correspond to `figure10_cohesion_hierarchy.csv` in the
  data deposit.
* `<specialty>_centrality.csv`: geometric centrality kappa_i(t) for each seed paper.
* `<specialty>_fit_parameters.json`: fixed-core Gamma2 fits (exponential and power law, bootstrap CIs, usable-resample
  count), Omega3 fit, tau3 bootstrap, and the run settings.

## The fixed core set (important)

The mean distance over all pairs valid at each epoch changes composition as pairs cross the threshold, which mixes a
change in membership with a change in distance. `cohesion.py` therefore fixes the pair set:

* start year s = first epoch with at least floor(max valid pairs / 2) valid pairs (half-coverage rule; 2000 with 14 of 28
  pairs for the PN set);
* core set S = pairs valid at s (they stay valid, since cumulative counts do not decrease);
* Gamma2(t) = 1 / mean over S of d_ij(t), for t >= s.

tau2 depends on the start year and is a window-dependent descriptor; the power-law exponent beta is the headline.
beta also depends on the assumed time origin, so both the free-origin value and the value at origin 1995 are reported.
In the bootstrap, resampled years are re-sorted before cumulative sums. Resamples in which a core pair falls below
`c_min` cannot be fitted and are counted as unusable, so the confidence intervals are conditional on the usable resamples.
The triple geometry (`geometry.py`) does not use the core set.

## Expected results (PN morphology seed set, `seed_set_pn_morphology.txt`)

| Quantity | Value |
|---|---|
| Core set | start 2000, 14 pairs, 26 epochs |
| beta (free origin) | 0.309, 95% CI [0.263, 0.386]; t0 = 1997.4, R2 = 0.998 |
| beta (origin 1995) | 0.389, 95% CI [0.349, 0.452] |
| tau2 | 6.95 yr, 95% CI [5.65, 9.26]; c_inf = 0.070, R2 = 0.995 |
| Usable bootstrap resamples | 564 of 1000 |
| tau3 | 15.3 yr, parametric-bootstrap CI about [8.3, 27.8] (the paper quotes [8.2, 27.5]; they differ by random resampling) |
| Tri-citation validation | r = 0.955 (56 triples) |
| Quad placement residual | mean 0.41, range 0.30-0.43 |

A full run with `--bootstrap` takes about 40 seconds. The bootstrap is seeded (42), so repeated runs give identical
intervals.

`python test_cohesion.py <n_boot>` is a regression check that compares `cohesion.analyse` with the audited reference
results; it needs the audit data layout and is included for transparency, not for general use.

## Module map

| File | Role |
|---|---|
| `run.py` | Entry point |
| `filter_citations.py` | Bulk file to filtered edge file |
| `seed_set.py` | Seed file parser |
| `counts.py` | Annual and cumulative citation, co-citation, tri- and quad-citation counts |
| `geometry.py` | Lens area, pairwise distance, circle placement, grid overlap |
| `metrics.py` | Centrality, Gamma2 (all valid pairs), Omega3, Gamma4, residuals |
| `cohesion.py` | Fixed-core Gamma2, exponential and power-law fits, year-block bootstrap |
| `fitting.py` | Omega3 fit, tau3 parametric bootstrap, sensitivity grid |

## Known limitations

* Gamma4 and the quad residual are exploratory. Four pairwise distances generally cannot be embedded in the plane; the
  residual (about 0.41 of the mean distance) measures this.
* tau3 is sensitive to `c_min` (roughly 4 to more than 50 yr over the 3x3 grid).
* The bootstrap intervals are conditional on usable resamples (44% unusable for PN morphology), so they understate the
  uncertainty.
* Seed papers with only a few valid pairs in the early years give a small core set; compare tau2 across fields only at
  equal window lengths, and compare beta.

## Citation and licence

Cite the paper and the Zenodo code deposit (DOI to be added). Add a licence file before publishing.
