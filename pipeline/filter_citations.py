from __future__ import annotations
"""
filter_citations.py — Algorithm A, Step 1: filter the ADS bulk citation file.

Reads the ADS bulk citation file and extracts all citation edges where the
cited paper is a member of the seed set, writing a filtered TSV that the
rest of the pipeline can consume.

Usage:
    python pipeline/filter_citations.py <seed_file> <bulk_citation_file> [options]

Options:
    --out       PATH   Output TSV path (default: filtered_<specialty>.tsv)
    --year-start INT   Earliest citing year to retain (default: 1996)
    --year-end   INT   Latest citing year to retain  (default: 2025)
    --encoding  STR    Bulk file encoding            (default: utf-8)
    --sep       STR    Bulk file column separator     (default: auto-detect)
    --col-cited INT    0-based index of cited bibcode column  (default: 0)
    --col-citing INT   0-based index of citing bibcode column (default: 1)
    --no-header        Bulk file has no header row (default: assume header)

ADS bulk file format:
    The standard ADS bulk citation file has two tab-separated columns:
        cited_bibcode <TAB> citing_bibcode
    with no header row. Year is parsed from characters 0-3 of the citing
    bibcode (standard ADS bibcode format: YYYYJournalVolumePageA).

    If your bulk file has a different layout, use --col-cited, --col-citing,
    and --no-header to adjust.

Output format (matches filtered_citations_pn_v2.tsv):
    citing_bibcode <TAB> cited_bibcode <TAB> year
    (one row per citation edge where cited_bibcode is in the seed set
     and year falls within [year_start, year_end])

Performance note:
    The ADS bulk file contains ~365 million rows. This script reads it in a
    single streaming pass without loading it into memory. On typical hardware
    expect roughly 5-15 minutes depending on disk speed. A progress report
    is printed every 10 million rows.
"""

import argparse
import csv
import sys
import os
from pathlib import Path

# Allow running from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import seed_set as ss


def detect_separator(path: str, n_lines: int = 5) -> str:
    """Sniff the column separator from the first few lines of the file."""
    with open(path, encoding='utf-8', errors='replace') as f:
        sample = ''.join(f.readline() for _ in range(n_lines))
    tabs   = sample.count('\t')
    commas = sample.count(',')
    pipes  = sample.count('|')
    if tabs >= commas and tabs >= pipes:
        return '\t'
    if commas >= tabs and commas >= pipes:
        return ','
    return '\t'   # default


def year_from_bibcode(bibcode: str) -> int | None:
    """Extract the 4-digit year from an ADS bibcode (characters 0-3)."""
    try:
        return int(bibcode[:4])
    except (ValueError, IndexError):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Filter ADS bulk citation file to seed set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('seed_file',        help='Seed set .txt file')
    p.add_argument('bulk_citation_file', help='ADS bulk citation file')
    p.add_argument('--out',        default=None,
                   help='Output TSV path (default: filtered_<stem>.tsv)')
    p.add_argument('--year-start', type=int, default=1996)
    p.add_argument('--year-end',   type=int, default=2025)
    p.add_argument('--encoding',   default='utf-8')
    p.add_argument('--sep',        default=None,
                   help='Column separator (default: auto-detect)')
    p.add_argument('--col-cited',  type=int, default=0,
                   help='0-based column index of cited bibcode (default: 0)')
    p.add_argument('--col-citing', type=int, default=1,
                   help='0-based column index of citing bibcode (default: 1)')
    p.add_argument('--no-header',  action='store_true',
                   help='Bulk file has no header row')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load seed set ─────────────────────────────────────────────────────
    seeds = set(ss.bibcodes(args.seed_file))
    specialty = Path(args.seed_file).stem \
                    .replace('seed_set_', '') \
                    .replace('seed_set', 'specialty')

    print(f"Seed set   : {args.seed_file}  ({len(seeds)} papers)")
    for b in sorted(seeds):
        print(f"             {b}")

    # ── Output path ───────────────────────────────────────────────────────
    out_path = args.out or f"filtered_{specialty}.tsv"
    print(f"\nBulk file  : {args.bulk_citation_file}")
    print(f"Output     : {out_path}")
    print(f"Window     : {args.year_start}–{args.year_end}")

    # ── Auto-detect separator ─────────────────────────────────────────────
    sep = args.sep or detect_separator(args.bulk_citation_file)
    sep_label = repr(sep)
    print(f"Separator  : {sep_label}")

    # ── Estimate file size for progress reporting ─────────────────────────
    file_bytes = os.path.getsize(args.bulk_citation_file)
    print(f"File size  : {file_bytes / 1e9:.1f} GB\n")
    print("Filtering... (progress every 10M rows)")

    # ── Streaming filter ──────────────────────────────────────────────────
    col_cited  = args.col_cited
    col_citing = args.col_citing
    year_start = args.year_start
    year_end   = args.year_end

    n_rows_read  = 0
    n_rows_kept  = 0
    n_bad_year   = 0
    n_bad_cols   = 0

    with open(args.bulk_citation_file, encoding=args.encoding,
              errors='replace') as infile, \
         open(out_path, 'w', newline='') as outfile:

        writer = csv.writer(outfile, delimiter='\t')
        writer.writerow(['citing_bibcode', 'cited_bibcode', 'year'])

        for i, raw_line in enumerate(infile):

            # Skip header row if present
            if i == 0 and not args.no_header:
                # Heuristic: if the first cell looks like a bibcode
                # (starts with 4 digits), there is no header — process it
                first_cell = raw_line.split(sep)[0].strip()
                if not (len(first_cell) >= 4 and first_cell[:4].isdigit()):
                    continue   # genuine header — skip

            n_rows_read += 1
            if n_rows_read % 10_000_000 == 0:
                print(f"  {n_rows_read / 1e6:6.0f}M rows read  |  "
                      f"{n_rows_kept:,} kept so far")

            cols = raw_line.rstrip('\n').split(sep)

            # Guard against malformed rows
            needed = max(col_cited, col_citing) + 1
            if len(cols) < needed:
                n_bad_cols += 1
                continue

            cited  = cols[col_cited].strip()
            citing = cols[col_citing].strip()

            # Keep only rows where the cited paper is in our seed set
            if cited not in seeds:
                continue

            yr = year_from_bibcode(citing)
            if yr is None:
                n_bad_year += 1
                continue
            if not (year_start <= yr <= year_end):
                continue

            writer.writerow([citing, cited, yr])
            n_rows_kept += 1

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Done.")
    print(f"  Rows read        : {n_rows_read:,}")
    print(f"  Rows kept        : {n_rows_kept:,}")
    print(f"  Bad year (skip)  : {n_bad_year:,}")
    print(f"  Bad cols (skip)  : {n_bad_cols:,}")
    print(f"  Output           : {out_path}")
    print(f"{'='*55}")

    if n_rows_kept == 0:
        print("\nWARNING: No rows matched the seed set.")
        print("Check that bibcodes in the seed file match the format in")
        print("the bulk citation file, and that --col-cited / --col-citing")
        print("point to the correct columns.")
        sys.exit(1)

    # Quick per-seed coverage check
    print("\nPer-seed citation edge counts:")
    counts: dict[str, int] = {b: 0 for b in seeds}
    with open(out_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cited = row['cited_bibcode']
            if cited in counts:
                counts[cited] += 1
    for bibcode, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = '█' * min(40, n // max(1, max(counts.values()) // 40))
        print(f"  {bibcode}  {n:6,}  {bar}")


if __name__ == '__main__':
    main()
