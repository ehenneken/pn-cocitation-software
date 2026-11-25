from __future__ import annotations
"""
seed_set.py — Parse a seed set definition file.

File format (# comments and blank lines ignored):
    bibcode | sub-theme | notes

Returns a list of dicts with keys: bibcode, subtheme, notes.
"""

from pathlib import Path


def load(path: str) -> list[dict]:
    """
    Parse a seed-set file and return a list of seed paper dicts.

    Parameters
    ----------
    path : str
        Path to the seed set .txt file.

    Returns
    -------
    list[dict]
        Each entry has keys:
          bibcode   (str)  — ADS bibcode
          subtheme  (str)  — sub-theme label
          notes     (str)  — free-text notes (may be empty)
    """
    seeds = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 2:
            continue  # malformed line — skip silently
        seeds.append({
            'bibcode':  parts[0],
            'subtheme': parts[1],
            'notes':    parts[2] if len(parts) > 2 else '',
        })
    return seeds


def bibcodes(path: str) -> list[str]:
    """Return just the ordered list of bibcodes from a seed set file."""
    return [s['bibcode'] for s in load(path)]


def summary(path: str) -> str:
    """Return a human-readable summary of a seed set file."""
    seeds = load(path)
    lines = [f"Seed set: {Path(path).name}  ({len(seeds)} papers)"]
    lines.append("-" * 60)
    for s in seeds:
        lines.append(f"  {s['bibcode']}  [{s['subtheme']}]")
        if s['notes']:
            # First fragment of notes (before semicolons) for brevity
            short = s['notes'].split(';')[0].strip()
            lines.append(f"    {short}")
    return '\n'.join(lines)


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'seed_set.txt'
    print(summary(path))
