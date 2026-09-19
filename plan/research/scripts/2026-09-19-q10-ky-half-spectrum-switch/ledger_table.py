"""Print the full-vs-half HLO ledger as one table.

usage: ledger_table.py <ledger_dir> [<committed_before_dir>]

The second argument, when given, is a directory of #248's ledger JSONs; the
``full`` arm is then also checked against it, which is what says the ``half``
numbers are a layout difference and not a drift in the harness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KEYS = ("fft", "concatenate", "copy", "transpose", "gather", "reverse", "bytes_written")
ROWS = [
    ("RHS", "32", None),
    ("diagnostics rk3", "32", "rk3"),
    ("diagnostics rk4", "32", "rk4"),
    ("RHS", "64", None),
    ("diagnostics rk3", "64", "rk3"),
    ("diagnostics rk4", "64", "rk4"),
]


def load(directory: Path, layout: str, route: str, grid: str) -> dict:
    return json.loads((directory / f"{layout}_{route}_{grid}.json").read_text())


def counts(payload: dict, method: str | None) -> dict:
    if method is None:
        return payload["rhs"]
    return payload["steps"][method]["counts"]


def main(argv: list[str]) -> int:
    directory = Path(argv[1])
    before = Path(argv[2]) if len(argv) > 2 else None
    for route in ("diagnostics", "runtime"):
        print(f"### route={route}")
        header = f"| {'graph':<20} | " + " | ".join(f"{k:>14}" for k in KEYS) + " |"
        print(header)
        print("|" + "-" * (len(header) - 2) + "|")
        for label, grid, method in ROWS:
            full = counts(load(directory, "full", route, grid), method)
            half = counts(load(directory, "half", route, grid), method)
            cells = []
            for key in KEYS:
                a, b = int(full[key]), int(half[key])
                pct = "" if a == 0 else f" ({100.0 * (b - a) / a:+.1f}%)"
                cells.append(f"{a}->{b}{pct}")
            print(f"| {label + ' ' + grid:<20} | " + " | ".join(cells) + " |")
        print()
    if before is not None:
        print("### full arm against the committed 'before'")
        ok = True
        for route in ("diagnostics", "runtime"):
            for label, grid, method in ROWS:
                mine = counts(load(directory, "full", route, grid), method)
                theirs = counts(
                    json.loads((before / f"{route}_{grid}.json").read_text()), method
                )
                same = all(int(mine[k]) == int(theirs[k]) for k in KEYS)
                ok = ok and same
                print(f"  {route:<12} {label} {grid}: {'same' if same else 'DIFFERS'}")
        print("full arm reproduces the committed ledger:", ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
