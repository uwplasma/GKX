"""Print the Q10 full-vs-half ledger: op counts, the byte split, buffer assignment.

usage: ledger_table.py <ledger_dir> [<earlier_ledger_dir>]

#259's script, with the chain-free control dropped (it belonged to that row's
diagnosis, not to this one) and the second argument repointed: it is now any
earlier ledger directory, and the check says whether this row's `src/` changes
moved the compiled modules the earlier row measured.  They may: this row folds
the ky reduction at fixed kx on a half axis, which the diagnostics route
compiles.  A difference is a result to report, not a failure.

Three byte totals appear, and they are not interchangeable.  ``bytes_written``
is the historical total -- every ``concatenate`` and ``copy`` in the module
text, fusion interiors included.  ``materialized`` is the subset that owns an
output buffer.  ``temp`` is XLA's own scratch arena for the executable and is
the independent check on ``materialized``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

COUNT_KEYS = ("fft", "concatenate", "copy", "transpose", "gather", "reverse")
BYTE_KEYS = ("bytes_written", "materialized_bytes", "fused_interior_bytes")
ROWS = [
    ("RHS", "32", None),
    ("rk3", "32", "rk3"),
    ("rk4", "32", "rk4"),
    ("RHS", "64", None),
    ("rk3", "64", "rk3"),
    ("rk4", "64", "rk4"),
]


def load(directory: Path, layout: str, route: str, grid: str) -> dict:
    return json.loads((directory / f"{layout}_{route}_{grid}.json").read_text())


def counts(payload: dict, method: str | None) -> dict:
    if method is None:
        return payload["rhs"]
    return payload["steps"][method]["counts"]


def memory(payload: dict, method: str | None) -> dict:
    if method is None:
        return payload.get("rhs_memory_analysis", {})
    return payload["steps"][method].get("memory_analysis", {})


def delta(a: int, b: int) -> str:
    pct = "" if a == 0 else f" ({100.0 * (b - a) / a:+.1f}%)"
    return f"{a}->{b}{pct}"


def table(directory: Path, route: str, keys: tuple[str, ...], temp: bool) -> None:
    header = f"| {'graph':<10} | " + " | ".join(f"{k:>26}" for k in keys)
    header += f" | {'temp_size_in_bytes':>26} |" if temp else " |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for label, grid, method in ROWS:
        full_payload = load(directory, "full", route, grid)
        half_payload = load(directory, "half", route, grid)
        full, half = counts(full_payload, method), counts(half_payload, method)
        cells = [delta(int(full[k]), int(half[k])) for k in keys]
        if temp:
            fm = memory(full_payload, method).get("temp_size_in_bytes", 0)
            hm = memory(half_payload, method).get("temp_size_in_bytes", 0)
            cells.append(delta(int(fm), int(hm)))
        print(f"| {label + ' ' + grid:<10} | " + " | ".join(cells) + " |")
    print()


def check_against_committed(directory: Path, before: Path) -> bool:
    """The compiled modules must be the ones #258 measured, both arms."""

    ok = True
    for route in ("diagnostics", "runtime"):
        for layout in ("full", "half"):
            for label, grid, method in ROWS:
                mine = counts(load(directory, layout, route, grid), method)
                theirs = counts(
                    load(before, layout, route, grid),
                    method,
                )
                same = all(int(mine[k]) == int(theirs[k]) for k in COUNT_KEYS) and int(
                    mine["bytes_written"]
                ) == int(theirs["bytes_written"])
                ok = ok and same
                print(
                    f"  {route:<12} {layout:<5} {label} {grid}: "
                    f"{'same' if same else 'DIFFERS'}"
                )
    return ok


def main(argv: list[str]) -> int:
    directory = Path(argv[1])
    for route in ("diagnostics", "runtime"):
        print(f"### op counts, route={route}")
        table(directory, route, COUNT_KEYS, temp=False)
        print(f"### bytes, route={route}")
        table(directory, route, BYTE_KEYS, temp=True)
    if len(argv) > 2:
        print("### both arms against the earlier ledger (counts + bytes_written)")
        ok = check_against_committed(directory, Path(argv[2]))
        print("compiled modules unchanged since that ledger:", ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
