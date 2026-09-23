"""Compare two NetCDF-bundle dumps variable by variable.

Usage: python compare_nc.py REF.npz NEW.npz OUT.json [--rtol 1e-5]

Prints, and records, one row per variable: shape agreement and the maximum
relative difference, with "bitwise" reserved for an exact zero.  A variable
present in one dump and not the other, or one whose shape moved, is a failure
on its own -- those are the ways a published file changes without any number
looking wrong.

Every row also carries ``max_abs`` and ``ref_peak``, because ``max_rel`` is
normalized by the variable's own peak and says nothing on its own about a
variable that is numerically zero.  ``TurbulentHeating`` is exactly that on an
electrostatic deck: the diagnostic is a difference of two O(1) terms that
cancel, it peaks at 3e-17 against a field energy of 2e-2, and the two layouts'
answers for it differ in the sixteenth significant figure of the terms.  Read
as ``max_rel`` alone that is 1.4e-3 and looks like a result moving; read with
``ref_peak`` it is a ratio of two roundoff residues.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ref", type=Path)
ap.add_argument("new", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--rtol", type=float, default=1.0e-5)
args = ap.parse_args()

ref = np.load(args.ref)
new = np.load(args.new)
keys = sorted(set(ref.files) | set(new.files))

rows = []
worst = 0.0
n_bitwise = 0
failures = []
for key in keys:
    if key not in ref.files or key not in new.files:
        failures.append({"key": key, "why": "missing", "in_ref": key in ref.files})
        continue
    a, b = ref[key], new[key]
    if a.shape != b.shape:
        failures.append(
            {"key": key, "why": "shape", "ref": list(a.shape), "new": list(b.shape)}
        )
        continue
    if a.size == 0:
        rows.append({"key": key, "max_rel": 0.0, "max_abs": 0.0, "ref_peak": 0.0})
        n_bitwise += 1
        continue
    x = np.asarray(a, dtype=np.complex128)
    y = np.asarray(b, dtype=np.complex128)
    # A non-finite entry is only a failure when the two arms disagree about
    # it.  Some shipped variables are NaN by construction on this deck
    # (`Inputs/surfarea`), and calling that a layout difference would bury the
    # real rows under a fixed one.
    bad_ref = ~np.isfinite(x)
    bad_new = ~np.isfinite(y)
    if not np.array_equal(bad_ref, bad_new):
        failures.append({"key": key, "why": "nonfinite-pattern"})
        continue
    if bad_ref.all():
        rows.append(
            {
                "key": key,
                "max_rel": 0.0,
                "max_abs": 0.0,
                "ref_peak": 0.0,
                "note": "non-finite in both arms",
            }
        )
        n_bitwise += 1
        continue
    x = np.where(bad_ref, 0.0, x)
    y = np.where(bad_new, 0.0, y)
    peak = float(np.max(np.abs(x)))
    scale = peak if peak > 0.0 else 1.0
    abs_diff = float(np.max(np.abs(x - y)))
    rel = abs_diff / scale
    rows.append({"key": key, "max_rel": rel, "max_abs": abs_diff, "ref_peak": peak})
    worst = max(worst, rel)
    if rel == 0.0:
        n_bitwise += 1

over = [r for r in rows if r["max_rel"] > args.rtol]
summary = {
    "ref": str(args.ref),
    "new": str(args.new),
    "variables": len(rows),
    "bitwise": n_bitwise,
    "max_rel": worst,
    "rtol": args.rtol,
    "over_rtol": sorted(over, key=lambda r: -r["max_rel"]),
    "failures": failures,
    "rows": sorted(rows, key=lambda r: -r["max_rel"]),
}
args.out.write_text(json.dumps(summary, indent=2))
print(f"variables {len(rows)}  bitwise {n_bitwise}  max_rel {worst:.3e}")
for row in summary["over_rtol"][:25]:
    print(
        f"  over rtol {row['max_rel']:.3e}"
        f"  (max_abs {row['max_abs']:.3e} of peak {row['ref_peak']:.3e})"
        f"  {row['key']}"
    )
for fail in failures[:25]:
    print(f"  FAIL {fail}")
raise SystemExit(1 if failures else 0)
