"""Compare two npz archives key by key: bitwise flag and ||a-b||/||a||.

Usage: python compare_npz.py REF.npz OTHER.npz [OUT.json]
"""

import json
import sys

import numpy as np

ref = np.load(sys.argv[1])
other = np.load(sys.argv[2])
rows = {}
n_equal = 0
for key in sorted(ref.files):
    a = np.asarray(ref[key])
    if key not in other.files:
        rows[key] = {"missing": True}
        continue
    b = np.asarray(other[key])
    equal = (
        a.shape == b.shape
        and a.dtype == b.dtype
        and np.array_equal(a, b, equal_nan=True)
    )
    n_equal += int(equal)
    a64 = a.astype(np.complex128)
    b64 = b.astype(np.complex128)
    norm = float(np.linalg.norm(a64))
    diff = float(np.linalg.norm(a64 - b64))
    rows[key] = {
        "bitwise": bool(equal),
        "rel": 0.0 if equal else (diff / norm if norm > 0 else float("inf")),
        "abs_max": float(np.max(np.abs(a64 - b64))) if a.size else 0.0,
        "ref_norm": norm,
        "dtype": str(a.dtype),
    }
summary = {
    "n": len(ref.files),
    "bitwise": n_equal,
    "max_rel": max((r["rel"] for r in rows.values() if "rel" in r), default=0.0),
}
for key, row in rows.items():
    flag = "==" if row.get("bitwise") else f"rel {row.get('rel', float('nan')):.2e}"
    print(f"{key:40s} {flag}")
print(json.dumps(summary))
if len(sys.argv) > 3:
    with open(sys.argv[3], "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2)
