#!/usr/bin/env python3
"""Final-window growth rates of the GS2/stella runs and GKX RESULT logs, in GKX units.

    python collect_xcode.py <root>      (run inside the gk-fortran env for netCDF4)

GS2/stella: mean of gamma over the last 20% of the trace, drift = change against
the preceding 20%; gamma_gkx = sqrt(2) gamma_code (vref = sqrt(2T/m)).
"""

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])


def window(t, g, o):
    ok = np.isfinite(g) & np.isfinite(o)
    t, g, o = t[ok], g[ok], o[ok]
    late, prev = t >= 0.8 * t[-1], (t >= 0.6 * t[-1]) & (t < 0.8 * t[-1])
    return (
        g[late].mean(),
        o[late].mean(),
        (g[late].mean() - g[prev].mean()) / abs(g[late].mean()),
    )


print("code,case,gamma_gkx,omega_gkx,drift")
for nc in sorted(glob.glob(str(root / "gs2" / "*" / "*.out.nc"))):
    if not (Path(nc).parent / "DONE").exists():
        continue
    from netCDF4 import Dataset

    ds = Dataset(nc)
    om = np.asarray(ds.variables["omega"][:], float)
    g, o, d = window(
        np.asarray(ds.variables["t"][:], float), om[:, 0, -1, 1], om[:, 0, -1, 0]
    )
    print(
        f"gs2,{Path(nc).parent.name},{math.sqrt(2) * g:.5f},{math.sqrt(2) * o:.5f},{d:+.1e}"
    )
for f in sorted(glob.glob(str(root / "stella" / "*" / "*.omega"))):
    if not (Path(f).parent / "DONE").exists():
        continue
    a = np.loadtxt(f)
    g, o, d = window(a[:, 0], a[:, 4], a[:, 3])
    print(
        f"stella,{Path(f).parent.name},{math.sqrt(2) * g:.5f},{math.sqrt(2) * o:.5f},{d:+.1e}"
    )
for log in sorted(glob.glob(str(root / "logs" / "*.log"))):
    for line in open(log):
        if line.startswith("RESULT"):
            r = json.loads(line[7:])
            print(
                f"gkx,{Path(log).stem},{r['gamma']:.5f},{r['omega']:.5f},{r['eigen_status'][:60]!r}"
            )
