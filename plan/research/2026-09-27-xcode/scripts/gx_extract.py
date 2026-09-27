#!/usr/bin/env python3
"""Print final gamma/omega per ky from GX .out.nc files (omega_kxkyt, kx index 0).

python gx_extract.py file.out.nc [...]
"""

import sys
import numpy as np
from netCDF4 import Dataset

for path in sys.argv[1:]:
    ds = Dataset(path)
    g = ds.groups["Grids"] if "Grids" in ds.groups else ds
    ky = np.asarray(g.variables["ky"][:])
    diag = ds.groups["Diagnostics"] if "Diagnostics" in ds.groups else ds
    om = np.asarray(diag.variables["omega_kxkyt"][:])  # (t, ky, kx, ri)
    t = np.asarray(g.variables["time"][:])
    print(f"# {path} t_end={t[-1]:.3f}")
    for i, k in enumerate(ky):
        if k == 0:
            continue
        w, gam = om[-1, i, 0, 0], om[-1, i, 0, 1]
        print(f"{k:.4f},{gam:.6f},{w:.6f}")
