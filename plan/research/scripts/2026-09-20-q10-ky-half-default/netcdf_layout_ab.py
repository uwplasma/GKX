"""Write the shipped nonlinear NetCDF bundle on one ky layout, and dump it.

Usage: python netcdf_layout_ab.py OUT.npz --ky-layout full|half [--Ny 16 ...]

The published bundle is the user-visible surface of plan 5.3 N3's default
flip: whatever else the half layout changes, a run's ``.out.nc``,
``.restart.nc`` and ``.big.nc`` have to carry the same variables, on the same
dimensions, with the same numbers.  This script runs one arm; ``compare_nc.py``
compares two dumps variable by variable.

Everything is dumped, including the Grids and Geometry groups, so a dimension
that silently shortened (an ``Nyc``-long ``y`` axis, a ``ky`` axis of
``1 + (Nyc - 1) // 3`` rows) shows up as a shape difference rather than as a
plausible-looking file.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from gkx.workflows.runtime.artifacts import run_runtime_nonlinear_with_artifacts
from gkx.workflows.runtime.toml import load_runtime_from_toml

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--ky-layout", choices=("full", "half"), required=True)
ap.add_argument("--deck", type=Path, default=None)
ap.add_argument("--Nx", type=int, default=16)
ap.add_argument("--Ny", type=int, default=16)
ap.add_argument("--Nz", type=int, default=12)
ap.add_argument("--Nl", type=int, default=2)
ap.add_argument("--Nm", type=int, default=4)
ap.add_argument("--dt", type=float, default=0.01)
ap.add_argument("--t-max", type=float, default=0.2)
ap.add_argument("--ky-target", type=float, default=0.3)
args = ap.parse_args()

DECK = args.deck or Path(
    "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"
)
cfg, _ = load_runtime_from_toml(DECK)
cfg = replace(
    cfg,
    grid=replace(
        cfg.grid, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz, ky_layout=args.ky_layout
    ),
    init=replace(cfg.init, init_amp=0.05),
    time=replace(
        cfg.time,
        dt=args.dt,
        t_max=args.t_max,
        diagnostics=True,
        fixed_dt=True,
        run_to="t_max",
    ),
)

tmp = Path(tempfile.mkdtemp(prefix="gkx-ky-nc-"))
out_nc = tmp / "arm.out.nc"
cfg = replace(cfg, output=replace(cfg.output, path=str(out_nc), save_for_restart=True))
_result, paths = run_runtime_nonlinear_with_artifacts(
    cfg, out=str(out_nc), ky_target=args.ky_target, Nl=args.Nl, Nm=args.Nm
)


def dump(path: Path, tag: str, sink: dict[str, np.ndarray]) -> None:
    from netCDF4 import Dataset

    def walk(group, prefix: str) -> None:
        for dim_name, dim in group.dimensions.items():
            sink[f"{tag}:{prefix}#dim:{dim_name}"] = np.asarray(len(dim))
        for var_name, var in group.variables.items():
            sink[f"{tag}:{prefix}{var_name}"] = np.asarray(var[...])
        for sub_name, sub in group.groups.items():
            walk(sub, f"{prefix}{sub_name}/")

    with Dataset(path, "r") as root:
        walk(root, "")


sink: dict[str, np.ndarray] = {}
meta = {"gkx": "src/gkx/__init__.py", "ky_layout": args.ky_layout, "paths": {}}
for key, path in sorted(paths.items()):
    p = Path(str(path))
    meta["paths"][key] = p.name
    if p.suffix == ".nc" and p.exists():
        dump(p, key, sink)
    elif p.suffix == ".json" and p.exists():
        meta[f"summary:{key}"] = json.loads(p.read_text())
np.savez(args.out, **sink)
args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
print(args.ky_layout, "variables dumped:", len(sink))
