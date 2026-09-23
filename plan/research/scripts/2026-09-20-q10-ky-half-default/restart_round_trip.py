"""Q10: a restart file written on either ky layout loads onto either one.

Usage: python restart_round_trip.py OUT.json [--Ny 16 ...]

The restart file has always stored the dealiased ``ky >= 0`` block, so the
layout flip must not make an existing file unreadable and must not make a new
file unreadable by the old axis.  This checks all four directions on both file
forms (NetCDF and the raw binary seed), and -- the part that matters -- that
the state that comes back is the *same physical field* whichever way it went:
the two-sided reload of a half-written file is the Hermitian widening of the
half reload, exactly.

A file written by GKX 2.1.0 is the ``full``-arm file here by construction:
2.1.0 had no layout key, its runtime built the two-sided axis, and its writer
stored the same dealiased ``ky >= 0`` block this one does.  The
``schema_version`` is checked on read and is unchanged.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from gkx.artifacts.io import load_netcdf_restart_state
from gkx.core_ky_layout import to_full, to_half
from gkx.workflows.runtime.artifacts import run_runtime_nonlinear_with_artifacts
from gkx.workflows.runtime.startup import _load_initial_state_from_file
from gkx.workflows.runtime.toml import load_runtime_from_toml

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--Nx", type=int, default=16)
ap.add_argument("--Ny", type=int, default=16)
ap.add_argument("--Nz", type=int, default=12)
ap.add_argument("--Nl", type=int, default=2)
ap.add_argument("--Nm", type=int, default=4)
args = ap.parse_args()

DECK = Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml")
tmp = Path(tempfile.mkdtemp(prefix="gkx-ky-restart-"))
report: dict[str, object] = {
    "gkx": "src/gkx/__init__.py",
    "grid": [args.Nx, args.Ny, args.Nz],
}
written: dict[str, Path] = {}
states: dict[str, np.ndarray] = {}

for layout in ("full", "half"):
    cfg, _ = load_runtime_from_toml(DECK)
    out_nc = tmp / f"{layout}.out.nc"
    cfg = replace(
        cfg,
        grid=replace(cfg.grid, Nx=args.Nx, Ny=args.Ny, Nz=args.Nz, ky_layout=layout),
        init=replace(cfg.init, init_amp=0.05),
        time=replace(
            cfg.time,
            dt=0.01,
            t_max=0.05,
            diagnostics=True,
            fixed_dt=True,
            run_to="t_max",
        ),
        output=replace(cfg.output, path=str(out_nc), save_for_restart=True),
    )
    result, paths = run_runtime_nonlinear_with_artifacts(
        cfg, out=str(out_nc), ky_target=0.3, Nl=args.Nl, Nm=args.Nm
    )
    written[layout] = Path(str(paths["restart"]))
    states[layout] = np.asarray(result.state)
    report[f"state_shape_{layout}"] = list(states[layout].shape)

    # The raw binary seed form, which `init_file` also accepts.
    raw = tmp / f"{layout}.seed.bin"
    raw.write_bytes(np.asarray(states[layout], dtype=np.complex64).tobytes())
    written[f"{layout}_bin"] = raw

checks: list[dict[str, object]] = []
for source, path in sorted(written.items()):
    for target in ("full", "half"):
        loaded = _load_initial_state_from_file(
            path,
            nspecies=states["full"].shape[0],
            Nl=args.Nl,
            Nm=args.Nm,
            ny=args.Ny,
            nx=args.Nx,
            nz=int(states["full"].shape[-1]),
            ky_layout=target,
        )
        arr = np.asarray(loaded)
        rows = int(arr.shape[3])
        expect = args.Ny if target == "full" else 1 + args.Ny // 2
        checks.append(
            {
                "file": path.name,
                "source_layout": source,
                "target_layout": target,
                "rows": rows,
                "rows_expected": expect,
                "ok": rows == expect,
            }
        )

# The two targets must be the same physical field, exactly.
pairs = []
for source, path in sorted(written.items()):
    kw = dict(
        nspecies=states["full"].shape[0],
        Nl=args.Nl,
        Nm=args.Nm,
        ny=args.Ny,
        nx=args.Nx,
        nz=int(states["full"].shape[-1]),
    )
    as_full = np.asarray(_load_initial_state_from_file(path, ky_layout="full", **kw))
    as_half = np.asarray(_load_initial_state_from_file(path, ky_layout="half", **kw))
    widened = np.asarray(to_full(as_half, ny_full=args.Ny))
    narrowed = np.asarray(to_half(as_full, ny_full=args.Ny))
    pairs.append(
        {
            "file": path.name,
            "widen(half) == full: max_abs": float(np.max(np.abs(widened - as_full))),
            "narrow(full) == half: max_abs": float(np.max(np.abs(narrowed - as_half))),
        }
    )

# A 2.1.0-shaped NetCDF restart (two-sided run, same writer, same schema)
# reloaded onto the half axis must be the half block of its own full reload.
nc_full = written["full"]
kw = dict(
    nspecies=states["full"].shape[0],
    Nl=args.Nl,
    Nm=args.Nm,
    ny=args.Ny,
    nx=args.Nx,
    nz=int(states["full"].shape[-1]),
)
legacy_full = np.asarray(load_netcdf_restart_state(nc_full, ky_layout="full", **kw))
legacy_half = np.asarray(load_netcdf_restart_state(nc_full, ky_layout="half", **kw))
report["legacy_netcdf"] = {
    "file": nc_full.name,
    "full_rows": int(legacy_full.shape[3]),
    "half_rows": int(legacy_half.shape[3]),
    "half == to_half(full): max_abs": float(
        np.max(np.abs(legacy_half - np.asarray(to_half(legacy_full, ny_full=args.Ny))))
    ),
}

report["row_counts"] = checks
report["layout_agreement"] = pairs
report["all_row_counts_ok"] = all(bool(c["ok"]) for c in checks)
args.out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
