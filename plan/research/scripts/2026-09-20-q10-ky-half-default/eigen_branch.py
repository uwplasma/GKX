"""Q10: what the ky layout does, and does not, do to the eigen branch.

Usage: python eigen_branch.py OUT.json

#258 left a question open: a two-sided ``ky`` axis carries every eigenvalue
and its ``lambda*`` copy, because ``L(-ky)`` is the conjugate of ``L(+ky)``,
and a half axis does not -- so does frequency-sign branch selection move?

The answer this script measures is *no, for a different reason than the
question assumed*.  Every eigen route in GKX selects **one** ``ky`` row before
it builds an operator (``select_ky_grid``), so the spectrum it diagonalises
never contained a conjugate pair on either axis.  What the layout can move is
**which row the selection lands on**, and that is a one-row question with a
sharp answer: an even grid stores ``|ky| = Ny/2`` once, as ``-Ny/2`` on the
two-sided axis and as ``+Ny/2`` on the half one (#258, "the one row the
layouts disagree on").  A request whose nearest magnitude is that row
therefore comes back with the opposite sign of ``ky``, and with it the
opposite sign of ``omega``; ``gamma`` is untouched, because ``gamma`` is even
under ``ky -> -ky`` and ``omega`` is odd.

Such a request is one the grid cannot resolve -- the Nyquist row is above the
two-thirds dealias cutoff and outside the physical band -- so this is a
mis-specified run reporting a sign convention, not a physics change.  It is
recorded rather than papered over.

The script reports three things:
  * every dealiased target on a sweep of ``Ny``: same row, same sign, both
    layouts;
  * the Nyquist case, explicitly, with both signs printed;
  * the shipped Cyclone linear deck at its own resolution, run end to end
    through the eigen solver on both layouts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from gkx.config import GridConfig
from gkx.core_grid import build_spectral_grid
from gkx.diagnostics.modes import select_ky_index
from gkx.geometry import apply_imported_geometry_grid_defaults
from gkx.runtime import build_runtime_geometry, run_runtime_linear
from gkx.workflows.runtime.toml import load_runtime_from_toml

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--deck", type=Path, default=None)
args = ap.parse_args()

report: dict[str, object] = {"gkx": "src/gkx/__init__.py"}

# --- 1. Every dealiased target picks the same physical row on both axes. ---
sweep = []
mismatch = 0
for ny in (8, 12, 16, 17, 24, 32, 33, 48):
    cfg = GridConfig(Nx=8, Ny=ny, Nz=8, Lx=6.0, Ly=6.0)
    grids = {
        layout: build_spectral_grid(replace(cfg, ky_layout=layout))
        for layout in ("full", "half")
    }
    ky_half = np.asarray(grids["half"].ky, dtype=float)
    # The dealiased band: |ky| < Ny/3, i.e. rows 0 .. (Ny - 1) // 3.
    for row in range(1 + (ny - 1) // 3):
        target = float(ky_half[row])
        if target == 0.0:
            continue
        picked = {
            layout: float(np.asarray(g.ky, dtype=float)[select_ky_index(g.ky, target)])
            for layout, g in grids.items()
        }
        same = picked["full"] == picked["half"]
        mismatch += 0 if same else 1
        sweep.append(
            {"Ny": ny, "row": row, "target": target, **picked, "same": bool(same)}
        )
report["dealiased_sweep_cases"] = len(sweep)
report["dealiased_sweep_mismatches"] = mismatch
report["dealiased_sweep"] = sweep

# --- 2. The Nyquist row, which is the one case that does move. ---
nyquist = []
for ny in (8, 16, 32):
    cfg = GridConfig(Nx=8, Ny=ny, Nz=8, Lx=6.0, Ly=6.0)
    grids = {
        layout: build_spectral_grid(replace(cfg, ky_layout=layout))
        for layout in ("full", "half")
    }
    nyq = float(np.abs(np.asarray(grids["half"].ky, dtype=float))[ny // 2])
    target = nyq * 1.2  # beyond the band: nothing resolvable is nearer
    picked = {
        layout: float(np.asarray(g.ky, dtype=float)[select_ky_index(g.ky, target)])
        for layout, g in grids.items()
    }
    nyquist.append(
        {
            "Ny": ny,
            "target": target,
            "nyquist_magnitude": nyq,
            "dealias_cutoff_row": (ny - 1) // 3,
            "nyquist_row": ny // 2,
            "inside_dealiased_band": (ny // 2) <= ((ny - 1) // 3),
            **picked,
            "sign_flips": picked["full"] != picked["half"],
        }
    )
report["nyquist"] = nyquist

# --- 3. The shipped linear deck, end to end, on both axes. ---
DECK = args.deck or Path("examples/linear/axisymmetric/cyclone.toml")
cfg, _ = load_runtime_from_toml(DECK)
geom = build_runtime_geometry(cfg)
grid_cfg = apply_imported_geometry_grid_defaults(geom, cfg.grid)
report["deck"] = {
    "path": str(DECK),
    "Nx": int(grid_cfg.Nx),
    "Ny": int(grid_cfg.Ny),
    "Nz": int(grid_cfg.Nz),
}
runs = {}
for layout in ("full", "half"):
    res = run_runtime_linear(
        replace(cfg, grid=replace(cfg.grid, ky_layout=layout)),
        ky_target=0.55,
        solver="eigen",
    )
    runs[layout] = {
        "ky": float(res.ky),
        "gamma": float(res.gamma),
        "omega": float(res.omega),
        "eigen_status": str(getattr(res, "eigen_status", None)),
    }
report["deck_runs"] = runs
report["deck_gamma_rel"] = abs(runs["half"]["gamma"] - runs["full"]["gamma"]) / max(
    abs(runs["full"]["gamma"]), 1e-300
)
report["deck_omega_rel"] = abs(runs["half"]["omega"] - runs["full"]["omega"]) / max(
    abs(runs["full"]["omega"]), 1e-300
)
report["deck_same_ky"] = runs["half"]["ky"] == runs["full"]["ky"]

args.out.write_text(json.dumps(report, indent=2))
print(
    json.dumps(
        {k: v for k, v in report.items() if k not in ("dealiased_sweep",)},
        indent=2,
    )
)
