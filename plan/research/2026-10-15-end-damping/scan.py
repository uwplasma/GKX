#!/usr/bin/env python3
"""End-damping rate / domain-length scan on four linear decks.

    python scan.py <deck> <ky> <Nl> <Nm> <rate|legacy|default> <nperiod> <solver> [dt]

deck: cyc (examples/01 adiabatic Cyclone s-alpha), ke (examples/05 kinetic e-,
s-alpha), kem (examples/05 with the GX Cyclone Miller surface), kbm
(examples/06 s-alpha KBM), tem (benchmarks/cases/tem_linear.toml with the
standard 0.125 layer). rate: explicit [time] damp_ends_rate; 'legacy' keeps the
deck's damp_ends_amp-only spelling (no rate), 'default' omits both and uses
the code's default. Nz = ntheta (2 nperiod - 1). Prints RESULT {json}.
"""

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import gkx  # noqa: E402
from gkx.runtime import run_runtime_linear  # noqa: E402
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402

deck, ky, nl, nm, rate, nper, solver = sys.argv[1:8]
ky, nl, nm, nper = float(ky), int(nl), int(nm), int(nper)
dt = float(sys.argv[8]) if len(sys.argv) > 8 else None
root = Path(gkx.__file__).parents[2]
path = {
    "cyc": "examples/01_linear_tokamak/case_full.toml",
    "ke": "examples/05_kinetic_electrons/case_full.toml",
    "kem": "examples/05_kinetic_electrons/case_full.toml",
    "kbm": "examples/06_electromagnetic/case_full.toml",
    "tem": "benchmarks/cases/tem_linear.toml",
}[deck]
cfg, _ = load_runtime_from_toml(root / path)
ntheta = int(cfg.grid.ntheta)
cfg = replace(cfg, grid=replace(cfg.grid, nperiod=nper, Nz=ntheta * (2 * nper - 1)))
coll = cfg.collisions
if deck == "tem":
    coll = replace(coll, damp_ends_amp=0.1, damp_ends_widthfrac=0.125)
if deck == "kem":
    cfg = replace(
        cfg,
        geometry=replace(
            cfg.geometry,
            model="miller",
            rhoc=0.5,
            q=1.4,
            s_hat=0.8,
            R0=2.77778,
            R_geo=2.77778,
            shift=0.0,
            akappa=1.0,
            akappri=0.0,
            tri=0.0,
            tripri=0.0,
            betaprim=0.0,
        ),
    )
import math  # noqa: E402

import numpy as np  # noqa: E402

from gkx.workflows.runtime.startup import build_runtime_geometry  # noqa: E402

kin = [sp for sp in cfg.species if sp.kinetic]
vmax = max(math.sqrt(sp.temperature / sp.mass) for sp in kin)
gradpar = abs(float(np.asarray(build_runtime_geometry(cfg).gradpar())))
omega_L = vmax * gradpar / (coll.damp_ends_widthfrac * (2 * nper - 1) * 2 * math.pi)
t = cfg.time
if rate.startswith("c"):
    rate_val = float(rate[1:]) * omega_L
    t = replace(t, damp_ends_rate=rate_val)
elif rate == "legacy":
    t = replace(t, damp_ends_rate=None)
elif rate == "default":
    t = replace(t, damp_ends_rate=None)
    coll = replace(coll, damp_ends_amp=None) if hasattr(coll, "damp_ends_amp") else coll
else:
    t = replace(t, damp_ends_rate=float(rate))
if solver == "sparse":
    import gkx.runtime as _rt

    sys.path.insert(0, str(Path(__file__).parent))
    import sparse_eig

    _rt.dominant_eigenpair = sparse_eig.dominant_eigenpair
    import gkx.workflows.linear as _wl

    for _name in dir(_wl):
        if _name == "dominant_eigenpair":
            _wl.dominant_eigenpair = sparse_eig.dominant_eigenpair
    solver = "krylov"
cfg = replace(cfg, time=t, collisions=coll)
kw = {}
if solver == "time" and dt is not None:
    kw = dict(dt=dt, steps=2 * int(round(cfg.time.t_max / dt / 2)))
t0 = time.perf_counter()
res = run_runtime_linear(cfg, ky_target=ky, Nl=nl, Nm=nm, solver=solver, **kw)
wall = time.perf_counter() - t0
sha = subprocess.run(
    ["git", "-C", str(root), "rev-parse", "--short=9", "HEAD"],
    capture_output=True,
    text=True,
).stdout.strip()
row = dict(
    deck=deck,
    ky=float(res.ky),
    Nl=nl,
    Nm=nm,
    rate=rate,
    nperiod=nper,
    Nz=int(cfg.grid.Nz),
    solver=sys.argv[7],
    omega_L=omega_L,
    rate_used=t.damp_ends_rate,
    dt=dt,
    gamma=float(res.gamma),
    omega=float(res.omega),
    wall_s=round(wall, 1),
    sha=sha or os.environ.get("SHA"),
    host=f"{platform.system()}-{platform.machine()}-{jax.default_backend()}",
    status=str(getattr(res, "eigen_status", None))[:160],
)
print("RESULT", json.dumps(row), flush=True)
