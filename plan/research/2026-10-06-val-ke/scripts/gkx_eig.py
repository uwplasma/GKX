#!/usr/bin/env python3
"""One GKX linear eigenpair: python gkx_eig.py <deck> <ky_gkx> <Nl> <Nm> [solver] [key=value ...]

key=value overrides set [grid]/[physics] fields (e.g. ntheta=48 nperiod=3 beta=6.67e-4).
Prints one line "RESULT {json}" with the git SHA and host type.
"""

import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
from gkx.runtime import run_runtime_linear  # noqa: E402
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402

deck, ky, nl, nm = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
solver = sys.argv[5] if len(sys.argv) > 5 else "krylov"
over = dict(a.split("=") for a in sys.argv[6:])
cfg, _ = load_runtime_from_toml(deck)
for key, val in over.items():
    sect = "grid" if hasattr(cfg.grid, key) else "physics"
    cur = getattr(getattr(cfg, sect), key)
    setattr_val = type(cur)(float(val)) if isinstance(cur, (int, float)) else val
    cfg = replace(cfg, **{sect: replace(getattr(cfg, sect), **{key: setattr_val})})
if "nperiod" in over or "ntheta" in over:
    g = cfg.grid
    cfg = replace(cfg, grid=replace(g, Nz=g.ntheta * (2 * g.nperiod - 1)))
sha = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
t0 = time.perf_counter()
r = run_runtime_linear(cfg, ky_target=ky, Nl=nl, Nm=nm, solver=solver)
print("RESULT", json.dumps(dict(
    deck=Path(deck).name, ky=float(r.ky), Nl=nl, Nm=nm, solver=solver, over=over,
    Nz=cfg.grid.Nz, gamma=float(r.gamma), omega=float(r.omega), sha=sha,
    host=f"{platform.system()}-{platform.machine()}-cpu", wall_s=round(time.perf_counter() - t0, 1))), flush=True)
