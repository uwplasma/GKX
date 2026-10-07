#!/usr/bin/env python3
"""One GKX linear eigenpair: python gkx_eig.py <deck> <ky_gkx> <Nl> <Nm> [solver] [key=value ...]

key=value overrides set fields; a bare key is [grid] or [physics], else section.key
(e.g. ntheta=48 nperiod=3 terms.end_damping=0).
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
    sect, _, field = key.rpartition(".")
    sect = sect or ("grid" if hasattr(cfg.grid, field) else "physics")
    cur = getattr(getattr(cfg, sect), field)
    new = (
        float(val)
        if cur is None
        else type(cur)(float(val))
        if isinstance(cur, (int, float)) and not isinstance(cur, bool)
        else val
    )
    cfg = replace(cfg, **{sect: replace(getattr(cfg, sect), **{field: new})})
if "nperiod" in over or "ntheta" in over:
    g = cfg.grid
    cfg = replace(cfg, grid=replace(g, Nz=g.ntheta * (2 * g.nperiod - 1)))
sha = subprocess.run(
    ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
    capture_output=True,
    text=True,
).stdout.strip()
t0 = time.perf_counter()
r = run_runtime_linear(cfg, ky_target=ky, Nl=nl, Nm=nm, solver=solver)
print(
    "RESULT",
    json.dumps(
        dict(
            deck=Path(deck).name,
            ky=float(r.ky),
            Nl=nl,
            Nm=nm,
            solver=solver,
            over=over,
            Nz=cfg.grid.Nz,
            gamma=float(r.gamma),
            omega=float(r.omega),
            sha=sha,
            host=f"{platform.system()}-{platform.machine()}-{jax.devices()[0].platform}",
            wall_s=round(time.perf_counter() - t0, 1),
            eigen_status=str(r.eigen_status),
            fit_settled=r.fit_settled,
            fit_r2=r.fit_r2,
            window=[r.fit_window_tmin, r.fit_window_tmax],
        )
    ),
    flush=True,
)
if r.t is not None and r.signal is not None:
    import numpy as np

    t, a = np.asarray(r.t), np.log(np.abs(np.asarray(r.signal)) + 1e-300)
    n = len(t) // 8
    print(
        "LOCAL_GROWTH",
        [
            round(float((a[i + n] - a[i]) / (t[i + n] - t[i])), 4)
            for i in range(0, len(t) - n, n)
        ],
    )
