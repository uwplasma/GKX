#!/usr/bin/env python3
"""python eig.py <deck> <ky> <Nl> <Nm> <method> [key=value ...]  (method: banded|adaptive|time)

key=value: section.field overrides (as gkx_eig.py); kr.shift=1+0j etc. for KrylovConfig.
Prints RESULT {json} with sha, host, wall, peak RSS, status."""
import json, platform, resource, subprocess, sys, time
from dataclasses import replace
from pathlib import Path
import jax
import os; jax.config.update("jax_enable_x64", os.environ.get("X64", "1") == "1")
from gkx.runtime import run_runtime_linear
from gkx.solvers_linear_krylov import KrylovConfig
from gkx.workflows.runtime.toml import load_runtime_from_toml

deck, ky, nl, nm, method = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
over = dict(a.split("=", 1) for a in sys.argv[6:])
cfg, _ = load_runtime_from_toml(deck)
kr = {}
for key, val in over.items():
    if key.startswith("kr."):
        kr[key[3:]] = complex(val) if key == "kr.shift" else val
        continue
    sect, _, field = key.rpartition(".")
    sect = sect or ("grid" if hasattr(cfg.grid, field) else "physics")
    cur = getattr(getattr(cfg, sect), field)
    if isinstance(cur, bool): new = val.lower() in ("1", "true")
    elif cur is None or isinstance(cur, (int, float)): new = float(val) if cur is None else type(cur)(float(val))
    else: new = val
    cfg = replace(cfg, **{sect: replace(getattr(cfg, sect), **{field: new})})
if "nperiod" in over or "ntheta" in over:
    g = cfg.grid
    cfg = replace(cfg, grid=replace(g, Nz=g.ntheta * (2 * g.nperiod - 1)))
solver = "time" if method == "time" else "krylov"
kcfg = None if method == "time" else KrylovConfig(method=method, **kr)
msgs = []
def cb(m):
    msgs.append(f"{time.perf_counter()-t0:7.1f}s {m}")
    print(msgs[-1], flush=True)
t0 = time.perf_counter()
try:
    r = run_runtime_linear(cfg, ky_target=ky, Nl=nl, Nm=nm, solver=solver, krylov_cfg=kcfg, status_callback=cb)
    out = dict(gamma=float(r.gamma), omega=float(r.omega), eigen_status=str(r.eigen_status), ky=float(r.ky))
except Exception as e:
    out = dict(error=f"{type(e).__name__}: {e}"[:400])
sha = subprocess.run(["git", "-C", str(Path(__file__).parent / "GKX"), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print("RESULT", json.dumps(dict(deck=Path(deck).name, Nl=nl, Nm=nm, method=method, over=over, sha=sha,
    host=f"{platform.system()}-{platform.machine()}-{jax.devices()[0].platform}",
    wall_s=round(time.perf_counter() - t0, 1), peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1), **out)), flush=True)
