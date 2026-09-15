#!/usr/bin/env python3
"""Q20 gyaradax (GKW port, 8d9dc2d2) s-alpha Cyclone growth rate for one ky and one grid rung.

    python gyaradax_salpha.py --ky-gx 0.30 --rung g2

Units: gyaradax/GKW use Lref = R, vth = sqrt(2T/m); krhomax = sqrt(2) ky_gx and
gamma_gx = gamma * sqrt(2) / 2.77778. Physics as manifest.toml: q 1.4, shat 0.8, eps 0.18,
R/LT = 2.49 * 2.77778, R/Ln = 0.8 * 2.77778, adiabatic electrons, collisionless, electrostatic.
Driven exactly like gyaradax's own CBC unit test (tests/unit/test_gk_cases.py): gk_run in windows of
naverage steps, float64 (mixed_precision=False), disp_par = 1 (GKW 4th-order parallel dissipation,
idisp 2), disp_vp = disp_x = disp_y = 0. gyaradax renormalizes each window and reports gamma for it;
no frequency is available on this route. Prints one line "RESULT {json}".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import time

T0 = time.perf_counter()
RUNGS = {
    "g1": dict(nperiod=3, ns=96, nvpar=32, nmu=8, dt=0.003),
    "g2": dict(nperiod=5, ns=144, nvpar=64, nmu=16, dt=0.003),
    "g3": dict(nperiod=5, ns=288, nvpar=96, nmu=24, dt=0.0015),
}
T_MAX = {0.15: 200.0, 0.30: 120.0, 0.40: 120.0, 0.50: 200.0, 0.55: 400.0}  # R/vth units
R_OVER_A = 2.77778

parser = argparse.ArgumentParser()
parser.add_argument("--ky-gx", type=float, required=True)
parser.add_argument("--rung", choices=sorted(RUNGS), required=True)
parser.add_argument("--naverage", type=int, default=100)
args = parser.parse_args()

from gyaradax.jax_config import enable_x64  # noqa: E402

enable_x64()
import jax  # noqa: E402
import numpy as np  # noqa: E402

import gyaradax  # noqa: E402
from gyaradax.geometry import compute_geometry  # noqa: E402
from gyaradax.params import GKParams  # noqa: E402
from gyaradax.precompute import linear_precompute  # noqa: E402
from gyaradax.simulate import gk_run  # noqa: E402
from gyaradax.solver import default_state, init_f  # noqa: E402

r = RUNGS[args.rung]
krho = math.sqrt(2.0) * args.ky_gx
t_max = T_MAX[round(args.ky_gx, 2)]
nwin = int(math.ceil(t_max / (r["dt"] * args.naverage)))
record = {
    "ky_gx": args.ky_gx,
    "krhomax": krho,
    "rung": args.rung,
    **r,
    "naverage": args.naverage,
    "t_max": nwin * r["dt"] * args.naverage,
    "host": platform.node(),
    "jax": jax.__version__,
    "devices": [str(d) for d in jax.devices()],
    "gyaradax_file": gyaradax.__file__,
    "x64": bool(jax.config.jax_enable_x64),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}

geom = compute_geometry(
    q=1.4,
    shat=0.8,
    eps=0.18,
    ns=r["ns"],
    nvpar=r["nvpar"],
    nmu=r["nmu"],
    vpar_max=3.0,
    nkx=1,
    nky=1,
    nperiod=r["nperiod"],
    kxmax=krho,
    signB=1.0,
    Rref=1.0,
    krhomax=krho,
    geom_type="s-alpha",
)
params = GKParams(
    dt=r["dt"],
    naverage=args.naverage,
    non_linear=False,
    adaptive_dt=False,
    adiabatic_electrons=True,
    disp_par=1.0,
    disp_vp=0.0,
    disp_x=0.0,
    disp_y=0.0,
    finit="cosine2",
    amp_init=1e-4,
    mas=1.0,
    signz=1.0,
    tmp=1.0,
    de=1.0,
    vthrat=1.0,
    rlt=2.49 * R_OVER_A,
    rln=0.8 * R_OVER_A,
    dgrid=1.0,
    tgrid=1.0,
    sgr_dist=float(geom["sgr_dist"]),
    dvp=float(geom["dvp"]),
    kxmax=krho,
    kymax=krho,
    norm_eps=1e-14,
    drive_scale=1.0,
    idisp=2,
    cfl_safety=0.95,
    mixed_precision=False,
    backend="jax",
    use_z2z=False,
)
df = init_f(geom, finit="cosine2", amp_init_real=1e-4)
pre = linear_precompute(geom, params)
state = default_state(nky=1)
growth = []
t_first = None
for i in range(nwin):
    df, phi, _, state = gk_run(df, geom, params, state, params.naverage, pre=pre)
    g = float(np.asarray(state.last_growth_rate)[0])
    if i == 0:
        t_first = time.perf_counter() - T0
    growth.append(g)
    if not math.isfinite(g):
        break
g = np.asarray(growth)
n = len(g)
last = g[int(0.8 * n) :]
prev = g[int(0.6 * n) : int(0.8 * n)]
gamma = float(last.mean())
record.update(
    {
        "windows": n,
        "gamma_code": gamma,
        "gamma_gx": gamma * math.sqrt(2.0) / R_OVER_A,
        "drift": float((last.mean() - prev.mean()) / gamma) if prev.size else None,
        "gamma_last_window": float(g[-1]),
        "growth_every_10th_window": [float(x) for x in g[::10]],
        "wall_s": time.perf_counter() - T0,
        "first_window_incl_compile_s": t_first,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
)
print("RESULT " + json.dumps(record), flush=True)
