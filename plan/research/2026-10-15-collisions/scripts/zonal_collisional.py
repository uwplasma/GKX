#!/usr/bin/env python3
"""Collisional decay of the Rosenbluth-Hinton zonal residual vs Xiao, Catto & Molvig (2007).

    python zonal_collisional.py <nu_ii R0/v_ti> <Nl> <Nm> [operator] [t_max R0/v_ti]

s-alpha, q = 1.4, eps = 0.18, kx rho_i = 0.05, adiabatic electrons, ion-ion
collisions only. Prints a RESULT line with the trace and the Xiao et al.
prediction (Frei et al. 2021, JPP 87, 905870501, Eq. 5.2):
phi(t)/phi(0) = beta/(1+beta) [1 + (1-Theta)/(Theta+beta) exp(-(1+beta) Gamma nu t/(Theta+beta))],
beta = eps^2/q^2, Theta = 1.635 eps^1.5 + eps^2/2 + 0.360 eps^2.5,
Gamma = 0.4 (1.46 sqrt(eps) + 1.32 eps). Convention: nu_D(v) = (3 sqrt(pi)/4)
nu_ii [erf(x) - G(x)]/x^3, so GKX nu = (3 sqrt(pi)/4) nu_ii (in GKX time units).
"""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from gkx.runtime import run_runtime_linear  # noqa: E402
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402

nu_ii, nl, nm = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
operator = sys.argv[4] if len(sys.argv) > 4 else "landau"
t_max = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
q, eps, kx, dt = 1.4, 0.18, 0.05, 2.5e-3
deck = (
    Path(__file__).resolve().parents[4] / "benchmarks/cases/miller_zonal_response.toml"
)
cfg, _ = load_runtime_from_toml(deck)
r0 = 0.5 / eps
nu_gkx = 0.75 * math.sqrt(math.pi) * nu_ii / r0  # rate in v_ti/a units
cfg = replace(
    cfg,
    species=tuple(replace(s, nu=nu_gkx) for s in cfg.species),
    geometry=replace(
        cfg.geometry, model="s-alpha", q=q, s_hat=0.8, epsilon=eps, R0=r0, rhoc=0.5
    ),
    grid=replace(cfg.grid, Nz=32, Lx=2 * np.pi / kx),
    physics=replace(cfg.physics, collisions=True, hypercollisions=True),
    terms=replace(cfg.terms, collisions=1.0, hypercollisions=1.0),
    collisions=replace(cfg.collisions, hypercollisions_kz=1.0),
    time=replace(cfg.time, collision_operator=operator),
)
stride = max(1, int(round(0.05 / dt)))
res = run_runtime_linear(
    cfg,
    ky_target=0.0,
    kx_target=kx,
    Nl=nl,
    Nm=nm,
    solver="time",
    dt=dt,
    steps=stride * int(np.ceil(t_max * r0 / dt / stride)),
    sample_stride=stride,
    fit_signal="phi",
    require_positive=False,
    min_points=4,
)
field = np.asarray(res.field_history)[:, 0]
k = int(np.argmax(np.abs(field[0]).max(axis=1)))
trace = np.real(field[:, k].mean(axis=1))
t = np.asarray(res.t) / r0
beta, theta = eps**2 / q**2, 1.635 * eps**1.5 + eps**2 / 2 + 0.360 * eps**2.5
gam = 0.4 * (1.46 * math.sqrt(eps) + 1.32 * eps)
xiao = (
    beta
    / (1 + beta)
    * (
        1
        + (1 - theta)
        / (theta + beta)
        * np.exp(-(1 + beta) * gam * nu_ii * t / (theta + beta))
    )
)
print(
    "RESULT",
    json.dumps(
        dict(
            nu_ii=nu_ii,
            Nl=nl,
            Nm=nm,
            operator=operator,
            t=t[::20].round(3).tolist(),
            phi=(trace / trace[0])[::20].round(5).tolist(),
            xiao=xiao[::20].round(5).tolist(),
        )
    ),
)
