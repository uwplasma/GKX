# ruff: noqa: E402
"""D3: Laguerre gyroaverage capture along the GX Nl24/32 discriminator chain (read-only review).

GX deck (office full96.in): s-alpha, eps .18, q 1.4, shat .8, shift 0, ntheta32,
nperiod2 (Nz96, three 2pi segments), nkx=1, y0=1.8181817787737895 (ky=.55).
Uses GKX's own cache b and J_l_all; no simulation.
"""

import sys
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO = Path("/Users/rogeriojorge/local/GKX-worktrees/perf-review")
sys.path.insert(0, str(REPO))
from gkx.runtime import _runtime_linear_dispatch_deps  # noqa: E402
from gkx.workflows.linear import _prepare_linear_runtime_context  # noqa: E402
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402
from gkx.core_velocity import J_l_all, gamma0  # noqa: E402

cfg, _ = load_runtime_from_toml(REPO / "examples/linear/axisymmetric/cyclone.toml")
print("geometry", cfg.geometry, flush=True)
cfg = replace(
    cfg,
    grid=replace(
        cfg.grid, Nx=1, Ny=4, Nz=96, ntheta=32, nperiod=2, y0=1.8181817787737895
    ),
)
deps = _runtime_linear_dispatch_deps().full_deps
ctx = _prepare_linear_runtime_context(
    cfg,
    deps=deps,
    ky_target=0.55,
    n_laguerre=32,
    n_hermite=8,
    solver="krylov",
    fit_signal="auto",
    return_state=False,
    initial_state=None,
    status_callback=None,
)
cache = deps.build_linear_cache(ctx.grid, ctx.geom, ctx.params, 32, 8)
b = np.asarray(cache.b)
z = np.asarray(ctx.grid.z).reshape(-1)
print(
    "ky",
    np.asarray(ctx.grid.ky).reshape(-1),
    "kx",
    np.asarray(ctx.grid.kx).reshape(-1),
    "b shape",
    b.shape,
    "z range",
    float(z.min()),
    float(z.max()),
    flush=True,
)
bz = b.reshape(-1, b.shape[-1]).max(axis=0)  # max over species/ky/kx at each z
print(
    f"b(theta): center={bz[np.argmin(np.abs(z))]:.3f} max={bz.max():.3f} at theta={z[np.argmax(bz)]:.3f} "
    f"(= {z[np.argmax(bz)] / np.pi:.2f} pi)",
    flush=True,
)
for theta_pi in (1, 2, 2.5, 3):
    i = np.argmin(np.abs(np.abs(z) - theta_pi * np.pi))
    print(f"  |theta|~{theta_pi}pi: b={bz[i]:.3f}", flush=True)
g0 = np.asarray(gamma0(jnp.asarray(bz)))
jl = np.asarray(J_l_all(jnp.asarray(bz), 63))
for nl in (4, 8, 16, 24, 32, 48):
    captured = (jl[:nl] ** 2).sum(axis=0) / g0
    print(
        f"Nl={nl:2d}: min Gamma0 capture along chain={captured.min():.6f} "
        f"(1-capture max={1 - captured.min():.2e})",
        flush=True,
    )
# Laguerre index where |J_l| peaks at the chain end, and J_{N-1} relative size
iend = np.argmax(bz)
print(
    f"at b_max: argmax_l |J_l|={int(np.argmax(np.abs(jl[:, iend])))}  "
    f"|J_23|/max={abs(jl[23, iend]) / np.abs(jl[:, iend]).max():.2e} "
    f"|J_31|/max={abs(jl[31, iend]) / np.abs(jl[:, iend]).max():.2e}",
    flush=True,
)
