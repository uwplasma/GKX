"""Q6 item 3: do linked kx rows outside the chains matter to linear time integration?

Pilot deck (linked Cyclone, Nx8/Ny16/Nz16, Nl4/Nm8, ky=.3, rate .1). Two arms:
A runtime initial condition; B the same plus off-chain content of equal peak
amplitude (what a user-supplied initial_state or restart could carry).
Reports off-chain amplitude before/after, fitted growth, and the CFL kx bound.
"""

import dataclasses
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gkx
from gkx.operators.linear.params import linear_terms_to_term_config
from gkx.runtime import _runtime_linear_dispatch_deps, run_runtime_linear
from gkx.solvers_time_explicit_cfl import _grid_frequency_bounds
from gkx.solvers_time_explicit_steps import _linear_explicit_step
from gkx.workflows.linear import _prepare_linear_runtime_context
from gkx.workflows.runtime.toml import load_runtime_from_toml

REPO = Path(gkx.__file__).resolve().parents[2]
print("gkx", gkx.__file__, flush=True)
cfg, _ = load_runtime_from_toml(REPO / "examples/linear/axisymmetric/cyclone.toml")
cfg = replace(
    cfg,
    grid=replace(cfg.grid, Nx=8, Ny=16, Nz=16, ntheta=16, nperiod=1, jtwist=1),
    time=replace(cfg.time, damp_ends_rate=0.1),
)
deps = _runtime_linear_dispatch_deps().full_deps
ctx = _prepare_linear_runtime_context(
    cfg,
    deps=deps,
    ky_target=0.3,
    n_laguerre=4,
    n_hermite=8,
    solver="explicit_time",
    fit_signal="auto",
    return_state=True,
    initial_state=None,
    status_callback=None,
)
ic = np.asarray(ctx.initial_state)
off = np.ones(ic.shape, dtype=bool)
off[..., [0, 1, 2, 6, 7], :] = False
kx = np.asarray(ctx.grid.kx)
bounds = _grid_frequency_bounds(ctx.grid)
print(
    f"state {ic.shape} n={ic.size} off-chain unknowns={int(off.sum())} "
    f"({off.mean():.3f}); IC off-chain max={np.max(np.abs(ic[off])):.3e} "
    f"IC max={np.max(np.abs(ic)):.3e}",
    flush=True,
)
print(
    f"kx grid {np.round(kx, 4).tolist()}; CFL kx_max={bounds.kx_max:.4f} "
    f"(max |kx| on grid {np.max(np.abs(kx)):.4f})",
    flush=True,
)
dt = float(cfg.time.dt)
steps = min(int(round(float(cfg.time.t_max) / dt)), 6000)
print(f"dt={dt} steps={steps} method={cfg.time.method}", flush=True)
rng = np.random.default_rng(11)
noise = rng.normal(size=ic.shape) + 1j * rng.normal(size=ic.shape)
contaminated = np.where(off, noise * np.max(np.abs(ic)) / np.max(np.abs(noise)), ic)
cache = deps.build_linear_cache(ctx.grid, ctx.geom, ctx.params, 4, 8)
term_cfg = linear_terms_to_term_config(ctx.terms)


@jax.jit
def evolve(state):
    def body(G, _):
        G_next, _fields = _linear_explicit_step(
            G, cache, ctx.params, term_cfg, dt, method=str(cfg.time.method)
        )
        return G_next, None

    return jax.lax.scan(body, state, None, length=steps)[0]


for label, state in (("runtime-ic", ic), ("ic+off-chain", contaminated)):
    t0 = time.perf_counter()
    # runtime fit (explicit_time refuses return_state, so the state comes from
    # a scan of the same explicit linear step below)
    result = run_runtime_linear(
        cfg,
        ky_target=0.3,
        Nl=4,
        Nm=8,
        solver="explicit_time",
        dt=dt,
        steps=steps,
        initial_state=np.asarray(state, dtype=ic.dtype),
    )
    fit_time = time.perf_counter() - t0
    fields = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
    final = np.asarray(evolve(jnp.asarray(state, dtype=jnp.complex128)))
    print(
        f"[{label}] fit gamma={fields.get('gamma')} omega={fields.get('omega')} "
        f"({fit_time:.1f}s); after {steps} {cfg.time.method} steps of "
        f"_linear_explicit_step: off-chain max={np.max(np.abs(final[off])):.3e} "
        f"(start {np.max(np.abs(np.asarray(state)[off])):.3e}), chain max="
        f"{np.max(np.abs(final[~off])):.3e}",
        flush=True,
    )
