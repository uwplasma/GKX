"""Q19: why intake is enough, and what the restart round trip carries.

Three measurements on the linked Cyclone pilot (Nx8/Ny16/Nz16, Nl4/Nm8, ky=.3,
rate .1) and a full linked nonlinear grid:

1. Linear time integration. From a masked state the off-chain rows stay exactly
   zero; from an unmasked one they persist as a neutral band while the chain
   rows stay bitwise identical. That is the evidence for masking at intake only.
2. Nonlinear RHS. On a full grid the off-chain kx rows are exactly the rows the
   two-thirds mask removes, so the bracket writes exactly zero there.
3. Restart round trip: write a state with off-chain content, read it back
   through the runtime's own init-file path, and check the contract.

Run once per worktree; ``before`` = origin/main, ``after`` = this branch.
"""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gkx
from gkx.artifacts.io import write_netcdf_restart_state
from gkx.core_grid import build_spectral_grid
from gkx.operators.linear.params import linear_terms_to_term_config
from gkx.runtime import (
    _build_initial_condition,
    _runtime_linear_dispatch_deps,
    apply_geometry_grid_defaults,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from gkx.solvers_nonlinear_state_integration import nonlinear_rhs_cached
from gkx.solvers_time_explicit_steps import _linear_explicit_step
from gkx.workflows.linear import _prepare_linear_runtime_context
from gkx.workflows.runtime.toml import load_runtime_from_toml


def cover_from_gather(gather: np.ndarray, *, ny: int, nx: int) -> np.ndarray:
    """Chain cover from a cache's flat gather mask, mirrors included.

    Spelled out here rather than imported so the same script runs on both arms.
    """

    mask = np.reshape(np.asarray(gather, dtype=bool), (nx, ny)).T
    if ny > 1:
        rows = np.any(mask, axis=1)
        mirror_rows = np.mod(-np.arange(ny), ny)
        fill = ~rows & rows[mirror_rows] & (np.arange(ny) != 0)
        mirrored = mask[mirror_rows][:, np.mod(-np.arange(nx), nx)]
        mask = mask | (fill[:, None] & mirrored)
    return mask


ARM = sys.argv[1] if len(sys.argv) > 1 else "after"
REPO = Path(gkx.__file__).resolve().parents[2]
print(f"arm={ARM} gkx={gkx.__file__}", flush=True)

cfg_base, _ = load_runtime_from_toml(REPO / "examples/linear/axisymmetric/cyclone.toml")
cfg = replace(
    cfg_base,
    grid=replace(cfg_base.grid, Nx=8, Ny=16, Nz=16, ntheta=16, nperiod=1, jtwist=1),
    time=replace(cfg_base.time, damp_ends_rate=0.1),
)
NL, NM = 4, 8
OFF_ROWS = [3, 4, 5]
deps = _runtime_linear_dispatch_deps().full_deps

ctx = _prepare_linear_runtime_context(
    cfg,
    deps=deps,
    ky_target=0.3,
    n_laguerre=NL,
    n_hermite=NM,
    solver="explicit_time",
    fit_signal="auto",
    return_state=True,
    initial_state=None,
    status_callback=None,
)
ic = np.asarray(ctx.initial_state)
off = np.zeros(ic.shape, dtype=bool)
off[..., OFF_ROWS, :] = True
rng = np.random.default_rng(1918)
noise = rng.normal(size=ic.shape) + 1j * rng.normal(size=ic.shape)
noise = noise.astype(np.complex64) * np.complex64(
    np.max(np.abs(ic)) / np.max(np.abs(noise))
)
supplied = np.where(off, noise, ic).astype(np.complex64)
masked = np.where(off, 0.0, supplied).astype(np.complex64)

# ---- 1. linear time integration -------------------------------------------
cache = deps.build_linear_cache(ctx.grid, ctx.geom, ctx.params, NL, NM)
term_cfg = linear_terms_to_term_config(ctx.terms)
dt = float(cfg.time.dt)
steps = 2000


@jax.jit
def evolve(state):
    def body(G, _):
        G_next, _fields = _linear_explicit_step(
            G, cache, ctx.params, term_cfg, dt, method=str(cfg.time.method)
        )
        return G_next, None

    return jax.lax.scan(body, state, None, length=steps)[0]


print(f"linear time integration: dt={dt} steps={steps} method={cfg.time.method}")
finals = {}
for label, state in (("masked", masked), ("unmasked", supplied)):
    final = np.asarray(evolve(jnp.asarray(state)))
    finals[label] = final
    print(
        f"  {label}: off-chain max start={np.max(np.abs(state[off])):.6e} "
        f"end={np.max(np.abs(final[off])):.6e}; "
        f"chain max end={np.max(np.abs(final[~off])):.6e}",
        flush=True,
    )
print(
    "  chain rows bitwise identical between the two arms="
    f"{bool(np.array_equal(finals['masked'][~off], finals['unmasked'][~off]))}",
    flush=True,
)
print(
    "  unmasked off-chain amplitude end/start="
    f"{np.max(np.abs(finals['unmasked'][off])) / np.max(np.abs(supplied[off])):.6f}"
    " (neutral band: no decay, no secular growth)",
    flush=True,
)

# ---- 2. nonlinear RHS on a full linked grid --------------------------------
cfg_nl = replace(
    cfg_base,
    grid=replace(cfg_base.grid, Nx=8, Ny=8, Nz=16, ntheta=16, nperiod=1, jtwist=1),
    time=replace(cfg_base.time, damp_ends_rate=0.1),
    physics=replace(cfg_base.physics, linear=False, nonlinear=True),
    terms=replace(cfg_base.terms, nonlinear=1.0),
)
geom_nl = build_runtime_geometry(cfg_nl)
grid_nl = build_spectral_grid(apply_geometry_grid_defaults(geom_nl, cfg_nl.grid))
params_nl = build_runtime_linear_params(cfg_nl, Nm=NM, geom=geom_nl)
terms_nl = build_runtime_term_config(cfg_nl)
cache_nl = deps.build_linear_cache(grid_nl, geom_nl, params_nl, NL, NM)
gather_nl = np.asarray(cache_nl.linked_gather_mask, dtype=bool)
ny_nl, nx_nl = int(np.asarray(grid_nl.ky).size), int(np.asarray(grid_nl.kx).size)
cover_nl = cover_from_gather(gather_nl, ny=ny_nl, nx=nx_nl)
dealias_nl = np.asarray(grid_nl.dealias_mask, dtype=bool)
rows_nl = np.any(cover_nl, axis=0)
print(
    f"nonlinear grid ny={ny_nl} nx={nx_nl}: chain kx rows="
    f"{sorted(int(i) for i in np.flatnonzero(rows_nl))}, "
    f"two-thirds kx rows="
    f"{sorted(int(i) for i in np.flatnonzero(np.any(dealias_nl, axis=0)))}",
    flush=True,
)
shape_nl = (1, NL, NM, ny_nl, nx_nl, int(np.asarray(grid_nl.z).size))
off_nl = np.zeros(shape_nl, dtype=bool)
off_nl[..., ~cover_nl, :] = True
rng2 = np.random.default_rng(77)
broad = (rng2.normal(size=shape_nl) + 1j * rng2.normal(size=shape_nl)).astype(
    np.complex64
)
on_chain_only = np.where(off_nl, 0.0, broad).astype(np.complex64)
dG, _fields = nonlinear_rhs_cached(
    jnp.asarray(on_chain_only), cache_nl, params_nl, terms_nl
)
dG = np.asarray(dG)
print(
    f"  nonlinear RHS of a chain-only state: off-chain |dG| max="
    f"{np.max(np.abs(dG[off_nl])):.6e} "
    f"(chain |dG| max {np.max(np.abs(dG[~off_nl])):.6e})",
    flush=True,
)
dG_broad = np.asarray(
    nonlinear_rhs_cached(jnp.asarray(broad), cache_nl, params_nl, terms_nl)[0]
)
chain_delta = np.max(np.abs(dG_broad[~off_nl] - dG[~off_nl])) / np.max(
    np.abs(dG[~off_nl])
)
print(
    "  chain rows of the RHS are bitwise the same with and without off-chain "
    f"content={bool(np.array_equal(dG_broad[~off_nl], dG[~off_nl]))}; "
    f"relative change {chain_delta:.6e}",
    flush=True,
)

# ---- 3. restart round trip (full nonlinear grid) ---------------------------
ic_nl = np.asarray(
    _build_initial_condition(
        grid_nl, geom_nl, cfg_nl, ky_index=1, kx_index=0, Nl=NL, Nm=NM, nspecies=1
    )
)
supplied_nl = np.where(off_nl, broad * np.complex64(np.max(np.abs(ic_nl))), ic_nl)
supplied_nl = supplied_nl.astype(np.complex64)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "pilot.restart.bin"
    write_netcdf_restart_state(path, supplied_nl)
    cfg_restart = replace(
        cfg_nl,
        init=replace(cfg_nl.init, init_file=str(path), init_file_mode="replace"),
    )
    loaded = np.asarray(
        _build_initial_condition(
            grid_nl,
            geom_nl,
            cfg_restart,
            ky_index=1,
            kx_index=0,
            Nl=NL,
            Nm=NM,
            nspecies=1,
        )
    )
print(
    f"restart round trip: wrote {supplied_nl.size} complex64 with off-chain max "
    f"{np.max(np.abs(supplied_nl[off_nl])):.6e}; read back off-chain max="
    f"{np.max(np.abs(loaded[off_nl])):.6e}; chain part bitwise equal to what was "
    f"written={bool(np.array_equal(loaded[~off_nl], supplied_nl[~off_nl]))}",
    flush=True,
)
