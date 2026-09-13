# ruff: noqa: E402
"""D7: load-independent HLO op counts for the nonlinear RHS and one RK3 step,
plus Hermitian-projector idempotence on the RHS output (read-only review)."""

import re
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path("/Users/rogeriojorge/local/GKX-worktrees/perf-review")
sys.path.insert(0, str(REPO))
from gkx.core_grid import build_spectral_grid  # noqa: E402
from gkx.geometry import apply_imported_geometry_grid_defaults  # noqa: E402
from gkx.operators.linear.cache_builder import build_linear_cache  # noqa: E402
from gkx.runtime import (  # noqa: E402
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from gkx.solvers_nonlinear_state_integration import (
    integrate_nonlinear,
    nonlinear_rhs_cached,
)  # noqa: E402
from gkx.operators.nonlinear.projection import _make_nonlinear_state_projector  # noqa: E402
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402

Nx, Ny, Nz, Nl, Nm = (int(a) for a in (sys.argv[1:6] or (32, 32, 16, 2, 4)))
cfg, _ = load_runtime_from_toml(
    REPO / "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml"
)
cfg = replace(cfg, grid=replace(cfg.grid, Nx=Nx, Ny=Ny, Nz=Nz))
geom = build_runtime_geometry(cfg)
grid = build_spectral_grid(apply_imported_geometry_grid_defaults(geom, cfg.grid))
params = build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
term_cfg = build_runtime_term_config(cfg)
kyi, kxi = _select_nonlinear_mode_indices(
    grid,
    ky_target=0.3,
    kx_target=None,
    use_dealias_mask=bool(cfg.time.nonlinear_dealias),
)
g0 = jnp.asarray(
    _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=kyi,
        kx_index=kxi,
        Nl=Nl,
        Nm=Nm,
        nspecies=len(cfg.species),
    )
)
cache = build_linear_cache(grid, geom, params, Nl, Nm)
print(
    "state",
    g0.shape,
    g0.dtype,
    "ky axis",
    np.asarray(grid.ky).round(3).tolist()[:6],
    "...",
    "compressed_real_fft",
    cfg.time.compressed_real_fft,
    "laguerre_mode",
    cfg.time.laguerre_nonlinear_mode,
    "method",
    cfg.time.method,
    flush=True,
)

TOKENS = (
    "fft",
    "concatenate",
    "gather",
    "scatter",
    "transpose",
    "copy",
    "dynamic-update-slice",
    "reverse",
    "reduce",
    "dot",
    "fusion",
    "custom-call",
    "while",
    "select",
)


def count(text):
    out = {}
    for tok in TOKENS:
        out[tok] = len(
            re.findall(
                r"^\s*%?[\w.]+ = [^\n]*?\b" + re.escape(tok) + r"\b", text, flags=re.M
            )
        )
    return out


rhs = jax.jit(
    lambda G: nonlinear_rhs_cached(
        G,
        cache,
        params,
        term_cfg,
        compressed_real_fft=bool(cfg.time.compressed_real_fft),
        laguerre_mode=cfg.time.laguerre_nonlinear_mode,
    )[0]
)
step = jax.jit(
    lambda G: integrate_nonlinear(
        G,
        grid,
        geom,
        params,
        dt=float(cfg.time.dt),
        steps=1,
        method=cfg.time.method,
        cache=cache,
        terms=term_cfg,
        compressed_real_fft=bool(cfg.time.compressed_real_fft),
        laguerre_mode=cfg.time.laguerre_nonlinear_mode,
        return_fields=False,
    )
)
for name, fn in (("rhs", rhs), ("rk_step(return_fields=False)", step)):
    txt = fn.lower(g0).compile().as_text()
    c = count(txt)
    print(f"{name}: " + " ".join(f"{k}={v}" for k, v in c.items()), flush=True)
    # bytes moved by concatenate/copy fusions: sum of output shapes of copy/concatenate ops (approx)
    tot = 0
    for m in re.finditer(
        r"= (c64|c128|f32|f64)\[([\d,]+)\][^\n]*\b(concatenate|copy)\b", txt
    ):
        dims = [int(d) for d in m.group(2).split(",") if d]
        tot += (
            int(np.prod(dims)) * {"c64": 8, "c128": 16, "f32": 4, "f64": 8}[m.group(1)]
        )
    print(
        f"   approx bytes written by concatenate/copy ops: {tot / 1e6:.1f} MB (state {g0.size * g0.dtype.itemsize / 1e6:.1f} MB)",
        flush=True,
    )

# projector idempotence: is the RHS of a Hermitian-complete state already complete?
project = _make_nonlinear_state_projector(
    None,
    ky_vals=np.asarray(grid.ky),
    nx=int(Nx),
    compressed_real_fft=bool(cfg.time.compressed_real_fft),
    fixed_mode_ky_index=None,
    fixed_mode_kx_index=None,
)
key = jax.random.PRNGKey(0)
G = project(
    jax.random.normal(key, g0.shape, dtype=jnp.float32).astype(g0.dtype)
    + 1j
    * jax.random.normal(jax.random.PRNGKey(1), g0.shape, dtype=jnp.float32).astype(
        g0.dtype
    )
)
dG = rhs(G)
err = float(jnp.linalg.norm(project(dG) - dG) / jnp.linalg.norm(dG))
G1 = G + 0.01 * dG
err2 = float(jnp.linalg.norm(project(G1) - G1) / jnp.linalg.norm(G1))
print(
    f"projector idempotence: ||P(rhs(PG)) - rhs(PG)||/||rhs|| = {err:.2e}; on G+dt*dG: {err2:.2e} (dtype {dG.dtype})",
    flush=True,
)
