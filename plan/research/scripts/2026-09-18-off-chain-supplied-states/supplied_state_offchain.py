"""Q19: what a supplied state's off-chain rows do to a linked run.

Pilot deck (linked Cyclone, Nx8/Ny16/Nz16, Nl4/Nm8, ky=.3, rate .1), the same
one Q6 used. Arm A is the runtime's own initial condition; arm B is that state
plus off-chain content of equal peak amplitude, which is what a user
``initial_state`` or a restart written elsewhere can carry.

Reports, for the state the runtime actually integrates: off-chain amplitude,
free energy, the ky and kx free-energy spectra, the certified eigenpair and the
explicit-time growth-rate fit. Run once per worktree (``before`` = origin/main,
``after`` = this branch); the numbers are compared across the two logs.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import gkx
from gkx.operators.moments import (
    distribution_free_energy,
    distribution_free_energy_resolved,
    fieldline_quadrature_weights,
)
from gkx.runtime import _runtime_linear_dispatch_deps, run_runtime_linear
from gkx.workflows.linear import _prepare_linear_runtime_context
from gkx.workflows.runtime.toml import load_runtime_from_toml

ARM = sys.argv[1] if len(sys.argv) > 1 else "after"
REPO = Path(gkx.__file__).resolve().parents[2]
print(f"arm={ARM} gkx={gkx.__file__}", flush=True)

cfg, _ = load_runtime_from_toml(REPO / "examples/linear/axisymmetric/cyclone.toml")
cfg = replace(
    cfg,
    grid=replace(cfg.grid, Nx=8, Ny=16, Nz=16, ntheta=16, nperiod=1, jtwist=1),
    time=replace(cfg.time, damp_ends_rate=0.1),
)
NL, NM = 4, 8
deps = _runtime_linear_dispatch_deps().full_deps


def context(initial_state, solver="krylov"):
    return _prepare_linear_runtime_context(
        cfg,
        deps=deps,
        ky_target=0.3,
        n_laguerre=NL,
        n_hermite=NM,
        solver=solver,
        fit_signal="auto",
        return_state=True,
        initial_state=initial_state,
        status_callback=None,
    )


base = context(None)
ic = np.asarray(base.initial_state)
grid, geom, params = base.grid, base.geom, base.params
vol_fac, _flux_fac = fieldline_quadrature_weights(geom, grid)

# The chain cover of this deck, read off the built cache rather than assumed.
cache = deps.build_linear_cache(grid, geom, params, NL, NM)
gather = np.asarray(cache.linked_gather_mask, dtype=bool)
ny, nx = int(np.asarray(grid.ky).size), int(np.asarray(grid.kx).size)
cover = np.reshape(gather, (nx, ny)).T
chain_rows = sorted(int(i) for i in np.flatnonzero(np.any(cover, axis=0)))
off_rows = sorted(set(range(nx)) - set(chain_rows))
off = np.zeros(ic.shape, dtype=bool)
off[..., off_rows, :] = True
print(
    f"grid ny={ny} nx={nx} nz={int(np.asarray(grid.z).size)}; "
    f"dealias_mask all-true={bool(np.all(np.asarray(grid.dealias_mask)))}; "
    f"chain kx rows={chain_rows} off-chain kx rows={off_rows}",
    flush=True,
)
print(
    f"state {ic.shape} n={ic.size} off-chain unknowns={int(off.sum())} "
    f"({off.mean():.4f} of the state)",
    flush=True,
)
print(
    f"runtime initial condition: max|G|={np.max(np.abs(ic)):.6e} "
    f"off-chain max={np.max(np.abs(ic[off])):.6e}",
    flush=True,
)

rng = np.random.default_rng(1918)
noise = rng.normal(size=ic.shape) + 1j * rng.normal(size=ic.shape)
noise = noise.astype(np.complex64) * np.complex64(
    np.max(np.abs(ic)) / np.max(np.abs(noise))
)
supplied = np.where(off, noise, ic).astype(np.complex64)
print(
    f"supplied state: off-chain max={np.max(np.abs(supplied[off])):.6e} "
    f"chain part bitwise equal to the runtime IC="
    f"{bool(np.array_equal(supplied[~off], ic[~off]))}",
    flush=True,
)


def report(label, state):
    state = np.asarray(state)
    wg = float(distribution_free_energy(state, grid, params, vol_fac))
    _st, kxst, kyst, _kxkyst, _zst, _lmst = distribution_free_energy_resolved(
        state, grid, params, vol_fac
    )
    kx_spec = np.asarray(kxst).sum(axis=0)
    ky_spec = np.asarray(kyst).sum(axis=0)
    off_share = float(kx_spec[off_rows].sum() / wg) if wg > 0 else 0.0
    print(f"--- {label}", flush=True)
    print(
        f"    off-chain max|G|={np.max(np.abs(state[off])):.6e} "
        f"Wg={wg:.12e} off-chain share of Wg={off_share:.6f}",
        flush=True,
    )
    print(f"    Wg(kx) = {np.array2string(kx_spec, precision=6)}", flush=True)
    print(f"    Wg(ky) = {np.array2string(ky_spec, precision=6)}", flush=True)
    return wg, kx_spec, ky_spec


report("runtime initial condition (arm A)", ic)
ctx = context(supplied)
report(
    "state the runtime integrates from the supplied array (arm B)", ctx.initial_state
)
print(
    "    chain part of the integrated state bitwise equal to the supplied one="
    f"{bool(np.array_equal(np.asarray(ctx.initial_state)[~off], supplied[~off]))}",
    flush=True,
)

for solver, tag in (("krylov", "certified eigenpair"), ("explicit_time", "time fit")):
    res = run_runtime_linear(
        cfg,
        ky_target=0.3,
        Nl=NL,
        Nm=NM,
        solver=solver,
        initial_state=supplied,
    )
    status = getattr(res, "eigen_status", None)
    extra = ""
    if status is not None:
        extra = (
            f" residual={float(status.residual):.6e} "
            f"certified={bool(status.certified)} route={status.route}"
        )
    print(
        f"{tag} ({solver}) from the supplied state: "
        f"gamma={float(res.gamma):.18e} omega={float(res.omega):.18e}{extra}",
        flush=True,
    )
