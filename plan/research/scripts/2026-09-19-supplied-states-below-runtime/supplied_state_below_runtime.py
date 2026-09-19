"""Q23: what the library entry points below the runtime do with a supplied state.

#247 (Q19) masks the off-chain rows of a user ``initial_state`` and of a restart
file at *runtime* intake. The entry points underneath -- ``gkx.prepare`` and the
prepared object's ``run(initial_state)``, the shared
``integrate_nonlinear_explicit_diagnostics_state``, and the differentiable
objective ``nonlinear_heat_flux_window`` -- took the state they were given.

This script measures, on the linked nonlinear Cyclone grid Q19 used
(Nx8/Ny8/Nz16, Nl4/Nm8, jtwist 1, rate .1), what that cost and what the contract
changes:

1. the supplied state's off-chain content, free energy and kx spectrum, and what
   each entry point actually integrates;
2. one nonlinear right-hand side, chain rows, with and without off-chain content;
3. the prepared object's trajectory and heat flux from a supplied state;
4. the reverse-mode cotangent of a supplied state through ``run_arrays``;
5. the objective adjoint's value and its gradient with respect to ``params``;
6. the no-op arms: an on-cover state, and a periodic deck, including the
   optimized-HLO hash of the compiled scan.

Run once per worktree; ``before`` = a detached checkout of ``origin/main``,
``after`` = this branch. The two logs are compared line for line.
"""

import hashlib
import re
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import gkx
from gkx.config import GridConfig
from gkx.core_grid import build_spectral_grid
from gkx.geometry import ensure_flux_tube_geometry_data
from gkx.operators.moments import (
    distribution_free_energy,
    distribution_free_energy_resolved,
    fieldline_quadrature_weights,
)
from gkx.runtime import (
    _build_initial_condition,
    _runtime_linear_dispatch_deps,
    apply_geometry_grid_defaults,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from gkx.solvers_nonlinear_diagnostic_integration import (
    integrate_nonlinear_explicit_diagnostics_state,
    prepare_nonlinear_explicit_diagnostics,
)
from gkx.solvers_nonlinear_state_integration import (
    nonlinear_heat_flux_window,
    nonlinear_rhs_cached,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml

ARM = sys.argv[1] if len(sys.argv) > 1 else "after"
REPO = Path(gkx.__file__).resolve().parents[2]
print(f"arm={ARM} gkx={gkx.__file__}", flush=True)

NL, NM = 4, 8
DT, STEPS = 1.0e-3, 8
deps = _runtime_linear_dispatch_deps().full_deps
base_cfg, _ = load_runtime_from_toml(REPO / "examples/linear/axisymmetric/cyclone.toml")


def cover_from_gather(gather, *, ny: int, nx: int) -> np.ndarray:
    """Chain cover from a cache's flat gather mask, conjugate mirrors included.

    Spelled out here rather than imported, so the same file runs on both arms.
    """

    mask = np.reshape(np.asarray(gather, dtype=bool), (nx, ny)).T
    if ny > 1:
        rows = np.any(mask, axis=1)
        mirror_rows = np.mod(-np.arange(ny), ny)
        fill = ~rows & rows[mirror_rows] & (np.arange(ny) != 0)
        mirrored = mask[mirror_rows][:, np.mod(-np.arange(nx), nx)]
        mask = mask | (fill[:, None] & mirrored)
    return mask


def build_deck(boundary: str, *, nx: int = 8, ny: int = 8, nz: int = 16):
    """Return the nonlinear deck, its grid/geometry/params/cache and its cover."""

    cfg = replace(
        base_cfg,
        grid=GridConfig(
            Nx=nx,
            Ny=ny,
            Nz=nz,
            ntheta=16,
            nperiod=1,
            boundary=boundary,
            jtwist=1 if boundary == "linked" else None,
        ),
        time=replace(base_cfg.time, damp_ends_rate=0.1),
        physics=replace(base_cfg.physics, linear=False, nonlinear=True),
        terms=replace(base_cfg.terms, nonlinear=1.0),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg.grid))
    params = build_runtime_linear_params(cfg, Nm=NM, geom=geom)
    terms = build_runtime_term_config(cfg)
    cache = deps.build_linear_cache(grid, geom, params, NL, NM)
    ny_g = int(np.asarray(grid.ky).size)
    nx_g = int(np.asarray(grid.kx).size)
    if bool(getattr(cache, "linked_use_gather", False)) and not bool(
        getattr(cache, "linked_full_cover", False)
    ):
        cover = cover_from_gather(cache.linked_gather_mask, ny=ny_g, nx=nx_g)
    else:
        cover = np.ones((ny_g, nx_g), dtype=bool)
    return cfg, grid, geom, params, terms, cache, cover


def hlo_fingerprint(lowered_text: str) -> str:
    """Instruction count and hash of optimized HLO, call-stack metadata stripped.

    A module's text also carries the Python call stack that produced it: a
    ``metadata={...}`` field per instruction and, at the top, string and
    ``FileLocations`` tables of file names, function names and *line numbers*.
    Those move whenever a source file is edited, which would make every arm
    differ for reasons that are not the graph -- the same distinction Q18 had to
    make. Keeping only the instruction lines (the ones with an ``=``) and
    dropping their metadata leaves the graph itself, and reproduces bit for bit
    across processes and across repeated compiles in one process.
    """

    kept = []
    for line in lowered_text.splitlines():
        if " = " not in line:
            continue
        line = re.sub(r"metadata=\{[^}]*\}", "", line)
        line = re.sub(r'source_file="[^"]*"|source_line=\d+', "", line)
        kept.append(line.rstrip())
    body = "\n".join(kept)
    return f"{len(kept)} instructions, sha256 {hashlib.sha256(body.encode()).hexdigest()[:16]}"


def prepared_for(cfg, grid, geom, params, terms, state):
    return prepare_nonlinear_explicit_diagnostics(
        jnp.asarray(state),
        grid,
        geom,
        params,
        DT,
        STEPS,
        method="rk3",
        terms=terms,
        fixed_dt=True,
        resolved_diagnostics=True,
    )


# ---- 0. the deck, its cover and the supplied state -------------------------

cfg, grid, geom, params, terms, cache, cover = build_deck("linked")
ny_g = int(np.asarray(grid.ky).size)
nx_g = int(np.asarray(grid.kx).size)
nz_g = int(np.asarray(grid.z).size)
shape = (1, NL, NM, ny_g, nx_g, nz_g)
off = np.broadcast_to(~cover[:, :, None], shape)
vol_fac, _flux_fac = fieldline_quadrature_weights(geom, grid)

ic = np.asarray(
    _build_initial_condition(
        grid, geom, cfg, ky_index=1, kx_index=0, Nl=NL, Nm=NM, nspecies=1
    )
)
# The state these entry points are documented to take is a saturated nonlinear
# one -- ``nonlinear_heat_flux_window`` says so in its signature -- so the
# supplied array is broadband at O(1), not the runtime's 1e-10 linear seed. At
# the seed amplitude the bracket is quadratically small and nothing downstream
# of it would be visible; that is a statement about the seed, not about the
# rule, and the amplitude scan at the end of this script shows both.
rng = np.random.default_rng(1923)
supplied = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
supplied = (supplied / np.max(np.abs(supplied))).astype(np.complex64)
on_cover = np.where(off, 0.0, supplied).astype(np.complex64)

print(
    f"deck: ny={ny_g} nx={nx_g} nz={nz_g}; cover {int(cover.sum())}/{cover.size} "
    f"(ky, kx) modes; chain kx rows="
    f"{sorted(int(i) for i in np.flatnonzero(np.any(cover, axis=0)))}; "
    f"off-chain kx rows="
    f"{sorted(int(i) for i in np.flatnonzero(~np.any(cover, axis=0)))}",
    flush=True,
)
print(
    f"state {shape} n={int(np.prod(shape))} off-chain unknowns={int(off.sum())} "
    f"({off.mean():.4f} of the state)",
    flush=True,
)
print(
    f"runtime initial condition: max|G|={np.max(np.abs(ic)):.6e} "
    f"off-chain max={np.max(np.abs(ic[off])):.6e}",
    flush=True,
)
print(
    f"supplied state: max|G|={np.max(np.abs(supplied)):.6e} "
    f"off-chain max={np.max(np.abs(supplied[off])):.6e}; off-chain share of "
    f"||G||^2={float(np.sum(np.abs(supplied[off]) ** 2) / np.sum(np.abs(supplied) ** 2)):.6f}",
    flush=True,
)


def report_state(label, state):
    state = np.asarray(state)
    wg = float(distribution_free_energy(state, grid, params, vol_fac))
    _st, kxst, _kyst, _kxkyst, _zst, _lmst = distribution_free_energy_resolved(
        state, grid, params, vol_fac
    )
    kx_spec = np.asarray(kxst).sum(axis=0)
    off_rows = sorted(int(i) for i in np.flatnonzero(~np.any(cover, axis=0)))
    share = float(kx_spec[off_rows].sum() / wg) if wg > 0 else 0.0
    print(f"--- {label}", flush=True)
    print(
        f"    off-chain max|G|={np.max(np.abs(state[off])):.6e} "
        f"Wg={wg:.12e} off-chain share of Wg={share:.6f}",
        flush=True,
    )
    print(f"    Wg(kx) = {np.array2string(kx_spec, precision=6)}", flush=True)
    return wg


wg_supplied = report_state("the supplied array itself", supplied)
wg_on_cover = report_state("the same state projected onto the chain cover", on_cover)
print(
    f"    inflation Wg(supplied)/Wg(on cover) = {wg_supplied / wg_on_cover:.4f}x",
    flush=True,
)

obj_supplied = prepared_for(cfg, grid, geom, params, terms, supplied)
obj_on_cover = prepared_for(cfg, grid, geom, params, terms, on_cover)
report_state(
    "what gkx.prepare's object integrates (prepared.initial_state)",
    obj_supplied.initial_state,
)
print(
    "    prepared.initial_state is bitwise equal to the one prepared from the "
    "already-on-cover array="
    f"{bool(np.array_equal(np.asarray(obj_supplied.initial_state), np.asarray(obj_on_cover.initial_state)))}"
    "; the setup's own Hermitian projection is why neither equals the raw input",
    flush=True,
)

# ---- 1. one nonlinear right-hand side --------------------------------------

dG_on_cover = np.asarray(
    nonlinear_rhs_cached(jnp.asarray(on_cover), cache, params, terms)[0]
)
dG_supplied = np.asarray(
    nonlinear_rhs_cached(jnp.asarray(supplied), cache, params, terms)[0]
)
chain_scale = np.max(np.abs(dG_on_cover[~off]))
rhs_rel = float(np.max(np.abs(dG_supplied[~off] - dG_on_cover[~off])) / chain_scale)
print(
    "one nonlinear RHS: chain rows change "
    f"{rhs_rel:.6e} relative when the off-chain rows are present "
    f"(chain |dG| max {chain_scale:.6e}); off-chain |dG| from an on-cover state="
    f"{np.max(np.abs(dG_on_cover[off])):.6e}",
    flush=True,
)
for amp in (1.0, 1.0e-2, 1.0e-4, 1.414214e-10):
    a = np.complex64(amp)
    d0 = np.asarray(
        nonlinear_rhs_cached(jnp.asarray(on_cover * a), cache, params, terms)[0]
    )
    d1 = np.asarray(
        nonlinear_rhs_cached(jnp.asarray(supplied * a), cache, params, terms)[0]
    )
    print(
        f"    amplitude max|G|={amp:.6e}: chain-row relative change="
        f"{float(np.max(np.abs(d1[~off] - d0[~off])) / np.max(np.abs(d0[~off]))):.6e}",
        flush=True,
    )

# ---- 2. the prepared object's trajectory from a supplied state -------------


def run_final(obj, state):
    t, diag, G_final, _fields = obj.run(None if state is None else jnp.asarray(state))
    return np.asarray(G_final), np.asarray(diag.heat_flux_t)


G_sup, q_sup = run_final(obj_supplied, supplied)
G_cov, q_cov = run_final(obj_on_cover, on_cover)
scale = np.max(np.abs(G_cov))
print(
    f"prepared run({STEPS} rk3 steps, dt={DT}) from the supplied state vs the "
    "on-cover state: final-state max relative difference="
    f"{float(np.max(np.abs(G_sup - G_cov)) / scale):.6e}; "
    f"final-state off-chain max={np.max(np.abs(G_sup[off])):.6e}; "
    f"heat-flux max relative difference="
    f"{float(np.max(np.abs(q_sup - q_cov)) / np.max(np.abs(q_cov))):.6e}",
    flush=True,
)

# The prepared object and the function entry point are one route (Q18); both
# must take the same view of a supplied state.
t_fn, diag_fn, G_fn, _f_fn = integrate_nonlinear_explicit_diagnostics_state(
    jnp.asarray(supplied),
    grid,
    geom,
    params,
    DT,
    STEPS,
    method="rk3",
    terms=terms,
    fixed_dt=True,
    resolved_diagnostics=True,
)
print(
    "integrate_nonlinear_explicit_diagnostics_state from the same supplied "
    "state is bitwise equal to the prepared route="
    f"{bool(np.array_equal(np.asarray(G_fn), G_sup))}",
    flush=True,
)

# ---- 3. the reverse-mode cotangent of a supplied state ---------------------


def final_state_energy(state):
    G_final, _scan_out, _fields = obj_on_cover.run_arrays(state)
    return jnp.sum(jnp.abs(G_final) ** 2)


grad_sup = np.asarray(jax.grad(final_state_energy)(jnp.asarray(supplied)))
grad_cov = np.asarray(jax.grad(final_state_energy)(jnp.asarray(on_cover)))
print(
    "VJP through run_arrays with respect to the supplied state "
    "(scalar = sum |G_final|^2): "
    f"off-chain cotangent max={np.max(np.abs(grad_sup[off])):.6e}; "
    f"chain cotangent max={np.max(np.abs(grad_sup[~off])):.6e}; "
    f"chain cotangent bitwise equal to the on-cover arm="
    f"{bool(np.array_equal(grad_sup[~off], grad_cov[~off]))}",
    flush=True,
)

# ---- 4. the objective adjoint ----------------------------------------------


geom_data = ensure_flux_tube_geometry_data(geom, grid.z)


def window(state, drift_scale):
    """The documented use: a detached saturated state, a traced geometry.

    ``drift_scale`` multiplies the curvature and grad-B drift profiles, which is
    a geometry knob a design loop would actually turn.
    """

    scaled = replace(
        geom_data,
        gb_profile=geom_data.gb_profile * drift_scale,
        cv_profile=geom_data.cv_profile * drift_scale,
    )
    return nonlinear_heat_flux_window(
        jnp.asarray(state), grid, scaled, params, DT, STEPS, terms=terms, method="rk2"
    )


one = jnp.asarray(1.0)
q_win_sup, g_win_sup = jax.value_and_grad(lambda a: window(supplied, a))(one)
q_win_cov, g_win_cov = jax.value_and_grad(lambda a: window(on_cover, a))(one)
q_win_sup, g_win_sup = float(q_win_sup), float(g_win_sup)
q_win_cov, g_win_cov = float(q_win_cov), float(g_win_cov)
print(
    f"nonlinear_heat_flux_window: supplied={q_win_sup:.12e} "
    f"on cover={q_win_cov:.12e} relative difference="
    f"{abs(q_win_sup - q_win_cov) / abs(q_win_cov):.6e}",
    flush=True,
)
print(
    "    adjoint gradient with respect to the drift scale: "
    f"supplied={g_win_sup:.12e} on cover={g_win_cov:.12e} relative difference="
    f"{abs(g_win_sup - g_win_cov) / abs(g_win_cov):.6e}",
    flush=True,
)
grad_state = np.asarray(jax.grad(lambda s: window(s, one))(jnp.asarray(supplied)))
print(
    "    gradient with respect to the supplied state (documented as detached): "
    f"max|dQ/dG|={np.max(np.abs(grad_state)):.6e}",
    flush=True,
)

# ---- 5. the no-op arms -----------------------------------------------------

G_default, q_default = run_final(prepared_for(cfg, grid, geom, params, terms, ic), None)
print(
    "no-op arm, linked deck from the runtime's own on-cover initial condition: "
    f"final-state checksum={hashlib.sha256(np.ascontiguousarray(G_default).tobytes()).hexdigest()[:16]} "
    f"heat-flux checksum={hashlib.sha256(np.ascontiguousarray(q_default).tobytes()).hexdigest()[:16]}",
    flush=True,
)
print(
    "    the compiled scan of that prepared object: optimized HLO = "
    f"{hlo_fingerprint(obj_on_cover._run_raw.lower(jnp.asarray(on_cover)).compile().as_text())}",
    flush=True,
)

pcfg, pgrid, pgeom, pparams, pterms, pcache, pcover = build_deck("periodic")
pshape = (
    1,
    NL,
    NM,
    int(np.asarray(pgrid.ky).size),
    int(np.asarray(pgrid.kx).size),
    int(np.asarray(pgrid.z).size),
)
prng = np.random.default_rng(2319)
pstate = (prng.normal(size=pshape) + 1j * prng.normal(size=pshape)).astype(np.complex64)
pobj = prepared_for(pcfg, pgrid, pgeom, pparams, pterms, pstate)
print(
    "no-op arm, periodic deck: cover is every mode="
    f"{bool(pcover.all())}; prepared.initial_state checksum="
    f"{hashlib.sha256(np.ascontiguousarray(np.asarray(pobj.initial_state)).tobytes()).hexdigest()[:16]}",
    flush=True,
)
pG, pdiag, pGf, _pf = pobj.run(jnp.asarray(pstate))
print(
    "    run(initial_state) final-state checksum="
    f"{hashlib.sha256(np.ascontiguousarray(np.asarray(pGf)).tobytes()).hexdigest()[:16]}; "
    "optimized HLO = "
    f"{hlo_fingerprint(pobj._run_raw.lower(jnp.asarray(pstate)).compile().as_text())}",
    flush=True,
)
p_window = float(
    nonlinear_heat_flux_window(
        jnp.asarray(pstate),
        pgrid,
        pgeom,
        pparams,
        DT,
        STEPS,
        terms=pterms,
        method="rk2",
    )
)
print(f"    periodic objective adjoint value={p_window:.12e}", flush=True)

# ---- 6. the linear runtime must not move -----------------------------------
#
# #247 recorded the pilot's certified eigenpair and explicit-time fit from a
# supplied state. Nothing here touches that route, and these lines are what
# says so rather than assumes it.

from gkx.runtime import run_runtime_linear  # noqa: E402
from gkx.workflows.linear import _prepare_linear_runtime_context  # noqa: E402

pilot = replace(
    base_cfg,
    grid=replace(base_cfg.grid, Nx=8, Ny=16, Nz=16, ntheta=16, nperiod=1, jtwist=1),
    time=replace(base_cfg.time, damp_ends_rate=0.1),
)
pilot_ctx = _prepare_linear_runtime_context(
    pilot,
    deps=deps,
    ky_target=0.3,
    n_laguerre=NL,
    n_hermite=NM,
    solver="krylov",
    fit_signal="auto",
    return_state=True,
    initial_state=None,
    status_callback=None,
)
pilot_ic = np.asarray(pilot_ctx.initial_state)
pilot_off = np.zeros(pilot_ic.shape, dtype=bool)
pilot_off[..., [3, 4, 5], :] = True
prng2 = np.random.default_rng(1918)
pilot_noise = (
    prng2.normal(size=pilot_ic.shape) + 1j * prng2.normal(size=pilot_ic.shape)
).astype(np.complex64)
pilot_noise = pilot_noise * np.complex64(
    np.max(np.abs(pilot_ic)) / np.max(np.abs(pilot_noise))
)
pilot_supplied = np.where(pilot_off, pilot_noise, pilot_ic).astype(np.complex64)
for solver, tag in (("krylov", "certified eigenpair"), ("explicit_time", "time fit")):
    res = run_runtime_linear(
        pilot,
        ky_target=0.3,
        Nl=NL,
        Nm=NM,
        solver=solver,
        initial_state=pilot_supplied,
    )
    status = getattr(res, "eigen_status", None)
    extra = (
        ""
        if status is None
        else (
            f" residual={float(status.residual):.6e} "
            f"certified={bool(status.certified)} route={status.route}"
        )
    )
    print(
        f"linear pilot {tag} ({solver}) from a supplied state: "
        f"gamma={float(res.gamma):.18e} omega={float(res.omega):.18e}{extra}",
        flush=True,
    )
