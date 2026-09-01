#!/usr/bin/env python
"""Vacuum QA optimization with physical nonlinear GKX heat flux."""

from dataclasses import replace
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import gkx
from gkx.workflows.runtime.warm_start import (
    SaturationRefreshPolicy,
    SaturationWarmStart,
    flux_tube_signature,
)
import vmex as vj
from vmex import optimize as opt
from vmex.core import turbulence

nfp = 2
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES, MAX_NFEV = [1, 2, 3, 4, 5], [10, 20, 40, 80, 120]
ASPECT_TARGET, IOTA_TARGET = 6.0, 0.42
MINIMUM_MPOL = 5
SEED_PERTURBATION = 0.01
QA_PRIORITY, ASPECT_PRIORITY, IOTA_PRIORITY = 1.0e3, 1.0e3, 1.0e5
TRANSPORT_PRIORITY = 20.0

# Finite ITG drive: GKX uses a/L, not R/L.
A_OVER_LT, A_OVER_LN = 3.0, 1.0
NX, NY, NZ, NL, NM = 8, 8, 16, 4, 8
DT, SATURATION_STEPS, WINDOW_STEPS = 0.05, 8_000, 1_024

ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    MAX_MODES, MAX_NFEV = [1], [2]
    MINIMUM_MPOL = 3
    NX, NY, NZ, NM = 4, 4, 8, 3
    SATURATION_STEPS, WINDOW_STEPS = 2, 2

DATA = Path(vj.__file__).resolve().parents[1] / "examples/data/input.minimal_seed_nfp2"
inp = vj.VmecInput.from_file(DATA)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor + 1, 1] += SEED_PERTURBATION
zbs[inp.ntor + 1, 1] += SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs, am=np.zeros_like(inp.am), pres_scale=0.0)
if ci_smoke:
    inp = replace(
        inp,
        ns_array=np.asarray([11]),
        ftol_array=np.asarray([1.0e-7]),
        niter_array=np.asarray([600]),
        delt=0.5,
    ).change_resolution(mpol=3, ntor=3, ntheta=12, nzeta=10)

grid = gkx.build_spectral_grid(gkx.GridConfig(Nx=NX, Ny=NY, Nz=NZ, Lx=62.8, Ly=62.8))
terms = gkx.TermConfig(
    collisions=1.0,
    hypercollisions=1.0,
    hyperdiffusion=1.0,
    end_damping=1.0,
    nonlinear=1.0,
    apar=0.0,
    bpar=0.0,
)


def initial_state():
    """Small Hermitian multimode density perturbation."""

    state = jnp.zeros((1, NL, NM, NY, NX, NZ), dtype=jnp.complex64)
    profile = jnp.asarray(1.0e-3 * (1.0 + 0.2 * jnp.cos(grid.z)), jnp.float32)
    mode0 = jnp.asarray(profile * (1.0 + 0.2j), state.dtype)
    mode1 = jnp.asarray(profile * (0.3 - 0.1j), state.dtype)
    state = state.at[0, 0, 0, 1, 0].set(mode0)
    state = state.at[0, 0, 0, -1, 0].set(jnp.conj(mode0))
    state = state.at[0, 0, 0, 1, 1].set(mode1)
    state = state.at[0, 0, 0, -1, -1].set(jnp.conj(mode1))
    return state


def geometry(state, runtime):
    return turbulence.flux_tube_geometry(
        state, runtime, s_index=7, alpha=0.0, ntheta=NZ
    )


def parameters(local_geometry):
    return gkx.LinearParams(
        tprim=A_OVER_LT,
        fprim=A_OVER_LN,
        nu=0.01,
        kpar_scale=local_geometry.gradpar_value,
        nu_hermite=1.0,
        nu_laguerre=2.0,
        p_hyper_m=float(min(20, max(NM // 2, 1))),
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
        D_hyper=0.05,
        p_hyper_kperp=2.0,
        damp_ends_amp=0.1,
        damp_ends_widthfrac=0.125,
    )


# Spin-up warm start, wired but not switched on. The saturated state is still
# detached and still refreshed exactly where it always was -- once per accepted
# VMEX stage -- so the objective remains a fixed function of (state, runtime)
# for the whole of each stage and never becomes a function of the optimizer's
# within-stage history. What a warm spin-up would change is only the cost of
# producing the refreshed state: a stage that barely moved the flux tube could
# reseed from the previous saturated state instead of climbing out of a 1e-3
# seed again, on a quarter of the step budget. The policy restores the full
# cold spin-up as soon as the geometry moves by more than `geometry_tolerance`,
# or after `max_reuse` consecutive warm spin-ups, so a run of individually
# small steps cannot drift away from the attractor.
#
# `max_reuse=0` disables it, so this script's numbers are exactly what they
# were. Raise it to opt in. It is off by default because the saving is real but
# its cost has not been measured here: a shortened spin-up still has to
# re-equilibrate to the new geometry, and whether a quarter budget suffices is
# a question about this objective's sensitivity that only a full optimization
# run answers. Turn it on alongside a comparison against a `max_reuse=0` run.
saturation_warm_start = SaturationWarmStart(
    policy=SaturationRefreshPolicy(
        max_reuse=0, geometry_tolerance=0.05, warm_step_fraction=0.25
    )
)


def saturate(equilibrium):
    local_geometry = geometry(equilibrium.state, equilibrium.runtime)
    signature = flux_tube_signature(local_geometry)
    plan = saturation_warm_start.plan(signature, cold_steps=SATURATION_STEPS)
    seed = initial_state() if plan.seed is None else jnp.asarray(plan.seed)
    print(
        f"saturation spin-up: {'warm' if plan.warm else 'cold'} seed, "
        f"{plan.steps} steps ({plan.reason or 'first spin-up'})",
        flush=True,
    )
    state = gkx.integrate_nonlinear(
        seed,
        grid,
        local_geometry,
        parameters(local_geometry),
        DT,
        plan.steps,
        method="rk3",
        terms=terms,
        checkpoint=False,
        return_fields=False,
    )
    saturation_warm_start.record(state, signature, warm=plan.warm)
    return state


print("solving vacuum seed equilibrium", flush=True)
equilibrium = opt.solve_equilibrium(inp)
print("running GKX saturation", flush=True)
saturated_state = saturate(equilibrium)
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)


def turbulent_transport(state, runtime):
    """Actual GKX heat flux, differentiated through the post-saturation window."""

    local_geometry = geometry(state, runtime)
    return gkx.nonlinear_heat_flux_window(
        saturated_state,
        grid,
        local_geometry,
        parameters(local_geometry),
        DT,
        WINDOW_STEPS,
        terms=terms,
        method="rk3",
    )


seed_flux = float(turbulent_transport(equilibrium.state, equilibrium.runtime))
flux_scale = max(abs(seed_flux), 1.0e-4 if ci_smoke else 1.0e-12)
transport_weight = TRANSPORT_PRIORITY / (flux_scale * flux_scale)
objective_function_terms = [
    (qs, 0.0, QA_PRIORITY),
    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_PRIORITY),
    (opt.mean_iota, IOTA_TARGET, IOTA_PRIORITY),
    (turbulent_transport, 0.0, transport_weight),
]


def report(label, local_equilibrium):
    values = {
        "QS total": float(qs.total(local_equilibrium)),
        "aspect": float(
            opt.aspect_ratio(local_equilibrium.state, local_equilibrium.runtime)
        ),
        "mean iota": float(
            opt.mean_iota(local_equilibrium.state, local_equilibrium.runtime)
        ),
        "GKX Q": float(
            turbulent_transport(local_equilibrium.state, local_equilibrium.runtime)
        ),
    }
    print(
        f"[{label}] QS={values['QS total']:.6e}, aspect={values['aspect']:.4f}, "
        f"iota={values['mean iota']:.4f}, GKX Q={values['GKX Q']:.6e}"
    )
    return values


for max_mode, max_nfev in zip(MAX_MODES, MAX_NFEV):
    print(f"\n===== QA stage, max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = replace(inp, delt=0.5).change_resolution(
        mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6, nzeta=2 * mpol + 4
    )
    problem = opt.VmecProblem.from_tuples(
        inp,
        objective_function_terms,
        max_mode=max_mode,
        use_ess=True,
        implicit_jacobian_method="auto",
    )
    result = least_squares(
        problem.residual,
        problem.x0,
        jac=problem.residual_jac,
        x_scale=problem.scales,
        max_nfev=max_nfev,
        ftol=1.0e-6,
        xtol=1.0e-10,
        verbose=2,
    )
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    saturated_state = saturate(equilibrium)
    report(f"mode {max_mode}", equilibrium)

final_total = report("final", equilibrium)["QS total"]
final_flux = float(turbulent_transport(equilibrium.state, equilibrium.runtime))
print(f"\nQS total {final_total:.3e}; GKX Q {seed_flux:.3e} -> {final_flux:.3e}")
input_path = inp.to_indata("input.QA_GKX_optimized")
wout_path = vj.write_wout("wout_QA_GKX_optimized.nc", equilibrium.wout)
print(f"wrote {input_path}")
print(f"wrote {wout_path}")
try:
    for path in vj.plot_wout(wout_path, ".").values():
        print(f"wrote {path}")
except (ImportError, ValueError) as error:
    print(f"skipped optional equilibrium plots: {error}")
