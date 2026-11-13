"""Vacuum QA stellarator optimization with a nonlinear GKX heat-flux objective.

Research path only: this is a campaign, not a quick run. VMEX optimizes the
boundary of the seed deck ``examples/vmec/input.minimal_seed_nfp2`` in five
boundary-mode stages for quasisymmetry, aspect ratio and mean iota, and an extra
objective term adds the ion heat flux of a GKX nonlinear run. The flux is the
average over a post-saturation window, differentiated step by step with a fixed
step rk3 integrator; the saturated state itself is detached and refreshed once per
accepted stage. The equilibrium is vacuum; ``A_OVER_LT`` and ``A_OVER_LN`` supply
the gyrokinetic drive (GKX uses a/L, not R/L). SciPy's ``least_squares`` consumes
the residuals and their analytic Jacobian. README.md in this directory states the
scope of the result and lists the maintained VMEX turbulence examples.

The default ladder (8,000 saturation steps per stage) is meant for a GPU. Set
``WIRING_RUN = True`` for a tiny CPU run that checks the wiring and does not reach
saturation. Requires ``pip install vmex``.

Prints one line per stage. Writes ``summary.json``, ``vmex_optimization.png``,
``input.QA_GKX_optimized``, ``wout_QA_GKX_optimized.nc`` and the vmex equilibrium
plots to ``outputs/10_vmex_optimization``.
"""

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import vmex as vj
from scipy.optimize import least_squares
from vmex import optimize as opt
from vmex.core import turbulence

import gkx

SEED = Path(__file__).parents[1] / "vmec" / "input.minimal_seed_nfp2"
OUTPUT = Path("outputs/10_vmex_optimization")
WIRING_RUN = False
SURFACES = np.linspace(0.1, 1.0, 10)
MAX_MODES, MAX_NFEV = [1, 2, 3, 4, 5], [10, 20, 40, 80, 120]
ASPECT_TARGET, IOTA_TARGET = 6.0, 0.42
MINIMUM_MPOL = 5
SEED_PERTURBATION = 0.01
QA_PRIORITY, ASPECT_PRIORITY, IOTA_PRIORITY = 1.0e3, 1.0e3, 1.0e5
TRANSPORT_PRIORITY = 20.0
A_OVER_LT, A_OVER_LN = 3.0, 1.0
NX, NY, NZ, NL, NM = 8, 8, 16, 4, 8
DT, SATURATION_STEPS, WINDOW_STEPS = 0.05, 8_000, 1_024

# ---- Setup ----
inp = vj.VmecInput.from_file(SEED)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor + 1, 1] += SEED_PERTURBATION
zbs[inp.ntor + 1, 1] += SEED_PERTURBATION
inp = replace(inp, rbc=rbc, zbs=zbs, am=np.zeros_like(inp.am), pres_scale=0.0)
if WIRING_RUN:
    MAX_MODES, MAX_NFEV, MINIMUM_MPOL = [1], [2], 3
    NX, NY, NZ, NM, SATURATION_STEPS, WINDOW_STEPS = 4, 4, 8, 3, 2, 2
    inp = replace(
        inp,
        ns_array=np.asarray([11]),
        ftol_array=np.asarray([1.0e-7]),
        niter_array=np.asarray([600]),
        delt=0.5,
    ).change_resolution(mpol=3, ntor=3, ntheta=12, nzeta=10)

grid = gkx.build_spectral_grid(
    gkx.GridConfig(Nx=NX, Ny=NY, Nz=NZ, Lx=62.8, Ly=62.8, ky_layout="full")
)
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
        # Pinned: the recorded single-tube campaign (README.md) ran at 0.1; a new objective
        # leaves it unset for the default rule (docs/guidance.rst, End damping).
        damp_ends_rate=0.1,
        damp_ends_widthfrac=0.125,
    )


def saturate(equilibrium):
    """Saturated nonlinear state on the equilibrium's flux tube, detached from the optimizer."""

    local_geometry = geometry(equilibrium.state, equilibrium.runtime)
    return gkx.integrate_nonlinear(
        initial_state(),
        grid,
        local_geometry,
        parameters(local_geometry),
        DT,
        SATURATION_STEPS,
        # Fixed-step rk3, not the CFL-controlled default: the window is differentiated step by
        # step, so the step count must not depend on the state.
        method="rk3",
        terms=terms,
        checkpoint=False,
        return_fields=False,
    )


def turbulent_transport(state, runtime):
    """GKX heat flux, differentiated through the post-saturation window."""

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


# ---- Research path: boundary-mode ladder ----
print("solving the seed equilibrium and its GKX saturation", flush=True)
equilibrium = opt.solve_equilibrium(inp)
saturated_state = saturate(equilibrium)
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
seed_flux = float(turbulent_transport(equilibrium.state, equilibrium.runtime))
flux_scale = max(abs(seed_flux), 1.0e-4 if WIRING_RUN else 1.0e-12)
transport_weight = TRANSPORT_PRIORITY / flux_scale**2
objective_function_terms = [
    (qs, 0.0, QA_PRIORITY),
    (opt.aspect_ratio, ASPECT_TARGET, ASPECT_PRIORITY),
    (opt.mean_iota, IOTA_TARGET, IOTA_PRIORITY),
    (turbulent_transport, 0.0, transport_weight),
]

history = []


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
    history.append({"stage": label, **values})


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
print(f"\nGKX Q {seed_flux:.3e} -> {history[-1]['GKX Q']:.3e}")

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "summary.json").write_text(json.dumps(history, indent=2) + "\n")
gkx.set_plot_style()
fig, (ax_qs, ax_q) = plt.subplots(1, 2, figsize=(9.0, 3.6))
stages = [row["stage"] for row in history]
ax_qs.semilogy(stages, [row["QS total"] for row in history], "o-")
ax_qs.set(ylabel="QS total", title="Quasisymmetry error")
ax_q.plot(stages, [row["GKX Q"] for row in history], "o-")
ax_q.axhline(seed_flux, color="0.5", linestyle="--", label="seed")
ax_q.set(ylabel=r"$Q_i/Q_{GB}$", title="Post-saturation heat flux")
ax_q.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT / "vmex_optimization.png", dpi=150)
input_path = inp.to_indata(str(OUTPUT / "input.QA_GKX_optimized"))
wout_path = vj.write_wout(str(OUTPUT / "wout_QA_GKX_optimized.nc"), equilibrium.wout)
print(
    f"wrote {OUTPUT}/summary.json, vmex_optimization.png, {input_path} and {wout_path}"
)
try:
    vj.plot_wout(wout_path, OUTPUT)
except (ImportError, ValueError) as error:
    print(f"skipped optional equilibrium plots: {error}")
