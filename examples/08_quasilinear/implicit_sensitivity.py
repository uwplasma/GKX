"""Implicit eigenpair sensitivities of the quasilinear observables.

Builds a tiny dense copy of the linear Cyclone ITG operator, selects its most
unstable eigenpair, and differentiates five observables -- gamma, omega,
<k_perp^2>, the heat-flux weight Q/|phi|^2 and the mixing-length flux --
with respect to a/L_n and a/L_Ti through the implicit left/right eigenpair
formula. The Jacobian is checked against central finite differences. Prints
the report, saves it as ``quasilinear_implicit_sensitivity.json`` and plots
observables, Jacobian and derivative parity. Well under a minute on a laptop
CPU. Set ``OUTPUT = Path("docs/_static")`` to regenerate the tracked docs
artifact. Production runs keep the operator matrix-free; the dense fixture
exists only so the derivative can be checked exactly.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import gkx
from gkx.core_grid import select_ky_grid
from gkx.diagnostics import fieldline_quadrature_weights, heat_flux_species

OUTPUT = Path("outputs/08_quasilinear")
A_OVER_LN, A_OVER_LTI = 2.2, 6.9  # differentiated gradients (a/L, not R/L)
NL, NM = 2, 3  # dense fixture velocity resolution
STEP, RTOL, ATOL = 1.0e-3, 5.0e-2, 2.0e-3  # finite-difference step and tolerances
OBSERVABLES = (
    r"$\gamma$",
    r"$\omega$",
    r"$k_{\perp,\mathrm{eff}}^2$",
    r"$\hat Q_i$",
    r"$Q_i^{ML}$",
)
PARAMETERS = (r"$a/L_n$", r"$a/L_{Ti}$")

# Geometry and a single-k_y grid small enough to build the operator densely.
config = gkx.CycloneBaseCase(grid=gkx.GridConfig(Nx=1, Ny=6, Nz=4, Lx=6.0, Ly=12.0))
grid = select_ky_grid(gkx.build_spectral_grid(config.grid), 1)
geometry = gkx.SAlphaGeometry.from_config(config.geometry)
state_shape = (NL, NM, grid.ky.size, grid.kx.size, grid.z.size)
terms = gkx.LinearTerms(
    streaming=1.0,
    mirror=1.0,
    curvature=1.0,
    gradb=1.0,
    diamagnetic=1.0,
    collisions=0.0,
    hypercollisions=0.0,
    end_damping=0.0,
    apar=0.0,
    bpar=0.0,
)


def parameters(x):
    """Collisionless electrostatic parameters with (a/L_n, a/L_Ti) = x."""

    return gkx.LinearParams(
        fprim=x[0],
        tprim=x[1],
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )


cache = gkx.build_linear_cache(
    grid, geometry, parameters(jnp.asarray([A_OVER_LN, A_OVER_LTI])), NL, NM
)
volume_weight, flux_weight = fieldline_quadrature_weights(geometry, grid)


def rhs(state, x):
    return gkx.linear_rhs_cached(
        state, cache, parameters(x), terms=terms, use_jit=False, use_custom_vjp=False
    )


def operator_matrix(x):
    return gkx.explicit_complex_operator_matrix(
        lambda state: rhs(state, x)[0], state_shape
    )


def observables(eigenvalue, eigenvector, x):
    """gamma, omega, <k_perp^2>, Q/|phi|^2 and the mixing-length flux."""

    state = jnp.reshape(eigenvector, state_shape)
    _, phi = rhs(state, x)
    zeros = jnp.zeros_like(phi)
    kperp2 = gkx.effective_kperp2(phi, cache, volume_weight)
    norm = gkx.phi_norm2(
        phi, cache, parameters(x), volume_weight, normalization="phi_rms"
    )
    weight = (
        jnp.sum(
            heat_flux_species(
                state, phi, zeros, zeros, cache, grid, parameters(x), flux_weight
            )
        )
        / norm
    )
    gamma, omega = jnp.real(eigenvalue), -jnp.imag(eigenvalue)
    return jnp.asarray(
        [
            gamma,
            omega,
            kperp2,
            weight,
            gkx.saturated_flux_from_linear_weight(weight, gamma, kperp2),
        ]
    )


x0 = jnp.asarray([A_OVER_LN, A_OVER_LTI])
report = gkx.implicit_eigenpair_observable_sensitivity_report(
    operator_matrix, observables, x0, step=STEP, rtol=RTOL, atol=ATOL, gap_floor=1.0e-6
)
eigenvalues, eigenvectors = jnp.linalg.eig(operator_matrix(x0))
index = int(report["selected_index"])
values = np.asarray(
    observables(eigenvalues[index], eigenvectors[:, index], x0), dtype=float
)
report.update(
    kind="quasilinear_implicit_sensitivity_demo",
    case="tiny_cyclone_linear_rhs",
    parameters={"fprim": A_OVER_LN, "tprim": A_OVER_LTI},
    observable_labels=list(OBSERVABLES),
    parameter_labels=list(PARAMETERS),
    observables=values.tolist(),
)

jac_implicit = np.asarray(report["jacobian_implicit"], dtype=float)
jac_fd = np.asarray(report["jacobian_fd"], dtype=float)
print(
    f"passed = {report['passed']}, branch gap = {float(report['eigenvalue_gap']):.2e}"
)
print(f"max relative derivative error = {float(report['max_rel_error']):.2e}")
for label, value, row in zip(OBSERVABLES, values, jac_implicit):
    print(
        f"{label:>28} = {value:+.4e}   d/d(a/L_n) = {row[0]:+.4e}   d/d(a/L_Ti) = {row[1]:+.4e}"
    )

OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "quasilinear_implicit_sensitivity.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)

gkx.set_plot_style()
fig, (ax_values, ax_jacobian, ax_parity) = plt.subplots(1, 3, figsize=(13.0, 3.9))
ax_values.bar(range(len(values)), values, color="tab:blue")
ax_values.set_xticks(range(len(values)), OBSERVABLES, rotation=20)
ax_values.set_title("Quasilinear observables")
limit = max(float(np.abs(jac_implicit).max()), 1.0e-12)
image = ax_jacobian.imshow(
    jac_implicit, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto"
)
ax_jacobian.set_xticks(range(len(PARAMETERS)), PARAMETERS)
ax_jacobian.set_yticks(range(len(OBSERVABLES)), OBSERVABLES)
ax_jacobian.set_title("Implicit sensitivity matrix")
fig.colorbar(image, ax=ax_jacobian, fraction=0.046, pad=0.04)
span = [min(jac_fd.min(), jac_implicit.min()), max(jac_fd.max(), jac_implicit.max())]
ax_parity.plot(span, span, "--", color="tab:red", linewidth=1.0)
ax_parity.scatter(jac_fd.ravel(), jac_implicit.ravel(), color="black", s=30)
ax_parity.set(
    xlabel="central finite difference",
    ylabel="implicit derivative",
    title="Derivative parity",
)
fig.tight_layout()
fig.savefig(OUTPUT / "quasilinear_implicit_sensitivity.png", dpi=150)
print(f"wrote {OUTPUT}/quasilinear_implicit_sensitivity.json and .png")
