"""Quasilinear heat-flux spectrum for the Cyclone ITG case and its derivatives.

Fast path: ``case.toml`` (Cyclone ITG with the ``[quasilinear]`` diagnostic on)
through ``gkx.scan``. Each ``k_y`` reports the quasilinear heat-flux weight
Q/|phi|^2 and the mixing-length estimate, ``gamma / <k_perp^2>`` times that
weight. The rule is uncalibrated: it ranks spectra and designs and does not
predict an absolute nonlinear flux. ``case_full.toml`` uses the Krylov
eigensolver at (Nl, Nm) = (8, 8).

Research path: implicit derivatives of five observables (gamma, omega,
<k_perp^2>, the weight, the mixing-length flux) with respect to a/L_n and a/L_Ti.
They come from the left and right eigenvectors of the most unstable mode of a
small dense copy of the linear operator and are checked against central finite
differences. Production runs keep the operator matrix-free; the dense copy exists
so the derivative can be checked exactly.

Prints the spectrum and the derivative report. Writes ``summary.json`` and
``quasilinear.png`` to ``outputs/08_quasilinear``. About a minute on a laptop CPU.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import gkx
from gkx.core_grid import select_ky_grid
from gkx.diagnostics import fieldline_quadrature_weights, heat_flux_species

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/08_quasilinear")
KY = [0.1, 0.2, 0.3, 0.4]  # k_y rho_i
A_OVER_LN, A_OVER_LTI = 2.2, 6.9  # differentiated gradients (a/L, not R/L)
NL, NM = 2, 3  # velocity resolution of the dense operator
STEP, RTOL, ATOL = 1.0e-3, 5.0e-2, 2.0e-3  # finite-difference step and tolerances
OBSERVABLES = (
    r"$\gamma$",
    r"$\omega$",
    r"$k_{\perp,\mathrm{eff}}^2$",
    r"$\hat Q_i$",
    r"$Q_i^{ML}$",
)
PARAMETERS = (r"$a/L_n$", r"$a/L_{Ti}$")

# ---- Fast path ----
case = gkx.load(CASE)
run = dict(
    Nl=case.run.Nl, Nm=case.run.Nm, solver=case.run.solver
)  # the deck's [run] block
scan = gkx.scan(case, KY, **run, show_progress=True)
weight = np.array([p["heat_flux_weight_total"] for p in scan.quasilinear])
kperp2 = np.array([p["kperp_eff2"] for p in scan.quasilinear])
flux = np.array([p["saturated_heat_flux_total"] for p in scan.quasilinear])
print(f"{'ky':>6} {'gamma':>9} {'<kperp^2>':>10} {'Q/|phi|^2':>10} {'Q_ML':>9}")
for row in zip(scan.ky, scan.gamma, kperp2, weight, flux):
    print("{:6.2f} {:9.5f} {:10.4f} {:10.4f} {:9.4f}".format(*row))
print(f"k_y-summed mixing-length heat flux (uncalibrated): {flux.sum():.4f}")

# ---- Research path: implicit eigenpair sensitivities ----
# A single-k_y grid small enough to build the operator densely.
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


x0 = jnp.asarray([A_OVER_LN, A_OVER_LTI])
cache = gkx.build_linear_cache(grid, geometry, parameters(x0), NL, NM)
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
    heat = heat_flux_species(
        state, phi, zeros, zeros, cache, grid, parameters(x), flux_weight
    )
    weight = jnp.sum(heat) / norm
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


report = gkx.implicit_eigenpair_observable_sensitivity_report(
    operator_matrix, observables, x0, step=STEP, rtol=RTOL, atol=ATOL, gap_floor=1.0e-6
)
eigenvalues, eigenvectors = jnp.linalg.eig(operator_matrix(x0))
index = int(report["selected_index"])
values = np.asarray(observables(eigenvalues[index], eigenvectors[:, index], x0), float)
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

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
scan.save(OUTPUT / "scan")
summary = {
    "case": CASE.name,
    "claim_level": scan.quasilinear[0]["metadata"]["claim_level"],
    "ky": np.asarray(scan.ky).tolist(),
    "gamma": np.asarray(scan.gamma).tolist(),
    "kperp_eff2": kperp2.tolist(),
    "heat_flux_weight": weight.tolist(),
    "saturated_heat_flux": flux.tolist(),
    "sensitivity": {**report, "observables": values.tolist()},
}
(OUTPUT / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)

gkx.set_plot_style()
fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
panels = (
    (scan.gamma, r"$\gamma\,a/v_{ti}$", "Growth rate"),
    (weight, r"$Q_i/|\phi|^2$", "Quasilinear weight"),
    (flux, r"$Q_i^{ML}$ (uncalibrated)", "Mixing-length flux"),
)
for ax, (y, label, title) in zip(axes[0], panels):
    ax.plot(scan.ky, y, "o-")
    ax.set(xlabel=r"$k_y\rho_i$", ylabel=label, title=title)
ax_values, ax_jacobian, ax_parity = axes[1]
ax_values.bar(range(len(values)), values)
ax_values.set_xticks(range(len(values)), OBSERVABLES, rotation=20)
ax_values.set_title("Observables of the dense operator")
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
fig.savefig(OUTPUT / "quasilinear.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/quasilinear.png")
