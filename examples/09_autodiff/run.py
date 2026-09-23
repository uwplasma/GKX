"""Autodiff inverse problem: recover a/L_Ti and a/L_n from two linear modes.

Differentiates a linear Cyclone ITG initial-value run end to end with JAX. The
observables are the growth rate and frequency of two ``k_y`` modes, generated
at planted gradients; a Gauss-Newton solve driven by the autodiff Jacobian
recovers the gradients from a wrong starting guess. One mode alone does not
identify both gradients; two modes do. The autodiff Jacobian is checked
against central finite differences, and the local covariance of the recovered
gradients is reported. Prints the recovery, saves a JSON summary and two sweep
CSVs, and plots loss, parameter path, derivative parity and sweeps. About a
minute on a laptop CPU. Set ``OUTPUT = Path("docs/_static")`` to regenerate
``autodiff_inverse_twomode.png``.
``geometry_bridge.py`` in this directory differentiates through VMEC geometry.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import gkx

OUTPUT = Path("outputs/09_autodiff")
STEPS = 120  # forward-integration steps per observable evaluation
DT = 0.05  # time step
KY_INDICES = (1, 3)  # the two probed k_y modes
TRUE = np.array([2.8, 0.8])  # planted (a/L_Ti, a/L_n)
GUESS = np.array([1.6, 1.1])  # starting guess for the inverse solve
GN_STEPS = 18  # Gauss-Newton iterations
DAMPING = 3.0e-3  # Levenberg damping of the Gauss-Newton step

config = gkx.CycloneBaseCase(grid=gkx.GridConfig(Nx=1, Ny=8, Nz=32, Lx=6.28, Ly=6.28))
grid = gkx.build_spectral_grid(config.grid)
geometry = gkx.SAlphaGeometry.from_config(config.geometry)
NL, NM = 2, 2
initial = jnp.zeros(
    (NL, NM, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
)
for index in KY_INDICES:
    initial = initial.at[0, 0, index, 0, :].set(1.0e-3)
cache = gkx.build_linear_cache(
    grid,
    geometry,
    gkx.build_linear_params([gkx.Species(1.0, 1.0, 1.0, 1.0, 2.0, 0.8)], tau_e=1.0),
    NL,
    NM,
)
time = jnp.arange(STEPS) * DT
window = slice(STEPS // 2, None)  # fit gamma and omega over the second half


def observables(gradients):
    """(gamma, omega) of both modes, as a differentiable function of the gradients."""

    parameters = gkx.LinearParams(
        charge_sign=jnp.asarray([1.0]),
        density=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        temp=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        rho=jnp.asarray([1.0]),
        tz=jnp.asarray([1.0]),
        tprim=jnp.asarray([gradients[0]]),
        fprim=jnp.asarray([gradients[1]]),
        tau_e=1.0,
    )
    _, phi = gkx.integrate_linear(
        initial, grid, geometry, parameters, dt=DT, steps=STEPS, cache=cache
    )
    t = time[window] - jnp.mean(time[window])
    rows = []
    for index in KY_INDICES:
        signal = phi[window, index, 0, 0]
        log_amplitude = jnp.log(jnp.abs(signal) + 1.0e-12)
        phase = jnp.unwrap(jnp.angle(signal))
        gamma = jnp.sum(t * (log_amplitude - jnp.mean(log_amplitude))) / jnp.sum(t**2)
        omega = -jnp.sum(t * (phase - jnp.mean(phase))) / jnp.sum(t**2)
        rows += [gamma, omega]
    return jnp.stack(rows)


forward = jax.jit(observables)
jacobian = jax.jit(jax.jacobian(observables))
target = np.asarray(forward(jnp.asarray(TRUE)))

# Damped Gauss-Newton with a backtracking line search.
path, losses = [GUESS.copy()], []
gradients = GUESS.copy()
for _ in range(GN_STEPS):
    residual = np.asarray(forward(jnp.asarray(gradients))) - target
    losses.append(float(residual @ residual))
    if losses[-1] < 1.0e-14:
        break
    J = np.asarray(jacobian(jnp.asarray(gradients)))
    step = np.linalg.solve(J.T @ J + DAMPING * np.eye(2), J.T @ residual)
    step *= min(1.0, 0.3 / max(np.linalg.norm(step), 1.0e-12))
    for alpha in 0.5 ** np.arange(10):
        candidate = gradients - alpha * step
        trial = np.asarray(forward(jnp.asarray(candidate))) - target
        if trial @ trial <= losses[-1]:
            gradients = candidate
            path.append(gradients.copy())
            break
    else:
        break
residual = np.asarray(forward(jnp.asarray(gradients))) - target

# Autodiff Jacobian against central finite differences at a fixed point.
center, eps = jnp.asarray([2.2, 0.9]), 1.0e-3
jac_ad = np.asarray(jacobian(center))
jac_fd = np.stack(
    [
        (np.asarray(forward(center + eps * e)) - np.asarray(forward(center - eps * e)))
        / (2 * eps)
        for e in jnp.eye(2)
    ],
    axis=1,
)
jac_rel_error = np.linalg.norm(jac_ad - jac_fd, axis=0) / (
    np.linalg.norm(jac_fd, axis=0) + 1.0e-12
)
uq = gkx.covariance_diagnostics(jac_ad, residual, regularization=1.0e-9)

print(f"planted  a/L_Ti = {TRUE[0]:.4f}, a/L_n = {TRUE[1]:.4f}")
print(f"guess    a/L_Ti = {GUESS[0]:.4f}, a/L_n = {GUESS[1]:.4f}")
print(
    f"recovered a/L_Ti = {gradients[0]:.4f}, a/L_n = {gradients[1]:.4f} after {len(path) - 1} steps"
)
print(f"AD vs FD Jacobian relative error per column: {jac_rel_error}")

sweep_t = np.linspace(1.2, 3.8, 16)
sweep_n = np.linspace(0.4, 1.6, 16)
values_t = np.asarray(
    jax.vmap(lambda v: forward(jnp.asarray([v, TRUE[1]])))(jnp.asarray(sweep_t))
)
values_n = np.asarray(
    jax.vmap(lambda v: forward(jnp.asarray([TRUE[0], v])))(jnp.asarray(sweep_n))
)

OUTPUT.mkdir(parents=True, exist_ok=True)
header = "gamma_ky0,omega_ky0,gamma_ky1,omega_ky1"
np.savetxt(
    OUTPUT / "autodiff_inverse_twomode_tprim_sweep.csv",
    np.column_stack([sweep_t, values_t]),
    delimiter=",",
    header="tprim," + header,
    comments="",
)
np.savetxt(
    OUTPUT / "autodiff_inverse_twomode_fprim_sweep.csv",
    np.column_stack([sweep_n, values_n]),
    delimiter=",",
    header="fprim," + header,
    comments="",
)
summary = {
    "target_observables": target.tolist(),
    "tprim_init": float(GUESS[0]),
    "fprim_init": float(GUESS[1]),
    "tprim_final": float(gradients[0]),
    "fprim_final": float(gradients[1]),
    "parameter_abs_error": np.abs(gradients - TRUE).tolist(),
    "observable_abs_error": np.abs(residual).tolist(),
    "loss_history": losses,
    "jac_autodiff": jac_ad.tolist(),
    "jac_finite_diff": jac_fd.tolist(),
    "jac_rel_error": jac_rel_error.tolist(),
    **uq,
}
(OUTPUT / "autodiff_inverse_twomode_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

gkx.set_plot_style()
path = np.asarray(path)
fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
axes[0, 0].semilogy(np.maximum(losses, 1.0e-16), "o-")
axes[0, 0].set(
    xlabel="Gauss-Newton iteration",
    ylabel="squared residual",
    title="Inverse-solve loss",
)
axes[0, 1].plot(path[:, 0], path[:, 1], "o-", label="Gauss-Newton path")
axes[0, 1].plot(*TRUE, "*", markersize=14, color="tab:red", label="planted")
axes[0, 1].set(xlabel=r"$a/L_{Ti}$", ylabel=r"$a/L_n$", title="Parameter recovery")
axes[0, 1].legend(frameon=False)
span = [min(jac_fd.min(), jac_ad.min()), max(jac_fd.max(), jac_ad.max())]
axes[1, 0].plot(span, span, "--", color="0.5")
axes[1, 0].scatter(jac_fd.ravel(), jac_ad.ravel(), color="black")
axes[1, 0].set(xlabel="finite difference", ylabel="autodiff", title="Jacobian parity")
for column, label in ((0, r"$\gamma$, mode 1"), (2, r"$\gamma$, mode 2")):
    axes[1, 1].plot(sweep_t, values_t[:, column], label=label)
axes[1, 1].axvline(TRUE[0], color="tab:red", linestyle=":")
axes[1, 1].set(
    xlabel=r"$a/L_{Ti}$", ylabel=r"$\gamma\,a/v_{ti}$", title="Growth-rate sweep"
)
axes[1, 1].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT / "autodiff_inverse_twomode.png", dpi=150)
print(f"wrote {OUTPUT}/autodiff_inverse_twomode_summary.json and .png")
