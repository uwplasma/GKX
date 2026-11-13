"""Autodiff of a linear Cyclone ITG run, and an inverse problem built on it.

The whole initial-value run, from the gradients a/L_Ti and a/L_n to the fitted
growth rate and frequency of two ``k_y`` modes, is differentiated end to end with
JAX.

Fast path: the autodiff Jacobian of the four observables (gamma and omega of each
mode) at one point, checked against central finite differences.

Research path: a damped Gauss-Newton solve driven by that Jacobian recovers the
gradients, planted at ``TRUE``, from the wrong starting guess ``GUESS``. One mode
alone does not identify both gradients; two modes do. The script then reports the
local covariance of the recovered gradients and sweeps the observables across each
gradient.

Prints the Jacobian error and the recovery. Writes ``summary.json`` and
``autodiff.png`` to ``outputs/09_autodiff``. About a minute on a laptop CPU.
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
POINT = np.array([2.2, 0.9])  # (a/L_Ti, a/L_n) of the Jacobian check
TRUE = np.array([2.8, 0.8])  # planted (a/L_Ti, a/L_n)
GUESS = np.array([1.6, 1.1])  # starting guess for the inverse solve
GN_STEPS = 18  # Gauss-Newton iterations
DAMPING = 3.0e-3  # Levenberg damping of the Gauss-Newton step

config = gkx.CycloneBaseCase(grid=gkx.GridConfig(Nx=1, Ny=8, Nz=32, Lx=6.28, Ly=6.28))
grid = gkx.build_spectral_grid(config.grid)
geometry = gkx.SAlphaGeometry.from_config(config.geometry)
NL, NM = 2, 2
initial = jnp.zeros((NL, NM, grid.ky.size, grid.kx.size, grid.z.size), jnp.complex64)
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


def values(x):
    return np.asarray(forward(jnp.asarray(x)))


# ---- Fast path: autodiff Jacobian against finite differences ----
eps = 1.0e-3
jac_ad = np.asarray(jacobian(jnp.asarray(POINT)))
jac_fd = np.stack(
    [
        (values(POINT + eps * e) - values(POINT - eps * e)) / (2 * eps)
        for e in np.eye(2)
    ],
    axis=1,
)
jac_rel_error = np.linalg.norm(jac_ad - jac_fd, axis=0) / (
    np.linalg.norm(jac_fd, axis=0) + 1.0e-12
)
print(
    f"autodiff vs finite-difference Jacobian, relative error per column: {jac_rel_error}"
)

# ---- Research path: Gauss-Newton inverse solve with a backtracking line search ----
target = values(TRUE)
gradients, path, losses = GUESS.copy(), [GUESS.copy()], []
for _ in range(GN_STEPS):
    residual = values(gradients) - target
    losses.append(float(residual @ residual))
    if losses[-1] < 1.0e-14:
        break
    J = np.asarray(jacobian(jnp.asarray(gradients)))
    step = np.linalg.solve(J.T @ J + DAMPING * np.eye(2), J.T @ residual)
    step *= min(1.0, 0.3 / max(np.linalg.norm(step), 1.0e-12))
    for alpha in 0.5 ** np.arange(10):
        trial = values(gradients - alpha * step) - target
        if trial @ trial <= losses[-1]:
            gradients = gradients - alpha * step
            path.append(gradients.copy())
            break
    else:
        break
residual = values(gradients) - target
uq = gkx.covariance_diagnostics(jac_ad, residual, regularization=1.0e-9)
print(f"planted   a/L_Ti = {TRUE[0]:.4f}, a/L_n = {TRUE[1]:.4f}")
print(f"guess     a/L_Ti = {GUESS[0]:.4f}, a/L_n = {GUESS[1]:.4f}")
print(
    f"recovered a/L_Ti = {gradients[0]:.4f}, a/L_n = {gradients[1]:.4f} after {len(path) - 1} steps"
)

sweep_t, sweep_n = np.linspace(1.2, 3.8, 16), np.linspace(0.4, 1.6, 16)
values_t = np.stack([values([v, TRUE[1]]) for v in sweep_t])
values_n = np.stack([values([TRUE[0], v]) for v in sweep_n])

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
summary = {
    "observables": "gamma_ky0, omega_ky0, gamma_ky1, omega_ky1",
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
    "tprim_sweep": {"tprim": sweep_t.tolist(), "observables": values_t.tolist()},
    "fprim_sweep": {"fprim": sweep_n.tolist(), "observables": values_n.tolist()},
    **uq,
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

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
fig.savefig(OUTPUT / "autodiff.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/autodiff.png")
