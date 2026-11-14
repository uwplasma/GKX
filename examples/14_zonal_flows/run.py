"""Zonal-flow response: Rosenbluth-Hinton residual, GAM and collisional damping.

Fast path: ``case.toml`` is the Rosenbluth-Hinton protocol (k_y = 0,
k_x rho_i = 0.05, adiabatic electrons, q = 1.4, epsilon = 0.1, an ion density
perturbation uniform along the line), collisionless. The flux-surface average
<phi>(t) / <phi>(0) rings at the geodesic acoustic mode (GAM) and settles to the
residual; the script prints the GAM frequency and the residual averaged over whole
GAM periods, and compares it with ``1 / (1 + 1.6 q^2 / sqrt(epsilon))`` (0.0916).
The formula is the epsilon -> 0 limit, and the residual falls towards it as
(Nl, Nm) grows (0.117, 0.100, 0.095 at (4, 8), (6, 16), (8, 24); docs/validation.rst).

Research path: the same initial state with collisions at ``NU`` for each operator in
``OPERATORS``. The residual decays instead of persisting, to about 0.006 of its initial
value at ``NU = 0.05`` whatever the operator. Both analyses use
``gkx.diagnostics.modes.zonal_response``. ``benchmarks/cases/collisional_zonal_response.toml``
is the 600-time-unit deck of the validation tables.

Prints the GAM frequency and the residuals. Writes ``summary.json`` and
``zonal_flows.png`` to ``outputs/14_zonal_flows``. About a minute on a laptop CPU.
"""

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import gkx
from gkx.diagnostics.modes import zonal_response
from gkx.geometry import ensure_flux_tube_geometry_data
from gkx.runtime import build_runtime_geometry

CASE = Path(__file__).with_name("case.toml")
OUTPUT = Path("outputs/14_zonal_flows")
T_MAX = 150.0  # time units of every run
OPERATORS = ("landau", "sugama")  # collision operators of the research path
NU = 0.05  # collisionality nu a / v_ti of the research path


def zonal_trace(case):
    """<phi>(t) / <phi>(0) of a zonal run: the Jacobian average over z."""
    run = gkx.solve(case, show_progress=True)
    z = jnp.asarray(run.z)
    jac = ensure_flux_tube_geometry_data(build_runtime_geometry(case), z).jacobian(z)
    t, response = zonal_response(
        run.t, run.field_history, run.selection, np.asarray(jac)
    )
    return t, response.real


def gam_average(t, x, t_min):
    """GAM angular frequency, and the mean of ``x`` over whole GAM periods after ``t_min``."""
    late = t > t_min
    tt, xx = t[late], x[late] - x[late].mean()
    up = np.flatnonzero((xx[:-1] < 0) & (xx[1:] >= 0))  # rising zero crossings
    cross = tt[up] - xx[up] * (tt[up + 1] - tt[up]) / (xx[up + 1] - xx[up])
    period = float(np.mean(np.diff(cross)))
    n = int((t[-1] - t_min) // period)
    window = (t >= t_min) & (t < t_min + n * period)
    return 2 * np.pi / period, float(x[window].mean())


# ---- Fast path: collisionless residual and GAM ----
case = gkx.load(CASE)
case = case.replace(time=replace(case.time, t_max=T_MAX, progress_bar=False))
t, collisionless = zonal_trace(case)
omega_gam, residual = gam_average(t, collisionless, t_min=T_MAX / 3)
q, eps = case.geometry.q, case.geometry.epsilon
formula = 1.0 / (1.0 + 1.6 * q**2 / np.sqrt(eps))
print(f"GAM frequency = {omega_gam:.3f} v_ti/a")
print(f"residual = {residual:.4f} (Rosenbluth-Hinton formula {formula:.4f})")

# ---- Research path: collisional damping ----
collisional = {}
for operator in OPERATORS:
    local = case.replace(
        species=tuple(replace(s, nu=NU) for s in case.species),
        physics=replace(case.physics, collisions=True),
        time=replace(case.time, collision_operator=operator),
    )
    _, response = zonal_trace(local)
    collisional[operator] = response
    print(
        f"{operator}: <phi>/<phi>(0) over the last third = {response[t > 2 * T_MAX / 3].mean():.4f}"
    )

# ---- Outputs ----
OUTPUT.mkdir(parents=True, exist_ok=True)
summary = {
    "case": CASE.name,
    "Nl": case.run.Nl,
    "Nm": case.run.Nm,
    "gam_omega": omega_gam,
    "residual": residual,
    "residual_formula": float(formula),
    "nu": NU,
    "collisional_late_mean": {
        op: float(r[t > 2 * T_MAX / 3].mean()) for op, r in collisional.items()
    },
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

gkx.set_plot_style()
fig, (ax_free, ax_coll) = plt.subplots(1, 2, figsize=(10.0, 3.8))
ax_free.plot(t, collisionless, label="collisionless")
ax_free.axhline(formula, color="tab:red", linestyle=":", label="Rosenbluth-Hinton")
ax_free.axhline(residual, color="0.5", linestyle="--", label="GAM-averaged")
for operator, response in collisional.items():
    ax_coll.plot(t, response, label=operator)
ax_coll.axhline(0.0, color="0.7", linewidth=0.8)
ax_free.set(
    xlabel=r"$t\,v_{ti}/a$", ylabel=r"$\langle\phi\rangle/\langle\phi\rangle_0$"
)
ax_free.set_title("Collisionless residual")
ax_coll.set(xlabel=r"$t\,v_{ti}/a$", title=rf"Collisional, $\nu a/v_{{ti}}$ = {NU}")
ax_free.legend(frameon=False, fontsize="small")
ax_coll.legend(frameon=False, fontsize="small")
fig.tight_layout()
fig.savefig(OUTPUT / "zonal_flows.png", dpi=150)
print(f"wrote {OUTPUT}/summary.json and {OUTPUT}/zonal_flows.png")
