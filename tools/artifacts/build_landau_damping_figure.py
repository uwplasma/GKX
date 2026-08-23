"""Landau damping validation: GKX's own linear operator against the exact roots.

Physics anchor: the slab gyrokinetic ion-acoustic dispersion relation with
adiabatic electrons at ``k_perp -> 0``,

    1 + T_i/T_e + zeta Z(zeta) = 0,   zeta = omega / (k_par sqrt(2 T_i/m_i)),

with ``Z`` the Fried-Conte plasma dispersion function. The roots are solved here
from ``scipy.special.wofz`` to double precision rather than read off a table.

Everything measured below is produced by ``linear_rhs_cached`` -- GKX's
production linear operator on a slab flux tube -- not by a reference hierarchy
written for the figure. Panel (b) eigen-decomposes that same operator by
applying it to Hermite basis vectors.

Three traps make the obvious measurement wrong, each in a way that still yields a
plausible number:

1. **A collisionless truncated Hermite system cannot damp asymptotically.** Free
   streaming is anti-Hermitian, so the truncated matrix has a purely real
   spectrum -- verified below to ~1e-13. Apparent damping is a transient that
   ends at recurrence, ``t_rec ~ 2 sqrt(N_m)/(k_par v_ti)``.

2. **The Landau root is not an eigenvalue.** It is a pole of the analytically
   continued response. Taking the least-damped eigenvalue of the collisional
   operator and extrapolating gives 76% error at ``T_e/T_i = 1``, because at
   strong damping the discrete modes are ballistic rather than collective.
   Landau damping is an initial-value phenomenon and must be measured as one.

3. **A density perturbation with no initial flow is a standing wave.** It splits
   into left- and right-going sound waves, so the signal stays real and beats
   through zeros. Unwrapping its phase returns ``omega ~ 0.13`` instead of
   ``2.05``; fitting ``A exp(gamma t) cos(omega t + phi)`` returns the root.

Reference for the protocol and the recurrence time: Kanekar, Schekochihin,
Dorland & Loureiro, J. Plasma Phys. 81, 305810104 (2015).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, fsolve
from scipy.signal import hilbert
from scipy.special import wofz

from gkx.artifacts.figure_style import (
    GKX_COLORS,
    REFERENCE_STYLE,
    annotate_reference,
    figure_style,
    panel_label,
    save_figure,
)
from gkx.config import GeometryConfig, GridConfig
from gkx.core.grid import build_spectral_grid
from gkx.geometry import SlabGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearTerms, linear_params_for_geometry
from gkx.operators.linear.rhs import linear_rhs_cached

# Z takes zeta = omega/(k v_t) with v_t = sqrt(2T/m); GKX normalizes to
# v_ti = sqrt(T/m). Roots are reported in the GKX convention, i.e. times sqrt(2).
# Mixing the two is a factor-1.414 error that still looks physical, and is the
# single most common source of false agreement in this test.
_CONVENTION = np.sqrt(2.0)

_NU_SCAN = np.array([0.020, 0.015, 0.010, 0.0075, 0.005, 0.004, 0.003, 0.002])

# Streaming and collisions only: no drifts, no mirror, no gradient drive.
_TERMS = LinearTerms(
    mirror=0.0,
    curvature=0.0,
    gradb=0.0,
    diamagnetic=0.0,
    collisions=1.0,
    hypercollisions=0.0,
    end_damping=0.0,
)


def plasma_dispersion(zeta: complex) -> complex:
    """Fried-Conte ``Z(zeta)`` via the Faddeeva function."""

    return 1j * np.sqrt(np.pi) * wofz(zeta)


def exact_root(te_over_ti: float, guess: complex) -> complex:
    """Root of ``1 + T_i/T_e + zeta Z(zeta) = 0`` as ``omega/(k_par v_ti)``."""

    tau = 1.0 / te_over_ti

    def residual(v: np.ndarray) -> list[float]:
        z = complex(v[0], v[1])
        f = 1.0 + tau + z * plasma_dispersion(z)
        return [f.real, f.imag]

    sol = fsolve(residual, [guess.real, guess.imag])
    return _CONVENTION * complex(sol[0], sol[1])


def _slab_setup(hermite: int, nu: float, te_over_ti: float):
    """Build the GKX slab flux tube, cache and jitted RHS for this problem.

    ``Ly`` is deliberately huge so the single retained ``ky`` is ~1e-4: that is
    the ``k_perp rho_i -> 0`` limit the dispersion relation is written in. The
    box length in ``z`` makes the parallel harmonic satisfy ``k_par v_ti = 1``,
    so time is directly comparable with the root.
    """

    grid = build_spectral_grid(
        GridConfig(Nx=1, Ny=2, Nz=32, Lx=1.0e4, Ly=1.0e4, boundary="periodic")
    )
    geometry = SlabGeometry.from_config(
        GeometryConfig(q=1.0, s_hat=0.0, epsilon=0.0, R0=1.0)
    )
    # GKX's tau_e is T_i/T_e (docs/theory.rst), the reciprocal of the ratio this
    # script is parameterized by. T_e = T_i cannot distinguish the two, so the
    # T_e/T_i = 10 case is what actually pins the convention.
    params = linear_params_for_geometry(geometry, tau_e=1.0 / te_over_ti, nu=nu)
    cache = build_linear_cache(grid, geometry, params, 1, hermite)
    rhs = jax.jit(
        lambda state: linear_rhs_cached(
            state, cache, params, terms=_TERMS, use_jit=False
        )[0]
    )
    return grid, rhs


def evolve(
    *, hermite: int, nu: float, te_over_ti: float, t_max: float, dt: float = 0.002
) -> tuple[np.ndarray, np.ndarray]:
    """RK4-evolve GKX's linear operator; return ``(t, Re g_0)`` for the mode."""

    grid, rhs = _slab_setup(hermite, nu, te_over_ti)
    z = np.asarray(grid.z)
    weight = jnp.asarray(np.exp(-1j * z))

    state = (
        jnp.zeros(
            (1, 1, hermite, grid.ky.size, grid.kx.size, grid.z.size),
            dtype=jnp.complex128,
        )
        .at[0, 0, 0, 1, 0, :]
        .set(jnp.asarray(np.exp(1j * z)))
    )

    def project(s: jnp.ndarray) -> float:
        return float(
            jnp.real(jnp.sum(jnp.asarray(s)[0, 0, 0, 1, 0, :] * weight) / z.size)
        )

    steps = int(round(t_max / dt))
    signal = np.empty(steps + 1)
    signal[0] = project(state)
    for index in range(steps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        signal[index + 1] = project(state)

    return np.arange(steps + 1) * dt, signal


def operator_matrix(hermite: int, nu: float, te_over_ti: float) -> np.ndarray:
    """Extract GKX's linear operator for this mode as a dense Hermite matrix.

    Applies ``linear_rhs_cached`` to each Hermite basis vector, so the spectrum
    in panel (b) belongs to the production operator rather than a model of it.
    """

    grid, rhs = _slab_setup(hermite, nu, te_over_ti)
    z = np.asarray(grid.z)
    profile = jnp.asarray(np.exp(1j * z))
    weight = jnp.asarray(np.exp(-1j * z))

    base = jnp.zeros(
        (1, 1, hermite, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex128
    )
    matrix = np.zeros((hermite, hermite), dtype=complex)
    for column in range(hermite):
        response = rhs(base.at[0, 0, column, 1, 0, :].set(profile))
        matrix[:, column] = np.asarray(
            jnp.sum(jnp.asarray(response)[0, 0, :, 1, 0, :] * weight, axis=-1) / z.size
        )
    return matrix


def fit_standing_wave(
    times: np.ndarray,
    signal: np.ndarray,
    window: tuple[float, float],
    guess: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Fit ``A exp(gamma t) cos(omega t + phi)`` and return ``(gamma, omega)``."""

    mask = (times >= window[0]) & (times <= window[1])

    def model(t: np.ndarray, amp: float, gamma: float, omega: float, phase: float):
        return amp * np.exp(gamma * t) * np.cos(omega * t + phase)

    popt, _ = curve_fit(model, times[mask], signal[mask], p0=guess, maxfev=200_000)
    return float(popt[1]), float(popt[2])


def measure(te_over_ti: float, guess: complex, *, hermite: int) -> dict[str, object]:
    """Run the nu-scan through GKX and extrapolate to the collisionless root."""

    exact = exact_root(te_over_ti, guess)
    seed = (1.0, exact.imag, exact.real, 0.0)

    gammas, omegas = [], []
    for nu in _NU_SCAN:
        times, signal = evolve(
            hermite=hermite, nu=float(nu), te_over_ti=te_over_ti, t_max=14.0
        )
        gamma, omega = fit_standing_wave(times, signal, (2.0, 10.0), seed)
        gammas.append(gamma)
        omegas.append(omega)

    gammas = np.array(gammas)
    omegas = np.array(omegas)
    gamma_fit = np.polyfit(_NU_SCAN, gammas, 1)
    omega_fit = np.polyfit(_NU_SCAN, omegas, 1)

    return {
        "te_over_ti": te_over_ti,
        "exact_omega": exact.real,
        "exact_gamma": exact.imag,
        "gammas": gammas,
        "omegas": omegas,
        "gamma_extrapolated": float(gamma_fit[1]),
        "omega_extrapolated": float(omega_fit[1]),
        "gamma_error_percent": 100.0 * abs(gamma_fit[1] - exact.imag) / abs(exact.imag),
        "omega_error_percent": 100.0 * abs(omega_fit[1] - exact.real) / abs(exact.real),
    }


def build_figure(output: Path, *, hermite: int = 96) -> dict[str, object]:
    unity = measure(1.0, complex(1.4, -0.6), hermite=hermite)
    hot_electrons = measure(10.0, complex(2.6, -0.04), hermite=hermite)
    gamma_exact = unity["exact_gamma"]

    collisionless = np.linalg.eigvals(operator_matrix(128, 0.0, 1.0))
    collisional = np.linalg.eigvals(operator_matrix(128, 0.02, 1.0))
    real_residual = float(np.abs(collisionless.real).max())

    with figure_style():
        fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.3))

        # ---- (a) transient damping, ballistic plateau, recurrence ---------
        ax = axes[0]
        for nm, color in zip(
            (16, 64, 256),
            (GKX_COLORS["sky"], GKX_COLORS["blue"], GKX_COLORS["purple"]),
        ):
            times, signal = evolve(hermite=nm, nu=0.0, te_over_ti=1.0, t_max=40.0)
            ax.semilogy(times, np.abs(signal), color=color, linewidth=0.5, alpha=0.22)
            ax.semilogy(
                times,
                np.abs(hilbert(signal)),
                color=color,
                linewidth=1.8,
                label=f"$N_m={nm}$",
            )
            ax.axvline(
                2.0 * np.sqrt(nm), color=color, linestyle=":", linewidth=1.3, alpha=0.85
            )

        # The Landau pole dominates for the first ~2 decades; after that the
        # ballistic part of the initial condition leaves a plateau. Drawing the
        # reference past that would show a real crossover as disagreement.
        reference_t = np.linspace(0.0, 6.0, 64)
        ax.semilogy(
            reference_t,
            np.exp(gamma_exact * reference_t),
            label="exact Landau rate",
            **REFERENCE_STYLE,
        )
        ax.set_xlim(0.0, 40.0)
        ax.set_ylim(2e-3, 6.0)
        ax.set_xlabel(r"$t\,k_\parallel v_{ti}$")
        ax.set_ylabel(r"$|g_0|$  (envelope)")
        ax.set_title("Damping is a transient, ended by recurrence")
        ax.legend(loc="upper right", ncol=2, fontsize=9)
        annotate_reference(
            ax,
            r"dotted: $t_{\mathrm{rec}}=2\sqrt{N_m}/(k_\parallel v_{ti})$"
            "\nplateau = ballistic response; rise at $t_{\\mathrm{rec}}$ = recurrence",
            loc="lower left",
        )
        panel_label(ax, "(a)")

        # ---- (b) the collisionless spectrum is purely real ----------------
        ax = axes[1]
        for values, color, marker, filled, label in (
            (collisionless, GKX_COLORS["vermillion"], "o", False, r"$\nu=0$"),
            (
                collisional,
                GKX_COLORS["blue"],
                "^",
                True,
                r"$\nu/(k_\parallel v_{ti})=0.02$",
            ),
        ):
            ax.scatter(
                np.abs(values.imag),
                values.real,
                s=26 if not filled else 20,
                facecolors=color if filled else "none",
                edgecolors=color,
                linewidths=1.2,
                marker=marker,
                label=label,
                zorder=3 if filled else 4,
            )
        ax.axhline(0.0, **REFERENCE_STYLE)
        ax.plot(
            [unity["exact_omega"]],
            [gamma_exact],
            marker="*",
            markersize=17,
            color=GKX_COLORS["black"],
            linestyle="none",
            label="exact Landau root",
            zorder=5,
        )
        ax.set_xlim(0.0, 12.0)
        ax.set_ylim(-1.35, 0.3)
        ax.set_xlabel(r"$|\omega|/(k_\parallel v_{ti})$")
        ax.set_ylabel(r"$\gamma/(k_\parallel v_{ti})$")
        ax.set_title("The Landau root is not an eigenvalue")
        ax.legend(
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="#CCCCCC",
            framealpha=0.95,
            fontsize=9,
        )
        annotate_reference(
            ax,
            f"$\\nu=0$ spectrum real to {real_residual:.0e} (anti-Hermitian)"
            "\nno eigenvalue sits at the root",
            loc="upper left",
        )
        panel_label(ax, "(b)")

        # ---- (c) nu -> 0 extrapolation ------------------------------------
        # Shown as deviation from each case's own exact root: the two roots
        # differ by a factor 15, so a shared linear axis would flatten one.
        ax = axes[2]
        for case, color, marker, label in (
            (unity, GKX_COLORS["blue"], "o", r"$T_e/T_i=1$"),
            (hot_electrons, GKX_COLORS["green"], "s", r"$T_e/T_i=10$"),
        ):
            exact = case["exact_gamma"]
            deviation = 100.0 * (case["gammas"] - exact) / abs(exact)
            ax.plot(_NU_SCAN, deviation, marker, color=color, zorder=4, label=label)
            fit = np.polyfit(_NU_SCAN, deviation, 1)
            dense = np.linspace(0.0, _NU_SCAN.max() * 1.05, 32)
            ax.plot(
                dense,
                np.polyval(fit, dense),
                color=color,
                linewidth=1.4,
                alpha=0.6,
                zorder=3,
            )
            ax.plot(
                [0.0],
                [fit[1]],
                marker="*",
                markersize=16,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.9,
                linestyle="none",
                zorder=6,
            )
        ax.axhline(0.0, label="exact root", **REFERENCE_STYLE)
        ax.set_xlim(-0.0012, _NU_SCAN.max() * 1.05)
        ax.set_xlabel(r"$\nu/(k_\parallel v_{ti})$")
        ax.set_ylabel(r"deviation from exact $\gamma$  [%]")
        ax.set_title(r"Extrapolating $\nu\to0$ recovers the root")
        ax.legend(loc="upper left", fontsize=9)
        annotate_reference(
            ax,
            r"extrapolated error in $\gamma$ / in $\omega$"
            "\n"
            rf"$T_e/T_i=1$:  {unity['gamma_error_percent']:.3f}% /"
            rf" {unity['omega_error_percent']:.3f}%"
            "\n"
            rf"$T_e/T_i=10$: {hot_electrons['gamma_error_percent']:.3f}% /"
            rf" {hot_electrons['omega_error_percent']:.3f}%",
            loc="lower right",
        )
        panel_label(ax, "(c)")

        fig.suptitle(
            "Landau damping through GKX's linear operator   "
            r"$1 + T_i/T_e + \zeta Z(\zeta) = 0$",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, output, palette_colors=256)

    return {
        "unity": unity,
        "hot_electrons": hot_electrons,
        "collisionless_spectrum_max_real": real_residual,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/landau_damping_validation.png"),
    )
    parser.add_argument("--hermite", type=int, default=96)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    result = build_figure(args.output, hermite=args.hermite)
    for name in ("unity", "hot_electrons"):
        case = result[name]
        print(f"[{name}]  T_e/T_i = {case['te_over_ti']:g}")
        print(
            f"    omega  exact {case['exact_omega']:.9f}"
            f"  GKX {case['omega_extrapolated']:.9f}"
            f"  ({case['omega_error_percent']:.4f}%)"
        )
        print(
            f"    gamma  exact {case['exact_gamma']:.9f}"
            f"  GKX {case['gamma_extrapolated']:.9f}"
            f"  ({case['gamma_error_percent']:.4f}%)"
        )
    print(
        "collisionless spectrum max |Re| = "
        f"{result['collisionless_spectrum_max_real']:.3e}"
    )
    print(f"written: {args.output}")

    if args.summary is not None:
        payload = {
            name: (
                {
                    key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in case.items()
                }
                if isinstance(case, dict)
                else case
            )
            for name, case in result.items()
        }
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
