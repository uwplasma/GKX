"""Recurrence and Hermite closure: why truncation reflects, and what fixes it.

Numerics anchor rather than a literature comparison: the quantities plotted here
are properties of the Hermite hierarchy itself, checked against their closed
forms.

* Recurrence time ``t_rec = 2 sqrt(N_m)/(k_par v_ti)`` -- the fitted exponent of
  the measured revival time against ``N_m`` is printed on the panel, so the
  square-root law is demonstrated rather than asserted.
* The reflectionless coefficient of Kanekar, Schekochihin, Dorland & Loureiro,
  J. Plasma Phys. 81, 305810104 (2015), Eq. (4.36),

      R_{M+1} = M/sqrt(2(M+1)) Gamma(M/2)/Gamma((M+1)/2) -> 1 - 1/(4M),

  whose ``M = 3`` value reproduces the Hammett-Perkins three-pole coefficient
  ``sqrt(8/pi)/sqrt(3)`` exactly. That coincidence is an independent check that
  the family is the right one.

The hypercollision comparison uses the GX Appendix-B normalization,
``nu_hyp = ln(10) f_hyp (p+1/2)/M^(p+1/2) |k| v_t`` with ``p = M/2`` applied for
``m > 2``. Without it the comparison is meaningless: an arbitrary ``nu`` made the
reflectionless closure look 21x better than it is when this was first measured.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gkx.artifacts.figure_style import (
    GKX_COLORS,
    REFERENCE_STYLE,
    annotate_reference,
    figure_style,
    panel_label,
    save_figure,
)
from gkx.terms.linear_terms import hermite_closure_coefficient

_RESOLUTIONS = (16, 32, 64, 128)


def _hypercollision_rate(hermite: int, *, f_hyper: float = 1.0) -> np.ndarray:
    """GX Appendix-B eq. (B12) hypercollision profile, applied for ``m > 2``."""

    order = hermite / 2.0
    m = np.arange(hermite, dtype=float)
    rate = (
        np.log(10.0)
        * f_hyper
        * (order + 0.5)
        / hermite ** (order + 0.5)
        * m ** (order + 0.5)
    )
    return np.where(m > 2, rate, 0.0)


def free_streaming_revival(
    hermite: int, closure: str, *, t_max: float | None = None, dt: float = 0.002
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve the free-streaming Hermite hierarchy from ``g_0 = 1``.

    ``k_par v_ti = 1``, so the recurrence time is ``2 sqrt(N_m)`` directly.
    """

    limit = t_max if t_max is not None else 3.2 * np.sqrt(hermite)
    m = np.arange(hermite, dtype=float)
    up = np.sqrt(m + 1.0)
    down = np.sqrt(m)

    damping = np.zeros(hermite)
    if closure == "hypercollisions":
        damping = _hypercollision_rate(hermite)
    tail = 0.0
    if closure == "reflectionless":
        tail = hermite_closure_coefficient(hermite) * np.sqrt(hermite)

    def rhs(state: np.ndarray) -> np.ndarray:
        out = np.zeros_like(state)
        out[:-1] += -1j * up[:-1] * state[1:]
        out[1:] += -1j * down[1:] * state[:-1]
        out -= damping * state
        if tail:
            # -R sqrt(M+1) v |k_par| G_M, confined to the last moment.
            out[-1] -= tail * state[-1]
        return out

    state = np.zeros(hermite, dtype=complex)
    state[0] = 1.0
    steps = int(round(limit / dt))
    times = np.arange(steps + 1) * dt
    amplitude = np.empty(steps + 1)
    amplitude[0] = 1.0
    for index in range(steps):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        amplitude[index + 1] = abs(state[0])
    return times, amplitude


def measure_revival(hermite: int, closure: str) -> tuple[float, float]:
    """Peak ``|g_0|`` in the recurrence window, and the time it occurs."""

    times, amplitude = free_streaming_revival(hermite, closure)
    predicted = 2.0 * np.sqrt(hermite)
    window = (times > 0.6 * predicted) & (times < 1.9 * predicted)
    peak = float(amplitude[window].max())
    when = float(times[window][np.argmax(amplitude[window])])
    return peak, when


def resolved_window_error(hermite: int, closure: str) -> float:
    """Max error in ``g_0`` before recurrence, against a converged reference.

    This is the metric that makes the comparison fair. Revival amplitude alone
    rewards any strong damping; a closure that flattens the whole hierarchy
    scores perfectly on it. Hypercollisions act on a BAND of moments (m > 2),
    so they can suppress the revival while also perturbing resolved physics,
    whereas the reflectionless condition touches only m = M. Measuring the
    resolved window against a reference charges for that.
    """

    horizon = 2.0 * np.sqrt(hermite) + 1.0
    _, reference = free_streaming_revival(1024, "truncation", t_max=horizon)
    _, trial = free_streaming_revival(hermite, closure, t_max=horizon)
    count = min(reference.size, trial.size)
    return float(np.abs(trial[:count] - reference[:count]).max())


def build_figure(output: Path) -> dict[str, object]:
    closures = ("truncation", "hypercollisions", "reflectionless")
    revivals = {
        closure: [measure_revival(nm, closure)[0] for nm in _RESOLUTIONS]
        for closure in closures
    }
    errors = {
        closure: [resolved_window_error(nm, closure) for nm in _RESOLUTIONS]
        for closure in closures
    }
    revival_times = [measure_revival(nm, "truncation")[1] for nm in _RESOLUTIONS]

    # Fit t_rec ~ N_m^alpha: alpha = 1/2 is the claim being demonstrated.
    exponent = float(np.polyfit(np.log(_RESOLUTIONS), np.log(revival_times), 1)[0])

    coefficients = {m: hermite_closure_coefficient(m) for m in (3, 8, 16, 32, 64, 128)}
    hammett_perkins = np.sqrt(8.0 / np.pi) / np.sqrt(3.0)

    with figure_style():
        fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.4))
        axes = axes.ravel()

        # ---- (a) the reflecting wall --------------------------------------
        ax = axes[0]
        for closure, color, style in (
            ("truncation", GKX_COLORS["vermillion"], "-"),
            ("hypercollisions", GKX_COLORS["green"], "-"),
            ("reflectionless", GKX_COLORS["blue"], "-"),
        ):
            times, amplitude = free_streaming_revival(64, closure)
            ax.semilogy(
                times, amplitude, style, color=color, linewidth=1.7, label=closure
            )
        reference_t, reference = free_streaming_revival(
            1024, "truncation", t_max=3.2 * np.sqrt(64)
        )
        ax.semilogy(
            reference_t, reference, label=r"converged ($N_m=1024$)", **REFERENCE_STYLE
        )
        ax.axvline(
            2.0 * np.sqrt(64),
            color=GKX_COLORS["grey"],
            linestyle=":",
            linewidth=1.4,
        )
        ax.set_xlim(0.0, 3.2 * np.sqrt(64))
        ax.set_ylim(1e-5, 3.0)
        ax.set_xlabel(r"$t\,k_\parallel v_{ti}$")
        ax.set_ylabel(r"$|g_0|$")
        ax.set_title(r"Free streaming at $N_m=64$")
        ax.legend(loc="upper left", fontsize=9)
        annotate_reference(
            ax,
            r"dotted: $t_{\mathrm{rec}}=2\sqrt{N_m}$"
            "\n"
            "truncation reflects the pulse back essentially intact",
            loc="lower left",
        )
        panel_label(ax, "(a)")

        # ---- (b) revival amplitude vs resolution --------------------------
        ax = axes[1]
        for closure, color, marker in (
            ("truncation", GKX_COLORS["vermillion"], "o"),
            ("hypercollisions", GKX_COLORS["green"], "s"),
            ("reflectionless", GKX_COLORS["blue"], "^"),
        ):
            ax.loglog(
                _RESOLUTIONS,
                revivals[closure],
                marker + "-",
                color=color,
                linewidth=1.6,
                label=closure,
            )
        ax.axhline(1.0, **REFERENCE_STYLE)
        ax.set_xticks(_RESOLUTIONS)
        ax.set_xticklabels([str(v) for v in _RESOLUTIONS])
        ax.set_xticks([], minor=True)
        ax.set_xlabel(r"$N_m$")
        ax.set_ylabel(r"revived $|g_0|$   (initial $=1$)")
        ax.set_title("Revival suppression")
        ax.legend(loc="center left", fontsize=9)
        annotate_reference(
            ax,
            "truncation recovers ~100% of the initial\n"
            "amplitude: the reflection is essentially perfect",
            loc="lower left",
        )
        panel_label(ax, "(b)")

        # ---- (c) fidelity on the resolved window --------------------------
        ax = axes[2]
        for closure, color, marker in (
            ("truncation", GKX_COLORS["vermillion"], "o"),
            ("hypercollisions", GKX_COLORS["green"], "s"),
            ("reflectionless", GKX_COLORS["blue"], "^"),
        ):
            ax.loglog(
                _RESOLUTIONS,
                errors[closure],
                marker + "-",
                color=color,
                linewidth=1.6,
                label=closure,
            )
        ax.set_xticks(_RESOLUTIONS)
        ax.set_xticklabels([str(v) for v in _RESOLUTIONS])
        ax.set_xticks([], minor=True)
        ax.set_xlabel(r"$N_m$")
        ax.set_ylabel(r"max $|g_0-g_0^{\rm ref}|$ before $t_{\rm rec}$")
        ax.set_title(r"Fidelity on the resolved window")
        ax.legend(loc="center left", fontsize=9)
        annotate_reference(
            ax,
            "the fair metric: revival alone rewards any strong damping.\n"
            "hypercollisions damp a band ($m>2$); the closure only $m=M$.",
            loc="lower left",
        )
        panel_label(ax, "(c)")

        # ---- (d) the closure coefficient ----------------------------------
        ax = axes[3]
        orders = np.arange(2, 160)
        exact = np.array([hermite_closure_coefficient(int(m)) for m in orders])
        ax.plot(
            orders, exact, color=GKX_COLORS["blue"], linewidth=2.0, label=r"$R_{M+1}$"
        )
        ax.plot(
            orders,
            1.0 - 1.0 / (4.0 * orders),
            label=r"asymptote $1-1/(4M)$",
            **REFERENCE_STYLE,
        )
        ax.plot(
            [3],
            [coefficients[3]],
            marker="*",
            markersize=17,
            color=GKX_COLORS["vermillion"],
            linestyle="none",
            zorder=5,
            label="Hammett-Perkins 3-pole",
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$M$")
        ax.set_ylabel(r"$R_{M+1}$")
        ax.set_ylim(0.55, 1.03)
        ax.set_title("Absorption becomes exact with resolution")
        ax.legend(loc="lower right", fontsize=9)
        annotate_reference(
            ax,
            rf"$R_4 = {coefficients[3]:.6f}$"
            "\n"
            rf"$\sqrt{{8/\pi}}/\sqrt{{3}} = {hammett_perkins:.6f}$"
            "\nequal to machine precision",
            loc="lower left",
        )
        panel_label(ax, "(d)")

        fig.suptitle(
            "Hermite recurrence: a reflecting wall, and two ways to absorb it   "
            rf"(measured $t_{{\mathrm{{rec}}}} \propto N_m^{{{exponent:.3f}}}$)",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, output)

    return {
        "resolutions": list(_RESOLUTIONS),
        "revivals": {k: [float(v) for v in vals] for k, vals in revivals.items()},
        "resolved_window_error": {
            k: [float(v) for v in vals] for k, vals in errors.items()
        },
        "recurrence_exponent": exponent,
        "closure_coefficients": {str(k): float(v) for k, v in coefficients.items()},
        "hammett_perkins": float(hammett_perkins),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/recurrence_hermite_closure.png"),
    )
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()

    summary = build_figure(args.output)
    print(
        f"measured t_rec exponent: {summary['recurrence_exponent']:.4f}  (expect 0.5)"
    )
    print(f"{'N_m':>6} {'truncation':>12} {'hypercoll':>12} {'reflectionless':>15}")
    for index, nm in enumerate(summary["resolutions"]):
        print(
            f"{nm:>6} {summary['revivals']['truncation'][index]:>12.4f}"
            f" {summary['revivals']['hypercollisions'][index]:>12.4f}"
            f" {summary['revivals']['reflectionless'][index]:>15.4f}"
        )
    print()
    print(f"{'N_m':>6} {'trunc err':>12} {'hyper err':>12} {'reflect err':>15}")
    for index, nm in enumerate(summary["resolutions"]):
        e = summary["resolved_window_error"]
        print(
            f"{nm:>6} {e['truncation'][index]:>12.3e}"
            f" {e['hypercollisions'][index]:>12.3e} {e['reflectionless'][index]:>15.3e}"
        )
    print(
        f"R_4 = {summary['closure_coefficients']['3']:.9f}  vs "
        f"Hammett-Perkins {summary['hammett_perkins']:.9f}"
    )
    print(f"written: {args.output}")

    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
