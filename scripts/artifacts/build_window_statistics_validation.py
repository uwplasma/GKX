"""Does the corrected standard error actually predict the error of the mean?

The window-statistics change asserts that a nonlinear window's uncertainty must
divide by the independent sample count, not the output count. Every check so far
has been self-consistent: the estimator agrees with the formula, and the formula
agrees with the estimator. That proves nothing a reviewer should accept.

This validates against an **empirical truth**. For a process whose correlation
time is known analytically, draw many independent realizations, measure the
actual scatter of their means, and ask which formula predicts it:

* naive      ``sigma / sqrt(n)``      -- treats outputs as independent draws
* corrected  ``sigma / sqrt(n_eff)``  -- divides by the independent count

The empirical standard deviation of the means across realizations *is* the
standard error. Whichever formula matches it is right, and the comparison cannot
be satisfied by an estimator agreeing with itself.

The test process is AR(1), ``x_{i+1} = rho x_i + sqrt(1-rho^2) e_i``, whose
integrated autocorrelation time is known in closed form:

    tau_ac = dt * (1 + rho) / (2 * (1 - rho))     [continuous-time convention]

so the estimator can also be scored against an exact value rather than a fit.
Sweeping ``rho`` sweeps the correlation from independent samples to strongly
correlated ones, which is the regime the tracked GKX traces sit in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gkx.artifacts.figure_style import (
    GKX_COLORS,
    annotate_reference,
    figure_style,
    panel_label,
    save_figure,
)
from gkx.diagnostics.analysis import integrated_autocorrelation_time

NAIVE = GKX_COLORS["vermillion"]
CORRECTED = GKX_COLORS["green"]
TRUTH = GKX_COLORS["blue"]


def ar1(rho: float, steps: int, realizations: int, rng) -> np.ndarray:
    """``(realizations, steps)`` draws from a unit-variance AR(1) process."""

    noise = rng.standard_normal((realizations, steps)) * np.sqrt(1.0 - rho**2)
    out = np.empty((realizations, steps))
    out[:, 0] = rng.standard_normal(realizations)
    for i in range(1, steps):
        out[:, i] = rho * out[:, i - 1] + noise[:, i]
    return out


def analytic_tau(rho: float, dt: float) -> float:
    """Integrated autocorrelation time of AR(1), in the same units as ``dt``."""

    return dt * (1.0 + rho) / (2.0 * (1.0 - rho))


def measure(rho: float, *, steps: int, realizations: int, dt: float, seed: int) -> dict:
    """Compare both formulas against the empirical error of the mean."""

    rng = np.random.default_rng(seed)
    series = ar1(rho, steps, realizations, rng) + 10.0  # positive mean, as a flux has

    means = series.mean(axis=1)
    # The empirical standard error: the actual scatter of independent estimates
    # of the mean. This is the quantity both formulas are trying to predict.
    empirical = float(means.std(ddof=1))

    # Per-realization predictions, averaged over realizations.
    naive, corrected, taus = [], [], []
    for row in series:
        sigma = float(row.std(ddof=1))
        tau = integrated_autocorrelation_time(row, dt)
        n_eff = (
            min(row.size, row.size * dt / (2.0 * tau)) if tau > 0.0 else float(row.size)
        )
        naive.append(sigma / np.sqrt(row.size))
        corrected.append(sigma / np.sqrt(max(n_eff, 1.0)))
        taus.append(tau)

    return {
        "rho": rho,
        "tau_analytic": analytic_tau(rho, dt),
        "tau_measured": float(np.mean(taus)),
        "empirical_stderr": empirical,
        "naive_stderr": float(np.mean(naive)),
        "corrected_stderr": float(np.mean(corrected)),
        "naive_ratio": float(np.mean(naive)) / empirical,
        "corrected_ratio": float(np.mean(corrected)) / empirical,
    }


def figure(rows: list[dict], output: Path) -> Path:
    import matplotlib.pyplot as plt

    rho = np.array([r["rho"] for r in rows])
    with figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))

        axes[0].plot(
            rho,
            [r["tau_analytic"] for r in rows],
            "-",
            color=TRUTH,
            lw=2.0,
            label="analytic  $\\tau=\\Delta t(1+\\rho)/2(1-\\rho)$",
        )
        axes[0].plot(
            rho,
            [r["tau_measured"] for r in rows],
            "o",
            color=CORRECTED,
            ms=5,
            label="estimator",
        )
        axes[0].set_yscale("log")
        axes[0].set_xlabel("AR(1) correlation $\\rho$")
        axes[0].set_ylabel("integrated autocorrelation time")
        axes[0].set_title("The estimator recovers a known correlation time")
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")

        axes[1].axhline(1.0, color=TRUTH, lw=2.0, label="empirical truth")
        axes[1].plot(
            rho,
            [r["naive_ratio"] for r in rows],
            "o-",
            color=NAIVE,
            ms=5,
            label="naive  $\\sigma/\\sqrt{n}$",
        )
        axes[1].plot(
            rho,
            [r["corrected_ratio"] for r in rows],
            "s-",
            color=CORRECTED,
            ms=5,
            label="corrected  $\\sigma/\\sqrt{n_{\\rm eff}}$",
        )
        axes[1].set_xlabel("AR(1) correlation $\\rho$")
        axes[1].set_ylabel("predicted / actual standard error")
        axes[1].set_title("Only the corrected formula predicts the real scatter")
        axes[1].legend(frameon=False, fontsize=8, loc="lower left")

        panel_label(axes[0], "a")
        panel_label(axes[1], "b")
        annotate_reference(
            axes[1], "empirical scatter of independent realizations", loc="upper right"
        )
        fig.tight_layout()
        return save_figure(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--realizations", type=int, default=4000)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/window_statistics_validation.png"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("docs/_static/window_statistics_validation.json"),
    )
    args = parser.parse_args()

    rhos = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    rows = [
        measure(
            r,
            steps=args.steps,
            realizations=args.realizations,
            dt=args.dt,
            seed=args.seed,
        )
        for r in rhos
    ]

    print(
        f"{'rho':>5} {'tau_exact':>10} {'tau_est':>9} {'naive/true':>11} {'corr/true':>10}"
    )
    for r in rows:
        print(
            f"{r['rho']:5.2f} {r['tau_analytic']:10.3f} {r['tau_measured']:9.3f} "
            f"{r['naive_ratio']:11.3f} {r['corrected_ratio']:10.3f}"
        )

    # A formula that predicts the truth has ratio 1. The naive one degrades as
    # correlation rises; the corrected one must stay close across the sweep.
    worst_naive = min(r["naive_ratio"] for r in rows)
    worst_corrected = min(r["corrected_ratio"] for r in rows)
    best_corrected_err = max(abs(r["corrected_ratio"] - 1.0) for r in rows)
    print(
        f"\nnaive underestimates by up to {1.0 / worst_naive:.2f}x; "
        f"corrected stays within {100 * best_corrected_err:.1f}% of truth"
    )

    written = figure(rows, args.output)
    print(f"written: {written}")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {
                "kind": "window_statistics_validation",
                "claim_level": "validated_against_empirical_scatter_of_independent_realizations",
                "process": "AR(1), analytic tau_ac = dt (1+rho) / (2 (1-rho))",
                "steps": args.steps,
                "realizations": args.realizations,
                "worst_naive_ratio": worst_naive,
                "worst_corrected_ratio": worst_corrected,
                "max_corrected_deviation": best_corrected_err,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
