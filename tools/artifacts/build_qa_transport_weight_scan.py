"""QA optimization: how hard must you push turbulence before it moves?

The shipped QA objective weights the quasilinear heat-flux proxy at 0.01 against
1.0 for quasisymmetry and 10.0 for iota. Measured at the seed, that makes the
transport term **0.00%** of the total sum of squares -- iota is 94.4%, aspect
5.4%, quasisymmetry 0.2%. The optimizer therefore never sees the turbulence, and
the resulting designs show no significant transport reduction. That is not a
null physics result; it is a weighting artifact.

This scans the transport weight and records where each optimization lands in the
(quasisymmetry, heat flux) plane, which is the honest way to present the
trade-off: reducing turbulence costs quasisymmetry, and the question is only how
much of one buys how much of the other.

The transport term is normalized by its seed value, so the weight means "how
much do I care about turbulence relative to quasisymmetry" rather than depending
on the arbitrary scale of the proxy.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from pathlib import Path as pathlib_Path
import time

import numpy as np

# Seed perturbation and targets follow examples/optimization/QA_optimization_*.
_SEED_PERTURBATION = 0.01
_QS_SURFACES = np.linspace(0.1, 1.0, 10)
_HELICITY = (1, 0)
_ASPECT_TARGET = 6.0
_IOTA_TARGET = 0.42

# Flux-tube sampling for the quasilinear proxy.
_PROXY = {
    "s_index": 7,
    "alpha": 0.0,
    "ntheta": 24,
    "selected_ky_index": 1,
    "n_laguerre": 2,
    "n_hermite": 3,
    "r_over_lt": 6.9,
    "r_over_ln": 2.2,
}


def _build_seed():
    import vmex as vj

    path = (
        Path(vj.__file__).resolve().parents[1] / "examples/data/input.minimal_seed_nfp2"
    )
    inp = vj.VmecInput.from_file(path)
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor + 1, 1] += _SEED_PERTURBATION
    zbs[inp.ntor + 1, 1] += _SEED_PERTURBATION
    return dataclasses.replace(inp, rbc=rbc, zbs=zbs)


def _metrics(equilibrium, qs, proxy) -> dict[str, float]:
    from vmex import optimize as opt

    return {
        "qs": float(qs.total(equilibrium)),
        "aspect": float(opt.aspect_ratio(equilibrium.state, equilibrium.runtime)),
        "iota": float(opt.mean_iota(equilibrium.state, equilibrium.runtime)),
        "flux": float(proxy(equilibrium.state, equilibrium.runtime)),
    }


def run_scan(
    weights: list[float], *, max_modes: tuple[int, ...], max_nfev: int
) -> dict[str, object]:
    from vmex import optimize as opt
    from vmex.core import turbulence as turb

    inp = _build_seed()
    qs = opt.QuasisymmetryRatioResidual(_QS_SURFACES, *_HELICITY)

    def proxy(state, runtime):
        return turb.quasilinear_flux_proxy(state, runtime, **_PROXY)

    seed_eq = opt.solve_equilibrium(inp)
    seed = _metrics(seed_eq, qs, proxy)
    print(
        f"seed: QS={seed['qs']:.4e} aspect={seed['aspect']:.3f} "
        f"iota={seed['iota']:.4f} Q_QL={seed['flux']:.4e}",
        flush=True,
    )

    # Normalizing by the seed flux makes the weight mean "priority relative to
    # quasisymmetry" instead of inheriting the proxy's arbitrary scale.
    flux_scale = max(abs(seed["flux"]), 1.0e-12)

    def normalized_flux(state, runtime):
        return proxy(state, runtime) / flux_scale

    results = []
    for weight in weights:
        terms = [
            (qs, 0.0, 1.0),
            (opt.aspect_ratio, _ASPECT_TARGET, 1.0),
            (opt.mean_iota, _IOTA_TARGET, 10.0),
            (normalized_flux, 0.0, float(weight)),
        ]
        started = time.time()
        current = inp
        for max_mode in max_modes:
            outcome = opt.least_squares(
                terms,
                current,
                max_mode=max_mode,
                jac=None,
                use_ess=True,
                verbose=0,
                max_nfev=max_nfev,
                ftol=1.0e-6,
                xtol=1.0e-10,
            )
            current = getattr(outcome, "input", getattr(outcome, "x", current))
        final = _metrics(opt.solve_equilibrium(current), qs, proxy)
        final["weight"] = float(weight)
        final["seconds"] = time.time() - started
        final["flux_ratio"] = final["flux"] / seed["flux"]
        final["qs_ratio"] = final["qs"] / seed["qs"]
        results.append(final)
        print(
            f"w={weight:<6g} QS={final['qs']:.4e} ({final['qs_ratio']:.2f}x)  "
            f"Q_QL={final['flux']:.4e} ({final['flux_ratio']:.2f}x)  "
            f"aspect={final['aspect']:.3f} iota={final['iota']:.4f}  "
            f"[{final['seconds']:.0f}s]",
            flush=True,
        )

    return {"seed": seed, "results": results, "max_modes": list(max_modes)}


def build_figure(summary: dict, output: pathlib_Path) -> None:
    """Draw the quasisymmetry / heat-flux trade-off the scan traced out."""

    import matplotlib.pyplot as plt

    from gkx.artifacts.figure_style import (
        GKX_COLORS,
        annotate_reference,
        figure_style,
        panel_label,
        save_figure,
    )

    seed = summary["seed"]
    rows = summary["results"]
    weights = [r["weight"] for r in rows]
    qs_ratio = [r["qs_ratio"] for r in rows]
    flux_ratio = [r["flux_ratio"] for r in rows]
    # A run that never reached the iota target did not optimize; it stalled, and
    # plotting it as a design point would misread a solver failure as a physics
    # trade-off.
    converged = [abs(r["iota"] - _IOTA_TARGET) < 0.1 for r in rows]

    with figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

        ax = axes[0]
        for x, y, w, ok in zip(qs_ratio, flux_ratio, weights, converged):
            color = GKX_COLORS["blue"] if ok else GKX_COLORS["grey"]
            ax.plot(
                [x],
                [y],
                "o" if ok else "x",
                color=color,
                markersize=11,
                markeredgewidth=2.0,
                linestyle="none",
            )
            ax.annotate(
                f"w={w:g}" + ("" if ok else "  (stalled)"),
                (x, y),
                textcoords="offset points",
                xytext=(9, 5),
                fontsize=9,
                color=color,
            )
        good = [(x, y) for x, y, ok in zip(qs_ratio, flux_ratio, converged) if ok]
        good.sort()
        ax.plot(
            [g[0] for g in good],
            [g[1] for g in good],
            color=GKX_COLORS["blue"],
            linewidth=1.3,
            alpha=0.5,
            zorder=1,
        )
        ax.plot(
            [1.0],
            [1.0],
            marker="*",
            markersize=17,
            color=GKX_COLORS["black"],
            linestyle="none",
            zorder=5,
        )
        ax.annotate(
            "seed", (1.0, 1.0), textcoords="offset points", xytext=(-34, -4), fontsize=9
        )
        ax.set_xlabel("quasisymmetry residual / seed")
        ax.set_ylabel("quasilinear heat flux / seed")
        ax.set_title("Trading quasisymmetry for turbulence")
        annotate_reference(
            ax,
            "down = less turbulence, right = worse quasisymmetry\n"
            "both axes below 1: better than the seed on both",
            loc="lower right",
        )
        panel_label(ax, "(a)")

        ax = axes[1]
        index = np.arange(len(rows), dtype=float)
        ax.bar(
            index - 0.2,
            qs_ratio,
            0.4,
            label="quasisymmetry",
            color=GKX_COLORS["orange"],
        )
        ax.bar(
            index + 0.2, flux_ratio, 0.4, label="heat flux", color=GKX_COLORS["blue"]
        )
        ax.axhline(
            1.0,
            color=GKX_COLORS["black"],
            linestyle=(0, (6, 3)),
            linewidth=1.5,
            alpha=0.85,
            zorder=1,
        )
        ax.set_xticks(index)
        ax.set_xticklabels(
            [
                f"{w:g}" + ("" if ok else "\nstalled")
                for w, ok in zip(weights, converged)
            ]
        )
        ax.set_xlabel("transport weight")
        ax.set_ylabel("value / seed")
        ax.set_title("Both metrics, relative to the seed")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(axis="x", visible=False)
        annotate_reference(
            ax,
            f"seed: QS={seed['qs']:.3e}, $Q_{{QL}}$={seed['flux']:.3e}",
            loc="upper right",
        )
        panel_label(ax, "(b)")

        fig.suptitle(
            "QA optimization: the shipped weight makes turbulence 0.00% of the objective",
            fontsize=12.5,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.01, 0.5, 2.0, 8.0],
        help="transport weights to scan; 0.01 reproduces the shipped default",
    )
    parser.add_argument("--max-modes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-nfev", type=int, default=150)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/qa_transport_weight_scan.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/_static/qa_transport_weight_scan.png"),
    )
    parser.add_argument("--from-json", type=Path, default=None)
    args = parser.parse_args()

    if args.from_json is not None:
        summary = json.loads(args.from_json.read_text())
        build_figure(summary, args.figure)
        print(f"figure:  {args.figure}")
        return 0

    summary = run_scan(
        list(args.weights),
        max_modes=tuple(args.max_modes),
        max_nfev=int(args.max_nfev),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"written: {args.output}")
    if args.figure is not None:
        build_figure(summary, args.figure)
        print(f"figure:  {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
