"""Do the quasilinear proxies actually predict nonlinear transport?

This audits the tracked ledger rather than fitting anything new, and it exists
because the answer decides whether quasilinear-guided optimization can work at
all.

Two tests, in increasing severity:

**Across devices** (12 nonlinear-validated windows spanning cyclone, HSX, W7-X,
ITER-model, Solovev, and several shaped tokamaks). Absolute flux is compared
across different normalizations, so scatter here is expected and only extreme
failure is meaningful.

**Within one family** (the RBC(1,1) boundary landscape: 24 replicated nonlinear
windows along a single-coefficient deformation, matched normalization, 1.6%
noise floor, 2.6x spread in heat flux). This is the controlled test, and it is
the one that matters for optimization: an objective only has to rank correctly
along the path an optimizer actually walks.

The controlled test is where the proxies fail hardest -- every one is
*anticorrelated* with the nonlinear flux, so minimizing them raises transport.
That is not a calibration error; a sign error cannot be fixed by a constant.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

_LEDGER = Path("docs/_static/quasilinear_stellarator_train_holdout_points.json")
_LANDSCAPE = Path("docs/_static/vmec_boundary_transport_landscape_rbc11_full.json")


def cross_device() -> dict[str, object]:
    """Correlate simple linear-spectrum proxies with flux across the ledger."""

    points = json.loads(_LEDGER.read_text())
    flux, proxies = [], {"mixing length": [], "peak growth": [], "linear weight": []}
    for point in points:
        rows = list(csv.DictReader(Path(point["quasilinear_artifact"]).open()))
        gamma = np.clip(np.array([float(r["gamma"]) for r in rows]), 0.0, None)
        kperp2 = np.maximum(np.array([float(r["kperp_eff2"]) for r in rows]), 1e-12)
        weight = np.array([float(r["heat_flux_weight_total"]) for r in rows])
        flux.append(point["observed_heat_flux"])
        proxies["mixing length"].append(float(np.sum(gamma / kperp2)))
        proxies["peak growth"].append(float(gamma.max()))
        proxies["linear weight"].append(float(np.sum(gamma * weight)))

    flux = np.array(flux)
    return {
        "n": len(flux),
        "flux_range": [float(flux.min()), float(flux.max())],
        "correlations": {
            name: {
                "pearson": float(pearsonr(np.array(values), flux)[0]),
                "spearman": float(spearmanr(np.array(values), flux)[0]),
            }
            for name, values in proxies.items()
        },
    }


def within_family() -> dict[str, object]:
    """Correlate the tracked reduced metrics with flux along the RBC(1,1) scan."""

    data = json.loads(_LANDSCAPE.read_text())
    rows = {round(r["coefficient_value"], 10): r for r in data["rows"]}
    records = []
    for entry in data["nonlinear_ensemble_points"]:
        if not entry.get("passed"):
            continue
        row = rows.get(round(entry["coefficient_value"], 10))
        if row is None:
            continue
        records.append((row["relative_fraction"], entry["mean"], entry["sem"],
                        row.get("reduced_metrics") or {}))
    records.sort()

    fraction = np.array([r[0] for r in records])
    flux = np.array([r[1] for r in records])
    sem = np.array([r[2] for r in records])

    correlations = {}
    for name in data["reduced_metric_kinds"]:
        values = np.array([r[3].get(name, np.nan) for r in records])
        if np.isnan(values).any() or np.allclose(values, values[0]):
            continue
        # Split-half robustness: a sign that survives both halves is structural,
        # not an artifact of one end of the scan.
        low, high = fraction < 0, fraction >= 0
        correlations[name] = {
            "pearson": float(pearsonr(values, flux)[0]),
            "spearman": float(spearmanr(values, flux)[0]),
            "spearman_lower_half": float(spearmanr(values[low], flux[low])[0]),
            "spearman_upper_half": float(spearmanr(values[high], flux[high])[0]),
        }

    return {
        "n": len(records),
        "flux_range": [float(flux.min()), float(flux.max())],
        "noise_floor_percent": float(100.0 * np.median(sem) / flux.mean()),
        # If flux tracked mere distance from the baseline, any monotone quantity
        # would appear correlated; it does not.
        "spearman_abs_fraction": float(spearmanr(np.abs(fraction), flux)[0]),
        "correlations": correlations,
        "series": {
            "fraction": fraction.tolist(),
            "flux": flux.tolist(),
            "flux_sem": sem.tolist(),
            "linear_weight": [
                float(r[3].get("quasilinear_flux_linear_weight", np.nan)) for r in records
            ],
        },
    }


def build_figure(summary: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    from gkx.artifacts.figure_style import (
        GKX_COLORS,
        annotate_reference,
        figure_style,
        panel_label,
        save_figure,
    )

    family = summary["within_family"]
    series = family["series"]
    fraction = np.array(series["fraction"])
    flux = np.array(series["flux"])
    sem = np.array(series["flux_sem"])
    proxy = np.array(series["linear_weight"])

    with figure_style():
        fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.4))

        ax = axes[0]
        ax.errorbar(fraction, flux, yerr=sem, marker="o", markersize=5,
                    color=GKX_COLORS["blue"], linewidth=1.6, capsize=2.5,
                    label="nonlinear heat flux")
        twin = ax.twinx()
        twin.plot(fraction, proxy, marker="s", markersize=4.5,
                  color=GKX_COLORS["orange"], linewidth=1.6, label="quasilinear proxy")
        twin.set_ylabel("quasilinear proxy", color=GKX_COLORS["orange"])
        twin.tick_params(axis="y", colors=GKX_COLORS["orange"])
        twin.grid(False)
        ax.set_xlabel("RBC(1,1) relative change")
        ax.set_ylabel(r"$Q_{\rm nl}$", color=GKX_COLORS["blue"])
        ax.tick_params(axis="y", colors=GKX_COLORS["blue"])
        ax.set_title("They move in opposite directions")
        annotate_reference(
            ax,
            "flux falls 2.6x as the proxy rises\n"
            f"noise floor {family['noise_floor_percent']:.1f}%",
            loc="upper right",
        )
        panel_label(ax, "(a)")

        ax = axes[1]
        ax.errorbar(proxy, flux, yerr=sem, fmt="o", markersize=6,
                    color=GKX_COLORS["blue"], capsize=2.5, linestyle="none")
        fit = np.polyfit(proxy, flux, 1)
        dense = np.linspace(proxy.min(), proxy.max(), 32)
        ax.plot(dense, np.polyval(fit, dense), color=GKX_COLORS["vermillion"],
                linewidth=1.6, alpha=0.75)
        rho = family["correlations"]["quasilinear_flux_linear_weight"]["spearman"]
        ax.set_xlabel("quasilinear proxy  (the objective being minimized)")
        ax.set_ylabel(r"$Q_{\rm nl}$  (what we actually want lower)")
        ax.set_title("Minimizing the proxy raises the flux")
        annotate_reference(
            ax,
            f"Spearman = {rho:+.3f}\n"
            "an optimizer following this walks the wrong way",
            loc="upper right",
        )
        panel_label(ax, "(b)")

        ax = axes[2]
        names, values = [], []
        for name, stats in family["correlations"].items():
            names.append(name.replace("quasilinear_flux_", "").replace("_", " "))
            values.append(stats["spearman"])
        for name, stats in summary["cross_device"]["correlations"].items():
            names.append(f"{name} (cross-device)")
            values.append(stats["spearman"])
        order = np.argsort(values)
        position = np.arange(len(names), dtype=float)
        colors = [
            GKX_COLORS["vermillion"] if values[i] < 0 else GKX_COLORS["green"]
            for i in order
        ]
        ax.barh(position, [values[i] for i in order], color=colors)
        ax.set_yticks(position)
        ax.set_yticklabels([names[i] for i in order], fontsize=8.5)
        ax.axvline(0.0, color=GKX_COLORS["black"], linewidth=1.2)
        ax.set_xlabel(r"Spearman correlation with $Q_{\rm nl}$")
        ax.set_title("No proxy has useful skill")
        ax.grid(axis="y", visible=False)
        annotate_reference(
            ax, "red = wrong sign\nan objective needs positive skill here",
            loc="lower left",
        )
        panel_label(ax, "(c)", dx=-0.42)

        fig.suptitle(
            "Quasilinear proxies are anticorrelated with nonlinear transport",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        save_figure(fig, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/_static/quasilinear_skill_audit.json")
    )
    parser.add_argument(
        "--figure", type=Path, default=Path("docs/_static/quasilinear_skill_audit.png")
    )
    args = parser.parse_args()

    summary = {"cross_device": cross_device(), "within_family": within_family()}

    family = summary["within_family"]
    print(f"within-family: n={family['n']}, flux {family['flux_range'][0]:.2f}"
          f"-{family['flux_range'][1]:.2f}, noise {family['noise_floor_percent']:.1f}%")
    print(f"{'metric':<48}{'Pearson':>9}{'Spearman':>10}{'lower':>8}{'upper':>8}")
    for name, stats in family["correlations"].items():
        print(f"{name:<48}{stats['pearson']:>9.3f}{stats['spearman']:>10.3f}"
              f"{stats['spearman_lower_half']:>8.3f}{stats['spearman_upper_half']:>8.3f}")
    print(f"\ncontrol -- Spearman(|fraction|, flux) = {family['spearman_abs_fraction']:+.3f}")
    print("\ncross-device:")
    for name, stats in summary["cross_device"]["correlations"].items():
        print(f"  {name:<20} Pearson {stats['pearson']:+.3f}  Spearman {stats['spearman']:+.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    build_figure(summary, args.figure)
    print(f"\nwritten: {args.output}\nfigure:  {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
