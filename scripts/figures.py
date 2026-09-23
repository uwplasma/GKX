"""Regenerate the README figures from tracked inputs or bounded runs.

    python scripts/figures.py                 # every figure in scripts/figures.toml
    python scripts/figures.py gx_defects      # one figure by name
    python scripts/figures.py --list          # names, outputs and inputs

Run from the repository root with ``PYTHONPATH=src:.`` (the ``proof_tests``
builder imports two generators under ``scripts/artifacts``) and
``JAX_ENABLE_X64=true``: the exact-identity residuals are float64 statements.
Each builder also writes ``<output>.json`` beside the PNG with every number it
plots, so the README can cite a machine-readable companion.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_suffix(".toml")

for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from gkx.artifacts.figure_style import (  # noqa: E402
    GKX_COLORS,
    figure_style,
    panel_label,
    save_figure,
)

REFERENCE = GKX_COLORS["black"]
GKX = GKX_COLORS["blue"]
ALT = GKX_COLORS["vermillion"]
MUTED = GKX_COLORS["grey"]


def _read_csv(path: str) -> list[dict[str, float]]:
    with (ROOT / path).open(encoding="utf-8", newline="") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _read_json(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _column(rows: list[dict[str, float]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows])


def _peak_relative_percent(ref: np.ndarray, gkx: np.ndarray) -> float:
    return float(100.0 * np.max(np.abs(ref - gkx)) / np.max(np.abs(ref)))


# --------------------------------------------------------------------- linear


def build_linear(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cyclone = _read_csv("docs/_static/cyclone_mismatch_table.csv")
    w7x = _read_csv("docs/_static/w7x_linear_t2_scan.csv")
    kbm = _read_csv("docs/_static/kbm_mismatch_table.csv")
    mode = _read_csv("docs/_static/reference_modes/w7x_linear_gkx_ky0p3000.csv")
    reference = np.load(
        ROOT
        / "docs/_static/comparison/reference_modes/w7x_linear_reference_ky0p3000.npz"
    )
    overlay = _read_json(
        "docs/_static/reference_modes/w7x_eigenfunction_reference_overlay_ky0p3000.json"
    )

    record: dict[str, Any] = {}
    with figure_style():
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.7), constrained_layout=True)

        def scan(
            ax,
            rows,
            keys,
            title,
            label,
            ref_label,
            where=(0.97, 0.04, "right", "bottom"),
            note="",
        ):
            ky = _column(rows, "ky")
            g_ref, g_gkx, w_ref, w_gkx = (_column(rows, key) for key in keys)
            ax.plot(ky, g_gkx, "-", color=GKX, label=r"GKX $\gamma$")
            ax.plot(
                ky, g_ref, "o", color=REFERENCE, mfc="none", mew=1.2, label=ref_label
            )
            ax.plot(ky, np.abs(w_gkx), "--", color=ALT, label=r"GKX $|\omega|$")
            ax.plot(
                ky, np.abs(w_ref), "s", color=REFERENCE, mfc="none", mew=1.2, ms=4.5
            )
            ax.set_xlabel(r"$k_y\rho_i$")
            ax.set_title(title)
            gamma_pct = _peak_relative_percent(g_ref, g_gkx)
            omega_pct = _peak_relative_percent(w_ref, w_gkx)
            x, y, ha, va = where
            ax.text(
                x,
                y,
                f"max diff / peak: {gamma_pct:.3g}% ($\\gamma$), {omega_pct:.3g}% ($\\omega$)"
                + (f"\n{note}" if note else ""),
                transform=ax.transAxes,
                ha=ha,
                va=va,
                fontsize=8.5,
            )
            record[label] = {"gamma_percent": gamma_pct, "omega_percent": omega_pct}
            return ax

        keys = ("gamma_ref", "gamma_gkx", "omega_ref", "omega_gkx")
        scan(
            axes[0],
            cyclone,
            keys,
            "Cyclone ITG, s-alpha",
            "cyclone_salpha",
            "GX golden",
            note="reference provisional: GX end-damping cap binds",
        )
        axes[0].set_ylabel(r"$\gamma,\ |\omega|$  [$v_{ti}/a$]")
        axes[0].legend(loc="upper left", fontsize=8.5)
        scan(
            axes[1],
            w7x,
            ("gamma_ref_last", "gamma_last", "omega_ref_last", "omega_last"),
            "W7-X ITG, adiabatic electrons",
            "w7x",
            "GX",
        )
        scan(
            axes[3],
            kbm,
            keys,
            r"KBM, Miller ($\phi$, $A_\parallel$)",
            "kbm",
            "GX golden",
            where=(0.03, 0.97, "left", "top"),
            note="reference provisional: GX end-damping cap binds",
        )

        ax = axes[2]
        z = _column(mode, "z")
        gkx_abs = _column(mode, "eigen_abs")
        ref_mode = np.asarray(reference["mode"])
        ref_theta = np.asarray(reference["theta"])
        ax.plot(
            ref_theta,
            np.abs(ref_mode) / np.abs(ref_mode).max(),
            color=REFERENCE,
            lw=3.2,
            alpha=0.35,
            label="GX",
        )
        ax.plot(z, gkx_abs / gkx_abs.max(), color=GKX, lw=1.4, label="GKX")
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$|\phi| / \max|\phi|$")
        ax.set_title(r"W7-X eigenfunction, $k_y\rho_i = 0.3$")
        ax.text(
            0.03,
            0.92,
            f"overlap {overlay['overlap']:.10f}\nrel. L2 {overlay['relative_l2']:.1e}",
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
        )
        ax.legend(loc="upper right", fontsize=8.5)
        record["w7x_eigenfunction"] = {
            "overlap": overlay["overlap"],
            "relative_l2": overlay["relative_l2"],
        }
        for index, ax in enumerate(axes):
            panel_label(ax, "abcd"[index])
        _save(fig, spec, config)
    return record


# ------------------------------------------------------------------ nonlinear


def build_nonlinear(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    gate = _read_json(
        "docs/_static/external_vmec_circular_replicates/circular_replicate_t700_ensemble_gate.json"
    )
    cases = [
        ("Cyclone", "nonlinear_cyclone_gate_summary.json"),
        ("Cyclone Miller", "nonlinear_cyclone_miller_gate_summary.json"),
        ("W7-X", "nonlinear_w7x_gate_summary.json"),
        ("HSX", "nonlinear_hsx_gate_summary.json"),
        ("KBM", "nonlinear_kbm_gate_summary.json"),
    ]
    metrics = ("HeatFlux", "Wg", "Wphi")
    record: dict[str, Any] = {"replicates": [], "gx_windows": {}}

    with figure_style():
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12.5, 3.9),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.35, 1.0]},
        )
        ax = axes[0]
        colors = (GKX, ALT, GKX_COLORS["green"])
        for color, row in zip(colors, gate["rows"]):
            trace = _read_csv(row["source_artifact"])
            summary = _read_json(row["summary_artifact"])
            t, q = _column(trace, "t"), _column(trace, "heat_flux")
            label = summary.get("variant_label", "").replace("dt0p", "dt 0.")
            ax.plot(
                t,
                q,
                color=color,
                lw=0.9,
                alpha=0.9,
                label=f"{label}: {row['late_mean']:.1f} ± {row['sem']:.1f}",
            )
            ax.hlines(
                row["late_mean"], summary["tmin"], summary["tmax"], color=color, lw=2.2
            )
            record["replicates"].append(
                {
                    "variant": label,
                    "late_mean": row["late_mean"],
                    "sem": row["sem"],
                    "window": [summary["tmin"], summary["tmax"]],
                }
            )
        window = gate["rows"][0]
        first = _read_json(window["summary_artifact"])
        ax.axvspan(first["tmin"], first["tmax"], color=MUTED, alpha=0.12, lw=0)
        spread = next(
            g["detail"] for g in gate["gates"] if g["metric"] == "mean_relative_spread"
        )
        record["replicate_spread"] = spread
        ax.set_xlabel(r"$t\ v_{ti}/a$")
        ax.set_ylabel(r"ion heat flux $Q_i$")
        ax.set_title("Circular tokamak from a VMEC wout: replicated heat flux")
        ax.legend(
            loc="upper left",
            fontsize=8.5,
            title="window mean ± SEM",
            title_fontsize=8.5,
        )

        ax = axes[1]
        width = 0.26
        x = np.arange(len(cases))
        for offset, metric, color in zip((-width, 0.0, width), metrics, colors):
            values = []
            for name, filename in cases:
                summary = _read_json(f"docs/_static/{filename}")
                entry = next(s for s in summary["summary"] if s["metric"] == metric)
                values.append(100.0 * entry["mean_rel_abs"])
                record["gx_windows"].setdefault(name, {})[metric] = entry[
                    "mean_rel_abs"
                ]
            label = {"HeatFlux": "heat flux", "Wg": r"$W_g$", "Wphi": r"$W_\phi$"}[
                metric
            ]
            ax.bar(x + offset, values, width, color=color, label=label)
        ax.axhline(10.0, color=REFERENCE, ls="--", lw=1.0)
        ax.text(len(cases) - 0.5, 10.3, "gate 10%", ha="right", fontsize=8.5)
        ax.set_xticks(x, [name for name, _ in cases], fontsize=9.5)
        ax.set_ylabel("mean |GKX − GX| / |GX|  [%]")
        ax.set_title("Nonlinear trajectories against self-run GX, same decks")
        ax.legend(loc="upper left", fontsize=8.5, ncols=3)
        ax.set_ylim(0, 14)
        panel_label(axes[0], "a")
        panel_label(axes[1], "b")
        _save(fig, spec, config)
    return record


# ---------------------------------------------------------------- proof tests


def _collision_identities() -> dict[str, float]:
    import jax.numpy as jnp

    from gkx.operators.linear.collisions import (
        assemble_drift_kinetic_improved_sugama_matrix,
        assemble_drift_kinetic_sugama_matrix,
        load_collision_moment_matrix,
    )

    one = jnp.asarray([1.0])
    matrices = {
        "sugama": np.asarray(assemble_drift_kinetic_sugama_matrix(one, one, one))[0, 0],
        "improved_sugama": np.asarray(
            assemble_drift_kinetic_improved_sugama_matrix(one, one, one)
        )[0, 0],
        "coulomb": np.asarray(load_collision_moment_matrix("coulomb")),
    }
    basis = np.eye(8)
    invariants = (basis[0], basis[2], basis[1] + basis[4] / np.sqrt(2.0))
    conservation = max(
        float(np.abs(functional @ matrix).max())
        for matrix in matrices.values()
        for functional in invariants
    )
    symmetry = max(
        float(np.abs(m - m.T).max() / np.abs(m).max()) for m in matrices.values()
    )
    h_theorem = max(
        float(np.linalg.eigvalsh(0.5 * (m + m.T)).max()) for m in matrices.values()
    )
    # Frei, Ernst & Ricci (2022) Eqs. (C9a)-(C9f), like-species Coulomb; GKX stores
    # the opposite Laguerre sign, so entries map through (-1)^(j + j').
    s2, s1, s3 = np.sqrt(2 / np.pi), np.sqrt(1 / np.pi), np.sqrt(1 / (3 * np.pi))
    index = {(2, 0): 4, (0, 1): 1, (3, 0): 6, (1, 1): 3}
    published = {
        ((2, 0), (2, 0)): -(16 / 15) * s2,
        ((2, 0), (0, 1)): -(16 / 15) * s1,
        ((0, 1), (0, 1)): -(8 / 15) * s2,
        ((3, 0), (3, 0)): -(8 / 5) * s2,
        ((3, 0), (1, 1)): -(8 / 5) * s3,
        ((1, 1), (1, 1)): -(28 / 15) * s2,
    }
    coefficient = max(
        abs(
            float(matrices["coulomb"][index[a], index[b]])
            - (-1.0) ** (a[1] + b[1]) * value
        )
        for (a, b), value in published.items()
    )
    return {
        "conservation": conservation,
        "self_adjointness": symmetry,
        "h_theorem": max(h_theorem, 0.0),
        "published_coefficients": coefficient,
    }


def _laguerre_identities() -> dict[str, float]:
    from gkx.core_velocity import laguerre_quadrature_count, laguerre_transform

    round_trip = 0.0
    for nl in (4, 8, 16, 32, 64):
        to_grid, to_spectral, _ = laguerre_transform(nl)
        round_trip = max(
            round_trip, float(np.abs(to_grid @ to_spectral - np.eye(nl)).max())
        )
    to_grid, to_spectral, roots = laguerre_transform(16)
    nj = laguerre_quadrature_count(16)
    weights = to_spectral[:, 0] * np.exp(-0.5 * roots)
    moment, factorial = 0.0, 1.0
    for k in range(min(2 * nj, 12)):
        factorial *= max(k, 1)
        moment = max(moment, abs(float(np.sum(weights * roots**k)) / factorial - 1.0))
    return {"laguerre_round_trip": round_trip, "gauss_laguerre_moments": moment}


def _spitzer_harm() -> dict[str, float]:
    import jax.numpy as jnp

    from gkx.operators.linear.collisions import solve_driven_collision_response
    from scripts.artifacts.build_linear_validation_artifacts import (
        coulomb_drift_kinetic_moment_matrices,
    )

    hermite, laguerre, digits = 7, 2, 40
    self_test, self_field = (
        np.asarray(m, dtype=float)
        for m in coulomb_drift_kinetic_moment_matrices(
            hermite, laguerre, 1.0, 1.0, digits=digits
        )[:2]
    )
    ion = np.asarray(
        coulomb_drift_kinetic_moment_matrices(
            hermite, laguerre, 1.0e-12, 1.0, digits=digits
        )[0],
        dtype=float,
    )

    def flow(matrix: np.ndarray) -> float:
        momentum = laguerre + 1
        source = np.zeros(matrix.shape[0])
        source[momentum] = np.sqrt(2.0)
        active = tuple(range(1, matrix.shape[0]))
        moments = np.asarray(
            solve_driven_collision_response(
                jnp.asarray(matrix), jnp.asarray(-source), active_modes=active
            )
        )
        return float(moments[momentum])

    # Spitzer & Harm, Phys. Rev. 89, 977 (1953): gamma_E(Z).
    tabulated = {1: 0.5816, 2: 0.6833, 4: 0.7849, 16: 0.9225}
    errors = {}
    for charge, value in tabulated.items():
        ratio = flow(charge * ion + self_test + self_field) / flow(charge * ion)
        errors[f"Z={charge}"] = 100.0 * abs(ratio - value) / value
    return errors


def _landau() -> dict[str, float]:
    from scripts.artifacts.build_landau_damping_figure import measure, operator_matrix

    spectrum = np.linalg.eigvals(operator_matrix(64, 0.0, 1.0))
    result = {"collisionless_real_part": float(np.abs(spectrum.real).max())}
    for te_over_ti, guess in ((1.0, complex(1.4, -0.6)), (10.0, complex(2.6, -0.04))):
        measured = measure(te_over_ti, guess, hermite=96)
        tag = f"Te/Ti={te_over_ti:g}"
        result[f"{tag} gamma"] = float(measured["gamma_error_percent"])
        result[f"{tag} omega"] = float(measured["omega_error_percent"])
    return result


def build_proof_tests(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("JAX_ENABLE_X64", "").lower() not in {"1", "true"}:
        raise SystemExit("proof_tests needs JAX_ENABLE_X64=true")
    identities = {**_collision_identities(), **_laguerre_identities()}
    landau = _landau()
    identities["collisionless_real_part"] = landau.pop("collisionless_real_part")
    spitzer = _spitzer_harm()

    # (label, measured, gate, test file)
    exact_rows = [
        ("collision invariants (n, u, E)", identities["conservation"], 1e-12),
        ("collision self-adjointness", identities["self_adjointness"], 1e-12),
        ("H-theorem: max eig of sym. part", identities["h_theorem"], 1e-12),
        ("Coulomb vs published (C9a-f)", identities["published_coefficients"], 1e-10),
        (
            "collisionless Hermite: max |Re λ|",
            identities["collisionless_real_part"],
            1e-11,
        ),
        ("Laguerre transform round trip", identities["laguerre_round_trip"], 1e-10),
        ("Gauss-Laguerre moments k ≤ 11", identities["gauss_laguerre_moments"], 1e-10),
    ]
    limit_rows = [
        (r"Landau $T_e/T_i=1$, $\gamma$", landau["Te/Ti=1 gamma"], 1.0),
        (r"Landau $T_e/T_i=1$, $\omega$", landau["Te/Ti=1 omega"], 0.5),
        (r"Landau $T_e/T_i=10$, $\gamma$", landau["Te/Ti=10 gamma"], 1.0),
        (r"Landau $T_e/T_i=10$, $\omega$", landau["Te/Ti=10 omega"], 0.5),
    ] + [
        (f"Spitzer-Härm $\\gamma_E$, {key}", value, 1.5)
        for key, value in spitzer.items()
    ]

    floor = 1e-18
    with figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.1), constrained_layout=True)
        for ax, rows, xlabel, title in (
            (
                axes[0],
                exact_rows,
                "measured residual (absolute or relative)",
                "Exact identities, float64",
            ),
            (axes[1], limit_rows, "relative error [%]", "Analytic limits"),
        ):
            y = np.arange(len(rows))[::-1]
            values = [max(row[1], floor) for row in rows]
            ax.barh(y, values, color=GKX, height=0.6, label="measured")
            ax.scatter(
                [row[2] for row in rows],
                y,
                marker="|",
                s=260,
                color=ALT,
                linewidths=2.2,
                label="test tolerance",
                zorder=3,
            )
            ax.set_yticks(y, [row[0] for row in rows], fontsize=9.5)
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_title(title)
        axes[0].set_xlim(floor, 1e-8)
        axes[1].set_xlim(1e-4, 3.0)
        axes[0].legend(loc="upper right", fontsize=8.5)
        panel_label(axes[0], "a")
        panel_label(axes[1], "b")
        _save(fig, spec, config)
    return {
        "exact_identities": identities,
        "landau_percent": landau,
        "spitzer_harm_percent": spitzer,
    }


# ----------------------------------------------------------------- GX defects


def build_gx_defects(spec: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    measured = config["measured"]
    clamp = measured["gx_clamp"]
    growth = measured["gx_clamp_growth"]
    hyper = measured["gx_hyper_overflow"]
    cap = int(clamp["launch_cap"])

    record: dict[str, Any] = {
        "sources": {
            "clamp": clamp["source"],
            "growth": [growth["gx_clamped_source"], growth["gx_repaired_source"]],
            "kernel": measured["gx_clamp_kernel"]["source"],
            "hyper": hyper["source"],
        }
    }
    with figure_style():
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.9), constrained_layout=True)

        # (a) share of the (z, l, m) index space dampEnds_linked never visits.
        ax = axes[0]
        count = np.logspace(4, 6, 400)
        ax.plot(count, 100.0 * np.clip(1.0 - cap / count, 0.0, None), color=REFERENCE)
        ax.axvline(cap, color=MUTED, ls=":", lw=1.0)
        record["undamped_percent"] = {}
        for marker, entry in zip(("o", "s", "D"), clamp["goldens"]):
            n = int(entry["count"])
            share = 100.0 * max(0.0, 1.0 - cap / n)
            ax.plot(
                n,
                share,
                marker,
                color=ALT,
                ms=7,
                label=f"{entry['label']}: {share:.1f}%",
            )
            record["undamped_percent"][entry["label"]] = share
        ax.set_xscale("log")
        ax.set_xlabel(r"$N_z N_l N_m$")
        ax.set_ylabel("indices without end damping [%]")
        ax.set_title("GX linked end damping: launch cap 65,535")
        ax.legend(loc="upper left", fontsize=8)

        # (b) the growth rate that clamp produces, before and after the repair.
        ax = axes[1]
        labels = ["GX, shipped\ndampEnds_linked", "GX, grid-stride\nloop added", "GKX"]
        gammas = [
            growth["gx_clamped_gamma"],
            growth["gx_repaired_gamma"],
            growth["gkx_gamma"],
        ]
        ax.bar(labels, gammas, color=[MUTED, REFERENCE, GKX], width=0.6)
        for index, value in enumerate(gammas):
            diff = 100.0 * (growth["gkx_gamma"] - value) / value
            text = f"{value:.5f}" + ("" if index == 2 else f"\nGKX {diff:+.2f}%")
            ax.text(index, value * 1.002, text, ha="center", va="bottom", fontsize=8.5)
        ax.set_ylim(0.0225, 0.0258)
        ax.set_ylabel(r"$\gamma$  [$v_{ti}/a$]")
        ax.set_title(r"Cyclone $k_y\rho_i=0.55$, $N_l=32$, $N_m=96$")
        record["cyclone_ky055"] = {
            "gamma": dict(zip(("gx_clamped", "gx_repaired", "gkx"), gammas)),
            "gkx_minus_gx_percent": {
                "clamped": 100.0 * (gammas[2] - gammas[0]) / gammas[0],
                "repaired": 100.0 * (gammas[2] - gammas[1]) / gammas[1],
            },
        }

        # (c) GX's float32 kz-hypercollision coefficient at the top moment.
        ax = axes[2]
        p = np.float32(hyper["p_hyper_m"])
        nm = np.arange(int(hyper["nm_range"][0]), int(hyper["nm_range"][1]) + 1)
        gx_values, gkx_values = [], []
        with np.errstate(over="ignore", invalid="ignore"):
            for n in nm:
                big_m = np.float32(n - 1)
                prefactor = np.float32(p + 0.5) / np.power(big_m, np.float32(p + 0.5))
                gx_values.append(float(prefactor * np.power(big_m, p)))
                gkx_values.append(float((p + 0.5) / np.sqrt(np.float64(big_m))))
        gx_values_arr = np.asarray(gx_values)
        gkx_values_arr = np.asarray(gkx_values)
        finite = np.isfinite(gx_values_arr) & (gx_values_arr > 0)
        zero = gx_values_arr == 0.0
        nan = ~np.isfinite(gx_values_arr)
        ax.plot(nm, gkx_values_arr, color=GKX, label=r"GKX: $(p+1/2)\,M^{-1/2}(m/M)^p$")
        ax.plot(
            nm[finite],
            gx_values_arr[finite],
            "o",
            color=REFERENCE,
            ms=4,
            mfc="none",
            mew=1.0,
            label="GX float32, finite",
        )
        ax.plot(
            nm[zero],
            np.full(zero.sum(), 0.0),
            "v",
            color=ALT,
            ms=6,
            clip_on=False,
            label="GX = 0: hypercollisions off",
        )
        ax.plot(
            nm[nan],
            np.full(nan.sum(), 0.5),
            "x",
            color=ALT,
            ms=6,
            mew=1.5,
            label="GX = NaN: run fails",
        )
        ax.set_xlabel(r"$N_m$  ($p = 20$, top moment $m = M = N_m - 1$)")
        ax.set_ylabel("coefficient / (ν 2.3 v_t |∇∥|)")
        ax.set_title("GX kz hypercollisions in float32")
        ax.set_ylim(0.0, 3.6)
        ax.legend(loc="upper right", fontsize=8)
        record["hyper_overflow"] = {
            "p_hyper_m": int(p),
            "last_finite_nm": int(nm[finite].max()),
            "zero_nm": [int(nm[zero].min()), int(nm[zero].max())] if zero.any() else [],
            "first_nan_nm": int(nm[nan].min()) if nan.any() else None,
        }
        for index, ax in enumerate(axes):
            panel_label(ax, "abc"[index])
        _save(fig, spec, config)
    return record


# ---------------------------------------------------------------------- driver

BUILDERS: dict[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]] = {
    "linear": build_linear,
    "nonlinear": build_nonlinear,
    "proof_tests": build_proof_tests,
    "gx_defects": build_gx_defects,
}


def _output(spec: dict[str, Any], config: dict[str, Any]) -> Path:
    return ROOT / config["output_dir"] / spec["output"]


def _save(fig: plt.Figure, spec: dict[str, Any], config: dict[str, Any]) -> None:
    save_figure(
        fig, _output(spec, config), palette_colors=int(config["palette_colors"])
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="figure names; default all")
    parser.add_argument("--list", action="store_true", help="list figures and exit")
    args = parser.parse_args(argv)

    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    figures = {spec["name"]: spec for spec in config["figure"]}
    if args.list:
        for spec in figures.values():
            print(f"{spec['name']:12s} {config['output_dir']}/{spec['output']}")
            for path in spec["inputs"]:
                print(f"{'':12s}   <- {path}")
        return 0
    unknown = sorted(set(args.names) - set(figures))
    if unknown:
        parser.error(f"unknown figure(s): {unknown}; choose from {sorted(figures)}")
    for name in args.names or list(figures):
        spec = figures[name]
        missing = [path for path in spec["inputs"] if not (ROOT / path).exists()]
        if missing:
            raise SystemExit(f"{name}: missing tracked inputs {missing}")
        record = BUILDERS[spec["builder"]](spec, config)
        output = _output(spec, config)
        companion = output.with_suffix(".json")
        companion.write_text(
            json.dumps(
                {"figure": spec["output"], "inputs": spec["inputs"], **record},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"{name}: wrote {output.relative_to(ROOT)} "
            f"({output.stat().st_size / 1024:.0f} KiB)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
