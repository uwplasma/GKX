#!/usr/bin/env python3
"""Build a bounded nonlinear startup-response finite-difference audit.

This tool intentionally audits a narrow plumbing path: a real GKX
nonlinear Cyclone runtime is run at ``a/LTi = base +/- step`` plus a repeated
base point, but only for a compact startup window.  The output checks that this
startup-window heat-flux response is finite, repeatable, conditioned, and has a
resolved central finite-difference response.  It is not a transport-average,
VMEC/Boozer optimized-equilibrium, or nonlinear heat-flux optimization claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gkx.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)  # noqa: E402
from gkx.artifacts.plotting import set_plot_style  # noqa: E402
from gkx.runtime import run_runtime_nonlinear  # noqa: E402
from gkx.workflows.runtime.config import (  # noqa: E402
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "docs" / "_static" / "nonlinear_window_fd_audit.png"
DEFAULT_TRANSPORT_MIN_TOTAL_TIME = 50.0
DEFAULT_TRANSPORT_TRANSIENT_TIME = 20.0
DEFAULT_TRANSPORT_MIN_POST_TRANSIENT_SAMPLES = 64
DEFAULT_TRANSPORT_MAX_RUNNING_MEAN_REL_CHANGE = 0.05
DEFAULT_TRANSPORT_MIN_ABS_MEAN_HEAT_FLUX = 1.0e-6


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def cyclone_runtime_config(
    *,
    tprim: float,
    fprim: float = 0.8,
    random_seed: int = 22,
    init_amp: float = 1.0e-4,
    nx: int = 8,
    ny: int = 8,
    nz: int = 12,
    dt: float = 0.01,
) -> RuntimeConfig:
    """Return the compact nonlinear Cyclone startup-audit configuration."""

    return RuntimeConfig(
        grid=GridConfig(
            Nx=int(nx), Ny=int(ny), Nz=int(nz), Lx=20.0, Ly=20.0, boundary="periodic"
        ),
        time=TimeConfig(
            t_max=float(dt),
            dt=float(dt),
            method="rk2",
            use_diffrax=False,
            fixed_dt=True,
            sample_stride=1,
            diagnostics_stride=1,
        ),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(
            init_field="density",
            init_amp=float(init_amp),
            gaussian_init=False,
            random_seed=int(random_seed),
            init_single=False,
        ),
        species=(
            RuntimeSpeciesConfig(name="ion", tprim=float(tprim), fprim=float(fprim)),
        ),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="rho_star"
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )


def late_window_metrics(
    time: np.ndarray,
    heat_flux: np.ndarray,
    *,
    tail_fraction: float,
) -> dict[str, float | int]:
    """Return scalar late-window metrics for a nonlinear heat-flux trace."""

    t = np.asarray(time, dtype=float).reshape(-1)
    q = np.asarray(heat_flux, dtype=float).reshape(-1)
    if t.size != q.size or t.size < 4:
        raise ValueError(
            "time and heat_flux must have the same length with at least four samples"
        )
    if not (0.0 < float(tail_fraction) <= 1.0):
        raise ValueError("tail_fraction must be in (0, 1]")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(q))):
        raise ValueError("time and heat_flux must be finite")

    n = int(q.size)
    start = max(0, min(n - 2, int(round((1.0 - float(tail_fraction)) * n))))
    tw = t[start:]
    qw = q[start:]
    mean = float(np.mean(qw))
    std = float(np.std(qw))
    centered_t = tw - float(np.mean(tw))
    denom = float(np.sum(centered_t**2))
    slope = 0.0 if denom <= 0.0 else float(np.sum(centered_t * (qw - mean)) / denom)
    span = float(tw[-1] - tw[0])
    trend = abs(slope) * span / max(abs(mean), 1.0e-300)
    return {
        "start_index": int(start),
        "n_samples": int(qw.size),
        "t_min": float(tw[0]),
        "t_max": float(tw[-1]),
        "mean": mean,
        "std": std,
        "cv": std / max(abs(mean), 1.0e-300),
        "slope": slope,
        "trend": float(trend),
        "last": float(q[-1]),
    }


def transport_average_requirements(
    runs: list[dict[str, Any]],
    *,
    min_total_time: float = DEFAULT_TRANSPORT_MIN_TOTAL_TIME,
    transient_time: float = DEFAULT_TRANSPORT_TRANSIENT_TIME,
    min_post_transient_samples: int = DEFAULT_TRANSPORT_MIN_POST_TRANSIENT_SAMPLES,
    max_running_mean_rel_change: float = DEFAULT_TRANSPORT_MAX_RUNNING_MEAN_REL_CHANGE,
    min_abs_mean_heat_flux: float = DEFAULT_TRANSPORT_MIN_ABS_MEAN_HEAT_FLUX,
) -> dict[str, Any]:
    """Return long-window transport-average acceptance diagnostics.

    A meaningful nonlinear heat flux must be averaged after the initial
    transient, and the running mean over that post-transient window must be
    stable.  The compact FD audit defaults intentionally do not satisfy this
    contract; this helper records that fact in the artifact so startup-response
    checks cannot be mistaken for transport validation.
    """

    per_run: list[dict[str, Any]] = []
    for run in runs:
        time = np.asarray(run["time"], dtype=float)
        heat = np.asarray(run["heat_flux"], dtype=float)
        post_mask = time >= float(transient_time)
        post_heat = heat[post_mask]
        total_time = float(time[-1] - time[0]) if time.size else 0.0
        n_post = int(post_heat.size)
        final_mean = float(np.mean(post_heat)) if n_post else None
        running_change = None
        if n_post >= 4:
            running = np.cumsum(post_heat) / np.arange(1, n_post + 1, dtype=float)
            midpoint = max(1, n_post // 2) - 1
            running_change = float(
                abs(running[-1] - running[midpoint]) / max(abs(running[-1]), 1.0e-300)
            )
        gates = {
            "total_time": bool(total_time >= float(min_total_time)),
            "post_transient_samples": bool(n_post >= int(min_post_transient_samples)),
            "running_mean_converged": bool(
                running_change is not None
                and running_change <= float(max_running_mean_rel_change)
            ),
            "mean_heat_flux_above_noise_floor": bool(
                final_mean is not None
                and abs(final_mean) >= float(min_abs_mean_heat_flux)
            ),
        }
        per_run.append(
            {
                "label": str(run.get("label", "")),
                "total_time": total_time,
                "post_transient_samples": n_post,
                "post_transient_mean": final_mean,
                "running_mean_rel_change": running_change,
                "gates": gates,
                "passed": bool(all(gates.values())),
            }
        )
    return {
        "passed": bool(per_run and all(bool(item["passed"]) for item in per_run)),
        "requirements": {
            "min_total_time": float(min_total_time),
            "transient_time": float(transient_time),
            "min_post_transient_samples": int(min_post_transient_samples),
            "max_running_mean_rel_change": float(max_running_mean_rel_change),
            "min_abs_mean_heat_flux": float(min_abs_mean_heat_flux),
        },
        "runs": per_run,
    }


def run_cyclone_window(
    *,
    label: str,
    tprim: float,
    steps: int,
    tail_fraction: float,
    random_seed: int,
    nl: int,
    nm: int,
    ky: float,
    dt: float,
    nx: int,
    ny: int,
    nz: int,
) -> dict[str, Any]:
    """Run one compact nonlinear Cyclone startup point and return metrics."""

    cfg = cyclone_runtime_config(
        tprim=tprim, random_seed=random_seed, nx=nx, ny=ny, nz=nz, dt=dt
    )
    result = run_runtime_nonlinear(
        cfg,
        ky_target=float(ky),
        Nl=int(nl),
        Nm=int(nm),
        dt=float(dt),
        steps=int(steps),
        sample_stride=1,
        diagnostics_stride=1,
        resolved_diagnostics=False,
    )
    if result.diagnostics is None:
        raise RuntimeError("nonlinear audit requires diagnostics")
    time = np.asarray(result.diagnostics.t, dtype=float)
    heat = np.asarray(result.diagnostics.heat_flux_t, dtype=float)
    metrics = late_window_metrics(time, heat, tail_fraction=tail_fraction)
    return {
        "label": str(label),
        "tprim": float(tprim),
        "random_seed": int(random_seed),
        "time": time.tolist(),
        "heat_flux": heat.tolist(),
        "window": metrics,
    }


def build_audit_payload(
    runs: list[dict[str, Any]],
    *,
    base_tprim: float,
    perturbation_step: float,
    tail_fraction: float,
    repeatability_rtol: float,
    max_window_cv: float,
    max_window_trend: float,
    min_response_fraction: float,
) -> dict[str, Any]:
    """Build a startup-window nonlinear FD audit from completed runs."""

    by_label = {str(run["label"]): run for run in runs}
    required = {"minus", "base", "plus", "base_repeat"}
    missing = sorted(required.difference(by_label))
    if missing:
        raise ValueError(f"missing required audit runs: {missing}")
    means = {label: float(by_label[label]["window"]["mean"]) for label in required}
    base = means["base"]
    plus = means["plus"]
    minus = means["minus"]
    repeat = means["base_repeat"]
    step = float(perturbation_step)
    if step <= 0.0:
        raise ValueError("perturbation_step must be positive")

    central = (plus - minus) / (2.0 * step)
    forward = (plus - base) / step
    backward = (base - minus) / step
    repeat_rel = abs(repeat - base) / max(abs(base), 1.0e-300)
    response = abs(central) * step
    response_fraction = response / max(abs(base), 1.0e-300)
    max_cv = max(float(run["window"]["cv"]) for run in runs)
    max_trend = max(float(run["window"]["trend"]) for run in runs)
    derivative_asymmetry = abs(forward - backward) / max(abs(central), 1.0e-300)

    gates = {
        "finite_outputs": all(
            np.all(np.isfinite(np.asarray(run["heat_flux"], dtype=float)))
            and all(
                math.isfinite(float(run["window"][key]))
                for key in ("mean", "cv", "trend", "slope")
            )
            for run in runs
        ),
        "repeatability": bool(repeat_rel <= float(repeatability_rtol)),
        "monotonic_drive_response": bool(plus > base > minus),
        "window_cv": bool(max_cv <= float(max_window_cv)),
        "window_trend": bool(max_trend <= float(max_window_trend)),
        "resolved_fd_response": bool(response_fraction >= float(min_response_fraction)),
    }
    startup_gate = bool(all(gates.values()))
    transport_gate = transport_average_requirements(runs)
    return {
        "kind": "nonlinear_startup_window_finite_difference_audit",
        "case": "compact_cyclone_startup_nonlinear_window",
        "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
        "passed": startup_gate,
        "startup_nonlinear_plumbing_fd_path_gate": startup_gate,
        "transport_average_gate": bool(transport_gate["passed"]),
        "production_nonlinear_observable_fd_path_gate": False,
        "production_nonlinear_window_gradient_gate": False,
        "base_tprim": float(base_tprim),
        "perturbation_step": step,
        "tail_fraction": float(tail_fraction),
        "thresholds": {
            "repeatability_rtol": float(repeatability_rtol),
            "max_window_cv": float(max_window_cv),
            "max_window_trend": float(max_window_trend),
            "min_response_fraction": float(min_response_fraction),
        },
        "metrics": {
            "central_fd_dq_dtprim": float(central),
            "forward_dq_dtprim": float(forward),
            "backward_dq_dtprim": float(backward),
            "derivative_asymmetry": float(derivative_asymmetry),
            "repeatability_relative_error": float(repeat_rel),
            "response_fraction": float(response_fraction),
            "max_window_cv": float(max_cv),
            "max_window_trend": float(max_trend),
        },
        "gates": gates,
        "transport_average_requirements": transport_gate,
        "runs": runs,
        "next_action": (
            "Use this only as a compact startup-response plumbing and FD-conditioning audit. "
            "Meaningful nonlinear heat-flux claims require long post-transient simulations, "
            "running-average convergence, block/window stability, and comparison against the "
            "tracked long-window nonlinear reference gates before optimization or manuscript use."
        ),
    }


def audit_figure(payload: dict[str, Any]) -> plt.Figure:
    """Create the nonlinear startup-response FD audit figure."""

    set_plot_style()
    runs = list(payload["runs"])
    colors = {
        "minus": "#457b9d",
        "base": "#1b4332",
        "plus": "#d1495b",
        "base_repeat": "#6c757d",
    }
    labels = {
        "minus": "base - step",
        "base": "base",
        "plus": "base + step",
        "base_repeat": "base repeat",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    ax0, ax1 = axes
    for run in runs:
        label = str(run["label"])
        t = np.asarray(run["time"], dtype=float)
        q = np.asarray(run["heat_flux"], dtype=float)
        window = run["window"]
        ax0.plot(
            t,
            q,
            linewidth=2.0,
            color=colors.get(label, "#333333"),
            label=labels.get(label, label),
        )
        ax0.axvspan(
            float(window["t_min"]),
            float(window["t_max"]),
            color=colors.get(label, "#333333"),
            alpha=0.055,
        )
    ax0.set_title("Startup nonlinear heat-flux traces")
    ax0.set_xlabel("time")
    ax0.set_ylabel("heat flux")
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=True, framealpha=0.92)

    order = ["minus", "base", "plus", "base_repeat"]
    x = np.arange(len(order), dtype=float)
    means = [
        float(next(run for run in runs if str(run["label"]) == label)["window"]["mean"])
        for label in order
    ]
    stds = [
        float(next(run for run in runs if str(run["label"]) == label)["window"]["std"])
        for label in order
    ]
    ax1.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=[colors[label] for label in order],
        edgecolor="#222222",
        linewidth=0.7,
    )
    ax1.set_xticks(x, [labels[label] for label in order], rotation=18, ha="right")
    ax1.set_ylabel("startup-window heat-flux mean")
    ax1.set_title("Finite-difference conditioning")
    ax1.grid(True, axis="y", alpha=0.25)
    metrics = payload["metrics"]
    gates = payload["gates"]
    text = "\n".join(
        [
            f"central dQ/d(a/LTi): {float(metrics['central_fd_dq_dtprim']):.3e}",
            f"response/base: {float(metrics['response_fraction']):.3f}",
            f"repeat rel. err.: {float(metrics['repeatability_relative_error']):.1e}",
            f"max CV/trend: {float(metrics['max_window_cv']):.3f}/{float(metrics['max_window_trend']):.3f}",
            f"startup gate: {'PASS' if payload['passed'] else 'BLOCKED'}",
            f"transport gate: {'PASS' if payload['transport_average_gate'] else 'not claimed'}",
            f"gradient claim: {'PASS' if gates.get('vmec_boozer_gradient', False) else 'not claimed'}",
        ]
    )
    ax1.text(
        0.98,
        0.04,
        text,
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
            "boxstyle": "round,pad=0.35",
        },
    )
    fig.suptitle(
        "Nonlinear startup-window finite-difference audit",
        y=1.04,
        fontsize=14,
        fontweight="bold",
    )
    return fig


def write_audit_artifacts(payload: dict[str, Any], out: Path) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF artifacts for one audit payload."""

    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    csv_path = out.with_suffix(".csv")
    pdf_path = out.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "label",
            "tprim",
            "random_seed",
            "window_mean",
            "window_std",
            "window_cv",
            "window_trend",
            "t_min",
            "t_max",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for run in payload["runs"]:
            window = run["window"]
            writer.writerow(
                {
                    "label": run["label"],
                    "tprim": run["tprim"],
                    "random_seed": run["random_seed"],
                    "window_mean": window["mean"],
                    "window_std": window["std"],
                    "window_cv": window["cv"],
                    "window_trend": window["trend"],
                    "t_min": window["t_min"],
                    "t_max": window["t_max"],
                }
            )
    fig = audit_figure(payload)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": str(out),
        "pdf": str(pdf_path),
        "csv": str(csv_path),
        "json": str(json_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-tprim", type=float, default=2.49)
    parser.add_argument("--perturbation-step", type=float, default=0.18)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--tail-fraction", type=float, default=0.30)
    parser.add_argument("--random-seed", type=int, default=22)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=3)
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--repeatability-rtol", type=float, default=1.0e-6)
    parser.add_argument("--max-window-cv", type=float, default=0.15)
    parser.add_argument("--max-window-trend", type=float, default=0.40)
    parser.add_argument("--min-response-fraction", type=float, default=0.03)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = float(args.base_tprim)
    step = float(args.perturbation_step)
    run_kwargs = {
        "steps": int(args.steps),
        "tail_fraction": float(args.tail_fraction),
        "random_seed": int(args.random_seed),
        "nl": int(args.nl),
        "nm": int(args.nm),
        "ky": float(args.ky),
        "dt": float(args.dt),
        "nx": int(args.nx),
        "ny": int(args.ny),
        "nz": int(args.nz),
    }
    runs = [
        run_cyclone_window(label="minus", tprim=base - step, **run_kwargs),
        run_cyclone_window(label="base", tprim=base, **run_kwargs),
        run_cyclone_window(label="plus", tprim=base + step, **run_kwargs),
        run_cyclone_window(label="base_repeat", tprim=base, **run_kwargs),
    ]
    payload = build_audit_payload(
        runs,
        base_tprim=base,
        perturbation_step=step,
        tail_fraction=float(args.tail_fraction),
        repeatability_rtol=float(args.repeatability_rtol),
        max_window_cv=float(args.max_window_cv),
        max_window_trend=float(args.max_window_trend),
        min_response_fraction=float(args.min_response_fraction),
    )
    paths = write_audit_artifacts(payload, Path(args.out))
    print(
        f"passed={payload['passed']} response_fraction={payload['metrics']['response_fraction']:.6g}"
    )
    for path in paths.values():
        print(f"Wrote {path}")
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
