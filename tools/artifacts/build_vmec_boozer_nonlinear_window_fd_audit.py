#!/usr/bin/env python3
"""Build a VMEC/Boozer-perturbed nonlinear startup FD audit.

This is a bounded startup-path audit.  It starts from the existing mode-21
``vmex -> booz_xform_jax`` geometry bridge, perturbs one VMEC state
coefficient, materializes the resulting sampled geometry to temporary NetCDF
files, and runs compact GKX nonlinear startup windows at
``x = base +/- step`` plus a repeated base point.  Passing this gate validates
the finite-difference plumbing from VMEC/Boozer geometry perturbations into
short nonlinear diagnostics.  It is not a transport-average, promoted
optimized-equilibrium nonlinear heat-flux gradient, or stellarator heat-flux
optimization claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from gkx.objectives.vmec_boozer_gradients import (
    _mode21_vmec_boozer_linear_context,
)  # noqa: E402

from tools.artifacts.build_nonlinear_window_fd_audit import (  # noqa: E402
    late_window_metrics,
    transport_average_requirements as late_transport_requirements,
)


DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_nonlinear_window_fd_audit.png"
PROFILE_NAMES = (
    "bmag",
    "bgrad",
    "gds2",
    "gds21",
    "gds22",
    "cvdrift",
    "gbdrift",
    "cvdrift0",
    "gbdrift0",
    "jacobian",
    "grho",
)


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


def _profile_arrays(geom: Any) -> dict[str, np.ndarray]:
    return {
        "bmag": np.asarray(geom.bmag_profile, dtype=float),
        "bgrad": np.asarray(geom.bgrad_profile, dtype=float),
        "gds2": np.asarray(geom.gds2_profile, dtype=float),
        "gds21": np.asarray(geom.gds21_profile, dtype=float),
        "gds22": np.asarray(geom.gds22_profile, dtype=float),
        "cvdrift": np.asarray(geom.cv_profile, dtype=float),
        "gbdrift": np.asarray(geom.gb_profile, dtype=float),
        "cvdrift0": np.asarray(geom.cv0_profile, dtype=float),
        "gbdrift0": np.asarray(geom.gb0_profile, dtype=float),
        "jacobian": np.asarray(geom.jacobian_profile, dtype=float),
        "grho": np.asarray(geom.grho_profile, dtype=float),
    }


def geometry_response_metrics(base_geom: Any, perturbed_geom: Any) -> dict[str, float]:
    """Return normalized geometry-change metrics for one perturbation."""

    base_profiles = _profile_arrays(base_geom)
    perturbed_profiles = _profile_arrays(perturbed_geom)
    per_profile: dict[str, float] = {}
    for name in PROFILE_NAMES:
        base = base_profiles[name]
        perturbed = perturbed_profiles[name]
        if base.shape != perturbed.shape:
            raise ValueError(
                f"geometry profile {name!r} shape changed under perturbation"
            )
        scale = max(float(np.max(np.abs(base))), 1.0)
        per_profile[name] = float(np.max(np.abs(perturbed - base)) / scale)
    scalar_values = {
        "gradpar": abs(float(perturbed_geom.gradpar()) - float(base_geom.gradpar()))
        / max(abs(float(base_geom.gradpar())), 1.0),
        "q": abs(float(perturbed_geom.q) - float(base_geom.q))
        / max(abs(float(base_geom.q)), 1.0),
        "s_hat": abs(float(perturbed_geom.s_hat) - float(base_geom.s_hat))
        / max(abs(float(base_geom.s_hat)), 1.0),
    }
    return {
        "max_profile_relative_change": float(max(per_profile.values())),
        "max_scalar_relative_change": float(max(scalar_values.values())),
        "max_relative_change": float(
            max(max(per_profile.values()), max(scalar_values.values()))
        ),
        "per_profile": per_profile,
        "per_scalar": scalar_values,
    }


def write_flux_tube_geometry_netcdf(geom: Any, path: Path) -> None:
    """Write ``FluxTubeGeometryData`` as a grouped imported-geometry NetCDF."""

    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "netCDF4 is required for VMEC/Boozer nonlinear-window FD audits"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    theta = np.asarray(geom.theta, dtype=np.float64)
    profiles = _profile_arrays(geom)
    with Dataset(path, "w") as root:
        grids = root.createGroup("Grids")
        geometry = root.createGroup("Geometry")
        grids.createDimension("theta", theta.size)
        geometry.createDimension("theta", theta.size)
        grids.createVariable("theta", "f8", ("theta",))[:] = theta
        for name, profile in profiles.items():
            geometry.createVariable(name, "f8", ("theta",))[:] = np.asarray(
                profile, dtype=np.float64
            )
        scalars = {
            "gradpar": float(geom.gradpar()),
            "q": float(geom.q),
            "shat": float(geom.s_hat),
            "rmaj": float(geom.R0),
            "aminor": float(geom.epsilon) * float(geom.R0),
            "alpha": float(geom.alpha),
            "kxfac": float(geom.kxfac),
            "theta_scale": float(geom.theta_scale),
        }
        for name, value in scalars.items():
            geometry.createVariable(name, "f8", ())[:] = value
        geometry.createVariable("nfp", "i4", ())[:] = int(geom.nfp)


def vmec_boozer_runtime_config(
    geometry_file: Path,
    *,
    tprim: float,
    fprim: float,
    random_seed: int,
    init_amp: float,
    nx: int,
    ny: int,
    nz: int,
    dt: float,
) -> RuntimeConfig:
    """Return a compact imported-geometry nonlinear startup-audit configuration."""

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
        geometry=GeometryConfig(
            model="imported-netcdf", geometry_file=str(Path(geometry_file))
        ),
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


def run_vmec_boozer_window(
    *,
    label: str,
    perturbation: float,
    context: dict[str, Any],
    base_geom: Any,
    workdir: Path,
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
    tprim: float,
    fprim: float,
    init_amp: float,
) -> dict[str, Any]:
    """Run one VMEC/Boozer-perturbed compact nonlinear startup window."""

    geom = context["geometry_for"](jnp.asarray([float(perturbation)]))
    geometry_metrics = geometry_response_metrics(base_geom, geom)
    geometry_file = workdir / f"{label}_geometry.nc"
    write_flux_tube_geometry_netcdf(geom, geometry_file)
    cfg = vmec_boozer_runtime_config(
        geometry_file,
        tprim=tprim,
        fprim=fprim,
        random_seed=random_seed,
        init_amp=init_amp,
        nx=nx,
        ny=ny,
        nz=nz,
        dt=dt,
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
        raise RuntimeError("VMEC/Boozer nonlinear-window audit requires diagnostics")
    time = np.asarray(result.diagnostics.t, dtype=float)
    heat = np.asarray(result.diagnostics.heat_flux_t, dtype=float)
    metrics = late_window_metrics(time, heat, tail_fraction=tail_fraction)
    return {
        "label": str(label),
        "perturbation": float(perturbation),
        "geometry_file_name": geometry_file.name,
        "geometry_response": geometry_metrics,
        "time": time.tolist(),
        "heat_flux": heat.tolist(),
        "window": metrics,
    }


def build_vmec_boozer_audit_payload(
    runs: list[dict[str, Any]],
    *,
    case_name: str,
    parameter_name: str,
    perturbation_step: float,
    tail_fraction: float,
    repeatability_rtol: float,
    max_window_cv: float,
    max_window_trend: float,
    min_response_fraction: float,
    min_geometry_relative_change: float,
) -> dict[str, Any]:
    """Build a JSON-ready VMEC/Boozer nonlinear startup FD audit payload."""

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
    response_fraction = abs(central) * step / max(abs(base), 1.0e-300)
    max_cv = max(float(run["window"]["cv"]) for run in runs)
    max_trend = max(float(run["window"]["trend"]) for run in runs)
    derivative_asymmetry = abs(forward - backward) / max(abs(central), 1.0e-300)
    geom_changes = [
        float(by_label[label]["geometry_response"]["max_relative_change"])
        for label in ("minus", "plus")
    ]
    min_geom_change = min(geom_changes)

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
        "window_cv": bool(max_cv <= float(max_window_cv)),
        "window_trend": bool(max_trend <= float(max_window_trend)),
        "resolved_fd_response": bool(response_fraction >= float(min_response_fraction)),
        "geometry_perturbation_resolved": bool(
            min_geom_change >= float(min_geometry_relative_change)
        ),
    }
    startup_gate = bool(all(gates.values()))
    transport_gate = late_transport_requirements(runs)
    return {
        "kind": "vmec_boozer_nonlinear_startup_finite_difference_audit",
        "case": "mode21_qh_vmec_boozer_imported_geometry_compact_startup_window",
        "case_name": str(case_name),
        "parameter_name": str(parameter_name),
        "claim_level": "vmec_boozer_geometry_perturbed_startup_plumbing_fd_audit_not_transport_average",
        "passed": startup_gate,
        "vmec_boozer_startup_nonlinear_plumbing_fd_path_gate": startup_gate,
        "transport_average_gate": bool(transport_gate["passed"]),
        "vmec_boozer_production_nonlinear_observable_fd_path_gate": False,
        "production_nonlinear_window_gradient_gate": False,
        "perturbation_step": step,
        "tail_fraction": float(tail_fraction),
        "thresholds": {
            "repeatability_rtol": float(repeatability_rtol),
            "max_window_cv": float(max_window_cv),
            "max_window_trend": float(max_window_trend),
            "min_response_fraction": float(min_response_fraction),
            "min_geometry_relative_change": float(min_geometry_relative_change),
        },
        "metrics": {
            "central_fd_dq_dparameter": float(central),
            "forward_dq_dparameter": float(forward),
            "backward_dq_dparameter": float(backward),
            "derivative_asymmetry": float(derivative_asymmetry),
            "repeatability_relative_error": float(repeat_rel),
            "response_fraction": float(response_fraction),
            "max_window_cv": float(max_cv),
            "max_window_trend": float(max_trend),
            "min_geometry_relative_change": float(min_geom_change),
        },
        "diagnostics": {
            "observable_ordered_response": bool(plus > base > minus),
            "forward_backward_same_sign": bool(
                (forward >= 0.0 and backward >= 0.0)
                or (forward <= 0.0 and backward <= 0.0)
            ),
        },
        "gates": gates,
        "transport_average_requirements": transport_gate,
        "runs": runs,
        "next_action": (
            "Use this only as a VMEC/Boozer-perturbed startup-response plumbing FD audit. "
            "Promoted nonlinear stellarator optimization still requires long post-transient "
            "running-average heat-flux windows, optimized-equilibrium audits, and a validated "
            "gradient/adjoint strategy."
        ),
    }


def audit_figure(payload: dict[str, Any]) -> plt.Figure:
    """Create the VMEC/Boozer nonlinear startup FD audit figure."""

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
    ax0.set_title("VMEC/Boozer-perturbed startup windows")
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
    ax1.set_title("VMEC-state FD conditioning")
    ax1.grid(True, axis="y", alpha=0.25)
    metrics = payload["metrics"]
    text = "\n".join(
        [
            f"dQ/d{payload['parameter_name']}: {float(metrics['central_fd_dq_dparameter']):.3e}",
            f"response/base: {float(metrics['response_fraction']):.3f}",
            f"repeat rel. err.: {float(metrics['repeatability_relative_error']):.1e}",
            f"max CV/trend: {float(metrics['max_window_cv']):.3f}/{float(metrics['max_window_trend']):.3f}",
            f"geom response: {float(metrics['min_geometry_relative_change']):.1e}",
            f"startup gate: {'PASS' if payload['passed'] else 'BLOCKED'}",
            f"transport gate: {'PASS' if payload['transport_average_gate'] else 'not claimed'}",
            "gradient claim: not promoted",
        ]
    )
    ax1.text(
        0.98,
        0.04,
        text,
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        bbox={
            "facecolor": "white",
            "edgecolor": "#cccccc",
            "alpha": 0.9,
            "boxstyle": "round,pad=0.35",
        },
    )
    fig.suptitle(
        "VMEC/Boozer nonlinear startup finite-difference audit",
        y=1.04,
        fontsize=14,
        fontweight="bold",
    )
    return fig


def write_audit_artifacts(payload: dict[str, Any], out: Path) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for one audit payload."""

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
            "perturbation",
            "window_mean",
            "window_std",
            "window_cv",
            "window_trend",
            "geometry_max_relative_change",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for run in payload["runs"]:
            writer.writerow(
                {
                    "label": run["label"],
                    "perturbation": run["perturbation"],
                    "window_mean": run["window"]["mean"],
                    "window_std": run["window"]["std"],
                    "window_cv": run["window"]["cv"],
                    "window_trend": run["window"]["trend"],
                    "geometry_max_relative_change": run["geometry_response"][
                        "max_relative_change"
                    ],
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
    parser.add_argument("--case-name", default="nfp4_QH_warm_start")
    parser.add_argument("--radial-index", type=int, default=None)
    parser.add_argument("--mode-index", type=int, default=1)
    parser.add_argument("--surface-index", type=int, default=None)
    parser.add_argument("--surface-stencil-width", type=int, default=3)
    parser.add_argument("--mboz", type=int, default=21)
    parser.add_argument("--nboz", type=int, default=21)
    parser.add_argument("--ntheta", type=int, default=8)
    parser.add_argument("--perturbation-step", type=float, default=1.0e-5)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--tail-fraction", type=float, default=0.30)
    parser.add_argument("--random-seed", type=int, default=22)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=3)
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--tprim", type=float, default=2.49)
    parser.add_argument("--fprim", type=float, default=0.8)
    parser.add_argument("--init-amp", type=float, default=1.0e-4)
    parser.add_argument("--repeatability-rtol", type=float, default=1.0e-6)
    parser.add_argument("--max-window-cv", type=float, default=0.15)
    parser.add_argument("--max-window-trend", type=float, default=0.40)
    parser.add_argument("--min-response-fraction", type=float, default=0.03)
    parser.add_argument("--min-geometry-relative-change", type=float, default=1.0e-7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if 0 < int(args.surface_stencil_width) < 3:
        raise SystemExit("--surface-stencil-width must be 0 or at least 3")
    step = float(args.perturbation_step)
    if step <= 0.0:
        raise SystemExit("--perturbation-step must be positive")
    context = _mode21_vmec_boozer_linear_context(
        case_name=args.case_name,
        radial_index=args.radial_index,
        mode_index=args.mode_index,
        surface_index=args.surface_index,
        ntheta=int(args.ntheta),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        surface_stencil_width=None
        if int(args.surface_stencil_width) <= 0
        else int(args.surface_stencil_width),
        n_laguerre=int(args.nl),
        n_hermite=int(args.nm),
    )
    base_geom = context["geometry_for"](jnp.asarray([0.0]))
    run_kwargs = {
        "context": context,
        "base_geom": base_geom,
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
        "tprim": float(args.tprim),
        "fprim": float(args.fprim),
        "init_amp": float(args.init_amp),
    }
    with tempfile.TemporaryDirectory(prefix="gkx_vmec_boozer_nl_fd_") as tmp:
        workdir = Path(tmp)
        runs = [
            run_vmec_boozer_window(
                label="minus", perturbation=-step, workdir=workdir, **run_kwargs
            ),
            run_vmec_boozer_window(
                label="base", perturbation=0.0, workdir=workdir, **run_kwargs
            ),
            run_vmec_boozer_window(
                label="plus", perturbation=step, workdir=workdir, **run_kwargs
            ),
            run_vmec_boozer_window(
                label="base_repeat", perturbation=0.0, workdir=workdir, **run_kwargs
            ),
        ]
    payload = build_vmec_boozer_audit_payload(
        runs,
        case_name=str(args.case_name),
        parameter_name=str(context["parameter_names"][0]),
        perturbation_step=step,
        tail_fraction=float(args.tail_fraction),
        repeatability_rtol=float(args.repeatability_rtol),
        max_window_cv=float(args.max_window_cv),
        max_window_trend=float(args.max_window_trend),
        min_response_fraction=float(args.min_response_fraction),
        min_geometry_relative_change=float(args.min_geometry_relative_change),
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
