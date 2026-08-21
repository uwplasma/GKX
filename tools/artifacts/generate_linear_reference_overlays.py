#!/usr/bin/env python3
"""Generate linear eigenfunction reference overlays for validation cases.

Subcommands:
  kbm   Run the KBM raw eigenfunction overlay against a frozen reference.
  w7x   Run the W7-X imported-geometry overlay against a frozen reference.
  overlap-summary   Plot overlap and relative-L2 values from a candidate table.
  reference-overlay Plot a saved mode against a frozen reference bundle.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from tools.comparison.compare_gx_imported_linear import (  # noqa: E402
    _build_imported_linear_terms,
    _build_sample_steps,
    _gx_has_uniform_linear_dt,
    _infer_gx_linear_dt,
    _load_gx_input_contract,
    _load_gx_reference,
    _match_local_kx_index,
    _resolve_imported_boundary,
    _resolve_imported_real_fft_ny,
    _run_single_ky,
    _select_gx_kx_index,
)
from tools.comparison.compare_gx_kbm import (  # noqa: E402
    _build_cfg,
    _load_gx_eigenfunction,
    _load_kbm_reference_input_contract,
    _normalize_mode as _normalize_kbm_mode,
    _prepare_gx_reference,
    _runtime_config_from_kbm_case,
)
from gkx.artifacts.plotting import (  # noqa: E402
    eigenfunction_overlap_summary_figure,
    eigenfunction_reference_overlay_figure,
)
from gkx.diagnostics.modes import (  # noqa: E402
    compare_eigenfunctions,
    load_eigenfunction_reference_bundle,
    save_eigenfunction_reference_bundle,
)
from gkx.diagnostics.validation_gates import (  # noqa: E402
    eigenfunction_gate_report,
    gate_report_to_dict,
)
from gkx.artifacts.spectral_layout import infer_triple_dealiased_ny  # noqa: E402
from gkx.diagnostics.growth_windows import late_time_window  # noqa: E402
from gkx.benchmarking.shared import _apply_reference_hypercollisions  # noqa: E402
from gkx.runtime import run_runtime_linear  # noqa: E402
from gkx.config import GridConfig, resolve_cfl_fac  # noqa: E402
from gkx.core.grid import build_spectral_grid  # noqa: E402
from gkx.operators.linear.params import build_linear_params  # noqa: E402
from gkx.diagnostics.analysis import extract_eigenfunction  # noqa: E402
from gkx.geometry import (  # noqa: E402
    apply_imported_geometry_grid_defaults,
    load_imported_geometry_netcdf,
)
from gkx.solvers.time.explicit import ExplicitTimeConfig  # noqa: E402

KBM_EIGENFUNCTION_GATE_TOLERANCES = {
    "min_overlap": 0.95,
    "max_relative_l2": 0.25,
}
W7X_EIGENFUNCTION_GATE_TOLERANCES = {
    "min_overlap": 0.95,
    "max_relative_l2": 0.25,
}
DEFAULT_OVERLAP_CSV = (
    ROOT / "docs" / "_static" / "comparison" / "kbm_reference_candidates.csv"
)
DEFAULT_OVERLAP_OUT = ROOT / "docs" / "_static" / "kbm_eigenfunction_overlap_summary.png"
DEFAULT_REFERENCE_OVERLAY_OUT = (
    ROOT / "docs" / "_static" / "eigenfunction_reference_overlay.png"
)


def _artifact_path(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_eigenfunction_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a complex mode from the documented eigenfunction CSV schema."""

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    required = {"z", "eigen_real", "eigen_imag"}
    names = set(data.dtype.names or ())
    if not required.issubset(names):
        raise ValueError("eigenfunction CSV must contain z,eigen_real,eigen_imag columns")
    theta = np.asarray(data["z"], dtype=float)
    mode = np.asarray(data["eigen_real"], dtype=float) + 1j * np.asarray(
        data["eigen_imag"], dtype=float
    )
    return theta, mode


def plot_saved_reference_overlay(
    *,
    reference: Path,
    gkx: Path,
    out: Path = DEFAULT_REFERENCE_OVERLAY_OUT,
    title: str = "Eigenfunction overlay against frozen reference",
) -> dict[str, str]:
    """Write a phase-aligned saved-mode overlay."""

    bundle = load_eigenfunction_reference_bundle(reference)
    theta, mode = load_eigenfunction_csv(gkx)
    fig, _axes = eigenfunction_reference_overlay_figure(
        theta, mode, bundle.theta, bundle.mode, title=title
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    pdf = out.with_suffix(".pdf")
    if out.suffix.lower() != ".pdf":
        fig.savefig(pdf, bbox_inches="tight")
    return {"png_or_pdf": str(out), "pdf": str(pdf)}


def plot_overlap_summary(
    *,
    csv: Path = DEFAULT_OVERLAP_CSV,
    out: Path = DEFAULT_OVERLAP_OUT,
    title: str = "KBM eigenfunction overlap against reference",
) -> dict[str, str]:
    """Write overlap and relative-L2 panels from a candidate table."""

    table = pd.read_csv(csv)
    if "selected" in table.columns:
        table = table[table["selected"].astype(bool)]
    required = {"ky", "eig_overlap_gx", "eig_rel_l2"}
    missing = required.difference(table.columns)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")
    table = table[
        np.isfinite(table["eig_overlap_gx"]) & np.isfinite(table["eig_rel_l2"])
    ].sort_values("ky")
    fig, _axes = eigenfunction_overlap_summary_figure(
        np.asarray(table["ky"], dtype=float),
        np.asarray(table["eig_overlap_gx"], dtype=float),
        np.asarray(table["eig_rel_l2"], dtype=float),
        title=title,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    return {"png": str(out), "pdf": str(pdf)}


def build_reference_overlay_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a saved mode against a reference bundle.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("gkx", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_REFERENCE_OVERLAY_OUT)
    parser.add_argument("--title", default="Eigenfunction overlay against frozen reference")
    return parser


def build_overlap_summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot eigenfunction candidate metrics.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_OVERLAP_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OVERLAP_OUT)
    parser.add_argument("--title", default="KBM eigenfunction overlap against reference")
    return parser


def main_reference_overlay(argv: list[str] | None = None) -> int:
    args = build_reference_overlay_parser().parse_args(argv)
    plot_saved_reference_overlay(
        reference=args.reference, gkx=args.gkx, out=args.out, title=args.title
    )
    return 0


def main_overlap_summary(argv: list[str] | None = None) -> int:
    args = build_overlap_summary_parser().parse_args(argv)
    plot_overlap_summary(csv=args.csv, out=args.out, title=args.title)
    return 0


def _selected_candidate_row(candidate_csv: Path, ky: float) -> pd.Series:
    table = pd.read_csv(candidate_csv)
    mask = table["selected"].astype(bool) & (
        table["ky"].round(4) == round(float(ky), 4)
    )
    if not mask.any():
        raise ValueError(f"no selected KBM candidate found for ky={ky:.4f}")
    return table.loc[mask].iloc[0]


def _steps_for_fit_window(
    *, fit_tmax: float, dt: float, fit_padding: float, sample_stride: int
) -> int:
    steps = max(int(math.ceil((float(fit_tmax) + float(fit_padding)) / float(dt))), 1)
    stride = max(int(sample_stride), 1)
    rem = steps % stride
    if rem != 0:
        steps += stride - rem
    return steps


def _kbm_eigenfunction_gate_report(metrics):
    return eigenfunction_gate_report(
        metrics,
        case="kbm_linear_eigenfunction_ky0p3000",
        source="GX raw eigenfunction bundle",
        min_overlap=float(KBM_EIGENFUNCTION_GATE_TOLERANCES["min_overlap"]),
        max_relative_l2=float(KBM_EIGENFUNCTION_GATE_TOLERANCES["max_relative_l2"]),
    )


def _normalize_mode(theta: np.ndarray, mode: np.ndarray) -> np.ndarray:
    finite = np.isfinite(mode)
    if not np.any(finite):
        return np.zeros_like(mode)
    idx0 = int(np.argmin(np.abs(theta)))
    ref = mode[idx0]
    if not np.isfinite(ref) or abs(ref) < 1.0e-14:
        idx = int(np.nanargmax(np.abs(np.where(finite, mode, 0.0))))
        ref = mode[idx]
    if not np.isfinite(ref) or abs(ref) < 1.0e-14:
        scale = float(np.nanmax(np.abs(np.where(finite, mode, 0.0))))
        return mode if scale <= 0.0 else mode / scale
    return mode / ref


def _freeze_gx_big_reference(gx_big: Path, *, ky: float, out: Path) -> Path:
    with Dataset(gx_big, "r") as root:
        theta = np.asarray(root.groups["Grids"].variables["theta"][:], dtype=float)
        ky_grid = np.asarray(root.groups["Grids"].variables["ky"][:], dtype=float)
        phi = root.groups["Diagnostics"].variables["Phi"]
        ky_idx = int(np.argmin(np.abs(ky_grid - float(ky))))
        raw = np.asarray(phi[-1, ky_idx, 0, :, :], dtype=float)
    mode = _normalize_mode(theta, raw[:, 0] + 1j * raw[:, 1])
    if not np.all(np.isfinite(mode)):
        raise ValueError(f"{gx_big} produced a non-finite W7-X reference eigenfunction")
    return save_eigenfunction_reference_bundle(
        out,
        theta=theta,
        mode=mode,
        source="GX",
        case="w7x_linear",
        metadata={"ky": float(ky_grid[ky_idx]), "gx_big": str(gx_big)},
    )


def _load_finite_reference(bundle_path: Path):
    bundle = load_eigenfunction_reference_bundle(bundle_path)
    if not np.all(np.isfinite(bundle.mode)):
        raise ValueError(
            f"{bundle_path} contains a non-finite reference mode; rerun with --gx-big"
        )
    return bundle


def _run_w7x_gkx_mode(
    args: argparse.Namespace, *, reference_times: np.ndarray, output_steps: np.ndarray
):
    gx_contract = _load_gx_input_contract(args.gx_input)
    gx_time, gx_ky, gx_kx, _gx_omega, _gx_wg, _gx_wphi, _gx_wapar, _gx_phi2 = (
        _load_gx_reference(args.gx)
    )
    geom = load_imported_geometry_netcdf(args.geometry_file.expanduser().resolve())

    ny = _resolve_imported_real_fft_ny(gx_ky, gx_contract)
    boundary_eff = _resolve_imported_boundary(
        gx_contract.boundary, zero_shat=bool(gx_contract.zero_shat)
    )
    y0 = float(gx_contract.y0)
    lx = 2.0 * np.pi * y0 if boundary_eff == "periodic" else 62.8
    grid_cfg = GridConfig(
        Nx=max(1, int(gx_contract.Nx)),
        Ny=ny,
        Nz=int(np.asarray(geom.theta).size),
        Lx=lx,
        Ly=2.0 * np.pi * y0,
        boundary=boundary_eff,
        y0=y0,
        nperiod=max(1, int(gx_contract.nperiod)),
        ntheta=int(gx_contract.ntheta),
    )
    grid_full = build_spectral_grid(
        apply_imported_geometry_grid_defaults(geom, grid_cfg)
    )
    nl_use = int(args.Nl) if args.Nl is not None else int(gx_contract.nlaguerre)
    nm_use = int(args.Nm) if args.Nm is not None else int(gx_contract.nhermite)

    params = build_linear_params(
        list(gx_contract.species),
        tau_e=float(gx_contract.tau_e),
        kpar_scale=float(geom.gradpar()),
        beta=float(gx_contract.beta),
        fapar=float(gx_contract.fapar),
    )
    if gx_contract.hypercollisions:
        params = _apply_reference_hypercollisions(params, nhermite=nm_use)
    params = replace(
        params,
        D_hyper=float(gx_contract.D_hyper),
        damp_ends_amp=float(gx_contract.damp_ends_amp),
        damp_ends_widthfrac=float(gx_contract.damp_ends_widthfrac),
    )
    time_cfg = ExplicitTimeConfig(
        dt=_infer_gx_linear_dt(gx_time, gx_contract),
        t_max=float(gx_time[-1]),
        method=gx_contract.scheme,
        sample_stride=gx_contract.nwrite,
        fixed_dt=bool(
            gx_contract.dt is not None
            or _gx_has_uniform_linear_dt(gx_time, gx_contract)
        ),
        cfl_fac=resolve_cfl_fac(gx_contract.scheme, None),
    )
    gx_kx_idx = _select_gx_kx_index(gx_kx, gx_contract)
    kx_idx = _match_local_kx_index(np.asarray(grid_full.kx), float(gx_kx[gx_kx_idx]))
    result = _run_single_ky(
        ky_target=float(args.ky),
        geom=geom,
        grid_full=grid_full,
        params=params,
        time_cfg=time_cfg,
        gx_contract=gx_contract,
        species=tuple(gx_contract.species),
        Nl=nl_use,
        Nm=nm_use,
        reference_times=reference_times,
        output_steps=output_steps,
        mode_method=str(args.mode_method),
        kx_index=kx_idx,
        terms=_build_imported_linear_terms(gx_contract),
        return_phi_samples=True,
    )
    gamma, omega, wg, wphi, wapar, phi2, phi_sample_times, phi_samples = result
    theta = np.asarray(grid_full.z, dtype=float)
    mode = np.asarray(phi_samples[-1, 0, 0, :], dtype=np.complex128)
    return {
        "mode": _normalize_mode(theta, mode),
        "theta": theta,
        "gamma_last": float(np.asarray(gamma)[-1]),
        "omega_last": float(np.asarray(omega)[-1]),
        "Wg_last": float(np.asarray(wg)[-1]),
        "Wphi_last": float(np.asarray(wphi)[-1]),
        "Wapar_last": float(np.asarray(wapar)[-1]),
        "Phi2_last": float(np.asarray(phi2)[-1]),
        "t_final": float(np.asarray(phi_sample_times)[-1]),
        "nl": nl_use,
        "nm": nm_use,
        "ny": int(ny),
        "kx_local": float(np.asarray(grid_full.kx)[kx_idx]),
        "kx_ref": float(gx_kx[gx_kx_idx]),
    }


def _w7x_eigenfunction_gate_report(metrics):
    return eigenfunction_gate_report(
        metrics,
        case="w7x_linear_eigenfunction_ky0p3000",
        source="GX raw eigenfunction bundle",
        min_overlap=float(W7X_EIGENFUNCTION_GATE_TOLERANCES["min_overlap"]),
        max_relative_l2=float(W7X_EIGENFUNCTION_GATE_TOLERANCES["max_relative_l2"]),
    )


def build_kbm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a raw KBM eigenfunction overlay."
    )
    parser.add_argument(
        "--gx", type=Path, required=True, help="Path to GX .out.nc file."
    )
    parser.add_argument(
        "--gx-big", type=Path, required=True, help="Path to GX .big.nc file."
    )
    parser.add_argument(
        "--gx-input", type=Path, required=True, help="Path to the GX input deck."
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "comparison"
        / "kbm_reference_candidates.csv",
        help="Per-candidate KBM comparison table with the selected branch rows.",
    )
    parser.add_argument("--ky", type=float, default=0.3, help="Target ky value.")
    parser.add_argument("--dt", type=float, default=0.01, help="GKX time step.")
    parser.add_argument(
        "--fit-padding",
        type=float,
        default=0.5,
        help="Extra time beyond the selected growth-fit window before stopping the run.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=2,
        help="Sample stride for the bounded reference-time extraction path.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.4,
        help="Late-time fraction used for eigenfunction extraction.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "reference_modes"
        / "kbm_linear_gkx_ky0p3000.csv",
        help="Output CSV path for the GKX eigenfunction.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "kbm_eigenfunction_reference_overlay_ky0p3000.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "reference_modes"
        / "kbm_eigenfunction_reference_overlay_ky0p3000.json",
        help="Output JSON path for overlay metrics.",
    )
    parser.add_argument(
        "--bundle-out",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "comparison"
        / "reference_modes"
        / "kbm_linear_reference_ky0p3000.npz",
        help="Output path for the frozen reference bundle.",
    )
    return parser


def build_w7x_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a W7-X raw eigenfunction overlay."
    )
    parser.add_argument(
        "--gx",
        type=Path,
        required=True,
        help="GX .out.nc file for the matched W7-X linear run.",
    )
    parser.add_argument(
        "--gx-input",
        type=Path,
        required=True,
        help="GX input deck for the matched W7-X linear run.",
    )
    parser.add_argument(
        "--gx-big",
        type=Path,
        default=None,
        help="Optional GX .big.nc file used to refresh the frozen reference bundle.",
    )
    parser.add_argument(
        "--geometry-file",
        type=Path,
        required=True,
        help="Imported GX/VMEC geometry file for the GKX run.",
    )
    parser.add_argument("--ky", type=float, default=0.3, help="Target ky value.")
    parser.add_argument(
        "--Nl",
        type=int,
        default=None,
        help="Laguerre moments. Defaults to the GX input contract.",
    )
    parser.add_argument(
        "--Nm",
        type=int,
        default=None,
        help="Hermite moments. Defaults to the GX input contract.",
    )
    parser.add_argument(
        "--mode-method", choices=("z_index", "max", "project", "svd"), default="z_index"
    )
    parser.add_argument("--sample-step-stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-window", choices=("head", "tail"), default="head")
    parser.add_argument(
        "--bundle-out",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "comparison"
        / "reference_modes"
        / "w7x_linear_reference_ky0p3000.npz",
        help="Frozen reference bundle path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "reference_modes"
        / "w7x_linear_gkx_ky0p3000.csv",
        help="Output CSV path for the GKX eigenfunction.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "w7x_eigenfunction_reference_overlay_ky0p3000.png",
        help="Output overlay figure path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT
        / "docs"
        / "_static"
        / "reference_modes"
        / "w7x_eigenfunction_reference_overlay_ky0p3000.json",
        help="Output JSON path for overlay metrics.",
    )
    return parser


def main_kbm(argv: list[str] | None = None) -> int:
    args = build_kbm_parser().parse_args(argv)
    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")

    candidate = _selected_candidate_row(args.candidate_csv, args.ky)
    fit_tmin = float(candidate["fit_window_tmin"])
    fit_tmax = float(candidate["fit_window_tmax"])
    steps = _steps_for_fit_window(
        fit_tmax=fit_tmax,
        dt=args.dt,
        fit_padding=args.fit_padding,
        sample_stride=args.sample_stride,
    )

    _gx_time, _gx_ky, _gx_omega, beta, _q, _shat, _eps, _rmaj, nky_full, _y0 = (
        _prepare_gx_reference(args.gx, ky_arg=str(args.ky), y0_fallback=10.0)
    )
    contract = _load_kbm_reference_input_contract(args.gx_input)
    ny = infer_triple_dealiased_ny(int(nky_full))
    cfg = _build_cfg(
        beta=float(contract.beta),
        q=float(contract.q),
        shat=float(contract.shat),
        eps=float(contract.eps),
        rmaj=float(contract.rmaj),
        ny=ny,
        ntheta=int(contract.ntheta),
        nperiod=int(contract.nperiod),
        y0=float(contract.y0),
        mass_ratio=float(contract.mass_ratio),
        ion_tprim=float(contract.ion_tprim),
        ele_tprim=float(contract.ele_tprim),
        ion_fprim=float(contract.ion_fprim),
        ele_fprim=float(contract.ele_fprim),
        te_over_ti=float(contract.te_over_ti),
        init_field=str(contract.init_field),
        gaussian_init=bool(contract.gaussian_init),
        init_electrons_only=bool(contract.init_electrons_only),
    )

    theta_gx, mode_gx = _load_gx_eigenfunction(args.gx_big, float(args.ky))
    save_eigenfunction_reference_bundle(
        args.bundle_out,
        theta=theta_gx,
        mode=mode_gx,
        source="GX",
        case="kbm_linear",
        metadata={"ky": float(args.ky)},
    )

    runtime_cfg = _runtime_config_from_kbm_case(cfg)
    runtime_cfg = replace(
        runtime_cfg,
        time=replace(runtime_cfg.time, fixed_dt=False),
    )
    result = run_runtime_linear(
        runtime_cfg,
        ky_target=float(args.ky),
        Nl=int(contract.nlaguerre),
        Nm=int(contract.nhermite),
        dt=float(args.dt),
        steps=int(steps),
        method="rk4",
        solver="explicit_time",
        fit_signal="phi",
        mode_method="project",
        auto_window=False,
        tmin=fit_tmin,
        tmax=fit_tmax,
        sample_stride=int(args.sample_stride),
    )

    t_arr = np.asarray(result.t, dtype=float)
    eig_tmin, eig_tmax = late_time_window(
        t_arr, tail_fraction=float(args.tail_fraction)
    )
    mode_sp = extract_eigenfunction(
        np.asarray(result.field_history),
        t_arr,
        result.selection,
        z=theta_gx,
        method="svd",
        tmin=eig_tmin,
        tmax=eig_tmax,
    )
    mode_sp = _normalize_kbm_mode(theta_gx, np.asarray(mode_sp, dtype=np.complex128))
    metrics = compare_eigenfunctions(mode_sp, mode_gx)
    validation_gate_report = _kbm_eigenfunction_gate_report(metrics)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "z": theta_gx,
            "eigen_real": mode_sp.real,
            "eigen_imag": mode_sp.imag,
            "eigen_abs": np.abs(mode_sp),
        }
    ).to_csv(args.out_csv, index=False)

    fig, _axes = eigenfunction_reference_overlay_figure(
        theta_gx,
        mode_sp,
        theta_gx,
        mode_gx,
        title=f"KBM raw eigenfunction overlay (ky={float(args.ky):.3f})",
    )
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "ky": float(args.ky),
                "ny": int(ny),
                "nky_full": int(nky_full),
                "dt": float(args.dt),
                "steps": int(steps),
                "sample_stride": int(args.sample_stride),
                "fit_tmin": fit_tmin,
                "fit_tmax": fit_tmax,
                "eig_tmin": eig_tmin,
                "eig_tmax": eig_tmax,
                "gamma": float(result.gamma),
                "omega": float(result.omega),
                "overlap": float(metrics.overlap),
                "relative_l2": float(metrics.relative_l2),
                "phase_shift": float(metrics.phase_shift),
                "gate_tolerances": dict(KBM_EIGENFUNCTION_GATE_TOLERANCES),
                "gate_report": gate_report_to_dict(validation_gate_report),
                "eigenfunction_gate_passed": bool(validation_gate_report.passed),
                "validation_status": "closed"
                if validation_gate_report.passed
                else "open",
                "t_final": float(t_arr[-1]),
                "nsamples": int(t_arr.size),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main_w7x(argv: list[str] | None = None) -> int:
    args = build_w7x_parser().parse_args(argv)
    if args.gx_big is not None:
        _freeze_gx_big_reference(args.gx_big, ky=float(args.ky), out=args.bundle_out)
    bundle = _load_finite_reference(args.bundle_out)
    gx_time, _gx_ky, _gx_kx, _gx_omega, _gx_wg, _gx_wphi, _gx_wapar, _gx_phi2 = (
        _load_gx_reference(args.gx)
    )
    output_steps = _build_sample_steps(
        gx_time,
        sample_step_stride=int(args.sample_step_stride),
        max_samples=args.max_samples,
        sample_window=str(args.sample_window),
    )
    reference_times = np.asarray(gx_time[: int(output_steps[-1]) + 1], dtype=float)
    run = _run_w7x_gkx_mode(
        args, reference_times=reference_times, output_steps=output_steps
    )

    if np.asarray(run["theta"]).shape != np.asarray(
        bundle.theta
    ).shape or not np.allclose(run["theta"], bundle.theta):
        mode = np.interp(bundle.theta, run["theta"], run["mode"].real) + 1j * np.interp(
            bundle.theta, run["theta"], run["mode"].imag
        )
    else:
        mode = np.asarray(run["mode"], dtype=np.complex128)
    metrics = compare_eigenfunctions(mode, bundle.mode)
    validation_gate_report = _w7x_eigenfunction_gate_report(metrics)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "z": bundle.theta,
            "eigen_real": mode.real,
            "eigen_imag": mode.imag,
            "eigen_abs": np.abs(mode),
        }
    ).to_csv(args.out_csv, index=False)

    fig, _axes = eigenfunction_reference_overlay_figure(
        bundle.theta,
        mode,
        bundle.theta,
        bundle.mode,
        title=f"W7-X raw eigenfunction overlay (ky={float(args.ky):.3f})",
    )
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "ky": float(args.ky),
                "kx_ref": float(run["kx_ref"]),
                "kx_local": float(run["kx_local"]),
                "ny": int(run["ny"]),
                "Nl": int(run["nl"]),
                "Nm": int(run["nm"]),
                "mode_method": str(args.mode_method),
                "gamma_last": float(run["gamma_last"]),
                "omega_last": float(run["omega_last"]),
                "Wg_last": float(run["Wg_last"]),
                "Wphi_last": float(run["Wphi_last"]),
                "Wapar_last": float(run["Wapar_last"]),
                "Phi2_last": float(run["Phi2_last"]),
                "t_final": float(run["t_final"]),
                "overlap": float(metrics.overlap),
                "relative_l2": float(metrics.relative_l2),
                "phase_shift": float(metrics.phase_shift),
                "gate_tolerances": dict(W7X_EIGENFUNCTION_GATE_TOLERANCES),
                "gate_report": gate_report_to_dict(validation_gate_report),
                "eigenfunction_gate_passed": bool(validation_gate_report.passed),
                "validation_status": "closed"
                if validation_gate_report.passed
                else "open",
                "reference_bundle": _artifact_path(args.bundle_out),
                "gkx_csv": _artifact_path(args.out_csv),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "command", choices=("kbm", "w7x", "overlap-summary", "reference-overlay")
        )
        parser.print_help()
        return 2
    command, rest = tokens[0], tokens[1:]
    if command == "kbm":
        return main_kbm(rest)
    if command == "w7x":
        return main_w7x(rest)
    if command == "overlap-summary":
        return main_overlap_summary(rest)
    if command == "reference-overlay":
        return main_reference_overlay(rest)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
