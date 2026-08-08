#!/usr/bin/env python3
"""Compare GX KBM linear outputs against GKX."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from gkx.diagnostics.analysis import (
    ModeSelection,
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    windowed_growth_rate_from_omega_series,
    select_ky_index,
)
from gkx.benchmarking.shared import KBM_KRYLOV_DEFAULT
from gkx.runtime import run_runtime_linear
from gkx.diagnostics.analysis import branch_continuity_metrics
from gkx.diagnostics.validation_gates import branch_continuity_gate_report, gate_report_to_dict
from gkx.config import KBMBaseCase, GeometryConfig, GridConfig, KineticElectronModelConfig
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.workflows.runtime.toml import load_runtime_from_toml, load_toml

LATE_PROJECT_WINDOW_FRACTION = 0.3
LATE_PROJECT_MIN_POINTS = 80
LATE_PROJECT_START_FRACTION = 0.4
LATE_PROJECT_GROWTH_WEIGHT = 1.0
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class KBMGXInputContract:
    y0: float
    ntheta: int
    nperiod: int
    nlaguerre: int
    nhermite: int
    q: float
    shat: float
    eps: float
    rmaj: float
    beta: float
    ion_tprim: float
    ele_tprim: float
    ion_fprim: float
    ele_fprim: float
    te_over_ti: float
    mass_ratio: float
    init_field: str
    gaussian_init: bool
    init_electrons_only: bool


def _runtime_config_from_kbm_case(cfg: KBMBaseCase):
    """Translate one parsed comparison case into the canonical runtime contract."""

    runtime_cfg, _raw = load_runtime_from_toml(
        ROOT / "examples/linear/axisymmetric/runtime_kbm.toml"
    )
    ion, electron = runtime_cfg.species
    model = cfg.model
    return replace(
        runtime_cfg,
        grid=cfg.grid,
        geometry=cfg.geometry,
        init=cfg.init,
        physics=replace(runtime_cfg.physics, beta=float(model.beta)),
        species=(
            replace(
                ion, tprim=float(model.tprim_i), fprim=float(model.fprim)
            ),
            replace(
                electron,
                mass=1.0 / float(model.mass_ratio),
                temperature=float(model.Te_over_Ti),
                tprim=float(model.tprim_e),
                fprim=float(model.fprim),
            ),
        ),
    )


def _load_kbm_reference_input_contract(path: Path) -> KBMGXInputContract:
    data = load_toml(path)
    dims = data.get("Dimensions", {})
    domain = data.get("Domain", {})
    physics = data.get("Physics", {})
    geometry = data.get("Geometry", {})
    species = data.get("species", {})
    init = data.get("Initialization", {})

    charge = np.asarray(species.get("z", [1.0, -1.0]), dtype=float)
    mass = np.asarray(species.get("mass", [1.0, 2.7e-4]), dtype=float)
    temp = np.asarray(species.get("temp", [1.0, 1.0]), dtype=float)
    tprim = np.asarray(species.get("tprim", [2.49, 2.49]), dtype=float)
    fprim = np.asarray(species.get("fprim", [0.8, 0.8]), dtype=float)
    ele_candidates = np.flatnonzero(charge < 0.0)
    ele_idx = int(ele_candidates[0]) if ele_candidates.size else min(1, charge.size - 1)
    ion_idx = 0 if ele_idx != 0 else min(1, charge.size - 1)
    me = float(mass[ele_idx])
    te = float(temp[ele_idx])
    ti = float(temp[ion_idx])
    rmaj = float(geometry.get("Rmaj", 2.77778))
    return KBMGXInputContract(
        y0=float(domain.get("y0", 10.0)),
        ntheta=int(dims.get("ntheta", 32)),
        nperiod=int(dims.get("nperiod", 2)),
        nlaguerre=int(dims.get("nlaguerre", 16)),
        nhermite=int(dims.get("nhermite", 48)),
        q=float(geometry.get("qinp", geometry.get("q", 1.4))),
        shat=float(geometry.get("shat", 0.8)),
        eps=float(geometry.get("rhoc", 0.5)) / rmaj,
        rmaj=rmaj,
        beta=float(physics.get("beta", 0.015)),
        ion_tprim=float(tprim[ion_idx]),
        ele_tprim=float(tprim[ele_idx]),
        ion_fprim=float(fprim[ion_idx]),
        ele_fprim=float(fprim[ele_idx]),
        te_over_ti=float(te / ti) if ti != 0.0 else 1.0,
        mass_ratio=float(1.0 / me) if me != 0.0 else float("inf"),
        init_field=str(init.get("init_field", "all")),
        gaussian_init=bool(init.get("gaussian_init", True)),
        init_electrons_only=bool(init.get("init_electrons_only", False)),
    )


def _load_gx_omega_gamma(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    def _maybe_scalar_array(data: np.ndarray | float) -> float | None:
        val = float(np.asarray(data))
        if abs(val) < 1.0e-12:
            return None
        return val

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        time = np.asarray(data["time"], dtype=float)
        ky = np.asarray(data["ky"], dtype=float)
        omega = np.asarray(data["omega_series"], dtype=float)
        beta = float(np.asarray(data["beta"]))
        q = None if "q" not in data else _maybe_scalar_array(data["q"])
        shat = None if "shat" not in data else _maybe_scalar_array(data["shat"])
        eps = None if "eps" not in data else _maybe_scalar_array(data["eps"])
        rmaj = None if "rmaj" not in data else _maybe_scalar_array(data["rmaj"])
        return time, ky, omega, beta, q, shat, eps, rmaj

    root = Dataset(path, "r")
    try:
        grids = root.groups["Grids"]
        diagnostics = root.groups["Diagnostics"]
        inputs = root.groups["Inputs"]
    except KeyError as exc:
        raise ValueError(f"{path} missing expected GX groups") from exc

    ky = np.asarray(grids.variables["ky"][:], dtype=float)
    time = np.asarray(grids.variables["time"][:], dtype=float)
    omega_kxkyt = np.asarray(diagnostics.variables["omega_kxkyt"][:], dtype=float)
    if omega_kxkyt.ndim != 4 or omega_kxkyt.shape[-1] != 2:
        raise ValueError(f"unexpected omega_kxkyt shape: {omega_kxkyt.shape}")

    omega = omega_kxkyt[:, :, 0, :]

    beta = float(np.asarray(inputs.variables["beta"][:]))

    def _maybe_scalar(name: str) -> float | None:
        if name not in inputs.variables:
            return None
        data = np.asarray(inputs.variables[name][:])
        if np.ma.is_masked(data):
            return None
        return _maybe_scalar_array(data)

    q = _maybe_scalar("q")
    shat = _maybe_scalar("shat")
    eps = _maybe_scalar("eps")
    rmaj = _maybe_scalar("Rmaj")

    root.close()
    mask = ky > 0.0
    return time, ky[mask], omega[:, mask], beta, q, shat, eps, rmaj


def _infer_y0(ky: np.ndarray) -> float:
    if ky.size < 2:
        raise ValueError("Need at least two ky values to infer y0.")
    ky_min = float(np.min(ky[ky > 0.0]))
    if ky_min <= 0.0:
        raise ValueError("ky array does not contain positive values.")
    return 1.0 / ky_min


def _prepare_gx_reference(
    path: Path,
    *,
    ky_arg: str,
    y0_fallback: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float | None,
    float | None,
    float | None,
    float | None,
    int,
    float,
]:
    gx_time, gx_ky_full, gx_omega_series, beta, q_gx, shat_gx, eps_gx, rmaj_gx = _load_gx_omega_gamma(path)
    nky_full = int(len(gx_ky_full))
    if nky_full < 2:
        raise ValueError("GX output must contain at least two positive ky points.")
    y0 = _infer_y0(gx_ky_full) if len(gx_ky_full) > 1 else float(y0_fallback)
    if ky_arg:
        ky_req = np.asarray([float(k.strip()) for k in ky_arg.split(",") if k.strip()], dtype=float)
        if ky_req.size == 0:
            raise ValueError("No ky values parsed from --ky")
        idx = [int(np.argmin(np.abs(gx_ky_full - k))) for k in ky_req]
        gx_ky = gx_ky_full[idx]
        gx_omega_series = gx_omega_series[:, idx]
    else:
        gx_ky = gx_ky_full
    return gx_time, gx_ky, gx_omega_series, beta, q_gx, shat_gx, eps_gx, rmaj_gx, nky_full, y0


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


def _load_gx_eigenfunction(path: Path, ky_target: float) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        theta = np.asarray(data["theta"], dtype=float)
        ky = np.asarray(data["ky"], dtype=float)
        modes = np.asarray(data["phi_modes"])
        ky_idx = int(np.argmin(np.abs(ky - float(ky_target))))
        mode = np.asarray(modes[ky_idx], dtype=np.complex128)
        return theta, _normalize_mode(theta, mode)

    root = Dataset(path, "r")
    grids = root.groups["Grids"]
    diag = root.groups["Diagnostics"]
    theta = np.asarray(grids.variables["theta"][:], dtype=float)
    ky = np.asarray(grids.variables["ky"][:], dtype=float)
    ky_idx = int(np.argmin(np.abs(ky - float(ky_target))))
    phi = np.asarray(diag.variables["Phi"][-1, ky_idx, 0, :, :], dtype=float)
    root.close()
    mode = phi[:, 0] + 1j * phi[:, 1]
    return theta, _normalize_mode(theta, mode)


def _trajectory_path(base_dir: Path, ky_value: float) -> Path:
    tag = f"{float(ky_value):0.4f}".replace(".", "p")
    return base_dir / f"kbm_ky_{tag}_trajectory.npz"


def _field_history(result) -> np.ndarray:
    """Return fresh runtime fields or a historical cached trajectory."""

    history = getattr(result, "field_history", None)
    if history is None:
        history = getattr(result, "phi_t", None)
    if history is None:
        raise ValueError("linear result does not contain a field trajectory")
    return np.asarray(history)


def _save_trajectory(path: Path, result) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    gamma_t = getattr(result, "gamma_t", None)
    omega_t = getattr(result, "omega_t", None)
    np.savez_compressed(
        path,
        t=np.asarray(result.t, dtype=float),
        phi_t=_field_history(result),
        gamma_t=np.asarray(gamma_t, dtype=float) if gamma_t is not None else np.array([], dtype=float),
        omega_t=np.asarray(omega_t, dtype=float) if omega_t is not None else np.array([], dtype=float),
        ky=float(result.ky),
        sel_ky=int(result.selection.ky_index),
        sel_kx=int(result.selection.kx_index),
        sel_z=int(result.selection.z_index),
    )
    return path


def _load_trajectory(path: Path):
    data = np.load(path, allow_pickle=False)
    gamma_t = np.asarray(data["gamma_t"], dtype=float)
    omega_t = np.asarray(data["omega_t"], dtype=float)
    return SimpleNamespace(
        t=np.asarray(data["t"], dtype=float),
        field_history=np.asarray(data["phi_t"]),
        gamma=float("nan"),
        omega=float("nan"),
        ky=float(np.asarray(data["ky"])),
        gamma_t=None if gamma_t.size == 0 else gamma_t,
        omega_t=None if omega_t.size == 0 else omega_t,
        selection=ModeSelection(
            ky_index=int(np.asarray(data["sel_ky"])),
            kx_index=int(np.asarray(data["sel_kx"])),
            z_index=int(np.asarray(data["sel_z"])),
        ),
    )


def _mode_overlap(lhs: np.ndarray, rhs: np.ndarray) -> float:
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom <= 0.0:
        return float("nan")
    return float(np.abs(np.vdot(lhs, rhs)) / denom)


def _mode_rel_l2(lhs: np.ndarray, rhs: np.ndarray) -> float:
    rhs_norm = float(np.linalg.norm(rhs))
    if rhs_norm <= 0.0:
        return float("nan")
    phase = np.vdot(lhs, rhs)
    lhs_align = lhs if abs(phase) <= 1.0e-30 else lhs * np.exp(-1j * np.angle(phase))
    return float(np.linalg.norm(lhs_align - rhs) / rhs_norm)


def _interp_complex(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y.real) + 1j * np.interp(x_new, x, y.imag)


def _candidate_objective(
    *,
    rel_gamma: float,
    rel_omega: float,
    eig_overlap_gx: float,
    eig_overlap_prev: float,
    gamma_weight: float,
    omega_weight: float,
    gx_overlap_weight: float,
    prev_overlap_weight: float,
) -> float:
    obj = gamma_weight * rel_gamma + omega_weight * rel_omega
    if np.isfinite(eig_overlap_gx):
        obj += gx_overlap_weight * (1.0 - eig_overlap_gx)
    if np.isfinite(eig_overlap_prev):
        obj += prev_overlap_weight * (1.0 - eig_overlap_prev)
    return float(obj)


def _extract_mode(
    result,
    theta: np.ndarray,
    *,
    method: str,
    tmin: float | None,
    tmax: float | None,
) -> np.ndarray:
    phi_t = _field_history(result)
    t = np.asarray(result.t, dtype=float)
    if t.size <= 1:
        return _normalize_mode(theta, np.asarray(phi_t[-1, 0, 0, :], dtype=np.complex128))
    mode = extract_eigenfunction(
        phi_t,
        t,
        result.selection,
        z=theta,
        method=method,
        tmin=tmin,
        tmax=tmax,
    )
    return _normalize_mode(theta, np.asarray(mode, dtype=np.complex128))


def _mode_metrics(
    result,
    *,
    grid_full,
    ky_value: float,
    gx_big: Path,
    eigen_method: str,
    eigen_tmin: float | None,
    eigen_tmax: float | None,
    prev_mode: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    ky_idx = select_ky_index(np.asarray(grid_full.ky), float(ky_value))
    grid = select_ky_grid(grid_full, ky_idx)
    theta = np.asarray(grid.z, dtype=float)
    if eigen_tmin is not None or eigen_tmax is not None:
        eig_tmin = eigen_tmin
        eig_tmax = eigen_tmax
    elif np.asarray(result.t).size > 1:
        eig_tmin = float(np.asarray(result.t)[-1]) * 0.6
        eig_tmax = float(np.asarray(result.t)[-1])
    else:
        eig_tmin = None
        eig_tmax = None

    mode_sp = _extract_mode(
        result,
        theta,
        method=eigen_method,
        tmin=eig_tmin,
        tmax=eig_tmax,
    )
    if gx_big.exists():
        theta_gx, mode_gx = _load_gx_eigenfunction(gx_big, float(ky_value))
        if theta_gx.shape != theta.shape or not np.allclose(theta_gx, theta, atol=1.0e-6, rtol=1.0e-6):
            mode_gx = _normalize_mode(theta, _interp_complex(theta, theta_gx, mode_gx))
        eig_overlap = _mode_overlap(mode_sp, mode_gx)
        eig_rel_l2 = _mode_rel_l2(mode_sp, mode_gx)
    else:
        eig_overlap = float("nan")
        eig_rel_l2 = float("nan")
    prev_overlap = float("nan") if prev_mode is None else _mode_overlap(mode_sp, prev_mode)
    return theta, mode_sp, eig_overlap, eig_rel_l2, prev_overlap


def _build_cfg(
    *,
    beta: float,
    q: float,
    shat: float,
    eps: float,
    rmaj: float,
    ny: int,
    ntheta: int,
    nperiod: int,
    y0: float,
    mass_ratio: float,
    ion_tprim: float,
    ele_tprim: float,
    ion_fprim: float,
    ele_fprim: float,
    te_over_ti: float,
    init_field: str,
    gaussian_init: bool,
    init_electrons_only: bool,
) -> KBMBaseCase:
    grid = GridConfig(
        Nx=1,
        Ny=ny,
        Nz=ntheta * (2 * nperiod - 1),
        Lx=62.8,
        Ly=62.8,
        y0=y0,
        ntheta=ntheta,
        nperiod=nperiod,
        boundary="linked",
    )
    geom = GeometryConfig(
        q=q,
        s_hat=shat,
        epsilon=eps,
        R0=rmaj,
    )
    model = KineticElectronModelConfig(
        tprim_i=ion_tprim,
        tprim_e=ele_tprim,
        fprim=ele_fprim,
        Te_over_Ti=te_over_ti,
        mass_ratio=mass_ratio,
        nu_i=0.0,
        nu_e=0.0,
        beta=beta,
    )
    return KBMBaseCase(
        grid=grid,
        geometry=geom,
        model=model,
        init=replace(
            KBMBaseCase().init,
            init_field=init_field,
            gaussian_init=gaussian_init,
            init_electrons_only=init_electrons_only,
        ),
    )


def _replace_result(result, /, **updates):
    if is_dataclass(result):
        field_names = {field.name for field in fields(result)}
        replace_updates = {name: value for name, value in updates.items() if name in field_names}
        extra_updates = {name: value for name, value in updates.items() if name not in field_names}
        result_new = replace(result, **replace_updates)
        if not extra_updates:
            return result_new
        values = dict(vars(result_new))
        values.update(extra_updates)
        return SimpleNamespace(**values)
    values = dict(vars(result))
    values.update(updates)
    return SimpleNamespace(**values)


def _split_mode_method_policy(mode_method: str) -> tuple[str, str]:
    if mode_method.endswith("_late"):
        return mode_method[: -len("_late")], "late"
    return mode_method, "default"


def _benchmark_solver_for_candidate(solver_name: str) -> str:
    """Map comparison-candidate labels to canonical GKX solver keys."""

    return "explicit_time" if solver_name == "gx_time" else solver_name


def _run_candidate(
    args,
    cfg: KBMBaseCase,
    ky_value: float,
    beta_value: float,
    solver_name: str,
    *,
    mode_method_override: str | None = None,
    gx_gamma: float | None = None,
    gx_omega: float | None = None,
):
    fit_signal = args.time_fit_signal if solver_name in {"time", "gx_time"} else "phi"
    benchmark_solver = _benchmark_solver_for_candidate(solver_name)
    mode_method_use = str(mode_method_override or args.mode_method)
    mode_method_base, _fit_policy = _split_mode_method_policy(mode_method_use)
    krylov_cfg = None
    if (
        solver_name == "krylov"
        and bool(getattr(args, "krylov_gx_shift", False))
        and gx_gamma is not None
        and gx_omega is not None
        and np.isfinite(gx_gamma)
        and np.isfinite(gx_omega)
    ):
        krylov_cfg = replace(
            KBM_KRYLOV_DEFAULT,
            shift=complex(float(gx_gamma), -float(gx_omega)),
            shift_source=str(getattr(args, "krylov_gx_shift_source", "target")),
            shift_selection="shift",
            omega_sign=0,
            omega_target_factor=0.0,
        )
    runtime_cfg = _runtime_config_from_kbm_case(cfg)
    runtime_cfg = replace(
        runtime_cfg,
        physics=replace(runtime_cfg.physics, beta=float(beta_value)),
        time=replace(
            runtime_cfg.time,
            fixed_dt=solver_name != "gx_time",
        ),
    )
    return run_runtime_linear(
        runtime_cfg,
        ky_target=float(ky_value),
        Nl=args.Nl,
        Nm=args.Nm,
        dt=args.dt,
        steps=args.steps,
        method=args.method,
        solver=benchmark_solver,
        krylov_cfg=krylov_cfg,
        fit_signal=fit_signal,
        mode_method=mode_method_base,
        auto_window=not args.no_auto_window,
        tmin=args.tmin,
        tmax=args.tmax,
        sample_stride=args.sample_stride,
    )


def _recompute_time_history_growth(args, result, *, mode_method: str):
    """Recompute growth rates from a cached time-history result."""

    t = np.asarray(result.t, dtype=float)
    if t.size <= 1:
        return result

    mode_method_base, fit_policy = _split_mode_method_policy(mode_method)

    if mode_method_base in {"project", "svd"}:
        signal = extract_mode_time_series(_field_history(result), result.selection, method=mode_method_base)
        if args.tmin is not None and args.tmax is not None:
            gamma, omega = fit_growth_rate(signal=signal, t=t, tmin=args.tmin, tmax=args.tmax)
            fit_tmin = float(args.tmin)
            fit_tmax = float(args.tmax)
        else:
            if fit_policy == "late":
                gamma, omega, fit_tmin, fit_tmax = fit_growth_rate_auto(
                    t,
                    signal,
                    window_method="fixed",
                    window_fraction=LATE_PROJECT_WINDOW_FRACTION,
                    min_points=LATE_PROJECT_MIN_POINTS,
                    start_fraction=LATE_PROJECT_START_FRACTION,
                    growth_weight=LATE_PROJECT_GROWTH_WEIGHT,
                    require_positive=True,
                    min_amp_fraction=0.0,
                )
            else:
                gamma, omega, fit_tmin, fit_tmax = fit_growth_rate_auto(
                    t,
                    signal,
                    window_method="fixed",
                    window_fraction=0.4,
                    min_points=40,
                    start_fraction=0.2,
                    growth_weight=1.0,
                    require_positive=True,
                    min_amp_fraction=0.0,
                )
        return _replace_result(
            result,
            gamma=float(gamma),
            omega=float(omega),
            fit_window_tmin=float(fit_tmin),
            fit_window_tmax=float(fit_tmax),
        )

    if args.tmin is not None and args.tmax is not None:
        try:
            signal = extract_mode_time_series(_field_history(result), result.selection, method=mode_method_base)
            gamma, omega = fit_growth_rate(signal=signal, t=t, tmin=args.tmin, tmax=args.tmax)
            return _replace_result(
                result,
                gamma=float(gamma),
                omega=float(omega),
                fit_window_tmin=float(args.tmin),
                fit_window_tmax=float(args.tmax),
            )
        except ValueError:
            pass

    try:
        gamma, omega, _g_t, _o_t, _t_mid = instantaneous_growth_rate_from_phi(
            _field_history(result),
            t,
            result.selection,
            navg_fraction=float(args.gx_avg_fraction),
            mode_method=mode_method_base,
        )
        fit_tmin = float(t[-1]) * (1.0 - float(args.gx_avg_fraction))
        fit_tmax = float(t[-1])
        return _replace_result(
            result,
            gamma=float(gamma),
            omega=float(omega),
            fit_window_tmin=fit_tmin,
            fit_window_tmax=fit_tmax,
        )
    except ValueError:
        signal = extract_mode_time_series(_field_history(result), result.selection, method=mode_method_base)
        gamma, omega, fit_tmin, fit_tmax = fit_growth_rate_auto(
            t,
            signal,
            window_fraction=0.4,
            min_points=40,
            start_fraction=0.2,
            growth_weight=1.0,
            require_positive=True,
            min_amp_fraction=0.0,
        )
        return _replace_result(
            result,
            gamma=float(gamma),
            omega=float(omega),
            fit_window_tmin=float(fit_tmin),
            fit_window_tmax=float(fit_tmax),
        )


def _interp_phi_t(phi_t: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    phi_src = np.asarray(phi_t, dtype=np.complex128)
    if np.array_equal(t_src, t_dst):
        return phi_src
    flat = phi_src.reshape(phi_src.shape[0], -1)
    out = np.empty((t_dst.size, flat.shape[1]), dtype=np.complex128)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, flat[:, j].real) + 1j * np.interp(
            t_dst, t_src, flat[:, j].imag
        )
    return out.reshape((t_dst.size,) + phi_src.shape[1:])


def _interp_real_tseries(data_t: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    data_src = np.asarray(data_t, dtype=float)
    if np.array_equal(t_src, t_dst):
        return data_src
    flat = data_src.reshape(data_src.shape[0], -1)
    out = np.empty((t_dst.size, flat.shape[1]), dtype=float)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, flat[:, j])
    return out.reshape((t_dst.size,) + data_src.shape[1:])


def _recompute_time_history_growth_on_grid(
    args,
    result,
    *,
    mode_method: str,
    t_ref: np.ndarray | None = None,
):
    mode_method_base, _fit_policy = _split_mode_method_policy(mode_method)
    gamma_t = getattr(result, "gamma_t", None)
    omega_t = getattr(result, "omega_t", None)
    if (
        mode_method_base in {"z_index", "max"}
        and gamma_t is not None
        and omega_t is not None
    ):
        gamma_arr = np.asarray(gamma_t, dtype=float)
        omega_arr = np.asarray(omega_t, dtype=float)
        t_src = np.asarray(result.t, dtype=float)
        if t_ref is not None and np.asarray(t_ref).size > 1 and t_src.size > 1:
            t_dst = np.asarray(t_ref, dtype=float)
            gamma_arr = _interp_real_tseries(gamma_arr, t_src, t_dst)
            omega_arr = _interp_real_tseries(omega_arr, t_src, t_dst)
        gamma, omega, _g_t, _o_t = windowed_growth_rate_from_omega_series(
            gamma_arr,
            omega_arr,
            result.selection,
            navg_fraction=float(args.gx_avg_fraction),
        )
        fit_tmin = float(t_dst[-1]) * (1.0 - float(args.gx_avg_fraction)) if t_ref is not None else float(t_src[-1]) * (
            1.0 - float(args.gx_avg_fraction)
        )
        fit_tmax = float(t_dst[-1]) if t_ref is not None else float(t_src[-1])
        return _replace_result(
            result,
            gamma=float(gamma),
            omega=float(omega),
            fit_window_tmin=fit_tmin,
            fit_window_tmax=fit_tmax,
        )
    if mode_method_base in {"project", "svd"}:
        return _recompute_time_history_growth(args, result, mode_method=mode_method)
    if t_ref is None or np.asarray(t_ref).size <= 1:
        return _recompute_time_history_growth(args, result, mode_method=mode_method)
    t_src = np.asarray(result.t, dtype=float)
    t_dst = np.asarray(t_ref, dtype=float)
    if t_src.size <= 1:
        return result
    phi_t = _interp_phi_t(_field_history(result), t_src, t_dst)
    sampled = _replace_result(result, t=t_dst, field_history=phi_t)
    updated = _recompute_time_history_growth(args, sampled, mode_method=mode_method)
    return _replace_result(result, gamma=float(updated.gamma), omega=float(updated.omega))


def _run_candidate_cached(
    args,
    cfg: KBMBaseCase,
    ky_value: float,
    beta_value: float,
    solver_name: str,
    *,
    mode_method_override: str | None,
    result_cache: dict[tuple[object, ...], object],
    gx_time_ref: np.ndarray | None = None,
    gx_gamma: float | None = None,
    gx_omega: float | None = None,
):
    mode_method_use = str(mode_method_override or args.mode_method)
    cache_key: tuple[object, ...]
    if solver_name == "gx_time":
        cache_key = ("gx_time", float(ky_value), float(beta_value), args.dt, args.steps, args.method)
        if cache_key not in result_cache:
            traj_dir = getattr(args, "trajectory_dir", None)
            reuse_traj = bool(getattr(args, "reuse_trajectory", False))
            traj_path = None if traj_dir is None else _trajectory_path(Path(traj_dir), float(ky_value))
            if reuse_traj and traj_path is not None and traj_path.exists():
                result_cache[cache_key] = _load_trajectory(traj_path)
            else:
                result_cache[cache_key] = _run_candidate(
                    args,
                    cfg,
                    ky_value,
                    beta_value,
                    solver_name,
                    mode_method_override="z_index",
                    gx_gamma=gx_gamma,
                    gx_omega=gx_omega,
                )
                if traj_path is not None:
                    _save_trajectory(traj_path, result_cache[cache_key])
        return _recompute_time_history_growth_on_grid(
            args,
            result_cache[cache_key],
            mode_method=mode_method_use,
            t_ref=gx_time_ref,
        )
    return _run_candidate(
        args,
        cfg,
        ky_value,
        beta_value,
        solver_name,
        mode_method_override=mode_method_override,
        gx_gamma=gx_gamma,
        gx_omega=gx_omega,
    )


def _parse_candidate_spec(spec: str) -> tuple[str, str | None, str]:
    label = spec.strip()
    if not label:
        raise ValueError("Empty branch candidate spec")
    if "@" not in label:
        return label, None, label
    solver_name, mode_method = label.split("@", 1)
    solver_name = solver_name.strip()
    mode_method = mode_method.strip()
    if not solver_name:
        raise ValueError(f"Missing solver name in candidate spec '{spec}'")
    if not mode_method:
        raise ValueError(f"Missing mode method in candidate spec '{spec}'")
    return solver_name, mode_method, label


def _write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _branch_gate_report_from_rows(
    rows: list[dict[str, object]],
    *,
    max_rel_gamma_jump: float = 0.5,
    max_rel_omega_jump: float = 0.5,
    min_successive_overlap: float | None = 0.95,
) -> dict[str, object] | None:
    """Build a branch-continuity gate report from selected KBM scan rows."""

    if len(rows) < 2:
        return None
    table = pd.DataFrame(rows).sort_values("ky")
    overlap: np.ndarray | None = None
    if "eig_overlap_prev" in table.columns:
        overlap_values = np.asarray(table["eig_overlap_prev"], dtype=float)[1:]
        if overlap_values.size == table.shape[0] - 1 and np.all(np.isfinite(overlap_values)):
            overlap = overlap_values
    metrics = branch_continuity_metrics(
        np.asarray(table["ky"], dtype=float),
        np.asarray(table["gamma"], dtype=float),
        np.asarray(table["omega"], dtype=float),
        successive_overlap=overlap,
    )
    report = branch_continuity_gate_report(
        metrics,
        case="kbm_linear_branch_continuity",
        source="selected KBM comparison rows",
        max_rel_gamma_jump=float(max_rel_gamma_jump),
        max_rel_omega_jump=float(max_rel_omega_jump),
        min_successive_overlap=min_successive_overlap if overlap is not None else None,
    )
    return gate_report_to_dict(report)


def _candidate_row(
    *,
    ky: float,
    solver: str,
    result,
    gx_gamma: float,
    gx_omega: float,
    eig_overlap_gx: float,
    eig_rel_l2: float,
    eig_overlap_prev: float,
    branch_score: float,
    selected: bool,
) -> dict[str, float | str | bool]:
    gamma = float(result.gamma)
    omega = float(result.omega)
    rel_gamma = abs(gamma - float(gx_gamma)) / max(abs(float(gx_gamma)), 1.0e-12)
    rel_omega = abs(omega - float(gx_omega)) / max(abs(float(gx_omega)), 1.0e-12)
    return {
        "ky": float(ky),
        "solver": solver,
        "gamma_gx": float(gx_gamma),
        "gamma": gamma,
        "rel_gamma": rel_gamma,
        "omega_gx": float(gx_omega),
        "omega": omega,
        "rel_omega": rel_omega,
        "eig_overlap_gx": eig_overlap_gx,
        "eig_rel_l2": eig_rel_l2,
        "eig_overlap_prev": eig_overlap_prev,
        "branch_score": branch_score,
        "selected": bool(selected),
        "fit_window_tmin": float(getattr(result, "fit_window_tmin", float("nan"))),
        "fit_window_tmax": float(getattr(result, "fit_window_tmax", float("nan"))),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare GX KBM output against GKX.")
    parser.add_argument("--gx", type=Path, required=True, help="Path to GX .out.nc file")
    parser.add_argument(
        "--gx-big",
        type=Path,
        default=Path(".cache/gx/kbm_salpha.big.nc"),
        help="Path to GX .big.nc file for eigenfunction comparisons",
    )
    parser.add_argument("--Nl", type=int, default=None)
    parser.add_argument("--Nm", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of GKX time steps. Defaults to the GX reference horizon rounded up by --dt.",
    )
    parser.add_argument("--method", type=str, default="rk4")
    parser.add_argument("--solver", type=str, default="gx_time")
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--nperiod", type=int, default=2)
    parser.add_argument("--y0", type=float, default=10.0, help="Fallback y0 when ky list is truncated.")
    parser.add_argument("--gx-input", type=Path, default=None, help="Optional GX input file for exact benchmark contract overrides.")
    parser.add_argument(
        "--nky",
        type=int,
        default=None,
        help="Override GX nky when ky list is truncated (sets Ny = 3*(nky-1)+1).",
    )
    parser.add_argument(
        "--gx-avg-fraction",
        type=float,
        default=0.5,
        help="Average GX omega/gamma over the last fraction of time samples.",
    )
    parser.add_argument("--q", type=float, default=1.4)
    parser.add_argument("--shat", type=float, default=0.8)
    parser.add_argument("--eps", type=float, default=0.18)
    parser.add_argument("--Rmaj", type=float, default=2.77778)
    parser.add_argument(
        "--ky",
        type=str,
        default="",
        help="Comma-separated ky values to compare (default: all from GX output)",
    )
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=None,
        help="Optional directory for cached GX-time KBM trajectory npz files.",
    )
    parser.add_argument(
        "--reuse-trajectory",
        action="store_true",
        help="Reuse cached GX-time trajectories from --trajectory-dir when available.",
    )
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--no-auto-window", action="store_true")
    parser.add_argument(
        "--branch-policy",
        type=str,
        choices=["fixed", "single", "gx-ref-auto", "auto", "continuation"],
        default="continuation",
        help="fixed/single: use one solver for all ky; gx-ref-auto/auto: legacy GX-scored solver selection; continuation: choose the most continuous candidate branch across ky.",
    )
    parser.add_argument(
        "--branch-solvers",
        type=str,
        default="gx_time@project,gx_time@project_late,gx_time@svd,gx_time@svd_late,gx_time@max,gx_time@z_index,krylov,time",
        help="Comma-separated candidate solvers used when --branch-policy=gx-ref-auto. Candidates can override time/gx_time extraction with the form solver@mode_method, e.g. gx_time@max.",
    )
    parser.add_argument("--branch-gamma-weight", type=float, default=1.0)
    parser.add_argument("--branch-omega-weight", type=float, default=1.0)
    parser.add_argument("--branch-overlap-gx-weight", type=float, default=1.0)
    parser.add_argument("--branch-overlap-prev-weight", type=float, default=2.0)
    parser.add_argument(
        "--krylov-gx-shift",
        action="store_true",
        help="When evaluating a Krylov candidate, use the GX reference eigenvalue as the shift target.",
    )
    parser.add_argument(
        "--krylov-gx-shift-source",
        type=str,
        default="target",
        choices=["target", "power", "propagator"],
        help="Seed source to pair with --krylov-gx-shift when probing an explicit shift-invert target.",
    )
    parser.add_argument(
        "--time-fit-signal",
        type=str,
        default="auto",
        choices=["auto", "phi", "density"],
        help="Fit signal used for time/gx_time candidate runs.",
    )
    parser.add_argument(
        "--mode-method",
        type=str,
        default="project",
        choices=["z_index", "max", "project", "svd", "project_late", "svd_late"],
        help="Mode-extraction method for reference-time/fallback fits. The default projects onto the late-time KBM structure.",
    )
    parser.add_argument(
        "--eigen-method",
        type=str,
        default="svd",
        choices=["svd", "snapshot"],
        help="Method used to extract GKX eigenfunctions from time histories.",
    )
    parser.add_argument("--eigen-tmin", type=float, default=None)
    parser.add_argument("--eigen-tmax", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV path for mismatch table")
    parser.add_argument(
        "--candidate-out",
        type=Path,
        default=None,
        help="Optional CSV path for per-candidate branch metrics (one row per ky/solver).",
    )
    parser.add_argument(
        "--branch-summary-json",
        type=Path,
        default=None,
        help="Optional JSON branch-continuity gate report for the selected KBM scan rows.",
    )
    parser.add_argument("--branch-max-rel-gamma-jump", type=float, default=0.5)
    parser.add_argument("--branch-max-rel-omega-jump", type=float, default=0.5)
    parser.add_argument("--branch-min-successive-overlap", type=float, default=0.95)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    gx_time, gx_ky, gx_omega_series, beta, q_gx, shat_gx, eps_gx, rmaj_gx, nky_full, y0 = _prepare_gx_reference(
        args.gx,
        ky_arg=str(args.ky),
        y0_fallback=float(args.y0),
    )
    gx_input_contract = None if args.gx_input is None else _load_kbm_reference_input_contract(args.gx_input)
    args.Nl = int(args.Nl) if args.Nl is not None else int(16 if gx_input_contract is None else gx_input_contract.nlaguerre)
    args.Nm = int(args.Nm) if args.Nm is not None else int(48 if gx_input_contract is None else gx_input_contract.nhermite)
    nky = int(args.nky) if args.nky is not None else nky_full
    ny = 3 * (nky - 1) + 1

    if gx_omega_series.ndim != 3:
        raise ValueError(f"Unexpected GX omega series shape: {gx_omega_series.shape}")
    start = int((1.0 - float(args.gx_avg_fraction)) * gx_omega_series.shape[0])
    start = max(0, min(start, gx_omega_series.shape[0] - 1))
    gx_avg = np.mean(gx_omega_series[start:, :, :], axis=0)
    gx_omega = gx_avg[:, 0]
    gx_gamma = gx_avg[:, 1]

    cfg = _build_cfg(
        beta=float(beta if gx_input_contract is None else gx_input_contract.beta),
        q=float(q_gx if q_gx is not None else (args.q if gx_input_contract is None else gx_input_contract.q)),
        shat=float(
            shat_gx if shat_gx is not None else (args.shat if gx_input_contract is None else gx_input_contract.shat)
        ),
        eps=float(eps_gx if eps_gx is not None else (args.eps if gx_input_contract is None else gx_input_contract.eps)),
        rmaj=float(
            rmaj_gx if rmaj_gx is not None else (args.Rmaj if gx_input_contract is None else gx_input_contract.rmaj)
        ),
        ny=ny,
        ntheta=int(args.ntheta if gx_input_contract is None else gx_input_contract.ntheta),
        nperiod=int(args.nperiod if gx_input_contract is None else gx_input_contract.nperiod),
        y0=float(y0 if gx_input_contract is None else gx_input_contract.y0),
        mass_ratio=float(3703.7037037037035 if gx_input_contract is None else gx_input_contract.mass_ratio),
        ion_tprim=float(2.49 if gx_input_contract is None else gx_input_contract.ion_tprim),
        ele_tprim=float(2.49 if gx_input_contract is None else gx_input_contract.ele_tprim),
        ion_fprim=float(0.8 if gx_input_contract is None else gx_input_contract.ion_fprim),
        ele_fprim=float(0.8 if gx_input_contract is None else gx_input_contract.ele_fprim),
        te_over_ti=float(1.0 if gx_input_contract is None else gx_input_contract.te_over_ti),
        init_field=str("all" if gx_input_contract is None else gx_input_contract.init_field),
        gaussian_init=bool(True if gx_input_contract is None else gx_input_contract.gaussian_init),
        init_electrons_only=bool(False if gx_input_contract is None else gx_input_contract.init_electrons_only),
    )
    steps_use = int(args.steps) if args.steps is not None else max(int(np.ceil(float(gx_time[-1]) / float(args.dt))), 1)
    args.steps = steps_use

    grid_full = build_spectral_grid(cfg.grid)
    use_legacy_auto = args.branch_policy in {"gx-ref-auto", "auto"}
    use_continuation = args.branch_policy == "continuation"
    branch_candidates = [_parse_candidate_spec(s) for s in args.branch_solvers.split(",") if s.strip()]
    if (use_legacy_auto or use_continuation) and not branch_candidates:
        raise ValueError("No candidate solvers parsed from --branch-solvers")

    rows: list[dict[str, float | str]] = []
    candidate_rows: list[dict[str, float | str | bool]] = []
    prev_mode: np.ndarray | None = None
    for i, ky_val in enumerate(gx_ky):
        branch_score = float("nan")
        result_cache: dict[tuple[object, ...], object] = {}
        if use_legacy_auto:
            best_row = None
            best_obj = np.inf
            candidate_start = len(candidate_rows)
            for solver_name, mode_method_override, solver_label in branch_candidates:
                result = _run_candidate_cached(
                    args,
                    cfg,
                    float(ky_val),
                    beta,
                    solver_name,
                    mode_method_override=mode_method_override,
                    result_cache=result_cache,
                    gx_time_ref=gx_time,
                    gx_gamma=float(gx_gamma[i]),
                    gx_omega=float(gx_omega[i]),
                )
                rel_g = abs(float(result.gamma) - float(gx_gamma[i])) / max(abs(float(gx_gamma[i])), 1.0e-12)
                rel_o = abs(float(result.omega) - float(gx_omega[i])) / max(abs(float(gx_omega[i])), 1.0e-12)
                obj = rel_g + rel_o
                candidate_rows.append(
                    _candidate_row(
                        ky=float(ky_val),
                        solver=solver_label,
                        result=result,
                        gx_gamma=float(gx_gamma[i]),
                        gx_omega=float(gx_omega[i]),
                        eig_overlap_gx=float("nan"),
                        eig_rel_l2=float("nan"),
                        eig_overlap_prev=float("nan"),
                        branch_score=float(obj),
                        selected=False,
                    )
                )
                if args.candidate_out is not None:
                    _write_rows(args.candidate_out, candidate_rows)
                if obj < best_obj:
                    best_obj = obj
                    best_row = (solver_label, result)
            if best_row is None:
                raise RuntimeError(f"No valid solver candidate for ky={float(ky_val):.4f}")
            solver_used, result = best_row
            for row in candidate_rows[candidate_start:]:
                row["selected"] = row["solver"] == solver_used
        elif use_continuation:
            best_candidate = None
            best_obj = np.inf
            candidate_start = len(candidate_rows)
            for solver_name, mode_method_override, solver_label in branch_candidates:
                result_c = _run_candidate_cached(
                    args,
                    cfg,
                    float(ky_val),
                    beta,
                    solver_name,
                    mode_method_override=mode_method_override,
                    result_cache=result_cache,
                    gx_time_ref=gx_time,
                    gx_gamma=float(gx_gamma[i]),
                    gx_omega=float(gx_omega[i]),
                )
                _theta, mode_c, eig_overlap_c, eig_rel_l2_c, prev_overlap_c = _mode_metrics(
                    result_c,
                    grid_full=grid_full,
                    ky_value=float(ky_val),
                    gx_big=args.gx_big,
                    eigen_method=args.eigen_method,
                    eigen_tmin=args.eigen_tmin,
                    eigen_tmax=args.eigen_tmax,
                    prev_mode=prev_mode,
                )
                rel_g = abs(float(result_c.gamma) - float(gx_gamma[i])) / max(abs(float(gx_gamma[i])), 1.0e-12)
                rel_o = abs(float(result_c.omega) - float(gx_omega[i])) / max(abs(float(gx_omega[i])), 1.0e-12)
                obj = _candidate_objective(
                    rel_gamma=rel_g,
                    rel_omega=rel_o,
                    eig_overlap_gx=eig_overlap_c,
                    eig_overlap_prev=prev_overlap_c,
                    gamma_weight=float(args.branch_gamma_weight),
                    omega_weight=float(args.branch_omega_weight),
                    gx_overlap_weight=float(args.branch_overlap_gx_weight),
                    prev_overlap_weight=float(args.branch_overlap_prev_weight),
                )
                candidate_rows.append(
                    _candidate_row(
                        ky=float(ky_val),
                        solver=solver_label,
                        result=result_c,
                        gx_gamma=float(gx_gamma[i]),
                        gx_omega=float(gx_omega[i]),
                        eig_overlap_gx=eig_overlap_c,
                        eig_rel_l2=eig_rel_l2_c,
                        eig_overlap_prev=prev_overlap_c,
                        branch_score=float(obj),
                        selected=False,
                    )
                )
                if args.candidate_out is not None:
                    _write_rows(args.candidate_out, candidate_rows)
                if obj < best_obj:
                    best_obj = obj
                    best_candidate = (solver_label, result_c, mode_c, eig_overlap_c, eig_rel_l2_c, prev_overlap_c)
            if best_candidate is None:
                raise RuntimeError(f"No valid solver candidate for ky={float(ky_val):.4f}")
            solver_used, result, mode_sp, eig_overlap, eig_rel_l2, prev_overlap = best_candidate
            branch_score = float(best_obj)
            for row in candidate_rows[candidate_start:]:
                row["selected"] = row["solver"] == solver_used
        else:
            solver_used = args.solver
            result = _run_candidate_cached(
                args,
                cfg,
                float(ky_val),
                beta,
                solver_used,
                mode_method_override=None,
                result_cache=result_cache,
                gx_time_ref=gx_time,
                gx_gamma=float(gx_gamma[i]),
                gx_omega=float(gx_omega[i]),
            )
            candidate_rows.append(
                _candidate_row(
                    ky=float(ky_val),
                    solver=solver_used,
                    result=result,
                    gx_gamma=float(gx_gamma[i]),
                    gx_omega=float(gx_omega[i]),
                    eig_overlap_gx=float("nan"),
                    eig_rel_l2=float("nan"),
                    eig_overlap_prev=float("nan"),
                    branch_score=float("nan"),
                    selected=True,
                )
            )
            if args.candidate_out is not None:
                _write_rows(args.candidate_out, candidate_rows)

        if not use_continuation:
            _theta, mode_sp, eig_overlap, eig_rel_l2, prev_overlap = _mode_metrics(
                result,
                grid_full=grid_full,
                ky_value=float(ky_val),
                gx_big=args.gx_big,
                eigen_method=args.eigen_method,
                eigen_tmin=args.eigen_tmin,
                eigen_tmax=args.eigen_tmax,
                prev_mode=prev_mode,
            )
        prev_mode = mode_sp

        fit_window_tmin = float(getattr(result, "fit_window_tmin", float("nan")))
        fit_window_tmax = float(getattr(result, "fit_window_tmax", float("nan")))
        if not np.isfinite(fit_window_tmin) or not np.isfinite(fit_window_tmax):
            t_series = np.asarray(result.t, dtype=float)
            if t_series.size > 1 and solver_used == "gx_time":
                fit_window_tmin = float(t_series[-1]) * (1.0 - float(args.gx_avg_fraction))
                fit_window_tmax = float(t_series[-1])
            elif args.tmin is not None and args.tmax is not None:
                fit_window_tmin = float(args.tmin)
                fit_window_tmax = float(args.tmax)

        row = {
            "ky": float(ky_val),
            "solver": solver_used,
            "gamma_gx": float(gx_gamma[i]),
            "gamma": float(result.gamma),
            "rel_gamma": abs(float(result.gamma) - float(gx_gamma[i])) / max(abs(float(gx_gamma[i])), 1.0e-12),
            "omega_gx": float(gx_omega[i]),
            "omega": float(result.omega),
            "rel_omega": abs(float(result.omega) - float(gx_omega[i])) / max(abs(float(gx_omega[i])), 1.0e-12),
            "eig_overlap_gx": eig_overlap,
            "eig_rel_l2": eig_rel_l2,
            "eig_overlap_prev": prev_overlap,
            "branch_score": branch_score,
            "fit_window_tmin": fit_window_tmin,
            "fit_window_tmax": fit_window_tmax,
        }
        rows.append(row)
        if args.out is not None:
            _write_rows(args.out, rows)
        if args.candidate_out is not None:
            _write_rows(args.candidate_out, candidate_rows)

    table = pd.DataFrame(rows)
    print(
        "ky    solver      gx_gamma   sp_gamma   rel_gamma    gx_omega   sp_omega   rel_omega   eig_ovlp   eig_rel"
    )
    for entry in table.itertuples(index=False):
        print(
            f"{entry.ky:5.3f} {entry.solver:10s} {entry.gamma_gx:9.5f} {entry.gamma:9.5f} {entry.rel_gamma:9.3f} "
            f"{entry.omega_gx:9.5f} {entry.omega:9.5f} {entry.rel_omega:9.3f} {entry.eig_overlap_gx:9.3f} {entry.eig_rel_l2:9.3f}"
        )

    if args.out is not None:
        _write_rows(args.out, rows)
    if args.candidate_out is not None:
        _write_rows(args.candidate_out, candidate_rows)
    if args.branch_summary_json is not None:
        report = _branch_gate_report_from_rows(
            rows,
            max_rel_gamma_jump=float(args.branch_max_rel_gamma_jump),
            max_rel_omega_jump=float(args.branch_max_rel_omega_jump),
            min_successive_overlap=float(args.branch_min_successive_overlap),
        )
        args.branch_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.branch_summary_json.write_text(
            json.dumps(
                {
                    "rows": rows,
                    "gate_report": report,
                    "gate_passed": None if report is None else bool(report["passed"]),
                    "notes": (
                        "Branch-continuity gate built from selected KBM rows. "
                        "A null gate_report means fewer than two ky samples were available."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
