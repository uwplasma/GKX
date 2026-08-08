"""Generate CSV tables for documentation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from pprint import pformat
import argparse
import csv
import sys
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gkx.benchmarking.shared import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    REFERENCE_DAMP_ENDS_AMP,
    REFERENCE_DAMP_ENDS_WIDTHFRAC,
    KINETIC_KRYLOV_REFERENCE_ALIGNED,
    TEM_KRYLOV_DEFAULT,
    _apply_reference_hypercollisions,
    _build_initial_condition,
    _midplane_index,
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_tem_reference,
    LinearScanResult,
)
from gkx.config import (
    CycloneBaseCase,
    GridConfig,
    TimeConfig,
)
from gkx.geometry import SAlphaGeometry
from gkx.runtime import run_runtime_linear, run_runtime_scan
from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit,
)
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.solvers.linear.krylov import KrylovConfig
from gkx.diagnostics.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate_auto,
    select_ky_index,
)

KINETIC_SOLVER = "krylov"
ETG_SOLVER = "time"
TEM_SOLVER = "time"
DIAGNOSTIC_NORM = "rho_star"
DEFAULT_RUN_KW = {"diagnostic_norm": DIAGNOSTIC_NORM}


KINETIC_KRYLOV = KINETIC_KRYLOV_REFERENCE_ALIGNED
TEM_KRYLOV = TEM_KRYLOV_DEFAULT


def _build_rows(scan, ref):
    rows = ["ky,gamma_ref,omega_ref,gamma_gkx,omega_gkx,rel_gamma,rel_omega"]
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        gamma_ref = float(ref.gamma[idx])
        omega_ref = float(ref.omega[idx])
        rel_gamma = (gamma - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
        rel_omega = (omega - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
        rows.append(
            f"{ky:.3f},{gamma_ref:.6f},{omega_ref:.6f},{gamma:.6f},{omega:.6f},{rel_gamma:.3f},{rel_omega:.3f}"
        )
    return rows


REQUIRED_LASTVALUE_SCAN_COLUMNS = {
    "ky",
    "gamma_last",
    "omega_last",
    "gamma_ref_last",
    "omega_ref_last",
}


def _load_lastvalue_scan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_LASTVALUE_SCAN_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df.copy()


def _build_lastvalue_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert imported-linear scan diagnostics into the last-value table."""

    out = pd.DataFrame(
        {
            "ky": df["ky"].astype(float),
            "gamma": df["gamma_last"].astype(float),
            "omega": df["omega_last"].astype(float),
            "gamma_gx": df["gamma_ref_last"].astype(float),
            "omega_gx": df["omega_ref_last"].astype(float),
        }
    )
    out["rel_gamma"] = (out["gamma"] - out["gamma_gx"]) / out["gamma_gx"].where(
        out["gamma_gx"] != 0.0
    )
    out["rel_omega"] = (out["omega"] - out["omega_gx"]) / out["omega_gx"].where(
        out["omega_gx"] != 0.0
    )
    return out.sort_values("ky").reset_index(drop=True)


def _write_lastvalue_table(scan: Path, out: Path) -> pd.DataFrame:
    df = _build_lastvalue_table(_load_lastvalue_scan(scan.expanduser().resolve()))
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"saved {out}")
    return df


def _rows_from_reference_columns(
    x: np.ndarray,
    gamma_ref: np.ndarray,
    omega_ref: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
) -> list[str]:
    rows = ["ky,gamma_ref,omega_ref,gamma_gkx,omega_gkx,rel_gamma,rel_omega"]
    for ky, g_ref, w_ref, g, w in zip(x, gamma_ref, omega_ref, gamma, omega):
        rel_gamma = (g - g_ref) / g_ref if g_ref != 0.0 else np.nan
        rel_omega = (w - w_ref) / w_ref if w_ref != 0.0 else np.nan
        rows.append(
            f"{float(ky):.3f},{float(g_ref):.6f},{float(w_ref):.6f},{float(g):.6f},{float(w):.6f},{float(rel_gamma):.3f},{float(rel_omega):.3f}"
        )
    return rows


def _kbm_public_rows_from_gx_mismatch(
    csv_path: Path, lowky_ckpt_path: Path | None = None
) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"no rows found in {csv_path}")
    selected_rows = [
        row
        for row in rows
        if str(row.get("selected", "")).strip().lower() in {"true", "1", "yes"}
    ]
    if selected_rows:
        rows = selected_rows
    by_ky = {float(row["ky"]): dict(row) for row in rows}
    if lowky_ckpt_path is not None and lowky_ckpt_path.exists():
        with lowky_ckpt_path.open("r", encoding="utf-8", newline="") as handle:
            ckpt_rows = list(csv.DictReader(handle))
        for row in ckpt_rows:
            ky_val = float(row["ky"])
            current = by_ky.get(ky_val)
            if current is None:
                continue
            current_score = abs(float(current["rel_gamma"])) + abs(
                float(current["rel_omega"])
            )
            candidate_score = abs(float(row["rel_gamma"])) + abs(
                float(row["rel_omega"])
            )
            if candidate_score + 1.0e-12 >= current_score:
                continue
            by_ky[ky_val] = {
                "ky": row["ky"],
                "solver": row["solver"],
                "gamma_gx": row["gamma_gx"],
                "gamma": row["gamma"],
                "rel_gamma": row["rel_gamma"],
                "omega_gx": row["omega_gx"],
                "omega": row["omega"],
                "rel_omega": row["rel_omega"],
                "eig_overlap_gx": row.get("eig_overlap_gx", ""),
                "eig_rel_l2": row.get("eig_rel_l2", ""),
                "eig_overlap_prev": "",
                "branch_score": "",
                "fit_window_tmin": "",
                "fit_window_tmax": "",
            }
    rows = sorted(by_ky.values(), key=lambda row: float(row["ky"]))
    ky = np.array([float(row["ky"]) for row in rows], dtype=float)
    gamma_ref = np.array([float(row["gamma_gx"]) for row in rows], dtype=float)
    omega_ref = np.array([float(row["omega_gx"]) for row in rows], dtype=float)
    gamma = np.array([float(row["gamma"]) for row in rows], dtype=float)
    omega = np.array([float(row["omega"]) for row in rows], dtype=float)
    return _rows_from_reference_columns(ky, gamma_ref, omega_ref, gamma, omega)


def _write_kbm_public_mismatch_table(
    outdir: Path,
    *,
    verbose: bool,
    progress: bool,
    stiff_spot_check: bool,
    stiff_spot_topk: int,
    stiff_spot_dt: float,
    stiff_spot_tmax: float,
    stiff_spot_replace: bool,
) -> None:
    kbm_table = outdir / "kbm_mismatch_table.csv"
    comparison_dir = outdir / "comparison"
    kbm_candidates = comparison_dir / "kbm_reference_candidates.csv"
    kbm_reference_mismatch = (
        kbm_candidates
        if kbm_candidates.exists()
        else comparison_dir / "kbm_reference_mismatch.csv"
    )
    if not kbm_reference_mismatch.exists():
        legacy_reference_mismatch = outdir / "kbm_reference_mismatch.csv"
        if legacy_reference_mismatch.exists():
            kbm_reference_mismatch = legacy_reference_mismatch
    kbm_lowky_ckpt = outdir / "kbm_probe_lowky_ckpt.csv"
    if kbm_reference_mismatch.exists():
        kbm_table.write_text(
            "\n".join(
                _kbm_public_rows_from_gx_mismatch(
                    kbm_reference_mismatch,
                    lowky_ckpt_path=(
                        None if kbm_reference_mismatch == kbm_candidates else kbm_lowky_ckpt
                    ),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return

    raise FileNotFoundError(
        "KBM publication tables require comparison/kbm_reference_candidates.csv "
        "or comparison/kbm_reference_mismatch.csv; "
        "regenerate it with tools/comparison/compare_gx_kbm.py"
    )


def _cyclone_refresh_reference(ref: LinearScanResult) -> LinearScanResult:
    keep = np.asarray(ref.ky) <= 0.45 + 1.0e-12
    return LinearScanResult(
        ky=np.asarray(ref.ky)[keep],
        gamma=np.asarray(ref.gamma)[keep],
        omega=np.asarray(ref.omega)[keep],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validation tables.")
    parser.add_argument(
        "--case",
        choices=["all", "cyclone", "etg"],
        default="all",
        help="Limit table generation to a specific case.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging.")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable tqdm progress bars."
    )
    parser.add_argument(
        "--stiff-spot-check",
        action="store_true",
        help=(
            "After Krylov scans, re-run the worst-mismatch high-ky points with a "
            "robust implicit time integrator (GMRES + hermite-line preconditioner) "
            "to diagnose instability-driven outliers."
        ),
    )
    parser.add_argument(
        "--stiff-spot-check-topk",
        type=int,
        default=2,
        help="Number of outlier points to re-run per scan when --stiff-spot-check is enabled.",
    )
    parser.add_argument(
        "--stiff-spot-check-min-ky",
        type=float,
        default=0.5,
        help="Only consider ky >= this threshold when selecting stiff spot-check points.",
    )
    parser.add_argument(
        "--stiff-spot-check-dt",
        type=float,
        default=0.01,
        help="Fixed dt to use for implicit stiff spot-check runs.",
    )
    parser.add_argument(
        "--stiff-spot-check-tmax",
        type=float,
        default=4.0,
        help="Total time horizon to use for implicit stiff spot-check runs.",
    )
    parser.add_argument(
        "--stiff-spot-check-replace",
        action="store_true",
        help="Replace Krylov results with implicit spot-check results when they reduce mismatch.",
    )
    parser.add_argument(
        "--refresh-minimal",
        action="store_true",
        help="Only build the tracked outputs required by the benchmark refresh workflow.",
    )
    parser.add_argument(
        "--lastvalue-scan",
        type=Path,
        help="Imported-linear scan CSV to convert into a last-value mismatch table.",
    )
    parser.add_argument(
        "--lastvalue-out",
        type=Path,
        help="Output CSV for --lastvalue-scan conversion.",
    )
    return parser.parse_args()


def _log(msg: str, *, verbose: bool, use_tqdm: bool) -> None:
    if not verbose:
        return
    if use_tqdm:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _format_cfg(cfg) -> str:
    if is_dataclass(cfg):
        return pformat(asdict(cfg), width=120, sort_dicts=False)
    return pformat(cfg, width=120, sort_dicts=False)


def _window_value(val, idx: int) -> float | None:
    if val is None:
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        return float(val[idx])
    return float(val)


def _run_runtime_linear_adapter(
    *,
    cfg,
    ky_target: float,
    Nl: int,
    Nm: int,
    dt: float,
    steps: int,
    method: str,
    solver: str,
    krylov_cfg=None,
    time_cfg: TimeConfig | None = None,
    diagnostic_norm: str | None = None,
    **fit_options,
):
    """Adapt table-scan controls to the canonical runtime linear API."""

    del diagnostic_norm
    cfg_use = replace(cfg, time=time_cfg) if time_cfg is not None else cfg
    return run_runtime_linear(
        cfg_use,
        ky_target=ky_target,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        solver=solver,
        krylov_cfg=krylov_cfg,
        **fit_options,
    )


def _scan_linear_verbose(
    *,
    ky_values: np.ndarray,
    run_linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    label: str,
    run_kwargs: dict | None = None,
    ref=None,
    verbose: bool,
    progress: bool,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], KrylovConfig] | None = None,
    stiff_spot_check: bool = False,
    stiff_spot_check_topk: int = 0,
    stiff_spot_check_min_ky: float = 0.0,
    stiff_spot_check_dt: float = 0.01,
    stiff_spot_check_tmax: float = 4.0,
    stiff_spot_check_replace: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _log(f"\n=== {label} scan ===", verbose=verbose, use_tqdm=progress)
    _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
    _log(
        f"Numerics: Nl={Nl} Nm={Nm} method={method} solver={solver} dt={dt} steps={steps}",
        verbose=verbose,
        use_tqdm=progress,
    )
    _log(f"Window params: {window_kw}", verbose=verbose, use_tqdm=progress)
    if run_kwargs:
        _log(f"Extra kwargs: {run_kwargs}", verbose=verbose, use_tqdm=progress)
    if tmin is not None or tmax is not None:
        _log(
            f"Manual window tmin={tmin} tmax={tmax}", verbose=verbose, use_tqdm=progress
        )

    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    mismatch_scores: list[float] = []
    ref_pairs: list[tuple[float, float]] = []
    base_extra = dict(DEFAULT_RUN_KW)
    if run_kwargs:
        base_extra.update(run_kwargs)
    iterator = tqdm(ky_values, desc=f"{label} ky scan") if progress else ky_values
    for i, ky in enumerate(iterator):
        if resolution_policy is not None:
            Nl_i, Nm_i = resolution_policy(float(ky))
        else:
            Nl_i, Nm_i = int(Nl), int(Nm)
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        tmin_i = _window_value(tmin, i)
        tmax_i = _window_value(tmax, i)
        _log(
            f"[{label}] start ky={float(ky):.4g} dt={dt_i:.4g} steps={steps_i} tmax={dt_i * steps_i:.4g}",
            verbose=verbose,
            use_tqdm=progress,
        )
        extra = dict(base_extra)
        krylov_cfg_use = (
            krylov_policy(float(ky)) if krylov_policy is not None else krylov_cfg
        )
        try:
            result = run_linear_fn(
                ky_target=float(ky),
                cfg=cfg,
                Nl=int(Nl_i),
                Nm=int(Nm_i),
                dt=dt_i,
                steps=steps_i,
                method=method,
                solver=solver,
                krylov_cfg=krylov_cfg_use,
                auto_window=auto_window,
                tmin=tmin_i,
                tmax=tmax_i,
                **window_kw,
                **extra,
            )
            gamma_val = float(result.gamma)
            omega_val = float(result.omega)
            ky_val = float(result.ky)
            msg = f"[{label}] done ky={ky_val:.4g} gamma={gamma_val:.6g} omega={omega_val:.6g}"
        except Exception as exc:
            gamma_val = float("nan")
            omega_val = float("nan")
            ky_val = float(ky)
            msg = f"[{label}] failed ky={ky_val:.4g} error={type(exc).__name__}: {exc}"
        gammas.append(gamma_val)
        omegas.append(omega_val)
        ky_out.append(ky_val)
        if ref is not None:
            idx = int(np.argmin(np.abs(ref.ky - ky_val)))
            gamma_ref = float(ref.gamma[idx])
            omega_ref = float(ref.omega[idx])
            rel_gamma = (
                (gamma_val - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
            )
            rel_omega = (
                (omega_val - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
            )
            mismatch_scores.append(float(np.nanmax(np.abs([rel_gamma, rel_omega]))))
            ref_pairs.append((gamma_ref, omega_ref))
            msg += (
                f" | ref gamma={gamma_ref:.6g} omega={omega_ref:.6g}"
                f" rel_gamma={rel_gamma:.3g} rel_omega={rel_omega:.3g}"
            )
        _log(msg, verbose=verbose, use_tqdm=progress)

    if (
        stiff_spot_check
        and str(solver).lower() == "krylov"
        and ref is not None
        and stiff_spot_check_topk > 0
        and len(ky_out) == len(mismatch_scores)
    ):
        ky_arr = np.asarray(ky_out)
        score_arr = np.asarray(mismatch_scores)
        eligible = ky_arr >= float(stiff_spot_check_min_ky)
        if np.any(eligible):
            eligible_idx = np.where(eligible)[0]
            ranked = eligible_idx[np.argsort(score_arr[eligible_idx])[::-1]]
            top_idx = ranked[: int(stiff_spot_check_topk)]
            _log(
                f"\n[{label}] stiff spot-check (implicit + hermite-line) for {len(top_idx)} outliers",
                verbose=verbose,
                use_tqdm=progress,
            )
            for local_i in top_idx:
                ky_val = float(ky_arr[local_i])
                gamma_ref, omega_ref = ref_pairs[local_i]
                gamma_k = float(gammas[local_i])
                omega_k = float(omegas[local_i])
                t_max = float(stiff_spot_check_tmax)
                dt_spot = float(stiff_spot_check_dt)
                steps_spot = max(int(round(t_max / dt_spot)), 1)
                time_cfg_spot = TimeConfig(
                    t_max=t_max,
                    dt=dt_spot,
                    method="implicit",
                    use_diffrax=False,
                    implicit_preconditioner="hermite-line",
                    progress_bar=False,
                    sample_stride=1,
                )
                extra_spot = dict(base_extra)
                extra_spot.pop("time_cfg", None)
                spot = run_linear_fn(
                    ky_target=ky_val,
                    cfg=cfg,
                    Nl=int(Nl),
                    Nm=int(Nm),
                    dt=dt_spot,
                    steps=steps_spot,
                    method="implicit",
                    solver="time",
                    krylov_cfg=None,
                    time_cfg=time_cfg_spot,
                    auto_window=True,
                    tmin=None,
                    tmax=None,
                    **window_kw,
                    **extra_spot,
                )
                gamma_i = float(spot.gamma)
                omega_i = float(spot.omega)
                rel_k = (
                    abs((gamma_k - gamma_ref) / gamma_ref)
                    if gamma_ref != 0.0
                    else np.inf
                )
                rel_i = (
                    abs((gamma_i - gamma_ref) / gamma_ref)
                    if gamma_ref != 0.0
                    else np.inf
                )
                relw_k = (
                    abs((omega_k - omega_ref) / omega_ref)
                    if omega_ref != 0.0
                    else np.inf
                )
                relw_i = (
                    abs((omega_i - omega_ref) / omega_ref)
                    if omega_ref != 0.0
                    else np.inf
                )
                _log(
                    f"[{label}] ky={ky_val:.4g} krylov(g={gamma_k:.4g}, w={omega_k:.4g}) "
                    f"implicit(g={gamma_i:.4g}, w={omega_i:.4g}) "
                    f"| ref(g={gamma_ref:.4g}, w={omega_ref:.4g}) "
                    f"rel_g(k={rel_k:.3g}, i={rel_i:.3g}) rel_w(k={relw_k:.3g}, i={relw_i:.3g})",
                    verbose=verbose,
                    use_tqdm=progress,
                )
                if stiff_spot_check_replace and (rel_i + relw_i) < (rel_k + relw_k):
                    gammas[local_i] = gamma_i
                    omegas[local_i] = omega_i
                    _log(
                        f"[{label}] replaced krylov result at ky={ky_val:.4g} with implicit spot-check.",
                        verbose=verbose,
                        use_tqdm=progress,
                    )

    return np.array(ky_out), np.array(gammas), np.array(omegas)


WINDOWS = {
    "cyclone": dict(
        window_fraction=0.3, min_points=80, start_fraction=0.58,
        growth_weight=0.0, require_positive=True, min_amp_fraction=0.05,
        max_fraction=0.6, end_fraction=0.8, max_amp_fraction=0.8,
        late_penalty=0.3, window_method="loglinear", mode_method="project",
    ),
    "kinetic": dict(
        window_fraction=0.3, min_points=160, start_fraction=0.45,
        growth_weight=0.1, require_positive=True, min_amp_fraction=0.05,
    ),
    "etg": dict(
        window_fraction=0.25, min_points=120, start_fraction=0.4,
        growth_weight=0.2, require_positive=True, min_amp_fraction=0.1,
    ),
    "kbm": dict(
        window_fraction=0.3, min_points=120, start_fraction=0.35,
        growth_weight=0.0, require_positive=False, min_amp_fraction=0.05,
    ),
    "tem": dict(
        window_fraction=0.35, min_points=120, start_fraction=0.5,
        growth_weight=0.2, require_positive=True, min_amp_fraction=0.1,
    ),
}

REFERENCE_CYCLONE_WINDOW = dict(
    window_method="loglinear", min_points=40, start_fraction=0.1,
    max_fraction=0.8, end_fraction=0.8, require_positive=True,
    min_amp_fraction=0.05, max_amp_fraction=0.8, growth_weight=0.1,
    late_penalty=0.1,
)

CYCLONE_PUBLIC_TIME = TimeConfig(
    t_max=150.0, dt=0.01, use_diffrax=True, diffrax_solver="Tsit5",
    diffrax_adaptive=False, diffrax_rtol=1.0e-4, diffrax_atol=1.0e-7,
    diffrax_max_steps=20000, progress_bar=False, fixed_dt=True,
)
CYCLONE_PUBLIC_NL = 16
CYCLONE_PUBLIC_NM = 48


def _gx_balanced_policy(ky: float) -> tuple[int, int, float]:
    if ky < 0.08:
        return 16, 8, 320.0
    if ky < 0.15:
        return 16, 8, 80.0
    if ky <= 0.25:
        return 24, 12, 40.0
    return 24, 12, 10.0


def _gx_mode_policy(ky: float) -> str:
    return "max" if ky < 0.3 else "project"


def _gx_window_policy(ky: float, base_window: dict) -> dict:
    window = dict(base_window)
    if ky < 0.08:
        window["start_fraction"] = 0.65
        window["end_fraction"] = 0.95
        window["min_points"] = max(int(window.get("min_points", 0)), 80)
        window["min_slope_frac"] = 0.25
        window["late_penalty"] = 0.0
    if ky >= 0.3:
        window["start_fraction"] = 0.3
        window["end_fraction"] = 0.9
        window["min_amp_fraction"] = 0.0
        window["max_amp_fraction"] = 1.0
        window["late_penalty"] = 0.0
    return window


def _fit_cyclone_reference_signal(
    *,
    t: np.ndarray,
    phi_t: np.ndarray,
    sel: ModeSelection,
    ky_val: float,
    window_kw: dict,
) -> tuple[float, float, float, float, str]:
    gamma_floor = 1.0e-6
    omega_floor = 1.0e-6
    methods = [_gx_mode_policy(ky_val)]
    if methods[0] != "max":
        methods.append("max")
    last: tuple[float, float, float, float, str] | None = None
    for method in methods:
        signal = extract_mode_time_series(phi_t, sel, method=method)
        fit_kw = _gx_window_policy(ky_val, window_kw)
        fit_kw = {k: v for k, v in fit_kw.items() if k != "mode_method"}
        gamma, omega, tmin, tmax_fit = fit_growth_rate_auto(t, signal, **fit_kw)
        last = (float(gamma), float(omega), float(tmin), float(tmax_fit), method)
        if (
            np.isfinite(gamma)
            and np.isfinite(omega)
            and gamma > gamma_floor
            and abs(omega) > omega_floor
        ):
            return last
    assert last is not None
    return last


def _cyclone_reference_scan(
    ky_values: np.ndarray,
    cfg: CycloneBaseCase,
    window_kw: dict,
    *,
    verbose: bool,
    progress: bool,
) -> LinearScanResult:
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_full = build_spectral_grid(cfg.grid)
    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    iterator = tqdm(ky_values, desc="Cyclone GX ky scan") if progress else ky_values
    for ky in iterator:
        ky_val = float(ky)
        Nl, Nm, tmax = _gx_balanced_policy(ky_val)
        params = LinearParams(
            fprim=cfg.model.fprim,
            tprim=cfg.model.tprim_i,
            tprim_e=cfg.model.tprim_e,
            omega_d_scale=CYCLONE_OMEGA_D_SCALE,
            omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
            rho_star=CYCLONE_RHO_STAR,
            kpar_scale=float(geom.gradpar()),
            nu=cfg.model.nu_i,
            damp_ends_amp=REFERENCE_DAMP_ENDS_AMP,
            damp_ends_widthfrac=REFERENCE_DAMP_ENDS_WIDTHFRAC,
        )
        params = _apply_reference_hypercollisions(params, nhermite=Nm)
        terms = LinearTerms()

        ky_idx = select_ky_index(np.asarray(grid_full.ky), ky_val)
        grid = select_ky_grid(grid_full, ky_idx)
        G0 = _build_initial_condition(
            grid,
            geom,
            ky_index=0,
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            init_cfg=cfg.init,
        )
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        time_cfg = ExplicitTimeConfig(
            t_max=tmax,
            dt=0.01,
            fixed_dt=False,
            dt_min=1.0e-7,
            dt_max=0.1,
            cfl_fac=0.3,
        )
        _log(
            f"[Cyclone GX] ky={ky_val:.3f} Nl={Nl} Nm={Nm} tmax={tmax}",
            verbose=verbose,
            use_tqdm=progress,
        )
        t, phi_t, _gamma_t, _omega_t = integrate_linear_explicit(
            G0,
            grid,
            cache,
            params,
            geom,
            time_cfg,
            terms,
            mode_method="z_index",
            z_index=_midplane_index(grid),
        )
        sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
        gamma, omega, tmin, tmax_fit, mode_method = _fit_cyclone_reference_signal(
            t=np.asarray(t),
            phi_t=np.asarray(phi_t),
            sel=sel,
            ky_val=ky_val,
            window_kw=window_kw,
        )
        _log(
            f"[Cyclone GX] ky={ky_val:.3f} method={mode_method} fit=[{tmin:.3g}, {tmax_fit:.3g}] "
            f"gamma={gamma:.6g} omega={omega:.6g}",
            verbose=verbose,
            use_tqdm=progress,
        )
        ky_out.append(ky_val)
        gammas.append(float(gamma))
        omegas.append(float(omega))
    return LinearScanResult(
        ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas)
    )


def _cyclone_reference_mismatch_scan(
    ref: LinearScanResult,
    cfg,
    *,
    verbose: bool,
    progress: bool,
) -> LinearScanResult:
    _log("\n=== Cyclone mismatch scan ===", verbose=verbose, use_tqdm=progress)
    _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
    _log(
        "Numerics: canonical TOML-backed combined-ky runtime scan",
        verbose=verbose,
        use_tqdm=progress,
    )
    scan = _runtime_cyclone_scan(
        cfg,
        np.asarray(ref.ky),
        Nl=CYCLONE_PUBLIC_NL,
        Nm=CYCLONE_PUBLIC_NM,
    )
    for ky_val, gamma_val, omega_val in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky_val)))
        gamma_ref = float(ref.gamma[idx])
        omega_ref = float(ref.omega[idx])
        rel_gamma = (
            (float(gamma_val) - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
        )
        rel_omega = (
            (float(omega_val) - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
        )
        _log(
            f"[Cyclone mismatch] done ky={float(ky_val):.4g} gamma={float(gamma_val):.6g} omega={float(omega_val):.6g}"
            f" | ref gamma={gamma_ref:.6g} omega={omega_ref:.6g}"
            f" rel_gamma={rel_gamma:.3g} rel_omega={rel_omega:.3g}",
            verbose=verbose,
            use_tqdm=progress,
        )
    return LinearScanResult(
        ky=np.asarray(scan.ky),
        gamma=np.asarray(scan.gamma),
        omega=np.asarray(scan.omega),
    )


def _runtime_cyclone_scan(
    cfg,
    ky_values: np.ndarray,
    *,
    Nl: int,
    Nm: int,
    progress: bool = False,
) -> LinearScanResult:
    ky = np.asarray(ky_values, dtype=float)
    gamma = np.empty_like(ky)
    omega = np.empty_like(ky)

    # Slowly growing low-ky modes need twice the transient horizon. Keeping the
    # two groups batched avoids paying that cost for the converged main branch.
    for mask, steps in ((ky < 0.15, 17160), (ky >= 0.15, 8580)):
        if not np.any(mask):
            continue
        scan = run_runtime_scan(
            cfg,
            ky[mask],
            Nl=Nl,
            Nm=Nm,
            dt=0.004663,
            steps=steps,
            method="rk4",
            solver="time",
            batch_ky=True,
            sample_stride=10,
            auto_window=True,
            fit_signal="phi",
            mode_method="z_index",
            window_fraction=0.3,
            min_points=80,
            start_fraction=0.58,
            growth_weight=0.0,
            require_positive=True,
            min_amp_fraction=0.05,
            show_progress=progress,
        )
        gamma[mask] = np.asarray(scan.gamma)
        omega[mask] = np.asarray(scan.omega)
    return LinearScanResult(
        ky=ky,
        gamma=gamma,
        omega=omega,
    )


def _etg_reference_mismatch_scan(
    ref: LinearScanResult,
    cfg,
    *,
    verbose: bool,
    progress: bool,
) -> LinearScanResult:
    _log("\n=== ETG mismatch scan ===", verbose=verbose, use_tqdm=progress)
    _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
    _log(
        "Numerics: canonical TOML-backed ETG runtime scan",
        verbose=verbose,
        use_tqdm=progress,
    )
    scan = run_runtime_scan(
        cfg,
        np.asarray(ref.ky, dtype=float),
        Nl=24,
        Nm=8,
        solver="time",
        batch_ky=True,
        method=cfg.time.method,
        dt=cfg.time.dt,
        steps=int(round(cfg.time.t_max / cfg.time.dt)),
        sample_stride=cfg.time.sample_stride,
        auto_window=False,
        tmin=1.0,
        tmax=cfg.time.t_max,
        fit_signal="phi",
        mode_method="z_index",
        show_progress=progress,
    )
    for ky_val, gamma_val, omega_val in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky_val)))
        gamma_ref = float(ref.gamma[idx])
        omega_ref = float(ref.omega[idx])
        rel_gamma = (
            (float(gamma_val) - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
        )
        rel_omega = (
            (float(omega_val) - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
        )
        _log(
            f"[ETG mismatch] done ky={float(ky_val):.4g} gamma={float(gamma_val):.6g} omega={float(omega_val):.6g}"
            f" | ref gamma={gamma_ref:.6g} omega={omega_ref:.6g}"
            f" rel_gamma={rel_gamma:.3g} rel_omega={rel_omega:.3g}",
            verbose=verbose,
            use_tqdm=progress,
        )
    return LinearScanResult(
        ky=np.asarray(scan.ky),
        gamma=np.asarray(scan.gamma),
        omega=np.asarray(scan.omega),
    )


def _scale_steps(
    ky: np.ndarray, base_steps: int, ky_ref: float, max_steps: int
) -> np.ndarray:
    scale = ky_ref / np.maximum(ky, 1.0e-6)
    steps = base_steps * np.maximum(1.0, scale)
    return np.clip(steps.astype(int), base_steps, max_steps)


def _scale_dt(ky: np.ndarray, base_dt: float, ky_ref: float) -> np.ndarray:
    scale = np.minimum(1.0, ky_ref / np.maximum(ky, 1.0e-6))
    return base_dt * scale


def _etg_time_controls(
    ky: np.ndarray,
    *,
    base_dt: float = 0.01,
    ky_ref: float = 20.0,
    base_steps: int = 500,
    max_steps: int = 2000,
    tmin_frac: float = 0.2,
    tmax_frac: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = _scale_dt(ky, base_dt=base_dt, ky_ref=ky_ref)
    steps = _scale_steps(ky, base_steps=base_steps, ky_ref=ky_ref, max_steps=max_steps)
    t_total = dt * steps
    tmin = tmin_frac * t_total
    tmax = tmax_frac * t_total
    return dt, steps, tmin, tmax


def _etg_runtime_case():
    cfg, _ = load_runtime_from_toml(ROOT / "examples/linear/axisymmetric/etg.toml")
    return cfg


def _run_etg_tables(*, outdir: Path, verbose: bool, progress: bool) -> None:
    base_cfg = _etg_runtime_case()
    etg_R = np.array([4.0, 6.0, 8.0, 10.0])
    etg_rows = ["tprim_e,gamma,omega"]
    for R in etg_R:
        electron = replace(base_cfg.species[0], tprim=float(R))
        cfg = replace(base_cfg, species=(electron,))
        _log(
            f"\n=== ETG trend a/LTe={float(R):.2f} ===",
            verbose=verbose,
            use_tqdm=progress,
        )
        _log(f"Config:\n{_format_cfg(cfg)}", verbose=verbose, use_tqdm=progress)
        res = run_runtime_linear(
            cfg,
            ky_target=5.0,
            Nl=24,
            Nm=8,
            steps=5000,
            dt=4.0e-4,
            method="rk4",
            sample_stride=10,
            solver="time",
            mode_method="z_index",
            fit_signal="phi",
            auto_window=False,
            tmin=1.0,
            tmax=2.0,
        )
        etg_rows.append(f"{R:.2f},{res.gamma:.6f},{res.omega:.6f}")
        _log(
            f"[ETG trend] gamma={res.gamma:.6g} omega={res.omega:.6g}",
            verbose=verbose,
            use_tqdm=progress,
        )
    (outdir / "etg_trend_table.csv").write_text(
        "\n".join(etg_rows) + "\n", encoding="utf-8"
    )

    etg_ref = load_etg_reference()
    etg_mismatch = _etg_reference_mismatch_scan(
        etg_ref,
        base_cfg,
        verbose=verbose,
        progress=progress,
    )
    (outdir / "etg_mismatch_table.csv").write_text(
        "\n".join(_build_rows(etg_mismatch, etg_ref)) + "\n", encoding="utf-8"
    )


def _run_tem_tables(*, outdir: Path, verbose: bool, progress: bool) -> None:
    tem_ref = load_tem_reference()
    tem_dt = _scale_dt(tem_ref.ky, base_dt=0.001, ky_ref=0.3)
    tem_steps = _scale_steps(tem_ref.ky, base_steps=2000, ky_ref=0.3, max_steps=6000)
    tem_tmax = tem_dt * tem_steps
    tem_tmin = 0.4 * tem_tmax
    tem_tmax = 0.85 * tem_tmax
    tem_cfg, _raw = load_runtime_from_toml(
        ROOT / "examples" / "linear" / "axisymmetric" / "runtime_tem.toml"
    )
    tem_ky, tem_g, tem_w = _scan_linear_verbose(
        ky_values=tem_ref.ky,
        run_linear_fn=_run_runtime_linear_adapter,
        cfg=tem_cfg,
        Nl=48,
        Nm=16,
        dt=tem_dt,
        steps=tem_steps,
        method="imex2",
        solver=TEM_SOLVER,
        krylov_cfg=TEM_KRYLOV,
        window_kw=WINDOWS["tem"],
        tmin=tem_tmin,
        tmax=tem_tmax,
        auto_window=False,
        run_kwargs={"mode_method": "z_index"},
        label="TEM mismatch",
        ref=tem_ref,
        verbose=verbose,
        progress=progress,
        stiff_spot_check=False,
        stiff_spot_check_topk=0,
        stiff_spot_check_min_ky=0.0,
        stiff_spot_check_dt=0.0,
        stiff_spot_check_tmax=0.0,
        stiff_spot_check_replace=False,
    )
    tem_mismatch = LinearScanResult(ky=tem_ky, gamma=tem_g, omega=tem_w)
    (outdir / "tem_mismatch_table.csv").write_text(
        "\n".join(_build_rows(tem_mismatch, tem_ref)) + "\n", encoding="utf-8"
    )


def _run_kinetic_tables(
    *,
    outdir: Path,
    verbose: bool,
    progress: bool,
    stiff_spot_check: bool,
    stiff_spot_topk: int,
    stiff_spot_min_ky: float,
    stiff_spot_dt: float,
    stiff_spot_tmax: float,
    stiff_spot_replace: bool,
) -> None:
    kinetic_ref = load_cyclone_reference_kinetic()
    kinetic_ny = 2 * int(kinetic_ref.ky.size) + 1
    kinetic_steps = _scale_steps(
        kinetic_ref.ky, base_steps=20000, ky_ref=0.3, max_steps=30000
    )
    kinetic_dt = _scale_dt(kinetic_ref.ky, base_dt=0.0005, ky_ref=0.3)
    kinetic_ttotal = kinetic_dt * kinetic_steps
    kinetic_tmin = 0.6 * kinetic_ttotal
    kinetic_tmax = 0.95 * kinetic_ttotal
    kinetic_cfg, _raw = load_runtime_from_toml(
        ROOT / "examples" / "linear" / "axisymmetric" / "runtime_kinetic_electron.toml"
    )
    kinetic_cfg = replace(
        kinetic_cfg,
        grid=GridConfig(
            Nx=1,
            Ny=kinetic_ny,
            Nz=96,
            Lx=62.8,
            Ly=62.8,
            y0=10.0,
            ntheta=32,
            nperiod=2,
        ),
    )
    kin_ky, kin_g, kin_w = _scan_linear_verbose(
        ky_values=kinetic_ref.ky,
        run_linear_fn=_run_runtime_linear_adapter,
        cfg=kinetic_cfg,
        Nl=48,
        Nm=16,
        dt=kinetic_dt,
        steps=kinetic_steps,
        method="imex2",
        solver=KINETIC_SOLVER,
        krylov_cfg=KINETIC_KRYLOV,
        window_kw=WINDOWS["kinetic"],
        tmin=kinetic_tmin,
        tmax=kinetic_tmax,
        auto_window=False,
        label="Kinetic ITG mismatch",
        ref=kinetic_ref,
        verbose=verbose,
        progress=progress,
        stiff_spot_check=stiff_spot_check,
        stiff_spot_check_topk=stiff_spot_topk,
        stiff_spot_check_min_ky=stiff_spot_min_ky,
        stiff_spot_check_dt=stiff_spot_dt,
        stiff_spot_check_tmax=stiff_spot_tmax,
        stiff_spot_check_replace=stiff_spot_replace,
    )
    kinetic_mismatch = LinearScanResult(ky=kin_ky, gamma=kin_g, omega=kin_w)
    (outdir / "kinetic_mismatch_table.csv").write_text(
        "\n".join(_build_rows(kinetic_mismatch, kinetic_ref)) + "\n", encoding="utf-8"
    )


def _write_cyclone_runtime_tables(
    *, outdir: Path, minimal: bool, verbose: bool, progress: bool
) -> None:
    cfg, _ = load_runtime_from_toml(ROOT / "examples/linear/axisymmetric/cyclone.toml")
    ref = _cyclone_refresh_reference(load_cyclone_reference())
    mismatch = _cyclone_reference_mismatch_scan(
        ref, cfg, verbose=verbose, progress=progress
    )
    (outdir / "cyclone_mismatch_table.csv").write_text(
        "\n".join(_build_rows(mismatch, ref)) + "\n", encoding="utf-8"
    )
    if minimal:
        return

    ky_convergence = np.array([0.3, 0.4])
    low = _runtime_cyclone_scan(cfg, ky_convergence, Nl=8, Nm=24, progress=progress)
    high = _runtime_cyclone_scan(cfg, ky_convergence, Nl=16, Nm=48, progress=progress)
    (outdir / "cyclone_scan_table_lowres.csv").write_text(
        "\n".join(_build_rows(low, ref)) + "\n", encoding="utf-8"
    )
    (outdir / "cyclone_scan_table_highres.csv").write_text(
        "\n".join(_build_rows(high, ref)) + "\n", encoding="utf-8"
    )
    convergence_rows = [
        "ky,gamma_low,gamma_high,omega_low,omega_high,rel_gamma_change,rel_omega_change"
    ]
    for ky, g_lo, g_hi, w_lo, w_hi in zip(
        low.ky, low.gamma, high.gamma, low.omega, high.omega
    ):
        rel_g = (g_hi - g_lo) / g_hi if g_hi != 0.0 else np.nan
        rel_w = (w_hi - w_lo) / w_hi if w_hi != 0.0 else np.nan
        convergence_rows.append(
            f"{ky:.3f},{g_lo:.6f},{g_hi:.6f},{w_lo:.6f},{w_hi:.6f},{rel_g:.3f},{rel_w:.3f}"
        )
    (outdir / "cyclone_scan_convergence.csv").write_text(
        "\n".join(convergence_rows) + "\n", encoding="utf-8"
    )

    full = _runtime_cyclone_scan(
        cfg, np.array([0.2, 0.3, 0.4]), Nl=16, Nm=48, progress=progress
    )
    full_rows = [
        "ky,gamma_ref,omega_ref,gamma_full,omega_full,abs_gamma,abs_omega,rel_gamma,rel_omega"
    ]
    for ky, gamma, omega in zip(full.ky, full.gamma, full.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        gamma_ref = float(ref.gamma[idx])
        omega_ref = float(ref.omega[idx])
        rel_gamma = (abs(gamma) - gamma_ref) / gamma_ref
        rel_omega = (abs(omega) - abs(omega_ref)) / abs(omega_ref)
        full_rows.append(
            f"{ky:.3f},{gamma_ref:.6f},{omega_ref:.6f},{gamma:.6f},{omega:.6f},"
            f"{abs(gamma):.6f},{abs(omega):.6f},{rel_gamma:.3f},{rel_omega:.3f}"
        )
    (outdir / "cyclone_full_operator_scan_table.csv").write_text(
        "\n".join(full_rows) + "\n", encoding="utf-8"
    )

    rho_rows = ["rho_star,mean_gamma_ratio,mean_omega_ratio"]
    for rho in (0.8, 1.0, 1.2):
        rho_cfg = replace(
            cfg, normalization=replace(cfg.normalization, rho_star=float(rho))
        )
        scan = _runtime_cyclone_scan(rho_cfg, full.ky, Nl=16, Nm=48, progress=progress)
        ref_indices = [int(np.argmin(np.abs(ref.ky - ky))) for ky in scan.ky]
        gamma_ref = np.asarray(ref.gamma)[ref_indices]
        omega_ref = np.asarray(ref.omega)[ref_indices]
        gamma_ratio = np.mean(np.abs(scan.gamma) / np.maximum(gamma_ref, 1.0e-30))
        omega_ratio = np.mean(
            np.abs(scan.omega) / np.maximum(np.abs(omega_ref), 1.0e-30)
        )
        rho_rows.append(f"{rho:.2f},{gamma_ratio:.3f},{omega_ratio:.3f}")
    (outdir / "cyclone_rhostar_convergence.csv").write_text(
        "\n".join(rho_rows) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse_args()
    if args.lastvalue_scan is not None or args.lastvalue_out is not None:
        if args.lastvalue_scan is None or args.lastvalue_out is None:
            raise SystemExit(
                "--lastvalue-scan and --lastvalue-out must be provided together"
            )
        _write_lastvalue_table(args.lastvalue_scan, args.lastvalue_out)
        return 0

    verbose = not args.quiet
    progress = not args.no_progress
    stiff_spot_check = bool(args.stiff_spot_check)
    stiff_spot_topk = int(args.stiff_spot_check_topk)
    stiff_spot_min_ky = float(args.stiff_spot_check_min_ky)
    stiff_spot_dt = float(args.stiff_spot_check_dt)
    stiff_spot_tmax = float(args.stiff_spot_check_tmax)
    stiff_spot_replace = bool(args.stiff_spot_check_replace)

    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.case == "etg":
        _run_etg_tables(outdir=outdir, verbose=verbose, progress=progress)
        return 0

    _write_cyclone_runtime_tables(
        outdir=outdir,
        minimal=bool(args.refresh_minimal),
        verbose=verbose,
        progress=progress,
    )
    if args.case == "cyclone" or args.refresh_minimal:
        return 0

    _run_etg_tables(outdir=outdir, verbose=verbose, progress=progress)

    _run_kinetic_tables(
        outdir=outdir,
        verbose=verbose,
        progress=progress,
        stiff_spot_check=stiff_spot_check,
        stiff_spot_topk=stiff_spot_topk,
        stiff_spot_min_ky=stiff_spot_min_ky,
        stiff_spot_dt=stiff_spot_dt,
        stiff_spot_tmax=stiff_spot_tmax,
        stiff_spot_replace=stiff_spot_replace,
    )

    _write_kbm_public_mismatch_table(
        outdir,
        verbose=verbose,
        progress=progress,
        stiff_spot_check=stiff_spot_check,
        stiff_spot_topk=stiff_spot_topk,
        stiff_spot_dt=stiff_spot_dt,
        stiff_spot_tmax=stiff_spot_tmax,
        stiff_spot_replace=stiff_spot_replace,
    )

    _run_tem_tables(outdir=outdir, verbose=verbose, progress=progress)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
