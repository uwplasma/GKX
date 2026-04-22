"""High-level benchmark helpers for scans and eigenfunction extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from spectraxgk.analysis import (
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
)
from spectraxgk.benchmarks import LinearRunResult, LinearScanResult
from spectraxgk.grids import SpectralGrid, build_spectral_grid


@dataclass(frozen=True)
class ScanAndModeResult:
    scan: LinearScanResult
    eigenfunction: np.ndarray
    grid: SpectralGrid
    ky_selected: float
    tmin: float | None
    tmax: float | None


@dataclass(frozen=True)
class LateTimeLinearMetrics:
    """Late-time growth/frequency metrics for a linear run."""

    gamma_fit: float
    omega_fit: float
    gamma_tail_mean: float
    omega_tail_mean: float
    gamma_tail_std: float
    omega_tail_std: float
    tmin: float | None
    tmax: float | None
    nsamples: int
    signal_source: str


@dataclass(frozen=True)
class NonlinearWindowMetrics:
    """Windowed transport/envelope metrics for a nonlinear run."""

    tmin: float
    tmax: float
    nsamples: int
    heat_flux_mean: float
    heat_flux_std: float
    heat_flux_rms: float
    wphi_mean: float
    wphi_std: float
    wg_mean: float
    wg_std: float
    phi_mode_envelope_mean: float | None
    phi_mode_envelope_std: float | None
    phi_mode_envelope_max: float | None


@dataclass(frozen=True)
class ObservedOrderMetrics:
    """Observed-order convergence summary from step sizes and errors."""

    step_sizes: np.ndarray
    errors: np.ndarray
    orders: np.ndarray
    asymptotic_order: float


@dataclass(frozen=True)
class EigenfunctionComparisonMetrics:
    """Phase-aligned eigenfunction comparison summary."""

    overlap: float
    relative_l2: float
    phase_shift: float


@dataclass(frozen=True)
class EigenfunctionReferenceBundle:
    """Frozen reference eigenfunction bundle for manuscript-grade overlays."""

    theta: np.ndarray
    mode: np.ndarray
    source: str
    case: str
    metadata: dict[str, object]


def normalize_eigenfunction(eigenfunction: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Normalize an eigenfunction by its value at theta=0 (nearest z=0)."""

    idx = int(np.argmin(np.abs(z)))
    scale = eigenfunction[idx]
    if scale == 0:
        return eigenfunction
    return eigenfunction / scale


def phase_align_eigenfunction(eigenfunction: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """Phase-align ``eigenfunction`` to ``reference`` using the global complex phase."""

    lhs = np.asarray(eigenfunction, dtype=np.complex128)
    rhs = np.asarray(reference, dtype=np.complex128)
    if lhs.shape != rhs.shape:
        raise ValueError("eigenfunction and reference must have the same shape")
    phase = np.vdot(lhs, rhs)
    if abs(phase) <= 1.0e-30:
        return lhs, 0.0
    phase_shift = float(np.angle(phase))
    return lhs * np.exp(1j * phase_shift), phase_shift


def compare_eigenfunctions(eigenfunction: np.ndarray, reference: np.ndarray) -> EigenfunctionComparisonMetrics:
    """Return normalized overlap and relative L2 error after global phase alignment."""

    lhs = np.asarray(eigenfunction, dtype=np.complex128)
    rhs = np.asarray(reference, dtype=np.complex128)
    if lhs.shape != rhs.shape:
        raise ValueError("eigenfunction and reference must have the same shape")
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return EigenfunctionComparisonMetrics(
            overlap=float("nan"),
            relative_l2=float("nan"),
            phase_shift=0.0,
        )
    lhs_aligned, phase_shift = phase_align_eigenfunction(lhs, rhs)
    overlap = float(np.abs(np.vdot(lhs, rhs)) / (lhs_norm * rhs_norm))
    rel_l2 = float(np.linalg.norm(lhs_aligned - rhs) / rhs_norm)
    return EigenfunctionComparisonMetrics(
        overlap=overlap,
        relative_l2=rel_l2,
        phase_shift=phase_shift,
    )


def save_eigenfunction_reference_bundle(
    path: str | Path,
    *,
    theta: np.ndarray,
    mode: np.ndarray,
    source: str,
    case: str,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Write a frozen reference eigenfunction bundle as ``.npz``."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        theta=np.asarray(theta, dtype=float),
        mode=np.asarray(mode, dtype=np.complex128),
        source=np.asarray(str(source)),
        case=np.asarray(str(case)),
        metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    )
    return out


def load_eigenfunction_reference_bundle(path: str | Path) -> EigenfunctionReferenceBundle:
    """Load a frozen reference eigenfunction bundle."""

    data = np.load(Path(path), allow_pickle=False)
    metadata_json = str(np.asarray(data["metadata_json"]).item()) if "metadata_json" in data else "{}"
    return EigenfunctionReferenceBundle(
        theta=np.asarray(data["theta"], dtype=float),
        mode=np.asarray(data["mode"], dtype=np.complex128),
        source=str(np.asarray(data["source"]).item()),
        case=str(np.asarray(data["case"]).item()),
        metadata=json.loads(metadata_json),
    )


def _tail_window(t: np.ndarray, tail_fraction: float) -> tuple[np.ndarray, float | None, float | None]:
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if t.size == 0:
        raise ValueError("t must be non-empty")
    start = max(0, int(np.floor((1.0 - float(tail_fraction)) * t.size)))
    mask = np.zeros_like(t, dtype=bool)
    mask[start:] = True
    if not np.any(mask):
        mask[-1] = True
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]) if tt.size else None, float(tt[-1]) if tt.size else None


def late_time_window(t: np.ndarray, *, tail_fraction: float = 0.4) -> tuple[float, float]:
    """Return the start/end of a late-time tail window.

    This is the windowing convention used for manuscript-facing eigenfunction
    extraction when the growth-rate fit window is not the same object as the
    late-time mode-shape window.
    """

    _mask, tmin, tmax = _tail_window(np.asarray(t, dtype=float), float(tail_fraction))
    if tmin is None or tmax is None:
        raise ValueError("late-time window requires a non-empty time axis")
    return float(tmin), float(tmax)


def infer_triple_dealiased_ny(nky_positive: int) -> int:
    """Infer the full ``Ny`` from the number of positive ``k_y`` points.

    GX-style reference outputs typically store only the non-negative
    ``k_y`` branch. For the linked-boundary spectral grid used here, the
    corresponding real-space ``Ny`` follows ``Ny = 3 * (nky - 1) + 1``.
    """

    nky = int(nky_positive)
    if nky < 2:
        raise ValueError("nky_positive must be >= 2")
    return 3 * (nky - 1) + 1


def _tail_stats(arr: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def late_time_linear_metrics(
    result: object,
    *,
    tail_fraction: float = 0.5,
    mode_method: str = "project",
) -> LateTimeLinearMetrics:
    """Return late-time growth/frequency metrics from a linear benchmark/runtime result."""

    t = getattr(result, "t", None)
    if t is None:
        gamma = float(getattr(result, "gamma"))
        omega = float(getattr(result, "omega"))
        return LateTimeLinearMetrics(
            gamma_fit=gamma,
            omega_fit=omega,
            gamma_tail_mean=gamma,
            omega_tail_mean=omega,
            gamma_tail_std=0.0,
            omega_tail_std=0.0,
            tmin=None,
            tmax=None,
            nsamples=1,
            signal_source="scalar",
        )

    t_arr = np.asarray(t, dtype=float)
    mask, tmin, tmax = _tail_window(t_arr, tail_fraction)

    gamma_series = getattr(result, "gamma_t", None)
    omega_series = getattr(result, "omega_t", None)
    signal_source = "scalar"
    gamma_fit = float(getattr(result, "gamma"))
    omega_fit = float(getattr(result, "omega"))

    signal = getattr(result, "signal", None)
    if signal is not None:
        signal_arr = np.asarray(signal, dtype=np.complex128)
        signal_source = "signal"
    elif hasattr(result, "phi_t") and hasattr(result, "selection"):
        signal_arr = np.asarray(
            extract_mode_time_series(
                np.asarray(getattr(result, "phi_t")),
                getattr(result, "selection"),
                method=mode_method,
            ),
            dtype=np.complex128,
        )
        signal_source = f"phi_t:{mode_method}"
    else:
        signal_arr = None

    if signal_arr is not None:
        finite = np.isfinite(signal_arr)
        signal_tail = signal_arr[mask & finite]
        t_tail = t_arr[mask & finite]
        if t_tail.size >= 2:
            gamma_fit, omega_fit = fit_growth_rate(t_tail, signal_tail)

    if gamma_series is not None:
        gamma_mean, gamma_std = _tail_stats(np.asarray(gamma_series), mask)
    else:
        gamma_mean, gamma_std = gamma_fit, 0.0
    if omega_series is not None:
        omega_mean, omega_std = _tail_stats(np.asarray(omega_series), mask)
    else:
        omega_mean, omega_std = omega_fit, 0.0

    nsamples = int(np.count_nonzero(mask))
    return LateTimeLinearMetrics(
        gamma_fit=float(gamma_fit),
        omega_fit=float(omega_fit),
        gamma_tail_mean=float(gamma_mean),
        omega_tail_mean=float(omega_mean),
        gamma_tail_std=float(gamma_std),
        omega_tail_std=float(omega_std),
        tmin=tmin,
        tmax=tmax,
        nsamples=nsamples,
        signal_source=signal_source,
    )


def windowed_nonlinear_metrics(
    result: object,
    *,
    start_fraction: float = 0.5,
) -> NonlinearWindowMetrics:
    """Return late-window transport and envelope metrics from a nonlinear runtime result."""

    diagnostics = getattr(result, "diagnostics", result)
    if diagnostics is None:
        raise ValueError("nonlinear diagnostics are required")
    if not 0.0 <= float(start_fraction) < 1.0:
        raise ValueError("start_fraction must be in [0, 1)")
    t = np.asarray(getattr(diagnostics, "t", None), dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("diagnostics.t must be a non-empty one-dimensional array")
    tail_fraction = max(np.finfo(float).eps, 1.0 - float(start_fraction))
    mask, tmin, tmax = _tail_window(t, tail_fraction)
    heat_flux = np.asarray(getattr(diagnostics, "heat_flux_t"), dtype=float)[mask]
    wphi = np.asarray(getattr(diagnostics, "Wphi_t"), dtype=float)[mask]
    wg = np.asarray(getattr(diagnostics, "Wg_t"), dtype=float)[mask]
    heat_flux = heat_flux[np.isfinite(heat_flux)]
    wphi = wphi[np.isfinite(wphi)]
    wg = wg[np.isfinite(wg)]
    if heat_flux.size == 0 or wphi.size == 0 or wg.size == 0:
        raise ValueError("windowed diagnostics must contain finite heat/Wphi/Wg samples")

    phi_mode = getattr(diagnostics, "phi_mode_t", None)
    envelope_mean: float | None = None
    envelope_std: float | None = None
    envelope_max: float | None = None
    if phi_mode is not None:
        envelope = np.abs(np.asarray(phi_mode)[mask])
        envelope = envelope[np.isfinite(envelope)]
        if envelope.size:
            envelope_mean = float(np.mean(envelope))
            envelope_std = float(np.std(envelope))
            envelope_max = float(np.max(envelope))

    return NonlinearWindowMetrics(
        tmin=float(tmin if tmin is not None else t[0]),
        tmax=float(tmax if tmax is not None else t[-1]),
        nsamples=int(np.count_nonzero(mask)),
        heat_flux_mean=float(np.mean(heat_flux)),
        heat_flux_std=float(np.std(heat_flux)),
        heat_flux_rms=float(np.sqrt(np.mean(np.square(heat_flux)))),
        wphi_mean=float(np.mean(wphi)),
        wphi_std=float(np.std(wphi)),
        wg_mean=float(np.mean(wg)),
        wg_std=float(np.std(wg)),
        phi_mode_envelope_mean=envelope_mean,
        phi_mode_envelope_std=envelope_std,
        phi_mode_envelope_max=envelope_max,
    )


def estimate_observed_order(step_sizes: np.ndarray, errors: np.ndarray) -> ObservedOrderMetrics:
    """Estimate observed order from successive step-size refinements."""

    h = np.asarray(step_sizes, dtype=float)
    err = np.asarray(errors, dtype=float)
    if h.ndim != 1 or err.ndim != 1 or h.size != err.size or h.size < 2:
        raise ValueError("step_sizes and errors must be one-dimensional arrays of equal length >= 2")
    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(err)):
        raise ValueError("step_sizes and errors must be finite")
    if np.any(h <= 0.0):
        raise ValueError("step_sizes must be positive")
    if np.any(err <= 0.0):
        raise ValueError("errors must be positive")

    orders: list[float] = []
    for i in range(h.size - 1):
        if np.isclose(h[i], h[i + 1]):
            raise ValueError("successive step sizes must differ")
        orders.append(float(np.log(err[i] / err[i + 1]) / np.log(h[i] / h[i + 1])))
    orders_arr = np.asarray(orders, dtype=float)
    return ObservedOrderMetrics(
        step_sizes=h,
        errors=err,
        orders=orders_arr,
        asymptotic_order=float(orders_arr[-1]),
    )


def run_linear_scan(
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
    run_kwargs: dict | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
) -> LinearScanResult:
    """Run a linear scan over ky values."""

    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    for i, ky in enumerate(ky_values):
        if resolution_policy is not None:
            Nl_i, Nm_i = resolution_policy(float(ky))
        else:
            Nl_i, Nm_i = int(Nl), int(Nm)
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        tmin_i = tmin[i] if isinstance(tmin, np.ndarray) else tmin
        tmax_i = tmax[i] if isinstance(tmax, np.ndarray) else tmax
        krylov_cfg_use = krylov_policy(float(ky)) if krylov_policy is not None else krylov_cfg
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
            **(run_kwargs or {}),
        )
        gammas.append(float(result.gamma))
        omegas.append(float(result.omega))
        ky_out.append(float(result.ky))

    return LinearScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


def run_scan_and_mode(
    *,
    ky_values: np.ndarray,
    scan_fn,
    linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    mode_solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict | None = None,
    mode_kwargs: dict | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
    select_ky: Callable[[LinearScanResult], float] | None = None,
) -> ScanAndModeResult:
    """Run a scan and extract a representative eigenfunction."""

    if select_ky is None:
        select_ky = lambda scan: float(scan.ky[int(np.nanargmax(scan.gamma))])

    scan = run_linear_scan(
        ky_values=ky_values,
        run_linear_fn=linear_fn,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        solver=solver,
        krylov_cfg=krylov_cfg,
        window_kw=window_kw,
        tmin=tmin,
        tmax=tmax,
        auto_window=auto_window,
        run_kwargs=run_kwargs,
        resolution_policy=resolution_policy,
        krylov_policy=krylov_policy,
    )
    ky_sel = float(select_ky(scan))
    if resolution_policy is not None:
        Nl_mode, Nm_mode = resolution_policy(ky_sel)
    else:
        Nl_mode, Nm_mode = int(Nl), int(Nm)
    idx = int(np.argmin(np.abs(scan.ky - ky_sel)))
    dt_mode = float(dt[idx]) if isinstance(dt, np.ndarray) else float(dt)
    steps_mode = int(steps[idx]) if isinstance(steps, np.ndarray) else int(steps)
    run: LinearRunResult = linear_fn(
        cfg=cfg,
        ky_target=ky_sel,
        Nl=int(Nl_mode),
        Nm=int(Nm_mode),
        dt=dt_mode,
        steps=steps_mode,
        method=method,
        solver=mode_solver,
        **window_kw,
        **(mode_kwargs or {}),
    )
    grid = build_spectral_grid(cfg.grid)
    if run.t.size < 2:
        tmin_fit = None
        tmax_fit = None
    else:
        signal = extract_mode_time_series(run.phi_t, run.selection, method="project")
        _g, _w, tmin_fit, tmax_fit = fit_growth_rate_auto(run.t, signal, **window_kw)
    z_np = np.asarray(grid.z)
    eig = extract_eigenfunction(
        run.phi_t, run.t, run.selection, z=z_np, method="snapshot", tmin=tmin_fit, tmax=tmax_fit
    )
    return ScanAndModeResult(
        scan=scan,
        eigenfunction=eig,
        grid=grid,
        ky_selected=ky_sel,
        tmin=tmin_fit,
        tmax=tmax_fit,
    )
