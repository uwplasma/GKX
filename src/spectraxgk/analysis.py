"""Analysis helpers for extracting growth rates and mode signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ModeSelection:
    ky_index: int
    kx_index: int
    z_index: int = 0


@dataclass(frozen=True)
class ModeSelectionBatch:
    ky_indices: np.ndarray
    kx_index: int
    z_index: int = 0


def select_ky_index(ky: np.ndarray, ky_target: float) -> int:
    """Return the index of ky closest to ky_target."""

    return int(np.argmin(np.abs(ky - ky_target)))


def extract_mode_time_series(
    phi_t: np.ndarray, sel: ModeSelection, method: str = "z_index"
) -> np.ndarray:
    """Extract a complex mode time series from phi_t(t, ky, kx, z)."""

    data = phi_t[:, sel.ky_index, sel.kx_index, :]
    if method == "z_index":
        return data[:, sel.z_index]
    if method == "max":
        idx = np.argmax(np.abs(data), axis=1)
        return data[np.arange(data.shape[0]), idx]
    if method == "project":
        n = data.shape[0]
        tail_start = int(0.6 * n)
        tail = data[tail_start:] if tail_start < n else data
        finite_rows = np.isfinite(tail).all(axis=1)
        if not finite_rows.any():
            return data[:, sel.z_index]
        tail = tail[finite_rows]
        ref_idx = int(np.argmax(np.linalg.norm(tail, axis=1)))
        ref = tail[ref_idx]
        denom = np.vdot(ref, ref)
        denom = denom if denom != 0.0 else 1.0
        return (data @ ref.conj()) / denom
    if method == "svd":
        if not np.isfinite(data).all():
            return extract_mode_time_series(phi_t, sel, method="z_index")
        try:
            u, s, _vh = np.linalg.svd(data, full_matrices=False)
        except np.linalg.LinAlgError:
            return extract_mode_time_series(phi_t, sel, method="project")
        return u[:, 0] * s[0]
    raise ValueError("method must be one of {'z_index', 'max', 'project', 'svd'}")


def extract_mode(phi_t: np.ndarray, sel: ModeSelection) -> np.ndarray:
    """Extract a complex mode time series from phi_t(t, ky, kx, z)."""

    return extract_mode_time_series(phi_t, sel, method="z_index")


def density_moment(
    G: np.ndarray,
    Jl: np.ndarray,
    *,
    species_index: int | None = None,
) -> np.ndarray:
    """Compute the m=0 density moment for a selected species (or summed if None)."""

    if G.ndim == 5:
        Gm0 = G[:, 0, ...]
        return np.sum(Jl * Gm0, axis=0)
    if G.ndim == 6:
        if species_index is None:
            Gm0 = G[:, :, 0, ...]
            return np.sum(Jl[None, ...] * Gm0, axis=1).sum(axis=0)
        Gm0 = G[species_index, :, 0, ...]
        return np.sum(Jl * Gm0, axis=0)
    raise ValueError("G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")


def extract_eigenfunction(
    phi_t: np.ndarray,
    t: np.ndarray,
    sel: ModeSelection,
    z: np.ndarray | None = None,
    method: str = "svd",
    tmin: float | None = None,
    tmax: float | None = None,
) -> np.ndarray:
    """Extract a normalized eigenfunction in z from phi_t(t, ky, kx, z)."""

    if phi_t.ndim != 4:
        raise ValueError("phi_t must have shape (t, ky, kx, z)")
    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if t.shape[0] != phi_t.shape[0]:
        raise ValueError("t and phi_t must have consistent time dimension")

    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= tmin
    if tmax is not None:
        mask &= t <= tmax
    data = phi_t[mask, sel.ky_index, sel.kx_index, :]
    if data.shape[0] == 0:
        raise ValueError("empty time window for eigenfunction extraction")

    def _snapshot_mode(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1)
        idx = int(np.argmax(norms))
        return arr[idx]

    if method == "snapshot":
        finite_rows = np.isfinite(data).all(axis=1)
        data_finite = data[finite_rows] if finite_rows.any() else data
        mode = _snapshot_mode(data_finite)
    elif method == "svd":
        if not np.isfinite(data).all():
            finite_rows = np.isfinite(data).all(axis=1)
            data_finite = data[finite_rows] if finite_rows.any() else data
            mode = _snapshot_mode(data_finite)
        else:
            try:
                _u, _s, vh = np.linalg.svd(data, full_matrices=False)
                mode = vh[0]
                ref = _snapshot_mode(data)
                phase = np.vdot(mode, ref)
                if phase != 0.0:
                    mode = mode * np.exp(-1j * np.angle(phase))
            except np.linalg.LinAlgError:
                mode = _snapshot_mode(data)
    else:
        raise ValueError("method must be one of {'svd', 'snapshot'}")

    if z is not None:
        if z.ndim != 1:
            raise ValueError("z must be 1D when provided")
        if z.shape[0] != mode.shape[0]:
            raise ValueError("z must have the same length as the eigenfunction")
        idx0 = int(np.argmin(np.abs(z)))
        ref = mode[idx0]
        if ref != 0.0:
            mode = mode / ref
        else:
            scale = np.max(np.abs(mode))
            if scale > 0.0:
                mode = mode / scale
    else:
        scale = np.max(np.abs(mode))
        if scale > 0.0:
            mode = mode / scale
    return mode


def fit_growth_rate(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
) -> Tuple[float, float]:
    """Fit gamma and omega from a complex signal ~ exp((gamma - i*omega) t)."""

    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    if t.shape[0] != signal.shape[0]:
        raise ValueError("t and signal must have same length")

    finite = np.isfinite(signal)
    if not np.all(finite):
        t = t[finite]
        signal = signal[finite]
        if t.size < 2:
            raise ValueError("not enough finite points to fit")

    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= tmin
    if tmax is not None:
        mask &= t <= tmax

    tt = t[mask]
    yy = signal[mask]
    if tt.size < 2:
        raise ValueError("not enough points to fit")

    log_amp, phase = _log_amp_phase(yy)

    A = np.vstack([tt, np.ones_like(tt)]).T
    gamma, _ = np.linalg.lstsq(A, log_amp, rcond=None)[0]
    omega, _ = np.linalg.lstsq(A, phase, rcond=None)[0]
    return float(gamma), float(-omega)


def fit_growth_rate_with_stats(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
) -> Tuple[float, float, float, float]:
    """Fit gamma/omega and return (gamma, omega, r2_log_amp, r2_phase)."""

    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    if t.shape[0] != signal.shape[0]:
        raise ValueError("t and signal must have same length")

    finite = np.isfinite(signal)
    if not np.all(finite):
        t = t[finite]
        signal = signal[finite]
        if t.size < 2:
            raise ValueError("not enough finite points to fit")

    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= tmin
    if tmax is not None:
        mask &= t <= tmax

    tt = t[mask]
    yy = signal[mask]
    if tt.size < 2:
        raise ValueError("not enough points to fit")

    log_amp, phase = _log_amp_phase(yy)
    A = np.vstack([tt, np.ones_like(tt)]).T
    gamma, offset = np.linalg.lstsq(A, log_amp, rcond=None)[0]
    phase_slope, phase_off = np.linalg.lstsq(A, phase, rcond=None)[0]
    log_fit = gamma * tt + offset
    phase_fit = phase_slope * tt + phase_off

    def r2_score(y: np.ndarray, yfit: np.ndarray) -> float:
        ss_res = float(np.sum((y - yfit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 0.0:
            return -np.inf
        return 1.0 - ss_res / ss_tot

    r2_log = r2_score(log_amp, log_fit)
    r2_phase = r2_score(phase, phase_fit)
    return float(gamma), float(-phase_slope), r2_log, r2_phase


def select_fit_window(
    t: np.ndarray,
    signal: np.ndarray,
    window_fraction: float = 0.3,
    min_points: int = 20,
    start_fraction: float = 0.0,
    growth_weight: float = 0.0,
    require_positive: bool = False,
    min_amp_fraction: float = 0.0,
) -> Tuple[float, float]:
    """Pick a time window with the most exponential-like behavior."""

    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    if t.shape[0] != signal.shape[0]:
        raise ValueError("t and signal must have same length")

    finite = np.isfinite(signal)
    if not np.all(finite):
        t = t[finite]
        signal = signal[finite]
        if t.size < 2:
            raise ValueError("not enough finite points to fit")

    n = t.shape[0]
    if n < 2:
        raise ValueError("not enough points to fit")
    window = max(min_points, int(window_fraction * n))
    window = min(window, n)
    if window < 2:
        raise ValueError("window too short")
    if not (0.0 <= start_fraction < 1.0):
        raise ValueError("start_fraction must be in [0, 1)")
    if growth_weight < 0.0:
        raise ValueError("growth_weight must be >= 0")
    if not (0.0 <= min_amp_fraction < 1.0):
        raise ValueError("min_amp_fraction must be in [0, 1)")

    log_amp, phase = _log_amp_phase(signal)
    amp_lin = np.abs(signal)

    def r2_score(y: np.ndarray, yfit: np.ndarray) -> float:
        ss_res = float(np.sum((y - yfit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 0.0:
            return -np.inf
        return 1.0 - ss_res / ss_tot

    best_score = -np.inf
    best_slice = (0, window)
    found_positive = False
    start_index = int(start_fraction * n)
    if min_amp_fraction > 0.0:
        amp_thresh = min_amp_fraction * float(np.max(amp_lin))
        above = np.where(amp_lin >= amp_thresh)[0]
        if above.size:
            start_index = max(start_index, int(above[0]))
    for start in range(start_index, n - window + 1):
        end = start + window
        tt = t[start:end]
        A = np.vstack([tt, np.ones_like(tt)]).T
        gamma, offset = np.linalg.lstsq(A, log_amp[start:end], rcond=None)[0]
        phase_slope, phase_off = np.linalg.lstsq(A, phase[start:end], rcond=None)[0]
        amp_fit = gamma * tt + offset
        phase_fit = phase_slope * tt + phase_off
        if require_positive and gamma <= 0.0:
            continue
        if require_positive:
            found_positive = True
        score = r2_score(log_amp[start:end], amp_fit) + r2_score(phase[start:end], phase_fit)
        if growth_weight > 0.0:
            score += growth_weight * float(gamma)
        score += 0.01 * (start / max(1, n - window))
        if score > best_score:
            best_score = score
            best_slice = (start, end)
    if require_positive and not found_positive:
        return select_fit_window(
            t,
            signal,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=False,
            min_amp_fraction=min_amp_fraction,
        )

    tmin = float(t[best_slice[0]])
    tmax = float(t[best_slice[1] - 1])
    return tmin, tmax


def select_fit_window_loglinear(
    t: np.ndarray,
    signal: np.ndarray,
    min_points: int = 20,
    start_fraction: float = 0.0,
    max_fraction: float = 0.8,
    end_fraction: float = 1.0,
    num_windows: int = 8,
    growth_weight: float = 0.0,
    require_positive: bool = False,
    min_amp_fraction: float = 0.0,
    max_amp_fraction: float = 1.0,
    phase_weight: float = 0.2,
    length_weight: float = 0.05,
    min_r2: float = 0.0,
    late_penalty: float = 0.0,
    min_slope: float | None = None,
    min_slope_frac: float = 0.0,
    slope_var_weight: float = 0.0,
) -> Tuple[float, float]:
    """Select a window where log-amplitude is closest to linear."""

    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if signal.ndim != 1:
        raise ValueError("signal must be 1D")
    if t.shape[0] != signal.shape[0]:
        raise ValueError("t and signal must have same length")

    finite = np.isfinite(signal)
    if not np.all(finite):
        t = t[finite]
        signal = signal[finite]
        if t.size < 2:
            raise ValueError("not enough finite points to fit")

    n = t.shape[0]
    if n < 2:
        raise ValueError("not enough points to fit")
    if min_points < 2:
        raise ValueError("min_points must be >= 2")
    if not (0.0 <= start_fraction < 1.0):
        raise ValueError("start_fraction must be in [0, 1)")
    if not (0.0 < max_fraction <= 1.0):
        raise ValueError("max_fraction must be in (0, 1]")
    if not (0.0 < end_fraction <= 1.0):
        raise ValueError("end_fraction must be in (0, 1]")
    if num_windows < 1:
        raise ValueError("num_windows must be >= 1")
    if growth_weight < 0.0:
        raise ValueError("growth_weight must be >= 0")
    if not (0.0 <= min_amp_fraction < 1.0):
        raise ValueError("min_amp_fraction must be in [0, 1)")
    if not (0.0 < max_amp_fraction <= 1.0):
        raise ValueError("max_amp_fraction must be in (0, 1]")
    if late_penalty < 0.0:
        raise ValueError("late_penalty must be >= 0")
    if min_slope_frac < 0.0:
        raise ValueError("min_slope_frac must be >= 0")
    if slope_var_weight < 0.0:
        raise ValueError("slope_var_weight must be >= 0")

    max_points = max(min_points, int(max_fraction * n))
    max_points = min(max_points, n)
    end_index_max = max(int(end_fraction * n), 2)
    lengths = np.unique(np.linspace(min_points, max_points, num=num_windows).astype(int))
    lengths = lengths[lengths >= 2]
    if lengths.size == 0:
        raise ValueError("no valid window lengths")

    log_amp, phase = _log_amp_phase(signal)
    amp_lin = np.abs(signal)
    slope_series = np.gradient(log_amp, t)
    slope_pos = slope_series[np.isfinite(slope_series) & (slope_series > 0.0)]
    slope_ref = float(np.percentile(slope_pos, 90)) if slope_pos.size else 0.0
    slope_thresh = min_slope if min_slope is not None else 0.0
    if min_slope_frac > 0.0 and slope_ref > 0.0:
        slope_thresh = max(slope_thresh, min_slope_frac * slope_ref)

    def r2_score(y: np.ndarray, yfit: np.ndarray) -> float:
        ss_res = float(np.sum((y - yfit) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot <= 0.0:
            return -np.inf
        return 1.0 - ss_res / ss_tot

    best_score = -np.inf
    best_slice = (0, int(lengths[0]))
    found_positive = False
    start_index = int(start_fraction * n)
    if min_amp_fraction > 0.0:
        amp_thresh = min_amp_fraction * float(np.max(amp_lin))
        above = np.where(amp_lin >= amp_thresh)[0]
        if above.size:
            start_index = max(start_index, int(above[0]))
    amp_finite = amp_lin[np.isfinite(amp_lin)]
    if amp_finite.size:
        amp_ref = float(np.percentile(amp_finite, 95.0))
        if not np.isfinite(amp_ref) or amp_ref <= 0.0:
            amp_ref = float(np.max(amp_finite))
    else:
        amp_ref = float(np.nanmax(amp_lin))
    max_amp = amp_ref if amp_ref > 0.0 else float(np.max(amp_lin))

    for window in lengths:
        if window > n:
            continue
        for start in range(start_index, n - window + 1):
            end = start + window
            if end > end_index_max:
                continue
            tt = t[start:end]
            A = np.vstack([tt, np.ones_like(tt)]).T
            gamma, offset = np.linalg.lstsq(A, log_amp[start:end], rcond=None)[0]
            phase_slope, phase_off = np.linalg.lstsq(A, phase[start:end], rcond=None)[0]
            log_fit = gamma * tt + offset
            phase_fit = phase_slope * tt + phase_off
            r2_log = r2_score(log_amp[start:end], log_fit)
            r2_phase = r2_score(phase[start:end], phase_fit)
            if r2_log < min_r2:
                continue
            if require_positive and gamma <= 0.0:
                continue
            if slope_thresh > 0.0 and gamma < slope_thresh:
                continue
            if require_positive:
                found_positive = True
            if max_amp_fraction < 1.0:
                if float(np.max(amp_lin[start:end])) > max_amp_fraction * max_amp:
                    continue
            score = r2_log + phase_weight * r2_phase
            if growth_weight > 0.0:
                score += growth_weight * float(gamma)
            if length_weight > 0.0:
                score += length_weight * float(window) / float(n)
            if slope_var_weight > 0.0:
                slope_std = float(np.std(slope_series[start:end]))
                score -= slope_var_weight * (slope_std / (abs(gamma) + 1.0e-12))
            if late_penalty > 0.0:
                score -= late_penalty * (start / max(1, n - window))
            if score > best_score:
                best_score = score
                best_slice = (start, end)

    if best_score == -np.inf:
        fallback_start = 0
        fallback_end_index_max = end_index_max if end_index_max >= min_points else n
        for window in lengths:
            if window > n:
                continue
            for start in range(fallback_start, n - window + 1):
                end = start + window
                if end > fallback_end_index_max:
                    continue
                tt = t[start:end]
                A = np.vstack([tt, np.ones_like(tt)]).T
                gamma, offset = np.linalg.lstsq(A, log_amp[start:end], rcond=None)[0]
                phase_slope, phase_off = np.linalg.lstsq(A, phase[start:end], rcond=None)[0]
                log_fit = gamma * tt + offset
                phase_fit = phase_slope * tt + phase_off
                r2_log = r2_score(log_amp[start:end], log_fit)
                r2_phase = r2_score(phase[start:end], phase_fit)
                score = r2_log + phase_weight * r2_phase
                if score > best_score:
                    best_score = score
                    best_slice = (start, end)

    if require_positive and not found_positive:
        return select_fit_window_loglinear(
            t,
            signal,
            min_points=min_points,
            start_fraction=start_fraction,
            max_fraction=max_fraction,
            end_fraction=end_fraction,
            num_windows=num_windows,
            growth_weight=growth_weight,
            require_positive=False,
            min_amp_fraction=min_amp_fraction,
            max_amp_fraction=max_amp_fraction,
            phase_weight=phase_weight,
            length_weight=length_weight,
            min_r2=min_r2,
            late_penalty=late_penalty,
            min_slope=min_slope,
            min_slope_frac=min_slope_frac,
            slope_var_weight=slope_var_weight,
        )

    tmin = float(t[best_slice[0]])
    tmax = float(t[best_slice[1] - 1])
    return tmin, tmax


def fit_growth_rate_auto(
    t: np.ndarray,
    signal: np.ndarray,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.3,
    min_points: int = 20,
    start_fraction: float = 0.0,
    growth_weight: float = 0.0,
    require_positive: bool = False,
    min_amp_fraction: float = 0.0,
    max_amp_fraction: float = 1.0,
    window_method: str = "loglinear",
    max_fraction: float = 0.8,
    end_fraction: float = 1.0,
    num_windows: int = 8,
    phase_weight: float = 0.2,
    length_weight: float = 0.05,
    min_r2: float = 0.0,
    late_penalty: float = 0.0,
    min_slope: float | None = None,
    min_slope_frac: float = 0.0,
    slope_var_weight: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Fit gamma/omega with optional auto-selected window."""

    if tmin is None and tmax is None:
        if window_method == "loglinear":
            tmin, tmax = select_fit_window_loglinear(
                t,
                signal,
                min_points=min_points,
                start_fraction=start_fraction,
                max_fraction=max_fraction,
                end_fraction=end_fraction,
                num_windows=num_windows,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
                max_amp_fraction=max_amp_fraction,
                phase_weight=phase_weight,
                length_weight=length_weight,
                min_r2=min_r2,
                late_penalty=late_penalty,
                min_slope=min_slope,
                min_slope_frac=min_slope_frac,
                slope_var_weight=slope_var_weight,
            )
        elif window_method == "fixed":
            tmin, tmax = select_fit_window(
                t,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
        else:
            raise ValueError("window_method must be 'loglinear' or 'fixed'")
    gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
    tmin_out = float(tmin) if tmin is not None else float(t[0])
    tmax_out = float(tmax) if tmax is not None else float(t[-1])
    return gamma, omega, tmin_out, tmax_out


def gx_growth_rate_from_phi(
    phi_t: np.ndarray,
    t: np.ndarray | None,
    sel: ModeSelection,
    *,
    navg_fraction: float = 0.5,
    mode_method: str = "z_index",
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute GX-style instantaneous growth rates from phi ratios.

    Returns (gamma_avg, omega_avg, gamma_t, omega_t, t_mid).
    """

    if phi_t.ndim != 4:
        raise ValueError("phi_t must have shape (t, ky, kx, z)")
    if t is None:
        t = np.arange(phi_t.shape[0], dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be 1D")
    if t.shape[0] != phi_t.shape[0]:
        raise ValueError("t and phi_t must have consistent time dimension")

    if phi_t.shape[0] < 2:
        raise ValueError("phi_t must have at least two time samples")

    if mode_method not in {"z_index", "max"}:
        raise ValueError("mode_method must be 'z_index' or 'max'")

    signal = extract_mode_time_series(phi_t, sel, method=mode_method)
    phi_now = signal[1:]
    phi_prev = signal[:-1]
    dt = np.diff(t)
    dt = np.where(dt == 0.0, 1.0, dt)

    ratio = np.full_like(phi_now, np.nan + 1.0j * np.nan)
    mask = (phi_prev != 0.0) & np.isfinite(phi_prev) & np.isfinite(phi_now)
    ratio[mask] = phi_now[mask] / phi_prev[mask]

    log_amp = np.log(np.abs(ratio))
    phase = np.angle(ratio)
    gamma = log_amp / dt
    omega = -phase / dt
    t_mid = 0.5 * (t[1:] + t[:-1])

    finite = np.isfinite(gamma) & np.isfinite(omega)
    gamma = gamma[finite]
    omega = omega[finite]
    t_mid = t_mid[finite]
    if gamma.size == 0:
        raise ValueError("No finite GX growth-rate samples available")

    istart = int(len(gamma) * navg_fraction)
    gamma_avg = float(np.mean(gamma[istart:]))
    omega_avg = float(np.mean(omega[istart:]))
    return gamma_avg, omega_avg, gamma, omega, t_mid


def _log_amp_phase(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (log|signal|, unwrapped phase) with robust scaling."""

    if signal.size == 0:
        raise ValueError("signal must be non-empty")
    signal = np.asarray(signal)
    finite = np.isfinite(signal)
    if np.any(finite):
        scale = float(np.max(np.abs(signal[finite])))
    else:
        scale = 1.0
    if not np.all(finite):
        signal = np.where(finite, signal, 0.0)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    scaled = signal / scale
    amp = np.abs(scaled)
    log_amp = np.log(np.maximum(amp, 1.0e-30)) + np.log(scale)
    phase = np.unwrap(np.angle(scaled))
    return log_amp, phase
