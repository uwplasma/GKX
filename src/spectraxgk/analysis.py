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

    amp = np.abs(yy)
    amp = np.maximum(amp, 1.0e-30)
    phase = np.unwrap(np.angle(yy))

    A = np.vstack([tt, np.ones_like(tt)]).T
    gamma, _ = np.linalg.lstsq(A, np.log(amp), rcond=None)[0]
    omega, _ = np.linalg.lstsq(A, phase, rcond=None)[0]
    return float(gamma), float(-omega)


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

    amp_lin = np.abs(signal)
    amp = np.log(np.maximum(amp_lin, 1.0e-30))
    phase = np.unwrap(np.angle(signal))

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
        gamma, offset = np.linalg.lstsq(A, amp[start:end], rcond=None)[0]
        phase_slope, phase_off = np.linalg.lstsq(A, phase[start:end], rcond=None)[0]
        amp_fit = gamma * tt + offset
        phase_fit = phase_slope * tt + phase_off
        if require_positive and gamma <= 0.0:
            continue
        if require_positive:
            found_positive = True
        score = r2_score(amp[start:end], amp_fit) + r2_score(phase[start:end], phase_fit)
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
) -> Tuple[float, float, float, float]:
    """Fit gamma/omega with optional auto-selected window."""

    if tmin is None and tmax is None:
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
    gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)
    tmin_out = float(tmin) if tmin is not None else float(t[0])
    tmax_out = float(tmax) if tmax is not None else float(t[-1])
    return gamma, omega, tmin_out, tmax_out
