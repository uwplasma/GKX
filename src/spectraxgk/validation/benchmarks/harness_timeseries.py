"""Diagnostic time-series loading and benchmark window helpers."""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np

from spectraxgk.validation.gates import DiagnosticTimeSeries


def load_diagnostic_time_series(
    path: str | Path,
    *,
    variable: str,
    diagnostics_group: str = "Diagnostics",
    time_group: str = "Grids",
    time_var: str = "time",
    kx_index: int | None = None,
    component: str = "real",
    align_phase: bool = False,
) -> DiagnosticTimeSeries:
    """Load a 1D diagnostics time series from a grouped NetCDF output artifact."""

    src = Path(path)
    with nc.Dataset(src) as ds:
        diag_group = ds.groups.get(diagnostics_group)
        if diag_group is None:
            raise ValueError(f"missing NetCDF group {diagnostics_group!r} in {src}")
        if variable not in diag_group.variables:
            raise ValueError(f"missing diagnostics variable {variable!r} in {src}")
        var = diag_group.variables[variable]
        raw = np.asarray(var[:])
        dims = tuple(getattr(var, "dimensions", ()))
        if dims and dims[-1] == "ri":
            raw = np.asarray(raw[..., 0] + 1j * raw[..., 1], dtype=np.complex128)
        if raw.ndim == 1:
            values = raw
        elif raw.ndim == 2:
            if kx_index is None:
                raise ValueError(
                    f"diagnostics variable {variable!r} requires kx_index for 2D extraction"
                )
            values = raw[:, int(kx_index)]
        else:
            raise ValueError(
                f"diagnostics variable {variable!r} must reduce to a 1D time series"
            )

        if time_group in ds.groups and time_var in ds.groups[time_group].variables:
            t = np.asarray(ds.groups[time_group].variables[time_var][:], dtype=float)
        elif time_var in ds.variables:
            t = np.asarray(ds.variables[time_var][:], dtype=float)
        else:
            raise ValueError(f"missing time variable {time_group}/{time_var} in {src}")

    if t.ndim != 1 or t.size != values.size:
        raise ValueError(
            f"time axis for {variable!r} must be one-dimensional and match the diagnostics length"
        )

    values_arr = np.asarray(values)
    if np.iscomplexobj(values_arr):
        if align_phase:
            finite = np.isfinite(values_arr)
            nz = finite & (np.abs(values_arr) > 1.0e-30)
            if np.any(nz):
                first = values_arr[np.flatnonzero(nz)[0]]
                values_arr = values_arr * np.exp(-1j * np.angle(first))
        component_key = str(component).lower()
        if component_key == "complex":
            selected = values_arr
        elif component_key == "real":
            selected = np.real(values_arr)
        elif component_key == "imag":
            selected = np.imag(values_arr)
        elif component_key == "abs":
            selected = np.abs(values_arr)
        else:
            raise ValueError(
                "component must be one of {'real', 'imag', 'abs', 'complex'}"
            )
    else:
        if component not in {"real", "abs"}:
            raise ValueError("real diagnostics only support component='real' or 'abs'")
        selected = (
            np.abs(values_arr)
            if component == "abs"
            else np.asarray(values_arr, dtype=float)
        )

    return DiagnosticTimeSeries(
        t=t,
        values=np.asarray(selected),
        variable=str(variable),
        source_path=str(src),
    )


def _tail_window(
    t: np.ndarray, tail_fraction: float
) -> tuple[np.ndarray, float | None, float | None]:
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


def late_time_window(
    t: np.ndarray, *, tail_fraction: float = 0.4
) -> tuple[float, float]:
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

    Reference real-FFT outputs typically store only the non-negative
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


def _leading_window(
    t: np.ndarray,
    lead_fraction: float,
) -> tuple[np.ndarray, float | None, float | None]:
    if not 0.0 < float(lead_fraction) <= 1.0:
        raise ValueError("lead_fraction must be in (0, 1]")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if t.size == 0:
        raise ValueError("t must be non-empty")
    stop = max(1, int(np.ceil(float(lead_fraction) * t.size)))
    mask = np.zeros_like(t, dtype=bool)
    mask[:stop] = True
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]) if tt.size else None, float(tt[-1]) if tt.size else None


def _explicit_time_window(
    t: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> tuple[np.ndarray, float, float]:
    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= float(tmin)
    if tmax is not None:
        mask &= t <= float(tmax)
    if not np.any(mask):
        raise ValueError("explicit fit window is empty")
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]), float(tt[-1])


def _analytic_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    spec = np.fft.fft(x)
    filt = np.zeros(x.size, dtype=float)
    if x.size % 2 == 0:
        filt[0] = 1.0
        filt[x.size // 2] = 1.0
        filt[1 : x.size // 2] = 2.0
    else:
        filt[0] = 1.0
        filt[1 : (x.size + 1) // 2] = 2.0
    return np.fft.ifft(spec * filt)


__all__ = [
    "infer_triple_dealiased_ny",
    "late_time_window",
    "load_diagnostic_time_series",
]
