"""Diagnostic time-series loading and benchmark window helpers."""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np

from spectraxgk.validation.gates import DiagnosticTimeSeries


def _decode_netcdf_values(var) -> np.ndarray:
    raw = np.asarray(var[:])
    dims = tuple(getattr(var, "dimensions", ()))
    if dims and dims[-1] == "ri":
        return np.asarray(raw[..., 0] + 1j * raw[..., 1], dtype=np.complex128)
    return raw


def _extract_diagnostic_values(
    values: np.ndarray,
    *,
    variable: str,
    kx_index: int | None,
) -> np.ndarray:
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        if kx_index is None:
            raise ValueError(
                f"diagnostics variable {variable!r} requires kx_index for 2D extraction"
            )
        return values[:, int(kx_index)]
    raise ValueError(
        f"diagnostics variable {variable!r} must reduce to a 1D time series"
    )


def _load_netcdf_time_axis(
    ds,
    *,
    src: Path,
    time_group: str,
    time_var: str,
) -> np.ndarray:
    if time_group in ds.groups and time_var in ds.groups[time_group].variables:
        return np.asarray(ds.groups[time_group].variables[time_var][:], dtype=float)
    if time_var in ds.variables:
        return np.asarray(ds.variables[time_var][:], dtype=float)
    raise ValueError(f"missing time variable {time_group}/{time_var} in {src}")


def _load_diagnostic_variable(
    ds,
    *,
    src: Path,
    diagnostics_group: str,
    variable: str,
    kx_index: int | None,
) -> np.ndarray:
    diag_group = ds.groups.get(diagnostics_group)
    if diag_group is None:
        raise ValueError(f"missing NetCDF group {diagnostics_group!r} in {src}")
    if variable not in diag_group.variables:
        raise ValueError(f"missing diagnostics variable {variable!r} in {src}")
    raw = _decode_netcdf_values(diag_group.variables[variable])
    return _extract_diagnostic_values(raw, variable=variable, kx_index=kx_index)


def _align_complex_phase(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    nz = finite & (np.abs(values) > 1.0e-30)
    if np.any(nz):
        first = values[np.flatnonzero(nz)[0]]
        return values * np.exp(-1j * np.angle(first))
    return values


def _select_complex_component(
    values: np.ndarray,
    *,
    component: str,
    align_phase: bool,
) -> np.ndarray:
    values_arr = _align_complex_phase(values) if align_phase else values
    component_key = str(component).lower()
    if component_key == "complex":
        return values_arr
    if component_key == "real":
        return np.real(values_arr)
    if component_key == "imag":
        return np.imag(values_arr)
    if component_key == "abs":
        return np.abs(values_arr)
    raise ValueError("component must be one of {'real', 'imag', 'abs', 'complex'}")


def _select_real_component(values: np.ndarray, *, component: str) -> np.ndarray:
    if component not in {"real", "abs"}:
        raise ValueError("real diagnostics only support component='real' or 'abs'")
    if component == "abs":
        return np.abs(values)
    return np.asarray(values, dtype=float)


def _select_series_component(
    values: np.ndarray,
    *,
    component: str,
    align_phase: bool,
) -> np.ndarray:
    values_arr = np.asarray(values)
    if np.iscomplexobj(values_arr):
        return _select_complex_component(
            values_arr,
            component=component,
            align_phase=align_phase,
        )
    return _select_real_component(values_arr, component=component)


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
        values = _load_diagnostic_variable(
            ds,
            src=src,
            diagnostics_group=diagnostics_group,
            variable=variable,
            kx_index=kx_index,
        )
        t = _load_netcdf_time_axis(
            ds,
            src=src,
            time_group=time_group,
            time_var=time_var,
        )

    if t.ndim != 1 or t.size != values.size:
        raise ValueError(
            f"time axis for {variable!r} must be one-dimensional and match the diagnostics length"
        )
    selected = _select_series_component(
        values,
        component=component,
        align_phase=align_phase,
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
