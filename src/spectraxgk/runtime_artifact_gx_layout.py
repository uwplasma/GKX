"""Pure GX-style runtime artifact layout helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

def _require_netcdf4():
    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "netCDF4 is required to write GX-style NetCDF runtime artifacts"
        ) from exc
    return Dataset

def _real_space_axis(length: int, extent: float) -> np.ndarray:
    return np.linspace(
        0.0, float(extent), int(length), endpoint=False, dtype=np.float32
    )

def _gx_active_kx_count(nx: int) -> int:
    return 1 + 2 * ((int(nx) - 1) // 3)

def _gx_active_ky_count(ny: int) -> int:
    return 1 + ((int(ny) - 1) // 3)

def _gx_active_kx_indices(nx: int) -> np.ndarray:
    nx_use = int(nx)
    split = 1 + ((nx_use - 1) // 3)
    if nx_use <= 1:
        return np.array([0], dtype=np.int32)
    neg = np.arange(2 * nx_use // 3 + 1, nx_use, dtype=np.int32)
    pos = np.arange(0, split, dtype=np.int32)
    return np.concatenate([neg, pos], axis=0)

def _gx_active_ky_indices(ny: int) -> np.ndarray:
    return np.arange(_gx_active_ky_count(int(ny)), dtype=np.int32)

def _gx_active_kx_values(kx: np.ndarray) -> np.ndarray:
    kx_arr = np.asarray(kx, dtype=np.float32)
    return kx_arr[_gx_active_kx_indices(kx_arr.shape[0])]

def _gx_active_ky_values(ky: np.ndarray) -> np.ndarray:
    ky_arr = np.asarray(ky, dtype=np.float32)
    nyc = 1 + ky_arr.shape[0] // 2
    return np.abs(ky_arr[:nyc])[: _gx_active_ky_count(int(ky_arr.shape[0]))]

def _take_axis(arr: np.ndarray, indices: np.ndarray, axis: int) -> np.ndarray:
    return np.take(np.asarray(arr), indices.astype(np.int32, copy=False), axis=axis)

def _spectral_to_ri(field: np.ndarray) -> np.ndarray:
    field_arr = np.asarray(field)
    if field_arr.ndim != 3:
        raise ValueError("field must have shape (Ny, Nx, Nz)")
    return np.stack([np.real(field_arr), np.imag(field_arr)], axis=-1).astype(
        np.float32, copy=False
    )

def _complex_to_ri(field: np.ndarray) -> np.ndarray:
    field_arr = np.asarray(field)
    return np.stack([np.real(field_arr), np.imag(field_arr)], axis=-1).astype(
        np.float32, copy=False
    )

def _spectral_to_xy(field: np.ndarray) -> np.ndarray:
    xy = np.fft.ifft2(np.asarray(field), axes=(0, 1))
    return np.real(xy).astype(np.float32, copy=False)

def _restart_to_gx_layout(state: np.ndarray) -> np.ndarray:
    state_arr = np.asarray(state)
    if state_arr.ndim == 5:
        state_arr = state_arr[None, ...]
    if state_arr.ndim != 6:
        raise ValueError(
            "nonlinear state must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )
    ky_idx = _gx_active_ky_indices(state_arr.shape[3])
    kx_idx = _gx_active_kx_indices(state_arr.shape[4])
    state_arr = _take_axis(state_arr, ky_idx, axis=3)
    state_arr = _take_axis(state_arr, kx_idx, axis=4)
    gx = np.transpose(state_arr, (0, 2, 1, 5, 4, 3))
    return np.stack([np.real(gx), np.imag(gx)], axis=-1).astype(np.float32, copy=False)

def _species_matrix(
    total: np.ndarray, nspecies: int, species_values: np.ndarray | None
) -> np.ndarray:
    total_arr = np.asarray(total, dtype=np.float32)
    ns = max(int(nspecies), 1)
    if species_values is not None:
        arr = np.asarray(species_values, dtype=np.float32)
        if arr.ndim == 1:
            return arr[:, None]
        return arr
    return np.broadcast_to(
        (total_arr / float(ns))[:, None], (total_arr.shape[0], ns)
    ).copy()

def _maybe_var(
    group: Any, name: str, dtype: str, dims: tuple[str, ...], values: np.ndarray
) -> None:
    var = group.createVariable(name, dtype, dims)
    var[...] = values

def _write_runtime_root_metadata(
    root: Any, cfg: Any, *, nspecies: int, nl: int, nm: int
) -> None:
    root.createVariable("ny", "i4", ())[:] = np.int32(cfg.grid.Ny)
    root.createVariable("nx", "i4", ())[:] = np.int32(cfg.grid.Nx)
    root.createVariable("ntheta", "i4", ())[:] = np.int32(
        cfg.grid.ntheta if cfg.grid.ntheta is not None else cfg.grid.Nz
    )
    root.createVariable("nhermite", "i4", ())[:] = np.int32(nm)
    root.createVariable("nlaguerre", "i4", ())[:] = np.int32(nl)
    root.createVariable("nspecies", "i4", ())[:] = np.int32(nspecies)
    root.createVariable("nperiod", "i4", ())[:] = np.int32(
        cfg.grid.nperiod if cfg.grid.nperiod is not None else 1
    )
    root.createVariable("debug", "i4", ())[:] = np.int32(0)
    code_info = root.createVariable("code_info", "i4", ())
    code_info[:] = np.int32(1)
    code_info.setncattr("value", "spectrax-gk")

def _gx_active_field(
    field: np.ndarray, *, ky_axis: int = 0, kx_axis: int = 1
) -> np.ndarray:
    field_arr = np.asarray(field)
    ky_idx = _gx_active_ky_indices(field_arr.shape[ky_axis])
    kx_idx = _gx_active_kx_indices(field_arr.shape[kx_axis])
    return _take_axis(_take_axis(field_arr, ky_idx, axis=ky_axis), kx_idx, axis=kx_axis)

def _spectral_species_to_ri(field: np.ndarray) -> np.ndarray:
    field_arr = np.asarray(field)
    if field_arr.ndim != 4:
        raise ValueError("field must have shape (Ns, Ny, Nx, Nz)")
    return np.stack([np.real(field_arr), np.imag(field_arr)], axis=-1).astype(
        np.float32, copy=False
    )

def _state_basis_moments(state: np.ndarray) -> dict[str, np.ndarray]:
    state_arr = np.asarray(state)
    if state_arr.ndim != 6:
        raise ValueError("state must have shape (Ns, Nl, Nm, Ny, Nx, Nz)")
    ns, nl, nm, _ny, _nx, nz = state_arr.shape
    zeros = np.zeros(
        (ns, state_arr.shape[3], state_arr.shape[4], nz), dtype=state_arr.dtype
    )
    density = state_arr[:, 0, 0, ...] if nl >= 1 and nm >= 1 else zeros
    upar = state_arr[:, 0, 1, ...] if nl >= 1 and nm >= 2 else zeros
    tpar = (
        np.sqrt(2.0, dtype=np.float32) * state_arr[:, 0, 2, ...]
        if nl >= 1 and nm >= 3
        else zeros
    )
    tperp = state_arr[:, 1, 0, ...] if nl >= 2 and nm >= 1 else zeros
    return {
        "Density": density,
        "Upar": upar,
        "Tpar": tpar,
        "Tperp": tperp,
    }

def _condense_kx(arr: np.ndarray) -> np.ndarray:
    return _take_axis(arr, _gx_active_kx_indices(np.asarray(arr).shape[-1]), axis=-1)

def _condense_ky(arr: np.ndarray) -> np.ndarray:
    return _take_axis(arr, _gx_active_ky_indices(np.asarray(arr).shape[-1]), axis=-1)

def _condense_kykx(arr: np.ndarray) -> np.ndarray:
    out = _take_axis(arr, _gx_active_ky_indices(np.asarray(arr).shape[-2]), axis=-2)
    return _take_axis(out, _gx_active_kx_indices(np.asarray(arr).shape[-1]), axis=-1)

def _condense_kx_for_output(
    arr: np.ndarray, *, full_nx: int, active_nx: int
) -> np.ndarray:
    """Return kx-resolved data on the GX-active output axis.

    Fresh in-memory diagnostics carry the full spectral ``kx`` axis, while
    history loaded from an existing GX-style ``out.nc`` bundle is already
    condensed.  External restart continuation appends both forms, so the writer
    must not apply the active-index selection a second time.
    """

    arr_np = np.asarray(arr)
    nx = int(arr_np.shape[-1])
    if nx == int(active_nx):
        return arr_np
    if nx == int(full_nx):
        return _take_axis(arr_np, _gx_active_kx_indices(int(full_nx)), axis=-1)
    raise ValueError(
        f"kx-resolved diagnostic has length {nx}; expected full Nx={full_nx} or active Nkx={active_nx}"
    )

def _condense_ky_for_output(
    arr: np.ndarray, *, full_ny: int, active_ny: int
) -> np.ndarray:
    """Return ky-resolved data on the GX-active positive-ky output axis."""

    arr_np = np.asarray(arr)
    ny = int(arr_np.shape[-1])
    if ny == int(active_ny):
        return arr_np
    if ny == int(full_ny):
        return _take_axis(arr_np, _gx_active_ky_indices(int(full_ny)), axis=-1)
    raise ValueError(
        f"ky-resolved diagnostic has length {ny}; expected full Ny={full_ny} or active Nky={active_ny}"
    )

def _condense_kykx_for_output(
    arr: np.ndarray,
    *,
    full_ny: int,
    full_nx: int,
    active_ny: int,
    active_nx: int,
) -> np.ndarray:
    """Return ky-kx-resolved data on GX-active output axes."""

    arr_np = np.asarray(arr)
    ny = int(arr_np.shape[-2])
    nx = int(arr_np.shape[-1])
    if ny == int(active_ny) and nx == int(active_nx):
        return arr_np
    if ny == int(full_ny):
        arr_np = _take_axis(arr_np, _gx_active_ky_indices(int(full_ny)), axis=-2)
    elif ny != int(active_ny):
        raise ValueError(
            f"ky-kx diagnostic ky length {ny}; expected full Ny={full_ny} or active Nky={active_ny}"
        )
    if nx == int(full_nx):
        arr_np = _take_axis(arr_np, _gx_active_kx_indices(int(full_nx)), axis=-1)
    elif nx != int(active_nx):
        raise ValueError(
            f"ky-kx diagnostic kx length {nx}; expected full Nx={full_nx} or active Nkx={active_nx}"
        )
    return arr_np

__all__ = [
    "_complex_to_ri",
    "_condense_kx",
    "_condense_kx_for_output",
    "_condense_ky",
    "_condense_ky_for_output",
    "_condense_kykx",
    "_condense_kykx_for_output",
    "_gx_active_field",
    "_gx_active_kx_count",
    "_gx_active_kx_indices",
    "_gx_active_kx_values",
    "_gx_active_ky_count",
    "_gx_active_ky_indices",
    "_gx_active_ky_values",
    "_maybe_var",
    "_real_space_axis",
    "_require_netcdf4",
    "_restart_to_gx_layout",
    "_species_matrix",
    "_spectral_species_to_ri",
    "_spectral_to_ri",
    "_spectral_to_xy",
    "_state_basis_moments",
    "_take_axis",
    "_write_runtime_root_metadata",
]
