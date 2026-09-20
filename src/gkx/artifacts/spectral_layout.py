"""NetCDF spectral-output layout helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from gkx.core_ky_layout import half_ky_values, is_half, ky_row_weights, nyc_from_ny

NETCDF_SCHEMA_VERSION = 1

#: How a ``ky``-resolved diagnostic array counts a conjugate pair.
#:
#: ``"per_row"`` -- the value of that one stored row.  This is what the
#: two-sided reductions produce, because their weight is 1 on every row
#: (:func:`gkx.core_ky_layout.hermitian_mode_weights`): ``Phi2``, ``Wg``,
#: ``Wphi``, ``Wapar`` and ``TurbulentHeating``.  On a half axis the same
#: reduction carries the pair weight 2, so it has to be divided out before it
#: is published or the file's numbers would double.
#:
#: ``"pair"`` -- the sum over the conjugate pair, which the transport weights
#: (:func:`gkx.core_ky_layout.transport_mode_weights`) already produce on
#: *both* axes: ``HeatFlux*`` and ``ParticleFlux*``.  Nothing to undo.
#:
#: The distinction is not cosmetic and cannot be inferred from the array: two
#: arrays of the same shape holding the same modes differ by a factor of two,
#: and only the reduction that built them knows which.
KY_WEIGHTING_PER_ROW = "per_row"
KY_WEIGHTING_PAIR = "pair"
_KY_WEIGHTINGS = (KY_WEIGHTING_PER_ROW, KY_WEIGHTING_PAIR)


def _validate_netcdf_schema_version(root: Any) -> None:
    raw = root.getncattr("schema_version") if "schema_version" in root.ncattrs() else 0
    if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
        raise ValueError("NetCDF schema_version must be the integer 1")
    version = int(raw)
    if version not in (0, NETCDF_SCHEMA_VERSION):
        raise ValueError(
            f"unsupported GKX NetCDF schema_version {version}; upgrade GKX or "
            "migrate the artifact"
        )


def _require_netcdf4():
    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "netCDF4 is required to write nonlinear NetCDF output artifacts"
        ) from exc
    return Dataset


def _real_space_axis(length: int, extent: float) -> np.ndarray:
    return np.linspace(
        0.0, float(extent), int(length), endpoint=False, dtype=np.float32
    )


def _dealiased_kx_count(nx: int) -> int:
    return 1 + 2 * ((int(nx) - 1) // 3)


def _dealiased_ky_count(ny: int) -> int:
    return 1 + ((int(ny) - 1) // 3)


def _dealiased_kx_indices(nx: int) -> np.ndarray:
    nx_use = int(nx)
    split = 1 + ((nx_use - 1) // 3)
    if nx_use <= 1:
        return np.array([0], dtype=np.int32)
    neg = np.arange(2 * nx_use // 3 + 1, nx_use, dtype=np.int32)
    pos = np.arange(0, split, dtype=np.int32)
    return np.concatenate([neg, pos], axis=0)


def _dealiased_ky_indices(ny: int) -> np.ndarray:
    return np.arange(_dealiased_ky_count(int(ny)), dtype=np.int32)


def _dealiased_kx_values(kx: np.ndarray) -> np.ndarray:
    kx_arr = np.asarray(kx, dtype=np.float32)
    return kx_arr[_dealiased_kx_indices(kx_arr.shape[0])]


def _dealiased_ky_values(ky: np.ndarray, *, ny_full: int | None = None) -> np.ndarray:
    """Return the published ``ky`` axis: the dealiased ``ky >= 0`` magnitudes.

    ``ny_full`` is the length of the two-sided axis.  It is what makes the
    answer the same from either layout's ``grid.ky``: the number of dealiased
    rows is a property of ``Ny``, and a half-spectrum axis already holds
    ``Nyc`` rows, so taking the count from ``ky.shape[0]`` would publish
    ``1 + (Nyc - 1) // 3`` of them -- about a third of the band -- with an
    axis of the right dtype and the wrong length.
    """

    ky_arr = np.asarray(ky, dtype=np.float32)
    rows = int(ky_arr.shape[0])
    ny = rows if ny_full is None else int(ny_full)
    half = ky_arr if is_half(rows, ny_full) else half_ky_values(ky_arr)
    return np.abs(half[: _dealiased_ky_count(ny)])


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


def _spectral_species_to_xy(
    values: np.ndarray, *, ny_full: int | None = None
) -> np.ndarray:
    """Real-space ``(s, y, x, z)`` field of a ``(s, ky, kx, z)`` spectrum.

    The species-axis twin of :func:`_spectral_to_xy`; see its docstring for why
    a half-spectrum input must not go through the complex ``ifft2``.
    """

    arr = np.asarray(values)
    rows = int(arr.shape[1])
    ny = rows if ny_full is None else int(ny_full)
    if is_half(rows, ny_full):
        nkx = int(arr.shape[2])
        real = np.fft.irfft2(arr, s=(nkx, ny), axes=(2, 1))
        return real.astype(np.float32, copy=False)
    return np.real(np.fft.ifft2(arr, axes=(1, 2))).astype(np.float32, copy=False)


def _spectral_to_xy(field: np.ndarray, *, ny_full: int | None = None) -> np.ndarray:
    """Return the real-space ``(y, x, z)`` field of a ``(ky, kx, z)`` spectrum.

    ``ny_full`` is the length of the two-sided ``ky`` axis.  A two-sided field
    keeps the full complex ``ifft2``, whose imaginary part is discarded; a
    half-spectrum field (:mod:`gkx.core_ky_layout`) has no negative rows to
    transform and goes through ``irfft2``, which produces the same ``Ny``
    real-space rows and the same values without materialising them.  Taking
    ``np.real`` of a complex ``ifft2`` of ``Nyc`` rows instead would silently
    return an ``Nyc``-row image against an ``Ny``-long ``y`` axis.
    """

    arr = np.asarray(field)
    rows = int(arr.shape[0])
    ny = rows if ny_full is None else int(ny_full)
    if is_half(rows, ny_full):
        nkx = int(arr.shape[1])
        real = np.fft.irfft2(arr, s=(nkx, ny), axes=(1, 0))
        return real.astype(np.float32, copy=False)
    xy = np.fft.ifft2(arr, axes=(0, 1))
    return np.real(xy).astype(np.float32, copy=False)


def _restart_to_netcdf_layout(
    state: np.ndarray, *, ny_full: int | None = None
) -> np.ndarray:
    """Pack an evolved state into the restart file's dealiased ky/kx block.

    ``ny_full`` is the length of the two-sided ``ky`` axis.  The retained rows
    are ``0 .. 1 + (Ny - 1) // 3``, which is the same index range in both
    layouts -- the restart file has always stored the non-negative dealiased
    block -- but the count is a property of ``Ny``, not of how many rows the
    state carries.  Deriving it from ``state.shape[3]`` on a half-spectrum
    state keeps only ``1 + (Nyc - 1) // 3`` rows, about a third of the band,
    and writes that file without any error, so the length is taken from
    ``ny_full`` when it is supplied.
    """

    state_arr = np.asarray(state)
    if state_arr.ndim == 5:
        state_arr = state_arr[None, ...]
    if state_arr.ndim != 6:
        raise ValueError(
            "nonlinear state must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )
    rows = int(state_arr.shape[3])
    ky_idx = _dealiased_ky_indices(rows if ny_full is None else int(ny_full))
    if ky_idx.size > rows:
        raise ValueError(
            f"state stores {rows} ky rows but the dealiased restart block of "
            f"ny_full={ny_full} needs {int(ky_idx.size)}"
        )
    kx_idx = _dealiased_kx_indices(state_arr.shape[4])
    state_arr = _take_axis(state_arr, ky_idx, axis=3)
    state_arr = _take_axis(state_arr, kx_idx, axis=4)
    packed = np.transpose(state_arr, (0, 2, 1, 5, 4, 3))
    return np.stack([np.real(packed), np.imag(packed)], axis=-1).astype(
        np.float32, copy=False
    )


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
    root.setncattr("schema_version", NETCDF_SCHEMA_VERSION)
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
    code_info.setncattr("value", "gkx")


def _dealiased_spectral_field(
    field: np.ndarray,
    *,
    ky_axis: int = 0,
    kx_axis: int = 1,
    ny_full: int | None = None,
) -> np.ndarray:
    """Return the dealiased ``ky >= 0`` / ``kx`` block of a spectral field.

    A field is not a reduction, so no weight is involved: the retained rows
    are the same indices holding the same coefficients in either layout.  What
    still has to come from ``ny_full`` is *how many* of them there are, which
    is a property of the two-sided axis.
    """

    field_arr = np.asarray(field)
    rows = int(field_arr.shape[ky_axis])
    ky_idx = _dealiased_ky_indices(rows if ny_full is None else int(ny_full))
    if ky_idx.size > rows:
        raise ValueError(
            f"spectral field stores {rows} ky rows but the dealiased block of "
            f"ny_full={ny_full} needs {int(ky_idx.size)}"
        )
    kx_idx = _dealiased_kx_indices(field_arr.shape[kx_axis])
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
    return _take_axis(arr, _dealiased_kx_indices(np.asarray(arr).shape[-1]), axis=-1)


def _condense_ky(arr: np.ndarray) -> np.ndarray:
    return _take_axis(arr, _dealiased_ky_indices(np.asarray(arr).shape[-1]), axis=-1)


def _condense_kykx(arr: np.ndarray) -> np.ndarray:
    out = _take_axis(arr, _dealiased_ky_indices(np.asarray(arr).shape[-2]), axis=-2)
    return _take_axis(out, _dealiased_kx_indices(np.asarray(arr).shape[-1]), axis=-1)


def _condense_kx_for_output(
    arr: np.ndarray, *, full_nx: int, active_nx: int
) -> np.ndarray:
    """Return kx-resolved data on the dealiased output axis.

    Fresh in-memory diagnostics carry the full spectral ``kx`` axis, while
    history loaded from an existing existing NetCDF output bundle is already
    condensed.  External restart continuation appends both forms, so the writer
    must not apply the active-index selection a second time.
    """

    arr_np = np.asarray(arr)
    nx = int(arr_np.shape[-1])
    if nx == int(active_nx):
        return arr_np
    if nx == int(full_nx):
        return _take_axis(arr_np, _dealiased_kx_indices(int(full_nx)), axis=-1)
    raise ValueError(
        f"kx-resolved diagnostic has length {nx}; expected full Nx={full_nx} or active Nkx={active_nx}"
    )


def _half_ky_publish_factor(
    full_ny: int, *, ky_weighting: str | None, what: str
) -> np.ndarray | None:
    """Return the per-row factor that republishes a half-spectrum reduction.

    The published ``ky`` axis is the dealiased ``ky >= 0`` block, which is the
    *same rows* in both layouts -- so condensing a half-spectrum diagnostic is
    a row selection and nothing more, **as long as the two layouts agree on
    what a row's number means**.  They do not, for the reductions weighted by
    :func:`gkx.core_ky_layout.hermitian_mode_weights`: the two-sided axis
    stores each row once at weight 1, and the half axis carries the conjugate
    partner's share at weight 2.  Publishing the half value unchanged would
    double every paired row of ``Phi2``, ``Wg``, ``Wphi``, ``Wapar`` and
    ``TurbulentHeating`` -- and with them ``Phi2_t``, which the writer derives
    from the condensed spectrum.

    Returning the reciprocal weights restores the two-sided per-row value
    exactly: the weight is 1 or 2, and division by two is exact in binary
    floating point, so the published block is bitwise what the two-sided run
    would have written from the same state.

    ``None`` means "publish as it stands", which is what the transport
    reductions need: their weight is already the pair weight on both axes.
    """

    if ky_weighting is None:
        raise ValueError(
            f"{what} is a half-spectrum (Nyc-row) array, so publishing it needs "
            f"to know whether its rows count one row or a conjugate pair; pass "
            f"ky_weighting={KY_WEIGHTING_PER_ROW!r} or {KY_WEIGHTING_PAIR!r}"
        )
    if ky_weighting not in _KY_WEIGHTINGS:
        raise ValueError(
            f"unknown ky_weighting {ky_weighting!r}; expected one of {_KY_WEIGHTINGS}"
        )
    if ky_weighting == KY_WEIGHTING_PAIR:
        return None
    rows = _dealiased_ky_indices(int(full_ny))
    return 1.0 / ky_row_weights(int(full_ny))[rows]


def _condense_ky_for_output(
    arr: np.ndarray,
    *,
    full_ny: int,
    active_ny: int,
    ky_weighting: str | None = None,
) -> np.ndarray:
    """Return ky-resolved data on the dealiased positive-ky output axis.

    The input may already be condensed (``active_ny`` rows, which is how
    history reloaded from an existing bundle arrives), two-sided (``full_ny``
    rows), or half-spectrum (``Nyc = 1 + Ny // 2`` rows).  All three publish
    the same numbers; see :func:`_half_ky_publish_factor` for why the last one
    needs ``ky_weighting``.
    """

    arr_np = np.asarray(arr)
    ny = int(arr_np.shape[-1])
    if ny == int(active_ny):
        return arr_np
    if ny == int(full_ny):
        return _take_axis(arr_np, _dealiased_ky_indices(int(full_ny)), axis=-1)
    if ny == nyc_from_ny(int(full_ny)):
        factor = _half_ky_publish_factor(
            int(full_ny), ky_weighting=ky_weighting, what="ky-resolved diagnostic"
        )
        out = _take_axis(arr_np, _dealiased_ky_indices(int(full_ny)), axis=-1)
        return out if factor is None else out * factor.astype(out.dtype, copy=False)
    raise ValueError(
        f"ky-resolved diagnostic has length {ny}; expected full Ny={full_ny}, "
        f"half Nyc={nyc_from_ny(int(full_ny))} or active Nky={active_ny}"
    )


def _condense_kykx_for_output(
    arr: np.ndarray,
    *,
    full_ny: int,
    full_nx: int,
    active_ny: int,
    active_nx: int,
    ky_weighting: str | None = None,
) -> np.ndarray:
    """Return ky-kx-resolved data on dealiased output axes.

    ``ky_weighting`` carries the same contract as in
    :func:`_condense_ky_for_output` and is needed for the same reason.
    """

    arr_np = np.asarray(arr)
    ny = int(arr_np.shape[-2])
    nx = int(arr_np.shape[-1])
    if ny == int(active_ny) and nx == int(active_nx):
        return arr_np
    if ny == int(full_ny):
        arr_np = _take_axis(arr_np, _dealiased_ky_indices(int(full_ny)), axis=-2)
    elif ny == nyc_from_ny(int(full_ny)):
        factor = _half_ky_publish_factor(
            int(full_ny), ky_weighting=ky_weighting, what="ky-kx diagnostic"
        )
        arr_np = _take_axis(arr_np, _dealiased_ky_indices(int(full_ny)), axis=-2)
        if factor is not None:
            arr_np = arr_np * factor.astype(arr_np.dtype, copy=False)[:, None]
    elif ny != int(active_ny):
        raise ValueError(
            f"ky-kx diagnostic ky length {ny}; expected full Ny={full_ny}, "
            f"half Nyc={nyc_from_ny(int(full_ny))} or active Nky={active_ny}"
        )
    if nx == int(full_nx):
        arr_np = _take_axis(arr_np, _dealiased_kx_indices(int(full_nx)), axis=-1)
    elif nx != int(active_nx):
        raise ValueError(
            f"ky-kx diagnostic kx length {nx}; expected full Nx={full_nx} or active Nkx={active_nx}"
        )
    return arr_np


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


__all__ = [
    "KY_WEIGHTING_PAIR",
    "KY_WEIGHTING_PER_ROW",
    "infer_triple_dealiased_ny",
    "_half_ky_publish_factor",
    "_complex_to_ri",
    "_condense_kx",
    "_condense_kx_for_output",
    "_condense_ky",
    "_condense_ky_for_output",
    "_condense_kykx",
    "_condense_kykx_for_output",
    "_dealiased_spectral_field",
    "_dealiased_kx_count",
    "_dealiased_kx_indices",
    "_dealiased_kx_values",
    "_dealiased_ky_count",
    "_dealiased_ky_indices",
    "_dealiased_ky_values",
    "_maybe_var",
    "_real_space_axis",
    "_require_netcdf4",
    "_restart_to_netcdf_layout",
    "_species_matrix",
    "_spectral_species_to_ri",
    "_spectral_to_ri",
    "_spectral_species_to_xy",
    "_spectral_to_xy",
    "_state_basis_moments",
    "_take_axis",
    "_write_runtime_root_metadata",
]
