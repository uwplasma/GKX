"""Readers for legacy GX grouped NetCDF outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GXLegacyCetgOutput:
    """Minimal legacy GX cETG diagnostic contract."""

    time: np.ndarray
    ky: np.ndarray
    kx: np.ndarray
    kz: np.ndarray
    x: np.ndarray
    y: np.ndarray
    W: np.ndarray
    Phi2: np.ndarray
    qflux: np.ndarray
    pflux: np.ndarray


@dataclass(frozen=True)
class GXLegacyCetgRestart:
    """Legacy GX cETG restart state in GX's positive-ky in-memory layout."""

    time: float
    state_positive_ky: np.ndarray
    nakx_active: int
    naky_active: int


_NETCDF_FILL_FLOAT = np.float64(9.969209968386869e36)


def _read_var(group, name: str) -> np.ndarray:
    return np.asarray(group.variables[name][:], dtype=float)


def _looks_like_fill(arr: np.ndarray) -> bool:
    arr_f = np.asarray(arr, dtype=float)
    if arr_f.size == 0:
        return True
    finite = np.isfinite(arr_f)
    if not np.any(finite):
        return True
    vals = arr_f[finite]
    return bool(np.all(np.abs(vals) >= 0.99 * _NETCDF_FILL_FLOAT))


def _legacy_active_kx_count(nx_full: int) -> int:
    return 1 + 2 * ((int(nx_full) - 1) // 3)


def _legacy_active_ky_count(ny_full: int) -> int:
    return 1 + ((int(ny_full) - 1) // 3)


def expand_gx_legacy_positive_ky_state(
    state_positive_ky: np.ndarray,
    *,
    ny_full: int,
) -> np.ndarray:
    """Expand GX's positive-ky real-FFT layout to a full Hermitian ``ky`` grid."""

    state = np.asarray(state_positive_ky)
    if state.ndim != 6 or state.shape[0] != 1 or state.shape[2] != 1:
        raise ValueError("state_positive_ky must have shape (1, 2, 1, Nyc, Nx, Nz)")
    pos = state[0, :, 0]
    nyc = pos.shape[1]
    expected_nyc = int(ny_full) // 2 + 1
    if nyc != expected_nyc:
        raise ValueError(f"positive-ky state Nyc={nyc} does not match ny_full={ny_full}")
    nx = pos.shape[2]
    neg_hi = nyc - 1 if (int(ny_full) % 2) == 0 else nyc
    neg = np.conj(pos[:, 1:neg_hi, :, :])[:, ::-1, :, :]
    if nx > 1:
        kx_neg = np.concatenate([np.array([0], dtype=np.int32), np.arange(nx - 1, 0, -1, dtype=np.int32)])
        neg = neg[:, :, kx_neg, :]
    full = np.concatenate([pos, neg], axis=1)
    return full[None, :, None, :, :, :]


def load_gx_legacy_cetg_restart(
    path: str | Path,
    *,
    nx_full: int,
    ny_full: int,
) -> GXLegacyCetgRestart:
    """Load a legacy GX cETG restart file into GX's positive-ky in-memory layout."""

    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("netCDF4 is required to load legacy GX cETG restart files") from exc

    root = Dataset(Path(path), "r")
    try:
        G_var = root.variables["G"]
        raw = np.asarray(G_var[:], dtype=float)
        if raw.ndim != 7:
            raise ValueError(f"Legacy GX cETG restart G has unsupported rank {raw.ndim}")
        nspec, nm, nl, nz, nakx, naky, ri = raw.shape
        if nspec != 1 or nm != 1 or nl != 2 or ri != 2:
            raise ValueError(
                "Legacy GX cETG restart must have shape (1, 1, 2, Nz, Nkx, Nky, 2); "
                f"got {raw.shape}"
            )
        nyc_full = int(ny_full) // 2 + 1
        state = np.zeros((1, nl, 1, nyc_full, int(nx_full), int(nz)), dtype=np.complex64)
        G_complex = raw[..., 0] + 1j * raw[..., 1]
        G_complex = G_complex[0, 0]  # (Nl, Nz, Nkx, Nky)

        expected_nakx = _legacy_active_kx_count(nx_full)
        expected_naky = _legacy_active_ky_count(ny_full)
        if int(nakx) != expected_nakx:
            raise ValueError(f"restart Nkx={nakx} does not match nx_full={nx_full} (expected {expected_nakx})")
        if int(naky) != expected_naky:
            raise ValueError(f"restart Nky={naky} does not match ny_full={ny_full} (expected {expected_naky})")

        for l in range(nl):
            for iz in range(nz):
                for i in range(1 + ((int(nx_full) - 1) // 3)):
                    for j in range(naky):
                        state[0, l, 0, j, i, iz] = G_complex[l, iz, i, j]
                for i in range(2 * int(nx_full) // 3 + 1, int(nx_full)):
                    it = i - 2 * int(nx_full) // 3 + ((int(nx_full) - 1) // 3)
                    for j in range(naky):
                        state[0, l, 0, j, i, iz] = G_complex[l, iz, it, j]

        time_var = np.asarray(root.variables["time"][:], dtype=float)
        time = float(time_var.reshape(-1)[0]) if time_var.size else 0.0
        return GXLegacyCetgRestart(
            time=time,
            state_positive_ky=state,
            nakx_active=int(nakx),
            naky_active=int(naky),
        )
    finally:
        root.close()


def load_gx_legacy_cetg_output(path: str | Path) -> GXLegacyCetgOutput:
    """Load the grouped legacy GX cETG NetCDF format."""

    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("netCDF4 is required to load legacy GX cETG outputs") from exc

    root = Dataset(Path(path), "r")
    try:
        spectra = root.groups["Spectra"]
        fluxes = root.groups["Fluxes"]
        W = _read_var(spectra, "W")
        if _looks_like_fill(W):
            Wkx = _read_var(spectra, "Wkxst")
            W = np.sum(Wkx, axis=tuple(range(1, Wkx.ndim)))
        Phi2 = _read_var(spectra, "Phi2t")
        if _looks_like_fill(Phi2):
            Phi2kx = _read_var(spectra, "Phi2kxt")
            Phi2 = np.sum(Phi2kx, axis=tuple(range(1, Phi2kx.ndim)))
        qflux = _read_var(fluxes, "qflux")
        pflux = _read_var(fluxes, "pflux")
        if _looks_like_fill(pflux):
            pflux = np.zeros_like(qflux)
        return GXLegacyCetgOutput(
            time=np.asarray(root.variables["time"][:], dtype=float),
            ky=np.asarray(root.variables["ky"][:], dtype=float),
            kx=np.asarray(root.variables["kx"][:], dtype=float),
            kz=np.asarray(root.variables["kz"][:], dtype=float),
            x=np.asarray(root.variables["x"][:], dtype=float),
            y=np.asarray(root.variables["y"][:], dtype=float),
            W=W,
            Phi2=Phi2,
            qflux=qflux,
            pflux=pflux,
        )
    finally:
        root.close()
