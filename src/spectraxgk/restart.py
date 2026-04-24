"""Restart-state IO helpers.

SPECTRAX-GK reuses GX's flat complex64 restart layout so that:
- runtime `init_file` can consume restart files directly
- users can roundtrip state between GX and SPECTRAX for audits

The file format is a raw `complex64` buffer with no header. Consumers must
already know the target shape from (nspecies, Nl, Nm, Ny, Nx, Nz).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from jax.typing import ArrayLike


def _gx_active_kx_count(nx_full: int) -> int:
    return 1 + 2 * ((int(nx_full) - 1) // 3)


def _gx_active_ky_count(ny_full: int) -> int:
    return 1 + ((int(ny_full) - 1) // 3)


def _gx_active_kx_indices(nx_full: int) -> np.ndarray:
    nx = int(nx_full)
    split = 1 + ((nx - 1) // 3)
    if nx <= 1:
        return np.array([0], dtype=np.int32)
    neg = np.arange(2 * nx // 3 + 1, nx, dtype=np.int32)
    pos = np.arange(0, split, dtype=np.int32)
    return np.concatenate([neg, pos], axis=0)


def _expand_positive_ky_to_full(state_positive_ky: np.ndarray, *, ny_full: int) -> np.ndarray:
    state = np.asarray(state_positive_ky)
    if state.ndim != 6:
        raise ValueError("state_positive_ky must have shape (Ns, Nl, Nm, Nyc, Nx, Nz)")
    nyc = state.shape[3]
    expected_nyc = int(ny_full) // 2 + 1
    if nyc != expected_nyc:
        raise ValueError(f"positive-ky state Nyc={nyc} does not match ny_full={ny_full}")
    neg_hi = nyc - 1 if (int(ny_full) % 2) == 0 else nyc
    neg = np.conj(state[..., 1:neg_hi, :, :])[..., ::-1, :, :]
    nx = state.shape[4]
    if nx > 1:
        kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
        neg = neg[..., kx_neg, :]
    return np.concatenate([state, neg], axis=3)


def _expand_gx_restart_state_to_full_positive_ky(
    state_active: np.ndarray,
    *,
    ny_full: int,
    nx_full: int,
) -> np.ndarray:
    state = np.asarray(state_active)
    if state.ndim != 6:
        raise ValueError("state_active must have shape (Ns, Nl, Nm, Naky, Nakx, Nz)")
    nspec, nl, nm, naky, nakx, nz = state.shape
    nyc_full = int(ny_full) // 2 + 1
    expected_naky = _gx_active_ky_count(int(ny_full))
    expected_nakx = _gx_active_kx_count(int(nx_full))
    if naky != expected_naky:
        raise ValueError(f"restart Nky={naky} does not match ny_full={ny_full} (expected {expected_naky})")
    if nakx != expected_nakx:
        raise ValueError(f"restart Nkx={nakx} does not match nx_full={nx_full} (expected {expected_nakx})")
    out = np.zeros((nspec, nl, nm, nyc_full, int(nx_full), nz), dtype=np.complex64)
    out[..., :naky, _gx_active_kx_indices(int(nx_full)), :] = state
    return out


def _expand_gx_restart_state_full_ky(
    state_active: np.ndarray,
    *,
    nx_full: int,
) -> np.ndarray:
    """Expand a GX restart that already stores the full ``ky`` axis."""

    state = np.asarray(state_active)
    if state.ndim != 6:
        raise ValueError("state_active must have shape (Ns, Nl, Nm, Ny, Nakx, Nz)")
    nspec, nl, nm, ny_full, nakx, nz = state.shape
    expected_nakx = _gx_active_kx_count(int(nx_full))
    if nakx != expected_nakx:
        raise ValueError(f"restart Nkx={nakx} does not match nx_full={nx_full} (expected {expected_nakx})")
    out = np.zeros((nspec, nl, nm, int(ny_full), int(nx_full), nz), dtype=np.complex64)
    out[..., _gx_active_kx_indices(int(nx_full)), :] = state
    return out


def write_gx_restart_state(path: str | Path, state: ArrayLike) -> Path:
    """Write a restart state in GX-compatible flat complex64 layout."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(state, dtype=np.complex64).tofile(out)
    return out


def load_gx_restart_state(
    path: str | Path,
    *,
    nspecies: int,
    Nl: int,
    Nm: int,
    ny: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    """Load a GX NetCDF restart file into SPECTRAX's full Hermitian layout."""

    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("netCDF4 is required to load GX restart files") from exc

    with Dataset(Path(path), "r") as root:
        if "G" not in root.variables:
            raise ValueError(f"restart file {path} does not contain variable 'G'")
        raw = np.asarray(root.variables["G"][:], dtype=float)
    if raw.ndim != 7 or raw.shape[-1] != 2:
        raise ValueError(f"unexpected GX restart G shape {raw.shape}")
    state_active = raw[..., 0] + 1j * raw[..., 1]
    state_active = np.asarray(np.transpose(state_active, (0, 2, 1, 5, 4, 3)), dtype=np.complex64)
    if state_active.shape[:3] != (int(nspecies), int(Nl), int(Nm)):
        raise ValueError(
            f"restart state shape {state_active.shape[:3]} does not match requested {(int(nspecies), int(Nl), int(Nm))}"
        )
    if state_active.shape[-1] != int(nz):
        raise ValueError(f"restart Nz={state_active.shape[-1]} does not match requested {int(nz)}")
    if state_active.shape[3] == int(ny):
        return _expand_gx_restart_state_full_ky(state_active, nx_full=nx)
    positive_ky = _expand_gx_restart_state_to_full_positive_ky(state_active, ny_full=ny, nx_full=nx)
    return _expand_positive_ky_to_full(positive_ky, ny_full=ny)
