"""Tests for NetCDF restart-state IO helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset
import pytest

from gkx.artifacts.io import (
    _expand_netcdf_restart_state_full_ky,
    _expand_netcdf_restart_state_to_full_positive_ky,
    _expand_positive_ky_to_full,
    load_netcdf_restart_state,
    write_netcdf_restart_state,
)


def test_load_netcdf_restart_state_accepts_full_ky_reduced_kx_layout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gx.restart.nc"
    root = Dataset(path, "w")
    root.createDimension("Nspecies", 1)
    root.createDimension("Nm", 2)
    root.createDimension("Nl", 2)
    root.createDimension("Nz", 3)
    root.createDimension("Nkx", 3)
    root.createDimension("Nky", 4)
    root.createDimension("ri", 2)
    data = np.zeros((1, 2, 2, 3, 3, 4, 2), dtype=np.float32)
    data[0, 0, 0, 0, 0, 1, 0] = 1.5
    data[0, 0, 0, 1, 2, 3, 1] = -0.25
    root.createVariable("G", "f4", ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri"))[
        :
    ] = data
    root.close()

    state = load_netcdf_restart_state(path, nspecies=1, Nl=2, Nm=2, ny=4, nx=4, nz=3)

    assert state.shape == (1, 2, 2, 4, 4, 3)
    # GX's reduced kx axis is stored in active spectral order [negative, zero,
    # positive]. For nx=4 that maps active Nkx indices [0, 1, 2] to full kx
    # indices [3, 0, 1], matching the restart writer.
    assert state[0, 0, 0, 1, 3, 0] == np.complex64(1.5 + 0.0j)
    assert state[0, 0, 0, 3, 1, 1] == np.complex64(0.0 - 0.25j)
    assert state[0, 0, 0, 1, 0, 0] == np.complex64(0.0 + 0.0j)


def test_raw_restart_writer_and_positive_ky_expansion(tmp_path: Path) -> None:
    state = np.arange(1 * 1 * 1 * 3 * 2 * 2, dtype=np.float32).reshape(
        (1, 1, 1, 3, 2, 2)
    )
    state = state.astype(np.complex64) * (1.0 + 1.0j)
    path = write_netcdf_restart_state(tmp_path / "nested" / "state.restart", state)

    loaded = np.fromfile(path, dtype=np.complex64).reshape(state.shape)
    full = _expand_positive_ky_to_full(state, ny_full=4)

    assert path.exists()
    np.testing.assert_allclose(loaded, state)
    assert full.shape == (1, 1, 1, 4, 2, 2)
    np.testing.assert_allclose(full[..., :3, :, :], state)


def test_restart_expansion_helpers_fail_closed_on_shape_mismatches() -> None:
    with pytest.raises(ValueError, match="state_positive_ky"):
        _expand_positive_ky_to_full(np.zeros((2, 3)), ny_full=4)
    with pytest.raises(ValueError, match="does not match ny_full"):
        _expand_positive_ky_to_full(np.zeros((1, 1, 1, 2, 2, 1)), ny_full=4)

    with pytest.raises(ValueError, match="state_active"):
        _expand_netcdf_restart_state_to_full_positive_ky(
            np.zeros((2, 3)), ny_full=4, nx_full=4
        )
    with pytest.raises(ValueError, match="Nky"):
        _expand_netcdf_restart_state_to_full_positive_ky(
            np.zeros((1, 1, 1, 1, 3, 1)), ny_full=4, nx_full=4
        )
    with pytest.raises(ValueError, match="Nkx"):
        _expand_netcdf_restart_state_to_full_positive_ky(
            np.zeros((1, 1, 1, 2, 2, 1)), ny_full=4, nx_full=4
        )

    with pytest.raises(ValueError, match="state_active"):
        _expand_netcdf_restart_state_full_ky(np.zeros((2, 3)), nx_full=4)
    with pytest.raises(ValueError, match="Nkx"):
        _expand_netcdf_restart_state_full_ky(np.zeros((1, 1, 1, 4, 2, 1)), nx_full=4)


def test_load_netcdf_restart_state_rejects_malformed_netcdf(tmp_path: Path) -> None:
    missing_g = tmp_path / "missing_g.restart.nc"
    root = Dataset(missing_g, "w")
    root.close()
    with pytest.raises(ValueError, match="does not contain variable"):
        load_netcdf_restart_state(missing_g, nspecies=1, Nl=1, Nm=1, ny=4, nx=4, nz=1)

    bad_shape = tmp_path / "bad_shape.restart.nc"
    root = Dataset(bad_shape, "w")
    root.createDimension("x", 2)
    root.createVariable("G", "f4", ("x",))[:] = np.zeros(2, dtype=np.float32)
    root.close()
    with pytest.raises(ValueError, match="unexpected NetCDF restart G shape"):
        load_netcdf_restart_state(bad_shape, nspecies=1, Nl=1, Nm=1, ny=4, nx=4, nz=1)

    shape_mismatch = tmp_path / "shape_mismatch.restart.nc"
    root = Dataset(shape_mismatch, "w")
    root.createDimension("Nspecies", 2)
    root.createDimension("Nm", 1)
    root.createDimension("Nl", 1)
    root.createDimension("Nz", 1)
    root.createDimension("Nkx", 3)
    root.createDimension("Nky", 2)
    root.createDimension("ri", 2)
    root.createVariable("G", "f4", ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri"))[
        :
    ] = np.zeros((2, 1, 1, 1, 3, 2, 2), dtype=np.float32)
    root.close()
    with pytest.raises(ValueError, match="does not match requested"):
        load_netcdf_restart_state(
            shape_mismatch, nspecies=1, Nl=1, Nm=1, ny=4, nx=4, nz=1
        )

    nz_mismatch = tmp_path / "nz_mismatch.restart.nc"
    root = Dataset(nz_mismatch, "w")
    root.createDimension("Nspecies", 1)
    root.createDimension("Nm", 1)
    root.createDimension("Nl", 1)
    root.createDimension("Nz", 2)
    root.createDimension("Nkx", 3)
    root.createDimension("Nky", 2)
    root.createDimension("ri", 2)
    root.createVariable("G", "f4", ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri"))[
        :
    ] = np.zeros((1, 1, 1, 2, 3, 2, 2), dtype=np.float32)
    root.close()
    with pytest.raises(ValueError, match="restart Nz"):
        load_netcdf_restart_state(nz_mismatch, nspecies=1, Nl=1, Nm=1, ny=4, nx=4, nz=1)
