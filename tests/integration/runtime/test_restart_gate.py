from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gkx.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from gkx.artifacts.io import (
    load_netcdf_restart_state,
    write_netcdf_restart_state,
)
from gkx.runtime import run_runtime_nonlinear
from gkx.workflows.runtime.artifacts import (
    _condense_kx,
    _condense_kykx,
    _restart_to_netcdf_layout,
    load_nonlinear_netcdf_diagnostics,
    run_runtime_nonlinear_with_artifacts,
    write_runtime_nonlinear_artifacts,
)
from gkx.workflows.runtime.config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimeOutputConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)


def _restart_base_cfg() -> RuntimeConfig:
    ion = RuntimeSpeciesConfig(
        name="ion",
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        tprim=2.49,
        fprim=0.8,
    )
    return RuntimeConfig(
        grid=GridConfig(Nx=4, Ny=8, Nz=16, Lx=6.28, Ly=6.28, boundary="periodic"),
        time=TimeConfig(
            t_max=1.0,
            dt=0.02,
            method="rk2",
            use_diffrax=False,
            fixed_dt=True,
            diagnostics=True,
            sample_stride=1,
            diagnostics_stride=1,
        ),
        geometry=GeometryConfig(
            model="s-alpha", q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778
        ),
        init=InitializationConfig(
            init_field="density", init_amp=1.0e-8, init_single=True, gaussian_init=False
        ),
        species=(ion,),
        physics=RuntimePhysicsConfig(
            linear=False,
            nonlinear=True,
            adiabatic_electrons=True,
            tau_e=1.0,
            electrostatic=True,
            electromagnetic=False,
            collisions=False,
            hypercollisions=False,
        ),
        collisions=RuntimeCollisionConfig(damp_ends_amp=0.0),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
        terms=RuntimeTermsConfig(nonlinear=1.0, end_damping=0.0, hypercollisions=0.0),
    )


def test_netcdf_restart_roundtrips_zonal_radial_modes(tmp_path: Path) -> None:
    nc = pytest.importorskip("netCDF4")

    state = np.zeros((1, 3, 4, 8, 8, 6), dtype=np.complex64)
    state[0, 0, 0, 0, 1, :] = (1.0 + 0.25j) * np.arange(1, 7, dtype=np.float32)
    state[0, 1, 2, 0, 7, :] = (-0.5 + 0.75j) * np.arange(1, 7, dtype=np.float32)
    gx_state = _restart_to_netcdf_layout(state)

    path = tmp_path / "zonal_restart.nc"
    with nc.Dataset(path, "w") as root:
        root.createDimension("Nspecies", gx_state.shape[0])
        root.createDimension("Nm", gx_state.shape[1])
        root.createDimension("Nl", gx_state.shape[2])
        root.createDimension("Nz", gx_state.shape[3])
        root.createDimension("Nkx", gx_state.shape[4])
        root.createDimension("Nky", gx_state.shape[5])
        root.createDimension("ri", 2)
        root.createVariable(
            "G", "f4", ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri")
        )[:] = gx_state

    loaded = load_netcdf_restart_state(path, nspecies=1, Nl=3, Nm=4, ny=8, nx=8, nz=6)

    np.testing.assert_array_equal(loaded, state)


def test_restart_gate_nonlinear_matches_continuous(tmp_path: Path) -> None:
    cfg = _restart_base_cfg()
    Nl = 4
    Nm = 6
    dt = 0.02
    steps1 = 7
    steps2 = 9
    ky = 0.2
    kx = 0.0

    full = run_runtime_nonlinear(
        cfg,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps1 + steps2,
        sample_stride=1,
        diagnostics_stride=1,
        return_state=True,
    )
    part1 = run_runtime_nonlinear(
        cfg,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps1,
        sample_stride=1,
        diagnostics_stride=1,
        return_state=True,
    )
    assert part1.state is not None
    assert full.state is not None

    restart_path = tmp_path / "restart.bin"
    write_netcdf_restart_state(
        restart_path, np.asarray(part1.state, dtype=np.complex64)
    )

    cfg_restart = replace(
        cfg,
        init=replace(
            cfg.init,
            init_file=str(restart_path),
            init_file_scale=1.0,
            init_file_mode="replace",
        ),
    )
    part2 = run_runtime_nonlinear(
        cfg_restart,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps2,
        sample_stride=1,
        diagnostics_stride=1,
        return_state=True,
    )
    assert part2.state is not None

    np.testing.assert_array_equal(np.asarray(part2.state), np.asarray(full.state))


def test_restart_gate_nonlinear_matches_continuous_from_gx_netcdf(
    tmp_path: Path,
) -> None:
    pytest.importorskip("netCDF4")

    cfg = _restart_base_cfg()
    Nl = 4
    Nm = 6
    dt = 0.02
    steps1 = 7
    steps2 = 9
    ky = 0.2
    kx = 0.0

    full = run_runtime_nonlinear(
        cfg,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps1 + steps2,
        sample_stride=1,
        diagnostics_stride=1,
        return_state=True,
    )
    part1 = run_runtime_nonlinear(
        cfg,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps1,
        sample_stride=1,
        diagnostics_stride=1,
        return_state=True,
    )
    assert part1.state is not None
    assert full.state is not None

    paths = write_runtime_nonlinear_artifacts(tmp_path / "roundtrip.out.nc", part1, cfg)

    cfg_restart = replace(
        cfg,
        init=replace(
            cfg.init,
            init_file=str(paths["restart"]),
            init_file_scale=1.0,
            init_file_mode="replace",
        ),
    )
    part2 = run_runtime_nonlinear(
        cfg_restart,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps2,
        sample_stride=1,
        diagnostics_stride=1,
        return_state=True,
    )
    assert part2.state is not None

    np.testing.assert_array_equal(np.asarray(part2.state), np.asarray(full.state))


def test_restart_gate_append_on_restart_preserves_full_history(tmp_path: Path) -> None:
    pytest.importorskip("netCDF4")

    cfg = _restart_base_cfg()
    cfg = replace(
        cfg,
        output=RuntimeOutputConfig(
            path=str(tmp_path / "history.out.nc"),
            restart_if_exists=True,
            append_on_restart=True,
            save_for_restart=True,
            nsave=7,
        ),
    )

    Nl = 4
    Nm = 6
    dt = 0.02
    steps1 = 7
    steps2 = 9
    ky = 0.2
    kx = 0.0

    full = run_runtime_nonlinear(
        cfg,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps1 + steps2,
        sample_stride=1,
        diagnostics_stride=1,
        return_state=True,
    )
    out_path = tmp_path / "history.out.nc"

    part1, part1_paths = run_runtime_nonlinear_with_artifacts(
        cfg,
        out=out_path,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps1,
        sample_stride=1,
        diagnostics_stride=1,
        diagnostics=True,
    )
    assert part1.state is not None
    assert "restart" in part1_paths

    part2, _paths = run_runtime_nonlinear_with_artifacts(
        cfg,
        out=out_path,
        ky_target=ky,
        kx_target=kx,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps2,
        sample_stride=1,
        diagnostics_stride=1,
        diagnostics=True,
    )
    assert part2.state is not None
    assert full.state is not None

    np.testing.assert_array_equal(np.asarray(part2.state), np.asarray(full.state))
    loaded = load_nonlinear_netcdf_diagnostics(out_path)
    assert full.diagnostics is not None
    np.testing.assert_allclose(
        np.asarray(loaded.t), np.asarray(full.diagnostics.t), rtol=1.0e-6, atol=1.0e-8
    )
    np.testing.assert_allclose(
        np.asarray(loaded.Wg_t),
        np.asarray(full.diagnostics.Wg_t),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(loaded.Wphi_t),
        np.asarray(full.diagnostics.Wphi_t),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(loaded.Wapar_t),
        np.asarray(full.diagnostics.Wapar_t),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(loaded.heat_flux_t),
        np.asarray(full.diagnostics.heat_flux_t),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(loaded.particle_flux_t),
        np.asarray(full.diagnostics.particle_flux_t),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert full.diagnostics.turbulent_heating_t is not None
    np.testing.assert_allclose(
        np.asarray(loaded.turbulent_heating_t),
        np.asarray(full.diagnostics.turbulent_heating_t),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert loaded.resolved is not None
    assert full.diagnostics.resolved is not None
    np.testing.assert_allclose(
        np.asarray(loaded.resolved.Phi_zonal_line_kxt),
        _condense_kx(np.asarray(full.diagnostics.resolved.Phi_zonal_line_kxt)),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.asarray(loaded.resolved.Phi_zonal_mode_kxt),
        _condense_kx(np.asarray(full.diagnostics.resolved.Phi_zonal_mode_kxt)),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.asarray(loaded.resolved.Phi2_kxkyt),
        _condense_kykx(np.asarray(full.diagnostics.resolved.Phi2_kxkyt)),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
