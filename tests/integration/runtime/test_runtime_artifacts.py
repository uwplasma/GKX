from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gkx.diagnostics.analysis import ModeSelection
from gkx.config import GridConfig, TimeConfig
from gkx.diagnostics import SimulationDiagnostics, ResolvedDiagnostics
from gkx.geometry import FluxTubeGeometryData
from gkx.runtime import RuntimeLinearResult, RuntimeNonlinearResult
import gkx.workflows.runtime.artifacts as runtime_artifacts
import gkx.artifacts.spectral_layout as spectral_layout
import gkx.artifacts.nonlinear_netcdf as nonlinear_netcdf
import gkx.artifacts as artifact_package
import gkx.artifacts.io as artifact_io
import gkx.artifacts.io as artifact_linear
import gkx.artifacts.io as artifact_nonlinear
import gkx.artifacts.io as artifact_nonlinear_diag
from gkx.workflows.runtime.config import RuntimeConfig, RuntimeOutputConfig
from gkx.workflows.runtime.artifacts import (
    _ensure_parent,
    _artifact_base,
    _condense_kx,
    _condense_ky,
    _condense_kykx,
    _condense_diagnostics_for_netcdf_output,
    _condense_kx_for_output,
    _condense_ky_for_output,
    _condense_kykx_for_output,
    _condense_resolved_for_output,
    _flatten_series,
    _dealiased_spectral_field,
    _dealiased_ky_count,
    _dealiased_ky_indices,
    _dealiased_ky_values,
    _dealiased_kx_count,
    _dealiased_kx_indices,
    _dealiased_kx_values,
    _netcdf_bundle_base,
    _is_netcdf_output_target,
    _maybe_var,
    _nonlinear_summary,
    _particle_moments,
    _real_space_axis,
    _read_optional_var,
    _require_netcdf4,
    _resolved_species_time,
    _resolve_restart_path,
    _restart_to_netcdf_layout,
    _spectral_species_to_ri,
    _species_matrix,
    _state_basis_moments,
    _complex_to_ri,
    _spectral_to_ri,
    _spectral_to_xy,
    _take_axis,
    _write_csv,
    _write_geometry_group,
    _write_input_parameters_group,
    _write_json,
    write_runtime_linear_scan_artifacts,
    _write_nonlinear_netcdf_outputs,
    _write_runtime_root_metadata,
    _write_state,
    load_nonlinear_netcdf_diagnostics,
    run_runtime_nonlinear_with_artifacts,
    write_runtime_linear_artifacts,
    write_runtime_nonlinear_artifacts,
)
from gkx.artifacts.io import (
    validate_finite_array,
    validate_finite_runtime_result,
)
from gkx.workflows.runtime.diagnostic_arrays import concat_runtime_diagnostics
from gkx.workflows.runtime.orchestration_artifacts import (
    resolve_nonlinear_artifact_policy,
    run_runtime_nonlinear_artifact_handoff,
)


def test_runtime_artifacts_facade_reexports_split_helper_contracts() -> None:
    assert artifact_package.write_runtime_linear_artifacts is (
        artifact_linear.write_runtime_linear_artifacts
    )
    assert artifact_package.write_runtime_nonlinear_table_artifacts is (
        artifact_nonlinear.write_runtime_nonlinear_table_artifacts
    )
    assert artifact_package.load_nonlinear_netcdf_diagnostics is (
        artifact_nonlinear_diag.load_nonlinear_netcdf_diagnostics
    )
    assert runtime_artifacts._artifact_base is artifact_io._artifact_base
    assert runtime_artifacts._write_json is artifact_io._write_json
    assert runtime_artifacts._write_csv is artifact_io._write_csv
    assert runtime_artifacts._write_state is artifact_io._write_state
    assert runtime_artifacts._netcdf_bundle_base is artifact_io._netcdf_bundle_base
    assert artifact_package.validate_finite_array is artifact_io.validate_finite_array
    assert artifact_package.validate_finite_runtime_result is (
        artifact_io.validate_finite_runtime_result
    )
    assert (
        runtime_artifacts._is_netcdf_output_target
        is artifact_io._is_netcdf_output_target
    )
    assert (
        runtime_artifacts._dealiased_kx_indices is spectral_layout._dealiased_kx_indices
    )
    assert artifact_io._dealiased_kx_count is spectral_layout._dealiased_kx_count
    assert artifact_io._dealiased_kx_indices is spectral_layout._dealiased_kx_indices
    assert artifact_io._dealiased_ky_count is spectral_layout._dealiased_ky_count
    assert (
        runtime_artifacts._dealiased_ky_indices is spectral_layout._dealiased_ky_indices
    )
    assert (
        runtime_artifacts._condense_kykx_for_output
        is spectral_layout._condense_kykx_for_output
    )
    assert runtime_artifacts._spectral_to_ri is spectral_layout._spectral_to_ri
    assert (
        runtime_artifacts._restart_to_netcdf_layout
        is spectral_layout._restart_to_netcdf_layout
    )
    assert (
        runtime_artifacts._write_runtime_root_metadata
        is spectral_layout._write_runtime_root_metadata
    )
    assert (
        runtime_artifacts._resolve_restart_path
        is artifact_nonlinear_diag._resolve_restart_path
    )
    assert (
        runtime_artifacts._read_optional_var
        is artifact_nonlinear_diag._read_optional_var
    )
    assert (
        runtime_artifacts._condense_diagnostics_for_netcdf_output
        is artifact_nonlinear_diag._condense_diagnostics_for_netcdf_output
    )
    assert (
        runtime_artifacts.load_nonlinear_netcdf_diagnostics
        is artifact_nonlinear_diag.load_nonlinear_netcdf_diagnostics
    )
    assert (
        runtime_artifacts.write_quasilinear_artifacts
        is artifact_linear.write_quasilinear_artifacts
    )
    assert (
        runtime_artifacts.write_runtime_linear_artifacts
        is artifact_linear.write_runtime_linear_artifacts
    )
    assert (
        runtime_artifacts.write_runtime_linear_scan_artifacts
        is artifact_linear.write_runtime_linear_scan_artifacts
    )
    assert runtime_artifacts._nonlinear_summary is artifact_nonlinear._nonlinear_summary
    assert (
        runtime_artifacts.write_runtime_nonlinear_table_artifacts
        is artifact_nonlinear.write_runtime_nonlinear_table_artifacts
    )
    assert (
        runtime_artifacts._build_output_grid_and_geometry
        is nonlinear_netcdf._build_output_grid_and_geometry
    )
    assert runtime_artifacts._particle_moments is nonlinear_netcdf._particle_moments
    assert (
        runtime_artifacts._write_geometry_group
        is nonlinear_netcdf._write_geometry_group
    )
    assert (
        runtime_artifacts._write_input_parameters_group
        is nonlinear_netcdf._write_input_parameters_group
    )
    assert (
        runtime_artifacts._write_nonlinear_netcdf_outputs
        is nonlinear_netcdf._write_nonlinear_netcdf_outputs
    )


def test_write_runtime_linear_artifacts_writes_bundle(tmp_path: Path) -> None:
    result = RuntimeLinearResult(
        ky=0.2,
        gamma=0.3,
        omega=-0.4,
        selection=ModeSelection(ky_index=1, kx_index=2, z_index=3),
        t=np.asarray([0.1, 0.2, 0.3]),
        signal=np.asarray([1.0, 2.0, 4.0]),
        state=np.zeros((1, 2, 3), dtype=np.complex64),
        z=np.asarray([-1.0, 0.0, 1.0]),
        eigenfunction=np.asarray([0.5 + 0.0j, 1.0 + 0.2j, 0.5 + 0.1j]),
        fit_window_tmin=0.1,
        fit_window_tmax=0.3,
        fit_signal_used="phi",
        quasilinear={
            "ky": 0.2,
            "gamma": 0.3,
            "omega": -0.4,
            "mode": "saturated",
            "saturation_rule": "mixing_length",
            "amplitude_normalization": "phi_rms",
            "channels": ["es"],
            "kperp_average": "phi_weighted",
            "kperp_eff2": 0.5,
            "phi_norm2": 2.0,
            "amplitude2": 0.6,
            "species": ["ion"],
            "heat_flux_weight_species": [1.5],
            "particle_flux_weight_species": [0.25],
            "heat_flux_weight_total": 1.5,
            "particle_flux_weight_total": 0.25,
            "saturated_heat_flux_species": [0.9],
            "saturated_particle_flux_species": [0.15],
            "saturated_heat_flux_total": 0.9,
            "saturated_particle_flux_total": 0.15,
            "metadata": {"claim_level": "uncalibrated_saturation_rule"},
        },
    )

    paths = write_runtime_linear_artifacts(tmp_path / "linear_run", result)

    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["kind"] == "linear"
    assert summary["gamma"] == 0.3
    assert summary["fit_window_tmin"] == 0.1
    assert summary["fit_window_tmax"] == 0.3
    assert summary["fit_signal_used"] == "phi"
    assert summary["has_eigenfunction"] is True
    assert summary["has_quasilinear"] is True
    assert summary["quasilinear"]["heat_flux_weight_total"] == pytest.approx(1.5)
    assert (
        summary["quasilinear"]["metadata"]["claim_level"]
        == "uncalibrated_saturation_rule"
    )
    assert summary["selection"]["ky_index"] == 1
    csv_lines = Path(paths["timeseries"]).read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == "t,signal_real,signal_imag,signal_abs"
    assert Path(paths["timeseries"]).exists()
    assert Path(paths["eigenfunction"]).exists()
    assert Path(paths["state"]).exists()
    assert Path(paths["quasilinear_summary"]).exists()
    assert Path(paths["quasilinear_species"]).exists()
    ql_summary = json.loads(
        Path(paths["quasilinear_summary"]).read_text(encoding="utf-8")
    )
    assert ql_summary["metadata"]["claim_level"] == "uncalibrated_saturation_rule"


def test_runtime_artifact_file_helpers_and_nonlinear_summary(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "run.summary.json"
    csv_path = tmp_path / "nested" / "run.timeseries.csv"
    base = tmp_path / "nested" / "run"
    payload = {"beta": 0.1, "alpha": 2.0}

    _ensure_parent(json_path)
    assert json_path.parent.exists()

    _write_json(json_path, payload)
    assert json.loads(json_path.read_text(encoding="utf-8")) == payload

    _write_csv(
        csv_path, ["t", "value"], [np.asarray([0.0, 1.0]), np.asarray([2.0, 3.0])]
    )
    csv_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(csv_data, np.asarray([[0.0, 2.0], [1.0, 3.0]]))

    state = np.asarray([[1.0 + 2.0j]], dtype=np.complex64)
    state_path = _write_state(base, state)
    assert state_path is not None
    np.testing.assert_allclose(np.load(state_path), state)
    assert _write_state(base, None) is None

    diag = SimulationDiagnostics(
        t=np.asarray([0.1, 0.2]),
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.01, 0.02]),
        omega_t=np.asarray([0.03, 0.04]),
        Wg_t=np.asarray([1.0, 1.1]),
        Wphi_t=np.asarray([2.0, 2.1]),
        Wapar_t=np.asarray([0.5, 0.6]),
        heat_flux_t=np.asarray([3.0, 3.1]),
        particle_flux_t=np.asarray([4.0, 4.1]),
        energy_t=np.asarray([3.5, 3.8]),
    )
    summary = _nonlinear_summary(
        SimpleNamespace(
            diagnostics=diag,
            state=np.zeros((1, 2, 3), dtype=np.complex64),
            ky_selected=0.2,
            kx_selected=0.1,
            phi2=None,
        )
    )
    assert summary["kind"] == "nonlinear"
    assert summary["n_samples"] == 2
    assert summary["heat_flux_last"] == pytest.approx(3.1)
    assert summary["n_state_shape"] == [1, 2, 3]

    scalar_only = _nonlinear_summary(
        SimpleNamespace(
            diagnostics=None,
            state=None,
            ky_selected=0.2,
            kx_selected=0.0,
            phi2=np.asarray(7.0),
        )
    )
    assert scalar_only["phi2_last"] == pytest.approx(7.0)


def test_write_runtime_linear_artifacts_splits_complex_signal_columns(
    tmp_path: Path,
) -> None:
    result = RuntimeLinearResult(
        ky=0.2,
        gamma=0.3,
        omega=-0.4,
        selection=ModeSelection(ky_index=1, kx_index=2, z_index=3),
        t=np.asarray([0.1, 0.2]),
        signal=np.asarray([1.0 + 2.0j, 3.0 + 4.0j]),
        state=None,
        fit_signal_used="phi",
    )

    paths = write_runtime_linear_artifacts(tmp_path / "linear_complex", result)

    rows = Path(paths["timeseries"]).read_text(encoding="utf-8").splitlines()
    assert rows[0] == "t,signal_real,signal_imag,signal_abs"
    data = np.loadtxt(paths["timeseries"], delimiter=",", skiprows=1)
    np.testing.assert_allclose(data[:, 1], np.asarray([1.0, 3.0]))
    np.testing.assert_allclose(data[:, 2], np.asarray([2.0, 4.0]))
    np.testing.assert_allclose(data[:, 3], np.abs(np.asarray([1.0 + 2.0j, 3.0 + 4.0j])))


def test_write_runtime_linear_scan_artifacts_with_quasilinear_spectrum(
    tmp_path: Path,
) -> None:
    result = SimpleNamespace(
        ky=np.asarray([0.2, 0.3]),
        gamma=np.asarray([0.1, 0.2]),
        omega=np.asarray([-0.4, -0.5]),
        parallel={
            "requested_workers": 2,
            "effective_workers": 2,
            "executor": "thread",
            "identity_contract": "independent ky workers must preserve serial ky ordering and values",
            "quasilinear_state_extraction": True,
        },
        quasilinear=(
            {
                "ky": 0.2,
                "gamma": 0.1,
                "omega": -0.4,
                "kperp_eff2": 0.8,
                "heat_flux_weight_total": 1.2,
                "particle_flux_weight_total": 0.1,
                "amplitude2": None,
                "saturated_heat_flux_total": None,
                "saturated_particle_flux_total": None,
            },
            {
                "ky": -0.3,
                "gamma": 0.2,
                "omega": -0.5,
                "kperp_eff2": 1.0,
                "heat_flux_weight_total": 1.5,
                "particle_flux_weight_total": 0.2,
                "amplitude2": 0.4,
                "saturated_heat_flux_total": 0.6,
                "saturated_particle_flux_total": 0.08,
            },
        ),
    )

    paths = write_runtime_linear_scan_artifacts(tmp_path / "scan_bundle", result)

    assert Path(paths["summary"]).exists()
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["parallel"]["requested_workers"] == 2
    assert summary["parallel"]["quasilinear_state_extraction"] is True
    assert set(summary["parallel"]) == {
        "requested_workers",
        "effective_workers",
        "executor",
        "identity_contract",
        "quasilinear_state_extraction",
    }
    assert "serial ky ordering" in summary["parallel"]["identity_contract"]
    assert Path(paths["scan"]).exists()
    assert Path(paths["quasilinear_spectrum"]).exists()
    spectrum = np.loadtxt(paths["quasilinear_spectrum"], delimiter=",", skiprows=1)
    np.testing.assert_allclose(spectrum[:, 0], [0.2, 0.3])
    np.testing.assert_allclose(spectrum[:, 1], [0.2, -0.3])
    np.testing.assert_allclose(spectrum[:, 5], [1.2, 1.5])
    assert np.isnan(spectrum[0, 7])
    assert spectrum[1, 8] == pytest.approx(0.6)


def test_write_runtime_linear_scan_artifacts_handles_quasilinear_length_mismatch(
    tmp_path: Path,
) -> None:
    result = SimpleNamespace(
        ky=np.asarray([0.2, 0.3]),
        gamma=np.asarray([0.1, 0.2]),
        omega=np.asarray([-0.4, -0.5]),
        quasilinear=(
            {
                "ky": -0.25,
                "gamma": 0.11,
                "omega": -0.44,
                "kperp_eff2": 0.8,
                "heat_flux_weight_total": 1.2,
                "particle_flux_weight_total": 0.1,
            },
        ),
    )

    paths = write_runtime_linear_scan_artifacts(tmp_path / "mismatch_scan", result)

    spectrum = np.loadtxt(paths["quasilinear_spectrum"], delimiter=",", skiprows=1)
    assert spectrum.shape == (10,)
    assert spectrum[0] == pytest.approx(-0.25)
    assert spectrum[1] == pytest.approx(-0.25)
    assert spectrum[5] == pytest.approx(1.2)


def test_runtime_artifact_helper_paths_and_flattening(tmp_path: Path) -> None:
    assert _artifact_base(tmp_path / "case.summary.json") == tmp_path / "case.summary"
    assert (
        _artifact_base(tmp_path / "case.timeseries.csv") == tmp_path / "case.timeseries"
    )
    assert (
        _artifact_base(tmp_path / "case.eigenfunction.csv")
        == tmp_path / "case.eigenfunction"
    )
    assert (
        _artifact_base(tmp_path / "case.diagnostics.csv")
        == tmp_path / "case.diagnostics"
    )
    assert _artifact_base(tmp_path / "case.out.nc") == tmp_path / "case.out.nc"
    assert _netcdf_bundle_base(tmp_path / "case.nc") == tmp_path / "case"
    assert _netcdf_bundle_base(tmp_path / "case.restart.nc") == tmp_path / "case"
    assert _netcdf_bundle_base(tmp_path / "case.big.nc") == tmp_path / "case"
    assert _is_netcdf_output_target(tmp_path / "case.out.nc") is True
    assert _is_netcdf_output_target(tmp_path / "case.csv") is False

    assert np.allclose(_flatten_series(np.array([1.0, 2.0])), np.array([1.0, 2.0]))
    assert np.allclose(_flatten_series(np.array([[1.0], [2.0]])), np.array([1.0, 2.0]))
    assert np.allclose(
        _flatten_series(np.array([[1.0, 3.0], [2.0, 4.0]])), np.array([2.0, 3.0])
    )


def test_runtime_artifact_restart_resolution_and_species_helpers() -> None:
    cfg = RuntimeConfig(output=RuntimeOutputConfig(path="tools_out/run.out.nc"))
    assert (
        _resolve_restart_path("tools_out/run.out.nc", cfg, for_write=True).name
        == "run.restart.nc"
    )
    assert (
        _resolve_restart_path("tools_out/run.out.nc", cfg, for_write=False).name
        == "run.restart.nc"
    )

    cfg_custom = RuntimeConfig(
        output=RuntimeOutputConfig(
            path="tools_out/run.out.nc",
            restart_to_file="custom_to.nc",
            restart_from_file="custom_from.nc",
        )
    )
    assert (
        _resolve_restart_path("tools_out/run.out.nc", cfg_custom, for_write=True).name
        == "custom_to.nc"
    )
    assert (
        _resolve_restart_path("tools_out/run.out.nc", cfg_custom, for_write=False).name
        == "custom_from.nc"
    )

    policy_cfg = RuntimeConfig(
        time=TimeConfig(dt=0.2, t_max=1.0, fixed_dt=True, diagnostics=True),
        output=RuntimeOutputConfig(
            path="tools_out/policy.out.nc", save_for_restart=True, nsave=2
        ),
    )
    deps = SimpleNamespace(
        is_netcdf_output_target=lambda path: Path(path).suffix == ".nc",
        resolve_restart_path=lambda _path, _cfg: Path("policy_from.restart.nc"),
        resolve_restart_write_path=lambda _path, _cfg: Path("policy_to.restart.nc"),
    )
    policy = resolve_nonlinear_artifact_policy(
        policy_cfg,
        out="tools_out/policy.out.nc",
        diagnostics=None,
        steps=None,
        dt=None,
        deps=deps,
    )
    assert policy.netcdf_output_target is True
    assert policy.diagnostics_on is True
    assert policy.remaining_steps == 5
    assert policy.checkpoint_steps == 2
    assert policy.restart_from == Path("policy_from.restart.nc")
    assert policy.restart_to == Path("policy_to.restart.nc")

    total = np.array([2.0, 4.0], dtype=np.float32)
    assert np.allclose(
        _species_matrix(total, 2, None),
        np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
    )
    assert np.allclose(
        _species_matrix(total, 2, np.array([3.0, 4.0], dtype=np.float32)),
        np.array([[3.0], [4.0]], dtype=np.float32),
    )
    assert np.allclose(_resolved_species_time(None, fallback=total), total)
    assert np.allclose(
        _resolved_species_time(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), fallback=total
        ),
        np.array([3.0, 7.0], dtype=np.float32),
    )

    species_matrix = _species_matrix(
        total,
        2,
        np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        species_matrix,
        np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )


def test_runtime_artifact_finite_validation_covers_state_and_fields() -> None:
    validate_finite_array(None, label="empty")
    validate_finite_array(np.asarray([]), label="empty")
    validate_finite_array(np.asarray([1.0, 2.0]), label="finite")

    with pytest.raises(RuntimeError, match="bad field contains non-finite values"):
        validate_finite_array(np.asarray([1.0, np.inf]), label="bad field")

    finite = RuntimeNonlinearResult(
        t=np.asarray([]),
        diagnostics=None,
        state=np.ones((1,), dtype=np.complex64),
        fields=SimpleNamespace(
            phi=np.ones((1,), dtype=np.complex64),
            apar=None,
            bpar=np.zeros((1,), dtype=np.complex64),
        ),
    )
    validate_finite_runtime_result(finite, label="finite nonlinear")

    bad_state = replace(finite, state=np.asarray([np.nan], dtype=np.float32))
    with pytest.raises(RuntimeError, match="finite nonlinear state"):
        validate_finite_runtime_result(bad_state, label="finite nonlinear")

    bad_phi = replace(
        finite,
        fields=SimpleNamespace(
            phi=np.asarray([1.0 + np.nan * 1j]),
            apar=np.asarray([0.0]),
            bpar=np.asarray([0.0]),
        ),
    )
    with pytest.raises(RuntimeError, match="finite nonlinear phi"):
        validate_finite_runtime_result(bad_phi, label="finite nonlinear")

    bad_apar = replace(
        finite,
        fields=SimpleNamespace(
            phi=np.asarray([0.0]),
            apar=np.asarray([np.inf]),
            bpar=np.asarray([0.0]),
        ),
    )
    with pytest.raises(RuntimeError, match="finite nonlinear apar"):
        validate_finite_runtime_result(bad_apar, label="finite nonlinear")

    bad_bpar = replace(
        finite,
        fields=SimpleNamespace(
            phi=np.asarray([0.0]),
            apar=np.asarray([0.0]),
            bpar=np.asarray([np.nan]),
        ),
    )
    with pytest.raises(RuntimeError, match="finite nonlinear bpar"):
        validate_finite_runtime_result(bad_bpar, label="finite nonlinear")

    bad_diag = SimulationDiagnostics(
        t=np.asarray([0.1, 0.2]),
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.zeros(2),
        omega_t=np.zeros(2),
        Wg_t=np.ones(2),
        Wphi_t=np.ones(2),
        Wapar_t=np.zeros(2),
        heat_flux_t=np.zeros(2),
        particle_flux_t=np.asarray([0.0, np.nan]),
        energy_t=np.ones(2),
    )
    bad_diag_result = replace(finite, diagnostics=bad_diag, fields=None, state=None)
    with pytest.raises(
        RuntimeError,
        match=r"finite nonlinear produced non-finite diagnostics in particle_flux_t at sample 1 at t=0.2",
    ):
        validate_finite_runtime_result(bad_diag_result, label="finite nonlinear")


def test_runtime_artifact_condense_output_helpers_reject_bad_axis_lengths() -> None:
    full_kx = np.arange(7, dtype=np.float32)
    active_kx = full_kx[_dealiased_kx_indices(full_kx.size)]
    np.testing.assert_allclose(
        _condense_kx_for_output(full_kx, full_nx=7, active_nx=active_kx.size),
        active_kx,
    )
    np.testing.assert_allclose(
        _condense_kx_for_output(active_kx, full_nx=7, active_nx=active_kx.size),
        active_kx,
    )
    with pytest.raises(ValueError, match="kx-resolved diagnostic"):
        _condense_kx_for_output(np.arange(6), full_nx=7, active_nx=active_kx.size)

    full_ky = np.arange(5, dtype=np.float32)
    active_ky = full_ky[_dealiased_ky_indices(full_ky.size)]
    np.testing.assert_allclose(
        _condense_ky_for_output(full_ky, full_ny=5, active_ny=active_ky.size),
        active_ky,
    )
    np.testing.assert_allclose(
        _condense_ky_for_output(active_ky, full_ny=5, active_ny=active_ky.size),
        active_ky,
    )
    with pytest.raises(ValueError, match="ky-resolved diagnostic"):
        _condense_ky_for_output(np.arange(4), full_ny=5, active_ny=active_ky.size)

    full = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    active = full[_dealiased_ky_indices(5)][:, _dealiased_kx_indices(7)]
    np.testing.assert_allclose(
        _condense_kykx_for_output(
            full,
            full_ny=5,
            full_nx=7,
            active_ny=active.shape[0],
            active_nx=active.shape[1],
        ),
        active,
    )
    partially_condensed = full[_dealiased_ky_indices(5)]
    np.testing.assert_allclose(
        _condense_kykx_for_output(
            partially_condensed,
            full_ny=5,
            full_nx=7,
            active_ny=active.shape[0],
            active_nx=active.shape[1],
        ),
        active,
    )
    with pytest.raises(ValueError, match="ky-kx diagnostic ky length"):
        _condense_kykx_for_output(
            np.zeros((3, 7)),
            full_ny=5,
            full_nx=7,
            active_ny=active.shape[0],
            active_nx=active.shape[1],
        )
    with pytest.raises(ValueError, match="ky-kx diagnostic kx length"):
        _condense_kykx_for_output(
            np.zeros((active.shape[0], 6)),
            full_ny=5,
            full_nx=7,
            active_ny=active.shape[0],
            active_nx=active.shape[1],
        )


def test_runtime_artifact_spectral_helpers() -> None:
    field = np.array([[[1.0 + 2.0j, 3.0 + 4.0j]]], dtype=np.complex64)
    ri = _spectral_to_ri(field)
    assert ri.shape == (1, 1, 2, 2)
    assert np.allclose(ri[0, 0, 0], np.array([1.0, 2.0]))

    xy = _spectral_to_xy(np.ones((2, 2, 1), dtype=np.complex64))
    assert xy.shape == (2, 2, 1)

    state = np.ones((1, 2, 3, 4, 4, 5), dtype=np.complex64)
    netcdf_layout = _restart_to_netcdf_layout(state)
    assert netcdf_layout.shape[-1] == 2
    netcdf_from_5d = _restart_to_netcdf_layout(
        np.ones((2, 3, 4, 4, 5), dtype=np.complex64)
    )
    assert netcdf_from_5d.shape[0] == 1

    with pytest.raises(ValueError):
        _spectral_to_ri(np.ones((2, 2), dtype=np.complex64))
    with pytest.raises(ValueError):
        _spectral_species_to_ri(np.ones((2, 2), dtype=np.complex64))
    with pytest.raises(ValueError):
        _restart_to_netcdf_layout(np.ones((2, 2), dtype=np.complex64))
    with pytest.raises(ValueError):
        _state_basis_moments(np.ones((2, 2), dtype=np.complex64))

    ri_series = _complex_to_ri(np.array([[1.0 + 2.0j, 3.0 + 4.0j]], dtype=np.complex64))
    assert ri_series.shape == (1, 2, 2)


def test_write_runtime_nonlinear_artifacts_preserves_active_resolved_axes(
    tmp_path: Path,
) -> None:
    nc = pytest.importorskip("netCDF4")

    cfg = RuntimeConfig(
        grid=GridConfig(Nx=6, Ny=4, Nz=4, Lx=6.0, Ly=6.0, boundary="periodic"),
        time=TimeConfig(dt=0.1, t_max=0.2, diagnostics=True, fixed_dt=True),
        output=RuntimeOutputConfig(
            path=str(tmp_path / "active.out.nc"), save_for_restart=False
        ),
    )
    active_kx = _dealiased_kx_count(cfg.grid.Nx)
    active_ky = _dealiased_ky_count(cfg.grid.Ny)
    line = np.asarray(
        [
            [1.0 + 0.5j, 2.0 + 0.25j, 3.0 + 0.125j],
            [4.0 + 0.5j, 5.0 + 0.25j, 6.0 + 0.125j],
        ],
        dtype=np.complex64,
    )
    phi2_kykx = np.arange(2 * active_ky * active_kx, dtype=np.float32).reshape(
        2, active_ky, active_kx
    )
    wg_kxst = np.arange(2 * active_kx, dtype=np.float32).reshape(2, 1, active_kx)
    diag = SimulationDiagnostics(
        t=np.asarray([0.1, 0.2]),
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.zeros(2),
        omega_t=np.zeros(2),
        Wg_t=np.asarray([1.0, 2.0]),
        Wphi_t=np.asarray([0.5, 0.6]),
        Wapar_t=np.zeros(2),
        heat_flux_t=np.zeros(2),
        particle_flux_t=np.zeros(2),
        energy_t=np.asarray([1.5, 2.6]),
        resolved=ResolvedDiagnostics(
            Phi2_kxkyt=phi2_kykx,
            Phi_zonal_line_kxt=line,
            Phi_zonal_mode_kxt=2.0 * line,
            Wg_kxst=wg_kxst,
        ),
    )
    result = RuntimeNonlinearResult(
        t=np.asarray(diag.t),
        diagnostics=diag,
        state=None,
        fields=None,
        ky_selected=0.0,
        kx_selected=0.0,
    )

    paths = write_runtime_nonlinear_artifacts(tmp_path / "active.out.nc", result, cfg)

    with nc.Dataset(paths["out"], "r") as root:
        assert root.dimensions["kx"].size == active_kx
        assert root.dimensions["ky"].size == active_ky
        diagnostics = root.groups["Diagnostics"]
        stored_line_ri = np.asarray(diagnostics.variables["Phi_zonal_line_kxt"][:])
        stored_line = stored_line_ri[..., 0] + 1j * stored_line_ri[..., 1]
        stored_wg = np.asarray(diagnostics.variables["Wg_kxst"][:])
        stored_phi2 = np.asarray(diagnostics.variables["Phi2_kxkyt"][:])

    np.testing.assert_allclose(stored_line, line)
    np.testing.assert_allclose(stored_wg, wg_kxst)
    np.testing.assert_allclose(stored_phi2, phi2_kykx)


def test_runtime_artifact_read_optional_var() -> None:
    class _Group:
        variables = {"present": np.array([1.0, 2.0])}

    assert _read_optional_var(_Group, "missing") is None


def test_runtime_artifact_read_optional_var_converts_ri_dimension() -> None:
    class _Var:
        dimensions = ("time", "ri")

        def __getitem__(self, _key):
            return np.array([[1.0, 2.0], [3.0, 4.0]])

    class _Group:
        variables = {"present": _Var()}

    out = _read_optional_var(_Group, "present")
    assert np.allclose(out, np.array([1.0 + 2.0j, 3.0 + 4.0j]))


def test_runtime_artifact_condense_helpers() -> None:
    assert _condense_resolved_for_output(None) is None

    resolved = ResolvedDiagnostics(
        Phi2_kxt=np.ones((2, 8), dtype=float),
        Phi2_kyt=np.ones((2, 8), dtype=float),
        Phi2_kxkyt=np.ones((2, 8, 8), dtype=float),
        Phi2_zt=np.ones((2, 6), dtype=float),
        Phi2_zonal_t=np.ones((2,), dtype=float),
        Phi2_zonal_kxt=np.ones((2, 8), dtype=float),
        Phi2_zonal_zt=np.ones((2, 6), dtype=float),
        Phi_zonal_mode_kxt=np.ones((2, 8), dtype=np.complex64),
        Wg_kxst=np.ones((2, 1, 8), dtype=float),
        Wg_kyst=np.ones((2, 1, 8), dtype=float),
        Wg_kxkyst=np.ones((2, 1, 8, 8), dtype=float),
        Wg_zst=np.ones((2, 1, 6), dtype=float),
        Wg_lmst=np.ones((2, 1, 8, 4), dtype=float),
        Wphi_kxst=np.ones((2, 1, 8), dtype=float),
        Wphi_kyst=np.ones((2, 1, 8), dtype=float),
        Wphi_kxkyst=np.ones((2, 1, 8, 8), dtype=float),
        Wphi_zst=np.ones((2, 1, 6), dtype=float),
    )
    condensed = _condense_resolved_for_output(resolved)
    assert condensed is not None
    assert condensed.Wg_kxst.shape[-1] <= resolved.Wg_kxst.shape[-1]
    assert (
        condensed.Phi_zonal_mode_kxt.shape[-1] <= resolved.Phi_zonal_mode_kxt.shape[-1]
    )
    diag = SimulationDiagnostics(
        t=np.asarray([0.0, 0.1]),
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.0, 0.0]),
        omega_t=np.asarray([0.0, 0.0]),
        Wg_t=np.asarray([1.0, 1.1]),
        Wphi_t=np.asarray([2.0, 2.1]),
        Wapar_t=np.asarray([0.0, 0.0]),
        heat_flux_t=np.asarray([3.0, 3.1]),
        particle_flux_t=np.asarray([4.0, 4.1]),
        energy_t=np.asarray([3.0, 3.2]),
        resolved=resolved,
    )
    diag_condensed = _condense_diagnostics_for_netcdf_output(diag)
    assert diag_condensed.resolved is not None


def test_runtime_artifact_small_helpers() -> None:
    assert _dealiased_kx_count(1) == 1
    assert np.array_equal(_dealiased_kx_indices(1), np.array([0], dtype=np.int32))

    class _Group:
        def __init__(self):
            self.created = {}

        def createVariable(self, name, _dtype, _dims):
            class _Var:
                def __init__(self, store, key):
                    self._store = store
                    self._key = key

                def __setitem__(self, _idx, value):
                    self._store[self._key] = np.asarray(value)

            return _Var(self.created, name)

    group = _Group()
    _maybe_var(group, "foo", "f4", ("x",), np.array([1.0, 2.0], dtype=np.float32))
    assert np.allclose(group.created["foo"], np.array([1.0, 2.0], dtype=np.float32))


def test_runtime_artifact_axis_and_condense_helpers() -> None:
    axis = _real_space_axis(4, 2.0)
    np.testing.assert_allclose(axis, np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32))

    ky = np.array([0.0, 0.2, 0.4, -0.4, -0.2], dtype=np.float32)
    kx = np.array([0.0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1], dtype=np.float32)
    assert _dealiased_ky_count(ky.size) == 2
    np.testing.assert_array_equal(
        _dealiased_ky_indices(ky.size), np.array([0, 1], dtype=np.int32)
    )
    np.testing.assert_allclose(
        _dealiased_ky_values(ky), np.array([0.0, 0.2], dtype=np.float32)
    )
    np.testing.assert_allclose(
        _dealiased_kx_values(kx),
        np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float32),
    )

    arr_kx = np.arange(7, dtype=np.float32)
    np.testing.assert_allclose(
        _condense_kx(arr_kx), np.array([5, 6, 0, 1, 2], dtype=np.float32)
    )

    arr_ky = np.arange(10, dtype=np.float32).reshape(2, 5)
    np.testing.assert_allclose(_condense_ky(arr_ky), arr_ky[..., :2])

    arr_kykx = np.arange(35, dtype=np.float32).reshape(5, 7)
    np.testing.assert_allclose(
        _condense_kykx(arr_kykx), arr_kykx[:2][:, [5, 6, 0, 1, 2]]
    )

    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.testing.assert_allclose(
        _take_axis(arr, np.array([2, 0]), axis=1), arr[:, [2, 0], :]
    )


def test_runtime_artifact_root_metadata_and_active_field() -> None:
    class _Var:
        def __init__(self):
            self.values = None
            self.attrs = {}

        def __setitem__(self, _idx, value):
            self.values = np.asarray(value)

        def setncattr(self, key, value):
            self.attrs[key] = value

    class _Root:
        def __init__(self):
            self.vars = {}

        def createVariable(self, name, _dtype, _dims=()):
            var = _Var()
            self.vars[name] = var
            return var

    cfg = SimpleNamespace(
        grid=SimpleNamespace(Ny=5, Nx=7, Nz=8, ntheta=None, nperiod=None)
    )
    root = _Root()
    _write_runtime_root_metadata(root, cfg, nspecies=2, nl=3, nm=4)
    assert int(root.vars["ny"].values) == 5
    assert int(root.vars["nx"].values) == 7
    assert int(root.vars["ntheta"].values) == 8
    assert int(root.vars["nperiod"].values) == 1
    assert root.vars["code_info"].attrs["value"] == "gkx"

    field = np.arange(5 * 7 * 2, dtype=np.float32).reshape(5, 7, 2)
    active = _dealiased_spectral_field(field, ky_axis=0, kx_axis=1)
    np.testing.assert_allclose(active, field[:2][:, [5, 6, 0, 1, 2], :])


def test_runtime_artifact_geometry_and_input_group_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Var:
        def __init__(self, store, key):
            self._store = store
            self._key = key

        def __setitem__(self, _idx, value):
            self._store[self._key] = np.asarray(value)

    class _Group:
        def __init__(self):
            self.values = {}

        def createVariable(self, name, _dtype, _dims=()):
            return _Var(self.values, name)

    grid = SimpleNamespace(
        z=np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
        kx=np.asarray([-0.2, 0.0, 0.2], dtype=np.float32),
        ky=np.asarray([0.0, 0.3, -0.3], dtype=np.float32),
    )
    geom = SimpleNamespace(
        bmag_profile=np.asarray([1.0, 1.1, 1.2], dtype=np.float32),
        bgrad_profile=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        gb_profile=np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
        gb0_profile=np.asarray([0.7, 0.8, 0.9], dtype=np.float32),
        cv_profile=np.asarray([1.0, 1.1, 1.2], dtype=np.float32),
        cv0_profile=np.asarray([1.3, 1.4, 1.5], dtype=np.float32),
        gds2_profile=np.asarray([1.6, 1.7, 1.8], dtype=np.float32),
        gds21_profile=np.asarray([1.9, 2.0, 2.1], dtype=np.float32),
        gds22_profile=np.asarray([2.2, 2.3, 2.4], dtype=np.float32),
        grho_profile=np.asarray([0.9, 1.0, 1.1], dtype=np.float32),
        jacobian_profile=np.asarray([1.0, 1.5, 2.0], dtype=np.float32),
        gradpar_value=0.75,
        q=1.4,
        s_hat=0.6,
        R0=3.0,
        epsilon=0.2,
        kxfac=1.1,
        theta_scale=1.0,
        nfp=5,
        alpha=0.3,
        B0=2.5,
    )
    cfg = SimpleNamespace(
        grid=SimpleNamespace(nperiod=2),
        geometry=SimpleNamespace(
            model="miller", shift=0.15, kappa=1.7, akappri=0.05, tri=0.2, tripri=0.03
        ),
        physics=SimpleNamespace(beta=0.02),
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.apply_geometry_grid_defaults",
        lambda _geom, grid_cfg: grid_cfg,
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.build_spectral_grid", lambda _cfg: grid
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.build_runtime_geometry",
        lambda _cfg: object(),
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.ensure_flux_tube_geometry_data",
        lambda _geom, _theta: geom,
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.real_fft_ordered_kx",
        lambda arr: np.asarray([-0.2, 0.0, 0.2], dtype=np.float32),
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.real_fft_unique_ky",
        lambda arr: np.asarray([0.0, 0.3], dtype=np.float32),
    )

    geom_group = _Group()
    theta, kx_vals, ky_vals, geom_out = _write_geometry_group(geom_group, cfg)
    np.testing.assert_allclose(theta, grid.z)
    np.testing.assert_allclose(kx_vals, np.asarray([-0.2, 0.0, 0.2], dtype=np.float32))
    np.testing.assert_allclose(ky_vals, np.asarray([0.0, 0.3], dtype=np.float32))
    assert geom_out is geom
    np.testing.assert_allclose(geom_group.values["bmag"], geom.bmag_profile)
    assert float(geom_group.values["gradpar"]) == pytest.approx(0.75)
    assert int(geom_group.values["nfp"]) == 5

    inputs_group = _Group()
    _write_input_parameters_group(inputs_group, cfg, geom)
    assert int(inputs_group.values["igeo"]) == 0
    assert int(inputs_group.values["slab"]) == 0
    assert float(inputs_group.values["kxfac"]) == pytest.approx(1.1)
    assert float(inputs_group.values["beta"]) == pytest.approx(0.02)
    assert int(inputs_group.values["zero_shat"]) == 0
    assert float(inputs_group.values["grhoavg"]) == pytest.approx(
        np.mean(geom.grho_profile)
    )


def test_runtime_artifact_geometry_writer_applies_imported_grid_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Var:
        def __init__(self, store, key):
            self._store = store
            self._key = key

        def __setitem__(self, _idx, value):
            self._store[self._key] = np.asarray(value)

    class _Group:
        def __init__(self):
            self.values = {}

        def createVariable(self, name, _dtype, _dims=()):
            return _Var(self.values, name)

    theta_closed = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    geom = FluxTubeGeometryData(
        theta=theta_closed,
        gradpar_value=0.5,
        bmag_profile=np.linspace(1.0, 1.4, theta_closed.size),
        bgrad_profile=np.linspace(0.1, 0.5, theta_closed.size),
        gds2_profile=np.linspace(2.0, 2.4, theta_closed.size),
        gds21_profile=np.linspace(-0.2, 0.2, theta_closed.size),
        gds22_profile=np.linspace(0.7, 0.9, theta_closed.size),
        cv_profile=np.linspace(0.3, 0.7, theta_closed.size),
        gb_profile=np.linspace(0.4, 0.8, theta_closed.size),
        cv0_profile=np.linspace(-0.1, 0.1, theta_closed.size),
        gb0_profile=np.linspace(-0.2, 0.2, theta_closed.size),
        jacobian_profile=np.linspace(3.0, 3.4, theta_closed.size),
        grho_profile=np.linspace(1.0, 1.2, theta_closed.size),
        q=1.6,
        s_hat=0.0,
        epsilon=0.12,
        R0=5.5,
        kxfac=1.25,
        nfp=5,
        theta_closed_interval=True,
        source_model="imported-netcdf",
    )
    cfg = SimpleNamespace(
        grid=GridConfig(Nx=4, Ny=4, Nz=17, Lx=6.28, Ly=6.28, boundary="periodic"),
        geometry=SimpleNamespace(model="vmec", shift=0.0),
        physics=SimpleNamespace(beta=0.0),
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.build_runtime_geometry",
        lambda _cfg: geom,
    )

    group = _Group()
    theta, _kx, _ky, geom_out = _write_geometry_group(group, cfg)

    assert theta.shape == (theta_closed.size - 1,)
    np.testing.assert_allclose(theta, theta_closed[:-1].astype(np.float32))
    np.testing.assert_allclose(
        group.values["bmag"], np.asarray(geom.bmag_profile[:-1], dtype=np.float32)
    )
    assert geom_out.theta_closed_interval is False
    assert float(group.values["kxfac"]) == pytest.approx(1.25)


def test_runtime_artifact_particle_moments(monkeypatch) -> None:
    state = np.ones((1, 2, 3, 2, 4, 3), dtype=np.complex64)
    cfg = SimpleNamespace(grid=SimpleNamespace())
    grid = SimpleNamespace(z=np.asarray([-1.0, 0.0, 1.0], dtype=np.float32))
    geom = object()
    cache = SimpleNamespace(
        Jl=np.ones((1, 2, 2, 4, 3), dtype=np.float32),
        JlB=np.ones((1, 2, 2, 4, 3), dtype=np.float32),
        kperp2=np.ones((2, 4, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.apply_geometry_grid_defaults",
        lambda _geom, grid_cfg: grid_cfg,
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.build_spectral_grid", lambda _grid: grid
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.build_runtime_geometry",
        lambda _cfg: geom,
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.ensure_flux_tube_geometry_data",
        lambda _geom, _theta: geom,
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.build_runtime_linear_params",
        lambda _cfg, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "gkx.artifacts.nonlinear_netcdf.build_linear_cache",
        lambda *_args, **_kwargs: cache,
    )

    moments = _particle_moments(state, cfg)
    np.testing.assert_allclose(moments["ParticleDensity"], np.full((1, 2, 4, 3), 2.0))
    np.testing.assert_allclose(moments["ParticleUpar"], np.full((1, 2, 4, 3), 2.0))
    np.testing.assert_allclose(moments["ParticleUperp"], np.full((1, 2, 4, 3), 2.0))
    np.testing.assert_allclose(
        moments["ParticleTemp"], np.full((1, 2, 4, 3), 2.0 * np.sqrt(2.0))
    )


def test_write_runtime_nonlinear_artifacts_preserves_csv_target(tmp_path: Path) -> None:
    diag = SimulationDiagnostics(
        t=np.asarray([0.1, 0.2]),
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.01, 0.02]),
        omega_t=np.asarray([0.03, 0.04]),
        Wg_t=np.asarray([1.0, 1.1]),
        Wphi_t=np.asarray([2.0, 2.1]),
        Wapar_t=np.asarray([0.5, 0.6]),
        heat_flux_t=np.asarray([3.0, 3.1]),
        particle_flux_t=np.asarray([4.0, 4.1]),
        energy_t=np.asarray([3.5, 3.8]),
        heat_flux_species_t=np.asarray([[3.0], [3.1]]),
        particle_flux_species_t=np.asarray([[4.0], [4.1]]),
        phi_mode_t=None,
    )
    result = RuntimeNonlinearResult(
        t=np.asarray([0.1, 0.2]),
        diagnostics=diag,
        state=np.zeros((2, 2), dtype=np.complex64),
        ky_selected=0.2,
        kx_selected=0.0,
    )

    csv_path = tmp_path / "diag.csv"
    paths = write_runtime_nonlinear_artifacts(csv_path, result)

    assert paths["diagnostics"] == str(csv_path)
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "heat_flux_s0" in header
    assert "particle_flux_s0" in header
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["kind"] == "nonlinear"
    assert summary["n_samples"] == 2
    assert Path(paths["state"]).exists()


def test_write_runtime_nonlinear_artifacts_handles_scalar_result_and_1d_species(
    tmp_path: Path,
) -> None:
    diag = SimulationDiagnostics(
        t=np.asarray([0.1, 0.2]),
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.01, 0.02]),
        omega_t=np.asarray([0.03, 0.04]),
        Wg_t=np.asarray([1.0, 1.1]),
        Wphi_t=np.asarray([2.0, 2.1]),
        Wapar_t=np.asarray([0.5, 0.6]),
        heat_flux_t=np.asarray([3.0, 3.1]),
        particle_flux_t=np.asarray([4.0, 4.1]),
        energy_t=np.asarray([3.5, 3.8]),
        heat_flux_species_t=np.asarray([3.0, 3.1]),
        particle_flux_species_t=np.asarray([4.0, 4.1]),
        turbulent_heating_t=np.asarray([5.0, 5.1]),
        turbulent_heating_species_t=np.asarray([5.0, 5.1]),
        phi_mode_t=None,
    )
    result = RuntimeNonlinearResult(
        t=np.asarray([0.1, 0.2]),
        diagnostics=diag,
        state=np.zeros((2, 2), dtype=np.complex64),
        ky_selected=0.2,
        kx_selected=0.0,
        phi2=7.0,
    )

    paths = write_runtime_nonlinear_artifacts(tmp_path / "diag1d", result)
    header = Path(paths["diagnostics"]).read_text(encoding="utf-8").splitlines()[0]
    assert "heat_flux_s0" in header
    assert "particle_flux_s0" in header
    assert "turbulent_heating_s0" in header

    result_no_diag = RuntimeNonlinearResult(
        t=np.asarray([0.1]),
        diagnostics=None,
        phi2=7.0,
        ky_selected=0.2,
        kx_selected=0.0,
    )
    paths_no_diag = write_runtime_nonlinear_artifacts(
        tmp_path / "scalar_only", result_no_diag
    )
    summary = json.loads(Path(paths_no_diag["summary"]).read_text(encoding="utf-8"))
    assert summary["phi2_last"] == 7.0


def test_write_runtime_nonlinear_artifacts_writes_nonlinear_netcdf_bundle(
    tmp_path: Path,
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    diag = SimulationDiagnostics(
        t=np.asarray([0.0, 0.1], dtype=float),
        dt_t=np.asarray([0.05, 0.05], dtype=float),
        dt_mean=np.asarray(0.05),
        gamma_t=np.asarray([0.0, 0.0], dtype=float),
        omega_t=np.asarray([0.0, 0.0], dtype=float),
        Wg_t=np.asarray([1.0, 1.1], dtype=float),
        Wphi_t=np.asarray([2.0, 2.1], dtype=float),
        Wapar_t=np.asarray([0.0, 0.0], dtype=float),
        heat_flux_t=np.asarray([3.0, 3.1], dtype=float),
        particle_flux_t=np.asarray([4.0, 4.1], dtype=float),
        energy_t=np.asarray([3.0, 3.2], dtype=float),
        heat_flux_species_t=np.asarray([[3.0], [3.1]], dtype=float),
        particle_flux_species_t=np.asarray([[4.0], [4.1]], dtype=float),
        turbulent_heating_t=np.asarray([8.0, 8.1], dtype=float),
        turbulent_heating_species_t=np.asarray([[8.0], [8.1]], dtype=float),
        phi_mode_t=None,
        resolved=ResolvedDiagnostics(
            Phi2_kxt=np.ones((2, 8), dtype=float),
            Phi2_kyt=np.ones((2, 8), dtype=float),
            Phi2_kxkyt=np.ones((2, 8, 8), dtype=float),
            Phi2_zt=np.ones((2, 6), dtype=float),
            Phi2_zonal_t=np.ones((2,), dtype=float),
            Phi2_zonal_kxt=np.ones((2, 8), dtype=float),
            Phi2_zonal_zt=np.ones((2, 6), dtype=float),
            Wg_kxst=np.ones((2, 1, 8), dtype=float),
            Wg_kyst=np.ones((2, 1, 8), dtype=float),
            Wg_kxkyst=np.ones((2, 1, 8, 8), dtype=float),
            Wg_zst=np.ones((2, 1, 6), dtype=float),
            Wg_lmst=np.ones((2, 1, 8, 4), dtype=float),
            Wphi_kxst=np.ones((2, 1, 8), dtype=float),
            Wphi_kyst=np.ones((2, 1, 8), dtype=float),
            Wphi_kxkyst=np.ones((2, 1, 8, 8), dtype=float),
            Wphi_zst=np.ones((2, 1, 6), dtype=float),
            Wapar_kxst=np.zeros((2, 1, 8), dtype=float),
            Wapar_kyst=np.zeros((2, 1, 8), dtype=float),
            Wapar_kxkyst=np.zeros((2, 1, 8, 8), dtype=float),
            Wapar_zst=np.zeros((2, 1, 6), dtype=float),
            HeatFlux_kxst=np.ones((2, 1, 8), dtype=float),
            HeatFlux_kyst=np.ones((2, 1, 8), dtype=float),
            HeatFlux_kxkyst=np.ones((2, 1, 8, 8), dtype=float),
            HeatFlux_zst=np.ones((2, 1, 6), dtype=float),
            HeatFluxES_kxst=np.full((2, 1, 8), 2.0, dtype=float),
            HeatFluxES_kyst=np.full((2, 1, 8), 2.0, dtype=float),
            HeatFluxES_kxkyst=np.full((2, 1, 8, 8), 2.0, dtype=float),
            HeatFluxES_zst=np.full((2, 1, 6), 2.0, dtype=float),
            HeatFluxApar_kxst=np.full((2, 1, 8), 3.0, dtype=float),
            HeatFluxApar_kyst=np.full((2, 1, 8), 3.0, dtype=float),
            HeatFluxApar_kxkyst=np.full((2, 1, 8, 8), 3.0, dtype=float),
            HeatFluxApar_zst=np.full((2, 1, 6), 3.0, dtype=float),
            HeatFluxBpar_kxst=np.full((2, 1, 8), 4.0, dtype=float),
            HeatFluxBpar_kyst=np.full((2, 1, 8), 4.0, dtype=float),
            HeatFluxBpar_kxkyst=np.full((2, 1, 8, 8), 4.0, dtype=float),
            HeatFluxBpar_zst=np.full((2, 1, 6), 4.0, dtype=float),
            ParticleFlux_kxst=np.ones((2, 1, 8), dtype=float),
            ParticleFlux_kyst=np.ones((2, 1, 8), dtype=float),
            ParticleFlux_kxkyst=np.ones((2, 1, 8, 8), dtype=float),
            ParticleFlux_zst=np.ones((2, 1, 6), dtype=float),
            ParticleFluxES_kxst=np.full((2, 1, 8), 5.0, dtype=float),
            ParticleFluxES_kyst=np.full((2, 1, 8), 5.0, dtype=float),
            ParticleFluxES_kxkyst=np.full((2, 1, 8, 8), 5.0, dtype=float),
            ParticleFluxES_zst=np.full((2, 1, 6), 5.0, dtype=float),
            ParticleFluxApar_kxst=np.full((2, 1, 8), 6.0, dtype=float),
            ParticleFluxApar_kyst=np.full((2, 1, 8), 6.0, dtype=float),
            ParticleFluxApar_kxkyst=np.full((2, 1, 8, 8), 6.0, dtype=float),
            ParticleFluxApar_zst=np.full((2, 1, 6), 6.0, dtype=float),
            ParticleFluxBpar_kxst=np.full((2, 1, 8), 7.0, dtype=float),
            ParticleFluxBpar_kyst=np.full((2, 1, 8), 7.0, dtype=float),
            ParticleFluxBpar_kxkyst=np.full((2, 1, 8, 8), 7.0, dtype=float),
            ParticleFluxBpar_zst=np.full((2, 1, 6), 7.0, dtype=float),
            TurbulentHeating_kxst=np.full((2, 1, 8), 8.0, dtype=float),
            TurbulentHeating_kyst=np.full((2, 1, 8), 8.0, dtype=float),
            TurbulentHeating_kxkyst=np.full((2, 1, 8, 8), 8.0, dtype=float),
            TurbulentHeating_zst=np.full((2, 1, 6), 8.0, dtype=float),
        ),
    )
    state = np.zeros((1, 4, 8, 8, 8, 6), dtype=np.complex64)
    fields = type(
        "Fields",
        (),
        {"phi": np.zeros((8, 8, 6), dtype=np.complex64), "apar": None, "bpar": None},
    )()
    result = RuntimeNonlinearResult(
        t=np.asarray([0.0, 0.1]),
        diagnostics=diag,
        fields=fields,
        state=state,
        ky_selected=0.2,
        kx_selected=0.0,
    )

    cfg = RuntimeConfig(
        grid=GridConfig(Nx=8, Ny=8, Nz=6, Lx=1.0, Ly=1.0),
        time=TimeConfig(compressed_real_fft=True),
    )

    paths = write_runtime_nonlinear_artifacts(tmp_path / "probe.out.nc", result, cfg)

    assert Path(paths["out"]).exists()
    assert Path(paths["restart"]).exists()
    assert Path(paths["big"]).exists()

    with Dataset(paths["out"], "r") as root:
        assert set(root.groups) == {"Diagnostics", "Geometry", "Grids", "Inputs"}
        assert int(root.variables["ny"][()]) == 8
        assert int(root.variables["nx"][()]) == 8
        assert int(root.variables["ntheta"][()]) == 6
        assert int(root.variables["nhermite"][()]) == 8
        assert int(root.variables["nlaguerre"][()]) == 4
        assert int(root.variables["nspecies"][()]) == 1
        assert root.variables["code_info"].getncattr("value") == "gkx"
        assert root.dimensions["kx"].size == 5
        assert root.dimensions["ky"].size == 3
        assert "Phi2_t" in root.groups["Diagnostics"].variables
        assert "Phi2_kxt" in root.groups["Diagnostics"].variables
        np.testing.assert_allclose(
            root.groups["Diagnostics"].variables["Phi2_t"][:], np.full(2, 15.0)
        )
        np.testing.assert_allclose(
            root.groups["Diagnostics"].variables["Phi2_kxt"][:],
            np.full((2, 5), 3.0),
        )
        np.testing.assert_allclose(
            root.groups["Diagnostics"].variables["Phi2_kyt"][:],
            np.full((2, 3), 5.0),
        )
        np.testing.assert_allclose(
            root.groups["Diagnostics"].variables["Phi2_kxkyt"][:],
            np.ones((2, 3, 5)),
        )
        assert "Wg_st" in root.groups["Diagnostics"].variables
        assert "Wg_kyst" in root.groups["Diagnostics"].variables
        assert "Wg_lmst" in root.groups["Diagnostics"].variables
        assert "HeatFlux_st" in root.groups["Diagnostics"].variables
        np.testing.assert_allclose(
            root.groups["Diagnostics"].variables["HeatFluxES_st"][:],
            np.full((2, 1), 16.0),
        )
        np.testing.assert_allclose(
            root.groups["Diagnostics"].variables["HeatFluxApar_st"][:],
            np.full((2, 1), 24.0),
        )
        np.testing.assert_allclose(
            root.groups["Diagnostics"].variables["HeatFluxBpar_st"][:],
            np.full((2, 1), 32.0),
        )
        np.testing.assert_allclose(
            root.groups["Diagnostics"].variables["TurbulentHeating_st"][:],
            np.full((2, 1), 64.0),
        )
        assert "ParticleFluxBpar_kxkyst" in root.groups["Diagnostics"].variables
        assert "TurbulentHeating_kxkyst" in root.groups["Diagnostics"].variables

    with Dataset(paths["restart"], "r") as root:
        assert root.dimensions["Nkx"].size == 5
        assert root.dimensions["Nky"].size == 3
        assert root.variables["G"].shape[-1] == 2
        assert "time" in root.variables

    with Dataset(paths["big"], "r") as root:
        assert root.variables["code_info"].getncattr("value") == "gkx"
        assert "Phi" in root.groups["Diagnostics"].variables
        assert "PhiXY" in root.groups["Diagnostics"].variables
        assert "Density" in root.groups["Diagnostics"].variables
        assert "Upar" in root.groups["Diagnostics"].variables
        assert "Tpar" in root.groups["Diagnostics"].variables
        assert "Tperp" in root.groups["Diagnostics"].variables
        assert "ParticleDensity" in root.groups["Diagnostics"].variables

    loaded = load_nonlinear_netcdf_diagnostics(paths["out"])
    np.testing.assert_allclose(
        np.asarray(loaded.Wg_t), np.asarray([1.0, 1.1], dtype=float)
    )
    np.testing.assert_allclose(
        np.asarray(loaded.turbulent_heating_t), np.asarray([64.0, 64.0], dtype=float)
    )
    assert loaded.resolved is not None
    assert loaded.resolved.HeatFluxApar_kxst is not None
    assert loaded.resolved.TurbulentHeating_kxst is not None


def test_write_nonlinear_netcdf_outputs_requires_diagnostics(
    tmp_path: Path,
) -> None:
    pytest.importorskip("netCDF4")
    cfg = RuntimeConfig(
        grid=GridConfig(Nx=4, Ny=4, Nz=4, Lx=1.0, Ly=1.0),
        time=TimeConfig(compressed_real_fft=True),
    )
    result = RuntimeNonlinearResult(
        t=np.asarray([0.0]),
        diagnostics=None,
        state=np.zeros((1, 1, 1, 1, 1, 1), dtype=np.complex64),
        ky_selected=0.2,
        kx_selected=0.0,
    )

    Dataset = _require_netcdf4()
    assert Dataset.__name__ == "Dataset"
    with pytest.raises(ValueError, match="require nonlinear diagnostics"):
        _write_nonlinear_netcdf_outputs(tmp_path / "probe.out.nc", result, cfg)


def test_run_runtime_nonlinear_with_artifacts_uses_restart_if_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []
    out_path = tmp_path / "resume.out.nc"
    restart_path = tmp_path / "resume.restart.nc"
    restart_path.write_bytes(b"stub")

    cfg = RuntimeConfig(
        time=TimeConfig(dt=0.1, t_max=0.2),
        output=RuntimeOutputConfig(
            path=str(out_path),
            restart_if_exists=True,
            restart_with_perturb=True,
            restart_scale=0.25,
            restart_to_file=str(restart_path),
            restart_from_file=str(restart_path),
            nsave=1,
        ),
    )

    diag = SimulationDiagnostics(
        t=np.asarray([0.1]),
        dt_t=np.asarray([0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.0]),
        omega_t=np.asarray([0.0]),
        Wg_t=np.asarray([1.0]),
        Wphi_t=np.asarray([0.0]),
        Wapar_t=np.asarray([0.0]),
        heat_flux_t=np.asarray([0.0]),
        particle_flux_t=np.asarray([0.0]),
        energy_t=np.asarray([1.0]),
    )

    def _fake_run_runtime_nonlinear(run_cfg, **kwargs):
        calls.append(
            {
                "init_file": run_cfg.init.init_file,
                "init_file_mode": run_cfg.init.init_file_mode,
                "init_file_scale": run_cfg.init.init_file_scale,
                "steps": kwargs.get("steps"),
            }
        )
        return RuntimeNonlinearResult(
            t=np.asarray([0.1]),
            diagnostics=diag,
            state=np.zeros((1, 1, 1, 1, 1, 1), dtype=np.complex64),
            fields=type(
                "Fields",
                (),
                {
                    "phi": np.zeros((1, 1, 1), dtype=np.complex64),
                    "apar": None,
                    "bpar": None,
                },
            )(),
            ky_selected=0.2,
            kx_selected=0.0,
        )

    def _fake_write_runtime_nonlinear_artifacts(_out, _result, _cfg):
        restart_path.write_bytes(b"stub")
        return {"out": str(out_path), "restart": str(restart_path)}

    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.run_runtime_nonlinear",
        _fake_run_runtime_nonlinear,
    )
    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.write_runtime_nonlinear_artifacts",
        _fake_write_runtime_nonlinear_artifacts,
    )

    _result, _paths = run_runtime_nonlinear_with_artifacts(
        cfg,
        out=out_path,
        ky_target=0.2,
        steps=2,
        diagnostics=True,
    )

    assert calls[0]["init_file"] == str(restart_path)
    assert calls[0]["init_file_mode"] == "add"
    assert calls[0]["init_file_scale"] == 0.25
    assert calls[1]["init_file"] == str(restart_path)
    assert calls[1]["init_file_mode"] == "replace"
    assert calls[1]["init_file_scale"] == 1.0


def test_runtime_orchestration_handoff_chunks_and_restarts(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    writes: list[float] = []
    out_path = tmp_path / "direct.out.nc"
    restart_path = tmp_path / "direct.restart.nc"
    cfg = RuntimeConfig(
        time=TimeConfig(dt=0.1, t_max=1.2, fixed_dt=True, diagnostics=True),
        output=RuntimeOutputConfig(
            path=str(out_path),
            save_for_restart=True,
            nsave=5,
            resolved_diagnostics=False,
        ),
    )

    def _diag(sample_t: float) -> SimulationDiagnostics:
        return SimulationDiagnostics(
            t=np.asarray([sample_t]),
            dt_t=np.asarray([0.1]),
            dt_mean=np.asarray(0.1),
            gamma_t=np.zeros(1),
            omega_t=np.zeros(1),
            Wg_t=np.ones(1),
            Wphi_t=np.ones(1),
            Wapar_t=np.zeros(1),
            heat_flux_t=np.zeros(1),
            particle_flux_t=np.zeros(1),
            energy_t=np.ones(1),
        )

    def _run(run_cfg, **kwargs):
        chunk_steps = int(kwargs["steps"])
        calls.append(
            {
                "init_file": run_cfg.init.init_file,
                "init_file_mode": run_cfg.init.init_file_mode,
                "steps": chunk_steps,
                "return_state": kwargs["return_state"],
                "resolved_diagnostics": kwargs["resolved_diagnostics"],
            }
        )
        return RuntimeNonlinearResult(
            t=np.asarray([0.1 * chunk_steps]),
            diagnostics=_diag(0.1 * chunk_steps),
            state=np.zeros((1, 1, 1, 1, 1, 1), dtype=np.complex64),
        )

    def _write(_out, result, _cfg):
        assert result.diagnostics is not None
        writes.append(float(np.asarray(result.diagnostics.t)[-1]))
        restart_path.write_bytes(b"restart")
        return {"out": str(out_path), "restart": str(restart_path)}

    deps = SimpleNamespace(
        is_netcdf_output_target=lambda path: Path(path).suffix == ".nc",
        resolve_restart_path=lambda _path, _cfg: restart_path,
        resolve_restart_write_path=lambda _path, _cfg: restart_path,
        netcdf_bundle_base=lambda path: Path(path).with_suffix("").with_suffix(""),
        load_nonlinear_netcdf_diagnostics=lambda _path: _diag(0.0),
        condense_diagnostics_for_netcdf_output=lambda diag: diag,
        concat_runtime_diagnostics=lambda diags: concat_runtime_diagnostics(diags),
        validate_finite_runtime_result=lambda _result: None,
        run_runtime_nonlinear=_run,
        write_runtime_nonlinear_artifacts=_write,
    )

    result, paths = run_runtime_nonlinear_artifact_handoff(
        cfg,
        out=out_path,
        ky_target=0.2,
        steps=12,
        diagnostics=True,
        deps=deps,
    )

    assert result.diagnostics is not None
    assert paths["restart"] == str(restart_path)
    assert [call["steps"] for call in calls] == [5, 5, 2]
    assert calls[0]["init_file"] is None
    assert calls[1]["init_file"] == str(restart_path)
    assert calls[1]["init_file_mode"] == "replace"
    assert all(call["return_state"] is True for call in calls)
    assert all(call["resolved_diagnostics"] is False for call in calls)
    assert writes == pytest.approx([0.5, 1.0, 1.2])
    assert float(np.asarray(result.diagnostics.t)[-1]) == pytest.approx(1.2)


def test_run_runtime_nonlinear_with_artifacts_keeps_adaptive_steps_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_path = tmp_path / "adaptive.out.nc"
    cfg = RuntimeConfig(
        time=TimeConfig(dt=0.1, t_max=0.3, fixed_dt=False, diagnostics=True),
        output=RuntimeOutputConfig(
            path=str(out_path), save_for_restart=True, nsave=10000
        ),
    )
    diag = SimulationDiagnostics(
        t=np.asarray([0.1, 0.2, 0.3]),
        dt_t=np.asarray([0.1, 0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.zeros(3),
        omega_t=np.zeros(3),
        Wg_t=np.ones(3),
        Wphi_t=np.ones(3),
        Wapar_t=np.zeros(3),
        heat_flux_t=np.zeros(3),
        particle_flux_t=np.zeros(3),
        energy_t=2.0 * np.ones(3),
    )
    captured_steps: list[int | None] = []

    def _fake_run_runtime_nonlinear(_cfg, **kwargs):
        captured_steps.append(kwargs.get("steps"))
        return RuntimeNonlinearResult(
            t=np.asarray(diag.t),
            diagnostics=diag,
            state=np.zeros((1, 1, 1, 1, 1, 1), dtype=np.complex64),
        )

    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.run_runtime_nonlinear",
        _fake_run_runtime_nonlinear,
    )
    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.write_runtime_nonlinear_artifacts",
        lambda *_args, **_kwargs: {"out": str(out_path)},
    )

    run_runtime_nonlinear_with_artifacts(
        cfg, out=out_path, ky_target=0.2, diagnostics=True
    )

    assert captured_steps == [None]


def test_run_runtime_nonlinear_with_artifacts_forwards_live_output_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = RuntimeConfig(
        time=TimeConfig(dt=0.1, t_max=0.1, fixed_dt=True, diagnostics=True),
        output=RuntimeOutputConfig(
            path=str(tmp_path / "live.out.nc"), save_for_restart=False
        ),
    )
    diag = SimulationDiagnostics(
        t=np.asarray([0.1]),
        dt_t=np.asarray([0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.0]),
        omega_t=np.asarray([0.0]),
        Wg_t=np.asarray([1.0]),
        Wphi_t=np.asarray([2.0]),
        Wapar_t=np.asarray([0.0]),
        heat_flux_t=np.asarray([3.0]),
        particle_flux_t=np.asarray([4.0]),
        energy_t=np.asarray([3.0]),
    )
    messages: list[str] = []
    captured: dict[str, object] = {}

    def _fake_run_runtime_nonlinear(_cfg, **kwargs):
        captured["show_progress"] = kwargs["show_progress"]
        captured["status_callback"] = kwargs["status_callback"]
        callback = kwargs["status_callback"]
        assert callback is not None
        callback("live status propagated")
        return RuntimeNonlinearResult(
            t=np.asarray(diag.t),
            diagnostics=diag,
            state=None,
            ky_selected=0.2,
            kx_selected=0.0,
        )

    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.run_runtime_nonlinear",
        _fake_run_runtime_nonlinear,
    )

    result, paths = run_runtime_nonlinear_with_artifacts(
        cfg,
        out=None,
        ky_target=0.2,
        show_progress=True,
        status_callback=messages.append,
    )

    assert result.diagnostics is diag
    assert paths == {}
    assert captured["show_progress"] is True
    assert callable(captured["status_callback"])
    assert messages == ["live status propagated"]


def test_run_runtime_nonlinear_with_artifacts_rejects_nonfinite_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_path = tmp_path / "bad.out.nc"
    cfg = RuntimeConfig(
        time=TimeConfig(dt=0.1, t_max=0.2, fixed_dt=True, diagnostics=True),
        output=RuntimeOutputConfig(path=str(out_path), save_for_restart=True, nsave=1),
    )
    bad_diag = SimulationDiagnostics(
        t=np.asarray([0.1]),
        dt_t=np.asarray([0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.zeros(1),
        omega_t=np.zeros(1),
        Wg_t=np.asarray([np.nan]),
        Wphi_t=np.ones(1),
        Wapar_t=np.zeros(1),
        heat_flux_t=np.zeros(1),
        particle_flux_t=np.zeros(1),
        energy_t=np.asarray([np.nan]),
    )
    calls = {"run": 0, "write": 0}

    def _fake_run_runtime_nonlinear(_cfg, **_kwargs):
        calls["run"] += 1
        return RuntimeNonlinearResult(
            t=np.asarray(bad_diag.t),
            diagnostics=bad_diag,
            state=np.zeros((1, 1, 1, 1, 1, 1), dtype=np.complex64),
        )

    def _fake_write(*_args, **_kwargs):
        calls["write"] += 1
        return {"out": str(out_path)}

    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.run_runtime_nonlinear",
        _fake_run_runtime_nonlinear,
    )
    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.write_runtime_nonlinear_artifacts",
        _fake_write,
    )

    with pytest.raises(
        RuntimeError, match=r"non-finite diagnostics in Wg_t at sample 0"
    ):
        run_runtime_nonlinear_with_artifacts(
            cfg, out=out_path, ky_target=0.2, steps=2, diagnostics=True
        )

    assert calls == {"run": 1, "write": 0}


def test_write_runtime_nonlinear_artifacts_requires_cfg_and_diagnostics_for_netcdf_output_target(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        write_runtime_nonlinear_artifacts(
            tmp_path / "case.out.nc",
            RuntimeNonlinearResult(t=np.asarray([]), diagnostics=None),
            cfg=None,
        )

    cfg = RuntimeConfig()
    with pytest.raises(ValueError):
        write_runtime_nonlinear_artifacts(
            tmp_path / "case.out.nc",
            RuntimeNonlinearResult(
                t=np.asarray([]),
                diagnostics=None,
                state=np.zeros((1, 1), dtype=np.complex64),
            ),
            cfg=cfg,
        )


def test_load_nonlinear_netcdf_diagnostics_fills_missing_turbulent_heating(
    tmp_path: Path,
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset
    out_nc = tmp_path / "case.out.nc"
    with Dataset(out_nc, "w") as root:
        root.createDimension("time", 2)
        root.createDimension("s", 1)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = np.array([0.0, 0.1])
        diag = root.createGroup("Diagnostics")
        for name, values in {
            "Wg_st": np.array([[1.0], [1.1]], dtype=np.float32),
            "Wphi_st": np.array([[2.0], [2.1]], dtype=np.float32),
            "Wapar_st": np.array([[0.0], [0.0]], dtype=np.float32),
            "HeatFlux_st": np.array([[3.0], [3.1]], dtype=np.float32),
            "ParticleFlux_st": np.array([[4.0], [4.1]], dtype=np.float32),
        }.items():
            diag.createVariable(name, "f4", ("time", "s"))[:] = values
    loaded = load_nonlinear_netcdf_diagnostics(out_nc)
    assert np.allclose(loaded.turbulent_heating_t, np.zeros(2, dtype=np.float32))


def test_run_runtime_nonlinear_with_artifacts_append_preserves_loaded_netcdf_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset
    out = tmp_path / "append.out.nc"
    restart = tmp_path / "append.restart.nc"
    restart.write_bytes(b"restart")
    cfg = RuntimeConfig(
        grid=GridConfig(Nx=6, Ny=4, Nz=4, Lx=1.0, Ly=1.0, boundary="periodic"),
        time=TimeConfig(dt=0.1, t_max=0.2, diagnostics=True, fixed_dt=True),
        output=RuntimeOutputConfig(
            path=str(out),
            restart_if_exists=True,
            append_on_restart=True,
            save_for_restart=False,
        ),
    )
    full_ny = int(cfg.grid.Ny)
    full_nx = int(cfg.grid.Nx)
    ky_idx = _dealiased_ky_indices(full_ny)
    kx_idx = _dealiased_kx_indices(full_nx)

    def _diag(sample_t: float, base: float) -> SimulationDiagnostics:
        phi2_kxkyt = base + np.arange(full_ny * full_nx, dtype=np.float32).reshape(
            1, full_ny, full_nx
        )
        phi2_kxt = np.sum(phi2_kxkyt[:, ky_idx, :], axis=1)
        phi2_kyt = np.sum(phi2_kxkyt[:, :, kx_idx], axis=2)
        wg_kxst = base + np.arange(full_nx, dtype=np.float32).reshape(1, 1, full_nx)
        wg_kyst = base + np.arange(full_ny, dtype=np.float32).reshape(1, 1, full_ny)
        wg_kxkyst = base + np.arange(full_ny * full_nx, dtype=np.float32).reshape(
            1, 1, full_ny, full_nx
        )
        return SimulationDiagnostics(
            t=np.asarray([sample_t]),
            dt_t=np.asarray([0.1]),
            dt_mean=np.asarray(0.1),
            gamma_t=np.asarray([0.0]),
            omega_t=np.asarray([0.0]),
            Wg_t=np.asarray([base + 1.0]),
            Wphi_t=np.asarray([base + 2.0]),
            Wapar_t=np.asarray([0.0]),
            heat_flux_t=np.asarray([base + 3.0]),
            particle_flux_t=np.asarray([base + 4.0]),
            energy_t=np.asarray([base + 3.0]),
            heat_flux_species_t=np.asarray([[base + 3.0]]),
            particle_flux_species_t=np.asarray([[base + 4.0]]),
            turbulent_heating_t=np.asarray([base + 5.0]),
            turbulent_heating_species_t=np.asarray([[base + 5.0]]),
            phi_mode_t=np.asarray([base + 1.0j * base]),
            resolved=ResolvedDiagnostics(
                Phi2_kxt=phi2_kxt,
                Phi2_kyt=phi2_kyt,
                Phi2_kxkyt=phi2_kxkyt,
                Phi_zonal_line_kxt=(
                    base + np.arange(full_nx, dtype=np.float32)[None, :]
                ).astype(np.complex64),
                Wg_kxst=wg_kxst,
                Wg_kyst=wg_kyst,
                Wg_kxkyst=wg_kxkyst,
            ),
        )

    write_runtime_nonlinear_artifacts(
        out,
        RuntimeNonlinearResult(
            t=np.asarray([0.1]),
            diagnostics=_diag(0.1, 1.0),
            state=None,
            ky_selected=0.2,
            kx_selected=0.0,
        ),
        cfg,
    )
    with Dataset(out, "r") as root:
        before_schema = {
            name: tuple(var.dimensions)
            for name, var in root.groups["Diagnostics"].variables.items()
        }
        assert root.dimensions["time"].size == 1
    assert "Phi_zonal_line_kxt" in before_schema
    assert all("phi_mode" not in name.lower() for name in before_schema)

    def _fake_run_runtime_nonlinear(run_cfg, **_kwargs):
        assert run_cfg.init.init_file == str(restart)
        return RuntimeNonlinearResult(
            t=np.asarray([0.1]),
            diagnostics=_diag(0.1, 20.0),
            state=None,
            ky_selected=0.2,
            kx_selected=0.0,
        )

    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.run_runtime_nonlinear",
        _fake_run_runtime_nonlinear,
    )

    result, paths = run_runtime_nonlinear_with_artifacts(
        cfg,
        out=out,
        ky_target=0.2,
        steps=1,
        diagnostics=True,
    )

    assert result.diagnostics is not None
    assert result.diagnostics.phi_mode_t is None
    with Dataset(paths["out"], "r") as root:
        after_schema = {
            name: tuple(var.dimensions)
            for name, var in root.groups["Diagnostics"].variables.items()
        }
        assert after_schema == before_schema
        assert root.dimensions["time"].size == 2
        np.testing.assert_allclose(
            root.groups["Grids"].variables["time"][:], np.asarray([0.1, 0.2])
        )
        assert root.groups["Diagnostics"].variables["Phi_zonal_line_kxt"].shape == (
            2,
            int(kx_idx.size),
            2,
        )


def test_run_runtime_nonlinear_with_artifacts_validation_branches(
    tmp_path: Path,
) -> None:
    cfg = RuntimeConfig(
        time=TimeConfig(dt=0.2, t_max=1.0, diagnostics=False, fixed_dt=True),
        output=RuntimeOutputConfig(path=str(tmp_path / "case.out.nc"), restart=True),
    )
    with pytest.raises(ValueError):
        run_runtime_nonlinear_with_artifacts(
            cfg, out=tmp_path / "case.out.nc", ky_target=0.2, diagnostics=False
        )

    cfg_missing_restart = RuntimeConfig(
        time=TimeConfig(dt=0.2, t_max=1.0, diagnostics=True, fixed_dt=True),
        output=RuntimeOutputConfig(path=str(tmp_path / "case.out.nc"), restart=True),
    )
    with pytest.raises(FileNotFoundError):
        run_runtime_nonlinear_with_artifacts(
            cfg_missing_restart,
            out=tmp_path / "case.out.nc",
            ky_target=0.2,
            diagnostics=True,
        )


def test_run_runtime_nonlinear_with_artifacts_history_and_restart_paths(
    monkeypatch, tmp_path: Path
) -> None:
    out = tmp_path / "case.out.nc"
    restart_path = tmp_path / "case.restart.nc"
    restart_path.write_bytes(b"restart")
    out.write_bytes(b"history")

    cfg = RuntimeConfig(
        time=TimeConfig(dt=0.2, t_max=1.0, diagnostics=True, fixed_dt=True),
        output=RuntimeOutputConfig(
            path=str(out),
            restart_if_exists=True,
            restart_with_perturb=True,
            append_on_restart=True,
            save_for_restart=True,
            nsave=1,
        ),
    )
    cumulative = SimulationDiagnostics(
        t=np.asarray([0.5]),
        dt_t=np.asarray([0.5]),
        dt_mean=np.asarray(0.5),
        gamma_t=np.asarray([0.0]),
        omega_t=np.asarray([0.0]),
        Wg_t=np.asarray([1.0]),
        Wphi_t=np.asarray([2.0]),
        Wapar_t=np.asarray([0.0]),
        heat_flux_t=np.asarray([3.0]),
        particle_flux_t=np.asarray([4.0]),
        energy_t=np.asarray([3.0]),
        resolved=ResolvedDiagnostics(Phi2_kxt=np.ones((1, 4), dtype=float)),
    )
    chunk_diag = SimulationDiagnostics(
        t=np.asarray([0.5]),
        dt_t=np.asarray([0.5]),
        dt_mean=np.asarray(0.5),
        gamma_t=np.asarray([0.0]),
        omega_t=np.asarray([0.0]),
        Wg_t=np.asarray([1.1]),
        Wphi_t=np.asarray([2.1]),
        Wapar_t=np.asarray([0.0]),
        heat_flux_t=np.asarray([3.1]),
        particle_flux_t=np.asarray([4.1]),
        energy_t=np.asarray([3.2]),
        resolved=ResolvedDiagnostics(Phi2_kxt=np.ones((1, 4), dtype=float)),
    )
    result_chunk = RuntimeNonlinearResult(
        t=np.asarray([0.5]),
        diagnostics=chunk_diag,
        state=np.zeros((1, 1, 1, 1, 1, 1), dtype=np.complex64),
    )
    captured = {"writes": 0}

    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.load_nonlinear_netcdf_diagnostics",
        lambda _path: cumulative,
    )
    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.run_runtime_nonlinear",
        lambda *_args, **_kwargs: result_chunk,
    )
    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts._concat_runtime_diagnostics",
        lambda diags: diags[-1],
    )
    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.write_runtime_nonlinear_artifacts",
        lambda *_args, **_kwargs: (
            captured.__setitem__("writes", captured["writes"] + 1) or {"out": str(out)}
        ),
    )

    result, paths = run_runtime_nonlinear_with_artifacts(
        cfg, out=out, ky_target=0.2, diagnostics=True
    )
    assert isinstance(result, RuntimeNonlinearResult)
    assert paths["out"] == str(out)
    assert captured["writes"] >= 1


def test_linear_summary_carries_fit_quality_fields(tmp_path: Path) -> None:
    """gamma/omega stderr and R^2 reach the summary, with None for eigensolves."""

    base = dict(
        ky=0.2,
        omega=-0.4,
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        t=np.asarray([0.1, 0.2, 0.3]),
        signal=np.asarray([1.0, 2.0, 4.0]),
        fit_window_tmin=0.1,
        fit_window_tmax=0.3,
        fit_signal_used="phi",
    )
    fitted = RuntimeLinearResult(
        gamma=0.3, gamma_stderr=0.004, omega_stderr=0.007, fit_r2=0.999, **base
    )
    paths = write_runtime_linear_artifacts(tmp_path / "fitted", fitted)
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["gamma_stderr"] == pytest.approx(0.004)
    assert summary["omega_stderr"] == pytest.approx(0.007)
    assert summary["fit_r2"] == pytest.approx(0.999)

    # An eigensolve has no fit window statistics, and a degenerate fit reports
    # an infinite stderr; both must serialize as JSON null rather than NaN.
    eigen = RuntimeLinearResult(gamma=0.3, gamma_stderr=float("inf"), **base)
    paths = write_runtime_linear_artifacts(tmp_path / "eigen", eigen)
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert summary["gamma_stderr"] is None
    assert summary["omega_stderr"] is None
    assert summary["fit_r2"] is None


def test_write_runtime_linear_scan_artifacts_records_warm_start(
    tmp_path: Path,
) -> None:
    """A reader with only the artifact can tell how each point was seeded."""

    warm = {
        "enabled": True,
        "visit_order": [1, 0],
        "warm_points": 1,
        "cold_points": 1,
    }
    result = SimpleNamespace(
        ky=np.asarray([0.3, 0.2]),
        gamma=np.asarray([0.2, 0.1]),
        omega=np.asarray([-0.5, -0.4]),
        quasilinear=None,
        parallel=None,
        warm_start=warm,
    )

    paths = write_runtime_linear_scan_artifacts(tmp_path / "warm_bundle", result)
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))

    assert summary["warm_start"] == warm

    cold = SimpleNamespace(
        ky=np.asarray([0.3]),
        gamma=np.asarray([0.2]),
        omega=np.asarray([-0.5]),
        quasilinear=None,
        parallel=None,
        warm_start=None,
    )
    cold_paths = write_runtime_linear_scan_artifacts(tmp_path / "cold_bundle", cold)
    cold_summary = json.loads(Path(cold_paths["summary"]).read_text(encoding="utf-8"))

    assert "warm_start" not in cold_summary
