"""Runtime output layer: the diagnostics a run computes and the artifacts they are persisted into, including NetCDF restart-state IO."""

from __future__ import annotations

from support.helpers import patch_runtime

from dataclasses import fields, replace
from gkx.artifacts.io import (
    _expand_netcdf_restart_state_full_ky,
    _expand_netcdf_restart_state_to_full_positive_ky,
    _expand_positive_ky_to_full,
    load_netcdf_restart_state,
    write_netcdf_restart_state,
)
from gkx.artifacts.io import (
    validate_finite_array,
    validate_finite_runtime_result,
)
from scripts.comparison.compare_gx_rhs_terms import (
    _build_initial_condition,
)
from gkx.config import CycloneBaseCase
from gkx.config import GridConfig, TimeConfig
from gkx.config import InitializationConfig
from gkx.core_grid import build_spectral_grid, select_ky_grid
from gkx.core_velocity import gamma0
from gkx.diagnostics import (
    ResolvedDiagnostics,
    SimulationDiagnostics,
    _cached_hermitian_mode_weight,
    _jl_family,
    _transport_mode_weight,
    _heat_flux_channel_contrib_species,
    _particle_flux_channel_contrib_species,
    _turbulent_heating_contrib_species,
    _reduce_scalar_kykxz,
    _reduce_species_kykxz,
    magnetic_vector_potential_energy,
    distribution_free_energy,
    distribution_free_energy_resolved,
    electrostatic_field_energy,
    electrostatic_field_energy_resolved,
    total_energy,
    magnetic_vector_potential_energy_resolved,
    heat_flux_total,
    heat_flux_channel_resolved_species,
    heat_flux_species,
    particle_flux_total,
    particle_flux_channel_resolved_species,
    particle_flux_species,
    phi2_resolved,
    zonal_phi_line_kxt,
    zonal_phi_mode_kxt,
    fieldline_quadrature_weights,
    turbulent_heating_resolved_species,
    turbulent_heating_species,
)
from scripts.checks._gates.validation_gates import (
    heat_flux_channel_species,
    particle_flux_channel_species,
    turbulent_heating_total,
)
from gkx.diagnostics.analysis import ModeSelection
from gkx.diagnostics.analysis import select_ky_index
from gkx.geometry import FluxTubeGeometryData
from gkx.geometry import SAlphaGeometry, sample_flux_tube_geometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from gkx.operators.linear.params import Species, build_linear_params
from gkx.runtime import (
    RuntimeLinearResult,
    RuntimeNonlinearResult,
)
from gkx.solvers_time_explicit import (
    ExplicitTimeConfig,
    _apply_completed_step_state_mask,
    _growth_rate_mode_mask,
    _linear_frequency_bound,
    _instantaneous_growth_rate_step,
    _gradient_ratio_max,
    _geometry_frequency_maxima,
    _cfl_wavenumber_arrays,
    _non_twist_shift_frequency_max,
    _diagnostic_midplane_index,
    _completed_step_state_mask,
    _linear_term_config,
    _parallel_periods_from_grid,
    integrate_linear_explicit_diagnostics,
)
from gkx.terms.assembly import assemble_rhs_cached
from gkx.terms.config import FieldState
from gkx.artifacts.spectral_layout import (
    KY_WEIGHTING_PAIR,
    KY_WEIGHTING_PER_ROW,
    _condense_kx,
    _condense_ky,
    _condense_kykx,
    _condense_kx_for_output,
    _condense_ky_for_output,
    _condense_kykx_for_output,
    _dealiased_spectral_field,
    _dealiased_ky_count,
    _dealiased_ky_indices,
    _dealiased_ky_values,
    _dealiased_kx_count,
    _dealiased_kx_indices,
    _dealiased_kx_values,
    _real_space_axis,
    _require_netcdf4,
    _restart_to_netcdf_layout,
    _spectral_species_to_ri,
    _species_matrix,
    _state_basis_moments,
    _complex_to_ri,
    _spectral_to_ri,
    _spectral_to_xy,
    _take_axis,
    _write_runtime_root_metadata,
)
from gkx.artifacts.io import (
    _ensure_parent,
    _artifact_base,
    _condense_diagnostics_for_netcdf_output,
    _condense_resolved_for_output,
    _flatten_series,
    _netcdf_bundle_base,
    _is_netcdf_output_target,
    _nonlinear_summary,
    _read_optional_var,
    _resolved_species_time,
    _resolve_restart_path,
    _write_csv,
    _write_json,
    _write_state,
    load_nonlinear_netcdf_diagnostics,
)
from gkx.artifacts.nonlinear_netcdf import (
    _particle_moments,
    _write_geometry_group,
    _write_input_parameters_group,
    _write_nonlinear_netcdf_outputs,
)
from gkx.workflows.runtime.artifacts import (
    write_runtime_linear_scan_artifacts,
    run_runtime_nonlinear_with_artifacts,
    write_runtime_linear_artifacts,
    write_runtime_nonlinear_artifacts,
)
from gkx.config import RuntimeConfig, RuntimeOutputConfig
from gkx.workflows.runtime.diagnostic_arrays import (
    validate_finite_runtime_diagnostics,
)
from gkx.workflows.runtime.artifacts import resolve_nonlinear_artifact_policy
from netCDF4 import Dataset
from pathlib import Path
from types import SimpleNamespace
import gkx.artifacts.nonlinear_netcdf as nonlinear_netcdf
import gkx.operators.moments as diagnostics_moments
import gkx.solvers_time_explicit as explicit_time_integrators
import gkx.workflows.runtime.artifacts as runtime_artifacts
import jax.numpy as jnp
import json
import numpy as np
import pytest


def _diag(n: int = 2, **overrides) -> SimulationDiagnostics:
    """Finite ``SimulationDiagnostics`` with ``n`` samples; ``overrides`` win."""

    fields_ = dict(
        t=np.linspace(0.1, 0.1 * n, n),
        dt_t=np.full(n, 0.1),
        dt_mean=np.asarray(0.1),
        gamma_t=np.zeros(n),
        omega_t=np.zeros(n),
        Wg_t=np.ones(n),
        Wphi_t=np.ones(n),
        Wapar_t=np.zeros(n),
        heat_flux_t=np.zeros(n),
        particle_flux_t=np.zeros(n),
        energy_t=np.ones(n),
    )
    fields_.update(
        {k: np.asarray(v) if isinstance(v, list) else v for k, v in overrides.items()}
    )
    return SimulationDiagnostics(**fields_)


# The two-sample trace several artifact writers are exercised on.
_TRACE = dict(
    gamma_t=[0.01, 0.02],
    omega_t=[0.03, 0.04],
    Wg_t=[1.0, 1.1],
    Wphi_t=[2.0, 2.1],
    Wapar_t=[0.5, 0.6],
    heat_flux_t=[3.0, 3.1],
    particle_flux_t=[4.0, 4.1],
    energy_t=[3.5, 3.8],
)
_STATE6 = np.zeros((1, 1, 1, 1, 1, 1), dtype=np.complex64)


def _resolved_bundle() -> ResolvedDiagnostics:
    """Every resolved family on an (Ny, Nx, Nz) = (8, 8, 6) one-species grid."""

    shapes = {
        "kxst": (2, 1, 8),
        "kyst": (2, 1, 8),
        "kxkyst": (2, 1, 8, 8),
        "zst": (2, 1, 6),
    }
    fill = {
        "Wg": 1.0,
        "Wphi": 1.0,
        "Wapar": 0.0,
        "HeatFlux": 1.0,
        "HeatFluxES": 2.0,
        "HeatFluxApar": 3.0,
        "HeatFluxBpar": 4.0,
        "ParticleFlux": 1.0,
        "ParticleFluxES": 5.0,
        "ParticleFluxApar": 6.0,
        "ParticleFluxBpar": 7.0,
        "TurbulentHeating": 8.0,
    }
    payload = {
        f"{family}_{suffix}": np.full(shape, value, dtype=float)
        for family, value in fill.items()
        for suffix, shape in shapes.items()
    }
    return ResolvedDiagnostics(
        **payload,
        Wg_lmst=np.ones((2, 1, 8, 4)),
        Phi2_kxt=np.ones((2, 8)),
        Phi2_kyt=np.ones((2, 8)),
        Phi2_kxkyt=np.ones((2, 8, 8)),
        Phi2_zt=np.ones((2, 6)),
        Phi2_zonal_t=np.ones((2,)),
        Phi2_zonal_kxt=np.ones((2, 8)),
        Phi2_zonal_zt=np.ones((2, 6)),
    )


class _FakeVar:
    def __init__(self) -> None:
        self.value = None
        self.attrs: dict = {}

    def __setitem__(self, _idx, value) -> None:
        self.value = np.asarray(value)

    def setncattr(self, key, value) -> None:
        self.attrs[key] = value


class _FakeGroup:
    """Records what a writer puts into a netCDF4 group or root."""

    def __init__(self) -> None:
        self.vars: dict[str, _FakeVar] = {}
        self.attrs: dict = {}

    def createVariable(self, name, _dtype, _dims=()):
        self.vars[name] = _FakeVar()
        return self.vars[name]

    def setncattr(self, key, value) -> None:
        self.attrs[key] = value

    def __getitem__(self, name):
        return self.vars[name].value


def _linear_result(**overrides) -> RuntimeLinearResult:
    base = dict(
        ky=0.2,
        gamma=0.3,
        omega=-0.4,
        selection=ModeSelection(ky_index=1, kx_index=2, z_index=3),
        t=np.asarray([0.1, 0.2]),
        signal=np.asarray([1.0, 2.0]),
    )
    base.update(overrides)
    return RuntimeLinearResult(**base)


def _summary(paths) -> dict:
    return json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))


def _patch(monkeypatch, target, **values) -> None:
    for name, value in values.items():
        monkeypatch.setattr(target, name, value)


def test_write_runtime_linear_artifacts_writes_bundle(tmp_path: Path) -> None:
    result = _linear_result(
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

    summary = _summary(paths)
    assert summary["kind"] == "linear"
    assert summary["gamma"] == 0.3
    assert summary["fit_window_tmin"] == 0.1
    assert summary["fit_window_tmax"] == 0.3
    assert summary["fit_signal_used"] == "phi"
    assert summary["has_eigenfunction"] is True
    assert summary["has_quasilinear"] is True
    assert summary["quasilinear"]["heat_flux_weight_total"] == pytest.approx(1.5)
    claim = "uncalibrated_saturation_rule"
    assert summary["quasilinear"]["metadata"]["claim_level"] == claim
    assert summary["selection"]["ky_index"] == 1
    csv_lines = Path(paths["timeseries"]).read_text(encoding="utf-8").splitlines()
    assert csv_lines[0] == "t,signal_real,signal_imag,signal_abs"
    for key in (
        "timeseries",
        "eigenfunction",
        "state",
        "quasilinear_summary",
        "quasilinear_species",
    ):
        assert Path(paths[key]).exists()
    ql_summary = json.loads(
        Path(paths["quasilinear_summary"]).read_text(encoding="utf-8")
    )
    assert ql_summary["metadata"]["claim_level"] == claim


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

    def summary_of(**kw):
        ns = dict(diagnostics=None, state=None, ky_selected=0.2, kx_selected=0.0)
        return _nonlinear_summary(SimpleNamespace(**{**ns, "phi2": None, **kw}))

    summary = summary_of(
        diagnostics=_diag(**_TRACE),
        state=np.zeros((1, 2, 3), dtype=np.complex64),
        kx_selected=0.1,
    )
    assert summary["kind"] == "nonlinear"
    assert summary["n_samples"] == 2
    assert summary["heat_flux_last"] == pytest.approx(3.1)
    assert summary["n_state_shape"] == [1, 2, 3]
    assert summary_of(phi2=np.asarray(7.0))["phi2_last"] == pytest.approx(7.0)


def test_write_runtime_linear_artifacts_splits_complex_signal_columns(
    tmp_path: Path,
) -> None:
    signal = np.asarray([1.0 + 2.0j, 3.0 + 4.0j])
    result = _linear_result(signal=signal, state=None, fit_signal_used="phi")

    paths = write_runtime_linear_artifacts(tmp_path / "linear_complex", result)

    rows = Path(paths["timeseries"]).read_text(encoding="utf-8").splitlines()
    assert rows[0] == "t,signal_real,signal_imag,signal_abs"
    data = np.loadtxt(paths["timeseries"], delimiter=",", skiprows=1)
    np.testing.assert_allclose(
        data[:, 1:], np.c_[signal.real, signal.imag, np.abs(signal)]
    )


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
    for suffix in (
        "summary.json",
        "timeseries.csv",
        "eigenfunction.csv",
        "diagnostics.csv",
    ):
        stem = suffix.split(".")[0]
        assert _artifact_base(tmp_path / f"case.{suffix}") == tmp_path / f"case.{stem}"
    assert _artifact_base(tmp_path / "case.out.nc") == tmp_path / "case.out.nc"
    for name in ("case.nc", "case.restart.nc", "case.big.nc"):
        assert _netcdf_bundle_base(tmp_path / name) == tmp_path / "case"
    assert _is_netcdf_output_target(tmp_path / "case.out.nc") is True
    assert _is_netcdf_output_target(tmp_path / "case.csv") is False

    for series, flat in (
        ([1.0, 2.0], [1.0, 2.0]),
        ([[1.0], [2.0]], [1.0, 2.0]),
        ([[1.0, 3.0], [2.0, 4.0]], [2.0, 3.0]),
    ):
        assert np.allclose(_flatten_series(np.array(series)), np.array(flat))


def test_runtime_artifact_restart_resolution_and_species_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = "tools_out/run.out.nc"
    cfg = RuntimeConfig(output=RuntimeOutputConfig(path=out))
    cfg_custom = RuntimeConfig(
        output=RuntimeOutputConfig(
            path=out, restart_to_file="custom_to.nc", restart_from_file="custom_from.nc"
        )
    )
    for config, for_write, name in (
        (cfg, True, "run.restart.nc"),
        (cfg, False, "run.restart.nc"),
        (cfg_custom, True, "custom_to.nc"),
        (cfg_custom, False, "custom_from.nc"),
    ):
        assert _resolve_restart_path(out, config, for_write=for_write).name == name

    policy_cfg = RuntimeConfig(
        time=TimeConfig(dt=0.2, t_max=1.0, fixed_dt=True, diagnostics=True),
        output=RuntimeOutputConfig(
            path="tools_out/policy.out.nc", save_for_restart=True, nsave=2
        ),
    )
    monkeypatch.setattr(
        runtime_artifacts,
        "_resolve_restart_path",
        lambda _path, _cfg, *, for_write: Path(
            "policy_to.restart.nc" if for_write else "policy_from.restart.nc"
        ),
    )
    policy = resolve_nonlinear_artifact_policy(
        policy_cfg, out="tools_out/policy.out.nc", diagnostics=None, steps=None, dt=None
    )
    assert policy.netcdf_output_target is True
    assert policy.diagnostics_on is True
    assert policy.remaining_steps == 5
    assert policy.checkpoint_steps == 2
    assert policy.restart_from == Path("policy_from.restart.nc")
    assert policy.restart_to == Path("policy_to.restart.nc")

    total = np.array([2.0, 4.0], dtype=np.float32)
    per_species = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    for species, expected in (
        (None, [[1.0, 1.0], [2.0, 2.0]]),
        (np.array([3.0, 4.0], dtype=np.float32), [[3.0], [4.0]]),
        (per_species, per_species),
    ):
        np.testing.assert_allclose(_species_matrix(total, 2, species), expected)
    assert np.allclose(_resolved_species_time(None, fallback=total), total)
    assert np.allclose(
        _resolved_species_time(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), fallback=total
        ),
        np.array([3.0, 7.0], dtype=np.float32),
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

    for name, bad in (
        ("phi", np.asarray([1.0 + np.nan * 1j])),
        ("apar", np.asarray([np.inf])),
        ("bpar", np.asarray([np.nan])),
    ):
        fields_ = {
            "phi": np.asarray([0.0]),
            "apar": np.asarray([0.0]),
            "bpar": np.asarray([0.0]),
        }
        fields_[name] = bad
        bad_field = replace(finite, fields=SimpleNamespace(**fields_))
        with pytest.raises(RuntimeError, match=f"finite nonlinear {name}"):
            validate_finite_runtime_result(bad_field, label="finite nonlinear")

    bad_diag = _diag(particle_flux_t=[0.0, np.nan])
    bad_diag_result = replace(finite, diagnostics=bad_diag, fields=None, state=None)
    with pytest.raises(
        RuntimeError,
        match=r"finite nonlinear produced non-finite diagnostics in particle_flux_t at sample 1 at t=0.2",
    ):
        validate_finite_runtime_result(bad_diag_result, label="finite nonlinear")


def test_runtime_artifact_condense_output_helpers_reject_bad_axis_lengths() -> None:
    for condense, n, axis_name, kw in (
        (_condense_kx_for_output, 7, "kx", ("full_nx", "active_nx")),
        (_condense_ky_for_output, 5, "ky", ("full_ny", "active_ny")),
    ):
        full_axis = np.arange(n, dtype=np.float32)
        indices = (
            _dealiased_kx_indices if axis_name == "kx" else _dealiased_ky_indices
        )(n)
        active_axis = full_axis[indices]
        lengths = {kw[0]: n, kw[1]: active_axis.size}
        for given in (full_axis, active_axis):
            np.testing.assert_allclose(condense(given, **lengths), active_axis)
        with pytest.raises(ValueError, match=f"{axis_name}-resolved diagnostic"):
            condense(np.arange(n - 1), **lengths)

    full = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    active = full[_dealiased_ky_indices(5)][:, _dealiased_kx_indices(7)]

    def condense(arr, **kw):
        return _condense_kykx_for_output(
            arr,
            full_ny=5,
            full_nx=7,
            active_ny=active.shape[0],
            active_nx=active.shape[1],
            **kw,
        )

    np.testing.assert_allclose(condense(full), active)
    np.testing.assert_allclose(condense(full[_dealiased_ky_indices(5)]), active)
    with pytest.raises(ValueError, match="ky-kx diagnostic ky length"):
        condense(np.zeros((4, 7)))
    # Nyc = 1 + 5 // 2 = 3 is a *valid* length: it is the half-spectrum block.
    # What it cannot be served without is the family's ky weighting, because a
    # half-axis row of a Hermitian-weighted reduction carries the conjugate
    # partner's share and a transport row does not. The refusal names both
    # conventions rather than guessing one.
    with pytest.raises(ValueError, match="ky_weighting"):
        condense(np.zeros((3, 7)))
    half_block = full[:3]
    np.testing.assert_allclose(
        condense(half_block, ky_weighting=KY_WEIGHTING_PAIR), active
    )
    # The other convention divides the pair weight back out, so a paired row
    # publishes the per-row value the two-sided axis stores. Ny = 5 is odd, so
    # only row 0 is self-conjugate and rows 1 and 2 carry the weight 2.
    np.testing.assert_allclose(
        condense(half_block, ky_weighting=KY_WEIGHTING_PER_ROW),
        active / np.asarray([1.0, 2.0])[:, None],
    )
    with pytest.raises(ValueError, match="ky-kx diagnostic kx length"):
        condense(np.zeros((active.shape[0], 6)))


def test_runtime_artifact_spectral_helpers() -> None:
    field = np.array([[[1.0 + 2.0j, 3.0 + 4.0j]]], dtype=np.complex64)
    ri = _spectral_to_ri(field)
    assert ri.shape == (1, 1, 2, 2)
    assert np.allclose(ri[0, 0, 0], np.array([1.0, 2.0]))

    assert _spectral_to_xy(np.ones((2, 2, 1), dtype=np.complex64)).shape == (2, 2, 1)
    six_d = np.ones((1, 2, 3, 4, 4, 5), dtype=np.complex64)
    assert _restart_to_netcdf_layout(six_d).shape[-1] == 2
    assert _restart_to_netcdf_layout(six_d[0]).shape[0] == 1

    for helper in (
        _spectral_to_ri,
        _spectral_species_to_ri,
        _restart_to_netcdf_layout,
        _state_basis_moments,
    ):
        with pytest.raises(ValueError):
            helper(np.ones((2, 2), dtype=np.complex64))

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
    diag = _diag(
        Wg_t=[1.0, 2.0],
        Wphi_t=[0.5, 0.6],
        energy_t=[1.5, 2.6],
        resolved=ResolvedDiagnostics(
            Phi2_kxkyt=phi2_kykx,
            Phi_zonal_line_kxt=line,
            Phi_zonal_mode_kxt=2.0 * line,
            Wg_kxst=wg_kxst,
        ),
    )
    result = RuntimeNonlinearResult(
        t=np.asarray(diag.t), diagnostics=diag, ky_selected=0.0, kx_selected=0.0
    )

    paths = write_runtime_nonlinear_artifacts(tmp_path / "active.out.nc", result, cfg)

    with nc.Dataset(paths["out"], "r") as root:
        assert root.dimensions["kx"].size == active_kx
        assert root.dimensions["ky"].size == active_ky
        variables = root.groups["Diagnostics"].variables
        stored_line_ri = np.asarray(variables["Phi_zonal_line_kxt"][:])
        stored_line = stored_line_ri[..., 0] + 1j * stored_line_ri[..., 1]
        np.testing.assert_allclose(stored_line, line)
        np.testing.assert_allclose(np.asarray(variables["Wg_kxst"][:]), wg_kxst)
        np.testing.assert_allclose(np.asarray(variables["Phi2_kxkyt"][:]), phi2_kykx)


def test_runtime_artifact_read_optional_var() -> None:
    class _Var:
        dimensions = ("time", "ri")

        def __getitem__(self, _key):
            return np.array([[1.0, 2.0], [3.0, 4.0]])

    group = SimpleNamespace(variables={"present": _Var()})
    assert _read_optional_var(group, "missing") is None
    out = _read_optional_var(group, "present")
    assert np.allclose(out, np.array([1.0 + 2.0j, 3.0 + 4.0j]))


def test_runtime_artifact_condense_helpers() -> None:
    assert _condense_resolved_for_output(None) is None

    resolved = replace(
        _resolved_bundle(),
        Phi_zonal_mode_kxt=np.ones((2, 8), dtype=np.complex64),
    )
    condensed = _condense_resolved_for_output(resolved)
    assert condensed is not None
    assert condensed.Wg_kxst.shape[-1] <= resolved.Wg_kxst.shape[-1]
    assert (
        condensed.Phi_zonal_mode_kxt.shape[-1] <= resolved.Phi_zonal_mode_kxt.shape[-1]
    )
    diag_condensed = _condense_diagnostics_for_netcdf_output(_diag(resolved=resolved))
    assert diag_condensed.resolved is not None


def test_runtime_artifact_small_helpers() -> None:
    assert _dealiased_kx_count(1) == 1
    assert np.array_equal(_dealiased_kx_indices(1), np.array([0], dtype=np.int32))


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
    cfg = SimpleNamespace(
        grid=SimpleNamespace(Ny=5, Nx=7, Nz=8, ntheta=None, nperiod=None)
    )
    root = _FakeGroup()
    _write_runtime_root_metadata(root, cfg, nspecies=2, nl=3, nm=4)
    for name, value in (("ny", 5), ("nx", 7), ("ntheta", 8), ("nperiod", 1)):
        assert int(root[name]) == value
    assert root.attrs["schema_version"] == 1
    assert root.vars["code_info"].attrs["value"] == "gkx"

    field = np.arange(5 * 7 * 2, dtype=np.float32).reshape(5, 7, 2)
    active = _dealiased_spectral_field(field, ky_axis=0, kx_axis=1)
    np.testing.assert_allclose(active, field[:2][:, [5, 6, 0, 1, 2], :])


def test_runtime_artifact_geometry_and_input_group_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = SimpleNamespace(
        z=np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
        kx=np.asarray([-0.2, 0.0, 0.2], dtype=np.float32),
        ky=np.asarray([0.0, 0.3, -0.3], dtype=np.float32),
    )
    profiles = {
        "bmag": [1.0, 1.1, 1.2],
        "bgrad": [0.1, 0.2, 0.3],
        "gb": [0.4, 0.5, 0.6],
        "gb0": [0.7, 0.8, 0.9],
        "cv": [1.0, 1.1, 1.2],
        "cv0": [1.3, 1.4, 1.5],
        "gds2": [1.6, 1.7, 1.8],
        "gds21": [1.9, 2.0, 2.1],
        "gds22": [2.2, 2.3, 2.4],
        "grho": [0.9, 1.0, 1.1],
        "jacobian": [1.0, 1.5, 2.0],
    }
    geom = SimpleNamespace(
        **{
            f"{k}_profile": np.asarray(v, dtype=np.float32) for k, v in profiles.items()
        },
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
    _patch(
        monkeypatch,
        nonlinear_netcdf,
        apply_geometry_grid_defaults=lambda _geom, grid_cfg: grid_cfg,
        build_spectral_grid=lambda _cfg: grid,
        build_runtime_geometry=lambda _cfg: object(),
        ensure_flux_tube_geometry_data=lambda _geom, _theta: geom,
        real_fft_ordered_kx=lambda arr: np.asarray([-0.2, 0.0, 0.2], dtype=np.float32),
        _half_ky_values_of=lambda _grid: np.asarray([0.0, 0.3], dtype=np.float32),
    )

    geom_group = _FakeGroup()
    theta, kx_vals, ky_vals, geom_out = _write_geometry_group(geom_group, cfg)
    np.testing.assert_allclose(theta, grid.z)
    np.testing.assert_allclose(kx_vals, np.asarray([-0.2, 0.0, 0.2], dtype=np.float32))
    np.testing.assert_allclose(ky_vals, np.asarray([0.0, 0.3], dtype=np.float32))
    assert geom_out is geom
    np.testing.assert_allclose(geom_group["bmag"], geom.bmag_profile)
    assert float(geom_group["gradpar"]) == pytest.approx(0.75)
    assert int(geom_group["nfp"]) == 5

    inputs = _FakeGroup()
    _write_input_parameters_group(inputs, cfg, geom)
    for name, value in (("igeo", 0), ("slab", 0), ("zero_shat", 0)):
        assert int(inputs[name]) == value
    assert float(inputs["kxfac"]) == pytest.approx(1.1)
    assert float(inputs["beta"]) == pytest.approx(0.02)
    assert float(inputs["grhoavg"]) == pytest.approx(np.mean(geom.grho_profile))


def test_runtime_artifact_geometry_writer_applies_imported_grid_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    theta_closed = np.linspace(-1.0, 1.0, 5, dtype=np.float64)
    ramps = {
        "bmag": (1.0, 1.4),
        "bgrad": (0.1, 0.5),
        "gds2": (2.0, 2.4),
        "gds21": (-0.2, 0.2),
        "gds22": (0.7, 0.9),
        "cv": (0.3, 0.7),
        "gb": (0.4, 0.8),
        "cv0": (-0.1, 0.1),
        "gb0": (-0.2, 0.2),
        "jacobian": (3.0, 3.4),
        "grho": (1.0, 1.2),
        "cylindrical_R": (5.0, 6.0),
        "cylindrical_Z": (-0.5, 0.5),
        "toroidal_angle": (-1.0, 1.0),
    }
    geom = FluxTubeGeometryData(
        theta=theta_closed,
        gradpar_value=0.5,
        **{
            f"{k}_profile": np.linspace(a, b, theta_closed.size)
            for k, (a, b) in ramps.items()
        },
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
    monkeypatch.setattr(nonlinear_netcdf, "build_runtime_geometry", lambda _cfg: geom)

    group = _FakeGroup()
    theta, _kx, _ky, geom_out = _write_geometry_group(group, cfg)

    assert theta.shape == (theta_closed.size - 1,)
    np.testing.assert_allclose(theta, theta_closed[:-1].astype(np.float32))
    np.testing.assert_allclose(
        group["bmag"], np.asarray(geom.bmag_profile[:-1], dtype=np.float32)
    )
    assert geom_out.theta_closed_interval is False
    np.testing.assert_allclose(group["Rplot"], geom.cylindrical_R_profile[:-1])
    np.testing.assert_allclose(group["zeta_plot"], geom.toroidal_angle_profile[:-1])
    assert float(group["kxfac"]) == pytest.approx(1.25)


def test_runtime_artifact_particle_moments(monkeypatch) -> None:
    state = np.ones((1, 2, 3, 2, 4, 3), dtype=np.complex64)
    grid = SimpleNamespace(z=np.asarray([-1.0, 0.0, 1.0], dtype=np.float32))
    geom = object()
    cache = SimpleNamespace(
        Jl=np.ones((1, 2, 2, 4, 3), dtype=np.float32),
        JlB=np.ones((1, 2, 2, 4, 3), dtype=np.float32),
        kperp2=np.ones((2, 4, 3), dtype=np.float32),
    )
    _patch(
        monkeypatch,
        nonlinear_netcdf,
        apply_geometry_grid_defaults=lambda _geom, grid_cfg: grid_cfg,
        build_spectral_grid=lambda _grid: grid,
        build_runtime_geometry=lambda _cfg: geom,
        ensure_flux_tube_geometry_data=lambda _geom, _theta: geom,
        build_runtime_linear_params=lambda _cfg, **_kwargs: object(),
        build_linear_cache=lambda *_args, **_kwargs: cache,
    )

    moments = _particle_moments(state, SimpleNamespace(grid=SimpleNamespace()))
    for name in ("ParticleDensity", "ParticleUpar", "ParticleUperp"):
        np.testing.assert_allclose(moments[name], np.full((1, 2, 4, 3), 2.0))
    np.testing.assert_allclose(
        moments["ParticleTemp"], np.full((1, 2, 4, 3), 2.0 * np.sqrt(2.0))
    )


@pytest.mark.parametrize(
    ("species_extra", "expected_columns", "target"),
    [
        pytest.param(
            dict(
                heat_flux_species_t=[[3.0], [3.1]],
                particle_flux_species_t=[[4.0], [4.1]],
            ),
            ("heat_flux_s0", "particle_flux_s0"),
            "diag.csv",
            id="2d-species",
        ),
        pytest.param(
            dict(
                heat_flux_species_t=[3.0, 3.1],
                particle_flux_species_t=[4.0, 4.1],
                turbulent_heating_t=[5.0, 5.1],
                turbulent_heating_species_t=[5.0, 5.1],
            ),
            ("heat_flux_s0", "particle_flux_s0", "turbulent_heating_s0"),
            "diag1d",
            id="1d-species",
        ),
    ],
)
def test_write_runtime_nonlinear_artifacts_writes_csv_target(
    tmp_path: Path, species_extra: dict, expected_columns: tuple, target: str
) -> None:
    diag = _diag(**_TRACE, **species_extra, phi_mode_t=None)
    result = RuntimeNonlinearResult(
        t=np.asarray([0.1, 0.2]),
        diagnostics=diag,
        state=np.zeros((2, 2), dtype=np.complex64),
        ky_selected=0.2,
        kx_selected=0.0,
        phi2=7.0,
    )

    paths = write_runtime_nonlinear_artifacts(tmp_path / target, result)

    csv_path = Path(paths["diagnostics"])
    if target.endswith(".csv"):
        assert csv_path == tmp_path / target
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    for column in expected_columns:
        assert column in header
    summary = _summary(paths)
    assert summary["kind"] == "nonlinear"
    assert summary["n_samples"] == 2
    assert Path(paths["state"]).exists()


def test_write_runtime_nonlinear_artifacts_handles_scalar_only_result(
    tmp_path: Path,
) -> None:
    result_no_diag = RuntimeNonlinearResult(
        t=np.asarray([0.1]),
        diagnostics=None,
        phi2=7.0,
        ky_selected=0.2,
        kx_selected=0.0,
    )
    paths = write_runtime_nonlinear_artifacts(tmp_path / "scalar_only", result_no_diag)
    assert _summary(paths)["phi2_last"] == 7.0


def test_write_runtime_nonlinear_artifacts_writes_nonlinear_netcdf_bundle(
    tmp_path: Path,
) -> None:
    Dataset = pytest.importorskip("netCDF4").Dataset

    diag = _diag(
        t=[0.0, 0.1],
        dt_t=[0.05, 0.05],
        dt_mean=np.asarray(0.05),
        Wg_t=[1.0, 1.1],
        Wphi_t=[2.0, 2.1],
        heat_flux_t=[3.0, 3.1],
        particle_flux_t=[4.0, 4.1],
        energy_t=[3.0, 3.2],
        heat_flux_species_t=[[3.0], [3.1]],
        particle_flux_species_t=[[4.0], [4.1]],
        turbulent_heating_t=[8.0, 8.1],
        turbulent_heating_species_t=[[8.0], [8.1]],
        phi_mode_t=None,
        resolved=_resolved_bundle(),
    )
    fields_ = SimpleNamespace(
        phi=np.zeros((8, 8, 6), dtype=np.complex64), apar=None, bpar=None
    )
    result = RuntimeNonlinearResult(
        t=np.asarray([0.0, 0.1]),
        diagnostics=diag,
        fields=fields_,
        state=np.zeros((1, 4, 8, 8, 8, 6), dtype=np.complex64),
        ky_selected=0.2,
        kx_selected=0.0,
    )

    cfg = RuntimeConfig(
        # Every diagnostic above is hand-built with Ny = 8 ky rows, so this
        # case describes a two-sided run and has to say so; the layout-
        # invariance of the published bundle is gated separately, on a real
        # run of both axes, in test_ky_half_spectrum_output.py.
        grid=GridConfig(Nx=8, Ny=8, Nz=6, Lx=1.0, Ly=1.0, ky_layout="full"),
        time=TimeConfig(compressed_real_fft=True),
    )

    paths = write_runtime_nonlinear_artifacts(tmp_path / "probe.out.nc", result, cfg)

    for key in ("out", "restart", "big"):
        assert Path(paths[key]).exists()

    with Dataset(paths["out"], "r") as root:
        assert root.getncattr("schema_version") == 1
        assert set(root.groups) == {"Diagnostics", "Geometry", "Grids", "Inputs"}
        for name, value in (
            ("ny", 8),
            ("nx", 8),
            ("ntheta", 6),
            ("nhermite", 8),
            ("nlaguerre", 4),
            ("nspecies", 1),
        ):
            assert int(root.variables[name][()]) == value
        assert root.variables["code_info"].getncattr("value") == "gkx"
        assert root.dimensions["kx"].size == 5
        assert root.dimensions["ky"].size == 3
        variables = root.groups["Diagnostics"].variables
        for name in (
            "Phi2_t",
            "Phi2_kxt",
            "Wg_st",
            "Wg_kyst",
            "Wg_lmst",
            "HeatFlux_st",
            "ParticleFluxBpar_kxkyst",
            "TurbulentHeating_kxkyst",
        ):
            assert name in variables
        # Phi2_t weights the two paired rows (ky > 0) by 2: 5 * (1 + 2 + 2);
        # Phi2_kxt is the in-memory kx reduction, condensed.
        for name, expected in (
            ("Phi2_t", np.full(2, 25.0)),
            ("Phi2_kxt", np.full((2, 5), 1.0)),
            ("Phi2_kyt", np.full((2, 3), 5.0)),
            ("Phi2_kxkyt", np.ones((2, 3, 5))),
            ("HeatFluxES_st", np.full((2, 1), 16.0)),
            ("HeatFluxApar_st", np.full((2, 1), 24.0)),
            ("HeatFluxBpar_st", np.full((2, 1), 32.0)),
            ("TurbulentHeating_st", np.full((2, 1), 64.0)),
        ):
            np.testing.assert_allclose(variables[name][:], expected)

    with Dataset(paths["restart"], "r") as root:
        assert root.getncattr("schema_version") == 1
        assert root.dimensions["Nkx"].size == 5
        assert root.dimensions["Nky"].size == 3
        assert root.variables["G"].shape[-1] == 2
        assert "time" in root.variables

    with Dataset(paths["big"], "r") as root:
        assert root.getncattr("schema_version") == 1
        assert root.variables["code_info"].getncattr("value") == "gkx"
        for name in (
            "Phi",
            "PhiXY",
            "Density",
            "Upar",
            "Tpar",
            "Tperp",
            "ParticleDensity",
        ):
            assert name in root.groups["Diagnostics"].variables

    loaded = load_nonlinear_netcdf_diagnostics(paths["out"])
    np.testing.assert_allclose(np.asarray(loaded.Wg_t), [1.0, 1.1])
    np.testing.assert_allclose(np.asarray(loaded.turbulent_heating_t), [64.0, 64.0])
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

    def _fake_run_runtime_nonlinear(run_cfg, **kwargs):
        init = run_cfg.init
        calls.append((init.init_file, init.init_file_mode, init.init_file_scale))
        return RuntimeNonlinearResult(
            t=np.asarray([0.1]),
            diagnostics=_diag(1, Wphi_t=[0.0], energy_t=[1.0]),
            state=_STATE6,
            fields=SimpleNamespace(
                phi=np.zeros((1, 1, 1), dtype=np.complex64), apar=None, bpar=None
            ),
            ky_selected=0.2,
            kx_selected=0.0,
        )

    def _fake_write_runtime_nonlinear_artifacts(_out, _result, _cfg):
        restart_path.write_bytes(b"stub")
        return {"out": str(out_path), "restart": str(restart_path)}

    patch_runtime(monkeypatch, "run_runtime_nonlinear", _fake_run_runtime_nonlinear)
    patch_runtime(
        monkeypatch,
        "write_runtime_nonlinear_artifacts",
        _fake_write_runtime_nonlinear_artifacts,
    )

    run_runtime_nonlinear_with_artifacts(
        cfg, out=out_path, ky_target=0.2, steps=2, diagnostics=True
    )

    assert calls[:2] == [
        (str(restart_path), "add", 0.25),
        (str(restart_path), "replace", 1.0),
    ]


def test_runtime_orchestration_handoff_chunks_and_restarts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
            diagnostics=_diag(1, t=[0.1 * chunk_steps]),
            state=_STATE6,
        )

    def _write(_out, result, _cfg):
        assert result.diagnostics is not None
        writes.append(float(np.asarray(result.diagnostics.t)[-1]))
        restart_path.write_bytes(b"restart")
        return {"out": str(out_path), "restart": str(restart_path)}

    _patch(
        monkeypatch,
        runtime_artifacts,
        _resolve_restart_path=lambda _path, _cfg, *, for_write: restart_path,
        load_nonlinear_netcdf_diagnostics=lambda _path: _diag(1, t=[0.0]),
        _condense_diagnostics_for_netcdf_output=lambda diag, **_kw: diag,
        validate_finite_runtime_result=lambda _result, **_kw: None,
        run_runtime_nonlinear=_run,
        write_runtime_nonlinear_artifacts=_write,
    )

    result, paths = run_runtime_nonlinear_with_artifacts(
        cfg, out=out_path, ky_target=0.2, steps=12, diagnostics=True
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
    diag = _diag(3, energy_t=2.0 * np.ones(3))
    captured_steps: list[int | None] = []

    def _fake_run_runtime_nonlinear(_cfg, **kwargs):
        captured_steps.append(kwargs.get("steps"))
        return RuntimeNonlinearResult(
            t=np.asarray(diag.t), diagnostics=diag, state=_STATE6
        )

    patch_runtime(monkeypatch, "run_runtime_nonlinear", _fake_run_runtime_nonlinear)
    patch_runtime(
        monkeypatch,
        "write_runtime_nonlinear_artifacts",
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
    diag = _diag(
        1, Wphi_t=[2.0], heat_flux_t=[3.0], particle_flux_t=[4.0], energy_t=[3.0]
    )
    messages: list[str] = []
    captured: dict[str, object] = {}

    def _fake_run_runtime_nonlinear(_cfg, **kwargs):
        captured["show_progress"] = kwargs["show_progress"]
        callback = captured["status_callback"] = kwargs["status_callback"]
        assert callback is not None
        callback("live status propagated")
        return RuntimeNonlinearResult(
            t=np.asarray(diag.t), diagnostics=diag, ky_selected=0.2, kx_selected=0.0
        )

    patch_runtime(monkeypatch, "run_runtime_nonlinear", _fake_run_runtime_nonlinear)

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
    bad_diag = _diag(1, Wg_t=[np.nan], energy_t=[np.nan])
    calls = {"run": 0, "write": 0}

    def _fake_run_runtime_nonlinear(_cfg, **_kwargs):
        calls["run"] += 1
        return RuntimeNonlinearResult(
            t=np.asarray(bad_diag.t), diagnostics=bad_diag, state=_STATE6
        )

    def _fake_write(*_args, **_kwargs):
        calls["write"] += 1
        return {"out": str(out_path)}

    patch_runtime(monkeypatch, "run_runtime_nonlinear", _fake_run_runtime_nonlinear)
    patch_runtime(monkeypatch, "write_runtime_nonlinear_artifacts", _fake_write)

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
    for state, cfg in (
        (None, None),
        (np.zeros((1, 1), dtype=np.complex64), RuntimeConfig()),
    ):
        result = RuntimeNonlinearResult(t=np.asarray([]), diagnostics=None, state=state)
        with pytest.raises(ValueError):
            write_runtime_nonlinear_artifacts(tmp_path / "case.out.nc", result, cfg=cfg)


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
    with Dataset(out_nc, "a") as root:
        root.setncattr("schema_version", 2)
    with pytest.raises(ValueError, match="unsupported GKX NetCDF schema_version 2"):
        load_nonlinear_netcdf_diagnostics(out_nc)


def test_run_runtime_nonlinear_with_artifacts_append_preserves_loaded_netcdf_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    Dataset = pytest.importorskip("netCDF4").Dataset
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

    def _chunk(base: float) -> RuntimeNonlinearResult:
        ramp = np.arange(full_ny * full_nx, dtype=np.float32)
        phi2_kxkyt = base + ramp.reshape(1, full_ny, full_nx)
        diag = _diag(
            1,
            Wg_t=[base + 1.0],
            Wphi_t=[base + 2.0],
            heat_flux_t=[base + 3.0],
            particle_flux_t=[base + 4.0],
            energy_t=[base + 3.0],
            heat_flux_species_t=[[base + 3.0]],
            particle_flux_species_t=[[base + 4.0]],
            turbulent_heating_t=[base + 5.0],
            turbulent_heating_species_t=[[base + 5.0]],
            phi_mode_t=np.asarray([base + 1.0j * base]),
            resolved=ResolvedDiagnostics(
                Phi2_kxt=np.sum(phi2_kxkyt[:, ky_idx, :], axis=1),
                Phi2_kyt=np.sum(phi2_kxkyt[:, :, kx_idx], axis=2),
                Phi2_kxkyt=phi2_kxkyt,
                Phi_zonal_line_kxt=(
                    base + np.arange(full_nx, dtype=np.float32)[None, :]
                ).astype(np.complex64),
                Wg_kxst=base
                + np.arange(full_nx, dtype=np.float32).reshape(1, 1, full_nx),
                Wg_kyst=base
                + np.arange(full_ny, dtype=np.float32).reshape(1, 1, full_ny),
                Wg_kxkyst=base + ramp.reshape(1, 1, full_ny, full_nx),
            ),
        )
        return RuntimeNonlinearResult(
            t=np.asarray([0.1]), diagnostics=diag, ky_selected=0.2, kx_selected=0.0
        )

    def _schema(root) -> dict:
        variables = root.groups["Diagnostics"].variables
        return {name: tuple(var.dimensions) for name, var in variables.items()}

    write_runtime_nonlinear_artifacts(out, _chunk(1.0), cfg)
    with Dataset(out, "r") as root:
        before_schema = _schema(root)
        assert root.dimensions["time"].size == 1
    assert "Phi_zonal_line_kxt" in before_schema
    assert all("phi_mode" not in name.lower() for name in before_schema)

    def _fake_run_runtime_nonlinear(run_cfg, **_kwargs):
        assert run_cfg.init.init_file == str(restart)
        return _chunk(20.0)

    patch_runtime(monkeypatch, "run_runtime_nonlinear", _fake_run_runtime_nonlinear)

    result, paths = run_runtime_nonlinear_with_artifacts(
        cfg, out=out, ky_target=0.2, steps=1, diagnostics=True
    )

    assert result.diagnostics is not None
    assert result.diagnostics.phi_mode_t is None
    with Dataset(paths["out"], "r") as root:
        assert _schema(root) == before_schema
        assert root.dimensions["time"].size == 2
        np.testing.assert_allclose(
            root.groups["Grids"].variables["time"][:], np.asarray([0.1, 0.2])
        )
        line = root.groups["Diagnostics"].variables["Phi_zonal_line_kxt"]
        assert line.shape == (2, int(kx_idx.size), 2)


def test_run_runtime_nonlinear_with_artifacts_validation_branches(
    tmp_path: Path,
) -> None:
    out = tmp_path / "case.out.nc"
    for diagnostics, error in ((False, ValueError), (True, FileNotFoundError)):
        cfg = RuntimeConfig(
            time=TimeConfig(dt=0.2, t_max=1.0, diagnostics=diagnostics, fixed_dt=True),
            output=RuntimeOutputConfig(path=str(out), restart=True),
        )
        with pytest.raises(error):
            run_runtime_nonlinear_with_artifacts(
                cfg, out=out, ky_target=0.2, diagnostics=diagnostics
            )


def test_run_runtime_nonlinear_with_artifacts_history_and_restart_paths(
    monkeypatch, tmp_path: Path
) -> None:
    out = tmp_path / "case.out.nc"
    (tmp_path / "case.restart.nc").write_bytes(b"restart")
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

    def _sample(offset: float) -> SimulationDiagnostics:
        return _diag(
            1,
            t=[0.5],
            dt_t=[0.5],
            dt_mean=np.asarray(0.5),
            Wg_t=[1.0 + offset],
            Wphi_t=[2.0 + offset],
            heat_flux_t=[3.0 + offset],
            particle_flux_t=[4.0 + offset],
            energy_t=[3.0 + 2 * offset],
            resolved=ResolvedDiagnostics(Phi2_kxt=np.ones((1, 4), dtype=float)),
        )

    result_chunk = RuntimeNonlinearResult(
        t=np.asarray([0.5]), diagnostics=_sample(0.1), state=_STATE6
    )
    captured = {"writes": 0}

    def _write(*_args, **_kwargs):
        captured["writes"] += 1
        return {"out": str(out)}

    patch_runtime(
        monkeypatch, "load_nonlinear_netcdf_diagnostics", lambda _path: _sample(0.0)
    )
    patch_runtime(monkeypatch, "run_runtime_nonlinear", lambda *_a, **_k: result_chunk)
    patch_runtime(monkeypatch, "concat_runtime_diagnostics", lambda diags: diags[-1])
    patch_runtime(monkeypatch, "write_runtime_nonlinear_artifacts", _write)

    result, paths = run_runtime_nonlinear_with_artifacts(
        cfg, out=out, ky_target=0.2, diagnostics=True
    )
    assert isinstance(result, RuntimeNonlinearResult)
    assert paths["out"] == str(out)
    assert captured["writes"] >= 1


def test_linear_summary_carries_fit_quality_fields(tmp_path: Path) -> None:
    """gamma/omega stderr and R^2 reach the summary, with None for eigensolves."""

    base = dict(
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        t=np.asarray([0.1, 0.2, 0.3]),
        signal=np.asarray([1.0, 2.0, 4.0]),
        fit_window_tmin=0.1,
        fit_window_tmax=0.3,
        fit_signal_used="phi",
    )
    fitted = _linear_result(
        gamma_stderr=0.004, omega_stderr=0.007, fit_r2=0.999, **base
    )
    summary = _summary(write_runtime_linear_artifacts(tmp_path / "fitted", fitted))
    assert summary["gamma_stderr"] == pytest.approx(0.004)
    assert summary["omega_stderr"] == pytest.approx(0.007)
    assert summary["fit_r2"] == pytest.approx(0.999)

    # An eigensolve has no fit window statistics, and a degenerate fit reports
    # an infinite stderr; both must serialize as JSON null rather than NaN.
    eigen = _linear_result(gamma_stderr=float("inf"), **base)
    summary = _summary(write_runtime_linear_artifacts(tmp_path / "eigen", eigen))
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

    def scan(n: int, warm_start):
        return SimpleNamespace(
            ky=np.asarray([0.3, 0.2][:n]),
            gamma=np.asarray([0.2, 0.1][:n]),
            omega=np.asarray([-0.5, -0.4][:n]),
            quasilinear=None,
            parallel=None,
            warm_start=warm_start,
        )

    paths = write_runtime_linear_scan_artifacts(tmp_path / "warm_bundle", scan(2, warm))
    assert _summary(paths)["warm_start"] == warm

    cold = write_runtime_linear_scan_artifacts(tmp_path / "cold_bundle", scan(1, None))
    assert "warm_start" not in _summary(cold)


# ---- from test_runtime_diagnostics.py ----


def test_jl_family_accepts_four_dimensional_arrays_and_rejects_bad_ranks() -> None:
    cache_4d = SimpleNamespace(
        Jl=jnp.ones((2, 2, 1, 3), dtype=jnp.float32),
        JlB=2.0 * jnp.ones((2, 2, 1, 3), dtype=jnp.float32),
    )

    jl, jlb, jfac = _jl_family(cache_4d)

    assert jl.shape == (1, 2, 2, 1, 3)
    assert jlb.shape == (1, 2, 2, 1, 3)
    assert jfac.shape == jl.shape

    with pytest.raises(ValueError, match="unexpected Jl rank"):
        _jl_family(SimpleNamespace(Jl=jnp.ones((2, 3, 4)), JlB=cache_4d.JlB))
    with pytest.raises(ValueError, match="unexpected JlB rank"):
        _jl_family(SimpleNamespace(Jl=cache_4d.Jl, JlB=jnp.ones((2, 3, 4))))


def test_flux_fac_nonzero_matches_positive_ky_convention() -> None:
    """Flux representatives on a two-sided axis: ``ky > 0``, plus Nyquist.

    The rule used to be exactly ``ky > 0``, which drops an even grid's Nyquist
    row -- ``fftfreq`` stores ``|ky| = Ny/2`` once, as ``-Ny/2`` -- from the
    flux, while a half-spectrum axis stores the same row as ``+Ny/2`` and
    counted it. The two layouts therefore named different sums. Plan 5.3 N3
    settles it in favour of counting the row, at the self-conjugate weight
    ``0.5`` that the kernel's own factor of two restores to one: the flux
    kernel carries an explicit ``i*ky``, so that row's contribution is odd
    under a sign choice that is pure convention, and a flux may not be.

    Nothing shipped moves. The contribution is identically zero on any state
    representing a real field, and two-thirds dealiasing zeroes the row in
    every run; only an explicitly undealiased reduction sees the weight at all.
    """

    cfg = CycloneBaseCase()
    # The two-sided axis by name: this test is about how the two-sided rule
    # treats a row that only the two-sided axis stores with a negative sign.
    grid = build_spectral_grid(replace(cfg.grid, Ny=8, Nx=4, ky_layout="full"))
    fac = np.asarray(_transport_mode_weight(grid, use_dealias=False))
    ky = np.asarray(grid.ky, dtype=float)
    nyquist = np.argmin(ky)  # the single -Ny/2 entry
    assert ky[nyquist] < 0.0

    pos = ky > 0.0
    assert np.allclose(fac[pos], 1.0)
    assert np.allclose(fac[nyquist], 0.5)
    other = ~pos
    other[nyquist] = False
    assert np.allclose(fac[other], 0.0)

    # Dealiased -- which is every shipped reduction -- the row is gone again.
    dealiased = np.asarray(_transport_mode_weight(grid, use_dealias=True))
    assert np.allclose(dealiased[nyquist], 0.0)


def test_state_mask_and_apply_mask_remove_dealiased_and_zonal00_modes() -> None:
    cache = SimpleNamespace(
        ky=jnp.asarray([0.0, 0.25], dtype=jnp.float32),
        kx=jnp.asarray([0.0, 0.5], dtype=jnp.float32),
        dealias_mask=jnp.asarray([[1, 1], [0, 1]], dtype=bool),
    )
    state = jnp.asarray(
        np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3) + 1.0j,
        dtype=jnp.complex64,
    )

    mask = np.asarray(_completed_step_state_mask(cache))
    expected_mask = np.asarray([[False, True], [False, True]])
    np.testing.assert_array_equal(mask, expected_mask)

    masked = np.asarray(_apply_completed_step_state_mask(state, cache))
    np.testing.assert_allclose(masked[0, 0], 0.0)
    np.testing.assert_allclose(masked[1, 0], 0.0)
    np.testing.assert_allclose(masked[0, 1], np.asarray(state[0, 1]))
    np.testing.assert_allclose(masked[1, 1], np.asarray(state[1, 1]))


def test_validate_finite_runtime_diagnostics_covers_optional_and_resolved_schema() -> (
    None
):
    n = 3
    t = np.asarray([0.0, 1.0, 2.0])
    resolved_payload = {
        field.name: np.full((n, 2), 1.0 + idx, dtype=np.float64)
        for idx, field in enumerate(fields(ResolvedDiagnostics))
    }
    resolved = ResolvedDiagnostics(**resolved_payload)
    diag = SimulationDiagnostics(
        t=t,
        dt_t=np.full(n, 0.1),
        dt_mean=np.asarray(0.1),
        gamma_t=np.linspace(0.0, 0.2, n),
        omega_t=np.linspace(0.3, 0.5, n),
        Wg_t=np.linspace(1.0, 1.2, n),
        Wphi_t=np.linspace(0.5, 0.7, n),
        Wapar_t=np.zeros(n),
        heat_flux_t=np.linspace(0.0, 0.2, n),
        particle_flux_t=np.linspace(0.1, 0.3, n),
        energy_t=np.linspace(1.5, 1.9, n),
        heat_flux_species_t=np.ones((n, 2)),
        particle_flux_species_t=np.ones((n, 2)),
        turbulent_heating_t=np.ones(n),
        turbulent_heating_species_t=np.ones((n, 2)),
        phi_mode_t=np.asarray([1.0 + 0.0j, 0.5 + 0.25j, 0.25 + 0.5j]),
        resolved=resolved,
    )

    validate_finite_runtime_diagnostics(diag, label="bounded")

    expected_resolved = {
        "Phi_zonal_mode_kxt",
        "Phi_zonal_line_kxt",
        "HeatFluxES_kxst",
        "ParticleFluxBpar_zst",
        "TurbulentHeating_zst",
    }
    assert expected_resolved.issubset(
        {field.name for field in fields(ResolvedDiagnostics)}
    )

    bad_scalar = replace(diag, heat_flux_t=np.asarray([0.0, np.inf, 0.2]))
    with pytest.raises(RuntimeError, match="heat_flux_t at sample 1 at t=1"):
        validate_finite_runtime_diagnostics(bad_scalar, label="bounded")

    bad_resolved_payload = {
        field.name: np.asarray(getattr(resolved, field.name)).copy()
        for field in fields(ResolvedDiagnostics)
    }
    bad_resolved_payload["Phi_zonal_line_kxt"][2, 0] = np.nan
    bad_resolved = replace(diag, resolved=ResolvedDiagnostics(**bad_resolved_payload))
    with pytest.raises(
        RuntimeError, match=r"resolved\.Phi_zonal_line_kxt at sample 2 at t=2"
    ):
        validate_finite_runtime_diagnostics(bad_resolved, label="bounded")


def test_cached_hermitian_mode_weight_matches_full_and_one_sided_conventions() -> None:
    dealias = jnp.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)
    full_cache = SimpleNamespace(
        ky=jnp.asarray([-0.25, 0.0, 0.25], dtype=jnp.float32),
        kx=jnp.asarray([-0.5, 0.5], dtype=jnp.float32),
        dealias_mask=dealias,
    )
    one_sided_cache = SimpleNamespace(
        ky=jnp.asarray([0.0, 0.25, 0.5], dtype=jnp.float32),
        kx=jnp.asarray([-0.5, 0.5], dtype=jnp.float32),
        dealias_mask=dealias,
    )

    np.testing.assert_allclose(
        np.asarray(_cached_hermitian_mode_weight(full_cache, use_dealias=True)),
        np.asarray(dealias),
    )
    np.testing.assert_allclose(
        np.asarray(_cached_hermitian_mode_weight(one_sided_cache, use_dealias=False)),
        np.asarray([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]], dtype=np.float32),
    )


def test_grid_helper_contracts_preserve_selected_ky_and_fft_ordering() -> None:
    grid = SimpleNamespace(
        kx=jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32),
        ky=jnp.asarray([-0.3], dtype=jnp.float32),
        z=jnp.asarray([-np.pi, -0.5 * np.pi, 0.0, 0.5 * np.pi], dtype=jnp.float32),
        ky_mode=np.asarray([2], dtype=np.int32),
    )

    assert _parallel_periods_from_grid(grid) == pytest.approx(1.0)
    kx, ky, kz = _cfl_wavenumber_arrays(grid)
    np.testing.assert_allclose(kx, [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(ky, [0.3])
    np.testing.assert_allclose(kz, [0.0, 1.0, 2.0, -1.0])

    assert _parallel_periods_from_grid(
        SimpleNamespace(z=jnp.asarray([0.0], dtype=jnp.float32))
    ) == pytest.approx(1.0)


def test_eta_geometry_and_ntft_helpers_match_manual_limits() -> None:
    assert _gradient_ratio_max(
        np.asarray([2.0, 4.0]), np.asarray([1.0, 0.0])
    ) == pytest.approx(1.0e6)
    assert _diagnostic_midplane_index(1) == 0
    assert _diagnostic_midplane_index(4) == 3

    cfg = CycloneBaseCase()
    grid = build_spectral_grid(
        replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None, non_twist=True)
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    theta = np.asarray(grid.z, dtype=float)
    cv_j, gb_j, cv0_j, gb0_j = geom.drift_coeffs(jnp.asarray(theta))
    bmag_j = geom.bmag(jnp.asarray(theta))

    maxima = _geometry_frequency_maxima(geom, theta)
    np.testing.assert_allclose(
        np.asarray(maxima),
        np.asarray(
            [
                np.max(np.abs(np.asarray(bmag_j))),
                np.max(np.abs(np.asarray(cv_j))),
                np.max(np.abs(np.asarray(gb_j))),
                np.max(np.abs(np.asarray(cv0_j))),
                np.max(np.abs(np.asarray(gb0_j))),
                float(geom.gradpar()),
            ]
        ),
    )

    m0_max, cv0_max, gb0_max = _non_twist_shift_frequency_max(
        geom, grid, ky_max=0.0, vpar_max=2.0, muB_max=1.5
    )
    assert m0_max == pytest.approx(0.0)
    assert cv0_max == pytest.approx(float(np.max(np.abs(np.asarray(cv0_j)))))
    assert gb0_max == pytest.approx(float(np.max(np.abs(np.asarray(gb0_j)))))


def _small_setup():
    cfg = CycloneBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), 0.2)
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        fprim=cfg.model.fprim,
        tprim=cfg.model.tprim_i,
        tprim_e=cfg.model.tprim_e,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
    )
    cache = build_linear_cache(grid, geom, params, 4, 4)
    return cfg, grid, geom, params, cache


def _multispecies_setup(*, Nl: int = 3, Nm: int = 4):
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(
        replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None)
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    common = dict(density=1.0, temperature=1.0, tprim=1.0, fprim=1.0)
    params = build_linear_params(
        [
            Species(charge=1.0, mass=1.0, **common),
            Species(charge=-1.0, mass=0.00027, **common),
        ],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    return cfg, grid, geom, params, cache, vol_fac, flux_fac


def _ramp_state(grid, Nl: int, Nm: int, *, g_offset: float):
    """Two-species ramp state ``G`` and a ramp ``phi`` on ``grid``."""

    shape = (2, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + g_offset), dtype=jnp.complex64)
    field_shape = (grid.ky.size, grid.kx.size, grid.z.size)
    field_base = np.arange(np.prod(field_shape), dtype=np.float32).reshape(field_shape)
    phi = jnp.asarray(field_base + 1.0j * (field_base + 1.0), dtype=jnp.complex64)
    return G, phi


def _small_G0():
    cfg, grid, geom, params, cache = _small_setup()
    G0 = _build_initial_condition(
        grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=cfg.init
    )
    return cfg, grid, geom, params, cache, G0


def _linear_fields(G, cache, params):
    _dG, fields_ = assemble_rhs_cached(G, cache, params, terms=LinearTerms())
    phi = fields_.phi
    apar = fields_.apar if fields_.apar is not None else jnp.zeros_like(phi)
    bpar = fields_.bpar if fields_.bpar is not None else jnp.zeros_like(phi)
    return phi, apar, bpar


def test_volume_and_flux_weights_are_finite_positive_and_normalized() -> None:
    _cfg, grid, geom, _params, _cache = _small_setup()

    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)

    assert np.all(np.isfinite(np.asarray(vol_fac)))
    assert np.all(np.isfinite(np.asarray(flux_fac)))
    assert np.all(np.asarray(vol_fac) > 0.0)
    assert np.all(np.asarray(flux_fac) > 0.0)
    np.testing.assert_allclose(np.asarray(vol_fac).sum(), 1.0, rtol=1.0e-7, atol=5.0e-7)
    np.testing.assert_allclose(
        np.asarray(flux_fac).sum(), 1.0, rtol=1.0e-7, atol=5.0e-7
    )


def test_resolved_energy_reductions_sum_to_scalar_totals() -> None:
    _cfg, grid, _geom, params, cache, vol_fac, _flux_fac = _multispecies_setup(
        Nl=3, Nm=4
    )
    shape = (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
    G = jnp.asarray(base + 1.0j * (0.25 * base + 1.0), dtype=jnp.complex64)
    field_base = np.arange(
        grid.ky.size * grid.kx.size * grid.z.size, dtype=np.float32
    ).reshape(grid.ky.size, grid.kx.size, grid.z.size)
    phi = jnp.asarray(field_base + 1.0j * (field_base + 0.5), dtype=jnp.complex64)
    apar = (0.2 - 0.1j) * phi

    k1, k12 = 1, (1, 2)
    cases = (
        (
            distribution_free_energy(G, grid, params, vol_fac, use_dealias=False),
            distribution_free_energy_resolved(
                G, grid, params, vol_fac, use_dealias=False
            ),
            (k1, k1, k12, k1, k12),
            1.0e-5,
        ),
        (
            electrostatic_field_energy(phi, cache, params, vol_fac, use_dealias=False),
            electrostatic_field_energy_resolved(
                phi, cache, params, vol_fac, use_dealias=False
            ),
            (k1, k1, k12, k1),
            1.0e-6,
        ),
        (
            magnetic_vector_potential_energy(apar, cache, vol_fac, use_dealias=False),
            magnetic_vector_potential_energy_resolved(
                apar, cache, vol_fac, nspecies=2, use_dealias=False
            ),
            (k1, k1, k12, k1),
            1.0e-6,
        ),
    )
    for total, (per_species, *spectra), axes, tol in cases:
        np.testing.assert_allclose(
            np.asarray(per_species).sum(), np.asarray(total), rtol=tol, atol=tol
        )
        for spectrum, axis in zip(spectra, axes, strict=True):
            np.testing.assert_allclose(
                np.asarray(spectrum).sum(axis=axis),
                np.asarray(per_species),
                rtol=tol,
                atol=tol,
            )

    phi2_t, phi2_kxt, phi2_kyt, phi2_kxkyt, phi2_zt, *_zonal = phi2_resolved(
        phi, grid, vol_fac, use_dealias=False
    )
    for spectrum, axes in (
        (phi2_kxt, 0),
        (phi2_kyt, 0),
        (phi2_kxkyt, (0, 1)),
        (phi2_zt, 0),
    ):
        np.testing.assert_allclose(
            np.asarray(spectrum).sum(axis=axes),
            np.asarray(phi2_t),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_diagnostics_mask_dealiased_nonfinite_modes_before_reduction() -> None:
    _cfg, grid, _geom, params, cache, vol_fac, flux_fac = _multispecies_setup(
        Nl=2, Nm=2
    )
    shape = (2, 2, 2, grid.ky.size, grid.kx.size, grid.z.size)
    field_shape = (grid.ky.size, grid.kx.size, grid.z.size)
    masked_indices = np.argwhere(~np.asarray(grid.dealias_mask, dtype=bool))
    assert masked_indices.size > 0
    ky_idx, kx_idx = masked_indices[0]

    def states(fill):
        """(G, phi, apar, bpar) with the first dealiased mode set to ``fill``."""

        G = np.ones(shape, dtype=np.complex64)
        G[:, :, :, ky_idx, kx_idx, :] = fill
        out = [jnp.asarray(G)]
        for value in (1.0 + 0.25j, 0.1 - 0.05j, 0.02 + 0.03j):
            field = np.ones(field_shape, dtype=np.complex64) * value
            field[ky_idx, kx_idx, :] = fill
            out.append(jnp.asarray(field))
        return tuple(out)

    dirty, clean = states(np.inf + 0.0j), states(0.0)
    reductions = (
        lambda G, phi, apar, bpar: distribution_free_energy(
            G, grid, params, vol_fac, use_dealias=True
        ),
        lambda G, phi, apar, bpar: distribution_free_energy_resolved(
            G, grid, params, vol_fac, use_dealias=True
        ),
        lambda G, phi, apar, bpar: electrostatic_field_energy(
            phi, cache, params, vol_fac, use_dealias=True
        ),
        lambda G, phi, apar, bpar: magnetic_vector_potential_energy(
            apar, cache, vol_fac, use_dealias=True
        ),
        lambda G, phi, apar, bpar: electrostatic_field_energy_resolved(
            phi, cache, params, vol_fac, use_dealias=True
        ),
        lambda G, phi, apar, bpar: magnetic_vector_potential_energy_resolved(
            apar, cache, vol_fac, nspecies=2, use_dealias=True
        ),
        lambda G, phi, apar, bpar: phi2_resolved(phi, grid, vol_fac),
        lambda G, phi, apar, bpar: heat_flux_channel_resolved_species(
            G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=True
        )[0],
        lambda G, phi, apar, bpar: heat_flux_species(
            G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=True
        ),
        lambda G, phi, apar, bpar: particle_flux_species(
            G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=True
        ),
        lambda G, phi, apar, bpar: turbulent_heating_species(
            G,
            0.9 * G,
            phi,
            apar,
            bpar,
            0.9 * phi,
            0.9 * apar,
            0.9 * bpar,
            cache,
            grid,
            params,
            vol_fac,
            0.1,
            use_dealias=True,
        ),
    )
    for reduce in reductions:
        got, expected = reduce(*dirty), reduce(*clean)
        if not isinstance(got, tuple):
            got, expected = (got,), (expected,)
        for got_arr, expected_arr in zip(got, expected, strict=True):
            assert np.all(np.isfinite(np.asarray(got_arr)))
            np.testing.assert_allclose(np.asarray(got_arr), np.asarray(expected_arr))


def test_zero_field_state_has_zero_transport_and_heating() -> None:
    _cfg, grid, _geom, params, cache, vol_fac, flux_fac = _multispecies_setup(
        Nl=3, Nm=4
    )
    G = jnp.zeros(
        (2, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    phi = jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    apar = jnp.zeros_like(phi)
    bpar = jnp.zeros_like(phi)
    flux = (G, phi, apar, bpar, cache, grid, params, flux_fac)
    heating = (
        G,
        G,
        phi,
        apar,
        bpar,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        vol_fac,
        0.125,
    )

    outputs = [
        heat_flux_species(*flux, use_dealias=False),
        heat_flux_total(*flux),
        *heat_flux_channel_species(*flux, use_dealias=False),
        particle_flux_species(*flux, use_dealias=False),
        particle_flux_total(*flux),
        *particle_flux_channel_species(*flux, use_dealias=False),
        turbulent_heating_species(*heating, use_dealias=False),
        turbulent_heating_total(*heating, use_dealias=False),
    ]
    for output in outputs:
        np.testing.assert_allclose(np.asarray(output), 0.0)


def test_energy_components_finite():
    _cfg, grid, geom, params, cache, G0 = _small_G0()
    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    phi, apar, bpar = _linear_fields(G0, cache, params)

    Wg = distribution_free_energy(G0, grid, params, vol_fac)
    Wphi = electrostatic_field_energy(phi, cache, params, vol_fac)
    Wapar = magnetic_vector_potential_energy(apar, cache, vol_fac)
    heat = heat_flux_total(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    pflux = particle_flux_total(G0, phi, apar, bpar, cache, grid, params, flux_fac)
    energy = total_energy(Wg, Wphi, Wapar)

    for value in (Wg, Wphi, Wapar, heat, pflux, energy):
        assert np.isfinite(np.asarray(value))
    assert energy == Wg + Wphi + Wapar


def test_fieldline_quadrature_weights_accept_sampled_geometry_contract():
    _cfg, grid, geom, _params, _cache = _small_setup()
    sampled = sample_flux_tube_geometry(geom, grid.z)

    vol_ref, flux_ref = fieldline_quadrature_weights(geom, grid)
    vol_s, flux_s = fieldline_quadrature_weights(sampled, grid)

    assert np.allclose(np.asarray(vol_s), np.asarray(vol_ref))
    assert np.allclose(np.asarray(flux_s), np.asarray(flux_ref))


def test_fieldline_quadrature_weights_trim_closed_sampled_geometry_contract():
    _cfg, grid, geom, _params, _cache = _small_setup()
    sampled = sample_flux_tube_geometry(geom, grid.z)
    profiles = (
        "bmag",
        "bgrad",
        "gds2",
        "gds21",
        "gds22",
        "cv",
        "gb",
        "cv0",
        "gb0",
        "jacobian",
        "grho",
    )
    closed = replace(
        sampled,
        theta=jnp.concatenate([sampled.theta, jnp.asarray([jnp.pi])]),
        theta_closed_interval=True,
        **{
            f"{name}_profile": jnp.concatenate(
                [
                    getattr(sampled, f"{name}_profile"),
                    getattr(sampled, f"{name}_profile")[:1],
                ]
            )
            for name in profiles
        },
    )

    vol_ref, flux_ref = fieldline_quadrature_weights(sampled, grid)
    vol_closed, flux_closed = fieldline_quadrature_weights(closed, grid)

    assert np.allclose(np.asarray(vol_closed), np.asarray(vol_ref))
    assert np.allclose(np.asarray(flux_closed), np.asarray(flux_ref))


def test_zonal_phi_mode_kxt_recovers_signed_zonal_average() -> None:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(replace(cfg.grid, Ny=8, Nx=4))
    geom = SAlphaGeometry.from_config(cfg.geometry)
    vol_fac, _flux_fac = fieldline_quadrature_weights(geom, grid)
    phi = jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    zonal_profile = (0.3 - 0.1j) + (0.2 + 0.05j) * jnp.cos(grid.z)
    phi = phi.at[0, 1, :].set(zonal_profile)

    out = zonal_phi_mode_kxt(phi, grid, vol_fac)
    out_line = zonal_phi_line_kxt(phi, grid)

    expected = jnp.sum(zonal_profile * vol_fac)
    expected_line = jnp.mean(zonal_profile)
    assert np.allclose(np.asarray(out[1]), np.asarray(expected))
    assert np.allclose(np.asarray(out_line[1]), np.asarray(expected_line))
    assert np.allclose(np.asarray(out[0]), 0.0)
    assert np.allclose(np.asarray(out_line[0]), 0.0)


def test_fieldline_quadrature_weights_use_grho_for_flux_weights():
    _cfg, grid, geom, _params, _cache = _small_setup()
    sampled = sample_flux_tube_geometry(geom, grid.z)
    sampled = replace(
        sampled,
        jacobian_profile=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        grho_profile=jnp.asarray([1.0, 2.0, 1.0, 2.0]),
        theta=jnp.asarray(grid.z[:4]),
    )
    grid_small = replace(grid, z=grid.z[:4])

    vol_fac, flux_fac = fieldline_quadrature_weights(sampled, grid_small)

    jac = np.array([1.0, 2.0, 3.0, 4.0])
    grho = np.array([1.0, 2.0, 1.0, 2.0])
    assert np.allclose(np.asarray(vol_fac), jac / np.sum(jac))
    assert np.allclose(np.asarray(flux_fac), jac / np.sum(jac * grho))


def test_standard_field_energies_match_geometry_weighted_formula():
    _cfg, grid, geom, params, cache = _small_setup()
    vol_fac, _flux_fac = fieldline_quadrature_weights(geom, grid)
    phi = jnp.ones((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    apar = (1.0 + 2.0j) * jnp.ones_like(phi)

    fac = np.where(np.asarray(grid.ky, dtype=float)[:, None] == 0.0, 1.0, 2.0)
    fac = fac * np.asarray(grid.dealias_mask, dtype=float)
    weight = fac[:, :, None] * np.asarray(vol_fac, dtype=float)[None, None, :]
    phi2 = np.abs(np.asarray(phi)) ** 2
    apar2 = np.abs(np.asarray(apar)) ** 2

    rho = np.atleast_1d(np.asarray(params.rho, dtype=float))
    wphi_expected = 0.0
    for rho_s in rho:
        b = np.asarray(cache.kperp2, dtype=float) * (rho_s * rho_s)
        wphi_expected += 0.5 * np.sum(phi2 * (1.0 - np.asarray(gamma0(b))) * weight)

    bmag2 = (
        np.asarray(cache.bmag, dtype=float)[None, None, :] ** 2
        if cache.kperp2_bmag
        else 1.0
    )
    wapar_expected = 0.5 * np.sum(
        apar2 * np.asarray(cache.kperp2, dtype=float) * bmag2 * weight
    )

    assert np.allclose(
        np.asarray(electrostatic_field_energy(phi, cache, params, vol_fac)),
        wphi_expected,
    )
    assert np.allclose(
        np.asarray(magnetic_vector_potential_energy(apar, cache, vol_fac)),
        wapar_expected,
    )


def test_reduce_scalar_and_species_kykxz_preserve_manual_sums() -> None:
    scalar = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    scalar_reduced = _reduce_scalar_kykxz(jnp.asarray(scalar))
    for got, axes in zip(
        scalar_reduced, ((0, 2), (1, 2), 2, (0, 1), None), strict=True
    ):
        np.testing.assert_allclose(np.asarray(got), scalar.sum(axis=axes))

    species = np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
    species_reduced = _reduce_species_kykxz(jnp.asarray(species))
    for got, axes in zip(
        species_reduced, ((1, 2, 3), (1, 3), (2, 3), 3, (1, 2)), strict=True
    ):
        np.testing.assert_allclose(np.asarray(got), species.sum(axis=axes))


def test_species_flux_sums_to_total():
    _cfg, grid, geom, params, cache, G0 = _small_G0()
    _vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    phi, apar, bpar = _linear_fields(G0, cache, params)
    flux = (G0, phi, apar, bpar, cache, grid, params, flux_fac)

    heat_s = heat_flux_species(*flux)
    pflux_s = particle_flux_species(*flux)

    assert heat_s.shape == (1,)
    assert pflux_s.shape == (1,)
    assert np.allclose(np.asarray(jnp.sum(heat_s)), np.asarray(heat_flux_total(*flux)))
    assert np.allclose(
        np.asarray(jnp.sum(pflux_s)), np.asarray(particle_flux_total(*flux))
    )


@pytest.mark.parametrize(
    ("contrib_fn", "resolved_fn", "Nm", "g_offset", "apar_scale", "bpar_scale"),
    [
        pytest.param(
            _heat_flux_channel_contrib_species,
            heat_flux_channel_resolved_species,
            4,
            1.0,
            0.3,
            -0.2,
            id="heat",
        ),
        pytest.param(
            _particle_flux_channel_contrib_species,
            particle_flux_channel_resolved_species,
            3,
            0.25,
            0.1,
            -0.3,
            id="particle",
        ),
    ],
)
def test_flux_channel_helper_matches_public_split_reductions(
    contrib_fn, resolved_fn, Nm, g_offset, apar_scale, bpar_scale
) -> None:
    _cfg, grid, _geom, params, cache, _vol_fac, flux_fac = _multispecies_setup(
        Nl=3, Nm=Nm
    )
    G, phi = _ramp_state(grid, 3, Nm, g_offset=g_offset)
    flux = (G, phi, apar_scale * phi, bpar_scale * phi, cache, grid, params, flux_fac)

    contribs = contrib_fn(*flux, use_dealias=False, flux_scale=1.0)
    resolved = resolved_fn(*flux, use_dealias=False)

    for contrib, reduced in zip(contribs, resolved, strict=True):
        expected = _reduce_species_kykxz(contrib)
        for got_arr, expected_arr in zip(reduced, expected, strict=True):
            np.testing.assert_allclose(
                np.asarray(got_arr), np.asarray(expected_arr), rtol=1.0e-6, atol=1.0e-6
            )


def test_particle_flux_total_channel_helper_single_species_short_circuits_to_zero() -> (
    None
):
    _cfg, grid, geom, params, cache, G = _small_G0()
    _vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    phi, apar, bpar = _linear_fields(G, cache, params)

    contribs = _particle_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=False,
        flux_scale=1.0,
    )

    for contrib in contribs:
        np.testing.assert_allclose(np.asarray(contrib), 0.0)


def test_jl_family_preserves_species_axis() -> None:
    cache = _multispecies_setup(Nl=3, Nm=3)[4]
    Jl, JlB, Jfac = _jl_family(cache)

    assert np.asarray(Jl).shape == np.asarray(cache.Jl).shape
    assert np.asarray(JlB).shape == np.asarray(cache.JlB).shape
    assert np.asarray(Jfac).shape == np.asarray(cache.Jl).shape
    assert np.allclose(np.asarray(Jl), np.asarray(cache.Jl))
    assert np.allclose(np.asarray(JlB), np.asarray(cache.JlB))


def test_particle_flux_species_matches_manual_multispecies_formula() -> None:
    _cfg, grid, _geom, params, cache, _vol_fac, flux_fac = _multispecies_setup(
        Nl=3, Nm=3
    )
    G, phi = _ramp_state(grid, 3, 3, g_offset=1.0)
    apar = 0.3 * phi
    bpar = -0.2 * phi

    got = np.asarray(
        particle_flux_species(
            G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=False
        )
    )

    fac = np.asarray(_transport_mode_weight(grid, use_dealias=False), dtype=np.float32)[
        :, :, None
    ]
    flx = np.asarray(flux_fac, dtype=np.float32)[None, None, :]
    ky = np.asarray(grid.ky, dtype=np.float32)[:, None, None]
    vphi = 1.0j * ky * np.asarray(phi)
    vapar = 1.0j * ky * np.asarray(apar)
    vbpar = 1.0j * ky * np.asarray(bpar)
    Jl = np.asarray(cache.Jl)
    JlB = np.asarray(cache.JlB)
    dens = np.asarray(params.density, dtype=np.float32)
    vth = np.asarray(params.vth, dtype=np.float32)
    tz = np.asarray(params.tz, dtype=np.float32)
    G_np = np.asarray(G)

    expected = []
    for s in range(2):
        G0 = G_np[s, :, 0, ...]
        G1 = G_np[s, :, 1, ...]
        n_bar = np.sum(Jl[s] * G0, axis=0)
        u_bar = np.sum(Jl[s] * G1, axis=0)
        uB_bar = np.sum(JlB[s] * G0, axis=0)
        fg = (
            np.conj(vphi) * n_bar
            - vth[s] * np.conj(vapar) * u_bar
            + tz[s] * np.conj(vbpar) * uB_bar
        )
        expected.append(np.sum((fg * 2.0 * flx * fac).real) * dens[s])

    assert np.allclose(got, np.asarray(expected), rtol=1.0e-6, atol=1.0e-6)


def test_flux_channel_splits_sum_to_total_multispecies() -> None:
    _cfg, grid, _geom, params, cache, _vol_fac, flux_fac = _multispecies_setup(
        Nl=3, Nm=4
    )
    G, phi = _ramp_state(grid, 3, 4, g_offset=1.0)
    flux = (G, phi, 0.3 * phi, -0.2 * phi, cache, grid, params, flux_fac)

    for total_fn, split_fn in (
        (heat_flux_species, heat_flux_channel_species),
        (particle_flux_species, particle_flux_channel_species),
    ):
        total = np.asarray(total_fn(*flux, use_dealias=False))
        es, apar_part, bpar_part = (
            np.asarray(arr) for arr in split_fn(*flux, use_dealias=False)
        )
        np.testing.assert_allclose(
            total, es + apar_part + bpar_part, rtol=1.0e-6, atol=1.0e-6
        )


def test_turbulent_heating_total_zero_for_steady_state() -> None:
    _cfg, grid, _geom, params, cache, vol_fac, _flux_fac = _multispecies_setup(
        Nl=3, Nm=4
    )
    G, phi = _ramp_state(grid, 3, 4, g_offset=0.5)
    apar = 0.2 * phi
    bpar = -0.1 * phi
    heating = (
        G,
        G,
        phi,
        apar,
        bpar,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        vol_fac,
        0.05,
    )

    heat_species = turbulent_heating_species(*heating, use_dealias=False)
    heat_total = turbulent_heating_total(*heating, use_dealias=False)

    np.testing.assert_allclose(np.asarray(heat_species), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(heat_total), 0.0, atol=1.0e-7)


def test_turbulent_heating_total_resolved_sums_to_species_total() -> None:
    _cfg, grid, _geom, params, cache, vol_fac, _flux_fac = _multispecies_setup(
        Nl=3, Nm=4
    )
    G_old, phi_old = _ramp_state(grid, 3, 4, g_offset=0.5)
    G = 1.03 * G_old + (0.02 - 0.01j)
    phi = 1.01 * phi_old + (0.03 + 0.02j)
    apar_old = 0.2 * phi_old
    apar = 0.2 * phi
    bpar_old = -0.1 * phi_old
    bpar = -0.1 * phi + (0.01 - 0.02j)
    heating = (
        G,
        G_old,
        phi,
        apar,
        bpar,
        phi_old,
        apar_old,
        bpar_old,
        cache,
        grid,
        params,
        vol_fac,
        0.05,
    )

    heat_species = turbulent_heating_species(*heating, use_dealias=False)
    heat_st, heat_kxst, heat_kyst, heat_kxkyst, heat_zst = (
        turbulent_heating_resolved_species(*heating, use_dealias=False)
    )

    for spectrum, axes in (
        (heat_st, None),
        (heat_kxst, 1),
        (heat_kyst, 1),
        (heat_kxkyst, (1, 2)),
        (heat_zst, 1),
    ):
        reduced = np.asarray(spectrum)
        if axes is not None:
            reduced = reduced.sum(axis=axes)
        np.testing.assert_allclose(
            reduced, np.asarray(heat_species), rtol=1.0e-5, atol=1.0e-6
        )
    assert np.max(np.abs(np.asarray(heat_species))) > 0.0


def test_turbulent_heating_is_weighted_by_density_times_charge() -> None:
    """GX weights the heating kernel by ``n_s Z_s`` (``sp.nz``), not ``n_s``.

    ``Q_s = Z_s n_s <h_s dchi/dt>`` is invariant under ``(Z, h) -> (-Z, -h)``,
    so a charge-mirrored copy of the ion carrying ``-G`` under the same fields
    must report the same heating (``bpar = 0``: its drive is not odd in ``Z``).
    """

    cfg = CycloneBaseCase()
    grid = build_spectral_grid(
        replace(cfg.grid, Nx=4, Ny=8, Nz=8, ntheta=None, nperiod=None)
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    ion = dict(mass=1.0, density=0.7, temperature=1.3, tprim=1.0, fprim=1.0)
    params = build_linear_params(
        [Species(charge=1.0, **ion), Species(charge=-1.0, **ion)],
        kpar_scale=float(geom.gradpar()),
    )
    cache = build_linear_cache(grid, geom, params, 3, 4)
    vol_fac, _flux_fac = fieldline_quadrature_weights(geom, grid)
    rng = np.random.default_rng(5)
    shape = (3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    G_ion = rng.normal(size=shape) + 1.0j * rng.normal(size=shape)
    G_ion_old = G_ion + 0.1 * (rng.normal(size=shape) + 1.0j * rng.normal(size=shape))
    phi = jnp.asarray(rng.normal(size=shape[2:]) + 1.0j * rng.normal(size=shape[2:]))
    apar = 0.3 * phi
    zero = jnp.zeros_like(phi)

    heat = np.asarray(
        turbulent_heating_species(
            jnp.asarray(np.stack([G_ion, -G_ion])),
            jnp.asarray(np.stack([G_ion_old, -G_ion_old])),
            phi,
            apar,
            zero,
            0.8 * phi,
            0.8 * apar,
            zero,
            cache,
            grid,
            params,
            vol_fac,
            0.05,
        )
    )

    assert abs(heat[0]) > 1.0e-3
    np.testing.assert_allclose(heat[1], heat[0], rtol=1.0e-10)


@pytest.mark.parametrize("ky_layout", ["full", "half"])
def test_netcdf_phi2_total_and_kx_spectrum_match_in_memory(ky_layout: str) -> None:
    """Published ``Phi2_t`` and ``Phi2_kxt`` carry the ``-ky`` partners.

    The per-row ``ky`` spectrum stores one row of each conjugate pair, so a
    plain sum of it is not the total: the published total and ``kx`` spectrum
    must agree with the in-memory reduction on either layout.
    """

    nx, ny, nz = 10, 12, 4
    grid = build_spectral_grid(
        GridConfig(Nx=nx, Ny=ny, Nz=nz, Lx=1.0, Ly=1.0, ky_layout=ky_layout)
    )
    rng = np.random.default_rng(11)
    real = rng.normal(size=(ny, nx, nz))
    phi_full = np.fft.fft2(real, axes=(0, 1)) / (nx * ny)
    phi = jnp.asarray(phi_full[: grid.ky.size])
    vol_fac = jnp.full((nz,), 1.0 / nz)
    phi2_t, phi2_kxt, phi2_kyt, phi2_kxkyt, *_rest = diagnostics_moments.phi2_resolved(
        phi, grid, vol_fac
    )
    resolved = SimpleNamespace(
        Phi2_kxkyt=np.asarray(phi2_kxkyt)[None],
        Phi2_kxt=np.asarray(phi2_kxt)[None],
        Phi2_kyt=np.asarray(phi2_kyt)[None],
    )
    out_t, out_kx, _out_ky, _out_kykx = nonlinear_netcdf._phi2_outputs_for_netcdf(
        resolved,
        SimpleNamespace(Wphi_t=np.zeros(1)),
        full_nx=nx,
        full_ny=ny,
        active_nx=_dealiased_kx_count(nx),
        active_ny=_dealiased_ky_count(ny),
    )

    np.testing.assert_allclose(out_t, [float(phi2_t)], rtol=1.0e-5)
    np.testing.assert_allclose(
        out_kx,
        np.asarray(phi2_kxt)[None][:, _dealiased_kx_indices(nx)],
        rtol=1.0e-5,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.sum(out_kx), float(phi2_t), rtol=1.0e-5)


def test_turbulent_heating_total_helper_zero_dt_guard_returns_zero_for_changed_state() -> (
    None
):
    _cfg, grid, _geom, params, cache, vol_fac, _flux_fac = _multispecies_setup(
        Nl=3, Nm=4
    )
    G_old, phi_old = _ramp_state(grid, 3, 4, g_offset=0.5)
    G = 1.02 * G_old + (0.03 - 0.01j)
    phi = 1.01 * phi_old + (0.02 + 0.01j)
    bpar = -0.1 * phi + (0.01 - 0.02j)

    contrib = _turbulent_heating_contrib_species(
        G,
        G_old,
        phi,
        0.2 * phi,
        bpar,
        phi_old,
        0.2 * phi_old,
        -0.1 * phi_old,
        cache,
        grid,
        params,
        vol_fac,
        0.0,
        use_dealias=False,
    )

    assert np.all(np.isfinite(np.asarray(contrib)))
    np.testing.assert_allclose(np.asarray(contrib), 0.0, atol=1.0e-7)


def test_init_all_scaling_matches_reference():
    cfg = CycloneBaseCase()
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = select_ky_index(np.asarray(grid_full.ky), float(grid_full.ky[1]))
    grid = select_ky_grid(grid_full, ky_index)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    init_cfg = InitializationConfig(
        init_field="all",
        init_amp=1.0,
        gaussian_init=False,
    )
    G0 = _build_initial_condition(
        grid, geom, ky_index=0, kx_index=0, Nl=4, Nm=4, init_cfg=init_cfg
    )
    base = 1.0 + 1.0j
    # density (l=0,m=0) should be unscaled
    assert np.allclose(G0[0, 0, 0, 0, 0], base)
    # tpar (l=0,m=2) scaled by 1/sqrt(2)
    assert np.allclose(G0[0, 2, 0, 0, 0], base / np.sqrt(2.0))
    # qpar (l=0,m=3) scaled by 1/sqrt(6)
    assert np.allclose(G0[0, 3, 0, 0, 0], base / np.sqrt(6.0))


def test_integrate_linear_explicit_diagnostics_shapes():
    _cfg, grid, geom, params, cache, G0 = _small_G0()
    time_cfg = ExplicitTimeConfig(dt=0.01, t_max=0.1, sample_stride=1, fixed_dt=True)

    t, phi_t, gamma_t, omega_t, diag = integrate_linear_explicit_diagnostics(
        G0, grid, cache, params, geom, time_cfg, terms=LinearTerms(), jit=False
    )
    n = t.shape[0]
    assert phi_t.shape[0] == gamma_t.shape[0] == omega_t.shape[0] == n
    for series in (
        diag.t,
        diag.Wg_t,
        diag.Wphi_t,
        diag.Wapar_t,
        diag.heat_flux_t,
        diag.particle_flux_t,
    ):
        assert series.shape[0] == n


def test_integrate_linear_explicit_diagnostics_honors_rk3_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _cfg, grid, geom, params, cache, G0 = _small_G0()
    calls: list[str] = []

    def _fake_step(G, cache, params, term_cfg, dt, *, method):
        calls.append(str(method))
        _dG, fields = assemble_rhs_cached(G, cache, params, terms=term_cfg)
        return G, fields

    monkeypatch.setattr(explicit_time_integrators, "_linear_explicit_step", _fake_step)

    time_cfg = ExplicitTimeConfig(
        dt=0.01, t_max=0.01, method="rk3", sample_stride=1, fixed_dt=True
    )
    integrate_linear_explicit_diagnostics(
        G0, grid, cache, params, geom, time_cfg, terms=LinearTerms(), jit=False
    )

    assert calls
    assert set(calls) == {"rk3"}


def test_term_config_and_rk3_wrapper_delegate_to_linear_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terms = LinearTerms(apar=0.0, bpar=1.0, hyperdiffusion=1.0)
    assert _linear_term_config(terms) == linear_terms_to_term_config(terms)

    captured: dict[str, object] = {}

    def _fake_step(G, cache, params, term_cfg, dt, *, method):
        captured.update(
            G=G, cache=cache, params=params, term_cfg=term_cfg, dt=dt, method=method
        )
        return G, FieldState(phi=G[0, 0])

    monkeypatch.setattr(explicit_time_integrators, "_linear_explicit_step", _fake_step)

    G0 = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.complex64)
    cache = SimpleNamespace()
    params = object()
    term_cfg = linear_terms_to_term_config(terms)

    G_next, fields = explicit_time_integrators._linear_explicit_step(
        G0, cache, params, term_cfg, 0.125, method="rk3"
    )

    for key, value in (
        ("G", G0),
        ("cache", cache),
        ("params", params),
        ("term_cfg", term_cfg),
    ):
        assert captured[key] is value
    assert captured["dt"] == pytest.approx(0.125)
    assert captured["method"] == "rk3"
    assert G_next is G0
    np.testing.assert_allclose(np.asarray(fields.phi), np.asarray(G0[0, 0]))


def test_linear_explicit_step_applies_completed_step_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SimpleNamespace(
        dealias_mask=jnp.asarray([[True, True], [True, False]]),
        ky=jnp.asarray([0.0, 0.2]),
        kx=jnp.asarray([0.0, 0.1]),
    )
    G0 = jnp.ones((1, 1, 2, 2, 1), dtype=jnp.complex64)
    seen_states: list[np.ndarray] = []

    def _fake_assemble_rhs(state, _cache, _params, *, terms, dt=None):
        seen_states.append(np.asarray(state))
        return jnp.zeros_like(state), FieldState(phi=state[0, 0])

    monkeypatch.setattr(
        explicit_time_integrators, "assemble_rhs_cached", _fake_assemble_rhs
    )

    G_next, fields = explicit_time_integrators._linear_explicit_step(
        G0,
        cache,
        object(),
        linear_terms_to_term_config(LinearTerms()),
        0.1,
        method="euler",
    )

    expected = np.ones((2, 2, 1), dtype=np.complex64)
    expected[0, 0, 0] = 0.0
    expected[1, 1, 0] = 0.0

    assert len(seen_states) == 2
    assert np.allclose(seen_states[0][0, 0], 1.0)
    assert np.allclose(np.asarray(G_next[0, 0]), expected)
    assert np.allclose(np.asarray(seen_states[-1][0, 0]), expected)
    assert np.allclose(np.asarray(fields.phi), expected)


def _only_terms(**on) -> LinearTerms:
    """LinearTerms with every listed channel off except the ones in ``on``."""

    names = (
        "streaming",
        "mirror",
        "curvature",
        "gradb",
        "diamagnetic",
        "collisions",
        "hypercollisions",
        "end_damping",
        "apar",
        "bpar",
    )
    return LinearTerms(**{name: on.get(name, 0.0) for name in names})


def test_energy_drift_small_no_drive():
    cfg, grid, geom, params, cache, G0 = _small_G0()
    params = replace(params, fprim=0.0, tprim=0.0, tprim_e=0.0, nu=0.0)
    time_cfg = ExplicitTimeConfig(dt=0.01, t_max=0.2, sample_stride=1, fixed_dt=True)

    _, _, _, _, diag = integrate_linear_explicit_diagnostics(
        G0,
        grid,
        cache,
        params,
        geom,
        time_cfg,
        terms=_only_terms(streaming=1.0),
        jit=False,
    )
    energy = np.asarray(diag.energy_t)
    assert np.all(np.isfinite(energy))
    if energy.size > 1:
        rel = np.abs((energy[-1] - energy[0]) / max(abs(energy[0]), 1.0e-12))
        assert rel < 0.05


def test_growth_rate_step_matches_real_imag_validity_mask():
    """Instantaneous growth-rate kernel should require non-zero real and imaginary parts."""

    phi_prev = jnp.asarray([[[1.0 + 1.0j, 1.0 + 1.0j]]], dtype=jnp.complex64)
    phi_now_invalid = jnp.asarray([[[2.0 + 0.0j, 2.0 + 0.0j]]], dtype=jnp.complex64)
    mask = jnp.asarray([[True]])
    gamma_bad, omega_bad = _instantaneous_growth_rate_step(
        phi_now_invalid, phi_prev, 0.1, z_index=0, mask=mask
    )
    assert np.allclose(np.asarray(gamma_bad), 0.0)
    assert np.allclose(np.asarray(omega_bad), 0.0)

    phi_now_valid = jnp.asarray([[[2.0 + 2.0j, 2.0 + 2.0j]]], dtype=jnp.complex64)
    gamma_ok, omega_ok = _instantaneous_growth_rate_step(
        phi_now_valid, phi_prev, 0.1, z_index=0, mask=mask
    )
    assert np.isfinite(np.asarray(gamma_ok)).all()
    assert np.isfinite(np.asarray(omega_ok)).all()
    assert not np.allclose(np.asarray(gamma_ok), 0.0)


def test_growth_rate_step_validity_depends_on_current_phi_only():
    """Instantaneous kernel checks real/imag nonzero on current phi only."""

    phi_prev = jnp.asarray([[[1.0 + 0.0j, 1.0 + 0.0j]]], dtype=jnp.complex64)
    phi_now = jnp.asarray([[[2.0 + 2.0j, 2.0 + 2.0j]]], dtype=jnp.complex64)
    mask = jnp.asarray([[True]])
    gamma, omega = _instantaneous_growth_rate_step(
        phi_now, phi_prev, 0.1, z_index=0, mask=mask
    )
    assert np.isfinite(np.asarray(gamma)).all()
    assert np.isfinite(np.asarray(omega)).all()
    assert not np.allclose(np.asarray(gamma), 0.0)


def test_growth_rate_step_max_uses_per_step_peak():
    """Max-mode diagnostics should follow each step's peak-z sample."""

    phi_prev = jnp.asarray([[[1.0 + 1.0j, 8.0 + 8.0j]]], dtype=jnp.complex64)
    phi_now = jnp.asarray([[[9.0 + 9.0j, 2.0 + 2.0j]]], dtype=jnp.complex64)
    mask = jnp.asarray([[True]])
    gamma, omega = _instantaneous_growth_rate_step(
        phi_now,
        phi_prev,
        0.1,
        z_index=0,
        mask=mask,
        mode_method="max",
    )
    ratio = (9.0 + 9.0j) / (8.0 + 8.0j)
    assert np.allclose(np.asarray(gamma), np.log(np.abs(ratio)) / 0.1)
    assert np.allclose(np.asarray(omega), -np.angle(ratio) / 0.1)


@pytest.mark.parametrize(("ky", "expected"), [(-0.01, True), (0.0, False)])
def test_growth_mask_promotes_only_a_single_selected_nonzonal_slice(
    ky: float, expected: bool
) -> None:
    mask = _growth_rate_mode_mask(
        jnp.asarray([ky]), jnp.asarray([0.0]), jnp.asarray([[False]])
    )
    assert np.asarray(mask).item() is expected


def test_rk4_step_uses_runtime_scaled_end_damping_once() -> None:
    _cfg, _grid, _geom, params, cache, G0 = _small_G0()
    params = replace(params, damp_ends_amp=0.5)
    term_cfg = linear_terms_to_term_config(_only_terms(end_damping=1.0))
    dt = 0.2

    G_step, fields_step = explicit_time_integrators._linear_explicit_step(
        G0, cache, params, term_cfg, dt, method="rk4"
    )

    def rhs(state: jnp.ndarray) -> jnp.ndarray:
        dG, _fields = assemble_rhs_cached(state, cache, params, terms=term_cfg)
        return dG

    k1 = rhs(G0)
    k2 = rhs(G0 + 0.5 * dt * k1)
    k3 = rhs(G0 + 0.5 * dt * k2)
    k4 = rhs(G0 + dt * k3)
    G_manual = G0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    _dG_manual, fields_manual = assemble_rhs_cached(
        G_manual, cache, params, terms=term_cfg
    )

    assert np.allclose(
        np.asarray(G_step), np.asarray(G_manual), rtol=1.0e-6, atol=1.0e-6
    )
    assert np.allclose(
        np.asarray(fields_step.phi),
        np.asarray(fields_manual.phi),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_linear_explicit_adaptive_default_dt_max_clamps_to_nominal_dt():
    """When dt_max is unset, the adaptive explicit path should clamp to dt."""

    _cfg, grid, geom, params, cache, G0 = _small_G0()
    time_cfg = ExplicitTimeConfig(
        dt=0.01, t_max=0.05, sample_stride=1, fixed_dt=False, dt_max=None, cfl=10.0
    )
    *_series, diag = integrate_linear_explicit_diagnostics(
        G0, grid, cache, params, geom, time_cfg, terms=LinearTerms(), jit=False
    )
    dt_t = np.asarray(diag.dt_t, dtype=float)
    assert dt_t.size > 0
    assert np.nanmax(dt_t) <= float(time_cfg.dt) + 1.0e-12


def test_linear_omega_max_preserves_selected_ky_mode():
    _cfg, grid, geom, params, _cache = _small_setup()

    omega_sel = _linear_frequency_bound(grid, geom, params, 4, 4)

    assert omega_sel[1] > 0.0


# ---- from test_restart.py ----
# Tests for NetCDF restart-state IO helpers.


_RESTART_DIMS = ("Nspecies", "Nm", "Nl", "Nz", "Nkx", "Nky", "ri")


def _write_restart_G(path: Path, data: np.ndarray) -> None:
    with Dataset(path, "w") as root:
        for name, size in zip(_RESTART_DIMS, data.shape, strict=True):
            root.createDimension(name, size)
        root.createVariable("G", "f4", _RESTART_DIMS)[:] = data


def test_load_netcdf_restart_state_accepts_full_ky_reduced_kx_layout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gx.restart.nc"
    data = np.zeros((1, 2, 2, 3, 3, 4, 2), dtype=np.float32)
    data[0, 0, 0, 0, 0, 1, 0] = 1.5
    data[0, 0, 0, 1, 2, 3, 1] = -0.25
    _write_restart_G(path, data)

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
    def load(path):
        return load_netcdf_restart_state(path, nspecies=1, Nl=1, Nm=1, ny=4, nx=4, nz=1)

    missing_g = tmp_path / "missing_g.restart.nc"
    Dataset(missing_g, "w").close()
    with pytest.raises(ValueError, match="does not contain variable"):
        load(missing_g)

    bad_shape = tmp_path / "bad_shape.restart.nc"
    with Dataset(bad_shape, "w") as root:
        root.createDimension("x", 2)
        root.createVariable("G", "f4", ("x",))[:] = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match="unexpected NetCDF restart G shape"):
        load(bad_shape)

    for name, shape, match in (
        ("shape_mismatch", (2, 1, 1, 1, 3, 2, 2), "does not match requested"),
        ("nz_mismatch", (1, 1, 1, 2, 3, 2, 2), "restart Nz"),
    ):
        path = tmp_path / f"{name}.restart.nc"
        _write_restart_G(path, np.zeros(shape, dtype=np.float32))
        with pytest.raises(ValueError, match=match):
            load(path)


def test_restart_reader_rejects_unsupported_future_schema(tmp_path: Path) -> None:
    path = tmp_path / "future.restart.nc"
    with Dataset(path, "w") as root:
        root.setncattr("schema_version", 2)
    with pytest.raises(ValueError, match="unsupported GKX NetCDF schema_version 2"):
        load_netcdf_restart_state(path, nspecies=1, Nl=1, Nm=1, ny=1, nx=1, nz=1)


# --- Q29: the saved summary carries the solver status the result carries ------


def test_saved_linear_summary_carries_the_result_solver_status(
    tmp_path: Path,
) -> None:
    """A run read back from disk must have the same convergence channel."""

    from gkx.solvers_linear_implicit import ImplicitSolveSummary
    from gkx.solvers_linear_krylov import EigenSolveStatus

    result = _linear_result(
        eigen_status=EigenSolveStatus(
            method="adaptive",
            route="shift_invert",
            residual=4.5e-9,
            tolerance=1.0e-7,
            certified=True,
            inner={"converged": True},
        ),
        implicit_solve=ImplicitSolveSummary(
            max_relative_residual=2.5e-9,
            max_iterations=4,
            solves=40,
            unconverged_solves=0,
        ),
    )

    summary = _summary(write_runtime_linear_artifacts(tmp_path / "linear_run", result))

    assert summary["eigen_route"] == "shift_invert"
    assert summary["eigen_residual"] == pytest.approx(4.5e-9)
    assert summary["eigen_tolerance"] == pytest.approx(1.0e-7)
    assert summary["eigen_certified"] is True
    assert summary["eigen_inner_converged"] is True
    assert summary["implicit_converged"] is True
    assert summary["implicit_max_relative_residual"] == pytest.approx(2.5e-9)
    assert summary["implicit_max_iterations"] == 4
    assert summary["implicit_unconverged_solves"] == 0
    # The saved keys are the result's own keys, not a second spelling of them.
    reported = result.summary()
    for key, value in summary.items():
        if key in reported:
            assert value == reported[key]


def test_saved_linear_summary_reports_no_status_as_null(tmp_path: Path) -> None:
    """An explicit-time run without an implicit solve saves ``None``, not a fake."""

    paths = write_runtime_linear_artifacts(tmp_path / "linear_run", _linear_result())
    summary = _summary(paths)

    for key in (
        "eigen_route",
        "eigen_residual",
        "eigen_certified",
        "implicit_converged",
        "implicit_unconverged_solves",
    ):
        assert key in summary and summary[key] is None


def test_saved_nonlinear_summary_carries_the_imex_solve_status() -> None:
    from gkx.solvers_linear_implicit import ImplicitSolveSummary

    def summary_of(**extra):
        return _nonlinear_summary(
            SimpleNamespace(
                diagnostics=None,
                state=None,
                ky_selected=0.2,
                kx_selected=0.0,
                phi2=np.asarray(7.0),
                **extra,
            )
        )

    summary = summary_of(
        implicit_solve=ImplicitSolveSummary(
            max_relative_residual=9.5e-7,
            max_iterations=11,
            solves=8,
            unconverged_solves=0,
        )
    )
    assert summary["implicit_converged"] is True
    assert summary["implicit_max_relative_residual"] == pytest.approx(9.5e-7)
    assert summary["implicit_max_iterations"] == 11
    assert summary["implicit_unconverged_solves"] == 0
    assert summary_of()["implicit_converged"] is None


def test_artifact_checkpoint_helpers_cover_their_early_returns() -> None:
    from types import SimpleNamespace

    from gkx.workflows.runtime import artifacts as art

    assert "no measurable window (no_window)" in art._format_saturation_summary(
        {"mean": None}
    )
    assert "drift" in art._format_saturation_summary(
        {"mean": None, "reasons": ["drift"]}
    )
    assert art._next_runtime_chunk_steps(remaining_steps=7, checkpoint_steps=None) == 7
    assert art._next_runtime_chunk_steps(remaining_steps=None, checkpoint_steps=5) == 5
    assert art._next_runtime_chunk_steps(remaining_steps=9, checkpoint_steps=5) == 5
    cfg = object()
    assert art._advance_restart_run_config(cfg, None) is cfg
    chunk = SimpleNamespace(diagnostics=None)
    assert art._merge_chunk_diagnostics(
        chunk, cumulative_diag=None, time_offset=1.5, history_from_file=False
    )[1:] == (None, 1.5)
    policy = SimpleNamespace(checkpoint_steps=None)
    assert art._checkpoint_loop_done(
        policy=policy,
        result_effective=chunk,
        remaining_steps=None,
        time_offset=0.0,
        cfg=cfg,
    )
