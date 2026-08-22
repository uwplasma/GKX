"""Unit contracts: differentiable geometry objectives."""

from __future__ import annotations

import inspect

# ---- test_differentiable_geometry_bridge.py ----

import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gkx
import gkx.geometry.autodiff_checks as diff_autodiff
import gkx.geometry.backend_discovery as backend_discovery
import gkx.geometry.booz_xform_bridge as booz_bridge
import gkx.geometry.differentiable as diff_geom
import gkx.geometry.flux_tube_contract as geom_contract
import gkx.geometry.numerics as geom_numerics
import gkx.geometry.sensitivity as geom_sensitivity
import gkx.geometry.vmec_boozer_core as vmec_boozer_core
import gkx.geometry.vmec_boozer_constants as vmec_boozer_constants
import gkx.geometry.vmec_boozer_derivatives as vmec_boozer_derivatives
import gkx.geometry.vmec_field_line_sampling as vmec_field_line_sampling
import gkx.geometry.vmec_flux_tube_reports as vmec_flux_tube_reports
import gkx.geometry.vmec_state_controls as vmec_state_controls
import gkx.geometry.vmec_state_sensitivity as vmec_state_sensitivity
import gkx.geometry.vmec_tensor_mapping as vmec_tensor_mapping
from gkx.geometry.differentiable import (
    _array_parity_metrics,
    _boozer_half_mesh_s_grid,
    _candidate_paths,
    _cumulative_trapezoid,
    _find_importable_module,
    _interp_equal_arc_profile,
    _interp_radial,
    _is_traced,
    _periodic_bilinear_sample_2d,
    _radial_derivative_array,
    _radial_derivative_profile,
    _scalar_parity_metrics,
    booz_xform_flux_tube_sensitivity_report,
    booz_xform_spectral_sensitivity_report,
    discover_differentiable_geometry_backends,
    finite_difference_jacobian,
    flux_tube_geometry_from_mapping,
    flux_tube_geometry_from_vmec_boozer_state,
    flux_tube_geometry_observables,
    geometry_inverse_design_report,
    geometry_observable_names,
    geometry_sensitivity_report,
    vmex_boozer_equal_arc_core_profiles_from_state,
    vmex_boozer_flux_tube_sensitivity_report,
    vmex_field_line_tensor_sensitivity_report,
    vmex_flux_tube_array_parity_report,
    vmex_flux_tube_sensitivity_report,
    vmex_metric_tensor_sensitivity_report,
    vmec_boundary_aspect_sensitivity_report,
    vmec_field_line_tensor_observable_names,
    vmec_metric_tensor_observable_names,
)


def _sample_mapping() -> dict[str, object]:
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    ones = np.ones_like(theta)
    zeros = np.zeros_like(theta)
    return {
        "theta": theta,
        "gradpar": 0.7 * ones,
        "bmag": 1.0 + 0.05 * np.cos(theta),
        "bgrad": 0.05 * np.sin(theta),
        "gds2": ones,
        "gds21": zeros,
        "gds22": ones,
        "cvdrift": 0.2 * np.cos(theta),
        "gbdrift": 0.2 * np.cos(theta),
        "cvdrift0": zeros,
        "gbdrift0": zeros,
        "jacobian": ones,
        "grho": ones,
        "q": 1.7,
        "s_hat": 0.4,
        "R0": 1.5,
        "nfp": 5,
    }


def test_differentiable_geometry_facade_preserves_split_symbol_identity() -> None:
    """The public geometry bridge remains a stable facade."""

    assert diff_geom._candidate_paths is backend_discovery._candidate_paths
    assert (
        diff_geom._find_importable_module is backend_discovery._find_importable_module
    )
    assert diff_geom._is_traced is backend_discovery._is_traced
    assert (
        diff_geom.discover_differentiable_geometry_backends
        is backend_discovery.discover_differentiable_geometry_backends
    )
    assert (
        diff_geom.finite_difference_jacobian is diff_autodiff.finite_difference_jacobian
    )
    assert (
        diff_geom.observable_gradient_validation_report
        is diff_autodiff.observable_gradient_validation_report
    )
    assert diff_geom._array_parity_metrics is geom_numerics._array_parity_metrics
    assert diff_geom._scalar_parity_metrics is geom_numerics._scalar_parity_metrics
    assert diff_geom._interp_radial is geom_numerics._interp_radial
    assert (
        diff_geom._interp_equal_arc_profile is geom_numerics._interp_equal_arc_profile
    )
    assert diff_geom._boozer_half_mesh_s_grid is geom_numerics._boozer_half_mesh_s_grid
    assert (
        diff_geom._radial_derivative_profile is geom_numerics._radial_derivative_profile
    )
    assert diff_geom._radial_derivative_array is geom_numerics._radial_derivative_array
    assert diff_geom._cumulative_trapezoid is geom_numerics._cumulative_trapezoid
    assert (
        diff_geom._periodic_bilinear_sample_2d
        is geom_numerics._periodic_bilinear_sample_2d
    )
    assert diff_geom._array is geom_contract._array
    assert diff_geom._scalar is geom_contract._scalar
    assert (
        diff_geom.flux_tube_geometry_from_mapping
        is geom_contract.flux_tube_geometry_from_mapping
    )
    assert (
        diff_geom.flux_tube_geometry_observables
        is geom_contract.flux_tube_geometry_observables
    )
    assert (
        diff_geom.geometry_observable_names is geom_contract.geometry_observable_names
    )
    assert (
        diff_geom.vmec_metric_tensor_observable_names
        is geom_contract.vmec_metric_tensor_observable_names
    )
    assert callable(diff_geom.vmec_boundary_aspect_sensitivity_report)
    assert callable(diff_geom.booz_xform_spectral_sensitivity_report)
    assert callable(diff_geom.booz_xform_flux_tube_mapping_from_inputs)
    assert callable(diff_geom.booz_xform_flux_tube_sensitivity_report)
    assert (
        diff_geom.evaluate_boozer_bmag_on_field_line
        is booz_bridge.evaluate_boozer_bmag_on_field_line
    )
    assert (
        diff_geom.vmec_boundary_aspect_sensitivity_report
        is not booz_bridge.vmec_boundary_aspect_sensitivity_report
    )
    assert (
        diff_geom.booz_xform_spectral_sensitivity_report
        is not booz_bridge.booz_xform_spectral_sensitivity_report
    )
    assert callable(diff_geom.vmex_boozer_flux_tube_sensitivity_report)
    assert callable(diff_geom.vmex_metric_tensor_sensitivity_report)
    assert callable(diff_geom.vmex_field_line_tensor_sensitivity_report)
    assert (
        diff_geom.vmex_boozer_flux_tube_sensitivity_report
        is not vmec_state_sensitivity.vmex_boozer_flux_tube_sensitivity_report
    )
    assert (
        diff_geom.vmex_metric_tensor_sensitivity_report
        is not vmec_state_sensitivity.vmex_metric_tensor_sensitivity_report
    )
    assert (
        diff_geom.vmex_field_line_tensor_sensitivity_report
        is not vmec_state_sensitivity.vmex_field_line_tensor_sensitivity_report
    )
    assert callable(diff_geom.vmex_flux_tube_mapping_from_state)
    assert (
        diff_geom.vmex_flux_tube_mapping_from_state
        is not vmec_tensor_mapping.vmex_flux_tube_mapping_from_state
    )
    assert callable(diff_geom.prewarm_vmec_boozer_equal_arc_cache)
    assert callable(diff_geom.vmex_boozer_equal_arc_core_profiles_from_state)
    assert (
        diff_geom.prewarm_vmec_boozer_equal_arc_cache
        is not vmec_boozer_core.prewarm_vmec_boozer_equal_arc_cache
    )
    assert vmec_boozer_core.prewarm_vmec_boozer_equal_arc_cache is (
        vmec_boozer_constants.prewarm_vmec_boozer_equal_arc_cache
    )
    assert vmec_boozer_core._cached_booz_xform_constants is (
        vmec_boozer_constants._cached_booz_xform_constants
    )
    assert (
        diff_geom.vmex_boozer_equal_arc_core_profiles_from_state
        is not vmec_boozer_core.vmex_boozer_equal_arc_core_profiles_from_state
    )
    assert callable(diff_geom.vmex_flux_tube_sensitivity_report)
    assert callable(diff_geom.vmex_flux_tube_array_parity_report)
    assert (
        diff_geom.vmex_flux_tube_sensitivity_report
        is not vmec_flux_tube_reports.vmex_flux_tube_sensitivity_report
    )
    assert (
        diff_geom.vmex_flux_tube_array_parity_report
        is not vmec_flux_tube_reports.vmex_flux_tube_array_parity_report
    )
    assert (
        diff_geom.vmec_field_line_tensor_observable_names
        is geom_contract.vmec_field_line_tensor_observable_names
    )
    assert (
        diff_geom.geometry_sensitivity_report
        is geom_sensitivity.geometry_sensitivity_report
    )
    assert (
        diff_geom.geometry_inverse_design_report
        is geom_sensitivity.geometry_inverse_design_report
    )


def test_differentiable_geometry_patch_context_restores_module_attrs() -> None:
    module = types.SimpleNamespace(first="original-first", second="original-second")

    with diff_geom._patched_module_attrs(
        module, {"first": "patched-first", "second": "patched-second"}
    ):
        assert module.first == "patched-first"
        assert module.second == "patched-second"

    assert module.first == "original-first"
    assert module.second == "original-second"

    with pytest.raises(RuntimeError, match="forced"):
        with diff_geom._patched_module_attrs(module, {"first": "patched-again"}):
            assert module.first == "patched-again"
            raise RuntimeError("forced")

    assert module.first == "original-first"
    assert module.second == "original-second"


def test_flux_tube_geometry_from_mapping_builds_solver_contract() -> None:
    assert gkx.flux_tube_geometry_from_mapping is flux_tube_geometry_from_mapping
    assert (
        gkx.flux_tube_geometry_from_vmec_boozer_state
        is flux_tube_geometry_from_vmec_boozer_state
    )
    assert gkx.geometry_observable_names is geometry_observable_names
    assert gkx.flux_tube_geometry_observables is flux_tube_geometry_observables
    geom = flux_tube_geometry_from_mapping(
        _sample_mapping(), source_model="vmex:test"
    )

    assert geom.source_model == "vmex:test"
    assert geom.theta.shape == (8,)
    assert geom.nfp == 5
    assert geom.gradpar() == pytest.approx(0.7)
    assert np.allclose(
        np.asarray(geom.bmag(jnp.asarray(geom.theta))), np.asarray(geom.bmag_profile)
    )
    observables = np.asarray(flux_tube_geometry_observables(geom))
    assert observables.shape == (len(geometry_observable_names()),)
    assert np.all(np.isfinite(observables))


def test_equal_arc_interpolation_keeps_value_gradients_finite() -> None:
    theta_uniform = jnp.linspace(-jnp.pi, jnp.pi, 9)
    theta_base = jnp.linspace(-jnp.pi, jnp.pi, 9)

    def remapped_mean(scale: jnp.ndarray) -> jnp.ndarray:
        theta_equal_arc = theta_base + 0.05 * scale * jnp.sin(theta_base)
        values = (1.0 + scale) * jnp.cos(theta_base)
        return jnp.mean(
            _interp_equal_arc_profile(theta_uniform, theta_equal_arc, values)
        )

    scale = jnp.asarray(0.2)
    step = jnp.asarray(1.0e-3)
    grad = jax.grad(remapped_mean)(scale)
    fd = (remapped_mean(scale + step) - remapped_mean(scale - step)) / (2.0 * step)

    assert np.isfinite(float(grad))
    assert float(grad) == pytest.approx(float(fd), rel=2.0e-3, abs=2.0e-5)


def test_boozer_field_line_derivative_helpers_match_circular_surface() -> None:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 9)
    out = {"ixm_b": jnp.asarray([0, 1]), "ixn_b": jnp.asarray([0, 0])}
    r0, r1, z1 = 2.0, 0.2, 0.3

    spectral = vmec_boozer_derivatives.evaluate_boozer_field_line_derivatives(
        out,
        theta_closed=theta,
        alpha=0.0,
        iota_safe=jnp.asarray(1.0),
        base_dtype=theta.dtype,
        bmnc_b=jnp.asarray([1.0, 0.1]),
        d_bmnc_b_d_s=jnp.asarray([0.01, 0.02]),
        rmnc_b=jnp.asarray([r0, r1]),
        d_rmnc_b_d_s=jnp.asarray([0.05, 0.01]),
        zmns_b=jnp.asarray([0.0, z1]),
        d_zmns_b_d_s=jnp.asarray([0.0, 0.02]),
        numns_b=jnp.asarray([0.0, 0.0]),
        d_numns_b_d_s=jnp.asarray([0.0, 0.0]),
    )

    expected_r = r0 + r1 * jnp.cos(theta)
    np.testing.assert_allclose(np.asarray(spectral.r_b), np.asarray(expected_r))
    np.testing.assert_allclose(
        np.asarray(spectral.d_r_b_d_theta),
        np.asarray(-r1 * jnp.sin(theta)),
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(spectral.d_z_b_d_theta),
        np.asarray(z1 * jnp.cos(theta)),
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(spectral.d_mod_b_d_phi), np.zeros(theta.shape), atol=2.0e-7
    )

    cartesian = vmec_boozer_derivatives.boozer_cartesian_derivatives(spectral)
    np.testing.assert_allclose(
        np.asarray(cartesian.d_x_d_theta),
        np.asarray(-r1 * jnp.sin(theta) * jnp.cos(theta)),
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(cartesian.d_x_d_phi),
        np.asarray(-expected_r * jnp.sin(theta)),
        atol=2.0e-7,
    )

    gradients = vmec_boozer_derivatives.boozer_coordinate_gradients(
        spectral=spectral,
        cartesian=cartesian,
        sqrt_g_booz=jnp.ones_like(theta),
        etf_safe=jnp.asarray(2.0),
    )
    np.testing.assert_allclose(
        np.asarray(gradients.grad_psi_z),
        np.asarray(
            cartesian.d_x_d_theta * cartesian.d_y_d_phi
            - cartesian.d_y_d_theta * cartesian.d_x_d_phi
        ),
        atol=2.0e-7,
    )


def test_flux_tube_geometry_from_vmec_boozer_state_wraps_in_memory_bridge(
    monkeypatch,
) -> None:
    assert inspect.signature(flux_tube_geometry_from_vmec_boozer_state) == (
        inspect.signature(vmec_boozer_core.flux_tube_geometry_from_vmec_boozer_state)
    )
    assert getattr(flux_tube_geometry_from_vmec_boozer_state, "__wrapped__") is (
        vmec_boozer_core.flux_tube_geometry_from_vmec_boozer_state
    )
    calls: list[dict[str, object]] = []

    def fake_core_profiles(state, static, indata, wout, **kwargs):  # noqa: ANN001, ANN202
        calls.append(
            {
                "state": state,
                "static": static,
                "indata": indata,
                "wout": wout,
                **kwargs,
            }
        )
        return _sample_mapping()

    monkeypatch.setattr(
        diff_geom,
        "vmex_boozer_equal_arc_core_profiles_from_state",
        fake_core_profiles,
    )

    geom = flux_tube_geometry_from_vmec_boozer_state(
        "state",
        "static",
        "indata",
        "wout",
        surface_index=3,
        torflux=0.42,
        alpha=0.25,
        ntheta=16,
        mboz=21,
        nboz=23,
        jit=True,
        surface_stencil_width=5,
        reference_length=1.7,
        reference_b=2.3,
        source_model="unit-test-vmec-boozer",
    )

    assert geom.source_model == "unit-test-vmec-boozer"
    assert geom.theta.shape == (8,)
    assert len(calls) == 1
    assert calls[0]["surface_index"] == 3
    assert calls[0]["torflux"] == 0.42
    assert calls[0]["alpha"] == 0.25
    assert calls[0]["ntheta"] == 16
    assert calls[0]["mboz"] == 21
    assert calls[0]["nboz"] == 23
    assert calls[0]["jit"] is True
    assert calls[0]["surface_stencil_width"] == 5
    assert calls[0]["reference_length"] == 1.7
    assert calls[0]["reference_b"] == 2.3


def test_flux_tube_geometry_from_mapping_rejects_bad_contracts() -> None:
    bad = _sample_mapping()
    bad.pop("bmag")
    with pytest.raises(ValueError, match="bmag"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bad["gds2"] = np.ones((2, 2))
    with pytest.raises(ValueError, match="gds2"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    gradpar = np.asarray(bad["gradpar"]).copy()
    gradpar[-1] = 0.9
    bad["gradpar"] = gradpar
    with pytest.raises(ValueError, match="gradpar"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bad["gds21"] = np.ones(7)
    with pytest.raises(ValueError, match="length"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bad["q"] = np.ones(2)
    with pytest.raises(ValueError, match="q"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bmag = np.asarray(bad["bmag"]).copy()
    bmag[2] = np.nan
    bad["bmag"] = bmag
    with pytest.raises(ValueError, match="non-finite"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bad["q"] = np.nan
    with pytest.raises(ValueError, match="q.*non-finite"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bad["nfp"] = 0
    with pytest.raises(ValueError, match="nfp"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    for key, value in list(bad.items()):
        if isinstance(value, np.ndarray):
            bad[key] = value[:0]
    with pytest.raises(ValueError, match="theta"):
        flux_tube_geometry_from_mapping(bad)


def test_flux_tube_geometry_from_mapping_uses_jax_native_defaults_and_shat_alias() -> (
    None
):
    data = _sample_mapping()
    data.pop("jacobian")
    data.pop("grho")
    data["shat"] = data.pop("s_hat")

    geom = flux_tube_geometry_from_mapping(data)

    expected_jacobian = 1.0 / np.asarray(data["gradpar"]) / np.asarray(data["bmag"])
    assert np.allclose(np.asarray(geom.jacobian_profile), expected_jacobian)
    assert np.allclose(np.asarray(geom.grho_profile), 1.0)
    assert geom.s_hat == pytest.approx(0.4)


def test_differentiable_backend_path_helpers_handle_missing_modules(
    tmp_path: Path, monkeypatch
) -> None:
    existing = tmp_path / "backend"
    (existing / "src").mkdir(parents=True)
    monkeypatch.setenv("GKX_VMEX_PATH", str(existing))

    paths = _candidate_paths(
        ("GKX_VMEX_PATH",), (existing, tmp_path / "missing")
    )

    assert paths == [existing.resolve(), (existing / "src").resolve()]
    assert (
        _find_importable_module("gkx_definitely_missing_backend", paths) is None
    )


def test_differentiable_backend_path_helpers_prefer_configured_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    installed_root = tmp_path / "installed"
    local_root = tmp_path / "local_vmec"
    installed_pkg = installed_root / "vmex"
    local_pkg = local_root / "vmex"
    installed_pkg.mkdir(parents=True)
    local_pkg.mkdir(parents=True)
    (installed_pkg / "__init__.py").write_text(
        "marker = 'installed'\n", encoding="utf-8"
    )
    (local_pkg / "__init__.py").write_text("marker = 'local'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(installed_root))
    sys.modules.pop("vmex", None)

    installed = __import__("vmex")
    assert installed.marker == "installed"

    module = _find_importable_module("vmex", [local_root])

    assert module is not None
    assert module.marker == "local"
    assert str(local_pkg) in str(module.__file__)


def test_discover_differentiable_geometry_backends_reports_optional_apis(
    tmp_path: Path, monkeypatch
) -> None:
    vmec_root = tmp_path / "vmex" / "src" / "vmex"
    booz_root = tmp_path / "booz_xform_jax" / "src" / "booz_xform_jax"
    vmec_root.mkdir(parents=True)
    booz_root.mkdir(parents=True)
    (vmec_root / "__init__.py").write_text("marker = 'vmec'\n", encoding="utf-8")
    (booz_root / "__init__.py").write_text("marker = 'booz'\n", encoding="utf-8")
    (booz_root / "jax_api.py").write_text(
        "def prepare_booz_xform_constants_from_inputs(*args, **kwargs): return None\n"
        "def booz_xform_from_inputs(*args, **kwargs): return None\n"
        "def booz_xform_jax_impl(*args, **kwargs): return None\n",
        encoding="utf-8",
    )
    for name in ("vmex", "booz_xform_jax", "booz_xform_jax.jax_api"):
        sys.modules.pop(name, None)
    monkeypatch.setenv("GKX_VMEX_PATH", str(tmp_path / "vmex"))
    monkeypatch.setenv("GKX_BOOZ_XFORM_JAX_PATH", str(tmp_path / "booz_xform_jax"))

    info = discover_differentiable_geometry_backends()

    assert info["vmex_available"] is True
    assert info["vmex_boundary_api_available"] is False
    assert info["booz_xform_jax_available"] is True
    assert info["booz_xform_jax_api_available"] is True


def test_vmec_boundary_aspect_sensitivity_report_uses_discovered_jax_api(
    tmp_path: Path, monkeypatch
) -> None:
    vmec_root = tmp_path / "vmex" / "src" / "vmex"
    vmec_root.mkdir(parents=True)
    (vmec_root / "__init__.py").write_text(
        "import jax.numpy as jnp\n"
        "class BoundaryCoeffs:\n"
        "    def __init__(self, R_cos, R_sin, Z_cos, Z_sin):\n"
        "        self.R_cos = R_cos; self.R_sin = R_sin; self.Z_cos = Z_cos; self.Z_sin = Z_sin\n"
        "class _Modes:\n"
        "    K = 2\n"
        "def vmec_mode_table(*args, **kwargs): return _Modes()\n"
        "def make_angle_grid(*args, **kwargs): return object()\n"
        "def build_helical_basis(*args, **kwargs): return object()\n"
        "def boundary_aspect_ratio(boundary, basis):\n"
        "    return 2.0 * boundary.R_cos[1] + 0.5 * boundary.Z_sin[1]\n",
        encoding="utf-8",
    )
    sys.modules.pop("vmex", None)
    monkeypatch.setenv("GKX_VMEX_PATH", str(tmp_path / "vmex"))

    report = vmec_boundary_aspect_sensitivity_report(
        jnp.asarray([0.08, 0.2]), fd_step=1.0e-3
    )

    assert report["available"] is True
    assert report["backend_info"]["vmex_boundary_api_available"] is True
    assert float(report["max_abs_ad_fd_error"]) < 2.0e-5
    conditioning = report["conditioning"]
    assert conditioning["jacobian_shape"] == [1, 2]
    assert conditioning["sensitivity_map_rank"] == 1
    assert conditioning["worst_abs_error"]["observable_name"] == "aspect_ratio"
    assert conditioning["finite_difference_step_by_parameter"][0]["parameter_name"] == (
        "ripple"
    )


def test_vmec_boundary_aspect_sensitivity_report_validates_parameter_shape() -> None:
    with pytest.raises(ValueError, match="length-2"):
        vmec_boundary_aspect_sensitivity_report(jnp.ones(3))


def test_booz_xform_spectral_sensitivity_report_is_bounded_when_available() -> None:
    for name in ("booz_xform_jax", "booz_xform_jax.jax_api"):
        sys.modules.pop(name, None)

    report = booz_xform_spectral_sensitivity_report(ripple=0.05, fd_step=2.0e-5)

    assert (
        gkx.booz_xform_spectral_sensitivity_report
        is booz_xform_spectral_sensitivity_report
    )
    assert "available" in report
    if not report["available"]:
        assert report["objective"] is None
        return

    assert float(report["objective"]) > 0.0
    assert float(report["max_abs_ad_fd_error"]) < 1.0e-7
    assert np.asarray(report["bmnc_b"]).shape == (1, 2)


def test_booz_xform_flux_tube_sensitivity_report_is_bounded_when_available() -> None:
    for name in ("booz_xform_jax", "booz_xform_jax.jax_api"):
        sys.modules.pop(name, None)

    report = booz_xform_flux_tube_sensitivity_report(ntheta=32, fd_step=2.0e-5)

    assert (
        gkx.booz_xform_flux_tube_sensitivity_report
        is booz_xform_flux_tube_sensitivity_report
    )
    assert "available" in report
    if not report["available"]:
        assert report["sensitivity"] is None
        return

    sensitivity = report["sensitivity"]
    assert sensitivity["observable_names"] == list(geometry_observable_names())
    assert np.asarray(sensitivity["jacobian_ad"]).shape == (
        len(geometry_observable_names()),
        2,
    )
    assert float(sensitivity["max_abs_ad_fd_error"]) < 2.0e-6
    assert float(sensitivity["max_rel_ad_fd_error"]) < 2.0e-4
    assert sensitivity["conditioning"]["jacobian_shape"] == [
        len(geometry_observable_names()),
        2,
    ]
    assert np.asarray(report["bmnc_b"]).shape == (5,)


def test_vmex_boozer_flux_tube_sensitivity_report_starts_from_real_vmec_state_when_available() -> (
    None
):
    for name in (
        "vmex",
        "vmex.driver",
        "vmex.config",
        "vmex.static",
        "vmex.wout",
        "vmex.booz_input",
        "booz_xform_jax",
        "booz_xform_jax.jax_api",
    ):
        sys.modules.pop(name, None)

    report = vmex_boozer_flux_tube_sensitivity_report(ntheta=16, fd_step=2.0e-5)

    assert (
        gkx.vmex_boozer_flux_tube_sensitivity_report
        is vmex_boozer_flux_tube_sensitivity_report
    )
    assert "available" in report
    if not report["available"]:
        assert report["sensitivity"] is None
        return

    sensitivity = report["sensitivity"]
    assert report["case_name"] == "circular_tokamak"
    assert report["param_names"] == ["delta_Rcos", "delta_Zsin"]
    assert sensitivity["observable_names"] == list(geometry_observable_names())
    assert np.asarray(sensitivity["jacobian_ad"]).shape == (
        len(geometry_observable_names()),
        2,
    )
    assert float(sensitivity["max_abs_ad_fd_error"]) < 2.0e-5
    assert float(sensitivity["max_rel_ad_fd_error"]) < 2.0e-4
    assert sensitivity["conditioning"]["finite_ad_jacobian"] is True
    assert np.asarray(report["bmnc_b"]).shape == (2,)


def test_vmec_state_sensitivity_report_helpers_are_fail_closed_and_json_ready() -> None:
    backend_info = {"vmex_available": False}

    unavailable = vmec_state_sensitivity._unavailable_vmec_state_sensitivity_report(
        backend_info=backend_info,
        fd_step=2.0e-5,
        case_name="case",
        reason="missing backend",
    )
    assert unavailable == {
        "available": False,
        "backend_info": backend_info,
        "sensitivity": None,
        "fd_step": 2.0e-5,
        "case_name": "case",
        "reason": "missing backend",
    }

    failed = vmec_state_sensitivity._failed_vmec_state_sensitivity_report(
        backend_info=backend_info,
        fd_step=1.0e-6,
        case_name="case",
        exc=ValueError("bad probe"),
    )
    assert failed["available"] is False
    assert failed["error"] == "ValueError: bad probe"

    ctx = vmec_state_sensitivity._VMECStateContext(
        input_path=Path("input.example"),
        wout_path=Path("wout_example.nc"),
        inp=object(),
        runtime=object(),
        wout=object(),
        state=object(),
        base_Rcos=jnp.ones((3, 4)),
        base_Zsin=jnp.ones((3, 4)),
    )
    metadata = vmec_state_sensitivity._vmec_state_sensitivity_metadata(
        backend_info={"vmex_available": True},
        ctx=ctx,
        case_name="case",
        params=jnp.asarray([0.1, -0.2]),
        radial_index=1,
        mode_index=2,
        surface_index=0,
        fd_step=3.0e-5,
    )
    assert metadata["available"] is True
    assert metadata["param_names"] == ["delta_Rcos", "delta_Zsin"]
    np.testing.assert_allclose(metadata["params"], [0.1, -0.2])
    assert metadata["state_shape"] == [3, 4]
    assert metadata["radial_index"] == 1
    assert metadata["surface_index"] == 0


def test_vmec_state_control_helpers_have_canonical_owner() -> None:
    assert vmec_state_sensitivity._VMECStateContext is (
        vmec_state_controls._VMECStateContext
    )
    assert vmec_state_sensitivity._load_vmec_state_context is (
        vmec_state_controls._load_vmec_state_context
    )
    assert vmec_state_sensitivity._resolve_vmec_state_indices is (
        vmec_state_controls._resolve_vmec_state_indices
    )
    assert vmec_state_sensitivity._perturb_vmec_state is (
        vmec_state_controls._perturb_vmec_state
    )
    assert vmec_state_sensitivity._length_two_params is (
        vmec_state_controls._length_two_params
    )
    assert vmec_flux_tube_reports._load_vmec_state_context is (
        vmec_state_controls._load_vmec_state_context
    )


def test_vmec_field_line_sampling_helpers_have_canonical_owner() -> None:
    assert vmec_state_sensitivity._rms_with_floor is (
        vmec_field_line_sampling._rms_with_floor
    )


def test_vmec_tensor_mapping_builds_finite_mapping_from_mocked_vmec_modules(
    monkeypatch,
) -> None:
    ntheta = 8
    dtype = jnp.float32
    theta = jnp.linspace(-jnp.pi, jnp.pi, ntheta, endpoint=False, dtype=dtype)
    ones = jnp.ones(ntheta, dtype=dtype)
    captured: dict[str, object] = {}

    def gk_fieldline_geometry(state, runtime, **kwargs):  # noqa: ANN001, ANN202
        captured["state"] = state
        captured["runtime"] = runtime
        captured.update(kwargs)
        return {
            "theta": theta,
            "gradpar": 0.5 * ones,
            "bmag": 1.0 + 0.05 * jnp.cos(theta),
            "bgrad": 0.01 * jnp.sin(theta),
            "gds2": 1.1 * ones,
            "gds21": -0.2 * ones,
            "gds22": 0.9 * ones,
            "cvdrift": 0.3 * ones,
            "cvdrift0": 0.0 * ones,
            "gbdrift": 0.25 * ones,
            "gbdrift0": 0.0 * ones,
            "grho": ones,
            "jacobian": 1.2 * ones,
            "q": 1.0 / 0.6,
            "s_hat": -0.1,
            "epsilon": 0.3,
            "R0": 1.5,
            "B0": 2.0,
            "alpha": 0.2,
            "nfp": 4,
            "vmex": {
                "surface_index": 2,
                "L_ref": 1.5,
                "B_ref": 2.0,
                "gradpar_profile": 0.5 * ones,
            },
        }

    monkeypatch.setattr(
        vmec_tensor_mapping,
        "_import_vmex_turbulence",
        lambda: types.SimpleNamespace(gk_fieldline_geometry=gk_fieldline_geometry),
    )

    state = object()
    runtime = object()
    mapping = vmec_tensor_mapping.vmex_flux_tube_mapping_from_state(
        state,
        runtime,
        surface_index=2,
        alpha=0.2,
        ntheta=ntheta,
    )

    assert captured["state"] is state
    assert captured["runtime"] is runtime
    assert captured["s_index"] == 2
    assert captured["alpha"] == pytest.approx(0.2)
    assert captured["ntheta"] == ntheta
    assert captured["equal_arc"] is True
    for key in (
        "theta",
        "gradpar",
        "bmag",
        "gds2",
        "gds21",
        "gds22",
        "gbdrift",
        "gbdrift0",
        "jacobian",
        "grho",
    ):
        arr = np.asarray(mapping[key])
        assert arr.shape == (ntheta,)
        assert np.all(np.isfinite(arr))
    assert mapping["R0"] == pytest.approx(1.5)
    assert mapping["B0"] == pytest.approx(2.0)
    assert mapping["nfp"] == 4
    assert float(mapping["q"]) == pytest.approx(1.0 / 0.6, rel=2.0e-6)
    # The raw gk_fieldline_geometry "vmex" diagnostics are repackaged in place
    # under the same "vmex" key with reference_length/reference_b aliases added.
    assert mapping["vmex"]["surface_index"] == 2
    assert mapping["vmex"]["reference_length"] == pytest.approx(1.5)
    assert mapping["vmex"]["reference_b"] == pytest.approx(2.0)


def test_vmec_state_sensitivity_ad_fd_diagnostics_match_analytic_jacobian() -> None:
    def observables(params: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            [
                params[0] + 2.0 * params[1],
                params[0] * params[0] - params[1],
            ]
        )

    report = vmec_state_sensitivity._ad_fd_jacobian_diagnostics(
        observables,
        jnp.asarray([0.3, -0.2]),
        fd_step=1.0e-3,
        observable_names=("linear_combo", "quadratic_combo"),
        relative_floor=1.0e-10,
    )

    np.testing.assert_allclose(
        np.asarray(report["jacobian_ad"]),
        np.asarray([[1.0, 2.0], [0.6, -1.0]]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(report["jacobian_fd"]),
        np.asarray(report["jacobian_ad"]),
        rtol=1.0e-4,
        atol=1.0e-4,
    )
    assert float(report["max_abs_ad_fd_error"]) < 1.0e-4
    assert report["conditioning"]["sensitivity_map_rank"] == 2
    assert report["conditioning"]["worst_rel_error"]["observable_name"] in {
        "linear_combo",
        "quadratic_combo",
    }


def test_vmec_state_rms_with_floor_matches_tensor_rms_contract() -> None:
    values = jnp.asarray([3.0, 4.0])

    rms = vmec_state_sensitivity._rms_with_floor(values, 1.0e-6)

    assert float(rms) == pytest.approx(float(jnp.sqrt(12.5 + 1.0e-6)))
    assert float(vmec_state_sensitivity._rms_with_floor(jnp.zeros(4), 1.0e-8)) == (
        pytest.approx(1.0e-4)
    )


def test_vmex_flux_tube_sensitivity_report_starts_from_real_vmec_state_when_available() -> (
    None
):
    for name in (
        "vmex",
        "vmex.driver",
        "vmex.config",
        "vmex.static",
        "vmex.wout",
        "vmex.geom",
        "vmex.vmec_bcovar",
        "vmex.field",
    ):
        sys.modules.pop(name, None)

    report = vmex_flux_tube_sensitivity_report(ntheta=12, fd_step=2.0e-6)

    assert (
        gkx.vmex_flux_tube_sensitivity_report
        is vmex_flux_tube_sensitivity_report
    )
    assert "available" in report
    if not report["available"]:
        assert report["sensitivity"] is None
        return

    sensitivity = report["sensitivity"]
    assert report["case_name"] == "nfp4_QH_warm_start"
    assert report["param_names"] == ["delta_Rcos", "delta_Zsin"]
    assert sensitivity["observable_names"] == list(geometry_observable_names())
    assert np.asarray(sensitivity["jacobian_ad"]).shape == (
        len(geometry_observable_names()),
        2,
    )
    assert float(sensitivity["max_abs_ad_fd_error"]) < 1.0e1
    assert float(sensitivity["max_rel_ad_fd_error"]) < 1.0e-3
    assert sensitivity["conditioning"]["finite_fd_jacobian"] is True
    assert int(report["surface_index"]) > 0
    assert float(report["reference_length"]) > 0.0
    assert float(report["reference_b"]) > 0.0


def test_vmex_flux_tube_array_parity_report_tracks_production_gap_when_available() -> (
    None
):
    for name in (
        "vmex",
        "vmex.driver",
        "vmex.config",
        "vmex.static",
        "vmex.wout",
        "vmex.geom",
        "vmex.vmec_bcovar",
        "vmex.field",
        "booz_xform_jax",
        "booz_xform_jax.jax_api",
    ):
        sys.modules.pop(name, None)

    report = vmex_flux_tube_array_parity_report(ntheta=8)

    assert (
        gkx.vmex_flux_tube_array_parity_report
        is vmex_flux_tube_array_parity_report
    )
    assert "available" in report
    if not report["available"]:
        assert "reason" in report or "error" in report
        return

    assert report["case_name"] == "nfp4_QH_warm_start"
    assert report["status"] in {"diagnostic_open", "passed"}
    assert set(report["array_metrics"]) >= {
        "bmag",
        "gds2",
        "gds21",
        "gds22",
        "gbdrift",
        "jacobian",
        "grho",
    }
    assert set(report["scalar_metrics"]) == {"gradpar", "q", "s_hat"}
    assert "equal_arc_core_array_metrics" in report
    assert "equal_arc_metric_array_metrics" in report
    assert "equal_arc_drift_array_metrics" in report
    assert "equal_arc_core_scalar_metrics" in report
    assert np.isfinite(float(report["worst_core_normalized_max_abs"]))
    assert np.isfinite(float(report["worst_scalar_rel"]))
    assert bool(report["array_metrics"]["bmag"]["shape_match"])
    if report["equal_arc_core_array_metrics"]:
        assert (
            gkx.vmex_boozer_equal_arc_core_profiles_from_state
            is vmex_boozer_equal_arc_core_profiles_from_state
        )
        assert set(report["equal_arc_core_array_metrics"]) >= {
            "bmag",
            "bgrad",
            "jacobian",
        }
        assert set(report["equal_arc_metric_array_metrics"]) == {
            "gds2",
            "gds21",
            "gds22",
            "grho",
        }
        assert set(report["equal_arc_drift_array_metrics"]) == {
            "cvdrift",
            "gbdrift",
            "cvdrift0",
            "gbdrift0",
        }
        assert set(report["equal_arc_core_scalar_metrics"]) == {"gradpar", "q", "s_hat"}
        assert np.isfinite(float(report["equal_arc_core_worst_normalized_max_abs"]))
        assert np.isfinite(float(report["equal_arc_core_worst_scalar_rel"]))
        assert np.isfinite(
            float(report["equal_arc_derivative_worst_normalized_max_abs"])
        )
        assert np.isfinite(float(report["equal_arc_metric_worst_normalized_max_abs"]))
        assert np.isfinite(float(report["equal_arc_drift_worst_normalized_max_abs"]))
        assert float(report["equal_arc_core_worst_normalized_max_abs"]) < 5.0e-2
        assert float(report["equal_arc_core_worst_scalar_rel"]) < 5.0e-2
        assert float(report["equal_arc_derivative_worst_normalized_max_abs"]) < 1.0e-1
        assert float(report["equal_arc_metric_worst_normalized_max_abs"]) < 1.2e-1
        assert float(report["equal_arc_drift_worst_normalized_max_abs"]) < 1.2e-1


def test_vmex_flux_tube_array_parity_report_enforces_boozer_resolution_floor() -> (
    None
):
    with pytest.raises(ValueError, match="mboz and nboz"):
        vmex_flux_tube_array_parity_report(mboz=20, nboz=21)


def test_boozer_half_mesh_s_grid_uses_fortran_half_mesh_indices() -> None:
    s_half = _boozer_half_mesh_s_grid(
        jnp.asarray([2, 3, 4]),
        ns_b=3,
        ns_b_full=3,
        dtype=jnp.float64,
    )

    np.testing.assert_allclose(
        np.asarray(s_half),
        np.asarray([1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0]),
    )
    fallback = _boozer_half_mesh_s_grid(
        None,
        ns_b=3,
        ns_b_full=3,
        dtype=jnp.float64,
    )
    np.testing.assert_allclose(np.asarray(fallback), np.asarray(s_half))


def test_vmex_boozer_equal_arc_core_profiles_supports_surface_stencil(
    monkeypatch,
) -> None:
    boozer_tables_mod = types.ModuleType("vmex.core.boozer_tables")
    booz_api = types.ModuleType("booz_xform_jax.jax_api")
    captured: dict[str, list[int] | None] = {}

    def boozer_input_tables(state, runtime, j):
        return {
            "rmnc": jnp.asarray([2.0 + 0.02 * j, 0.03], dtype=jnp.float64),
            "zmns": jnp.asarray([0.0, 0.12], dtype=jnp.float64),
            "lmns": jnp.asarray([0.0, 0.01], dtype=jnp.float64),
            "bmnc": jnp.asarray([1.0 + 0.01 * j, 0.04], dtype=jnp.float64),
            "bsubumnc": jnp.asarray([0.08, 0.0], dtype=jnp.float64),
            "bsubvmnc": jnp.asarray([1.1, 0.0], dtype=jnp.float64),
            "iota": jnp.asarray(0.42 + 0.01 * j, dtype=jnp.float64),
            "G": jnp.asarray(1.1, dtype=jnp.float64),
            "I": jnp.asarray(0.08, dtype=jnp.float64),
            "xm": np.asarray([0, 1], dtype=np.int32),
            "xn": np.asarray([0, 0], dtype=np.int32),
        }

    def prepare_booz_xform_constants_from_inputs(*args, **kwargs):
        return object(), object()

    def booz_xform_from_inputs(*, surface_indices=None, **kwargs):
        if surface_indices is None:
            idx = jnp.arange(5, dtype=jnp.int32)
            captured["surface_indices"] = None
        else:
            idx = jnp.asarray(surface_indices, dtype=jnp.int32)
            captured["surface_indices"] = [int(x) for x in np.asarray(idx)]
        s = idx.astype(jnp.float64)
        rows = idx.size
        return {
            "bmnc_b": jnp.stack((1.0 + 0.01 * s, 0.04 + 0.002 * s), axis=1),
            "rmnc_b": jnp.stack((2.0 + 0.02 * s, 0.03 + 0.001 * s), axis=1),
            "zmns_b": jnp.stack((jnp.zeros(rows), 0.12 + 0.003 * s), axis=1),
            "pmns_b": jnp.stack((jnp.zeros(rows), 0.01 + 0.001 * s), axis=1),
            "iota_b": 0.42 + 0.01 * s,
            "buco_b": 0.08 + 0.002 * s,
            "bvco_b": 1.1 + 0.01 * s,
            "ixm_b": jnp.asarray([0, 1], dtype=jnp.int32),
            "ixn_b": jnp.asarray([0, 0], dtype=jnp.int32),
            "ns_b": 5,
            "jlist": idx + 2,
        }

    boozer_tables_mod.boozer_input_tables = boozer_input_tables
    booz_api.prepare_booz_xform_constants_from_inputs = (
        prepare_booz_xform_constants_from_inputs
    )
    booz_api.booz_xform_from_inputs = booz_xform_from_inputs
    monkeypatch.setitem(sys.modules, "vmex.core.boozer_tables", boozer_tables_mod)
    monkeypatch.setitem(sys.modules, "booz_xform_jax.jax_api", booz_api)

    state = types.SimpleNamespace(R_cos=jnp.ones((6, 2), dtype=jnp.float64))
    runtime = object()  # no .resolution: forces the from-inputs constants path
    wout = types.SimpleNamespace(
        signgs=1, Aminor_p=1.0, phi=np.asarray([0.0, -np.pi]), nfp=4
    )
    mapping = vmex_boozer_equal_arc_core_profiles_from_state(
        state,
        runtime,
        inp=object(),
        wout=wout,
        ntheta=8,
        surface_stencil_width=3,
    )

    assert captured["surface_indices"] == [1, 2, 3]
    assert mapping["surface_stencil_width"] == 3
    assert mapping["boozer_surface_indices"] == [1, 2, 3]
    assert np.all(np.isfinite(np.asarray(mapping["bmag"])))

    zero_flux_mapping = vmex_boozer_equal_arc_core_profiles_from_state(
        state,
        runtime,
        inp=object(),
        wout=types.SimpleNamespace(
            signgs=1, Aminor_p=1.0, phi=np.asarray([0.0, 0.0]), nfp=4
        ),
        ntheta=8,
        surface_stencil_width=3,
    )
    for key in ("gds2", "gds21", "gds22", "grho", "cvdrift", "gbdrift", "jacobian"):
        assert np.all(np.isfinite(np.asarray(zero_flux_mapping[key])))

    with pytest.raises(ValueError, match="surface_stencil_width"):
        vmex_boozer_equal_arc_core_profiles_from_state(
            state,
            runtime,
            inp=object(),
            wout=wout,
            ntheta=8,
            surface_stencil_width=2,
        )


def test_vmex_boozer_surface_stencil_indices_remain_static_under_ad() -> None:
    indices = vmec_boozer_core._surface_indices_for_stencil(
        surface_stencil_width=3, ns_full=6, torflux=0.5
    )

    assert isinstance(indices, np.ndarray)
    np.testing.assert_array_equal(indices, [1, 2, 3])
    derivative = jax.jit(jax.grad(lambda value: value * jnp.sum(jnp.asarray(indices))))(
        jnp.asarray(2.0)
    )
    np.testing.assert_allclose(derivative, 6.0)


def test_vmex_metric_tensor_sensitivity_report_checks_real_metric_tensors_when_available() -> (
    None
):
    for name in (
        "vmex",
        "vmex.driver",
        "vmex.config",
        "vmex.static",
        "vmex.wout",
        "vmex.geom",
    ):
        sys.modules.pop(name, None)

    report = vmex_metric_tensor_sensitivity_report(fd_step=2.0e-5)

    assert (
        gkx.vmex_metric_tensor_sensitivity_report
        is vmex_metric_tensor_sensitivity_report
    )
    assert (
        gkx.vmec_metric_tensor_observable_names
        is vmec_metric_tensor_observable_names
    )
    assert "available" in report
    if not report["available"]:
        assert report["sensitivity"] is None
        return

    assert report["case_name"] == "circular_tokamak"
    assert report["param_names"] == ["delta_Rcos", "delta_Zsin"]
    assert report["observable_names"] == list(vmec_metric_tensor_observable_names())
    assert np.asarray(report["jacobian_ad"]).shape == (
        len(vmec_metric_tensor_observable_names()),
        2,
    )
    assert float(report["max_abs_ad_fd_error"]) < 2.0e-5
    assert float(report["max_rel_ad_fd_error"]) < 2.0e-4
    assert report["conditioning"]["jacobian_shape"] == [
        len(vmec_metric_tensor_observable_names()),
        2,
    ]
    assert len(report["metric_grid_shape"]) == 3


def test_vmex_field_line_tensor_sensitivity_report_checks_stellarator_tensors_when_available() -> (
    None
):
    for name in (
        "vmex",
        "vmex.driver",
        "vmex.config",
        "vmex.static",
        "vmex.wout",
        "vmex.geom",
        "vmex.vmec_bcovar",
        "vmex.field",
    ):
        sys.modules.pop(name, None)

    report = vmex_field_line_tensor_sensitivity_report(ntheta=24, fd_step=1.0e-6)

    assert (
        gkx.vmex_field_line_tensor_sensitivity_report
        is vmex_field_line_tensor_sensitivity_report
    )
    assert (
        gkx.vmec_field_line_tensor_observable_names
        is vmec_field_line_tensor_observable_names
    )
    assert "available" in report
    if not report["available"]:
        assert report["sensitivity"] is None
        return

    assert report["case_name"] == "nfp4_QH_warm_start"
    assert report["param_names"] == ["delta_Rcos", "delta_Zsin"]
    assert report["observable_names"] == list(vmec_field_line_tensor_observable_names())
    assert np.asarray(report["jacobian_ad"]).shape == (
        len(vmec_field_line_tensor_observable_names()),
        2,
    )
    assert float(report["max_abs_ad_fd_error"]) < 5.0e-3
    assert float(report["max_rel_ad_fd_error"]) < 5.0e-4
    assert report["conditioning"]["worst_rel_error"]["parameter_name"] in {
        "delta_Rcos",
        "delta_Zsin",
    }
    assert len(report["metric_grid_shape"]) == 3
    assert int(report["metric_grid_shape"][2]) > 1


def _differentiable_mapping(params: jnp.ndarray) -> dict[str, object]:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 16, endpoint=False)
    ripple, elongation = params
    ones = jnp.ones_like(theta)
    shear = 0.35 + 0.1 * elongation
    bmag = 1.0 + ripple * jnp.cos(theta) + 0.03 * jnp.cos(2.0 * theta)
    return {
        "theta": theta,
        "gradpar": (0.8 + 0.02 * ripple) * ones,
        "bmag": bmag,
        "bgrad": -ripple * jnp.sin(theta),
        "gds2": 1.0 + (shear * theta - 0.2 * elongation * jnp.sin(theta)) ** 2,
        "gds21": -shear * (shear * theta - 0.2 * elongation * jnp.sin(theta)),
        "gds22": (shear * shear) * ones,
        "cvdrift": jnp.cos(theta) + 0.1 * elongation * jnp.sin(theta) ** 2,
        "gbdrift": jnp.cos(theta) + 0.1 * elongation * jnp.sin(theta) ** 2,
        "cvdrift0": -shear * jnp.sin(theta),
        "gbdrift0": -shear * jnp.sin(theta),
        "jacobian": 1.0 / ((0.8 + 0.02 * ripple) * bmag),
        "grho": ones,
        "q": 1.4 + 0.05 * elongation,
        "s_hat": shear,
        "epsilon": ripple,
        "R0": 1.0,
        "nfp": 5,
    }


def test_flux_tube_geometry_from_mapping_is_tracer_safe_for_geometry_sensitivities() -> (
    None
):
    x64_enabled = bool(jax.config.jax_enable_x64)
    fd_step = 2.0e-5 if x64_enabled else 1.0e-3
    abs_tol = 5.0e-6 if x64_enabled else 1.0e-3
    rel_tol = 5.0e-4 if x64_enabled else 2.0e-3
    params = jnp.asarray([0.08, 0.4], dtype=jnp.float64 if x64_enabled else jnp.float32)

    report = geometry_sensitivity_report(
        _differentiable_mapping, params, fd_step=fd_step
    )

    assert gkx.geometry_sensitivity_report is geometry_sensitivity_report
    assert report["observable_names"] == list(geometry_observable_names())
    assert np.asarray(report["jacobian_ad"]).shape == (
        len(geometry_observable_names()),
        2,
    )
    assert np.asarray(report["jacobian_fd"]).shape == (
        len(geometry_observable_names()),
        2,
    )
    assert float(report["max_abs_ad_fd_error"]) < abs_tol
    assert float(report["max_rel_ad_fd_error"]) < rel_tol
    conditioning = report["conditioning"]
    assert conditioning["jacobian_shape"] == [len(geometry_observable_names()), 2]
    assert conditioning["finite_ad_jacobian"] is True
    assert conditioning["finite_fd_jacobian"] is True
    assert conditioning["sensitivity_map_rank"] == 2
    assert np.all(np.isfinite(conditioning["jacobian_singular_values"]))
    assert (
        conditioning["worst_abs_error"]["observable_name"]
        in geometry_observable_names()
    )
    assert conditioning["finite_difference_step_by_parameter"][0]["absolute_step"] == (
        pytest.approx(fd_step)
    )


def test_geometry_tracer_detection_recurses_through_containers() -> None:
    """Container tracer detection keeps validation JAX-transform safe."""

    assert bool(jax.jit(lambda x: _is_traced([x]))(jnp.asarray(1.0)))
    assert bool(jax.jit(lambda x: _is_traced({"x": x}))(jnp.asarray(1.0)))


def test_finite_difference_jacobian_matches_closed_form_linear_map() -> None:
    assert gkx.finite_difference_jacobian is finite_difference_jacobian
    x64_enabled = bool(jax.config.jax_enable_x64)
    fd_step = 1.0e-5 if x64_enabled else 1.0e-3
    rtol = 1.0e-10 if x64_enabled else 2.0e-4
    atol = 1.0e-10 if x64_enabled else 2.0e-4

    def fn(params: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([2.0 * params[0] - params[1], params[0] + 3.0 * params[1]])

    jac = finite_difference_jacobian(fn, jnp.asarray([0.2, -0.5]), step=fd_step)

    np.testing.assert_allclose(
        np.asarray(jac), np.asarray([[2.0, -1.0], [1.0, 3.0]]), rtol=rtol, atol=atol
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        finite_difference_jacobian(fn, jnp.ones((2, 1)))
    with pytest.raises(ValueError, match="step"):
        finite_difference_jacobian(fn, jnp.ones(2), step=0.0)


def test_observable_gradient_report_preserves_raw_relative_zero_scale_error() -> None:
    @jax.custom_jvp
    def observable(params: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([0.0, 2.0 * params[0]])

    @observable.defjvp
    def _observable_jvp(
        primals: tuple[jnp.ndarray],
        tangents: tuple[jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        params = primals[0]
        tangent = tangents[0]
        return observable(params), jnp.asarray([5.0e-7 * tangent[0], 2.0 * tangent[0]])

    report = diff_autodiff.observable_gradient_validation_report(
        observable,
        jnp.asarray([0.25]),
        fd_step=1.0e-3,
        atol=1.0e-6,
        rtol=1.0e-4,
        condition_number_max=None,
    )

    assert report["passed"] is True
    assert float(report["max_rel_ad_fd_error"]) < 1.0e-4
    assert float(report["max_rel_ad_fd_error_raw"]) > 1.0
    zero_scale_check = report["gradient_checks"][0]
    assert zero_scale_check["passed"] is True
    assert float(zero_scale_check["abs_error"]) < 1.0e-6
    assert float(zero_scale_check["rel_error"]) > 1.0


def test_low_level_radial_and_sampling_helpers_cover_edge_contracts() -> None:
    s_grid = jnp.asarray([0.0, 0.5, 1.0])
    profile = jnp.asarray([1.0, 2.0, 5.0])
    modes = jnp.asarray([[1.0, 10.0], [2.0, 20.0], [5.0, 50.0]])

    assert float(_interp_radial(profile, s_grid, 0.25)) == pytest.approx(1.5)
    assert np.allclose(np.asarray(_interp_radial(modes, s_grid, 0.25)), [1.5, 15.0])
    with pytest.raises(ValueError, match="radial interpolation"):
        _interp_radial(jnp.ones((2, 2, 2)), s_grid[:2], 0.25)

    assert np.allclose(
        np.asarray(_radial_derivative_profile(profile, 0.5)), [2.0, 4.0, 6.0]
    )
    assert np.allclose(
        np.asarray(_radial_derivative_profile(jnp.asarray([3.0]), 0.5)), [0.0]
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        _radial_derivative_profile(jnp.ones((2, 2)), 0.5)

    assert np.allclose(
        np.asarray(_radial_derivative_array(modes, 0.5)),
        [[2.0, 20.0], [4.0, 40.0], [6.0, 60.0]],
    )
    assert np.allclose(
        np.asarray(_radial_derivative_array(jnp.ones((1, 2)), 0.5)), [[0.0, 0.0]]
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        _radial_derivative_array(jnp.ones(2), 0.5)

    cumulative = _cumulative_trapezoid(
        jnp.asarray([0.0, 2.0, 2.0]), jnp.asarray([0.0, 1.0, 3.0])
    )
    assert np.allclose(np.asarray(cumulative), [0.0, 1.0, 5.0])
    assert np.allclose(
        np.asarray(_cumulative_trapezoid(jnp.asarray([7.0]), jnp.asarray([0.0]))), [0.0]
    )
    with pytest.raises(ValueError, match="cumulative trapezoid"):
        _cumulative_trapezoid(jnp.ones((2, 1)), jnp.ones(2))

    grid = jnp.asarray([[0.0, 1.0], [2.0, 3.0]])
    sampled = _periodic_bilinear_sample_2d(
        grid,
        jnp.asarray([0.0, jnp.pi / 2.0]),
        jnp.asarray([0.0, jnp.pi / 2.0]),
    )
    assert sampled.shape == (2,)
    assert np.all(np.isfinite(np.asarray(sampled)))
    with pytest.raises(ValueError, match="two-dimensional"):
        _periodic_bilinear_sample_2d(jnp.ones(2), jnp.ones(2), jnp.ones(2))
    with pytest.raises(ValueError, match="same shape"):
        _periodic_bilinear_sample_2d(grid, jnp.ones(2), jnp.ones(3))
    with pytest.raises(ValueError, match="non-empty"):
        _periodic_bilinear_sample_2d(jnp.ones((0, 2)), jnp.ones(1), jnp.ones(1))


def test_parity_metric_helpers_report_shape_and_error_scales() -> None:
    mismatch = _array_parity_metrics(np.ones((2,)), np.ones((2, 1)))
    assert mismatch["shape_match"] is False
    assert mismatch["candidate_shape"] == [2]

    metrics = _array_parity_metrics(np.asarray([1.0, 3.0]), np.asarray([1.0, 2.0]))
    assert metrics["shape_match"] is True
    assert metrics["max_abs"] == pytest.approx(1.0)
    assert metrics["normalized_max_abs"] == pytest.approx(0.5)
    assert metrics["candidate_max"] == pytest.approx(3.0)

    scalar = _scalar_parity_metrics(3.0, 2.0)
    assert scalar["abs"] == pytest.approx(1.0)
    assert scalar["rel"] == pytest.approx(0.5)


def test_optional_vmec_boundary_report_unavailable_path(monkeypatch) -> None:
    monkeypatch.setattr(
        diff_geom,
        "discover_differentiable_geometry_backends",
        lambda: {"vmex_boundary_api_available": False},
    )

    report = vmec_boundary_aspect_sensitivity_report(jnp.asarray([0.1, 0.2]))

    assert report["available"] is False
    assert report["aspect"] is None
    assert report["grad_ad"] is None


def test_evaluate_boozer_bmag_on_field_line_matches_axisymmetric_series() -> None:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 16, endpoint=False)
    bmag, dbmag = diff_geom.evaluate_boozer_bmag_on_field_line(
        theta,
        bmnc_b=jnp.asarray([1.0, 0.2]),
        ixm_b=jnp.asarray([0, 1]),
        ixn_b=jnp.asarray([0, 0]),
        iota=0.41,
    )

    np.testing.assert_allclose(
        np.asarray(bmag),
        np.asarray(1.0 + 0.2 * jnp.cos(theta)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(dbmag),
        np.asarray(-0.2 * jnp.sin(theta)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_geometry_sensitivity_report_rejects_nondesign_parameter_array() -> None:
    with pytest.raises(ValueError, match="params"):
        geometry_sensitivity_report(_differentiable_mapping, jnp.ones((2, 1)))


def test_geometry_inverse_design_report_recovers_selected_observables() -> None:
    x64_enabled = bool(jax.config.jax_enable_x64)
    dtype = jnp.float64 if x64_enabled else jnp.float32
    fd_step = 2.0e-5 if x64_enabled else 1.0e-3
    rel_tol = 5.0e-4 if x64_enabled else 3.0e-3
    initial = jnp.asarray([0.035, 0.12], dtype=dtype)
    target_params = jnp.asarray([0.085, 0.34], dtype=dtype)
    target_geom = flux_tube_geometry_from_mapping(
        _differentiable_mapping(target_params),
        validate_finite=False,
    )
    target = flux_tube_geometry_observables(target_geom)[jnp.asarray([1, 2])]

    report = geometry_inverse_design_report(
        _differentiable_mapping,
        initial,
        target,
        observable_indices=[1, 2],
        max_steps=6,
        damping=1.0e-8,
        fd_step=fd_step,
    )

    assert gkx.geometry_inverse_design_report is geometry_inverse_design_report
    assert report["observable_names"] == [
        "relative_bmag_ripple",
        "metric_frobenius_rms",
    ]
    assert len(report["history"]) == 7
    assert float(report["final_residual_norm"]) < 1.0e-5
    assert float(report["max_rel_ad_fd_error"]) < rel_tol
    assert report["uq"]["sensitivity_map_rank"] == 2
    assert report["conditioning"]["sensitivity_map_rank"] == 2
    assert report["conditioning"]["worst_rel_error"]["observable_name"] in {
        "relative_bmag_ripple",
        "metric_frobenius_rms",
    }
    covariance = np.asarray(report["uq"]["covariance"])
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1.0e-16)
    assert float(report["uq"]["jacobian_condition_number"]) < 1.0e5


def test_geometry_inverse_design_report_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="initial_params"):
        geometry_inverse_design_report(
            _differentiable_mapping,
            jnp.ones((2, 1)),
            jnp.ones(2),
            observable_indices=[1, 2],
        )
    with pytest.raises(ValueError, match="max_steps"):
        geometry_inverse_design_report(
            _differentiable_mapping,
            jnp.ones(2),
            jnp.ones(2),
            observable_indices=[1, 2],
            max_steps=-1,
        )
    with pytest.raises(ValueError, match="damping"):
        geometry_inverse_design_report(
            _differentiable_mapping,
            jnp.ones(2),
            jnp.ones(2),
            observable_indices=[1, 2],
            damping=-1.0,
        )
    with pytest.raises(ValueError, match="observable_indices"):
        geometry_inverse_design_report(
            _differentiable_mapping, jnp.ones(2), jnp.ones(2), observable_indices=[]
        )
    with pytest.raises(ValueError, match="target_observables"):
        geometry_inverse_design_report(
            _differentiable_mapping, jnp.ones(2), jnp.ones(1), observable_indices=[1, 2]
        )
    with pytest.raises(ValueError, match="out-of-range"):
        geometry_inverse_design_report(
            _differentiable_mapping, jnp.ones(2), jnp.ones(1), observable_indices=[99]
        )


def test_geometry_inverse_design_report_defaults_to_all_observables_for_square_problem() -> (
    None
):
    initial = jnp.asarray([0.08, 0.40])
    target = flux_tube_geometry_observables(
        flux_tube_geometry_from_mapping(
            _differentiable_mapping(initial), validate_finite=False
        )
    )

    report = geometry_inverse_design_report(
        _differentiable_mapping,
        initial,
        target,
        max_steps=0,
    )

    assert report["observable_names"] == list(geometry_observable_names())
    assert report["history"][0]["step"] == 0
    assert float(report["history"][0]["residual_norm"]) == pytest.approx(0.0)


# ---- test_check_vmec_boozer_gates.py differentiability-claim ----

import json

from tools.release.check_vmec_boozer_gates import (
    build_vmec_boozer_differentiability_claim_guard,
)


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _objectives_for_gate(gate_type: str) -> tuple[dict[str, bool], dict[str, float]]:
    if gate_type == "frequency":
        names = ("gamma", "omega")
        rel_error = 1.0e-3
    elif gate_type == "quasilinear":
        names = (
            "gamma",
            "omega",
            "kperp_eff2",
            "linear_heat_flux_weight",
            "mixing_length_heat_flux_proxy",
        )
        rel_error = 2.0e-3
    elif gate_type == "nonlinear-window estimator":
        names = (
            "gamma",
            "omega",
            "kperp_eff2",
            "linear_heat_flux_weight",
            "mixing_length_heat_flux_proxy",
            "nonlinear_window_heat_flux_mean",
            "nonlinear_window_heat_flux_cv",
            "nonlinear_window_heat_flux_trend",
        )
        rel_error = 2.5e-2
    else:
        raise AssertionError(f"unexpected gate_type: {gate_type}")
    return {name: True for name in names}, {name: rel_error for name in names}


def _minimal_artifacts(root: Path) -> None:
    _write_json(
        root,
        "docs/_static/vmec_boozer_parity_matrix.json",
        {
            "claim_level": "multi_equilibrium_zero_beta_equal_arc_parity_gate_not_full_transport_gradient_claim",
            "minimum_boozer_mode_count": 21,
            "summary": {
                "all_available": True,
                "all_equal_arc_passed": True,
                "n_cases": 3,
                "n_equal_arc_passed": 3,
            },
            "rows": [
                {
                    "case_name": "case_a",
                    "available": True,
                    "family": "quasi-helical",
                    "equal_arc_all_passed": True,
                    "mboz": 21,
                    "nboz": 21,
                    "mode_floor_passed": True,
                    "production_parity_passed": False,
                    "status": "diagnostic_open",
                },
                {
                    "case_name": "case_b",
                    "available": True,
                    "family": "quasi-isodynamic",
                    "equal_arc_all_passed": True,
                    "mboz": 21,
                    "nboz": 21,
                    "mode_floor_passed": True,
                    "production_parity_passed": False,
                    "status": "diagnostic_open",
                },
                {
                    "case_name": "shaped_tokamak_pressure",
                    "available": True,
                    "family": "axisymmetric finite-beta",
                    "equal_arc_all_passed": True,
                    "mboz": 21,
                    "nboz": 21,
                    "mode_floor_passed": True,
                    "production_parity_passed": False,
                    "status": "diagnostic_open",
                },
            ],
        },
    )
    gradient_rows = []
    for case_name in ("case_a", "case_b"):
        for gate_type in ("frequency", "quasilinear", "nonlinear-window estimator"):
            objectives, objective_rel_error = _objectives_for_gate(gate_type)
            gradient_rows.append(
                {
                    "case_name": case_name,
                    "gate_type": gate_type,
                    "max_rel_error": max(objective_rel_error.values()),
                    "mboz": 21,
                    "nboz": 21,
                    "objective_rel_error": objective_rel_error,
                    "objectives": objectives,
                    "passed": True,
                    "source_scope": "mode21_vmec_boozer_state",
                    "path": f"{case_name}_{gate_type}.json",
                }
            )
    _write_json(
        root,
        "docs/_static/vmec_boozer_gradient_holdout_matrix.json",
        {
            "passed": True,
            "claim_level": (
                "multi_equilibrium_reduced_linear_quasilinear_and_nonlinear_window_"
                "estimator_gradient_gate_not_production_nonlinear_optimization"
            ),
            "summary": {
                "all_gates_passed": True,
                "all_mboz_nboz_at_least_21": True,
                "all_mode21_source_scope": True,
            },
            "rows": gradient_rows,
        },
    )
    _write_json(
        root,
        "docs/_static/differentiable_geometry_bridge.json",
        {
            "vmex_flux_tube_array_parity": {
                "production_parity_passed": False,
                "status": "diagnostic_open",
                "interpretation": "Direct tensor path is diagnostic; equal-arc Boozer path carries the claim.",
                "equal_arc_core_passed": True,
                "equal_arc_derivative_passed": True,
                "equal_arc_metric_passed": True,
                "equal_arc_drift_passed": True,
            },
            "vmex_boozer_flux_tube": {"available": True},
            "booz_xform_flux_tube": {"available": True},
        },
    )
    _write_json(
        root,
        "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json",
        {
            "passed": True,
            "claim_level": "vmec_boozer_geometry_perturbed_startup_plumbing_fd_audit_not_transport_average",
            "vmec_boozer_startup_nonlinear_plumbing_fd_path_gate": True,
            "transport_average_gate": False,
            "production_nonlinear_window_gradient_gate": False,
            "vmec_boozer_production_nonlinear_observable_fd_path_gate": False,
        },
    )
    _write_json(
        root,
        "docs/_static/vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json",
        {
            "case_name": "shaped_tokamak_pressure",
            "eigenpair_gate": {"max_rel_error": 1.0e-8},
            "kind": "mode21_vmec_boozer_linear_frequency_gradient_gate",
            "linear_frequency_gradient_gate": True,
            "linear_growth_gradient_gate": True,
            "mboz": 21,
            "nboz": 21,
            "nonlinear_window_gradient_gate": False,
            "objective_gates": [
                {
                    "objective": "gamma",
                    "passed": True,
                    "rel_error": 0.0,
                },
                {
                    "objective": "omega",
                    "passed": True,
                    "rel_error": 1.0e-8,
                },
            ],
            "passed": True,
            "quasilinear_weight_gradient_gate": False,
            "source_scope": "mode21_vmec_boozer_state",
            "surface_stencil_width": 3,
        },
    )
    _write_json(
        root,
        "docs/_static/vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json",
        {
            "case_name": "shaped_tokamak_pressure",
            "eigenpair_gate": {"max_rel_error": 2.0e-4},
            "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
            "linear_frequency_gradient_gate": True,
            "linear_growth_gradient_gate": True,
            "mboz": 21,
            "nboz": 21,
            "nonlinear_window_gradient_gate": False,
            "objective_gates": [
                {
                    "objective": "gamma",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "omega",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "kperp_eff2",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "linear_heat_flux_weight",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "mixing_length_heat_flux_proxy",
                    "passed": True,
                    "rel_error": 2.0e-4,
                },
            ],
            "passed": True,
            "quasilinear_weight_gradient_gate": True,
            "source_scope": "mode21_vmec_boozer_state",
            "surface_stencil_width": 3,
        },
    )
    _write_json(
        root,
        "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json",
        {
            "case_name": "shaped_tokamak_pressure",
            "eigenpair_gate": {"max_rel_error": 2.5e-4},
            "kind": "mode21_vmec_boozer_nonlinear_window_gradient_gate",
            "linear_frequency_gradient_gate": True,
            "linear_growth_gradient_gate": True,
            "mboz": 21,
            "nboz": 21,
            "nonlinear_window_gradient_gate": True,
            "objective_gates": [
                {
                    "objective": "gamma",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "omega",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "kperp_eff2",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "linear_heat_flux_weight",
                    "passed": True,
                    "rel_error": 1.0e-4,
                },
                {
                    "objective": "mixing_length_heat_flux_proxy",
                    "passed": True,
                    "rel_error": 2.0e-4,
                },
                {
                    "objective": "nonlinear_window_heat_flux_mean",
                    "passed": True,
                    "rel_error": 2.0e-4,
                },
                {
                    "objective": "nonlinear_window_heat_flux_cv",
                    "passed": True,
                    "rel_error": 2.0e-4,
                },
                {
                    "objective": "nonlinear_window_heat_flux_trend",
                    "passed": True,
                    "rel_error": 2.0e-4,
                },
            ],
            "passed": True,
            "production_nonlinear_window_gradient_gate": False,
            "quasilinear_weight_gradient_gate": True,
            "source_scope": "mode21_vmec_boozer_state",
            "surface_stencil_width": 3,
        },
    )


def test_vmec_boozer_differentiability_claim_guard_accepts_scoped_artifacts(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is True
    assert not report["blockers"]
    assert (
        report["checks"]["differentiable_geometry_bridge"][
            "direct_tensor_gap_explicitly_scoped"
        ]
        is True
    )
    assert (
        report["checks"]["nonlinear_fd_audit"][
            "production_nonlinear_window_gradient_gate"
        ]
        is False
    )
    assert report["checks"]["parity_matrix"]["finite_beta_pressure_equal_arc_rows"] == [
        "shaped_tokamak_pressure"
    ]
    assert report["checks"]["finite_beta_frequency_gate"]["case_name"] == (
        "shaped_tokamak_pressure"
    )
    assert report["checks"]["finite_beta_quasilinear_gate"]["case_name"] == (
        "shaped_tokamak_pressure"
    )
    assert report["checks"]["finite_beta_nonlinear_window_gate"]["case_name"] == (
        "shaped_tokamak_pressure"
    )


def test_vmec_boozer_differentiability_claim_guard_rejects_hidden_direct_gap(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    bridge_path = tmp_path / "docs/_static/differentiable_geometry_bridge.json"
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    bridge["vmex_flux_tube_array_parity"]["status"] = "failed"
    bridge["vmex_flux_tube_array_parity"]["interpretation"] = ""
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "direct_tensor_parity_gap_not_explicitly_scoped" in report["blockers"]


def test_vmec_boozer_differentiability_claim_guard_requires_finite_beta_parity(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    parity_path = tmp_path / "docs/_static/vmec_boozer_parity_matrix.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["rows"] = [
        row for row in parity["rows"] if row["case_name"] != "shaped_tokamak_pressure"
    ]
    parity_path.write_text(json.dumps(parity), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "parity_matrix_missing_required_family" in report["blockers"]
    assert "parity_matrix_missing_finite_beta_pressure_row" in report["blockers"]


def test_vmec_boozer_differentiability_claim_guard_rejects_unscoped_nonlinear_claim(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    audit_path = tmp_path / "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["production_nonlinear_window_gradient_gate"] = True
    audit["transport_average_gate"] = True
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "startup_fd_audit_attempts_production_nonlinear_claim" in report["blockers"]


def test_vmec_boozer_differentiability_claim_guard_requires_mode21_gradient_scope(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gradient_path = tmp_path / "docs/_static/vmec_boozer_gradient_holdout_matrix.json"
    gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
    gradient["rows"][0]["source_scope"] = "reduced_fixture"
    gradient_path.write_text(json.dumps(gradient), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "gradient_holdout_wrong_source_scope" in report["blockers"]


def test_vmec_boozer_differentiability_claim_guard_requires_finite_beta_frequency_gate(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gate_path = (
        tmp_path
        / "docs/_static/vmec_boozer_shaped_pressure_solver_frequency_gradient_gate.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["case_name"] = "nfp4_QH_warm_start"
    gate["quasilinear_weight_gradient_gate"] = True
    gate["objective_gates"][1]["rel_error"] = 0.2
    gate["eigenpair_gate"]["max_rel_error"] = 0.2
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "finite_beta_frequency_gate_wrong_case" in report["blockers"]
    assert "finite_beta_frequency_gate_error_threshold_failed" in report["blockers"]
    assert (
        "finite_beta_frequency_gate_attempts_transport_gradient_claim"
        in report["blockers"]
    )


def test_vmec_boozer_differentiability_claim_guard_requires_finite_beta_ql_gate(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gate_path = (
        tmp_path
        / "docs/_static/vmec_boozer_shaped_pressure_quasilinear_gradient_gate.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["case_name"] = "nfp4_QH_warm_start"
    gate["nonlinear_window_gradient_gate"] = True
    gate["objective_gates"] = [
        row
        for row in gate["objective_gates"]
        if row["objective"] != "mixing_length_heat_flux_proxy"
    ]
    gate["eigenpair_gate"]["max_rel_error"] = 0.2
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "finite_beta_quasilinear_gate_wrong_case" in report["blockers"]
    assert "finite_beta_quasilinear_gate_missing_objective" in report["blockers"]
    assert "finite_beta_quasilinear_gate_error_threshold_failed" in report["blockers"]
    assert (
        "finite_beta_quasilinear_gate_attempts_nonlinear_gradient_claim"
        in report["blockers"]
    )


def test_vmec_boozer_differentiability_claim_guard_requires_finite_beta_nonlinear_window_gate(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gate_path = (
        tmp_path
        / "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["case_name"] = "nfp4_QH_warm_start"
    gate["production_nonlinear_window_gradient_gate"] = True
    gate["objective_gates"] = [
        row
        for row in gate["objective_gates"]
        if row["objective"] != "nonlinear_window_heat_flux_trend"
    ]
    gate["eigenpair_gate"]["max_rel_error"] = 0.2
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "finite_beta_nonlinear_window_gate_wrong_case" in report["blockers"]
    assert "finite_beta_nonlinear_window_gate_missing_objective" in report["blockers"]
    assert (
        "finite_beta_nonlinear_window_gate_error_threshold_failed" in report["blockers"]
    )
    assert (
        "finite_beta_nonlinear_window_gate_attempts_production_transport_claim"
        in report["blockers"]
    )


def test_vmec_boozer_differentiability_claim_guard_requires_finite_beta_nonlinear_window_artifact(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gate_path = (
        tmp_path
        / "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json"
    )
    gate_path.unlink()

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "finite_beta_nonlinear_window_gate_unreadable" in report["blockers"]


def test_vmec_boozer_differentiability_claim_guard_blocks_malformed_finite_beta_nonlinear_window_gate(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gate_path = (
        tmp_path
        / "docs/_static/vmec_boozer_shaped_pressure_nonlinear_window_gradient_gate.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["kind"] = "mode21_vmec_boozer_quasilinear_gradient_gate"
    gate["source_scope"] = "reduced_fixture"
    gate["mboz"] = 19
    gate["nboz"] = 21
    gate["passed"] = False
    gate["nonlinear_window_gradient_gate"] = False
    gate["objective_gates"][0]["passed"] = False
    gate["eigenpair_gate"].pop("max_rel_error")
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "finite_beta_nonlinear_window_gate_failed" in report["blockers"]
    assert "finite_beta_nonlinear_window_gate_wrong_kind" in report["blockers"]
    assert "finite_beta_nonlinear_window_gate_wrong_source_scope" in report["blockers"]
    assert "finite_beta_nonlinear_window_gate_mode_floor_failed" in report["blockers"]
    assert "finite_beta_nonlinear_window_gate_objective_failed" in report["blockers"]
    assert (
        "finite_beta_nonlinear_window_gate_error_threshold_failed" in report["blockers"]
    )


def test_vmec_boozer_differentiability_claim_guard_requires_ql_objectives(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gradient_path = tmp_path / "docs/_static/vmec_boozer_gradient_holdout_matrix.json"
    gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
    ql_row = next(row for row in gradient["rows"] if row["gate_type"] == "quasilinear")
    ql_row["objectives"].pop("mixing_length_heat_flux_proxy")
    ql_row["objective_rel_error"].pop("mixing_length_heat_flux_proxy")
    gradient_path.write_text(json.dumps(gradient), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "gradient_holdout_missing_required_objective" in report["blockers"]


def test_vmec_boozer_differentiability_claim_guard_rejects_large_objective_error(
    tmp_path: Path,
) -> None:
    _minimal_artifacts(tmp_path)
    gradient_path = tmp_path / "docs/_static/vmec_boozer_gradient_holdout_matrix.json"
    gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
    estimator_row = next(
        row
        for row in gradient["rows"]
        if row["gate_type"] == "nonlinear-window estimator"
    )
    estimator_row["max_rel_error"] = 0.2
    estimator_row["objective_rel_error"]["nonlinear_window_heat_flux_mean"] = 0.2
    gradient_path.write_text(json.dumps(gradient), encoding="utf-8")

    report = build_vmec_boozer_differentiability_claim_guard(tmp_path)

    assert report["passed"] is False
    assert "gradient_holdout_error_threshold_failed" in report["blockers"]
