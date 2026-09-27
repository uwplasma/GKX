"""Geometry helper tests."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.config import GeometryConfig, GridConfig
import gkx.geometry as geometry_pkg
import gkx.geometry.core as geometry_core
from gkx.geometry import (
    ZERO_SHAT_THRESHOLD,
    FluxTubeGeometryData,
    SAlphaGeometry,
    SlabGeometry,
    _bgrad_from_bmag,
    _periodic_spectral_derivative,
    apply_geometry_grid_defaults,
    build_flux_tube_geometry,
    ensure_flux_tube_geometry_data,
    load_imported_geometry_netcdf,
    sample_flux_tube_geometry,
    twist_shift_params,
)
from gkx.core_grid import build_spectral_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
import json
from gkx.geometry.differentiable import observable_gradient_validation_report
from gkx.geometry.kernels import (
    finite_diff_nonuniform,
    centered_reflected_difference,
    weighted_centered_difference,
    extend_nperiod_data,
    reflect_and_append,
    nperiod_contract,
    nperiod_mask,
)
import os
from pathlib import Path
import sys
import netCDF4 as nc
from unittest.mock import MagicMock
from gkx.config import (
    InitializationConfig,
    TimeConfig,
)
from gkx.config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
)
from gkx.geometry.vmec_eik import (
    build_vmec_geometry_request,
    default_vmec_eik_output_path,
    generate_runtime_vmec_eik,
)
import gkx.geometry.imported_vmec as vmec_backend
from gkx.geometry.imported_vmec import internal_vmec_backend_available
import dataclasses
from types import SimpleNamespace
import gkx.geometry.vmec_state_controls as controls
import gkx.geometry.vmec_boozer_derivatives as vmec_derivatives
import gkx.geometry.backend_discovery as vmec_backend_discovery
import gkx.geometry.imported_vmec as vmec_facade
import gkx.geometry.vmec_field_line_sampling as vmec_fieldline_numerics
from gkx.geometry.imported_vmec import _Struct
from gkx.geometry.imported_vmec import (
    _apply_flux_tube_cut,
    _booz_read_wout_square_layout_failure,
    _booz_xform_jax_search_paths,
    _equal_arc_remap,
    _import_booz_backend,
    _import_module_with_search_paths,
    _vmec_fieldlines,
    _vmec_splines,
    dermv,
    generate_vmec_eik_internal,
    nperiod_set,
    write_vmec_eik_netcdf,
)
from gkx.operators.linear.params import (
    LinearTerms,
    linear_params_for_geometry,
)
from gkx.operators.linear.rhs import linear_rhs_cached
import tempfile
from gkx.geometry.numerics import _array_parity_metrics
from gkx.geometry.vmec_boozer_derivatives import (
    _MU_0,
    boozer_pressure_gradient,
    raw_drift_profiles,
)
from support.paths import load_release_tool
import subprocess
import tomllib
from support.paths import REPO_ROOT


def test_geometry_package_facade_preserves_core_symbol_identity() -> None:
    """The geometry package should remain a compatibility facade."""

    assert geometry_pkg.SAlphaGeometry is geometry_core.SAlphaGeometry
    assert geometry_pkg.SlabGeometry is geometry_core.SlabGeometry
    assert geometry_pkg.FluxTubeGeometryData is geometry_core.FluxTubeGeometryData
    assert (
        geometry_pkg.sample_flux_tube_geometry
        is geometry_core.sample_flux_tube_geometry
    )
    assert (
        geometry_pkg.load_imported_geometry_netcdf
        is geometry_core.load_imported_geometry_netcdf
    )
    assert (
        geometry_pkg.build_flux_tube_geometry is geometry_core.build_flux_tube_geometry
    )
    assert (
        geometry_pkg.apply_geometry_grid_defaults
        is geometry_core.apply_geometry_grid_defaults
    )
    assert geometry_pkg._bgrad_from_bmag is geometry_core._bgrad_from_bmag


def test_kperp2_matches_s_alpha():
    """k_perp^2 should match the s-alpha formula for kx(theta)."""
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.0)
    kx0 = jnp.array(0.0)
    ky = jnp.array(1.0)
    theta = jnp.array([0.0, 2.0])
    kperp2 = geom.k_perp2(kx0, ky, theta)
    assert jnp.allclose(kperp2[0], 1.0)
    assert jnp.allclose(kperp2[1], 5.0)


def test_geometry_from_config():
    """Geometry config should map cleanly into the geometry class."""
    cfg = GeometryConfig(q=1.7, s_hat=0.9, epsilon=0.2, R0=3.0, B0=2.0, alpha=0.1)
    geom = SAlphaGeometry.from_config(cfg)
    assert geom.q == 1.7
    assert geom.R0 == 3.0
    assert geom.alpha == 0.1


def test_build_flux_tube_geometry_analytic_from_config():
    cfg = GeometryConfig(q=1.7, s_hat=0.9, epsilon=0.2, R0=3.0, B0=2.0, alpha=0.1)
    geom = build_flux_tube_geometry(cfg)

    assert isinstance(geom, SAlphaGeometry)
    assert geom.q == 1.7


def test_build_flux_tube_geometry_slab_from_config():
    cfg = GeometryConfig(model="slab", s_hat=0.3, z0=2.5, zero_shat=False)
    geom = build_flux_tube_geometry(cfg)

    assert isinstance(geom, SlabGeometry)
    assert geom.s_hat == pytest.approx(0.3)
    assert geom.gradpar() == pytest.approx(0.4)


def test_slab_geometry_matches_reference_contract():
    geom = SlabGeometry(s_hat=0.4, z0=3.0)
    theta = jnp.array([-jnp.pi, 0.0, jnp.pi / 2.0])
    gds2, gds21, gds22 = geom.metric_coeffs(theta)
    cv, gb, cv0, gb0 = geom.drift_coeffs(theta)

    assert jnp.allclose(geom.bmag(theta), jnp.ones_like(theta))
    assert jnp.allclose(geom.bgrad(theta), jnp.zeros_like(theta))
    assert geom.gradpar() == pytest.approx(1.0 / 3.0)
    assert jnp.allclose(cv, jnp.zeros_like(theta))
    assert jnp.allclose(gb, jnp.zeros_like(theta))
    assert jnp.allclose(cv0, jnp.zeros_like(theta))
    assert jnp.allclose(gb0, jnp.zeros_like(theta))
    assert jnp.allclose(gds2, 1.0 + (0.4 * theta) ** 2)
    assert jnp.allclose(gds21, -(0.4 * 0.4) * theta)
    assert jnp.allclose(gds22, jnp.full_like(theta, 0.16))

    kx = jnp.array([0.0, 0.2])
    ky = jnp.array([0.1, 0.3])
    assert geom.kx_effective(kx, ky, theta[:2]).shape == (2,)
    assert geom.k_perp2(
        kx[None, :, None], ky[:, None, None], theta[None, None, :]
    ).shape == (2, 2, 3)
    cv_d, gb_d = geom.drift_components(kx, ky, theta)
    assert cv_d.shape == (2, 2, 3)
    assert gb_d.shape == (2, 2, 3)
    assert geom.omega_d(kx, ky, theta).shape == (2, 2, 3)


def test_slab_geometry_pytree_roundtrip_and_z0_default() -> None:
    geom = SlabGeometry(s_hat=0.2, z0=None, zero_shat=False)
    children, aux = geom.tree_flatten()
    restored = SlabGeometry.tree_unflatten(aux, children)

    assert restored.s_hat == pytest.approx(geom.s_hat)
    assert restored.gradpar() == pytest.approx(1.0)


def test_zero_shat_slab_geometry_matches_zero_shear_override():
    geom = SlabGeometry.from_config(
        GeometryConfig(model="slab", s_hat=0.8, zero_shat=True)
    )
    theta = jnp.array([-1.0, 0.0, 1.0])
    gds2, gds21, gds22 = geom.metric_coeffs(theta)

    assert geom.s_hat == pytest.approx(0.0)
    assert geom.gradpar() == pytest.approx(1.0)
    assert jnp.allclose(gds2, jnp.ones_like(theta))
    assert jnp.allclose(gds21, jnp.zeros_like(theta))
    assert jnp.allclose(gds22, jnp.ones_like(theta))


def test_slab_geometry_auto_zero_shat_threshold_matches_reference_default():
    geom = SlabGeometry.from_config(
        GeometryConfig(model="slab", s_hat=0.1 * ZERO_SHAT_THRESHOLD, zero_shat=False)
    )

    assert geom.zero_shat is True
    assert geom.s_hat == pytest.approx(0.0)


def test_salpha_geometry_auto_zero_shat_threshold_matches_reference_default():
    geom = SAlphaGeometry.from_config(
        GeometryConfig(s_hat=0.1 * ZERO_SHAT_THRESHOLD, zero_shat=False)
    )
    theta = jnp.array([-1.0, 0.0, 1.0])
    _gds2, _gds21, gds22 = geom.metric_coeffs(theta)

    assert geom.s_hat == pytest.approx(0.0)
    assert jnp.allclose(gds22, jnp.ones_like(theta))


@pytest.mark.parametrize("geometry_type", [SAlphaGeometry, SlabGeometry])
@pytest.mark.parametrize("shear_value", [0.8, -0.8])
def test_analytic_metric_finite_shear_jit_gradient_matches_finite_difference(
    geometry_type,
    shear_value: float,
) -> None:
    dtype = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32
    theta = jnp.asarray([-0.7, 0.2, 1.1], dtype=dtype)
    shear = jnp.asarray(shear_value, dtype=dtype)
    step = jnp.asarray(1.0e-5 if dtype == jnp.float64 else 1.0e-3, dtype=dtype)

    def metric_vector(s_hat: jnp.ndarray) -> jnp.ndarray:
        if geometry_type is SAlphaGeometry:
            geom = SAlphaGeometry(q=1.4, s_hat=s_hat, epsilon=0.18, alpha=0.2)
        else:
            geom = SlabGeometry(s_hat=s_hat)
        return jnp.concatenate([coeff.ravel() for coeff in geom.metric_coeffs(theta)])

    gradient = jax.jacrev(metric_vector)(shear)
    compiled_gradient = jax.jit(jax.jacrev(metric_vector))(shear)
    finite_difference = (metric_vector(shear + step) - metric_vector(shear - step)) / (
        2.0 * step
    )

    np.testing.assert_allclose(
        np.asarray([gradient, compiled_gradient]),
        np.stack([finite_difference, finite_difference]),
        rtol=2.0e-3 if dtype == jnp.float32 else 2.0e-8,
        atol=2.0e-4 if dtype == jnp.float32 else 2.0e-10,
    )


def test_bmag_and_omega_d_shapes():
    """Magnetic field and drift frequency should have consistent shapes."""
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    theta = jnp.array([0.0])
    bmag = geom.bmag(theta)
    assert jnp.isclose(bmag[0], 1.0 / (1.0 + geom.epsilon))
    assert jnp.isclose(geom.gradpar(), jnp.abs(1.0 / (geom.q * geom.R0)))

    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0))
    omega_d = geom.omega_d(grid.kx, grid.ky, grid.z)
    assert omega_d.shape == (grid.ky.size, grid.kx.size, grid.z.size)


def test_metric_and_drift_coeffs_at_midplane():
    """Metric and drift coefficients should reduce cleanly at theta=0."""
    geom = SAlphaGeometry(q=1.4, s_hat=0.7, epsilon=0.0, R0=2.0, alpha=0.2)
    theta = jnp.array([0.0])
    gds2, gds21, gds22 = geom.metric_coeffs(theta)
    assert jnp.isclose(gds2[0], 1.0)
    assert jnp.isclose(gds21[0], 0.0)
    assert jnp.isclose(gds22, geom.s_hat * geom.s_hat)

    cv, gb, cv0, gb0 = geom.drift_coeffs(theta)
    expected = geom.drift_scale * (1.0 / geom.R0)
    assert jnp.isclose(cv[0], expected)
    assert jnp.isclose(gb[0], cv[0])
    assert jnp.isclose(cv0[0], 0.0)
    assert jnp.isclose(gb0[0], 0.0)

    cv_d, gb_d = geom.drift_components(jnp.array([0.0]), jnp.array([1.0]), theta)
    assert cv_d.shape == (1, 1, 1)
    assert gb_d.shape == (1, 1, 1)
    bgrad = geom.bgrad(theta)
    assert jnp.isfinite(bgrad[0])


def test_kx_effective_shear_shift():
    """kx_effective should include the s-alpha shear shift."""
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.0, alpha=0.5)
    kx0 = jnp.array([0.2])
    ky = jnp.array([0.3])
    theta = jnp.array([1.0])
    kx_eff = geom.kx_effective(kx0, ky, theta)
    shear = geom.s_hat * theta - geom.alpha * jnp.sin(theta)
    assert jnp.isclose(kx_eff[0], kx0[0] - shear[0] * ky[0])


def test_twist_shift_params_for_analytic_geometry_avoids_metric_coeffs(
    monkeypatch: pytest.MonkeyPatch,
):
    """Analytic twist/shift defaults should stay on host scalars."""

    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.0, alpha=0.2)
    grid = GridConfig(
        Nx=4,
        Ny=4,
        Nz=16,
        Lx=6.28,
        Ly=6.28,
        boundary="linked",
        y0=10.0,
        ntheta=16,
        nperiod=2,
    )

    def _fail(self, _theta):
        raise AssertionError(
            "metric_coeffs should not be called for analytic twist-shift defaults"
        )

    monkeypatch.setattr(SAlphaGeometry, "metric_coeffs", _fail)

    jtwist, x0 = twist_shift_params(geom, grid)

    theta_min = -np.pi * (2 * int(grid.nperiod) - 1)
    shear = float(geom.s_hat) * theta_min - float(geom.alpha) * np.sin(theta_min)
    expected_fac = (
        2.0
        * float(geom.s_hat)
        * (-float(geom.s_hat) * shear)
        / (float(geom.s_hat) ** 2)
    )
    expected_jtwist = int(round(expected_fac))
    if expected_jtwist == 0:
        expected_jtwist = 1
    expected_y0 = float(grid.y0)
    expected_x0 = expected_y0 * abs(expected_jtwist) / abs(expected_fac)

    assert jtwist == expected_jtwist
    assert x0 == pytest.approx(expected_x0)


def test_geometry_tree_roundtrip():
    """Geometry pytree should round-trip through flatten/unflatten."""
    geom = SAlphaGeometry(q=1.5, s_hat=0.8, epsilon=0.2, R0=3.0, B0=1.8, alpha=0.1)
    children, aux = geom.tree_flatten()
    geom2 = SAlphaGeometry.tree_unflatten(aux, children)
    assert geom2.q == geom.q
    assert geom2.s_hat == geom.s_hat
    assert geom2.epsilon == geom.epsilon
    assert geom2.R0 == geom.R0
    assert geom2.B0 == geom.B0
    assert geom2.alpha == geom.alpha


def test_sampled_flux_tube_geometry_matches_salpha_profiles():
    """Sampled geometry data should preserve the analytic s-alpha profiles."""
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, alpha=0.1)
    theta = jnp.linspace(-jnp.pi, jnp.pi, 17)
    sampled = sample_flux_tube_geometry(geom, theta)

    assert jnp.allclose(sampled.bmag(theta), geom.bmag(theta))
    assert jnp.allclose(sampled.bgrad(theta), geom.bgrad(theta))
    gds2_s, gds21_s, gds22_s = sampled.metric_coeffs(theta)
    gds2_g, gds21_g, gds22_g = geom.metric_coeffs(theta)
    assert jnp.allclose(gds2_s, gds2_g)
    assert jnp.allclose(gds21_s, gds21_g)
    assert jnp.allclose(gds22_s, jnp.full_like(theta, gds22_g))

    kx = jnp.array([0.0, 0.2])
    ky = jnp.array([0.1, 0.3])
    theta_b = theta[None, None, :]
    assert jnp.allclose(
        sampled.k_perp2(kx[None, :, None], ky[:, None, None], theta_b),
        geom.k_perp2(kx[None, :, None], ky[:, None, None], theta_b),
    )


def test_sampled_flux_tube_geometry_matches_slab_profiles():
    geom = SlabGeometry(s_hat=0.5, z0=2.0)
    theta = jnp.linspace(-jnp.pi, jnp.pi, 17)
    sampled = sample_flux_tube_geometry(geom, theta)

    assert sampled.source_model == "slab"
    assert jnp.allclose(sampled.bmag(theta), jnp.ones_like(theta))
    assert jnp.allclose(sampled.bgrad(theta), jnp.zeros_like(theta))
    gds2_s, gds21_s, gds22_s = sampled.metric_coeffs(theta)
    gds2_g, gds21_g, gds22_g = geom.metric_coeffs(theta)
    assert jnp.allclose(gds2_s, gds2_g)
    assert jnp.allclose(gds21_s, gds21_g)
    assert jnp.allclose(gds22_s, gds22_g)


def test_sampled_flux_tube_geometry_tree_roundtrip():
    """Sampled geometry should behave as a pytree for JAX transforms."""

    geom = SAlphaGeometry(q=1.8, s_hat=0.6, epsilon=0.14, R0=2.4, alpha=0.2)
    theta = jnp.linspace(-jnp.pi, jnp.pi, 9)
    sampled = sample_flux_tube_geometry(geom, theta)

    leaves, treedef = jax.tree_util.tree_flatten(sampled)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert restored.source_model == sampled.source_model
    assert restored.kperp2_bmag is sampled.kperp2_bmag
    assert jnp.allclose(restored.theta, sampled.theta)
    assert jnp.allclose(restored.bmag_profile, sampled.bmag_profile)
    assert jnp.allclose(restored.cv_profile, sampled.cv_profile)


def test_ensure_flux_tube_geometry_data_reuses_sampled_input():
    """The geometry contract helper should preserve pre-sampled geometry objects."""

    geom = SAlphaGeometry(q=1.7, s_hat=0.9, epsilon=0.1)
    theta = jnp.linspace(-jnp.pi, jnp.pi, 13)
    sampled = sample_flux_tube_geometry(geom, theta)

    ensured = ensure_flux_tube_geometry_data(sampled, theta)

    assert ensured is sampled


def test_ensure_flux_tube_geometry_data_trims_closed_theta_interval():
    """Imported geometry should drop the terminal theta point for solver grids."""

    geom = SAlphaGeometry(q=1.7, s_hat=0.9, epsilon=0.1)
    theta_closed = jnp.linspace(-jnp.pi, jnp.pi, 17)
    theta_solver = theta_closed[:-1]
    sampled = replace(
        sample_flux_tube_geometry(geom, theta_closed),
        cylindrical_R_profile=np.linspace(4.0, 5.0, theta_closed.size),
        cylindrical_Z_profile=np.linspace(-1.0, 1.0, theta_closed.size),
        toroidal_angle_profile=np.linspace(-2.0, 2.0, theta_closed.size),
    )

    ensured = ensure_flux_tube_geometry_data(sampled, theta_solver)

    assert ensured is not sampled
    assert ensured.theta.shape == theta_solver.shape
    assert jnp.allclose(ensured.theta, theta_solver)
    assert jnp.allclose(ensured.bmag_profile, sampled.bmag_profile[:-1])
    np.testing.assert_allclose(
        ensured.cylindrical_R_profile, sampled.cylindrical_R_profile[:-1]
    )


def test_sampled_geometry_rejects_mismatched_theta_and_trim_too_short() -> None:
    geom = sample_flux_tube_geometry(
        SAlphaGeometry(q=1.0, s_hat=0.0, epsilon=0.0), jnp.asarray([0.0])
    )

    with pytest.raises(ValueError, match="fewer than two"):
        geom.trim_terminal_theta_point()
    with pytest.raises(ValueError, match="same last dimension"):
        geom.bmag(jnp.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="does not match"):
        geom.bmag(jnp.asarray([1.0]))


def test_sampled_geometry_broadcasts_profiles_on_batched_theta() -> None:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 5)
    geom = sample_flux_tube_geometry(
        SAlphaGeometry(q=1.0, s_hat=0.0, epsilon=0.0), theta
    )
    theta_batched = jnp.stack([theta, theta], axis=0)

    bmag = geom.bmag(theta_batched)

    assert bmag.shape == theta_batched.shape
    assert jnp.allclose(bmag[0], geom.bmag(theta))


def test_periodic_derivative_and_bgrad_validation_paths() -> None:
    assert np.allclose(_periodic_spectral_derivative(np.asarray([1.0]), 1.0), [0.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        _periodic_spectral_derivative(np.ones((2, 2)), 1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        _bgrad_from_bmag(np.ones((2, 2)), np.ones(2), 1.0, closed=False)
    with pytest.raises(ValueError, match="same shape"):
        _bgrad_from_bmag(np.ones(3), np.ones(2), 1.0, closed=False)
    assert np.allclose(
        _bgrad_from_bmag(np.asarray([0.0]), np.asarray([1.0]), 1.0, closed=False),
        [0.0],
    )


def test_load_imported_geometry_netcdf_reads_sampled_contract(tmp_path):
    """imported grouped NetCDF geometry output should map into the sampled contract."""

    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom.out.nc"
    theta = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    jacobian = np.linspace(2.0, 3.0, theta.size)
    with Dataset(path, "w") as root:
        root.createDimension("theta", theta.size)
        grids = root.createGroup("Grids")
        geom = root.createGroup("Geometry")
        grids.createVariable("theta", "f8", ("theta",))[:] = theta
        for name, values in {
            "bmag": np.linspace(1.0, 1.2, theta.size),
            "bgrad": np.linspace(-0.1, 0.1, theta.size),
            "gds2": np.linspace(1.0, 2.0, theta.size),
            "gds21": np.linspace(-0.2, 0.2, theta.size),
            "gds22": np.full(theta.size, 0.8),
            "cvdrift": np.linspace(0.3, 0.5, theta.size),
            "gbdrift": np.linspace(0.3, 0.5, theta.size),
            "cvdrift0": np.linspace(-0.1, 0.1, theta.size),
            "gbdrift0": np.linspace(-0.1, 0.1, theta.size),
            "jacobian": jacobian,
            "grho": np.linspace(1.0, 1.4, theta.size),
        }.items():
            geom.createVariable(name, "f8", ("theta",))[:] = values
        for name, value in {
            "gradpar": 0.4,
            "q": 1.7,
            "shat": 0.6,
            "rmaj": 5.0,
            "aminor": 1.0,
            "kxfac": 1.3,
            "theta_scale": 2.0,
            "nfp": 5.0,
            "alpha": 0.2,
        }.items():
            geom.createVariable(name, "f8", ())[:] = value

    loaded = load_imported_geometry_netcdf(path)

    assert loaded.source_model == "imported-netcdf"
    assert loaded.theta_closed_interval is False
    assert jnp.allclose(loaded.theta, theta)
    assert jnp.allclose(loaded.jacobian_profile, jacobian)
    assert jnp.allclose(loaded.grho_profile, np.linspace(1.0, 1.4, theta.size))
    assert loaded.kxfac == pytest.approx(1.3)
    assert loaded.theta_scale == pytest.approx(2.0)
    assert loaded.nfp == 5
    assert loaded.epsilon == pytest.approx(0.2)


def test_load_imported_geometry_netcdf_reads_root_level_eik_layout(tmp_path):
    """Root-level eik.nc geometry should map into the sampled contract."""

    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom.eik.nc"
    theta = np.linspace(-np.pi, np.pi, 5)
    bmag = np.array([1.0, 1.1, 1.2, 1.1, 1.0])
    gds2 = np.array([1.0, 1.4, 1.8, 1.4, 1.0])
    gds21 = np.array([0.0, -0.2, 0.0, 0.2, 0.0])
    gds22 = np.full(theta.size, 0.8)
    cvdrift = np.array([0.3, 0.4, 0.5, 0.4, 0.3])
    cvdrift0 = np.array([0.0, -0.1, 0.0, 0.1, 0.0])
    drhodpsi = 1.7
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = bmag
        root.createVariable("gds2", "f8", ("z",))[:] = gds2
        root.createVariable("gds21", "f8", ("z",))[:] = gds21
        root.createVariable("gds22", "f8", ("z",))[:] = gds22
        root.createVariable("cvdrift", "f8", ("z",))[:] = cvdrift
        root.createVariable("gbdrift", "f8", ("z",))[:] = cvdrift
        root.createVariable("cvdrift0", "f8", ("z",))[:] = cvdrift0
        root.createVariable("gbdrift0", "f8", ("z",))[:] = cvdrift0
        root.createVariable("jacob", "f8", ("z",))[:] = np.linspace(
            2.0, 3.0, theta.size
        )
        root.createVariable("grho", "f8", ("z",))[:] = np.linspace(1.0, 1.4, theta.size)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, 0.4)
        root.createVariable("drhodpsi", "f8", ())[:] = drhodpsi
        root.createVariable("q", "f8", ())[:] = 1.7
        root.createVariable("shat", "f8", ())[:] = 0.6
        root.createVariable("Rmaj", "f8", ())[:] = 5.0
        root.createVariable("kxfac", "f8", ())[:] = 1.3
        root.createVariable("scale", "f8", ())[:] = 2.0
        root.createVariable("nfp", "f8", ())[:] = 5.0
        root.createVariable("alpha", "f8", ())[:] = 0.2
        root.createVariable("Rplot", "f8", ("z",))[:] = np.linspace(
            4.0, 6.0, theta.size
        )
        root.createVariable("Zplot", "f8", ("z",))[:] = np.linspace(
            -1.0, 1.0, theta.size
        )
        root.createVariable("theta_PEST", "f8", ("z",))[:] = theta + 0.1
        root.createVariable("zeta_center", "f8", ())[:] = 0.3

    loaded = load_imported_geometry_netcdf(path)

    assert loaded.source_model == "imported-netcdf"
    assert loaded.theta_closed_interval is True
    assert jnp.allclose(loaded.theta, theta)
    expected_jacobian = 1.0 / np.abs(drhodpsi * 0.4 * bmag)
    assert jnp.allclose(loaded.jacobian_profile, expected_jacobian)
    assert jnp.allclose(loaded.grho_profile, np.linspace(1.0, 1.4, theta.size))
    assert jnp.allclose(loaded.cv_profile, 0.5 * cvdrift)
    assert jnp.allclose(loaded.gb_profile, 0.5 * cvdrift)
    assert loaded.kxfac == pytest.approx(1.3)
    assert loaded.theta_scale == pytest.approx(2.0)
    assert loaded.nfp == 5
    assert loaded.R0 == pytest.approx(5.0)
    np.testing.assert_allclose(
        loaded.cylindrical_R_profile, np.linspace(4.0, 6.0, theta.size)
    )
    np.testing.assert_allclose(loaded.toroidal_angle_profile, 0.3 + 1.7 * (theta + 0.1))
    assert np.all(np.isfinite(np.asarray(loaded.bgrad_profile)))


def test_load_imported_geometry_netcdf_detects_open_root_level_eik_layout(tmp_path):
    """Root-level eik files can already be on the open solver grid."""

    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom_open.eik.nc"
    theta = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    bmag = np.linspace(1.0, 1.2, theta.size)
    drhodpsi = 1.7
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = bmag
        root.createVariable("gds2", "f8", ("z",))[:] = np.linspace(1.0, 2.0, theta.size)
        root.createVariable("gds21", "f8", ("z",))[:] = np.linspace(
            -0.2, 0.2, theta.size
        )
        root.createVariable("gds22", "f8", ("z",))[:] = np.full(theta.size, 0.8)
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.linspace(
            0.3, 0.5, theta.size
        )
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.linspace(
            0.3, 0.5, theta.size
        )
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.linspace(
            -0.1, 0.1, theta.size
        )
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.linspace(
            -0.1, 0.1, theta.size
        )
        root.createVariable("jacob", "f8", ("z",))[:] = np.linspace(
            2.0, 3.0, theta.size
        )
        root.createVariable("grho", "f8", ("z",))[:] = np.linspace(1.0, 1.4, theta.size)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, 0.4)
        root.createVariable("drhodpsi", "f8", ())[:] = drhodpsi
        root.createVariable("q", "f8", ())[:] = 1.7
        root.createVariable("shat", "f8", ())[:] = 0.6
        root.createVariable("Rmaj", "f8", ())[:] = 5.0
        root.createVariable("kxfac", "f8", ())[:] = 1.3
        root.createVariable("scale", "f8", ())[:] = 2.0
        root.createVariable("nfp", "f8", ())[:] = 5.0
        root.createVariable("alpha", "f8", ())[:] = 0.2

    loaded = load_imported_geometry_netcdf(path)

    assert loaded.source_model == "imported-netcdf"
    assert loaded.theta_closed_interval is False
    assert jnp.allclose(loaded.theta, theta)
    expected_jacobian = 1.0 / np.abs(drhodpsi * 0.4 * bmag)
    assert jnp.allclose(loaded.jacobian_profile, expected_jacobian)
    assert jnp.allclose(loaded.cv_profile, 0.5 * np.linspace(0.3, 0.5, theta.size))
    assert jnp.allclose(loaded.gb_profile, 0.5 * np.linspace(0.3, 0.5, theta.size))


def test_root_level_eik_import_matches_sampled_contract_after_trim(tmp_path):
    """VMEC-style closed-interval imported geometry should recover the open solver contract."""

    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    analytic = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, alpha=0.1)
    theta_closed = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 65)
    sampled_closed = sample_flux_tube_geometry(analytic, jnp.asarray(theta_closed))
    path = tmp_path / "geom.eik.nc"
    with Dataset(path, "w") as root:
        root.createDimension("z", theta_closed.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta_closed
        root.createVariable("bmag", "f8", ("z",))[:] = np.asarray(
            sampled_closed.bmag_profile
        )
        root.createVariable("gds2", "f8", ("z",))[:] = np.asarray(
            sampled_closed.gds2_profile
        )
        root.createVariable("gds21", "f8", ("z",))[:] = np.asarray(
            sampled_closed.gds21_profile
        )
        root.createVariable("gds22", "f8", ("z",))[:] = np.asarray(
            sampled_closed.gds22_profile
        )
        root.createVariable("cvdrift", "f8", ("z",))[:] = 2.0 * np.asarray(
            sampled_closed.cv_profile
        )
        root.createVariable("gbdrift", "f8", ("z",))[:] = 2.0 * np.asarray(
            sampled_closed.gb_profile
        )
        root.createVariable("cvdrift0", "f8", ("z",))[:] = 2.0 * np.asarray(
            sampled_closed.cv0_profile
        )
        root.createVariable("gbdrift0", "f8", ("z",))[:] = 2.0 * np.asarray(
            sampled_closed.gb0_profile
        )
        root.createVariable("jacob", "f8", ("z",))[:] = np.full(theta_closed.size, 7.0)
        root.createVariable("grho", "f8", ("z",))[:] = np.asarray(
            sampled_closed.grho_profile
        )
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(
            theta_closed.size, sampled_closed.gradpar_value
        )
        root.createVariable("drhodpsi", "f8", ())[:] = 1.0
        root.createVariable("q", "f8", ())[:] = sampled_closed.q
        root.createVariable("shat", "f8", ())[:] = sampled_closed.s_hat
        root.createVariable("Rmaj", "f8", ())[:] = sampled_closed.R0
        root.createVariable("kxfac", "f8", ())[:] = sampled_closed.kxfac
        root.createVariable("scale", "f8", ())[:] = sampled_closed.theta_scale
        root.createVariable("nfp", "f8", ())[:] = sampled_closed.nfp
        root.createVariable("alpha", "f8", ())[:] = sampled_closed.alpha

    loaded = load_imported_geometry_netcdf(path)
    theta_solver = jnp.asarray(theta_closed[:-1])
    loaded_open = ensure_flux_tube_geometry_data(loaded, theta_solver)
    sampled_open = ensure_flux_tube_geometry_data(sampled_closed, theta_solver)

    assert loaded.theta_closed_interval is True
    assert jnp.allclose(loaded_open.theta, sampled_open.theta)
    assert jnp.allclose(
        loaded_open.bmag_profile, sampled_open.bmag_profile, rtol=1.0e-6, atol=1.0e-6
    )
    assert jnp.allclose(
        loaded_open.gds2_profile, sampled_open.gds2_profile, rtol=1.0e-6, atol=1.0e-6
    )
    assert jnp.allclose(
        loaded_open.gds21_profile, sampled_open.gds21_profile, rtol=1.0e-6, atol=1.0e-6
    )
    assert jnp.allclose(
        loaded_open.gds22_profile, sampled_open.gds22_profile, rtol=1.0e-6, atol=1.0e-6
    )
    assert jnp.allclose(
        loaded_open.cv_profile, sampled_open.cv_profile, rtol=1.0e-6, atol=1.0e-6
    )
    assert jnp.allclose(
        loaded_open.gb_profile, sampled_open.gb_profile, rtol=1.0e-6, atol=1.0e-6
    )
    assert jnp.allclose(
        loaded_open.cv0_profile, sampled_open.cv0_profile, rtol=1.0e-6, atol=1.0e-6
    )
    assert jnp.allclose(
        loaded_open.gb0_profile, sampled_open.gb0_profile, rtol=1.0e-6, atol=1.0e-6
    )
    assert jnp.allclose(
        loaded_open.jacobian_profile,
        sampled_open.jacobian_profile,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert jnp.allclose(
        loaded_open.grho_profile, sampled_open.grho_profile, rtol=1.0e-6, atol=1.0e-6
    )
    assert jnp.allclose(
        loaded_open.bgrad_profile, sampled_open.bgrad_profile, rtol=1.0e-4, atol=1.0e-4
    )


def test_build_flux_tube_geometry_loads_imported_netcdf(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom.out.nc"
    theta = np.linspace(-np.pi, np.pi, 5, endpoint=False)
    with Dataset(path, "w") as root:
        root.createDimension("theta", theta.size)
        grids = root.createGroup("Grids")
        geom = root.createGroup("Geometry")
        grids.createVariable("theta", "f8", ("theta",))[:] = theta
        for name in (
            "bmag",
            "bgrad",
            "gds2",
            "gds21",
            "gds22",
            "cvdrift",
            "gbdrift",
            "cvdrift0",
            "gbdrift0",
            "jacobian",
            "grho",
        ):
            geom.createVariable(name, "f8", ("theta",))[:] = np.ones(theta.size)
        geom.createVariable("gradpar", "f8", ())[:] = 0.4
        geom.createVariable("q", "f8", ())[:] = 1.7
        geom.createVariable("shat", "f8", ())[:] = 0.6
        geom.createVariable("rmaj", "f8", ())[:] = 5.0
        geom.createVariable("aminor", "f8", ())[:] = 1.0

    loaded = build_flux_tube_geometry(
        GeometryConfig(model="imported-netcdf", geometry_file=str(path))
    )

    assert loaded.source_model == "imported-netcdf"


def test_build_flux_tube_geometry_rejects_missing_import_file_and_unknown_model() -> (
    None
):
    with pytest.raises(ValueError, match="geometry_file"):
        build_flux_tube_geometry(GeometryConfig(model="imported-netcdf"))
    with pytest.raises(ValueError, match="geometry.model"):
        build_flux_tube_geometry(GeometryConfig(model="banana"))


def test_twist_shift_params_slab_and_imported_geometry_branches() -> None:
    slab = SlabGeometry(s_hat=0.0)
    jtwist, x0 = twist_shift_params(
        slab, GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0, jtwist=0)
    )
    assert jtwist == 1
    assert x0 == pytest.approx(6.0 / (2.0 * np.pi))

    theta = jnp.linspace(-jnp.pi, jnp.pi, 5)
    imported = FluxTubeGeometryData(
        theta=theta,
        gradpar_value=1.0,
        bmag_profile=jnp.ones(5),
        bgrad_profile=jnp.zeros(5),
        gds2_profile=jnp.ones(5),
        gds21_profile=jnp.full(5, -0.3),
        gds22_profile=jnp.full(5, 0.4),
        cv_profile=jnp.zeros(5),
        gb_profile=jnp.zeros(5),
        cv0_profile=jnp.zeros(5),
        gb0_profile=jnp.zeros(5),
        jacobian_profile=jnp.ones(5),
        grho_profile=jnp.ones(5),
        q=1.0,
        s_hat=0.7,
        epsilon=0.0,
        R0=1.0,
    )
    jtwist, x0 = twist_shift_params(
        imported,
        GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0, y0=2.0, jtwist=None),
    )

    assert jtwist != 0
    assert x0 > 0.0


@pytest.mark.parametrize("model", ["imported-eik", "vmec-eik", "desc-eik", "eik"])
def test_build_flux_tube_geometry_accepts_imported_eik_aliases(tmp_path, model: str):
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom.eik.nc"
    theta = np.linspace(-np.pi, np.pi, 5)
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.array(
            [1.0, 1.1, 1.2, 1.1, 1.0]
        )
        root.createVariable("gds2", "f8", ("z",))[:] = np.array(
            [1.0, 1.5, 2.0, 1.5, 1.0]
        )
        root.createVariable("gds21", "f8", ("z",))[:] = np.array(
            [-0.2, 0.0, 0.2, 0.0, -0.2]
        )
        root.createVariable("gds22", "f8", ("z",))[:] = np.full(theta.size, 0.8)
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.array(
            [0.3, 0.4, 0.5, 0.4, 0.3]
        )
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.array(
            [0.3, 0.4, 0.5, 0.4, 0.3]
        )
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.array(
            [-0.1, 0.0, 0.1, 0.0, -0.1]
        )
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.array(
            [-0.1, 0.0, 0.1, 0.0, -0.1]
        )
        root.createVariable("jacob", "f8", ("z",))[:] = np.array(
            [2.0, 2.5, 3.0, 2.5, 2.0]
        )
        root.createVariable("grho", "f8", ("z",))[:] = np.array(
            [1.0, 1.2, 1.4, 1.2, 1.0]
        )
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, 0.4)
        root.createVariable("drhodpsi", "f8", ())[:] = 1.0
        root.createVariable("q", "f8", ())[:] = 1.7
        root.createVariable("shat", "f8", ())[:] = 0.6
        root.createVariable("Rmaj", "f8", ())[:] = 5.0
        root.createVariable("kxfac", "f8", ())[:] = 1.3
        root.createVariable("scale", "f8", ())[:] = 2.0
        root.createVariable("nfp", "f8", ())[:] = 5.0

    loaded = build_flux_tube_geometry(
        GeometryConfig(model=model, geometry_file=str(path))
    )

    assert not isinstance(loaded, (SAlphaGeometry, SlabGeometry))
    assert loaded.source_model == "imported-netcdf"
    assert loaded.theta_closed_interval is True
    assert loaded.theta_scale == pytest.approx(2.0)
    assert loaded.nfp == 5


def test_ensure_flux_tube_geometry_data_trims_closed_imported_vmec_grid(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom_vmec.eik.nc"
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 9)
    bmag_val = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1.0])
    jacob_val = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 2.6, 2.4, 2.2, 2.0])
    grho_val = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.2, 1.1, 1.0])
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = bmag_val
        root.createVariable("gds2", "f8", ("z",))[:] = np.array(
            [1.0, 1.2, 1.4, 1.6, 1.8, 1.6, 1.4, 1.2, 1.0]
        )
        root.createVariable("gds21", "f8", ("z",))[:] = np.array(
            [-0.2, -0.1, 0.0, 0.1, 0.2, 0.1, 0.0, -0.1, -0.2]
        )
        root.createVariable("gds22", "f8", ("z",))[:] = np.full(theta.size, 0.8)
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.array(
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3]
        )
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.array(
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3]
        )
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.array(
            [-0.1, -0.05, 0.0, 0.05, 0.1, 0.05, 0.0, -0.05, -0.1]
        )
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.array(
            [-0.1, -0.05, 0.0, 0.05, 0.1, 0.05, 0.0, -0.05, -0.1]
        )
        root.createVariable("jacob", "f8", ("z",))[:] = jacob_val
        root.createVariable("grho", "f8", ("z",))[:] = grho_val
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, 0.4)
        root.createVariable("q", "f8", ())[:] = 1.7
        root.createVariable("shat", "f8", ())[:] = 0.6
        root.createVariable("Rmaj", "f8", ())[:] = 5.0
        root.createVariable("scale", "f8", ())[:] = 2.0
        root.createVariable("nfp", "f8", ())[:] = 5.0

    geom = build_flux_tube_geometry(
        GeometryConfig(model="vmec-eik", geometry_file=str(path))
    )
    grid_cfg = apply_geometry_grid_defaults(
        geom,
        GridConfig(Nx=4, Ny=4, Nz=16, Lx=6.28, Ly=6.28, boundary="linked", y0=10.0),
    )
    grid = build_spectral_grid(grid_cfg)
    sampled = ensure_flux_tube_geometry_data(geom, grid.z)

    assert geom.theta_closed_interval is True
    assert sampled.theta_closed_interval is False
    assert sampled.theta.shape[0] == grid.z.size
    assert jnp.allclose(sampled.theta, jnp.asarray(grid.z))
    assert sampled.theta_scale == pytest.approx(2.0)
    assert sampled.nfp == 5
    expected_jacob = 1.0 / (0.4 * bmag_val[:-1])
    assert jnp.allclose(sampled.jacobian_profile, jnp.asarray(expected_jacob))
    assert jnp.allclose(sampled.grho_profile, jnp.asarray(grho_val[:-1]))


def test_apply_geometry_grid_defaults_uses_imported_theta_and_kxfac(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom.eik.nc"
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 9)
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gds2", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gds21", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gds22", "f8", ("z",))[:] = np.full(theta.size, 0.5)
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("jacob", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("grho", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, 0.4)
        root.createVariable("q", "f8", ())[:] = 1.7
        root.createVariable("shat", "f8", ())[:] = 0.5
        root.createVariable("Rmaj", "f8", ())[:] = 5.0
        root.createVariable("kxfac", "f8", ())[:] = 1.7

    geom = load_imported_geometry_netcdf(path)
    grid = GridConfig(Nx=4, Ny=4, Nz=16, Lx=6.28, Ly=6.28, boundary="linked", y0=10.0)
    adjusted = apply_geometry_grid_defaults(geom, grid)
    jtwist, x0 = twist_shift_params(geom, adjusted)

    assert adjusted.Nz == theta.size - 1
    assert adjusted.z_min == pytest.approx(theta[0])
    assert adjusted.z_max == pytest.approx(theta[-1])
    assert adjusted.kxfac == pytest.approx(1.7)
    assert adjusted.jtwist == jtwist
    assert adjusted.Lx == pytest.approx(2.0 * np.pi * x0)


def test_apply_geometry_grid_defaults_applies_twist_shift_for_fix_aspect(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom_fix_aspect.eik.nc"
    theta = np.linspace(-np.pi, np.pi, 9)
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gds2", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gds21", "f8", ("z",))[:] = np.full(theta.size, -0.6)
        root.createVariable("gds22", "f8", ("z",))[:] = np.full(theta.size, 0.2)
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("jacob", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("grho", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, 0.4)
        root.createVariable("q", "f8", ())[:] = 1.7
        root.createVariable("shat", "f8", ())[:] = 0.5
        root.createVariable("Rmaj", "f8", ())[:] = 5.0
        root.createVariable("kxfac", "f8", ())[:] = 1.0

    geom = load_imported_geometry_netcdf(path)
    grid = GridConfig(
        Nx=4, Ny=4, Nz=16, Lx=6.28, Ly=6.28, boundary="fix aspect", y0=10.0
    )
    adjusted = apply_geometry_grid_defaults(geom, grid)
    jtwist, x0 = twist_shift_params(geom, adjusted)

    assert adjusted.jtwist == jtwist
    assert adjusted.Lx == pytest.approx(2.0 * np.pi * x0)


def test_apply_geometry_grid_defaults_promotes_near_zero_shat_to_periodic():
    geom = SlabGeometry.from_config(GeometryConfig(model="slab", s_hat=1.0e-8))
    grid = GridConfig(
        Nx=1, Ny=4, Nz=16, Lx=62.8, Ly=2.0 * np.pi * 100.0, boundary="linked", y0=100.0
    )

    adjusted = apply_geometry_grid_defaults(geom, grid)

    assert adjusted.boundary == "periodic"
    assert adjusted.jtwist is None


def test_build_linear_cache_uses_linked_streaming_for_fix_aspect_imported_geometry(
    tmp_path,
):
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom_fix_aspect_cache.eik.nc"
    theta = np.linspace(-np.pi, np.pi, 9)
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gds2", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gds21", "f8", ("z",))[:] = np.full(theta.size, -0.6)
        root.createVariable("gds22", "f8", ("z",))[:] = np.full(theta.size, 0.2)
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("jacob", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("grho", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, 0.4)
        root.createVariable("q", "f8", ())[:] = 1.7
        root.createVariable("shat", "f8", ())[:] = 0.5
        root.createVariable("Rmaj", "f8", ())[:] = 5.0
        root.createVariable("kxfac", "f8", ())[:] = 1.0

    geom = load_imported_geometry_netcdf(path)
    grid_cfg = apply_geometry_grid_defaults(
        geom,
        GridConfig(Nx=4, Ny=4, Nz=16, Lx=6.28, Ly=6.28, boundary="fix aspect", y0=10.0),
    )
    grid = build_spectral_grid(grid_cfg)
    cache = build_linear_cache(grid, geom, LinearParams(), Nl=2, Nm=4)

    assert cache.use_twist_shift is True
    assert cache.jtwist != 0
    assert cache.linked_indices


def test_apply_geometry_grid_defaults_preserves_open_solver_theta(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom.out.nc"
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    with Dataset(path, "w") as root:
        root.createDimension("theta", theta.size)
        grids = root.createGroup("Grids")
        geom = root.createGroup("Geometry")
        grids.createVariable("theta", "f8", ("theta",))[:] = theta
        for name in (
            "bmag",
            "bgrad",
            "gds2",
            "gds21",
            "gds22",
            "cvdrift",
            "gbdrift",
            "cvdrift0",
            "gbdrift0",
            "jacobian",
            "grho",
        ):
            geom.createVariable(name, "f8", ("theta",))[:] = np.ones(theta.size)
        geom.createVariable("gradpar", "f8", ())[:] = 0.4
        geom.createVariable("q", "f8", ())[:] = 1.7
        geom.createVariable("shat", "f8", ())[:] = 0.5
        geom.createVariable("rmaj", "f8", ())[:] = 5.0
        geom.createVariable("kxfac", "f8", ())[:] = 1.7

    geom = load_imported_geometry_netcdf(path)
    grid = GridConfig(Nx=4, Ny=4, Nz=16, Lx=6.28, Ly=6.28, boundary="linked", y0=10.0)
    adjusted = apply_geometry_grid_defaults(geom, grid)

    spacing = theta[1] - theta[0]
    assert adjusted.Nz == theta.size
    assert adjusted.z_min == pytest.approx(theta[0])
    assert adjusted.z_max == pytest.approx(theta[-1] + spacing)


# ---- from test_differentiable_geometry.py ----


def _x64_enabled() -> bool:
    return bool(jax.config.read("jax_enable_x64"))


def test_observable_gradient_validation_report_passes_with_conditioning_metadata() -> (
    None
):
    dtype = jnp.float64 if _x64_enabled() else jnp.float32
    fd_step = 2.0e-5 if _x64_enabled() else 1.0e-3
    atol = 1.0e-7 if _x64_enabled() else 5.0e-4
    rtol = 5.0e-5 if _x64_enabled() else 5.0e-3
    params = jnp.asarray([0.24, -0.37], dtype=dtype)

    def observables(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            [
                x[0] * x[0] + 2.0 * x[1],
                jnp.sin(x[0]) + x[1] * x[1],
            ]
        )

    report = observable_gradient_validation_report(
        observables,
        params,
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
        observable_names=("quadratic_drive", "mixed_wave"),
        param_names=("ripple", "shear"),
        tangent=jnp.asarray([0.6, -0.8], dtype=dtype),
    )

    assert report["passed"] is True
    assert report["finite_passed"] is True
    assert report["derivative_tolerance_passed"] is True
    assert report["tangent_tolerance_passed"] is True
    assert report["conditioning_passed"] is True
    assert report["finite_flags"]["autodiff_jacobian"] is True
    assert report["finite_flags"]["finite_difference_jacobian"] is True
    assert report["conditioning"]["jacobian_shape"] == [2, 2]
    assert report["conditioning"]["sensitivity_map_rank"] == 2
    assert np.isfinite(float(report["conditioning"]["jacobian_condition_number"]))
    assert report["conditioning_gate"]["rank_passed"] is True
    assert report["tangent_direction_norm"] == pytest.approx(
        np.linalg.norm([0.6, -0.8])
    )
    assert np.isfinite(float(report["tangent_ad_norm"]))
    assert len(report["gradient_checks"]) == 4
    assert all(row["passed"] for row in report["gradient_checks"])
    json.dumps(report, allow_nan=False)


def test_observable_gradient_chunking_preserves_jacobian_and_records_policy() -> None:
    dtype = jnp.float64 if _x64_enabled() else jnp.float32
    params = jnp.linspace(0.1, 0.6, 6, dtype=dtype)

    def observables(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            [jnp.sum(jnp.sin(x)), jnp.dot(x, x), jnp.prod(1.0 + 0.1 * x)]
        )

    common = {
        "fd_step": 2.0e-4 if _x64_enabled() else 2.0e-3,
        "rtol": 2.0e-3 if _x64_enabled() else 2.0e-2,
        "atol": 2.0e-5 if _x64_enabled() else 2.0e-3,
        "condition_number_max": None,
    }
    full = observable_gradient_validation_report(
        observables, params, jacobian_chunk_size=None, **common
    )
    chunked = observable_gradient_validation_report(
        observables, params, jacobian_chunk_size=2, **common
    )

    assert full["jacobian_chunk_size"] is None
    assert chunked["jacobian_chunk_size"] == 2
    assert full["jacobian_mode"] == "reverse"
    assert chunked["jacobian_mode"] == "forward"
    assert full["passed"] is True and chunked["passed"] is True
    np.testing.assert_allclose(
        np.asarray(chunked["jacobian_ad"]),
        np.asarray(full["jacobian_ad"]),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    forward = observable_gradient_validation_report(
        observables, params, jacobian_mode="forward", **common
    )
    reverse = observable_gradient_validation_report(
        observables, params, jacobian_mode="reverse", **common
    )
    assert forward["jacobian_mode"] == "forward"
    assert reverse["jacobian_mode"] == "reverse"
    np.testing.assert_allclose(
        np.asarray(forward["jacobian_ad"]),
        np.asarray(reverse["jacobian_ad"]),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_observable_gradient_validation_rejects_invalid_mode_policy() -> None:
    def fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([jnp.sum(x)])

    params = jnp.ones(3)
    with pytest.raises(ValueError, match="jacobian_mode"):
        observable_gradient_validation_report(fn, params, jacobian_mode="adjoint")
    with pytest.raises(ValueError, match="only valid for forward"):
        observable_gradient_validation_report(
            fn, params, jacobian_mode="reverse", jacobian_chunk_size=1
        )


def test_observable_gradient_validation_report_fails_strict_json_on_nonfinite_data() -> (
    None
):
    def nonfinite_observables(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([x[0] / 0.0, x[1] * x[1]])

    report = observable_gradient_validation_report(
        nonfinite_observables,
        jnp.asarray([1.0, 0.2]),
        fd_step=1.0e-3,
        rtol=1.0e-3,
        atol=1.0e-5,
        observable_names=("bad", "finite"),
        param_names=("p0", "p1"),
    )

    assert report["passed"] is False
    assert report["finite_passed"] is False
    assert report["finite_flags"]["observables"] is False
    assert report["finite_flags"]["autodiff_jacobian"] is False
    assert any(
        str(reason).startswith("nonfinite:") for reason in report["failure_reasons"]
    )
    assert report["conditioning"]["finite_ad_jacobian"] is False
    assert report["conditioning_gate"]["passed"] is False
    json.dumps(report, allow_nan=False)


def test_observable_gradient_validation_report_fails_ill_conditioned_synthetic_map() -> (
    None
):
    def rank_deficient_observables(x: jnp.ndarray) -> jnp.ndarray:
        shared = x[0] + x[1]
        return jnp.asarray([shared, 2.0 * shared])

    report = observable_gradient_validation_report(
        rank_deficient_observables,
        jnp.asarray([0.1, -0.2]),
        fd_step=1.0e-3,
        rtol=1.0e-3,
        atol=1.0e-5,
        observable_names=("row0", "row1"),
        param_names=("p0", "p1"),
        min_rank=2,
        condition_number_max=1.0e6,
    )

    assert report["finite_passed"] is True
    assert report["derivative_tolerance_passed"] is True
    assert report["passed"] is False
    assert report["conditioning"]["sensitivity_map_rank"] == 1
    assert report["conditioning_gate"]["rank_passed"] is False
    assert report["conditioning_gate"]["condition_number_passed"] is False
    assert "rank_below_required" in report["failure_reasons"]
    assert "ill_conditioned" in report["failure_reasons"]
    json.dumps(report, allow_nan=False)


# ---- from test_miller_geometry.py ----
# Miller geometry backend, eik-file, and low-level kernel tests.


def test_nperiod_helpers_contract_arrays() -> None:
    theta = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    values = np.arange(theta.size)
    mask = np.asarray(nperiod_mask(theta, 1.0))
    assert mask.tolist() == [False, True, True, True, False]
    contracted_values, contracted_theta = nperiod_contract.__wrapped__(
        values, theta, 1.0
    )
    np.testing.assert_allclose(np.asarray(contracted_values), [1, 2, 3])
    np.testing.assert_allclose(np.asarray(contracted_theta), [-1.0, 0.0, 1.0])


def test_finite_diff_nonuniform_matches_quadratic_derivative() -> None:
    grid = np.array([0.0, 0.5, 1.5, 3.0])
    values = grid**2
    diff = np.asarray(finite_diff_nonuniform(values, grid))
    np.testing.assert_allclose(diff[1:-1], 2.0 * grid[1:-1], atol=1.0e-6)
    assert np.isfinite(diff[[0, -1]]).all()
    np.testing.assert_allclose(
        np.asarray(finite_diff_nonuniform(np.array([1.0, 2.0]), np.array([0.0, 1.0]))),
        [0.0, 0.0],
    )


def test_centered_reflected_difference_handles_1d_and_2d_axes() -> None:
    arr = np.array([1.0, 2.0, 4.0, 7.0])
    np.testing.assert_allclose(
        np.asarray(centered_reflected_difference(arr, axis="l", parity="e")),
        [0.0, 3.0, 5.0, 0.0],
    )
    np.testing.assert_allclose(
        np.asarray(centered_reflected_difference(arr, axis="r")), [2.0, 3.0, 5.0, 6.0]
    )
    arr2 = np.arange(12.0).reshape(3, 4)
    out_r = np.asarray(centered_reflected_difference(arr2, axis="r"))
    out_l = np.asarray(centered_reflected_difference(arr2, axis="l", parity="o"))
    assert out_r.shape == arr2.shape
    assert out_l.shape == arr2.shape
    with pytest.raises(ValueError):
        centered_reflected_difference(np.ones((2, 2, 2)), axis="l")
    with pytest.raises(ValueError):
        centered_reflected_difference(arr, axis="bad")


def test_weighted_centered_difference_handles_weighted_derivatives() -> None:
    grid = np.array([0.0, 1.0, 2.0, 4.0])
    values = grid**2
    out_l = np.asarray(weighted_centered_difference(values, grid, axis="l", parity="o"))
    assert out_l.shape == values.shape
    assert np.isfinite(out_l).all()

    arr2 = np.vstack([values, values + 1.0])
    grid2 = np.vstack([grid, grid])
    out2 = np.asarray(weighted_centered_difference(arr2, grid2, axis="r"))
    assert out2.shape == arr2.shape

    with pytest.raises(ValueError):
        weighted_centered_difference(np.ones(3), np.ones(4), axis="l")
    with pytest.raises(ValueError):
        weighted_centered_difference(np.ones((2, 2, 2)), np.ones((2, 2, 2)), axis="l")
    with pytest.raises(ValueError):
        weighted_centered_difference(values, grid, axis="bad")


def test_nperiod_extension_and_reflection_helpers() -> None:
    base = np.array([0.0, 1.0, 2.0])
    even = np.asarray(extend_nperiod_data(base, 2, istheta=False, parity="e"))
    odd = np.asarray(extend_nperiod_data(base, 2, istheta=False, parity="o"))
    theta = np.asarray(extend_nperiod_data(base, 2, istheta=True))
    assert even.size == 7
    assert odd.size == 7
    assert theta.size == 7
    np.testing.assert_allclose(
        np.asarray(reflect_and_append(base, "e")), [2.0, 1.0, 0.0, 1.0, 2.0]
    )
    np.testing.assert_allclose(
        np.asarray(reflect_and_append(base, "o")), [-2.0, -1.0, 0.0, 1.0, 2.0]
    )


def test_backend_kernel_exports_canonical_helpers() -> None:
    arr = np.array([1.0, 2.0, 4.0, 7.0])
    np.testing.assert_allclose(
        np.asarray(centered_reflected_difference(arr, axis="r")),
        np.asarray(centered_reflected_difference(arr, axis="r")),
    )
    assert callable(centered_reflected_difference)


# Internal Miller backend request, collocation, and NetCDF contracts.


from gkx.geometry.analytic import MillerCoreParams, build_collocation_surfaces
from gkx.geometry.kernels import _safe_denom, cumulative_trapezoid
from gkx.geometry.imported_miller import (
    _request_attr,
    generate_miller_eik_internal,
)


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        ntheta=24,
        nperiod=1,
        rhoc=0.5,
        q=1.4,
        s_hat=0.8,
        R0=3.0,
        R_geo=3.0,
        shift=0.0,
        akappa=1.0,
        tri=0.0,
        akappri=0.0,
        tripri=0.0,
        betaprim=0.0,
    )


def test_safe_denom_and_request_attr() -> None:
    assert _safe_denom(0.0) > 0.0
    assert _safe_denom(-0.0) > 0.0
    arr = _safe_denom(np.array([0.0, -1.0e-40, 2.0]))
    assert np.all(np.abs(arr) > 0.0)

    req = SimpleNamespace(q=1.4, s_hat=0.9)
    assert _request_attr(req, "qinp", "q") == 1.4
    assert _request_attr(req, "shat", "s_hat") == 0.9
    with pytest.raises(AttributeError):
        _request_attr(req, "missing_a", "missing_b")


def test_cumulative_trapezoid_supports_1d_and_2d() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(cumulative_trapezoid(y, x), [0.0, 0.5, 2.0])

    y2 = np.vstack([y, 2.0 * y])
    out = cumulative_trapezoid(y2, x, axis=1)
    np.testing.assert_allclose(out[0], [0.0, 0.5, 2.0])
    np.testing.assert_allclose(out[1], [0.0, 1.0, 4.0])

    x2 = np.vstack([x, x + 0.5])
    out2 = cumulative_trapezoid(y2, x2, axis=1)
    assert out2.shape == y2.shape

    with pytest.raises(ValueError):
        cumulative_trapezoid(np.array([1.0, 2.0]), np.array([[0.0, 1.0]]))
    with pytest.raises(NotImplementedError):
        cumulative_trapezoid(np.ones((2, 2)), np.ones(2), axis=0)
    with pytest.raises(ValueError):
        cumulative_trapezoid(np.ones((2, 2, 2)), np.ones((2, 2, 2)))


def test_miller_collocation_preserves_signed_shift_derivative() -> None:
    params = MillerCoreParams(
        ntgrid=8,
        nperiod=1,
        rhoc=0.5,
        qinp=1.4,
        shat=0.8,
        rmaj=3.0,
        r_geo=3.0,
        shift=-0.2,
        akappa=1.0,
        tri=0.0,
        akappri=0.0,
        tripri=0.0,
        betaprim=0.0,
        delrho=1.0e-3,
    )

    state = build_collocation_surfaces(params)
    r_midplane = np.asarray(state["r"])[:, 0]
    rho = np.asarray(state["rho"])
    d_rgeom_drho = np.gradient(r_midplane - rho, rho)[1]

    assert d_rgeom_drho == pytest.approx(params.shift, rel=1.0e-6)


def test_generate_miller_eik_internal_writes_netcdf(tmp_path: Path) -> None:
    out = generate_miller_eik_internal(
        output_path=tmp_path / "miller.eiknc.nc", request=_request()
    )
    assert out.exists()

    import netCDF4 as nc

    with nc.Dataset(out) as ds:
        assert "theta" in ds.variables
        assert "bmag" in ds.variables
        assert ds.variables["theta"][:].size > 0
        assert float(ds.variables["q"].getValue()) == pytest.approx(1.4)
        assert float(ds.variables["shat"].getValue()) == pytest.approx(0.8)


# Runtime Miller eik request and generation contracts.


from gkx.geometry.miller_eik import (
    build_miller_geometry_request,
    generate_runtime_miller_eik,
)


def _miller_runtime_cfg(
    tmp_path: Path, *, geometry_file: str | None = None
) -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(
            Nx=32,
            Ny=16,
            Nz=24,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            ntheta=24,
            nperiod=1,
        ),
        time=TimeConfig(t_max=1.0, dt=0.1, method="rk3", fixed_dt=True),
        geometry=GeometryConfig(
            model="miller",
            geometry_file=geometry_file,
            q=1.4,
            s_hat=0.8,
            rhoc=0.5,
            R0=2.77778,
            R_geo=2.77778,
            shift=0.0,
            akappa=1.0,
            akappri=0.0,
            tri=0.0,
            tripri=0.0,
            betaprim=0.0,
        ),
        init=InitializationConfig(init_field="density", init_amp=1.0e-6),
        species=(
            RuntimeSpeciesConfig(
                name="ion", charge=1.0, mass=1.0, tprim=2.49, fprim=0.8
            ),
        ),
        physics=RuntimePhysicsConfig(
            linear=False,
            nonlinear=True,
            adiabatic_electrons=True,
            tau_e=1.0,
            electrostatic=True,
            electromagnetic=False,
            beta=0.0,
            collisions=False,
        ),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="rho_star"
        ),
    )


def test_build_miller_geometry_request_creates_expected_request(tmp_path: Path) -> None:
    cfg = _miller_runtime_cfg(tmp_path)
    request = build_miller_geometry_request(cfg)

    assert request.q == 1.4
    assert request.s_hat == 0.8
    assert request.rhoc == 0.5
    assert request.ntheta == 24
    assert request.nperiod == 1


def test_generate_runtime_miller_eik_invokes_internal_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "geom.eiknc.nc"
    cfg = _miller_runtime_cfg(tmp_path, geometry_file=str(out_path))

    mock_gen = MagicMock(return_value=out_path.resolve())
    monkeypatch.setattr(
        "gkx.geometry.miller_eik.generate_miller_eik_internal", mock_gen
    )
    out_path.write_bytes(b"stale")
    out = generate_runtime_miller_eik(cfg, force=True)

    assert out == out_path.resolve()
    assert mock_gen.called
    _, kwargs = mock_gen.call_args
    request = kwargs["request"]
    assert request.ntheta == 24
    assert request.q == 1.4


def test_generate_runtime_miller_eik_reuses_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "cached.eiknc.nc"
    out_path.write_bytes(b"cached")
    cfg = _miller_runtime_cfg(tmp_path, geometry_file=str(out_path))
    mock_gen = MagicMock()
    monkeypatch.setattr(
        "gkx.geometry.miller_eik.generate_miller_eik_internal", mock_gen
    )

    out = generate_runtime_miller_eik(cfg)

    assert out == out_path.resolve()
    mock_gen.assert_not_called()


def test_internal_miller_request_attr_accepts_runtime_aliases() -> None:
    class Req:
        q = 1.4
        s_hat = 0.8

    req = Req()
    assert _request_attr(req, "qinp", "q") == 1.4
    assert _request_attr(req, "shat", "s_hat") == 0.8


# ---- from test_vmec_eik.py ----


def _vmec_runtime_cfg(
    tmp_path: Path, *, geometry_file: str | None = None
) -> RuntimeConfig:
    vmec_path = tmp_path / "wout_test.nc"
    vmec_path.write_text("stub", encoding="utf-8")
    return RuntimeConfig(
        grid=GridConfig(
            Nx=1,
            Ny=8,
            Nz=32,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            ntheta=32,
            nperiod=1,
        ),
        time=TimeConfig(t_max=1.0, dt=0.1, method="rk4", fixed_dt=True),
        geometry=GeometryConfig(
            model="vmec",
            vmec_file=str(vmec_path),
            geometry_file=geometry_file,
            torflux=0.64,
            npol=2.0,
            alpha=0.1,
        ),
        init=InitializationConfig(init_field="density", init_amp=1.0e-6),
        species=(
            RuntimeSpeciesConfig(
                name="ion", charge=1.0, mass=1.0, tprim=3.0, fprim=1.0
            ),
        ),
        physics=RuntimePhysicsConfig(
            linear=True,
            nonlinear=False,
            adiabatic_electrons=True,
            tau_e=1.0,
            electrostatic=True,
            electromagnetic=False,
            beta=0.0,
            collisions=False,
        ),
        normalization=RuntimeNormalizationConfig(
            contract="kinetic", diagnostic_norm="rho_star"
        ),
    )


def _write_minimal_eik_cache(path: Path) -> None:
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("z", 1)
        for name in ("theta", "bmag", "gradpar"):
            ds.createVariable(name, "f8", ("z",))[:] = [1.0]
        ds.createVariable("q", "f8").assignValue(1.4)
        ds.createVariable("shat", "f8").assignValue(0.8)


def test_build_vmec_geometry_request_creates_expected_request(tmp_path: Path) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    request = build_vmec_geometry_request(cfg)

    assert request.vmec_file == str(Path(cfg.geometry.vmec_file).resolve())
    assert request.torflux == 0.64
    assert request.npol == 2.0
    assert request.ntheta == 32
    assert request.alpha == 0.1
    assert request.y0 == pytest.approx(10.0)
    assert request.z == (1.0, -1.0)  # Ion + adiabatic electron


def test_build_vmec_geometry_request_expands_env_vmec_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vmec_path = tmp_path / "wout_env.nc"
    vmec_path.write_text("stub", encoding="utf-8")
    monkeypatch.setenv("HSX_VMEC_FILE", str(vmec_path))
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg = replace(cfg, geometry=replace(cfg.geometry, vmec_file="$HSX_VMEC_FILE"))

    request = build_vmec_geometry_request(cfg)

    assert request.vmec_file == str(vmec_path.resolve())


@pytest.mark.integration
def test_vmec_roundtrip_gate_is_deterministic(tmp_path: Path) -> None:
    vmec_file = os.environ.get("GKX_VMEC_FILE", "").strip()
    if not vmec_file:
        pytest.skip("Set GKX_VMEC_FILE to enable VMEC roundtrip parity gate.")

    geometry_helper_repo = (
        os.environ.get("GKX_GEOMETRY_HELPER_REPO", "").strip() or None
    )
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg = replace(
        cfg,
        grid=replace(cfg.grid, Nz=64, ntheta=64),
        geometry=replace(
            cfg.geometry,
            vmec_file=vmec_file,
            geometry_file=None,
            npol=1.0,
            alpha=0.0,
            geometry_helper_repo=geometry_helper_repo,
        ),
    )

    out1 = tmp_path / "geom1.eik.nc"
    out2 = tmp_path / "geom2.eik.nc"
    generate_runtime_vmec_eik(cfg, output_path=out1, force=True)
    generate_runtime_vmec_eik(cfg, output_path=out2, force=True)

    g1 = load_imported_geometry_netcdf(out1)
    g2 = load_imported_geometry_netcdf(out2)

    # This is a determinism gate, not a physics gate: any drift here will break
    # VMEC-backed parity workflows in hard-to-debug ways.
    for name in (
        "theta",
        "bmag_profile",
        "gds2_profile",
        "gds21_profile",
        "gds22_profile",
        "cv_profile",
        "gb_profile",
        "jacobian_profile",
        "grho_profile",
    ):
        a = np.asarray(getattr(g1, name))
        b = np.asarray(getattr(g2, name))
        np.testing.assert_allclose(a, b, rtol=0.0, atol=0.0)


def test_build_vmec_geometry_request_infers_npol_from_nperiod(tmp_path: Path) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg = replace(
        cfg,
        grid=replace(cfg.grid, nperiod=3),
        geometry=replace(cfg.geometry, npol=None),
    )

    request = build_vmec_geometry_request(cfg)

    assert request.npol == pytest.approx(5.0)


def test_default_vmec_eik_output_path_tracks_vmec_file_metadata(tmp_path: Path) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    request = build_vmec_geometry_request(cfg)

    first = default_vmec_eik_output_path(request)
    vmec_path = Path(request.vmec_file)
    vmec_path.write_text("updated-stub", encoding="utf-8")
    os.utime(
        vmec_path,
        ns=(vmec_path.stat().st_atime_ns, vmec_path.stat().st_mtime_ns + 1_000_000),
    )

    second = default_vmec_eik_output_path(request)

    assert first.parent.name == "vmec_eik"
    assert first.suffixes == [".eik", ".nc"]
    assert first != second


def test_atomic_vmec_eik_write_replaces_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "geom.eik.nc"
    out_path.write_text("old", encoding="utf-8")
    temp_paths: list[Path] = []

    def fake_write(path: Path, _profiles: dict, *, request: object) -> None:
        assert request == "request"
        assert path != out_path
        temp_paths.append(path)
        path.write_text("new", encoding="utf-8")

    monkeypatch.setattr("gkx.geometry.imported_vmec.write_vmec_eik_netcdf", fake_write)

    vmec_backend._write_vmec_eik_netcdf_atomically(out_path, {}, request="request")

    assert out_path.read_text(encoding="utf-8") == "new"
    assert temp_paths
    assert not temp_paths[0].exists()


def test_generate_runtime_vmec_eik_invokes_internal_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "geom.eik.nc"
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=str(out_path))

    mock_gen = MagicMock(return_value=out_path.resolve())
    monkeypatch.setattr("gkx.geometry.vmec_eik.generate_vmec_eik_internal", mock_gen)
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.internal_vmec_backend_available", lambda: True
    )

    out = generate_runtime_vmec_eik(cfg)

    assert out == out_path.resolve()
    assert mock_gen.called
    _, kwargs = mock_gen.call_args
    request = kwargs["request"]
    assert request.ntheta == 32
    assert request.vmec_file == str(Path(cfg.geometry.vmec_file).resolve())


def test_generate_runtime_vmec_eik_reuses_default_cache_without_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=None)
    expected_out = tmp_path / "cached.eik.nc"
    _write_minimal_eik_cache(expected_out)

    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.default_vmec_eik_output_path",
        lambda _request: expected_out,
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.internal_vmec_backend_available", lambda: False
    )
    mock_gen = MagicMock(
        side_effect=AssertionError("cached VMEC geometry should be reused")
    )
    monkeypatch.setattr("gkx.geometry.vmec_eik.generate_vmec_eik_internal", mock_gen)

    out = generate_runtime_vmec_eik(cfg)

    assert out == expected_out.resolve()
    assert not mock_gen.called


def test_generate_runtime_vmec_eik_regenerates_invalid_default_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=None)
    expected_out = tmp_path / "cached.eik.nc"
    expected_out.write_text("partial-not-netcdf", encoding="utf-8")

    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.default_vmec_eik_output_path",
        lambda _request: expected_out,
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.internal_vmec_backend_available", lambda: True
    )
    mock_gen = MagicMock(return_value=expected_out.resolve())
    monkeypatch.setattr("gkx.geometry.vmec_eik.generate_vmec_eik_internal", mock_gen)

    out = generate_runtime_vmec_eik(cfg)

    assert out == expected_out.resolve()
    assert mock_gen.called


def test_generate_runtime_vmec_eik_uses_default_output_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path, geometry_file=None)
    expected_out = tmp_path / "cached.eik.nc"

    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.default_vmec_eik_output_path",
        lambda _request: expected_out,
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.internal_vmec_backend_available", lambda: True
    )
    mock_gen = MagicMock(return_value=expected_out.resolve())
    monkeypatch.setattr("gkx.geometry.vmec_eik.generate_vmec_eik_internal", mock_gen)

    out = generate_runtime_vmec_eik(cfg)

    assert out == expected_out.resolve()
    _, kwargs = mock_gen.call_args
    assert Path(kwargs["output_path"]) == expected_out


@pytest.mark.parametrize(
    ("backend", "expected_message"),
    [
        ("mystery", "Unknown geometry backend"),
    ],
)
def test_generate_runtime_vmec_eik_rejects_invalid_backends(
    tmp_path: Path, backend: str, expected_message: str
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    cfg = replace(cfg, geometry=replace(cfg.geometry, geometry_backend=backend))

    with pytest.raises(ValueError, match=expected_message):
        generate_runtime_vmec_eik(cfg)


def test_generate_runtime_vmec_eik_requires_internal_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _vmec_runtime_cfg(tmp_path)
    monkeypatch.setattr(
        "gkx.geometry.vmec_eik.internal_vmec_backend_available", lambda: False
    )

    with pytest.raises(
        RuntimeError, match="Internal VMEC geometry backend dependencies are missing"
    ):
        generate_runtime_vmec_eik(cfg)


def test_internal_vmec_backend_available_detects_env_provided_booz_xform_jax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg_root = tmp_path / "booz_xform_jax_checkout"
    pkg_dir = pkg_root / "src" / "booz_xform_jax"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text(
        "class Booz_xform:\n    pass\n", encoding="utf-8"
    )

    for name in ("booz_xform_jax", "booz_xform"):
        sys.modules.pop(name, None)
    original_sys_path = list(sys.path)
    monkeypatch.setenv("GKX_BOOZ_XFORM_JAX_PATH", str(pkg_root))
    monkeypatch.delenv("BOOZ_XFORM_JAX_PATH", raising=False)

    try:
        assert internal_vmec_backend_available() is True
    finally:
        sys.path[:] = original_sys_path
        for name in ("booz_xform_jax", "booz_xform"):
            sys.modules.pop(name, None)


# ---- from test_vmec_state_controls_coverage.py ----
# Unit contracts for the vmex state-control helpers in ``vmec_state_controls``.
#
# These exercise the pure coefficient accessor / replacement / index-resolution
# controls that drive the differentiable VMEC-Boozer sensitivity gates.  Every
# case uses synthetic ``SpectralState``-like frozen dataclasses and small arrays
# so no real equilibrium solve is required; numeric expectations, resolved index


# ---------------------------------------------------------------------------
# Synthetic-input factories
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _FakeSpectralState:
    """Minimal frozen vmex-``SpectralState`` stand-in for replace/perturb gates.

    ``R_sin`` is a witness family that no tested control touches; it must
    survive ``dataclasses.replace`` unchanged.
    """

    R_cos: jnp.ndarray
    Z_sin: jnp.ndarray
    R_sin: jnp.ndarray


class _ClosableDataset:
    """netCDF-like handle that records exactly one ``close()``."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _solved_case_bundle(r_cos: object, z_sin: object) -> tuple[object, ...]:
    """Return an ``(inp, state, runtime, wout)`` bundle like a solved vmex case."""

    inp = SimpleNamespace(tag="inp")
    runtime = SimpleNamespace(tag="runtime")
    wout = SimpleNamespace(tag="wout")
    state = SimpleNamespace(R_cos=r_cos, Z_sin=z_sin)
    return inp, state, runtime, wout


_SQUARE_LAYOUT_MESSAGE = (
    "rmnc0 has unexpected shape (50, 50); one dimension must equal ns=50"
)


# ---------------------------------------------------------------------------
# _new_boozer_object_with_auto_fallback: classic-reader fallback policy
# ---------------------------------------------------------------------------
def test_auto_fallback_closes_dataset_when_classic_reader_also_fails(monkeypatch):
    """A square-layout failure tries the classic reader, then re-raises + closes."""

    monkeypatch.delenv("GKX_BOOZ_BACKEND", raising=False)
    calls = {"new_booz": 0, "import_preferred": "<unset>"}

    def _always_square(_backend: object, _path: object) -> object:
        calls["new_booz"] += 1
        raise ValueError(_SQUARE_LAYOUT_MESSAGE)

    def _fake_import(preferred: str | None = None) -> object:
        calls["import_preferred"] = preferred
        return SimpleNamespace(name="classic-booz")

    monkeypatch.setattr(controls, "_new_booz_object", _always_square)
    monkeypatch.setattr(controls, "_import_booz_backend", _fake_import)

    nc_obj = _ClosableDataset()
    with pytest.raises(ValueError, match="rmnc0 has unexpected shape"):
        controls._new_boozer_object_with_auto_fallback(
            SimpleNamespace(name="jax-booz"), Path("square.nc"), nc_obj
        )

    # Primary attempt + classic fallback attempt both ran; the classic reader
    # was requested explicitly, and the dataset handle is closed on failure.
    assert calls["new_booz"] == 2
    assert calls["import_preferred"] == "booz_xform"
    assert nc_obj.closed is True


@pytest.mark.parametrize(
    ("env_backend", "error"),
    [
        # auto mode, but a non-square read failure must not touch the classic reader
        (None, ValueError("wout file is corrupt")),
        # square failure, but an explicitly forced backend stays fail-fast
        ("jax", ValueError(_SQUARE_LAYOUT_MESSAGE)),
    ],
)
def test_auto_fallback_skips_classic_reader_and_closes(monkeypatch, env_backend, error):
    """Only auto-mode square-layout failures may fall back to booz_xform."""

    if env_backend is None:
        monkeypatch.delenv("GKX_BOOZ_BACKEND", raising=False)
    else:
        monkeypatch.setenv("GKX_BOOZ_BACKEND", env_backend)

    def _raise(_backend: object, _path: object) -> object:
        raise error

    def _forbidden_import(preferred: str | None = None) -> object:
        raise AssertionError("classic booz_xform reader must not be imported")

    monkeypatch.setattr(controls, "_new_booz_object", _raise)
    monkeypatch.setattr(controls, "_import_booz_backend", _forbidden_import)

    nc_obj = _ClosableDataset()
    with pytest.raises(ValueError):
        controls._new_boozer_object_with_auto_fallback(
            SimpleNamespace(name="jax-booz"), Path("wout.nc"), nc_obj
        )
    assert nc_obj.closed is True


# ---------------------------------------------------------------------------
# _load_vmec_state_context: solved-case wiring + 2-D validation
# ---------------------------------------------------------------------------
def test_load_vmec_state_context_exposes_differentiable_state_arrays(monkeypatch):
    r_cos = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    )
    z_sin = np.array([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 1.0, 1.1]])
    bundle = _solved_case_bundle(r_cos, z_sin)
    seen: dict[str, str] = {}

    def _fake_resolve(name: str) -> Path:
        seen["resolve"] = name
        return Path("input.synthetic")

    def _fake_load(name: str) -> tuple[object, ...]:
        seen["load"] = name
        return bundle

    monkeypatch.setattr(controls, "resolve_vmex_case_input_path", _fake_resolve)
    monkeypatch.setattr(controls, "load_solved_vmex_case", _fake_load)

    ctx = controls._load_vmec_state_context("synthetic_case")

    # The case name is forwarded (as a string) to both resolver and loader.
    assert seen == {"resolve": "synthetic_case", "load": "synthetic_case"}
    assert ctx.input_path == Path("input.synthetic")
    assert ctx.wout_path == controls.VMEC_STATE_IN_MEMORY_WOUT_PATH
    assert ctx.inp is bundle[0]
    assert ctx.state is bundle[1]
    assert ctx.runtime is bundle[2]
    assert ctx.wout is bundle[3]
    assert ctx.base_Rcos.shape == (3, 4)
    assert ctx.base_Zsin.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(ctx.base_Rcos), r_cos)
    np.testing.assert_allclose(np.asarray(ctx.base_Zsin), z_sin)


def test_load_vmec_state_context_rejects_non_2d_state_arrays(monkeypatch):
    bundle = _solved_case_bundle(np.ones(4), np.ones((3, 4)))  # R_cos is 1-D

    monkeypatch.setattr(
        controls, "resolve_vmex_case_input_path", lambda name: Path("x")
    )
    monkeypatch.setattr(controls, "load_solved_vmex_case", lambda name: bundle)

    with pytest.raises(
        RuntimeError, match="R_cos/Z_sin arrays must be two-dimensional"
    ):
        controls._load_vmec_state_context("synthetic_case")


# ---------------------------------------------------------------------------
# _vmec_state_family_attribute + _vmec_boozer_state_array
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("family", "attribute"),
    [
        ("Rcos", "R_cos"),
        ("Rsin", "R_sin"),
        ("Zcos", "Z_cos"),
        ("Zsin", "Z_sin"),
        ("Lcos", "L_cos"),
        ("Lsin", "L_sin"),
    ],
)
def test_vmec_state_family_attribute_maps_public_family_to_state_attr(
    family, attribute
):
    assert controls._vmec_state_family_attribute(family) == attribute


def test_vmec_state_family_attribute_rejects_unknown_family():
    with pytest.raises(ValueError, match="parameter_family must be one of"):
        controls._vmec_state_family_attribute("Bcos")


def test_vmec_boozer_state_array_returns_validated_family_table():
    state = SimpleNamespace(
        R_cos=np.arange(12.0).reshape(3, 4),
        Z_sin=np.arange(6.0).reshape(3, 2),
    )

    r_table = controls._vmec_boozer_state_array(state, "Rcos")
    assert r_table.shape == (3, 4)
    np.testing.assert_allclose(np.asarray(r_table), np.arange(12.0).reshape(3, 4))

    z_table = controls._vmec_boozer_state_array(state, "Zsin")
    assert z_table.shape == (3, 2)
    np.testing.assert_allclose(np.asarray(z_table), np.arange(6.0).reshape(3, 2))


def test_vmec_boozer_state_array_reports_missing_family_attribute():
    state = SimpleNamespace(Z_sin=np.ones((3, 4)))  # no R_cos attribute

    with pytest.raises(RuntimeError, match="vmex state does not expose R_cos"):
        controls._vmec_boozer_state_array(state, "Rcos")


@pytest.mark.parametrize("bad", [np.ones(4), np.ones((3, 1))])
def test_vmec_boozer_state_array_requires_two_dim_non_axisymmetric_mode(bad):
    state = SimpleNamespace(R_cos=bad)

    with pytest.raises(
        RuntimeError,
        match="R_cos array must expose at least one non-axisymmetric mode",
    ):
        controls._vmec_boozer_state_array(state, "Rcos")


# ---------------------------------------------------------------------------
# _replace_vmec_boozer_state_coefficient (get/replace round-trip)
# ---------------------------------------------------------------------------
def test_replace_vmec_boozer_state_coefficient_round_trips_single_entry():
    base = jnp.asarray(np.arange(12.0).reshape(3, 4))
    witness = jnp.asarray(np.full((3, 4), 7.0))
    state = _FakeSpectralState(R_cos=base, Z_sin=jnp.zeros((3, 4)), R_sin=witness)

    fetched = controls._vmec_boozer_state_array(state, "Rcos")
    np.testing.assert_allclose(np.asarray(fetched), np.asarray(base))

    new_state = controls._replace_vmec_boozer_state_coefficient(
        state, "Rcos", fetched, radial_index=2, mode_index=3, delta=0.25
    )

    expected = np.arange(12.0).reshape(3, 4)
    expected[2, 3] += 0.25
    np.testing.assert_allclose(
        np.asarray(controls._vmec_boozer_state_array(new_state, "Rcos")), expected
    )
    # Untouched family, witness field, and the original frozen state are intact.
    np.testing.assert_allclose(np.asarray(new_state.Z_sin), np.zeros((3, 4)))
    assert new_state.R_sin is witness
    np.testing.assert_allclose(np.asarray(state.R_cos), np.arange(12.0).reshape(3, 4))


# ---------------------------------------------------------------------------
# _vmec_boozer_state_parameter_name (mid-surface vs off-surface naming)
# ---------------------------------------------------------------------------
def test_vmec_boozer_state_parameter_name_switches_on_mid_surface():
    assert (
        controls._vmec_boozer_state_parameter_name("Rcos", 4, 2, default_mid_surface=4)
        == "Rcos_mid_surface_m2"
    )
    assert (
        controls._vmec_boozer_state_parameter_name("Zsin", 3, 1, default_mid_surface=4)
        == "Zsin_r3_m1"
    )


# ---------------------------------------------------------------------------
# _resolve_vmec_state_indices (default resolution, clamps, validation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    (
        "shape",
        "radial_index",
        "mode_index",
        "surface_index",
        "surface_grid",
        "expected",
    ),
    [
        # Default radial index is ns // 2; each grid has its own default surface.
        ((8, 5), None, 2, None, "half_mesh", (4, 2, 3)),
        ((8, 5), None, 2, None, "field_line", (4, 2, 4)),
        ((8, 5), None, 2, None, "metric", (4, 2, 3)),
        # Explicit, in-range indices pass straight through.
        ((8, 5), 2, 1, 5, "metric", (2, 1, 5)),
        # Low-radius clamps: half_mesh/metric floor at 0, field_line floors at 1.
        ((8, 5), 0, 0, None, "half_mesh", (0, 0, 0)),
        ((2, 3), 0, 0, None, "metric", (0, 0, 0)),
        ((2, 3), 0, 0, None, "field_line", (0, 0, 1)),
    ],
)
def test_resolve_vmec_state_indices_resolves_defaults_and_clamps(
    shape, radial_index, mode_index, surface_index, surface_grid, expected
):
    base = jnp.zeros(shape)

    resolved = controls._resolve_vmec_state_indices(
        base,
        radial_index=radial_index,
        mode_index=mode_index,
        surface_index=surface_index,
        surface_grid=surface_grid,
    )

    assert resolved == expected
    assert all(isinstance(value, int) for value in resolved)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            dict(
                radial_index=8, mode_index=0, surface_index=None, surface_grid="metric"
            ),
            "radial_index is outside the VMEC state radial grid",
        ),
        (
            dict(
                radial_index=-1, mode_index=0, surface_index=None, surface_grid="metric"
            ),
            "radial_index is outside the VMEC state radial grid",
        ),
        (
            dict(
                radial_index=None,
                mode_index=5,
                surface_index=None,
                surface_grid="metric",
            ),
            "mode_index is outside the VMEC state mode table",
        ),
        (
            dict(
                radial_index=None,
                mode_index=0,
                surface_index=None,
                surface_grid="bogus",
            ),
            "unknown VMEC surface grid",
        ),
        (
            dict(
                radial_index=None,
                mode_index=0,
                surface_index=7,
                surface_grid="half_mesh",
            ),
            "half-mesh Boozer surface grid",
        ),
        (
            dict(
                radial_index=None,
                mode_index=0,
                surface_index=8,
                surface_grid="field_line",
            ),
            "VMEC metric radial grid",
        ),
        (
            dict(
                radial_index=None,
                mode_index=0,
                surface_index=-1,
                surface_grid="metric",
            ),
            "VMEC metric radial grid",
        ),
    ],
)
def test_resolve_vmec_state_indices_rejects_out_of_range_and_unknown_grid(
    kwargs, message
):
    base = jnp.zeros((8, 5))

    with pytest.raises(ValueError, match=message):
        controls._resolve_vmec_state_indices(base, **kwargs)


# ---------------------------------------------------------------------------
# _perturb_vmec_state (two-control perturbation + immutability)
# ---------------------------------------------------------------------------
def _perturb_context() -> tuple[object, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    base_Rcos = jnp.asarray(np.arange(1.0, 13.0).reshape(3, 4))
    base_Zsin = jnp.asarray(
        np.array([[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 1.0, 1.1]])
    )
    witness = jnp.asarray(np.full((3, 4), 3.0))
    # The state tables deliberately differ from base_* so the assertions prove
    # the perturbation is applied to ctx.base_Rcos/base_Zsin, not ctx.state.*.
    state = _FakeSpectralState(
        R_cos=jnp.zeros((3, 4)), Z_sin=jnp.zeros((3, 4)), R_sin=witness
    )
    ctx = controls._VMECStateContext(
        input_path=Path("input.synthetic"),
        wout_path=controls.VMEC_STATE_IN_MEMORY_WOUT_PATH,
        inp=object(),
        runtime=object(),
        wout=object(),
        state=state,
        base_Rcos=base_Rcos,
        base_Zsin=base_Zsin,
    )
    return ctx, base_Rcos, base_Zsin, witness


def test_perturb_vmec_state_increments_two_controls_from_base_tables():
    ctx, base_Rcos, base_Zsin, witness = _perturb_context()
    x = jnp.asarray([0.5, -0.3])

    perturbed = controls._perturb_vmec_state(ctx, x, radial_index=1, mode_index=2)

    expected_Rcos = np.asarray(base_Rcos).copy()
    expected_Rcos[1, 2] += 0.5
    expected_Zsin = np.asarray(base_Zsin).copy()
    expected_Zsin[1, 2] += -0.3

    assert isinstance(perturbed, _FakeSpectralState)
    np.testing.assert_allclose(np.asarray(perturbed.R_cos), expected_Rcos)
    np.testing.assert_allclose(np.asarray(perturbed.Z_sin), expected_Zsin)
    # Only the [1, 2] entries moved (the [0, 0] control is unchanged base data).
    assert float(np.asarray(perturbed.R_cos)[0, 0]) == pytest.approx(1.0)
    # The untouched family survives and the original frozen state is unchanged.
    assert perturbed.R_sin is witness
    np.testing.assert_allclose(np.asarray(ctx.state.R_cos), np.zeros((3, 4)))


def test_perturb_vmec_state_and_context_are_immutable():
    ctx, _base_Rcos, _base_Zsin, _witness = _perturb_context()

    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.base_Rcos = jnp.zeros((3, 4))

    perturbed = controls._perturb_vmec_state(
        ctx, jnp.asarray([0.0, 0.0]), radial_index=0, mode_index=1
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        perturbed.R_cos = jnp.zeros((3, 4))


# ---------------------------------------------------------------------------
# _length_two_params (default fill + length-2 validation)
# ---------------------------------------------------------------------------
def test_length_two_params_fills_default_and_preserves_length_two_vectors():
    filled = controls._length_two_params(None, 2.5)
    assert filled.shape == (2,)
    assert np.asarray(filled).dtype == np.float64  # x64: forced float64 contract
    np.testing.assert_allclose(np.asarray(filled), [2.5, 2.5])

    passed = controls._length_two_params(jnp.asarray([0.1, -0.2]), 0.0)
    np.testing.assert_allclose(np.asarray(passed), [0.1, -0.2])


@pytest.mark.parametrize(
    "params",
    [jnp.asarray([1.0]), jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([[1.0, 2.0]])],
)
def test_length_two_params_rejects_non_length_two_vectors(params):
    with pytest.raises(ValueError, match="params must be a length-2 vector"):
        controls._length_two_params(params, 0.0)


# ---- from test_imported_vmec_geometry.py ----


def test_imported_vmec_reuses_focused_backend_contracts() -> None:
    assert vmec_facade.internal_vmec_backend_available is (
        vmec_backend_discovery.internal_vmec_backend_available
    )
    assert (
        vmec_facade._import_booz_backend is vmec_backend_discovery._import_booz_backend
    )
    assert vmec_facade.nperiod_set is vmec_fieldline_numerics.nperiod_set
    assert vmec_facade.dermv is vmec_fieldline_numerics.dermv
    assert vmec_facade._vmec_splines is vmec_fieldline_numerics._vmec_splines


def test_vmec_struct_accepts_named_fields_and_remains_mutable() -> None:
    geom = _Struct(theta=np.array([0.0]), nfp=5)

    np.testing.assert_allclose(geom.theta, [0.0])
    assert geom.nfp == 5

    geom.iota = 0.41
    assert geom.iota == pytest.approx(0.41)

    with pytest.raises(AttributeError, match="missing"):
        _ = geom.missing


def test_booz_search_paths_include_env_and_src(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "booz_xform_jax"
    (checkout / "src").mkdir(parents=True)
    monkeypatch.setenv("GKX_BOOZ_XFORM_JAX_PATH", str(checkout))
    paths = _booz_xform_jax_search_paths()
    assert checkout.resolve() in paths
    assert (checkout / "src").resolve() in paths


def test_import_module_with_search_paths_loads_temp_module(tmp_path: Path) -> None:
    pkg = tmp_path / "mods"
    pkg.mkdir()
    (pkg / "demo_mod.py").write_text("VALUE = 7\n", encoding="utf-8")
    sys.modules.pop("demo_mod", None)
    mod = _import_module_with_search_paths("demo_mod", [pkg])
    assert mod.VALUE == 7
    sys.modules.pop("demo_mod", None)


def test_import_module_with_search_paths_replaces_namespace_without_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout_root = tmp_path / "checkout"
    pkg = checkout_root / "booz_xform_jax"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("class Booz_xform:\n    pass\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "booz_xform_jax",
        SimpleNamespace(
            __name__="booz_xform_jax", __path__=[str(tmp_path / "namespace")]
        ),
    )

    mod = _import_module_with_search_paths(
        "booz_xform_jax", [checkout_root], required_attr="Booz_xform"
    )

    assert hasattr(mod, "Booz_xform")
    assert Path(mod.__file__).resolve() == (pkg / "__init__.py").resolve()
    sys.modules.pop("booz_xform_jax", None)


def test_import_module_with_search_paths_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(ImportError):
        _import_module_with_search_paths("missing_demo_mod", [tmp_path / "missing"])


def test_import_booz_backend_falls_back_to_booz_xform(monkeypatch) -> None:
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._booz_xform_jax_search_paths",
        lambda: [],
    )
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_module_with_search_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ImportError("jax backend missing")
        ),
    )

    marker = SimpleNamespace(name="fallback", Booz_xform=object)

    def _import_module(name: str):
        if name == "booz_xform":
            return marker
        raise ImportError(name)

    monkeypatch.setattr(
        "gkx.geometry.backend_discovery.importlib.import_module",
        _import_module,
    )
    assert _import_booz_backend() is marker


def test_import_booz_backend_honors_fortran_override(monkeypatch) -> None:
    marker = SimpleNamespace(name="forced-fortran", Booz_xform=object)
    monkeypatch.setenv("GKX_BOOZ_BACKEND", "fortran")
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_backend",
        lambda: marker,
    )
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_jax_backend",
        lambda: (_ for _ in ()).throw(
            AssertionError("jax backend should not be imported")
        ),
    )

    assert _import_booz_backend() is marker


def test_square_layout_failure_matcher_is_specific() -> None:
    assert _booz_read_wout_square_layout_failure(
        ValueError(
            "rmnc0 has unexpected shape (50, 50); one dimension must equal ns=50"
        )
    )
    assert not _booz_read_wout_square_layout_failure(
        ValueError("rmnc0 has unexpected shape (50, 49)")
    )
    assert not _booz_read_wout_square_layout_failure(
        RuntimeError(
            "rmnc0 has unexpected shape (50, 50); one dimension must equal ns=50"
        )
    )


def test_import_booz_backend_honors_jax_override(monkeypatch) -> None:
    marker = SimpleNamespace(name="forced-jax", Booz_xform=object)
    monkeypatch.setenv("GKX_BOOZ_BACKEND", "jax")
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_jax_backend",
        lambda: marker,
    )
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_backend",
        lambda: (_ for _ in ()).throw(
            AssertionError("booz_xform should not be imported")
        ),
    )

    assert _import_booz_backend() is marker


def test_import_booz_backend_rejects_unknown_override(monkeypatch) -> None:
    monkeypatch.setenv("GKX_BOOZ_BACKEND", "unexpected-backend")

    with pytest.raises(ValueError, match="GKX_BOOZ_BACKEND"):
        _import_booz_backend()


def test_import_booz_backend_reports_missing_backends(monkeypatch) -> None:
    monkeypatch.delenv("GKX_BOOZ_BACKEND", raising=False)
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_jax_backend",
        lambda: (_ for _ in ()).throw(ImportError("jax missing")),
    )
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_xform_backend",
        lambda: (_ for _ in ()).throw(ImportError("booz missing")),
    )

    with pytest.raises(
        ImportError, match="booz_xform_jax/booz_xform backend unavailable"
    ):
        _import_booz_backend()


def test_internal_vmec_backend_available_uses_backend_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_backend",
        lambda: object(),
    )
    assert internal_vmec_backend_available() is True
    monkeypatch.setattr(
        "gkx.geometry.backend_discovery._import_booz_backend",
        lambda: (_ for _ in ()).throw(ImportError("missing")),
    )
    assert internal_vmec_backend_available() is False


def test_nperiod_set_and_dermv(monkeypatch) -> None:
    monkeypatch.setattr(
        "gkx.geometry.vmec_field_line_sampling.nperiod_contract",
        lambda values, theta, npol: (values[1:-1], theta[1:-1]),
    )
    values, theta = nperiod_set(
        np.array([0.0, 1.0, 2.0]), np.array([-2.0, 0.0, 2.0]), 1.0
    )
    np.testing.assert_allclose(values, [1.0])
    np.testing.assert_allclose(theta, [0.0])

    out = dermv(np.array([0.0, 1.0, 4.0, 9.0]), np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out[1:-1], [2.0, 4.0], atol=1.0e-6)

    with pytest.raises(ValueError):
        nperiod_set(np.array([1.0, 2.0]), np.array([1.0]), 1.0)
    with pytest.raises(ValueError):
        dermv(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        dermv(np.ones(2), np.ones(3))


def test_apply_flux_tube_cut_none_and_unknown() -> None:
    theta = np.linspace(-np.pi, np.pi, 7)
    base = np.linspace(1.0, 2.0, theta.size)
    geo = SimpleNamespace(
        bmag=base[None, None, :],
        gradpar_theta_b=np.abs(base)[None, None, :],
        cvdrift=base[None, None, :],
        gbdrift=base[None, None, :],
        cvdrift0=base[None, None, :],
        gbdrift0=base[None, None, :],
        gds2=(base + 1.0)[None, None, :],
        gds21=np.linspace(-1.0, 1.0, theta.size)[None, None, :],
        gds22=(base + 2.0)[None, None, :],
        grho=base[None, None, :],
        R_b=(base + 3.0)[None, None, :],
        Z_b=(base - 1.0)[None, None, :],
        grad_x=np.stack([base, base + 1.0, base + 2.0])[:, None, None, :],
        grad_y=np.stack([base + 3.0, base + 4.0, base + 5.0])[:, None, None, :],
        s_hat_input=0.8,
    )

    theta_cut, arrays = _apply_flux_tube_cut(
        theta,
        geo,
        ntheta=theta.size,
        flux_tube_cut="none",
        npol_min=None,
        which_crossing=0,
        y0=1.0,
        x0=1.0,
        jtwist_in=None,
    )
    np.testing.assert_allclose(theta_cut, theta)
    assert arrays["grad_x"].shape == (3, theta.size)
    assert arrays["b_vec"].shape == (3, theta.size)

    with pytest.raises(ValueError):
        _apply_flux_tube_cut(
            theta,
            geo,
            ntheta=theta.size,
            flux_tube_cut="bad",
            npol_min=None,
            which_crossing=0,
            y0=1.0,
            x0=1.0,
            jtwist_in=None,
        )


def test_equal_arc_remap_returns_constant_gradpar() -> None:
    theta = np.linspace(-np.pi, np.pi, 7)
    arrays = {
        "theta_PEST": theta.copy(),
        "bmag": np.linspace(1.0, 2.0, theta.size),
        "gradpar": np.full(theta.size, 2.0),
        "cvdrift": np.linspace(0.0, 1.0, theta.size),
        "gbdrift": np.linspace(1.0, 0.0, theta.size),
        "cvdrift0": np.linspace(0.5, 1.5, theta.size),
        "gbdrift0": np.linspace(1.5, 0.5, theta.size),
        "gds2": np.linspace(2.0, 3.0, theta.size),
        "gds21": np.linspace(-0.5, 0.5, theta.size),
        "gds22": np.linspace(3.0, 4.0, theta.size),
        "grho": np.linspace(0.8, 1.2, theta.size),
        "Rplot": np.linspace(5.0, 6.0, theta.size),
        "Zplot": np.linspace(-1.0, 1.0, theta.size),
        "grad_x": np.vstack(
            [np.ones(theta.size), 2.0 * np.ones(theta.size), 3.0 * np.ones(theta.size)]
        ),
        "grad_y": np.vstack(
            [
                4.0 * np.ones(theta.size),
                5.0 * np.ones(theta.size),
                6.0 * np.ones(theta.size),
            ]
        ),
    }

    gradpar_eqarc, out = _equal_arc_remap(theta, arrays, ntheta=9)
    assert np.isfinite(gradpar_eqarc)
    np.testing.assert_allclose(out["gradpar"], np.full(9, gradpar_eqarc))
    assert out["theta"].shape == (9,)
    assert out["grad_x"].shape == (3, 9)
    assert out["b_vec"].shape == (3, 9)
    assert np.isfinite(out["scale"])


def _mock_geo_for_cut(
    theta: np.ndarray,
    *,
    gds21: np.ndarray | None = None,
    gbdrift0: np.ndarray | None = None,
) -> SimpleNamespace:
    base = np.linspace(1.0, 2.0, theta.size)
    gds21_arr = gds21 if gds21 is not None else np.linspace(-1.0, 1.0, theta.size)
    gbdrift0_arr = (
        gbdrift0 if gbdrift0 is not None else np.linspace(-1.0, 1.0, theta.size)
    )
    return SimpleNamespace(
        bmag=base[None, None, :],
        gradpar_theta_b=np.abs(base)[None, None, :],
        cvdrift=base[None, None, :],
        gbdrift=base[None, None, :],
        cvdrift0=base[None, None, :],
        gbdrift0=gbdrift0_arr[None, None, :],
        gds2=(base + 1.0)[None, None, :],
        gds21=gds21_arr[None, None, :],
        gds22=np.ones(theta.size)[None, None, :],
        grho=base[None, None, :],
        R_b=(base + 3.0)[None, None, :],
        Z_b=(base - 1.0)[None, None, :],
        grad_x=np.stack([base, base + 1.0, base + 2.0])[:, None, None, :],
        grad_y=np.stack([base + 3.0, base + 4.0, base + 5.0])[:, None, None, :],
        s_hat_input=0.5,
    )


class _ScalarWithData:
    def __init__(self, value: float | int) -> None:
        self.data = np.array(value)


class _FakeVar:
    def __init__(self, data: object, *, with_data: bool = False) -> None:
        self._data = np.asarray(data)
        self._with_data = with_data

    def __getitem__(self, item: object) -> object:
        if self._data.ndim == 0:
            out = self._data
        else:
            out = self._data[item]
        if self._with_data:
            return _ScalarWithData(out)
        return out


class _FakeNCScalarVar:
    def __init__(self, value: float | int) -> None:
        self.value = value

    def __getitem__(self, _item: object) -> np.ndarray:
        return np.array(self.value)


class _FakeNCDataset:
    def __init__(self, mpol: int = 2, ntor: int = 1) -> None:
        self.variables = {
            "mpol": _FakeNCScalarVar(mpol),
            "ntor": _FakeNCScalarVar(ntor),
        }
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeBoozXform:
    def __init__(self) -> None:
        self.verbose = 0
        self.mboz: int | None = None
        self.nboz: int | None = None
        self.read_path: str | None = None
        self.ran = False

    def read_wout(self, path: str) -> None:
        self.read_path = path

    def run(self) -> None:
        self.ran = True


class _SquareLayoutFailingBoozXform(_FakeBoozXform):
    def read_wout(self, path: str) -> None:
        self.read_path = path
        raise ValueError(
            "rmnc0 has unexpected shape (50, 50); one dimension must equal ns=50"
        )


def _const_callable(value: float):
    def _inner(s: object, _value: float = value) -> np.ndarray:
        return np.full_like(np.asarray(s, dtype=float), _value, dtype=float)

    return _inner


def _fake_vmec_spline_struct() -> SimpleNamespace:
    return SimpleNamespace(
        d_pressure_d_s=_const_callable(0.2),
        iota=_const_callable(0.8),
        d_iota_d_s=_const_callable(0.1),
        Gfun=_const_callable(1.5),
        Ifun=_const_callable(0.3),
        phiedge=-2.0 * np.pi,
        Aminor_p=1.2,
        nfp=5,
        raxis_cc=np.array([3.0]),
        xm_b=np.array([0.0, 1.0]),
        xn_b=np.array([0.0, 0.0]),
        mnbooz=2,
        mboz=4,
        nboz=2,
        rmnc_b=[_const_callable(4.0), _const_callable(0.2)],
        zmns_b=[_const_callable(0.0), _const_callable(0.1)],
        numns_b=[_const_callable(0.0), _const_callable(0.05)],
        d_rmnc_b_d_s=[_const_callable(0.0), _const_callable(0.0)],
        d_zmns_b_d_s=[_const_callable(0.0), _const_callable(0.0)],
        d_numns_b_d_s=[_const_callable(0.0), _const_callable(0.0)],
        gmnc_b=[_const_callable(1.0), _const_callable(0.0)],
        bmnc_b=[_const_callable(1.3), _const_callable(0.1)],
        d_bmnc_b_d_s=[_const_callable(0.0), _const_callable(0.0)],
    )


@pytest.mark.parametrize(
    ("flux_tube_cut", "geo", "expected_cut", "jtwist_in"),
    [
        (
            "gds21",
            _mock_geo_for_cut(
                np.linspace(-2.0, 2.0, 9), gds21=np.linspace(-2.0, 2.0, 9) - 0.5
            ),
            0.5,
            None,
        ),
        (
            "gbdrift0",
            _mock_geo_for_cut(
                np.linspace(-2.0, 2.0, 9), gbdrift0=np.linspace(-2.0, 2.0, 9) - 0.75
            ),
            0.75,
            None,
        ),
        (
            "aspect",
            _mock_geo_for_cut(
                np.linspace(-2.0, 2.0, 9), gds21=np.linspace(-2.0, 2.0, 9)
            ),
            1.0,
            1,
        ),
    ],
)
def test_apply_flux_tube_cut_branch_specific_roots(
    flux_tube_cut: str,
    geo: SimpleNamespace,
    expected_cut: float,
    jtwist_in: int | None,
) -> None:
    theta = np.linspace(-2.0, 2.0, 9)
    theta_cut, arrays = _apply_flux_tube_cut(
        theta,
        geo,
        ntheta=11,
        flux_tube_cut=flux_tube_cut,
        npol_min=None,
        which_crossing=0,
        y0=1.0,
        x0=1.0,
        jtwist_in=jtwist_in,
    )

    assert theta_cut[0] == pytest.approx(-expected_cut, abs=1.0e-6)
    assert theta_cut[-1] == pytest.approx(expected_cut, abs=1.0e-6)
    assert arrays["theta"].shape == (11,)
    assert arrays["grad_x"].shape == (3, 11)


def test_apply_flux_tube_cut_reports_missing_crossings() -> None:
    theta = np.linspace(-2.0, 2.0, 9)
    geo = _mock_geo_for_cut(theta, gds21=np.ones_like(theta))

    with pytest.raises(ValueError, match="No positive gds21 flux-tube crossing"):
        _apply_flux_tube_cut(
            theta,
            geo,
            ntheta=11,
            flux_tube_cut="gds21",
            npol_min=None,
            which_crossing=0,
            y0=1.0,
            x0=1.0,
            jtwist_in=None,
        )


def test_apply_flux_tube_cut_reports_out_of_range_crossing() -> None:
    theta = np.linspace(-2.0, 2.0, 9)
    geo = _mock_geo_for_cut(theta, gds21=theta - 0.5)

    with pytest.raises(ValueError, match="which_crossing=2"):
        _apply_flux_tube_cut(
            theta,
            geo,
            ntheta=11,
            flux_tube_cut="gds21",
            npol_min=None,
            which_crossing=2,
            y0=1.0,
            x0=1.0,
            jtwist_in=None,
        )


def test_write_vmec_eik_netcdf_writes_expected_variables(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "geom.eik.nc"
    theta = np.linspace(-np.pi, np.pi, 5)
    bmag = np.linspace(1.0, 2.0, theta.size)
    gradpar = np.full(theta.size, 0.5)
    profiles = {
        "theta": theta,
        "theta_PEST": theta + 0.1,
        "bmag": bmag,
        "gradpar": gradpar,
        "grho": np.full(theta.size, 1.2),
        "gds2": np.linspace(2.0, 3.0, theta.size),
        "gds21": np.linspace(-0.5, 0.5, theta.size),
        "gds22": np.linspace(3.0, 4.0, theta.size),
        "gbdrift": np.linspace(0.1, 0.2, theta.size),
        "gbdrift0": np.linspace(0.2, 0.3, theta.size),
        "cvdrift": np.linspace(0.3, 0.4, theta.size),
        "cvdrift0": np.linspace(0.4, 0.5, theta.size),
        "Rplot": np.linspace(5.0, 6.0, theta.size),
        "Zplot": np.linspace(-1.0, 1.0, theta.size),
        "grad_x": np.vstack(
            [np.ones(theta.size), 2.0 * np.ones(theta.size), 3.0 * np.ones(theta.size)]
        ),
        "grad_y": np.vstack(
            [
                4.0 * np.ones(theta.size),
                5.0 * np.ones(theta.size),
                6.0 * np.ones(theta.size),
            ]
        ),
        "b_vec": np.vstack(
            [np.ones(theta.size), np.zeros(theta.size), np.zeros(theta.size)]
        ),
        "dpsidrho": 2.0,
        "kxfac": 1.7,
        "Rmaj": 5.5,
        "q": 1.4,
        "shat": 0.8,
        "scale": 2.0,
        "alpha": 0.25,
        "zeta_center": 0.125,
        "nfp": 5,
    }

    write_vmec_eik_netcdf(path, profiles, request=SimpleNamespace())

    with netcdf4.Dataset(path) as ds:
        np.testing.assert_allclose(ds.variables["theta"][:], theta)
        np.testing.assert_allclose(ds.variables["theta_PEST"][:], theta + 0.1)
        np.testing.assert_allclose(ds.variables["bmag"][:], bmag)
        np.testing.assert_allclose(ds.variables["gradpar"][:], gradpar)
        np.testing.assert_allclose(ds.variables["grad_x"][:, :], profiles["grad_x"])
        assert float(ds.variables["kxfac"].getValue()) == pytest.approx(1.7)
        assert float(ds.variables["Rmaj"].getValue()) == pytest.approx(5.5)
        assert int(ds.variables["nfp"].getValue()) == 5
        expected_jacob = 1.0 / abs(
            (1.0 / abs(profiles["dpsidrho"])) * gradpar[0] * bmag
        )
        np.testing.assert_allclose(ds.variables["jacob"][:], expected_jacob)


def test_vmec_splines_builds_interpolants_and_metadata() -> None:
    s_half = np.array([0.125, 0.375, 0.625, 0.875])
    booz_obj = SimpleNamespace(
        mnboz=2,
        rmnc_b=np.vstack([1.0 + s_half, 2.0 * s_half]),
        zmns_b=np.vstack([0.5 * s_half, -s_half]),
        numns_b=np.vstack([0.2 * s_half, 0.3 + 0.1 * s_half]),
        gmnc_b=np.vstack([1.0 + 0.5 * s_half, 0.1 * s_half]),
        bmnc_b=np.vstack([2.0 + 2.0 * s_half, 0.5 - 0.25 * s_half]),
        Boozer_G=5.0 + 0.2 * s_half,
        Boozer_I=1.0 - 0.1 * s_half,
        xm_b=np.array([0, 1]),
        xn_b=np.array([0, 5]),
        mboz=8,
        nboz=6,
    )
    nc_obj = SimpleNamespace(
        variables={
            "ns": _FakeVar(5, with_data=True),
            "pres": _FakeVar([0.0, 1.0, 2.0, 3.0, 4.0]),
            "phi": _FakeVar([0.0, 2.0 * np.pi, 4.0 * np.pi, 6.0 * np.pi, 8.0 * np.pi]),
            "iotas": _FakeVar([0.0, 0.5, 0.6, 0.7, 0.8]),
            "Aminor_p": _FakeVar(1.7),
            "nfp": _FakeVar(5),
            "raxis_cc": _FakeVar([3.2, 0.1]),
        }
    )

    out = _vmec_splines(nc_obj, booz_obj)

    assert out.mnbooz == 2
    assert out.mboz == 8
    assert out.nboz == 6
    assert out.nfp == 5
    np.testing.assert_allclose(out.raxis_cc, [3.2, 0.1])
    assert out.Aminor_p == pytest.approx(1.7)
    assert out.phiedge == pytest.approx(8.0 * np.pi)
    assert out.rmnc_b[0](0.5) == pytest.approx(1.5, abs=1.0e-10)
    assert out.d_rmnc_b_d_s[0](0.5) == pytest.approx(1.0, abs=1.0e-10)
    assert out.bmnc_b[1](0.5) == pytest.approx(0.375, abs=1.0e-10)
    assert out.d_bmnc_b_d_s[1](0.5) == pytest.approx(-0.25, abs=1.0e-10)
    assert out.psi(0.5) == pytest.approx(2.5, abs=1.0e-10)
    assert out.d_psi_d_s(0.5) == pytest.approx(4.0, abs=1.0e-10)
    assert out.iota(0.5) == pytest.approx(0.65, abs=1.0e-10)
    assert out.d_iota_d_s(0.5) == pytest.approx(0.4, abs=1.0e-10)


def test_vmec_fieldline_helper_angle_and_denominator_policies() -> None:
    theta = np.array([[[0.0, 1.0]]])
    phi = np.array([[[0.25, 0.25]]])
    xm = np.array([1.0, 2.0])
    xn = np.array([0.0, 1.0])

    angle = vmec_fieldline_numerics._boozer_mode_angle(xm, xn, theta, phi, flipit=False)
    np.testing.assert_allclose(angle[0, 0, 0], theta[0, 0])
    np.testing.assert_allclose(angle[1, 0, 0], 2.0 * theta[0, 0] - 0.25)

    flipped = vmec_fieldline_numerics._boozer_mode_angle(
        xm, xn, theta, phi, flipit=True
    )
    np.testing.assert_allclose(flipped[0] - angle[0], np.pi)
    np.testing.assert_allclose(flipped[1] - angle[1], 2.0 * np.pi)

    safe = vmec_derivatives._safe_mode_denominator(
        np.array([0.0, 2.0, 3.0]), np.array([0.0, 1.0, 1.0]), np.array([0.5])
    )
    assert safe.shape == (1, 2)
    assert safe[0, 0] == pytest.approx(1.0e-30)
    assert safe[0, 1] == pytest.approx(0.5)


def test_vmec_fieldline_boozer_mode_sum_preserves_surface_axis() -> None:
    coeff = np.array([[1.0, 2.0], [3.0, 4.0]])
    basis = np.arange(2 * 2 * 3 * 4, dtype=float).reshape(2, 2, 3, 4)

    out = vmec_fieldline_numerics._boozer_mode_sum(coeff, basis)

    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(
        out[0], coeff[0, 0] * basis[0, 0] + coeff[0, 1] * basis[1, 0]
    )
    np.testing.assert_allclose(
        out[1], coeff[1, 0] * basis[0, 1] + coeff[1, 1] * basis[1, 1]
    )


def test_vmec_fieldline_boozer_trig_basis_preserves_mode_axis() -> None:
    xm = np.array([0.0, 1.0, 2.0])
    xn = np.array([0.0, -1.0, 3.0])
    angle = np.linspace(0.0, 0.5, 3 * 2 * 4).reshape(3, 2, 4)

    cosangle, sinangle, mcos, msin, ncos, nsin = (
        vmec_fieldline_numerics._boozer_trig_basis(xm, xn, angle)
    )

    assert cosangle.shape == angle.shape
    assert sinangle.shape == angle.shape
    np.testing.assert_allclose(cosangle, np.cos(angle))
    np.testing.assert_allclose(sinangle, np.sin(angle))
    np.testing.assert_allclose(mcos[2], 2.0 * np.cos(angle[2]))
    np.testing.assert_allclose(msin[1], np.sin(angle[1]))
    np.testing.assert_allclose(ncos[1], -np.cos(angle[1]))
    np.testing.assert_allclose(nsin[2], 3.0 * np.sin(angle[2]))


def test_vmec_fieldline_tensor_helpers_match_circular_surface() -> None:
    theta = np.linspace(-np.pi, np.pi, 9)[None, None, :]
    phi = theta.copy()
    xm = np.array([0.0, 1.0])
    xn = np.array([0.0, 0.0])
    angle = vmec_fieldline_numerics._boozer_mode_angle(xm, xn, theta, phi, flipit=False)
    cosangle, sinangle, mcos, msin, ncos, nsin = (
        vmec_fieldline_numerics._boozer_trig_basis(xm, xn, angle)
    )
    r0, r1, z1 = 2.0, 0.2, 0.3

    tensors = vmec_fieldline_numerics._fieldline_boozer_tensors(
        rmnc_b=np.array([[r0, r1]]),
        zmns_b=np.array([[0.0, z1]]),
        numns_b=np.array([[0.0, 0.0]]),
        d_rmnc_b_d_s=np.array([[0.05, 0.01]]),
        d_zmns_b_d_s=np.array([[0.0, 0.02]]),
        d_numns_b_d_s=np.array([[0.0, 0.0]]),
        gmnc_b=np.array([[1.0, 0.0]]),
        bmnc_b=np.array([[1.0, 0.1]]),
        d_bmnc_b_d_s=np.array([[0.01, 0.02]]),
        cosangle_b=cosangle,
        sinangle_b=sinangle,
        mcosangle_b=mcos,
        msinangle_b=msin,
        ncosangle_b=ncos,
        nsinangle_b=nsin,
    )

    expected_r = r0 + r1 * np.cos(theta[0, 0])
    np.testing.assert_allclose(tensors.R_b[0, 0], expected_r)
    np.testing.assert_allclose(tensors.d_R_b_d_theta_b[0, 0], -r1 * np.sin(theta[0, 0]))
    np.testing.assert_allclose(tensors.d_Z_b_d_theta_b[0, 0], z1 * np.cos(theta[0, 0]))
    np.testing.assert_allclose(
        tensors.d_B_b_d_s[0, 0], 0.01 + 0.02 * np.cos(theta[0, 0])
    )

    cartesian = vmec_fieldline_numerics._fieldline_cartesian_derivatives(
        tensors=tensors, phi_b=phi
    )
    np.testing.assert_allclose(
        cartesian.d_X_d_phi_b[0, 0],
        -expected_r * np.sin(theta[0, 0]),
    )
    gradients = vmec_fieldline_numerics._fieldline_coordinate_gradients(
        tensors=tensors,
        cartesian=cartesian,
        edge_toroidal_flux_over_2pi=2.0,
    )
    np.testing.assert_allclose(
        gradients.grad_psi_Z,
        cartesian.d_X_d_theta_b * cartesian.d_Y_d_phi_b
        - cartesian.d_Y_d_theta_b * cartesian.d_X_d_phi_b,
    )


def test_vmec_fieldline_alpha_gradient_and_local_shear_helpers() -> None:
    shape = (1, 1, 3)
    gradients = vmec_fieldline_numerics._FieldlineCoordinateGradients(
        grad_psi_X=np.ones(shape),
        grad_psi_Y=2.0 * np.ones(shape),
        grad_psi_Z=3.0 * np.ones(shape),
        grad_theta_b_X=0.5 * np.ones(shape),
        grad_theta_b_Y=0.25 * np.ones(shape),
        grad_theta_b_Z=-0.75 * np.ones(shape),
        grad_phi_b_X=0.1 * np.ones(shape),
        grad_phi_b_Y=0.2 * np.ones(shape),
        grad_phi_b_Z=0.3 * np.ones(shape),
    )
    phi_b = np.array([[[0.0, 0.5, 1.0]]])
    zeta_center = 0.1
    d_iota_d_s = np.array([0.2])
    iota = np.array([0.4])
    etf = 2.0

    alpha = vmec_fieldline_numerics._fieldline_alpha_gradients(
        gradients=gradients,
        phi_b=phi_b,
        zeta_center=zeta_center,
        d_iota_d_s=d_iota_d_s,
        iota=iota,
        edge_toroidal_flux_over_2pi=etf,
    )

    radial_shear = -(phi_b - zeta_center) * d_iota_d_s[:, None, None] / etf
    expected_x = radial_shear * gradients.grad_psi_X + 0.5 - iota[:, None, None] * 0.1
    expected_y = radial_shear * gradients.grad_psi_Y + 0.25 - iota[:, None, None] * 0.2
    expected_z = radial_shear * gradients.grad_psi_Z - 0.75 - iota[:, None, None] * 0.3
    expected_g = 1.0**2 + 2.0**2 + 3.0**2
    expected_dot = expected_x + 2.0 * expected_y + 3.0 * expected_z

    np.testing.assert_allclose(alpha.grad_alpha_X, expected_x)
    np.testing.assert_allclose(alpha.grad_alpha_Y, expected_y)
    np.testing.assert_allclose(alpha.grad_alpha_Z, expected_z)
    np.testing.assert_allclose(alpha.g_sup_psi_psi, expected_g)
    np.testing.assert_allclose(alpha.grad_alpha_dot_grad_psi, expected_dot)

    d_iota_d_s_1 = np.array([0.3])
    d_pressure_d_s_1 = np.array([0.2])
    Vprime = np.array([1.2])
    G = np.array([2.0])
    boozer_i = np.array([0.1])
    intinv_g = np.array([[[0.1, 0.2, 0.3]]])
    int_lam_div_g = np.array([[[0.05, 0.07, 0.11]]])
    D1 = 1.7
    D2 = 0.4
    shear = vmec_fieldline_numerics._fieldline_local_shear(
        edge_toroidal_flux_over_2pi=etf,
        d_iota_d_s=d_iota_d_s,
        d_iota_d_s_1=d_iota_d_s_1,
        d_pressure_d_s_1=d_pressure_d_s_1,
        Vprime=Vprime,
        G=G,
        iota=iota,
        boozer_i=boozer_i,
        phi_b=phi_b,
        zeta_center=zeta_center,
        intinv_g=intinv_g,
        int_lam_div_g=int_lam_div_g,
        D1=D1,
        D2=D2,
        g_sup_psi_psi=alpha.g_sup_psi_psi,
        grad_alpha_dot_grad_psi=alpha.grad_alpha_dot_grad_psi,
    )
    expected_D = (
        d_iota_d_s_1[:, None, None] * (intinv_g / D1 - phi_b + zeta_center)
        - d_pressure_d_s_1[:, None, None]
        * Vprime[:, None, None]
        * (G[:, None, None] + iota[:, None, None] * boozer_i[:, None, None])
        * (int_lam_div_g - D2 * intinv_g / D1)
    ) / etf
    expected_L0 = -(
        expected_dot / expected_g
        + d_iota_d_s[:, None, None] * (phi_b - zeta_center) / etf
    )
    expected_L1 = (
        -d_iota_d_s_1[:, None, None] * (phi_b - zeta_center) / etf
        + expected_dot / expected_g
        - expected_D
    )

    np.testing.assert_allclose(shear.D_HNGC, expected_D)
    np.testing.assert_allclose(shear.L0, expected_L0)
    np.testing.assert_allclose(shear.L1, expected_L1)


def test_vmec_fieldline_helper_coordinates_and_axisym_flip_policy() -> None:
    theta1d = np.array([0.0, 1.0])
    alpha_arr = np.array([0.0, 0.5])
    iota = np.array([0.5, 1.0])

    theta_b, phi_b = vmec_derivatives._fieldline_boozer_coordinates(
        theta1d, alpha_arr, iota
    )

    assert theta_b.shape == (2, 2, 2)
    np.testing.assert_allclose(theta_b[1, 0], theta1d)
    np.testing.assert_allclose(phi_b[0, 1], (theta1d - 0.5) / 0.5)
    np.testing.assert_allclose(
        theta_b - iota[:, None, None] * phi_b,
        np.broadcast_to(alpha_arr[None, :, None], theta_b.shape),
    )

    xm = np.array([1.0])
    xn = np.array([0.0])
    rmnc_b = np.array([[1.0]])
    zmns_b = np.array([[0.25]])

    assert not vmec_derivatives._axisym_flip_required(
        isaxisym=False,
        xm_b=xm,
        xn_b=xn,
        theta_b=theta_b[:1, :1],
        phi_b=phi_b[:1, :1],
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
    )
    assert vmec_derivatives._axisym_flip_required(
        isaxisym=True,
        xm_b=xm,
        xn_b=xn,
        theta_b=theta_b[:1, :1],
        phi_b=phi_b[:1, :1],
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
    )


def test_vmec_fieldline_helper_surface_average_and_centered_integral() -> None:
    theta_grid = np.linspace(-np.pi, np.pi, 21)
    phi_grid = np.linspace(-np.pi, np.pi, 19)
    constant = np.full((phi_grid.size, theta_grid.size), 2.5)

    assert vmec_derivatives._surface_average_2d(
        constant, theta_grid, phi_grid
    ) == pytest.approx(2.5)

    theta = np.linspace(-np.pi, np.pi, 17)
    fieldline = theta[None, None, :]
    centered = vmec_derivatives._centered_fieldline_integral(
        np.ones_like(fieldline), fieldline, theta
    )
    np.testing.assert_allclose(centered[0, 0], theta, atol=1.0e-12)


def test_vmec_fieldline_reference_scale_and_override_policies() -> None:
    vs = SimpleNamespace(Aminor_p=2.0, raxis_cc=np.array([3.5]))

    L_reference, B_reference, R_mag_ax = vmec_derivatives._validated_reference_scales(
        vs, edge_toroidal_flux_over_2pi=-4.0
    )

    assert L_reference == pytest.approx(2.0)
    assert B_reference == pytest.approx(2.0)
    assert R_mag_ax == pytest.approx(3.5)

    with pytest.raises(ValueError, match="positive finite minor radius"):
        vmec_derivatives._validated_reference_scales(
            SimpleNamespace(Aminor_p=0.0, raxis_cc=np.array([3.5])),
            edge_toroidal_flux_over_2pi=-4.0,
        )

    assert vmec_derivatives._input_iota_shear(
        np.array([0.62]), np.array([0.0]), None, None
    ) == (pytest.approx(0.62), pytest.approx(1.0e-8))
    assert vmec_derivatives._input_iota_shear(
        np.array([0.62]), np.array([0.2]), 0.7, 0.3
    ) == (pytest.approx(0.7), pytest.approx(0.3))


def test_vmec_fieldline_hngc_shear_and_pressure_correction_policies() -> None:
    d_iota_d_s_1, sfac = vmec_derivatives._hngc_shear_correction(
        s_val=0.5,
        iota=np.array([0.6]),
        shat=np.array([0.2]),
        iota_input_val=0.7,
        s_hat_input_val=0.4,
        include_shear_variation=True,
    )

    np.testing.assert_allclose(d_iota_d_s_1, np.array([-0.16]))
    assert sfac == pytest.approx(0.5)

    disabled_shear, disabled_sfac = vmec_derivatives._hngc_shear_correction(
        s_val=0.5,
        iota=np.array([0.6]),
        shat=np.array([0.2]),
        iota_input_val=0.7,
        s_hat_input_val=0.4,
        include_shear_variation=False,
    )

    np.testing.assert_allclose(disabled_shear, np.zeros(1))
    assert disabled_sfac == pytest.approx(1.0)

    d_pressure_d_s_1, pfac = vmec_derivatives._hngc_pressure_correction(
        s_val=0.25,
        betaprim=0.2,
        B_reference=2.0,
        d_pressure_d_s=np.array([0.5]),
        include_pressure_variation=True,
    )

    drive = 0.2 * 2.0**2 / (4.0 * np.sqrt(0.25))
    np.testing.assert_allclose(
        d_pressure_d_s_1,
        np.array([drive - vmec_fieldline_numerics._MU_0 * 0.5]),
    )
    assert pfac == pytest.approx(drive / (vmec_fieldline_numerics._MU_0 * 0.5))

    floored_pressure, floored_pfac = vmec_derivatives._hngc_pressure_correction(
        s_val=0.25,
        betaprim=0.2,
        B_reference=2.0,
        d_pressure_d_s=np.array([0.0]),
        include_pressure_variation=True,
    )

    np.testing.assert_allclose(floored_pressure, np.array([drive]))
    assert floored_pfac == pytest.approx(
        drive / (vmec_fieldline_numerics._MU_0 * 1.0e-8)
    )

    disabled_pressure, disabled_pfac = vmec_derivatives._hngc_pressure_correction(
        s_val=0.25,
        betaprim=0.2,
        B_reference=2.0,
        d_pressure_d_s=np.array([0.5]),
        include_pressure_variation=False,
    )

    np.testing.assert_allclose(disabled_pressure, np.zeros(1))
    assert disabled_pfac == pytest.approx(1.0)


def test_vmec_fieldline_helper_flux_surface_hngc_averages() -> None:
    d1, d2 = vmec_derivatives._flux_surface_hngc_averages(
        xm_b=np.array([0.0, 1.0]),
        xn_b=np.array([0.0, 0.0]),
        flipit=False,
        lambmnc_b=np.array([[0.0, 0.0]]),
        rmnc_b=np.array([[3.0, 0.4]]),
        zmns_b=np.array([[0.0, 0.4]]),
        numns_b=np.array([[0.0, 0.0]]),
        gmnc_b=np.array([[1.0, 0.0]]),
        res_theta=31,
        res_phi=29,
    )

    assert np.isfinite(d1)
    assert d1 > 0.0
    assert d2 == pytest.approx(0.0, abs=1.0e-14)


def test_vmec_fieldline_helper_samples_boozer_mode_table() -> None:
    s = np.array([0.25, 0.75])

    def _family(scale: float) -> list:
        return [lambda x, j=j: scale * (j + np.asarray(x)) for j in range(3)]

    vs = SimpleNamespace(
        mnbooz=3,
        rmnc_b=_family(1.0),
        zmns_b=_family(2.0),
        numns_b=_family(3.0),
        d_rmnc_b_d_s=_family(4.0),
        d_zmns_b_d_s=_family(5.0),
        d_numns_b_d_s=_family(6.0),
        gmnc_b=_family(7.0),
        bmnc_b=_family(8.0),
        d_bmnc_b_d_s=_family(9.0),
    )

    rmnc_b, zmns_b, *_, d_bmnc_b_d_s = (
        vmec_fieldline_numerics._sample_boozer_mode_table(vs, s, ns=2)
    )

    assert rmnc_b.shape == (2, 3)
    np.testing.assert_allclose(rmnc_b[:, 2], 2.0 + s)
    np.testing.assert_allclose(zmns_b[:, 1], 2.0 * (1.0 + s))
    np.testing.assert_allclose(d_bmnc_b_d_s[:, 0], 9.0 * s)


def test_vmec_fieldlines_respects_overrides_and_closes_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nc = _FakeNCDataset(mpol=3, ntor=2)
    fake_backend = SimpleNamespace(Booz_xform=_FakeBoozXform)

    monkeypatch.setitem(
        sys.modules,
        "netCDF4",
        SimpleNamespace(Dataset=lambda *_args, **_kwargs: fake_nc),
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._import_booz_backend",
        lambda: fake_backend,
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._vmec_splines",
        lambda _nc, _booz: _fake_vmec_spline_struct(),
    )

    out = _vmec_fieldlines(
        vmec_fname="dummy.nc",
        s_val=0.5,
        betaprim=0.01,
        alpha=0.2,
        include_shear_variation=False,
        include_pressure_variation=False,
        theta1d=np.linspace(-np.pi, np.pi, 9),
        isaxisym=True,
        iota_input=0.9,
        s_hat_input=0.0,
        res_theta=21,
        res_phi=21,
    )

    assert fake_nc.closed is True
    assert out.iota_input == pytest.approx(0.9)
    assert out.s_hat_input == pytest.approx(1.0e-8)
    assert out.zeta_center == pytest.approx(-0.2 / 0.8)
    assert out.nfp == 5
    assert out.L_reference == pytest.approx(1.2)
    assert out.B_reference == pytest.approx(2.0 / (1.2**2))
    assert out.dpsidrho == pytest.approx(np.sqrt(2.0))
    assert out.theta_b.shape == (1, 1, 9)
    assert out.theta_PEST.shape == (1, 1, 9)
    assert out.grad_x.shape == (3, 1, 1, 9)
    assert out.grad_y.shape == (3, 1, 1, 9)
    assert np.isfinite(out.bmag).all()
    assert np.isfinite(out.gds2).all()
    assert np.isfinite(out.gbdrift).all()


def test_vmec_fieldlines_falls_back_for_square_vmex_wout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nc = _FakeNCDataset(mpol=3, ntor=2)
    jax_backend = SimpleNamespace(Booz_xform=_SquareLayoutFailingBoozXform)
    fortran_backend = SimpleNamespace(Booz_xform=_FakeBoozXform)
    calls: dict[str, object] = {}

    def _fake_import_backend(preferred: str | None = None) -> object:
        if preferred == "booz_xform":
            return fortran_backend
        return jax_backend

    def _fake_splines(_nc: object, booz_obj: object) -> SimpleNamespace:
        calls["booz_obj"] = booz_obj
        return _fake_vmec_spline_struct()

    monkeypatch.delenv("GKX_BOOZ_BACKEND", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "netCDF4",
        SimpleNamespace(Dataset=lambda *_args, **_kwargs: fake_nc),
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._import_booz_backend",
        _fake_import_backend,
    )
    monkeypatch.setattr("gkx.geometry.vmec_state_controls._vmec_splines", _fake_splines)

    out = _vmec_fieldlines(
        vmec_fname="square-vmec-jax.nc",
        s_val=0.5,
        betaprim=0.01,
        alpha=0.2,
        include_shear_variation=False,
        include_pressure_variation=False,
        theta1d=np.linspace(-np.pi, np.pi, 9),
        isaxisym=True,
        iota_input=0.9,
        s_hat_input=0.0,
        res_theta=21,
        res_phi=21,
    )

    assert fake_nc.closed is True
    assert isinstance(calls["booz_obj"], _FakeBoozXform)
    assert not isinstance(calls["booz_obj"], _SquareLayoutFailingBoozXform)
    assert out.iota_input == pytest.approx(0.9)


def test_vmec_fieldlines_does_not_fallback_when_booz_backend_is_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nc = _FakeNCDataset(mpol=3, ntor=2)
    jax_backend = SimpleNamespace(Booz_xform=_SquareLayoutFailingBoozXform)

    def _fake_import_backend(preferred: str | None = None) -> object:
        assert preferred is None
        return jax_backend

    monkeypatch.setenv("GKX_BOOZ_BACKEND", "jax")
    monkeypatch.setitem(
        sys.modules,
        "netCDF4",
        SimpleNamespace(Dataset=lambda *_args, **_kwargs: fake_nc),
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._import_booz_backend",
        _fake_import_backend,
    )

    with pytest.raises(ValueError, match="rmnc0 has unexpected shape"):
        _vmec_fieldlines(
            vmec_fname="square-vmec-jax.nc",
            s_val=0.5,
            betaprim=0.01,
            alpha=0.2,
            include_shear_variation=False,
            include_pressure_variation=False,
            theta1d=np.linspace(-np.pi, np.pi, 9),
            isaxisym=True,
            iota_input=0.9,
            s_hat_input=0.0,
            res_theta=21,
            res_phi=21,
        )
    assert fake_nc.closed is True


def test_vmec_fieldlines_rejects_degenerate_reference_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nc = _FakeNCDataset(mpol=3, ntor=2)
    fake_backend = SimpleNamespace(Booz_xform=_FakeBoozXform)
    bad_vs = _fake_vmec_spline_struct()
    bad_vs.Aminor_p = 0.0

    monkeypatch.setitem(
        sys.modules,
        "netCDF4",
        SimpleNamespace(Dataset=lambda *_args, **_kwargs: fake_nc),
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._import_booz_backend",
        lambda: fake_backend,
    )
    monkeypatch.setattr(
        "gkx.geometry.vmec_state_controls._vmec_splines",
        lambda _nc, _booz: bad_vs,
    )

    with pytest.raises(ValueError, match="Aminor_p"):
        _vmec_fieldlines(
            vmec_fname="dummy.nc",
            s_val=0.5,
            betaprim=0.01,
            alpha=0.2,
            include_shear_variation=False,
            include_pressure_variation=False,
            theta1d=np.linspace(-np.pi, np.pi, 9),
            isaxisym=True,
            iota_input=0.9,
            s_hat_input=0.0,
            res_theta=21,
            res_phi=21,
        )
    assert fake_nc.closed is True


def test_generate_vmec_eik_internal_maps_boundary_and_computes_betaprim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    theta_out = np.linspace(-np.pi, np.pi, 9)
    arrays_equal_arc = {
        "theta": theta_out,
        "theta_PEST": theta_out,
        "bmag": np.linspace(1.0, 2.0, theta_out.size),
        "gradpar": np.full(theta_out.size, 0.5),
        "cvdrift": np.linspace(0.0, 1.0, theta_out.size),
        "gbdrift": np.linspace(1.0, 2.0, theta_out.size),
        "cvdrift0": np.linspace(2.0, 3.0, theta_out.size),
        "gbdrift0": np.linspace(3.0, 4.0, theta_out.size),
        "gds2": np.linspace(4.0, 5.0, theta_out.size),
        "gds21": np.linspace(-1.0, 1.0, theta_out.size),
        "gds22": np.linspace(5.0, 6.0, theta_out.size),
        "grho": np.full(theta_out.size, 1.1),
        "Rplot": np.linspace(4.0, 6.0, theta_out.size),
        "Zplot": np.linspace(-1.0, 1.0, theta_out.size),
        "grad_x": np.vstack(
            [
                np.ones(theta_out.size),
                2.0 * np.ones(theta_out.size),
                3.0 * np.ones(theta_out.size),
            ]
        ),
        "grad_y": np.vstack(
            [
                4.0 * np.ones(theta_out.size),
                5.0 * np.ones(theta_out.size),
                6.0 * np.ones(theta_out.size),
            ]
        ),
        "b_vec": np.vstack(
            [
                np.ones(theta_out.size),
                np.zeros(theta_out.size),
                np.zeros(theta_out.size),
            ]
        ),
        "scale": 1.0,
    }

    def _mock_fieldlines(**kwargs):
        calls["fieldlines"] = kwargs
        return SimpleNamespace(
            dpsidrho=2.0,
            iota_input=0.5,
            s_hat_input=0.8,
            nfp=5,
            alpha=0.125,
            zeta_center=0.25,
        )

    def _mock_cut(**kwargs):
        calls["cut"] = kwargs
        return np.linspace(-1.0, 1.0, 9), {"theta_PEST": np.linspace(-1.0, 1.0, 9)}

    def _mock_remap(**kwargs):
        calls["remap"] = kwargs
        return 0.5, arrays_equal_arc

    def _mock_write(
        path: Path, profiles: dict[str, object], *, request: object
    ) -> None:
        calls["write"] = {"path": path, "profiles": profiles, "request": request}
        Path(path).write_bytes(b"mock eik data")

    monkeypatch.setattr("gkx.geometry.imported_vmec._vmec_fieldlines", _mock_fieldlines)
    monkeypatch.setattr("gkx.geometry.imported_vmec._apply_flux_tube_cut", _mock_cut)
    monkeypatch.setattr("gkx.geometry.imported_vmec._equal_arc_remap", _mock_remap)
    monkeypatch.setattr("gkx.geometry.imported_vmec.write_vmec_eik_netcdf", _mock_write)

    request = SimpleNamespace(
        vmec_file=str(tmp_path / "wout_test.nc"),
        torflux=0.64,
        beta=0.02,
        alpha=0.1,
        include_shear_variation=True,
        include_pressure_variation=False,
        npol=1.0,
        npol_min=None,
        ntheta=8,
        isaxisym=False,
        boundary="fix aspect",
        which_crossing=None,
        betaprim=None,
        z=(1.0, -1.0),
        dens=(2.0, 1.0),
        temp=(3.0, 0.5),
        tprim=(4.0, 5.0),
        fprim=(6.0, 7.0),
        y0=2.5,
        x0=None,
        jtwist=3,
    )

    out = generate_vmec_eik_internal(
        output_path=tmp_path / "out.eik.nc", request=request
    )

    assert out == (tmp_path / "out.eik.nc").resolve()
    assert calls["fieldlines"]["betaprim"] == pytest.approx(
        -0.02 * (2.0 * 3.0 * (4.0 + 6.0) + 1.0 * 0.5 * (5.0 + 7.0))
    )
    assert calls["cut"]["flux_tube_cut"] == "aspect"
    assert calls["cut"]["which_crossing"] == -1
    assert calls["cut"]["ntheta"] == 9
    assert calls["cut"]["x0"] == pytest.approx(2.5)
    profiles = calls["write"]["profiles"]
    assert profiles["q"] == pytest.approx(2.0)
    assert profiles["shat"] == pytest.approx(0.8)
    assert profiles["Rmaj"] == pytest.approx(5.0)
    assert profiles["alpha"] == pytest.approx(0.125)
    assert profiles["nfp"] == 5


# ---- from test_geometry_physics_contracts.py ----
# Physics gate: geometry contracts that silently corrupt results when broken.
#
# Both properties gated here were wrong in ways that produce plausible-looking
# numbers rather than errors, which is why they are worth a dedicated gate.


def test_s_alpha_keeps_radial_wavenumber_at_zero_shear() -> None:
    r"""``k_perp^2`` must depend on ``kx`` even when the magnetic shear vanishes.

    ``k_perp2`` divides ``kx`` by ``s_hat`` and multiplies by ``gds22``, so the
    two cancel at finite shear. At zero shear it uses ``kx`` directly, which
    means ``gds22`` has to be 1 rather than ``s_hat**2``. Leaving it at zero
    erased the ``kx`` dependence entirely: every radial mode then shares one
    perpendicular wavenumber, hence identical FLR factors and a degenerate
    zonal polarization. ``SlabGeometry`` always carried this guard.
    """

    kx = jnp.asarray([0.0, 0.3, 0.6])
    ky = jnp.asarray(0.5)
    theta = jnp.asarray(0.0)

    ratios = []
    for shear in (0.8, 0.4, 1.0e-6, 0.0):
        geometry = SAlphaGeometry.from_config(
            GeometryConfig(q=1.4, s_hat=shear, epsilon=0.18, R0=2.78)
        )
        kperp2 = np.asarray(geometry.k_perp2(kx, ky, theta))
        assert np.all(np.diff(kperp2) > 0.0), (
            f"s_hat={shear}: k_perp2 does not increase with kx: {kperp2}"
        )
        ratios.append(kperp2[2] / kperp2[0])

    # At theta = 0 the shear terms drop out of k_perp2, so the kx dependence
    # must be identical at every shear. That continuity is the sharpest
    # statement: the zero-shear branch cannot be special-cased incorrectly.
    assert max(ratios) - min(ratios) < 1.0e-9, f"kx scaling varies with shear: {ratios}"

    slab = SlabGeometry.from_config(
        GeometryConfig(q=1.4, s_hat=0.0, epsilon=0.18, R0=2.78)
    )
    assert np.all(np.diff(np.asarray(slab.k_perp2(kx, ky, theta))) > 0.0)


def test_s_alpha_retains_field_strength_variation() -> None:
    """s-alpha must keep ``B(theta)``, which is what makes particles trap.

    Trapping supplies the neoclassical polarization behind the
    Rosenbluth-Hinton residual, so a uniform ``|B|`` would silently remove that
    physics while every other term kept working.
    """

    theta = jnp.linspace(-np.pi, np.pi, 33)
    for epsilon in (0.1, 0.18):
        geometry = SAlphaGeometry.from_config(
            GeometryConfig(q=1.4, s_hat=0.8, epsilon=epsilon, R0=2.78)
        )
        bmag = np.asarray(geometry.bmag(theta))
        # The claim is physical -- B = B0 / (1 + eps cos theta) keeps its
        # variation -- so the tolerance has to be the arithmetic's, not a
        # constant. A flat 1e-9 is below float32 resolution: this ran green
        # under x64 and failed by 5.96e-8 at default precision, which is 2**-24,
        # exactly one float32 epsilon. That is the format, not the geometry, and
        # loosening the gate for everyone to accommodate it would hide a real
        # float64 regression. Scale with the dtype actually in use instead.
        tolerance = max(1.0e-9, 8.0 * float(np.finfo(bmag.dtype).eps))
        assert abs(bmag.min() - 1.0 / (1.0 + epsilon)) < tolerance
        assert abs(bmag.max() - 1.0 / (1.0 - epsilon)) < tolerance
        assert np.abs(np.asarray(geometry.bgrad(theta))).max() > 0.0


def _streaming_only_rhs(geometry, params) -> float:
    """Return the norm of a streaming-only linear RHS for a z-varying state."""

    grid = build_spectral_grid(GridConfig(Nx=3, Ny=2, Nz=16, Lx=20.0, Ly=20.0))
    cache = build_linear_cache(grid, geometry, params, 4, 4)
    state = jnp.zeros(
        (4, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex128
    )
    # A z-dependent perturbation: a constant one has zero parallel derivative
    # and would make this test vacuous.
    state = state.at[0, 0, 1, 1, :].set(jnp.sin(jnp.asarray(grid.z)) * 1.0e-3)
    terms = LinearTerms(
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
    )
    return float(
        jnp.linalg.norm(
            linear_rhs_cached(state, cache, params, terms=terms, use_jit=False)[0]
        )
    )


def test_geometry_params_helper_carries_the_parallel_scale() -> None:
    """``linear_params_for_geometry`` must make streaming follow ``gradpar``.

    ``kpar_scale`` multiplies the parallel derivative and defaults to 1.0,
    because ``LinearParams`` knows nothing about geometry. A hand-built
    ``LinearParams()`` therefore streams at the wrong rate -- by ``1/(q R0)``,
    a factor of 8.4 for the q=2.8, R0=3 case below -- with no error raised.
    The runtime path sets it from the geometry; this helper is the Python-API
    equivalent, and this test pins that it actually takes effect.
    """

    baseline = None
    for q, major_radius in ((1.4, 1.0), (1.4, 2.78), (2.8, 3.0)):
        geometry = SAlphaGeometry.from_config(
            GeometryConfig(q=q, s_hat=0.8, epsilon=0.18, R0=major_radius)
        )
        gradpar = float(np.asarray(geometry.gradpar()))

        plain = _streaming_only_rhs(geometry, LinearParams())
        matched = _streaming_only_rhs(geometry, linear_params_for_geometry(geometry))

        # The bare constructor ignores geometry entirely, so it gives the same
        # answer for every case; that is the defect this helper exists for.
        if baseline is None:
            baseline = plain
        assert abs(plain - baseline) < 1.0e-15 * baseline

        # The helper scales the parallel term by exactly gradpar.
        assert abs(matched / plain - gradpar) < 1.0e-9, (
            f"q={q}, R0={major_radius}: streaming scaled by {matched / plain:.6f}, "
            f"expected gradpar {gradpar:.6f}"
        )

    # An explicit override still wins, for published normalizations that fold
    # q R0 in elsewhere.
    geometry = SAlphaGeometry.from_config(
        GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.78)
    )
    assert linear_params_for_geometry(geometry, kpar_scale=1.0).kpar_scale == 1.0


def test_electromagnetic_zonal_solve_is_continuous_in_beta() -> None:
    """The field solve must not jump when beta becomes infinitesimally finite.

    The adiabatic species responds to ``phi - <phi>``, so quasineutrality
    carries a ``tau_e<phi>`` source at ``ky = 0``. The electrostatic branch
    solves for it exactly; the electromagnetic branch used to rebuild ``phi``
    from ``nbar`` alone and drop it, which over-screened the zonal potential by
    a factor of 7.3 and made the solve discontinuous as ``beta -> 0``.

    Continuity in ``beta`` is the sharp statement: no physical quantity may
    jump between ``beta = 0`` and ``beta = 1e-12``.
    """

    from gkx.terms.fields import solve_fields

    grid = build_spectral_grid(GridConfig(Nx=5, Ny=2, Nz=16, Lx=40.0, Ly=40.0))
    geometry = SAlphaGeometry.from_config(
        GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.78)
    )
    generator = np.random.default_rng(0)
    shape = (1, 4, 4, grid.ky.size, grid.kx.size, grid.z.size)
    state = jnp.asarray(
        generator.normal(size=shape) * 1.0e-3
        + 1j * generator.normal(size=shape) * 1.0e-3
    )
    unit = jnp.asarray([1.0])

    def zonal_potential(beta: float) -> np.ndarray:
        params = linear_params_for_geometry(geometry, beta=beta, tau_e=1.0)
        cache = build_linear_cache(grid, geometry, params, 4, 4)
        fields = solve_fields(
            state,
            cache,
            params,
            charge=unit,
            density=unit,
            temp=unit,
            mass=unit,
            tz=unit,
            vth=unit,
            fapar=jnp.asarray(0.0),
            w_bpar=jnp.asarray(1.0),
        )
        return np.asarray(fields.phi)

    electrostatic = zonal_potential(0.0)
    infinitesimal = zonal_potential(1.0e-12)

    zonal_es = np.abs(electrostatic[..., 0, 1, :]).max()
    zonal_em = np.abs(infinitesimal[..., 0, 1, :]).max()
    assert zonal_es > 0.0
    assert abs(zonal_em / zonal_es - 1.0) < 1.0e-9, (
        f"zonal potential jumps by {zonal_em / zonal_es:.4f} between beta=0 and "
        "beta=1e-12; the ky=0 adiabatic correction is missing from the "
        "electromagnetic branch"
    )

    # Finite beta must still do something physical rather than nothing.
    assert np.abs(zonal_potential(1.0e-2)[..., 0, 1, :]).max() / zonal_es < 0.99


def test_reference_electrostatic_solve_matches_production_including_zonal() -> None:
    """The reference field solve must reproduce production at ``ky = 0`` too.

    The existing equivalence test builds its fixture with ``Nx = 1``, which
    makes the ``kx > 0`` mask empty, so the zonal branch is never exercised and
    both implementations trivially return ``nbar/q_phi``. With ``Nx > 1`` the
    reference path returned 0.22 times the production zonal potential while
    agreeing exactly everywhere else.
    """

    from gkx.parallel.velocity_drive import electrostatic_phi_reference
    from gkx.terms.fields import solve_fields

    grid = build_spectral_grid(
        GridConfig(Nx=5, Ny=2, Nz=16, Lx=40.0, Ly=40.0, boundary="periodic")
    )
    geometry = SAlphaGeometry.from_config(
        GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.78)
    )
    assert grid.kx.size > 1, "the fixture must expose a nonzero kx to be meaningful"

    generator = np.random.default_rng(1)
    shape = (1, 4, 4, grid.ky.size, grid.kx.size, grid.z.size)
    state = jnp.asarray(
        generator.normal(size=shape) * 1.0e-3
        + 1j * generator.normal(size=shape) * 1.0e-3
    )
    unit = jnp.asarray([1.0])
    params = linear_params_for_geometry(geometry, tau_e=1.0)
    cache = build_linear_cache(grid, geometry, params, 4, 4)

    production = np.asarray(
        solve_fields(
            state,
            cache,
            params,
            charge=unit,
            density=unit,
            temp=unit,
            mass=unit,
            tz=unit,
            vth=unit,
            fapar=jnp.asarray(0.0),
            w_bpar=jnp.asarray(0.0),
        ).phi
    )
    reference = np.asarray(
        electrostatic_phi_reference(
            state,
            Jl=cache.Jl,
            tau_e=params.tau_e,
            charge=unit,
            density=unit,
            tz=unit,
            mask0=cache.mask0,
            jacobian=cache.jacobian,
            ky=cache.ky,
        )
    )

    zonal = np.abs(production[..., 0, 1, :]).max()
    assert zonal > 0.0, "the zonal mode must carry signal for this test to bite"
    assert np.abs(reference - production).max() < 1.0e-12 * np.abs(production).max()


# ---- from test_finite_beta_vmec_boozer_parity.py ----
# Finite-beta parity between the wout runtime path and the vmex-state bridge.
#
# GKX evaluates VMEC flux-tube geometry two ways: the wout runtime path
# (``imported_vmec`` + ``vmec_field_line_sampling``) and the differentiable
# vmex-state bridge (``vmec_boozer_core`` + ``vmec_boozer_drifts``) that the
# adjoint and objective stack uses.  The bridge used to alias ``gbdrift`` to


_FINITE_BETA_CASES = (
    ("LandremanPaul2021_QA_beta2", 0.25),
    ("LandremanPaul2021_QA_beta2", 0.64),
    ("LandremanPaul2021_QA_beta2_current", 0.64),
)
# The bridge and the runtime path disagree at this level on quantities that
# carry no pressure at all (bgrad, gds2), so it is the bridge's own numerical
# floor rather than a pressure tolerance. Before the pressure terms landed
# cvdrift missed by 2.1e-1 to 2.6e-1, an order of magnitude above this.
_DRIFT_TOLERANCE = 2.0e-2
_METRIC_TOLERANCE = 8.0e-2


def _stub_drift_inputs(d_pressure_ds: float) -> tuple[object, ...]:
    """Build the minimal duck-typed bundle ``raw_drift_profiles`` consumes."""

    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, dtype=jnp.float64)[:-1]
    mod_b = 1.0 + 0.1 * jnp.cos(theta)
    request = SimpleNamespace(
        base_Rcos=jnp.zeros((3, 2), dtype=jnp.float64), torflux=0.5
    )
    scales = SimpleNamespace(length=0.3, magnetic_field=1.2)
    profiles = SimpleNamespace(
        boozer_g=jnp.asarray(1.4, dtype=jnp.float64),
        boozer_i=jnp.asarray(0.07, dtype=jnp.float64),
        iota_safe=jnp.asarray(0.42, dtype=jnp.float64),
        d_iota_ds=jnp.asarray(0.11, dtype=jnp.float64),
        s_hat=jnp.asarray(-0.26, dtype=jnp.float64),
        d_pressure_ds=jnp.asarray(d_pressure_ds, dtype=jnp.float64),
    )
    equal_arc = SimpleNamespace(
        mod_b_safe=mod_b,
        sqrt_g_booz=(1.4 + 0.42 * 0.07) / (mod_b * mod_b),
    )
    spectral = SimpleNamespace(
        d_mod_b_d_theta=-0.1 * jnp.sin(theta),
        d_mod_b_d_phi=0.03 * jnp.cos(theta),
        d_mod_b_d_s=0.2 + 0.05 * jnp.cos(theta),
    )
    state = SimpleNamespace(
        spectral=spectral,
        eps=jnp.asarray(1.0e-30, dtype=jnp.float64),
        etf=jnp.asarray(-0.013, dtype=jnp.float64),
        etf_safe=jnp.asarray(-0.013, dtype=jnp.float64),
        local_shear_l1=0.05 * jnp.sin(theta),
        shear_phase=theta,
        metric_bmag_sq=mod_b * mod_b,
    )
    return request, scales, profiles, equal_arc, state


def test_zero_pressure_restores_the_zero_beta_drift_alias() -> None:
    """At zero beta the bridge must reproduce the old ``gbdrift = cvdrift``."""

    drifts = raw_drift_profiles(*_stub_drift_inputs(0.0))

    np.testing.assert_array_equal(
        np.asarray(drifts.gbdrift), np.asarray(drifts.cvdrift)
    )
    np.testing.assert_array_equal(
        np.asarray(drifts.gbdrift0), np.asarray(drifts.cvdrift0)
    )


def test_pressure_moves_the_curvature_drift_and_leaves_the_grad_b_drift() -> None:
    """``mu0 dp/ds`` belongs to the curvature, not to ``grad B``.

    Force balance puts the pressure gradient in ``kappa`` and nowhere else, so
    the same term that shifts ``cvdrift`` is subtracted back out of ``gbdrift``.
    The grad-B drift must therefore be numerically identical with and without
    pressure, and the split must equal the analytic offset.
    """

    d_pressure_ds = -3.1e4
    vacuum = raw_drift_profiles(*_stub_drift_inputs(0.0))
    finite = raw_drift_profiles(*_stub_drift_inputs(d_pressure_ds))

    grad_b = np.asarray(finite.gbdrift)
    # The invariant is exactness, so the tolerance tracks the working precision
    # rather than assuming float64: the suite runs both ways.
    tol = 8.0 * float(np.finfo(grad_b.dtype).eps)
    np.testing.assert_allclose(grad_b, np.asarray(vacuum.gbdrift), rtol=tol, atol=tol)
    np.testing.assert_allclose(
        np.asarray(finite.cvdrift0), np.asarray(vacuum.cvdrift0), rtol=tol, atol=tol
    )

    _request, scales, _profiles, _equal_arc, state = _stub_drift_inputs(d_pressure_ds)
    expected_offset = (
        2.0
        * scales.magnetic_field
        * scales.length**2
        * np.sqrt(0.5)
        * _MU_0
        * d_pressure_ds
        * np.sign(float(state.etf))
        / (float(state.etf_safe) * np.asarray(state.metric_bmag_sq))
    )
    np.testing.assert_allclose(
        np.asarray(finite.gbdrift) - np.asarray(finite.cvdrift),
        expected_offset,
        rtol=max(tol, 1e-10),
    )
    assert np.max(np.abs(np.asarray(finite.cvdrift) - np.asarray(vacuum.cvdrift))) > 0.0


def test_pressure_gradient_is_exact_for_a_quadratic_vmec_profile() -> None:
    """``p(s) = p0 (1-s)^2`` differentiates exactly on the VMEC half mesh."""

    ns = 50
    p0 = 2.2e4
    s_full = np.linspace(0.0, 1.0, ns)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    pres = np.concatenate([[0.0], p0 * (1.0 - s_half) ** 2])
    wout = SimpleNamespace(pres=pres)

    for s_value in (0.25, 0.64):
        got = float(boozer_pressure_gradient(wout, s_value=s_value, dtype=jnp.float64))
        assert got == pytest.approx(-2.0 * p0 * (1.0 - s_value), rel=2e-3)


def test_pressure_gradient_vanishes_for_a_vacuum_equilibrium() -> None:
    """A wout with no pressure profile must not perturb the vacuum drifts."""

    assert (
        float(
            boozer_pressure_gradient(SimpleNamespace(), s_value=0.5, dtype=jnp.float64)
        )
        == 0.0
    )
    assert (
        float(
            boozer_pressure_gradient(
                SimpleNamespace(pres=np.zeros(50)), s_value=0.5, dtype=jnp.float64
            )
        )
        == 0.0
    )


def test_pressure_gradient_rejects_a_malformed_profile() -> None:
    """Silently dropping pressure is the defect this module exists to remove."""

    with pytest.raises(ValueError, match="pressure profile"):
        boozer_pressure_gradient(
            SimpleNamespace(pres=np.zeros((4, 2))), s_value=0.5, dtype=jnp.float64
        )


def _finite_beta_artifact_dir() -> Path:
    raw = os.environ.get("GKX_FINITE_BETA_VMEC_DIR", "").strip()
    if not raw:
        pytest.skip(
            "Set GKX_FINITE_BETA_VMEC_DIR to a directory holding "
            "input.LandremanPaul2021_QA_beta2[_current] and their wout files "
            "to enable the finite-beta VMEC/Boozer parity gate."
        )
    directory = Path(raw).expanduser()
    if not directory.is_dir():
        pytest.skip(f"GKX_FINITE_BETA_VMEC_DIR is not a directory: {directory}")
    return directory


def _finite_beta_case_paths(directory: Path, case: str) -> tuple[Path, Path]:
    input_path = directory / f"input.{case}"
    wout_path = directory / f"wout_{case}.nc"
    missing = [p.name for p in (input_path, wout_path) if not p.is_file()]
    if missing:
        pytest.skip(f"missing finite-beta artifacts in {directory}: {missing}")
    return input_path, wout_path


def _imported_runtime_geometry(wout_path: Path, *, torflux: float, ntheta: int):
    from gkx.geometry import load_imported_geometry_netcdf
    from gkx.geometry.imported_vmec import generate_vmec_eik_internal

    # The report builders moved to scripts/campaigns; the default EIK request is
    # a constant this parity check needs, not report machinery.
    from gkx.geometry.vmec_eik import _VMEC_EIK_DEFAULT_REQUEST

    payload = dict(_VMEC_EIK_DEFAULT_REQUEST)
    payload.update(
        {
            "vmec_file": str(wout_path),
            "ntheta": int(ntheta),
            "boundary": "none",
            "alpha": 0.0,
            "torflux": float(torflux),
            "include_shear_variation": False,
            "include_pressure_variation": False,
        }
    )
    with tempfile.TemporaryDirectory(prefix="gkx_finite_beta_parity_") as tmp:
        eik_path = Path(tmp) / "finite_beta.eik.nc"
        generate_vmec_eik_internal(
            output_path=eik_path, request=SimpleNamespace(**payload)
        )
        return load_imported_geometry_netcdf(eik_path)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(("case", "torflux"), _FINITE_BETA_CASES)
def test_finite_beta_drifts_agree_between_wout_and_state_paths(
    case: str, torflux: float
) -> None:
    """The differentiable bridge must reproduce the runtime path at ~2% beta.

    Both routes describe one equilibrium, so a disagreement here is a missing
    physical term rather than a convention difference: this gate is what caught
    the bridge computing the grad-B drift and labelling it the curvature drift.
    """

    pytest.importorskip("vmex")
    pytest.importorskip("booz_xform_jax")
    directory = _finite_beta_artifact_dir()
    input_path, wout_path = _finite_beta_case_paths(directory, case)

    from gkx.geometry.vmec_boozer_core import (
        load_solved_vmex_case,
        vmex_boozer_equal_arc_core_profiles_from_state,
    )

    ntheta = 32
    inp, state, runtime, wout = load_solved_vmex_case(str(input_path))
    bridge = vmex_boozer_equal_arc_core_profiles_from_state(
        state,
        runtime,
        inp,
        wout,
        surface_index=None,
        torflux=torflux,
        alpha=0.0,
        ntheta=ntheta,
        mboz=24,
        nboz=24,
        jit=False,
    )
    imported = _imported_runtime_geometry(wout_path, torflux=torflux, ntheta=ntheta)
    if imported.theta.shape[0] == ntheta + 1:
        imported = imported.trim_terminal_theta_point()

    tolerances = {
        "bmag": 1.0e-3,
        "gds2": _METRIC_TOLERANCE,
        "gds21": _METRIC_TOLERANCE,
        "gds22": _METRIC_TOLERANCE,
        "grho": _METRIC_TOLERANCE,
        "cvdrift": _DRIFT_TOLERANCE,
        "gbdrift": _DRIFT_TOLERANCE,
        # cvdrift0/gbdrift0 are grad-psi components carrying no pressure term:
        # they were already at 2.2e-2 here before the pressure work and moved
        # by nothing, so they are held to the metric floor they actually share
        # with gds21/gds22 rather than to the drift tolerance.
        "cvdrift0": _METRIC_TOLERANCE,
        "gbdrift0": _METRIC_TOLERANCE,
    }
    attributes = {
        "bmag": "bmag_profile",
        "gds2": "gds2_profile",
        "gds21": "gds21_profile",
        "gds22": "gds22_profile",
        "grho": "grho_profile",
        "cvdrift": "cv_profile",
        "gbdrift": "gb_profile",
        "cvdrift0": "cv0_profile",
        "gbdrift0": "gb0_profile",
    }
    for name, tolerance in tolerances.items():
        metrics = _array_parity_metrics(
            np.asarray(bridge[name]), getattr(imported, attributes[name])
        )
        assert bool(metrics["shape_match"]), f"{name} shape mismatch"
        assert float(metrics["normalized_max_abs"]) <= tolerance, (
            f"{name} parity {metrics['normalized_max_abs']:.3e} exceeds {tolerance:.1e}"
        )

    # The split itself is the finite-beta physics: assert it is present in the
    # bridge and the same size as the runtime path, not merely that both are
    # close to zero.
    bridge_split = np.asarray(bridge["cvdrift"]) - np.asarray(bridge["gbdrift"])
    runtime_split = np.asarray(imported.cv_profile) - np.asarray(imported.gb_profile)
    drift_amplitude = float(np.max(np.abs(np.asarray(imported.cv_profile))))
    assert np.max(np.abs(runtime_split)) > 0.1 * drift_amplitude
    np.testing.assert_allclose(
        bridge_split, runtime_split, rtol=0.0, atol=1.0e-3 * drift_amplitude
    )


# ---- from test_external_vmec_validation_policy.py ----


def _load_admission_tool_module():
    return load_release_tool("check_vmec_boozer_gates")


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _full_grid_gate(
    tmp_path: Path, *, extra_failed_metric: str | None = None, passed: bool = False
) -> Path:
    failed_metrics = [
        "common_window_pairwise_heat_flux_symmetric_relative_difference",
        "least_window_pairwise_heat_flux_symmetric_relative_difference",
    ]
    if extra_failed_metric is not None:
        failed_metrics.append(extra_failed_metric)
    gates = [
        {"metric": metric, "passed": metric not in failed_metrics}
        for metric in [
            "common_window_max_relative_slope_per_time",
            "common_window_pairwise_heat_flux_symmetric_relative_difference",
            "least_window_pairwise_heat_flux_symmetric_relative_difference",
        ]
    ]
    return _write_json(
        tmp_path / "full_grid.json",
        {
            "kind": "external_vmec_nonlinear_grid_convergence_gate",
            "passed": passed,
            "runs": [{"label": "n48"}, {"label": "n64"}, {"label": "n80"}],
            "gate_report": {"passed": passed, "gates": gates},
        },
    )


def _high_grid_gate(
    tmp_path: Path, name: str, *, passed: bool = True, common: float = 0.05
) -> Path:
    return _write_json(
        tmp_path / f"{name}.json",
        {
            "kind": "external_vmec_nonlinear_grid_convergence_gate",
            "passed": passed,
            "thresholds": {"max_pairwise_relative_difference": 0.15},
            "common_window": {
                "max_pairwise_heat_flux_symmetric_relative_difference": common
            },
            "least_windows": {
                "max_pairwise_heat_flux_symmetric_relative_difference": 0.04
            },
            "runs": [{"label": "n64"}, {"label": "n80"}],
            "gate_report": {
                "passed": passed,
                "gates": [{"metric": "pairwise", "passed": passed}],
            },
        },
    )


def _time_horizon_gate(tmp_path: Path, *, passed: bool = True) -> Path:
    return _write_json(
        tmp_path / "time_horizon.json",
        {
            "kind": "external_vmec_time_horizon_gate",
            "passed": passed,
            "thresholds": {"max_relative_change": 0.15},
            "common_window_time_horizon_relative_change": 0.02,
            "least_window_time_horizon_relative_change": 0.03,
            "gate_report": {
                "passed": passed,
                "gates": [{"metric": "horizon", "passed": passed}],
            },
        },
    )


def _replicate_gate(
    tmp_path: Path, *, passed: bool = True, spread: float = 0.04
) -> Path:
    return _write_json(
        tmp_path / "replicate.json",
        {
            "kind": "nonlinear_window_ensemble_report",
            "passed": passed,
            "config": {
                "max_mean_rel_spread": 0.15,
                "max_combined_sem_rel": 0.25,
            },
            "statistics": {
                "n_reports": 4,
                "n_finite_means": 4,
                "ensemble_mean": 9.5,
                "mean_rel_spread": spread,
                "combined_sem_rel": 0.05,
            },
        },
    )


def _build_admission_payload(tmp_path: Path, **overrides: Path):
    mod = _load_admission_tool_module()
    return mod.build_high_grid_admission_payload(
        full_grid_gate_path=overrides.get("full_grid") or _full_grid_gate(tmp_path),
        high_grid_gate_paths=[
            overrides.get("high_grid_a") or _high_grid_gate(tmp_path, "t250"),
            overrides.get("high_grid_b") or _high_grid_gate(tmp_path, "t350"),
        ],
        time_horizon_gate_path=overrides.get("time_horizon")
        or _time_horizon_gate(tmp_path),
        replicate_ensemble_path=overrides.get("replicate") or _replicate_gate(tmp_path),
        excluded_grid_labels=["n48"],
        retained_grid_labels=["n64", "n80"],
        case="synthetic high-grid admission",
    )


def test_high_grid_admission_passes_with_coarse_exclusion_and_replicates(
    tmp_path: Path,
) -> None:
    payload = _build_admission_payload(tmp_path)

    assert payload["kind"] == "external_vmec_high_grid_admission_gate"
    assert payload["passed"] is True
    assert payload["promotion_gate"]["passed"] is True
    assert payload["promotion_gate"]["blockers"] == []
    assert (
        payload["policy"]["calibration_use"] == "eligible_as_scoped_high_grid_holdout"
    )
    assert (
        payload["claim_level"]
        == "passed_high_grid_transport_holdout_admission_under_coarse_grid_exclusion"
    )


def test_high_grid_admission_fails_unexpected_full_grid_failure(tmp_path: Path) -> None:
    payload = _build_admission_payload(
        tmp_path,
        full_grid=_full_grid_gate(
            tmp_path,
            extra_failed_metric="common_window_max_relative_slope_per_time",
        ),
    )

    assert payload["passed"] is False
    assert (
        "full_grid_failure_limited_to_grid_difference"
        in payload["promotion_gate"]["blockers"]
    )


def test_high_grid_admission_fails_when_high_grid_pair_does_not_pass(
    tmp_path: Path,
) -> None:
    payload = _build_admission_payload(
        tmp_path, high_grid_b=_high_grid_gate(tmp_path, "t350", passed=False)
    )

    assert payload["passed"] is False
    assert "high_grid_gate_failure_count" in payload["promotion_gate"]["blockers"]


def test_high_grid_admission_fails_replicate_spread(tmp_path: Path) -> None:
    payload = _build_admission_payload(
        tmp_path, replicate=_replicate_gate(tmp_path, spread=0.22)
    )

    assert payload["passed"] is False
    assert "replicate_mean_relative_spread" in payload["promotion_gate"]["blockers"]


def test_high_grid_admission_cli_writes_json(tmp_path: Path) -> None:
    mod = _load_admission_tool_module()
    out = tmp_path / "admission.json"
    rc = mod.main(
        [
            "high-grid-admission",
            "--full-grid-gate",
            str(_full_grid_gate(tmp_path)),
            "--high-grid-gate",
            str(_high_grid_gate(tmp_path, "t250")),
            "--high-grid-gate",
            str(_high_grid_gate(tmp_path, "t350")),
            "--time-horizon-gate",
            str(_time_horizon_gate(tmp_path)),
            "--replicate-ensemble",
            str(_replicate_gate(tmp_path)),
            "--excluded-grid-label",
            "n48",
            "--retained-grid-label",
            "n64",
            "--retained-grid-label",
            "n80",
            "--out",
            str(out),
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is True


# ---- from test_vmec_example_inventory.py ----


REPO = REPO_ROOT
EXAMPLES = REPO / "examples"
VMEC_INPUTS = EXAMPLES / "vmec"


def _walk_vmec_file_values(obj: object) -> list[str]:
    values: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "vmec_file" and isinstance(value, str):
                values.append(value)
            values.extend(_walk_vmec_file_values(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(_walk_vmec_file_values(value))
    return values


def test_example_tomls_do_not_ship_geometry_placeholders() -> None:
    forbidden = ("$HSX_VMEC_FILE", "$W7X_VMEC_FILE", "/path/to", "pth to")
    offenders: list[str] = []

    for path in EXAMPLES.rglob("*.toml"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path.relative_to(REPO)} contains {token!r}")

    assert offenders == []


def test_vmec_backed_examples_point_to_generated_wouts_with_input_decks() -> None:
    missing: list[str] = []

    for path in EXAMPLES.rglob("*.toml"):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        for vmec_file in _walk_vmec_file_values(data):
            if not vmec_file.endswith(".nc"):
                missing.append(
                    f"{path.relative_to(REPO)} has non-WOUT vmec_file={vmec_file!r}"
                )
                continue
            resolved = (path.parent / vmec_file).resolve()
            if resolved.parent != VMEC_INPUTS.resolve():
                missing.append(
                    f"{path.relative_to(REPO)} points outside examples/vmec: {vmec_file}"
                )
                continue
            stem = resolved.name.removeprefix("wout_").removesuffix(".nc")
            input_deck = VMEC_INPUTS / f"input.{stem}"
            if not input_deck.exists():
                missing.append(
                    f"{path.relative_to(REPO)} expects {resolved.name}, but {input_deck.relative_to(REPO)} is missing"
                )

    assert missing == []


def test_vmec_input_decks_are_small_text_inputs() -> None:
    inputs = sorted(VMEC_INPUTS.glob("input.*"))
    script_text = (VMEC_INPUTS / "generate_wouts.sh").read_text(encoding="utf-8")
    tracked = subprocess.run(
        ["git", "ls-files", "examples/vmec"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert {path.name for path in inputs} >= {
        "input.circular_tokamak",
        "input.NuhrenbergZille_1988_QHS",
        "input.nfp3_QI_fixed_resolution_final",
    }
    assert all(path.stat().st_size < 100_000 for path in inputs)
    assert [path for path in tracked if path.endswith(".nc")] == []
    assert "${input/input./wout_}.nc" in script_text
    for path in inputs:
        assert path.name in script_text
