"""Geometry helper tests."""

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
from gkx.core.grid import build_spectral_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams


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

    assert geom.s_hat == pytest.approx(0.0)


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
    sampled = sample_flux_tube_geometry(geom, theta_closed)

    ensured = ensure_flux_tube_geometry_data(sampled, theta_solver)

    assert ensured is not sampled
    assert ensured.theta.shape == theta_solver.shape
    assert jnp.allclose(ensured.theta, theta_solver)
    assert jnp.allclose(ensured.bmag_profile, sampled.bmag_profile[:-1])


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
