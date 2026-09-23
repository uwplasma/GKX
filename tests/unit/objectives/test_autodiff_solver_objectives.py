"""Unit contracts: autodiff solver objectives."""

from __future__ import annotations

import json
from typing import Any
from support.paired_solvax import requires_paired_solvax
from support.paths import REPO_ROOT

# ---- test_autodiff_validation.py ----

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import solvax

import gkx
import gkx.objectives.autodiff_validation as adv
from gkx.objectives.autodiff_validation import (
    autodiff_finite_difference_report,
    central_finite_difference_jacobian,
    covariance_diagnostics,
    explicit_complex_operator_matrix,
    implicit_eigenpair_observable_sensitivity_report,
    isolated_eigenpair_observable_sensitivity_report,
    isolated_eigenvalue_sensitivity_report,
)
from gkx.config import CycloneBaseCase, GridConfig
from gkx.geometry import SAlphaGeometry
from gkx.geometry.flux_tube import sample_flux_tube_geometry
from gkx.core_grid import build_spectral_grid, select_ky_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    Species,
    build_linear_params,
)
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.diagnostics import fieldline_quadrature_weights
from gkx.diagnostics.quasilinear_transport import (
    effective_kperp2,
    quasilinear_feature_objective,
    shape_aware_power_law_objective,
)


requires_solvax_reverse_eigenpair = requires_paired_solvax(
    "eigenpair_reverse", "propagator_eigenpairs"
)


def _actual_linear_rhs_objective_functions():
    """Build the shared physical operator and phase-invariant ITG objective."""

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=6, Nz=4, Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    state_shape = (2, 1, grid.ky.size, grid.kx.size, grid.z.size)
    base_params = LinearParams(
        fprim=2.2,
        tprim=6.9,
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )
    cache = build_linear_cache(grid, geom, base_params, 2, 1)
    vol_fac, _flux_fac = fieldline_quadrature_weights(geom, grid)
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    def params_from_features(x):
        return LinearParams(
            fprim=x[0],
            tprim=x[1],
            nu=0.0,
            nu_hyper=0.0,
            hypercollisions_const=0.0,
            hypercollisions_kz=0.0,
            D_hyper=0.0,
            beta=0.0,
            fapar=0.0,
        )

    def rhs_with_params(state, params):
        return linear_rhs_cached(
            state,
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    def matrix_fn(x):
        params = params_from_features(x)
        return explicit_complex_operator_matrix(
            lambda state: rhs_with_params(state, params)[0], state_shape
        )

    def objective_fn(eigenvalue, eigenvector, x):
        params = params_from_features(x)
        state = jnp.reshape(eigenvector, state_shape)
        _rhs, phi = rhs_with_params(state, params)
        kperp_eff = effective_kperp2(phi, cache, vol_fac)
        gamma = jnp.real(eigenvalue)
        return jnp.asarray([gamma, kperp_eff, gamma / jnp.maximum(kperp_eff, 1.0e-12)])

    return matrix_fn, objective_fn


def test_covariance_diagnostics_reports_uq_and_sensitivity_metadata() -> None:
    assert gkx.covariance_diagnostics is covariance_diagnostics
    jac = np.array([[1.0, 0.2], [0.1, 0.8], [0.4, -0.3]])
    residual = np.array([1.0e-3, -2.0e-3, 1.5e-3])

    out = covariance_diagnostics(jac, residual, regularization=1.0e-8)
    cov = np.asarray(out["covariance"])
    corr = np.asarray(out["covariance_correlation"])
    eig = np.asarray(out["covariance_eigenvalues"])

    assert out["sensitivity_map_rank"] == 2
    assert np.isfinite(float(out["jacobian_condition_number"]))
    assert np.allclose(cov, cov.T)
    assert np.all(eig > 0.0)
    assert np.all(np.asarray(out["covariance_std"]) > 0.0)
    assert np.allclose(np.diag(corr), 1.0)
    assert float(out["uq_ellipse_area_1sigma"]) > 0.0


def test_covariance_diagnostics_matches_closed_form_gauss_newton() -> None:
    jac = np.eye(2)
    residual = np.array([2.0e-3, -4.0e-3])

    out = covariance_diagnostics(jac, residual, regularization=0.0)
    sigma2 = float(np.mean(residual**2) + 1.0e-12)

    np.testing.assert_allclose(np.asarray(out["covariance"]), sigma2 * np.eye(2))
    np.testing.assert_allclose(
        np.asarray(out["covariance_std"]), np.sqrt(sigma2) * np.ones(2)
    )
    np.testing.assert_allclose(np.asarray(out["covariance_correlation"]), np.eye(2))
    np.testing.assert_allclose(float(out["uq_ellipse_area_1sigma"]), np.pi * sigma2)
    assert out["sensitivity_map_rank"] == 2
    assert float(out["jacobian_condition_number"]) == 1.0


def test_covariance_diagnostics_reports_full_rank_ill_conditioning() -> None:
    jac = np.diag(np.array([1.0, 1.0e-4]))
    residual = np.array([3.0e-3, -4.0e-3])

    out = covariance_diagnostics(jac, residual, regularization=0.0)
    cov = np.asarray(out["covariance"])
    singular_values = np.asarray(out["jacobian_singular_values"])

    assert out["sensitivity_map_rank"] == 2
    np.testing.assert_allclose(singular_values, np.array([1.0, 1.0e-4]))
    assert float(out["jacobian_condition_number"]) == pytest.approx(1.0e4)
    assert cov[1, 1] / cov[0, 0] == pytest.approx(1.0e8)
    np.testing.assert_allclose(np.asarray(out["covariance_correlation"]), np.eye(2))


def test_covariance_diagnostics_flags_rank_deficient_sensitivity_map() -> None:
    jac = np.array([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
    residual = np.array([1.0e-3, 2.0e-3, -1.0e-3])

    out = covariance_diagnostics(jac, residual, regularization=1.0e-6)

    assert out["sensitivity_map_rank"] == 1
    assert np.isinf(float(out["jacobian_condition_number"]))
    cov = np.asarray(out["covariance"])
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) >= 0.0)
    assert np.asarray(out["covariance_std"])[1] > np.asarray(out["covariance_std"])[0]


def test_covariance_diagnostics_handles_scalar_and_empty_parameter_maps() -> None:
    scalar = covariance_diagnostics(np.ones((3, 1)), np.array([1.0e-3, -1.0e-3, 0.0]))
    assert scalar["sensitivity_map_rank"] == 1
    assert np.asarray(scalar["covariance"]).shape == (1, 1)
    assert float(scalar["uq_ellipse_area_1sigma"]) == 0.0

    with pytest.raises(ValueError, match="at least one parameter column"):
        covariance_diagnostics(np.empty((2, 0)), np.array([1.0e-3, -1.0e-3]))


def test_covariance_diagnostics_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2, 1)), np.ones(2))
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2)), np.ones(3))
    with pytest.raises(ValueError):
        covariance_diagnostics(np.ones((2, 2)), np.ones(2), regularization=-1.0)
    with pytest.raises(ValueError, match="jacobian.*finite"):
        covariance_diagnostics(np.asarray([[1.0, np.nan]]), np.ones(1))
    with pytest.raises(ValueError, match="residual.*finite"):
        covariance_diagnostics(np.ones((1, 1)), np.asarray([np.inf]))


def test_autodiff_finite_difference_report_matches_closed_form_jacobian() -> None:
    assert gkx.autodiff_finite_difference_report is autodiff_finite_difference_report
    assert gkx.central_finite_difference_jacobian is central_finite_difference_jacobian

    def fn(x):
        return jnp.asarray([x[0] ** 2 + 3.0 * x[1], x[0] * x[1]])

    p = jnp.asarray([0.4, -0.2])
    report = autodiff_finite_difference_report(
        fn, p, step=1.0e-3, rtol=5.0e-4, atol=5.0e-6
    )
    parallel_report = autodiff_finite_difference_report(
        fn,
        p,
        step=1.0e-3,
        rtol=5.0e-4,
        atol=5.0e-6,
        workers=2,
    )

    assert report["passed"] is True
    assert parallel_report["passed"] is True
    assert parallel_report["finite_difference_parallel"]["requested_workers"] == 2
    jac_ad = np.asarray(report["jacobian_ad"])
    np.testing.assert_allclose(
        jac_ad, np.asarray([[0.8, 3.0], [-0.2, 0.4]]), rtol=1.0e-6
    )
    np.testing.assert_allclose(parallel_report["jacobian_fd"], report["jacobian_fd"])
    assert float(report["tangent_max_abs_error"]) < 1.0e-4


def test_autodiff_report_matches_jvp_and_vjp_on_tiny_analytic_function() -> None:
    def fn(x):
        return jnp.asarray([jnp.sin(x[0]) + x[1] ** 2, x[0] * jnp.exp(x[1])])

    p = jnp.asarray([0.3, -0.4])
    direction = jnp.asarray([0.6, -0.8])
    cotangent = jnp.asarray([1.25, -0.5])

    report = autodiff_finite_difference_report(
        fn,
        p,
        step=1.0e-3,
        rtol=2.0e-3,
        atol=2.0e-5,
        direction=direction,
    )
    _value, jvp_tangent = jax.jvp(fn, (p,), (direction,))
    _value, pullback = jax.vjp(fn, p)
    (vjp_cotangent,) = pullback(cotangent)
    jac_ad = np.asarray(report["jacobian_ad"])
    expected_jac = np.asarray(
        [
            [np.cos(0.3), -0.8],
            [np.exp(-0.4), 0.3 * np.exp(-0.4)],
        ]
    )

    assert report["passed"] is True
    np.testing.assert_allclose(jac_ad, expected_jac, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(
        report["tangent_ad"], jvp_tangent, rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        report["tangent_fd"], jvp_tangent, rtol=2.0e-3, atol=2.0e-5
    )
    np.testing.assert_allclose(
        cotangent @ jac_ad, vjp_cotangent, rtol=1.0e-6, atol=1.0e-6
    )


def test_central_finite_difference_handles_empty_parameter_vector() -> None:
    jac = central_finite_difference_jacobian(
        lambda x: jnp.asarray([1.0, 2.0]), jnp.asarray([])
    )
    assert jac.shape == (2, 0)
    assert jac.dtype == jnp.asarray([]).dtype


def test_finite_difference_report_rejects_invalid_worker_contracts() -> None:
    def fn(x):
        return jnp.asarray([x[0] ** 2])

    with pytest.raises(ValueError, match="step"):
        central_finite_difference_jacobian(fn, jnp.asarray([1.0]), step=0.0)
    with pytest.raises(ValueError, match="workers"):
        autodiff_finite_difference_report(fn, jnp.asarray([1.0]), workers=0)
    with pytest.raises(ValueError, match="parallel_executor"):
        autodiff_finite_difference_report(
            fn, jnp.asarray([1.0]), parallel_executor="mpi"
        )
    with pytest.raises(ValueError, match="thread executor"):
        central_finite_difference_jacobian(
            fn,
            jnp.asarray([1.0, 2.0]),
            workers=2,
            parallel_executor="process",
        )
    with pytest.raises(ValueError, match="direction"):
        autodiff_finite_difference_report(fn, jnp.asarray([1.0]), direction=jnp.ones(2))


def test_quasilinear_feature_objective_derivative_gate() -> None:
    features = jnp.asarray([0.2, 0.8, 1.5])
    report = autodiff_finite_difference_report(
        lambda x: quasilinear_feature_objective(x, csat=0.7),
        features,
        step=1.0e-3,
        rtol=1.0e-4,
        atol=1.0e-5,
    )

    assert report["passed"] is True
    jac = np.asarray(report["jacobian_ad"]).reshape(3)
    expected = np.asarray([0.7 * 1.5 / 0.8, -0.7 * 1.5 * 0.2 / 0.8**2, 0.7 * 0.2 / 0.8])
    np.testing.assert_allclose(jac, expected, rtol=1.0e-6)


def test_quasilinear_sweep_rule_objectives_have_fd_checked_derivatives() -> None:
    features = jnp.asarray([-0.25, 0.8, 1.5])
    for rule in ("linear_weight", "absolute_growth_mixing_length"):
        report = autodiff_finite_difference_report(
            lambda x, rule=rule: quasilinear_feature_objective(x, rule=rule, csat=0.7),
            features,
            step=1.0e-3,
            rtol=2.0e-4,
            atol=1.0e-5,
        )
        assert report["passed"] is True


def test_shape_aware_power_law_objective_has_fd_checked_derivatives() -> None:
    ky = jnp.asarray([0.1, 0.2, 0.4])

    def objective(x):
        features = jnp.stack(
            [
                jnp.asarray([0.1, 0.2, 0.3]),
                jnp.asarray([0.5, 0.6, 0.7]),
                x[:3],
            ],
            axis=-1,
        )
        return jnp.sum(
            shape_aware_power_law_objective(features, ky, exponent=x[3], csat=0.8)
        )

    x0 = jnp.asarray([1.0, 1.5, 2.0, -0.3])
    report = autodiff_finite_difference_report(
        objective, x0, step=1.0e-3, rtol=2.0e-4, atol=1.0e-5
    )

    assert report["passed"] is True


def test_isolated_eigenvalue_sensitivity_report_tracks_branch_derivatives() -> None:
    assert (
        gkx.isolated_eigenvalue_sensitivity_report
        is isolated_eigenvalue_sensitivity_report
    )

    def matrix_fn(x):
        return jnp.asarray(
            [
                [0.30 + x[0], 0.02],
                [0.00, -0.40 + 0.50 * x[1]],
            ]
        )

    report = isolated_eigenvalue_sensitivity_report(
        matrix_fn,
        jnp.asarray([0.1, -0.2]),
        step=1.0e-3,
        rtol=1.0e-4,
        atol=1.0e-5,
    )

    assert report["passed"] is True
    assert report["branch_isolated"] is True
    assert report["selected_index"] == 0
    jac = np.asarray(report["jacobian_ad"])
    np.testing.assert_allclose(jac, np.asarray([[1.0, 0.0], [0.0, 0.0]]), rtol=1.0e-6)


def test_actual_linear_rhs_eigenvalue_derivative_gate() -> None:
    """Gate AD through a tiny GKX linear RHS dense fixture."""

    assert gkx.explicit_complex_operator_matrix is explicit_complex_operator_matrix
    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=4, Lx=6.0, Ly=6.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    n_laguerre = 1
    n_hermite = 1
    state_shape = (n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
    base_params = LinearParams(
        fprim=2.2,
        tprim=6.9,
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )
    cache = build_linear_cache(grid, geom, base_params, n_laguerre, n_hermite)
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    def matrix_fn(x):
        params = LinearParams(
            fprim=x[0],
            tprim=x[1],
            nu=0.0,
            nu_hyper=0.0,
            hypercollisions_const=0.0,
            hypercollisions_kz=0.0,
            D_hyper=0.0,
            beta=0.0,
            fapar=0.0,
        )
        return explicit_complex_operator_matrix(
            lambda state: linear_rhs_cached(
                state,
                cache,
                params,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
            )[0],
            state_shape,
        )

    report = isolated_eigenvalue_sensitivity_report(
        matrix_fn,
        jnp.asarray([2.2, 6.9]),
        step=1.0e-3,
        rtol=2.0e-2,
        atol=2.0e-4,
        gap_floor=1.0e-6,
    )

    assert report["passed"] is True
    assert report["branch_isolated"] is True
    jac_ad = np.asarray(report["jacobian_ad"])
    jac_fd = np.asarray(report["jacobian_fd"])
    np.testing.assert_allclose(jac_ad, jac_fd, rtol=2.0e-2, atol=2.0e-4)


def test_actual_linear_rhs_branch_objective_derivative_gate() -> None:
    """Gate a phase-invariant reduced quasilinear objective on the RHS branch."""

    assert gkx.isolated_eigenpair_observable_sensitivity_report is (
        isolated_eigenpair_observable_sensitivity_report
    )
    matrix_fn, objective_fn = _actual_linear_rhs_objective_functions()

    report = isolated_eigenpair_observable_sensitivity_report(
        matrix_fn,
        objective_fn,
        jnp.asarray([2.2, 6.9]),
        step=1.0e-3,
        rtol=2.5e-2,
        atol=5.0e-4,
        gap_floor=1.0e-6,
    )

    assert report["passed"] is False
    assert report["ad_supported"] is False
    assert report["branch_isolated"] is True
    assert "non-symmetric eigenvectors" in str(report["failure_reason"])


def test_implicit_eigenpair_observable_gate_matches_closed_form_branch() -> None:
    assert gkx.implicit_eigenpair_observable_sensitivity_report is (
        implicit_eigenpair_observable_sensitivity_report
    )

    def matrix_fn(x):
        return jnp.asarray(
            [
                [0.7 + x[0] + 0.2j, 0.2 + 0.1j * x[1]],
                [0.0, -0.4 + 0.3 * x[1] - 0.1j],
            ],
            dtype=jnp.complex64,
        )

    def observable_fn(eigenvalue, eigenvector, x):
        norm = jnp.sum(jnp.abs(eigenvector) ** 2)
        participation = jnp.abs(eigenvector[0]) ** 2 / norm
        return jnp.asarray(
            [jnp.real(eigenvalue), jnp.imag(eigenvalue), participation + 0.1 * x[0]]
        )

    report = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        observable_fn,
        jnp.asarray([0.2, -0.1]),
        step=1.0e-3,
        rtol=1.0e-3,
        atol=2.0e-5,
    )

    assert report["passed"] is True
    assert report["ad_supported"] is True
    assert report["sensitivity_method"] == "implicit_left_right_eigenpair"
    assert report["observable_chain_rule"] == "split_eigenpair_and_explicit_parameter"
    jac_impl = np.asarray(report["jacobian_implicit"])
    jac_fd = np.asarray(report["jacobian_fd"])
    np.testing.assert_allclose(jac_impl, jac_fd, rtol=1.0e-3, atol=2.0e-5)


def test_implicit_eigenpair_observable_gate_handles_complex_observables() -> None:
    def matrix_fn(x):
        return jnp.asarray(
            [
                [0.8 + x[0] + 0.1j, 0.05 + 0.03j * x[1]],
                [0.0, -0.2 + 0.2 * x[1] - 0.05j],
            ],
            dtype=jnp.complex64,
        )

    def observable_fn(eigenvalue, eigenvector, x):
        norm = jnp.sum(jnp.abs(eigenvector) ** 2)
        phase_invariant_weight = jnp.abs(eigenvector[0]) ** 2 / norm
        return jnp.asarray([eigenvalue + 0.1 * x[0] + 1j * phase_invariant_weight])

    report = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        observable_fn,
        jnp.asarray([0.1, -0.2]),
        step=1.0e-3,
        rtol=1.0e-3,
        atol=3.0e-5,
    )

    assert report["passed"] is True
    assert report["observable_chain_rule"] == "split_eigenpair_and_explicit_parameter"
    assert np.asarray(report["jacobian_implicit"]).shape == (2, 2)
    np.testing.assert_allclose(
        report["jacobian_implicit"], report["jacobian_fd"], rtol=1.0e-3, atol=3.0e-5
    )


def test_actual_linear_rhs_branch_objective_implicit_derivative_gate() -> None:
    matrix_fn, objective_fn = _actual_linear_rhs_objective_functions()

    report = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        objective_fn,
        jnp.asarray([2.2, 6.9]),
        step=1.0e-3,
        rtol=3.0e-2,
        atol=7.5e-4,
        gap_floor=1.0e-6,
    )

    assert report["passed"] is True
    assert report["branch_isolated"] is True
    assert report["observable_chain_rule"] == "split_eigenpair_and_explicit_parameter"
    jac_impl = np.asarray(report["jacobian_implicit"])
    jac_fd = np.asarray(report["jacobian_fd"])
    np.testing.assert_allclose(jac_impl, jac_fd, rtol=3.0e-2, atol=7.5e-4)


def test_isolated_eigenvalue_sensitivity_report_flags_small_gaps() -> None:
    def matrix_fn(x):
        return jnp.diag(jnp.asarray([x[0], x[0] + 1.0e-9]))

    report = isolated_eigenvalue_sensitivity_report(
        matrix_fn,
        jnp.asarray([0.2]),
        step=1.0e-3,
        gap_floor=1.0e-6,
    )

    assert report["branch_isolated"] is False
    assert report["passed"] is False
    with pytest.raises(ValueError):
        isolated_eigenvalue_sensitivity_report(
            matrix_fn, jnp.asarray([0.2]), selector="index:4"
        )


def test_eigen_sensitivity_reports_validate_selectors_and_scalar_branches() -> None:
    def scalar_matrix_fn(x):
        return jnp.asarray([[0.5 + x[0] + 0.2j * x[1]]], dtype=jnp.complex64)

    value_report = isolated_eigenvalue_sensitivity_report(
        scalar_matrix_fn,
        jnp.asarray([0.1, -0.2]),
        selector="index:0",
        step=1.0e-3,
        rtol=1.0e-3,
        atol=2.0e-5,
    )
    assert value_report["passed"] is True
    assert value_report["eigenvalue_gap"] == float("inf")

    pair_report = isolated_eigenpair_observable_sensitivity_report(
        scalar_matrix_fn,
        lambda eigenvalue, eigenvector, x: jnp.asarray(
            [jnp.real(eigenvalue) + jnp.abs(eigenvector[0]) ** 2]
        ),
        jnp.asarray([0.1, -0.2]),
        selector="index:0",
        step=1.0e-3,
        rtol=1.0e-3,
        atol=2.0e-5,
    )
    assert pair_report["ad_supported"] is False
    assert pair_report["eigenvalue_gap"] == float("inf")

    with pytest.raises(ValueError):
        isolated_eigenvalue_sensitivity_report(scalar_matrix_fn, jnp.asarray([[0.1]]))
    with pytest.raises(ValueError):
        isolated_eigenvalue_sensitivity_report(
            scalar_matrix_fn, jnp.asarray([0.1]), selector="min_abs"
        )
    with pytest.raises(ValueError):
        isolated_eigenpair_observable_sensitivity_report(
            scalar_matrix_fn, lambda *_: jnp.asarray([1.0]), jnp.asarray([[0.1]])
        )
    with pytest.raises(ValueError):
        isolated_eigenpair_observable_sensitivity_report(
            scalar_matrix_fn,
            lambda *_: jnp.asarray([1.0]),
            jnp.asarray([0.1]),
            selector="index:3",
        )


def test_eigen_sensitivity_reports_cover_empty_matrices_and_ad_fallbacks(
    monkeypatch,
) -> None:
    def empty_matrix(x):
        return jnp.zeros((0, 0), dtype=jnp.complex64)

    with pytest.raises(ValueError, match="at least one eigenvalue"):
        isolated_eigenvalue_sensitivity_report(empty_matrix, jnp.asarray([0.1]))
    with pytest.raises(ValueError, match="at least one eigenvalue"):
        isolated_eigenpair_observable_sensitivity_report(
            empty_matrix, lambda *_: jnp.asarray([1.0]), jnp.asarray([0.1])
        )
    with pytest.raises(ValueError, match="at least one eigenvalue"):
        implicit_eigenpair_observable_sensitivity_report(
            empty_matrix, lambda *_: jnp.asarray([1.0]), jnp.asarray([0.1])
        )

    def matrix_fn(x):
        return jnp.asarray([[1.0 + x[0], 0.0], [0.0, -0.5 + x[0]]], dtype=jnp.complex64)

    def raise_not_implemented(*_args, **_kwargs):
        raise NotImplementedError("forced derivative fallback")

    monkeypatch.setattr(adv, "autodiff_finite_difference_report", raise_not_implemented)
    fallback = adv.isolated_eigenvalue_sensitivity_report(matrix_fn, jnp.asarray([0.1]))
    assert fallback["ad_supported"] is False
    assert "forced derivative fallback" in str(fallback["failure_reason"])


def test_isolated_eigenpair_report_realifies_complex_observable(monkeypatch) -> None:
    def matrix_fn(x):
        return jnp.asarray(
            [[1.0 + x[0] + 0.1j, 0.0], [0.0, -0.5 + 0.2j]], dtype=jnp.complex64
        )

    captured = {}

    def fake_fd_report(fn, params, **kwargs):
        values = fn(params)
        captured["values"] = np.asarray(values)
        return {
            "passed": True,
            "step": kwargs["step"],
            "rtol": kwargs["rtol"],
            "atol": kwargs["atol"],
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "tangent_max_abs_error": 0.0,
            "jacobian_ad": [[1.0], [0.0]],
            "jacobian_fd": [[1.0], [0.0]],
            "tangent_ad": [1.0, 0.0],
            "tangent_fd": [1.0, 0.0],
        }

    monkeypatch.setattr(adv, "autodiff_finite_difference_report", fake_fd_report)
    report = adv.isolated_eigenpair_observable_sensitivity_report(
        matrix_fn,
        lambda eigenvalue, eigenvector, x: jnp.asarray(
            [eigenvalue + 0.1j * jnp.abs(eigenvector[0]) ** 2]
        ),
        jnp.asarray([0.2]),
    )

    assert report["passed"] is True
    assert report["ad_supported"] is True
    assert captured["values"].shape == (2,)


def test_implicit_eigenpair_observable_report_validates_inputs() -> None:
    def matrix_fn(x):
        return jnp.asarray([[1.0 + x[0], 0.0], [0.0, -1.0 + x[0]]], dtype=jnp.complex64)

    def observable(eigenvalue, eigenvector, x):
        return jnp.asarray([jnp.real(eigenvalue)])

    with pytest.raises(ValueError):
        implicit_eigenpair_observable_sensitivity_report(
            matrix_fn, observable, jnp.asarray([[0.1]])
        )
    with pytest.raises(ValueError):
        implicit_eigenpair_observable_sensitivity_report(
            lambda x: jnp.ones((2, 3)), observable, jnp.asarray([0.1])
        )
    with pytest.raises(ValueError):
        implicit_eigenpair_observable_sensitivity_report(
            matrix_fn, observable, jnp.asarray([0.1]), selector="min_abs"
        )
    with pytest.raises(ValueError):
        implicit_eigenpair_observable_sensitivity_report(
            matrix_fn, observable, jnp.asarray([0.1]), selector="index:9"
        )


def test_autodiff_finite_difference_report_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        central_finite_difference_jacobian(lambda x: x, jnp.ones((2, 1)))
    with pytest.raises(ValueError):
        central_finite_difference_jacobian(lambda x: x, jnp.ones(2), step=0.0)
    with pytest.raises(ValueError):
        central_finite_difference_jacobian(lambda x: x, jnp.ones(2), workers=0)
    with pytest.raises(ValueError):
        central_finite_difference_jacobian(
            lambda x: x, jnp.ones(2), workers=2, parallel_executor="process"
        )
    with pytest.raises(ValueError):
        autodiff_finite_difference_report(lambda x: x, jnp.ones((2, 1)))
    with pytest.raises(ValueError):
        autodiff_finite_difference_report(
            lambda x: x, jnp.ones(2), direction=jnp.ones(3)
        )
    with pytest.raises(ValueError):
        autodiff_finite_difference_report(lambda x: x, jnp.ones(2), workers=0)
    with pytest.raises(ValueError):
        explicit_complex_operator_matrix(lambda x: x, (0,))
    with pytest.raises(ValueError):
        explicit_complex_operator_matrix(lambda x: jnp.zeros((2,), dtype=x.dtype), (1,))


# ---- test_solver_objective_gradients.py ----

import gkx.objectives.sampling as sampling
from gkx import (
    AdaptiveLinearEigensolverConfig,
    SOLVER_OBJECTIVE_NAMES,
    SolverScalarObjective,
    dominant_eigenvalue_branch_locality_report,
    dominant_real_eigenvalue,
    solver_linear_operator_matrix_from_geometry,
    solver_growth_rate_from_geometry,
    solver_objective_vector_from_geometry,
    solver_grid_options_from_ky_values,
    solver_scalar_objective_from_vector,
)
from gkx.geometry.vmec_state_controls import _vmec_boozer_state_parameter_name


def default_solver_geometry_design_params() -> jnp.ndarray:
    """Return the small two-parameter geometry design vector these tests use."""

    return jnp.asarray([0.05, 0.20], dtype=jnp.float32)


def solver_ready_geometry_mapping(
    params: jnp.ndarray, theta: jnp.ndarray
) -> dict[str, Any]:
    """Map ``[bmag_ripple, curvature_drift_scale]`` onto solver-ready arrays."""

    ripple, drift = jnp.asarray(params)[0], jnp.asarray(params)[1]
    theta_arr = jnp.asarray(theta)
    ones = jnp.ones_like(theta_arr)
    zeros = jnp.zeros_like(theta_arr)
    bmag = 1.0 + ripple * jnp.cos(theta_arr)
    return {
        "theta": theta_arr,
        "gradpar": 0.7 * ones,
        "bmag": bmag,
        "bgrad": -ripple * jnp.sin(theta_arr),
        "gds2": 1.0 + 0.1 * ripple * jnp.cos(theta_arr),
        "gds21": 0.05 * ripple * jnp.sin(theta_arr),
        "gds22": 1.0 + 0.05 * ripple * jnp.cos(theta_arr),
        "cvdrift": drift * jnp.cos(theta_arr),
        "gbdrift": drift * jnp.cos(theta_arr),
        "cvdrift0": zeros,
        "gbdrift0": zeros,
        "jacobian": ones / (0.7 * bmag),
        "grho": ones,
        "q": 1.4,
        "s_hat": 0.0,
        "R0": 1.0,
        "nfp": 1,
    }


def test_vmec_boozer_state_parameter_name_tracks_default_and_explicit_modes() -> None:
    assert (
        _vmec_boozer_state_parameter_name("Rcos", 17, 1, default_mid_surface=17)
        == "Rcos_mid_surface_m1"
    )
    assert (
        _vmec_boozer_state_parameter_name("Rcos", 11, 2, default_mid_surface=17)
        == "Rcos_r11_m2"
    )
    assert (
        _vmec_boozer_state_parameter_name("Zsin", 17, 2, default_mid_surface=17)
        == "Zsin_mid_surface_m2"
    )


def test_solver_objective_vector_from_geometry_is_finite_and_exported() -> None:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 4, endpoint=False)
    geom = gkx.flux_tube_geometry_from_mapping(
        solver_ready_geometry_mapping(default_solver_geometry_design_params(), theta),
        validate_finite=False,
    )

    vector = solver_objective_vector_from_geometry(
        geom,
        n_laguerre=1,
        n_hermite=1,
        ny=4,
        selected_ky_index=1,
    )

    assert (
        gkx.solver_objective_vector_from_geometry
        is solver_objective_vector_from_geometry
    )
    assert vector.shape == (len(SOLVER_OBJECTIVE_NAMES),)
    assert np.all(np.isfinite(np.asarray(vector)))
    with pytest.raises(ValueError, match="selected_ky_index"):
        solver_objective_vector_from_geometry(geom, selected_ky_index=99)
    with pytest.raises(ValueError, match="positive"):
        solver_objective_vector_from_geometry(geom, n_laguerre=0)


def _sparse_direct_geometry(parameter: jnp.ndarray):
    theta = jnp.linspace(-jnp.pi, jnp.pi, 16, endpoint=False)
    return gkx.flux_tube_geometry_from_mapping(
        solver_ready_geometry_mapping(parameter, theta), validate_finite=False
    )


_SPARSE_DIRECT_GRID = dict(n_laguerre=2, n_hermite=3, ny=4, selected_ky_index=1)


def _sparse_direct_growth(parameter: jnp.ndarray, **kwargs) -> jnp.ndarray:
    return solver_growth_rate_from_geometry(
        _sparse_direct_geometry(parameter), **_SPARSE_DIRECT_GRID, **kwargs
    )


def test_sparse_direct_growth_rate_rejects_bad_selection() -> None:
    base = jnp.asarray([0.05, 0.20])
    with pytest.raises(ValueError, match="requires a shift"):
        _sparse_direct_growth(base, eigensolver="sparse-direct")
    with pytest.raises(ValueError, match="eigensolver must be"):
        _sparse_direct_growth(base, eigensolver="arnoldi")


@requires_paired_solvax("sparse_eigenvalue", "csr_data_from_products")
def test_sparse_direct_growth_rate_matches_dense_value_and_gradient() -> None:
    """One shifted host factor gives the dense growth rate and its gradient."""

    if not bool(jax.config.read("jax_enable_x64")):
        pytest.skip("the 1e-8 agreement gate is a float64 gate")
    base = jnp.asarray([0.05, 0.20])
    dense_value, dense_grad = jax.value_and_grad(_sparse_direct_growth)(base)
    spectrum = np.linalg.eigvals(
        np.asarray(
            solver_linear_operator_matrix_from_geometry(
                _sparse_direct_geometry(base), **_SPARSE_DIRECT_GRID
            )
        )
    )
    top = spectrum[np.argmax(spectrum.real)]
    shift = complex(round(top.real, 2), round(top.imag, 2))

    def sparse(parameter):
        return _sparse_direct_growth(
            parameter, eigensolver="sparse-direct", shift=shift
        )

    value, grad = jax.value_and_grad(sparse)(base)
    np.testing.assert_allclose(value, dense_value, rtol=1e-10)
    np.testing.assert_allclose(grad, dense_grad, rtol=1e-8, atol=1e-12)


@requires_solvax_reverse_eigenpair
def test_adaptive_solver_objective_matches_dense_and_implicit_gradient() -> None:
    """The matrix-free primal and bordered tangent must preserve QL observables."""

    dtype = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32
    step = 2.0e-5 if dtype == jnp.float64 else 2.0e-3
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False, dtype=dtype)
    base = jnp.asarray([0.05, 0.20], dtype=dtype)
    direction = jnp.asarray([0.3, -0.2], dtype=dtype)
    weights = jnp.asarray([1.0, 0.2, 0.1, 0.05, 0.0, 0.01], dtype=dtype)

    def objective_vector(
        parameter: jnp.ndarray,
        *,
        eigensolver: str,
        adaptive_config: AdaptiveLinearEigensolverConfig | None = None,
    ) -> jnp.ndarray:
        geom = gkx.flux_tube_geometry_from_mapping(
            solver_ready_geometry_mapping(parameter, theta),
            validate_finite=False,
        )
        return solver_objective_vector_from_geometry(
            geom,
            n_laguerre=2,
            n_hermite=1,
            ny=4,
            selected_ky_index=1,
            eigensolver=eigensolver,  # type: ignore[arg-type]
            adaptive_config=adaptive_config,
        )

    dense = objective_vector(base, eigensolver="dense")
    adaptive = objective_vector(base, eigensolver="adaptive-propagator")
    np.testing.assert_allclose(
        np.asarray(adaptive),
        np.asarray(dense),
        rtol=2.0e-8 if dtype == jnp.float64 else 2.0e-3,
        atol=2.0e-9 if dtype == jnp.float64 else 2.0e-4,
    )
    exponential_config = None
    if callable(getattr(solvax, "exponential_eigenpairs", None)):
        exponential_config = AdaptiveLinearEigensolverConfig(
            krylov_dim=15,
            restart_krylov_dim=12,
            candidate_count=2,
            max_restarts=1,
            tolerance=1.0e-9 if dtype == jnp.float64 else 2.0e-4,
            exponential_krylov_dim=16,
            exponential_horizon=10.0,
            adjoint_krylov_dim=15,
            sensitivity_solver="gmres",
        )
        exponential = objective_vector(
            base,
            eigensolver="adaptive-propagator",
            adaptive_config=exponential_config,
        )
        np.testing.assert_allclose(
            np.asarray(exponential),
            np.asarray(dense),
            rtol=2.0e-8 if dtype == jnp.float64 else 2.0e-3,
            atol=2.0e-9 if dtype == jnp.float64 else 2.0e-4,
        )

    def scalar(offset: jnp.ndarray) -> jnp.ndarray:
        return jnp.vdot(
            weights,
            objective_vector(
                base + offset * direction,
                eigensolver="adaptive-propagator",
            ),
        )

    implicit = float(jax.grad(scalar)(jnp.asarray(0.0, dtype=dtype)))
    finite_difference = float(
        (
            scalar(jnp.asarray(step, dtype=dtype))
            - scalar(jnp.asarray(-step, dtype=dtype))
        )
        / (2.0 * step)
    )
    assert implicit == pytest.approx(
        finite_difference,
        rel=2.0e-5 if dtype == jnp.float64 else 2.0e-2,
        abs=2.0e-7 if dtype == jnp.float64 else 2.0e-3,
    )

    if exponential_config is not None:

        def exponential_scalar(offset: jnp.ndarray) -> jnp.ndarray:
            return jnp.vdot(
                weights,
                objective_vector(
                    base + offset * direction,
                    eigensolver="adaptive-propagator",
                    adaptive_config=exponential_config,
                ),
            )

        exponential_gradient = float(
            jax.grad(exponential_scalar)(jnp.asarray(0.0, dtype=dtype))
        )
        assert exponential_gradient == pytest.approx(
            implicit,
            rel=2.0e-5 if dtype == jnp.float64 else 2.0e-2,
            abs=2.0e-7 if dtype == jnp.float64 else 2.0e-3,
        )
    assert gkx.AdaptiveLinearEigensolverConfig is not None


def test_solver_objective_accepts_runtime_grid_and_multi_species_state() -> None:
    """The objective adapter must preserve linked-grid and species dimensions."""

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.0, Ly=12.0))
    full_grid = build_spectral_grid(cfg.grid)
    grid = select_ky_grid(full_grid, 1)
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    geometry = sample_flux_tube_geometry(analytic, grid.z)
    params = build_linear_params(
        (
            Species(1.0, 1.0, 1.0, 1.0, 6.9, 2.2),
            Species(-1.0, 5.0e-4, 1.0, 1.0, 6.9, 2.2),
        ),
        beta=0.01,
        fapar=1.0,
    )
    terms = LinearTerms(
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=1.0,
        bpar=1.0,
    )

    matrix = solver_linear_operator_matrix_from_geometry(
        geometry,
        spectral_grid=grid,
        n_laguerre=1,
        n_hermite=2,
        params_linear=params,
        terms=terms,
    )
    vector = solver_objective_vector_from_geometry(
        geometry,
        spectral_grid=grid,
        n_laguerre=1,
        n_hermite=2,
        params_linear=params,
        terms=terms,
    )

    assert matrix.shape == (32, 32)
    assert np.all(np.isfinite(np.asarray(vector)))
    with pytest.raises(ValueError, match="exactly one ky"):
        solver_objective_vector_from_geometry(
            geometry,
            spectral_grid=full_grid,
            n_laguerre=1,
            n_hermite=2,
            params_linear=params,
            terms=terms,
        )


def test_solver_scalar_objective_selector_aliases_and_errors() -> None:
    vector = jnp.asarray([1.0, -0.5, 2.0, 3.0, 4.0, 5.0])

    assert gkx.SolverScalarObjective is SolverScalarObjective
    assert (
        gkx.solver_scalar_objective_from_vector is solver_scalar_objective_from_vector
    )
    assert float(
        solver_scalar_objective_from_vector(vector, "growth")
    ) == pytest.approx(1.0)
    assert float(solver_scalar_objective_from_vector(vector, "gamma")) == pytest.approx(
        1.0
    )
    assert float(
        solver_scalar_objective_from_vector(vector, "frequency")
    ) == pytest.approx(-0.5)
    assert float(
        solver_scalar_objective_from_vector(vector, "quasilinear_flux")
    ) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="unknown solver objective"):
        solver_scalar_objective_from_vector(vector, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length"):
        solver_scalar_objective_from_vector(jnp.ones(2), "growth")


def test_dominant_real_eigenvalue_custom_vjp_matches_finite_difference() -> None:
    x64_enabled = bool(jax.config.read("jax_enable_x64"))
    dtype = jnp.float64 if x64_enabled else jnp.float32
    step = 1.0e-5 if x64_enabled else 2.0e-3
    rtol = 1.0e-4 if x64_enabled else 5.0e-2
    atol = 1.0e-6 if x64_enabled else 5.0e-4
    params = jnp.asarray([0.3, -0.2, 0.5, 0.1, 0.7], dtype=dtype)

    def matrix_from_params(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            [
                [1.0 + 0.2j * x[0], x[0] + 1j * x[1], 0.1 + 0.2j * x[4]],
                [0.2 * x[2] - 0.1j, -0.3 + 0.4j * x[3], 0.05 + 0.1j * x[1]],
                [0.01 + x[4], -0.2j * x[2], 0.2 + 0.3j],
            ],
            dtype=jnp.complex128 if x64_enabled else jnp.complex64,
        )

    def objective(x: jnp.ndarray) -> jnp.ndarray:
        return dominant_real_eigenvalue(matrix_from_params(x))

    grad_ad = np.asarray(jax.grad(objective)(params), dtype=float)
    eye = jnp.eye(int(params.size), dtype=params.dtype)
    grad_fd = []
    for index in range(int(params.size)):
        plus = objective(params + step * eye[index])
        minus = objective(params - step * eye[index])
        grad_fd.append(float((plus - minus) / (2.0 * step)))

    assert gkx.dominant_real_eigenvalue is dominant_real_eigenvalue
    assert np.all(np.isfinite(grad_ad))
    np.testing.assert_allclose(grad_ad, np.asarray(grad_fd), rtol=rtol, atol=atol)


def test_dominant_real_eigenvalue_validates_shape_and_casts_real_matrices() -> None:
    assert float(
        dominant_real_eigenvalue(jnp.diag(jnp.asarray([1.0, 2.0])))
    ) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="matrix must be square"):
        dominant_real_eigenvalue(jnp.ones((2, 3)))


def test_dominant_eigenvalue_branch_locality_report_accepts_isolated_branch() -> None:
    base = jnp.diag(jnp.asarray([1.0 + 0.2j, 0.5 - 0.1j, -0.2 + 0.0j]))
    plus = jnp.diag(jnp.asarray([1.02 + 0.21j, 0.48 - 0.1j, -0.2 + 0.0j]))
    minus = jnp.diag(jnp.asarray([0.98 + 0.19j, 0.52 - 0.1j, -0.2 + 0.0j]))

    report = dominant_eigenvalue_branch_locality_report(
        base,
        plus,
        minus,
        step=1.0e-2,
        gap_floor=1.0e-3,
    )

    assert (
        gkx.dominant_eigenvalue_branch_locality_report
        is dominant_eigenvalue_branch_locality_report
    )
    assert report["passed"] is True
    assert report["classification"] == "dominant_branch_locally_consistent"
    assert report["base_selected_index"] == 0
    assert report["dominant_growth_fd_slope"] == pytest.approx(2.0)
    assert report["nearest_branch_growth_fd_slope"] == pytest.approx(2.0)
    assert report["slope_relative_difference"] == pytest.approx(0.0)
    assert all(row["dominant_matches_nearest"] for row in report["branch_rows"])


def test_dominant_eigenvalue_branch_locality_report_rejects_branch_switch_fd() -> None:
    base = jnp.diag(jnp.asarray([1.0 + 0.0j, 0.8 + 0.0j, -0.1 + 0.0j]))
    plus = jnp.diag(jnp.asarray([0.92 + 0.0j, 1.12 + 0.0j, -0.1 + 0.0j]))
    minus = jnp.diag(jnp.asarray([1.08 + 0.0j, 0.7 + 0.0j, -0.1 + 0.0j]))

    report = dominant_eigenvalue_branch_locality_report(
        base,
        plus,
        minus,
        step=1.0e-2,
        gap_floor=1.0e-3,
    )

    assert report["passed"] is False
    assert report["classification"] == "dominant_branch_differs_from_nearest_branch"
    assert report["dominant_growth_fd_slope"] == pytest.approx(2.0)
    assert report["nearest_branch_growth_fd_slope"] == pytest.approx(-8.0)
    assert report["slope_relative_difference"] > 1.0
    branch_rows = list(report["branch_rows"])
    assert branch_rows[0]["side"] == "minus"
    assert branch_rows[0]["dominant_matches_nearest"] is True
    assert branch_rows[1]["side"] == "plus"
    assert branch_rows[1]["dominant_matches_nearest"] is False
    assert "do not use dominant-growth finite differences" in str(report["next_action"])

    with pytest.raises(ValueError, match="positive"):
        dominant_eigenvalue_branch_locality_report(base, plus, minus, step=0.0)
    with pytest.raises(ValueError, match="same eigenvalue count"):
        dominant_eigenvalue_branch_locality_report(
            base,
            jnp.diag(jnp.asarray([1.0, 2.0])),
            minus,
            step=1.0e-2,
        )


def test_solver_growth_rate_from_geometry_has_finite_fd_checked_gradient() -> None:
    x64_enabled = bool(jax.config.read("jax_enable_x64"))
    dtype = jnp.float64 if x64_enabled else jnp.float32
    step = 2.0e-4 if x64_enabled else 2.0e-3
    rtol = 5.0e-2 if x64_enabled else 2.0e-1
    atol = 2.0e-4 if x64_enabled else 2.0e-3
    params = default_solver_geometry_design_params().astype(dtype)
    theta = jnp.linspace(-jnp.pi, jnp.pi, 4, endpoint=False, dtype=dtype)

    def objective(x: jnp.ndarray) -> jnp.ndarray:
        geom = gkx.flux_tube_geometry_from_mapping(
            solver_ready_geometry_mapping(x, theta),
            source_model="solver_growth_custom_vjp_gate",
            validate_finite=False,
        )
        return solver_growth_rate_from_geometry(
            geom,
            n_laguerre=1,
            n_hermite=1,
            ny=4,
            selected_ky_index=1,
        )

    grad_ad = np.asarray(jax.grad(objective)(params), dtype=float)
    eye = jnp.eye(int(params.size), dtype=params.dtype)
    grad_fd = []
    for index in range(int(params.size)):
        plus = objective(params + step * eye[index])
        minus = objective(params - step * eye[index])
        grad_fd.append(float((plus - minus) / (2.0 * step)))

    assert np.all(np.isfinite(grad_ad))
    np.testing.assert_allclose(grad_ad, np.asarray(grad_fd), rtol=rtol, atol=atol)


def test_solver_linear_operator_matrix_matches_growth_rate_path() -> None:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 4, endpoint=False)
    geom = gkx.flux_tube_geometry_from_mapping(
        solver_ready_geometry_mapping(default_solver_geometry_design_params(), theta),
        validate_finite=False,
    )

    matrix = solver_linear_operator_matrix_from_geometry(
        geom,
        n_laguerre=1,
        n_hermite=1,
        ny=4,
        selected_ky_index=1,
    )
    growth = solver_growth_rate_from_geometry(
        geom,
        n_laguerre=1,
        n_hermite=1,
        ny=4,
        selected_ky_index=1,
    )

    assert (
        gkx.solver_linear_operator_matrix_from_geometry
        is solver_linear_operator_matrix_from_geometry
    )
    assert matrix.shape == (4, 4)
    assert np.all(np.isfinite(np.asarray(matrix)))
    assert float(
        np.max(np.real(np.linalg.eigvals(np.asarray(matrix))))
    ) == pytest.approx(float(growth))


def test_solver_growth_rate_from_geometry_validates_small_grid_contracts() -> None:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 4, endpoint=False)
    geom = gkx.flux_tube_geometry_from_mapping(
        solver_ready_geometry_mapping(default_solver_geometry_design_params(), theta),
        source_model="solver_growth_contract_gate",
        validate_finite=False,
    )

    with pytest.raises(ValueError, match="positive"):
        solver_growth_rate_from_geometry(geom, n_laguerre=0)
    with pytest.raises(ValueError, match="selected_ky_index"):
        solver_growth_rate_from_geometry(geom, ny=4, selected_ky_index=99)

    class EmptyThetaGeometry:
        theta = jnp.asarray([])

    with pytest.raises(ValueError, match="at least one theta"):
        solver_growth_rate_from_geometry(EmptyThetaGeometry())


def test_solver_grid_options_from_ky_values_maps_physical_scan_to_fft_rows() -> None:
    options = solver_grid_options_from_ky_values((0.1, 0.3, 0.5))

    assert gkx.solver_grid_options_from_ky_values is solver_grid_options_from_ky_values
    assert options["selected_ky_indices"] == (1, 3, 5)
    assert options["ny"] == 12
    assert float(options["ly"]) == pytest.approx(2.0 * np.pi / 0.1)
    np.testing.assert_allclose(
        options["resolved_ky_values"], (0.1, 0.3, 0.5), rtol=5.0e-6, atol=5.0e-8
    )

    shifted = solver_grid_options_from_ky_values((0.15, 0.35), ky_base=0.05)
    assert shifted["selected_ky_indices"] == (3, 7)
    assert shifted["ny"] == 16
    np.testing.assert_allclose(
        shifted["resolved_ky_values"], (0.15, 0.35), rtol=5.0e-6, atol=5.0e-8
    )
    with pytest.raises(ValueError, match="integer multiples"):
        solver_grid_options_from_ky_values((0.15, 0.35))
    with pytest.raises(ValueError, match="duplicate"):
        solver_grid_options_from_ky_values((0.1, 0.1))
    with pytest.raises(ValueError, match="ky_base"):
        solver_grid_options_from_ky_values((0.1, 0.2), ky_base=0.0)
    with pytest.raises(ValueError, match="positive"):
        solver_grid_options_from_ky_values((0.0, 0.1))


def test_solver_objective_sampling_helpers_validate_contracts() -> None:
    assert sampling._surface_index_tuple(None) == (None,)
    assert sampling._surface_index_tuple(3) == (3,)
    with pytest.raises(ValueError, match="surface_indices"):
        sampling._surface_index_tuple([])

    assert sampling._int_tuple(2, name="selected_ky_indices") == (2,)
    with pytest.raises(ValueError, match="selected_ky_indices"):
        sampling._int_tuple([], name="selected_ky_indices")

    assert sampling._float_tuple(0.3, name="ky_values") == (0.3,)
    with pytest.raises(ValueError, match="ky_values"):
        sampling._float_tuple([], name="ky_values")
    with pytest.raises(ValueError, match="finite"):
        sampling._float_tuple([0.1, float("nan")], name="ky_values")

    np.testing.assert_allclose(
        sampling._aggregate_weights(None, 3), np.full(3, 1.0 / 3.0)
    )
    np.testing.assert_allclose(sampling._aggregate_weights([1.0, 3.0], 2), [0.25, 0.75])
    with pytest.raises(ValueError, match="positive"):
        sampling._aggregate_weights([0.0, 0.0], 2)
    with pytest.raises(ValueError, match="finite"):
        sampling._aggregate_weights([1.0, float("nan")], 2)

    rows = sampling._aggregate_sample_metadata(
        (None, 4), (0.0,), (1, 2), np.asarray([0.1, 0.2, 0.3, 0.4])
    )
    assert rows == [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.1},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.2},
        {"surface_index": 4, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.3},
        {"surface_index": 4, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.4},
    ]


# ---- test_stellarator_objective_portfolio.py ----


from scripts.campaigns.portfolio_guard import (
    ReducedPortfolioArtifactGuardConfig,
    reduced_portfolio_artifact_guard_report,
)
from gkx.objectives.portfolio import (
    aggregate_objective_portfolio,
    portfolio_objective_weight_vector,
    portfolio_sample_weight_tensor,
    validate_objective_portfolio_contract,
)
from gkx.objectives.portfolio import (
    objective_portfolio_sensitivity_report,
)


def test_weighted_objective_portfolio_matches_manual_reduction() -> None:
    rows = jnp.arange(1.0, 1.0 + 2 * 2 * 2 * 3, dtype=jnp.float32).reshape((2, 2, 2, 3))
    surface_weights = jnp.asarray([2.0, 1.0])
    alpha_weights = jnp.asarray([1.0, 3.0])
    ky_weights = jnp.asarray([1.0, 2.0])
    objective_weights = jnp.asarray([0.5, 1.5, 1.0])

    value = aggregate_objective_portfolio(
        rows,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        objective_weights=objective_weights,
    )

    surface = np.asarray(surface_weights / jnp.sum(surface_weights))
    alpha = np.asarray(alpha_weights / jnp.sum(alpha_weights))
    ky = np.asarray(ky_weights / jnp.sum(ky_weights))
    objective = np.asarray(objective_weights / jnp.sum(objective_weights))
    sample = surface[:, None, None] * alpha[None, :, None] * ky[None, None, :]
    expected = np.sum(np.asarray(rows) * sample[..., None] * objective)

    np.testing.assert_allclose(float(value), float(expected), rtol=1.0e-6)
    np.testing.assert_allclose(
        float(
            jnp.sum(
                portfolio_sample_weight_tensor(rows, surface_weights=surface_weights)
            )
        ),
        1.0,
    )
    np.testing.assert_allclose(
        float(
            jnp.sum(
                portfolio_objective_weight_vector(
                    rows, objective_weights=objective_weights
                )
            )
        ),
        1.0,
    )

    contract = validate_objective_portfolio_contract(
        rows,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        objective_weights=objective_weights,
    )
    assert contract.row_shape == (2, 2, 2, 3)
    assert contract.sample_shape == (2, 2, 2)
    assert contract.n_samples == 8
    assert contract.uses_separable_sample_weights is True
    assert contract.uses_objective_weights is True
    assert contract.to_dict()["row_shape"] == [2, 2, 2, 3]


def test_objective_portfolio_gradient_jvp_and_finite_difference_parity() -> None:
    surface_weights = jnp.asarray([1.0, 2.0])
    alpha_weights = jnp.asarray([1.0, 3.0])
    ky_weights = jnp.asarray([2.0, 1.0, 1.5])
    objective_weights = jnp.asarray([0.75, 1.25])

    surface = jnp.asarray([0.2, 0.7])[:, None, None, None]
    alpha = jnp.asarray([-0.35, 0.45])[None, :, None, None]
    ky = jnp.asarray([0.15, 0.55, 0.9])[None, None, :, None]
    objective = jnp.asarray([0.8, 1.6])[None, None, None, :]

    def objective_fn(params: jnp.ndarray) -> jnp.ndarray:
        rows = (
            objective * params[0] ** 2
            + jnp.sin(params[1] + alpha) * (1.0 + surface)
            + params[2] * ky
            + 0.1 * params[0] * params[2] * surface * objective
        )
        return aggregate_objective_portfolio(
            rows,
            surface_weights=surface_weights,
            alpha_weights=alpha_weights,
            ky_weights=ky_weights,
            objective_weights=objective_weights,
        )

    params = jnp.asarray([0.42, -0.18, 0.31])
    direction = jnp.asarray([0.25, -0.40, 0.15])
    grad = jax.grad(objective_fn)(params)
    _value, tangent = jax.jvp(objective_fn, (params,), (direction,))
    step = 1.0e-3
    finite_difference = (
        objective_fn(params + step * direction)
        - objective_fn(params - step * direction)
    ) / (2.0 * step)

    np.testing.assert_allclose(
        float(tangent), float(jnp.vdot(grad, direction)), rtol=2.0e-5, atol=2.0e-5
    )
    np.testing.assert_allclose(
        float(tangent), float(finite_difference), rtol=1.5e-3, atol=1.5e-3
    )


def test_objective_portfolio_sensitivity_report_checks_fd_and_conditioning() -> None:
    surface = jnp.asarray([-0.4, 0.7])[:, None, None]
    alpha = jnp.asarray([-0.5, 0.3])[None, :, None]
    ky = jnp.asarray([0.2, 0.6, 1.0])[None, None, :]

    def row_fn(params: jnp.ndarray) -> jnp.ndarray:
        gamma = 0.15 + 0.08 * params[0] + 0.04 * alpha + 0.03 * ky
        kperp = 0.45 + 0.05 * params[1] ** 2 + 0.08 * ky + 0.02 * surface
        flux = (
            0.30
            + 0.09 * params[2]
            + 0.04 * jnp.sin(params[1] + alpha)
            + 0.02 * surface * ky
        )
        ql_flux = gamma * flux / kperp
        return jnp.stack(
            [
                gamma + jnp.zeros_like(ql_flux),
                kperp + jnp.zeros_like(ql_flux),
                ql_flux,
            ],
            axis=-1,
        )

    params = jnp.asarray([0.12, -0.20, 0.35])
    step = 1.0e-4 if bool(jax.config.jax_enable_x64) else 2.0e-3
    rtol = 5.0e-4 if bool(jax.config.jax_enable_x64) else 2.0e-2
    atol = 1.0e-5 if bool(jax.config.jax_enable_x64) else 2.0e-4

    report = objective_portfolio_sensitivity_report(
        row_fn,
        params,
        surface_weights=jnp.asarray([1.0, 2.0]),
        alpha_weights=jnp.asarray([2.0, 1.0]),
        ky_weights=jnp.asarray([1.0, 2.0, 3.0]),
        objective_weights=jnp.asarray([0.2, 0.2, 1.0]),
        step=step,
        rtol=rtol,
        atol=atol,
        min_rank=3,
        condition_number_limit=1.0e4,
        workers=2,
    )

    assert report["passed"] is True
    assert report["portfolio_contract"]["row_shape"] == [2, 2, 3, 3]
    assert report["scalar_gradient_gate"]["passed"] is True
    assert report["row_jacobian_gate"]["passed"] is True
    assert report["conditioning_gate"]["passed"] is True
    assert report["conditioning_gate"]["sensitivity_map_rank"] == 3
    assert (
        report["scalar_gradient_gate"]["finite_difference_parallel"][
            "requested_workers"
        ]
        == 2
    )
    assert report["covariance"]["source"] == "objective_portfolio_rows"


def test_objective_portfolio_sensitivity_report_fails_rank_deficient_rows() -> None:
    sample_axis = jnp.arange(4.0).reshape((1, 1, 4))

    def rank_deficient_row_fn(params: jnp.ndarray) -> jnp.ndarray:
        row = 0.2 + params[0] * (1.0 + sample_axis)
        return row[..., None]

    report = objective_portfolio_sensitivity_report(
        rank_deficient_row_fn,
        jnp.asarray([0.1, -0.2]),
        reduction="mean",
        step=1.0e-3,
        rtol=2.0e-2,
        atol=2.0e-4,
        min_rank=2,
        condition_number_limit=1.0e4,
    )

    assert report["passed"] is False
    assert report["scalar_gradient_gate"]["passed"] is True
    assert report["row_jacobian_gate"]["passed"] is True
    assert report["conditioning_gate"]["passed"] is False
    assert report["conditioning_gate"]["rank_deficiency"] == 1


def test_objective_portfolio_rejects_invalid_shape_and_weights() -> None:
    rows = jnp.ones((2, 2, 2, 2))

    with pytest.raises(ValueError, match="objective_rows"):
        aggregate_objective_portfolio(jnp.ones((2, 2, 2)))

    with pytest.raises(ValueError, match="sample_weights"):
        aggregate_objective_portfolio(rows, sample_weights=jnp.ones((2, 2)))

    with pytest.raises(ValueError, match="either sample_weights"):
        aggregate_objective_portfolio(
            rows, sample_weights=jnp.ones((2, 2, 2)), surface_weights=jnp.ones(2)
        )

    with pytest.raises(ValueError, match="surface_weights"):
        aggregate_objective_portfolio(rows, surface_weights=jnp.asarray([1.0, -0.2]))

    with pytest.raises(ValueError, match="alpha_weights"):
        aggregate_objective_portfolio(rows, alpha_weights=jnp.asarray([0.0, 0.0]))

    with pytest.raises(ValueError, match="ky_weights"):
        aggregate_objective_portfolio(rows, ky_weights=jnp.asarray([1.0, jnp.nan]))

    with pytest.raises(ValueError, match="objective_weights"):
        aggregate_objective_portfolio(rows, objective_weights=jnp.ones(3))

    with pytest.raises(ValueError, match="mean reduction"):
        aggregate_objective_portfolio(
            rows, surface_weights=jnp.ones(2), reduction="mean"
        )

    with pytest.raises(ValueError, match="max reduction"):
        aggregate_objective_portfolio(
            rows, sample_weights=jnp.ones((2, 2, 2)), reduction="max"
        )

    with pytest.raises(TypeError, match="real numeric"):
        aggregate_objective_portfolio(jnp.ones((1, 1, 1, 1), dtype=jnp.complex64))

    with pytest.raises(ValueError, match="params"):
        objective_portfolio_sensitivity_report(lambda _p: rows, jnp.ones((1, 1)))


def test_objective_portfolio_mean_and_max_reductions_are_explicit() -> None:
    rows = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    ).reshape((2, 2, 1, 2))

    mean_contract = validate_objective_portfolio_contract(rows, reduction="mean")
    max_contract = validate_objective_portfolio_contract(
        rows,
        objective_weights=jnp.asarray([1.0, 3.0]),
        reduction="max",
    )

    assert mean_contract.reduction == "mean"
    assert max_contract.reduction == "max"
    np.testing.assert_allclose(
        float(aggregate_objective_portfolio(rows, reduction="mean")), 4.5
    )
    np.testing.assert_allclose(
        float(
            aggregate_objective_portfolio(
                rows,
                objective_weights=jnp.asarray([1.0, 3.0]),
                reduction="max",
            )
        ),
        7.75,
    )


def test_objective_portfolio_helpers_are_exported_at_package_top_level() -> None:
    import gkx as sgk

    rows = jnp.ones((1, 1, 2, 2))
    contract = sgk.validate_objective_portfolio_contract(rows)

    assert isinstance(contract, sgk.StellaratorObjectivePortfolioContract)
    np.testing.assert_allclose(float(sgk.aggregate_objective_portfolio(rows)), 1.0)
    assert (
        sgk.objective_portfolio_sensitivity_report
        is objective_portfolio_sensitivity_report
    )
    # The reduced-portfolio artifact guard is campaign promotion policy, not
    # solver API: it ships in scripts/campaigns/ and is deliberately absent from
    # the installable package's top level.
    assert isinstance(
        ReducedPortfolioArtifactGuardConfig(), ReducedPortfolioArtifactGuardConfig
    )
    assert not hasattr(sgk, "ReducedPortfolioArtifactGuardConfig")
    assert not hasattr(sgk, "reduced_portfolio_artifact_guard_report")
    assert callable(reduced_portfolio_artifact_guard_report)


# ---- test_stellarator_optimization.py ----

import gkx.objectives.stellarator as so
from gkx.objectives.stellarator import (
    OBSERVABLE_NAMES,
    PARAMETER_NAMES,
    StellaratorITGOptimizationConfig,
    StellaratorITGOptimizationResult,
    StellaratorITGSampleSet,
    compare_stellarator_itg_objectives,
    default_stellarator_initial_params,
    nonlinear_heat_flux_trace,
    nonlinear_heat_flux_window_metrics,
    optimize_stellarator_itg,
    qa_observable_vector,
    stellarator_itg_density_gradient_scan,
    stellarator_itg_objective,
    stellarator_itg_objective_residual_names,
    stellarator_itg_objective_residual_vector,
    stellarator_itg_portfolio_gate_payload,
    stellarator_itg_portfolio_sensitivity_report,
    stellarator_itg_reduced_portfolio_objective,
    stellarator_itg_residual_sensitivity_report,
    stellarator_itg_sample_objective_table,
    stellarator_itg_vmec_boozer_portfolio_objective_from_state,
    stellarator_itg_vmec_boozer_sample_objective_table_from_state,
)


def _fast_config() -> StellaratorITGOptimizationConfig:
    return StellaratorITGOptimizationConfig(
        nonlinear_dt=0.18,
        nonlinear_steps=96,
        nonlinear_tail_fraction=0.30,
        fd_step=1.0e-4 if bool(jax.config.jax_enable_x64) else 5.0e-3,
    )


def _fd_tolerances() -> tuple[float, float]:
    if bool(jax.config.jax_enable_x64):
        return 5.0e-3, 6.0e-4
    return 5.0e-2, 6.0e-3


def _disable_optional_backend_discovery(monkeypatch) -> None:
    monkeypatch.setattr(
        so,
        "discover_differentiable_geometry_backends",
        lambda: {
            "vmex_available": False,
            "vmex_boundary_api_available": False,
            "booz_xform_jax_available": False,
            "booz_xform_jax_api_available": False,
        },
    )


def test_public_optimization_examples_exclude_reduced_synthetic_workflows() -> None:
    examples = REPO_ROOT / "examples" / "10_vmex_optimization"
    names = {path.name for path in examples.iterdir() if path.is_file()}

    assert "run.py" in names
    assert not any(name.startswith("stellarator_itg_") for name in names)
    assert "_stellarator_itg_plotting.py" not in names
    assert "compare_stellarator_itg_optimizations.py" not in names


def test_public_optimization_examples_keep_editable_constant_style() -> None:
    examples = REPO_ROOT / "examples" / "10_vmex_optimization"
    optimizer_scripts = {"run.py"}
    for script in sorted(examples.glob("*.py")):
        text = script.read_text(encoding="utf-8")
        assert "argparse" not in text
        assert "def main(" not in text
        assert "def _main(" not in text
        assert 'if __name__ == "__main__"' not in text
        if script.name in optimizer_scripts:
            assert "s_index=7" in text
            assert "alpha=0.0" in text
            assert "A_OVER_LT, A_OVER_LN = 3.0, 1.0" in text
            assert "WINDOW_STEPS" in text
        else:
            raise AssertionError(f"unexpected optimization example {script.name}")


def test_stellarator_itg_observable_contract_is_finite_and_exported() -> None:
    assert gkx.STELLARATOR_ITG_PARAMETER_NAMES == PARAMETER_NAMES
    assert gkx.STELLARATOR_ITG_OBSERVABLE_NAMES == OBSERVABLE_NAMES
    assert gkx.optimize_stellarator_itg is optimize_stellarator_itg
    assert (
        gkx.stellarator_itg_density_gradient_scan
        is stellarator_itg_density_gradient_scan
    )
    assert (
        gkx.stellarator_itg_residual_sensitivity_report
        is stellarator_itg_residual_sensitivity_report
    )

    cfg = _fast_config()
    params = default_stellarator_initial_params()
    observables = np.asarray(qa_observable_vector(params, cfg))

    assert observables.shape == (len(OBSERVABLE_NAMES),)
    assert np.all(np.isfinite(observables))
    obs = dict(zip(OBSERVABLE_NAMES, observables, strict=True))
    assert obs["aspect"] > 0.0
    assert obs["kperp_eff2"] > 0.0
    assert obs["growth_rate"] > 0.0
    assert obs["linear_heat_flux_weight"] > 0.0
    assert obs["quasilinear_heat_flux"] > 0.0
    assert obs["nonlinear_heat_flux_mean"] > 0.0


def test_stellarator_itg_density_gradient_scan_is_monotone_and_scoped() -> None:
    cfg = _fast_config()
    params = default_stellarator_initial_params()
    scan = stellarator_itg_density_gradient_scan(
        params,
        cfg,
        density_gradients=(0.8, cfg.reference_density_gradient, 4.8),
    )
    default_obs = dict(
        zip(OBSERVABLE_NAMES, qa_observable_vector(params, cfg), strict=True)
    )

    assert (
        scan["scope"]
        == "reduced_max_mode1_density_gradient_response_not_full_nonlinear_scan"
    )
    assert scan["fixed_temperature_gradient"] == cfg.reference_temperature_gradient
    assert scan["density_gradient_axis"] == [0.8, cfg.reference_density_gradient, 4.8]
    assert np.all(np.diff(scan["heat_flux_mean"]) > 0.0)
    assert np.all(np.diff(scan["growth_rate"]) > 0.0)
    assert scan["linear_slope_dQ_d_a_over_Ln"] > 0.0
    assert scan["heat_flux_mean"][1] == pytest.approx(
        default_obs["nonlinear_heat_flux_mean"]
    )


@pytest.mark.parametrize("density_gradients", [(), (0.8, np.nan), (-0.1, 1.0)])
def test_stellarator_itg_density_gradient_scan_rejects_invalid_axes(
    density_gradients: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="density_gradients"):
        stellarator_itg_density_gradient_scan(
            default_stellarator_initial_params(),
            _fast_config(),
            density_gradients=density_gradients,
        )


def test_stellarator_itg_objectives_have_fd_checked_gradients() -> None:
    cfg = _fast_config()
    params = jnp.asarray([0.18, 0.25, 0.22, -0.16])
    rtol, atol = _fd_tolerances()

    for kind in ("growth", "quasilinear_flux", "nonlinear_heat_flux"):
        value = stellarator_itg_objective(params, kind, cfg)
        residual = stellarator_itg_objective_residual_vector(params, kind, cfg)
        assert float(value) > 0.0
        assert residual.shape == (3 + len(PARAMETER_NAMES) + 1,)
        assert len(stellarator_itg_objective_residual_names(kind)) == residual.size
        np.testing.assert_allclose(
            float(value), float(jnp.dot(residual, residual)), rtol=1.0e-6
        )
        report = autodiff_finite_difference_report(
            lambda x, kind=kind: stellarator_itg_objective(x, kind, cfg),
            params,
            step=cfg.fd_step,
            rtol=rtol,
            atol=atol,
            workers=2,
        )
        assert report["passed"] is True
        assert report["finite_difference_parallel"]["requested_workers"] == 2


def test_quasilinear_residual_sensitivity_report_checks_fd_and_conditioning() -> None:
    cfg = _fast_config()
    params = jnp.asarray([0.16, 0.21, 0.24, -0.18])

    report = stellarator_itg_residual_sensitivity_report(
        params,
        "quasilinear_flux",
        cfg,
        finite_difference_workers=2,
    )

    assert report["passed"] is True
    assert report["objective_kind"] == "quasilinear_flux"
    assert report["finite_difference_gate"]["passed"] is True
    assert report["finite_difference_gate"]["step"] <= cfg.fd_step
    assert (
        report["finite_difference_gate"]["finite_difference_parallel"][
            "requested_workers"
        ]
        == 2
    )
    assert report["conditioning_gate"]["passed"] is True
    assert report["conditioning_gate"]["sensitivity_map_rank"] == len(PARAMETER_NAMES)
    assert report["covariance"]["source"] == "weighted_objective_residual"
    assert report["covariance"]["conditioning_gate"]["passed"] is True
    assert report["residual_names"] == list(
        stellarator_itg_objective_residual_names("quasilinear_flux")
    )


def test_stellarator_itg_sample_portfolio_is_rectangular_and_exported() -> None:
    assert gkx.StellaratorITGSampleSet is StellaratorITGSampleSet
    assert (
        gkx.stellarator_itg_sample_objective_table
        is stellarator_itg_sample_objective_table
    )
    assert (
        gkx.stellarator_itg_reduced_portfolio_objective
        is stellarator_itg_reduced_portfolio_objective
    )
    assert (
        gkx.stellarator_itg_portfolio_sensitivity_report
        is stellarator_itg_portfolio_sensitivity_report
    )
    assert (
        gkx.stellarator_itg_portfolio_gate_payload
        is stellarator_itg_portfolio_gate_payload
    )

    cfg = _fast_config()
    samples = StellaratorITGSampleSet(
        surfaces=(0.55, 0.64),
        alphas=(0.0, 0.7),
        ky_values=(0.1, 0.3, 0.5),
        surface_weights=(1.0, 2.0),
        ky_weights=(1.0, 2.0, 1.0),
    )
    params = jnp.asarray([0.18, 0.25, 0.22, -0.16])
    table = stellarator_itg_sample_objective_table(
        params,
        ("growth", "quasilinear_flux", "nonlinear_heat_flux"),
        cfg,
        samples,
    )
    reduced = stellarator_itg_reduced_portfolio_objective(
        params,
        ("growth", "quasilinear_flux"),
        cfg,
        samples,
        objective_weights=(2.0, 1.0),
    )

    assert samples.n_samples == 12
    assert table.shape == (2, 2, 3, 3)
    assert float(jnp.min(table)) > 0.0
    assert float(reduced) > 0.0
    assert samples.to_dict()["reduction"] == "weighted_mean"


def test_stellarator_itg_portfolio_sensitivity_report_checks_rows_and_scalar() -> None:
    cfg = _fast_config()
    samples = StellaratorITGSampleSet(
        surfaces=(0.55, 0.64),
        alphas=(0.0, 0.7),
        ky_values=(0.15, 0.35),
    )
    params = jnp.asarray([0.16, 0.21, 0.24, -0.18])

    report = stellarator_itg_portfolio_sensitivity_report(
        params,
        ("growth", "quasilinear_flux"),
        cfg,
        samples,
        workers=2,
    )
    inner = report["portfolio_report"]

    assert report["passed"] is True
    assert report["objective_names"] == ["growth", "quasilinear_flux"]
    assert report["sample_set"]["n_samples"] == 8
    assert inner["portfolio_contract"]["row_shape"] == [2, 2, 2, 2]
    assert inner["scalar_gradient_gate"]["passed"] is True
    assert inner["row_jacobian_gate"]["passed"] is True
    assert inner["conditioning_gate"]["sensitivity_map_rank"] == len(PARAMETER_NAMES)
    assert (
        inner["scalar_gradient_gate"]["finite_difference_parallel"]["requested_workers"]
        == 2
    )


def test_stellarator_itg_portfolio_gate_payload_is_json_ready() -> None:
    cfg = _fast_config()
    samples = StellaratorITGSampleSet(
        surfaces=(0.55, 0.64),
        alphas=(0.0, 0.7),
        ky_values=(0.15, 0.35),
    )
    params = jnp.asarray([0.16, 0.21, 0.24, -0.18])

    payload = stellarator_itg_portfolio_gate_payload(
        params,
        ("growth", "quasilinear_flux"),
        cfg,
        samples,
        objective_weights=(2.0, 1.0),
        finite_difference_workers=2,
    )

    assert payload["kind"] == "stellarator_itg_portfolio_gate"
    assert payload["passed"] is True
    assert payload["production_nonlinear_optimization_claim"] is False
    assert payload["sample_set"]["n_samples"] == 8
    assert len(payload["samples"]) == 8
    assert len(payload["base_sample_values"]) == 8
    json.dumps(payload, allow_nan=False)
    objective_table = np.asarray(payload["base_objective_table"], dtype=float)
    objective_tensor = np.asarray(payload["base_objective_tensor"], dtype=float)
    objective_weights = np.asarray(payload["objective_weights"], dtype=float)
    sample_values = np.asarray(payload["base_sample_values"], dtype=float)
    sample_rows = payload["samples"]

    assert objective_table.shape == (8, 2)
    assert objective_tensor.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(objective_table, objective_tensor.reshape((8, 2)))
    np.testing.assert_allclose(sample_values, objective_table @ objective_weights)
    assert [(row["surface"], row["alpha"], row["ky"]) for row in sample_rows] == [
        (surface, alpha, ky)
        for surface in samples.surfaces
        for alpha in samples.alphas
        for ky in samples.ky_values
    ]
    np.testing.assert_allclose(sum(payload["objective_weights"]), 1.0)
    np.testing.assert_allclose(sum(row["weight"] for row in payload["samples"]), 1.0)
    np.testing.assert_allclose(
        payload["base_value"],
        float(np.dot(sample_values, [row["weight"] for row in sample_rows])),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    assert payload["portfolio_report"]["scalar_gradient_gate"]["passed"] is True
    assert payload["portfolio_report"]["row_jacobian_gate"]["passed"] is True
    assert "real vmex" in payload["next_action"]


def test_stellarator_itg_vmec_boozer_portfolio_wraps_real_table_contract(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_table(_state, _static, _indata, _wout, **kwargs):  # noqa: ANN001, ANN202
        calls.update(kwargs)
        rows = []
        for index in range(8):
            value = float(index + 1)
            rows.append([value, 0.0, 1.0, 2.0, 0.0, 10.0 * value])
        metadata = [{"sample": index} for index in range(8)]
        return jnp.asarray(rows), metadata

    monkeypatch.setattr(
        so, "vmec_boozer_solver_objective_table_with_metadata_from_state", fake_table
    )
    samples = StellaratorITGSampleSet(
        surfaces=(0.50, 0.70),
        alphas=(0.0, 0.6),
        ky_values=(0.1, 0.3),
    )

    table = stellarator_itg_vmec_boozer_sample_objective_table_from_state(
        "state",
        "static",
        "indata",
        "wout",
        ("growth", "quasilinear_flux"),
        samples,
        ntheta=8,
    )
    reduced = stellarator_itg_vmec_boozer_portfolio_objective_from_state(
        "state",
        "static",
        "indata",
        "wout",
        ("growth", "quasilinear_flux"),
        samples,
        objective_weights=(1.0, 0.0),
        ntheta=8,
    )

    assert gkx.stellarator_itg_vmec_boozer_sample_objective_table_from_state is (
        stellarator_itg_vmec_boozer_sample_objective_table_from_state
    )
    assert gkx.stellarator_itg_vmec_boozer_portfolio_objective_from_state is (
        stellarator_itg_vmec_boozer_portfolio_objective_from_state
    )
    assert table.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(np.asarray(table)[0, 0, 0], (1.0, 10.0))
    assert float(reduced) == pytest.approx(4.5)
    assert calls["torflux_values"] == samples.surfaces
    assert calls["alphas"] == samples.alphas
    assert calls["ky_values"] == samples.ky_values
    assert calls["ntheta"] == 8


def test_nonlinear_heat_flux_window_metrics_use_late_stable_samples() -> None:
    cfg = _fast_config()
    times, heat_flux = nonlinear_heat_flux_trace(
        default_stellarator_initial_params(), cfg
    )
    metrics = nonlinear_heat_flux_window_metrics(
        times,
        heat_flux,
        tail_fraction=cfg.nonlinear_tail_fraction,
    )

    assert times.shape == heat_flux.shape == (cfg.nonlinear_steps + 1,)
    assert int(metrics["start_index"]) < cfg.nonlinear_steps - 1
    assert float(metrics["mean"]) > 0.0
    assert float(metrics["cv"]) < 0.15
    assert float(metrics["trend"]) < 0.35


def test_optimize_stellarator_itg_reduces_nonlinear_window_objective(
    monkeypatch,
) -> None:
    _disable_optional_backend_discovery(monkeypatch)
    cfg = _fast_config()

    result = optimize_stellarator_itg("nonlinear_heat_flux", config=cfg)

    assert result.objective_kind == "nonlinear_heat_flux"
    assert result.final_objective < 0.20 * result.initial_objective
    assert result.gradient_gate["passed"] is True
    assert result.covariance["source"] == "weighted_objective_residual"
    assert result.covariance["residual_sensitivity_passed"] is True
    assert result.covariance["residual_jacobian_gate"]["passed"] is True
    assert result.covariance["conditioning_gate"]["passed"] is True
    assert len(result.covariance["residual_names"]) == 3 + len(PARAMETER_NAMES) + 1
    assert result.covariance["sensitivity_map_rank"] == len(PARAMETER_NAMES)
    assert result.nonlinear_trace is not None
    assert result.nonlinear_trace["final_window"]["cv"] < 0.05
    assert result.nonlinear_trace["final_window"]["trend"] < 0.15
    serialized = result.to_dict()
    assert (
        serialized["claim_level"]
        == "reduced_nonlinear_window_estimator_optimization_not_transport_average"
    )
    assert serialized["nonlinear_transport_scope"]["transport_average_gate"] is False
    assert (
        serialized["nonlinear_transport_scope"][
            "production_nonlinear_optimization_claim"
        ]
        is False
    )

    initial = dict(zip(OBSERVABLE_NAMES, result.initial_observables, strict=True))
    final = dict(zip(OBSERVABLE_NAMES, result.final_observables, strict=True))
    assert final["growth_rate"] < initial["growth_rate"]
    assert final["quasilinear_heat_flux"] < initial["quasilinear_heat_flux"]
    assert final["nonlinear_heat_flux_mean"] < initial["nonlinear_heat_flux_mean"]


def test_compare_stellarator_itg_objectives_payload_is_json_ready(monkeypatch) -> None:
    _disable_optional_backend_discovery(monkeypatch)
    cfg = _fast_config()

    payload = compare_stellarator_itg_objectives(
        ("growth",), config=cfg, workers=2, finite_difference_workers=2
    )

    assert (
        payload["claim_level"]
        == "reduced_objective_optimization_comparison_not_full_production_vmec_gk"
    )
    assert payload["production_nonlinear_optimization_claim"] is False
    assert payload["parameter_names"] == list(PARAMETER_NAMES)
    assert payload["observable_names"] == list(OBSERVABLE_NAMES)
    assert payload["parallel"]["requested_workers"] == 2
    assert payload["parallel"]["finite_difference_workers"] == 2
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["objective_kind"] == "growth"
    assert result["final_objective"] < result["initial_objective"]
    assert result["gradient_gate"]["passed"] is True
    assert (
        result["gradient_gate"]["finite_difference_parallel"]["requested_workers"] == 2
    )


def test_compare_stellarator_itg_objectives_parallel_preserves_order(
    monkeypatch,
) -> None:
    _disable_optional_backend_discovery(monkeypatch)

    def fake_optimize(kind, initial_params=None, config=None, **kwargs):  # noqa: ANN001, ANN202
        idx = {"growth": 1.0, "quasilinear_flux": 2.0, "nonlinear_heat_flux": 3.0}[kind]
        return StellaratorITGOptimizationResult(
            objective_kind=kind,
            parameter_names=PARAMETER_NAMES,
            observable_names=OBSERVABLE_NAMES,
            initial_params=(0.0, 0.0, 0.0, 0.0),
            final_params=(idx, idx, idx, idx),
            initial_objective=idx + 1.0,
            final_objective=idx,
            initial_observables=tuple(0.0 for _ in OBSERVABLE_NAMES),
            final_observables=tuple(idx for _ in OBSERVABLE_NAMES),
            history=(),
            gradient_gate={
                "passed": True,
                "finite_difference_parallel": {
                    "requested_workers": kwargs["finite_difference_workers"],
                    "executor": kwargs["finite_difference_executor"],
                },
            },
            covariance={"source": "test"},
            nonlinear_trace=None,
            config={},
            backend_info={},
        )

    monkeypatch.setattr(so, "optimize_stellarator_itg", fake_optimize)
    payload = compare_stellarator_itg_objectives(
        ("growth", "quasilinear_flux", "nonlinear_heat_flux"),
        workers=3,
        finite_difference_workers=2,
    )

    assert [row["objective_kind"] for row in payload["results"]] == [
        "growth",
        "quasilinear_flux",
        "nonlinear_heat_flux",
    ]
    assert [row["final_objective"] for row in payload["results"]] == [1.0, 2.0, 3.0]
    assert payload["parallel"]["effective_workers"] == 3


# ---- test_zonal_objective.py ----


def test_reduced_stellarator_model_stays_deprecated_and_shallowly_depended_on() -> None:
    """The reduced model must stay marked, and must not grow production consumers.

    ``objectives.stellarator_reduced`` is a fitted feature map whose "nonlinear"
    trace is an ODE envelope, not gyrokinetics. It is slated for removal now that
    ``solver_objective_vector_from_geometry`` evaluates the production linear RHS
    differentiably. Two things can silently undo that: the deprecation notice
    being dropped in a refactor, and the real VMEC path acquiring a deeper
    dependency on the reduced physics than the two generic helpers it uses today.

    This asserts both. It fails if someone removes the marker, and it fails if
    ``vmec_transport`` starts importing reduced *physics* rather than
    ``smooth_positive`` and the sampling contract.
    """

    import ast
    import gkx.objectives.stellarator as reduced  # fused: stellarator_reduced now lives here

    assert "deprecated" in (reduced.__doc__ or "").lower(), (
        "the reduced-model deprecation notice was removed; it is a fitted "
        "feature map, not gyrokinetics, and must stay marked until deleted"
    )

    source = (
        REPO_ROOT / "src" / "gkx" / "objectives" / "vmec_transport.py"
    ).read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and "stellarator" in node.module
        ):
            imported.update(alias.name for alias in node.names)

    # Generic helpers only: a softplus and a portfolio-shape contract. Anything
    # else means the production VMEC path has taken a dependency on reduced
    # physics, which is what removal has to avoid.
    assert imported <= {"StellaratorITGSampleSet", "smooth_positive"}, (
        f"vmec_transport imports reduced-model physics: {sorted(imported)}"
    )
