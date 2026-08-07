"""Unit contracts: autodiff solver objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from support.paired_solvax import requires_paired_solvax
from support.paths import REPO_ROOT, load_artifact_tool, load_repo_script

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
from gkx.core.grid import build_spectral_grid, select_ky_grid
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
        R_over_Ln=2.2,
        R_over_LTi=6.9,
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
            R_over_Ln=x[0],
            R_over_LTi=x[1],
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
        R_over_Ln=2.2,
        R_over_LTi=6.9,
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
            R_over_Ln=x[0],
            R_over_LTi=x[1],
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

import gkx.objectives.gradient_gates as gradient_gates
import gkx.objectives.sampling as sampling
import gkx.objectives.solver_vmec as solver_vmec
from gkx import (
    AdaptiveLinearEigensolverConfig,
    SOLVER_GEOMETRY_PARAMETER_NAMES,
    SOLVER_OBJECTIVE_NAMES,
    SolverScalarObjective,
    VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES,
    VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES,
    VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES,
    VMEC_BOOZER_STATE_PARAMETER_FAMILIES,
    VMEC_BOOZER_STATE_PARAMETER_NAMES,
    default_solver_geometry_design_params,
    dominant_eigenvalue_branch_locality_report,
    dominant_real_eigenvalue,
    linear_solver_geometry_gradient_report,
    mode21_vmec_boozer_linear_frequency_gradient_report,
    mode21_vmec_boozer_nonlinear_window_gradient_report,
    mode21_vmec_boozer_quasilinear_gradient_report,
    solver_linear_operator_matrix_from_geometry,
    solver_objective_branch_gradient_report,
    solver_growth_rate_from_geometry,
    solver_objective_vector_from_geometry,
    solver_grid_options_from_ky_values,
    solver_scalar_objective_from_vector,
    solver_ready_geometry_mapping,
    vmec_boozer_aggregate_line_search_holdout_report,
    vmec_boozer_aggregate_scalar_objective_finite_difference_report,
    vmec_boozer_aggregate_scalar_objective_from_state,
    vmec_boozer_aggregate_scalar_objective_line_search_report,
    vmec_boozer_scalar_objective_finite_difference_report,
    vmec_boozer_scalar_objective_from_state,
    vmec_boozer_scalar_objective_line_search_report,
    vmec_boozer_solver_objective_table_from_state,
    vmec_boozer_solver_objective_table_with_metadata_from_state,
    vmec_boozer_solver_objective_vector_from_state,
)
from gkx.geometry.vmec_state_controls import _vmec_boozer_state_parameter_name
from gkx.objectives.geometry import (
    TINY_OBJECTIVE_NAMES,
    _objective_gate_rows,
    tiny_differentiable_objective_gradient_report,
)
from gkx.objectives.vmec_boozer_gradients import (
    _reduced_nonlinear_window_metrics_from_linear_observables,
)


ROOT = REPO_ROOT
mod = load_artifact_tool("build_solver_objective_gradient_gate")


def test_solver_ready_geometry_mapping_validates_contract() -> None:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    mapping = solver_ready_geometry_mapping(
        default_solver_geometry_design_params(), theta
    )

    assert gkx.solver_ready_geometry_mapping is solver_ready_geometry_mapping
    assert tuple(SOLVER_GEOMETRY_PARAMETER_NAMES) == (
        "bmag_ripple",
        "curvature_drift_scale",
    )
    assert mapping["theta"].shape == theta.shape
    assert np.all(np.asarray(mapping["bmag"]) > 0.0)
    with pytest.raises(ValueError, match="length-2"):
        solver_ready_geometry_mapping(jnp.ones(3), theta)


def test_solver_ready_linear_context_builds_operator_contract() -> None:
    context = gradient_gates._solver_ready_linear_context(
        n_laguerre=2,
        n_hermite=1,
        source_model="unit_test_solver_ready_context",
    )

    assert context.state_shape[:2] == (2, 1)
    assert context.grid.ky.size == 1
    assert context.grid.kx.size == 1
    assert context.source_model == "unit_test_solver_ready_context"

    matrix = context.matrix_fn(default_solver_geometry_design_params())
    assert matrix.shape == (int(np.prod(context.state_shape)),) * 2
    feature_context = context.quasilinear_feature_context()
    assert feature_context["state_shape"] == context.state_shape
    assert callable(feature_context["geometry_for"])
    assert callable(feature_context["rhs_phi"])


def test_tiny_differentiable_objective_gradient_report_is_finite_and_conditioned() -> (
    None
):
    x64_enabled = bool(jax.config.read("jax_enable_x64"))
    fd_step = 2.0e-5 if x64_enabled else 1.0e-3
    rtol = 5.0e-4 if x64_enabled else 1.0e-2
    atol = 1.0e-7 if x64_enabled else 1.0e-4

    report = tiny_differentiable_objective_gradient_report(
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
    )

    assert report["passed"] is True
    assert report["source_scope"] == "solver_ready_geometry_contract"
    assert report["observable_names"] == list(TINY_OBJECTIVE_NAMES)
    assert report["parameter_names"] == list(SOLVER_GEOMETRY_PARAMETER_NAMES)
    assert report["finite_flags"]["autodiff_jacobian"] is True
    assert report["finite_flags"]["finite_difference_jacobian"] is True
    assert report["conditioning"]["jacobian_shape"] == [
        len(TINY_OBJECTIVE_NAMES),
        len(SOLVER_GEOMETRY_PARAMETER_NAMES),
    ]
    assert report["conditioning"]["sensitivity_map_rank"] == 2
    assert np.isfinite(float(report["tangent_ad_norm"]))
    assert len(report["gradient_checks"]) == (
        len(TINY_OBJECTIVE_NAMES) * len(SOLVER_GEOMETRY_PARAMETER_NAMES)
    )
    json.dumps(report, allow_nan=False)

    with pytest.raises(ValueError, match="length-2"):
        tiny_differentiable_objective_gradient_report(jnp.ones(3))


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


def test_linear_solver_geometry_gradient_report_passes_actual_rhs_gate() -> None:
    report = linear_solver_geometry_gradient_report(
        fd_step=1.0e-3, rtol=1.0e-1, atol=2.0e-3
    )

    assert (
        gkx.linear_solver_geometry_gradient_report
        is linear_solver_geometry_gradient_report
    )
    assert report["passed"] is True
    assert report["source_scope"] == "solver_ready_geometry_contract"
    assert report["linear_growth_gradient_gate"] is True
    assert report["quasilinear_weight_gradient_gate"] is True
    assert report["nonlinear_window_gradient_gate"] is False
    assert report["objective_names"] == list(SOLVER_OBJECTIVE_NAMES)
    assert np.asarray(report["eigenpair_gate"]["jacobian_implicit"]).shape == (
        len(SOLVER_OBJECTIVE_NAMES),
        len(SOLVER_GEOMETRY_PARAMETER_NAMES),
    )

    with pytest.raises(ValueError, match="length-2"):
        linear_solver_geometry_gradient_report(jnp.ones(3))


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


def test_solver_objective_branch_gradient_report_gates_public_evaluator() -> None:
    report = solver_objective_branch_gradient_report(
        fd_step=1.0e-3,
        rtol=1.0e-1,
        atol=2.0e-3,
        n_laguerre=2,
        n_hermite=1,
    )

    assert (
        gkx.solver_objective_branch_gradient_report
        is solver_objective_branch_gradient_report
    )
    assert report["passed"] is True
    assert report["source_scope"] == "solver_ready_geometry_contract"
    assert report["value_evaluator_finite"] is True
    assert report["branch_continuity_gate"] is True
    assert report["ad_fd_gate"] is True
    assert len(report["branch_rows"]) == 2 * len(SOLVER_GEOMETRY_PARAMETER_NAMES)
    assert np.asarray(report["value_evaluator_objectives"]).shape == (
        len(SOLVER_OBJECTIVE_NAMES),
    )
    assert np.asarray(report["eigenpair_gate"]["jacobian_implicit"]).shape == (
        len(SOLVER_OBJECTIVE_NAMES),
        len(SOLVER_GEOMETRY_PARAMETER_NAMES),
    )
    with pytest.raises(ValueError, match="length-2"):
        solver_objective_branch_gradient_report(jnp.ones(3))


def test_vmec_boozer_solver_objective_vector_from_state_splits_options(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_geometry(state, static, indata, wout, **kwargs):  # noqa: ANN001, ANN202
        calls["geometry"] = (state, static, indata, wout, kwargs)
        return "geom"

    def fake_objective(geom, **kwargs):  # noqa: ANN001, ANN202
        calls["objective"] = (geom, kwargs)
        return jnp.arange(len(SOLVER_OBJECTIVE_NAMES), dtype=jnp.float32)

    monkeypatch.setattr(
        solver_vmec, "flux_tube_geometry_from_vmec_boozer_state", fake_geometry
    )
    monkeypatch.setattr(
        solver_vmec, "solver_objective_vector_from_geometry", fake_objective
    )

    vector = vmec_boozer_solver_objective_vector_from_state(
        "state",
        "static",
        "indata",
        "wout",
        surface_index=2,
        ntheta=8,
        mboz=21,
        nboz=21,
        selected_ky_index=1,
        n_laguerre=2,
    )

    assert gkx.vmec_boozer_solver_objective_vector_from_state is (
        vmec_boozer_solver_objective_vector_from_state
    )
    assert np.asarray(vector).tolist() == list(range(len(SOLVER_OBJECTIVE_NAMES)))
    assert calls["geometry"][4] == {
        "surface_index": 2,
        "ntheta": 8,
        "mboz": 21,
        "nboz": 21,
    }
    assert calls["objective"] == ("geom", {"selected_ky_index": 1, "n_laguerre": 2})
    with pytest.raises(TypeError, match="unknown VMEC/Boozer objective options"):
        vmec_boozer_solver_objective_vector_from_state(
            "state",
            "static",
            "indata",
            "wout",
            unexpected=True,
        )


def test_vmec_boozer_scalar_objective_from_state_uses_vector_selector(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        solver_vmec,
        "vmec_boozer_solver_objective_vector_from_state",
        lambda *_args, **_kwargs: jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )

    value = vmec_boozer_scalar_objective_from_state(
        "state",
        "static",
        "indata",
        "wout",
        objective="quasilinear_flux",
        ntheta=4,
    )

    assert (
        gkx.vmec_boozer_scalar_objective_from_state
        is vmec_boozer_scalar_objective_from_state
    )
    assert float(value) == pytest.approx(6.0)


def test_vmec_boozer_solver_objective_table_samples_surfaces_alphas_and_ky(
    monkeypatch,
) -> None:
    geometry_calls: list[dict[str, object]] = []
    objective_calls: list[tuple[str, dict[str, object]]] = []

    def fake_geometry(_state, _static, _indata, _wout, **kwargs):  # noqa: ANN001, ANN202
        geometry_calls.append(dict(kwargs))
        return f"geom-{len(geometry_calls)}"

    def fake_objective(geom, **kwargs):  # noqa: ANN001, ANN202
        objective_calls.append((geom, dict(kwargs)))
        ky = float(kwargs["selected_ky_index"])
        geom_index = float(str(geom).split("-")[-1])
        return jnp.asarray([geom_index + ky, 0.0, 1.0, 2.0, 0.0, 3.0])

    monkeypatch.setattr(
        solver_vmec, "flux_tube_geometry_from_vmec_boozer_state", fake_geometry
    )
    monkeypatch.setattr(
        solver_vmec, "solver_objective_vector_from_geometry", fake_objective
    )

    table = vmec_boozer_solver_objective_table_from_state(
        "state",
        "static",
        "indata",
        "wout",
        surface_indices=[1, 3],
        alphas=[0.0, 0.5],
        selected_ky_indices=[1, 2],
        ntheta=8,
        n_laguerre=2,
    )

    assert gkx.vmec_boozer_solver_objective_table_from_state is (
        vmec_boozer_solver_objective_table_from_state
    )
    assert np.asarray(table).shape == (8, len(SOLVER_OBJECTIVE_NAMES))
    assert len(geometry_calls) == 4
    assert len(objective_calls) == 8
    assert geometry_calls[0] == {"surface_index": 1, "alpha": 0.0, "ntheta": 8}
    assert objective_calls[0] == ("geom-1", {"n_laguerre": 2, "selected_ky_index": 1})
    assert objective_calls[1] == ("geom-1", {"n_laguerre": 2, "selected_ky_index": 2})
    with pytest.raises(TypeError, match="selected_ky_indices"):
        vmec_boozer_solver_objective_table_from_state(
            "state",
            "static",
            "indata",
            "wout",
            selected_ky_index=1,
            selected_ky_indices=[1, 2],
        )


def test_vmec_boozer_solver_objective_table_with_metadata_accepts_torflux_and_physical_ky(
    monkeypatch,
) -> None:
    geometry_calls: list[dict[str, object]] = []
    objective_calls: list[dict[str, object]] = []

    def fake_geometry(_state, _static, _indata, _wout, **kwargs):  # noqa: ANN001, ANN202
        geometry_calls.append(dict(kwargs))
        return f"geom-{len(geometry_calls)}"

    def fake_objective(_geom, **kwargs):  # noqa: ANN001, ANN202
        objective_calls.append(dict(kwargs))
        ky = float(kwargs["selected_ky_index"])
        return jnp.asarray([ky, 0.0, 1.0, 2.0, 0.0, 3.0])

    monkeypatch.setattr(
        solver_vmec, "flux_tube_geometry_from_vmec_boozer_state", fake_geometry
    )
    monkeypatch.setattr(
        solver_vmec, "solver_objective_vector_from_geometry", fake_objective
    )

    table, metadata = vmec_boozer_solver_objective_table_with_metadata_from_state(
        "state",
        "static",
        "indata",
        "wout",
        torflux_values=(0.45, 0.65),
        alphas=(0.0, 0.5),
        ky_values=(0.1, 0.3),
        ntheta=8,
        ny=4,
        n_laguerre=2,
    )

    assert gkx.vmec_boozer_solver_objective_table_with_metadata_from_state is (
        vmec_boozer_solver_objective_table_with_metadata_from_state
    )
    assert np.asarray(table).shape == (8, len(SOLVER_OBJECTIVE_NAMES))
    assert geometry_calls[0] == {"torflux": 0.45, "alpha": 0.0, "ntheta": 8}
    assert objective_calls[0]["selected_ky_index"] == 1
    assert objective_calls[0]["ny"] == 8
    assert objective_calls[0]["ly"] == pytest.approx(2.0 * np.pi / 0.1)
    assert metadata[0]["surface"] == pytest.approx(0.45)
    assert metadata[0]["torflux"] == pytest.approx(0.45)
    assert metadata[0]["ky"] == pytest.approx(0.1)
    assert metadata[1]["selected_ky_index"] == 3
    assert metadata[1]["selected_ky"] == pytest.approx(0.3, rel=5.0e-6, abs=5.0e-8)
    with pytest.raises(TypeError, match="torflux_values or surface_indices"):
        vmec_boozer_solver_objective_table_with_metadata_from_state(
            "state",
            "static",
            "indata",
            "wout",
            surface_indices=[1],
            torflux_values=[0.5],
        )
    with pytest.raises(TypeError, match="ky_values or selected_ky_indices"):
        vmec_boozer_solver_objective_table_with_metadata_from_state(
            "state",
            "static",
            "indata",
            "wout",
            selected_ky_indices=[1],
            ky_values=[0.1],
        )


def test_vmec_boozer_aggregate_scalar_objective_from_state_reductions(
    monkeypatch,
) -> None:
    table = jnp.asarray(
        [
            [1.0, 0.0, 1.0, 2.0, 0.0, 10.0],
            [2.0, 0.0, 1.0, 2.0, 0.0, 20.0],
            [4.0, 0.0, 1.0, 2.0, 0.0, 40.0],
        ]
    )
    monkeypatch.setattr(
        solver_vmec,
        "vmec_boozer_solver_objective_table_from_state",
        lambda *_args, **_kwargs: table,
    )

    mean_value = vmec_boozer_aggregate_scalar_objective_from_state(
        "state",
        "static",
        "indata",
        "wout",
        objective="growth",
    )
    weighted_value = vmec_boozer_aggregate_scalar_objective_from_state(
        "state",
        "static",
        "indata",
        "wout",
        objective="quasilinear_flux",
        reduction="weighted_mean",
        weights=[1.0, 1.0, 2.0],
    )
    max_value = vmec_boozer_aggregate_scalar_objective_from_state(
        "state",
        "static",
        "indata",
        "wout",
        objective="growth",
        reduction="max",
    )

    assert gkx.vmec_boozer_aggregate_scalar_objective_from_state is (
        vmec_boozer_aggregate_scalar_objective_from_state
    )
    assert float(mean_value) == pytest.approx(7.0 / 3.0)
    assert float(weighted_value) == pytest.approx(27.5)
    assert float(max_value) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="weights"):
        vmec_boozer_aggregate_scalar_objective_from_state(
            "state",
            "static",
            "indata",
            "wout",
            reduction="weighted_mean",
            weights=[1.0],
        )
    with pytest.raises(ValueError, match="reduction"):
        vmec_boozer_aggregate_scalar_objective_from_state(
            "state",
            "static",
            "indata",
            "wout",
            reduction="median",  # type: ignore[arg-type]
        )


def test_vmec_boozer_scalar_objective_finite_difference_report(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class FakeState:
        R_cos: jnp.ndarray

    fake_state = FakeState(R_cos=jnp.zeros((5, 3), dtype=jnp.float32))
    monkeypatch.setattr(
        solver_vmec,
        "_load_vmex_example_state_bundle",
        lambda case_name: {
            "case_name": case_name,
            "input_path": "input.test",
            "wout_path": "wout.test",
            "state": fake_state,
            "runtime": "runtime",
            "inp": "inp",
            "wout": "wout",
        },
    )

    def fake_vector(state, *_args, **_kwargs):  # noqa: ANN001, ANN202
        coeff = float(np.asarray(state.R_cos[2, 1]))
        return jnp.asarray([1.0 + 3.0 * coeff, 0.0, 2.0, 4.0, 0.5, 5.0 + coeff])

    monkeypatch.setattr(
        solver_vmec, "vmec_boozer_solver_objective_vector_from_state", fake_vector
    )

    report = vmec_boozer_scalar_objective_finite_difference_report(
        case_name="case",
        objective="growth",
        base_delta=0.1,
        perturbation_step=1.0e-3,
        response_atol=1.0e-6,
        ntheta=4,
    )

    assert gkx.vmec_boozer_scalar_objective_finite_difference_report is (
        vmec_boozer_scalar_objective_finite_difference_report
    )
    assert report["passed"] is True
    assert report["source_scope"] == "mode21_vmec_boozer_state"
    assert report["parameter_name"] == "Rcos_mid_surface_m1"
    assert report["base_delta"] == pytest.approx(0.1)
    assert report["central_derivative"] == pytest.approx(3.0, rel=1.0e-4)
    assert report["response_resolved"] is True
    assert report["finite_difference_consistent"] is True
    assert report["curvature_ratio"] < 1.0e-3
    assert report["options"] == {"ntheta": 4}

    with pytest.raises(ValueError, match="perturbation_step"):
        vmec_boozer_scalar_objective_finite_difference_report(perturbation_step=0.0)
    with pytest.raises(ValueError, match="max_curvature_ratio"):
        vmec_boozer_scalar_objective_finite_difference_report(max_curvature_ratio=-1.0)
    with pytest.raises(ValueError, match="radial_index"):
        vmec_boozer_scalar_objective_finite_difference_report(radial_index=99)


def test_vmec_boozer_scalar_objective_finite_difference_report_selects_state_family(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class FakeState:
        R_cos: jnp.ndarray
        Z_sin: jnp.ndarray

    fake_state = FakeState(
        R_cos=jnp.zeros((5, 3), dtype=jnp.float32),
        Z_sin=jnp.zeros((5, 3), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        solver_vmec,
        "_load_vmex_example_state_bundle",
        lambda case_name: {
            "case_name": case_name,
            "input_path": "input.test",
            "wout_path": "wout.test",
            "state": fake_state,
            "runtime": "runtime",
            "inp": "inp",
            "wout": "wout",
        },
    )

    def fake_vector(state, *_args, **_kwargs):  # noqa: ANN001, ANN202
        coeff = float(np.asarray(state.Z_sin[2, 1]))
        return jnp.asarray([2.0 + 4.0 * coeff, 0.0, 2.0, 4.0, 0.5, 5.0])

    monkeypatch.setattr(
        solver_vmec, "vmec_boozer_solver_objective_vector_from_state", fake_vector
    )

    report = vmec_boozer_scalar_objective_finite_difference_report(
        case_name="case",
        objective="growth",
        parameter_family="Zsin",
        base_delta=0.0,
        perturbation_step=1.0e-3,
        response_atol=1.0e-6,
    )

    assert report["passed"] is True
    assert report["parameter_name"] == "Zsin_mid_surface_m1"
    assert report["parameter_indices"] == {"Zsin": [2, 1]}
    assert report["central_derivative"] == pytest.approx(4.0, rel=1.0e-4)

    with pytest.raises(ValueError, match="parameter_family"):
        vmec_boozer_scalar_objective_finite_difference_report(
            parameter_family="BadFamily"
        )


def test_vmec_boozer_aggregate_scalar_objective_finite_difference_report(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class FakeState:
        R_cos: jnp.ndarray

    fake_state = FakeState(R_cos=jnp.zeros((5, 3), dtype=jnp.float32))
    monkeypatch.setattr(
        solver_vmec,
        "_load_vmex_example_state_bundle",
        lambda case_name: {
            "case_name": case_name,
            "input_path": "input.multi",
            "wout_path": "wout.multi",
            "state": fake_state,
            "runtime": "runtime",
            "inp": "inp",
            "wout": "wout",
        },
    )

    def fake_table(state, *_args, **kwargs):  # noqa: ANN001, ANN202
        coeff = float(np.asarray(state.R_cos[2, 1]))
        ky_indices = tuple(kwargs["selected_ky_indices"])
        rows = []
        metadata = []
        for ky in ky_indices:
            rows.append(
                [1.0 + 2.0 * coeff + float(ky), 0.0, 1.0, 3.0, 0.0, 5.0 + coeff]
            )
            metadata.append(
                {"surface_index": None, "alpha": 0.0, "selected_ky_index": int(ky)}
            )
        return jnp.asarray(rows), metadata

    monkeypatch.setattr(
        solver_vmec,
        "vmec_boozer_solver_objective_table_with_metadata_from_state",
        fake_table,
    )

    report = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
        case_name="case",
        objective="growth",
        reduction="weighted_mean",
        weights=[1.0, 3.0],
        selected_ky_indices=[1, 2],
        base_delta=0.1,
        perturbation_step=1.0e-3,
        response_atol=1.0e-6,
        ntheta=4,
    )

    assert gkx.vmec_boozer_aggregate_scalar_objective_finite_difference_report is (
        vmec_boozer_aggregate_scalar_objective_finite_difference_report
    )
    assert report["passed"] is True
    assert report["source_scope"] == "mode21_vmec_boozer_state_multi_point"
    assert report["n_samples"] == 2
    assert report["parameter_name"] == "Rcos_mid_surface_m1"
    assert report["base_value"] == pytest.approx(2.95)
    assert report["central_derivative"] == pytest.approx(2.0, rel=1.0e-4)
    assert report["curvature_ratio"] < 1.0e-3
    assert report["samples"] == [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.75},
    ]
    assert report["options"] == {"ntheta": 4}

    with pytest.raises(ValueError, match="perturbation_step"):
        vmec_boozer_aggregate_scalar_objective_finite_difference_report(
            perturbation_step=0.0
        )
    with pytest.raises(ValueError, match="weights"):
        vmec_boozer_aggregate_scalar_objective_finite_difference_report(
            selected_ky_indices=[1, 2],
            weights=[1.0],
        )


def test_vmec_boozer_aggregate_scalar_objective_line_search_report_accepts_safe_updates(
    monkeypatch,
) -> None:
    calls: list[float] = []

    def fake_fd_report(**kwargs):  # noqa: ANN003, ANN202
        delta = float(kwargs.get("base_delta", 0.0))
        calls.append(delta)
        value = 2.0 + 4.0 * delta
        return {
            "passed": True,
            "base_value": value,
            "central_derivative": 4.0,
            "curvature_ratio": 0.0,
            "n_samples": 2,
            "samples": [
                {
                    "surface_index": None,
                    "alpha": 0.0,
                    "selected_ky_index": 1,
                    "weight": 0.5,
                },
                {
                    "surface_index": None,
                    "alpha": 0.0,
                    "selected_ky_index": 2,
                    "weight": 0.5,
                },
            ],
        }

    monkeypatch.setattr(
        solver_vmec,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_fd_report,
    )

    report = vmec_boozer_aggregate_scalar_objective_line_search_report(
        objective="quasilinear_flux",
        reduction="mean",
        selected_ky_indices=[1, 2],
        update_step=0.05,
        max_steps=2,
        ntheta=4,
    )

    assert gkx.vmec_boozer_aggregate_scalar_objective_line_search_report is (
        vmec_boozer_aggregate_scalar_objective_line_search_report
    )
    assert report["passed"] is True
    assert report["accepted_steps"] == 2
    assert report["n_samples"] == 2
    assert report["final_delta"] == pytest.approx(-0.10)
    assert report["final_objective"] < report["initial_objective"]
    assert all(row["accepted"] for row in report["history"])
    assert calls[:2] == [0.0, -0.05]


def test_vmec_boozer_aggregate_scalar_objective_line_search_report_fails_closed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        solver_vmec,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        lambda **_kwargs: {
            "passed": False,
            "base_value": 1.0,
            "central_derivative": 2.0,
            "curvature_ratio": 9.0,
            "n_samples": 2,
            "samples": [],
        },
    )

    report = vmec_boozer_aggregate_scalar_objective_line_search_report(
        selected_ky_indices=[1, 2],
        max_steps=1,
    )

    assert report["passed"] is False
    assert report["accepted_steps"] == 0
    assert report["stop_reason"] == "finite_difference_gate_failed"
    with pytest.raises(ValueError, match="max_steps"):
        vmec_boozer_aggregate_scalar_objective_line_search_report(max_steps=0)
    with pytest.raises(ValueError, match="update_step"):
        vmec_boozer_aggregate_scalar_objective_line_search_report(update_step=0.0)
    with pytest.raises(ValueError, match="min_improvement"):
        vmec_boozer_aggregate_scalar_objective_line_search_report(min_improvement=-1.0)


def test_vmec_boozer_aggregate_line_search_holdout_report_passes_split(
    monkeypatch,
) -> None:
    calls: list[tuple[float, tuple[int, ...]]] = []

    def fake_line_search(**kwargs):  # noqa: ANN003, ANN202
        return {
            "passed": True,
            "initial_objective": 2.0,
            "final_objective": 1.9,
            "relative_reduction": 0.05,
            "final_delta": -0.1,
            "samples": [{"selected_ky_index": 1}],
        }

    def fake_fd_report(**kwargs):  # noqa: ANN003, ANN202
        delta = float(kwargs.get("base_delta", 0.0))
        ky = tuple(int(item) for item in kwargs.get("selected_ky_indices", ()))
        calls.append((delta, ky))
        return {
            "passed": True,
            "base_value": 1.0 + 0.5 * delta,
            "central_derivative": 0.5,
            "curvature_ratio": 0.0,
            "samples": [{"selected_ky_index": ky[0] if ky else 0}],
        }

    monkeypatch.setattr(
        solver_vmec,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_line_search,
    )
    monkeypatch.setattr(
        solver_vmec,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_fd_report,
    )

    report = vmec_boozer_aggregate_line_search_holdout_report(
        objective="quasilinear_flux",
        training_selected_ky_indices=[1],
        holdout_selected_ky_indices=[2],
        min_holdout_improvement=0.01,
    )

    assert gkx.vmec_boozer_aggregate_line_search_holdout_report is (
        vmec_boozer_aggregate_line_search_holdout_report
    )
    assert report["passed"] is True
    assert report["training_passed"] is True
    assert report["heldout_passed"] is True
    assert report["heldout_initial_objective"] == pytest.approx(1.0)
    assert report["heldout_final_objective"] == pytest.approx(0.95)
    assert report["heldout_relative_reduction"] == pytest.approx(0.05)
    assert calls == [(0.0, (2,)), (-0.1, (2,))]


def test_vmec_boozer_aggregate_line_search_holdout_report_fails_closed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        solver_vmec,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        lambda **_kwargs: {
            "passed": True,
            "initial_objective": 2.0,
            "final_objective": 1.9,
            "relative_reduction": 0.05,
            "final_delta": 0.1,
            "samples": [],
        },
    )
    monkeypatch.setattr(
        solver_vmec,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        lambda **kwargs: {
            "passed": True,
            "base_value": 1.0 + float(kwargs.get("base_delta", 0.0)),
            "central_derivative": 1.0,
            "curvature_ratio": 0.0,
            "samples": [],
        },
    )

    report = vmec_boozer_aggregate_line_search_holdout_report(
        training_selected_ky_indices=[1],
        holdout_selected_ky_indices=[2],
    )

    assert report["passed"] is False
    assert report["training_passed"] is True
    assert report["heldout_passed"] is False
    with pytest.raises(ValueError, match="min_holdout_improvement"):
        vmec_boozer_aggregate_line_search_holdout_report(min_holdout_improvement=-1.0)


def test_vmec_boozer_scalar_objective_line_search_report_accepts_safe_updates(
    monkeypatch,
) -> None:
    calls: list[float] = []

    def fake_fd_report(**kwargs):  # noqa: ANN003, ANN202
        delta = float(kwargs.get("base_delta", 0.0))
        calls.append(delta)
        value = 1.0 + 2.0 * delta
        return {
            "passed": True,
            "base_value": value,
            "central_derivative": 2.0,
            "curvature_ratio": 0.0,
        }

    monkeypatch.setattr(
        solver_vmec,
        "vmec_boozer_scalar_objective_finite_difference_report",
        fake_fd_report,
    )

    report = vmec_boozer_scalar_objective_line_search_report(
        objective="growth",
        update_step=0.05,
        max_steps=2,
        ntheta=4,
    )

    assert gkx.vmec_boozer_scalar_objective_line_search_report is (
        vmec_boozer_scalar_objective_line_search_report
    )
    assert report["passed"] is True
    assert report["accepted_steps"] == 2
    assert report["final_delta"] == pytest.approx(-0.10)
    assert report["final_objective"] < report["initial_objective"]
    assert all(row["accepted"] for row in report["history"])
    assert calls[:2] == [0.0, -0.05]


def test_vmec_boozer_scalar_objective_line_search_report_fails_closed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        solver_vmec,
        "vmec_boozer_scalar_objective_finite_difference_report",
        lambda **_kwargs: {
            "passed": False,
            "base_value": 1.0,
            "central_derivative": 2.0,
            "curvature_ratio": 9.0,
        },
    )

    report = vmec_boozer_scalar_objective_line_search_report(max_steps=1)

    assert report["passed"] is False
    assert report["accepted_steps"] == 0
    assert report["stop_reason"] == "finite_difference_gate_failed"
    with pytest.raises(ValueError, match="max_steps"):
        vmec_boozer_scalar_objective_line_search_report(max_steps=0)
    with pytest.raises(ValueError, match="update_step"):
        vmec_boozer_scalar_objective_line_search_report(update_step=0.0)
    with pytest.raises(ValueError, match="min_improvement"):
        vmec_boozer_scalar_objective_line_search_report(min_improvement=-1.0)


def test_reduced_nonlinear_window_metrics_are_smooth_and_fd_checked() -> None:
    x0 = jnp.asarray([0.22, 0.75, 1.10], dtype=jnp.float32)

    def metric_fn(x: jnp.ndarray) -> jnp.ndarray:
        return _reduced_nonlinear_window_metrics_from_linear_observables(
            x[0],
            x[1],
            x[2],
            dt=0.08,
            steps=18,
            tail_fraction=0.40,
        )

    metrics = np.asarray(metric_fn(x0))
    assert metrics.shape == (3,)
    assert np.all(np.isfinite(metrics))
    assert metrics[0] > 0.0
    assert metrics[1] >= 0.0

    ad = np.asarray(jax.jacfwd(metric_fn)(x0))
    fd_columns = []
    step = 1.0e-3
    eye = np.eye(3, dtype=np.float32)
    for direction in eye:
        forward = np.asarray(metric_fn(x0 + step * jnp.asarray(direction)))
        backward = np.asarray(metric_fn(x0 - step * jnp.asarray(direction)))
        fd_columns.append((forward - backward) / (2.0 * step))
    fd = np.stack(fd_columns, axis=1)
    np.testing.assert_allclose(ad, fd, rtol=2.0e-2, atol=5.0e-3)

    with pytest.raises(ValueError, match="steps"):
        _reduced_nonlinear_window_metrics_from_linear_observables(
            0.1, 1.0, 1.0, steps=3
        )
    with pytest.raises(ValueError, match="tail_fraction"):
        _reduced_nonlinear_window_metrics_from_linear_observables(
            0.1, 1.0, 1.0, tail_fraction=0.0
        )


def test_objective_gate_rows_are_json_ready_and_gate_each_parameter() -> None:
    report = {
        "jacobian_implicit": [[1.0, 2.0], [0.0, 1.1]],
        "jacobian_fd": [[1.0, 2.2], [0.0, 1.0]],
    }

    rows = _objective_gate_rows(
        report,
        parameter_names=("p0", "p1"),
        objective_names=("gamma", "ql"),
        rtol=0.05,
        atol=1.0e-6,
    )

    assert len(rows) == 4
    assert rows[0]["objective"] == "gamma"
    assert rows[0]["parameter"] == "p0"
    assert rows[0]["passed"] is True
    assert rows[1]["passed"] is False
    assert rows[1]["rel_error"] == pytest.approx(abs(2.0 - 2.2) / 2.2)
    assert all(isinstance(row["passed"], bool) for row in rows)


def test_mode21_vmec_boozer_frequency_gate_exports_and_scope() -> None:
    assert gkx.mode21_vmec_boozer_linear_frequency_gradient_report is (
        mode21_vmec_boozer_linear_frequency_gradient_report
    )
    assert gkx.mode21_vmec_boozer_quasilinear_gradient_report is (
        mode21_vmec_boozer_quasilinear_gradient_report
    )
    assert gkx.mode21_vmec_boozer_nonlinear_window_gradient_report is (
        mode21_vmec_boozer_nonlinear_window_gradient_report
    )
    assert tuple(VMEC_BOOZER_STATE_PARAMETER_NAMES) == ("Rcos_mid_surface_m1",)
    assert (
        gkx.VMEC_BOOZER_STATE_PARAMETER_FAMILIES is VMEC_BOOZER_STATE_PARAMETER_FAMILIES
    )
    assert tuple(VMEC_BOOZER_STATE_PARAMETER_FAMILIES) == (
        "Rcos",
        "Rsin",
        "Zcos",
        "Zsin",
        "Lcos",
        "Lsin",
    )
    assert (
        _vmec_boozer_state_parameter_name("Rcos", 17, 1, default_mid_surface=17)
        == "Rcos_mid_surface_m1"
    )
    assert (
        _vmec_boozer_state_parameter_name("Rcos", 17, 2, default_mid_surface=17)
        == "Rcos_mid_surface_m2"
    )
    assert (
        _vmec_boozer_state_parameter_name("Zsin", 17, 2, default_mid_surface=17)
        == "Zsin_mid_surface_m2"
    )
    assert (
        _vmec_boozer_state_parameter_name("Rcos", 16, 2, default_mid_surface=17)
        == "Rcos_r16_m2"
    )
    assert tuple(VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES) == ("gamma", "omega")
    assert tuple(VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES) == (
        "gamma",
        "omega",
        "kperp_eff2",
        "linear_heat_flux_weight",
        "mixing_length_heat_flux_proxy",
    )
    assert tuple(VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES) == (
        "gamma",
        "omega",
        "kperp_eff2",
        "linear_heat_flux_weight",
        "mixing_length_heat_flux_proxy",
        "nonlinear_window_heat_flux_mean",
        "nonlinear_window_heat_flux_cv",
        "nonlinear_window_heat_flux_trend",
    )


def test_write_solver_objective_gradient_artifacts(tmp_path: Path) -> None:
    payload = {
        "kind": "linear_solver_geometry_gradient_gate",
        "passed": True,
        "parameter_names": ["p0", "p1"],
        "objective_names": ["gamma", "omega"],
        "objective_gates": [
            {
                "objective": "gamma",
                "parameter": "p0",
                "implicit": 1.0,
                "finite_difference": 1.0,
                "abs_error": 0.0,
                "rel_error": 0.0,
                "passed": True,
            },
            {
                "objective": "omega",
                "parameter": "p1",
                "implicit": 2.0,
                "finite_difference": 2.0,
                "abs_error": 0.0,
                "rel_error": 0.0,
                "passed": True,
            },
        ],
        "eigenpair_gate": {
            "atol": 1.0e-6,
            "jacobian_implicit": [[1.0, 0.0], [0.0, 2.0]],
            "jacobian_fd": [[1.0, 0.0], [0.0, 2.0]],
        },
    }

    paths = mod.write_solver_objective_gradient_artifacts(
        payload, out=tmp_path / "gate.png"
    )

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "linear_solver_geometry_gradient_gate"


# ---- test_stellarator_objective_portfolio.py ----


from gkx.objectives.portfolio_guard import (
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
    assert isinstance(
        sgk.ReducedPortfolioArtifactGuardConfig(), ReducedPortfolioArtifactGuardConfig
    )
    np.testing.assert_allclose(float(sgk.aggregate_objective_portfolio(rows)), 1.0)
    assert (
        sgk.objective_portfolio_sensitivity_report
        is objective_portfolio_sensitivity_report
    )
    assert (
        sgk.reduced_portfolio_artifact_guard_report
        is reduced_portfolio_artifact_guard_report
    )


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


def _load_stellarator_itg_plotting_module():
    return load_repo_script(
        Path(
            "examples/theory_and_demos/reduced_stellarator_itg/_stellarator_itg_plotting.py"
        ),
        module_name="_stellarator_itg_plotting_for_test",
    )


def test_reduced_stellarator_itg_development_scripts_are_explicit_workflows() -> None:
    examples = REPO_ROOT / "examples" / "theory_and_demos" / "reduced_stellarator_itg"
    scripts = [
        examples / "stellarator_itg_growth_optimization.py",
        examples / "stellarator_itg_quasilinear_flux_optimization.py",
        examples / "stellarator_itg_nonlinear_heat_flux_optimization.py",
        examples / "compare_stellarator_itg_optimizations.py",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert (
            "run_stellarator_itg_adam" in text
            or "compare_scripted_stellarator_itg_objectives" in text
        )
        assert "optimize_stellarator_itg" not in text
        assert "QA_optimization.py" in text


def test_public_optimization_examples_exclude_reduced_synthetic_workflows() -> None:
    examples = REPO_ROOT / "examples" / "optimization"
    names = {path.name for path in examples.iterdir() if path.is_file()}

    assert "QA_optimization_linear_ITG.py" in names
    assert "QA_optimization_quasilinear_ITG.py" in names
    assert "QA_optimization_nonlinear_ITG.py" in names
    assert not any(name.startswith("stellarator_itg_") for name in names)
    assert "_stellarator_itg_plotting.py" not in names
    assert "compare_stellarator_itg_optimizations.py" not in names


def test_public_optimization_examples_keep_editable_constant_style() -> None:
    examples = REPO_ROOT / "examples" / "optimization"
    optimizer_scripts = {
        "QA_optimization_linear_ITG.py",
        "QA_optimization_quasilinear_ITG.py",
        "QA_optimization_nonlinear_ITG.py",
    }
    for script in sorted(examples.glob("*.py")):
        text = script.read_text(encoding="utf-8")
        assert "argparse" not in text
        assert "def main(" not in text
        assert "def _main(" not in text
        assert 'if __name__ == "__main__"' not in text
        if script.name in optimizer_scripts:
            assert "SURFACE_INDEX = 7" in text
            assert "ALPHA = 0.0" in text
            assert "NTHETA = 24" in text
            assert "SELECTED_KY_INDEX = 1" in text
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


def test_stellarator_itg_plotting_artifact_includes_reduced_diagnostics(
    tmp_path: Path,
) -> None:
    plotting = _load_stellarator_itg_plotting_module()
    cfg = _fast_config()
    p0 = np.asarray(default_stellarator_initial_params(), dtype=float)
    p1 = 0.55 * p0
    obs0 = np.asarray(qa_observable_vector(p0, cfg), dtype=float)
    obs1 = np.asarray(qa_observable_vector(p1, cfg), dtype=float)
    payload = {
        "objective_kind": "growth",
        "parameter_names": list(PARAMETER_NAMES),
        "observable_names": list(OBSERVABLE_NAMES),
        "initial_params": p0.tolist(),
        "final_params": p1.tolist(),
        "initial_objective": 1.0,
        "final_objective": 0.5,
        "initial_observables": obs0.tolist(),
        "final_observables": obs1.tolist(),
        "history": [
            {
                "step": 0,
                "objective": 1.0,
                "gradient_norm": 1.0,
                "params": p0.tolist(),
                "observables": obs0.tolist(),
            },
            {
                "step": 1,
                "objective": 0.5,
                "gradient_norm": 0.2,
                "params": p1.tolist(),
                "observables": obs1.tolist(),
            },
        ],
        "gradient_gate": {"passed": True, "max_abs_error": 1.0e-8},
        "covariance": {},
        "nonlinear_trace": None,
        "config": asdict(cfg),
        "backend_info": {
            "vmex_available": True,
            "booz_xform_jax_available": True,
            "vmex_paths": ["/Users/tester/local/vmex"],
            "booz_xform_jax_paths": ["/Users/tester/local/booz_xform_jax"],
        },
        "claim_level": "reduced_linear_or_quasilinear_objective_optimization",
    }
    result = SimpleNamespace(to_dict=lambda: payload)
    out = tmp_path / "stellarator_itg_panel"

    plotting.write_result_artifacts(result, out, title="test panel")

    written = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert "/Users/tester" not in out.with_suffix(".json").read_text(encoding="utf-8")
    assert written["backend_info"] == {
        "vmex_available": True,
        "booz_xform_jax_available": True,
    }
    assert out.with_suffix(".history.csv").exists()
    assert out.with_suffix(".png").exists()
    assert out.with_suffix(".pdf").exists()
    diagnostics = written["reduced_diagnostics"]
    assert (
        diagnostics["claim_level"]
        == "reduced_max_mode1_diagnostics_not_solved_vmec_or_full_nonlinear_scan"
    )
    assert diagnostics["final"]["density_gradient_scan"]["scope"].startswith(
        "reduced_max_mode1"
    )
    assert diagnostics["final"]["fixed_gradient_trace"]["trace_kind"] == (
        "smooth_reduced_nonlinear_envelope_not_full_turbulent_gk"
    )
    assert diagnostics["final"]["surface"]["scope"].endswith(
        "not_solved_vmec_equilibrium"
    )
    assert diagnostics["final"]["lcfs_bmag"]["scope"].startswith(
        "synthetic_reduced_lcfs_bmag"
    )
    assert len(diagnostics["final"]["lcfs_bmag"]["theta"]) >= 64


def test_stellarator_itg_comparison_artifact_has_publication_lcfs_diagnostics(
    tmp_path: Path,
) -> None:
    plotting = _load_stellarator_itg_plotting_module()
    cfg = _fast_config()
    p0 = np.asarray(default_stellarator_initial_params(), dtype=float)
    names = list(OBSERVABLE_NAMES)

    def row(kind: str, scale: float) -> dict[str, object]:
        p1 = scale * p0
        obs0 = np.asarray(qa_observable_vector(p0, cfg), dtype=float)
        obs1 = np.asarray(qa_observable_vector(p1, cfg), dtype=float)
        return {
            "objective_kind": kind,
            "parameter_names": list(PARAMETER_NAMES),
            "observable_names": names,
            "initial_params": p0.tolist(),
            "final_params": p1.tolist(),
            "initial_objective": 1.0,
            "final_objective": 0.5 * scale,
            "initial_observables": obs0.tolist(),
            "final_observables": obs1.tolist(),
            "history": [
                {
                    "step": 0,
                    "objective": 1.0,
                    "gradient_norm": 1.0,
                    "params": p0.tolist(),
                    "observables": obs0.tolist(),
                },
                {
                    "step": 1,
                    "objective": 0.5 * scale,
                    "gradient_norm": 0.2,
                    "params": p1.tolist(),
                    "observables": obs1.tolist(),
                },
            ],
            "gradient_gate": {"passed": True, "max_abs_error": 1.0e-8},
            "covariance": {},
            "nonlinear_trace": None,
            "config": asdict(cfg),
            "backend_info": {},
            "claim_level": "reduced_linear_or_quasilinear_objective_optimization",
        }

    payload = {
        "claim_level": "reduced_objective_optimization_comparison_not_full_production_vmec_gk",
        "production_nonlinear_optimization_claim": False,
        "parameter_names": list(PARAMETER_NAMES),
        "observable_names": names,
        "results": [
            row("growth", 0.70),
            row("quasilinear_flux", 0.55),
            row("nonlinear_heat_flux", 0.45),
        ],
        "backend_info": {},
        "parallel": {"requested_workers": 1},
    }
    out = tmp_path / "stellarator_compare"

    plotting.write_comparison_artifacts(payload, out)

    written = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert written["figure_scope"]["surface_grid"] == {
        "ntheta": 72,
        "nzeta": 72,
        "cmap": "jet",
    }
    assert "synthetic reduced" in written["figure_scope"]["surface_claim"]
    assert "not a solved VMEC LCFS" in written["figure_scope"]["surface_claim"]
    assert out.with_suffix(".png").exists()
    assert out.with_suffix(".pdf").exists()
    plotting_source = REPO_ROOT.joinpath(
        "examples/theory_and_demos/reduced_stellarator_itg/_stellarator_itg_plotting.py"
    ).read_text(encoding="utf-8")
    assert (
        "Reduced synthetic QA ITG optimization-plumbing comparison" in plotting_source
    )
    assert "Synthetic reduced LCFS" in plotting_source
    assert "Synthetic reduced Boozer-LCFS" in plotting_source
    for result in written["results"]:
        diagnostics = result["reduced_diagnostics"]["final"]
        assert diagnostics["surface_summary"]["theta_count"] == 72
        assert diagnostics["surface_summary"]["zeta_count"] == 72
        assert diagnostics["lcfs_bmag_summary"]["theta_count"] == 72
        assert diagnostics["lcfs_bmag_summary"]["zeta_count"] == 72
        assert diagnostics["lcfs_bmag_summary"]["cmap"] == "jet"
        assert "surface" not in diagnostics
        assert "lcfs_bmag" not in diagnostics
        assert diagnostics["fixed_gradient_trace"]["trace_kind"].startswith(
            "smooth_reduced"
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

from gkx.objectives.zonal import (
    ZONAL_FLOW_OBJECTIVE_NAMES,
    ZonalFlowObjectiveConfig,
    zonal_flow_objective_artifact_from_records,
    zonal_flow_objective_rows,
    zonal_flow_objective_sensitivity_report,
    zonal_flow_reduced_objective,
)
from gkx.objectives import zonal
from gkx.objectives.zonal import _finite_metric_tensor_from_records


def test_zonal_record_helpers_have_single_canonical_owner() -> None:
    assert inspect.getmodule(_finite_metric_tensor_from_records) is zonal


def test_zonal_flow_objective_prefers_large_residual_and_low_damping() -> None:
    residual_weak = jnp.asarray([[[0.18, 0.22], [0.20, 0.24]]])
    residual_strong = residual_weak + 0.18
    damping_high = jnp.asarray([[[0.12, 0.10], [0.11, 0.09]]])
    damping_low = damping_high * 0.35
    growth = jnp.asarray([[[0.30, 0.34], [0.28, 0.32]]])
    recurrence = jnp.asarray([[[0.04, 0.05], [0.03, 0.04]]])
    cfg = ZonalFlowObjectiveConfig(
        residual_weight=2.0,
        damping_weight=1.0,
        growth_over_residual_weight=0.5,
        recurrence_weight=0.25,
    )

    weak = zonal_flow_reduced_objective(
        residual_level=residual_weak,
        damping_rate=damping_high,
        linear_growth_rate=growth,
        recurrence_amplitude=recurrence,
        config=cfg,
    )
    strong = zonal_flow_reduced_objective(
        residual_level=residual_strong,
        damping_rate=damping_low,
        linear_growth_rate=growth,
        recurrence_amplitude=recurrence * 0.5,
        config=cfg,
    )

    assert float(strong) < float(weak)
    rows = zonal_flow_objective_rows(
        residual_level=residual_strong,
        damping_rate=damping_low,
        linear_growth_rate=growth,
        recurrence_amplitude=recurrence,
        config=cfg,
    )
    assert tuple(rows.shape) == (1, 2, 2, len(ZONAL_FLOW_OBJECTIVE_NAMES))
    np.testing.assert_allclose(
        np.asarray(rows[..., 0]), 1.0 / np.asarray(residual_strong)
    )


def test_zonal_flow_objective_sensitivity_report_checks_ad_fd_and_conditioning() -> (
    None
):
    surface = jnp.asarray([0.3, 0.7])[:, None, None]
    alpha = jnp.asarray([-0.4, 0.2])[None, :, None]
    kx = jnp.asarray([0.05, 0.11, 0.23])[None, None, :]

    def metric_fn(params: jnp.ndarray) -> dict[str, jnp.ndarray]:
        residual = (
            0.34
            + 0.05 * params[0] * (1.0 + surface)
            + 0.03 * jnp.sin(params[1] + alpha)
            - 0.02 * params[2] * kx
        )
        damping = 0.06 + 0.025 * params[1] ** 2 + 0.015 * surface + 0.01 * kx
        growth = 0.22 + 0.04 * params[2] + 0.02 * alpha + 0.01 * kx
        recurrence = 0.025 + 0.01 * params[0] ** 2 + 0.004 * surface * kx
        return {
            "residual_level": residual,
            "damping_rate": damping,
            "linear_growth_rate": growth,
            "recurrence_amplitude": recurrence,
        }

    cfg = ZonalFlowObjectiveConfig(
        residual_weight=1.5,
        damping_weight=1.0,
        growth_over_residual_weight=0.75,
        recurrence_weight=0.25,
    )
    params = jnp.asarray([0.12, -0.18, 0.26])
    step = 1.0e-4 if bool(jax.config.read("jax_enable_x64")) else 2.0e-3
    rtol = 7.5e-4 if bool(jax.config.read("jax_enable_x64")) else 2.5e-2
    atol = 1.0e-5 if bool(jax.config.read("jax_enable_x64")) else 3.0e-4

    report = zonal_flow_objective_sensitivity_report(
        metric_fn,
        params,
        config=cfg,
        surface_weights=jnp.asarray([1.0, 2.0]),
        alpha_weights=jnp.asarray([1.5, 1.0]),
        ky_weights=jnp.asarray([2.0, 1.0, 1.5]),
        step=step,
        rtol=rtol,
        atol=atol,
        min_rank=3,
        condition_number_limit=1.0e5,
        workers=2,
    )

    assert report["kind"] == "zonal_flow_objective_sensitivity_report"
    assert report["passed"] is True
    assert report["claim_level"].startswith("reduced_zonal_flow_objective")
    assert report["portfolio_contract"]["row_shape"] == [2, 2, 3, 4]
    assert report["scalar_gradient_gate"]["passed"] is True
    assert report["row_jacobian_gate"]["passed"] is True
    assert report["conditioning_gate"]["passed"] is True
    assert report["covariance"]["source"] == "objective_portfolio_rows"
    assert report["objective_config"]["objective_names"] == list(
        ZONAL_FLOW_OBJECTIVE_NAMES
    )
    json.dumps(report, allow_nan=False)


def test_zonal_flow_objective_rejects_invalid_contracts() -> None:
    good = jnp.ones((1, 1, 2)) * 0.3

    with pytest.raises(ValueError, match="at least one"):
        ZonalFlowObjectiveConfig(residual_weight=0.0, damping_weight=0.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        ZonalFlowObjectiveConfig(residual_weight=-1.0)
    with pytest.raises(ValueError, match="residual_floor"):
        ZonalFlowObjectiveConfig(residual_floor=0.0)
    with pytest.raises(ValueError, match="residual_level"):
        zonal_flow_objective_rows(residual_level=jnp.ones((1, 2)), damping_rate=good)
    with pytest.raises(ValueError, match="dimensions"):
        zonal_flow_objective_rows(
            residual_level=jnp.ones((0, 1, 1)), damping_rate=jnp.ones((0, 1, 1))
        )
    with pytest.raises(TypeError, match="real numeric"):
        zonal_flow_objective_rows(
            residual_level=jnp.ones((1, 1, 1), dtype=jnp.complex64),
            damping_rate=jnp.ones((1, 1, 1)),
        )
    with pytest.raises(ValueError, match="finite"):
        zonal_flow_objective_rows(
            residual_level=jnp.asarray([[[jnp.nan]]]),
            damping_rate=jnp.asarray([[[0.1]]]),
        )
    with pytest.raises(ValueError, match="strictly positive"):
        zonal_flow_objective_rows(
            residual_level=jnp.asarray([[[0.0]]]), damping_rate=jnp.asarray([[[0.1]]])
        )
    with pytest.raises(ValueError, match="broadcast-compatible"):
        zonal_flow_objective_rows(residual_level=good, damping_rate=jnp.ones((1, 1, 3)))
    with pytest.raises(ValueError, match="residual_level and damping_rate"):
        zonal_flow_objective_sensitivity_report(
            lambda _p: {"residual_level": good}, jnp.ones(1)
        )


def test_zonal_flow_objective_artifact_from_records_is_strict_and_ranked() -> None:
    records = [
        {
            "surface": 0.25,
            "alpha": 0.0,
            "kx_target": 0.05,
            "residual_level": 0.22,
            "gam_damping_rate": 0.07,
            "linear_growth_rate": 0.30,
            "tail_std_ratio": 1.8,
        },
        {
            "surface": 0.25,
            "alpha": 0.0,
            "kx_target": 0.10,
            "residual_level": 0.44,
            "gam_damping_rate": 0.03,
            "linear_growth_rate": 0.28,
            "tail_std_ratio": 0.8,
        },
    ]
    payload = zonal_flow_objective_artifact_from_records(
        records,
        config=ZonalFlowObjectiveConfig(
            residual_weight=1.0,
            damping_weight=1.0,
            growth_over_residual_weight=0.5,
            recurrence_weight=0.25,
        ),
        source_paths=["docs/_static/example.csv"],
    )

    assert payload["promotion_ready"] is True
    assert payload["missing_damping_count"] == 0
    assert payload["axes"] == {"surface": [0.25], "alpha": [0.0], "kx": [0.05, 0.1]}
    assert payload["sample_count"] == 2
    assert np.asarray(payload["objective_rows"]).shape == (
        1,
        1,
        2,
        len(ZONAL_FLOW_OBJECTIVE_NAMES),
    )
    assert (
        payload["row_table"][1]["sample_objective"]
        < payload["row_table"][0]["sample_objective"]
    )
    assert payload["source_paths"] == ["docs/_static/example.csv"]
    json.dumps(payload, allow_nan=False)


def test_zonal_flow_objective_artifact_missing_damping_policy_and_shape_guards() -> (
    None
):
    records = [
        {"kx": 0.05, "residual_level": 0.25, "tail_std_ratio": 1.0},
        {"kx": 0.10, "residual_level": 0.35, "tail_std_ratio": 0.7},
    ]

    with pytest.raises(ValueError, match="missing finite damping_rate"):
        zonal_flow_objective_artifact_from_records(records)

    payload = zonal_flow_objective_artifact_from_records(
        records, missing_damping_policy="zero"
    )
    assert payload["promotion_ready"] is False
    assert payload["missing_damping_count"] == 2
    np.testing.assert_allclose(np.asarray(payload["metrics"]["damping_rate"]), 0.0)

    with pytest.raises(ValueError, match="duplicate"):
        zonal_flow_objective_artifact_from_records(
            [records[0], records[0]],
            missing_damping_policy="zero",
        )

    with pytest.raises(ValueError, match="strictly positive"):
        zonal_flow_objective_artifact_from_records(
            [{"kx": 0.05, "residual_level": 0.0, "damping_rate": 0.1}],
        )

    recurrence_missing = zonal_flow_objective_artifact_from_records(
        [{"kx": 0.05, "residual_level": 0.25, "damping_rate": 0.1}],
    )
    assert recurrence_missing["missing_recurrence_count"] == 1

    with pytest.raises(ValueError, match="missing_damping_policy"):
        zonal_flow_objective_artifact_from_records(
            [{"kx": 0.05, "residual_level": 0.25, "damping_rate": 0.1}],
            missing_damping_policy="skip",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="at least one"):
        zonal_flow_objective_artifact_from_records([], missing_damping_policy="zero")

    with pytest.raises(ValueError, match="missing finite kx"):
        zonal_flow_objective_artifact_from_records(
            [{"residual_level": 0.25, "damping_rate": 0.1}],
        )

    with pytest.raises(ValueError, match="numeric"):
        zonal_flow_objective_artifact_from_records(
            [{"kx": "not-a-number", "residual_level": 0.25, "damping_rate": 0.1}],
        )

    with pytest.raises(ValueError, match="missing finite residual_level"):
        zonal_flow_objective_artifact_from_records(
            [{"kx": 0.05, "residual_level": "nan", "damping_rate": 0.1}],
        )

    with pytest.raises(ValueError, match="complete finite tensor"):
        zonal_flow_objective_artifact_from_records(
            [
                {
                    "surface": 0.0,
                    "kx": 0.05,
                    "residual_level": 0.25,
                    "damping_rate": 0.1,
                },
                {
                    "surface": 1.0,
                    "kx": 0.10,
                    "residual_level": 0.35,
                    "damping_rate": 0.1,
                },
            ],
        )


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
    import gkx.objectives.stellarator_reduced as reduced

    assert "deprecated" in (reduced.__doc__ or "").lower(), (
        "the reduced-model deprecation notice was removed; it is a fitted "
        "feature map, not gyrokinetics, and must stay marked until deleted"
    )

    source = (REPO_ROOT / "src" / "gkx" / "objectives" / "vmec_transport.py").read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module and "stellarator" in node.module:
            imported.update(alias.name for alias in node.names)

    # Generic helpers only: a softplus and a portfolio-shape contract. Anything
    # else means the production VMEC path has taken a dependency on reduced
    # physics, which is what removal has to avoid.
    assert imported <= {"StellaratorITGSampleSet", "smooth_positive"}, (
        f"vmec_transport imports reduced-model physics: {sorted(imported)}"
    )
