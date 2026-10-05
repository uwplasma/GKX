"""Unit contracts: autodiff solver objectives."""

from __future__ import annotations

from typing import Any
from support.helpers import requires_paired_solvax
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
)
import json
import sys
from types import SimpleNamespace
from scripts.campaigns.vmec_candidate_admission import (
    build_authoritative_wout_candidate_gate,
    build_solved_vmec_candidate_gate,
    build_wout_reproducibility_gate,
    final_iota_profiles_from_vmec_result,
)
import csv
from pathlib import Path
from support.paths import load_artifact_tool
import py_compile
import re


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

from gkx import (
    AdaptiveLinearEigensolverConfig,
    SOLVER_OBJECTIVE_NAMES,
    SolverScalarObjective,
    dominant_eigenvalue_branch_locality_report,
    dominant_real_eigenvalue,
    solver_linear_operator_matrix_from_geometry,
    solver_growth_rate_from_geometry,
    solver_objective_vector_from_geometry,
    solver_scalar_objective_from_vector,
)


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


def _central_difference_jacobian(function, base: jnp.ndarray, step: float):
    eye = jnp.eye(int(base.size), dtype=base.dtype)
    columns = [
        (function(base + step * row) - function(base - step * row)) / (2.0 * step)
        for row in eye
    ]
    return np.stack([np.asarray(column) for column in columns], axis=-1)


def test_dense_objectives_have_forward_mode_matching_reverse_and_fd() -> None:
    """VMEX's least-squares Jacobian is forward mode: jacfwd must work and agree."""

    x64 = bool(jax.config.read("jax_enable_x64"))
    base = jnp.asarray([0.05, 0.20], dtype=jnp.float64 if x64 else jnp.float32)

    def vector(parameter: jnp.ndarray) -> jnp.ndarray:
        return solver_objective_vector_from_geometry(
            _sparse_direct_geometry(parameter), **_SPARSE_DIRECT_GRID
        )

    for function in (_sparse_direct_growth, vector):
        forward = np.asarray(jax.jacfwd(function)(base))
        np.testing.assert_allclose(
            forward, np.asarray(jax.jacrev(function)(base)), rtol=1e-4, atol=1e-6
        )
        fd = _central_difference_jacobian(function, base, 1e-5 if x64 else 2e-3)
        np.testing.assert_allclose(
            forward, fd, rtol=1e-4 if x64 else 5e-2, atol=1e-6 if x64 else 5e-3
        )


@requires_paired_solvax("sparse_eigenvalue", "csr_data_from_products")
def test_sparse_direct_growth_rate_has_forward_mode() -> None:
    if not bool(jax.config.read("jax_enable_x64")):
        pytest.skip("the 1e-8 agreement gate is a float64 gate")
    base = jnp.asarray([0.05, 0.20])
    dense = jax.jacfwd(_sparse_direct_growth)(base)
    gamma, omega = solver_objective_vector_from_geometry(
        _sparse_direct_geometry(base), **_SPARSE_DIRECT_GRID
    )[:2]
    shift = complex(round(float(gamma), 2), round(float(omega), 2))
    sparse = jax.jacfwd(
        lambda p: _sparse_direct_growth(p, eigensolver="sparse-direct", shift=shift)
    )(base)
    np.testing.assert_allclose(sparse, dense, rtol=1e-8, atol=1e-12)


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


def test_dominant_real_eigenvalue_both_modes_match_finite_difference() -> None:
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
    grad_fd = _central_difference_jacobian(objective, params, step)

    assert gkx.dominant_real_eigenvalue is dominant_real_eigenvalue
    np.testing.assert_allclose(jax.jacfwd(objective)(params), grad_ad, rtol=rtol)
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


# ---- test_stellarator_objective_portfolio.py ----


from scripts.campaigns.portfolio_guard import (
    ReducedPortfolioArtifactGuardConfig,
    reduced_portfolio_artifact_guard_report,
)
from gkx.objectives.vmec_transport import (
    aggregate_objective_portfolio,
    portfolio_objective_weight_vector,
    portfolio_sample_weight_tensor,
    validate_objective_portfolio_contract,
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


# ---- from test_vmec_transport_objectives.py ----
# Unit contracts: vmec transport objectives.


POLICY = {
    "target_aspect": 6.0,
    "aspect_atol": 5.0e-2,
    "min_abs_mean_iota": 0.41,
    "qs_residual_max": 5.0e-2,
    "iota_profile_floor": 0.41,
}


def test_candidate_gate_rejects_bad_history_without_nonfinite_json() -> None:
    report = build_solved_vmec_candidate_gate(
        {"aspect_final": "bad", "iota_final": np.inf, "qs_final": np.nan},
        **POLICY,
        iota_profiles=(np.asarray([0.0, 0.412]), np.asarray([0.413])),
    )

    assert report["passed"] is False
    assert report["checks"]["aspect"]["value"] is None
    assert report["checks"]["aspect"]["passed"] is False
    assert report["checks"]["mean_iota"]["value"] is None
    assert report["checks"]["quasisymmetry"]["value"] is None
    json.dumps(report, allow_nan=False)


def test_candidate_gate_requires_iota_profiles_when_floor_is_enabled() -> None:
    report = build_solved_vmec_candidate_gate(
        {"aspect_final": 6.0, "iota_final": 0.42, "qs_final": 0.02},
        **POLICY,
    )

    assert report["passed"] is False
    assert report["checks"]["iota_profile"]["source"] == "missing"
    assert report["checks"]["iota_profile"]["passed"] is False


def test_candidate_gate_can_disable_profile_floor_for_fast_diagnostic_use() -> None:
    report = build_solved_vmec_candidate_gate(
        {"aspect_final": 6.0, "iota_final": 0.42, "qs_final": 0.02},
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=None,
    )

    assert report["passed"] is True
    assert report["checks"]["iota_profile"]["floor"] is None
    assert report["checks"]["iota_profile"]["passed"] is True


def test_final_iota_profiles_from_vmec_result_returns_none_without_solved_state() -> (
    None
):
    assert final_iota_profiles_from_vmec_result(SimpleNamespace(history={})) is None


def test_candidate_gate_extracts_iota_profiles_from_vmex_state(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "vmex", SimpleNamespace())
    wout = SimpleNamespace(
        iotas=np.asarray([0.0, 0.411, 0.415]),
        iotaf=np.asarray([0.412, 0.416]),
    )
    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": -0.42, "qs_final": 0.02},
        final_equilibrium=SimpleNamespace(wout=wout),
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert report["passed"] is True
    assert report["checks"]["mean_iota"]["value"] == 0.42
    assert report["checks"]["iota_profile"]["source"] == "vmex_state"
    assert report["checks"]["iota_profile"]["minimum_iotas_excluding_axis"] == 0.411


def test_candidate_gate_prefers_independent_state_qs_over_history(monkeypatch) -> None:
    class FakeQS:
        def __init__(self, surfaces, *, helicity_m=1, helicity_n=0, **_kwargs):
            assert np.asarray(surfaces).shape[0] == 11
            assert (helicity_m, helicity_n) == (1, 0)

        def total_state(self, state, runtime):
            assert state == "state"
            assert runtime == "runtime"
            return 0.013

    fake_vmex = SimpleNamespace(
        optimize=SimpleNamespace(QuasisymmetryRatioResidual=FakeQS)
    )
    monkeypatch.setitem(sys.modules, "vmex", fake_vmex)
    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": 0.428, "qs_final": 99.0},
        final_state="state",
        final_runtime="runtime",
        final_wout=SimpleNamespace(
            iotas=np.asarray([0.0, 0.411, 0.415]),
            iotaf=np.asarray([0.412, 0.416]),
        ),
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert report["passed"] is True
    assert report["checks"]["quasisymmetry"]["value"] == 0.013
    assert report["checks"]["quasisymmetry"]["source"] == "vmex_state"


def test_candidate_gate_uses_standalone_qs_not_assembled_transport_block(
    monkeypatch,
) -> None:
    class FakeQS:
        def __init__(self, surfaces, *, helicity_m=1, helicity_n=0, **_kwargs):
            assert np.asarray(surfaces).shape[0] == 11
            assert (helicity_m, helicity_n) == (1, 0)

        def total_state(self, state, runtime):
            assert state == "state"
            assert runtime == "runtime"
            return 0.009

    fake_vmex = SimpleNamespace(
        optimize=SimpleNamespace(QuasisymmetryRatioResidual=FakeQS)
    )
    monkeypatch.setitem(sys.modules, "vmex", fake_vmex)

    class FakeOptimizer:
        def quasisymmetry_objective(self, _params):
            raise AssertionError("assembled optimizer objective must not be used")

    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": 0.428, "qs_final": 99.0},
        final_state="state",
        final_runtime="runtime",
        final_params=(1.0, 2.0),
        final_optimizer=FakeOptimizer(),
        final_wout=SimpleNamespace(
            iotas=np.asarray([0.0, 0.411, 0.415]),
            iotaf=np.asarray([0.412, 0.416]),
        ),
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert report["passed"] is True
    assert report["checks"]["quasisymmetry"]["value"] == 0.009
    assert report["checks"]["quasisymmetry"]["source"] == "vmex_state"


def test_candidate_gate_state_qs_falls_back_to_optimizer_method(monkeypatch) -> None:
    class FailingQS:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("qs residual unavailable")

    fake_vmex = SimpleNamespace(
        optimize=SimpleNamespace(QuasisymmetryRatioResidual=FailingQS)
    )
    monkeypatch.setitem(sys.modules, "vmex", fake_vmex)

    class FakeOptimizer:
        def quasisymmetry_objective(self, params):
            assert params == (1.0, 2.0)
            return 0.017

    result = SimpleNamespace(
        history={"aspect_final": 6.0, "iota_final": 0.428, "qs_final": 99.0},
        final_state="state",
        final_runtime="runtime",
        final_params=(1.0, 2.0),
        final_optimizer=FakeOptimizer(),
        final_wout=SimpleNamespace(
            iotas=np.asarray([0.0, 0.411, 0.415]),
            iotaf=np.asarray([0.412, 0.416]),
        ),
    )

    report = build_solved_vmec_candidate_gate(result, **POLICY)

    assert report["passed"] is True
    assert report["checks"]["quasisymmetry"]["value"] == 0.017


def test_final_iota_profiles_from_vmec_result_handles_vmex_failure() -> None:
    class BrokenEquilibrium:
        @property
        def wout(self):
            raise RuntimeError("not converged")

    result = SimpleNamespace(final_equilibrium=BrokenEquilibrium())

    assert final_iota_profiles_from_vmec_result(result) is None


def test_wout_reproducibility_gate_rejects_iota_drift() -> None:
    report = build_wout_reproducibility_gate(
        {
            "source": "optimizer_state_wout",
            "aspect": 5.000154,
            "mean_iota": 0.41020,
            "min_iotas_excluding_axis": 0.40567,
            "min_iotaf": 0.40550,
        },
        {
            "source": "input_final_rerun_wout",
            "aspect": 5.000154,
            "mean_iota": 0.40851,
            "min_iotas_excluding_axis": 0.39598,
            "min_iotaf": 0.39581,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        iota_profile_floor=None,
        mean_iota_repro_atol=5.0e-4,
    )

    assert report["passed"] is False
    assert report["checks"]["rerun_mean_iota_admission"]["passed"] is False
    assert report["checks"]["mean_iota_reproducibility"]["passed"] is False
    assert report["checks"]["mean_iota_reproducibility"][
        "absolute_drift"
    ] == pytest.approx(0.00169)
    json.dumps(report, allow_nan=False)


def test_wout_reproducibility_gate_accepts_matching_rerun() -> None:
    report = build_wout_reproducibility_gate(
        {
            "source": "optimizer_state_wout",
            "aspect": 5.000154,
            "mean_iota": 0.41020,
            "min_iotas_excluding_axis": 0.40567,
            "min_iotaf": 0.40550,
        },
        {
            "source": "input_final_rerun_wout",
            "aspect": 5.0001542,
            "mean_iota": 0.41010,
            "min_iotas_excluding_axis": 0.40561,
            "min_iotaf": 0.40545,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        iota_profile_floor=None,
        mean_iota_repro_atol=5.0e-4,
        aspect_repro_atol=5.0e-7,
        profile_repro_atol=5.0e-4,
    )

    assert report["passed"] is True
    assert report["checks"]["rerun_mean_iota_admission"]["passed"] is True


def test_authoritative_wout_candidate_gate_accepts_mapping_with_qs() -> None:
    report = build_authoritative_wout_candidate_gate(
        {
            "source": "deterministic_rerun_wout",
            "aspect": 5.0001,
            "mean_iota": -0.411,
            "min_iotas_excluding_axis": 0.405,
            "min_iotaf": 0.404,
            "qs_residual": 2.0e-3,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=None,
    )

    assert report["passed"] is True
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["value"] == pytest.approx(0.411)
    assert report["checks"]["quasisymmetry"]["source"] == "mapping"
    json.dumps(report, allow_nan=False)


def test_authoritative_wout_candidate_gate_rejects_missing_qs() -> None:
    report = build_authoritative_wout_candidate_gate(
        {
            "source": "deterministic_rerun_wout",
            "aspect": 5.0001,
            "mean_iota": 0.411,
            "min_iotas_excluding_axis": 0.405,
            "min_iotaf": 0.404,
        },
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=None,
    )

    assert report["passed"] is False
    assert report["checks"]["quasisymmetry"]["passed"] is False
    assert report["checks"]["quasisymmetry"]["error"] == "missing_qs_residual"


def test_authoritative_wout_candidate_gate_reads_wout_file_with_profile_floor(
    tmp_path, monkeypatch
) -> None:
    class FakeVar:
        def __init__(self, value):
            self.value = value

        def __getitem__(self, _key):
            return np.asarray(self.value)

    class FakeDataset:
        def __init__(self, path):
            assert path == tmp_path / "wout_final_rerun.nc"
            self.variables = {
                "aspect": FakeVar(5.0002),
                "iotas": FakeVar([0.0, 0.412, 0.418]),
                "iotaf": FakeVar([0.413, 0.419]),
            }

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def fake_read_wout(path):
        assert path == tmp_path / "wout_final_rerun.nc"
        return "loaded-wout"

    class FakeQS:
        def __init__(self, surfaces, *, helicity_m, helicity_n, ntheta, nphi):
            assert tuple(np.asarray(surfaces, dtype=float)) == (0.0, 0.5, 1.0)
            assert (helicity_m, helicity_n, ntheta, nphi) == (1, 0, 31, 32)

        def total(self, wout):
            assert wout == "loaded-wout"
            return 0.003

    monkeypatch.setitem(sys.modules, "netCDF4", SimpleNamespace(Dataset=FakeDataset))
    monkeypatch.setitem(
        sys.modules,
        "vmex",
        SimpleNamespace(
            read_wout=fake_read_wout,
            optimize=SimpleNamespace(QuasisymmetryRatioResidual=FakeQS),
        ),
    )

    report = build_authoritative_wout_candidate_gate(
        tmp_path / "wout_final_rerun.nc",
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.411,
        qs_surfaces=(0.0, 0.5, 1.0),
        qs_ntheta=31,
        qs_nphi=32,
    )

    assert report["passed"] is True
    assert report["authoritative_wout"]["mean_iota"] == pytest.approx(0.415)
    assert report["checks"]["iota_profile"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["source"] == "vmex_wout"


def test_authoritative_wout_candidate_gate_reports_wout_load_errors(
    tmp_path, monkeypatch
) -> None:
    def broken_dataset(_path):
        raise OSError("missing variable")

    def broken_read_wout(_path):
        raise RuntimeError("bad wout")

    monkeypatch.setitem(sys.modules, "netCDF4", SimpleNamespace(Dataset=broken_dataset))
    monkeypatch.setitem(
        sys.modules, "vmex", SimpleNamespace(read_wout=broken_read_wout)
    )

    report = build_authoritative_wout_candidate_gate(
        tmp_path / "bad_wout.nc",
        target_aspect=5.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
    )

    assert report["passed"] is False
    assert report["authoritative_wout"]["aspect"] is None
    assert report["checks"]["iota_profile"]["passed"] is False
    assert report["checks"]["quasisymmetry"]["source"] == "vmex_wout_error"


# ---- test_vmex_transport_admission.py ----


import scripts.campaigns.stellarator_transport_reports as transport_reports
from scripts.campaigns.stellarator_transport_reports import (
    build_nonlinear_audit_redesign_report,
    build_nonlinear_campaign_admission_report,
    build_nonlinear_landscape_admission_report,
    build_reduced_nonlinear_audit_prelaunch_report,
)
from scripts.campaigns.vmec_transport_admission import (
    VMEXNonlinearAuditPolicy,
    VMEXNonlinearCampaignPolicy,
    VMEXReducedPrelaunchPolicy,
    VMEXTransportAdmissionPolicy,
)
from scripts.campaigns.vmec_transport_admission import (
    candidate_transport_metric,
    transport_objective_sample_summary,
)
from scripts.campaigns.vmec_transport_admission import (
    build_transport_admission_report,
    select_admitted_transport_candidate,
)


def _candidate(
    label: str,
    *,
    objective: float,
    weight: float | None = None,
    passed: bool = True,
    authoritative: bool = True,
    baseline: bool = False,
) -> dict[str, object]:
    return {
        "label": label,
        "baseline": baseline,
        "transport_weight": weight,
        "passed": passed and authoritative,
        "gate_reported_passed": passed,
        "gate_is_authoritative": authoritative,
        "gate_checks": {
            "aspect": passed,
            "mean_iota": True,
            "quasisymmetry": passed,
            "iota_profile": passed,
        },
        "objective_final": objective,
    }


def test_transport_metric_prefers_explicit_transport_metric_over_total_objective() -> (
    None
):
    metric = candidate_transport_metric(
        {
            "objective_final": 4.0,
            "gkx_objective_final": 2.0,
            "transport_objective_final": 1.0,
        }
    )

    assert metric["available"] is True
    assert metric["source"] == "transport_objective_final"
    assert metric["value"] == 1.0
    assert metric["uses_total_objective_proxy"] is False


def test_transport_admission_selects_largest_physical_improving_weight() -> None:
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("low", objective=0.8, weight=0.001),
        _candidate("high", objective=0.7, weight=0.005),
        _candidate("failed", objective=0.1, weight=0.01, passed=False),
    ]

    report = build_transport_admission_report(summaries)

    assert report["transport_candidate_admitted"] is True
    assert report["promoted_candidate"]["label"] == "high"
    assert report["promoted_candidate"]["transport_weight"] == 0.005
    assert report["admitted_transport_candidates"] == ["low", "high"]


def test_transport_admission_blocks_worse_transport_metric_even_if_gate_passes() -> (
    None
):
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("worse", objective=1.1, weight=0.001),
    ]

    report = build_transport_admission_report(summaries)
    worse = report["candidates"][1]

    assert report["transport_candidate_admitted"] is False
    assert report["promoted_candidate"]["label"] == "baseline"
    assert worse["relative_transport_improvement"] < 0.0
    assert "insufficient_transport_improvement" in worse["admission_blockers"]


def test_transport_admission_blocks_non_authoritative_gate() -> None:
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("legacy", objective=0.5, weight=0.001, authoritative=False),
    ]

    report = build_transport_admission_report(summaries)
    legacy = report["candidates"][1]

    assert report["transport_candidate_admitted"] is False
    assert "non_authoritative_gate" in legacy["admission_blockers"]
    assert report["promoted_candidate"]["label"] == "baseline"


def test_transport_admission_can_require_stronger_relative_improvement() -> None:
    policy = VMEXTransportAdmissionPolicy(minimum_relative_improvement=0.25)
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("small", objective=0.9, weight=0.001),
        _candidate("large", objective=0.7, weight=0.002),
    ]

    report = build_transport_admission_report(summaries, policy=policy)

    assert report["admitted_transport_candidates"] == ["large"]
    assert report["promoted_candidate"]["label"] == "large"
    assert (
        select_admitted_transport_candidate(summaries, policy=policy)
        == report["promoted_candidate"]
    )


def test_transport_admission_is_not_installable_public_api() -> None:
    """Campaign admission policy lives in scripts/campaigns, not in the package."""

    for name in (
        "VMEXTransportAdmissionPolicy",
        "build_transport_admission_report",
        "candidate_transport_metric",
        "select_admitted_transport_candidate",
    ):
        assert not hasattr(gkx, name)


def _matched_comparison(
    *,
    relative_reduction: float,
    z_score: float,
    passed: bool,
) -> dict[str, object]:
    return {
        "kind": "matched_nonlinear_transport_comparison",
        "case": "qa_projected_transport_step1e3",
        "passed": passed,
        "baseline": {"passed": True, "ensemble_mean": 9.833},
        "candidate": {"passed": True, "ensemble_mean": 9.891},
        "statistics": {
            "relative_reduction": relative_reduction,
            "uncertainty_z_score": z_score,
        },
    }


def _ensemble(
    mean: float, sem: float, *, passed: bool = True, n_reports: int = 3
) -> dict[str, object]:
    return {
        "case": f"ensemble_mean_{mean}",
        "passed": passed,
        "statistics": {
            "ensemble_mean": mean,
            "combined_sem": sem,
            "combined_sem_rel": sem / abs(mean),
            "n_reports": n_reports,
        },
    }


def test_nonlinear_landscape_admission_selects_uncertainty_resolved_candidate() -> None:
    report = build_nonlinear_landscape_admission_report(
        _ensemble(8.554362366164424, 0.11951503416978174),
        [
            _ensemble(6.274543846475065, 0.04213243251063571),
            _ensemble(6.42653555490751, 0.04399590111876854),
        ],
        candidate_labels=("+3%", "+6%"),
        policy=VMEXNonlinearAuditPolicy(
            minimum_relative_reduction=0.02,
            minimum_uncertainty_z_score=2.0,
            maximum_combined_sem_rel=0.05,
            minimum_replicate_count=3,
        ),
    )

    assert report["passed"] is True
    assert report["selected_candidate"]["label"] == "+3%"
    assert report["selected_candidate"]["relative_reduction"] > 0.26
    assert report["selected_candidate"]["uncertainty_z_score"] > 17.0
    assert all(row["admitted"] for row in report["candidates"])
    assert (
        build_nonlinear_landscape_admission_report
        is transport_reports.build_nonlinear_landscape_admission_report
    )
    json.dumps(report, allow_nan=False)


def test_nonlinear_landscape_admission_fails_closed_for_noisy_or_unresolved_candidates() -> (
    None
):
    report = build_nonlinear_landscape_admission_report(
        _ensemble(8.0, 0.5),
        [
            _ensemble(7.95, 0.5),
            _ensemble(6.0, 2.0, n_reports=2),
            _ensemble(5.0, 0.1, passed=False),
        ],
        policy=VMEXNonlinearAuditPolicy(
            minimum_relative_reduction=0.02,
            minimum_uncertainty_z_score=2.0,
            maximum_combined_sem_rel=0.2,
            minimum_replicate_count=3,
        ),
    )

    assert report["passed"] is False
    assert report["selected_candidate"] is None
    blockers = [set(row["admission_blockers"]) for row in report["candidates"]]
    assert "insufficient_relative_reduction" in blockers[0]
    assert "insufficient_uncertainty_separation" in blockers[0]
    assert "candidate_combined_sem_rel_too_large" in blockers[1]
    assert "candidate_insufficient_replicates" in blockers[1]
    assert "candidate_ensemble_failed" in blockers[2]


@pytest.mark.parametrize("invalid", ["false", "true", 1, None])
def test_nonlinear_landscape_admission_rejects_nonboolean_pass_flags(invalid) -> None:
    baseline = _ensemble(8.0, 0.1)
    candidate = _ensemble(6.0, 0.1)
    baseline["passed"] = invalid

    report = build_nonlinear_landscape_admission_report(baseline, [candidate])

    assert report["passed"] is False
    assert "baseline_ensemble_failed" in report["candidates"][0]["admission_blockers"]


@pytest.mark.parametrize("invalid", ["three", 2.5, -3, float("nan")])
def test_nonlinear_landscape_admission_rejects_invalid_replicate_counts(
    invalid,
) -> None:
    candidate = _ensemble(6.0, 0.1)
    candidate["statistics"]["n_reports"] = invalid

    report = build_nonlinear_landscape_admission_report(
        _ensemble(8.0, 0.1), [candidate]
    )

    assert report["passed"] is False
    assert (
        "candidate_insufficient_replicates"
        in report["candidates"][0]["admission_blockers"]
    )


def test_nonlinear_landscape_admission_validates_candidate_labels() -> None:
    try:
        build_nonlinear_landscape_admission_report(
            _ensemble(8.0, 0.1),
            [_ensemble(7.0, 0.1)],
            candidate_labels=("one", "two"),
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched candidate labels were accepted")


def test_reduced_nonlinear_audit_prelaunch_passes_calibrated_landscape_margin() -> None:
    baseline = 0.06558065223919245
    candidate = 0.06251277500404685

    report = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=baseline,
        candidate_metric=candidate,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        baseline_sample_statistics={
            "weighted_mean": 0.06777885259618041,
            "weighted_standard_error": 0.015344998342625694,
        },
        candidate_sample_statistics={
            "weighted_mean": 0.06450805792574345,
            "weighted_standard_error": 0.014457225619392737,
        },
        failed_reference_relative_reduction=0.022876,
        policy=VMEXReducedPrelaunchPolicy(minimum_relative_reduction=0.04),
    )

    assert report["passed"] is True
    assert report["relative_reduced_reduction"] > 0.046
    assert report["required_relative_reduced_reduction"] == 0.04
    assert report["blockers"] == []
    assert report["gates"][0]["passed"] is True
    assert report["reduced_cross_sample_statistics"]["passed"] is True
    assert report["gates"][2]["metric"] == "reduced_cross_sample_dispersion"
    assert (
        build_reduced_nonlinear_audit_prelaunch_report
        is transport_reports.build_reduced_nonlinear_audit_prelaunch_report
    )


def test_reduced_nonlinear_audit_prelaunch_blocks_weak_failed_transfer_margin() -> None:
    report = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=0.08010670290,
        candidate_metric=0.07827418221,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        failed_reference_relative_reduction=0.022876,
        policy=VMEXReducedPrelaunchPolicy(
            minimum_relative_reduction=0.04,
            failed_reference_safety_factor=1.5,
        ),
    )

    assert report["passed"] is False
    assert "insufficient_reduced_margin_for_nonlinear_audit" in report["blockers"]
    assert (
        report["relative_reduced_reduction"]
        < report["required_relative_reduced_reduction"]
    )


def test_reduced_prelaunch_blocks_excessive_reduced_cross_sample_spread() -> None:
    report = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=0.06558065223919245,
        candidate_metric=0.06251277500404685,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        baseline_sample_statistics={
            "weighted_mean": 0.067,
            "weighted_standard_error": 0.03,
        },
        candidate_sample_statistics={
            "weighted_mean": 0.064,
            "weighted_standard_error": 0.04,
        },
        policy=VMEXReducedPrelaunchPolicy(
            minimum_relative_reduction=0.04,
            maximum_cross_sample_sem_rel=0.35,
        ),
    )

    assert report["passed"] is False
    assert "candidate_cross_sample_sem_rel_too_large" in report["blockers"]
    assert report["gates"][2]["passed"] is False


def test_campaign_admission_combines_reduced_and_replicated_landscape_gates() -> None:
    prelaunch = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=0.06558065223919245,
        candidate_metric=0.06251277500404685,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        baseline_sample_statistics={
            "weighted_mean": 0.06777885259618041,
            "weighted_standard_error": 0.015344998342625694,
        },
        candidate_sample_statistics={
            "weighted_mean": 0.06450805792574345,
            "weighted_standard_error": 0.014457225619392737,
        },
        policy=VMEXReducedPrelaunchPolicy(minimum_relative_reduction=0.04),
    )
    landscape = build_nonlinear_landscape_admission_report(
        _ensemble(8.554362366164424, 0.11951503416978174),
        [_ensemble(6.274543846475065, 0.04213243251063571)],
        candidate_labels=("+3% RBC(0,1)",),
        policy=VMEXNonlinearAuditPolicy(
            minimum_relative_reduction=0.02,
            minimum_uncertainty_z_score=2.0,
            maximum_combined_sem_rel=0.05,
            minimum_replicate_count=3,
        ),
    )

    report = build_nonlinear_campaign_admission_report(
        reduced_prelaunch_report=prelaunch,
        landscape_admission_report=landscape,
    )

    assert report["campaign_admitted"] is True
    assert report["blockers"] == []
    assert report["selected_landscape_candidate"]["label"] == "+3% RBC(0,1)"
    assert report["claim_scope"].startswith(
        "next nonlinear optimizer-campaign admission"
    )
    assert (
        build_nonlinear_campaign_admission_report
        is transport_reports.build_nonlinear_campaign_admission_report
    )
    json.dumps(report, allow_nan=False)


def test_campaign_admission_fails_closed_without_cross_sample_gate_or_landscape_margin() -> (
    None
):
    prelaunch = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=1.0,
        candidate_metric=0.95,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        policy=VMEXReducedPrelaunchPolicy(minimum_relative_reduction=0.04),
    )
    landscape = build_nonlinear_landscape_admission_report(
        _ensemble(8.0, 0.3),
        [_ensemble(7.4, 0.3)],
        candidate_labels=("weak",),
        policy=VMEXNonlinearAuditPolicy(minimum_relative_reduction=0.02),
    )

    report = build_nonlinear_campaign_admission_report(
        reduced_prelaunch_report=prelaunch,
        landscape_admission_report=landscape,
        policy=VMEXNonlinearCampaignPolicy(
            minimum_landscape_relative_reduction=0.10,
            minimum_landscape_uncertainty_z_score=3.0,
        ),
    )

    assert report["campaign_admitted"] is False
    assert "reduced_cross_sample_statistics_missing" in report["blockers"]
    assert "selected_landscape_reduction_too_small" in report["blockers"]
    assert "selected_landscape_uncertainty_separation_too_small" in report["blockers"]


def test_campaign_admission_rejects_corrupt_persisted_gate_fields() -> None:
    prelaunch = {
        "passed": "true",
        "objective_sample_summary": {"passed": "true", "sample_count": 18},
        "reduced_cross_sample_statistics": {
            "available": "true",
            "passed": "true",
            "rows": [],
        },
    }
    landscape = {
        "passed": "true",
        "selected_candidate": {
            "relative_reduction": 0.2,
            "uncertainty_z_score": 4.0,
            "combined_sem_rel": 0.01,
            "n_reports": "three",
        },
    }

    report = build_nonlinear_campaign_admission_report(
        reduced_prelaunch_report=prelaunch,
        landscape_admission_report=landscape,
    )

    assert report["campaign_admitted"] is False
    assert "reduced_prelaunch_gate_failed" in report["blockers"]
    assert "reduced_objective_sample_coverage_failed" in report["blockers"]
    assert "reduced_cross_sample_statistics_missing" in report["blockers"]
    assert "replicated_landscape_admission_failed" in report["blockers"]
    assert "selected_landscape_insufficient_replicates" in report["blockers"]


def test_transport_sample_summary_requires_surface_alpha_and_ky_coverage() -> None:
    summary = transport_objective_sample_summary(
        {"surfaces": [0.5], "alphas": [0.0], "ky_values": [0.3]}
    )

    assert summary["passed"] is False
    assert summary["sample_count"] == 1
    assert "insufficient_surface_coverage" in summary["blockers"]
    assert "insufficient_field_line_coverage" in summary["blockers"]
    assert "insufficient_ky_coverage" in summary["blockers"]


def test_nonlinear_audit_redesign_blocks_negative_transfer_and_recommends_multisample_design() -> (
    None
):
    report = build_nonlinear_audit_redesign_report(
        _matched_comparison(relative_reduction=-0.00585, z_score=-0.20, passed=False),
        objective_sample_set={"surfaces": [0.64], "alphas": [0.0], "ky_values": [0.3]},
    )

    assert report["nonlinear_audit_promoted"] is False
    assert report["requires_objective_redesign"] is True
    assert "insufficient_matched_reduction" in report["blockers"]
    assert "insufficient_uncertainty_separation" in report["blockers"]
    assert "insufficient_total_sample_count" in report["blockers"]
    assert report["recommended_sample_set"]["sample_count"] == 18
    assert report["gates"][0]["passed"] is False
    json.dumps(report, allow_nan=False)


def test_nonlinear_audit_redesign_promotes_only_when_audit_and_sample_coverage_pass() -> (
    None
):
    policy = VMEXNonlinearAuditPolicy(
        minimum_relative_reduction=0.02,
        minimum_uncertainty_z_score=1.0,
        minimum_surface_count=3,
        minimum_alpha_count=2,
        minimum_ky_count=3,
        minimum_sample_count=12,
    )
    sample_set = {
        "surfaces": [0.45, 0.64, 0.78],
        "alphas": [0.0, 0.7853981633974483],
        "ky_values": [0.1, 0.3, 0.5],
    }

    report = build_nonlinear_audit_redesign_report(
        _matched_comparison(relative_reduction=0.08, z_score=2.5, passed=True),
        objective_sample_set=sample_set,
        policy=policy,
    )

    assert report["nonlinear_audit_promoted"] is True
    assert report["requires_objective_redesign"] is False
    assert report["blockers"] == []
    assert report["objective_sample_summary"]["sample_count"] == 18
    assert all(gate["passed"] for gate in report["gates"])
    assert (
        build_nonlinear_audit_redesign_report
        is transport_reports.build_nonlinear_audit_redesign_report
    )


def test_nonlinear_audit_redesign_rejects_nonboolean_persisted_pass_flags() -> None:
    comparison = _matched_comparison(
        relative_reduction=0.08,
        z_score=2.5,
        passed=True,
    )
    comparison["passed"] = "true"
    comparison["baseline"]["passed"] = "true"

    report = build_nonlinear_audit_redesign_report(
        comparison,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, np.pi / 4.0],
            "ky_values": [0.1, 0.3, 0.5],
        },
    )

    assert report["nonlinear_audit_promoted"] is False
    assert "baseline_ensemble_failed" in report["blockers"]
    assert "matched_comparison_not_passed" in report["blockers"]


def test_transport_sample_summary_rejects_ky_values_not_supported_by_single_solver_grid() -> (
    None
):
    summary = transport_objective_sample_summary(
        {
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.19, 0.3, 0.476],
        }
    )

    assert summary["passed"] is False
    assert "ky_values_not_single_grid_compatible" in summary["blockers"]


# ---- test_vmex_transport_gradient.py ----

from dataclasses import dataclass


@dataclass(frozen=True)
class FakeSpec:
    name: str
    kind: str
    index: int
    m: int
    n: int


class FakeOptimizer:
    _specs = (
        FakeSpec("rc01", "rc", 0, 0, 1),
        FakeSpec("zs10", "zs", 1, 1, 0),
        FakeSpec("rc11", "rc", 2, 1, 1),
    )

    def residual_fun(self, params):
        params = np.asarray(params, dtype=float)
        return np.asarray([0.4 + params[0] - 2.0 * params[1]])

    def objective_and_gradient_fun(self, params):
        residual = self.residual_fun(params)[0]
        jac = np.asarray([1.0, -2.0, 0.0])
        return 0.5 * residual**2, residual * jac

    def jacobian_fun(self, params):
        return np.asarray([[1.0, -2.0, 0.0]])


# ---- test_vmex_transport_line_search.py ----


def _gradient_report() -> dict[str, object]:
    return {
        "parameter_count": 4,
        "top_gradient_components": [
            {"parameter_index": 1, "gradient": -3.0, "name": "zs10"},
            {"parameter_index": 3, "gradient": 4.0, "name": "rc11"},
            {"parameter_index": 0, "gradient": 12.0, "name": "rc01"},
        ],
    }


def _boundary_chain_collection() -> dict[str, object]:
    return {
        "kind": "vmex_boundary_chain_collection_summary",
        "classification": "mixed_exact_fd_consistency_with_branch_sensitive_modes",
        "rows": [
            {
                "index": 1,
                "name": "zs10",
                "finite": True,
                "frozen_axis_jvp_vjp_consistent": True,
                "frozen_axis_matches_exact_fd": True,
                "frozen_axis_convention_verified": False,
                "growth_branch_locality_checked": True,
                "growth_branch_locality_passed": True,
            },
            {
                "index": 3,
                "name": "rc11",
                "finite": True,
                "frozen_axis_jvp_vjp_consistent": True,
                "frozen_axis_matches_exact_fd": False,
                "frozen_axis_convention_verified": True,
                "growth_branch_locality_checked": True,
                "growth_branch_locality_passed": False,
            },
            {
                "index": 0,
                "name": "rc01",
                "finite": True,
                "frozen_axis_jvp_vjp_consistent": False,
                "frozen_axis_matches_exact_fd": True,
                "frozen_axis_convention_verified": False,
                "growth_branch_locality_checked": False,
                "growth_branch_locality_passed": False,
            },
        ],
    }


# ---- test_vmex_transport_objective.py ----

"""Tests for VMEC-JAX to GKX transport objective plumbing."""


from types import ModuleType


from gkx import (
    StellaratorITGSampleSet,
    VMEXGKXTransportObjective,
    VMEXTransportObjectiveConfig,
    vmex_transport_objective_from_state,
)
import gkx.objectives.vmec_transport as transport_config
import gkx.objectives.vmec_transport as transport_tables


def _fake_geometry() -> SimpleNamespace:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    return SimpleNamespace(
        theta=theta,
        bmag_profile=1.0 + 0.05 * jnp.cos(theta),
        jacobian_profile=jnp.ones_like(theta),
        gds2_profile=1.2 + 0.1 * jnp.cos(theta),
        gds21_profile=0.05 * jnp.sin(theta),
        gds22_profile=1.0 + 0.08 * jnp.cos(2.0 * theta),
        cv_profile=0.03 * jnp.sin(theta),
        gb_profile=0.04 * jnp.cos(theta),
        cv0_profile=0.02 * jnp.sin(2.0 * theta),
        gb0_profile=0.02 * jnp.cos(2.0 * theta),
    )


def _fake_solver_rows(scale: float = 1.0) -> jnp.ndarray:
    rows = []
    idx = {name: i for i, name in enumerate(SOLVER_OBJECTIVE_NAMES)}
    for gamma in (0.08, 0.10, 0.12, 0.14):
        row = np.zeros(len(SOLVER_OBJECTIVE_NAMES), dtype=float)
        row[idx["gamma"]] = scale * gamma
        row[idx["omega"]] = -0.2
        row[idx["kperp_eff2"]] = 0.42
        row[idx["linear_heat_flux_weight"]] = 1.5
        row[idx["linear_particle_flux_weight"]] = 0.3
        row[idx["mixing_length_heat_flux_proxy"]] = scale * 0.04
        rows.append(row)
    return jnp.asarray(rows)


def test_vmex_transport_objective_reduces_fake_solver_rows(monkeypatch) -> None:

    calls: list[dict[str, object]] = []
    growth_calls: list[dict[str, object]] = []
    rows = _fake_solver_rows()
    row_counter = {"i": 0}

    def fake_geom(state, static, indata, wout, **kwargs):
        calls.append(
            {"state": state, "static": static, "indata": indata, "wout": wout, **kwargs}
        )
        return _fake_geometry()

    def fake_growth(_geom, **kwargs):
        growth_calls.append(kwargs)
        value = rows[row_counter["i"], SOLVER_OBJECTIVE_NAMES.index("gamma")]
        row_counter["i"] += 1
        return value

    monkeypatch.setattr(
        transport_tables, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )
    monkeypatch.setattr(
        transport_tables, "solver_growth_rate_from_geometry", fake_growth
    )
    samples = StellaratorITGSampleSet(
        surfaces=(0.5, 0.7), alphas=(0.0,), ky_values=(0.2, 0.4)
    )
    cfg = VMEXTransportObjectiveConfig(kind="growth", sample_set=samples, ny=4)

    value = vmex_transport_objective_from_state(
        object(),
        object(),
        object(),
        SimpleNamespace(signgs=1, nfp=2, Aminor_p=1.0, phi=np.asarray([0.0, -np.pi])),
        cfg,
    )

    assert np.isclose(float(value), np.mean([0.08, 0.10, 0.12, 0.14]))
    assert calls[0]["mboz"] == 21
    assert calls[0]["nboz"] == 21
    assert [call["torflux"] for call in calls] == list(samples.surfaces)
    assert [call["selected_ky_index"] for call in growth_calls] == [1, 2, 1, 2]
    assert np.isclose(growth_calls[0]["ly"], 2.0 * np.pi / min(samples.ky_values))
    assert int(growth_calls[0]["ny"]) >= 6


def test_vmex_transport_surface_chunking_matches_unchunked_weighted_mean(
    monkeypatch,
) -> None:

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    rows = _fake_solver_rows()

    def evaluate(*, chunk_size: int) -> float:
        row_counter = {"i": 0}

        def fake_growth(_geom, **_kwargs):
            value = rows[row_counter["i"], SOLVER_OBJECTIVE_NAMES.index("gamma")]
            row_counter["i"] += 1
            return value

        monkeypatch.setattr(
            transport_tables, "solver_growth_rate_from_geometry", fake_growth
        )
        samples = StellaratorITGSampleSet(
            surfaces=(0.5, 0.7),
            alphas=(0.0,),
            ky_values=(0.2, 0.4),
            surface_weights=(3.0, 1.0),
        )
        cfg = VMEXTransportObjectiveConfig(
            kind="growth",
            sample_set=samples,
            ny=4,
            objective_transform="log1p",
            surface_chunk_size=chunk_size,
        )
        value = vmex_transport_objective_from_state(
            object(),
            object(),
            object(),
            SimpleNamespace(
                signgs=1, nfp=2, Aminor_p=1.0, phi=np.asarray([0.0, -np.pi])
            ),
            cfg,
        )
        assert row_counter["i"] == 4
        return float(value)

    monkeypatch.setattr(
        transport_tables, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )

    assert evaluate(chunk_size=1) == pytest.approx(evaluate(chunk_size=0))


def test_vmex_transport_objective_nonlinear_proxy_is_positive_and_exported(
    monkeypatch,
) -> None:

    scale = {"value": 1.0}

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    def fake_growth(_geom, **_kwargs):
        return jnp.asarray(0.1 * scale["value"])

    monkeypatch.setattr(
        transport_tables, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )
    monkeypatch.setattr(
        transport_tables, "solver_growth_rate_from_geometry", fake_growth
    )
    samples = StellaratorITGSampleSet(
        surfaces=(0.5, 0.7), alphas=(0.0,), ky_values=(0.2, 0.4)
    )
    cfg = VMEXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux", sample_set=samples
    )

    low = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), cfg
    )
    scale["value"] = 2.0
    high = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), cfg
    )

    assert gkx.VMEXTransportObjectiveConfig is VMEXTransportObjectiveConfig
    assert gkx.VMEXGKXTransportObjective is VMEXGKXTransportObjective
    assert float(low) > 0.0
    assert float(high) > float(low)


def test_vmex_transport_objective_transform_scales_large_residuals(
    monkeypatch,
) -> None:

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    def fake_growth(_geom, **_kwargs):
        return jnp.asarray(20.0)

    monkeypatch.setattr(
        transport_tables, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )
    monkeypatch.setattr(
        transport_tables, "solver_growth_rate_from_geometry", fake_growth
    )
    samples = StellaratorITGSampleSet(surfaces=(0.5,), alphas=(0.0,), ky_values=(0.2,))
    raw_cfg = VMEXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux",
        sample_set=samples,
        objective_transform="raw",
    )
    scaled_cfg = VMEXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux",
        sample_set=samples,
        objective_transform="scaled",
        objective_scale=10.0,
    )
    log_cfg = VMEXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux",
        sample_set=samples,
        objective_transform="log1p",
        objective_scale=10.0,
    )

    raw = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), raw_cfg
    )
    scaled = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), scaled_cfg
    )
    logged = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), log_cfg
    )

    assert float(raw) > 1.0
    assert float(scaled) == pytest.approx(float(raw) / 10.0)
    assert float(logged) == pytest.approx(float(jnp.log1p(jnp.abs(scaled))))
    assert float(logged) < float(scaled)


def test_vmex_transport_objective_vmec_callback_builds_reference_wout(
    monkeypatch,
) -> None:
    import gkx.objectives.vmec_transport as mod

    captured: dict[str, object] = {}

    def fake_eval(state, static, indata, wout_reference, config):
        captured["state"] = state
        captured["static"] = static
        captured["indata"] = indata
        captured["wout"] = wout_reference
        captured["config"] = config
        return jnp.asarray(0.125)

    monkeypatch.setattr(mod, "vmex_transport_objective_from_state", fake_eval)
    objective = VMEXGKXTransportObjective()
    ctx = SimpleNamespace(
        static=SimpleNamespace(cfg=SimpleNamespace(nfp=3)), indata="indata", signgs=-1
    )

    value = objective.J(ctx, "state")

    assert float(value) == 0.125
    assert captured["state"] == "state"
    assert captured["indata"] == "indata"
    assert captured["wout"].nfp == 3
    assert captured["wout"].signgs == -1


def test_vmex_transport_config_rejects_underresolved_boozer_modes() -> None:
    assert (
        VMEXTransportObjectiveConfig(kind="growth").gradient_scope
        == "eigenvalue_growth_ad"
    )
    assert (
        VMEXTransportObjectiveConfig(kind="quasilinear_flux").gradient_scope
        == "eigenvalue_growth_ad_with_geometry_transport_weights"
    )
    try:
        VMEXTransportObjectiveConfig(mboz=12, nboz=21)
    except ValueError as exc:
        assert "at least 21" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("underresolved Boozer mode count should fail")
    with pytest.raises(ValueError, match="objective_scale"):
        VMEXTransportObjectiveConfig(objective_scale=0.0)
    with pytest.raises(ValueError, match="objective transform"):
        VMEXTransportObjectiveConfig(objective_transform="bad")  # type: ignore[arg-type]


def test_vmex_transport_objective_pins_imported_backend_paths(
    monkeypatch, tmp_path
) -> None:

    vmec_root = tmp_path / "vmex_repo"
    vmec_pkg = vmec_root / "vmex"
    vmec_pkg.mkdir(parents=True)
    vmec_file = vmec_pkg / "__init__.py"
    vmec_file.write_text("", encoding="utf-8")

    booz_root = tmp_path / "booz_xform_jax_repo" / "src"
    booz_pkg = booz_root / "booz_xform_jax"
    booz_pkg.mkdir(parents=True)
    booz_file = booz_pkg / "__init__.py"
    booz_file.write_text("", encoding="utf-8")

    vmec_module = ModuleType("vmex")
    vmec_module.__file__ = str(vmec_file)
    booz_module = ModuleType("booz_xform_jax")
    booz_module.__file__ = str(booz_file)
    monkeypatch.setitem(sys.modules, "vmex", vmec_module)
    monkeypatch.setitem(sys.modules, "booz_xform_jax", booz_module)
    monkeypatch.delenv("GKX_VMEX_PATH", raising=False)
    monkeypatch.delenv("VMEX_PATH", raising=False)
    monkeypatch.delenv("GKX_BOOZ_XFORM_JAX_PATH", raising=False)
    monkeypatch.delenv("BOOZ_XFORM_JAX_PATH", raising=False)

    transport_config._pin_current_optional_backend_paths()

    assert str(vmec_root) == transport_config.os.environ["GKX_VMEX_PATH"]
    assert str(booz_root) == transport_config.os.environ["GKX_BOOZ_XFORM_JAX_PATH"]


def test_module_search_root_handles_paths_and_missing_modules(
    monkeypatch, tmp_path
) -> None:

    namespace_root = tmp_path / "namespace_backend"
    namespace_root.mkdir()
    namespace_module = ModuleType("namespace_backend")
    namespace_module.__path__ = [str(namespace_root)]

    missing_path_module = ModuleType("missing_path_backend")
    missing_path_module.__path__ = [str(tmp_path / "does_not_exist")]

    no_path_module = ModuleType("no_path_backend")

    monkeypatch.setitem(sys.modules, "namespace_backend", namespace_module)
    monkeypatch.setitem(sys.modules, "missing_path_backend", missing_path_module)
    monkeypatch.setitem(sys.modules, "no_path_backend", no_path_module)

    assert transport_config._module_search_root(
        "namespace_backend"
    ) == namespace_root.resolve(strict=False)
    assert transport_config._module_search_root("missing_path_backend") is None
    assert transport_config._module_search_root("no_path_backend") is None
    assert transport_config._module_search_root("gkx_missing_backend_for_test") is None


def test_pin_current_optional_backend_paths_respects_explicit_environment(
    monkeypatch,
) -> None:

    def unexpected_search(module_name: str):
        raise AssertionError(f"backend search should be skipped for {module_name}")

    monkeypatch.setattr(transport_config, "_module_search_root", unexpected_search)
    monkeypatch.delenv("GKX_VMEX_PATH", raising=False)
    monkeypatch.setenv("VMEX_PATH", "/explicit/vmec-jax")
    monkeypatch.setenv("GKX_BOOZ_XFORM_JAX_PATH", "/explicit/booz-xform-jax")
    monkeypatch.delenv("BOOZ_XFORM_JAX_PATH", raising=False)

    transport_config._pin_current_optional_backend_paths()

    assert "GKX_VMEX_PATH" not in transport_config.os.environ
    assert transport_config.os.environ["VMEX_PATH"] == "/explicit/vmec-jax"
    assert (
        transport_config.os.environ["GKX_BOOZ_XFORM_JAX_PATH"]
        == "/explicit/booz-xform-jax"
    )


def test_static_grid_options_maps_integer_ky_multiples_to_solver_grid() -> None:

    options = transport_tables._static_grid_options_from_ky_values(
        (0.15, 0.45), min_ny=12
    )

    assert options["ky_base"] == pytest.approx(0.15)
    assert options["ly"] == pytest.approx(2.0 * np.pi / 0.15)
    assert options["ny"] == 12
    assert options["selected_ky_indices"] == (1, 3)


@pytest.mark.parametrize(
    ("ky_values", "message"),
    (
        ((), "finite non-empty vector"),
        ((0.2, np.nan), "finite non-empty vector"),
        ((0.0,), "positive"),
        ((0.2, 0.31), "integer multiples"),
        ((0.2, 0.2), "duplicate selected ky indices"),
    ),
)
def test_static_grid_options_rejects_invalid_ky_values(
    ky_values: tuple[float, ...],
    message: str,
) -> None:

    with pytest.raises(ValueError, match=message):
        transport_tables._static_grid_options_from_ky_values(ky_values, min_ny=3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"kind": "invalid"}, "unknown VMEC-JAX transport objective kind"),
        ({"ntheta": 3}, "ntheta must be >= 4"),
        ({"n_laguerre": 0}, "n_laguerre and n_hermite must be positive"),
        ({"ny": 2}, "nx must be positive and ny must be at least 3"),
        ({"nonlinear_csat": 0.0}, "nonlinear_csat must be positive"),
        ({"surface_chunk_size": -1}, "surface_chunk_size must be non-negative"),
        (
            {
                "sample_set": StellaratorITGSampleSet(reduction="max"),
                "surface_chunk_size": 1,
            },
            "surface_chunk_size currently supports only mean or weighted_mean reductions",
        ),
    ),
)
def test_vmex_transport_config_rejects_invalid_edges(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        VMEXTransportObjectiveConfig(**kwargs)


def test_vmex_transport_config_objective_options_filter_none_values() -> None:
    default_options = VMEXTransportObjectiveConfig().objective_options()
    configured_options = VMEXTransportObjectiveConfig(
        reference_length=2.5,
        reference_b=0.7,
        validate_finite=False,
    ).objective_options()

    assert "reference_length" not in default_options
    assert "reference_b" not in default_options
    assert configured_options["reference_length"] == 2.5
    assert configured_options["reference_b"] == 0.7
    assert configured_options["validate_finite"] is False


def test_geometry_transport_weights_use_safe_defaults_for_minimal_geometry() -> None:

    theta = jnp.linspace(-jnp.pi, jnp.pi, 6, endpoint=False)
    kperp, heat_weight, particle_weight = transport_tables._geometry_transport_weights(
        SimpleNamespace(theta=theta),
        selected_ky_index=2,
        ly=5.0,
    )

    assert np.isfinite(float(kperp))
    assert np.isfinite(float(heat_weight))
    assert np.isfinite(float(particle_weight))
    assert float(kperp) > 0.0
    assert float(heat_weight) > 0.0
    assert float(particle_weight) == pytest.approx(0.25 * float(heat_weight))


def test_transport_feature_table_rejects_empty_sample_rows() -> None:

    config = SimpleNamespace(
        sample_set=SimpleNamespace(surfaces=(), alphas=(0.0,), ky_values=(0.2,)),
        kind="growth",
    )

    with pytest.raises(RuntimeError, match="produced no sample rows"):
        transport_tables._transport_feature_table_from_state(
            "state",
            "static",
            "indata",
            object(),
            config,
            {"selected_ky_indices": (1,), "ny": 4, "ly": 2.0 * np.pi / 0.2},
        )


def test_quasilinear_flux_uses_geometry_transport_weights(monkeypatch) -> None:

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    def fake_growth(_geom, **_kwargs):
        return jnp.asarray(0.2)

    monkeypatch.setattr(
        transport_tables, "flux_tube_geometry_from_vmec_boozer_state", fake_geom
    )
    monkeypatch.setattr(
        transport_tables, "solver_growth_rate_from_geometry", fake_growth
    )
    samples = StellaratorITGSampleSet(surfaces=(0.5,), alphas=(0.0,), ky_values=(0.2,))
    cfg = VMEXTransportObjectiveConfig(kind="quasilinear_flux", sample_set=samples)

    value = vmex_transport_objective_from_state(
        "state", "static", "indata", object(), cfg
    )

    assert float(value) > 0.0


# ---- from test_vmex_qa_transport_optimization.py ----


ROOT = REPO_ROOT
EXAMPLES = ROOT / "examples" / "10_vmex_optimization"
QA_SCRIPT = EXAMPLES / "run.py"
TRANSPORT_SUMMARY = ROOT / "docs" / "_static" / "qa_transport_summary.csv"
TRANSPORT_TRACES = ROOT / "docs" / "_static" / "qa_transport_traces.csv"
TRANSPORT_TIMESERIES = ROOT / "docs" / "_static" / "qa_transport_nominal_timeseries.csv"


def _transport_summary() -> dict[str, dict[str, float]]:
    with TRANSPORT_SUMMARY.open(encoding="utf-8", newline="") as stream:
        return {
            row["case"]: {
                key: float(value) for key, value in row.items() if key != "case"
            }
            for row in csv.DictReader(stream)
        }


def test_qa_transport_stationarity_gate_is_per_trace(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_qa_transport_figures")
    time = np.linspace(1100.0, 1500.0, 401)
    drift = 10.0 + 3.0 * (time - time[0]) / (time[-1] - time[0])
    path = tmp_path / "nominal_baseline_seed000.npz"
    np.savez(path, time=time, heat_flux=drift, elapsed_seconds=1.0)

    report = mod.trace_stats(path)

    assert abs(float(report["trend_percent"])) > mod.MAX_FINAL_DRIFT_PERCENT
    assert report["stationary"] == 0


def test_vmex_style_qa_script_appends_physical_autodiff_transport() -> None:
    text = QA_SCRIPT.read_text(encoding="utf-8")

    py_compile.compile(str(QA_SCRIPT), doraise=True)
    assert "argparse" not in text
    assert "MAX_MODES, MAX_NFEV = [1, 2, 3, 4, 5]" in text
    assert "SEED_PERTURBATION = 0.01" in text
    assert "ASPECT_TARGET, IOTA_TARGET = 6.0, 0.42" in text
    assert "am=np.zeros_like(inp.am), pres_scale=0.0" in text
    assert "A_OVER_LT, A_OVER_LN = 3.0, 1.0" in text
    assert "kpar_scale=local_geometry.gradpar_value" in text
    assert "p_hyper_m=float(min(20, max(NM // 2, 1)))" in text
    assert "gkx.integrate_nonlinear(" in text
    assert "gkx.nonlinear_heat_flux_window(" in text
    assert 'implicit_jacobian_method="auto"' in text
    assert "objective_function_terms = [" in text
    assert "(qs, 0.0, QA_PRIORITY)," in text
    assert "(opt.aspect_ratio, ASPECT_TARGET, ASPECT_PRIORITY)," in text
    assert "(opt.mean_iota, IOTA_TARGET, IOTA_PRIORITY)," in text
    assert "(turbulent_transport, 0.0, transport_weight)," in text
    assert "result = least_squares(" in text
    assert "def report(label, local_equilibrium):" in text


def test_docs_name_the_single_qa_autodiff_script() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "stellarator_optimization.rst",
        EXAMPLES / "README.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert "QA_optimization.py" in text, path

    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert "exact discrete differentiation" in re.sub(r"\s+", " ", examples_readme)


def test_docs_scope_vmex_transport_optimizer_claims() -> None:
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "stellarator_optimization.rst",
        EXAMPLES / "README.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        normalized = re.sub(r"\s+", " ", text)
        assert "transport" in text, path
        assert "nonlinear" in text, path
        assert "post-saturation" in normalized, path


def test_readme_qa_figures_and_reproduction_inputs_are_checked_in() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "stellarator_optimization.rst").read_text(encoding="utf-8")
    figures = (
        "nonlinear_autodiff_validation.png",
        "qa_transport_equilibria.png",
        "qa_transport_reduction.svg",
    )
    for filename in figures:
        path = ROOT / "docs" / "_static" / filename
        assert f"docs/_static/{filename}" in readme
        assert path.stat().st_size > 0
    equilibria = ROOT / "docs" / "_static" / "qa_transport_equilibria.png"
    assert equilibria.stat().st_size < 100_000
    assert equilibria.read_bytes()[25] == 3  # indexed-color PNG

    for filename in (
        "input.qa_transport_baseline",
        "input.qa_transport_candidate",
    ):
        assert (EXAMPLES / filename).is_file()
        assert filename in docs
    for script in (
        ROOT / "scripts" / "campaigns" / "qa_transport_validation.py",
        ROOT / "scripts" / "artifacts" / "build_qa_transport_figures.py",
    ):
        py_compile.compile(str(script), doraise=True)
        assert script.name in docs

    with TRANSPORT_TIMESERIES.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    time = np.asarray([float(row["time"]) for row in rows])
    values = np.asarray(
        [
            [
                float(row["baseline_mean"]),
                float(row["baseline_sem"]),
                float(row["candidate_mean"]),
                float(row["candidate_sem"]),
            ]
            for row in rows
        ]
    )
    assert len(rows) == 301
    assert np.all(np.diff(time) > 0.0)
    assert time[0] < 1.0 and time[-1] > 1499.0
    assert np.all(np.isfinite(values))
    assert np.all(values[:, (1, 3)] >= 0.0)


def test_optimization_examples_document_user_customization_knobs() -> None:
    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")

    assert "QA nonlinear transport" in examples_readme
    assert "A_OVER_LT" in examples_readme
    assert "A_OVER_LN" in examples_readme
    assert "SATURATION_STEPS" in examples_readme
    assert "WINDOW_STEPS" in examples_readme
    assert "objective_function_terms" in examples_readme
    assert "least_squares" in examples_readme


def test_readme_uses_solved_vmec_qa_geometry_not_reduced_surface_panel() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "stellarator_optimization.rst").read_text(encoding="utf-8")
    manuscript = (ROOT / "docs" / "manuscript_figures.rst").read_text(encoding="utf-8")
    normalized_readme = re.sub(r"\s+", " ", readme)

    # The README presents the production method, not a reduced proxy panel.
    assert "docs/_static/qa_itg_optimization_summary_panel.png" not in readme
    assert "docs/_static/vmex_qa_solved_boundary_boozer_panel.png" not in readme
    assert "docs/_static/stellarator_itg_optimization_comparison.png" not in readme
    assert "docs/_static/stellarator_itg_optimization_uq.png" not in readme
    assert "independent matched runs validate" not in normalized_readme.lower()
    assert "preliminary 12.26% reduction" in normalized_readme.lower()
    assert "not statistically resolved" in normalized_readme.lower()
    assert "replicated" in normalized_readme.lower()
    assert "post-saturation" in normalized_readme.lower()

    assert "QA_optimization.py" in docs
    assert (
        "nonlinear_heat_flux_window" not in docs or "physical heat-flux window" in docs
    )
    assert ".. figure:: _static/stellarator_itg_optimization_comparison.png" not in docs
    assert "screening diagnostics" in docs
    assert (
        "current artifact bases: ``docs/_static/stellarator_itg_optimization_comparison.png``"
        not in manuscript
    )
    assert "is not a solved-geometry optimization figure" in manuscript
    assert (
        "production QA optimization examples are the VMEC-JAX-style scripts"
        in manuscript
    )


def test_reduced_surface_comparison_is_not_current_primary_optimization_figure() -> (
    None
):
    release_contract = (
        ROOT / "benchmarks" / "references" / "gkx_1_7_release_contract.json"
    ).read_text(encoding="utf-8")
    examples_readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "stellarator_optimization.rst").read_text(encoding="utf-8")

    reduced_png = '"docs/_static/stellarator_itg_optimization_comparison.png"'
    assert reduced_png not in release_contract
    assert "stellarator_itg_growth_optimization.py" not in examples_readme
    assert "reduced_stellarator_itg" not in examples_readme
    assert "screening diagnostics" in re.sub(r"\s+", " ", docs)


def test_matched_qa_transport_evidence_fails_closed() -> None:
    """Keep the preliminary campaign reproducible without promoting it."""
    rows = _transport_summary()
    assert set(rows) == {
        "nominal",
        "dt04",
        "dt025",
        "perp12",
        "perp20",
        "perp24",
        "perp24long",
        "z16",
        "z32",
        "v36",
        "v612",
    }
    assert all(np.isfinite(value) for row in rows.values() for value in row.values())
    with TRANSPORT_TRACES.open(encoding="utf-8", newline="") as stream:
        traces = list(csv.DictReader(stream))
    for row in rows.values():
        assert row["p_hyper_m"] == min(20, max(int(row["nm"]) // 2, 1))
    for case, row in rows.items():
        case_traces = [trace for trace in traces if trace["case"] == case]
        pairs = int(row["pairs"])
        assert len(case_traces) == 2 * pairs
        for design in ("baseline", "candidate"):
            seeds = {
                int(trace["seed"]) for trace in case_traces if trace["design"] == design
            }
            assert seeds == set(range(pairs))
        assert all(float(trace["tau"]) > 0.0 for trace in case_traces)
        assert all(float(trace["neff"]) > 0.0 for trace in case_traces)
        assert all(
            int(trace["stationary"]) == int(abs(float(trace["trend_percent"])) <= 20.0)
            for trace in case_traces
        )
        assert row["stationary_traces"] == sum(
            int(trace["stationary"]) for trace in case_traces
        )
        assert (
            row["nonstationary_traces"] == len(case_traces) - row["stationary_traces"]
        )
        assert row["resolved_spectra_available"] == 0.0
        assert row["promotion_ready"] == 0.0

    nominal = rows["nominal"]
    assert nominal["pairs"] == 24
    assert nominal["positive_pairs"] == nominal["pairs"]
    assert nominal["ci95_low_percent"] > 0.0
    assert nominal["minimum_window_in_tau"] > 10.0
    assert nominal["stationary_traces"] == 44
    assert nominal["nonstationary_traces"] == 4
    assert nominal["all_traces_stationary"] == 0.0
    assert rows["z32"]["all_traces_stationary"] == 1.0


def test_solved_wout_candidate_gate_passes_valid_qa_branch() -> None:
    result = SimpleNamespace(
        history={
            "aspect_final": 5.999233,
            "iota_final": 0.427011,
            "qs_final": 2.604013e-2,
        },
    )

    report = build_solved_vmec_candidate_gate(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.410131, 0.414]),
            np.asarray([0.410706, 0.414]),
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is True
    assert report["checks"]["iota_profile"]["passed"] is True
    json.dumps(report, allow_nan=False)


def test_solved_wout_candidate_gate_rejects_transport_branch_that_breaks_constraints() -> (
    None
):
    result = SimpleNamespace(
        history={
            "aspect_final": 5.996817,
            "iota_final": 0.425028,
            "qs_final": 1.091236e-1,
        },
    )

    report = build_solved_vmec_candidate_gate(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.402043, 0.414]),
            np.asarray([0.402493, 0.414]),
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is False
    assert report["checks"]["iota_profile"]["passed"] is False
    assert "do not promote" in report["next_action"]
    json.dumps(report, allow_nan=False)
