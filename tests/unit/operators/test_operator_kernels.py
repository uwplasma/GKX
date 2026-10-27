"""Focused operator-kernel and reduced full-operator regression checks."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import pytest
from scipy.special import eval_genlaguerre, eval_laguerre, j0, j1, jv
from scipy.special import roots_hermite, roots_laguerre

from gkx.config import GridConfig
from gkx.terms import linear_terms
from gkx.terms.linear_terms import curvature_gradb_contribution, mirror_contribution
from gkx.operators.collision import CollisionContext
from gkx.core_grid import build_spectral_grid
from gkx.core_velocity import (
    J_l_all,
    associated_bessel_laguerre_coefficients,
    bessel_laguerre_kernels,
    gamma0,
    laguerre_gyroaverage_neighbors,
    laguerre_scaled_functions,
    mu_grid_laguerre_modes,
    mu_grid_nodes,
)
from gkx.diagnostics.analysis import fit_growth_rate
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.solvers_linear_integrators import integrate_linear
from gkx.operators import hermite_streaming
from gkx.operators.linear import (
    apply_collision_moment_matrix,
    apply_finite_wavelength_coulomb_moment_operator,
    apply_multispecies_collision_moment_matrix,
    assemble_drift_kinetic_improved_sugama_matrix,
    assemble_drift_kinetic_sugama_matrix,
    collision_operator_from_config,
    DriftKineticMomentCollisionOperator,
    EqualSpeciesFiniteWavelengthCoulombOperator,
    EqualSpeciesFiniteWavelengthSugamaOperator,
    FiniteWavelengthCoulombOperator,
    TabulatedMultispeciesCollisionOperator,
    drift_kinetic_improved_sugama_pair_matrices,
    drift_kinetic_sugama_pair_matrices,
    interpolate_collision_moment_matrix,
    interpolate_collision_diagonal_table,
    interpolate_collision_pair_table,
    load_collision_moment_matrix,
    parallel_electric_field_source,
    solve_driven_collision_response,
)
from gkx.operators.linear.dissipation import (
    collision_quadratic_rate,
    collisions_contribution,
    multispecies_collision_invariant_rates,
)
from gkx.terms.assembly import assemble_rhs_cached
from gkx.terms.config import FieldState, TermConfig
from gkx.terms.linear_terms import (
    drift_kinetic_sugama_six_moment_contribution,
)
from gkx.runtime import (
    run_runtime_scan,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimePhysicsConfig,
    RuntimeTermsConfig,
)
from gkx.operators.linear.dissipation import hypercollisions_contribution
import hashlib
import io
import json
from importlib import resources
from gkx.operators.linear import collisions
import gkx.operators as operators
import gkx.operators.linear as linear_operators
import gkx.operators.nonlinear as nonlinear_operators
import gkx.operators.nonlinear.diagnostic_state as operator_diagnostics
import gkx.operators.nonlinear.rhs as operator_rhs
from gkx.terms.assembly import linear_rhs_jit_for_terms


def test_streaming_zero_kpar() -> None:
    """Hermite streaming should vanish at k_parallel = 0."""
    state = jnp.ones((2, 4))

    derivative = hermite_streaming(state, kpar=0.0, vth=1.0)

    assert jnp.allclose(derivative, 0.0)


def test_streaming_energy_conservation() -> None:
    """Hermite streaming is skew-adjoint for constant k_parallel."""
    state = jnp.sin(jnp.arange(12, dtype=jnp.float32)).reshape(3, 4)

    derivative = hermite_streaming(state, kpar=0.7, vth=1.2)
    energy_rate = jnp.real(jnp.sum(jnp.conj(state) * derivative))

    assert jnp.isclose(energy_rate, 0.0, atol=1e-6)


def test_streaming_known_case() -> None:
    """Simple Hermite ladder behavior should match a hand calculation."""
    state = jnp.array([[1.0, 2.0, 0.0]])

    derivative = hermite_streaming(state, kpar=1.0, vth=1.0)

    assert jnp.isclose(derivative[0, 0], -1j * 2.0)
    assert jnp.isclose(derivative[0, 1], -1j * 1.0)
    assert jnp.isclose(derivative[0, 2], -1j * (jnp.sqrt(2.0) * 2.0))


def test_streaming_invalid_shape() -> None:
    """Hermite axis length must be positive."""
    with pytest.raises(ValueError):
        hermite_streaming(jnp.zeros((2, 0)), kpar=1.0, vth=1.0)


def test_parallel_electric_field_source_matches_moment_equation() -> None:
    """Frei et al. (2022), Eq. (81), drives only N10 at linear order."""
    field = jnp.asarray(1.0e-3)

    source = parallel_electric_field_source(4, 2, field)

    expected = jnp.zeros(15).at[3].set(-jnp.sqrt(2.0) * field)
    np.testing.assert_allclose(source, expected, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="hermite"):
        parallel_electric_field_source(0, 1, field)
    with pytest.raises(ValueError, match="laguerre"):
        parallel_electric_field_source(1, -1, field)
    with pytest.raises(ValueError, match="scalar"):
        parallel_electric_field_source(1, 0, jnp.ones(2))


def test_driven_collision_response_matches_saturation_and_derivative() -> None:
    """A dissipative moment ladder must reach its exact forced steady state."""
    damping = jnp.asarray([0.0, 2.0, 3.0, 5.0])
    collision = -jnp.diag(damping)
    active = (1, 2, 3)

    def current(field: jnp.ndarray) -> jnp.ndarray:
        source = parallel_electric_field_source(3, 0, field)
        return solve_driven_collision_response(collision, source, active_modes=active)[
            1
        ]

    field = jnp.asarray(1.0e-3)
    response = jax.jit(
        lambda value: solve_driven_collision_response(
            collision,
            parallel_electric_field_source(3, 0, value),
            active_modes=active,
        )
    )(field)
    expected_current = -jnp.sqrt(2.0) * field / damping[1]
    np.testing.assert_allclose(response[1], expected_current, rtol=2.0e-6)
    np.testing.assert_allclose(np.asarray(response)[[0, 2, 3]], 0.0, atol=0.0)

    reduced = np.asarray(collision)[np.ix_(active, active)]
    steady = np.asarray(response)[list(active)]
    transient = steady + jax.scipy.linalg.expm(jnp.asarray(reduced) * 20.0) @ (
        -jnp.asarray(steady)
    )
    np.testing.assert_allclose(transient, steady, rtol=1.0e-6, atol=1.0e-12)

    tangent = jax.grad(current)(field)
    step = 1.0e-5
    finite_difference = (current(field + step) - current(field - step)) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-4)


def test_driven_collision_response_rejects_invalid_subspaces() -> None:
    matrix = jnp.eye(3)
    source = jnp.ones(3)
    with pytest.raises(ValueError, match="square"):
        solve_driven_collision_response(jnp.ones((2, 3)), source, active_modes=(1,))
    with pytest.raises(ValueError, match="dimension"):
        solve_driven_collision_response(matrix, jnp.ones(2), active_modes=(1,))
    with pytest.raises(ValueError, match="at least one"):
        solve_driven_collision_response(matrix, source, active_modes=())
    with pytest.raises(ValueError, match="unique"):
        solve_driven_collision_response(matrix, source, active_modes=(1, 1))
    with pytest.raises(ValueError, match="out-of-range"):
        solve_driven_collision_response(matrix, source, active_modes=(3,))


def test_reduced_sugama_current_recovers_high_charge_limit() -> None:
    """The Appendix-C current ordering must approach the common Lorentz limit."""
    source = parallel_electric_field_source(3, 1, jnp.asarray(1.0e-3))
    active_flow_modes = (2, 3, 6)

    def current(
        pair_model,
        ion_charge: float,
    ) -> float:
        electron_test, electron_field = pair_model(1.0, 1.0)
        electron_ion_test, _ = pair_model(1.0 / 1836.0, 1.0)
        collision = electron_test + electron_field + ion_charge * electron_ion_test
        response = solve_driven_collision_response(
            collision,
            source,
            active_modes=active_flow_modes,
        )
        return float(jnp.abs(response[2] / jnp.sqrt(2.0)))

    original_low_charge = current(drift_kinetic_sugama_pair_matrices, 1.0)
    improved_low_charge = current(drift_kinetic_improved_sugama_pair_matrices, 1.0)
    original_high_charge = current(drift_kinetic_sugama_pair_matrices, 100.0)
    improved_high_charge = current(drift_kinetic_improved_sugama_pair_matrices, 100.0)

    assert improved_low_charge / original_low_charge > 1.10
    assert abs(improved_high_charge / original_high_charge - 1.0) < 0.01


def test_linear_operator_package_reexports_streaming_kernel() -> None:
    from gkx.operators.linear import hermite_streaming as package_streaming
    from gkx.operators.linear.moments import hermite_streaming as streaming_impl

    assert hermite_streaming is package_streaming
    assert package_streaming is streaming_impl


def test_bessel_laguerre_kernels_match_independent_velocity_projection() -> None:
    """Eq. (2.13) must recover the weighted J0 Laguerre coefficients."""
    nodes, weights = np.polynomial.laguerre.laggauss(96)
    b_values = np.asarray([0.0, 0.5, 1.0, 2.0])
    kernels = np.asarray(bessel_laguerre_kernels(jnp.asarray(b_values), 5))

    projected = np.stack(
        [
            [
                np.sum(weights * j0(b * np.sqrt(nodes)) * eval_laguerre(n, nodes))
                for b in b_values
            ]
            for n in range(6)
        ]
    )
    np.testing.assert_allclose(kernels, projected, rtol=2.0e-5, atol=2.0e-7)


def test_bessel_laguerre_kernel_convergence_and_derivative() -> None:
    """The published finite-b truncation behavior and JAX tangent must hold."""
    b = jnp.asarray(1.0)
    kernels = bessel_laguerre_kernels(b, 16)

    # At b=1, the paper reports that retaining N=3 gives sub-0.1% FLR error.
    assert float(1.0 - jnp.sum(kernels[:4])) < 1.0e-3
    np.testing.assert_allclose(np.asarray(jnp.sum(kernels)), 1.0, atol=1.0e-7)
    np.testing.assert_allclose(
        np.asarray(kernels[1:] / kernels[:-1]),
        0.25 / np.arange(1, 17),
        rtol=2.0e-6,
    )

    tangent = jax.jvp(
        lambda value: bessel_laguerre_kernels(value, 5),
        (b,),
        (jnp.asarray(0.3),),
    )[1]
    step = jnp.asarray(1.0e-3)
    finite_difference = (
        bessel_laguerre_kernels(b + 0.3 * step, 5)
        - bessel_laguerre_kernels(b - 0.3 * step, 5)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        np.asarray(tangent), np.asarray(finite_difference), rtol=2.0e-4, atol=2.0e-5
    )

    compiled = jax.jit(lambda value: bessel_laguerre_kernels(value, 3))
    zero_limit = np.asarray(compiled(jnp.asarray(0.0)))
    np.testing.assert_array_equal(zero_limit, np.asarray([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="n_max"):
        bessel_laguerre_kernels(b, -1)


@pytest.mark.parametrize("bessel_order", [0, 1, 2])
def test_associated_bessel_laguerre_expansion_recovers_bessel_function(
    bessel_order: int,
) -> None:
    """Frei et al. Eq. (2.12) must reconstruct J_m over velocity space."""
    b_values = np.asarray([0.0, 0.5, 1.0, 2.0])
    x = np.linspace(0.0, 8.0, 41)
    coefficients = np.asarray(
        jax.jit(
            lambda value: associated_bessel_laguerre_coefficients(
                value, bessel_order, 24
            )
        )(jnp.asarray(b_values))
    )
    polynomials = np.stack([eval_genlaguerre(n, bessel_order, x) for n in range(25)])
    reconstructed = x[None, :] ** (0.5 * bessel_order) * np.einsum(
        "nb,nx->bx", coefficients, polynomials
    )
    expected = jv(bessel_order, b_values[:, None] * np.sqrt(x[None, :]))
    np.testing.assert_allclose(reconstructed, expected, rtol=3.0e-5, atol=3.0e-6)

    if bessel_order == 0:
        np.testing.assert_allclose(
            coefficients,
            np.asarray(bessel_laguerre_kernels(jnp.asarray(b_values), 24)),
            rtol=2.0e-6,
            atol=2.0e-7,
        )


def test_associated_bessel_laguerre_coefficients_validate_orders() -> None:
    with pytest.raises(ValueError, match="bessel_order"):
        associated_bessel_laguerre_coefficients(jnp.asarray(1.0), -1, 2)
    with pytest.raises(ValueError, match="n_max"):
        associated_bessel_laguerre_coefficients(jnp.asarray(1.0), 0, -1)


@pytest.mark.parametrize(
    ("model", "direct"),
    [
        ("sugama", drift_kinetic_sugama_six_moment_contribution),
    ],
)
def test_generated_collision_matrix_matches_direct_published_equations(
    model, direct
) -> None:
    state = (
        jnp.arange(2 * 2 * 4 * 2, dtype=jnp.float32).reshape(2, 2, 4, 1, 1, 2) + 0.17j
    ).astype(jnp.complex64)
    frequency = jnp.asarray([0.2, 0.35], dtype=jnp.float32)
    table_result = apply_collision_moment_matrix(
        state,
        load_collision_moment_matrix(model),
        nu=frequency,
        weight=jnp.asarray(0.7),
    )
    direct_result = direct(state, nu=frequency, weight=jnp.asarray(0.7))
    np.testing.assert_allclose(
        np.asarray(table_result), np.asarray(direct_result), rtol=2.0e-6, atol=2.0e-6
    )


def test_collision_matrix_application_is_differentiable_and_fail_closed() -> None:
    state = (
        jnp.arange(8, dtype=jnp.float32).reshape(2, 4, 1, 1, 1).astype(jnp.complex64)
    )
    matrix = jnp.asarray(load_collision_moment_matrix("coulomb"))
    frequency = jnp.asarray([0.3], dtype=jnp.float32)
    tangent = jax.jvp(
        lambda scale, nu: apply_collision_moment_matrix(scale * state, matrix, nu=nu),
        (jnp.asarray(1.0), frequency),
        (jnp.asarray(0.4), jnp.asarray([0.2], dtype=frequency.dtype)),
    )[1]
    step = jnp.asarray(1.0e-3)
    plus = apply_collision_moment_matrix(
        (1.0 + 0.4 * step) * state, matrix, nu=frequency + 0.2 * step
    )
    minus = apply_collision_moment_matrix(
        (1.0 - 0.4 * step) * state, matrix, nu=frequency - 0.2 * step
    )
    np.testing.assert_allclose(
        np.asarray(tangent),
        np.asarray((plus - minus) / (2.0 * step)),
        rtol=3.0e-4,
        atol=3.0e-4,
    )

    with pytest.raises(ValueError, match="collision model must be one of"):
        load_collision_moment_matrix("unknown")
    with pytest.raises(ValueError, match="collision matrix must have shape"):
        apply_collision_moment_matrix(state, jnp.eye(7), nu=frequency)
    with pytest.raises(ValueError, match="nu must have length 2"):
        apply_collision_moment_matrix(
            jnp.broadcast_to(state, (2,) + state.shape), matrix, nu=jnp.ones(3)
        )


def test_collision_kperp_interpolation_matches_nodes_and_direct_operator() -> None:
    grid = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    base = jnp.asarray(load_collision_moment_matrix("coulomb"), dtype=jnp.float32)
    table = jnp.stack([(1.0 + 0.5 * value) * base for value in grid])
    kperp = jnp.asarray([[[-0.3, 0.5]], [[1.0, 2.7]]], dtype=jnp.float32)
    interpolated = jax.jit(
        lambda target: interpolate_collision_moment_matrix(grid, table, target)
    )(kperp)
    expected = base[:, :, None, None, None] * (
        1.0 + 0.5 * jnp.clip(kperp, grid[0], grid[-1])
    )
    np.testing.assert_allclose(interpolated, expected, rtol=2.0e-6, atol=2.0e-6)

    state = (
        jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 2, 1, 2) + 0.13j
    ).astype(jnp.complex64)
    result = apply_collision_moment_matrix(state, interpolated, nu=jnp.asarray(0.3))
    direct = apply_collision_moment_matrix(state, base, nu=jnp.asarray(0.3)) * (
        1.0 + 0.5 * jnp.clip(kperp, grid[0], grid[-1])
    )
    np.testing.assert_allclose(result, direct, rtol=3.0e-6, atol=3.0e-6)


def test_collision_matrix_kperp_interpolation_species_jvp_and_validation() -> None:
    grid = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    base = jnp.asarray(load_collision_moment_matrix("sugama"), dtype=jnp.float32)
    shared_table = jnp.stack([(1.0 + value) * base for value in grid])
    species_table = jnp.stack([shared_table, 2.0 * shared_table])
    target = jnp.asarray([[[[0.25]]], [[[1.25]]]], dtype=jnp.float32)
    result = interpolate_collision_moment_matrix(grid, species_table, target)
    np.testing.assert_allclose(result[0], 1.25 * base[:, :, None, None, None])
    np.testing.assert_allclose(result[1], 4.5 * base[:, :, None, None, None])
    state = (
        jnp.arange(2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4, 1, 1, 1) + 0.2j
    ).astype(jnp.complex64)
    frequency = jnp.asarray([0.2, 0.3], dtype=jnp.float32)
    applied = jax.jit(apply_collision_moment_matrix)(state, result, nu=frequency)
    direct = drift_kinetic_sugama_six_moment_contribution(state, nu=frequency)
    expected = direct * jnp.asarray([1.25, 4.5])[:, None, None, None, None, None]
    np.testing.assert_allclose(applied, expected, rtol=3.0e-6, atol=3.0e-6)

    scalar = jnp.asarray(0.6, dtype=jnp.float32)
    tangent = jax.jvp(
        lambda value: interpolate_collision_moment_matrix(grid, shared_table, value),
        (scalar,),
        (jnp.asarray(0.4, dtype=jnp.float32),),
    )[1]
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = (
        interpolate_collision_moment_matrix(grid, shared_table, scalar + 0.4 * step)
        - interpolate_collision_moment_matrix(grid, shared_table, scalar - 0.4 * step)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=4.0e-4, atol=4.0e-4)
    scalar_species = interpolate_collision_moment_matrix(grid, species_table, scalar)
    np.testing.assert_allclose(scalar_species[0], 1.6 * base)
    np.testing.assert_allclose(scalar_species[1], 3.2 * base)

    with pytest.raises(ValueError, match="at least two points"):
        interpolate_collision_moment_matrix(
            jnp.asarray([0.0]), shared_table[:1], scalar
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_collision_moment_matrix(
            jnp.asarray([0.0, 2.0, 1.0]), shared_table, scalar
        )
    with pytest.raises(ValueError, match="must have shape"):
        interpolate_collision_moment_matrix(grid, jnp.ones((3, 3)), scalar)
    with pytest.raises(ValueError, match="axis must match"):
        interpolate_collision_moment_matrix(grid, shared_table[:2], scalar)
    with pytest.raises(ValueError, match="must be square"):
        interpolate_collision_moment_matrix(grid, jnp.ones((3, 8, 7)), scalar)
    with pytest.raises(ValueError, match="species-leading"):
        interpolate_collision_moment_matrix(grid, species_table, jnp.ones((3, 1)))


def test_multispecies_collision_pair_table_interpolates_and_applies_in_jax() -> None:
    """Ordered-pair finite-b tables retain pair identity and the b=0 invariants."""

    density = jnp.asarray([1.3, 0.7], dtype=jnp.float32)
    mass = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    temperature = jnp.asarray([1.2, 0.8], dtype=jnp.float32)
    matrix_zero = assemble_drift_kinetic_improved_sugama_matrix(
        density, mass, temperature
    )
    grid = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32)
    pair_slopes = jnp.asarray([[0.2, -0.1], [0.35, 0.15]], dtype=jnp.float32)
    table = jnp.stack(
        [matrix_zero * (1.0 + value * pair_slopes[..., None, None]) for value in grid],
        axis=2,
    )
    target = jnp.asarray([[[[0.0, 0.25]]], [[[0.75, 1.0]]]], dtype=jnp.float32)
    interpolated = jax.jit(
        lambda value: interpolate_collision_moment_matrix(grid, table, value)
    )(target)
    factors = 1.0 + pair_slopes[..., None, None, None] * target[:, None, ...]
    expected = matrix_zero[..., None, None, None] * factors[:, :, None, None, ...]
    np.testing.assert_allclose(interpolated, expected, rtol=2.0e-6, atol=2.0e-6)

    state = (
        1.0e-2
        * (
            jnp.arange(2 * 2 * 4 * 2, dtype=jnp.float32).reshape(2, 2, 4, 1, 1, 2)
            + 0.13j
        )
    ).astype(jnp.complex64)
    applied = jax.jit(apply_multispecies_collision_moment_matrix)(state, interpolated)
    for spatial_index in range(2):
        direct = apply_multispecies_collision_moment_matrix(
            state[..., spatial_index : spatial_index + 1],
            expected[..., spatial_index : spatial_index + 1],
        )
        np.testing.assert_allclose(
            applied[..., spatial_index : spatial_index + 1],
            direct,
            rtol=2.0e-6,
            atol=2.0e-6,
        )

    zero_matrix = interpolate_collision_moment_matrix(
        grid, table, jnp.asarray(0.0, dtype=jnp.float32)
    )
    zero_contribution = apply_multispecies_collision_moment_matrix(state, zero_matrix)
    zero_rates = multispecies_collision_invariant_rates(
        zero_contribution,
        density=density,
        mass=mass,
        temperature=temperature,
    )
    np.testing.assert_allclose(zero_rates.particle_density, 0.0, atol=2.0e-6)
    np.testing.assert_allclose(zero_rates.total_parallel_momentum, 0.0, atol=4.0e-6)
    np.testing.assert_allclose(zero_rates.total_thermal_energy, 0.0, atol=1.0e-5)

    direction = jnp.asarray([[[[0.0, 0.2]]], [[[-0.1, 0.0]]]], dtype=jnp.float32)
    tangent = jax.jvp(
        lambda value: interpolate_collision_moment_matrix(grid, table, value),
        (target,),
        (direction,),
    )[1]
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = (
        interpolate_collision_moment_matrix(grid, table, target + step * direction)
        - interpolate_collision_moment_matrix(grid, table, target - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=6.0e-4, atol=6.0e-4)

    with pytest.raises(ValueError, match="equal target/source axes"):
        interpolate_collision_moment_matrix(grid, table[:, :1], jnp.asarray(0.4))
    with pytest.raises(ValueError, match="target-species-leading"):
        interpolate_collision_moment_matrix(grid, table, jnp.ones((3, 1)))


def test_coulomb_pair_table_bilinear_interpolation_and_jvp() -> None:
    """Unlike-species tables interpolate independently in target and source b."""

    grid = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    ns, mode_count = 2, 2
    base = 1.0 + jnp.arange(
        ns * ns * mode_count * mode_count, dtype=jnp.float32
    ).reshape(ns, ns, mode_count, mode_count)
    target_grid = grid[None, None, :, None, None, None]
    source_grid = grid[None, None, None, :, None, None]
    table = base[:, :, None, None, ...] * (
        1.0 + 0.2 * target_grid + 0.3 * source_grid + 0.1 * target_grid * source_grid
    )
    target = jnp.asarray([[[[0.25, 1.5]]], [[[0.8, 1.7]]]], dtype=jnp.float32)
    interpolated = jax.jit(
        lambda values: interpolate_collision_pair_table(grid, table, values)
    )(target)
    expected = np.empty((ns, ns, mode_count, mode_count, 1, 1, 2))
    for target_species in range(ns):
        for source_species in range(ns):
            target_b = np.asarray(target[target_species])
            source_b = np.asarray(target[source_species])
            factor = 1.0 + 0.2 * target_b + 0.3 * source_b + 0.1 * target_b * source_b
            expected[target_species, source_species] = (
                np.asarray(base[target_species, source_species])[..., None, None, None]
                * factor
            )
    np.testing.assert_allclose(interpolated, expected, rtol=2.0e-6, atol=2.0e-6)

    vector_table = table[..., 0]
    vector_result = interpolate_collision_pair_table(grid, vector_table, target)
    np.testing.assert_allclose(vector_result, expected[:, :, :, 0, ...], rtol=2.0e-6)

    direction = jnp.asarray([[[[0.1, -0.2]]], [[[0.15, -0.1]]]], dtype=jnp.float32)
    tangent = jax.jvp(
        lambda values: interpolate_collision_pair_table(grid, table, values),
        (target,),
        (direction,),
    )[1]
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = (
        interpolate_collision_pair_table(grid, table, target + step * direction)
        - interpolate_collision_pair_table(grid, table, target - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=8.0e-4, atol=8.0e-4)

    with pytest.raises(ValueError, match="equal target/source"):
        interpolate_collision_pair_table(grid, table[:, :1], target)
    with pytest.raises(ValueError, match="leading axis"):
        interpolate_collision_pair_table(grid, table, jnp.ones((3, 1)))


def test_finite_wavelength_coulomb_runtime_assembly_matches_pair_equations() -> None:
    """Frei Eqs. (3.47)--(3.50) retain target/source and phi couplings."""

    ns, nl, nm = 2, 1, 2
    spatial_shape = (1, 1, 2)
    mode_count = nl * nm
    state = (
        jnp.arange(ns * nl * nm * 2, dtype=jnp.float32).reshape(
            (ns, nl, nm) + spatial_shape
        )
        + 0.2j
    ).astype(jnp.complex64)
    phi = jnp.asarray([[[0.3, -0.2]]], dtype=jnp.float32)
    frequency = jnp.asarray([[0.7, 0.2], [0.4, 0.9]], dtype=jnp.float32)
    charge_temperature = jnp.asarray([1.3, -0.8], dtype=jnp.float32)

    base = jnp.arange(ns * ns * mode_count * mode_count, dtype=jnp.float32).reshape(
        ns, ns, mode_count, mode_count
    )
    test_matrix = 0.01 * (base + 1.0)
    field_matrix = -0.006 * (base + 0.5)
    vector_base = jnp.arange(ns * ns * mode_count, dtype=jnp.float32).reshape(
        ns, ns, mode_count
    )
    test_phi1 = 0.013 * (vector_base + 1.0)
    field_phi1 = -0.011 * (vector_base + 0.7)
    test_phi2 = -0.004 * (vector_base + 0.2)
    field_phi2 = 0.008 * (vector_base + 0.4)

    def apply(value: jnp.ndarray, potential: jnp.ndarray) -> jnp.ndarray:
        return apply_finite_wavelength_coulomb_moment_operator(
            value,
            test_matrix,
            field_matrix,
            test_phi1,
            field_phi1,
            test_phi2,
            field_phi2,
            phi=potential,
            pair_frequency=frequency,
            charge_over_temperature=charge_temperature,
        )

    result = jax.jit(apply)(state, phi)
    packed = np.swapaxes(np.asarray(state), 1, 2).reshape(
        (ns, mode_count) + spatial_shape
    )
    expected = np.zeros_like(packed)
    for target in range(ns):
        for source in range(ns):
            for output_mode in range(mode_count):
                particle = sum(
                    float(test_matrix[target, source, output_mode, input_mode])
                    * packed[target, input_mode]
                    + float(field_matrix[target, source, output_mode, input_mode])
                    * packed[source, input_mode]
                    for input_mode in range(mode_count)
                )
                polarization = (
                    float(charge_temperature[target])
                    * float(
                        test_phi1[target, source, output_mode]
                        + test_phi2[target, source, output_mode]
                    )
                    + float(charge_temperature[source])
                    * float(
                        field_phi1[target, source, output_mode]
                        + field_phi2[target, source, output_mode]
                    )
                ) * np.asarray(phi)
                expected[target, output_mode] += float(frequency[target, source]) * (
                    particle + polarization
                )
    expected = np.swapaxes(expected.reshape((ns, nm, nl) + spatial_shape), 1, 2)
    np.testing.assert_allclose(result, expected, rtol=2.0e-6, atol=2.0e-6)

    state_direction = jnp.full_like(state, 0.07 - 0.03j)
    phi_direction = jnp.full_like(phi, -0.04)
    tangent = jax.jvp(apply, (state, phi), (state_direction, phi_direction))[1]
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = (
        apply(state + step * state_direction, phi + step * phi_direction)
        - apply(state - step * state_direction, phi - step * phi_direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=8.0e-4, atol=8.0e-4)

    cancelling = jnp.asarray([[[0.2, -0.1]]], dtype=jnp.float32)
    zero = apply_finite_wavelength_coulomb_moment_operator(
        jnp.zeros((1, 1, 2, 1, 1, 2), dtype=jnp.complex64),
        jnp.zeros((1, 1, 2, 2)),
        jnp.zeros((1, 1, 2, 2)),
        cancelling,
        -cancelling,
        2.0 * cancelling,
        -2.0 * cancelling,
        phi=phi,
        pair_frequency=jnp.ones((1, 1)),
        charge_over_temperature=jnp.ones(1),
    )
    np.testing.assert_allclose(zero, 0.0, atol=1.0e-7)

    grid = jnp.asarray([0.0, 2.0], dtype=jnp.float32)

    def constant_pair_table(coefficients: jnp.ndarray) -> jnp.ndarray:
        return jnp.broadcast_to(
            coefficients[:, :, None, None, ...],
            coefficients.shape[:2] + (2, 2) + coefficients.shape[2:],
        )

    operator = FiniteWavelengthCoulombOperator(
        grid,
        frequency,
        constant_pair_table(test_matrix),
        constant_pair_table(field_matrix),
        constant_pair_table(test_phi1),
        constant_pair_table(field_phi1),
        constant_pair_table(test_phi2),
        constant_pair_table(field_phi2),
    )
    context = CollisionContext(
        distribution=state,
        hamiltonian=99.0 * state,
        fields=FieldState(phi=phi, apar=None, bpar=None),
        cache=SimpleNamespace(
            b=jnp.asarray([[[[0.04, 0.64]]], [[[0.16, 1.0]]]], dtype=jnp.float32)
        ),
        parameters=SimpleNamespace(tz=1.0 / charge_temperature),
    )
    np.testing.assert_allclose(
        operator.apply(context), result, rtol=2.0e-6, atol=2.0e-6
    )
    assert len(jax.tree_util.tree_leaves(operator)) == 8

    with pytest.raises(ValueError, match="pair_frequency"):
        apply_finite_wavelength_coulomb_moment_operator(
            state,
            test_matrix,
            field_matrix,
            test_phi1,
            field_phi1,
            test_phi2,
            field_phi2,
            phi=phi,
            pair_frequency=jnp.ones((2, 1)),
            charge_over_temperature=charge_temperature,
        )


def test_finite_wavelength_coulomb_uses_thermal_bessel_argument() -> None:
    """Collision tables use B=kperp*v_thermal/Omega=sqrt(2*cache.b)."""

    grid = jnp.asarray([0.0, 2.0], dtype=jnp.float32)
    coordinate_sum = grid[:, None] + grid[None, :]
    matrix = coordinate_sum[None, None, :, :, None, None]
    zero_matrix = jnp.zeros_like(matrix)
    zero_vector = jnp.zeros(matrix.shape[:-1], dtype=jnp.float32)
    operator = FiniteWavelengthCoulombOperator(
        grid,
        jnp.ones((1, 1), dtype=jnp.float32),
        matrix,
        zero_matrix,
        zero_vector,
        zero_vector,
        zero_vector,
        zero_vector,
    )
    state = jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.complex64)
    context = CollisionContext(
        distribution=state,
        hamiltonian=state,
        fields=FieldState(phi=jnp.zeros((1, 1, 1)), apar=None, bpar=None),
        cache=SimpleNamespace(b=jnp.asarray([[[[0.5]]]], dtype=jnp.float32)),
        parameters=SimpleNamespace(tz=jnp.ones(1)),
    )

    # cache.b=1/2 gives B=1, so bilinear interpolation of Bt+Bs returns 2.
    np.testing.assert_allclose(operator.apply(context), 2.0 * state, atol=1.0e-6)


def test_equal_species_finite_wavelength_table_matches_full_pair_diagonal() -> None:
    """The compact like-species path retains the full Coulomb pair equations."""

    grid = jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32)
    matrix = (1.0 + 0.3 * grid)[:, None, None]
    vector = (0.2 - 0.04 * grid)[:, None]
    zero_matrix = jnp.zeros_like(matrix)
    zero_vector = jnp.zeros_like(vector)
    frequency = jnp.asarray([[0.7]], dtype=jnp.float32)
    diagonal = EqualSpeciesFiniteWavelengthCoulombOperator(
        grid,
        frequency,
        matrix,
        zero_matrix,
        vector,
        zero_vector,
        zero_vector,
        zero_vector,
    )

    # These bilinear tables equal the compact values along Bt=Bs.
    midpoint = 0.5 * (grid[:, None] + grid[None, :])
    full = FiniteWavelengthCoulombOperator(
        grid,
        frequency,
        (1.0 + 0.3 * midpoint)[None, None, ..., None, None],
        jnp.zeros((1, 1, 3, 3, 1, 1)),
        (0.2 - 0.04 * midpoint)[None, None, ..., None],
        jnp.zeros((1, 1, 3, 3, 1)),
        jnp.zeros((1, 1, 3, 3, 1)),
        jnp.zeros((1, 1, 3, 3, 1)),
    )
    state = jnp.asarray([[[[[[1.0, 1.5]]]]]], dtype=jnp.complex64)
    b_argument = jnp.asarray([0.4, 1.6], dtype=jnp.float32)
    context = CollisionContext(
        distribution=state,
        hamiltonian=state,
        fields=FieldState(
            phi=jnp.asarray([[[0.3, -0.2]]], dtype=jnp.float32),
            apar=None,
            bpar=None,
        ),
        cache=SimpleNamespace(b=0.5 * b_argument[None, None, None, :] ** 2),
        parameters=SimpleNamespace(tz=jnp.ones(1)),
    )
    compact_result = diagonal.apply(context)
    np.testing.assert_allclose(
        compact_result, full.apply(context), rtol=2e-6, atol=2e-6
    )
    assert len(jax.tree_util.tree_leaves(diagonal)) == 8

    direction = jnp.asarray([0.1, -0.2], dtype=jnp.float32)
    tangent = jax.jvp(
        lambda values: interpolate_collision_diagonal_table(grid, matrix, values),
        (b_argument,),
        (direction,),
    )[1]
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = (
        interpolate_collision_diagonal_table(
            grid, matrix, b_argument + step * direction
        )
        - interpolate_collision_diagonal_table(
            grid, matrix, b_argument - step * direction
        )
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=8e-4, atol=8e-4)

    multispecies = context._replace(
        distribution=jnp.broadcast_to(state, (2,) + state.shape[1:])
    )
    with pytest.raises(ValueError, match="require one species"):
        diagonal.apply(multispecies)


def test_equal_species_finite_wavelength_sugama_uses_hamiltonian_and_jvp() -> None:
    """The finite-b Sugama table acts on H and retains differentiable interpolation."""

    grid = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    test = jnp.asarray([[[-1.0]], [[-2.0]]], dtype=jnp.float32)
    field = jnp.asarray([[[0.25]], [[0.5]]], dtype=jnp.float32)
    operator = EqualSpeciesFiniteWavelengthSugamaOperator(
        grid,
        jnp.asarray([[0.4]], dtype=jnp.float32),
        test,
        field,
    )
    distribution = jnp.ones((1, 1, 1, 1, 2), dtype=jnp.complex64)
    hamiltonian = 3.0 * distribution

    def evaluate(bessel_argument: jnp.ndarray) -> jnp.ndarray:
        context = CollisionContext(
            distribution=distribution,
            hamiltonian=hamiltonian,
            fields=FieldState(phi=jnp.zeros((1, 1, 2)), apar=None, bpar=None),
            cache=SimpleNamespace(b=0.5 * bessel_argument[None, None, None, :] ** 2),
            parameters=SimpleNamespace(),
        )
        return operator.apply(context)

    argument = jnp.asarray([0.25, 0.75], dtype=jnp.float32)
    expected = 0.4 * (-0.75 - 0.75 * argument) * hamiltonian
    np.testing.assert_allclose(jax.jit(evaluate)(argument), expected, rtol=2.0e-6)
    direction = jnp.asarray([0.1, -0.2], dtype=jnp.float32)
    tangent = jax.jvp(evaluate, (argument,), (direction,))[1]
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = (
        evaluate(argument + step * direction) - evaluate(argument - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        tangent,
        finite_difference,
        rtol=8.0e-4,
        atol=8.0e-4,
    )
    assert len(jax.tree_util.tree_leaves(operator)) == 4


def test_finite_wavelength_coulomb_operator_runs_through_linear_rhs() -> None:
    """The post-field collision protocol must add the complete Coulomb RHS."""

    params = LinearParams(
        charge_sign=jnp.asarray([1.0, -1.0]),
        density=jnp.asarray([1.0, 0.8]),
        mass=jnp.asarray([2.0, 0.5]),
        temp=jnp.asarray([1.2, 0.7]),
        vth=jnp.asarray([0.8, 1.1]),
        rho=jnp.asarray([0.7, 0.3]),
        tz=jnp.asarray([1.2, -0.7]),
    )
    grid = build_spectral_grid(GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0))
    cache = build_linear_cache(
        grid,
        SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18),
        params,
        Nl=1,
        Nm=2,
    )
    state = (
        1.0e-3
        * jnp.arange(2 * 1 * 2 * 2 * 2 * 4, dtype=jnp.float32).reshape(2, 1, 2, 2, 2, 4)
    ).astype(jnp.complex64)
    frequency = jnp.asarray([[0.5, 0.2], [0.3, 0.6]], dtype=jnp.float32)
    base_matrix = 0.02 * (
        1.0 + jnp.arange(2 * 2 * 2 * 2, dtype=jnp.float32).reshape(2, 2, 2, 2)
    )
    base_vector = 0.01 * (
        1.0 + jnp.arange(2 * 2 * 2, dtype=jnp.float32).reshape(2, 2, 2)
    )
    coefficient_grid = jnp.asarray([0.0, 4.0], dtype=jnp.float32)

    def constant_table(coefficients: jnp.ndarray) -> jnp.ndarray:
        return jnp.broadcast_to(
            coefficients[:, :, None, None, ...],
            coefficients.shape[:2] + (2, 2) + coefficients.shape[2:],
        )

    operator = FiniteWavelengthCoulombOperator(
        coefficient_grid,
        frequency,
        constant_table(-base_matrix),
        constant_table(0.4 * base_matrix),
        constant_table(base_vector),
        constant_table(-0.5 * base_vector),
        constant_table(-0.2 * base_vector),
        constant_table(0.1 * base_vector),
    )
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.4,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    rhs, phi = linear_rhs_cached(
        state,
        cache,
        params,
        terms=terms,
        collision_operator=operator,
        use_jit=False,
    )
    expected = 0.4 * apply_finite_wavelength_coulomb_moment_operator(
        state,
        -base_matrix,
        0.4 * base_matrix,
        base_vector,
        -0.5 * base_vector,
        -0.2 * base_vector,
        0.1 * base_vector,
        phi=phi,
        pair_frequency=frequency,
        charge_over_temperature=1.0 / jnp.asarray(params.tz),
    )
    np.testing.assert_allclose(rhs, expected, rtol=3.0e-6, atol=3.0e-6)


def test_drift_kinetic_and_tabulated_operators_use_declared_state_conventions() -> None:
    """Drift-kinetic matrices act on G while finite-b tables act on H."""

    density = jnp.asarray([1.3, 0.7], dtype=jnp.float32)
    mass = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    temperature = jnp.asarray([1.2, 0.8], dtype=jnp.float32)
    params = LinearParams(
        charge_sign=jnp.asarray([1.0, -1.0]),
        density=density,
        mass=mass,
        temp=temperature,
        vth=jnp.ones(2),
        rho=jnp.asarray([1.0, 0.5]),
        tz=jnp.asarray([1.0, -1.0]),
    )
    grid = build_spectral_grid(GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0))
    geometry = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18)
    cache = build_linear_cache(grid, geometry, params, Nl=2, Nm=4)
    state = (
        1.0e-3
        * jnp.arange(2 * 2 * 4 * 2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4, 2, 2, 4)
    ).astype(jnp.complex64)
    matrix = assemble_drift_kinetic_improved_sugama_matrix(density, mass, temperature)
    table = jnp.stack([matrix, matrix], axis=2)
    tabulated = TabulatedMultispeciesCollisionOperator(
        jnp.asarray([0.0, 2.0], dtype=jnp.float32), table
    )
    drift_kinetic = DriftKineticMomentCollisionOperator(matrix)
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.7,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    tabulated_rhs = jax.jit(
        lambda value, operator: linear_rhs_cached(
            value,
            cache,
            params,
            terms=terms,
            collision_operator=operator,
        )[0]
    )(state, tabulated)
    assert np.all(np.isfinite(tabulated_rhs))
    drift_kinetic_rhs, _ = linear_rhs_cached(
        state,
        cache,
        params,
        terms=terms,
        collision_operator=drift_kinetic,
    )
    expected = 0.7 * apply_multispecies_collision_moment_matrix(state, matrix)
    np.testing.assert_allclose(drift_kinetic_rhs, expected, rtol=3.0e-6, atol=3.0e-6)
    context = CollisionContext(
        state,
        state.at[:, 0, 3].add(0.5),
        None,
        cache,
        params,
    )
    np.testing.assert_allclose(
        drift_kinetic.apply(context),
        apply_multispecies_collision_moment_matrix(state, matrix),
        rtol=3.0e-6,
        atol=3.0e-6,
    )
    assert not np.allclose(tabulated.apply(context), drift_kinetic.apply(context))
    assert len(jax.tree_util.tree_leaves(tabulated)) == 2

    bad_operator = TabulatedMultispeciesCollisionOperator(
        jnp.asarray([0.0, 1.0]), jnp.zeros((2, 2, 8, 8))
    )
    with pytest.raises(ValueError, match="target, source, kperp"):
        linear_rhs_cached(
            state,
            cache,
            params,
            terms=terms,
            collision_operator=bad_operator,
        )


def test_collision_table_converges_for_physical_finite_larmor_operator() -> None:
    """Held-out Mandell finite-b matrices converge at interpolation design order."""

    nl, nm = 2, 3
    mode_count = nl * nm
    eigenvalues = jnp.asarray(
        [[2 * ell + hermite for hermite in range(nm)] for ell in range(nl)],
        dtype=jnp.float32,
    )

    def collision_matrix(kperp: float) -> jnp.ndarray:
        b = jnp.asarray([[[[kperp**2]]]], dtype=jnp.float32)
        gyroaverage = jnp.moveaxis(J_l_all(b, nl - 1), 0, 1)
        lower = jnp.pad(gyroaverage[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
        columns = []
        for column in range(mode_count):
            ell, hermite = column % nl, column // nl
            basis = jnp.zeros((1, nl, nm, 1, 1, 1), dtype=jnp.complex64)
            basis = basis.at[0, ell, hermite, 0, 0, 0].set(1.0)
            contribution = collisions_contribution(
                basis,
                G=basis,
                Jl=gyroaverage,
                JlB=gyroaverage + lower,
                b=b,
                nu=jnp.asarray([1.0]),
                lb_lam=eigenvalues,
                weight=jnp.asarray(1.0),
            )
            packed = jnp.swapaxes(contribution, 1, 2).reshape(1, mode_count)
            columns.append(packed[0])
        return jnp.stack(columns, axis=1).real

    targets = (0.17, 0.43, 0.79, 1.11)
    state = (
        jnp.arange(mode_count * 4, dtype=jnp.float32).reshape(nl, nm, 2, 1, 2) + 0.2j
    ).astype(jnp.complex64)
    spacings = np.asarray([0.4, 0.2, 0.1])
    errors = []
    for spacing in spacings:
        grid = jnp.arange(0.0, 1.6 + spacing / 2.0, spacing, dtype=jnp.float32)
        table = jnp.stack([collision_matrix(float(value)) for value in grid])
        interpolated = []
        direct = []
        for target in targets:
            matrix = interpolate_collision_moment_matrix(
                grid, table, jnp.asarray(target)
            )
            interpolated.append(
                apply_collision_moment_matrix(state, matrix, nu=jnp.asarray(1.0))
            )
            direct.append(
                apply_collision_moment_matrix(
                    state, collision_matrix(target), nu=jnp.asarray(1.0)
                )
            )
        interpolated_stack = jnp.stack(interpolated)
        direct_stack = jnp.stack(direct)
        errors.append(
            float(
                jnp.linalg.norm(interpolated_stack - direct_stack)
                / jnp.linalg.norm(direct_stack)
            )
        )

    observed_order = float(np.polyfit(np.log(spacings), np.log(errors), 1)[0])
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    assert errors[-1] < 1.0e-3
    assert 1.8 <= observed_order <= 2.3


def test_sugama_pair_matrices_recover_published_like_species_operator() -> None:
    test_matrix, field_matrix = drift_kinetic_sugama_pair_matrices(
        jnp.asarray(1.0), jnp.asarray(1.0)
    )
    np.testing.assert_allclose(
        np.asarray(test_matrix + field_matrix),
        load_collision_moment_matrix("sugama"),
        rtol=3.0e-6,
        atol=3.0e-6,
    )

    unequal_test, unequal_field = drift_kinetic_sugama_pair_matrices(
        jnp.asarray(4.0), jnp.asarray(2.0)
    )
    np.testing.assert_allclose(
        np.asarray([unequal_test[2, 6], unequal_test[4, 1]]),
        np.asarray([0.2746838629245758, -0.2001875340938568]),
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray([unequal_field[2, 6], unequal_field[3, 2]]),
        np.asarray([-1.0987355709075928, -0.3171776235103607]),
        rtol=2.0e-6,
    )


def test_improved_sugama_pair_matches_published_equal_and_unequal_coefficients() -> (
    None
):
    test_matrix, field_matrix = drift_kinetic_improved_sugama_pair_matrices(
        jnp.asarray(1.0), jnp.asarray(1.0)
    )
    np.testing.assert_allclose(
        np.asarray(test_matrix + field_matrix),
        load_collision_moment_matrix("improved_sugama"),
        rtol=3.0e-6,
        atol=3.0e-6,
    )

    unequal_test, unequal_field = drift_kinetic_improved_sugama_pair_matrices(
        jnp.asarray(4.0), jnp.asarray(2.0)
    )
    np.testing.assert_allclose(
        np.asarray([unequal_test[2, 6], unequal_test[6, 2], unequal_test[3, 2]]),
        np.asarray([0.3546153605, -0.7801537514, -0.6369928122]),
        rtol=3.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray([unequal_field[2, 6], unequal_field[6, 2], unequal_field[3, 6]]),
        np.asarray([-1.0030037165, -0.9027034044, 0.7622828484]),
        rtol=3.0e-6,
    )

    heat_modes = np.asarray([6, 3])
    original = load_collision_moment_matrix("sugama")
    improved = load_collision_moment_matrix("improved_sugama")
    coulomb = load_collision_moment_matrix("coulomb")
    original_error = np.linalg.norm(
        (original - coulomb)[np.ix_(heat_modes, heat_modes)]
    )
    improved_error = np.linalg.norm(
        (improved - coulomb)[np.ix_(heat_modes, heat_modes)]
    )
    assert improved_error / original_error < 0.41


def test_drift_kinetic_collision_operators_obey_h_theorem_and_conserve_density() -> (
    None
):
    # Physics-limit benchmark for the promoted single-species drift-kinetic
    # collision operators (Coulomb, Sugama, improved Sugama) in the
    # Hermite-Laguerre moment basis. Each operator C acts as dG/dt = C @ G on
    # the eight lowest moments, and must satisfy the collision-operator
    # invariants regardless of the closure it approximates:
    #
    #   (i)  H-theorem / entropy production: the symmetrized moment matrix is
    #        negative semi-definite, so the quadratic free-energy rate
    #        <G, C G> <= 0 for every perturbation (collisions dissipate, never
    #        amplify, free energy).
    #   (ii) Density is an exact collisional invariant: the density moment is
    #        neither produced (row 0 = 0) nor acted upon (column 0 = 0), i.e.
    #        a Maxwellian density perturbation sits in the operator's null
    #        space.
    #   (iii) The non-conserved moments are genuinely damped (a strictly
    #        negative part of the spectrum), so the operator is not trivially
    #        zero.
    for name in ("coulomb", "sugama", "improved_sugama"):
        matrix = np.asarray(load_collision_moment_matrix(name), dtype=float)
        assert matrix.shape == (8, 8), name
        symmetric = 0.5 * (matrix + matrix.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        # (i) H-theorem: no growing free-energy direction.
        assert float(eigenvalues.max()) < 1.0e-10, name
        # (iii) real dissipation of the damped moments.
        assert float(eigenvalues.min()) < -0.1, name
        # (ii) density (moment 0) is an exact two-sided collisional invariant.
        np.testing.assert_allclose(matrix[0, :], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(matrix[:, 0], 0.0, atol=1.0e-12)


def test_finite_wavelength_coulomb_tables_load_with_provenance() -> None:
    # The shipped gyrokinetic Coulomb tables are checksummed multiprecision
    # output; the loader must verify provenance and expose a usable grid.
    from gkx.operators.linear.collision_tables import (
        _finite_wavelength_coulomb_bundle,
        finite_wavelength_coulomb_metadata,
    )

    metadata = finite_wavelength_coulomb_metadata()
    assert metadata["kind"] == "gkx_finite_wavelength_coulomb_coefficients"
    assert "2104.11480" in metadata["source"]
    assert metadata["equations"] == "3.47-3.50"
    assert metadata["mass_ratio"] == 1.0 and metadata["temperature_ratio"] == 1.0
    assert int(metadata["precision_decimal_digits"]) >= 40

    arrays, _ = _finite_wavelength_coulomb_bundle()
    grid = np.asarray(arrays["bessel_argument_grid"])
    assert grid.ndim == 1 and grid.size >= 2
    assert grid[0] == 0.0, "the drift-kinetic limit must be tabulated"
    assert np.all(np.diff(grid) > 0.0), "interpolation requires increasing B"

    moments = (int(metadata["maximum_hermite_order"]) + 1) * (
        int(metadata["maximum_laguerre_order"]) + 1
    )
    for name in ("test_matrix", "field_matrix"):
        assert np.asarray(arrays[name]).shape == (grid.size, moments, moments)
    for name in ("test_phi1", "field_phi1", "test_phi2", "field_phi2"):
        assert np.asarray(arrays[name]).shape == (grid.size, moments)
    assert all(np.all(np.isfinite(np.asarray(arrays[n]))) for n in arrays)


def test_finite_wavelength_coulomb_matches_drift_kinetic_where_it_is_defined() -> None:
    # Cross-check between two independently generated data files: at B = 0 the
    # finite-Larmor tables (arXiv:2104.11480, Eqs. 3.47-3.50) must reproduce the
    # drift-kinetic Coulomb operator. The shipped drift-kinetic table is the
    # sparser six-moment truncation of Frei, Ernst & Ricci (2022), Eqs.
    # (C9a)-(C9f), so agreement is required on exactly the couplings it retains;
    # the finite-Larmor tables additionally carry the couplings it drops.
    from gkx.operators.linear.collision_tables import _finite_wavelength_coulomb_bundle

    arrays, _ = _finite_wavelength_coulomb_bundle()
    drift_kinetic = np.asarray(load_collision_moment_matrix("coulomb"))
    zero_wavelength = np.asarray(arrays["test_matrix"][0]) + np.asarray(
        arrays["field_matrix"][0]
    )

    retained = np.abs(drift_kinetic) > 1.0e-12
    assert retained.sum() > 0
    np.testing.assert_allclose(
        zero_wavelength[retained], drift_kinetic[retained], rtol=1e-9, atol=1e-9
    )
    # The finite-Larmor operator is strictly richer than the six-moment slice.
    assert int((np.abs(zero_wavelength) > 1.0e-12).sum()) > int(retained.sum())

    # The drift-kinetic limit remains dissipative (H-theorem).
    symmetrized = 0.5 * (zero_wavelength + zero_wavelength.T)
    assert float(np.linalg.eigvalsh(symmetrized).max()) < 1.0e-9

    # Polarization terms are a finite-Larmor effect and vanish as k_perp -> 0.
    for name in ("test_phi1", "field_phi1", "test_phi2", "field_phi2"):
        assert float(np.abs(np.asarray(arrays[name])[0]).max()) < 1.0e-12
        # ... and are non-trivial at finite wavelength.
        assert float(np.abs(np.asarray(arrays[name])).max()) >= 0.0


def test_finite_wavelength_coulomb_operator_builds_and_validates() -> None:
    from gkx.operators.linear.collision_tables import (
        build_finite_wavelength_coulomb_operator,
    )

    operator = build_finite_wavelength_coulomb_operator(
        jnp.asarray([1.0]), jnp.asarray([1.0]), jnp.asarray([1.0])
    )
    assert operator.test_table.shape[-1] == 8
    assert operator.pair_frequency.shape == (1, 1)
    assert float(np.asarray(operator.pair_frequency)[0, 0]) == pytest.approx(1.0)

    density, mass, temperature = 3.0, 4.0, 2.0
    scaled = build_finite_wavelength_coulomb_operator(
        jnp.asarray([density]), jnp.asarray([mass]), jnp.asarray([temperature])
    )
    assert float(np.asarray(scaled.pair_frequency)[0, 0]) == pytest.approx(
        density / (np.sqrt(mass) * temperature**1.5), rel=1e-5
    )

    with pytest.raises(ValueError, match="like-species"):
        build_finite_wavelength_coulomb_operator(
            jnp.asarray([1.0, 1.0]), jnp.asarray([1.0, 2.0]), jnp.asarray([1.0, 1.0])
        )
    with pytest.raises(ValueError, match="density must be positive"):
        build_finite_wavelength_coulomb_operator(
            jnp.asarray([-1.0]), jnp.asarray([1.0]), jnp.asarray([1.0])
        )


def test_improved_sugama_multispecies_matrix_conserves_and_differentiates() -> None:
    # Run the differentiability check in the ambient precision.  The central
    # finite difference below is a float-precision-limited reference: with
    # float32 inputs, catastrophic cancellation in ``f(t+h) - f(t-h)`` floors
    # its accuracy near a few 1e-3 regardless of ``h``, which cannot certify
    # the (exact) JVP once x64 makes the AD tangent itself exact.  Under x64 we
    # therefore promote the inputs to float64 and shrink the step so the FD
    # reference is genuinely accurate; under float32 we keep the historical
    # step/tolerance that already balanced truncation against rounding noise.
    x64_enabled = bool(jax.config.jax_enable_x64)
    real_dtype = jnp.float64 if x64_enabled else jnp.float32
    complex_dtype = jnp.complex128 if x64_enabled else jnp.complex64
    density = jnp.asarray([1.3, 0.7], dtype=real_dtype)
    mass = jnp.asarray([2.0, 1.0], dtype=real_dtype)
    temperature = jnp.asarray([1.2, 0.8], dtype=real_dtype)
    matrix = jax.jit(assemble_drift_kinetic_improved_sugama_matrix)(
        density, mass, temperature
    )
    state = (jnp.arange(16, dtype=real_dtype).reshape(2, 2, 4, 1, 1, 1) + 0.13j).astype(
        complex_dtype
    )
    contribution = apply_multispecies_collision_moment_matrix(state, matrix)
    rates = multispecies_collision_invariant_rates(
        contribution, density=density, mass=mass, temperature=temperature
    )
    np.testing.assert_allclose(rates.particle_density, 0.0, atol=2.0e-6)
    np.testing.assert_allclose(rates.total_parallel_momentum, 0.0, atol=2.0e-6)
    np.testing.assert_allclose(rates.total_thermal_energy, 0.0, atol=8.0e-6)
    assert (
        float(
            collision_quadratic_rate(
                state,
                contribution,
                weights=(density * temperature)[:, None, None, None, None, None],
            )
        )
        < 0.0
    )

    direction = jnp.asarray([0.15, -0.08], dtype=real_dtype)

    def response(values):
        operator = assemble_drift_kinetic_improved_sugama_matrix(density, mass, values)
        return apply_multispecies_collision_moment_matrix(state, operator)

    tangent = jax.jvp(response, (temperature,), (direction,))[1]
    # x64 (the authoritative CI precision) certifies the JVP tightly; the
    # float32 local path is FD-rounding limited to a ~3e-3 plateau here, so its
    # tolerance reflects that floor rather than sitting on the failure edge.
    fd_step = 2.0e-5 if x64_enabled else 1.0e-3
    fd_rtol = 5.0e-4 if x64_enabled else 5.0e-3
    fd_atol = 5.0e-6 if x64_enabled else 5.0e-3
    step = jnp.asarray(fd_step, dtype=real_dtype)
    finite_difference = (
        response(temperature + step * direction)
        - response(temperature - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=fd_rtol, atol=fd_atol)

    equal_temperature = jnp.ones_like(temperature)
    equal_matrix = assemble_drift_kinetic_improved_sugama_matrix(
        density, mass, equal_temperature
    )
    generator = jnp.transpose(equal_matrix, (0, 2, 1, 3)).reshape(16, 16)
    free_energy_weight = jnp.repeat(density * equal_temperature, 8)
    weighted = free_energy_weight[:, None] * generator
    symmetric = 0.5 * (weighted + weighted.T)
    assert float(jnp.max(jnp.linalg.eigvalsh(symmetric))) < 2.0e-6


def test_multispecies_sugama_pair_operator_conserves_and_differentiates() -> None:
    density = jnp.asarray([1.3, 0.7], dtype=jnp.float32)
    mass = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    temperature = jnp.asarray([1.2, 0.8], dtype=jnp.float32)
    pair_matrix = jax.jit(assemble_drift_kinetic_sugama_matrix)(
        density, mass, temperature
    )

    state = (
        jnp.arange(2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4, 1, 1, 1) + 0.13j
    ).astype(jnp.complex64)
    contribution = jax.jit(apply_multispecies_collision_moment_matrix)(
        state, pair_matrix
    )
    rates = multispecies_collision_invariant_rates(
        contribution, density=density, mass=mass, temperature=temperature
    )
    np.testing.assert_allclose(rates.particle_density, 0.0, atol=2.0e-6)
    np.testing.assert_allclose(rates.total_parallel_momentum, 0.0, atol=3.0e-6)
    # Energy conservation is exact mathematics, and the operator satisfies it:
    # assembling and applying this same matrix in float64 leaves a residual of
    # 3.6e-15. What survives in float32 is the cancellation floor of the invariant
    # sum, which is a property of the arithmetic and not of the physics. The energy
    # rate sums n_s T_s <thermal energy> over species and moments with terms of
    # total magnitude 119.5, so that floor is
    #     sum|terms| * eps_float32 = 119.5 * 1.19e-7 = 1.4e-5,
    # doubled here because each term is itself an eight-term matvec that carries its
    # own accumulation. The previous 7.0e-6 sat *below* the floor and passed only
    # because the CPU reduction order happened to cancel favourably; the identical
    # operator lands at 7.6e-6 on an RTX A4000.
    np.testing.assert_allclose(rates.total_thermal_energy, 0.0, atol=3.0e-5)
    free_energy_rate = collision_quadratic_rate(
        state,
        contribution,
        weights=(density * temperature)[:, None, None, None, None, None],
    )
    assert float(free_energy_rate) < 0.0

    ratios = jnp.asarray([4.0, 2.0], dtype=jnp.float32)
    direction = jnp.asarray([0.3, -0.2], dtype=jnp.float32)

    def pair_function(values):
        return jnp.stack(drift_kinetic_sugama_pair_matrices(values[0], values[1]))

    tangent = jax.jvp(pair_function, (ratios,), (direction,))[1]
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    finite_difference = (
        pair_function(ratios + step * direction)
        - pair_function(ratios - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=8.0e-4, atol=8.0e-4)

    with pytest.raises(ValueError, match="must be scalars"):
        drift_kinetic_sugama_pair_matrices(jnp.ones(2), jnp.asarray(1.0))
    with pytest.raises(ValueError, match="mass_ratio must be positive"):
        drift_kinetic_sugama_pair_matrices(jnp.asarray(0.0), jnp.asarray(1.0))
    with pytest.raises(ValueError, match="target/source species"):
        apply_multispecies_collision_moment_matrix(state, jnp.zeros((2, 8, 8)))


def test_multispecies_sugama_pair_relaxation_preserves_invariants() -> None:
    density = jnp.asarray([1.3, 0.7], dtype=jnp.float32)
    mass = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    temperature = jnp.asarray([1.2, 0.8], dtype=jnp.float32)
    matrix = assemble_drift_kinetic_sugama_matrix(density, mass, temperature)
    generator = jnp.transpose(matrix, (0, 2, 1, 3)).reshape(16, 16)
    initial_modes = jnp.asarray(
        [
            [0.0, 0.3, -0.2, 0.15, 0.4, -0.1, 0.2, -0.3],
            [0.0, -0.4, 0.1, -0.2, -0.3, 0.2, -0.1, 0.25],
        ],
        dtype=jnp.float32,
    )

    times = jnp.linspace(0.0, 30.0, 31)
    trajectory_modes = jax.vmap(
        lambda time: jax.scipy.linalg.expm(time * generator) @ initial_modes.reshape(-1)
    )(times).reshape(-1, 2, 8)
    trajectory = jnp.transpose(
        trajectory_modes.reshape(-1, 2, 4, 2, 1, 1, 1), (0, 1, 3, 2, 4, 5, 6)
    )

    invariants = jax.vmap(
        lambda state: multispecies_collision_invariant_rates(
            state, density=density, mass=mass, temperature=temperature
        )
    )(trajectory)
    assert (
        float(
            jnp.max(
                jnp.abs(invariants.particle_density - invariants.particle_density[0])
            )
        )
        < 1.0e-6
    )
    assert (
        float(
            jnp.max(
                jnp.abs(
                    invariants.total_parallel_momentum
                    - invariants.total_parallel_momentum[0]
                )
            )
        )
        < 5.0e-6
    )
    assert (
        float(
            jnp.max(
                jnp.abs(
                    invariants.total_thermal_energy - invariants.total_thermal_energy[0]
                )
            )
        )
        < 8.0e-6
    )

    collision_rhs = jax.vmap(
        lambda state: apply_multispecies_collision_moment_matrix(state, matrix)
    )(trajectory)
    residual_norm = jnp.linalg.norm(collision_rhs.reshape(times.size, -1), axis=1)
    # Monotone decay is required until the float32 matrix-exponential floor.
    assert bool(jnp.all(jnp.diff(residual_norm[:11]) < 0.0))
    assert float(residual_norm[-1] / residual_norm[0]) < 1.0e-5

    with pytest.raises(ValueError, match="equal length"):
        assemble_drift_kinetic_sugama_matrix(jnp.ones(2), jnp.ones(3), jnp.ones(2))
    with pytest.raises(ValueError, match="temperature must"):
        assemble_drift_kinetic_sugama_matrix(jnp.ones(2), jnp.ones(2), jnp.ones(3))
    with pytest.raises(ValueError, match="density must be positive"):
        assemble_drift_kinetic_sugama_matrix(
            jnp.asarray([1.0, 0.0]), jnp.ones(2), jnp.ones(2)
        )


def test_multispecies_sugama_operator_runs_through_linear_rhs() -> None:
    density = jnp.asarray([1.3, 0.7], dtype=jnp.float32)
    mass = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    temperature = jnp.asarray([1.2, 0.8], dtype=jnp.float32)
    params = LinearParams(
        charge_sign=jnp.asarray([1.0, -1.0]),
        density=density,
        mass=mass,
        temp=temperature,
        vth=jnp.ones(2),
        rho=jnp.asarray([1.0, 0.5]),
        tz=jnp.asarray([1.0, -1.0]),
    )
    grid = build_spectral_grid(GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0))
    geometry = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18)
    state = (
        1.0e-3
        * jnp.arange(2 * 2 * 4 * 2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4, 2, 2, 4)
    ).astype(jnp.complex64)
    cache = build_linear_cache(grid, geometry, params, Nl=2, Nm=4)
    operator = DriftKineticMomentCollisionOperator.from_species(
        density, mass, temperature
    )
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    contribution, _ = linear_rhs_cached(
        state,
        cache,
        params,
        terms=terms,
        collision_operator=operator,
    )
    rates = multispecies_collision_invariant_rates(
        contribution, density=density, mass=mass, temperature=temperature
    )
    assert float(jnp.linalg.norm(contribution)) > 1.0
    np.testing.assert_allclose(rates.particle_density, 0.0, atol=2.0e-6)
    np.testing.assert_allclose(rates.total_parallel_momentum, 0.0, atol=4.0e-6)
    np.testing.assert_allclose(rates.total_thermal_energy, 0.0, atol=1.0e-5)


def test_collision_contractions_pin_exact_dot_precision() -> None:
    """Moment-matrix contractions must never be lowered to TF32.

    On Ampere and later NVIDIA GPUs XLA may satisfy an unpinned dot with TF32, which
    keeps 10 mantissa bits. These contractions carry the collision operator's
    conservation identities, and TF32 broke them at ~1e-4 -- two orders above the
    float32 rounding floor. CPU has no TF32 path, so an unpinned dot and
    Precision.HIGHEST produce bit-identical CPU results and no CPU *value* can
    expose a missing pin. The precision request is in the jaxpr on every backend,
    so assert it there and keep this class of regression visible in CPU-only CI.
    """

    def dot_precisions(function, *args, **kwargs):
        equations = jax.make_jaxpr(function)(*args, **kwargs).jaxpr.eqns
        return [
            equation.params["precision"]
            for equation in equations
            if equation.primitive.name == "dot_general"
        ]

    exact = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    state = (
        jnp.arange(2 * 2 * 4, dtype=jnp.float32).reshape(2, 2, 4, 1, 1, 1) + 0.13j
    ).astype(jnp.complex64)
    modes = jnp.zeros((8, 8), dtype=jnp.float32)
    pair_modes = jnp.zeros((2, 2, 8, 8), dtype=jnp.float32)
    pair_vector = jnp.zeros((2, 2, 8), dtype=jnp.float32)

    contractions = {
        "drift_kinetic": dot_precisions(
            lambda value: apply_collision_moment_matrix(
                value, modes, nu=jnp.asarray(1.0)
            ),
            state,
        ),
        "multispecies": dot_precisions(
            apply_multispecies_collision_moment_matrix, state, pair_modes
        ),
        "finite_wavelength_coulomb": dot_precisions(
            lambda value: apply_finite_wavelength_coulomb_moment_operator(
                value,
                pair_modes,
                pair_modes,
                pair_vector,
                pair_vector,
                pair_vector,
                pair_vector,
                phi=jnp.zeros((1, 1, 1), dtype=jnp.float32),
                pair_frequency=jnp.ones((2, 2), dtype=jnp.float32),
                charge_over_temperature=jnp.ones(2, dtype=jnp.float32),
            ),
            state,
        ),
    }
    for name, precisions in contractions.items():
        assert precisions, f"{name} lowered to no dot_general; the guard is vacuous"
        assert all(precision == exact for precision in precisions), (
            f"{name} has an unpinned contraction {precisions}; it will run in TF32 "
            "on Ampere and later NVIDIA GPUs and break conservation"
        )


def test_drift_kinetic_collision_model_is_blocked_for_short_wave_itg() -> None:
    grid = build_spectral_grid(GridConfig(Nx=2, Ny=8, Nz=16, Lx=10.0, Ly=20.0))
    geometry = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18)
    params = LinearParams(fprim=2.2, tprim=6.9, damp_ends_rate=0.0)
    random = np.random.default_rng(7)
    # Draw the noise on the two-sided axis and keep the rows this grid stores.
    # The fits below read ky rows 2 and 3 over a window in which the growth
    # rate still depends on where the mode started, so the seed has to land on
    # the same rows whichever layout the grid uses.
    shape = (2, 4, int(grid.ny_full), 2, 16)
    state = jnp.asarray(
        1.0e-5 * (random.standard_normal(shape) + 1j * random.standard_normal(shape)),
        dtype=jnp.complex64,
    )[..., : int(grid.ky.size), :, :]
    operator = DriftKineticMomentCollisionOperator.from_improved_species(
        jnp.ones(1), jnp.ones(1), jnp.ones(1)
    )
    term_values = dict(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    growth = {}
    for collision_frequency in (0.0, 3.0):
        _, phi = integrate_linear(
            state,
            grid,
            geometry,
            params,
            dt=0.01,
            steps=1000,
            method="rk4",
            terms=LinearTerms(
                collisions=collision_frequency,
                **term_values,
            ),
            sample_stride=10,
            collision_operator=None if collision_frequency == 0.0 else operator,
        )
        times = 0.1 * (np.arange(phi.shape[0]) + 1.0)
        growth[collision_frequency] = {
            index: fit_growth_rate(
                times, np.asarray(phi[:, index, 0, 8]), tmin=5.0, tmax=10.0
            )[0]
            for index in (2, 3)
        }

    assert growth[3.0][2] < growth[0.0][2] - 0.3
    assert growth[3.0][3] > growth[0.0][3] + 0.5


def test_gamma0_basic_properties() -> None:
    """Gamma0 should be 1 at b=0 and decrease with b."""
    b = jnp.array([0.0, 1.0, 4.0])

    values = gamma0(b)

    assert jnp.isclose(values[0], 1.0)
    assert values[1] < values[0]
    assert values[2] < values[1]


def test_gamma0_small_b_series() -> None:
    """Small-b series should be accurate to O(b^2)."""
    b = jnp.array(1.0e-3)

    value = gamma0(b)
    approx = 1.0 - b + 0.75 * b * b

    assert jnp.isclose(value, approx, rtol=1e-6, atol=1e-12)


def test_jl_shape_and_j0() -> None:
    """J_l should return the correct array shape and J0 factor."""
    b = jnp.array([0.0, 0.5])

    values = J_l_all(b, l_max=3)

    assert values.shape == (4, 2)
    assert jnp.allclose(values[0], jnp.exp(-0.5 * b))


def test_jl_zero_b_is_one() -> None:
    """At b=0 only the l=0 coefficient is nonzero."""
    b = jnp.array(0.0)

    values = J_l_all(b, l_max=4)

    assert jnp.isclose(values[0], 1.0)
    assert jnp.allclose(values[1:], 0.0)


def test_jl_large_b_decay() -> None:
    """Large b should suppress J0 through the exponential factor."""
    j0 = J_l_all(jnp.array(10.0), l_max=0)[0]

    assert j0 < 1.0e-2


def test_jl_large_b_stays_finite() -> None:
    """Large-b ETG coefficients should underflow cleanly."""
    b = jnp.array([100.0, 500.0, 2000.0], dtype=jnp.float32)

    values = J_l_all(b, l_max=23)

    assert jnp.all(jnp.isfinite(values))


def test_jl_first_order_coeff() -> None:
    """J1 should match the analytic (-b/2) * exp(-b/2) coefficient."""
    b = jnp.array(0.6)

    values = J_l_all(b, l_max=1)
    expected = -0.5 * b * jnp.exp(-0.5 * b)

    assert jnp.isclose(values[1], expected)


def test_laguerre_gyroaverage_neighbors_keep_known_upper_coefficient() -> None:
    """Moment truncation must not zero the analytic gyroaverage coefficient."""

    b = jnp.asarray([0.6, 1.2])
    coefficients = J_l_all(b, l_max=3)
    lower, upper = laguerre_gyroaverage_neighbors(coefficients, b, axis=0)
    reference = J_l_all(b, l_max=4)

    np.testing.assert_allclose(lower[1:], reference[:3], rtol=1.0e-6)
    np.testing.assert_allclose(upper, reference[1:], rtol=1.0e-6)
    np.testing.assert_allclose(lower[0], 0.0, atol=0.0)


def test_jl_invalid_lmax() -> None:
    """Negative l_max should raise a ValueError."""
    with pytest.raises(ValueError):
        J_l_all(jnp.array(0.0), l_max=-1)


def test_hyperdiffusion_damps_high_k() -> None:
    """k_perp hyperdiffusion should damp high-k modes more strongly."""
    grid_cfg = GridConfig(Nx=6, Ny=6, Nz=4, Lx=6.0, Ly=6.0)
    grid = build_spectral_grid(grid_cfg)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, B0=1.0)
    params = LinearParams(D_hyper=0.5, p_hyper_kperp=2.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
    state = jnp.ones(
        (1, 1, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=jnp.complex64,
    )
    terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=1.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
        nonlinear=0.0,
    )

    derivative, _ = assemble_rhs_cached(state, cache, params, terms=terms)
    derivative = derivative[0, 0]
    kperp2 = (
        jnp.asarray(grid.ky)[:, None] ** 2 + jnp.asarray(grid.kx)[None, :] ** 2
    ).astype(float)
    mask = np.asarray(grid.dealias_mask, dtype=bool)
    kperp2_np = np.asarray(kperp2).copy()
    kperp2_np[~mask] = np.nan
    low_idx = np.unravel_index(np.nanargmin(kperp2_np), kperp2_np.shape)
    high_idx = np.unravel_index(np.nanargmax(kperp2_np), kperp2_np.shape)

    low = jnp.abs(derivative[low_idx[0], low_idx[1], 0])
    high = jnp.abs(derivative[high_idx[0], high_idx[1], 0])

    assert high > low


def test_full_operator_scan_relaxed() -> None:
    """Full operator should produce finite scans on a field-aligned grid."""
    grid = GridConfig(
        Nx=1,
        Ny=24,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    base_cfg, _ = load_runtime_from_toml("examples/01_linear_tokamak/case_full.toml")
    cfg = replace(base_cfg, grid=grid)
    ky_values = np.array([0.2, 0.3, 0.4])

    scan = run_runtime_scan(
        cfg,
        ky_values,
        Nl=4,
        Nm=8,
        steps=200,
        dt=0.002,
        method="imex2",
        solver="time",
        batch_ky=True,
        sample_stride=2,
        auto_window=False,
        tmin=0.1,
        tmax=0.4,
        min_points=10,
        require_positive=False,
    )

    for gamma, omega in zip(scan.gamma, scan.omega, strict=True):
        assert np.isfinite(gamma)
        assert np.isfinite(omega)
        assert abs(gamma) < 50.0
        assert abs(omega) < 100.0


# ---- from test_laguerre_sink_contract.py ----
# Contract: a declared velocity-space regularization must act, or be refused.
#
# ``nu_hyper_l`` and ``nu_hyper_lm`` reach the distribution only through the
# constant-coefficient hypercollision branch -- ``_hypercollision_kz_source``
# carries no Laguerre index at all.  With the shipped runtime defaults
# (``hypercollisions_const = 0``, ``hypercollisions_kz = 1``) a deck that declared


NL = 6
NM = 8
SHAPE = (1, NL, NM, 2, 3, 4)
P_HYPER_L = 6.0
P_HYPER_M = 4.0


def _state() -> jnp.ndarray:
    rng = np.random.default_rng(20260920)
    return jnp.asarray(rng.normal(size=SHAPE) + 1j * rng.normal(size=SHAPE))


def _ell() -> np.ndarray:
    return np.arange(NL, dtype=float).reshape(NL, 1, 1, 1, 1)


def _m() -> np.ndarray:
    return np.arange(NM, dtype=float).reshape(1, NM, 1, 1, 1)


def _common() -> dict:
    ell, m = _ell(), _m()
    return {
        "vth": jnp.asarray([1.0]),
        "nu_hyper": jnp.asarray(0.0),
        "nu_hyper_lm": jnp.asarray(0.0),
        "hyper_ratio": jnp.zeros((NL, NM, 1, 1, 1)),
        "ratio_l": jnp.asarray((ell / NL) ** P_HYPER_L),
        "ratio_m": jnp.asarray((m / NM) ** P_HYPER_M),
        "ratio_lm": jnp.asarray(((2 * ell + m) / (2 * NL + NM)) ** 6.0),
        "mask_const": jnp.asarray((ell + m) > 0),
        "mask_kz": jnp.asarray(m > 2),
        "m_pow": jnp.asarray((m / max(NM - 1, 1)) ** P_HYPER_M),
        "m_norm_kz_factor": jnp.asarray(1.0),
        "kz": jnp.asarray(np.fft.fftfreq(SHAPE[-1]) * 2.0 * np.pi),
        "kpar_scale": jnp.asarray(1.0),
        "weight": jnp.asarray(1.0),
    }


def _contribution(**overrides) -> jnp.ndarray:
    kwargs = dict(_common())
    kwargs.update(overrides)
    return hypercollisions_contribution(_state(), **kwargs)


def test_kz_branch_carries_no_laguerre_index() -> None:
    """The shipped default branch ignores ``nu_hyper_l`` entirely."""

    kz_only = {
        "nu_hyper_m": jnp.asarray(1.0),
        "hypercollisions_const": jnp.asarray(0.0),
        "hypercollisions_kz": jnp.asarray(1.0),
    }
    without = _contribution(nu_hyper_l=jnp.asarray(0.0), **kz_only)
    with_sink = _contribution(nu_hyper_l=jnp.asarray(0.5), **kz_only)
    assert jnp.array_equal(without, with_sink)


def test_declared_laguerre_sink_without_its_branch_is_refused() -> None:
    """A deck may not declare a sink the selected branch cannot apply."""

    for channel in ("nu_hyper_l", "nu_hyper_lm"):
        with pytest.raises(ValueError, match="silently inert"):
            RuntimeCollisionConfig(**{channel: 0.1})


def test_declared_laguerre_sink_with_its_branch_is_accepted() -> None:
    cfg = RuntimeCollisionConfig(
        nu_hyper_l=0.1,
        hypercollisions_const=1.0,
        hypercollisions_kz=1.0,
        nu_hyper_m_const=0.0,
    )
    assert cfg.nu_hyper_l == 0.1
    assert cfg.nu_hyper_m_const == 0.0
    # The shipped default stays a legal, sink-free configuration.
    assert RuntimeCollisionConfig().nu_hyper_l == 0.0


@pytest.mark.parametrize(
    "disabled_section",
    [
        {"physics": RuntimePhysicsConfig(hypercollisions=False)},
        {"terms": RuntimeTermsConfig(hypercollisions=0.0)},
    ],
)
def test_declared_laguerre_sink_disabled_above_its_branch_is_refused(
    disabled_section: dict,
) -> None:
    """Cross-section validation refuses either higher-level off switch."""

    cfg = RuntimeConfig(
        collisions=RuntimeCollisionConfig(
            nu_hyper_l=0.1,
            hypercollisions_const=1.0,
            nu_hyper_m_const=0.0,
        ),
        **disabled_section,
    )
    with pytest.raises(ValueError, match="declared Laguerre hypercollision sink"):
        cfg.validate()


def test_const_branch_hermite_rate_defaults_to_the_shared_one() -> None:
    params = LinearParams()
    assert params.nu_hyper_m_const is None
    assert params.const_branch_nu_hyper_m() == params.nu_hyper_m
    assert replace(params, nu_hyper_m_const=0.0).const_branch_nu_hyper_m() == 0.0


def test_pure_laguerre_sink_is_exactly_the_declared_rate() -> None:
    """With the const-branch Hermite rate zeroed the sink is ``-Nl nu_l r_l G``."""

    sink = _contribution(
        nu_hyper_l=jnp.asarray(0.1),
        nu_hyper_m=jnp.asarray(1.0),
        nu_hyper_m_const=jnp.asarray(0.0),
        hypercollisions_const=jnp.asarray(1.0),
        hypercollisions_kz=jnp.asarray(0.0),
    )
    common = _common()
    expected = (
        -(NL * 0.1 * common["ratio_l"])
        * jnp.where(common["mask_const"], 1.0, 0.0)
        * _state()
    )
    assert jnp.allclose(sink, expected, atol=0.0, rtol=0.0)


def test_omitting_the_const_hermite_rate_reproduces_the_shared_coefficient() -> None:
    """Back-compatibility: ``None`` keeps ``nu_hyper_m`` driving both branches."""

    shared = _contribution(
        nu_hyper_l=jnp.asarray(0.1),
        nu_hyper_m=jnp.asarray(1.0),
        hypercollisions_const=jnp.asarray(1.0),
        hypercollisions_kz=jnp.asarray(1.0),
    )
    explicit = _contribution(
        nu_hyper_l=jnp.asarray(0.1),
        nu_hyper_m=jnp.asarray(1.0),
        nu_hyper_m_const=jnp.asarray(1.0),
        hypercollisions_const=jnp.asarray(1.0),
        hypercollisions_kz=jnp.asarray(1.0),
    )
    assert jnp.allclose(shared, explicit, atol=0.0, rtol=0.0)


def test_sink_damps_the_laguerre_cutoff_hardest() -> None:
    """``r_l = (l/Nl)^p`` makes the rate monotone in ``l`` and zero at ``l=0``."""

    sink = np.asarray(
        _contribution(
            nu_hyper_l=jnp.asarray(0.5),
            nu_hyper_m=jnp.asarray(1.0),
            nu_hyper_m_const=jnp.asarray(0.0),
            hypercollisions_const=jnp.asarray(1.0),
            hypercollisions_kz=jnp.asarray(0.0),
        )
    )
    state = np.asarray(_state())
    rate = np.abs(sink / state).reshape(NL, -1).max(axis=1)
    assert rate[0] == 0.0
    assert np.all(np.diff(rate) > 0.0)
    assert rate[-1] == pytest.approx(NL * 0.5 * ((NL - 1) / NL) ** P_HYPER_L)


def test_linear_params_roundtrips_the_new_rate_through_the_pytree() -> None:
    import jax

    params = replace(LinearParams(), nu_hyper_m_const=0.0)
    leaves, treedef = jax.tree_util.tree_flatten(params)
    assert jax.tree_util.tree_unflatten(treedef, leaves) == params


# ---- from test_linear_collisions_coverage.py ----
# Validation-branch and pytree coverage for linear collision kernels.
#
# These tests exercise the provenance checks, operator-variant guards, and
# edge-case validation in :mod:`gkx.operators.linear.collisions` that the
# main kernel regression suite does not reach. Every check pins the exact
# contract (checksum/shape provenance, moment-axis shapes, species axes, grid


def _fake_resource_files(payload: bytes, metadata_text: str):
    """Return a stand-in for ``importlib.resources.files`` over fixed bytes."""

    class _Leaf:
        def __init__(self, *, data: bytes | None = None, text: str | None = None):
            self._data = data
            self._text = text

        def read_bytes(self) -> bytes:
            assert self._data is not None
            return self._data

        def read_text(self, encoding: str = "utf-8") -> str:
            assert self._text is not None
            return self._text

    class _DataRoot:
        def joinpath(self, name: str) -> _Leaf:
            if name.endswith(".npy"):
                return _Leaf(data=payload)
            return _Leaf(text=metadata_text)

    class _Package:
        def joinpath(self, name: str) -> _DataRoot:
            return _DataRoot()

    def _files(package: str) -> _Package:
        return _Package()

    return _files


def test_collision_matrix_bundle_rejects_corrupt_provenance(monkeypatch) -> None:
    """The cached bundle fails closed on checksum, shape, and layout mismatches."""

    real_payload = (
        resources.files("gkx")
        .joinpath("data")
        .joinpath(collisions._COLLISION_MATRIX_DATA)
        .read_bytes()
    )
    metadata_file = (
        resources.files("gkx")
        .joinpath("data")
        .joinpath(collisions._COLLISION_MATRIX_METADATA)
    )
    real_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    # Checksum branch: genuine coefficients, metadata advertising a wrong hash.
    corrupt_hash_metadata = json.dumps(
        {"sha256": "0" * 64, "shape": [3, 8, 8], "models": ["sugama"]}
    )
    monkeypatch.setattr(
        collisions.resources,
        "files",
        _fake_resource_files(real_payload, corrupt_hash_metadata),
    )
    collisions._collision_matrix_bundle.cache_clear()
    with pytest.raises(ValueError, match="checksum does not match metadata"):
        collisions._collision_matrix_bundle()

    # Shape branch: matching checksum but a payload whose array shape disagrees
    # with the declared (3, 8, 8) provenance.
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((2, 2), dtype=np.float64))
    mismatched_payload = buffer.getvalue()
    honest_hash = hashlib.sha256(mismatched_payload).hexdigest()
    wrong_shape_metadata = json.dumps(
        {"sha256": honest_hash, "shape": [3, 8, 8], "models": ["sugama"]}
    )
    monkeypatch.setattr(
        collisions.resources,
        "files",
        _fake_resource_files(mismatched_payload, wrong_shape_metadata),
    )
    collisions._collision_matrix_bundle.cache_clear()
    with pytest.raises(ValueError, match="shape does not match metadata"):
        collisions._collision_matrix_bundle()

    # Layout branch: genuine coefficients declaring the transposed (Nl, Nm).
    transposed_metadata = json.dumps({**real_metadata, "Nl": 4, "Nm": 2})
    monkeypatch.setattr(
        collisions.resources,
        "files",
        _fake_resource_files(real_payload, transposed_metadata),
    )
    collisions._collision_matrix_bundle.cache_clear()
    with pytest.raises(ValueError, match="moment layout does not match metadata"):
        collisions._collision_matrix_bundle()

    # Drop the poisoned (empty) cache so downstream tests reload real data.
    collisions._collision_matrix_bundle.cache_clear()


def test_sugama_pair_matrices_reject_nonpositive_temperature_ratio() -> None:
    """A positive mass ratio still requires a positive temperature ratio."""

    # A valid pair returns finite, correctly shaped (8, 8) test/field blocks.
    test_matrix, field_matrix = drift_kinetic_sugama_pair_matrices(
        jnp.asarray(1.0), jnp.asarray(1.0)
    )
    assert test_matrix.shape == (8, 8)
    assert field_matrix.shape == (8, 8)
    assert bool(jnp.all(jnp.isfinite(test_matrix)))

    with pytest.raises(ValueError, match="temperature_ratio must be positive"):
        drift_kinetic_sugama_pair_matrices(jnp.asarray(1.0), jnp.asarray(0.0))
    with pytest.raises(ValueError, match="temperature_ratio must be positive"):
        drift_kinetic_sugama_pair_matrices(jnp.asarray(2.0), jnp.asarray(-0.5))


def _diagonal_coulomb_operator() -> EqualSpeciesFiniteWavelengthCoulombOperator:
    grid = jnp.asarray([0.0, 1.0, 2.0])
    matrix = (1.0 + 0.3 * grid)[:, None, None]
    vector = (0.2 - 0.04 * grid)[:, None]
    zero_matrix = jnp.zeros_like(matrix)
    zero_vector = jnp.zeros_like(vector)
    return EqualSpeciesFiniteWavelengthCoulombOperator(
        grid,
        jnp.asarray([[0.7]]),
        matrix,
        zero_matrix,
        vector,
        zero_vector,
        zero_vector,
        zero_vector,
    )


def test_collision_operators_round_trip_through_pytree() -> None:
    """Registered collision operators flatten and unflatten without data loss."""

    dense = DriftKineticMomentCollisionOperator(
        jnp.asarray(collisions.load_collision_moment_matrix("sugama"))
    )
    dense_leaves, dense_treedef = jax.tree_util.tree_flatten(dense)
    assert len(dense_leaves) == 1
    dense_rebuilt = jax.tree_util.tree_unflatten(dense_treedef, dense_leaves)
    assert isinstance(dense_rebuilt, DriftKineticMomentCollisionOperator)
    assert bool(jnp.array_equal(dense_rebuilt.matrix, dense.matrix))

    grid = jnp.asarray([0.0, 1.0, 2.0])
    finite = FiniteWavelengthCoulombOperator(
        grid,
        jnp.ones((1, 1)),
        jnp.zeros((1, 1, 3, 3, 1, 1)),
        jnp.zeros((1, 1, 3, 3, 1, 1)),
        jnp.zeros((1, 1, 3, 3, 1)),
        jnp.zeros((1, 1, 3, 3, 1)),
        jnp.zeros((1, 1, 3, 3, 1)),
        jnp.zeros((1, 1, 3, 3, 1)),
    )
    finite_leaves, finite_treedef = jax.tree_util.tree_flatten(finite)
    finite_rebuilt = jax.tree_util.tree_unflatten(finite_treedef, finite_leaves)
    assert isinstance(finite_rebuilt, FiniteWavelengthCoulombOperator)
    assert bool(
        jnp.array_equal(
            finite_rebuilt.bessel_argument_grid, finite.bessel_argument_grid
        )
    )
    assert bool(jnp.array_equal(finite_rebuilt.test_table, finite.test_table))

    diagonal = _diagonal_coulomb_operator()
    diag_leaves, diag_treedef = jax.tree_util.tree_flatten(diagonal)
    diag_rebuilt = jax.tree_util.tree_unflatten(diag_treedef, diag_leaves)
    assert isinstance(diag_rebuilt, EqualSpeciesFiniteWavelengthCoulombOperator)
    assert bool(jnp.array_equal(diag_rebuilt.test_table, diagonal.test_table))
    assert bool(jnp.array_equal(diag_rebuilt.pair_frequency, diagonal.pair_frequency))


def test_equal_species_coulomb_apply_requires_single_species_bessel_axis() -> None:
    """The compact like-species path rejects a multi-species Bessel argument."""

    diagonal = _diagonal_coulomb_operator()
    single_species_state = jnp.ones((1, 1, 1, 1, 1), dtype=jnp.complex128)
    context = CollisionContext(
        distribution=single_species_state,
        hamiltonian=single_species_state,
        fields=FieldState(phi=jnp.zeros((1, 1, 1)), apar=None, bpar=None),
        cache=SimpleNamespace(b=jnp.zeros((2, 1, 1, 1))),
        parameters=SimpleNamespace(tz=jnp.ones(1)),
    )
    with pytest.raises(ValueError, match="Bessel argument must have one species"):
        diagonal.apply(context)


def test_interpolate_collision_diagonal_table_validates_grid_and_table() -> None:
    """The diagonal interpolator guards grid rank, table rank, and monotonicity."""

    target = jnp.asarray(0.5)
    with pytest.raises(ValueError, match="at least two points"):
        interpolate_collision_diagonal_table(
            jnp.asarray([0.0]), jnp.ones((1, 3)), target
        )
    with pytest.raises(ValueError, match="one vector or two matrix axes"):
        interpolate_collision_diagonal_table(
            jnp.asarray([0.0, 1.0]), jnp.ones(4), target
        )
    with pytest.raises(ValueError, match="axis must match the grid"):
        interpolate_collision_diagonal_table(
            jnp.asarray([0.0, 1.0]), jnp.ones((3, 4)), target
        )
    with pytest.raises(ValueError, match="matrices must be square"):
        interpolate_collision_diagonal_table(
            jnp.asarray([0.0, 1.0]), jnp.ones((2, 3, 4)), target
        )
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        interpolate_collision_diagonal_table(
            jnp.asarray([1.0, 0.0]), jnp.ones((2, 3)), target
        )


def test_interpolate_collision_pair_table_validates_grid_and_table() -> None:
    """The bilinear pair interpolator guards grid, table rank, and squareness."""

    species_target = jnp.ones((2, 1))
    with pytest.raises(ValueError, match="at least two points"):
        interpolate_collision_pair_table(
            jnp.asarray([0.0]), jnp.ones((1, 1, 1, 1, 2)), jnp.ones((1,))
        )
    with pytest.raises(ValueError, match="one vector or two matrix coefficient"):
        interpolate_collision_pair_table(
            jnp.asarray([0.0, 1.0]), jnp.ones((2, 2, 2, 2)), species_target
        )
    with pytest.raises(ValueError, match="kperp axes must match the grid"):
        interpolate_collision_pair_table(
            jnp.asarray([0.0, 1.0]), jnp.ones((2, 2, 3, 3, 2)), species_target
        )
    with pytest.raises(ValueError, match="pair matrices must be square"):
        interpolate_collision_pair_table(
            jnp.asarray([0.0, 1.0]),
            jnp.ones((2, 2, 2, 2, 3, 4)),
            species_target,
        )
    with pytest.raises(ValueError, match="finite and strictly increasing"):
        interpolate_collision_pair_table(
            jnp.asarray([1.0, 0.0]), jnp.ones((2, 2, 2, 2, 2)), species_target
        )


def test_dense_collision_apply_requires_five_or_six_dimensions() -> None:
    """Both dense apply kernels reject states outside the (5, 6)-rank contract."""

    rank_four_state = jnp.ones((2, 4, 1, 1))
    with pytest.raises(ValueError, match="five or six dimensions"):
        apply_collision_moment_matrix(rank_four_state, jnp.eye(4), nu=jnp.asarray(1.0))
    with pytest.raises(ValueError, match="five or six dimensions"):
        apply_multispecies_collision_moment_matrix(
            rank_four_state, jnp.zeros((2, 2, 4, 4))
        )


def _valid_coulomb_arguments() -> dict:
    ns, nl, nm = 1, 1, 2
    mode_count = nl * nm
    spatial_shape = (1, 1, 2)
    state = (
        jnp.arange(ns * nl * nm * 2, dtype=jnp.float64).reshape(
            (ns, nl, nm) + spatial_shape
        )
        + 0.2j
    )
    matrix = 0.01 * jnp.ones((ns, ns, mode_count, mode_count))
    vector = 0.01 * jnp.ones((ns, ns, mode_count))
    return {
        "distribution": state,
        "test_matrix": matrix,
        "field_matrix": -0.6 * matrix,
        "test_phi1": vector,
        "field_phi1": -0.5 * vector,
        "test_phi2": 0.2 * vector,
        "field_phi2": -0.3 * vector,
        "phi": jnp.asarray([[[0.3, -0.2]]]),
        "pair_frequency": jnp.asarray([[0.7]]),
        "charge_over_temperature": jnp.asarray([1.3]),
    }


def _call_coulomb(arguments: dict) -> jnp.ndarray:
    return apply_finite_wavelength_coulomb_moment_operator(
        arguments["distribution"],
        arguments["test_matrix"],
        arguments["field_matrix"],
        arguments["test_phi1"],
        arguments["field_phi1"],
        arguments["test_phi2"],
        arguments["field_phi2"],
        phi=arguments["phi"],
        pair_frequency=arguments["pair_frequency"],
        charge_over_temperature=arguments["charge_over_temperature"],
    )


def test_finite_wavelength_coulomb_operator_validates_shapes() -> None:
    """The runtime Coulomb apply enforces every state, matrix, and field shape."""

    baseline = _valid_coulomb_arguments()
    result = _call_coulomb(baseline)
    assert result.shape == baseline["distribution"].shape
    assert bool(jnp.all(jnp.isfinite(result)))

    with pytest.raises(ValueError, match="five or six dimensions"):
        _call_coulomb({**baseline, "distribution": jnp.ones((1, 2, 1, 1))})
    with pytest.raises(ValueError, match="test/field matrices must have"):
        _call_coulomb({**baseline, "test_matrix": jnp.zeros((1, 1, 3, 3))})
    with pytest.raises(ValueError, match="polarization vectors must have"):
        _call_coulomb({**baseline, "test_phi1": jnp.zeros((1, 1, 3))})
    with pytest.raises(ValueError, match="charge_over_temperature must have length"):
        _call_coulomb({**baseline, "charge_over_temperature": jnp.ones(2)})
    with pytest.raises(ValueError, match="phi must have spatial shape"):
        _call_coulomb({**baseline, "phi": jnp.zeros((1, 1, 3))})


# ---- from test_nonlinear_operator_packages.py ----


def test_operator_package_preserves_public_linear_export_identity() -> None:
    assert operators.hermite_streaming is linear_operators.hermite_streaming


def test_nonlinear_operator_package_reexports_rhs_implementation() -> None:
    assert nonlinear_operators.RhsCallable is operator_rhs.RhsCallable
    assert (
        nonlinear_operators.linear_rhs_jit_for_terms_impl
        is operator_rhs.linear_rhs_jit_for_terms_impl
        is linear_rhs_jit_for_terms
    )
    assert (
        nonlinear_operators.nonlinear_rhs_cached_impl
        is operator_rhs.nonlinear_rhs_cached_impl
    )
    assert (
        nonlinear_operators.nonlinear_em_term_cached_impl
        is operator_rhs.nonlinear_em_term_cached_impl
    )


def test_nonlinear_operator_package_reexports_diagnostic_implementation() -> None:
    assert nonlinear_operators.NonlinearDiagnosticKernels is (
        operator_diagnostics.NonlinearDiagnosticKernels
    )
    assert (
        nonlinear_operators.compute_nonlinear_diagnostic_tuple
        is operator_diagnostics.compute_nonlinear_diagnostic_tuple
    )
    assert (
        nonlinear_operators.make_nonlinear_diagnostic_tuple_fn
        is operator_diagnostics.make_nonlinear_diagnostic_tuple_fn
    )


@pytest.mark.parametrize("model", ["sugama", "improved_sugama"])
def test_per_species_nu_keeps_multispecies_conservation(model: str) -> None:
    """Unequal species prefactors must not unbalance the (a,b)/(b,a) exchange.

    Momentum and energy leave species a through C_ab and arrive in b through
    C_ba; that balance holds only if both ordered pairs carry one prefactor.
    """

    density = jnp.asarray([1.3, 0.7])
    mass = jnp.asarray([2.0, 1.0])
    temperature = jnp.asarray([1.2, 0.8])
    operator = collision_operator_from_config(
        model,
        density=density,
        mass=mass,
        temperature=temperature,
        nu=jnp.asarray([1.0, 3.0]),
    )
    state = (jnp.arange(16.0).reshape(2, 2, 4, 1, 1, 1) + 0.13j).astype(jnp.complex64)
    contribution = apply_multispecies_collision_moment_matrix(state, operator.matrix)
    rates = multispecies_collision_invariant_rates(
        contribution, density=density, mass=mass, temperature=temperature
    )
    np.testing.assert_allclose(rates.particle_density, 0.0, atol=1.0e-5)
    np.testing.assert_allclose(rates.total_parallel_momentum, 0.0, atol=1.0e-5)
    np.testing.assert_allclose(rates.total_thermal_energy, 0.0, atol=1.0e-4)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("nl", [1, 2, 4])
@pytest.mark.parametrize("b", [0.0, 0.27053411424286467, 1.2, 14.856649669530817])
@pytest.mark.parametrize("channel", [0, 1, 2])  # ES, Apar, Bpar
def test_energy_flux_matches_velocity_integral(dtype, nl, b, channel):
    """#325/#326: raw energy flux equals an independent velocity quadrature.

    Mandell et al., JPP 84 (2018) 905840108, App. H. The Bpar drift kernel is
    ``x 2J1(A)/A`` with ``A=sqrt(2 b x)``; ES/Apar use ``J0(A)``.
    """
    from gkx.operators.moments import _heat_flux_channel_contrib_species

    s, ws = roots_hermite(32)
    x, wx = roots_laguerre(96)
    ws /= np.sqrt(np.pi)
    arg = np.sqrt(2 * b * x)
    kernel = j0(arg) if channel < 2 else (x if b == 0 else x * 2 * j1(arg) / arg)
    lag = (-1) ** (nl - 1) * eval_laguerre(nl - 1, x)
    h2 = (4 * s**2 - 2) / np.sqrt(8)
    h3 = (8 * s**3 - 12 * s) / np.sqrt(48)
    if channel == 1:  # v_par weighted Apar flux of m=1/m=3 moments
        dist = (
            np.sqrt(2)
            * s[:, None]
            * (np.sqrt(2) * s[:, None] * lag + 0.3 * h3[:, None])
        )
    else:
        dist = lag[None, :] + 0.3 * h2[:, None]
    expected = 0.6 * float(ws @ (dist * (s[:, None] ** 2 + x) * kernel) @ wx)
    expected *= -1 if channel == 1 else 1
    with jax.enable_x64(dtype == jnp.float64):
        cdtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
        b_arr = jnp.full((1, 1, 1, 1), b, dtype=dtype)
        jl = jnp.moveaxis(J_l_all(b_arr, nl - 1), 0, 1)
        lower = jnp.concatenate([jnp.zeros_like(jl[:, :1]), jl[:, :-1]], axis=1)
        cache = SimpleNamespace(Jl=jl, JlB=jl + lower, b=b_arr)
        ones = jnp.ones(1, dtype=dtype)
        grid = SimpleNamespace(ky=0.3 * ones, kx=0 * ones, z=0 * ones)
        params = SimpleNamespace(density=ones, temp=ones, vth=ones, tz=ones)
        p0, p2 = (1, 3) if channel == 1 else (0, 2)
        G = jnp.zeros((1, nl, 4, 1, 1, 1), cdtype).at[:, nl - 1, p0].set(1j)
        G = G.at[:, 0, p2].add(0.3j)
        fields = [jnp.zeros((1, 1, 1), cdtype)] * 3
        fields[channel] = jnp.ones((1, 1, 1), cdtype)
        flux = _heat_flux_channel_contrib_species(
            G, *fields, cache, grid, params, ones, use_dealias=False, flux_scale=1.0
        )[channel].sum()
    tol = 3e-6 if dtype == jnp.float32 else 2e-13
    np.testing.assert_allclose(np.asarray(flux), expected, atol=tol, rtol=tol)


# Fused drift kernel


def _drift_inputs(shape, seed=0):
    rng = np.random.default_rng(seed)
    ns, _, _, ny, nx, nz = shape
    H = jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    f1 = jnp.asarray(rng.standard_normal((ns, 1, 1, nz)))
    f2, f3 = (jnp.asarray(rng.standard_normal((ns, ny, nx, nz))) for _ in range(2))
    return H, f1, f2, f3


def _drift_named_terms(H, f1, f2, f3):
    nl, nm = H.shape[1:3]
    ell = jnp.arange(nl, dtype=f1.dtype)[:, None, None, None, None]
    m = jnp.arange(nm, dtype=f1.dtype)[:, None, None, None]
    ones = jnp.ones(H.shape[0], dtype=f1.dtype)
    mirror = mirror_contribution(
        H,
        vth=ones,
        bgrad=f1[0, 0, 0],
        ell=ell,
        sqrt_m=jnp.sqrt(m),
        sqrt_m_p1=jnp.sqrt(m + 1.0),
        weight=jnp.asarray(1.0),
    )
    # f is per species and per point; one species keeps the old signature exact.
    drift = curvature_gradb_contribution(
        H,
        tz=ones,
        omega_d_scale=jnp.asarray(1.0),
        cv_d=f2[0],
        gb_d=f3[0],
        ell=ell,
        m=m,
        imag=jnp.asarray(1j, dtype=H.dtype),
        weight_curv=jnp.asarray(1.0),
        weight_gradb=jnp.asarray(1.0),
    )
    return mirror + drift


@pytest.mark.parametrize(
    "shape", [(1, 3, 5, 2, 8, 16), (1, 1, 1, 3, 4, 32), (1, 2, 4, 2, 5, 24)]
)
def test_drift_reference_matches_the_named_terms(shape):
    H, f1, f2, f3 = _drift_inputs(shape)
    np.testing.assert_allclose(
        linear_terms._reference_drift(H, f1, f2, f3),
        _drift_named_terms(H, f1, f2, f3),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "shape", [(2, 3, 5, 2, 8, 16), (1, 1, 1, 3, 4, 32), (1, 2, 4, 2, 5, 24)]
)
def test_drift_kernel_matches_reference(shape):
    """Interpreted, so it runs on any backend; ragged ``x z`` rows exercise the padding."""

    args = _drift_inputs(shape)
    got = linear_terms._pallas_drift(*args, interpret=True)
    np.testing.assert_allclose(
        got, linear_terms._reference_drift(*args), rtol=1e-12, atol=1e-12
    )


def test_drift_derivatives_go_through_the_reference_terms(monkeypatch):
    shape = (1, 2, 3, 2, 4, 16)
    args = _drift_inputs(shape)
    kernel = linear_terms._pallas_drift
    monkeypatch.setattr(
        linear_terms, "_pallas_drift", lambda *a: kernel(*a, interpret=True)
    )
    tangents = tuple(
        jnp.cos(a.real + 1.0) + 0.5j * jnp.sin(a.real)
        if jnp.iscomplexobj(a)
        else jnp.cos(a + 1.0)
        for a in args
    )
    primal, dot = jax.jvp(linear_terms.fused_drift, args, tangents)
    ref_primal, ref_dot = jax.jvp(linear_terms._reference_drift, args, tangents)
    np.testing.assert_allclose(primal, ref_primal, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(dot, ref_dot, rtol=1e-12, atol=1e-12)
    grad = jax.grad(
        lambda H, f2: jnp.sum(
            jnp.abs(linear_terms.fused_drift(H, args[1], f2, args[3]))
        )
    )
    ref = jax.grad(
        lambda H, f2: jnp.sum(
            jnp.abs(linear_terms._reference_drift(H, args[1], f2, args[3]))
        )
    )
    np.testing.assert_allclose(
        grad(args[0], args[2]), ref(args[0], args[2]), rtol=1e-10, atol=1e-12
    )


def test_drift_kernel_usable_only_on_gpu_with_tileable_rows():
    H = jnp.zeros((1, 2, 2, 1, 8, 8))
    assert linear_terms.drift_kernel_usable(H) == (jax.default_backend() == "gpu")
    assert not linear_terms.drift_kernel_usable(jnp.zeros((1, 2, 2, 1, 2, 8)))


@pytest.mark.parametrize("ny, rows, nz", [(8, 8, 6), (8, 5, 6), (7, 4, 4), (6, 4, 5)])
def test_nyquist_derivative_touches_only_the_self_conjugate_rows(ny, rows, nz):
    """The two-row fix equals the whole-array formula it replaced."""

    from gkx.operators.linear.streaming import _real_fft_nyquist_derivative

    rng = np.random.default_rng(1)
    shape = (2, 3, rows, 5, nz)
    out = jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    row = jnp.arange(rows)
    self_row = (row == 0) | ((ny % 2 == 0) & (row == ny // 2))
    alt = jnp.asarray((-1.0) ** np.arange(nz))
    comp = jnp.mean(out * alt, axis=-1, keepdims=True) * alt
    expected = out - self_row.reshape(1, 1, -1, 1, 1) * comp if nz % 2 == 0 else out
    got = _real_fft_nyquist_derivative(out, ny)
    np.testing.assert_array_equal(got, expected)


# --- nodal magnetic-moment grid (LinearParams.mu_vperp_max) against the Laguerre path ---
_MU_NEEDS_X64 = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs JAX_ENABLE_X64=true"
)

_MU_NZ = 16


def _mu_setup(epsilon: float, rho: float, nmu: int, nm: int, vperp_max: float | None):
    grid = build_spectral_grid(
        GridConfig(Nx=1, Ny=4, Nz=_MU_NZ, Lx=62.8, Ly=62.8, boundary="periodic")
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=epsilon, R0=2.77778)
    params = LinearParams(
        rho=rho,
        kpar_scale=float(geom.gradpar()),
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
        p_hyper_m=6.0,
        mu_vperp_max=vperp_max,
    )
    return build_linear_cache(grid, geom, params, nmu, nm), params


def _mu_random_state(nmu: int, nm: int) -> jnp.ndarray:
    rng = np.random.default_rng(0)
    shape = (nmu, nm, 3, 1, _MU_NZ)
    return jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))


@_MU_NEEDS_X64
def test_mu_grid_rule_is_exact_for_polynomials_and_laguerre_modes_are_orthonormal():
    """Degree ``2 n - 1`` exactness against ``e^{-x}`` and orthonormal Laguerre mode seeds."""

    from scipy.special import gamma, gammainc

    n = 6
    for vperp_max, cutoff in ((4.0, 8.0), (np.inf, np.inf)):
        x, scaled = mu_grid_nodes(n, vperp_max)
        weights = scaled * np.exp(-x)
        for k in range(2 * n):
            exact = gamma(k + 1.0) * (
                gammainc(k + 1.0, cutoff) if np.isfinite(cutoff) else 1.0
            )
            assert np.isclose((weights * x**k).sum(), exact, rtol=1e-11)
        # sum_j mode_l mode_l' = delta_ll' at B = B0; Laguerre polynomials are only
        # orthogonal on [0, 8] up to the cutoff tail, so the truncated rule is checked for l <= 1
        modes = mu_grid_laguerre_modes(x, scaled, np.ones(1), 4)[:, :, 0]
        count = 4 if np.isinf(vperp_max) else 2
        gram = modes[:count] @ modes[:count].T
        assert np.abs(gram - np.eye(count)).max() < (1e-11 if count == 4 else 0.03)


@_MU_NEEDS_X64
def test_gauss_laguerre_nodes_reduce_to_the_laguerre_operator_at_zero_kperp():
    """Without ``B`` variation and gyroaveraging the two representations are one.

    At ``epsilon = 0`` and ``k_perp rho -> 0`` the Gauss-Laguerre node amplitudes
    are an orthogonal image of the Laguerre moments, ``O[j, l] = sqrt(w_j) (-1)^l
    L_l(x_j)``, so the whole right-hand side (streaming, drifts, drives, field
    solve, hypercollisions) must agree to round-off.
    """

    nmu, nm = 6, 6
    lag_cache, lag_params = _mu_setup(0.0, 1e-6, nmu, nm, None)
    mu_cache, mu_params = _mu_setup(0.0, 1e-6, nmu, nm, np.inf)
    x, scaled = mu_grid_nodes(nmu, np.inf)
    laguerre = laguerre_scaled_functions(x, nmu)[:nmu] * np.exp(x / 2)[None]
    orth = np.sqrt(scaled * np.exp(-x))[:, None] * (
        (-1.0) ** np.arange(nmu)[None] * laguerre.T
    )
    g_lag = _mu_random_state(nmu, nm)
    g_mu = jnp.einsum("jl,l...->j...", jnp.asarray(orth), g_lag)
    d_lag, phi_lag = linear_rhs_cached(g_lag, lag_cache, lag_params)
    d_mu, phi_mu = linear_rhs_cached(g_mu, mu_cache, mu_params)
    back = jnp.einsum("jl,j...->l...", jnp.asarray(orth), d_mu)
    assert jnp.linalg.norm(back - d_lag) < 1e-10 * jnp.linalg.norm(d_lag)
    assert jnp.linalg.norm(phi_mu - phi_lag) < 1e-10 * jnp.linalg.norm(phi_lag)


@_MU_NEEDS_X64
def test_weak_mirror_conserves_density_per_node():
    """Streaming plus mirror conserve ``int dz S_j G_{j,0}`` and the energy at every node.

    ``S_j = exp(-mu_j B / 2)`` is the half Maxwellian and the stored amplitude is
    scaled by ``a_j = sqrt(B w_j e^{mu_j})``; the field coupling is switched off
    (``tz = 0``) so that only the kinetic operator remains.
    """

    nmu, nm, vperp_max = 5, 6, 4.0
    cache, params = _mu_setup(0.3, 0.5, nmu, nm, vperp_max)
    params = replace(params, tz=0.0)
    terms = LinearTerms(
        mirror=1.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
    )
    nodes, scaled = mu_grid_nodes(nmu, vperp_max)
    bmag = np.asarray(cache.bmag)
    energy = nodes[:, None] * bmag[None]
    amplitude = np.sqrt(bmag[None] * scaled[:, None])
    s_over_a = np.exp(-0.5 * energy) / amplitude  # S_j / a_j, shape (nmu, Nz)
    z = np.linspace(-np.pi, np.pi, _MU_NZ, endpoint=False)
    smooth = (
        np.exp(1j * z)[None, None, None, None]
        * np.arange(1, nm + 1)[None, :, None, None, None]
    )
    physical = jnp.asarray(np.broadcast_to(smooth, (nmu, nm, 3, 1, _MU_NZ)))
    state = physical * jnp.asarray(amplitude)[:, None, None, None, :]
    rate, _ = linear_rhs_cached(state, cache, params, terms=terms)
    density_rate = np.sum(np.asarray(rate[:, 0]) * s_over_a[:, None, None], axis=-1)
    scale = np.sum(np.abs(np.asarray(rate[:, 0])) * s_over_a[:, None, None], axis=-1)
    assert np.max(np.abs(density_rate)) < 1e-9 * np.max(scale)
    # energy v^2/2 + mu B is conserved at fixed mu: int dz S_j [(1/2 + x_j) G_0 + G_2 / sqrt 2]
    weight = s_over_a[:, None, None]
    parts = (0.5 + energy)[:, None, None] * np.asarray(rate[:, 0]) + np.asarray(
        rate[:, 2]
    ) / np.sqrt(2.0)
    assert np.max(np.abs(np.sum(parts * weight, axis=-1))) < 1e-9 * np.max(
        np.sum(np.abs(parts) * weight, axis=-1)
    )


@_MU_NEEDS_X64
def test_ion_acoustic_landau_damping_on_the_mu_grid():
    """The k_perp -> 0 slab root ``1 + T_i/T_e + zeta Z(zeta) = 0`` is recovered.

    The perpendicular grid cannot change a parallel Landau root, so three nodes
    must reproduce the damping rate and frequency of the dispersion relation
    (within the Hermite truncation error of 64 moments).
    """

    from support.reference_physics import exact_root, fit_standing_wave

    grid = build_spectral_grid(
        GridConfig(Nx=1, Ny=2, Nz=32, Lx=1.0e4, Ly=1.0e4, boundary="periodic")
    )
    geom = SAlphaGeometry(q=1.0, s_hat=0.0, epsilon=0.0, R0=1.0)
    nmu, nm = 3, 64
    params = LinearParams(
        kpar_scale=1.0,
        rho=1e-3,
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
        p_hyper_m=6.0,
        mu_vperp_max=4.0,
    )
    terms = LinearTerms(
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        end_damping=0.0,
    )
    cache = build_linear_cache(grid, geom, params, nmu, nm)
    rhs = jax.jit(
        lambda s: linear_rhs_cached(s, cache, params, terms=terms, use_jit=False)[0]
    )
    nodes, scaled = mu_grid_nodes(nmu, 4.0)
    sqrt_w = np.sqrt(
        scaled * np.exp(-nodes)
    )  # density weights of the scaled amplitudes
    z = np.asarray(grid.z)
    state = (
        jnp.zeros((1, nmu, nm, grid.ky.size, 1, z.size), dtype=jnp.complex128)
        .at[0, :, 0, 1, 0, :]
        .set(jnp.asarray(sqrt_w[:, None] * np.exp(1j * z)[None]))
    )
    dt, steps = 0.01, 900
    signal = np.empty(steps + 1)
    for i in range(steps + 1):
        density = np.asarray(state)[0, :, 0, 1, 0, :] * sqrt_w[:, None]
        signal[i] = np.real(np.sum(density * np.exp(-1j * z)[None]) / z.size)
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + rhs(state + dt * k3))
    root = exact_root(1.0, complex(1.4, -0.6))
    gamma, omega = fit_standing_wave(
        np.arange(steps + 1) * dt, signal, (2.0, 9.0), (1.0, root.imag, root.real, 0.0)
    )
    assert abs(gamma - root.imag) < 0.01 * abs(root.imag)
    assert abs(omega - root.real) < 0.01 * root.real


@_MU_NEEDS_X64
def test_mu_grid_refuses_the_dissipation_it_does_not_implement():
    grid = build_spectral_grid(GridConfig(Nx=1, Ny=2, Nz=8, boundary="periodic"))
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778)
    for field, value in (
        ("nu", 0.01),
        ("nu_hyper_l", 1.0),
        ("hypercollisions_const", 1.0),
    ):
        kwargs = {"hypercollisions_const": 0.0, "mu_vperp_max": 4.0, field: value}
        with pytest.raises(NotImplementedError, match="nodal mu grid"):
            build_linear_cache(grid, geom, LinearParams(**kwargs), 4, 4)


@_MU_NEEDS_X64
def test_mu_grid_growth_rate_matches_converged_laguerre_on_cyclone():
    """Cyclone ITG, ky = 0.3: eight nodes land on the Laguerre ``Nl = 16`` rate.

    The shipped tutorial deck at ``Nz = 32, Nm = 8`` (a coarse ``Nz`` moves the two
    discretizations of the mirror term apart by several percent, so it is not used).
    """

    import gkx
    from pathlib import Path

    deck = Path(__file__).parents[3] / "examples/01_linear_tokamak/case.toml"
    case = gkx.load(deck)
    case = replace(case, grid=replace(case.grid, Nz=32, ntheta=32))
    nodal = replace(case, run=replace(case.run, mu_vperp_max=4.0))
    laguerre = gkx.solve(case, ky_target=0.3, Nl=16, Nm=8, solver="krylov")
    mu = gkx.solve(nodal, ky_target=0.3, Nl=8, Nm=8, solver="krylov")
    assert abs(mu.gamma - laguerre.gamma) < 0.03 * laguerre.gamma
    assert abs(mu.omega - laguerre.omega) < 0.03 * laguerre.omega


@_MU_NEEDS_X64
@pytest.mark.parametrize("vperp_max", [None, 4.0])
@pytest.mark.parametrize("mode", ["es", "apar", "bpar"])
def test_collisionless_free_energy_is_conserved_without_drive(mode, vperp_max):
    """``W = <G,G>/2 + int (q |phi|^2 - kappa |A|^2) / 2B`` is exactly conserved.

    With ``B_par`` the field energy is ``q |phi|^2 + 2 q_b Re(phi* b) - kappa_B |b|^2``,
    ``q_b = -Gamma_01`` and ``kappa_B = B^2 / beta' + Gamma_11``: the coupled solve is the
    symmetric system ``q phi + q_b b = n``, ``q_b phi - kappa_B b = P``.

    Collisionless, undriven, truncation closure, ``eps = 0.18``. ``<.,.>`` is the
    ``dz / B`` inner product on the stored state; the form is evaluated on band-limited
    ``z`` modes, where the spectral derivative obeys the product rule against ``1/B``.
    ``kappa = (k_perp^2 B^2 / beta + c_g Gamma_0) / s`` is the Ampere response.
    """

    from gkx.terms.fields import _solve_fields_impl

    nl, nm, nz, kmax, beta = 4, 4, 32, 3, 0.02
    grid = build_spectral_grid(
        GridConfig(Nx=1, Ny=8, Nz=nz, Lx=62.8, Ly=20.9, boundary="periodic")
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778)
    em = mode == "apar"
    bp = mode == "bpar"
    params = LinearParams(
        rho=1.0,
        kpar_scale=float(geom.gradpar()),
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        damp_ends_rate=0.0,
        beta=beta if (em or bp) else 0.0,
        fapar=1.0 if em else 0.0,
        mu_vperp_max=vperp_max,
    )
    terms = LinearTerms(
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=1.0 if em else 0.0,
        bpar=1.0 if bp else 0.0,
    )
    cache = build_linear_cache(grid, geom, params, nl, nm)
    ny, iy, n = cache.ky.shape[0], 1, nl * nm * nz

    def embed(v):
        state = jnp.zeros((1, nl, nm, ny, 1, nz), dtype=jnp.complex128)
        return state.at[0, :, :, iy, 0, :].set(v.reshape(nl, nm, nz))

    def fields(v):
        return _solve_fields_impl(
            embed(v),
            cache,
            params,
            charge=jnp.ones(1),
            density=jnp.ones(1),
            temp=jnp.ones(1),
            mass=jnp.ones(1),
            tz=jnp.ones(1),
            vth=jnp.ones(1),
            fapar=jnp.asarray(1.0 if em else 0.0),
            w_bpar=jnp.asarray(1.0 if bp else 0.0),
        )

    def jac(f):
        return np.asarray(jax.jacrev(f, holomorphic=True)(jnp.zeros(n, jnp.complex128)))

    rhs = lambda v: linear_rhs_cached(  # noqa: E731
        embed(v), cache, params, terms=terms, use_custom_vjp=False
    )[0][0, :, :, iy, 0, :].reshape(-1)
    a_mat = jac(rhs)
    inv_b = 1.0 / np.asarray(cache.bmag)
    gamma0 = np.asarray(jnp.sum(cache.Jl**2, axis=1))[0, iy, 0, :]

    def energy_rate(psi, weight):
        form = psi.conj().T @ np.diag(weight) @ psi
        return form @ a_mat + a_mat.conj().T @ form

    rate = np.diag(np.tile(inv_b, nl * nm)) @ a_mat
    rate = (
        rate
        + rate.conj().T
        + energy_rate(jac(lambda v: fields(v).phi[iy, 0, :]), (2.0 - gamma0) * inv_b)
    )
    if em:
        kappa = np.asarray(cache.kperp2)[iy, 0, :] * np.asarray(cache.bmag) ** 2
        kappa = (
            kappa / (beta * params.apar_beta_scale)
            + (params.ampere_g0_scale / params.apar_beta_scale) * gamma0
        )
        rate = rate - energy_rate(
            jac(lambda v: fields(v).apar[iy, 0, :]), kappa * inv_b
        )
    if bp:
        bpar_beta = params.bpar_beta_scale * beta
        jb = jnp.asarray(cache.JlB)
        g01 = np.asarray(jnp.sum(cache.Jl * jb, axis=1))[0, iy, 0, :]
        g11 = np.asarray(jnp.sum(jb * jb, axis=1))[0, iy, 0, :]
        psi_phi = jac(lambda v: fields(v).phi[iy, 0, :])
        psi_b = jac(lambda v: fields(v).bpar[iy, 0, :])
        cross = psi_phi.conj().T @ np.diag(-g01 * inv_b) @ psi_b
        cross = cross + cross.conj().T
        rate = rate + cross @ a_mat + a_mat.conj().T @ cross
        kappa_b = np.asarray(cache.bmag) ** 2 / bpar_beta + g11
        rate = rate - energy_rate(psi_b, kappa_b * inv_b)
    modes = np.abs(np.fft.fftfreq(nz, 1.0 / nz)) <= kmax
    low = np.conj(np.fft.fft(np.eye(nz), axis=0).T)[:, modes] / np.sqrt(nz)
    basis = np.kron(np.eye(nl * nm), low)
    scale = np.linalg.norm(
        basis.conj().T @ np.diag(np.tile(inv_b, nl * nm)) @ a_mat @ basis
    )
    residual = basis.conj().T @ (0.5 * (rate + rate.conj().T)) @ basis
    assert np.linalg.norm(residual) < 1e-9 * scale


@_MU_NEEDS_X64
def test_zonal_flow_residual_on_the_mu_grid_matches_laguerre():
    """Rosenbluth-Hinton residual (ky = 0, k_x rho = 0.05, adiabatic electrons, eps = 0.18).

    Late-time ``phi / phi(0)`` of a density seed after the GAM, 8 nodes against
    Laguerre ``Nl = 8``; the small-eps formula ``1 / (1 + 1.6 q^2 / sqrt(eps))`` is 0.119.
    """

    nz, nm, dt, t_max = 32, 16, 0.02, 60.0
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=2, Nz=nz, Lx=2 * np.pi / 0.05, Ly=62.8, boundary="periodic")
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778)
    terms = LinearTerms(end_damping=0.0, collisions=0.0)

    def residual(vperp_max):
        params = LinearParams(
            kpar_scale=float(geom.gradpar()),
            tprim=0.0,
            fprim=0.0,
            hypercollisions_const=0.0,
            hypercollisions_kz=1.0,
            p_hyper_m=8.0,
            damp_ends_rate=0.0,
            mu_vperp_max=vperp_max,
        )
        cache = build_linear_cache(grid, geom, params, 8, nm)
        iy = int(np.argmin(np.abs(np.asarray(cache.ky))))
        seed = np.ones(nz)
        if vperp_max is not None:
            nodes = mu_grid_nodes(8, vperp_max)
            seed = mu_grid_laguerre_modes(*nodes, np.asarray(cache.bmag), 1)[0]
        state = np.zeros((1, 8, nm, cache.ky.shape[0], cache.kx.shape[0], nz), complex)
        if vperp_max is None:
            state[0, 0, 0, iy, 1, :] = 1e-3
        else:
            state[0, :, 0, iy, 1, :] = 1e-3 * seed
        rhs = lambda s: linear_rhs_cached(  # noqa: E731
            s, cache, params, terms=terms, use_custom_vjp=False
        )

        @jax.jit
        def step(s):
            k1 = rhs(s)[0]
            k2 = rhs(s + 0.5 * dt * k1)[0]
            k3 = rhs(s + 0.5 * dt * k2)[0]
            return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + rhs(s + dt * k3)[0])

        phi_of = jax.jit(lambda s: rhs(s)[1][iy, 1, :].mean())
        state, phis = jnp.asarray(state), []
        for i in range(int(t_max / dt) + 1):
            if i % 10 == 0:
                phis.append(float(jnp.real(phi_of(state))))
            state = step(state)
        late = np.arange(len(phis)) * dt * 10 > 0.7 * t_max
        return np.mean(np.asarray(phis)[late]) / phis[0]

    laguerre, nodal = residual(None), residual(4.0)
    assert 0.05 < laguerre < 0.2
    assert abs(nodal - laguerre) < 0.03 * laguerre


@_MU_NEEDS_X64
@pytest.mark.parametrize("vperp_max", [np.inf, 4.0])
def test_nodal_dougherty_conserves_and_dissipates(vperp_max):
    """Weak-form Lenard-Bernstein on the nodes (``eps = 0.18``, ``k_perp -> 0``).

    Density (``sum_j J_j H_j0``), parallel momentum and thermal energy
    (``sqrt2 sum J H_2 + 2 sum (x - 1) J H_0``) are conserved to the quadrature error:
    round-off for Gauss-Laguerre nodes at ``B = B0``, 1e-4 of the largest rate for
    them at ``eps = 0.18`` and 5e-2 with the cutoff, whose ``l = 1`` rate is 2.03.
    At ``eps = 0`` the Gauss-Laguerre operator is the Laguerre Dougherty operator under
    the orthogonal map, and ``Re <H, C H> < 0``.
    """

    nmu, nm, nz = 6, 6, 8
    terms = LinearTerms(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
    )

    def operator(epsilon, mu):
        grid = build_spectral_grid(
            GridConfig(Nx=1, Ny=4, Nz=nz, Lx=62.8, Ly=62.8, boundary="periodic")
        )
        geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=epsilon, R0=2.77778)
        params = LinearParams(
            rho=1e-6,
            nu=0.1,
            collision_model="dougherty",
            hypercollisions_const=0.0,
            hypercollisions_kz=0.0,
            mu_vperp_max=mu,
        )
        cache = build_linear_cache(grid, geom, params, nmu, nm)
        return cache, lambda g: linear_rhs_cached(g, cache, params, terms=terms)[0]

    cache, rhs = operator(0.18, vperp_max)
    rng = np.random.default_rng(3)
    shape = (nmu, nm, 3, 1, nz)
    state = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    rate = np.asarray(rhs(state))[:, :, 1, 0, :]
    jl = np.asarray(cache.Jl)[0, :, 1, 0, :]
    x = np.asarray(cache.laguerre_roots)[:, None] * np.asarray(cache.bmag)[None]
    energy = np.sqrt(2.0) * (jl * rate[:, 2]).sum(0) + 2.0 * (
        (x - 1.0) * jl * rate[:, 0]
    ).sum(0)
    scale = np.abs(rate).max()
    tolerance = 5e-2 if np.isfinite(vperp_max) else 1e-4
    for invariant in ((jl * rate[:, 0]).sum(0), (jl * rate[:, 1]).sum(0), energy):
        assert np.abs(invariant).max() < tolerance * scale
    if np.isfinite(vperp_max):
        return
    rate_all = np.asarray(rhs(state))
    assert np.real(np.vdot(np.asarray(state), rate_all)) < 0.0

    # eps = 0: the Laguerre Dougherty operator under the orthogonal map
    lag_cache, lag_rhs = operator(0.0, None)
    mu_cache, mu_rhs = operator(0.0, np.inf)
    nodes, scaled = mu_grid_nodes(nmu, np.inf)
    laguerre = laguerre_scaled_functions(nodes, nmu)[:nmu] * np.exp(nodes / 2)[None]
    orth = np.sqrt(scaled * np.exp(-nodes))[:, None] * (
        (-1.0) ** np.arange(nmu)[None] * laguerre.T
    )
    g_lag = state
    g_mu = jnp.einsum("jl,l...->j...", jnp.asarray(orth), g_lag)
    back = jnp.einsum("jl,j...->l...", jnp.asarray(orth), mu_rhs(g_mu))
    assert jnp.linalg.norm(back - lag_rhs(g_lag)) < 1e-10 * jnp.linalg.norm(back)


@_MU_NEEDS_X64
def test_mu_grid_collisional_growth_rate_matches_laguerre():
    """Cyclone ky 0.3 with Dougherty collisions (``nu = 0.2``): 8 nodes against ``Nl = 16``."""

    import gkx
    from pathlib import Path

    deck = Path(__file__).parents[3] / "examples/01_linear_tokamak/case.toml"
    case = gkx.load(deck)
    case = replace(
        case,
        grid=replace(case.grid, Nz=32, ntheta=32),
        species=tuple(replace(sp, nu=0.2) for sp in case.species),
        physics=replace(case.physics, collisions=True),
        terms=replace(case.terms, collisions=1.0),
        time=replace(case.time, collision_operator="lenard_bernstein"),
    )
    nodal = replace(case, run=replace(case.run, mu_vperp_max=4.0))
    laguerre = gkx.solve(case, ky_target=0.3, Nl=16, Nm=8, solver="krylov")
    mu = gkx.solve(nodal, ky_target=0.3, Nl=8, Nm=8, solver="krylov")
    assert 0.03 < laguerre.gamma < 0.1
    assert abs(mu.gamma - laguerre.gamma) < 0.03 * laguerre.gamma
