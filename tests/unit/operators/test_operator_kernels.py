"""Focused operator-kernel and reduced full-operator regression checks."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import pytest
from scipy.special import eval_genlaguerre, eval_laguerre, j0, jv

from gkx.config import GridConfig
from gkx.operators.collision import CollisionContext
from gkx.core.grid import build_spectral_grid
from gkx.core.velocity import (
    J_l_all,
    associated_bessel_laguerre_coefficients,
    bessel_laguerre_kernels,
    gamma0,
    laguerre_gyroaverage_neighbors,
    sum_Jl2,
)
from gkx.diagnostics.analysis import fit_growth_rate
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.operators.linear.rhs import linear_rhs_cached
from gkx.solvers.linear.integrators import integrate_linear
from gkx.operators import hermite_streaming
from gkx.operators.linear import (
    apply_collision_moment_matrix,
    apply_finite_wavelength_coulomb_moment_operator,
    apply_multispecies_collision_moment_matrix,
    assemble_drift_kinetic_improved_sugama_matrix,
    assemble_drift_kinetic_sugama_matrix,
    COLLISION_OPERATOR_NAMES,
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
    drift_kinetic_coulomb_six_moment_contribution,
    drift_kinetic_sugama_six_moment_contribution,
)
from gkx.runtime import run_runtime_scan
from gkx.workflows.runtime.toml import load_runtime_from_toml


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
        ("coulomb", drift_kinetic_coulomb_six_moment_contribution),
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
    direct = drift_kinetic_coulomb_six_moment_contribution(
        state, nu=jnp.asarray(0.3)
    ) * (1.0 + 0.5 * jnp.clip(kperp, grid[0], grid[-1]))
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


def test_drift_kinetic_collision_operators_obey_h_theorem_and_conserve_density() -> None:
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


def test_collision_operator_from_config_selects_validates_and_stays_dissipative() -> None:
    # TOML ``collision_operator`` selection: the diagonal defaults resolve to
    # None (the linear RHS keeps its built-in Lenard-Bernstein term, re-enabled
    # exactly when collision_operator is None), the moment-matrix names build a
    # dense drift-kinetic operator, and an unknown name is rejected. The
    # selected operators must remain physically valid (H-theorem).
    from gkx.config import TimeConfig

    assert TimeConfig().collision_operator == "none"
    density = jnp.asarray([1.0])
    mass = jnp.asarray([1.0])
    temperature = jnp.asarray([1.0])

    assert (
        collision_operator_from_config(
            "none", density=density, mass=mass, temperature=temperature
        )
        is None
    )
    # Case/whitespace-insensitive; alias for the built-in diagonal term.
    assert (
        collision_operator_from_config(
            "  Lenard_Bernstein ", density=density, mass=mass, temperature=temperature
        )
        is None
    )

    for name in ("sugama", "improved_sugama", "coulomb"):
        operator = collision_operator_from_config(
            name, density=density, mass=mass, temperature=temperature
        )
        assert isinstance(operator, DriftKineticMomentCollisionOperator)
        # Single species -> (Ns, Ns, 8, 8) ordered-pair moment matrix.
        assert operator.matrix.shape == (1, 1, 8, 8)
        block = np.asarray(operator.matrix).reshape(8, 8)
        eigenvalues = np.linalg.eigvalsh(0.5 * (block + block.T))
        # H-theorem preserved (tolerance covers float32 eigenvalue noise ~2e-8;
        # any real growing direction would be O(0.1)).
        assert float(eigenvalues.max()) < 1.0e-6, name
        assert float(eigenvalues.min()) < -0.1, name

    # The three moment models are physically distinct operators.
    matrices = {
        name: np.asarray(
            collision_operator_from_config(
                name, density=density, mass=mass, temperature=temperature
            ).matrix
        )
        for name in ("sugama", "improved_sugama", "coulomb")
    }
    assert not np.allclose(matrices["coulomb"], matrices["sugama"])
    assert not np.allclose(matrices["coulomb"], matrices["improved_sugama"])

    assert set(COLLISION_OPERATOR_NAMES) == {
        "none",
        "lenard_bernstein",
        "sugama",
        "improved_sugama",
        "coulomb",
        "coulomb_finite_kperp",
    }
    with pytest.raises(ValueError, match="collision_operator must be one of"):
        collision_operator_from_config(
            "bogus", density=density, mass=mass, temperature=temperature
        )

    # The tabulated Coulomb table is a like-species vertical slice; a
    # multispecies request must be refused rather than silently extrapolated.
    with pytest.raises(ValueError, match="like-species"):
        collision_operator_from_config(
            "coulomb",
            density=jnp.asarray([1.0, 1.0]),
            mass=jnp.asarray([1.0, 2.0]),
            temperature=jnp.asarray([1.0, 1.0]),
        )


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


def test_drift_kinetic_coulomb_matrix_scales_with_collision_frequency() -> None:
    # The tabulated coefficients are normalized to unit frequency and must be
    # scaled by the same dimensionless nu_aa = n / (sqrt(m) T**1.5) that the
    # Sugama assemblers use, so all three drift-kinetic models share one
    # normalization.
    from gkx.operators.linear.collision_tables import (
        assemble_drift_kinetic_coulomb_matrix,
    )

    unit = assemble_drift_kinetic_coulomb_matrix(
        jnp.asarray([1.0]), jnp.asarray([1.0]), jnp.asarray([1.0])
    )
    assert np.allclose(
        np.asarray(unit).reshape(8, 8),
        np.asarray(load_collision_moment_matrix("coulomb")),
        atol=1e-6,
    )

    density, mass, temperature = 3.0, 4.0, 2.0
    scaled = assemble_drift_kinetic_coulomb_matrix(
        jnp.asarray([density]), jnp.asarray([mass]), jnp.asarray([temperature])
    )
    expected = density / (np.sqrt(mass) * temperature**1.5)
    assert np.allclose(np.asarray(scaled), expected * np.asarray(unit), rtol=1e-5)

    for bad in ("density", "mass", "temperature"):
        values = {
            "density": jnp.asarray([1.0]),
            "mass": jnp.asarray([1.0]),
            "temperature": jnp.asarray([1.0]),
        }
        values[bad] = jnp.asarray([-1.0])
        with pytest.raises(ValueError, match=f"{bad} must be positive"):
            assemble_drift_kinetic_coulomb_matrix(**values)


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
    state = (
        jnp.arange(16, dtype=real_dtype).reshape(2, 2, 4, 1, 1, 1) + 0.13j
    ).astype(complex_dtype)
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
    np.testing.assert_allclose(rates.total_thermal_energy, 0.0, atol=7.0e-6)
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


def test_drift_kinetic_collision_model_is_blocked_for_short_wave_itg() -> None:
    grid = build_spectral_grid(GridConfig(Nx=2, Ny=8, Nz=16, Lx=10.0, Ly=20.0))
    geometry = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18)
    params = LinearParams(fprim=2.2, tprim=6.9, damp_ends_amp=0.0)
    random = np.random.default_rng(7)
    state = jnp.asarray(
        1.0e-5
        * (
            random.standard_normal((2, 4, 8, 2, 16))
            + 1j * random.standard_normal((2, 4, 8, 2, 16))
        ),
        dtype=jnp.complex64,
    )
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


def test_sum_jl2_monotone_in_lmax() -> None:
    """Truncated sum of J_l^2 should increase with l_max."""
    b = jnp.array([0.2, 1.5])

    s2 = sum_Jl2(b, l_max=2)
    s5 = sum_Jl2(b, l_max=5)

    assert jnp.all(s5 >= s2)


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
    base_cfg, _ = load_runtime_from_toml("examples/linear/axisymmetric/cyclone.toml")
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
