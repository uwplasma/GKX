"""Fast invariants for Hermite-Laguerre moment primitives."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gkx.core.velocity import hermite_ladder_coeffs
import gkx.operators.linear.moments as linear_moments
import gkx.operators.linear.streaming as streaming_operators


def _complex_state(shape: tuple[int, ...]) -> jnp.ndarray:
    values = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    return (jnp.sin(0.17 * values) + 1j * jnp.cos(0.11 * values)).astype(jnp.complex64)


def _operator_matrix(fn, shape: tuple[int, ...]) -> np.ndarray:
    cols = []
    size = int(np.prod(shape))
    for idx in range(size):
        basis = (
            jnp.zeros(shape, dtype=jnp.float32)
            .reshape(-1)
            .at[idx]
            .set(1.0)
            .reshape(shape)
        )
        cols.append(np.asarray(fn(basis)).reshape(-1))
    return np.stack(cols, axis=1)


def test_linear_moments_reuse_canonical_velocity_space_algebra() -> None:
    """Moment assembly must reuse, rather than copy, velocity recurrences."""

    state = _complex_state((2, 4, 5, 2, 3, 4))
    assert linear_moments.shift_axis is streaming_operators.shift_axis
    assert linear_moments.apply_hermite_v is streaming_operators.apply_hermite_v
    assert linear_moments.apply_hermite_v2 is streaming_operators.apply_hermite_v2
    assert linear_moments.apply_laguerre_x is streaming_operators.apply_laguerre_x

    np.testing.assert_allclose(
        np.asarray(linear_moments.shift_axis(state, 1, axis=-4)),
        np.asarray(streaming_operators.shift_axis(state, 1, axis=-4)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(linear_moments.shift_axis(state, -2, axis=-5)),
        np.asarray(streaming_operators.shift_axis(state, -2, axis=-5)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(linear_moments.apply_hermite_v(state)),
        np.asarray(streaming_operators.apply_hermite_v(state)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(linear_moments.apply_hermite_v2(state)),
        np.asarray(streaming_operators.apply_hermite_v2(state)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(linear_moments.apply_laguerre_x(state)),
        np.asarray(streaming_operators.apply_laguerre_x(state)),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_velocity_multiplication_matrices_are_self_adjoint() -> None:
    """The truncated Hermite/Laguerre multiplication matrices stay symmetric."""

    shape = (5, 6, 1, 1, 1)
    hermite_v = _operator_matrix(linear_moments.apply_hermite_v, shape)
    hermite_v2 = _operator_matrix(linear_moments.apply_hermite_v2, shape)
    laguerre_x = _operator_matrix(linear_moments.apply_laguerre_x, shape)

    np.testing.assert_allclose(hermite_v, hermite_v.T, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(hermite_v2, hermite_v2.T, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(laguerre_x, laguerre_x.T, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        hermite_v2, hermite_v @ hermite_v, rtol=1.0e-6, atol=1.0e-6
    )
    assert np.min(np.linalg.eigvalsh(hermite_v2)) >= -1.0e-6
    assert np.min(np.linalg.eigvalsh(laguerre_x)) > 0.0


def test_periodic_streaming_matches_terms_path_and_conserves_quadratic_norm() -> None:
    """Periodic parallel streaming should be skew-adjoint in the free-energy norm."""

    ns, nl, nm, ny, nx, nz = 2, 3, 5, 2, 1, 16
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=dz)
    state = _complex_state((ns, nl, nm, ny, nx, nz))
    state = state * (1.0 + 0.1 * jnp.sin(z)).reshape((1, 1, 1, 1, 1, nz))
    vth = jnp.asarray([0.7, 1.3], dtype=jnp.float32)

    linear_path = linear_moments.streaming_term(state, dz=dz, vth=vth)
    sqrt_p, sqrt_m = hermite_ladder_coeffs(nm - 1)
    sqrt_shape = (1, 1, nm, 1, 1, 1)
    modular = streaming_operators.streaming_ladder_term(
        state,
        kz=kz,
        vth=vth.reshape((ns, 1, 1, 1, 1, 1)),
        sqrt_p=sqrt_p[:nm].reshape(sqrt_shape),
        sqrt_m=sqrt_m[:nm].reshape(sqrt_shape),
    )

    np.testing.assert_allclose(
        np.asarray(linear_path), np.asarray(modular), rtol=1.0e-6, atol=1.0e-6
    )
    energy_rate = jnp.real(jnp.vdot(state, linear_path))
    assert abs(float(energy_rate)) < 2.0e-4
