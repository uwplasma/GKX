"""Energy moments agree with independent Gaussian velocity quadrature."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import eval_laguerre, j0, roots_hermite, roots_laguerre

from gkx.core_velocity import J_l_all
from gkx.operators.fluxes import heat_flux_channel_species, particle_flux_species


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("nl", [1, 2, 4])
@pytest.mark.parametrize("b", [0.0, 0.27053411424286467, 1.2])
@pytest.mark.parametrize("channel", ["es", "apar"])
def test_energy_flux_matches_velocity_integral(dtype, nl, b, channel):
    with jax.enable_x64(dtype == jnp.float64):
        _check_velocity_integral(dtype, nl, b, channel)


def _check_velocity_integral(dtype, nl, b, channel):
    s, ws = roots_hermite(32)
    x, wx = roots_laguerre(64)
    ws /= np.sqrt(np.pi)
    laguerre = (-1) ** (nl - 1) * eval_laguerre(nl - 1, x)
    h2 = (4 * s**2 - 2) / np.sqrt(8)
    h3 = (8 * s**3 - 12 * s) / np.sqrt(48)
    if channel == "es":
        distribution = laguerre[None, :] + 0.3 * h2[:, None]
    else:
        distribution = np.sqrt(2) * s[:, None] * laguerre + 0.3 * h3[:, None]
        distribution *= np.sqrt(2) * s[:, None]
    integrand = distribution * (s[:, None] ** 2 + x) * j0(np.sqrt(2 * b * x))
    energy_moment = float(ws @ integrand @ wx)

    grid = SimpleNamespace(
        ky=jnp.array([0.3], dtype=dtype),
        kx=jnp.zeros(1, dtype=dtype),
        z=jnp.zeros(1, dtype=dtype),
    )
    b_array = jnp.full((2, 1, 1, 1), b, dtype=dtype)
    jl = jnp.moveaxis(J_l_all(b_array, nl - 1), 0, 1)
    cache = SimpleNamespace(Jl=jl, JlB=jnp.zeros_like(jl), b=b_array)
    params = SimpleNamespace(
        density=jnp.ones(2, dtype=dtype),
        temp=jnp.ones(2, dtype=dtype),
        vth=jnp.ones(2, dtype=dtype),
        tz=jnp.ones(2, dtype=dtype),
    )
    complex_dtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    state = jnp.zeros((2, nl, 4, 1, 1, 1), dtype=complex_dtype)
    p0, p2 = (0, 2) if channel == "es" else (1, 3)
    state = state.at[:, nl - 1, p0].set(1j).at[:, 0, p2].set(0.3j)
    field = jnp.ones((1, 1, 1), dtype=complex_dtype)
    zero = jnp.zeros_like(field)
    phi, apar = (field, zero) if channel == "es" else (zero, field)
    flux = heat_flux_channel_species(
        state,
        phi,
        apar,
        zero,
        cache,
        grid,
        params,
        jnp.ones(1, dtype=dtype),
        use_dealias=False,
    )
    assert flux[0 if channel == "es" else 1].dtype == dtype
    expected = 0.6 * energy_moment * (1 if channel == "es" else -1)
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-13
    np.testing.assert_allclose(
        np.asarray(flux[0 if channel == "es" else 1]),
        expected,
        atol=tolerance,
        rtol=tolerance,
    )
    if channel == "es":
        density_moment = float(ws @ (distribution * j0(np.sqrt(2 * b * x))) @ wx)
        particle_flux = particle_flux_species(
            state,
            phi,
            zero,
            zero,
            cache,
            grid,
            params,
            jnp.ones(1, dtype=dtype),
            use_dealias=False,
        )
        np.testing.assert_allclose(
            np.asarray(particle_flux),
            0.6 * density_moment,
            atol=tolerance,
            rtol=tolerance,
        )
