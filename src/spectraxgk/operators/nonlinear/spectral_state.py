"""Deterministic nonlinear spectral states and shape validation."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _validate_spectral_state_shape(
    shape: tuple[int, ...],
) -> tuple[int, int, int, int, int]:
    if len(shape) != 5:
        raise ValueError("spectral state shape must be (Nl, Nm, Ny, Nx, Nz)")
    nl, nm, ny, nx, nz = (int(item) for item in shape)
    if min(nl, nm, ny, nx, nz) <= 0:
        raise ValueError("spectral state dimensions must be positive")
    if ny < 2 or nx < 2:
        raise ValueError("spectral communication gate requires Ny and Nx >= 2")
    return (nl, nm, ny, nx, nz)


def deterministic_nonlinear_spectral_state(
    shape: tuple[int, int, int, int, int] = (2, 3, 6, 4, 2),
) -> jax.Array:
    """Return deterministic complex spectral coefficients for communication gates.

    The layout is ``(Nl, Nm, Ny, Nx, Nz)`` with the FFT axes in ``(Ny, Nx)``.
    """

    nl, nm, ny, nx, nz = _validate_spectral_state_shape(tuple(shape))
    laguerre = jnp.arange(nl, dtype=jnp.float32)[:, None, None, None, None]
    hermite = jnp.arange(nm, dtype=jnp.float32)[None, :, None, None, None]
    y = jnp.arange(ny, dtype=jnp.float32)[None, None, :, None, None]
    x = jnp.arange(nx, dtype=jnp.float32)[None, None, None, :, None]
    z = jnp.arange(nz, dtype=jnp.float32)[None, None, None, None, :]
    phase = (
        0.41 * (laguerre + 1.0)
        + 0.29 * (hermite + 1.0)
        + 0.17 * y
        + 0.23 * x
        + 0.31 * (z + 1.0)
    )
    envelope = 1.0 / (1.0 + laguerre + hermite + 0.25 * y + 0.5 * x + z)
    real_part = envelope * jnp.sin(phase)
    imag_part = 0.5 * envelope * jnp.cos(1.7 * phase)
    return real_part.astype(jnp.float32) + 1j * imag_part.astype(jnp.float32)


__all__ = [
    "_validate_spectral_state_shape",
    "deterministic_nonlinear_spectral_state",
]
