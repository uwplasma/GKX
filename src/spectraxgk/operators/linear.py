"""Linear operators used in the Hermite-Laguerre gyrokinetic system."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.basis import hermite_ladder_coeffs


def hermite_streaming(G: jnp.ndarray, kpar: jnp.ndarray, vth: float) -> jnp.ndarray:
    """Parallel streaming operator acting on the Hermite index.

    Parameters
    ----------
    G : jnp.ndarray
        Array with Hermite index on the last axis.
    kpar : jnp.ndarray
        Parallel wave number (broadcastable to G without Hermite axis).
    vth : float
        Thermal speed scaling.
    """

    Nm = G.shape[-1]
    if Nm < 1:
        raise ValueError("Hermite axis must have length >= 1")
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]

    pad_width = [(0, 0)] * (G.ndim - 1) + [(1, 1)]
    G_pad = jnp.pad(G, pad_width)
    G_mplus = G_pad[..., 2:]
    G_mminus = G_pad[..., :-2]

    ladder = sqrt_p * G_mplus + sqrt_m * G_mminus
    return -1j * kpar * vth * ladder
