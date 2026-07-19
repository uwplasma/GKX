"""Hermite-Laguerre moment and field-coupling primitives for linear operators."""

from __future__ import annotations

import jax.numpy as jnp

from gkx.core.velocity import hermite_ladder_coeffs
from gkx.geometry import FluxTubeGeometryLike
from gkx.core.grid import SpectralGrid
from gkx.operators.linear.params import _check_nonnegative, _check_positive
from gkx.operators.linear.streaming import (
    apply_hermite_v,
    apply_hermite_v2,
    apply_laguerre_x,
    grad_z_periodic,
    shift_axis,
    streaming_ladder_term,
)

__all__ = [
    "apply_hermite_v",
    "apply_hermite_v2",
    "apply_laguerre_x",
    "build_H",
    "compute_b",
    "diamagnetic_drive_coeffs",
    "energy_operator",
    "grad_z_periodic",
    "hermite_streaming",
    "lenard_bernstein_eigenvalues",
    "quasineutrality_phi",
    "shift_axis",
    "streaming_term",
]


def compute_b(
    grid: SpectralGrid, geom: FluxTubeGeometryLike, rho: float
) -> jnp.ndarray:
    """Compute b = rho^2 * k_perp^2(kx, ky, theta) for s-alpha geometry."""

    _check_positive(rho, "rho")
    kx0 = grid.kx[None, :, None]
    ky = grid.ky[:, None, None]
    theta = grid.z[None, None, :]
    kperp2 = geom.k_perp2(kx0, ky, theta)
    return (rho * rho) * kperp2


def lenard_bernstein_eigenvalues(
    Nl: int, Nm: int, nu_hermite: float, nu_laguerre: float
) -> jnp.ndarray:
    """Diagonal Lenard-Bernstein rates in Hermite-Laguerre space."""

    ell = jnp.arange(Nl)
    m = jnp.arange(Nm)
    return nu_laguerre * ell[:, None] + nu_hermite * m[None, :]


def hermite_streaming(G: jnp.ndarray, kpar: jnp.ndarray, vth: float) -> jnp.ndarray:
    """Parallel streaming operator acting on the Hermite index."""

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


def energy_operator(
    G: jnp.ndarray, coeff_const: float, coeff_par: float, coeff_perp: float
) -> jnp.ndarray:
    """Apply the energy operator (1 + v_par^2 + mu) in Hermite-Laguerre space."""

    return (
        coeff_const * G
        + coeff_par * apply_hermite_v2(G)
        + coeff_perp * apply_laguerre_x(G)
    )


def diamagnetic_drive_coeffs(
    Nl: int,
    Nm: int,
    eta_i: jnp.ndarray,
    coeff_const: float,
    coeff_par: float,
    coeff_perp: float,
) -> jnp.ndarray:
    """Return velocity-space coefficients for (1 + eta_i(E - 3/2))."""

    e00 = jnp.zeros((Nl, Nm, 1, 1, 1))
    e00 = e00.at[0, 0, 0, 0, 0].set(1.0)
    energy_e00 = energy_operator(e00, coeff_const, coeff_par, coeff_perp)
    coeffs = e00 + eta_i * (energy_e00 - 1.5 * e00)
    return coeffs[:, :, 0, 0, 0]


def quasineutrality_phi(
    G: jnp.ndarray,
    Jl: jnp.ndarray,
    tau_e: float | jnp.ndarray,
    charge: jnp.ndarray,
    density: jnp.ndarray,
    tz: jnp.ndarray,
) -> jnp.ndarray:
    """Solve electrostatic quasineutrality for phi with optional adiabatic closure."""

    _check_nonnegative(tau_e, "tau_e")
    Gm0 = G[:, :, 0, ...]
    num = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * jnp.sum(Jl * Gm0, axis=1),
        axis=0,
    )
    g0 = jnp.sum(Jl * Jl, axis=1)
    zt = jnp.where(tz == 0.0, 0.0, 1.0 / tz)
    den = tau_e + jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * zt[:, None, None, None]
        * (1.0 - g0),
        axis=0,
    )
    den_safe = jnp.where(den == 0.0, jnp.inf, den)
    return num / den_safe


def build_H(
    G: jnp.ndarray,
    Jl: jnp.ndarray,
    phi: jnp.ndarray,
    tz: jnp.ndarray,
    apar: jnp.ndarray | None = None,
    vth: jnp.ndarray | None = None,
    bpar: jnp.ndarray | None = None,
    JlB: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Map G -> H for mirror/curvature/grad-B/collision terms.

    The moment-space field transform adds electrostatic and compressional
    magnetic terms to ``m=0`` and the parallel-vector-potential term to
    ``m=1``. The streaming term applies its own pre-derivative field
    contributions.
    """

    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    if Jl.ndim == 4:
        Jl = Jl[None, ...]
    tz_arr = jnp.asarray(tz)
    if tz_arr.ndim == 0:
        tz_arr = tz_arr[None]
    zt_arr = jnp.where(tz_arr == 0.0, 0.0, 1.0 / tz_arr)
    Nm = G.shape[-4]
    m0_mask = (jnp.arange(Nm, dtype=jnp.int32) == 0).astype(G.dtype)
    m0_mask = m0_mask.reshape((1, 1, Nm, 1, 1, 1))
    phi_term = (zt_arr[:, None, None, None, None] * Jl * phi)[:, :, None, ...]
    H = G + m0_mask * phi_term
    if apar is not None:
        if vth is None:
            raise ValueError("vth must be provided when apar is supplied")
        m1_mask = (jnp.arange(Nm, dtype=jnp.int32) == 1).astype(G.dtype)
        m1_mask = m1_mask.reshape((1, 1, Nm, 1, 1, 1))
        vth_arr = jnp.asarray(vth)
        if vth_arr.ndim == 0:
            vth_arr = vth_arr[None]
        apar_term = (
            zt_arr[:, None, None, None, None]
            * vth_arr[:, None, None, None, None]
            * Jl
            * apar
        )[:, :, None, ...]
        H = H - m1_mask * apar_term
    if bpar is not None:
        if JlB is None:
            raise ValueError("JlB must be provided when bpar is supplied")
        bpar_term = (JlB * bpar)[:, :, None, ...]
        H = H + m0_mask * bpar_term
    return H[0] if squeeze_species else H


def streaming_term(
    H: jnp.ndarray, dz: float | jnp.ndarray, vth: float | jnp.ndarray
) -> jnp.ndarray:
    """Streaming term using Hermite ladder and real-space z derivative."""

    _check_positive(dz, "dz")
    _check_positive(vth, "vth")
    axis_m = -4
    Nm = H.shape[axis_m]
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]

    shape = [1] * H.ndim
    shape[axis_m] = Nm
    sqrt_p = sqrt_p.reshape(shape)
    sqrt_m = sqrt_m.reshape(shape)
    dz_val = jnp.asarray(dz, dtype=jnp.real(H).dtype)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(H.shape[-1], d=dz_val)
    vth_arr = jnp.asarray(vth)
    if vth_arr.ndim == 0:
        vth_arr = vth_arr[None]
    v_shape = [1] * H.ndim
    v_shape[0] = vth_arr.shape[0]
    vth_arr = vth_arr.reshape(v_shape)
    return streaming_ladder_term(
        H,
        kz,
        vth_arr,
        sqrt_p,
        sqrt_m,
        dz=dz,
    )
