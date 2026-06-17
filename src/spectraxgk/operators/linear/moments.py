"""Hermite-Laguerre moment and field-coupling primitives for linear operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectraxgk.core.velocity import hermite_ladder_coeffs
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.grids import SpectralGrid
from spectraxgk.operators.linear.params import _check_nonnegative, _check_positive

__all__ = [
    "apply_hermite_v",
    "apply_hermite_v2",
    "apply_laguerre_x",
    "build_H",
    "compute_b",
    "diamagnetic_drive_coeffs",
    "energy_operator",
    "grad_z_periodic",
    "lenard_bernstein_eigenvalues",
    "quasineutrality_phi",
    "shift_axis",
    "streaming_term",
]


def grad_z_periodic(f: jnp.ndarray, dz: float | jnp.ndarray) -> jnp.ndarray:
    """Spectral periodic derivative along the last axis."""

    _check_positive(dz, "dz")
    n = f.shape[-1]
    dz_val = jnp.asarray(dz, dtype=jnp.real(f).dtype)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dz_val)
    f_hat = jnp.fft.fft(f, axis=-1)
    df_hat = (1j * kz) * f_hat
    return jnp.fft.ifft(df_hat, axis=-1)


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


def apply_hermite_v(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Hermite coefficients by v_parallel (ladder form)."""

    axis_m = -4
    Nm = G.shape[axis_m]
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]
    G_plus = shift_axis(G, 1, axis_m)
    G_minus = shift_axis(G, -1, axis_m)
    shape = [1] * G.ndim
    shape[axis_m] = Nm
    sqrt_p = sqrt_p.reshape(shape)
    sqrt_m = sqrt_m.reshape(shape)
    return sqrt_p * G_plus + sqrt_m * G_minus


def apply_hermite_v2(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Hermite coefficients by v_parallel^2."""

    return apply_hermite_v(apply_hermite_v(G))


def apply_laguerre_x(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Laguerre coefficients by the perpendicular energy variable."""

    axis_l = -5
    Nl = G.shape[axis_l]
    ell = jnp.arange(Nl)
    G_plus = shift_axis(G, 1, axis_l)
    G_minus = shift_axis(G, -1, axis_l)
    ell_shape = [1] * G.ndim
    ell_shape[axis_l] = Nl
    ell_col = ell.reshape(ell_shape)
    return (2.0 * ell_col + 1.0) * G - (ell_col + 1.0) * G_plus - ell_col * G_minus


def shift_axis(arr: jnp.ndarray, offset: int, axis: int) -> jnp.ndarray:
    """Shift an array along an axis with zero padding (non-periodic)."""

    axis = axis % arr.ndim
    if offset == 0:
        return arr
    axis_len = arr.shape[axis]
    if abs(offset) >= axis_len:
        return jnp.zeros_like(arr)
    out = jnp.zeros_like(arr)
    if offset > 0:
        body = jax.lax.slice_in_dim(arr, offset, axis_len, axis=axis)
        starts = [0] * arr.ndim
        starts[axis] = 0
        return jax.lax.dynamic_update_slice(out, body, starts)
    body = jax.lax.slice_in_dim(arr, 0, axis_len + offset, axis=axis)
    starts = [0] * arr.ndim
    starts[axis] = -offset
    return jax.lax.dynamic_update_slice(out, body, starts)


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

    _check_positive(vth, "vth")
    dH_dz = grad_z_periodic(H, dz)
    axis_m = -4
    Nm = H.shape[axis_m]
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]

    pad = [(0, 0)] * H.ndim
    pad[axis_m] = (1, 1)
    dH_pad = jnp.pad(dH_dz, pad)
    slc_plus = [slice(None)] * H.ndim
    slc_minus = [slice(None)] * H.ndim
    slc_plus[axis_m] = slice(2, None)
    slc_minus[axis_m] = slice(0, -2)
    dH_plus = dH_pad[tuple(slc_plus)]
    dH_minus = dH_pad[tuple(slc_minus)]

    shape = [1] * H.ndim
    shape[axis_m] = Nm
    sqrt_p = sqrt_p.reshape(shape)
    sqrt_m = sqrt_m.reshape(shape)
    ladder = sqrt_p * dH_plus + sqrt_m * dH_minus
    vth_arr = jnp.asarray(vth)
    if vth_arr.ndim == 0:
        vth_arr = vth_arr[None]
    v_shape = [1] * H.ndim
    v_shape[0] = vth_arr.shape[0]
    vth_arr = vth_arr.reshape(v_shape)
    return vth_arr * ladder
