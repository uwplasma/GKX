"""Nonlinear E×B term placeholders (to be implemented)."""

from __future__ import annotations

import jax.numpy as jnp


def _fft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.fft2(x, axes=(-3, -2))


def _ifft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(x, axes=(-3, -2))


def _broadcast_mask(mask: jnp.ndarray, ndim: int) -> jnp.ndarray:
    shape = (1,) * (ndim - 3) + mask.shape + (1,)
    return jnp.reshape(mask, shape)


def _broadcast_grid(grid: jnp.ndarray, ndim: int) -> jnp.ndarray:
    shape = (1,) * (ndim - 3) + grid.shape + (1,)
    return jnp.reshape(grid, shape)


def _broadcast_to_G(x: jnp.ndarray, G: jnp.ndarray) -> jnp.ndarray:
    if x.ndim == G.ndim:
        return x
    if x.ndim == G.ndim - 1:
        return jnp.expand_dims(x, axis=-4)
    if x.ndim == 3:
        shape = (1,) * (G.ndim - 3) + x.shape
        x = jnp.reshape(x, shape)
        if x.ndim == G.ndim - 1:
            x = jnp.expand_dims(x, axis=-4)
        return x
    if x.ndim < G.ndim:
        shape = (1,) * (G.ndim - x.ndim) + x.shape
        x = jnp.reshape(x, shape)
        if x.ndim == G.ndim - 1:
            x = jnp.expand_dims(x, axis=-4)
        return x
    return x


def _apply_flutter(
    bracket_apar: jnp.ndarray,
    vth: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    sqrt_m_p1: jnp.ndarray,
) -> jnp.ndarray:
    axis_m = -4
    pad = [(0, 0)] * bracket_apar.ndim
    pad[axis_m] = (1, 1)
    b_pad = jnp.pad(bracket_apar, pad)
    slc_m1 = [slice(None)] * bracket_apar.ndim
    slc_p1 = [slice(None)] * bracket_apar.ndim
    slc_m1[axis_m] = slice(0, bracket_apar.shape[axis_m])
    slc_p1[axis_m] = slice(2, bracket_apar.shape[axis_m] + 2)
    b_m1 = b_pad[tuple(slc_m1)]
    b_p1 = b_pad[tuple(slc_p1)]
    vth_arr = jnp.asarray(vth)
    if vth_arr.ndim == 0:
        vth_arr = vth_arr[None]
    v_shape = [1] * bracket_apar.ndim
    v_shape[0] = vth_arr.shape[0]
    vth_s = vth_arr.reshape(v_shape)
    sqrt_m_b = sqrt_m
    sqrt_m_p1_b = sqrt_m_p1
    if sqrt_m_b.ndim < bracket_apar.ndim:
        sqrt_m_b = jnp.reshape(
            sqrt_m_b, (1,) * (bracket_apar.ndim - sqrt_m_b.ndim) + sqrt_m_b.shape
        )
    if sqrt_m_p1_b.ndim < bracket_apar.ndim:
        sqrt_m_p1_b = jnp.reshape(
            sqrt_m_p1_b, (1,) * (bracket_apar.ndim - sqrt_m_p1_b.ndim) + sqrt_m_p1_b.shape
        )
    return -vth_s * (sqrt_m_b * b_m1 + sqrt_m_p1_b * b_p1)


def _spectral_bracket(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
) -> jnp.ndarray:
    complex_dtype = jnp.result_type(G_hat, chi_hat, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)

    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat = jnp.asarray(chi_hat, dtype=complex_dtype)

    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    G_hat = G_hat * _broadcast_mask(mask, G_hat.ndim)
    chi_hat = chi_hat * _broadcast_mask(mask, chi_hat.ndim)

    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    kx_b = _broadcast_grid(kx, G_hat.ndim)
    ky_b = _broadcast_grid(ky, G_hat.ndim)
    kx_chi = _broadcast_grid(kx, chi_hat.ndim)
    ky_chi = _broadcast_grid(ky, chi_hat.ndim)

    dchi_dx = _ifft2_xy(imag * kx_chi * chi_hat)
    dchi_dy = _ifft2_xy(imag * ky_chi * chi_hat)
    dG_dx = _ifft2_xy(imag * kx_b * G_hat)
    dG_dy = _ifft2_xy(imag * ky_b * G_hat)

    dchi_dx_b = _broadcast_to_G(dchi_dx, G_hat)
    dchi_dy_b = _broadcast_to_G(dchi_dy, G_hat)
    bracket = dchi_dx_b * dG_dy - dchi_dy_b * dG_dx

    bracket_hat = _fft2_xy(bracket)
    bracket_hat = bracket_hat * _broadcast_mask(mask, bracket_hat.ndim)
    return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat


def exb_nonlinear_contribution(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Return the nonlinear E×B contribution using a pseudospectral bracket."""
    bracket_hat = _spectral_bracket(
        G,
        phi,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=jnp.asarray(1.0),
    )
    real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
    return -jnp.asarray(weight, dtype=real_dtype) * bracket_hat


def nonlinear_em_contribution(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    sqrt_m_p1: jnp.ndarray,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    weight: jnp.ndarray,
    apar_weight: float,
    bpar_weight: float,
) -> jnp.ndarray:
    """Nonlinear E×B + flutter contribution using GX-style gyroaveraging.

    ``apar_weight`` and ``bpar_weight`` are used as on/off toggles (nonzero
    enables the term); the fields themselves already include any scaling.
    """

    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    if Jl.ndim == 4:
        Jl = Jl[None, ...]
    if JlB.ndim == 4:
        JlB = JlB[None, ...]

    phi_hat = phi[None, None, ...]
    chi = Jl * phi_hat
    if bpar is not None and bpar_weight != 0.0:
        chi = chi + JlB * bpar[None, None, ...]

    bracket_phi = _spectral_bracket(
        G,
        chi,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=kxfac,
    )

    bracket_total = bracket_phi
    if apar is not None and apar_weight != 0.0:
        apar_hat = apar[None, None, ...]
        chi_apar = Jl * apar_hat
        bracket_apar = _spectral_bracket(
            G,
            chi_apar,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
        )
        flutter = _apply_flutter(bracket_apar, vth, sqrt_m, sqrt_m_p1)
        bracket_total = bracket_total + flutter

    real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
    out = -jnp.asarray(weight, dtype=real_dtype) * bracket_total
    return out[0] if squeeze_species else out


def nonlinear_em_components(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    sqrt_m_p1: jnp.ndarray,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    weight: jnp.ndarray,
    apar_weight: float,
    bpar_weight: float,
) -> dict[str, jnp.ndarray]:
    """Return nonlinear E×B/flutter components for diagnostics/parity checks."""

    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    if Jl.ndim == 4:
        Jl = Jl[None, ...]
    if JlB.ndim == 4:
        JlB = JlB[None, ...]

    phi_hat = phi[None, None, ...]
    chi_phi = Jl * phi_hat
    exb_phi = _spectral_bracket(
        G,
        chi_phi,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=kxfac,
    )

    if bpar is not None and bpar_weight != 0.0:
        chi_bpar = JlB * bpar[None, None, ...]
        exb_bpar = _spectral_bracket(
            G,
            chi_bpar,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
        )
    else:
        exb_bpar = jnp.zeros_like(exb_phi)

    bracket_apar = None
    flutter = jnp.zeros_like(exb_phi)
    if apar is not None and apar_weight != 0.0:
        apar_hat = apar[None, None, ...]
        chi_apar = Jl * apar_hat
        bracket_apar = _spectral_bracket(
            G,
            chi_apar,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
        )
        flutter = _apply_flutter(bracket_apar, vth, sqrt_m, sqrt_m_p1)

    total_bracket = exb_phi + exb_bpar + flutter
    real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
    total = -jnp.asarray(weight, dtype=real_dtype) * total_bracket

    if squeeze_species:
        exb_phi = exb_phi[0]
        exb_bpar = exb_bpar[0]
        flutter = flutter[0]
        total = total[0]
        if bracket_apar is not None:
            bracket_apar = bracket_apar[0]

    return {
        "exb_phi": exb_phi,
        "exb_bpar": exb_bpar,
        "bracket_apar": bracket_apar if bracket_apar is not None else jnp.zeros_like(exb_phi),
        "flutter": flutter,
        "total": total,
    }


def placeholder_nonlinear_contribution(
    G: jnp.ndarray,
    *,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Return a zero nonlinear contribution to validate IO shapes."""

    return jnp.zeros_like(G) * weight
