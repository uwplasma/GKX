"""Nonlinear E×B term placeholders (to be implemented)."""

from __future__ import annotations

from typing import cast, Sequence

import jax
import jax.numpy as jnp
from jax.scipy import special as jsp

from spectraxgk.gyroaverage import bessel_j0, bessel_j1
from spectraxgk.grids import gx_real_fft_mesh

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


def _apply_mask_xy(field: jnp.ndarray, mask: jnp.ndarray | None) -> jnp.ndarray:
    if mask is None:
        return field
    real_dtype = jnp.real(jnp.empty((), dtype=field.dtype)).dtype
    mask_b = _broadcast_mask(jnp.asarray(mask, dtype=real_dtype), field.ndim)
    return field * mask_b


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


def _laguerre_to_grid(G: jnp.ndarray, laguerre_to_grid: jnp.ndarray) -> jnp.ndarray:
    """Transform Laguerre moments to GX muB grid."""
    G = jnp.asarray(G)
    laguerre_to_grid = jnp.asarray(laguerre_to_grid)
    # (S, L, M, Y, X, Z) -> (S, M, Y, X, Z, L)
    G_perm = jnp.moveaxis(G, 1, -1)
    # (S, M, Y, X, Z, L) @ (L, J) -> (S, M, Y, X, Z, J)
    out = jnp.tensordot(G_perm, laguerre_to_grid, axes=([-1], [0]))
    # (S, J, M, Y, X, Z)
    return jnp.moveaxis(out, -1, 1)


def _laguerre_to_spectral(
    g_mu: jnp.ndarray, laguerre_to_spectral: jnp.ndarray
) -> jnp.ndarray:
    """Transform GX muB grid values back to Laguerre moments."""
    g_mu = jnp.asarray(g_mu)
    laguerre_to_spectral = jnp.asarray(laguerre_to_spectral)
    # (S, J, M, Y, X, Z) -> (S, M, Y, X, Z, J)
    g_perm = jnp.moveaxis(g_mu, 1, -1)
    # (S, M, Y, X, Z, J) @ (J, L) -> (S, M, Y, X, Z, L)
    out = jnp.tensordot(g_perm, laguerre_to_spectral, axes=([-1], [0]))
    # (S, L, M, Y, X, Z)
    return jnp.moveaxis(out, -1, 1)


def _gx_j0_field(
    field: jnp.ndarray,
    b: jnp.ndarray,
    roots: jnp.ndarray,
    factor: float,
) -> jnp.ndarray:
    """GX-style J0(field) on the Laguerre quadrature grid."""
    b = jnp.asarray(b)
    roots = jnp.asarray(roots)
    field = jnp.asarray(field)
    if b.ndim == 3:
        b = b[None, ...]
    if roots.ndim == 0:
        roots = roots[None]
    alpha = jnp.sqrt(
        jnp.maximum(0.0, 2.0 * roots[None, :, None, None, None] * b[:, None, ...])
    )
    j0 = bessel_j0(alpha)
    field_b = field[None, None, ...]
    return j0 * field_b * jnp.asarray(factor, dtype=field.dtype)


def _gx_j0_field_precomputed(
    field: jnp.ndarray,
    j0: jnp.ndarray,
    factor: float,
) -> jnp.ndarray:
    field = jnp.asarray(field)
    field_b = field[None, None, ...]
    return j0 * field_b * jnp.asarray(factor, dtype=field.dtype)


def _gx_bpar_term(
    bpar: jnp.ndarray,
    b: jnp.ndarray,
    roots: jnp.ndarray,
    tz: jnp.ndarray,
    factor: float,
) -> jnp.ndarray:
    """GX-style bpar correction term on Laguerre quadrature grid."""
    b = jnp.asarray(b)
    roots = jnp.asarray(roots)
    bpar = jnp.asarray(bpar)
    if b.ndim == 3:
        b = b[None, ...]
    if roots.ndim == 0:
        roots = roots[None]
    tz_arr = jnp.asarray(tz)
    if tz_arr.ndim == 0:
        tz_arr = tz_arr[None]
    alpha = jnp.sqrt(
        jnp.maximum(0.0, 2.0 * roots[None, :, None, None, None] * b[:, None, ...])
    )
    j1 = bessel_j1(alpha)
    j1_over_alpha = jnp.where(alpha < 1.0e-8, 0.5, j1 / alpha)
    coeff = (
        tz_arr[:, None, None, None, None]
        * 2.0
        * roots[None, :, None, None, None]
        * j1_over_alpha
    )
    bpar_b = bpar[None, None, ...]
    return coeff * bpar_b * jnp.asarray(factor, dtype=bpar.dtype)


def _gx_bpar_term_precomputed(
    bpar: jnp.ndarray,
    j1_over_alpha: jnp.ndarray,
    roots: jnp.ndarray,
    tz: jnp.ndarray,
    factor: float,
) -> jnp.ndarray:
    bpar = jnp.asarray(bpar)
    tz_arr = jnp.asarray(tz)
    if tz_arr.ndim == 0:
        tz_arr = tz_arr[None]
    coeff = (
        tz_arr[:, None, None, None, None]
        * 2.0
        * roots[None, :, None, None, None]
        * j1_over_alpha
    )
    bpar_b = bpar[None, None, ...]
    return coeff * bpar_b * jnp.asarray(factor, dtype=bpar.dtype)


def _apply_flutter(
    bracket_apar: jnp.ndarray,
    vth: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    sqrt_m_p1: jnp.ndarray,
) -> jnp.ndarray:
    axis_m = -4
    Nm = bracket_apar.shape[axis_m]
    zero_slice = jnp.zeros_like(jnp.take(bracket_apar, 0, axis=axis_m))
    zero_slice = jnp.expand_dims(zero_slice, axis=axis_m)
    b_lo = jax.lax.slice_in_dim(bracket_apar, 0, Nm - 1, axis=axis_m)
    b_hi = jax.lax.slice_in_dim(bracket_apar, 1, Nm, axis=axis_m)
    b_m1 = jnp.concatenate([zero_slice, b_lo], axis=axis_m)
    b_p1 = jnp.concatenate([b_hi, zero_slice], axis=axis_m)
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


def _stack_fields(G_hat: jnp.ndarray, fields: Sequence[jnp.ndarray]) -> jnp.ndarray:
    stacked = []
    for field in fields:
        stacked.append(_broadcast_to_G(jnp.asarray(field), G_hat))
    return jnp.stack(stacked, axis=0)


def _spectral_bracket_multi_gx(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
) -> jnp.ndarray:
    complex_dtype = jnp.result_type(G_hat, chi_hat_stack, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)

    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat_stack = jnp.asarray(chi_hat_stack, dtype=complex_dtype)

    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    if fft_norm is None:
        fft_norm_val = float(ky_grid.shape[0] * ky_grid.shape[1])
    else:
        fft_norm_val = float(fft_norm)
    ifft_scale = jnp.asarray(fft_norm_val, dtype=real_dtype)
    fft_scale = jnp.asarray(1.0 / fft_norm_val, dtype=real_dtype)

    ny_full = int(ky.shape[0])
    _, ky_vals, kx_nyc, ky_nyc = gx_real_fft_mesh(kx, ky)
    nyc = int(ky_vals.shape[0])

    G_nyc = G_hat[..., :nyc, :, :]
    chi_nyc = chi_hat_stack[..., :nyc, :, :]
    axes = (-2, -3)
    kx_b = _broadcast_grid(kx_nyc, G_nyc.ndim)
    ky_b = _broadcast_grid(ky_nyc, G_nyc.ndim)
    grad_G = jnp.stack([imag * kx_b * G_nyc, imag * ky_b * G_nyc], axis=0)
    grad_G = jnp.fft.irfft2(grad_G, s=(kx.shape[1], ny_full), axes=axes) * ifft_scale
    dG_dx = grad_G[0]
    dG_dy = grad_G[1]

    kx_chi = _broadcast_grid(kx_nyc, chi_nyc.ndim)
    ky_chi = _broadcast_grid(ky_nyc, chi_nyc.ndim)
    grad_chi = jnp.stack([imag * kx_chi * chi_nyc, imag * ky_chi * chi_nyc], axis=0)
    grad_chi = jnp.fft.irfft2(grad_chi, s=(kx.shape[1], ny_full), axes=axes) * ifft_scale
    dchi_dx = grad_chi[0]
    dchi_dy = grad_chi[1]

    bracket = dG_dx[None, ...] * dchi_dy - dG_dy[None, ...] * dchi_dx

    bracket_hat_nyc = jnp.fft.rfft2(bracket, axes=axes) * fft_scale
    mask_nyc = mask[:nyc, :]
    bracket_hat_nyc = bracket_hat_nyc * _broadcast_mask(mask_nyc, bracket_hat_nyc.ndim)
    if ny_full > 1:
        neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
        neg = jnp.conj(bracket_hat_nyc[..., 1:neg_hi, :, :])
        neg = neg[..., ::-1, :, :]
        if kx.shape[1] > 1:
            kx_neg = jnp.concatenate(
                [jnp.asarray([0], dtype=jnp.int32), jnp.arange(kx.shape[1] - 1, 0, -1, dtype=jnp.int32)]
            )
            neg = neg[..., kx_neg, :]
        bracket_hat = jnp.concatenate([bracket_hat_nyc, neg], axis=-3)
    else:
        bracket_hat = bracket_hat_nyc
    return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat


def _spectral_bracket_multi_full(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
) -> jnp.ndarray:
    complex_dtype = jnp.result_type(G_hat, chi_hat_stack, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)

    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat_stack = jnp.asarray(chi_hat_stack, dtype=complex_dtype)

    mask = jnp.asarray(dealias_mask, dtype=real_dtype)
    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    if fft_norm is None:
        fft_norm_val = float(ky_grid.shape[0] * ky_grid.shape[1])
    else:
        fft_norm_val = float(fft_norm)
    ifft_scale = jnp.asarray(fft_norm_val, dtype=real_dtype)
    fft_scale = jnp.asarray(1.0 / fft_norm_val, dtype=real_dtype)

    kx_b = _broadcast_grid(kx, G_hat.ndim)
    ky_b = _broadcast_grid(ky, G_hat.ndim)
    grad_G = jnp.stack([imag * kx_b * G_hat, imag * ky_b * G_hat], axis=0)
    grad_G = _ifft2_xy(grad_G) * ifft_scale
    dG_dx = grad_G[0]
    dG_dy = grad_G[1]

    kx_chi = _broadcast_grid(kx, chi_hat_stack.ndim)
    ky_chi = _broadcast_grid(ky, chi_hat_stack.ndim)
    grad_chi = jnp.stack([imag * kx_chi * chi_hat_stack, imag * ky_chi * chi_hat_stack], axis=0)
    grad_chi = _ifft2_xy(grad_chi) * ifft_scale
    dchi_dx = grad_chi[0]
    dchi_dy = grad_chi[1]

    bracket = dG_dx[None, ...] * dchi_dy - dG_dy[None, ...] * dchi_dx

    bracket_hat = _fft2_xy(bracket) * fft_scale
    bracket_hat = bracket_hat * _broadcast_mask(mask, bracket_hat.ndim)
    return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat


def _spectral_bracket(
    G_hat: jnp.ndarray,
    chi_hat: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
    gx_real_fft: bool = True,
) -> jnp.ndarray:
    return _spectral_bracket_multi(
        G_hat,
        _stack_fields(G_hat, [chi_hat]),
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=kxfac,
        fft_norm=fft_norm,
        gx_real_fft=gx_real_fft,
    )[0]


def _spectral_bracket_multi(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    *,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kxfac: jnp.ndarray,
    fft_norm: float | None = None,
    gx_real_fft: bool = True,
) -> jnp.ndarray:
    if gx_real_fft:
        return _spectral_bracket_multi_gx(
            G_hat,
            chi_hat_stack,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
            fft_norm=fft_norm,
        )
    return _spectral_bracket_multi_full(
        G_hat,
        chi_hat_stack,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=kxfac,
        fft_norm=fft_norm,
    )


def exb_nonlinear_contribution(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    weight: jnp.ndarray,
    gx_real_fft: bool = True,
) -> jnp.ndarray:
    """Return the nonlinear E×B contribution using a pseudospectral bracket."""
    phi = _apply_mask_xy(phi, dealias_mask)
    bracket_fn = _spectral_bracket_multi_gx if gx_real_fft else _spectral_bracket_multi_full
    bracket_hat = bracket_fn(
        G,
        _stack_fields(G, [phi]),
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=jnp.asarray(1.0),
    )[0]
    real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
    return jnp.asarray(weight, dtype=real_dtype) * bracket_hat


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
    laguerre_to_grid: jnp.ndarray | None = None,
    laguerre_to_spectral: jnp.ndarray | None = None,
    laguerre_roots: jnp.ndarray | None = None,
    laguerre_j0: jnp.ndarray | None = None,
    laguerre_j1_over_alpha: jnp.ndarray | None = None,
    b: jnp.ndarray | None = None,
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
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

    use_laguerre = (
        laguerre_to_grid is not None
        and laguerre_to_spectral is not None
        and laguerre_roots is not None
        and b is not None
    )
    mode = str(laguerre_mode).lower()
    if mode in {"spectral", "fast", "spectral_fast", "spectral-fast"}:
        use_laguerre = False

    phi = _apply_mask_xy(phi, dealias_mask)
    if apar is not None:
        apar = _apply_mask_xy(apar, dealias_mask)
    if bpar is not None:
        bpar = _apply_mask_xy(bpar, dealias_mask)

    if use_laguerre:
        laguerre_to_grid = cast(jnp.ndarray, laguerre_to_grid)
        laguerre_to_spectral = cast(jnp.ndarray, laguerre_to_spectral)
        laguerre_roots = cast(jnp.ndarray, laguerre_roots)
        laguerre_j0 = cast(jnp.ndarray | None, laguerre_j0)
        laguerre_j1_over_alpha = cast(jnp.ndarray | None, laguerre_j1_over_alpha)
        b = cast(jnp.ndarray, b)
        g_mu = _laguerre_to_grid(G, laguerre_to_grid)
        chi_fields: list[jnp.ndarray] = []
        idx_phi = 0
        if laguerre_j0 is not None:
            chi_fields.append(_gx_j0_field_precomputed(phi, laguerre_j0, 1.0))
        else:
            chi_fields.append(_gx_j0_field(phi, b, laguerre_roots, 1.0))
        idx_bpar = None
        if bpar is not None and bpar_weight != 0.0:
            idx_bpar = len(chi_fields)
            if laguerre_j1_over_alpha is not None:
                chi_fields.append(
                    _gx_bpar_term_precomputed(
                        bpar,
                        laguerre_j1_over_alpha,
                        laguerre_roots,
                        tz,
                        1.0,
                    )
                )
            else:
                chi_fields.append(_gx_bpar_term(bpar, b, laguerre_roots, tz, 1.0))
        idx_apar = None
        if apar is not None and apar_weight != 0.0:
            idx_apar = len(chi_fields)
            if laguerre_j0 is not None:
                chi_fields.append(_gx_j0_field_precomputed(apar, laguerre_j0, 1.0))
            else:
                chi_fields.append(_gx_j0_field(apar, b, laguerre_roots, 1.0))
        chi_stack = _stack_fields(g_mu, chi_fields)
        bracket_fn = _spectral_bracket_multi_gx if gx_real_fft else _spectral_bracket_multi_full
        brackets = bracket_fn(
            g_mu,
            chi_stack,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
        )
        exb_phi = brackets[idx_phi]
        exb_bpar = brackets[idx_bpar] if idx_bpar is not None else jnp.zeros_like(exb_phi)
        flutter = jnp.zeros_like(exb_phi)
        if idx_apar is not None:
            bracket_apar = brackets[idx_apar]
            flutter = _apply_flutter(bracket_apar, vth, sqrt_m, sqrt_m_p1)

        total_bracket = exb_phi + exb_bpar + flutter
        total = _laguerre_to_spectral(total_bracket, laguerre_to_spectral)
        real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
        out = jnp.asarray(weight, dtype=real_dtype) * total
        return out[0] if squeeze_species else out

    phi = _apply_mask_xy(phi, dealias_mask)
    if apar is not None:
        apar = _apply_mask_xy(apar, dealias_mask)
    if bpar is not None:
        bpar = _apply_mask_xy(bpar, dealias_mask)
    phi_hat = phi[None, None, ...]
    chi_fields = [Jl * phi_hat]
    idx_bpar = None
    if bpar is not None and bpar_weight != 0.0:
        idx_bpar = len(chi_fields)
        chi_fields.append(JlB * bpar[None, None, ...])
    idx_apar = None
    if apar is not None and apar_weight != 0.0:
        idx_apar = len(chi_fields)
        chi_fields.append(Jl * apar[None, None, ...])
    chi_stack = _stack_fields(G, chi_fields)
    bracket_fn = _spectral_bracket_multi_gx if gx_real_fft else _spectral_bracket_multi_full
    brackets = bracket_fn(
        G,
        chi_stack,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=kxfac,
    )
    bracket_total = brackets[0]
    if idx_bpar is not None:
        bracket_total = bracket_total + brackets[idx_bpar]
    if idx_apar is not None:
        flutter = _apply_flutter(brackets[idx_apar], vth, sqrt_m, sqrt_m_p1)
        bracket_total = bracket_total + flutter

    real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
    out = jnp.asarray(weight, dtype=real_dtype) * bracket_total
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
    laguerre_to_grid: jnp.ndarray | None = None,
    laguerre_to_spectral: jnp.ndarray | None = None,
    laguerre_roots: jnp.ndarray | None = None,
    laguerre_j0: jnp.ndarray | None = None,
    laguerre_j1_over_alpha: jnp.ndarray | None = None,
    b: jnp.ndarray | None = None,
    gx_real_fft: bool = True,
    laguerre_mode: str = "grid",
) -> dict[str, jnp.ndarray]:
    """Return nonlinear E×B/flutter components for diagnostics/comparison checks."""

    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    if Jl.ndim == 4:
        Jl = Jl[None, ...]
    if JlB.ndim == 4:
        JlB = JlB[None, ...]

    use_laguerre = (
        laguerre_to_grid is not None
        and laguerre_to_spectral is not None
        and laguerre_roots is not None
        and b is not None
    )
    mode = str(laguerre_mode).lower()
    if mode in {"spectral", "fast", "spectral_fast", "spectral-fast"}:
        use_laguerre = False

    phi = _apply_mask_xy(phi, dealias_mask)
    if apar is not None:
        apar = _apply_mask_xy(apar, dealias_mask)
    if bpar is not None:
        bpar = _apply_mask_xy(bpar, dealias_mask)

    if use_laguerre:
        laguerre_to_grid = cast(jnp.ndarray, laguerre_to_grid)
        laguerre_to_spectral = cast(jnp.ndarray, laguerre_to_spectral)
        laguerre_roots = cast(jnp.ndarray, laguerre_roots)
        laguerre_j0 = cast(jnp.ndarray | None, laguerre_j0)
        laguerre_j1_over_alpha = cast(jnp.ndarray | None, laguerre_j1_over_alpha)
        b = cast(jnp.ndarray, b)
        g_mu = _laguerre_to_grid(G, laguerre_to_grid)
        chi_fields: list[jnp.ndarray] = []
        idx_phi = 0
        if laguerre_j0 is not None:
            chi_fields.append(_gx_j0_field_precomputed(phi, laguerre_j0, 1.0))
        else:
            chi_fields.append(_gx_j0_field(phi, b, laguerre_roots, 1.0))
        idx_bpar = None
        if bpar is not None and bpar_weight != 0.0:
            idx_bpar = len(chi_fields)
            if laguerre_j1_over_alpha is not None:
                chi_fields.append(
                    _gx_bpar_term_precomputed(
                        bpar,
                        laguerre_j1_over_alpha,
                        laguerre_roots,
                        tz,
                        1.0,
                    )
                )
            else:
                chi_fields.append(_gx_bpar_term(bpar, b, laguerre_roots, tz, 1.0))
        idx_apar = None
        if apar is not None and apar_weight != 0.0:
            idx_apar = len(chi_fields)
            if laguerre_j0 is not None:
                chi_fields.append(_gx_j0_field_precomputed(apar, laguerre_j0, 1.0))
            else:
                chi_fields.append(_gx_j0_field(apar, b, laguerre_roots, 1.0))
        chi_stack = _stack_fields(g_mu, chi_fields)
        bracket_fn = _spectral_bracket_multi_gx if gx_real_fft else _spectral_bracket_multi_full
        brackets = bracket_fn(
            g_mu,
            chi_stack,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
        )
        exb_phi_mu = brackets[idx_phi]
        exb_bpar_mu = brackets[idx_bpar] if idx_bpar is not None else jnp.zeros_like(exb_phi_mu)
        bracket_apar_mu = brackets[idx_apar] if idx_apar is not None else None
        flutter_mu = (
            _apply_flutter(bracket_apar_mu, vth, sqrt_m, sqrt_m_p1)
            if bracket_apar_mu is not None
            else jnp.zeros_like(exb_phi_mu)
        )

        exb_phi = _laguerre_to_spectral(exb_phi_mu, laguerre_to_spectral)
        exb_bpar = _laguerre_to_spectral(exb_bpar_mu, laguerre_to_spectral)
        flutter = _laguerre_to_spectral(flutter_mu, laguerre_to_spectral)
        bracket_apar = (
            _laguerre_to_spectral(bracket_apar_mu, laguerre_to_spectral)
            if bracket_apar_mu is not None
            else None
        )
        total_bracket = exb_phi + exb_bpar + flutter
        real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
        total = jnp.asarray(weight, dtype=real_dtype) * total_bracket
    else:
        phi_hat = phi[None, None, ...]
        chi_fields = [Jl * phi_hat]
        idx_bpar = None
        if bpar is not None and bpar_weight != 0.0:
            idx_bpar = len(chi_fields)
            chi_fields.append(JlB * bpar[None, None, ...])
        idx_apar = None
        if apar is not None and apar_weight != 0.0:
            idx_apar = len(chi_fields)
            chi_fields.append(Jl * apar[None, None, ...])
        chi_stack = _stack_fields(G, chi_fields)
        bracket_fn = _spectral_bracket_multi_gx if gx_real_fft else _spectral_bracket_multi_full
        brackets = bracket_fn(
            G,
            chi_stack,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
        )
        exb_phi = brackets[0]
        exb_bpar = brackets[idx_bpar] if idx_bpar is not None else jnp.zeros_like(exb_phi)
        bracket_apar = brackets[idx_apar] if idx_apar is not None else None
        flutter = (
            _apply_flutter(bracket_apar, vth, sqrt_m, sqrt_m_p1)
            if bracket_apar is not None
            else jnp.zeros_like(exb_phi)
        )

        total_bracket = exb_phi + exb_bpar + flutter
        real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
        total = jnp.asarray(weight, dtype=real_dtype) * total_bracket

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
