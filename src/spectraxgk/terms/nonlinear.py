"""Nonlinear E×B term placeholders (to be implemented)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy import special as jsp


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


def _laguerre_to_grid(G: jnp.ndarray, laguerre_to_grid: jnp.ndarray) -> jnp.ndarray:
    """Transform Laguerre moments to GX muB grid."""
    return jnp.einsum("slmxyz,lj->sjmxyz", G, laguerre_to_grid)


def _laguerre_to_spectral(
    g_mu: jnp.ndarray, laguerre_to_spectral: jnp.ndarray
) -> jnp.ndarray:
    """Transform GX muB grid values back to Laguerre moments."""
    return jnp.einsum("sjmxyz,jl->slmxyz", g_mu, laguerre_to_spectral)


def _j0_series(x: jnp.ndarray, n_terms: int = 20) -> jnp.ndarray:
    x2 = (0.5 * x) ** 2
    term = jnp.ones_like(x)
    acc = term

    def body(k, state):
        term_k, acc_k = state
        term_k = term_k * (-x2) / (k * k)
        acc_k = acc_k + term_k
        return term_k, acc_k

    _, acc = jax.lax.fori_loop(1, n_terms, body, (term, acc))
    return acc


def _j1_series(x: jnp.ndarray, n_terms: int = 20) -> jnp.ndarray:
    x2 = (0.5 * x) ** 2
    term = 0.5 * x
    acc = term

    def body(k, state):
        term_k, acc_k = state
        term_k = term_k * (-x2) / (k * (k + 1))
        acc_k = acc_k + term_k
        return term_k, acc_k

    _, acc = jax.lax.fori_loop(1, n_terms, body, (term, acc))
    return acc


def _bessel_j0(x: jnp.ndarray) -> jnp.ndarray:
    j0_b = jsp.bessel_jn(x, v=0)[0]
    j0_s = _j0_series(x)
    mask = jnp.isnan(j0_b) | (jnp.abs(x) < 1.0)
    return jnp.where(mask, j0_s, j0_b)


def _bessel_j1(x: jnp.ndarray) -> jnp.ndarray:
    j1_b = jsp.bessel_jn(x, v=1)[1]
    j1_s = _j1_series(x)
    mask = jnp.isnan(j1_b) | (jnp.abs(x) < 1.0)
    return jnp.where(mask, j1_s, j1_b)


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
    j0 = _bessel_j0(alpha)
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
    j1 = _bessel_j1(alpha)
    j1_over_alpha = jnp.where(alpha < 1.0e-8, 0.5, j1 / alpha)
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
    fft_norm: float | None = None,
    gx_real_fft: bool = True,
) -> jnp.ndarray:
    complex_dtype = jnp.result_type(G_hat, chi_hat, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=complex_dtype)).dtype
    imag = jnp.asarray(1j, dtype=complex_dtype)

    G_hat = jnp.asarray(G_hat, dtype=complex_dtype)
    chi_hat = jnp.asarray(chi_hat, dtype=complex_dtype)

    mask = jnp.asarray(dealias_mask, dtype=real_dtype)

    kx = jnp.asarray(kx_grid, dtype=real_dtype)
    ky = jnp.asarray(ky_grid, dtype=real_dtype)
    if fft_norm is None:
        fft_norm_val = float(ky_grid.shape[0] * ky_grid.shape[1])
    else:
        fft_norm_val = float(fft_norm)
    ifft_scale = jnp.asarray(fft_norm_val, dtype=real_dtype)
    fft_scale = jnp.asarray(1.0 / fft_norm_val, dtype=real_dtype)

    if gx_real_fft:
        ny_full = int(ky.shape[0])
        nyc = ny_full // 2 + 1
        ky_nyc = ky[:nyc, :]
        kx_nyc = kx[:nyc, :]
        G_nyc = G_hat[..., :nyc, :, :]
        chi_nyc = chi_hat[..., :nyc, :, :]
        kx_b = _broadcast_grid(kx_nyc, G_nyc.ndim)
        ky_b = _broadcast_grid(ky_nyc, G_nyc.ndim)
        kx_chi = _broadcast_grid(kx_nyc, chi_nyc.ndim)
        ky_chi = _broadcast_grid(ky_nyc, chi_nyc.ndim)
        axes = (-2, -3)
        dchi_dx = jnp.fft.irfft2(imag * kx_chi * chi_nyc, s=(kx.shape[1], ny_full), axes=axes) * ifft_scale
        dchi_dy = jnp.fft.irfft2(imag * ky_chi * chi_nyc, s=(kx.shape[1], ny_full), axes=axes) * ifft_scale
        dG_dx = jnp.fft.irfft2(imag * kx_b * G_nyc, s=(kx.shape[1], ny_full), axes=axes) * ifft_scale
        dG_dy = jnp.fft.irfft2(imag * ky_b * G_nyc, s=(kx.shape[1], ny_full), axes=axes) * ifft_scale

        dchi_dx_b = _broadcast_to_G(dchi_dx, dG_dx)
        dchi_dy_b = _broadcast_to_G(dchi_dy, dG_dx)
        bracket = dG_dx * dchi_dy_b - dG_dy * dchi_dx_b
        bracket_hat_nyc = jnp.fft.rfft2(bracket, axes=axes) * fft_scale
        mask_nyc = mask[:nyc, :]
        bracket_hat_nyc = bracket_hat_nyc * _broadcast_mask(mask_nyc, bracket_hat_nyc.ndim)
        if ny_full > 1:
            neg = jnp.conj(bracket_hat_nyc[..., 1 : nyc - 1, :, :])
            neg = neg[..., ::-1, :, :]
            bracket_hat = jnp.concatenate([bracket_hat_nyc, neg], axis=-3)
        else:
            bracket_hat = bracket_hat_nyc
        return jnp.asarray(kxfac, dtype=real_dtype) * bracket_hat

    kx_b = _broadcast_grid(kx, G_hat.ndim)
    ky_b = _broadcast_grid(ky, G_hat.ndim)
    kx_chi = _broadcast_grid(kx, chi_hat.ndim)
    ky_chi = _broadcast_grid(ky, chi_hat.ndim)

    dchi_dx = _ifft2_xy(imag * kx_chi * chi_hat) * ifft_scale
    dchi_dy = _ifft2_xy(imag * ky_chi * chi_hat) * ifft_scale
    dG_dx = _ifft2_xy(imag * kx_b * G_hat) * ifft_scale
    dG_dy = _ifft2_xy(imag * ky_b * G_hat) * ifft_scale

    dchi_dx_b = _broadcast_to_G(dchi_dx, G_hat)
    dchi_dy_b = _broadcast_to_G(dchi_dy, G_hat)
    bracket = dG_dx * dchi_dy_b - dG_dy * dchi_dx_b

    bracket_hat = _fft2_xy(bracket) * fft_scale
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
    b: jnp.ndarray | None = None,
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

    if use_laguerre:
        g_mu = _laguerre_to_grid(G, laguerre_to_grid)
        chi_phi = _gx_j0_field(phi, b, laguerre_roots, 1.0)
        exb_phi = _spectral_bracket(
            g_mu,
            chi_phi,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
        )
        exb_bpar = jnp.zeros_like(exb_phi)
        if bpar is not None and bpar_weight != 0.0:
            chi_bpar = _gx_bpar_term(bpar, b, laguerre_roots, tz, 1.0)
            exb_bpar = _spectral_bracket(
                g_mu,
                chi_bpar,
                kx_grid=kx_grid,
                ky_grid=ky_grid,
                dealias_mask=dealias_mask,
                kxfac=kxfac,
            )

        flutter = jnp.zeros_like(exb_phi)
        if apar is not None and apar_weight != 0.0:
            chi_apar = _gx_j0_field(apar, b, laguerre_roots, 1.0)
            bracket_apar = _spectral_bracket(
                g_mu,
                chi_apar,
                kx_grid=kx_grid,
                ky_grid=ky_grid,
                dealias_mask=dealias_mask,
                kxfac=kxfac,
            )
            flutter = _apply_flutter(bracket_apar, vth, sqrt_m, sqrt_m_p1)

        total_bracket = exb_phi + exb_bpar + flutter
        total = _laguerre_to_spectral(total_bracket, laguerre_to_spectral)
        real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
        out = -jnp.asarray(weight, dtype=real_dtype) * total
        return out[0] if squeeze_species else out

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
    laguerre_to_grid: jnp.ndarray | None = None,
    laguerre_to_spectral: jnp.ndarray | None = None,
    laguerre_roots: jnp.ndarray | None = None,
    b: jnp.ndarray | None = None,
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

    use_laguerre = (
        laguerre_to_grid is not None
        and laguerre_to_spectral is not None
        and laguerre_roots is not None
        and b is not None
    )

    if use_laguerre:
        g_mu = _laguerre_to_grid(G, laguerre_to_grid)
        chi_phi = _gx_j0_field(phi, b, laguerre_roots, 1.0)
        exb_phi_mu = _spectral_bracket(
            g_mu,
            chi_phi,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
        )
        if bpar is not None and bpar_weight != 0.0:
            chi_bpar = _gx_bpar_term(bpar, b, laguerre_roots, tz, 1.0)
            exb_bpar_mu = _spectral_bracket(
                g_mu,
                chi_bpar,
                kx_grid=kx_grid,
                ky_grid=ky_grid,
                dealias_mask=dealias_mask,
                kxfac=kxfac,
            )
        else:
            exb_bpar_mu = jnp.zeros_like(exb_phi_mu)

        bracket_apar_mu = None
        flutter_mu = jnp.zeros_like(exb_phi_mu)
        if apar is not None and apar_weight != 0.0:
            chi_apar = _gx_j0_field(apar, b, laguerre_roots, 1.0)
            bracket_apar_mu = _spectral_bracket(
                g_mu,
                chi_apar,
                kx_grid=kx_grid,
                ky_grid=ky_grid,
                dealias_mask=dealias_mask,
                kxfac=kxfac,
            )
            flutter_mu = _apply_flutter(bracket_apar_mu, vth, sqrt_m, sqrt_m_p1)

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
