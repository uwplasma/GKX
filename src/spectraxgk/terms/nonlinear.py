"""Pseudo-spectral nonlinear E×B and electromagnetic bracket terms."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp

from spectraxgk.terms.brackets import (
    _apply_mask_xy,
    _spectral_bracket,
    _spectral_bracket_multi_full,
    _spectral_bracket_multi_real_fft,
    _stack_fields,
)

from spectraxgk.terms.gyroaveraging import (
    _laguerre_bpar_correction,
    _laguerre_bpar_correction_precomputed,
    _laguerre_j0_field,
    _laguerre_j0_field_precomputed,
    _laguerre_to_grid,
    _laguerre_to_spectral,
)
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


def exb_nonlinear_contribution(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    weight: jnp.ndarray,
    compressed_real_fft: bool = True,
) -> jnp.ndarray:
    """Return the nonlinear E×B contribution using a pseudospectral bracket."""
    phi = _apply_mask_xy(phi, dealias_mask)
    bracket_hat = _spectral_bracket(
        G,
        phi,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=compressed_real_fft,
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
    laguerre_j0: jnp.ndarray | None = None,
    laguerre_j1_over_alpha: jnp.ndarray | None = None,
    b: jnp.ndarray | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
) -> jnp.ndarray:
    """Nonlinear E×B + flutter contribution using Laguerre gyroaveraging.

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

    electrostatic_only = (bpar is None or bpar_weight == 0.0) and (apar is None or apar_weight == 0.0)

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
            chi_phi = _laguerre_j0_field_precomputed(phi, laguerre_j0, 1.0)
        else:
            chi_phi = _laguerre_j0_field(phi, b, laguerre_roots, 1.0)
        if electrostatic_only:
            exb_phi = _spectral_bracket(
                g_mu,
                chi_phi,
                kx_grid=kx_grid,
                ky_grid=ky_grid,
                dealias_mask=dealias_mask,
                kxfac=kxfac,
                compressed_real_fft=compressed_real_fft,
            )
            total = _laguerre_to_spectral(exb_phi, laguerre_to_spectral)
            real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
            out = jnp.asarray(weight, dtype=real_dtype) * total
            return out[0] if squeeze_species else out
        chi_fields.append(chi_phi)
        idx_bpar = None
        if bpar is not None and bpar_weight != 0.0:
            idx_bpar = len(chi_fields)
            if laguerre_j1_over_alpha is not None:
                chi_fields.append(
                    _laguerre_bpar_correction_precomputed(
                        bpar,
                        laguerre_j1_over_alpha,
                        laguerre_roots,
                        tz,
                        1.0,
                    )
                )
            else:
                chi_fields.append(_laguerre_bpar_correction(bpar, b, laguerre_roots, tz, 1.0))
        idx_apar = None
        if apar is not None and apar_weight != 0.0:
            idx_apar = len(chi_fields)
            if laguerre_j0 is not None:
                chi_fields.append(_laguerre_j0_field_precomputed(apar, laguerre_j0, 1.0))
            else:
                chi_fields.append(_laguerre_j0_field(apar, b, laguerre_roots, 1.0))
        chi_stack = _stack_fields(g_mu, chi_fields)
        bracket_fn = _spectral_bracket_multi_real_fft if compressed_real_fft else _spectral_bracket_multi_full
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

    phi_hat = phi[None, None, ...]
    chi_phi = Jl * phi_hat
    if electrostatic_only:
        bracket_total = _spectral_bracket(
            G,
            chi_phi,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
            compressed_real_fft=compressed_real_fft,
        )
        real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
        out = jnp.asarray(weight, dtype=real_dtype) * bracket_total
        return out[0] if squeeze_species else out
    chi_fields = [chi_phi]
    idx_bpar = None
    if bpar is not None and bpar_weight != 0.0:
        idx_bpar = len(chi_fields)
        chi_fields.append(JlB * bpar[None, None, ...])
    idx_apar = None
    if apar is not None and apar_weight != 0.0:
        idx_apar = len(chi_fields)
        chi_fields.append(Jl * apar[None, None, ...])
    chi_stack = _stack_fields(G, chi_fields)
    bracket_fn = _spectral_bracket_multi_real_fft if compressed_real_fft else _spectral_bracket_multi_full
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
    compressed_real_fft: bool = True,
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
            chi_fields.append(_laguerre_j0_field_precomputed(phi, laguerre_j0, 1.0))
        else:
            chi_fields.append(_laguerre_j0_field(phi, b, laguerre_roots, 1.0))
        idx_bpar = None
        if bpar is not None and bpar_weight != 0.0:
            idx_bpar = len(chi_fields)
            if laguerre_j1_over_alpha is not None:
                chi_fields.append(
                    _laguerre_bpar_correction_precomputed(
                        bpar,
                        laguerre_j1_over_alpha,
                        laguerre_roots,
                        tz,
                        1.0,
                    )
                )
            else:
                chi_fields.append(_laguerre_bpar_correction(bpar, b, laguerre_roots, tz, 1.0))
        idx_apar = None
        if apar is not None and apar_weight != 0.0:
            idx_apar = len(chi_fields)
            if laguerre_j0 is not None:
                chi_fields.append(_laguerre_j0_field_precomputed(apar, laguerre_j0, 1.0))
            else:
                chi_fields.append(_laguerre_j0_field(apar, b, laguerre_roots, 1.0))
        chi_stack = _stack_fields(g_mu, chi_fields)
        bracket_fn = _spectral_bracket_multi_real_fft if compressed_real_fft else _spectral_bracket_multi_full
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
        bracket_fn = _spectral_bracket_multi_real_fft if compressed_real_fft else _spectral_bracket_multi_full
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
    """Return a zero contribution for shape-only tests and disabled-term paths."""

    return jnp.zeros_like(G) * weight
