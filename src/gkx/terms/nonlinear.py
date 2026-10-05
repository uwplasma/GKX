"""Pseudo-spectral nonlinear E×B and electromagnetic bracket terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp

from gkx.operators.nonlinear.brackets import (
    _apply_mask_xy,
    _spectral_bracket,
    _spectral_bracket_multi_full,
    _spectral_bracket_multi_real_fft,
    _stack_fields,
)

from gkx.core_velocity import (
    _laguerre_bpar_correction,
    _laguerre_bpar_correction_precomputed,
    _laguerre_j0_field,
    _laguerre_j0_field_precomputed,
    _laguerre_to_grid,
    _laguerre_to_spectral,
)


@dataclass(frozen=True)
class _PreparedNonlinearInputs:
    G: jnp.ndarray
    phi: jnp.ndarray
    apar: jnp.ndarray | None
    bpar: jnp.ndarray | None
    Jl: jnp.ndarray
    JlB: jnp.ndarray
    squeeze_species: bool


@dataclass(frozen=True)
class _LaguerreGridContext:
    to_grid: jnp.ndarray
    to_spectral: jnp.ndarray
    roots: jnp.ndarray
    j0: jnp.ndarray | None
    j1_over_alpha: jnp.ndarray | None
    b: jnp.ndarray


@dataclass(frozen=True)
class _PreparedNonlinearPath:
    prep: _PreparedNonlinearInputs
    laguerre: _LaguerreGridContext | None
    electrostatic_only: bool


@dataclass(frozen=True)
class _NonlinearBracketContext:
    tz: jnp.ndarray
    vth: jnp.ndarray
    sqrt_m: jnp.ndarray
    sqrt_m_p1: jnp.ndarray
    kx_grid: jnp.ndarray
    ky_grid: jnp.ndarray
    dealias_mask: jnp.ndarray
    kxfac: jnp.ndarray
    weight: jnp.ndarray
    apar_weight: float
    bpar_weight: float
    compressed_real_fft: bool
    ny_full: int | None
    radial_phase: jnp.ndarray | None


def _prepare_nonlinear_inputs(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    dealias_mask: jnp.ndarray,
) -> _PreparedNonlinearInputs:
    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    if Jl.ndim == 4:
        Jl = Jl[None, ...]
    if JlB.ndim == 4:
        JlB = JlB[None, ...]
    phi = _apply_mask_xy(phi, dealias_mask)
    if apar is not None:
        apar = _apply_mask_xy(apar, dealias_mask)
    if bpar is not None:
        bpar = _apply_mask_xy(bpar, dealias_mask)
    return _PreparedNonlinearInputs(
        G=G,
        phi=phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        squeeze_species=squeeze_species,
    )


def _use_laguerre_grid(
    *,
    laguerre_to_grid: jnp.ndarray | None,
    laguerre_to_spectral: jnp.ndarray | None,
    laguerre_roots: jnp.ndarray | None,
    b: jnp.ndarray | None,
    laguerre_mode: str,
) -> bool:
    available = (
        laguerre_to_grid is not None
        and laguerre_to_spectral is not None
        and laguerre_roots is not None
        and b is not None
    )
    mode = str(laguerre_mode).lower()
    if mode in {"spectral", "fast", "spectral_fast", "spectral-fast"}:
        return False
    return bool(available)


def _multi_bracket_fn(compressed_real_fft: bool):
    return (
        _spectral_bracket_multi_real_fft
        if compressed_real_fft
        else _spectral_bracket_multi_full
    )


def _weighted_total(
    reference: jnp.ndarray, weight: jnp.ndarray, total: jnp.ndarray
) -> jnp.ndarray:
    real_dtype = jnp.real(jnp.empty((), dtype=reference.dtype)).dtype
    return jnp.asarray(weight, dtype=real_dtype) * total


def _squeeze_species_output(value: jnp.ndarray, squeeze_species: bool) -> jnp.ndarray:
    return value[0] if squeeze_species else value


def _laguerre_phi_field(phi: jnp.ndarray, ctx: _LaguerreGridContext) -> jnp.ndarray:
    if ctx.j0 is not None:
        return _laguerre_j0_field_precomputed(phi, ctx.j0, 1.0)
    return _laguerre_j0_field(phi, ctx.b, ctx.roots, 1.0)


def _laguerre_chi_fields(
    prep: _PreparedNonlinearInputs,
    ctx: _LaguerreGridContext,
    *,
    tz: jnp.ndarray,
    apar_weight: float,
    bpar_weight: float,
) -> tuple[list[jnp.ndarray], int | None, int | None]:
    chi_fields = [_laguerre_phi_field(prep.phi, ctx)]
    idx_bpar = None
    if prep.bpar is not None and bpar_weight != 0.0:
        idx_bpar = len(chi_fields)
        if ctx.j1_over_alpha is not None:
            chi_fields.append(
                _laguerre_bpar_correction_precomputed(
                    prep.bpar,
                    ctx.j1_over_alpha,
                    ctx.roots,
                    tz,
                    1.0,
                )
            )
        else:
            chi_fields.append(
                _laguerre_bpar_correction(prep.bpar, ctx.b, ctx.roots, tz, 1.0)
            )
    idx_apar = None
    if prep.apar is not None and apar_weight != 0.0:
        idx_apar = len(chi_fields)
        if ctx.j0 is not None:
            chi_fields.append(_laguerre_j0_field_precomputed(prep.apar, ctx.j0, 1.0))
        else:
            chi_fields.append(_laguerre_j0_field(prep.apar, ctx.b, ctx.roots, 1.0))
    return chi_fields, idx_bpar, idx_apar


def _spectral_chi_fields(
    prep: _PreparedNonlinearInputs,
    *,
    apar_weight: float,
    bpar_weight: float,
) -> tuple[list[jnp.ndarray], int | None, int | None]:
    phi_hat = prep.phi[None, None, ...]
    chi_fields = [prep.Jl * phi_hat]
    idx_bpar = None
    if prep.bpar is not None and bpar_weight != 0.0:
        idx_bpar = len(chi_fields)
        chi_fields.append(prep.JlB * prep.bpar[None, None, ...])
    idx_apar = None
    if prep.apar is not None and apar_weight != 0.0:
        idx_apar = len(chi_fields)
        chi_fields.append(prep.Jl * prep.apar[None, None, ...])
    return chi_fields, idx_bpar, idx_apar


def _prepare_nonlinear_path(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    apar_weight: float,
    bpar_weight: float,
    laguerre_to_grid: jnp.ndarray | None,
    laguerre_to_spectral: jnp.ndarray | None,
    laguerre_roots: jnp.ndarray | None,
    laguerre_j0: jnp.ndarray | None,
    laguerre_j1_over_alpha: jnp.ndarray | None,
    b: jnp.ndarray | None,
    laguerre_mode: str,
) -> _PreparedNonlinearPath:
    """Prepare common nonlinear bracket routing decisions."""

    prep = _prepare_nonlinear_inputs(
        G,
        phi=phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        dealias_mask=dealias_mask,
    )
    laguerre = None
    if _use_laguerre_grid(
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        b=b,
        laguerre_mode=laguerre_mode,
    ):
        laguerre = _LaguerreGridContext(
            to_grid=cast(jnp.ndarray, laguerre_to_grid),
            to_spectral=cast(jnp.ndarray, laguerre_to_spectral),
            roots=cast(jnp.ndarray, laguerre_roots),
            j0=cast(jnp.ndarray | None, laguerre_j0),
            j1_over_alpha=cast(jnp.ndarray | None, laguerre_j1_over_alpha),
            b=cast(jnp.ndarray, b),
        )
    electromagnetic = (prep.bpar is not None and bpar_weight != 0.0) or (
        prep.apar is not None and apar_weight != 0.0
    )
    return _PreparedNonlinearPath(
        prep=prep,
        laguerre=laguerre,
        electrostatic_only=not electromagnetic,
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
            sqrt_m_p1_b,
            (1,) * (bracket_apar.ndim - sqrt_m_p1_b.ndim) + sqrt_m_p1_b.shape,
        )
    return -vth_s * (sqrt_m_b * b_m1 + sqrt_m_p1_b * b_p1)


def _bracket_kwargs(c: _NonlinearBracketContext) -> dict:
    return {
        "kx_grid": c.kx_grid,
        "ky_grid": c.ky_grid,
        "dealias_mask": c.dealias_mask,
        "kxfac": c.kxfac,
        "radial_phase": c.radial_phase,
        "ny_full": c.ny_full,
    }


def _laguerre_contribution_from_prepared(
    prep: _PreparedNonlinearInputs,
    ctx: _LaguerreGridContext,
    electrostatic_only: bool,
    c: _NonlinearBracketContext,
) -> jnp.ndarray:
    g_mu = _laguerre_to_grid(prep.G, ctx.to_grid)
    chi_phi = _laguerre_phi_field(prep.phi, ctx)
    if electrostatic_only:
        total = _spectral_bracket(
            g_mu,
            chi_phi,
            compressed_real_fft=c.compressed_real_fft,
            **_bracket_kwargs(c),
        )
    else:
        chi_fields, idx_bpar, idx_apar = _laguerre_chi_fields(
            prep, ctx, tz=c.tz, apar_weight=c.apar_weight, bpar_weight=c.bpar_weight
        )
        brackets = _multi_bracket_fn(c.compressed_real_fft)(
            g_mu, _stack_fields(g_mu, chi_fields), **_bracket_kwargs(c)
        )
        exb_phi = brackets[0]
        exb_bpar = (
            brackets[idx_bpar] if idx_bpar is not None else jnp.zeros_like(exb_phi)
        )
        flutter = jnp.zeros_like(exb_phi)
        if idx_apar is not None:
            flutter = _apply_flutter(brackets[idx_apar], c.vth, c.sqrt_m, c.sqrt_m_p1)
        total = exb_phi + exb_bpar + flutter
    total = _laguerre_to_spectral(total, ctx.to_spectral)
    return _squeeze_species_output(
        _weighted_total(prep.G, c.weight, total),
        prep.squeeze_species,
    )


def _spectral_contribution_from_prepared(
    prep: _PreparedNonlinearInputs,
    electrostatic_only: bool,
    c: _NonlinearBracketContext,
) -> jnp.ndarray:
    chi_phi = prep.Jl * prep.phi[None, None, ...]
    if electrostatic_only:
        bracket_total = _spectral_bracket(
            prep.G,
            chi_phi,
            compressed_real_fft=c.compressed_real_fft,
            **_bracket_kwargs(c),
        )
    else:
        chi_fields, idx_bpar, idx_apar = _spectral_chi_fields(
            prep, apar_weight=c.apar_weight, bpar_weight=c.bpar_weight
        )
        brackets = _multi_bracket_fn(c.compressed_real_fft)(
            prep.G, _stack_fields(prep.G, chi_fields), **_bracket_kwargs(c)
        )
        bracket_total = brackets[0]
        if idx_bpar is not None:
            bracket_total = bracket_total + brackets[idx_bpar]
        if idx_apar is not None:
            bracket_total = bracket_total + _apply_flutter(
                brackets[idx_apar], c.vth, c.sqrt_m, c.sqrt_m_p1
            )
    return _squeeze_species_output(
        _weighted_total(prep.G, c.weight, bracket_total),
        prep.squeeze_species,
    )


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
    ny_full: int | None = None,
    laguerre_mode: str = "grid",
    radial_phase: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Nonlinear E×B + flutter contribution using Laguerre gyroaveraging.

    ``apar_weight`` and ``bpar_weight`` are used as on/off toggles (nonzero
    enables the term); the fields themselves already include any scaling.
    """

    path = _prepare_nonlinear_path(
        G,
        phi=phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        dealias_mask=dealias_mask,
        apar_weight=apar_weight,
        bpar_weight=bpar_weight,
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        laguerre_j0=laguerre_j0,
        laguerre_j1_over_alpha=laguerre_j1_over_alpha,
        b=b,
        laguerre_mode=laguerre_mode,
    )
    ctx = _NonlinearBracketContext(
        tz=tz,
        vth=vth,
        sqrt_m=sqrt_m,
        sqrt_m_p1=sqrt_m_p1,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=kxfac,
        weight=weight,
        apar_weight=apar_weight,
        bpar_weight=bpar_weight,
        compressed_real_fft=compressed_real_fft,
        ny_full=ny_full,
        radial_phase=radial_phase,
    )
    if path.laguerre is not None:
        return _laguerre_contribution_from_prepared(
            path.prep, path.laguerre, path.electrostatic_only, ctx
        )
    return _spectral_contribution_from_prepared(path.prep, path.electrostatic_only, ctx)
