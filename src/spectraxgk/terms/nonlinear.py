"""Pseudo-spectral nonlinear E×B and electromagnetic bracket terms."""

from __future__ import annotations

from dataclasses import dataclass
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


def _laguerre_context(
    *,
    laguerre_to_grid: jnp.ndarray | None,
    laguerre_to_spectral: jnp.ndarray | None,
    laguerre_roots: jnp.ndarray | None,
    laguerre_j0: jnp.ndarray | None,
    laguerre_j1_over_alpha: jnp.ndarray | None,
    b: jnp.ndarray | None,
) -> _LaguerreGridContext:
    return _LaguerreGridContext(
        to_grid=cast(jnp.ndarray, laguerre_to_grid),
        to_spectral=cast(jnp.ndarray, laguerre_to_spectral),
        roots=cast(jnp.ndarray, laguerre_roots),
        j0=cast(jnp.ndarray | None, laguerre_j0),
        j1_over_alpha=cast(jnp.ndarray | None, laguerre_j1_over_alpha),
        b=cast(jnp.ndarray, b),
    )


def _multi_bracket_fn(compressed_real_fft: bool):
    return (
        _spectral_bracket_multi_real_fft
        if compressed_real_fft
        else _spectral_bracket_multi_full
    )


def _weighted_total(reference: jnp.ndarray, weight: jnp.ndarray, total: jnp.ndarray) -> jnp.ndarray:
    real_dtype = jnp.real(jnp.empty((), dtype=reference.dtype)).dtype
    return jnp.asarray(weight, dtype=real_dtype) * total


def _squeeze_species_output(value: jnp.ndarray, squeeze_species: bool) -> jnp.ndarray:
    return value[0] if squeeze_species else value


def _electromagnetic_enabled(
    *,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
    apar_weight: float,
    bpar_weight: float,
) -> bool:
    return (bpar is not None and bpar_weight != 0.0) or (
        apar is not None and apar_weight != 0.0
    )


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


def _laguerre_contribution_from_prepared(
    prep: _PreparedNonlinearInputs,
    ctx: _LaguerreGridContext,
    *,
    electrostatic_only: bool,
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
    compressed_real_fft: bool,
) -> jnp.ndarray:
    g_mu = _laguerre_to_grid(prep.G, ctx.to_grid)
    chi_phi = _laguerre_phi_field(prep.phi, ctx)
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
        total = _laguerre_to_spectral(exb_phi, ctx.to_spectral)
        return _squeeze_species_output(
            _weighted_total(prep.G, weight, total),
            prep.squeeze_species,
        )
    chi_fields, idx_bpar, idx_apar = _laguerre_chi_fields(
        prep,
        ctx,
        tz=tz,
        apar_weight=apar_weight,
        bpar_weight=bpar_weight,
    )
    brackets = _multi_bracket_fn(compressed_real_fft)(
        g_mu,
        _stack_fields(g_mu, chi_fields),
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=kxfac,
    )
    exb_phi = brackets[0]
    exb_bpar = brackets[idx_bpar] if idx_bpar is not None else jnp.zeros_like(exb_phi)
    flutter = jnp.zeros_like(exb_phi)
    if idx_apar is not None:
        flutter = _apply_flutter(brackets[idx_apar], vth, sqrt_m, sqrt_m_p1)
    total = _laguerre_to_spectral(exb_phi + exb_bpar + flutter, ctx.to_spectral)
    return _squeeze_species_output(
        _weighted_total(prep.G, weight, total),
        prep.squeeze_species,
    )


def _spectral_contribution_from_prepared(
    prep: _PreparedNonlinearInputs,
    *,
    electrostatic_only: bool,
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
    compressed_real_fft: bool,
) -> jnp.ndarray:
    chi_phi = prep.Jl * prep.phi[None, None, ...]
    if electrostatic_only:
        bracket_total = _spectral_bracket(
            prep.G,
            chi_phi,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=dealias_mask,
            kxfac=kxfac,
            compressed_real_fft=compressed_real_fft,
        )
        return _squeeze_species_output(
            _weighted_total(prep.G, weight, bracket_total),
            prep.squeeze_species,
        )
    chi_fields, idx_bpar, idx_apar = _spectral_chi_fields(
        prep,
        apar_weight=apar_weight,
        bpar_weight=bpar_weight,
    )
    brackets = _multi_bracket_fn(compressed_real_fft)(
        prep.G,
        _stack_fields(prep.G, chi_fields),
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=kxfac,
    )
    bracket_total = brackets[0]
    if idx_bpar is not None:
        bracket_total = bracket_total + brackets[idx_bpar]
    if idx_apar is not None:
        bracket_total = bracket_total + _apply_flutter(
            brackets[idx_apar], vth, sqrt_m, sqrt_m_p1
        )
    return _squeeze_species_output(
        _weighted_total(prep.G, weight, bracket_total),
        prep.squeeze_species,
    )


def _laguerre_components_from_prepared(
    prep: _PreparedNonlinearInputs,
    ctx: _LaguerreGridContext,
    *,
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
    compressed_real_fft: bool,
) -> dict[str, jnp.ndarray | None]:
    g_mu = _laguerre_to_grid(prep.G, ctx.to_grid)
    chi_fields, idx_bpar, idx_apar = _laguerre_chi_fields(
        prep,
        ctx,
        tz=tz,
        apar_weight=apar_weight,
        bpar_weight=bpar_weight,
    )
    brackets = _multi_bracket_fn(compressed_real_fft)(
        g_mu,
        _stack_fields(g_mu, chi_fields),
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=kxfac,
    )
    exb_phi_mu = brackets[0]
    exb_bpar_mu = (
        brackets[idx_bpar] if idx_bpar is not None else jnp.zeros_like(exb_phi_mu)
    )
    bracket_apar_mu = brackets[idx_apar] if idx_apar is not None else None
    flutter_mu = (
        _apply_flutter(bracket_apar_mu, vth, sqrt_m, sqrt_m_p1)
        if bracket_apar_mu is not None
        else jnp.zeros_like(exb_phi_mu)
    )
    exb_phi = _laguerre_to_spectral(exb_phi_mu, ctx.to_spectral)
    exb_bpar = _laguerre_to_spectral(exb_bpar_mu, ctx.to_spectral)
    flutter = _laguerre_to_spectral(flutter_mu, ctx.to_spectral)
    bracket_apar = (
        _laguerre_to_spectral(bracket_apar_mu, ctx.to_spectral)
        if bracket_apar_mu is not None
        else None
    )
    total_bracket = exb_phi + exb_bpar + flutter
    return {
        "exb_phi": exb_phi,
        "exb_bpar": exb_bpar,
        "bracket_apar": bracket_apar,
        "flutter": flutter,
        "total": _weighted_total(prep.G, weight, total_bracket),
    }


def _spectral_components_from_prepared(
    prep: _PreparedNonlinearInputs,
    *,
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
    compressed_real_fft: bool,
) -> dict[str, jnp.ndarray | None]:
    chi_fields, idx_bpar, idx_apar = _spectral_chi_fields(
        prep,
        apar_weight=apar_weight,
        bpar_weight=bpar_weight,
    )
    brackets = _multi_bracket_fn(compressed_real_fft)(
        prep.G,
        _stack_fields(prep.G, chi_fields),
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
    return {
        "exb_phi": exb_phi,
        "exb_bpar": exb_bpar,
        "bracket_apar": bracket_apar,
        "flutter": flutter,
        "total": _weighted_total(prep.G, weight, total_bracket),
    }


def _squeeze_component_payload(
    components: dict[str, jnp.ndarray | None],
    *,
    squeeze_species: bool,
) -> dict[str, jnp.ndarray]:
    exb_phi = cast(jnp.ndarray, components["exb_phi"])
    exb_bpar = cast(jnp.ndarray, components["exb_bpar"])
    flutter = cast(jnp.ndarray, components["flutter"])
    total = cast(jnp.ndarray, components["total"])
    bracket_apar = components["bracket_apar"]
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
        "bracket_apar": (
            cast(jnp.ndarray, bracket_apar)
            if bracket_apar is not None
            else jnp.zeros_like(exb_phi)
        ),
        "flutter": flutter,
        "total": total,
    }


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

    prep = _prepare_nonlinear_inputs(
        G,
        phi=phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        dealias_mask=dealias_mask,
    )
    use_laguerre = _use_laguerre_grid(
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        b=b,
        laguerre_mode=laguerre_mode,
    )
    electrostatic_only = not _electromagnetic_enabled(
        apar=prep.apar,
        bpar=prep.bpar,
        apar_weight=apar_weight,
        bpar_weight=bpar_weight,
    )

    if use_laguerre:
        ctx = _laguerre_context(
            laguerre_to_grid=laguerre_to_grid,
            laguerre_to_spectral=laguerre_to_spectral,
            laguerre_roots=laguerre_roots,
            laguerre_j0=laguerre_j0,
            laguerre_j1_over_alpha=laguerre_j1_over_alpha,
            b=b,
        )
        return _laguerre_contribution_from_prepared(
            prep,
            ctx,
            electrostatic_only=electrostatic_only,
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
        )

    return _spectral_contribution_from_prepared(
        prep,
        electrostatic_only=electrostatic_only,
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
    )


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

    prep = _prepare_nonlinear_inputs(
        G,
        phi=phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        dealias_mask=dealias_mask,
    )
    use_laguerre = _use_laguerre_grid(
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        b=b,
        laguerre_mode=laguerre_mode,
    )

    if use_laguerre:
        ctx = _laguerre_context(
            laguerre_to_grid=laguerre_to_grid,
            laguerre_to_spectral=laguerre_to_spectral,
            laguerre_roots=laguerre_roots,
            laguerre_j0=laguerre_j0,
            laguerre_j1_over_alpha=laguerre_j1_over_alpha,
            b=b,
        )
        components = _laguerre_components_from_prepared(
            prep,
            ctx,
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
        )
    else:
        components = _spectral_components_from_prepared(
            prep,
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
        )
    return _squeeze_component_payload(
        components,
        squeeze_species=prep.squeeze_species,
    )


def placeholder_nonlinear_contribution(
    G: jnp.ndarray,
    *,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Return a zero contribution for shape-only tests and disabled-term paths."""

    return jnp.zeros_like(G) * weight
