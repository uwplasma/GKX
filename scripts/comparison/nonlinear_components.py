"""Per-term split of the nonlinear E×B/flutter bracket for GX comparisons.

Only the comparison tooling and its tests need the split; the solver uses
``gkx.terms.nonlinear.nonlinear_em_contribution``, whose total these
components reproduce.
"""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp

from gkx.core_velocity import _laguerre_to_grid, _laguerre_to_spectral
from gkx.operators.nonlinear.brackets import _stack_fields
from gkx.terms.nonlinear import (
    _apply_flutter,
    _laguerre_chi_fields,
    _LaguerreGridContext,
    _multi_bracket_fn,
    _NonlinearBracketContext,
    _prepare_nonlinear_path,
    _PreparedNonlinearInputs,
    _PreparedNonlinearPath,
    _spectral_chi_fields,
    _weighted_total,
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
    ny_full: int | None = None,
    radial_phase: jnp.ndarray | None,
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
        radial_phase=radial_phase,
        ny_full=ny_full,
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
    ny_full: int | None = None,
    radial_phase: jnp.ndarray | None,
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
        radial_phase=radial_phase,
        ny_full=ny_full,
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


def _nonlinear_em_components_from_path(
    path: _PreparedNonlinearPath,
    ctx: _NonlinearBracketContext,
) -> dict[str, jnp.ndarray | None]:
    if path.laguerre is not None:
        return _laguerre_components_from_prepared(
            path.prep,
            path.laguerre,
            tz=ctx.tz,
            vth=ctx.vth,
            sqrt_m=ctx.sqrt_m,
            sqrt_m_p1=ctx.sqrt_m_p1,
            kx_grid=ctx.kx_grid,
            ky_grid=ctx.ky_grid,
            dealias_mask=ctx.dealias_mask,
            kxfac=ctx.kxfac,
            weight=ctx.weight,
            apar_weight=ctx.apar_weight,
            bpar_weight=ctx.bpar_weight,
            compressed_real_fft=ctx.compressed_real_fft,
            ny_full=ctx.ny_full,
            radial_phase=ctx.radial_phase,
        )
    return _spectral_components_from_prepared(
        path.prep,
        vth=ctx.vth,
        sqrt_m=ctx.sqrt_m,
        sqrt_m_p1=ctx.sqrt_m_p1,
        kx_grid=ctx.kx_grid,
        ky_grid=ctx.ky_grid,
        dealias_mask=ctx.dealias_mask,
        kxfac=ctx.kxfac,
        weight=ctx.weight,
        apar_weight=ctx.apar_weight,
        bpar_weight=ctx.bpar_weight,
        compressed_real_fft=ctx.compressed_real_fft,
        ny_full=ctx.ny_full,
        radial_phase=ctx.radial_phase,
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
    ny_full: int | None = None,
    laguerre_mode: str = "grid",
    radial_phase: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Return nonlinear E×B/flutter components for diagnostics/comparison checks."""

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
    components = _nonlinear_em_components_from_path(path, ctx)
    return _squeeze_component_payload(
        components,
        squeeze_species=path.prep.squeeze_species,
    )
