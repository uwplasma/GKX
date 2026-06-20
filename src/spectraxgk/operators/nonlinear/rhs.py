"""Nonlinear RHS assembly helpers.

This module owns the performance-critical RHS composition path.  The public
``spectraxgk.nonlinear`` facade passes its module-level callables into these
helpers so existing tests and downstream monkeypatch/debug workflows keep the
same seams while the implementation stays isolated and easier to profile.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.assembly import (
    _is_static_zero,
    assemble_rhs_cached_electrostatic_jit,
    assemble_rhs_cached_jit,
)
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.nonlinear import nonlinear_em_contribution

RhsCallable = Callable[..., tuple[jnp.ndarray, FieldState]]
StaticZeroCallable = Callable[[object], bool]
NonlinearContributionCallable = Callable[..., jnp.ndarray]
FieldSolveCallable = Callable[..., FieldState]


def linear_rhs_jit_for_terms_impl(
    term_cfg: TermConfig,
    *,
    electrostatic_rhs_fn: RhsCallable = assemble_rhs_cached_electrostatic_jit,
    full_rhs_fn: RhsCallable = assemble_rhs_cached_jit,
    is_static_zero_fn: StaticZeroCallable = _is_static_zero,
) -> RhsCallable:
    """Return the narrowest compiled linear RHS path compatible with ``term_cfg``."""

    return (
        electrostatic_rhs_fn
        if is_static_zero_fn(term_cfg.apar) and is_static_zero_fn(term_cfg.bpar)
        else full_rhs_fn
    )


def nonlinear_rhs_cached_impl(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: TermConfig | None = None,
    *,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    external_phi: jnp.ndarray | float | None = None,
    electrostatic_rhs_fn: RhsCallable = assemble_rhs_cached_electrostatic_jit,
    full_rhs_fn: RhsCallable = assemble_rhs_cached_jit,
    is_static_zero_fn: StaticZeroCallable = _is_static_zero,
    nonlinear_contribution_fn: NonlinearContributionCallable = nonlinear_em_contribution,
) -> tuple[jnp.ndarray, FieldState]:
    """Compute the assembled nonlinear RHS and electromagnetic field state."""

    term_cfg = terms or TermConfig()
    linear_rhs_fn = linear_rhs_jit_for_terms_impl(
        term_cfg,
        electrostatic_rhs_fn=electrostatic_rhs_fn,
        full_rhs_fn=full_rhs_fn,
        is_static_zero_fn=is_static_zero_fn,
    )
    dG, fields = linear_rhs_fn(G, cache, params, term_cfg, external_phi=external_phi)
    if term_cfg.nonlinear != 0.0:
        real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
        weight = jnp.asarray(term_cfg.nonlinear, dtype=real_dtype)
        apar = (
            None
            if fields.apar is None or is_static_zero_fn(term_cfg.apar)
            else fields.apar
        )
        bpar = (
            None
            if fields.bpar is None or is_static_zero_fn(term_cfg.bpar)
            else fields.bpar
        )
        dG = dG + nonlinear_contribution_fn(
            G,
            phi=fields.phi,
            apar=apar,
            bpar=bpar,
            Jl=cache.Jl,
            JlB=cache.JlB,
            tz=jnp.asarray(params.tz),
            vth=jnp.asarray(params.vth),
            sqrt_m=cache.sqrt_m,
            sqrt_m_p1=cache.sqrt_m_p1,
            kx_grid=cache.kx_grid,
            ky_grid=cache.ky_grid,
            dealias_mask=cache.dealias_mask,
            kxfac=cache.kxfac,
            weight=weight,
            apar_weight=float(term_cfg.apar),
            bpar_weight=float(term_cfg.bpar),
            laguerre_to_grid=cache.laguerre_to_grid,
            laguerre_to_spectral=cache.laguerre_to_spectral,
            laguerre_roots=cache.laguerre_roots,
            laguerre_j0=cache.laguerre_j0,
            laguerre_j1_over_alpha=cache.laguerre_j1_over_alpha,
            b=cache.b,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
        )
    return dG, fields


def nonlinear_em_term_cached_impl(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    *,
    real_dtype: Any | None = None,
    external_phi: jnp.ndarray | float | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    fields_fn: FieldSolveCallable,
    nonlinear_contribution_fn: NonlinearContributionCallable = nonlinear_em_contribution,
) -> jnp.ndarray:
    """Return the explicit electromagnetic nonlinear term used by IMEX paths."""

    if term_cfg.nonlinear == 0.0:
        return jnp.zeros_like(G)
    dtype = real_dtype or jnp.real(jnp.empty((), dtype=G.dtype)).dtype
    weight = jnp.asarray(term_cfg.nonlinear, dtype=dtype)
    fields = fields_fn(G, cache, params, terms=term_cfg, external_phi=external_phi)
    return nonlinear_contribution_fn(
        G,
        phi=fields.phi,
        apar=fields.apar,
        bpar=fields.bpar,
        Jl=cache.Jl,
        JlB=cache.JlB,
        tz=jnp.asarray(params.tz),
        vth=jnp.asarray(params.vth),
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        kx_grid=cache.kx_grid,
        ky_grid=cache.ky_grid,
        dealias_mask=cache.dealias_mask,
        kxfac=cache.kxfac,
        weight=weight,
        apar_weight=float(term_cfg.apar),
        bpar_weight=float(term_cfg.bpar),
        laguerre_to_grid=cache.laguerre_to_grid,
        laguerre_to_spectral=cache.laguerre_to_spectral,
        laguerre_roots=cache.laguerre_roots,
        laguerre_j0=cache.laguerre_j0,
        laguerre_j1_over_alpha=cache.laguerre_j1_over_alpha,
        b=cache.b,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
    )


__all__ = [
    "RhsCallable",
    "linear_rhs_jit_for_terms_impl",
    "nonlinear_em_term_cached_impl",
    "nonlinear_rhs_cached_impl",
]
