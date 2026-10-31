"""Nonlinear RHS assembly helpers.

This module owns the performance-critical RHS composition path.  The nonlinear
integration solvers in :mod:`gkx.solvers_nonlinear_diagnostics` pass their
module-level callables into these helpers so existing tests and downstream
monkeypatch/debug workflows keep the same seams while the implementation stays
isolated and easier to profile.
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

from gkx.operators.collision import CollisionOperator
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import LinearParams
from gkx.terms.assembly import (
    _is_static_zero,
    assemble_linear_rhs_cached,
    assemble_rhs_cached,
    assemble_rhs_cached_electrostatic_jit,
    assemble_rhs_cached_jit,
    linear_rhs_jit_for_terms as linear_rhs_jit_for_terms_impl,
    split_species,
)
from gkx.terms.config import FieldState, TermConfig
from gkx.terms.nonlinear import nonlinear_em_contribution

RhsCallable = Callable[..., tuple[jnp.ndarray, FieldState]]
StaticZeroCallable = Callable[[object], bool]
NonlinearContributionCallable = Callable[..., jnp.ndarray]
FieldSolveCallable = Callable[..., FieldState]


def _bracket(
    fn: NonlinearContributionCallable,
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    **kwargs: Any,
) -> jnp.ndarray:
    """The nonlinear bracket of ``G``, run per species group when moments differ."""

    def one(
        g: jnp.ndarray, c: LinearCache, p: LinearParams, _: Any = None
    ) -> jnp.ndarray:
        return fn(
            g,
            Jl=c.Jl,
            JlB=c.JlB,
            tz=jnp.asarray(p.tz),
            vth=jnp.asarray(p.vth),
            sqrt_m=c.sqrt_m,
            sqrt_m_p1=c.sqrt_m_p1,
            kx_grid=c.kx_grid,
            ky_grid=c.ky_grid,
            ny_full=getattr(c, "ny_full", None),
            dealias_mask=c.dealias_mask,
            kxfac=c.kxfac,
            laguerre_to_grid=c.laguerre_to_grid,
            laguerre_to_spectral=c.laguerre_to_spectral,
            laguerre_roots=c.laguerre_roots,
            laguerre_j0=c.laguerre_j0,
            laguerre_j1_over_alpha=c.laguerre_j1_over_alpha,
            b=c.b,
            **kwargs,
        )

    if getattr(cache, "groups", None) is None:
        return one(G, cache, params)
    return split_species(G, cache, params, one)


def nonlinear_rhs_cached_impl(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: TermConfig | None = None,
    *,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    external_phi: jnp.ndarray | float | None = None,
    collision_operator: CollisionOperator | None = None,
    radial_phase: jnp.ndarray | None = None,
    differentiable: bool = False,
    electrostatic_rhs_fn: RhsCallable = assemble_rhs_cached_electrostatic_jit,
    full_rhs_fn: RhsCallable = assemble_rhs_cached_jit,
    differentiable_rhs_fn: RhsCallable = assemble_rhs_cached,
    is_static_zero_fn: StaticZeroCallable = _is_static_zero,
    nonlinear_contribution_fn: NonlinearContributionCallable = nonlinear_em_contribution,
) -> tuple[jnp.ndarray, FieldState]:
    """Compute the assembled nonlinear RHS and electromagnetic field state."""

    term_cfg = terms or TermConfig()
    dG, fields = assemble_linear_rhs_cached(
        G,
        cache,
        params,
        terms=term_cfg,
        use_jit=not differentiable,
        use_custom_vjp=not differentiable,
        external_phi=external_phi,
        electrostatic_fields=None,
        collision_operator=collision_operator,
        electrostatic_rhs_fn=electrostatic_rhs_fn,
        full_rhs_fn=full_rhs_fn,
        eager_rhs_fn=differentiable_rhs_fn,
        is_static_zero_fn=is_static_zero_fn,
    )
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
        dG = dG + _bracket(
            nonlinear_contribution_fn,
            G,
            cache,
            params,
            phi=fields.phi,
            apar=apar,
            bpar=bpar,
            weight=weight,
            apar_weight=float(term_cfg.apar),
            bpar_weight=float(term_cfg.bpar),
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            radial_phase=radial_phase,
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
    radial_phase: jnp.ndarray | None = None,
    fields_fn: FieldSolveCallable,
    nonlinear_contribution_fn: NonlinearContributionCallable = nonlinear_em_contribution,
) -> jnp.ndarray:
    """Return the explicit electromagnetic nonlinear term used by IMEX paths."""

    if term_cfg.nonlinear == 0.0:
        return jnp.zeros_like(G)
    dtype = real_dtype or jnp.real(jnp.empty((), dtype=G.dtype)).dtype
    weight = jnp.asarray(term_cfg.nonlinear, dtype=dtype)
    fields = fields_fn(G, cache, params, terms=term_cfg, external_phi=external_phi)
    return _bracket(
        nonlinear_contribution_fn,
        G,
        cache,
        params,
        phi=fields.phi,
        apar=fields.apar,
        bpar=fields.bpar,
        weight=weight,
        apar_weight=float(term_cfg.apar),
        bpar_weight=float(term_cfg.bpar),
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        radial_phase=radial_phase,
    )


__all__ = [
    "RhsCallable",
    "linear_rhs_jit_for_terms_impl",
    "nonlinear_em_term_cached_impl",
    "nonlinear_rhs_cached_impl",
]
