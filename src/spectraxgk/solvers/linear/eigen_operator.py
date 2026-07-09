"""Matrix-free linear operator applications used by eigenmode solvers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.cache_arrays import (
    collision_damping,
    hypercollision_damping,
)
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.assembly import assemble_rhs_cached
from spectraxgk.terms.config import TermConfig


def _normalize(v: jnp.ndarray) -> jnp.ndarray:
    norm = jnp.linalg.norm(v)
    norm_safe = jnp.where(norm == 0.0, 1.0, norm)
    return v / norm_safe


@jax.jit
def _assemble_rhs_cached_novjp(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
) -> tuple[jnp.ndarray, object]:
    return assemble_rhs_cached(G, cache, params, terms=term_cfg, use_custom_vjp=False)


def _apply_operator(
    v: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
) -> jnp.ndarray:
    dG, _fields = _assemble_rhs_cached_novjp(v, cache, params, term_cfg)
    return dG


def _compute_damping(
    v: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
) -> jnp.ndarray:
    real_dtype = jnp.real(v).dtype
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if v.ndim == 5 and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    damping = collision_damping(cache, params, real_dtype, squeeze_species=(v.ndim == 5)) + hyper_damp
    return damping.astype(real_dtype)


def _advance_imex2(
    v: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    damping = _compute_damping(v, cache, params)
    dG = _apply_operator(v, cache, params, term_cfg)
    dG_explicit = dG + damping * v
    v_half = (v + 0.5 * dt * dG_explicit) / (1.0 + 0.5 * dt * damping)
    dG_half = _apply_operator(v_half, cache, params, term_cfg)
    dG_half_exp = dG_half + damping * v_half
    return (v + dt * dG_half_exp) / (1.0 + dt * damping)


__all__ = [
    "_advance_imex2",
    "_apply_operator",
    "_assemble_rhs_cached_novjp",
    "_compute_damping",
    "_normalize",
]
