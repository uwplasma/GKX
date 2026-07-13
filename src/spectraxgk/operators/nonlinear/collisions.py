"""Collision and hypercollision split helpers for nonlinear integrations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import jax.numpy as jnp

from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.cache_arrays import (
    hypercollision_damping,
)
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.config import TermConfig

__all__ = [
    "NonlinearCollisionSplitPolicy",
    "_apply_collision_split",
    "_collision_damping",
    "build_nonlinear_collision_split_policy",
]


@dataclass(frozen=True)
class NonlinearCollisionSplitPolicy:
    """Collision split settings shared by explicit and IMEX diagnostics."""

    active: bool
    rhs_terms: TermConfig
    damping: jnp.ndarray | None


def _collision_damping(
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    real_dtype: jnp.dtype,
    *,
    squeeze_species: bool,
) -> jnp.ndarray:
    """Assemble the diagonal hypercollision damping safe to split.

    The conserving collision operator includes non-diagonal field-particle
    corrections and must remain in the RHS unless an operator supplies its own
    mathematically valid split update.
    """

    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    hyper_w = jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype)
    if squeeze_species and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    return (hyper_w * hyper_damp).astype(real_dtype)


def build_nonlinear_collision_split_policy(
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    real_dtype: jnp.dtype,
    *,
    squeeze_species: bool,
    collision_split: bool,
    collision_damping_fn: Callable[..., jnp.ndarray] = _collision_damping,
) -> NonlinearCollisionSplitPolicy:
    """Build collision splitting weights and RHS terms for nonlinear scans."""

    active = bool(collision_split) and float(term_cfg.hypercollisions) != 0.0
    rhs_terms = replace(term_cfg, hypercollisions=0.0) if active else term_cfg
    damping = (
        collision_damping_fn(
            cache, params, term_cfg, real_dtype, squeeze_species=squeeze_species
        )
        if active
        else None
    )
    return NonlinearCollisionSplitPolicy(
        active=active,
        rhs_terms=rhs_terms,
        damping=damping,
    )


def _apply_collision_split(
    G: jnp.ndarray,
    damping: jnp.ndarray,
    dt_local: jnp.ndarray,
    scheme: str,
) -> jnp.ndarray:
    """Apply a diagonal collision/hypercollision split update."""

    scheme_key = scheme.strip().lower()
    if scheme_key in {"implicit", "imex"}:
        return G / (1.0 + dt_local * damping)
    if scheme_key in {"exp", "sts", "rkc", "rkc2"}:
        # For diagonal collision operators the exponential update is exact and
        # behaves like a stabilized explicit (STS/RKC) limit.
        return G * jnp.exp(-dt_local * damping)
    raise ValueError("collision_scheme must be one of {'implicit', 'exp', 'sts', 'rkc'}")
