"""Preconditioners for shifted linear eigenmode solves."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.solvers.linear.eigen_operator import _compute_damping
from spectraxgk.terms.config import TermConfig


def _build_shift_invert_precond(
    v: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    sigma: jnp.ndarray,
    mode: str | None,
) -> tuple[jnp.ndarray | None, Callable[[jnp.ndarray], jnp.ndarray] | None]:
    if mode is None or mode.lower() == "none":
        return None, None
    mode_key = mode.lower()
    if mode_key == "damping":
        damping = _compute_damping(v, cache, params)
        diag = -damping.astype(v.dtype) - sigma
        safe = jnp.where(jnp.abs(diag) > 0.0, diag, 1.0 + 0.0j)
        precond = 1.0 / safe
        shape = v.shape
        size = v.size

        def apply_precond_damping(x_flat: jnp.ndarray) -> jnp.ndarray:
            x = x_flat.reshape(shape)
            return (x * precond).reshape(size)

        return precond, apply_precond_damping

    if mode_key not in {
        "hermite-line",
        "hermite_line",
        "hermite",
        "streaming-line",
        "streaming_line",
        "hermite-line-coarse",
        "hermite_line_coarse",
        "hermite_coarse",
        "streaming-line-coarse",
    }:
        return None, None

    # Hermite-line preconditioners rely on real-valued tridiagonal solves, but
    # shift-invert uses complex coefficients (streaming i*kz). Until a complex
    # tridiagonal solve is implemented, fall back to the diagonal damping
    # preconditioner when hermite-line is requested.
    damping = _compute_damping(v, cache, params)
    diag = -damping.astype(v.dtype) - sigma
    safe = jnp.where(jnp.abs(diag) > 0.0, diag, 1.0 + 0.0j)
    precond = 1.0 / safe
    shape = v.shape
    size = v.size

    def apply_precond_fallback(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (x * precond).reshape(size)

    return precond, apply_precond_fallback


__all__ = ["_build_shift_invert_precond"]
