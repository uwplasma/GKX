"""Validation helpers shared across term modules."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _is_tracer(x) -> bool:
    return isinstance(x, jax.core.Tracer)


def _check_positive(x, name: str) -> None:
    arr = jnp.asarray(x)
    if _is_tracer(x) or _is_tracer(arr):
        return
    if arr.ndim == 0:
        if float(arr) <= 0.0:
            raise ValueError(f"{name} must be > 0")
        return
    if np.any(np.asarray(arr) <= 0.0):
        raise ValueError(f"{name} must be > 0")


def _check_nonnegative(x, name: str) -> None:
    arr = jnp.asarray(x)
    if _is_tracer(x) or _is_tracer(arr):
        return
    if arr.ndim == 0:
        if float(arr) < 0.0:
            raise ValueError(f"{name} must be >= 0")
        return
    if np.any(np.asarray(arr) < 0.0):
        raise ValueError(f"{name} must be >= 0")
