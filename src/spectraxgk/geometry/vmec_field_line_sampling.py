"""JAX field-line sampling helpers for VMEC tensor sensitivity gates."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def _rms_with_floor(arr: jnp.ndarray, epsilon: jnp.ndarray | float) -> jnp.ndarray:
    """Return a differentiable RMS with a small floor for zero-valued tensors."""

    values = jnp.asarray(arr)
    return jnp.sqrt(jnp.mean(values * values) + epsilon)


def _vmec_field_line_sampling_coordinates(
    wout: Any,
    *,
    surface_index: int,
    alpha: float,
    ntheta: int,
    dtype: Any,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return VMEC field-line coordinates used by tensor sensitivity gates."""

    ntheta_int = int(ntheta)
    if ntheta_int < 4:
        raise ValueError("ntheta must be >= 4")

    iota_profile = jnp.asarray(getattr(wout, "iotas"))
    sidx = int(surface_index)
    if iota_profile.ndim != 1 or int(iota_profile.shape[0]) <= sidx:
        raise RuntimeError(
            "vmec_jax wout iotas profile is missing or incompatible with the state grid"
        )
    iota_line = iota_profile[sidx]
    iota_safe = jnp.where(
        jnp.abs(iota_line) < 1.0e-12,
        jnp.sign(iota_line + 1.0e-30) * 1.0e-12,
        iota_line,
    )
    theta_line = jnp.linspace(-jnp.pi, jnp.pi, ntheta_int, endpoint=False, dtype=dtype)
    theta_vmec = jnp.mod(theta_line + jnp.pi, 2.0 * jnp.pi)
    zeta_line = jnp.mod(
        (theta_vmec - jnp.asarray(float(alpha), dtype=dtype)) / iota_safe,
        2.0 * jnp.pi,
    )
    return iota_line, iota_safe, theta_line, theta_vmec, zeta_line


__all__ = [
    "_rms_with_floor",
    "_vmec_field_line_sampling_coordinates",
]
