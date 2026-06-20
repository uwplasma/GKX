"""Eigenmode branch-selection rules for linear solves."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import LinearParams


def _omega_scale(cache: LinearCache, params: LinearParams) -> jnp.ndarray:
    ky_scale = jnp.max(jnp.abs(cache.ky))
    rlt_i = jnp.abs(params.R_over_LTi)
    rlt_e = jnp.abs(params.R_over_LTe)
    rln = jnp.abs(params.R_over_Ln)

    def _max_scalar(arr: jnp.ndarray) -> jnp.ndarray:
        return arr if arr.ndim == 0 else jnp.max(arr)

    drive_i = _max_scalar(rlt_i)
    drive_e = _max_scalar(rlt_e)
    drive_n = _max_scalar(rln)
    drive = jnp.maximum(drive_i, jnp.maximum(drive_e, drive_n))
    return ky_scale * jnp.maximum(drive, 1.0e-8)


def _mode_family_sign(mode_family: str) -> int:
    key = mode_family.strip().lower()
    if key in {"ion", "itg", "cyclone", "positive"}:
        return 1
    if key in {"electron", "etg", "tem", "kbm", "negative"}:
        return -1
    return 0


def _physical_omega(imag_part: jnp.ndarray) -> jnp.ndarray:
    """Map eigenvalue imaginary part to reported physical frequency."""

    return -imag_part


def _select_by_overlap(
    eigvecs: jnp.ndarray,
    V: jnp.ndarray,
    v_ref: jnp.ndarray,
    mask: jnp.ndarray,
    fallback_idx: jnp.ndarray,
) -> jnp.ndarray:
    """Select eigenpair with maximal overlap to v_ref within mask."""
    beta = jnp.tensordot(jnp.conj(V), v_ref, axes=v_ref.ndim)
    overlap = jnp.abs(jnp.dot(jnp.conj(beta), eigvecs))
    overlap_masked = jnp.where(mask, overlap, -jnp.inf)
    has_mask = jnp.any(mask)
    idx = jnp.argmax(overlap_masked)
    return jnp.where(has_mask, idx, fallback_idx)


def _select_by_target(
    real_part: jnp.ndarray,
    imag_part: jnp.ndarray,
    mask: jnp.ndarray,
    omega_scale: jnp.ndarray,
    omega_target_factor: float,
    omega_sign: int,
    fallback_idx: jnp.ndarray,
) -> jnp.ndarray:
    omega_target_factor_val = jnp.asarray(omega_target_factor, dtype=imag_part.dtype)
    use_target = omega_target_factor_val > 0.0
    omega_target = omega_target_factor_val * omega_scale
    omega_phys = _physical_omega(imag_part)
    omega_sign_val = jnp.asarray(omega_sign, dtype=imag_part.dtype)
    use_sign = omega_sign_val != 0.0
    omega_target = jnp.where(use_sign, jnp.sign(omega_sign_val) * jnp.abs(omega_target), omega_target)
    use_mask = jnp.any(mask)
    mask_use = jnp.where(use_mask, mask, jnp.ones_like(mask, dtype=bool))
    mask_pos = real_part >= 0.0
    mask_target = mask_use
    has_pos = jnp.any(mask_target & mask_pos)
    mask_target = jnp.where(has_pos, mask_target & mask_pos, mask_target)
    dist = jnp.abs(omega_phys - omega_target)
    dist_masked = jnp.where(mask_target, dist, jnp.inf)
    idx_target = jnp.argmin(dist_masked)
    has_target = jnp.any(mask_target)
    use_choice = use_target & has_target
    return jnp.where(use_choice, idx_target, fallback_idx)


__all__ = [
    "_mode_family_sign",
    "_omega_scale",
    "_physical_omega",
    "_select_by_overlap",
    "_select_by_target",
]
