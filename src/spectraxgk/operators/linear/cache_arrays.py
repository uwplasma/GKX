"""Array factories and damping factors for the linear cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.core.velocity import J_l_all
from spectraxgk.operators.linear.params import LinearParams, _as_species_array

if TYPE_CHECKING:
    from spectraxgk.operators.linear.cache_model import LinearCache


def _shift_axis_for_cache(arr: jnp.ndarray, offset: int, axis: int) -> jnp.ndarray:
    """Shift an array along one axis with zeros introduced at the boundary."""

    axis = axis % arr.ndim
    if offset == 0:
        return arr
    n = arr.shape[axis]
    if abs(offset) >= n:
        return jnp.zeros_like(arr)
    out = jnp.zeros_like(arr)
    if offset > 0:
        body = jax.lax.slice_in_dim(arr, offset, n, axis=axis)
        starts = [0] * arr.ndim
        starts[axis] = 0
        return jax.lax.dynamic_update_slice(out, body, starts)
    body = jax.lax.slice_in_dim(arr, 0, n + offset, axis=axis)
    starts = [0] * arr.ndim
    starts[axis] = -offset
    return jax.lax.dynamic_update_slice(out, body, starts)


def _numpy_dtype_for_jax(real_dtype: jnp.dtype) -> type[np.float32] | type[np.float64]:
    return np.float64 if real_dtype == jnp.float64 else np.float32


def _build_low_rank_moment_cache_arrays(
    Nl: int,
    Nm: int,
    params: LinearParams,
    real_dtype: jnp.dtype,
) -> dict[str, jnp.ndarray]:
    """Build small moment-space cache arrays without many eager JAX dispatches."""

    np_dtype: Any = _numpy_dtype_for_jax(real_dtype)
    ell: Any = np.arange(Nl, dtype=np_dtype).reshape(Nl, 1, 1, 1, 1)
    m: Any = np.arange(Nm, dtype=np_dtype).reshape(1, Nm, 1, 1, 1)
    lb_lam_np = (
        float(params.nu_laguerre) * np.arange(Nl, dtype=np_dtype)[:, None]
        + float(params.nu_hermite) * np.arange(Nm, dtype=np_dtype)[None, :]
    )
    sqrt_shape = (1, 1, Nm, 1, 1, 1)
    hermite_index: Any = np.arange(Nm, dtype=np_dtype)
    sqrt_p_np = np.sqrt(hermite_index + np_dtype(1.0)).reshape(sqrt_shape)
    sqrt_m_ladder_np = np.sqrt(hermite_index).reshape(sqrt_shape)
    l_norm = np_dtype(max(Nl - 1, 1))
    m_norm = np_dtype(max(Nm - 1, 1))
    l_norm_full = np_dtype(max(Nl, 1))
    m_norm_full = np_dtype(max(Nm, 1))
    m_norm_kz = np_dtype(max(Nm - 1, 1))
    p_hyper_l = np_dtype(params.p_hyper_l)
    p_hyper_m = np_dtype(params.p_hyper_m)
    p_hyper_lm = np_dtype(params.p_hyper_lm)
    normalized_m = m / m_norm_kz
    return {
        "lb_lam": jnp.asarray(lb_lam_np, dtype=real_dtype),
        "l": jnp.asarray(ell, dtype=real_dtype),
        "m": jnp.asarray(m, dtype=real_dtype),
        "l4": jnp.asarray(
            np.arange(Nl, dtype=np_dtype).reshape(Nl, 1, 1, 1), dtype=real_dtype
        ),
        "sqrt_m": jnp.asarray(np.sqrt(m), dtype=real_dtype),
        "sqrt_m_p1": jnp.asarray(np.sqrt(m + np_dtype(1.0)), dtype=real_dtype),
        "sqrt_p": jnp.asarray(sqrt_p_np, dtype=real_dtype),
        "sqrt_m_ladder": jnp.asarray(sqrt_m_ladder_np, dtype=real_dtype),
        "hyper_ratio": jnp.asarray(
            (ell / l_norm) ** params.p_hyper + (m / m_norm) ** params.p_hyper,
            dtype=real_dtype,
        ),
        "ratio_l": jnp.asarray((ell / l_norm_full) ** p_hyper_l, dtype=real_dtype),
        "ratio_m": jnp.asarray((m / m_norm_full) ** p_hyper_m, dtype=real_dtype),
        "ratio_lm": jnp.asarray(
            ((2.0 * ell + m) / (2.0 * l_norm_full + m_norm_full)) ** p_hyper_lm,
            dtype=real_dtype,
        ),
        "mask_const": jnp.asarray((m > 2.0) | (ell > 1.0), dtype=bool),
        "mask_kz": jnp.asarray(m > 2.0, dtype=bool),
        "m_pow": jnp.asarray(normalized_m**p_hyper_m, dtype=real_dtype),
        "m_norm_kz_factor": jnp.asarray(
            (p_hyper_m + 0.5) / np.sqrt(m_norm_kz), dtype=real_dtype
        ),
    }


def _build_end_damping_profile_array(
    Nz: int,
    widthfrac: float,
    boundary: str,
    real_dtype: jnp.dtype,
) -> jnp.ndarray:
    """Build the one-dimensional end-damping profile as one host array."""

    np_dtype = np.float32
    width = max(1, int(np.floor(float(widthfrac) * int(Nz))))
    idx = np.arange(Nz, dtype=np_dtype)
    width_f = np_dtype(width)
    left_mask = idx <= width_f
    right_mask = idx >= (Nz - width_f)
    x_left = np.where(left_mask, idx / width_f, 0.0)
    x_right = np.where(right_mask, (Nz - idx) / width_f, 0.0)
    nu_left = np.where(left_mask, 1.0 - 2.0 * x_left * x_left / (1.0 + x_left**4), 0.0)
    nu_right = np.where(
        right_mask, 1.0 - 2.0 * x_right * x_right / (1.0 + x_right**4), 0.0
    )
    damp_profile_np = np.maximum(nu_left, nu_right).astype(np_dtype)
    if boundary == "periodic":
        damp_profile_np = np.zeros_like(damp_profile_np)
    return jnp.asarray(damp_profile_np, dtype=real_dtype)


def _build_gyroaverage_cache_arrays(
    b: jnp.ndarray,
    Nl: int,
    real_dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build species-major gyroaverage factors without a Python-level vmap."""

    Jl = jnp.moveaxis(J_l_all(b, l_max=Nl - 1), 0, 1).astype(real_dtype)
    JlB = Jl + _shift_axis_for_cache(Jl, -1, axis=1)
    return Jl, JlB.astype(real_dtype)


def hypercollision_damping(
    cache: "LinearCache",
    params: "LinearParams",
    real_dtype: jnp.dtype,
) -> jnp.ndarray:
    """Assemble reference-aligned hypercollision damping factors."""

    Nl = jnp.asarray(max(int(cache.l.shape[0]), 1), dtype=real_dtype)
    Nm = jnp.asarray(max(int(cache.m.shape[1]), 1), dtype=real_dtype)

    nu_hyper = jnp.asarray(params.nu_hyper, dtype=real_dtype)
    nu_hyper_l = jnp.asarray(params.nu_hyper_l, dtype=real_dtype)
    nu_hyper_m = jnp.asarray(params.nu_hyper_m, dtype=real_dtype)
    nu_hyper_lm = jnp.asarray(params.nu_hyper_lm, dtype=real_dtype)
    w_const = jnp.asarray(params.hypercollisions_const, dtype=real_dtype)
    w_kz = jnp.asarray(params.hypercollisions_kz, dtype=real_dtype)

    vth = jnp.asarray(params.vth, dtype=real_dtype)
    vth_s = vth if vth.ndim == 0 else vth[:, None, None, None, None, None]

    ratio_l = cache.ratio_l.astype(real_dtype)
    ratio_m = cache.ratio_m.astype(real_dtype)
    ratio_lm = cache.ratio_lm.astype(real_dtype)
    scaled_nu_l = Nl * nu_hyper_l
    scaled_nu_m = Nm * nu_hyper_m
    mask_const = cache.mask_const
    const_coeff = (
        vth_s * (scaled_nu_l * ratio_l + scaled_nu_m * ratio_m) + nu_hyper_lm * ratio_lm
    )

    hyper = nu_hyper * cache.hyper_ratio.astype(real_dtype)
    hyper = hyper + w_const * jnp.where(mask_const, const_coeff, 0.0)

    abs_kz = jnp.abs(cache.kz).astype(real_dtype)[None, None, None, None, None, :]
    nu_hyp_m = (
        nu_hyper_m
        * cache.m_norm_kz_factor.astype(real_dtype)
        * 2.3
        * vth_s
        * jnp.abs(jnp.asarray(params.kpar_scale, dtype=real_dtype))
    )
    kz_term = nu_hyp_m * cache.m_pow.astype(real_dtype) * abs_kz
    hyper = hyper + w_kz * jnp.where(cache.mask_kz, kz_term, 0.0)
    return hyper


def collision_damping(
    cache: "LinearCache",
    params: "LinearParams",
    real_dtype: jnp.dtype,
    *,
    squeeze_species: bool = False,
) -> jnp.ndarray:
    """Assemble collision damping from cached low-rank factors.

    Runtime caches store ``lb_lam`` as the Hermite-Laguerre Lenard-Bernstein
    diagonal only, with shape ``(Nl, Nm)``. Older tests may still provide a
    pre-expanded array; support that for compatibility.
    """

    lb_lam = cache.lb_lam.astype(real_dtype)
    if lb_lam.ndim == 2:
        b = jnp.asarray(cache.b, dtype=real_dtype)
        ns = int(b.shape[0])
        nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
        nu_s = nu[:, None, None, None, None, None]
        damping = nu_s * lb_lam[None, :, :, None, None, None]
        damping = damping + nu_s * b[:, None, None, ...]
        if squeeze_species:
            damping = damping[0]
        return damping.astype(real_dtype)

    if lb_lam.ndim == 6:
        ns = int(lb_lam.shape[0])
        nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
        damping = nu[:, None, None, None, None, None] * lb_lam
        if squeeze_species:
            damping = damping[0]
        return damping.astype(real_dtype)

    return (jnp.asarray(params.nu, dtype=real_dtype) * lb_lam).astype(real_dtype)

