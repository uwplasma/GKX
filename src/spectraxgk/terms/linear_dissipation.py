"""Linear collisional, hypercollisional, and damping term contributions."""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from spectraxgk.terms.operators import abs_z_linked_fft, shift_axis


def _is_static_zero(value: jnp.ndarray, dtype: jnp.dtype | None = None) -> bool:
    arr = jnp.asarray(value, dtype=dtype)
    if isinstance(arr, jax.core.Tracer):
        return False
    return bool(np.all(np.asarray(arr) == 0.0))


def _zeros_like_result(x: jnp.ndarray, *values: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(x, dtype=jnp.result_type(x, *values))


def _hermite_mode_drive(
    template: jnp.ndarray,
    mode: int,
    drive: jnp.ndarray,
) -> jnp.ndarray:
    """Embed a single-Hermite-mode drive into a full state-shaped array."""

    mask = jnp.arange(template.shape[2], dtype=jnp.int32)[None, None, :, None, None, None]
    return (mask == int(mode)).astype(template.dtype) * drive[:, :, None, ...]


def _species_collision_frequency(
    nu: jnp.ndarray,
    *,
    ns: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Return one collision frequency per species."""

    nu_arr = jnp.asarray(nu, dtype=dtype).reshape(-1)
    if nu_arr.size == 1:
        return jnp.broadcast_to(nu_arr, (ns,))
    if int(nu_arr.size) != int(ns):
        raise ValueError(f"nu must have length {ns} (got {nu_arr.size})")
    return nu_arr


def _collision_magnetic_shift(
    b: jnp.ndarray | None,
    *,
    H_ndim: int,
    ns: int,
    dtype: jnp.dtype,
) -> jnp.ndarray | None:
    if b is None:
        return None
    b_s = jnp.asarray(b, dtype=dtype)
    if H_ndim == 6:
        if b_s.ndim == 3:
            b_s = jnp.broadcast_to(b_s, (ns,) + b_s.shape)
        return b_s[:, None, None, ...]
    if b_s.ndim == 4:
        b_s = b_s[0]
    return b_s[None, None, ...]


def _laguerre_collision_base(
    H: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    lb_lam: jnp.ndarray,
    b: jnp.ndarray | None,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    lb_arr = jnp.asarray(lb_lam, dtype=dtype)
    if lb_arr.ndim == 2 and H.ndim == 6:
        ns = int(H.shape[0])
        nu_s = _species_collision_frequency(nu, ns=ns, dtype=dtype)[
            :, None, None, None, None, None
        ]
        base = nu_s * lb_arr[None, :, :, None, None, None]
        b_shift = _collision_magnetic_shift(b, H_ndim=H.ndim, ns=ns, dtype=dtype)
        return base if b_shift is None else base + nu_s * b_shift
    if lb_arr.ndim == 2:
        nu0 = _species_collision_frequency(nu, ns=1, dtype=dtype)[0]
        base = nu0 * lb_arr[:, :, None, None, None]
        b_shift = _collision_magnetic_shift(b, H_ndim=H.ndim, ns=1, dtype=dtype)
        return base if b_shift is None else base + nu0 * b_shift
    if lb_arr.ndim == 6:
        ns = int(lb_arr.shape[0])
        base = (
            _species_collision_frequency(nu, ns=ns, dtype=dtype)[
                :, None, None, None, None, None
            ]
            * lb_arr
        )
        return base[0] if H.ndim == 5 else base
    return jnp.asarray(nu, dtype=dtype) * lb_arr


def _collision_base_operator(
    H: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    collision_lam: jnp.ndarray | None,
    lb_lam: jnp.ndarray | None,
    b: jnp.ndarray | None,
    dtype: jnp.dtype,
) -> jnp.ndarray | None:
    if collision_lam is not None:
        collision_arr = jnp.asarray(collision_lam, dtype=dtype)
        if collision_arr.size != 0:
            return collision_arr
    if _is_static_zero(nu, dtype):
        return None
    if lb_lam is None:
        return jnp.zeros_like(H, dtype=dtype)
    return _laguerre_collision_base(H, nu=nu, lb_lam=lb_lam, b=b, dtype=dtype)


def _laguerre_temperature_coupling(
    H_m0: jnp.ndarray,
    G_m2: jnp.ndarray,
    Jl: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    Jl_m1 = shift_axis(Jl, -1, axis=1)
    Jl_p1 = shift_axis(Jl, 1, axis=1)
    coeff_t = jnp.arange(Jl.shape[1], dtype=jnp.real(H_m0).dtype)[
        None, :, None, None, None
    ]
    coeff_t = coeff_t * Jl_m1 + 2.0 * coeff_t * Jl + (coeff_t + 1.0) * Jl_p1
    if int(Jl.shape[1]) == 1:
        t_bar = jnp.sqrt(2.0) * jnp.sum(Jl * G_m2, axis=1)
    else:
        t_bar = (jnp.sqrt(2.0) / 3.0) * jnp.sum(Jl * G_m2, axis=1) + (
            2.0 / 3.0
        ) * jnp.sum(coeff_t * H_m0, axis=1)
    return coeff_t, t_bar


def _collision_moment_correction(
    H: jnp.ndarray,
    *,
    G: jnp.ndarray,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    b: jnp.ndarray,
    nu: jnp.ndarray,
) -> jnp.ndarray:
    nu_s = nu[:, None, None, None, None]
    b_s = jnp.asarray(b, dtype=jnp.real(H).dtype)
    sqrt_b = jnp.sqrt(jnp.maximum(b_s, 0.0))
    H_m0 = H[:, :, 0, ...]
    Nm = H.shape[2]
    H_m1 = H[:, :, 1, ...] if Nm > 1 else jnp.zeros_like(H_m0)
    G_m2 = G[:, :, 2, ...] if Nm > 2 else jnp.zeros_like(H_m0)

    coeff_t, t_bar = _laguerre_temperature_coupling(H_m0, G_m2, Jl)
    uperp_bar = sqrt_b * jnp.sum(JlB * H_m0, axis=1)
    upar_bar = jnp.sum(Jl * H_m1, axis=1)

    corr = jnp.zeros_like(H)
    corr_m0 = (
        nu_s * sqrt_b[:, None, ...] * JlB * uperp_bar[:, None, ...]
        + nu_s * 2.0 * coeff_t * t_bar[:, None, ...]
    )
    corr = corr + _hermite_mode_drive(corr, 0, corr_m0)
    if Nm > 1:
        corr = corr + _hermite_mode_drive(corr, 1, nu_s * Jl * upar_bar[:, None, ...])
    if Nm > 2:
        corr = corr + _hermite_mode_drive(
            corr, 2, nu_s * jnp.sqrt(2.0) * Jl * t_bar[:, None, ...]
        )
    return corr


def collisions_contribution(
    H: jnp.ndarray,
    *,
    G: jnp.ndarray | None = None,
    Jl: jnp.ndarray | None = None,
    JlB: jnp.ndarray | None = None,
    b: jnp.ndarray | None = None,
    nu: jnp.ndarray,
    collision_lam: jnp.ndarray | None = None,
    lb_lam: jnp.ndarray | None = None,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    real_dtype = jnp.real(H).dtype
    if _is_static_zero(weight, real_dtype):
        return jnp.zeros_like(H)
    collision_base = _collision_base_operator(
        H,
        nu=nu,
        collision_lam=collision_lam,
        lb_lam=lb_lam,
        b=b,
        dtype=real_dtype,
    )
    if collision_base is None:
        return jnp.zeros_like(H)

    base = -(H * collision_base) * weight
    if G is None or Jl is None or JlB is None or b is None:
        return base

    corr = _collision_moment_correction(H, G=G, Jl=Jl, JlB=JlB, b=b, nu=nu)
    return base + weight * corr


def _hypercollision_zero_result(
    G: jnp.ndarray,
    *,
    weight: jnp.ndarray,
    nu_hyper: jnp.ndarray,
    nu_hyper_l: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    nu_hyper_lm: jnp.ndarray,
    hypercollisions_const: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
) -> jnp.ndarray:
    return _zeros_like_result(
        G,
        weight,
        nu_hyper,
        nu_hyper_l,
        nu_hyper_m,
        nu_hyper_lm,
        hypercollisions_const,
        hypercollisions_kz,
    )


def _hypercollision_operator_is_static_zero(
    *,
    weight: jnp.ndarray,
    nu_hyper: jnp.ndarray,
    nu_hyper_l: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    nu_hyper_lm: jnp.ndarray,
    hypercollisions_const: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
    dtype: jnp.dtype,
) -> bool:
    const_branch_zero = (
        _is_static_zero(weight * hypercollisions_const * nu_hyper_l, dtype)
        and _is_static_zero(weight * hypercollisions_const * nu_hyper_m, dtype)
        and _is_static_zero(weight * hypercollisions_const * nu_hyper_lm, dtype)
    )
    isotropic_branch_zero = _is_static_zero(weight * nu_hyper, dtype)
    kz_branch_zero = _is_static_zero(weight * hypercollisions_kz * nu_hyper_m, dtype)
    return const_branch_zero and isotropic_branch_zero and kz_branch_zero


def _constant_hypercollision_contribution(
    G: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    nu_hyper: jnp.ndarray,
    nu_hyper_l: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    nu_hyper_lm: jnp.ndarray,
    hyper_ratio: jnp.ndarray,
    ratio_l: jnp.ndarray,
    ratio_m: jnp.ndarray,
    ratio_lm: jnp.ndarray,
    mask_const: jnp.ndarray,
    hypercollisions_const: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    l_norm = jnp.asarray(max(G.shape[1], 1), dtype=ratio_l.dtype)
    m_norm = jnp.asarray(max(G.shape[2], 1), dtype=ratio_m.dtype)
    scaled_nu_l = l_norm * nu_hyper_l
    scaled_nu_m = m_norm * nu_hyper_m
    vth_s = vth[:, None, None, None, None, None]
    const_term = -(
        vth_s * (scaled_nu_l * ratio_l + scaled_nu_m * ratio_m) + nu_hyper_lm * ratio_lm
    )
    dG = weight * hypercollisions_const * jnp.where(mask_const, const_term, 0.0) * G
    return dG - weight * nu_hyper * hyper_ratio * G


def _hypercollision_kz_source(
    G: jnp.ndarray,
    *,
    weight: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    m_norm_kz_factor: jnp.ndarray,
    vth: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    mask_kz: jnp.ndarray,
    m_pow: jnp.ndarray,
) -> jnp.ndarray:
    """Return the pre-``|k_z|`` source used by hypercollisional damping."""

    vth_s = vth[:, None, None, None, None, None]
    kz_weight = jnp.asarray(weight) * jnp.asarray(hypercollisions_kz)
    nu_hyp_m = nu_hyper_m * m_norm_kz_factor * 2.3 * vth_s * jnp.abs(kpar_scale)
    return kz_weight * jnp.where(mask_kz, -nu_hyp_m * m_pow, 0.0) * G


def _apply_parallel_hypercollision(
    kz_source: jnp.ndarray,
    *,
    kz: jnp.ndarray,
    linked_indices: tuple[jnp.ndarray, ...] | None,
    linked_kz: tuple[jnp.ndarray, ...] | None,
    linked_inverse_permutation: jnp.ndarray | None,
    linked_full_cover: bool,
    linked_gather_map: jnp.ndarray | None,
    linked_gather_mask: jnp.ndarray | None,
    linked_use_gather: bool,
) -> jnp.ndarray:
    if linked_indices and linked_kz:
        return abs_z_linked_fft(
            kz_source,
            linked_indices=linked_indices,
            linked_kz=linked_kz,
            linked_inverse_permutation=linked_inverse_permutation,
            linked_full_cover=linked_full_cover,
            linked_gather_map=linked_gather_map,
            linked_gather_mask=linked_gather_mask,
            linked_use_gather=linked_use_gather,
        )
    abs_kz = jnp.abs(kz)[None, None, None, None, None, :]
    return abs_kz * kz_source


def _inactive_hypercollision_result(
    G: jnp.ndarray,
    *,
    weight: jnp.ndarray,
    nu_hyper: jnp.ndarray,
    nu_hyper_l: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    nu_hyper_lm: jnp.ndarray,
    hypercollisions_const: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
    dtype: jnp.dtype,
) -> jnp.ndarray | None:
    """Return a zero result when all hypercollision branches are statically off."""

    if _is_static_zero(weight, dtype) or _hypercollision_operator_is_static_zero(
        weight=weight,
        nu_hyper=nu_hyper,
        nu_hyper_l=nu_hyper_l,
        nu_hyper_m=nu_hyper_m,
        nu_hyper_lm=nu_hyper_lm,
        hypercollisions_const=hypercollisions_const,
        hypercollisions_kz=hypercollisions_kz,
        dtype=dtype,
    ):
        return _hypercollision_zero_result(
            G,
            weight=weight,
            nu_hyper=nu_hyper,
            nu_hyper_l=nu_hyper_l,
            nu_hyper_m=nu_hyper_m,
            nu_hyper_lm=nu_hyper_lm,
            hypercollisions_const=hypercollisions_const,
            hypercollisions_kz=hypercollisions_kz,
        )
    return None


def _hypercollision_kz_weight_is_static_zero(
    *,
    weight: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
) -> bool:
    """Return true when the parallel hypercollision branch is statically off."""

    kz_weight = jnp.asarray(weight) * jnp.asarray(hypercollisions_kz)
    return not isinstance(kz_weight, jax.core.Tracer) and bool(
        np.all(np.asarray(kz_weight) == 0.0)
    )


def _parallel_hypercollision_contribution(
    G: jnp.ndarray,
    *,
    weight: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    m_norm_kz_factor: jnp.ndarray,
    vth: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    mask_kz: jnp.ndarray,
    m_pow: jnp.ndarray,
    kz: jnp.ndarray,
    linked_indices: tuple[jnp.ndarray, ...] | None,
    linked_kz: tuple[jnp.ndarray, ...] | None,
    linked_inverse_permutation: jnp.ndarray | None,
    linked_full_cover: bool,
    linked_gather_map: jnp.ndarray | None,
    linked_gather_mask: jnp.ndarray | None,
    linked_use_gather: bool,
) -> jnp.ndarray:
    """Compute the ``|k_z|`` hypercollision branch, including linked tubes."""

    kz_source = _hypercollision_kz_source(
        G,
        weight=weight,
        hypercollisions_kz=hypercollisions_kz,
        nu_hyper_m=nu_hyper_m,
        m_norm_kz_factor=m_norm_kz_factor,
        vth=vth,
        kpar_scale=kpar_scale,
        mask_kz=mask_kz,
        m_pow=m_pow,
    )
    return _apply_parallel_hypercollision(
        kz_source,
        kz=kz,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_inverse_permutation=linked_inverse_permutation,
        linked_full_cover=linked_full_cover,
        linked_gather_map=linked_gather_map,
        linked_gather_mask=linked_gather_mask,
        linked_use_gather=linked_use_gather,
    )


def hypercollisions_contribution(
    G: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    nu_hyper: jnp.ndarray,
    nu_hyper_l: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    nu_hyper_lm: jnp.ndarray,
    hyper_ratio: jnp.ndarray,
    ratio_l: jnp.ndarray,
    ratio_m: jnp.ndarray,
    ratio_lm: jnp.ndarray,
    mask_const: jnp.ndarray,
    mask_kz: jnp.ndarray,
    m_pow: jnp.ndarray,
    m_norm_kz_factor: jnp.ndarray,
    kz: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    hypercollisions_const: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
    weight: jnp.ndarray,
    linked_indices: tuple[jnp.ndarray, ...] | None = None,
    linked_kz: tuple[jnp.ndarray, ...] | None = None,
    linked_inverse_permutation: jnp.ndarray | None = None,
    linked_full_cover: bool = False,
    linked_gather_map: jnp.ndarray | None = None,
    linked_gather_mask: jnp.ndarray | None = None,
    linked_use_gather: bool = False,
) -> jnp.ndarray:
    real_dtype = jnp.real(G).dtype
    inactive_result = _inactive_hypercollision_result(
        G,
        weight=weight,
        nu_hyper=nu_hyper,
        nu_hyper_l=nu_hyper_l,
        nu_hyper_m=nu_hyper_m,
        nu_hyper_lm=nu_hyper_lm,
        hypercollisions_const=hypercollisions_const,
        hypercollisions_kz=hypercollisions_kz,
        dtype=real_dtype,
    )
    if inactive_result is not None:
        return inactive_result

    dG = _constant_hypercollision_contribution(
        G,
        vth=vth,
        nu_hyper=nu_hyper,
        nu_hyper_l=nu_hyper_l,
        nu_hyper_m=nu_hyper_m,
        nu_hyper_lm=nu_hyper_lm,
        hyper_ratio=hyper_ratio,
        ratio_l=ratio_l,
        ratio_m=ratio_m,
        ratio_lm=ratio_lm,
        mask_const=mask_const,
        hypercollisions_const=hypercollisions_const,
        weight=weight,
    )

    if _hypercollision_kz_weight_is_static_zero(
        weight=weight, hypercollisions_kz=hypercollisions_kz
    ):
        return dG

    return dG + _parallel_hypercollision_contribution(
        G,
        weight=weight,
        hypercollisions_kz=hypercollisions_kz,
        nu_hyper_m=nu_hyper_m,
        m_norm_kz_factor=m_norm_kz_factor,
        vth=vth,
        kpar_scale=kpar_scale,
        mask_kz=mask_kz,
        m_pow=m_pow,
        kz=kz,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_inverse_permutation=linked_inverse_permutation,
        linked_full_cover=linked_full_cover,
        linked_gather_map=linked_gather_map,
        linked_gather_mask=linked_gather_mask,
        linked_use_gather=linked_use_gather,
    )


def hyperdiffusion_contribution(
    G: jnp.ndarray,
    *,
    kx: jnp.ndarray,
    ky: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    D_hyper: jnp.ndarray,
    p_hyper_kperp: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Hyperdiffusion in k_perp following Laguerre-Hermite conventions."""

    real_dtype = jnp.real(G).dtype
    if _is_static_zero(weight, real_dtype) or _is_static_zero(D_hyper, real_dtype):
        return _zeros_like_result(G, weight, D_hyper)

    kx2 = kx * kx
    ky2 = ky * ky
    kperp2 = ky2[:, None] + kx2[None, :]

    nx = kx.size
    ny = ky.size
    kx_idx = max((nx - 1) // 3, 0)
    ky_idx = max((ny - 1) // 3, 0)
    kperp2_max = kx2[kx_idx] + ky2[ky_idx]
    kperp2_max = jnp.where(kperp2_max > 0.0, kperp2_max, 1.0)

    Dfac = D_hyper * (kperp2 / kperp2_max) ** p_hyper_kperp
    mask = dealias_mask.astype(Dfac.dtype)
    Dfac = Dfac * mask
    return -weight * Dfac[None, None, :, :, None] * G


def end_damping_contribution(
    H: jnp.ndarray,
    *,
    ky: jnp.ndarray,
    damp_profile: jnp.ndarray,
    linked_damp_profile: jnp.ndarray | None,
    damp_amp: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    real_dtype = jnp.real(H).dtype
    if _is_static_zero(weight, real_dtype) or _is_static_zero(damp_amp, real_dtype):
        return _zeros_like_result(H, weight, damp_amp)

    if linked_damp_profile is not None and getattr(linked_damp_profile, "size", 0) != 0:
        damp = weight * damp_amp * linked_damp_profile[None, None, None, ...]
        return -(damp * H)
    damp = weight * damp_amp * damp_profile[None, None, None, None, None, :]
    ky_mask = (ky > 0.0).astype(damp.dtype)[None, None, None, :, None, None]
    return -(ky_mask * damp * H)


__all__ = [
    "collisions_contribution",
    "end_damping_contribution",
    "hypercollisions_contribution",
    "hyperdiffusion_contribution",
]
