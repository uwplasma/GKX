"""Linear term implementations for gyrokinetic RHS assembly."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.terms.operators import shift_axis, streaming_term


def streaming_contribution(
    H: jnp.ndarray,
    *,
    kz: jnp.ndarray,
    vth: jnp.ndarray,
    sqrt_p: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    vth_s = vth if vth.ndim == 0 else vth[:, None, None, None, None, None]
    return -weight * kpar_scale * streaming_term(H, kz, vth_s, sqrt_p, sqrt_m)


def mirror_contribution(
    H: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    bgrad: jnp.ndarray,
    omega_d_scale: jnp.ndarray,
    l: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    sqrt_m_p1: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    axis_l = -5
    axis_m = -4
    l_p1 = l + 1.0
    H_m_p1 = shift_axis(H, 1, axis=axis_m)
    H_m_m1 = shift_axis(H, -1, axis=axis_m)
    mirror_term = (
        -sqrt_m_p1 * l_p1 * H_m_p1
        - sqrt_m_p1 * l * shift_axis(H_m_p1, -1, axis=axis_l)
        + sqrt_m * l * H_m_m1
        + sqrt_m * l_p1 * shift_axis(H_m_m1, 1, axis=axis_l)
    )
    bgrad_s = omega_d_scale * bgrad[None, None, None, None, None, :]
    vth_s = vth[:, None, None, None, None, None]
    return -weight * vth_s * bgrad_s * mirror_term


def curvature_gradb_contribution(
    H: jnp.ndarray,
    *,
    tz: jnp.ndarray,
    omega_d_scale: jnp.ndarray,
    cv_d: jnp.ndarray,
    gb_d: jnp.ndarray,
    l: jnp.ndarray,
    m: jnp.ndarray,
    imag: jnp.ndarray,
    weight_curv: jnp.ndarray,
    weight_gradb: jnp.ndarray,
) -> jnp.ndarray:
    axis_m = -4
    H_m_p2 = shift_axis(H, 2, axis=axis_m)
    H_m_m2 = shift_axis(H, -2, axis=axis_m)
    curv_term = (
        jnp.sqrt((m + 1.0) * (m + 2.0)) * H_m_p2
        + (2.0 * m + 1.0) * H
        + jnp.sqrt(m * (m - 1.0)) * H_m_m2
    )
    axis_l = -5
    gradb_term = (
        (l + 1.0) * shift_axis(H, 1, axis=axis_l)
        + (2.0 * l + 1.0) * H
        + l * shift_axis(H, -1, axis=axis_l)
    )
    icv = imag * tz[:, None, None, None, None, None] * omega_d_scale * cv_d[None, None, None, ...]
    igb = imag * tz[:, None, None, None, None, None] * omega_d_scale * gb_d[None, None, None, ...]
    return -weight_curv * icv * curv_term - weight_gradb * igb * gradb_term


def diamagnetic_contribution(
    dG: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    l4: jnp.ndarray,
    tprim: jnp.ndarray,
    fprim: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    omega_star_scale: jnp.ndarray,
    ky: jnp.ndarray,
    imag: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    Nm = dG.shape[2]
    Jl_m1 = shift_axis(Jl, -1, axis=1)
    Jl_p1 = shift_axis(Jl, 1, axis=1)
    JlB_m1 = shift_axis(JlB, -1, axis=1)
    JlB_p1 = shift_axis(JlB, 1, axis=1)
    omega_star = imag * omega_star_scale * ky
    tprim_s = tprim[:, None, None, None, None]
    fprim_s = fprim[:, None, None, None, None]
    tz_s = tz[:, None, None, None, None]
    omega_star_s = omega_star[None, None, :, None, None]
    omega_star_bpar = omega_star_s / tz_s
    drive_m0 = omega_star_s * phi * (
        Jl_m1 * (l4 * tprim_s)
        + Jl * (fprim_s + 2.0 * l4 * tprim_s)
        + Jl_p1 * ((l4 + 1.0) * tprim_s)
    )
    drive_m0 = drive_m0 + omega_star_bpar * bpar * (
        JlB_m1 * (l4 * tprim_s)
        + JlB * (fprim_s + 2.0 * l4 * tprim_s)
        + JlB_p1 * ((l4 + 1.0) * tprim_s)
    )
    dG = dG.at[:, :, 0, ...].add(weight * drive_m0)
    if Nm > 2:
        drive_m2 = omega_star_s * phi * Jl * (tprim_s / jnp.sqrt(2.0))
        drive_m2 = drive_m2 + omega_star_bpar * bpar * JlB * (tprim_s / jnp.sqrt(2.0))
        dG = dG.at[:, :, 2, ...].add(weight * drive_m2)
    if Nm > 1:
        vth_s = vth[:, None, None, None, None]
        apar_drive = -vth_s * omega_star_s * apar * (
            Jl_m1 * (l4 * tprim_s)
            + Jl * (fprim_s + (2.0 * l4 + 1.0) * tprim_s)
            + Jl_p1 * ((l4 + 1.0) * tprim_s)
        )
        dG = dG.at[:, :, 1, ...].add(weight * apar_drive)
    if Nm > 3:
        vth_s = vth[:, None, None, None, None]
        drive_m3 = -vth_s * omega_star_s * apar * Jl * (tprim_s * jnp.sqrt(3.0 / 2.0))
        dG = dG.at[:, :, 3, ...].add(weight * drive_m3)
    return dG


def collisions_contribution(
    H: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    lb_lam: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    return -weight * nu[:, None, None, None, None, None] * lb_lam * H


def hypercollisions_contribution(
    G: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    l: jnp.ndarray,
    m: jnp.ndarray,
    nu_hyper: jnp.ndarray,
    nu_hyper_l: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    nu_hyper_lm: jnp.ndarray,
    p_hyper_l: jnp.ndarray,
    p_hyper_m: jnp.ndarray,
    p_hyper_lm: jnp.ndarray,
    hyper_ratio: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    Nl = G.shape[1]
    Nm = G.shape[2]
    l_norm = jnp.asarray(max(Nl, 1), dtype=l.dtype)
    m_norm = jnp.asarray(max(Nm, 1), dtype=m.dtype)
    p_hyper_m_eff = jnp.minimum(p_hyper_m.astype(m.dtype), 0.5 * m_norm)
    ratio_l = (l / l_norm) ** p_hyper_l
    ratio_m = (m / m_norm) ** p_hyper_m_eff
    ratio_lm = ((2.0 * l + m) / (2.0 * l_norm + m_norm)) ** p_hyper_lm
    scaled_nu_l = l_norm * nu_hyper_l
    scaled_nu_m = m_norm * nu_hyper_m
    vth_s = vth[:, None, None, None, None, None]
    hyper_term = -vth_s * (scaled_nu_l * ratio_l + scaled_nu_m * ratio_m) - nu_hyper_lm * ratio_lm
    mask = (m > 2.0) | (l > 1.0)
    dG = weight * jnp.where(mask, hyper_term, 0.0) * G
    dG = dG - weight * nu_hyper * hyper_ratio * G
    return dG


def end_damping_contribution(
    H: jnp.ndarray,
    *,
    ky: jnp.ndarray,
    damp_profile: jnp.ndarray,
    damp_amp: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    damp = weight * damp_amp * damp_profile[None, None, None, None, None, :]
    ky_mask = (ky > 0.0).astype(damp.dtype)[None, None, None, :, None, None]
    return -(ky_mask * damp * H)
