"""Linear term implementations for gyrokinetic RHS assembly."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.terms.operators import grad_z_linked_fft, grad_z_periodic, shift_axis, streaming_term


def streaming_contribution(
    H: jnp.ndarray,
    *,
    kz: jnp.ndarray,
    dz: jnp.ndarray,
    vth: jnp.ndarray,
    sqrt_p: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    weight: jnp.ndarray,
    kx_link_plus: jnp.ndarray | None = None,
    kx_link_minus: jnp.ndarray | None = None,
    kx_mask_plus: jnp.ndarray | None = None,
    kx_mask_minus: jnp.ndarray | None = None,
    linked_indices: tuple[jnp.ndarray, ...] | None = None,
    linked_kz: tuple[jnp.ndarray, ...] | None = None,
    use_twist_shift: bool = False,
) -> jnp.ndarray:
    vth_s = vth if vth.ndim == 0 else vth[:, None, None, None, None, None]
    return -weight * kpar_scale * streaming_term(
        H,
        kz,
        vth_s,
        sqrt_p,
        sqrt_m,
        dz=dz,
        kx_link_plus=kx_link_plus,
        kx_link_minus=kx_link_minus,
        kx_mask_plus=kx_mask_plus,
        kx_mask_minus=kx_mask_minus,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        use_twist_shift=use_twist_shift,
    )


def streaming_contribution_gx(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
    sqrt_p: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    weight: jnp.ndarray,
    kz: jnp.ndarray,
    dz: jnp.ndarray,
    use_twist_shift: bool = False,
    linked_indices: tuple[jnp.ndarray, ...] | None = None,
    linked_kz: tuple[jnp.ndarray, ...] | None = None,
) -> jnp.ndarray:
    """GX-style streaming: ladder on g, add field terms, then apply parallel derivative."""

    axis_m = -4
    G_p1 = shift_axis(G, 1, axis=axis_m)
    G_m1 = shift_axis(G, -1, axis=axis_m)
    vth_s = vth[:, None, None, None, None, None]
    rhs = -vth_s * (sqrt_p * G_p1 + sqrt_m * G_m1)

    tz_arr = tz[:, None, None, None, None, None]
    zt = jnp.where(tz_arr == 0.0, 0.0, 1.0 / tz_arr)
    zt5 = zt[:, 0, 0, 0, 0, 0][:, None, None, None, None]
    vth5 = vth[:, None, None, None, None]
    phi_s = phi[None, None, ...]
    apar_s = apar[None, None, ...]
    bpar_s = bpar[None, None, ...]

    # field terms (pre-derivative); use contiguous m-axis masks instead of scatter updates.
    Nm = rhs.shape[2]
    m_idx = jnp.arange(Nm, dtype=jnp.int32)[None, None, :, None, None, None]
    field_rhs = jnp.zeros_like(rhs)
    drive_m0 = zt5 * (vth5 * vth5) * Jl * apar_s
    field_rhs = field_rhs + (m_idx == 0).astype(field_rhs.dtype) * drive_m0[:, :, None, ...]
    if Nm > 1:
        drive_m1 = -zt5 * vth5 * Jl * phi_s - vth5 * JlB * bpar_s
        field_rhs = field_rhs + (m_idx == 1).astype(field_rhs.dtype) * drive_m1[:, :, None, ...]
    if Nm > 2:
        drive_m2 = jnp.sqrt(2.0) * zt5 * (vth5 * vth5) * Jl * apar_s
        field_rhs = field_rhs + (m_idx == 2).astype(field_rhs.dtype) * drive_m2[:, :, None, ...]
    rhs = rhs + field_rhs

    rhs = kpar_scale * rhs

    if use_twist_shift:
        if linked_indices is None or linked_kz is None:
            raise ValueError("linked_indices and linked_kz must be provided for linked streaming")
        dG = grad_z_linked_fft(rhs, dz=dz, linked_indices=linked_indices, linked_kz=linked_kz)
    else:
        dG = grad_z_periodic(rhs, kz=kz)

    return weight * dG


def mirror_contribution(
    H: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    bgrad: jnp.ndarray,
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
    bgrad_s = bgrad[None, None, None, None, None, :]
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
    tz_s = tz[:, None, None, None, None, None]
    icv = imag * tz_s * omega_d_scale * cv_d[None, None, None, ...]
    igb = imag * tz_s * omega_d_scale * gb_d[None, None, None, ...]
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
    m_idx = jnp.arange(Nm, dtype=jnp.int32)[None, None, :, None, None, None]
    Jl_m1 = shift_axis(Jl, -1, axis=1)
    Jl_p1 = shift_axis(Jl, 1, axis=1)
    JlB_m1 = shift_axis(JlB, -1, axis=1)
    JlB_p1 = shift_axis(JlB, 1, axis=1)
    omega_star = imag * omega_star_scale * ky
    tprim_s = tprim[:, None, None, None, None]
    fprim_s = fprim[:, None, None, None, None]
    tz_s = tz[:, None, None, None, None]
    omega_star_s = omega_star[None, None, :, None, None]
    omega_star_bpar = omega_star_s * tz_s
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
    drive = (m_idx == 0).astype(dG.dtype) * drive_m0[:, :, None, ...]
    if Nm > 2:
        drive_m2 = omega_star_s * phi * Jl * (tprim_s / jnp.sqrt(2.0))
        drive_m2 = drive_m2 + omega_star_bpar * bpar * JlB * (tprim_s / jnp.sqrt(2.0))
        drive = drive + (m_idx == 2).astype(dG.dtype) * drive_m2[:, :, None, ...]
    if Nm > 1:
        vth_s = vth[:, None, None, None, None]
        apar_drive = -vth_s * omega_star_s * apar * (
            Jl_m1 * (l4 * tprim_s)
            + Jl * (fprim_s + (2.0 * l4 + 1.0) * tprim_s)
            + Jl_p1 * ((l4 + 1.0) * tprim_s)
        )
        drive = drive + (m_idx == 1).astype(dG.dtype) * apar_drive[:, :, None, ...]
    if Nm > 3:
        vth_s = vth[:, None, None, None, None]
        drive_m3 = -vth_s * omega_star_s * apar * Jl * (tprim_s * jnp.sqrt(3.0 / 2.0))
        drive = drive + (m_idx == 3).astype(dG.dtype) * drive_m3[:, :, None, ...]
    return dG + weight * drive


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
    dG = dG - weight * nu_hyper * hyper_ratio * G

    abs_kz = jnp.abs(kz)[None, None, None, None, None, :]
    nu_hyp_m = (
        nu_hyper_m
        * m_norm_kz_factor
        * 2.3
        * vth_s
        * jnp.abs(kpar_scale)
    )
    kz_term = -nu_hyp_m * m_pow * abs_kz
    dG = dG + weight * hypercollisions_kz * jnp.where(mask_kz, kz_term, 0.0) * G
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
