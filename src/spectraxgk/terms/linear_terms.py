"""Linear term implementations for gyrokinetic RHS assembly."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.terms.linear_dissipation import (
    _hermite_mode_drive,
    _hypercollision_kz_source as _hypercollision_kz_source,
    _is_static_zero,
    _zeros_like_result,
    collision_invariant_rates as collision_invariant_rates,
    collision_quadratic_rate as collision_quadratic_rate,
    collisions_contribution as collisions_contribution,
    end_damping_contribution as end_damping_contribution,
    hypercollisions_contribution as hypercollisions_contribution,
    hyperdiffusion_contribution as hyperdiffusion_contribution,
    multispecies_collision_invariant_rates as multispecies_collision_invariant_rates,
)
from spectraxgk.terms.operators import (
    grad_z_linked_fft,
    grad_z_periodic,
    shift_axis,
    streaming_term,
)


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
    linked_inverse_permutation: jnp.ndarray | None = None,
    linked_full_cover: bool = False,
    linked_gather_map: jnp.ndarray | None = None,
    linked_gather_mask: jnp.ndarray | None = None,
    linked_use_gather: bool = False,
    use_twist_shift: bool = False,
) -> jnp.ndarray:
    if _is_static_zero(weight, jnp.real(H).dtype):
        return _zeros_like_result(H, weight)
    vth_s = vth if vth.ndim == 0 else vth[:, None, None, None, None, None]
    return (
        -weight
        * kpar_scale
        * streaming_term(
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
            linked_inverse_permutation=linked_inverse_permutation,
            linked_full_cover=linked_full_cover,
            linked_gather_map=linked_gather_map,
            linked_gather_mask=linked_gather_mask,
            linked_use_gather=linked_use_gather,
            use_twist_shift=use_twist_shift,
        )
    )


def _streaming_ladder_rhs(
    G: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    sqrt_p: jnp.ndarray,
    sqrt_m: jnp.ndarray,
) -> jnp.ndarray:
    axis_m = -4
    G_p1 = shift_axis(G, 1, axis=axis_m)
    G_m1 = shift_axis(G, -1, axis=axis_m)
    vth_s = vth[:, None, None, None, None, None]
    return -vth_s * (sqrt_p * G_p1 + sqrt_m * G_m1)


def _field_inverse_temperature(
    tz: jnp.ndarray,
) -> jnp.ndarray:
    tz_arr = tz[:, None, None, None, None, None]
    zt = jnp.where(tz_arr == 0.0, 0.0, 1.0 / tz_arr)
    return zt[:, 0, 0, 0, 0, 0][:, None, None, None, None]


def _streaming_field_drive(
    template: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    tz: jnp.ndarray,
    vth: jnp.ndarray,
) -> jnp.ndarray:
    zt5 = _field_inverse_temperature(tz)
    vth5 = vth[:, None, None, None, None]
    phi_s = phi[None, None, ...]
    Nm = template.shape[2]
    field_rhs = jnp.zeros_like(template)
    if apar is not None:
        apar_s = apar[None, None, ...]
        drive_m0 = zt5 * (vth5 * vth5) * Jl * apar_s
        field_rhs = field_rhs + _hermite_mode_drive(field_rhs, 0, drive_m0)
    if Nm > 1:
        drive_m1 = -zt5 * vth5 * Jl * phi_s
        if bpar is not None:
            drive_m1 = drive_m1 - vth5 * JlB * bpar[None, None, ...]
        field_rhs = field_rhs + _hermite_mode_drive(field_rhs, 1, drive_m1)
    if Nm > 2 and apar is not None:
        drive_m2 = jnp.sqrt(2.0) * zt5 * (vth5 * vth5) * Jl * apar_s
        field_rhs = field_rhs + _hermite_mode_drive(field_rhs, 2, drive_m2)
    return field_rhs


def _streaming_parallel_derivative(
    rhs: jnp.ndarray,
    *,
    kz: jnp.ndarray,
    dz: jnp.ndarray,
    use_twist_shift: bool,
    linked_indices: tuple[jnp.ndarray, ...] | None,
    linked_kz: tuple[jnp.ndarray, ...] | None,
    linked_inverse_permutation: jnp.ndarray | None,
    linked_full_cover: bool,
    linked_gather_map: jnp.ndarray | None,
    linked_gather_mask: jnp.ndarray | None,
    linked_use_gather: bool,
) -> jnp.ndarray:
    if not use_twist_shift:
        return grad_z_periodic(rhs, kz=kz)
    if linked_indices is None or linked_kz is None:
        raise ValueError("linked_indices and linked_kz must be provided for linked streaming")
    return grad_z_linked_fft(
        rhs,
        dz=dz,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_inverse_permutation=linked_inverse_permutation,
        linked_full_cover=linked_full_cover,
        linked_gather_map=linked_gather_map,
        linked_gather_mask=linked_gather_mask,
        linked_use_gather=linked_use_gather,
    )


def linked_streaming_contribution(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
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
    linked_inverse_permutation: jnp.ndarray | None = None,
    linked_full_cover: bool = False,
    linked_gather_map: jnp.ndarray | None = None,
    linked_gather_mask: jnp.ndarray | None = None,
    linked_use_gather: bool = False,
) -> jnp.ndarray:
    """Hermite-Laguerre streaming: ladder on g, add field terms, then apply parallel derivative."""

    if _is_static_zero(weight, jnp.real(G).dtype):
        return _zeros_like_result(G, weight)

    ladder_rhs = _streaming_ladder_rhs(G, vth=vth, sqrt_p=sqrt_p, sqrt_m=sqrt_m)
    field_rhs = _streaming_field_drive(
        ladder_rhs,
        phi=phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        tz=tz,
        vth=vth,
    )
    rhs = kpar_scale * (ladder_rhs + field_rhs)
    return weight * _streaming_parallel_derivative(
        rhs,
        kz=kz,
        dz=dz,
        use_twist_shift=use_twist_shift,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_inverse_permutation=linked_inverse_permutation,
        linked_full_cover=linked_full_cover,
        linked_gather_map=linked_gather_map,
        linked_gather_mask=linked_gather_mask,
        linked_use_gather=linked_use_gather,
    )


def mirror_contribution(
    H: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    bgrad: jnp.ndarray,
    ell: jnp.ndarray,
    sqrt_m: jnp.ndarray,
    sqrt_m_p1: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    axis_l = -5
    axis_m = -4
    ell_p1 = ell + 1.0
    H_m_p1 = shift_axis(H, 1, axis=axis_m)
    H_m_m1 = shift_axis(H, -1, axis=axis_m)
    mirror_term = (
        -sqrt_m_p1 * ell_p1 * H_m_p1
        - sqrt_m_p1 * ell * shift_axis(H_m_p1, -1, axis=axis_l)
        + sqrt_m * ell * H_m_m1
        + sqrt_m * ell_p1 * shift_axis(H_m_m1, 1, axis=axis_l)
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
    ell: jnp.ndarray,
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
        (ell + 1.0) * shift_axis(H, 1, axis=axis_l)
        + (2.0 * ell + 1.0) * H
        + ell * shift_axis(H, -1, axis=axis_l)
    )
    tz_s = tz[:, None, None, None, None, None]
    icv = imag * tz_s * omega_d_scale * cv_d[None, None, None, ...]
    igb = imag * tz_s * omega_d_scale * gb_d[None, None, None, ...]
    return -weight_curv * icv * curv_term - weight_gradb * igb * gradb_term


def _laguerre_gradient_profile(
    J: jnp.ndarray,
    *,
    l4: jnp.ndarray,
    tprim_s: jnp.ndarray,
    fprim_s: jnp.ndarray,
    thermal_shift: float,
) -> jnp.ndarray:
    J_m1 = shift_axis(J, -1, axis=1)
    J_p1 = shift_axis(J, 1, axis=1)
    return (
        J_m1 * (l4 * tprim_s)
        + J * (fprim_s + (2.0 * l4 + thermal_shift) * tprim_s)
        + J_p1 * ((l4 + 1.0) * tprim_s)
    )


def _diamagnetic_scalar_factors(
    *,
    tprim: jnp.ndarray,
    fprim: jnp.ndarray,
    tz: jnp.ndarray,
    omega_star_scale: jnp.ndarray,
    ky: jnp.ndarray,
    imag: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    omega_star = imag * omega_star_scale * ky
    tprim_s = tprim[:, None, None, None, None]
    fprim_s = fprim[:, None, None, None, None]
    tz_s = tz[:, None, None, None, None]
    omega_star_s = omega_star[None, None, :, None, None]
    return tprim_s, fprim_s, omega_star_s, omega_star_s * tz_s


def _diamagnetic_m0_drive(
    *,
    phi: jnp.ndarray,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    l4: jnp.ndarray,
    tprim_s: jnp.ndarray,
    fprim_s: jnp.ndarray,
    omega_star_s: jnp.ndarray,
    omega_star_bpar: jnp.ndarray,
) -> jnp.ndarray:
    drive_m0 = (
        omega_star_s
        * phi
        * _laguerre_gradient_profile(
            Jl, l4=l4, tprim_s=tprim_s, fprim_s=fprim_s, thermal_shift=0.0
        )
    )
    if bpar is None:
        return drive_m0
    return drive_m0 + omega_star_bpar * bpar * _laguerre_gradient_profile(
        JlB, l4=l4, tprim_s=tprim_s, fprim_s=fprim_s, thermal_shift=0.0
    )


def _diamagnetic_m2_drive(
    *,
    phi: jnp.ndarray,
    bpar: jnp.ndarray | None,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    tprim_s: jnp.ndarray,
    omega_star_s: jnp.ndarray,
    omega_star_bpar: jnp.ndarray,
) -> jnp.ndarray:
    thermal_factor = tprim_s / jnp.sqrt(2.0)
    drive_m2 = omega_star_s * phi * Jl * thermal_factor
    if bpar is None:
        return drive_m2
    return drive_m2 + omega_star_bpar * bpar * JlB * thermal_factor


def _diamagnetic_apar_profile_drive(
    *,
    apar: jnp.ndarray,
    Jl: jnp.ndarray,
    l4: jnp.ndarray,
    tprim_s: jnp.ndarray,
    fprim_s: jnp.ndarray,
    vth: jnp.ndarray,
    omega_star_s: jnp.ndarray,
) -> jnp.ndarray:
    vth_s = vth[:, None, None, None, None]
    return (
        -vth_s
        * omega_star_s
        * apar
        * _laguerre_gradient_profile(
            Jl, l4=l4, tprim_s=tprim_s, fprim_s=fprim_s, thermal_shift=1.0
        )
    )


def _diamagnetic_apar_temperature_drive(
    *,
    apar: jnp.ndarray,
    Jl: jnp.ndarray,
    tprim_s: jnp.ndarray,
    vth: jnp.ndarray,
    omega_star_s: jnp.ndarray,
) -> jnp.ndarray:
    vth_s = vth[:, None, None, None, None]
    thermal_factor = tprim_s * jnp.sqrt(3.0 / 2.0)
    return -vth_s * omega_star_s * apar * Jl * thermal_factor


def diamagnetic_contribution(
    dG: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    apar: jnp.ndarray | None,
    bpar: jnp.ndarray | None,
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
    tprim_s, fprim_s, omega_star_s, omega_star_bpar = _diamagnetic_scalar_factors(
        tprim=tprim,
        fprim=fprim,
        tz=tz,
        omega_star_scale=omega_star_scale,
        ky=ky,
        imag=imag,
    )
    drive_m0 = _diamagnetic_m0_drive(
        phi=phi,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        l4=l4,
        tprim_s=tprim_s,
        fprim_s=fprim_s,
        omega_star_s=omega_star_s,
        omega_star_bpar=omega_star_bpar,
    )
    drive = _hermite_mode_drive(dG, 0, drive_m0)
    if Nm > 2:
        drive_m2 = _diamagnetic_m2_drive(
            phi=phi,
            bpar=bpar,
            Jl=Jl,
            JlB=JlB,
            tprim_s=tprim_s,
            omega_star_s=omega_star_s,
            omega_star_bpar=omega_star_bpar,
        )
        drive = drive + _hermite_mode_drive(dG, 2, drive_m2)
    if Nm > 1 and apar is not None:
        apar_drive = _diamagnetic_apar_profile_drive(
            apar=apar,
            Jl=Jl,
            l4=l4,
            tprim_s=tprim_s,
            fprim_s=fprim_s,
            vth=vth,
            omega_star_s=omega_star_s,
        )
        drive = drive + _hermite_mode_drive(dG, 1, apar_drive)
    if Nm > 3 and apar is not None:
        drive_m3 = _diamagnetic_apar_temperature_drive(
            apar=apar,
            Jl=Jl,
            tprim_s=tprim_s,
            vth=vth,
            omega_star_s=omega_star_s,
        )
        drive = drive + _hermite_mode_drive(dG, 3, drive_m3)
    return dG + weight * drive
