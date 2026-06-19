"""Geometry-dependent construction of :class:`LinearCache`."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from spectraxgk.core.velocity import bessel_j0, bessel_j1, laguerre_transform
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.linear.cache_arrays import (
    _build_end_damping_profile_array,
    _build_gyroaverage_cache_arrays,
    _build_low_rank_moment_cache_arrays,
)
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.linked import (
    _build_linked_end_damping_profile,
    _build_linked_fft_maps,
)
from spectraxgk.operators.linear.params import LinearParams, _is_tracer, _x64_enabled


def _resolve_twist_shift_policy(
    grid: SpectralGrid,
    geom_data: Any,
    *,
    gds21: jnp.ndarray,
    gds22: jnp.ndarray,
    kx_eff: jnp.ndarray,
    kx_grid: jnp.ndarray,
) -> tuple[
    str,
    bool,
    bool,
    float,
    jnp.ndarray,
    float,
    int,
    float,
    jnp.ndarray,
    jnp.ndarray,
]:
    boundary = str(getattr(grid, "boundary", "periodic")).lower()
    use_twist_shift = boundary in {"linked", "fix aspect", "continuous drifts"}
    use_ntft = bool(getattr(grid, "non_twist", False))
    y0 = getattr(grid, "y0", None)
    if y0 is None:
        if grid.ky.size > 1:
            y0 = float(1.0 / float(grid.ky[1] - grid.ky[0]))
        else:
            y0 = 1.0
    shat_arr = jnp.asarray(geom_data.s_hat, dtype=kx_eff.dtype)
    shat_host = None if _is_tracer(shat_arr) else float(np.asarray(shat_arr))
    x0_eff = float(getattr(grid, "x0", 1.0))
    jtwist = 0
    x0_target = x0_eff
    if use_twist_shift:
        if shat_host is None:
            raise ValueError(
                "traced magnetic shear is not supported with twist-shift boundaries"
            )
        shat = shat_host
        gds21_min = float(gds21[0]) if gds21.ndim else float(gds21)
        gds22_min = float(gds22[0]) if gds22.ndim else float(gds22)
        twist_shift_geo_fac = 0.0
        if gds22_min != 0.0:
            twist_shift_geo_fac = float(2.0 * shat * gds21_min / gds22_min)
        if twist_shift_geo_fac != 0.0:
            jtwist_val = getattr(grid, "jtwist", None)
            if jtwist_val is not None:
                jtwist = int(jtwist_val)
            else:
                jtwist = int(np.round(twist_shift_geo_fac))
            if jtwist == 0:
                jtwist = 1
            x0_target = float(y0) * abs(jtwist) / abs(twist_shift_geo_fac)
            if use_ntft:
                x0_eff = x0_target
        else:
            jtwist_val = getattr(grid, "jtwist", None)
            if jtwist_val is not None:
                jtwist = int(jtwist_val)
            else:
                jtwist = 1
        if use_ntft and float(getattr(grid, "x0", x0_eff)) != 0.0:
            kx_eff = kx_eff * (float(getattr(grid, "x0", x0_eff)) / float(x0_eff))
        if not use_ntft and x0_target != 0.0 and x0_target != x0_eff:
            scale = float(x0_eff) / float(x0_target)
            kx_eff = kx_eff * scale
            kx_grid = kx_grid * scale
            x0_eff = x0_target
    kxfac_val = float(getattr(grid, "kxfac", 1.0))
    return (
        boundary,
        use_twist_shift,
        use_ntft,
        float(y0),
        shat_arr,
        x0_eff,
        jtwist,
        kxfac_val,
        kx_eff,
        kx_grid,
    )


def _build_kperp_and_drift_arrays(
    grid: SpectralGrid,
    geom_data: Any,
    *,
    theta: jnp.ndarray,
    kx_eff: jnp.ndarray,
    ky_eff: jnp.ndarray,
    ky_raw: jnp.ndarray,
    rho_star: jnp.ndarray,
    gds2: jnp.ndarray,
    gds21: jnp.ndarray,
    gds22_arr: jnp.ndarray,
    bmag: jnp.ndarray,
    cv: jnp.ndarray,
    gb: jnp.ndarray,
    cv0: jnp.ndarray,
    gb0: jnp.ndarray,
    shat_arr: jnp.ndarray,
    x0_eff: float,
    kperp2_bmag: bool,
    use_ntft: bool,
    dealias_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if use_ntft:
        ftwist = (geom_data.s_hat * gds21 / gds22_arr).astype(kx_eff.dtype)
        delta = jnp.asarray(0.01313, dtype=kx_eff.dtype)
        ftwist_next = jnp.roll(ftwist, -1)
        mid_idx = int(grid.z.size // 2)
        mid_next = (mid_idx + 1) % grid.z.size
        ftwist_mid = ftwist[mid_idx]
        ftwist_mid_next = ftwist[mid_next]
        m0 = -jnp.rint(
            float(x0_eff)
            * ky_raw[:, None]
            * ((1.0 - delta) * ftwist[None, :] + delta * ftwist_next[None, :])
        ) + jnp.rint(
            float(x0_eff)
            * ky_raw[:, None]
            * ((1.0 - delta) * ftwist_mid + delta * ftwist_mid_next)
        )
        m0 = m0.astype(kx_eff.dtype)
        shat_inv = 1.0 / shat_arr
        delta_kx = ky_eff[:, None] * ftwist[None, :] + (
            rho_star * m0 / float(x0_eff)
        )
        term_ky = ky_eff[:, None, None] ** 2 * (
            gds2[None, None, :]
            - 2.0 * ftwist[None, None, :] * gds21[None, None, :] * shat_inv
            + (ftwist[None, None, :] ** 2)
            * gds22_arr[None, None, :]
            * shat_inv
            * shat_inv
        )
        term_kx = (
            (kx_eff[None, :, None] + delta_kx[:, None, :]) ** 2
            * gds22_arr[None, None, :]
            * shat_inv
            * shat_inv
        )
        bmag_inv = 1.0 / bmag
        kperp2 = term_ky + term_kx
        if kperp2_bmag:
            kperp2 = kperp2 * (bmag_inv[None, None, :] ** 2)
        kx_shift = kx_eff[None, :, None] + (rho_star * m0 / float(x0_eff))[:, None, :]
        cv_d = (
            ky_eff[:, None, None] * cv[None, None, :]
            + shat_inv * kx_shift * cv0[None, None, :]
        )
        gb_d = (
            ky_eff[:, None, None] * gb[None, None, :]
            + shat_inv * kx_shift * gb0[None, None, :]
        )
        omega_d = cv_d + gb_d
    else:
        kx0 = kx_eff[None, :, None]
        ky0 = ky_eff[:, None, None]
        theta_b = theta[None, None, :]
        kperp2 = geom_data.k_perp2(kx0, ky0, theta_b).astype(kx_eff.dtype)
        cv_d, gb_d = geom_data.drift_components(kx_eff, ky_eff, theta)
        cv_d = cv_d.astype(kx_eff.dtype)
        gb_d = gb_d.astype(kx_eff.dtype)
        omega_d = (cv_d + gb_d).astype(kx_eff.dtype)
    apply_dealias_mask = dealias_mask is not None and int(grid.ky.size) > 1
    if apply_dealias_mask:
        mask = dealias_mask[:, :, None]
        kperp2 = kperp2 * mask
        cv_d = cv_d * mask
        gb_d = gb_d * mask
        omega_d = omega_d * mask
    return kperp2, cv_d, gb_d, omega_d


def _build_laguerre_gyro_cache(
    params: LinearParams,
    *,
    geom_data: Any,
    kperp2: jnp.ndarray,
    bmag: jnp.ndarray,
    Nl: int,
    real_dtype: Any,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    rho = jnp.asarray(params.rho, dtype=real_dtype)
    if rho.ndim == 0:
        rho = rho[None]
    b = (rho[:, None, None, None] * rho[:, None, None, None]) * kperp2[None, ...]
    bessel_bmag_power = float(getattr(geom_data, "bessel_bmag_power", 0.0))
    if bessel_bmag_power != 0.0:
        bmag_factor = bmag[None, None, None, :] ** (-bessel_bmag_power)
        b = b * bmag_factor
    Jl, JlB = _build_gyroaverage_cache_arrays(b, Nl, real_dtype)
    lag_to_grid_np, lag_to_spec_np, lag_roots_np = laguerre_transform(Nl)
    laguerre_to_grid = jnp.asarray(lag_to_grid_np, dtype=real_dtype)
    laguerre_to_spectral = jnp.asarray(lag_to_spec_np, dtype=real_dtype)
    laguerre_roots = jnp.asarray(lag_roots_np, dtype=real_dtype)
    alpha = jnp.sqrt(
        jnp.maximum(
            0.0,
            2.0 * laguerre_roots[None, :, None, None, None] * b[:, None, ...],
        )
    )
    laguerre_j0 = bessel_j0(alpha).astype(real_dtype)
    laguerre_j1 = bessel_j1(alpha)
    laguerre_j1_over_alpha = jnp.where(alpha < 1.0e-8, 0.5, laguerre_j1 / alpha).astype(
        real_dtype
    )
    return (
        b,
        Jl,
        JlB,
        laguerre_to_grid,
        laguerre_to_spectral,
        laguerre_roots,
        laguerre_j0,
        laguerre_j1_over_alpha,
    )


def _build_linked_boundary_cache(
    grid: SpectralGrid,
    params: LinearParams,
    *,
    boundary: str,
    use_twist_shift: bool,
    y0: float,
    jtwist: int,
    dz: jnp.ndarray,
    real_dtype: Any,
) -> dict[str, Any]:
    damp_profile = _build_end_damping_profile_array(
        int(grid.z.size),
        float(params.damp_ends_widthfrac),
        boundary,
        real_dtype,
    )
    linked_damp_profile = jnp.asarray([], dtype=real_dtype)
    if use_twist_shift:
        iky = jnp.rint(grid.ky * float(y0)).astype(jnp.int32)
        shift = jnp.asarray(jtwist, dtype=jnp.int32) * iky
        kx_idx = jnp.arange(grid.kx.size, dtype=jnp.int32)[None, :]
        kx_link_plus = kx_idx + shift[:, None]
        kx_link_minus = kx_idx - shift[:, None]
        kx_link_mask_plus = (kx_link_plus >= 0) & (kx_link_plus < grid.kx.size)
        kx_link_mask_minus = (kx_link_minus >= 0) & (kx_link_minus < grid.kx.size)
        kx_link_plus = jnp.clip(kx_link_plus, 0, grid.kx.size - 1)
        kx_link_minus = jnp.clip(kx_link_minus, 0, grid.kx.size - 1)
    else:
        jtwist = 0
        kx_idx = jnp.arange(grid.kx.size, dtype=jnp.int32)[None, :]
        kx_link_plus = jnp.broadcast_to(kx_idx, (grid.ky.size, grid.kx.size))
        kx_link_minus = kx_link_plus
        kx_link_mask_plus = jnp.ones((grid.ky.size, grid.kx.size), dtype=bool)
        kx_link_mask_minus = kx_link_mask_plus

    linked_indices: tuple[jnp.ndarray, ...] = ()
    linked_kz: tuple[jnp.ndarray, ...] = ()
    linked_inverse_permutation = jnp.asarray([], dtype=jnp.int32)
    linked_full_cover = False
    linked_gather_map = jnp.asarray([], dtype=jnp.int32)
    linked_gather_mask = jnp.asarray([], dtype=bool)
    linked_use_gather = False
    if use_twist_shift:
        ky_mode = getattr(grid, "ky_mode", None)
        linked_indices, linked_kz = _build_linked_fft_maps(
            np.asarray(grid.kx),
            np.asarray(grid.ky),
            float(y0),
            int(jtwist),
            float(dz),
            int(grid.z.size),
            real_dtype,
            None if ky_mode is None else np.asarray(ky_mode),
        )
        if linked_indices:
            idx_flat = np.concatenate(
                [np.asarray(idx, dtype=np.int32).reshape(-1) for idx in linked_indices],
                axis=0,
            )
            n_modes = int(grid.ky.size * grid.kx.size)
            if idx_flat.size == n_modes:
                ref = np.arange(n_modes, dtype=np.int32)
                if np.array_equal(np.sort(idx_flat), ref):
                    linked_inverse_permutation = jnp.asarray(
                        np.argsort(idx_flat).astype(np.int32)
                    )
                    linked_full_cover = True
            if idx_flat.size > 0:
                gather_map = np.zeros(n_modes, dtype=np.int32)
                gather_mask = np.zeros(n_modes, dtype=bool)
                gather_map[idx_flat] = np.arange(idx_flat.size, dtype=np.int32)
                gather_mask[idx_flat] = True
                linked_gather_map = jnp.asarray(gather_map, dtype=jnp.int32)
                linked_gather_mask = jnp.asarray(gather_mask, dtype=bool)
                linked_use_gather = True
        if boundary != "periodic":
            linked_damp_profile = jnp.asarray(
                _build_linked_end_damping_profile(
                    linked_indices=linked_indices,
                    ny=int(grid.ky.size),
                    nx=int(grid.kx.size),
                    nz=int(grid.z.size),
                    widthfrac=float(params.damp_ends_widthfrac),
                    ky_mode=(
                        None
                        if getattr(grid, "ky_mode", None) is None
                        else np.asarray(grid.ky_mode, dtype=np.int32)
                    ),
                ),
                dtype=real_dtype,
            )

    return {
        "damp_profile": damp_profile,
        "linked_damp_profile": linked_damp_profile,
        "kx_link_plus": kx_link_plus,
        "kx_link_minus": kx_link_minus,
        "kx_link_mask_plus": kx_link_mask_plus,
        "kx_link_mask_minus": kx_link_mask_minus,
        "linked_full_cover": linked_full_cover,
        "linked_inverse_permutation": linked_inverse_permutation,
        "linked_gather_map": linked_gather_map,
        "linked_gather_mask": linked_gather_mask,
        "linked_use_gather": linked_use_gather,
        "linked_indices": linked_indices,
        "linked_kz": linked_kz,
        "jtwist": jtwist,
    }


def build_linear_cache(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    Nl: int,
    Nm: int,
) -> LinearCache:
    """Build reusable arrays for the linear RHS."""

    real_dtype = jnp.float64 if _x64_enabled() else jnp.float32
    dz = jnp.asarray(grid.z[1] - grid.z[0], dtype=real_dtype)
    kz = jnp.asarray(
        2.0 * jnp.pi * jnp.fft.fftfreq(grid.z.size, d=dz), dtype=real_dtype
    )
    rho_star = jnp.asarray(params.rho_star, dtype=real_dtype)
    kx_raw = jnp.asarray(grid.kx, dtype=real_dtype)
    ky_raw = jnp.asarray(grid.ky, dtype=real_dtype)
    kx_eff = rho_star * kx_raw
    ky_eff = rho_star * ky_raw
    kx_grid = jnp.asarray(grid.kx_grid, dtype=real_dtype) * rho_star
    ky_grid = jnp.asarray(grid.ky_grid, dtype=real_dtype) * rho_star
    dealias_mask = jnp.asarray(grid.dealias_mask, dtype=bool)
    theta = jnp.asarray(grid.z, dtype=real_dtype)
    geom_data = ensure_flux_tube_geometry_data(geom, theta)
    gds2, gds21, gds22 = geom_data.metric_coeffs(theta)
    gds22_arr = gds22 if gds22.ndim else jnp.full_like(theta, gds22)
    bmag = geom_data.bmag(theta).astype(real_dtype)
    bgrad = geom_data.bgrad(theta).astype(real_dtype)
    jacobian = geom_data.jacobian(theta).astype(real_dtype)
    cv, gb, cv0, gb0 = geom_data.drift_coeffs(theta)
    (
        boundary,
        use_twist_shift,
        use_ntft,
        y0,
        shat_arr,
        x0_eff,
        jtwist,
        kxfac_val,
        kx_eff,
        kx_grid,
    ) = _resolve_twist_shift_policy(
        grid,
        geom_data,
        gds21=gds21,
        gds22=gds22,
        kx_eff=kx_eff,
        kx_grid=kx_grid,
    )
    kperp2_bmag = bool(getattr(geom_data, "kperp2_bmag", True))
    kperp2, cv_d, gb_d, omega_d = _build_kperp_and_drift_arrays(
        grid,
        geom_data,
        theta=theta,
        kx_eff=kx_eff,
        ky_eff=ky_eff,
        ky_raw=ky_raw,
        rho_star=rho_star,
        gds2=gds2,
        gds21=gds21,
        gds22_arr=gds22_arr,
        bmag=bmag,
        cv=cv,
        gb=gb,
        cv0=cv0,
        gb0=gb0,
        shat_arr=shat_arr,
        x0_eff=x0_eff,
        kperp2_bmag=kperp2_bmag,
        use_ntft=use_ntft,
        dealias_mask=dealias_mask,
    )
    (
        b,
        Jl,
        JlB,
        laguerre_to_grid,
        laguerre_to_spectral,
        laguerre_roots,
        laguerre_j0,
        laguerre_j1_over_alpha,
    ) = _build_laguerre_gyro_cache(
        params,
        geom_data=geom_data,
        kperp2=kperp2,
        bmag=bmag,
        Nl=Nl,
        real_dtype=real_dtype,
    )
    mask0 = (grid.ky == 0.0)[:, None, None] & (grid.kx == 0.0)[None, :, None]
    moment_cache = _build_low_rank_moment_cache_arrays(Nl, Nm, params, real_dtype)
    lb_lam = moment_cache["lb_lam"]
    ell_cache = moment_cache["l"]
    m = moment_cache["m"]
    l4 = moment_cache["l4"]
    sqrt_m = moment_cache["sqrt_m"]
    sqrt_m_p1 = moment_cache["sqrt_m_p1"]
    sqrt_p = moment_cache["sqrt_p"]
    sqrt_m_ladder = moment_cache["sqrt_m_ladder"]
    hyper_ratio = moment_cache["hyper_ratio"]
    ratio_l = moment_cache["ratio_l"]
    ratio_m = moment_cache["ratio_m"]
    ratio_lm = moment_cache["ratio_lm"]
    mask_const = moment_cache["mask_const"]
    mask_kz = moment_cache["mask_kz"]
    m_pow = moment_cache["m_pow"]
    m_norm_kz_factor = moment_cache["m_norm_kz_factor"]
    linked_cache = _build_linked_boundary_cache(
        grid,
        params,
        boundary=boundary,
        use_twist_shift=use_twist_shift,
        y0=y0,
        jtwist=jtwist,
        dz=dz,
        real_dtype=real_dtype,
    )
    return LinearCache(
        Jl=Jl,
        b=b.astype(real_dtype),
        kperp2=kperp2,
        kperp2_bmag=kperp2_bmag,
        bmag=bmag,
        omega_d=omega_d,
        cv_d=cv_d,
        gb_d=gb_d,
        bgrad=bgrad,
        jacobian=jacobian,
        mask0=mask0,
        dz=dz,
        kz=kz,
        ky=ky_eff.astype(real_dtype),
        kx=kx_eff.astype(real_dtype),
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=jnp.asarray(kxfac_val, dtype=real_dtype),
        lb_lam=lb_lam,
        collision_lam=jnp.asarray([], dtype=real_dtype),
        hyper_ratio=hyper_ratio.astype(real_dtype),
        ratio_l=ratio_l.astype(real_dtype),
        ratio_m=ratio_m.astype(real_dtype),
        ratio_lm=ratio_lm.astype(real_dtype),
        mask_const=mask_const,
        mask_kz=mask_kz,
        m_pow=m_pow.astype(real_dtype),
        m_norm_kz_factor=m_norm_kz_factor.astype(real_dtype),
        damp_profile=linked_cache["damp_profile"],
        linked_damp_profile=linked_cache["linked_damp_profile"],
        l=ell_cache,
        m=m,
        l4=l4,
        sqrt_m=sqrt_m.astype(real_dtype),
        sqrt_m_p1=sqrt_m_p1.astype(real_dtype),
        sqrt_p=sqrt_p,
        sqrt_m_ladder=sqrt_m_ladder,
        JlB=JlB.astype(real_dtype),
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        laguerre_j0=laguerre_j0,
        laguerre_j1_over_alpha=laguerre_j1_over_alpha,
        kx_link_plus=linked_cache["kx_link_plus"],
        kx_link_minus=linked_cache["kx_link_minus"],
        kx_link_mask_plus=linked_cache["kx_link_mask_plus"],
        kx_link_mask_minus=linked_cache["kx_link_mask_minus"],
        linked_full_cover=linked_cache["linked_full_cover"],
        linked_inverse_permutation=linked_cache["linked_inverse_permutation"],
        linked_gather_map=linked_cache["linked_gather_map"],
        linked_gather_mask=linked_cache["linked_gather_mask"],
        linked_use_gather=linked_cache["linked_use_gather"],
        linked_indices=linked_cache["linked_indices"],
        linked_kz=linked_cache["linked_kz"],
        use_twist_shift=use_twist_shift,
        jtwist=int(linked_cache["jtwist"]),
    )
