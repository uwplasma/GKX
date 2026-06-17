"""Linear operator cache construction and damping helpers."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from spectraxgk.core.velocity import J_l_all, bessel_j0, bessel_j1, laguerre_transform
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.linear.linked import (
    _build_linked_end_damping_profile,
    _build_linked_fft_maps,
)
from spectraxgk.operators.linear.params import (
    LinearParams,
    _as_species_array,
    _is_tracer,
    _x64_enabled,
)

__all__ = [
    "LinearCache",
    "_build_end_damping_profile_array",
    "_build_gyroaverage_cache_arrays",
    "_build_low_rank_moment_cache_arrays",
    "_numpy_dtype_for_jax",
    "build_linear_cache",
    "collision_damping",
    "hypercollision_damping",
]


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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearCache:
    """Precomputed arrays for the linear operator."""

    Jl: jnp.ndarray
    b: jnp.ndarray
    kperp2: jnp.ndarray
    kperp2_bmag: bool
    bmag: jnp.ndarray
    omega_d: jnp.ndarray
    cv_d: jnp.ndarray
    gb_d: jnp.ndarray
    bgrad: jnp.ndarray
    jacobian: jnp.ndarray
    mask0: jnp.ndarray
    dz: jnp.ndarray
    kz: jnp.ndarray
    ky: jnp.ndarray
    kx: jnp.ndarray
    kx_grid: jnp.ndarray
    ky_grid: jnp.ndarray
    dealias_mask: jnp.ndarray
    kxfac: jnp.ndarray
    lb_lam: jnp.ndarray
    collision_lam: jnp.ndarray
    hyper_ratio: jnp.ndarray
    ratio_l: jnp.ndarray
    ratio_m: jnp.ndarray
    ratio_lm: jnp.ndarray
    mask_const: jnp.ndarray
    mask_kz: jnp.ndarray
    m_pow: jnp.ndarray
    m_norm_kz_factor: jnp.ndarray
    damp_profile: jnp.ndarray
    linked_damp_profile: jnp.ndarray
    l: jnp.ndarray  # noqa: E741 - public cache field for the Laguerre index.
    m: jnp.ndarray
    l4: jnp.ndarray
    sqrt_m: jnp.ndarray
    sqrt_m_p1: jnp.ndarray
    sqrt_p: jnp.ndarray
    sqrt_m_ladder: jnp.ndarray
    JlB: jnp.ndarray
    laguerre_to_grid: jnp.ndarray
    laguerre_to_spectral: jnp.ndarray
    laguerre_roots: jnp.ndarray
    laguerre_j0: jnp.ndarray
    laguerre_j1_over_alpha: jnp.ndarray
    kx_link_plus: jnp.ndarray
    kx_link_minus: jnp.ndarray
    kx_link_mask_plus: jnp.ndarray
    kx_link_mask_minus: jnp.ndarray
    linked_inverse_permutation: jnp.ndarray = dataclass_field(
        default_factory=lambda: jnp.asarray([], dtype=jnp.int32)
    )
    linked_gather_map: jnp.ndarray = dataclass_field(
        default_factory=lambda: jnp.asarray([], dtype=jnp.int32)
    )
    linked_gather_mask: jnp.ndarray = dataclass_field(
        default_factory=lambda: jnp.asarray([], dtype=bool)
    )
    linked_full_cover: bool = False
    linked_use_gather: bool = False
    linked_indices: tuple[jnp.ndarray, ...] = ()
    linked_kz: tuple[jnp.ndarray, ...] = ()
    use_twist_shift: bool = False
    jtwist: int = 0

    def tree_flatten(self):
        children = (
            self.Jl,
            self.b,
            self.kperp2,
            self.kperp2_bmag,
            self.bmag,
            self.omega_d,
            self.cv_d,
            self.gb_d,
            self.bgrad,
            self.jacobian,
            self.mask0,
            self.dz,
            self.kz,
            self.ky,
            self.kx,
            self.kx_grid,
            self.ky_grid,
            self.dealias_mask,
            self.kxfac,
            self.lb_lam,
            self.collision_lam,
            self.hyper_ratio,
            self.ratio_l,
            self.ratio_m,
            self.ratio_lm,
            self.mask_const,
            self.mask_kz,
            self.m_pow,
            self.m_norm_kz_factor,
            self.damp_profile,
            self.linked_damp_profile,
            self.l,
            self.m,
            self.l4,
            self.sqrt_m,
            self.sqrt_m_p1,
            self.sqrt_p,
            self.sqrt_m_ladder,
            self.JlB,
            self.laguerre_to_grid,
            self.laguerre_to_spectral,
            self.laguerre_roots,
            self.laguerre_j0,
            self.laguerre_j1_over_alpha,
            self.kx_link_plus,
            self.kx_link_minus,
            self.kx_link_mask_plus,
            self.kx_link_mask_minus,
            self.linked_inverse_permutation,
            self.linked_gather_map,
            self.linked_gather_mask,
        )
        linked_idx = self.linked_indices or ()
        linked_kz = self.linked_kz or ()
        children = children + tuple(linked_idx) + tuple(linked_kz)
        aux_data = (
            self.use_twist_shift,
            self.jtwist,
            len(linked_idx),
            len(linked_kz),
            self.linked_full_cover,
            self.linked_use_gather,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            use_twist_shift,
            jtwist,
            n_linked_idx,
            n_linked_kz,
            linked_full_cover,
            linked_use_gather,
        ) = aux_data
        base_count = 51
        base_children = children[:base_count]
        linked_idx = tuple(children[base_count : base_count + n_linked_idx])
        linked_kz = tuple(
            children[
                base_count + n_linked_idx : base_count + n_linked_idx + n_linked_kz
            ]
        )
        return cls(
            *base_children,
            linked_indices=linked_idx,
            linked_kz=linked_kz,
            use_twist_shift=use_twist_shift,
            jtwist=jtwist,
            linked_full_cover=linked_full_cover,
            linked_use_gather=linked_use_gather,
        )


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
    kxfac_val = float(getattr(grid, "kxfac", 1.0))
    theta = jnp.asarray(grid.z, dtype=real_dtype)
    geom_data = ensure_flux_tube_geometry_data(geom, theta)
    gds2, gds21, gds22 = geom_data.metric_coeffs(theta)
    gds22_arr = gds22 if gds22.ndim else jnp.full_like(theta, gds22)
    bmag = geom_data.bmag(theta).astype(real_dtype)
    bgrad = geom_data.bgrad(theta).astype(real_dtype)
    jacobian = geom_data.jacobian(theta).astype(real_dtype)
    cv, gb, cv0, gb0 = geom_data.drift_coeffs(theta)
    boundary = str(getattr(grid, "boundary", "periodic")).lower()
    use_twist_shift = boundary in {"linked", "fix aspect", "continuous drifts"}
    use_ntft = bool(getattr(grid, "non_twist", False))
    y0 = getattr(grid, "y0", None)
    if y0 is None:
        if grid.ky.size > 1:
            y0 = float(1.0 / float(grid.ky[1] - grid.ky[0]))
        else:
            y0 = 1.0
    shat_arr = jnp.asarray(geom_data.s_hat, dtype=real_dtype)
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
    kperp2_bmag = bool(getattr(geom_data, "kperp2_bmag", True))
    if use_ntft:
        ftwist = (geom_data.s_hat * gds21 / gds22_arr).astype(real_dtype)
        kxfac_val = float(getattr(grid, "kxfac", 1.0))
        delta = jnp.asarray(0.01313, dtype=real_dtype)
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
        m0 = m0.astype(real_dtype)
        shat_inv = 1.0 / shat_arr
        delta_kx = ky_eff[:, None] * ftwist[None, :] + (rho_star * m0 / float(x0_eff))
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
        kperp2 = geom_data.k_perp2(kx0, ky0, theta_b).astype(real_dtype)
        cv_d, gb_d = geom_data.drift_components(kx_eff, ky_eff, theta)
        cv_d = cv_d.astype(real_dtype)
        gb_d = gb_d.astype(real_dtype)
        omega_d = (cv_d + gb_d).astype(real_dtype)
    apply_dealias_mask = dealias_mask is not None and int(grid.ky.size) > 1
    if apply_dealias_mask:
        mask = dealias_mask[:, :, None]
        kperp2 = kperp2 * mask
        cv_d = cv_d * mask
        gb_d = gb_d * mask
        omega_d = omega_d * mask
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
        damp_profile=damp_profile,
        linked_damp_profile=linked_damp_profile,
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
        kx_link_plus=kx_link_plus,
        kx_link_minus=kx_link_minus,
        kx_link_mask_plus=kx_link_mask_plus,
        kx_link_mask_minus=kx_link_mask_minus,
        linked_full_cover=linked_full_cover,
        linked_inverse_permutation=linked_inverse_permutation,
        linked_gather_map=linked_gather_map,
        linked_gather_mask=linked_gather_mask,
        linked_use_gather=linked_use_gather,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        use_twist_shift=use_twist_shift,
        jtwist=int(jtwist),
    )
