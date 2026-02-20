"""Linear electrostatic gyrokinetic building blocks (Hermite-Laguerre)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres

from spectraxgk.basis import hermite_ladder_coeffs
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.gyroaverage import J_l_all
from spectraxgk.grids import SpectralGrid


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearParams:
    """Parameters for the linear gyrokinetic operator (supports multi-species arrays)."""

    charge_sign: float | jnp.ndarray = 1.0
    density: float | jnp.ndarray = 1.0
    mass: float | jnp.ndarray = 1.0
    temp: float | jnp.ndarray = 1.0
    tau_e: float = 1.0
    vth: float | jnp.ndarray = 1.0
    rho: float | jnp.ndarray = 1.0
    kpar_scale: float = 1.0
    R_over_Ln: float | jnp.ndarray = 2.2
    R_over_LTi: float | jnp.ndarray = 6.9
    R_over_LTe: float | jnp.ndarray = 0.0
    omega_d_scale: float = 1.0
    omega_star_scale: float = 1.0
    energy_const: float = 0.0
    energy_par_coef: float = 0.5
    energy_perp_coef: float = 1.0
    nu: float | jnp.ndarray = 0.0
    nu_hermite: float = 1.0
    nu_laguerre: float = 2.0
    nu_hyper: float = 0.0
    p_hyper: float = 4.0
    nu_hyper_l: float = 0.0
    nu_hyper_m: float = 1.0
    nu_hyper_lm: float = 0.0
    p_hyper_l: float = 6.0
    p_hyper_m: float = 20.0
    p_hyper_lm: float = 6.0
    hypercollisions_const: float = 1.0
    hypercollisions_kz: float = 0.0
    damp_ends_widthfrac: float = 0.125
    damp_ends_amp: float = 0.1
    tz: float | jnp.ndarray = 1.0
    rho_star: float = 1.0
    beta: float = 0.0
    fapar: float = 0.0

    def tree_flatten(self):
        children = (
            self.charge_sign,
            self.density,
            self.mass,
            self.temp,
            self.tau_e,
            self.vth,
            self.rho,
            self.kpar_scale,
            self.R_over_Ln,
            self.R_over_LTi,
            self.R_over_LTe,
            self.omega_d_scale,
            self.omega_star_scale,
            self.energy_const,
            self.energy_par_coef,
            self.energy_perp_coef,
            self.nu,
            self.nu_hermite,
            self.nu_laguerre,
            self.nu_hyper,
            self.p_hyper,
            self.nu_hyper_l,
            self.nu_hyper_m,
            self.nu_hyper_lm,
            self.p_hyper_l,
            self.p_hyper_m,
            self.p_hyper_lm,
            self.hypercollisions_const,
            self.hypercollisions_kz,
            self.damp_ends_widthfrac,
            self.damp_ends_amp,
            self.tz,
            self.rho_star,
            self.beta,
            self.fapar,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearTerms:
    """Switches for linear-operator components (1.0 = on, 0.0 = off)."""

    streaming: float = 1.0
    mirror: float = 1.0
    curvature: float = 1.0
    gradb: float = 1.0
    diamagnetic: float = 1.0
    collisions: float = 1.0
    hypercollisions: float = 1.0
    end_damping: float = 1.0
    apar: float = 1.0
    bpar: float = 1.0

    def tree_flatten(self):
        children = (
            self.streaming,
            self.mirror,
            self.curvature,
            self.gradb,
            self.diamagnetic,
            self.collisions,
            self.hypercollisions,
            self.end_damping,
            self.apar,
            self.bpar,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _is_tracer(x) -> bool:
    return isinstance(x, jax.core.Tracer)

def _x64_enabled() -> bool:
    return bool(getattr(jax.config, "jax_enable_x64", False))


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


def _as_species_array(value: float | jnp.ndarray, ns: int, name: str) -> jnp.ndarray:
    """Ensure a parameter is a 1D array of length ns for multi-species handling."""

    arr = jnp.asarray(value)
    if arr.ndim == 0:
        arr = arr[None]
    if arr.size == 1:
        return jnp.broadcast_to(arr, (ns,))
    if int(arr.size) != int(ns):
        raise ValueError(f"{name} must have length {ns} (got {arr.size})")
    return arr


Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]
PreconditionerSpec = Preconditioner | str | None


def _resolve_implicit_preconditioner(preconditioner: PreconditionerSpec) -> PreconditionerSpec:
    if preconditioner is None:
        return "auto"
    if isinstance(preconditioner, str):
        return preconditioner.strip().lower()
    return preconditioner


def grad_z_periodic(f: jnp.ndarray, dz: float | jnp.ndarray) -> jnp.ndarray:
    """Spectral periodic derivative along the last axis."""

    _check_positive(dz, "dz")
    n = f.shape[-1]
    dz_val = jnp.asarray(dz, dtype=jnp.real(f).dtype)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dz_val)
    f_hat = jnp.fft.fft(f, axis=-1)
    df_hat = (1j * kz) * f_hat
    return jnp.fft.ifft(df_hat, axis=-1)


def compute_b(grid: SpectralGrid, geom: SAlphaGeometry, rho: float) -> jnp.ndarray:
    """Compute b = rho^2 * k_perp^2(kx, ky, theta) for s-alpha geometry."""

    _check_positive(rho, "rho")
    kx0 = grid.kx[None, :, None]
    ky = grid.ky[:, None, None]
    theta = grid.z[None, None, :]
    kperp2 = geom.k_perp2(kx0, ky, theta)
    return (rho * rho) * kperp2


def lenard_bernstein_eigenvalues(
    Nl: int, Nm: int, nu_hermite: float, nu_laguerre: float
) -> jnp.ndarray:
    """Diagonal Lenard-Bernstein rates in Hermite-Laguerre space."""

    l = jnp.arange(Nl)
    m = jnp.arange(Nm)
    return nu_laguerre * l[:, None] + nu_hermite * m[None, :]


def hypercollision_damping(
    cache: "LinearCache",
    params: "LinearParams",
    real_dtype: jnp.dtype,
) -> jnp.ndarray:
    """Assemble GX-style hypercollision damping factors."""

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
        vth_s * (scaled_nu_l * ratio_l + scaled_nu_m * ratio_m)
        + nu_hyper_lm * ratio_lm
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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearCache:
    """Precomputed arrays for the linear operator."""

    Jl: jnp.ndarray
    b: jnp.ndarray
    kperp2: jnp.ndarray
    bmag: jnp.ndarray
    omega_d: jnp.ndarray
    cv_d: jnp.ndarray
    gb_d: jnp.ndarray
    bgrad: jnp.ndarray
    mask0: jnp.ndarray
    dz: jnp.ndarray
    kz: jnp.ndarray
    ky: jnp.ndarray
    lb_lam: jnp.ndarray
    hyper_ratio: jnp.ndarray
    ratio_l: jnp.ndarray
    ratio_m: jnp.ndarray
    ratio_lm: jnp.ndarray
    mask_const: jnp.ndarray
    mask_kz: jnp.ndarray
    m_pow: jnp.ndarray
    m_norm_kz_factor: jnp.ndarray
    damp_profile: jnp.ndarray
    l: jnp.ndarray
    m: jnp.ndarray
    l4: jnp.ndarray
    sqrt_m: jnp.ndarray
    sqrt_m_p1: jnp.ndarray
    sqrt_p: jnp.ndarray
    sqrt_m_ladder: jnp.ndarray
    JlB: jnp.ndarray
    kx_link_plus: jnp.ndarray
    kx_link_minus: jnp.ndarray
    kx_link_mask_plus: jnp.ndarray
    kx_link_mask_minus: jnp.ndarray
    use_twist_shift: bool = False
    jtwist: int = 0

    def tree_flatten(self):
        children = (
            self.Jl,
            self.b,
            self.kperp2,
            self.bmag,
            self.omega_d,
            self.cv_d,
            self.gb_d,
            self.bgrad,
            self.mask0,
            self.dz,
            self.kz,
            self.ky,
            self.lb_lam,
            self.hyper_ratio,
            self.ratio_l,
            self.ratio_m,
            self.ratio_lm,
            self.mask_const,
            self.mask_kz,
            self.m_pow,
            self.m_norm_kz_factor,
            self.damp_profile,
            self.l,
            self.m,
            self.l4,
            self.sqrt_m,
            self.sqrt_m_p1,
            self.sqrt_p,
            self.sqrt_m_ladder,
            self.JlB,
            self.kx_link_plus,
            self.kx_link_minus,
            self.kx_link_mask_plus,
            self.kx_link_mask_minus,
        )
        aux_data = (self.use_twist_shift, self.jtwist)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        use_twist_shift, jtwist = aux_data
        return cls(*children, use_twist_shift=use_twist_shift, jtwist=jtwist)


def build_linear_cache(
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    Nl: int,
    Nm: int,
) -> LinearCache:
    """Build reusable arrays for the linear RHS."""

    real_dtype = jnp.float64 if _x64_enabled() else jnp.float32
    dz = jnp.asarray(grid.z[1] - grid.z[0], dtype=real_dtype)
    kz = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(grid.z.size, d=dz), dtype=real_dtype)
    rho_star = jnp.asarray(params.rho_star, dtype=real_dtype)
    kx_raw = jnp.asarray(grid.kx, dtype=real_dtype)
    ky_raw = jnp.asarray(grid.ky, dtype=real_dtype)
    kx_eff = rho_star * kx_raw
    ky_eff = rho_star * ky_raw
    theta = jnp.asarray(grid.z, dtype=real_dtype)
    gds2, gds21, gds22 = geom.metric_coeffs(theta)
    gds22_arr = gds22 if gds22.ndim else jnp.full_like(theta, gds22)
    bmag = geom.bmag(theta).astype(real_dtype)
    bgrad = geom.bgrad(theta).astype(real_dtype)
    cv, gb, cv0, gb0 = geom.drift_coeffs(theta)
    boundary = str(getattr(grid, "boundary", "periodic")).lower()
    use_twist_shift = boundary == "linked"
    y0 = getattr(grid, "y0", None)
    if y0 is None:
        if grid.ky.size > 1:
            y0 = float(1.0 / float(grid.ky[1] - grid.ky[0]))
        else:
            y0 = 1.0
    shat = float(geom.s_hat)
    if use_twist_shift and abs(shat) < 1.0e-12:
        use_twist_shift = False
    x0_eff = float(getattr(grid, "x0", 1.0))
    jtwist = 0
    if use_twist_shift:
        gds21_min = float(gds21[0]) if gds21.ndim else float(gds21)
        gds22_min = float(gds22[0]) if gds22.ndim else float(gds22)
        twist_shift_geo_fac = 0.0
        if gds22_min != 0.0:
            twist_shift_geo_fac = float(2.0 * shat * gds21_min / gds22_min)
        if twist_shift_geo_fac != 0.0:
            jtwist = grid.jtwist if getattr(grid, "jtwist", None) is not None else int(
                np.round(twist_shift_geo_fac)
            )
            if jtwist == 0:
                jtwist = 1
            x0_eff = float(y0) * abs(jtwist) / abs(twist_shift_geo_fac)
        else:
            jtwist = grid.jtwist if getattr(grid, "jtwist", None) is not None else 1
        if float(getattr(grid, "x0", x0_eff)) != 0.0:
            kx_eff = kx_eff * (float(getattr(grid, "x0", x0_eff)) / float(x0_eff))
    if use_twist_shift:
        ftwist = (geom.s_hat * gds21 / gds22_arr).astype(real_dtype)
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
        shat_inv = 1.0 / shat
        delta_kx = ky_eff[:, None] * ftwist[None, :] + (rho_star * m0 / float(x0_eff))
        term_ky = ky_eff[:, None, None] ** 2 * (
            gds2[None, None, :]
            - 2.0 * ftwist[None, None, :] * gds21[None, None, :] * shat_inv
            + (ftwist[None, None, :] ** 2) * gds22_arr[None, None, :] * shat_inv * shat_inv
        )
        term_kx = (kx_eff[None, :, None] + delta_kx[:, None, :]) ** 2 * gds22_arr[
            None, None, :
        ] * shat_inv * shat_inv
        bmag_inv = 1.0 / bmag
        kperp2 = (term_ky + term_kx) * (bmag_inv[None, None, :] ** 2)
        kx_shift = kx_eff[None, :, None] + (rho_star * m0 / float(x0_eff))[:, None, :]
        cv_d = ky_eff[:, None, None] * cv[None, None, :] + shat_inv * kx_shift * cv0[
            None, None, :
        ]
        gb_d = ky_eff[:, None, None] * gb[None, None, :] + shat_inv * kx_shift * gb0[
            None, None, :
        ]
        omega_d = cv_d + gb_d
    else:
        kx0 = kx_eff[None, :, None]
        ky0 = ky_eff[:, None, None]
        theta_b = theta[None, None, :]
        kperp2 = geom.k_perp2(kx0, ky0, theta_b).astype(real_dtype)
        cv_d, gb_d = geom.drift_components(kx_eff, ky_eff, theta)
        cv_d = cv_d.astype(real_dtype)
        gb_d = gb_d.astype(real_dtype)
        omega_d = (cv_d + gb_d).astype(real_dtype)
    rho = jnp.asarray(params.rho, dtype=real_dtype)
    if rho.ndim == 0:
        rho = rho[None]
    b = (rho[:, None, None, None] * rho[:, None, None, None]) * kperp2[None, ...]
    Jl = jax.vmap(lambda bs: J_l_all(bs, l_max=Nl - 1))(b).astype(real_dtype)
    JlB = Jl + shift_axis(Jl, -1, axis=1)
    mask0 = (grid.ky == 0.0)[:, None, None] & (grid.kx == 0.0)[None, :, None]
    lb_base = lenard_bernstein_eigenvalues(Nl, Nm, params.nu_hermite, params.nu_laguerre)[
        None, :, :, None, None, None
    ]
    lb_lam = lb_base + b[:, None, None, ...]
    l = jnp.arange(Nl, dtype=real_dtype)[:, None, None, None, None]
    m = jnp.arange(Nm, dtype=real_dtype)[None, :, None, None, None]
    l4 = jnp.arange(Nl, dtype=real_dtype)[:, None, None, None]
    m_p1 = m + 1.0
    sqrt_m = jnp.sqrt(m)
    sqrt_m_p1 = jnp.sqrt(m_p1)
    sqrt_p, sqrt_m_ladder = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m_ladder = sqrt_m_ladder[:Nm]
    sqrt_shape = [1] * 6
    sqrt_shape[2] = Nm
    sqrt_p = sqrt_p.reshape(sqrt_shape).astype(real_dtype)
    sqrt_m_ladder = sqrt_m_ladder.reshape(sqrt_shape).astype(real_dtype)
    l_norm = jnp.maximum(Nl - 1, 1)
    m_norm = jnp.maximum(Nm - 1, 1)
    hyper_ratio = (l / l_norm) ** params.p_hyper + (m / m_norm) ** params.p_hyper
    l_norm_full = jnp.asarray(max(Nl, 1), dtype=real_dtype)
    m_norm_full = jnp.asarray(max(Nm, 1), dtype=real_dtype)
    m_norm_kz = jnp.asarray(max(Nm - 1, 1), dtype=real_dtype)
    p_hyper_l = jnp.asarray(params.p_hyper_l, dtype=real_dtype)
    p_hyper_m = jnp.asarray(params.p_hyper_m, dtype=real_dtype)
    p_hyper_lm = jnp.asarray(params.p_hyper_lm, dtype=real_dtype)
    ratio_l = (l / l_norm_full) ** p_hyper_l
    ratio_m = (m / m_norm_full) ** p_hyper_m
    ratio_lm = ((2.0 * l + m) / (2.0 * l_norm_full + m_norm_full)) ** p_hyper_lm
    mask_const = (m > 2.0) | (l > 1.0)
    mask_kz = m > 2.0
    m_pow = m ** p_hyper_m
    m_norm_kz_factor = (p_hyper_m + 0.5) / (m_norm_kz ** (p_hyper_m + 0.5))
    Nz = grid.z.size
    width = jnp.maximum(1, jnp.asarray(jnp.floor(params.damp_ends_widthfrac * Nz), dtype=jnp.int32))
    idx = jnp.arange(Nz, dtype=jnp.float32)
    width_f = jnp.asarray(width, dtype=jnp.float32)
    left_mask = idx <= width_f
    right_mask = idx >= (Nz - width_f)
    x_left = jnp.where(left_mask, idx / width_f, 0.0)
    x_right = jnp.where(right_mask, (Nz - idx) / width_f, 0.0)
    nu_left = jnp.where(left_mask, 1.0 - 2.0 * x_left * x_left / (1.0 + x_left**4), 0.0)
    nu_right = jnp.where(right_mask, 1.0 - 2.0 * x_right * x_right / (1.0 + x_right**4), 0.0)
    damp_profile = jnp.maximum(nu_left, nu_right).astype(real_dtype)
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
    return LinearCache(
        Jl=Jl,
        b=b.astype(real_dtype),
        kperp2=kperp2,
        bmag=bmag,
        omega_d=omega_d,
        cv_d=cv_d,
        gb_d=gb_d,
        bgrad=bgrad,
        mask0=mask0,
        dz=dz,
        kz=kz,
        ky=ky_eff.astype(real_dtype),
        lb_lam=lb_lam.astype(real_dtype),
        hyper_ratio=hyper_ratio.astype(real_dtype),
        ratio_l=ratio_l.astype(real_dtype),
        ratio_m=ratio_m.astype(real_dtype),
        ratio_lm=ratio_lm.astype(real_dtype),
        mask_const=mask_const,
        mask_kz=mask_kz,
        m_pow=m_pow.astype(real_dtype),
        m_norm_kz_factor=m_norm_kz_factor.astype(real_dtype),
        damp_profile=damp_profile,
        l=l,
        m=m,
        l4=l4,
        sqrt_m=sqrt_m.astype(real_dtype),
        sqrt_m_p1=sqrt_m_p1.astype(real_dtype),
        sqrt_p=sqrt_p,
        sqrt_m_ladder=sqrt_m_ladder,
        JlB=JlB.astype(real_dtype),
        kx_link_plus=kx_link_plus,
        kx_link_minus=kx_link_minus,
        kx_link_mask_plus=kx_link_mask_plus,
        kx_link_mask_minus=kx_link_mask_minus,
        use_twist_shift=use_twist_shift,
        jtwist=int(jtwist),
    )


def apply_hermite_v(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Hermite coefficients by v_parallel (ladder form)."""

    axis_m = -4
    Nm = G.shape[axis_m]
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]

    pad = [(0, 0)] * G.ndim
    pad[axis_m] = (1, 1)
    G_pad = jnp.pad(G, pad)
    slc_plus = [slice(None)] * G.ndim
    slc_minus = [slice(None)] * G.ndim
    slc_plus[axis_m] = slice(2, None)
    slc_minus[axis_m] = slice(0, -2)
    G_plus = G_pad[tuple(slc_plus)]
    G_minus = G_pad[tuple(slc_minus)]
    shape = [1] * G.ndim
    shape[axis_m] = Nm
    sqrt_p = sqrt_p.reshape(shape)
    sqrt_m = sqrt_m.reshape(shape)
    return sqrt_p * G_plus + sqrt_m * G_minus


def apply_hermite_v2(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Hermite coefficients by v_parallel^2."""

    return apply_hermite_v(apply_hermite_v(G))


def apply_laguerre_x(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Laguerre coefficients by the perpendicular energy variable."""

    axis_l = -5
    Nl = G.shape[axis_l]
    l = jnp.arange(Nl)
    pad = [(0, 0)] * G.ndim
    pad[axis_l] = (1, 1)
    G_pad = jnp.pad(G, pad)
    slc_plus = [slice(None)] * G.ndim
    slc_minus = [slice(None)] * G.ndim
    slc_plus[axis_l] = slice(2, None)
    slc_minus[axis_l] = slice(0, -2)
    G_plus = G_pad[tuple(slc_plus)]
    G_minus = G_pad[tuple(slc_minus)]
    l_shape = [1] * G.ndim
    l_shape[axis_l] = Nl
    l_col = l.reshape(l_shape)
    return (
        (2.0 * l_col + 1.0) * G
        - (l_col + 1.0) * G_plus
        - l_col * G_minus
    )


def shift_axis(arr: jnp.ndarray, offset: int, axis: int) -> jnp.ndarray:
    """Shift an array along an axis with zero padding (non-periodic)."""

    if offset == 0:
        return arr
    pad = [(0, 0)] * arr.ndim
    if offset > 0:
        pad[axis] = (0, offset)
        arr_pad = jnp.pad(arr, pad)
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(offset, offset + arr.shape[axis])
        return arr_pad[tuple(slc)]
    pad[axis] = (-offset, 0)
    arr_pad = jnp.pad(arr, pad)
    slc = [slice(None)] * arr.ndim
    slc[axis] = slice(0, arr.shape[axis])
    return arr_pad[tuple(slc)]


def energy_operator(
    G: jnp.ndarray, coeff_const: float, coeff_par: float, coeff_perp: float
) -> jnp.ndarray:
    """Apply the energy operator (1 + v_par^2 + mu) in Hermite-Laguerre space."""

    return coeff_const * G + coeff_par * apply_hermite_v2(G) + coeff_perp * apply_laguerre_x(G)


def diamagnetic_drive_coeffs(
    Nl: int,
    Nm: int,
    eta_i: jnp.ndarray,
    coeff_const: float,
    coeff_par: float,
    coeff_perp: float,
) -> jnp.ndarray:
    """Return velocity-space coefficients for (1 + eta_i(E - 3/2))."""

    e00 = jnp.zeros((Nl, Nm, 1, 1, 1))
    e00 = e00.at[0, 0, 0, 0, 0].set(1.0)
    energy_e00 = energy_operator(e00, coeff_const, coeff_par, coeff_perp)
    coeffs = e00 + eta_i * (energy_e00 - 1.5 * e00)
    return coeffs[:, :, 0, 0, 0]


def quasineutrality_phi(
    G: jnp.ndarray,
    Jl: jnp.ndarray,
    tau_e: float | jnp.ndarray,
    charge: jnp.ndarray,
    density: jnp.ndarray,
    tz: jnp.ndarray,
) -> jnp.ndarray:
    """Solve electrostatic quasineutrality for phi with optional adiabatic closure."""

    _check_nonnegative(tau_e, "tau_e")
    Gm0 = G[:, :, 0, ...]
    num = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * jnp.sum(Jl * Gm0, axis=1),
        axis=0,
    )
    g0 = jnp.sum(Jl * Jl, axis=1)
    zt = jnp.where(tz == 0.0, 0.0, 1.0 / tz)
    den = tau_e + jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * zt[:, None, None, None]
        * (1.0 - g0),
        axis=0,
    )
    den_safe = jnp.where(den == 0.0, jnp.inf, den)
    return num / den_safe


def build_H(
    G: jnp.ndarray,
    Jl: jnp.ndarray,
    phi: jnp.ndarray,
    tz: jnp.ndarray,
    apar: jnp.ndarray | None = None,
    vth: jnp.ndarray | None = None,
    bpar: jnp.ndarray | None = None,
    JlB: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Map G -> H = G + tz * J_l(b) * phi * delta_{m0} (+ Apar term in m=1)."""

    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    if Jl.ndim == 4:
        Jl = Jl[None, ...]
    tz_arr = jnp.asarray(tz)
    if tz_arr.ndim == 0:
        tz_arr = tz_arr[None]
    zt_arr = jnp.where(tz_arr == 0.0, 0.0, 1.0 / tz_arr)
    H = G.at[:, :, 0, ...].add(zt_arr[:, None, None, None, None] * Jl * phi)
    if bpar is not None:
        if JlB is None:
            raise ValueError("JlB must be provided when bpar is supplied")
        H = H.at[:, :, 0, ...].add(JlB * bpar)
    if apar is not None and vth is not None:
        vth_arr = jnp.asarray(vth)
        if vth_arr.ndim == 0:
            vth_arr = vth_arr[None]
        H = H.at[:, :, 1, ...].add(
            -zt_arr[:, None, None, None, None] * vth_arr[:, None, None, None, None] * Jl * apar
        )
    return H[0] if squeeze_species else H


def streaming_term(
    H: jnp.ndarray, dz: float | jnp.ndarray, vth: float | jnp.ndarray
) -> jnp.ndarray:
    """Streaming term using Hermite ladder and real-space z derivative."""

    _check_positive(vth, "vth")
    dH_dz = grad_z_periodic(H, dz)
    axis_m = -4
    Nm = H.shape[axis_m]
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]

    pad = [(0, 0)] * H.ndim
    pad[axis_m] = (1, 1)
    dH_pad = jnp.pad(dH_dz, pad)
    slc_plus = [slice(None)] * H.ndim
    slc_minus = [slice(None)] * H.ndim
    slc_plus[axis_m] = slice(2, None)
    slc_minus[axis_m] = slice(0, -2)
    dH_plus = dH_pad[tuple(slc_plus)]
    dH_minus = dH_pad[tuple(slc_minus)]

    shape = [1] * H.ndim
    shape[axis_m] = Nm
    sqrt_p = sqrt_p.reshape(shape)
    sqrt_m = sqrt_m.reshape(shape)
    ladder = sqrt_p * dH_plus + sqrt_m * dH_minus
    vth_arr = jnp.asarray(vth)
    if vth_arr.ndim == 0:
        vth_arr = vth_arr[None]
    v_shape = [1] * H.ndim
    v_shape[0] = vth_arr.shape[0]
    vth_arr = vth_arr.reshape(v_shape)
    return vth_arr * ladder


def linear_rhs(
    G: jnp.ndarray,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    terms: LinearTerms | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS and electrostatic potential.

    Parameters
    ----------
    G : jnp.ndarray
        Laguerre-Hermite moments with shape (Nl, Nm, Ny, Nx, Nz).
    grid : SpectralGrid
        Flux-tube spectral grid.
    geom : SAlphaGeometry
        Analytic s-alpha geometry.
    params : LinearParams
        Physical and normalization parameters.
    """

    if G.ndim == 5:
        Nl, Nm = G.shape[0], G.shape[1]
    elif G.ndim == 6:
        Nl, Nm = G.shape[1], G.shape[2]
    else:
        raise ValueError("G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    return linear_rhs_cached(G, cache, params, terms=terms)


def linear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    use_jit: bool = True,
    use_custom_vjp: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS using precomputed geometry arrays."""

    if terms is None:
        terms = LinearTerms()

    from spectraxgk.terms.assembly import assemble_rhs_cached, assemble_rhs_cached_jit
    from spectraxgk.terms.config import TermConfig

    term_cfg = TermConfig(
        streaming=terms.streaming,
        mirror=terms.mirror,
        curvature=terms.curvature,
        gradb=terms.gradb,
        diamagnetic=terms.diamagnetic,
        collisions=terms.collisions,
        hypercollisions=terms.hypercollisions,
        end_damping=terms.end_damping,
        apar=terms.apar,
        bpar=terms.bpar,
        nonlinear=0.0,
    )

    if use_jit:
        dG, fields = assemble_rhs_cached_jit(G, cache, params, term_cfg)
    else:
        dG, fields = assemble_rhs_cached(
            G,
            cache,
            params,
            terms=term_cfg,
            use_custom_vjp=use_custom_vjp,
        )
    return dG, fields.phi


def _integrate_linear_cached_impl(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using cached geometry arrays."""
    if method not in {"euler", "rk2", "rk4", "imex", "imex2"}:
        raise ValueError("method must be one of {'euler', 'rk2', 'rk4', 'imex', 'imex2'}")
    if terms is None:
        terms = LinearTerms()

    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    lb_lam = cache.lb_lam.astype(real_dtype)
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if lb_lam.ndim == 6:
        ns = lb_lam.shape[0]
        nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
        damping = nu[:, None, None, None, None, None] * lb_lam + hyper_damp
        if G0.ndim == 5:
            damping = damping[0]
    else:
        damping = jnp.asarray(params.nu, dtype=real_dtype) * lb_lam + hyper_damp
    damping = damping.astype(real_dtype)

    def advance(G):
        dG, _phi = linear_rhs_cached(G, cache, params, terms=terms)
        if method == "imex":
            dG_explicit = dG + damping * G
            return (G + dt_val * dG_explicit) / (1.0 + dt_val * damping)
        if method == "imex2":
            dG_explicit = dG + damping * G
            G_half = (G + 0.5 * dt_val * dG_explicit) / (1.0 + 0.5 * dt_val * damping)
            dG_half, _phi = linear_rhs_cached(G_half, cache, params, terms=terms)
            dG_half_exp = dG_half + damping * G_half
            return (G + dt_val * dG_half_exp) / (1.0 + dt_val * damping)
        if method == "euler":
            return G + dt_val * dG
        if method == "rk2":
            k1 = dG
            k2, _ = linear_rhs_cached(G + 0.5 * dt_val * k1, cache, params, terms=terms)
            return G + dt_val * k2
        k1 = dG
        k2, _ = linear_rhs_cached(G + 0.5 * dt_val * k1, cache, params, terms=terms)
        k3, _ = linear_rhs_cached(G + 0.5 * dt_val * k2, cache, params, terms=terms)
        k4, _ = linear_rhs_cached(G + dt_val * k3, cache, params, terms=terms)
        return G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(G, _):
        G_new = advance(G)
        _dG_new, phi_new = linear_rhs_cached(G_new, cache, params, terms=terms)
        return G_new, phi_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    if sample_stride <= 1:
        return jax.lax.scan(step_fn, G0, None, length=steps)

    def sample_step(G, _):
        def inner_step(i, state):
            return advance(state)

        G_out = jax.lax.fori_loop(0, sample_stride, inner_step, G)
        _dG_out, phi_out = linear_rhs_cached(G_out, cache, params, terms=terms)
        return G_out, phi_out

    num_samples = steps // sample_stride
    return jax.lax.scan(sample_step, G0, None, length=num_samples)


@partial(
    jax.jit,
    static_argnames=("steps", "method", "checkpoint", "sample_stride"),
)
def _integrate_linear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
    )


@partial(
    jax.jit,
    static_argnames=("steps", "method", "checkpoint", "sample_stride"),
    donate_argnums=(0,),
)
def _integrate_linear_cached_donate(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    checkpoint: bool = False,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
    )


def _build_implicit_operator(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    terms: LinearTerms | None,
    implicit_preconditioner: PreconditionerSpec,
) -> tuple[jnp.ndarray, tuple[int, ...], int, jnp.ndarray, Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray], bool]:
    if terms is None:
        terms = LinearTerms()
    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)

    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    shape = G.shape
    size = int(np.prod(np.asarray(shape)))

    ns = shape[0]
    nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
    lb_lam = cache.lb_lam.astype(real_dtype)
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    damping = nu[:, None, None, None, None, None] * lb_lam + hyper_damp
    damping = damping.astype(real_dtype)

    l = cache.l.astype(real_dtype)
    m = cache.m.astype(real_dtype)
    cv_d = cache.cv_d.astype(real_dtype)
    gb_d = cache.gb_d.astype(real_dtype)
    bgrad = cache.bgrad.astype(real_dtype)
    w_mirror = jnp.asarray(terms.mirror, dtype=real_dtype)
    w_curv = jnp.asarray(terms.curvature, dtype=real_dtype)
    w_gradb = jnp.asarray(terms.gradb, dtype=real_dtype)
    diag = jnp.zeros_like(damping, dtype=state_dtype)
    imag = jnp.asarray(1j, dtype=state_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tz_b = tz[:, None, None, None, None, None]
    vth_b = vth[:, None, None, None, None, None]
    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    diag = diag - imag * tz_b * omega_d_scale * (
        w_curv * cv_d[None, None, None, ...] * (2.0 * m + 1.0)
        + w_gradb * gb_d[None, None, None, ...] * (2.0 * l + 1.0)
    )
    bgrad = bgrad[None, None, None, None, None, :]
    mirror_diag = vth_b * (2.0 * l + 1.0) * (2.0 * m + 1.0)
    mirror_weight = 0.2
    diag = diag - w_mirror * mirror_weight * bgrad * mirror_diag

    precond_full = 1.0 / (1.0 + dt_val * damping - dt_val * diag)
    precond_full = precond_full.astype(G.dtype)
    precond_damp = (1.0 / (1.0 + dt_val * damping)).astype(G.dtype)
    resolved_precond = _resolve_implicit_preconditioner(implicit_preconditioner)

    def apply_precond_full(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (x * precond_full).reshape(size)

    def apply_precond_damp(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        return (x * precond_damp).reshape(size)

    def apply_identity(x_flat: jnp.ndarray) -> jnp.ndarray:
        return x_flat

    precond_op: Callable[[jnp.ndarray], jnp.ndarray]
    if callable(resolved_precond):
        precond_op = resolved_precond
    else:
        key = resolved_precond or "auto"
        if key in {"auto", "diag", "diagonal", "physics", "block"}:
            precond_op = apply_precond_full
        elif key in {"damping", "collisional", "hyper"}:
            precond_op = apply_precond_damp
        elif key in {"identity", "none", "off"}:
            precond_op = apply_identity
        else:
            raise ValueError(f"Unknown implicit_preconditioner '{resolved_precond}'")

    def matvec(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(shape)
        dG, _phi = linear_rhs_cached(
            x,
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )
        return (x - dt_val * dG).reshape(size)

    return G, shape, size, dt_val, precond_op, matvec, squeeze_species


def _integrate_linear_implicit_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: LinearTerms | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: PreconditionerSpec = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Implicit linear integrator using GMRES with a diagonal preconditioner."""
    if terms is None:
        terms = LinearTerms()
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")

    G, shape, size, dt_val, precond_op, matvec, squeeze_species = _build_implicit_operator(
        G0, cache, params, dt, terms, implicit_preconditioner
    )

    def fixed_point(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        def body(_i, g):
            dG, _phi = linear_rhs_cached(
                g,
                cache,
                params,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
            )
            g_next = G_rhs + dt_val * dG
            return (1.0 - implicit_relax) * g + implicit_relax * g_next

        return jax.lax.fori_loop(0, max(int(implicit_iters), 0), body, G_in)

    def solve_step(G_in: jnp.ndarray) -> jnp.ndarray:
        G_guess = fixed_point(G_in, G_in)
        sol, _info = gmres(
            matvec,
            G_in.reshape(size),
            x0=G_guess.reshape(size),
            tol=implicit_tol,
            maxiter=implicit_maxiter,
            restart=implicit_restart,
            M=precond_op,
            solve_method=implicit_solve_method,
        )
        return sol.reshape(shape)

    def step(G_in, _):
        G_new = solve_step(G_in)
        _dG_new, phi_new = linear_rhs_cached(
            G_new,
            cache,
            params,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )
        return G_new, phi_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    if sample_stride <= 1:
        G_out, phi_t = jax.lax.scan(step_fn, G, None, length=steps)
    else:
        def sample_step(G_in, _):
            def inner_step(_i, g):
                return solve_step(g)

            G_out_local = jax.lax.fori_loop(0, sample_stride, inner_step, G_in)
            _dG_out, phi_out = linear_rhs_cached(
                G_out_local,
                cache,
                params,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
            )
            return G_out_local, phi_out

        num_samples = steps // sample_stride
        G_out, phi_t = jax.lax.scan(sample_step, G, None, length=num_samples)

    G_out = G_out[0] if squeeze_species else G_out
    return G_out, phi_t


def integrate_linear(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    cache: LinearCache | None = None,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: PreconditionerSpec = None,
    terms: LinearTerms | None = None,
    checkpoint: bool = False,
    sample_stride: int = 1,
    donate: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using a fixed-step scheme."""
    if terms is None:
        terms = LinearTerms()
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
    if method == "semi-implicit":
        method = "imex"
    if method == "implicit":
        return _integrate_linear_implicit_cached(
            G0,
            cache,
            params,
            dt=dt,
            steps=steps,
            terms=terms,
            implicit_tol=implicit_tol,
            implicit_maxiter=implicit_maxiter,
            implicit_iters=implicit_iters,
            implicit_relax=implicit_relax,
            implicit_restart=implicit_restart,
            implicit_solve_method=implicit_solve_method,
            implicit_preconditioner=implicit_preconditioner,
            checkpoint=checkpoint,
            sample_stride=sample_stride,
        )
    integrator = _integrate_linear_cached_donate if donate else _integrate_linear_cached
    return integrator(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        checkpoint=checkpoint,
        terms=terms,
        sample_stride=sample_stride,
    )


def integrate_linear_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    method: str = "rk4",
    cache: LinearCache | None = None,
    terms: LinearTerms | None = None,
    sample_stride: int = 1,
    species_index: int | None = 0,
    record_hl_energy: bool = False,
) -> (
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    """Integrate and return (G_out, phi_t, density_t) for diagnostics."""

    if terms is None:
        terms = LinearTerms()
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if steps % sample_stride != 0:
        raise ValueError("steps must be divisible by sample_stride")
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError("G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
        cache = build_linear_cache(grid, geom, params, Nl, Nm)

    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    lb_lam = cache.lb_lam.astype(real_dtype)
    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    if lb_lam.ndim == 6:
        ns = lb_lam.shape[0]
        nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
        damping = nu[:, None, None, None, None, None] * lb_lam + hyper_damp
        if G0.ndim == 5:
            damping = damping[0]
    else:
        damping = jnp.asarray(params.nu, dtype=real_dtype) * lb_lam + hyper_damp
    damping = damping.astype(real_dtype)

    def advance(G_in: jnp.ndarray) -> jnp.ndarray:
        dG, _phi = linear_rhs_cached(G_in, cache, params, terms=terms, use_jit=False)
        if method == "imex":
            dG_explicit = dG + damping * G_in
            return (G_in + dt_val * dG_explicit) / (1.0 + dt_val * damping)
        if method == "imex2":
            dG_explicit = dG + damping * G_in
            G_half = (G_in + 0.5 * dt_val * dG_explicit) / (1.0 + 0.5 * dt_val * damping)
            dG_half, _phi = linear_rhs_cached(G_half, cache, params, terms=terms, use_jit=False)
            dG_half_exp = dG_half + damping * G_half
            return (G_in + dt_val * dG_half_exp) / (1.0 + dt_val * damping)
        if method == "euler":
            return G_in + dt_val * dG
        if method == "rk2":
            k1 = dG
            k2, _ = linear_rhs_cached(G_in + 0.5 * dt_val * k1, cache, params, terms=terms, use_jit=False)
            return G_in + dt_val * k2
        if method == "rk4":
            k1 = dG
            k2, _ = linear_rhs_cached(G_in + 0.5 * dt_val * k1, cache, params, terms=terms, use_jit=False)
            k3, _ = linear_rhs_cached(G_in + 0.5 * dt_val * k2, cache, params, terms=terms, use_jit=False)
            k4, _ = linear_rhs_cached(G_in + dt_val * k3, cache, params, terms=terms, use_jit=False)
            return G_in + (dt_val / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        raise ValueError(f"Unsupported method '{method}'")

    def density_from_G(G_in: jnp.ndarray) -> jnp.ndarray:
        Jl = cache.Jl
        if G_in.ndim == 5:
            if Jl.ndim == 5:
                Jl_s = Jl[0]
            else:
                Jl_s = Jl
            return jnp.sum(Jl_s * G_in[:, 0, ...], axis=0)
        if Jl.ndim == 5:
            if species_index is None:
                return jnp.sum(jnp.sum(Jl * G_in[:, :, 0, ...], axis=1), axis=0)
            Jl_s = Jl[int(species_index)]
            return jnp.sum(Jl_s * G_in[int(species_index), :, 0, ...], axis=0)
        if species_index is None:
            return jnp.sum(jnp.sum(Jl[None, ...] * G_in[:, :, 0, ...], axis=1), axis=0)
        return jnp.sum(Jl * G_in[int(species_index), :, 0, ...], axis=0)

    def hl_energy_from_G(G_in: jnp.ndarray) -> jnp.ndarray:
        if G_in.ndim == 5:
            return jnp.sum(jnp.abs(G_in) ** 2, axis=(2, 3, 4))
        return jnp.sum(jnp.abs(G_in) ** 2, axis=(0, 3, 4, 5))

    def step(G_in, _):
        G_out = advance(G_in)
        _dG, phi = linear_rhs_cached(G_out, cache, params, terms=terms, use_jit=False)
        density = density_from_G(G_out)
        if record_hl_energy:
            hl_energy = hl_energy_from_G(G_out)
            return G_out, (phi, density, hl_energy)
        return G_out, (phi, density)

    if sample_stride <= 1:
        G_out, outputs = jax.lax.scan(step, G0, None, length=steps)
    else:
        def sample_step(G_in, _):
            def inner_step(_i, g):
                return step(g, None)[0]

            G_out_local = jax.lax.fori_loop(0, sample_stride, inner_step, G_in)
            _dG, phi_out = linear_rhs_cached(G_out_local, cache, params, terms=terms, use_jit=False)
            density_out = density_from_G(G_out_local)
            if record_hl_energy:
                hl_out = hl_energy_from_G(G_out_local)
                return G_out_local, (phi_out, density_out, hl_out)
            return G_out_local, (phi_out, density_out)

        num_samples = steps // sample_stride
        G_out, outputs = jax.lax.scan(sample_step, G0, None, length=num_samples)

    if record_hl_energy:
        phi_t, density_t, hl_t = outputs
        return G_out, phi_t, density_t, hl_t
    phi_t, density_t = outputs
    return G_out, phi_t, density_t
