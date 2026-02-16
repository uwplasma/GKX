"""Linear electrostatic gyrokinetic building blocks (Hermite-Laguerre)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

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
    ky: jnp.ndarray
    lb_lam: jnp.ndarray
    hyper_ratio: jnp.ndarray
    damp_profile: jnp.ndarray
    l: jnp.ndarray
    m: jnp.ndarray
    l4: jnp.ndarray
    sqrt_m: jnp.ndarray
    sqrt_m_p1: jnp.ndarray

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
            self.ky,
            self.lb_lam,
            self.hyper_ratio,
            self.damp_profile,
            self.l,
            self.m,
            self.l4,
            self.sqrt_m,
            self.sqrt_m_p1,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def build_linear_cache(
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    Nl: int,
    Nm: int,
) -> LinearCache:
    """Build reusable arrays for the linear RHS."""

    dz = jnp.asarray(grid.z[1] - grid.z[0])
    kx_eff = params.rho_star * grid.kx
    ky_eff = params.rho_star * grid.ky
    kx0 = kx_eff[None, :, None]
    ky0 = ky_eff[:, None, None]
    theta = grid.z[None, None, :]
    kperp2 = geom.k_perp2(kx0, ky0, theta)
    rho = jnp.asarray(params.rho)
    if rho.ndim == 0:
        rho = rho[None]
    b = (rho[:, None, None, None] * rho[:, None, None, None]) * kperp2[None, ...]
    Jl = jax.vmap(lambda bs: J_l_all(bs, l_max=Nl - 1))(b)
    omega_d = geom.omega_d(kx_eff, ky_eff, grid.z)
    cv_d, gb_d = geom.drift_components(kx_eff, ky_eff, grid.z)
    bgrad = geom.bgrad(grid.z)
    bmag = geom.bmag(grid.z)
    mask0 = (grid.ky == 0.0)[:, None, None] & (grid.kx == 0.0)[None, :, None]
    lb_base = lenard_bernstein_eigenvalues(Nl, Nm, params.nu_hermite, params.nu_laguerre)[
        None, :, :, None, None, None
    ]
    lb_lam = lb_base + b[:, None, None, ...]
    l = jnp.arange(Nl, dtype=jnp.float32)[:, None, None, None, None]
    m = jnp.arange(Nm, dtype=jnp.float32)[None, :, None, None, None]
    l4 = jnp.arange(Nl, dtype=jnp.float32)[:, None, None, None]
    m_p1 = m + 1.0
    sqrt_m = jnp.sqrt(m)
    sqrt_m_p1 = jnp.sqrt(m_p1)
    l_norm = jnp.maximum(Nl - 1, 1)
    m_norm = jnp.maximum(Nm - 1, 1)
    hyper_ratio = (l / l_norm) ** params.p_hyper + (m / m_norm) ** params.p_hyper
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
    damp_profile = jnp.maximum(nu_left, nu_right)
    return LinearCache(
        Jl=Jl,
        b=b,
        kperp2=kperp2,
        bmag=bmag,
        omega_d=omega_d,
        cv_d=cv_d,
        gb_d=gb_d,
        bgrad=bgrad,
        mask0=mask0,
        dz=dz,
        ky=ky_eff,
        lb_lam=lb_lam,
        hyper_ratio=hyper_ratio,
        damp_profile=damp_profile,
        l=l,
        m=m,
        l4=l4,
        sqrt_m=sqrt_m,
        sqrt_m_p1=sqrt_m_p1,
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
    tau_e: float,
    charge: jnp.ndarray,
    density: jnp.ndarray,
    tz: jnp.ndarray,
) -> jnp.ndarray:
    """Solve electrostatic quasineutrality for phi with optional adiabatic closure."""

    _check_positive(tau_e, "tau_e")
    Gm0 = G[:, :, 0, ...]
    num = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * jnp.sum(Jl * Gm0, axis=1),
        axis=0,
    )
    g0 = jnp.sum(Jl * Jl, axis=1)
    den = tau_e + jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * tz[:, None, None, None]
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
    H = G.at[:, :, 0, ...].add(tz_arr[:, None, None, None, None] * Jl * phi)
    if bpar is not None:
        if JlB is None:
            raise ValueError("JlB must be provided when bpar is supplied")
        H = H.at[:, :, 0, ...].add(JlB * bpar)
    if apar is not None and vth is not None:
        vth_arr = jnp.asarray(vth)
        if vth_arr.ndim == 0:
            vth_arr = vth_arr[None]
        H = H.at[:, :, 1, ...].add(
            -tz_arr[:, None, None, None, None] * vth_arr[:, None, None, None, None] * Jl * apar
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS using precomputed geometry arrays."""

    out_dtype = jnp.result_type(G, jnp.complex64)
    G = jnp.asarray(G, dtype=out_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=out_dtype)).dtype
    imag = jnp.asarray(1j, dtype=out_dtype)
    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    if G.ndim != 6:
        raise ValueError("G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
    if terms is None:
        terms = LinearTerms()

    ns = G.shape[0]
    charge = _as_species_array(params.charge_sign, ns, "charge_sign").astype(real_dtype)
    density = _as_species_array(params.density, ns, "density").astype(real_dtype)
    mass = _as_species_array(params.mass, ns, "mass").astype(real_dtype)
    temp = _as_species_array(params.temp, ns, "temp").astype(real_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tprim = _as_species_array(params.R_over_LTi, ns, "R_over_LTi").astype(real_dtype)
    fprim = _as_species_array(params.R_over_Ln, ns, "R_over_Ln").astype(real_dtype)
    nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    omega_star_scale = jnp.asarray(params.omega_star_scale, dtype=real_dtype)
    kpar_scale = jnp.asarray(params.kpar_scale, dtype=real_dtype)
    nu_hyper = jnp.asarray(params.nu_hyper, dtype=real_dtype)
    nu_hyper_l = jnp.asarray(params.nu_hyper_l, dtype=real_dtype)
    nu_hyper_m = jnp.asarray(params.nu_hyper_m, dtype=real_dtype)
    nu_hyper_lm = jnp.asarray(params.nu_hyper_lm, dtype=real_dtype)
    p_hyper_l = jnp.asarray(params.p_hyper_l, dtype=real_dtype)
    p_hyper_m = jnp.asarray(params.p_hyper_m, dtype=real_dtype)
    p_hyper_lm = jnp.asarray(params.p_hyper_lm, dtype=real_dtype)
    damp_amp = jnp.asarray(params.damp_ends_amp, dtype=real_dtype)
    w_stream = jnp.asarray(terms.streaming, dtype=real_dtype)
    w_mirror = jnp.asarray(terms.mirror, dtype=real_dtype)
    w_curv = jnp.asarray(terms.curvature, dtype=real_dtype)
    w_gradb = jnp.asarray(terms.gradb, dtype=real_dtype)
    w_dia = jnp.asarray(terms.diamagnetic, dtype=real_dtype)
    w_coll = jnp.asarray(terms.collisions, dtype=real_dtype)
    w_hyper = jnp.asarray(terms.hypercollisions, dtype=real_dtype)
    w_damp = jnp.asarray(terms.end_damping, dtype=real_dtype)
    w_apar = jnp.asarray(terms.apar, dtype=real_dtype)
    w_bpar = jnp.asarray(terms.bpar, dtype=real_dtype)

    if cache.Jl.shape[0] != ns:
        raise ValueError("Cache species dimension does not match G")

    Jl = cache.Jl.astype(real_dtype)
    Jl_m1 = shift_axis(Jl, -1, axis=1)
    JlB = Jl + Jl_m1
    omega_d = cache.omega_d.astype(real_dtype)
    cv_d = cache.cv_d.astype(real_dtype)
    gb_d = cache.gb_d.astype(real_dtype)
    bgrad = cache.bgrad.astype(real_dtype)
    bmag = cache.bmag.astype(real_dtype)
    kperp2 = cache.kperp2.astype(real_dtype)
    ky = cache.ky.astype(real_dtype)
    lb_lam = cache.lb_lam.astype(real_dtype)
    hyper_ratio = cache.hyper_ratio.astype(real_dtype)
    nu_hyper = jnp.asarray(params.nu_hyper, dtype=real_dtype)
    damp_profile = cache.damp_profile.astype(real_dtype)
    sqrt_m_p1 = cache.sqrt_m_p1.astype(real_dtype)
    sqrt_m = cache.sqrt_m.astype(real_dtype)
    l = cache.l.astype(real_dtype)
    m = cache.m.astype(real_dtype)

    beta = jnp.asarray(params.beta, dtype=real_dtype)
    fapar = jnp.asarray(params.fapar, dtype=real_dtype) * w_apar
    g0 = jnp.sum(Jl * Jl, axis=1)
    g01 = jnp.sum(Jl * JlB, axis=1)
    g11 = jnp.sum(JlB * JlB, axis=1)
    Gm1 = G[:, :, 1, ...]
    Gm0 = G[:, :, 0, ...]

    phi_es = quasineutrality_phi(G, Jl, params.tau_e, charge, density, tz)
    phi_es = jnp.where(cache.mask0, 0.0, phi_es)

    nbar = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * jnp.sum(Jl * Gm0, axis=1),
        axis=0,
    )
    bmag_inv2 = 1.0 / (bmag * bmag)
    jperpbar = jnp.sum(
        (-0.5 * beta)
        * density[:, None, None, None]
        * temp[:, None, None, None]
        * bmag_inv2[None, None, :]
        * jnp.sum(JlB * Gm0, axis=1),
        axis=0,
    )
    qphi = params.tau_e + jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * tz[:, None, None, None]
        * (1.0 - g0),
        axis=0,
    )
    qb = -jnp.sum(density[:, None, None, None] * charge[:, None, None, None] * g01, axis=0)
    aphi = 0.5 * beta * jnp.sum(
        density[:, None, None, None] * charge[:, None, None, None] * g01, axis=0
    ) * bmag_inv2[None, None, :]
    ab = 1.0 + 0.5 * beta * jnp.sum(
        density[:, None, None, None] * temp[:, None, None, None] * g11, axis=0
    ) * bmag_inv2[None, None, :]
    denom = qphi * ab - qb * aphi
    denom_safe = jnp.where(denom == 0.0, jnp.inf, denom)
    phi_em = (ab * nbar - qb * jperpbar) / denom_safe
    bpar_em = (-aphi * nbar + qphi * jperpbar) / denom_safe
    use_bpar = jnp.where((beta > 0.0) & (w_bpar > 0.0), 1.0, 0.0)
    phi = phi_es * (1.0 - use_bpar) + phi_em * use_bpar
    bpar = bpar_em * use_bpar
    phi = jnp.where(cache.mask0, 0.0, phi)
    bpar = jnp.where(cache.mask0, 0.0, bpar)
    jpar = jnp.sum(
        density[:, None, None, None]
        * charge[:, None, None, None]
        * vth[:, None, None, None]
        * jnp.sum(Jl * Gm1, axis=1),
        axis=0,
    )
    jpar = 0.5 * beta * jpar
    bmag2 = bmag[None, None, :] * bmag[None, None, :]
    ampere_denom = kperp2 * bmag2 + 0.5 * beta * jnp.sum(
        density[:, None, None, None]
        * (charge * charge / mass)[:, None, None, None]
        * g0,
        axis=0,
    )
    ampere_safe = jnp.where(ampere_denom == 0.0, jnp.inf, ampere_denom)
    apar = fapar * jpar / ampere_safe
    apar = jnp.where(cache.mask0, 0.0, apar)

    H = build_H(G, Jl, phi, tz, apar=apar, vth=vth, bpar=bpar, JlB=JlB)
    stream = streaming_term(H, cache.dz.astype(real_dtype), vth)
    dG = -kpar_scale * stream * w_stream

    axis_l = -5
    axis_m = -4
    Nm = G.shape[2]
    l_p1 = l + 1.0
    m_p1 = m + 1.0
    H_m_p1 = shift_axis(H, 1, axis=axis_m)
    H_m_m1 = shift_axis(H, -1, axis=axis_m)
    mirror_term = (
        -sqrt_m_p1 * l_p1 * H_m_p1
        - sqrt_m_p1 * l * shift_axis(H_m_p1, -1, axis=axis_l)
        + sqrt_m * l * H_m_m1
        + sqrt_m * l_p1 * shift_axis(H_m_m1, 1, axis=axis_l)
    )
    bgrad = omega_d_scale * bgrad[None, None, None, None, None, :]
    vth_b = vth[:, None, None, None, None, None]
    vth_s = vth[:, None, None, None, None]
    dG = dG - w_mirror * vth_b * bgrad * mirror_term

    icv = imag * tz[:, None, None, None, None, None] * omega_d_scale * cv_d[None, None, None, ...]
    igb = imag * tz[:, None, None, None, None, None] * omega_d_scale * gb_d[None, None, None, ...]
    H_m_p2 = shift_axis(H, 2, axis=axis_m)
    H_m_m2 = shift_axis(H, -2, axis=axis_m)
    curv_term = (
        jnp.sqrt((m + 1.0) * (m + 2.0)) * H_m_p2
        + (2.0 * m + 1.0) * H
        + jnp.sqrt(m * (m - 1.0)) * H_m_m2
    )
    gradb_term = (
        (l + 1.0) * shift_axis(H, 1, axis=axis_l)
        + (2.0 * l + 1.0) * H
        + l * shift_axis(H, -1, axis=axis_l)
    )
    dG = dG - w_curv * icv * curv_term - w_gradb * igb * gradb_term

    iky = imag * omega_star_scale * ky
    l4 = cache.l4.astype(real_dtype)
    Jl_m1 = shift_axis(Jl, -1, axis=1)
    Jl_p1 = shift_axis(Jl, 1, axis=1)
    JlB_m1 = shift_axis(JlB, -1, axis=1)
    JlB_p1 = shift_axis(JlB, 1, axis=1)
    tprim_s = tprim[:, None, None, None, None]
    fprim_s = fprim[:, None, None, None, None]
    tz_s = tz[:, None, None, None, None]
    iky_s = iky[None, None, :, None, None]
    drive_m0 = iky_s * phi * (
        Jl_m1 * (l4 * tprim_s)
        + Jl * (fprim_s + 2.0 * l4 * tprim_s)
        + Jl_p1 * ((l4 + 1.0) * tprim_s)
    )
    drive_m0 = drive_m0 + iky_s / tz_s * bpar * (
        JlB_m1 * (l4 * tprim_s)
        + JlB * (fprim_s + 2.0 * l4 * tprim_s)
        + JlB_p1 * ((l4 + 1.0) * tprim_s)
    )
    dG = dG.at[:, :, 0, ...].add(w_dia * drive_m0)
    if Nm > 2:
        drive_m2 = iky_s * phi * Jl * (tprim_s / jnp.sqrt(2.0))
        drive_m2 = drive_m2 + iky_s / tz_s * bpar * JlB * (tprim_s / jnp.sqrt(2.0))
        dG = dG.at[:, :, 2, ...].add(w_dia * drive_m2)
    if Nm > 1:
        apar_drive = -vth_s * iky_s * apar * (
            Jl_m1 * (l4 * tprim_s)
            + Jl * (fprim_s + (2.0 * l4 + 1.0) * tprim_s)
            + Jl_p1 * ((l4 + 1.0) * tprim_s)
        )
        dG = dG.at[:, :, 1, ...].add(w_dia * apar_drive)
    if Nm > 3:
        drive_m3 = -vth_s * iky_s * apar * Jl * (tprim_s * jnp.sqrt(3.0 / 2.0))
        dG = dG.at[:, :, 3, ...].add(w_dia * drive_m3)

    dG = dG - w_coll * nu[:, None, None, None, None, None] * lb_lam * H

    Nl = G.shape[0]
    Nm = G.shape[1]
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
    dG = dG + w_hyper * jnp.where(mask, hyper_term, 0.0) * G
    dG = dG - w_hyper * nu_hyper * hyper_ratio * G
    damp = w_damp * damp_amp * damp_profile[None, None, None, None, None, :]
    ky_mask = (ky > 0.0)[None, None, None, :, None, None]
    dG = dG - ky_mask * damp * H
    if squeeze_species:
        dG = dG[0]
    return dG.astype(out_dtype), phi.astype(out_dtype)


@partial(jax.jit, static_argnames=("steps", "method"))
def _integrate_linear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    terms: LinearTerms | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using cached geometry arrays."""
    if method not in {"euler", "rk2", "rk4", "imex"}:
        raise ValueError("method must be one of {'euler', 'rk2', 'rk4', 'imex'}")
    if terms is None:
        terms = LinearTerms()

    base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
    state_dtype = jnp.result_type(G0, base_dtype)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_val = jnp.asarray(dt, dtype=real_dtype)
    lb_lam = cache.lb_lam.astype(real_dtype)
    hyper_ratio = cache.hyper_ratio.astype(real_dtype)
    nu_hyper = jnp.asarray(params.nu_hyper, dtype=real_dtype)
    if lb_lam.ndim == 6:
        ns = lb_lam.shape[0]
        nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
        damping = nu[:, None, None, None, None, None] * lb_lam + nu_hyper * hyper_ratio
        if G0.ndim == 5:
            damping = damping[0]
    else:
        damping = jnp.asarray(params.nu, dtype=real_dtype) * lb_lam + nu_hyper * hyper_ratio
    damping = damping.astype(real_dtype)

    def step(G, _):
        dG, _phi = linear_rhs_cached(G, cache, params, terms=terms)
        if method == "imex":
            dG_explicit = dG + damping * G
            G_new = (G + dt_val * dG_explicit) / (1.0 + dt_val * damping)
        elif method == "euler":
            G_new = G + dt_val * dG
        elif method == "rk2":
            k1 = dG
            k2, _ = linear_rhs_cached(G + 0.5 * dt_val * k1, cache, params, terms=terms)
            G_new = G + dt_val * k2
        else:
            k1 = dG
            k2, _ = linear_rhs_cached(G + 0.5 * dt_val * k1, cache, params, terms=terms)
            k3, _ = linear_rhs_cached(G + 0.5 * dt_val * k2, cache, params, terms=terms)
            k4, _ = linear_rhs_cached(G + dt_val * k3, cache, params, terms=terms)
            G_new = G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        _dG_new, phi_new = linear_rhs_cached(G_new, cache, params, terms=terms)
        return G_new, phi_new

    return jax.lax.scan(step, G0, None, length=steps)


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
    terms: LinearTerms | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using a fixed-step scheme."""
    if terms is None:
        terms = LinearTerms()
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
        base_dtype = jnp.complex128 if _x64_enabled() else jnp.complex64
        state_dtype = jnp.result_type(G0, base_dtype)
        G = jnp.asarray(G0, dtype=state_dtype)
        real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
        squeeze_species = False
        if G.ndim == 5:
            G = G[None, ...]
            squeeze_species = True
        shape = G.shape
        size = 1
        for dim in shape:
            size *= int(dim)
        dt_val = jnp.asarray(dt, dtype=real_dtype)
        ns = shape[0]
        nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)
        lb_lam = cache.lb_lam.astype(real_dtype)
        hyper_ratio = cache.hyper_ratio.astype(real_dtype)
        nu_hyper = jnp.asarray(params.nu_hyper, dtype=real_dtype)
        damping = nu[:, None, None, None, None, None] * lb_lam + nu_hyper * hyper_ratio
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
        mirror_diag = vth_b * omega_d_scale * (2.0 * l + 1.0) * (2.0 * m + 1.0)
        mirror_weight = 0.2
        diag = diag - w_mirror * mirror_weight * bgrad * mirror_diag
        precond = 1.0 / (1.0 + dt_val * damping - dt_val * diag)
        precond = precond.astype(G.dtype)
        phi_out = []

        def matvec(x_flat: jnp.ndarray) -> jnp.ndarray:
            x = x_flat.reshape(shape)
            dG, _phi = linear_rhs_cached(x, cache, params, terms=terms)
            return (x - dt_val * dG).reshape(size)

        def apply_precond(x_flat: jnp.ndarray) -> jnp.ndarray:
            x = x_flat.reshape(shape)
            return (x * precond).reshape(size)

        for _ in range(steps):
            G_guess = G
            for _iter in range(max(implicit_iters, 0)):
                dG, _phi = linear_rhs_cached(G_guess, cache, params, terms=terms)
                G_next = G + dt_val * dG
                G_guess = (1.0 - implicit_relax) * G_guess + implicit_relax * G_next

            sol, _ = gmres(
                matvec,
                G.reshape(size),
                x0=G_guess.reshape(size),
                tol=implicit_tol,
                maxiter=implicit_maxiter,
                M=apply_precond,
            )
            G = sol.reshape(shape)
            _dG, phi = linear_rhs_cached(G, cache, params, terms=terms)
            phi_out.append(phi)
        G_out = G[0] if squeeze_species else G
        return G_out, jnp.stack(phi_out, axis=0)
    return _integrate_linear_cached(G0, cache, params, dt, steps, method=method, terms=terms)
