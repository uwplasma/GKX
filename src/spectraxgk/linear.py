"""Linear electrostatic gyrokinetic building blocks (Hermite-Laguerre)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from spectraxgk.basis import hermite_ladder_coeffs
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.gyroaverage import J_l_all
from spectraxgk.grids import SpectralGrid


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LinearParams:
    """Parameters for the linear electrostatic operator."""

    tau_e: float = 1.0
    vth: float = 1.0
    rho: float = 1.0
    kpar_scale: float = 1.0
    R_over_Ln: float = 2.2
    R_over_LTi: float = 6.9
    R_over_LTe: float = 0.0
    omega_d_scale: float = 1.0
    omega_star_scale: float = 1.0
    energy_const: float = 0.0
    energy_par_coef: float = 0.5
    energy_perp_coef: float = 1.0
    nu: float = 0.0
    nu_hermite: float = 1.0
    nu_laguerre: float = 2.0
    nu_hyper: float = 0.0
    p_hyper: float = 4.0
    tz: float = 1.0
    rho_star: float = 1.0

    def tree_flatten(self):
        children = (
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
            self.tz,
            self.rho_star,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def _is_tracer(x) -> bool:
    return isinstance(x, jax.core.Tracer)


def _check_positive(x, name: str) -> None:
    if _is_tracer(x):
        return
    if float(x) <= 0.0:
        raise ValueError(f"{name} must be > 0")


def grad_z_periodic(f: jnp.ndarray, dz: float | jnp.ndarray) -> jnp.ndarray:
    """Centered periodic derivative along the last axis."""

    _check_positive(dz, "dz")
    return (jnp.roll(f, -1, axis=-1) - jnp.roll(f, 1, axis=-1)) / (2.0 * dz)


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
    omega_d: jnp.ndarray
    cv_d: jnp.ndarray
    gb_d: jnp.ndarray
    bgrad: jnp.ndarray
    mask0: jnp.ndarray
    dz: jnp.ndarray
    ky: jnp.ndarray
    lb_lam: jnp.ndarray
    hyper_ratio: jnp.ndarray
    l: jnp.ndarray
    m: jnp.ndarray
    l4: jnp.ndarray
    sqrt_m: jnp.ndarray
    sqrt_m_p1: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.Jl,
            self.b,
            self.omega_d,
            self.cv_d,
            self.gb_d,
            self.bgrad,
            self.mask0,
            self.dz,
            self.ky,
            self.lb_lam,
            self.hyper_ratio,
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
    b = (params.rho * params.rho) * kperp2
    Jl = J_l_all(b, l_max=Nl - 1)
    omega_d = geom.omega_d(kx_eff, ky_eff, grid.z)
    cv_d, gb_d = geom.drift_components(kx_eff, ky_eff, grid.z)
    bgrad = geom.bgrad(grid.z)
    mask0 = (grid.ky == 0.0)[:, None, None] & (grid.kx == 0.0)[None, :, None]
    lb_lam = lenard_bernstein_eigenvalues(Nl, Nm, params.nu_hermite, params.nu_laguerre)[
        :, :, None, None, None
    ] + b[None, None, ...]
    l = jnp.arange(Nl, dtype=jnp.float32)[:, None, None, None, None]
    m = jnp.arange(Nm, dtype=jnp.float32)[None, :, None, None, None]
    l4 = jnp.arange(Nl, dtype=jnp.float32)[:, None, None, None]
    m_p1 = m + 1.0
    sqrt_m = jnp.sqrt(m)
    sqrt_m_p1 = jnp.sqrt(m_p1)
    l_norm = jnp.maximum(Nl - 1, 1)
    m_norm = jnp.maximum(Nm - 1, 1)
    hyper_ratio = (l / l_norm) ** params.p_hyper + (m / m_norm) ** params.p_hyper
    return LinearCache(
        Jl=Jl,
        b=b,
        omega_d=omega_d,
        cv_d=cv_d,
        gb_d=gb_d,
        bgrad=bgrad,
        mask0=mask0,
        dz=dz,
        ky=ky_eff,
        lb_lam=lb_lam,
        hyper_ratio=hyper_ratio,
        l=l,
        m=m,
        l4=l4,
        sqrt_m=sqrt_m,
        sqrt_m_p1=sqrt_m_p1,
    )


def apply_hermite_v(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Hermite coefficients by v_parallel (ladder form)."""

    Nm = G.shape[1]
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]

    pad = ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0))
    G_pad = jnp.pad(G, pad)
    G_plus = G_pad[:, 2:, ...]
    G_minus = G_pad[:, :-2, ...]
    return sqrt_p[None, :, None, None, None] * G_plus + sqrt_m[None, :, None, None, None] * G_minus


def apply_hermite_v2(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Hermite coefficients by v_parallel^2."""

    return apply_hermite_v(apply_hermite_v(G))


def apply_laguerre_x(G: jnp.ndarray) -> jnp.ndarray:
    """Multiply Laguerre coefficients by the perpendicular energy variable."""

    Nl = G.shape[0]
    l = jnp.arange(Nl)
    pad = ((1, 1), (0, 0), (0, 0), (0, 0), (0, 0))
    G_pad = jnp.pad(G, pad)
    G_plus = G_pad[2:, ...]
    G_minus = G_pad[:-2, ...]
    l_col = l[:, None, None, None, None]
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


def quasineutrality_phi(G: jnp.ndarray, Jl: jnp.ndarray, tau_e: float) -> jnp.ndarray:
    """Solve electrostatic quasineutrality for phi.

    Uses an adiabatic electron closure:
        (tau_e + 1 - sum_l J_l^2) * phi = sum_l J_l * G_{l,m=0}
    """

    _check_positive(tau_e, "tau_e")
    Gm0 = G[:, 0, ...]
    num = jnp.sum(Jl * Gm0, axis=0)
    den = tau_e + 1.0 - jnp.sum(Jl * Jl, axis=0)
    den_safe = jnp.where(den == 0.0, jnp.inf, den)
    return num / den_safe


def build_H(G: jnp.ndarray, Jl: jnp.ndarray, phi: jnp.ndarray, tz: float = 1.0) -> jnp.ndarray:
    """Map G -> H = G + tz * J_l(b) * phi * delta_{m0}."""

    return G.at[:, 0, ...].add(tz * Jl * phi)


def streaming_term(H: jnp.ndarray, dz: float | jnp.ndarray, vth: float) -> jnp.ndarray:
    """Streaming term using Hermite ladder and real-space z derivative."""

    _check_positive(vth, "vth")
    dH_dz = grad_z_periodic(H, dz)
    Nm = H.shape[1]
    sqrt_p, sqrt_m = hermite_ladder_coeffs(Nm - 1)
    sqrt_p = sqrt_p[:Nm]
    sqrt_m = sqrt_m[:Nm]

    pad = ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0))
    dH_pad = jnp.pad(dH_dz, pad)
    dH_plus = dH_pad[:, 2:, ...]
    dH_minus = dH_pad[:, :-2, ...]

    ladder = sqrt_p[None, :, None, None, None] * dH_plus + sqrt_m[None, :, None, None, None] * dH_minus
    return vth * ladder


def linear_rhs(
    G: jnp.ndarray,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    operator: str = "gx",
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

    if G.ndim != 5:
        raise ValueError("G must have shape (Nl, Nm, Ny, Nx, Nz)")
    cache = build_linear_cache(grid, geom, params, G.shape[0], G.shape[1])
    return linear_rhs_cached(G, cache, params, operator=operator)


def linear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    operator: str = "gx",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS using precomputed geometry arrays."""

    out_dtype = jnp.result_type(G, jnp.complex64)
    G = jnp.asarray(G, dtype=out_dtype)
    imag = jnp.asarray(1j, dtype=out_dtype)
    if G.ndim != 5:
        raise ValueError("G must have shape (Nl, Nm, Ny, Nx, Nz)")
    if operator == "full":
        operator = "gx"
    if operator not in {"gx", "energy"}:
        raise ValueError("operator must be one of {'full', 'energy'}")
    phi = quasineutrality_phi(G, cache.Jl, params.tau_e)
    phi = jnp.where(cache.mask0, 0.0, phi)
    H = build_H(G, cache.Jl, phi, params.tz)
    stream = streaming_term(H, cache.dz, params.vth)
    dG = -params.kpar_scale * stream

    if operator == "gx":
        l = cache.l
        m = cache.m
        Nm = G.shape[1]
        l_p1 = l + 1.0
        m_p1 = m + 1.0
        sqrt_m_p1 = cache.sqrt_m_p1
        sqrt_m = cache.sqrt_m

        H_m_p1 = shift_axis(H, 1, axis=1)
        H_m_m1 = shift_axis(H, -1, axis=1)
        mirror_term = (
            -sqrt_m_p1 * l_p1 * H_m_p1
            - sqrt_m_p1 * l * shift_axis(H_m_p1, -1, axis=0)
            + sqrt_m * l * H_m_m1
            + sqrt_m * l_p1 * shift_axis(H_m_m1, 1, axis=0)
        )
        bgrad = params.omega_d_scale * cache.bgrad[None, None, None, None, :]
        dG = dG - params.vth * bgrad * mirror_term

        icv = imag * params.tz * params.omega_d_scale * cache.cv_d[None, None, ...]
        igb = imag * params.tz * params.omega_d_scale * cache.gb_d[None, None, ...]
        H_m_p2 = shift_axis(H, 2, axis=1)
        H_m_m2 = shift_axis(H, -2, axis=1)
        curv_term = (
            jnp.sqrt((m + 1.0) * (m + 2.0)) * H_m_p2
            + (2.0 * m + 1.0) * H
            + jnp.sqrt(m * (m - 1.0)) * H_m_m2
        )
        gradb_term = (
            (l + 1.0) * shift_axis(H, 1, axis=0)
            + (2.0 * l + 1.0) * H
            + l * shift_axis(H, -1, axis=0)
        )
        dG = dG - icv * params.energy_par_coef * curv_term - igb * params.energy_perp_coef * gradb_term

        iky = imag * params.omega_star_scale * cache.ky[:, None, None]
        l4 = cache.l4
        Jl_m1 = shift_axis(cache.Jl, -1, axis=0)
        Jl_p1 = shift_axis(cache.Jl, 1, axis=0)
        tprim = jnp.asarray(params.R_over_LTi)
        fprim = jnp.asarray(params.R_over_Ln)
        drive_m0 = iky * phi * (
            Jl_m1 * (l4 * tprim)
            + cache.Jl * (fprim + 2.0 * l4 * tprim)
            + Jl_p1 * ((l4 + 1.0) * tprim)
        )
        dG = dG.at[:, 0, ...].add(drive_m0)
        if Nm > 2:
            drive_m2 = iky * phi * cache.Jl * (tprim / jnp.sqrt(2.0))
            dG = dG.at[:, 2, ...].add(drive_m2)
    else:
        phi_component = jnp.zeros_like(G).at[:, 0, ...].set(cache.Jl * phi)
        energy_phi = energy_operator(
            phi_component, params.energy_const, params.energy_par_coef, params.energy_perp_coef
        )
        dG = dG + imag * params.omega_d_scale * cache.omega_d[None, None, ...] * (G + energy_phi)

        R_over_Ln = jnp.asarray(params.R_over_Ln)
        eta_i = jnp.where(R_over_Ln == 0.0, 0.0, params.R_over_LTi / R_over_Ln)
        drive_coeffs = diamagnetic_drive_coeffs(
            G.shape[0],
            G.shape[1],
            eta_i,
            params.energy_const,
            params.energy_par_coef,
            params.energy_perp_coef,
        )
        omega_star = params.omega_star_scale * cache.ky[:, None, None] * R_over_Ln
        phi_drive = omega_star * phi
        dG = dG + imag * drive_coeffs[:, :, None, None, None] * cache.Jl[:, None, ...] * phi_drive

    dG = dG - params.nu * cache.lb_lam * H
    dG = dG - params.nu_hyper * cache.hyper_ratio * G
    return dG.astype(out_dtype), phi.astype(out_dtype)


@partial(jax.jit, static_argnames=("steps", "method", "operator"))
def _integrate_linear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    operator: str = "gx",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using cached geometry arrays."""

    if operator == "full":
        operator = "gx"
    if method not in {"euler", "rk2", "rk4", "imex"}:
        raise ValueError("method must be one of {'euler', 'rk2', 'rk4', 'imex'}")

    state_dtype = jnp.result_type(G0, cache.lb_lam, cache.ky, jnp.complex64)
    G0 = jnp.asarray(G0, dtype=state_dtype)
    damping = params.nu * cache.lb_lam + params.nu_hyper * cache.hyper_ratio
    damping = damping.astype(jnp.real(G0).dtype)

    def step(G, _):
        dG, _phi = linear_rhs_cached(G, cache, params, operator=operator)
        if method == "imex":
            dG_explicit = dG + damping * G
            G_new = (G + dt * dG_explicit) / (1.0 + dt * damping)
        elif method == "euler":
            G_new = G + dt * dG
        elif method == "rk2":
            k1 = dG
            k2, _ = linear_rhs_cached(G + 0.5 * dt * k1, cache, params, operator=operator)
            G_new = G + dt * k2
        else:
            k1 = dG
            k2, _ = linear_rhs_cached(G + 0.5 * dt * k1, cache, params, operator=operator)
            k3, _ = linear_rhs_cached(G + 0.5 * dt * k2, cache, params, operator=operator)
            k4, _ = linear_rhs_cached(G + dt * k3, cache, params, operator=operator)
            G_new = G + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        _dG_new, phi_new = linear_rhs_cached(G_new, cache, params, operator=operator)
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
    operator: str = "gx",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using a fixed-step scheme."""

    if operator == "full":
        operator = "gx"
    if cache is None:
        cache = build_linear_cache(grid, geom, params, G0.shape[0], G0.shape[1])
    if method == "semi-implicit":
        method = "imex"
    if method == "implicit":
        shape = G0.shape
        size = 1
        for dim in shape:
            size *= int(dim)
        state_dtype = jnp.result_type(G0, cache.lb_lam, cache.ky, jnp.complex64)
        G = jnp.asarray(G0, dtype=state_dtype)
        dt_val = jnp.asarray(dt, dtype=jnp.real(G).dtype)
        damping = params.nu * cache.lb_lam + params.nu_hyper * cache.hyper_ratio
        damping = damping.astype(jnp.real(G).dtype)
        l = cache.l
        m = cache.m
        diag = jnp.zeros_like(damping, dtype=state_dtype)
        imag = jnp.asarray(1j, dtype=state_dtype)
        if operator == "gx":
            diag = diag - imag * params.tz * params.omega_d_scale * (
                cache.cv_d[None, None, ...] * (2.0 * m + 1.0)
                + cache.gb_d[None, None, ...] * (2.0 * l + 1.0)
            )
            bgrad = cache.bgrad[None, None, None, None, :]
            mirror_diag = params.vth * params.omega_d_scale * (2.0 * l + 1.0) * (2.0 * m + 1.0)
            mirror_weight = 0.2
            diag = diag - mirror_weight * bgrad * mirror_diag
        elif operator == "energy":
            diag = diag + imag * params.omega_d_scale * cache.omega_d[None, None, ...]
        else:
            raise ValueError("operator must be one of {'gx', 'energy'}")
        precond = 1.0 / (1.0 + dt_val * damping - dt_val * diag)
        precond = precond.astype(G.dtype)
        phi_out = []

        def matvec(x_flat: jnp.ndarray) -> jnp.ndarray:
            x = x_flat.reshape(shape)
            dG, _phi = linear_rhs_cached(x, cache, params, operator=operator)
            return (x - dt_val * dG).reshape(size)

        def apply_precond(x_flat: jnp.ndarray) -> jnp.ndarray:
            x = x_flat.reshape(shape)
            return (x * precond).reshape(size)

        for _ in range(steps):
            G_guess = G
            for _iter in range(max(implicit_iters, 0)):
                dG, _phi = linear_rhs_cached(G_guess, cache, params, operator=operator)
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
            _dG, phi = linear_rhs_cached(G, cache, params, operator=operator)
            phi_out.append(phi)
        return G, jnp.stack(phi_out, axis=0)
    return _integrate_linear_cached(G0, cache, params, dt, steps, method=method, operator=operator)
