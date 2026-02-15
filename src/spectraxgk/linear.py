"""Linear electrostatic gyrokinetic building blocks (Hermite-Laguerre)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

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


def grad_z_periodic(f: jnp.ndarray, dz: float) -> jnp.ndarray:
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
    omega_d: jnp.ndarray
    mask0: jnp.ndarray
    dz: jnp.ndarray
    ky: jnp.ndarray
    lb_lam: jnp.ndarray

    def tree_flatten(self):
        children = (self.Jl, self.omega_d, self.mask0, self.dz, self.ky, self.lb_lam)
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
    b = compute_b(grid, geom, params.rho)
    Jl = J_l_all(b, l_max=Nl - 1)
    omega_d = geom.omega_d(grid.kx, grid.ky, grid.z)
    mask0 = (grid.ky == 0.0)[:, None, None] & (grid.kx == 0.0)[None, :, None]
    lb_lam = lenard_bernstein_eigenvalues(Nl, Nm, params.nu_hermite, params.nu_laguerre)[
        :, :, None, None, None
    ]
    return LinearCache(Jl=Jl, omega_d=omega_d, mask0=mask0, dz=dz, ky=grid.ky, lb_lam=lb_lam)


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


def build_H(G: jnp.ndarray, Jl: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """Map G -> H = G + J_l(b) * phi * delta_{m0}."""

    return G.at[:, 0, ...].add(Jl * phi)


def streaming_term(H: jnp.ndarray, dz: float, vth: float) -> jnp.ndarray:
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
    dz = grid.z[1] - grid.z[0]
    b = compute_b(grid, geom, params.rho)
    Jl = J_l_all(b, l_max=G.shape[0] - 1)
    phi = quasineutrality_phi(G, Jl, params.tau_e)
    mask0 = (grid.ky == 0.0)[:, None, None] & (grid.kx == 0.0)[None, :, None]
    phi = jnp.where(mask0, 0.0, phi)
    H = build_H(G, Jl, phi)
    stream = streaming_term(H, dz, params.vth)
    dG = -params.kpar_scale * stream

    omega_d = geom.omega_d(grid.kx, grid.ky, grid.z)
    phi_component = jnp.zeros_like(G).at[:, 0, ...].set(Jl * phi)
    energy_phi = energy_operator(
        phi_component, params.energy_const, params.energy_par_coef, params.energy_perp_coef
    )
    dG = dG + 1j * params.omega_d_scale * omega_d[None, None, ...] * (G + energy_phi)

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
    omega_star = params.omega_star_scale * grid.ky[:, None, None] * R_over_Ln
    phi_drive = omega_star * phi
    dG = dG + 1j * drive_coeffs[:, :, None, None, None] * Jl[:, None, ...] * phi_drive
    lb_lam = lenard_bernstein_eigenvalues(G.shape[0], G.shape[1], params.nu_hermite, params.nu_laguerre)[
        :, :, None, None, None
    ]
    dG = dG - params.nu * lb_lam * G
    l = jnp.arange(G.shape[0])[:, None, None, None, None]
    m = jnp.arange(G.shape[1])[None, :, None, None, None]
    l_norm = jnp.maximum(G.shape[0] - 1, 1)
    m_norm = jnp.maximum(G.shape[1] - 1, 1)
    ratio = (l / l_norm) ** params.p_hyper + (m / m_norm) ** params.p_hyper
    dG = dG - params.nu_hyper * ratio * G
    return dG, phi


def linear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the linear RHS using precomputed geometry arrays."""

    if G.ndim != 5:
        raise ValueError("G must have shape (Nl, Nm, Ny, Nx, Nz)")
    phi = quasineutrality_phi(G, cache.Jl, params.tau_e)
    phi = jnp.where(cache.mask0, 0.0, phi)
    H = build_H(G, cache.Jl, phi)
    stream = streaming_term(H, cache.dz, params.vth)
    dG = -params.kpar_scale * stream

    phi_component = jnp.zeros_like(G).at[:, 0, ...].set(cache.Jl * phi)
    energy_phi = energy_operator(
        phi_component, params.energy_const, params.energy_par_coef, params.energy_perp_coef
    )
    dG = dG + 1j * params.omega_d_scale * cache.omega_d[None, None, ...] * (G + energy_phi)

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
    dG = dG + 1j * drive_coeffs[:, :, None, None, None] * cache.Jl[:, None, ...] * phi_drive
    dG = dG - params.nu * cache.lb_lam * G
    l = jnp.arange(G.shape[0])[:, None, None, None, None]
    m = jnp.arange(G.shape[1])[None, :, None, None, None]
    l_norm = jnp.maximum(G.shape[0] - 1, 1)
    m_norm = jnp.maximum(G.shape[1] - 1, 1)
    ratio = (l / l_norm) ** params.p_hyper + (m / m_norm) ** params.p_hyper
    dG = dG - params.nu_hyper * ratio * G
    return dG, phi


@partial(jax.jit, static_argnames=("steps", "method"))
def _integrate_linear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using cached geometry arrays."""

    if method not in {"euler", "rk2", "rk4"}:
        raise ValueError("method must be one of {'euler', 'rk2', 'rk4'}")

    G0 = jnp.asarray(G0, dtype=jnp.complex64)

    def step(G, _):
        dG, _phi = linear_rhs_cached(G, cache, params)
        if method == "euler":
            G_new = G + dt * dG
        elif method == "rk2":
            k1 = dG
            k2, _ = linear_rhs_cached(G + 0.5 * dt * k1, cache, params)
            G_new = G + dt * k2
        else:
            k1 = dG
            k2, _ = linear_rhs_cached(G + 0.5 * dt * k1, cache, params)
            k3, _ = linear_rhs_cached(G + 0.5 * dt * k2, cache, params)
            k4, _ = linear_rhs_cached(G + dt * k3, cache, params)
            G_new = G + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        _dG_new, phi_new = linear_rhs_cached(G_new, cache, params)
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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Time integrate the linear system using a fixed-step explicit scheme."""

    if cache is None:
        cache = build_linear_cache(grid, geom, params, G0.shape[0], G0.shape[1])
    return _integrate_linear_cached(G0, cache, params, dt, steps, method=method)
