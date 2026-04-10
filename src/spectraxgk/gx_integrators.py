"""Reference-aligned linear time integrator implemented in JAX."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.laguerre import laggauss

from spectraxgk.diagnostics import SimulationDiagnostics
from spectraxgk.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import (
    LinearCache,
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.terms.assembly import assemble_rhs_cached
from spectraxgk.terms.config import FieldState, TermConfig


@dataclass(frozen=True)
class ExplicitTimeConfig:
    """Explicit time integration configuration."""

    t_max: float
    dt: float
    method: str = "rk4"
    sample_stride: int = 1
    fixed_dt: bool = False
    use_dealias_mask: bool = False
    dt_min: float = 1.0e-7
    dt_max: float | None = None
    cfl: float = 0.9
    cfl_fac: float = 2.82


_SSPX3_ADT = float((1.0 / 6.0) ** (1.0 / 3.0))
_SSPX3_WGTFAC = float((9.0 - 2.0 * (6.0 ** (2.0 / 3.0))) ** 0.5)
_SSPX3_W1 = 0.5 * (_SSPX3_WGTFAC - 1.0)
_SSPX3_W2 = 0.5 * ((6.0 ** (2.0 / 3.0)) - 1.0 - _SSPX3_WGTFAC)
_SSPX3_W3 = (1.0 / _SSPX3_ADT) - 1.0 - _SSPX3_W2 * (_SSPX3_W1 + 1.0)


def _gx_state_mask(cache: LinearCache) -> jnp.ndarray:
    """Return the GX state-space mask applied after each completed step."""

    mask = _gx_growth_mask(cache.ky, cache.kx, cache.dealias_mask)
    ky_zero = jnp.isclose(jnp.asarray(cache.ky), 0.0)
    kx_zero = jnp.isclose(jnp.asarray(cache.kx), 0.0)
    zonal00 = ky_zero[:, None] & kx_zero[None, :]
    return mask & ~zonal00


def _apply_gx_state_mask(state: jnp.ndarray, cache: LinearCache) -> jnp.ndarray:
    """Apply GX's end-of-step mask to a spectral state array."""

    mask = _gx_state_mask(cache).astype(state.real.dtype)[..., None]
    shape = (1,) * (state.ndim - mask.ndim) + mask.shape
    return state * jnp.reshape(mask, shape)


def _gx_growth_mask(
    ky: jnp.ndarray,
    kx: jnp.ndarray,
    dealias_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Return the diagnostic mask used by GX-style growth-rate extraction.

    Single selected nonzonal ``ky`` slices should remain diagnosable even when
    the originating full nonlinear mesh would mark that representative row as
    dealiased away.
    """

    mask = jnp.asarray(dealias_mask, dtype=bool)
    ky_arr = jnp.asarray(ky, dtype=float).reshape(-1)
    if mask.ndim == 2 and int(mask.shape[0]) == 1:
        promote = (~jnp.any(mask)) & jnp.any(jnp.abs(ky_arr) > 0.0)
        mask = jnp.where(promote, jnp.ones_like(mask, dtype=bool), mask)
    ky_zero = jnp.isclose(ky_arr, 0.0)
    kx_zero = jnp.isclose(jnp.asarray(kx, dtype=float).reshape(-1), 0.0)
    zonal00 = ky_zero[:, None] & kx_zero[None, :]
    return mask & ~zonal00


def _gx_zp_from_grid(grid: SpectralGrid) -> float:
    if grid.z.size <= 1:
        return 1.0
    dz = float(np.asarray(grid.z[1] - grid.z[0]))
    extent = float(np.asarray(grid.z[-1] - grid.z[0] + dz))
    return extent / (2.0 * np.pi)


def _gx_k_arrays(grid: SpectralGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx = np.asarray(grid.kx, dtype=float).reshape(-1)
    ky_full = np.asarray(grid.ky, dtype=float).reshape(-1)
    nx = int(kx.size)
    nz = int(grid.z.size)
    zp = _gx_zp_from_grid(grid)
    kz = np.zeros(nz, dtype=float)

    # Preserve actual mode values on sliced ky grids. Reconstructing ky from
    # (Ny, y0) maps a single selected positive ky back to ky=0 and breaks GX's
    # CFL estimate for one-mode linear benchmark runs.
    if ky_full.size == 0:
        ky = np.zeros(0, dtype=float)
    elif grid.ky_mode is not None:
        ky = np.abs(ky_full)
    else:
        nyc = 1 + ky_full.size // 2
        ky = np.abs(ky_full[:nyc])

    for idx in range(nz):
        if idx < nz / 2 + 1:
            kz[idx] = float(idx) / zp
        else:
            kz[idx] = float(idx - nz) / zp
    return kx, ky, kz


def _gx_laguerre_vmax(nl: int) -> float:
    if nl <= 0:
        return 0.0
    nj = max(1, (3 * nl) // 2 - 1)
    roots, _weights = laggauss(nj)
    idx = min(max(nl - 1, 0), roots.size - 1)
    return float(roots[idx])


def _gx_eta_max(tprim: np.ndarray, fprim: np.ndarray) -> float:
    if tprim.size == 0:
        return 0.0
    eta = np.zeros_like(tprim, dtype=float)
    mask = np.abs(fprim) > 0.0
    eta[mask] = tprim[mask] / fprim[mask]
    eta[~mask] = 1.0e6
    return float(np.max(eta))


def _gx_geometry_maxima(
    geom: FluxTubeGeometryLike, theta: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    theta_j = jnp.asarray(theta)
    cv_j, gb_j, cv0_j, gb0_j = geom.drift_coeffs(theta_j)
    bmag_j = geom.bmag(theta_j)
    cv = np.asarray(cv_j, dtype=float)
    gb = np.asarray(gb_j, dtype=float)
    cv0 = np.asarray(cv0_j, dtype=float)
    gb0 = np.asarray(gb0_j, dtype=float)
    bmag = np.asarray(bmag_j, dtype=float)
    bmag_max = float(np.max(np.abs(bmag)))
    cvdrift_max = float(np.max(np.abs(cv)))
    gbdrift_max = float(np.max(np.abs(gb)))
    cvdrift0_max = float(np.max(np.abs(cv0)))
    gbdrift0_max = float(np.max(np.abs(gb0)))
    return bmag_max, cvdrift_max, gbdrift_max, cvdrift0_max, gbdrift0_max, float(geom.gradpar())


def _gx_m0_max_ntft(
    geom: FluxTubeGeometryLike,
    grid: SpectralGrid,
    ky_max: float,
    vpar_max: float,
    muB_max: float,
) -> tuple[float, float, float]:
    theta = np.asarray(grid.z, dtype=float)
    theta_j = jnp.asarray(theta)
    _gds2, gds21_j, gds22_j = geom.metric_coeffs(theta_j)
    gds21 = np.asarray(gds21_j, dtype=float)
    gds22 = np.asarray(gds22_j, dtype=float)
    shat = float(geom.s_hat)
    ftwist = shat * gds21 / gds22
    nz = theta.size
    if nz <= 1:
        _cv_j, _gb_j, cv0_j, gb0_j = geom.drift_coeffs(theta_j)
        return 0.0, float(np.max(np.abs(np.asarray(cv0_j)))), float(
            np.max(np.abs(np.asarray(gb0_j)))
        )
    delta = 0.01313
    x0 = float(grid.x0)
    kxfac = float(grid.kxfac)
    zp = _gx_zp_from_grid(grid)
    mid = nz // 2
    mid_next = min(mid + 1, nz - 1)
    ref_term = (1.0 - delta) * ftwist[mid] + delta * ftwist[mid_next]

    cv_j, gb_j, cv0_j, gb0_j = geom.drift_coeffs(theta_j)
    cv0 = np.asarray(cv0_j, dtype=float)
    gb0 = np.asarray(gb0_j, dtype=float)

    m0_max = 0.0
    cv0_max = float(np.max(np.abs(cv0)))
    gb0_max = float(np.max(np.abs(gb0)))
    m0_omega0 = 0.0
    for idz in range(nz):
        term1 = ftwist[idz] - 2.0 * np.pi * zp * kxfac * shat * np.floor(idz / (1.0 * nz))
        term2 = ftwist[(idz + 1) % nz] - 2.0 * np.pi * zp * kxfac * shat * np.floor(
            (idz + 1) / (1.0 * nz)
        )
        m0 = -np.rint(x0 * ky_max * ((1.0 - delta) * term1 + delta * term2)) + np.rint(
            x0 * ky_max * ref_term
        )
        omega0 = float(m0) * (
            vpar_max * vpar_max * abs(cv0[idz]) + muB_max * abs(gb0[idz])
        )
        if omega0 > m0_omega0:
            m0_omega0 = omega0
            m0_max = abs(float(m0))
            cv0_max = abs(float(cv0[idz]))
            gb0_max = abs(float(gb0[idz]))
    return m0_max, cv0_max, gb0_max


def _gx_linear_omega_max(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    nl: int,
    nm: int,
    *,
    include_diamagnetic_drive: bool = True,
) -> np.ndarray:
    kx, ky, kz = _gx_k_arrays(grid)
    nz = kz.size
    nx = kx.size
    ny = int(grid.ky.shape[0])
    kx_max = float(abs(kx[(nx - 1) // 3])) if nx > 1 else 0.0
    ky_max = float(abs(ky[(ny - 1) // 3])) if ky.size > 0 else 0.0
    kz_max = float(abs(kz[nz // 2])) if nz > 0 else 0.0
    kx_min = float(abs(kx[1])) if nx > 1 else np.inf
    ky_min = float(abs(ky[1])) if ky.size > 1 else np.inf
    kperp_min = float(min(kx_min, ky_min)) if np.isfinite(min(kx_min, ky_min)) else 0.0

    tprim = np.atleast_1d(np.asarray(params.R_over_LTi, dtype=float))
    fprim = np.atleast_1d(np.asarray(params.R_over_Ln, dtype=float))
    tz = np.atleast_1d(np.asarray(params.tz, dtype=float))
    vth = np.atleast_1d(np.asarray(params.vth, dtype=float))
    temp = np.atleast_1d(np.asarray(params.temp, dtype=float))
    dens = np.atleast_1d(np.asarray(params.density, dtype=float))
    charge = np.atleast_1d(np.asarray(params.charge_sign, dtype=float))

    tzmax = float(np.max(np.abs(tz))) if tz.size else 0.0
    vtmax = float(np.max(np.abs(vth))) if vth.size else 0.0
    vtmin = float(np.min(np.abs(vth))) if vth.size else 1.0
    etamax = _gx_eta_max(tprim, fprim)
    vpar_max = 2.0 * float(np.sqrt(max(nm, 1)))
    muB_max = _gx_laguerre_vmax(nl)
    bmag_max, cvdrift_max, gbdrift_max, cvdrift0_max, gbdrift0_max, gradpar = (
        _gx_geometry_maxima(geom, np.asarray(grid.z, dtype=float))
    )

    shat = float(geom.s_hat)
    non_twist = bool(getattr(grid, "non_twist", False))
    m0_max = 0.0
    if non_twist and abs(shat) > 0.0 and ky.size > 0:
        ky_m0 = float(ky[-1])
        m0_max, cvdrift0_max, gbdrift0_max = _gx_m0_max_ntft(
            geom,
            grid,
            ky_m0,
            vpar_max=vpar_max,
            muB_max=muB_max,
        )

    omega_max = np.zeros(3, dtype=float)
    if abs(shat) == 0.0:
        omega_max[0] = tzmax * kx_max * (
            vpar_max * vpar_max * abs(cvdrift0_max) + muB_max * abs(gbdrift0_max)
        )
    else:
        if non_twist:
            omega_max[0] = tzmax * (kx_max + m0_max / float(grid.x0)) * (
                vpar_max * vpar_max * abs(cvdrift0_max) + muB_max * abs(gbdrift0_max)
            )
        else:
            omega_max[0] = tzmax * kx_max / abs(shat) * (
                vpar_max * vpar_max * abs(cvdrift0_max) + muB_max * abs(gbdrift0_max)
            )

    omega_max[1] = tzmax * ky_max * (
        vpar_max * vpar_max * cvdrift_max + muB_max * gbdrift_max
    )
    if include_diamagnetic_drive and etamax < 1.0e5:
        omega_max[1] = omega_max[1] + ky_max * (
            1.0 + etamax * (vpar_max * vpar_max / 2.0 + muB_max - 1.5)
        )

    beta = float(params.beta)
    nspec_in = int(max(charge.size, 1))
    if charge.size:
        neg = charge < 0.0
        if np.any(neg):
            ne = float(dens[neg][0])
            Te = float(temp[neg][0])
        else:
            ne = float(dens[0])
            Te = float(temp[0])
    else:
        ne = 1.0
        Te = 1.0
    nte = ne * Te
    mime = (vtmax * vtmax) / (vtmin * vtmin) if vtmin > 0.0 else 0.0
    kperprho2 = kperp_min * kperp_min / (bmag_max * bmag_max) if bmag_max > 0.0 else 0.0
    if nspec_in > 1:
        denom = beta * nte / 2.0 * mime + kperprho2
        guard = 1.0 / np.sqrt(denom) if denom > 0.0 else 0.0
    else:
        guard = 0.0
    omega_max[2] = vtmax * kz_max * abs(gradpar) * max(vpar_max, guard)

    return omega_max


def _gx_midplane_index(nz: int) -> int:
    if nz <= 1:
        return 0
    idx = nz // 2 + 1
    return min(idx, nz - 1)


def _gx_growth_rate_step(
    phi_now: jnp.ndarray,
    phi_prev: jnp.ndarray,
    dt: float | jax.Array,
    *,
    z_index: int,
    mask: jnp.ndarray,
    mode_method: str = "z_index",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """GX instantaneous growth rates from phi ratios at the midplane."""

    if mode_method == "z_index":
        phi_now_z = phi_now[..., z_index]
        phi_prev_z = phi_prev[..., z_index]
    elif mode_method == "max":
        now_idx = jnp.argmax(jnp.abs(phi_now), axis=-1, keepdims=True)
        prev_idx = jnp.argmax(jnp.abs(phi_prev), axis=-1, keepdims=True)
        phi_now_z = jnp.take_along_axis(phi_now, now_idx, axis=-1)[..., 0]
        phi_prev_z = jnp.take_along_axis(phi_prev, prev_idx, axis=-1)[..., 0]
    else:
        raise ValueError("mode_method must be 'z_index' or 'max'")
    # Match GX growthRates kernel logic: require non-zero real and imaginary
    # parts of phi at the current step only.
    valid = (jnp.abs(jnp.real(phi_now_z)) > 0.0) & (jnp.abs(jnp.imag(phi_now_z)) > 0.0)
    ratio = jnp.where(phi_prev_z != 0.0, phi_now_z / phi_prev_z, 0.0 + 0.0j)
    log_amp = jnp.log(jnp.abs(ratio))
    phase = jnp.angle(ratio)
    gamma = jnp.where(mask & valid, log_amp / dt, 0.0)
    omega = jnp.where(mask & valid, -phase / dt, 0.0)
    return gamma, omega


def _gx_term_config(terms: LinearTerms | None) -> TermConfig:
    return linear_terms_to_term_config(terms)


def _rk4_step(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt: float,
) -> tuple[jnp.ndarray, FieldState]:
    """Single GX-style RK4 step for linear dynamics."""

    return _linear_explicit_step(G, cache, params, term_cfg, dt, method="rk4")


def _rk3_gx_step(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt: float,
) -> tuple[jnp.ndarray, FieldState]:
    """Single GX-style RK3/Heun step for linear dynamics."""

    return _linear_explicit_step(G, cache, params, term_cfg, dt, method="rk3")


def _linear_explicit_step(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    dt: float,
    *,
    method: str,
) -> tuple[jnp.ndarray, FieldState]:
    """Single explicit linear step matching GX-style staged schemes."""

    dt_val = jnp.asarray(dt)
    method_key = method.strip().lower()

    def rhs(state: jnp.ndarray) -> jnp.ndarray:
        dG, _fields = assemble_rhs_cached(state, cache, params, terms=term_cfg, dt=dt_val)
        return dG

    k1 = rhs(G)
    if method_key == "euler":
        G_next = G + dt_val * k1
    elif method_key == "rk2":
        G_half = G + 0.5 * dt_val * k1
        k2 = rhs(G_half)
        G_next = G + dt_val * k2
    elif method_key == "rk3_classic":
        G1 = G + dt_val * k1
        k2 = rhs(G1)
        G2 = 0.75 * G + 0.25 * (G1 + dt_val * k2)
        k3 = rhs(G2)
        G_next = (1.0 / 3.0) * G + (2.0 / 3.0) * (G2 + dt_val * k3)
    elif method_key in {"rk3", "rk3_gx"}:
        G1 = G + (dt_val / 3.0) * k1
        k2 = rhs(G1)
        G2 = G + (2.0 * dt_val / 3.0) * k2
        k3 = rhs(G2)
        G3 = G + 0.75 * dt_val * k3
        G_next = G3 + 0.25 * dt_val * k1
    elif method_key == "rk4":
        k2 = rhs(G + 0.5 * dt_val * k1)
        k3 = rhs(G + 0.5 * dt_val * k2)
        k4 = rhs(G + dt_val * k3)
        G_next = G + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    elif method_key == "sspx3":
        def _sspx3_euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
            dG_state = rhs(G_state)
            return G_state + (_SSPX3_ADT * dt_val) * dG_state

        G1 = _sspx3_euler_step(G)
        G2_euler = _sspx3_euler_step(G1)
        G2 = (1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler
        G3 = _sspx3_euler_step(G2)
        G_next = (1.0 - _SSPX3_W2 - _SSPX3_W3) * G + _SSPX3_W3 * G1 + (_SSPX3_W2 - 1.0) * G2 + G3
    elif method_key == "k10":
        def _k10_euler_step(G_state: jnp.ndarray) -> jnp.ndarray:
            dG_state = rhs(G_state)
            return G_state + (dt_val / 6.0) * dG_state

        G_q1 = G
        G_q2 = G
        for _ in range(5):
            G_q1 = _k10_euler_step(G_q1)
        G_q2 = 0.04 * G_q2 + 0.36 * G_q1
        G_q1 = 15.0 * G_q2 - 5.0 * G_q1
        for _ in range(4):
            G_q1 = _k10_euler_step(G_q1)
        dG_final = rhs(G_q1)
        G_next = G_q2 + 0.6 * G_q1 + 0.1 * dt_val * dG_final
    else:
        raise ValueError(
            "GX linear method must be one of {'euler', 'rk2', 'rk3', 'rk3_classic', 'rk3_gx', 'rk4', 'k10', 'sspx3'}"
        )

    # GX masks G only after the full explicit step, before the next field solve.
    G_next = _apply_gx_state_mask(jnp.asarray(G_next), cache)

    # fields at the end of step
    _, fields = assemble_rhs_cached(G_next, cache, params, terms=term_cfg, dt=dt_val)
    return G_next, fields


def integrate_linear_gx(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    cache: LinearCache,
    params: LinearParams,
    geom: FluxTubeGeometryLike,
    time_cfg: ExplicitTimeConfig,
    terms: LinearTerms | None = None,
    *,
    mode_method: str = "z_index",
    z_index: int | None = None,
    jit: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GX-style RK4 integrator with GX growth-rate diagnostics."""

    if mode_method not in {"z_index", "max"}:
        raise ValueError("mode_method must be 'z_index' or 'max'")

    method = time_cfg.method.strip().lower()
    if method not in {"euler", "rk2", "rk3", "rk3_classic", "rk3_gx", "rk4", "k10", "sspx3"}:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_classic', 'rk3_gx', 'rk4', 'k10', 'sspx3'}"
        )
    term_cfg = _gx_term_config(terms)
    t_max = float(time_cfg.t_max)
    dt = float(time_cfg.dt)
    dt_min = float(time_cfg.dt_min)
    # GX default behavior: when dt_max is unset, dt_max == dt.
    dt_max = float(time_cfg.dt_max) if time_cfg.dt_max is not None else dt
    sample_stride = int(max(time_cfg.sample_stride, 1))

    z_idx = _gx_midplane_index(grid.z.size) if z_index is None else int(z_index)
    mask = _gx_growth_mask(grid.ky, grid.kx, grid.dealias_mask)

    G = jnp.asarray(G0)
    t = 0.0
    step = 0

    # compute initial fields for growth-rate ratio
    _, fields0 = assemble_rhs_cached(G, cache, params, terms=term_cfg, dt=dt)
    phi_prev = fields0.phi

    omega_max = _gx_linear_omega_max(grid, geom, params, G.shape[-5], G.shape[-4])
    wmax = float(np.sum(omega_max))
    if not time_cfg.fixed_dt and wmax > 0.0:
        dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
        dt = min(max(dt_guess, dt_min), dt_max)

    ts: list[float] = []
    phi_list: list[np.ndarray] = []
    gamma_list: list[np.ndarray] = []
    omega_list: list[np.ndarray] = []
    dt_list: list[float] = []

    def _step(G_state, cache_state, params_state, term_cfg_state, dt_state):
        return _linear_explicit_step(
            G_state, cache_state, params_state, term_cfg_state, dt_state, method=method
        )

    stepper = _step
    if jit:
        stepper = jax.jit(_step, donate_argnums=(0,))

    while t < t_max - 1.0e-12:
        if not time_cfg.fixed_dt and wmax > 0.0:
            dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
            dt = min(max(dt_guess, dt_min), dt_max)

        G, fields = stepper(G, cache, params, term_cfg, dt)
        phi = fields.phi
        step += 1
        t += dt

        if step % sample_stride == 0 or t >= t_max:
            gamma, omega = _gx_growth_rate_step(
                phi,
                phi_prev,
                dt,
                z_index=z_idx,
                mask=mask,
                mode_method=mode_method,
            )
            ts.append(t)
            phi_list.append(np.asarray(phi))
            gamma_list.append(np.asarray(gamma))
            omega_list.append(np.asarray(omega))
        phi_prev = phi

    return (
        np.asarray(ts),
        np.asarray(phi_list),
        np.asarray(gamma_list),
        np.asarray(omega_list),
    )


def integrate_linear_gx_diagnostics(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    cache: LinearCache,
    params: LinearParams,
    geom: FluxTubeGeometryLike,
    time_cfg: ExplicitTimeConfig,
    terms: LinearTerms | None = None,
    *,
    mode_method: str = "z_index",
    z_index: int | None = None,
    jit: bool = True,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, SimulationDiagnostics]:
    """GX-style RK4 integrator with GX growth-rate + energy/flux diagnostics."""

    if show_progress:
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel.fit("[bold blue]SPECTRAX-GK[/bold blue] | [bold green]GX Linear Simulation Started[/bold green]", border_style="blue"))

    from spectraxgk.diagnostics import (
        SimulationDiagnostics,
        gx_energy_total,
        gx_heat_flux,
        gx_particle_flux,
        gx_volume_factors,
        gx_Wapar,
        gx_Wg,
        gx_Wphi,
    )

    if mode_method not in {"z_index", "max"}:
        raise ValueError("mode_method must be 'z_index' or 'max'")

    if terms is None:
        terms = LinearTerms()
    method = time_cfg.method.strip().lower()
    if method not in {"euler", "rk2", "rk3", "rk3_classic", "rk3_gx", "rk4", "k10", "sspx3"}:
        raise ValueError(
            "method must be one of {'euler', 'rk2', 'rk3', 'rk3_classic', 'rk3_gx', 'rk4', 'k10', 'sspx3'}"
        )
    term_cfg = _gx_term_config(terms)
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)

    t_max = float(time_cfg.t_max)
    dt = float(time_cfg.dt)
    dt_min = float(time_cfg.dt_min)
    # GX default behavior: when dt_max is unset, dt_max == dt.
    dt_max = float(time_cfg.dt_max) if time_cfg.dt_max is not None else dt
    sample_stride = int(max(time_cfg.sample_stride, 1))

    z_idx = _gx_midplane_index(grid.z.size) if z_index is None else int(z_index)
    mask = _gx_growth_mask(grid.ky, grid.kx, grid.dealias_mask)

    G = jnp.asarray(G0)
    t = 0.0
    step = 0

    _, fields0 = assemble_rhs_cached(G, cache, params, terms=term_cfg, dt=dt)
    phi_prev = fields0.phi

    omega_max = _gx_linear_omega_max(grid, geom_eff, params, G.shape[-5], G.shape[-4])
    wmax = float(np.sum(omega_max))
    if not time_cfg.fixed_dt and wmax > 0.0:
        dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
        dt = min(max(dt_guess, dt_min), dt_max)

    ts: list[float] = []
    phi_list: list[np.ndarray] = []
    gamma_list: list[np.ndarray] = []
    omega_list: list[np.ndarray] = []
    dt_list: list[float] = []
    Wg_list: list[float] = []
    Wphi_list: list[float] = []
    Wapar_list: list[float] = []
    heat_list: list[float] = []
    pflux_list: list[float] = []

    vol_fac, flux_fac = gx_volume_factors(geom_eff, grid)
    use_dealias = bool(time_cfg.use_dealias_mask)

    def _step(G_state, cache_state, params_state, term_cfg_state, dt_state):
        return _linear_explicit_step(
            G_state, cache_state, params_state, term_cfg_state, dt_state, method=method
        )

    stepper = _step
    if jit:
        stepper = jax.jit(_step, donate_argnums=(0,))

    while t < t_max - 1.0e-12:
        if not time_cfg.fixed_dt and wmax > 0.0:
            dt_guess = float(time_cfg.cfl_fac) * float(time_cfg.cfl) / wmax
            dt = min(max(dt_guess, dt_min), dt_max)

        G, fields = stepper(G, cache, params, term_cfg, dt)
        step += 1
        t += dt

        if step % sample_stride == 0 or t >= t_max:
            phi = fields.phi
            apar = fields.apar if fields.apar is not None else jnp.zeros_like(phi)
            bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(phi)
            gamma, omega = _gx_growth_rate_step(
                phi,
                phi_prev,
                dt,
                z_index=z_idx,
                mask=mask,
                mode_method=mode_method,
            )
            ts.append(t)
            dt_list.append(float(dt))
            phi_list.append(np.asarray(phi))
            gamma_list.append(np.asarray(gamma))
            omega_list.append(np.asarray(omega))

            Wg_val = gx_Wg(G, grid, params, vol_fac, use_dealias=use_dealias)
            Wphi_val = gx_Wphi(
                phi,
                cache,
                params,
                vol_fac,
                use_dealias=use_dealias,
            )
            Wapar_val = gx_Wapar(apar, cache, vol_fac, use_dealias=use_dealias)
            heat_val = gx_heat_flux(
                G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=use_dealias
            )
            pflux_val = gx_particle_flux(
                G, phi, apar, bpar, cache, grid, params, flux_fac, use_dealias=use_dealias
            )

            Wg_list.append(float(Wg_val))
            Wphi_list.append(float(Wphi_val))
            Wapar_list.append(float(Wapar_val))
            heat_list.append(float(heat_val))
            pflux_list.append(float(pflux_val))

            if show_progress:
                from spectraxgk.utils.callbacks import print_callback
                # We can't easily use jax.debug.callback here because we are in a python loop
                # but we can just use the internal logic of print_callback if it's pure python
                # or just use rich directly.
                from rich.table import Table
                from rich import box
                pct = (t / t_max) * 100
                table = Table(box=box.HORIZONTALS, show_header=(step <= sample_stride), header_style="bold magenta")
                if step <= sample_stride:
                    table.add_column("Progress", justify="right", style="cyan")
                    table.add_column("Step", justify="right", style="green")
                    table.add_column("Time", justify="right", style="yellow")
                    table.add_column("Wg", justify="right")
                    table.add_column("Wphi", justify="right")
                    table.add_column("Heat", justify="right")
                table.add_row(
                    f"{pct:>3.0f}%",
                    str(step),
                    f"{float(t):.2f}",
                    f"{float(Wg_val):.4e}",
                    f"{float(Wphi_val):.4e}",
                    f"{float(heat_val):.4e}"
                )
                console.print(table)

        phi_prev = fields.phi

    if show_progress:
        from rich.panel import Panel
        console.print(Panel.fit("[bold green]Simulation Complete![/bold green]", border_style="green"))

    diag = SimulationDiagnostics(
        t=np.asarray(ts),
        dt_t=np.asarray(dt_list),
        dt_mean=np.asarray(np.mean(dt_list)) if dt_list else np.asarray(0.0),
        gamma_t=np.asarray(gamma_list),
        omega_t=np.asarray(omega_list),
        Wg_t=np.asarray(Wg_list),
        Wphi_t=np.asarray(Wphi_list),
        Wapar_t=np.asarray(Wapar_list),
        heat_flux_t=np.asarray(heat_list),
        particle_flux_t=np.asarray(pflux_list),
        energy_t=np.asarray(
            gx_energy_total(np.asarray(Wg_list), np.asarray(Wphi_list), np.asarray(Wapar_list))
        ),
    )
    return (
        np.asarray(ts),
        np.asarray(phi_list),
        np.asarray(gamma_list),
        np.asarray(omega_list),
        diag,
    )


# Compatibility alias retained while callers migrate off the older name.
GXTimeConfig = ExplicitTimeConfig
