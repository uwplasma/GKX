"""CFL and linear-frequency bound policy for explicit time integration."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.laguerre import laggauss
import jax.numpy as jnp

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.operators.linear.params import LinearParams

__all__ = [
    "_cfl_wavenumber_arrays",
    "_geometry_frequency_maxima",
    "_gradient_ratio_max",
    "_laguerre_velocity_max",
    "_linear_frequency_bound",
    "_non_twist_shift_frequency_max",
    "_parallel_periods_from_grid",
]


def _parallel_periods_from_grid(grid: SpectralGrid) -> float:
    if grid.z.size <= 1:
        return 1.0
    dz = float(np.asarray(grid.z[1] - grid.z[0]))
    extent = float(np.asarray(grid.z[-1] - grid.z[0] + dz))
    return extent / (2.0 * np.pi)


def _cfl_wavenumber_arrays(
    grid: SpectralGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx = np.asarray(grid.kx, dtype=float).reshape(-1)
    ky_full = np.asarray(grid.ky, dtype=float).reshape(-1)
    nz = int(grid.z.size)
    zp = _parallel_periods_from_grid(grid)
    kz = np.zeros(nz, dtype=float)

    # Preserve actual mode values on sliced ky grids. Reconstructing ky from
    # (Ny, y0) maps a single selected positive ky back to ky=0 and breaks the
    # one-mode CFL estimate used by linear benchmark runs.
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


def _laguerre_velocity_max(nl: int) -> float:
    if nl <= 0:
        return 0.0
    nj = max(1, (3 * nl) // 2 - 1)
    roots, _weights = laggauss(nj)
    idx = min(max(nl - 1, 0), roots.size - 1)
    return float(roots[idx])


def _gradient_ratio_max(tprim: np.ndarray, fprim: np.ndarray) -> float:
    if tprim.size == 0:
        return 0.0
    eta = np.zeros_like(tprim, dtype=float)
    mask = np.abs(fprim) > 0.0
    eta[mask] = tprim[mask] / fprim[mask]
    eta[~mask] = 1.0e6
    return float(np.max(eta))


def _geometry_frequency_maxima(
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
    return (
        bmag_max,
        cvdrift_max,
        gbdrift_max,
        cvdrift0_max,
        gbdrift0_max,
        float(geom.gradpar()),
    )


def _non_twist_shift_frequency_max(
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
        return (
            0.0,
            float(np.max(np.abs(np.asarray(cv0_j)))),
            float(np.max(np.abs(np.asarray(gb0_j)))),
        )
    delta = 0.01313
    x0 = float(grid.x0)
    kxfac = float(grid.kxfac)
    zp = _parallel_periods_from_grid(grid)
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
        term1 = ftwist[idz] - 2.0 * np.pi * zp * kxfac * shat * np.floor(
            idz / (1.0 * nz)
        )
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


def _linear_frequency_bound(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    nl: int,
    nm: int,
    *,
    include_diamagnetic_drive: bool = True,
) -> np.ndarray:
    kx, ky, kz = _cfl_wavenumber_arrays(grid)
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
    etamax = _gradient_ratio_max(tprim, fprim)
    vpar_max = 2.0 * float(np.sqrt(max(nm, 1)))
    muB_max = _laguerre_velocity_max(nl)
    bmag_max, cvdrift_max, gbdrift_max, cvdrift0_max, gbdrift0_max, gradpar = (
        _geometry_frequency_maxima(geom, np.asarray(grid.z, dtype=float))
    )

    shat = float(geom.s_hat)
    non_twist = bool(getattr(grid, "non_twist", False))
    m0_max = 0.0
    if non_twist and abs(shat) > 0.0 and ky.size > 0:
        ky_m0 = float(ky[-1])
        m0_max, cvdrift0_max, gbdrift0_max = _non_twist_shift_frequency_max(
            geom,
            grid,
            ky_m0,
            vpar_max=vpar_max,
            muB_max=muB_max,
        )

    omega_max = np.zeros(3, dtype=float)
    if abs(shat) == 0.0:
        omega_max[0] = (
            tzmax
            * kx_max
            * (vpar_max * vpar_max * abs(cvdrift0_max) + muB_max * abs(gbdrift0_max))
        )
    else:
        if non_twist:
            omega_max[0] = (
                tzmax
                * (kx_max + m0_max / float(grid.x0))
                * (
                    vpar_max * vpar_max * abs(cvdrift0_max)
                    + muB_max * abs(gbdrift0_max)
                )
            )
        else:
            omega_max[0] = (
                tzmax
                * kx_max
                / abs(shat)
                * (
                    vpar_max * vpar_max * abs(cvdrift0_max)
                    + muB_max * abs(gbdrift0_max)
                )
            )

    omega_max[1] = (
        tzmax * ky_max * (vpar_max * vpar_max * cvdrift_max + muB_max * gbdrift_max)
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


