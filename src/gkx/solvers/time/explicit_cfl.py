"""CFL and linear-frequency bound policy for explicit time integration."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any

import numpy as np
from numpy.polynomial.laguerre import laggauss
import jax
import jax.numpy as jnp

from gkx.core_grid import SpectralGrid
from gkx.geometry import FluxTubeGeometryLike
from gkx.operators.linear.params import LinearParams

# Ratio of dt to the explicit CFL bound above which a fixed-step nonlinear run
# is warned about. Not 1.0: the bound is deliberately conservative -- it takes
# maxima over the whole grid and carries the 0.9 cfl safety factor -- so decks
# sit above it and still integrate. Measured on the shipped decks: 1.33x runs
# clean over 500 steps, 1.60x (the W7-X zonal benchmark) goes non-finite at
# t=5.65 of 60, and 2.77x (the ETG nonlinear example) at t=0.021 of 0.5. 1.5
# sits in that gap, so it fires on both known failures and on neither known
# good deck. The linear path's equivalent warning uses 1.0 and predates this
# measurement; the two are not reconciled.
FIXED_DT_CFL_WARN_RATIO = 1.5

__all__ = [
    "FIXED_DT_CFL_WARN_RATIO",
    "_cfl_wavenumber_arrays",
    "_geometry_frequency_maxima",
    "_gradient_ratio_max",
    "_laguerre_velocity_max",
    "_linear_frequency_bound",
    "_non_twist_shift_frequency_max",
    "_parallel_periods_from_grid",
]


@dataclass(frozen=True)
class _CFLGridBounds:
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    kx_max: float
    ky_max: float
    kz_max: float
    kperp_min: float


@dataclass(frozen=True)
class _SpeciesFrequencyScales:
    tprim: np.ndarray
    fprim: np.ndarray
    temp: np.ndarray
    dens: np.ndarray
    charge: np.ndarray
    tzmax: float
    vtmax: float
    vtmin: float
    etamax: float
    vpar_max: float
    muB_max: float


@dataclass(frozen=True)
class _GeometryFrequencyScales:
    bmag_max: float
    cvdrift_max: float
    gbdrift_max: float
    cvdrift0_max: float
    gbdrift0_max: float
    gradpar: float
    m0_max: float


def _parallel_periods_from_grid(grid: SpectralGrid) -> float:
    """Return the parallel extent of the flux tube in 2*pi periods.

    Take the spacing off the host buffer rather than out of ``grid.z`` by
    ``jnp`` arithmetic. Inside a trace every ``jnp`` call stages out, so
    ``grid.z[1] - grid.z[0]`` is a tracer even when ``grid.z`` is the concrete
    axis built from the run's own config, and reading it back on the host then
    fails. Nothing here is differentiable -- the number of periods is grid
    topology -- so this refused every adaptive-dt run under ``jit`` for a
    quantity that was never traced.
    """

    if grid.z.size <= 1:
        return 1.0
    z_host = np.asarray(grid.z, dtype=float)
    dz = float(z_host[1] - z_host[0])
    return float(z_host[-1] - z_host[0] + dz) / (2.0 * np.pi)


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


_TRACED_CFL_INPUT_MESSAGE = (
    "the adaptive time step cannot read a traced {name}: the CFL bound is a "
    "host scalar that fixes dt_max before the scan is staged, so it needs the "
    "value and not a promise of one. Run with fixed_dt=True, where the step is "
    "the dt you pass and the trajectory is still exact and still "
    "differentiable, or keep {name} concrete. An analytic geometry sampled "
    "inside the trace arrives here traced even when nothing physical is: "
    "sample it once outside with ensure_flux_tube_geometry_data and pass the "
    "sampled geometry in."
)


def _host_cfl_value(value: Any, name: str) -> np.ndarray:
    """Return host samples of one CFL input, naming it if it is traced."""

    if isinstance(value, jax.core.Tracer):
        raise ValueError(_TRACED_CFL_INPUT_MESSAGE.format(name=name))
    return np.asarray(value, dtype=float)


def _geometry_frequency_maxima(
    geom: FluxTubeGeometryLike, theta: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    # Sample the geometry as a compile-time constant. ``theta`` is the host
    # axis and an analytic geometry evaluates it with ``jnp``, which stages out
    # inside a trace; the maxima below then could not be read back, so an
    # adaptive-dt run refused under ``jit`` for a geometry that was never
    # traced. Genuinely traced geometry stays traced here and is named below.
    with jax.ensure_compile_time_eval():
        theta_j = jnp.asarray(theta)
        cv_j, gb_j, cv0_j, gb0_j = geom.drift_coeffs(theta_j)
        bmag_j = geom.bmag(theta_j)
        gradpar_value = geom.gradpar()
    cv = _host_cfl_value(cv_j, "curvature drift")
    gb = _host_cfl_value(gb_j, "grad-B drift")
    cv0 = _host_cfl_value(cv0_j, "radial curvature drift")
    gb0 = _host_cfl_value(gb0_j, "radial grad-B drift")
    bmag = _host_cfl_value(bmag_j, "bmag")
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
        float(_host_cfl_value(gradpar_value, "gradpar")),
    )


def _non_twist_shift_frequency_max(
    geom: FluxTubeGeometryLike,
    grid: SpectralGrid,
    ky_max: float,
    vpar_max: float,
    muB_max: float,
) -> tuple[float, float, float]:
    theta = np.asarray(grid.z, dtype=float)
    # Same compile-time sampling as _geometry_frequency_maxima: the metric on a
    # host theta axis is a constant, not a stage-out.
    with jax.ensure_compile_time_eval():
        theta_j = jnp.asarray(theta)
        _gds2, gds21_j, gds22_j = geom.metric_coeffs(theta_j)
        _cv_j, _gb_j, cv0_j, gb0_j = geom.drift_coeffs(theta_j)
    gds21 = _host_cfl_value(gds21_j, "gds21")
    gds22 = _host_cfl_value(gds22_j, "gds22")
    shat = float(_host_cfl_value(geom.s_hat, "s_hat"))
    ftwist = shat * gds21 / gds22
    nz = theta.size
    if nz <= 1:
        return (
            0.0,
            float(np.max(np.abs(_host_cfl_value(cv0_j, "radial curvature drift")))),
            float(np.max(np.abs(_host_cfl_value(gb0_j, "radial grad-B drift")))),
        )
    delta = 0.01313
    x0 = float(grid.x0)
    kxfac = float(grid.kxfac)
    zp = _parallel_periods_from_grid(grid)
    mid = nz // 2
    mid_next = min(mid + 1, nz - 1)
    ref_term = (1.0 - delta) * ftwist[mid] + delta * ftwist[mid_next]

    cv0 = _host_cfl_value(cv0_j, "radial curvature drift")
    gb0 = _host_cfl_value(gb0_j, "radial grad-B drift")

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


def _grid_frequency_bounds(grid: SpectralGrid) -> _CFLGridBounds:
    kx, ky, kz = _cfl_wavenumber_arrays(grid)
    nz = kz.size
    nx = kx.size
    ny = int(grid.ky.shape[0])
    kx_max = float(abs(kx[(nx - 1) // 3])) if nx > 1 else 0.0
    ky_max = float(abs(ky[(ny - 1) // 3])) if ky.size > 0 else 0.0
    kz_max = float(abs(kz[nz // 2])) if nz > 0 else 0.0
    kx_min = float(abs(kx[1])) if nx > 1 else np.inf
    ky_min = float(abs(ky[1])) if ky.size > 1 else np.inf
    finite_min = min(kx_min, ky_min)
    kperp_min = float(finite_min) if np.isfinite(finite_min) else 0.0
    return _CFLGridBounds(
        kx=kx,
        ky=ky,
        kz=kz,
        kx_max=kx_max,
        ky_max=ky_max,
        kz_max=kz_max,
        kperp_min=kperp_min,
    )


def _species_frequency_scales(
    params: LinearParams,
    *,
    nl: int,
    nm: int,
) -> _SpeciesFrequencyScales:
    tprim = np.atleast_1d(_host_cfl_value(params.tprim, "tprim"))
    fprim = np.atleast_1d(_host_cfl_value(params.fprim, "fprim"))
    tz = np.atleast_1d(_host_cfl_value(params.tz, "tz"))
    vth = np.atleast_1d(_host_cfl_value(params.vth, "vth"))
    temp = np.atleast_1d(_host_cfl_value(params.temp, "temp"))
    dens = np.atleast_1d(_host_cfl_value(params.density, "density"))
    charge = np.atleast_1d(_host_cfl_value(params.charge_sign, "charge_sign"))

    tzmax = float(np.max(np.abs(tz))) if tz.size else 0.0
    vtmax = float(np.max(np.abs(vth))) if vth.size else 0.0
    vtmin = float(np.min(np.abs(vth))) if vth.size else 1.0
    return _SpeciesFrequencyScales(
        tprim=tprim,
        fprim=fprim,
        temp=temp,
        dens=dens,
        charge=charge,
        tzmax=tzmax,
        vtmax=vtmax,
        vtmin=vtmin,
        etamax=_gradient_ratio_max(tprim, fprim),
        vpar_max=2.0 * float(np.sqrt(max(nm, 1))),
        muB_max=_laguerre_velocity_max(nl),
    )


def _effective_geometry_frequency_scales(
    geom: FluxTubeGeometryLike,
    grid: SpectralGrid,
    grid_bounds: _CFLGridBounds,
    species: _SpeciesFrequencyScales,
) -> _GeometryFrequencyScales:
    (
        bmag_max,
        cvdrift_max,
        gbdrift_max,
        cvdrift0_max,
        gbdrift0_max,
        gradpar,
    ) = _geometry_frequency_maxima(geom, np.asarray(grid.z, dtype=float))

    shat = float(_host_cfl_value(geom.s_hat, "s_hat"))
    non_twist = bool(getattr(grid, "non_twist", False))
    m0_max = 0.0
    if non_twist and abs(shat) > 0.0 and grid_bounds.ky.size > 0:
        m0_max, cvdrift0_max, gbdrift0_max = _non_twist_shift_frequency_max(
            geom,
            grid,
            float(grid_bounds.ky[-1]),
            vpar_max=species.vpar_max,
            muB_max=species.muB_max,
        )
    return _GeometryFrequencyScales(
        bmag_max=bmag_max,
        cvdrift_max=cvdrift_max,
        gbdrift_max=gbdrift_max,
        cvdrift0_max=cvdrift0_max,
        gbdrift0_max=gbdrift0_max,
        gradpar=gradpar,
        m0_max=m0_max,
    )


def _radial_drift_frequency(
    geom: FluxTubeGeometryLike,
    grid: SpectralGrid,
    grid_bounds: _CFLGridBounds,
    species: _SpeciesFrequencyScales,
    geometry: _GeometryFrequencyScales,
) -> float:
    drift_strength = species.vpar_max * species.vpar_max * abs(
        geometry.cvdrift0_max
    ) + species.muB_max * abs(geometry.gbdrift0_max)
    shat = float(_host_cfl_value(geom.s_hat, "s_hat"))
    if abs(shat) == 0.0:
        kx_effective = grid_bounds.kx_max
    elif bool(getattr(grid, "non_twist", False)):
        kx_effective = grid_bounds.kx_max + geometry.m0_max / float(grid.x0)
    else:
        kx_effective = grid_bounds.kx_max / abs(shat)
    return species.tzmax * kx_effective * drift_strength


def _binormal_drift_frequency(
    grid_bounds: _CFLGridBounds,
    species: _SpeciesFrequencyScales,
    geometry: _GeometryFrequencyScales,
    *,
    include_diamagnetic_drive: bool,
) -> float:
    omega = (
        species.tzmax
        * grid_bounds.ky_max
        * (
            species.vpar_max * species.vpar_max * geometry.cvdrift_max
            + species.muB_max * geometry.gbdrift_max
        )
    )
    if include_diamagnetic_drive and species.etamax < 1.0e5:
        omega += grid_bounds.ky_max * (
            1.0
            + species.etamax
            * (species.vpar_max * species.vpar_max / 2.0 + species.muB_max - 1.5)
        )
    return omega


def _electron_pressure_product(species: _SpeciesFrequencyScales) -> float:
    if species.charge.size:
        neg = species.charge < 0.0
        if np.any(neg):
            ne = float(species.dens[neg][0])
            Te = float(species.temp[neg][0])
        else:
            ne = float(species.dens[0])
            Te = float(species.temp[0])
    else:
        ne = 1.0
        Te = 1.0
    return ne * Te


def _parallel_streaming_frequency(
    params: LinearParams,
    grid_bounds: _CFLGridBounds,
    species: _SpeciesFrequencyScales,
    geometry: _GeometryFrequencyScales,
) -> float:
    beta = float(_host_cfl_value(params.beta, "beta"))
    nspec_in = int(max(species.charge.size, 1))
    nte = _electron_pressure_product(species)
    mime = (
        (species.vtmax * species.vtmax) / (species.vtmin * species.vtmin)
        if species.vtmin > 0.0
        else 0.0
    )
    kperprho2 = (
        grid_bounds.kperp_min
        * grid_bounds.kperp_min
        / (geometry.bmag_max * geometry.bmag_max)
        if geometry.bmag_max > 0.0
        else 0.0
    )
    if nspec_in > 1:
        denom = beta * nte / 2.0 * mime + kperprho2
        guard = 1.0 / np.sqrt(denom) if denom > 0.0 else 0.0
    else:
        guard = 0.0
    return (
        species.vtmax
        * grid_bounds.kz_max
        * abs(geometry.gradpar)
        * max(species.vpar_max, guard)
    )


def _linear_frequency_bound(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    nl: int,
    nm: int,
    *,
    include_diamagnetic_drive: bool = True,
) -> np.ndarray:
    grid_bounds = _grid_frequency_bounds(grid)
    species = _species_frequency_scales(params, nl=nl, nm=nm)
    geometry = _effective_geometry_frequency_scales(geom, grid, grid_bounds, species)
    omega_max = np.zeros(3, dtype=float)
    omega_max[0] = _radial_drift_frequency(geom, grid, grid_bounds, species, geometry)
    omega_max[1] = _binormal_drift_frequency(
        grid_bounds,
        species,
        geometry,
        include_diamagnetic_drive=include_diamagnetic_drive,
    )
    omega_max[2] = _parallel_streaming_frequency(params, grid_bounds, species, geometry)
    return omega_max


def warn_if_fixed_dt_exceeds_cfl(
    *, grid: Any, geom: Any, params: Any, n_laguerre: int, n_hermite: int, tcfg: Any
) -> None:
    """Warn at startup when a fixed linear dt exceeds the explicit CFL estimate.

    The fixed-step linear paths advance the whole physical RHS explicitly (the
    imex methods only treat damping implicitly), so the explicit-time frequency
    bound applies to them too. An over-CFL dt overflows the trajectory, and the
    growth fit then fails loudly; this names the cause before the run starts.

    Geometry is conformed to this grid and any failure is swallowed: an
    imported equilibrium carries a closed theta interval one point longer than
    the grid, which made this hint abort every ``gkx wout.nc --linear`` run. A
    diagnostic that can stop a working run is worse than no diagnostic.
    """

    from gkx.config import resolve_cfl_fac
    from gkx.geometry import ensure_flux_tube_geometry_data

    try:
        geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
        wmax = float(
            np.sum(
                _linear_frequency_bound(grid, geom_eff, params, n_laguerre, n_hermite)
            )
        )
    except Exception:  # noqa: BLE001 - a startup hint must never stop a run
        return
    if not np.isfinite(wmax) or wmax <= 0.0:
        return
    cfl_fac = resolve_cfl_fac(str(tcfg.method), tcfg.cfl_fac)
    dt_stable = cfl_fac * float(tcfg.cfl) / wmax
    if float(tcfg.dt) > dt_stable:
        warnings.warn(
            f"requested dt={float(tcfg.dt):.4g} exceeds the estimated "
            f"CFL-stable step {dt_stable:.4g} (max linear frequency "
            f"{wmax:.4g}); the integration may overflow -- reduce dt",
            RuntimeWarning,
            stacklevel=2,
        )
