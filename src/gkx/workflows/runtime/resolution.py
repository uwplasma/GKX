"""Deterministic minimum-grid estimator for adiabatic-electron ITG runs.

The criterion is the binormal reach ky_max*rho, met by (y0, Ny) jointly:
the box fixes dky = 1/y0 and the 2/3 rule resolves ky_max = ((Ny-1)//3)*dky.
The target is CLASS-based (tokamak vs stellarator, from nfp), calibrated on
the 2026-08 y0=14 ladder (dky*rho = 0.071, the published stella/GENE W7-X
spacing; Nx=Ny in {64, 96, 128}, Nz=48, Nl=4, Nm=8, t_max=400 with
saturation auto-stop): DIII-D saturated at every rung and its 96/128 fluxes
agree within errors (16.86 +/- 0.68 vs 17.50 +/- 0.87), with 64^2 only ~8%
above that plateau; every stellarator case instead converged FROM ABOVE and
was still falling at 128^2 (QA-vacuum 6.73 -> 5.61 -> 4.90, QHS 5.91 ->
4.49 -> 3.64, QA-beta0.5 7.20 -> 6.45 -> 5.92), so stellarator tiers carry
measured upper-estimate bias annotations rather than convergence claims.
A finer per-geometry metric was tried and FALSIFIED by this ladder: the
anisotropy rms|grad x|/max|grad y| ordered the earlier dky=0.048 ladder's
minima 17/17 but not the corrected one (QHS has the lowest stellarator
anisotropy yet the steepest continuing decline). The metric is retained
only as a reported diagnostic and a tokamak-class corroboration. The
cautious stellarator tier matches published W7-X practice, ky_max ~ 4.4
(JPP 2024, App. G); no rung of this scan validates it directly yet.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from gkx.config import resolve_cfl_fac
from gkx.core_grid import build_spectral_grid
from gkx.geometry import FluxTubeGeometryLike
from gkx.geometry.core import (
    apply_geometry_grid_defaults,
    ensure_flux_tube_geometry_data,
)
from gkx.solvers_time_explicit_cfl import _linear_frequency_bound
from gkx.config import RuntimeConfig
from gkx.workflows.runtime.startup import (
    build_runtime_geometry,
    build_runtime_linear_params,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml

#: Perpendicular grid rungs, extending the scan ladder upward.
PERP_LADDER = (32, 64, 96, 128, 192, 256)

#: ky_max*rho per (class, tier). At the default dky*rho = 0.071 these land on
#: rungs 64/96/128 (tokamak) and 96/128/192 (stellarator). Measured y0=14
#: ladder bias per tier -- tokamak: preview ~+8% above the converged 96/128
#: plateau, standard converged within SEM, cautious headroom; stellarator:
#: preview ~+15-25% and standard ~+8-19% (both still upper estimates, flux
#: falling at 128^2), cautious = published W7-X reach ~4.4 (unvalidated by
#: this scan until the 192^2 references land).
KY_TARGETS_BY_CLASS = {
    "tokamak": {"preview": 1.5, "standard": 2.2, "cautious": 2.9},
    "stellarator": {"preview": 2.2, "standard": 2.9, "cautious": 4.4},
}

# Anisotropy corroborates the nfp class split only (measured: DIII-D 0.138
# vs stellarators 0.55-0.80); its finer per-case ordering was falsified.
_TOKAMAK_ANISOTROPY_MAX = 0.30

_TARGET_ERRORS = ("preview", "standard", "cautious")
_WELL_PROMINENCE = 0.10  # fraction of the |B| range a well must dip to count
_PUBLISHED_STELLARATOR_DKY = 0.071  # published W7-X spacing; 0.100 also in use
_SATURATION_TIME = 50.0  # typical a/v_ti units to saturation in the scan
_T_MAX_FACTOR = 8.0  # scan cap 400 = 8 * t_sat; tokamak rungs stopped early


@dataclass(frozen=True)
class GeometryFeatures:
    """Host-side geometry scalars the estimator maps to a grid."""

    anisotropy: float
    shat: float
    q: float
    nfp: int
    bmag_wells: int
    zp: float


def _count_bmag_wells(bmag: np.ndarray) -> int:
    """Count ``|B|`` wells deeper than ``_WELL_PROMINENCE`` of the ``|B|`` range."""

    n, b_range = int(bmag.size), float(bmag.max() - bmag.min()) if bmag.size else 0.0
    if n < 4 or b_range <= 0.0:
        return 1
    prev, nxt = np.roll(bmag, 1), np.roll(bmag, -1)
    minima = np.flatnonzero((bmag < prev) & (bmag < nxt))
    maxima = np.flatnonzero((bmag > prev) & (bmag > nxt))
    if minima.size == 0 or maxima.size == 0:
        return 1
    wells = 0
    for i in minima:
        left = maxima[np.argmin((int(i) - maxima) % n)]
        right = maxima[np.argmin((maxima - int(i)) % n)]
        wells += min(bmag[left], bmag[right]) - bmag[i] >= _WELL_PROMINENCE * b_range
    return max(int(wells), 1)


def geometry_features(geom: Any, *, zp: float = 1.0) -> GeometryFeatures:
    """Extract estimator features from sampled flux-tube geometry data."""

    sl = slice(None, -1) if bool(geom.theta_closed_interval) else slice(None)
    gds2 = np.asarray(geom.gds2_profile, dtype=float)[sl]
    gds22 = np.asarray(geom.gds22_profile, dtype=float)[sl]
    bmag = np.asarray(geom.bmag_profile, dtype=float)[sl]
    shat = float(geom.s_hat)
    gradx_rms = float(np.sqrt(np.mean(gds22))) / max(abs(shat), 1.0e-8)
    grady_max = float(np.sqrt(np.max(gds2)))
    return GeometryFeatures(
        anisotropy=gradx_rms / grady_max,
        shat=shat,
        q=float(geom.q),
        nfp=int(geom.nfp),
        bmag_wells=_count_bmag_wells(bmag),
        zp=float(zp),
    )


def geometry_class(features: GeometryFeatures) -> str:
    """Classify the equilibrium; nfp decides, anisotropy corroborates."""

    if features.nfp > 1:
        return "stellarator"
    return (
        "tokamak" if features.anisotropy <= _TOKAMAK_ANISOTROPY_MAX else "stellarator"
    )


def ky_max_target(features: GeometryFeatures, target_error: str = "standard") -> float:
    """Required binormal reach ky_max*rho for this class and error tier."""

    if target_error not in _TARGET_ERRORS:
        raise ValueError(
            f"target_error must be one of {sorted(_TARGET_ERRORS)}, got {target_error!r}"
        )
    return KY_TARGETS_BY_CLASS[geometry_class(features)][target_error]


def perp_points_for(dky: float, ky_target: float) -> int:
    """Smallest ladder rung whose dealiased ky grid reaches ``ky_target``."""

    for rung in PERP_LADDER:
        if ((rung - 1) // 3) * dky >= 0.98 * ky_target:
            return rung
    return PERP_LADDER[-1]


def resolution_from_features(
    features: GeometryFeatures,
    *,
    dky: float,
    target_error: str = "standard",
    nz_default: int = 48,
    hypercollisions: bool = True,
    kinetic_electrons: bool = False,
) -> dict[str, Any]:
    """Map geometry features to grid hints with a rationale per number."""

    klass = geometry_class(features)
    ky_target = ky_max_target(features, target_error)
    nx = perp_points_for(dky, ky_target)
    floor = max(16.0 * features.zp, 6.0 * features.bmag_wells)
    nz_floor = int(-(-int(np.ceil(floor)) // 8) * 8)  # round up to multiple of 8
    nz = max(nz_default, nz_floor) if target_error == "cautious" else nz_default
    nl, nm = (4, 8) if hypercollisions else (6, 12)
    if kinetic_electrons:
        nm = max(nm, 16)
    notes = []
    if nz_floor > nz:
        notes.append(
            f"parallel floor unmet: {features.bmag_wells} |B| wells x 6 pts and 16/2pi-"
            f"period ask Nz >= {nz_floor}; the scan left Nz unvalidated here"
        )
    if dky < 0.9 * _PUBLISHED_STELLARATOR_DKY:
        alt = perp_points_for(_PUBLISHED_STELLARATOR_DKY, ky_target)
        notes.append(
            f"cheaper box: dky*rho = {_PUBLISHED_STELLARATOR_DKY} (y0 = 14, the "
            f"calibrated ladder's spacing) reaches ky_max {ky_target:g} at "
            f"Nx=Ny={alt}"
        )
    if klass == "stellarator":
        notes.append(
            "stellarator fluxes converge from above: the calibration ladder was "
            "still falling at 128^2, so this tier's flux is an upper estimate "
            "(measured bias: preview ~+15-25%, standard ~+8-19% vs the next rung)"
        )
    if abs(features.shat) < 0.3:
        notes.append(
            "low-shear tube: check npol>=2 and a second field line before "
            "promoting any flux (Faber 2018; Ajay 2020; Kim 2024)"
        )
    hyper = "hypercollisions" if hypercollisions else "no hypercollisions"
    rationale = {
        "nx": "square perpendicular box (Lx = Ly), so Nx tracks Ny",
        "ny": (
            f"{klass} class (nfp = {features.nfp}, anisotropy "
            f"{features.anisotropy:.3f}) asks ky_max*rho >= {ky_target:g} "
            f"({target_error}); reach ((Ny-1)//3)*dky = "
            f"{((nx - 1) // 3) * dky:.2f} at dky = {dky:.3f}"
        ),
        "nz": (
            f"scan found Nz weakly coupled (flux 8.47/8.78/8.39 at 24/32/48); "
            f"floors: 16/2pi-period x {features.zp:g}, 6 x {features.bmag_wells} |B| wells"
        ),
        "nl": f"Laguerre FLR floor with {hyper}; the scan converged at Nl=4",
        "nm": (
            f"{hyper}: t_quiet ~ 5.5*sqrt(Nm) recurrence sets the published floor "
            f"({'4,8' if hypercollisions else '6,12'})"
            + ("; kinetic electrons floor Nm at 16" if kinetic_electrons else "")
        ),
        "t_max": f"{_T_MAX_FACTOR:g} x t_sat ~ {_SATURATION_TIME:g} hard cap; "
        'run_to = "saturation" stops earlier',
    }
    t_max = _T_MAX_FACTOR * _SATURATION_TIME
    return {
        "nx": nx,
        "ny": nx,
        "nz": nz,
        "nl": nl,
        "nm": nm,
        "t_max": 0.5 * t_max if target_error == "preview" else t_max,
        "geometry_class": klass,
        "ky_max_target": ky_target,
        "rationale": rationale,
        "features": features,
        "notes": notes,
    }


def _dt_hint(
    cfg: RuntimeConfig, geom: FluxTubeGeometryLike, est: dict[str, Any]
) -> float:
    """Initial-dt bound from the solver's own linear CFL frequency estimator."""

    grid_cfg = replace(cfg.grid, Nx=int(est["nx"]), Ny=int(est["ny"]))
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, grid_cfg))
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    params = build_runtime_linear_params(cfg, Nm=int(est["nm"]), geom=geom_eff)
    nl, nm = int(est["nl"]), int(est["nm"])
    wmax = float(np.sum(_linear_frequency_bound(grid, geom_eff, params, nl, nm)))
    if wmax <= 0.0:
        return float(cfg.time.dt)
    fac = resolve_cfl_fac(cfg.time.method, cfg.time.cfl_fac)
    return min(float(cfg.time.dt), fac * float(cfg.time.cfl) / wmax)


def estimate_resolution(
    wout_path: str | Path,
    *,
    torflux: float | None = None,
    target_error: str = "standard",
    deck_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate the minimum adequate grid for one VMEC/VMEX equilibrium."""

    from gkx.workflows.runtime.wout import default_wout_deck_path

    cfg, _ = load_runtime_from_toml(
        deck_path if deck_path is not None else default_wout_deck_path()
    )
    geometry = replace(cfg.geometry, vmec_file=str(Path(wout_path).resolve()))
    if torflux is not None:
        geometry = replace(geometry, torflux=float(torflux))
    cfg = replace(cfg, geometry=geometry)
    geom = build_runtime_geometry(cfg)

    grid = cfg.grid
    zp = float(grid.zp if grid.zp is not None else 2 * (grid.nperiod or 1) - 1)
    y0 = float(grid.y0) if grid.y0 is not None else grid.Ly / (2.0 * np.pi)
    est = resolution_from_features(
        geometry_features(geom, zp=zp),
        dky=1.0 / y0,
        target_error=target_error,
        nz_default=int(grid.ntheta if grid.ntheta is not None else grid.Nz),
        hypercollisions=bool(cfg.physics.hypercollisions),
        kinetic_electrons=any(
            sp.kinetic and float(sp.charge) < 0.0 for sp in cfg.species
        ),
    )
    est["dt"] = _dt_hint(cfg, geom, est)
    est["rationale"]["dt"] = (
        "explicit CFL bound cfl_fac*cfl/sum(omega_max) at this grid; "
        "the adaptive stepper raises it toward the measured ExB limit"
    )
    est["torflux"] = (
        None if cfg.geometry.torflux is None else float(cfg.geometry.torflux)
    )
    return est
