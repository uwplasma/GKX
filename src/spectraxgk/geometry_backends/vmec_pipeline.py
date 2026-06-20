"""High-level VMEC-to-imported-geometry pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.geometry_backends.vmec_fieldlines import _vmec_fieldlines
from spectraxgk.geometry_backends.vmec_io import _write_vmec_eik_netcdf_atomically
from spectraxgk.geometry_backends.vmec_remap import _apply_flux_tube_cut, _equal_arc_remap


@dataclass(frozen=True)
class _VmecThetaGrid:
    npol: float
    npol_min: float | None
    ntheta_in: int
    ntheta: int
    theta: np.ndarray


@dataclass(frozen=True)
class _VmecFluxTubeCut:
    kind: str
    which_crossing: int
    y0: float
    x0: float
    jtwist_in: Any


def _require_vmec_request(request: Any | None) -> Any:
    if request is None:
        raise NotImplementedError(
            "Internal VMEC geometry backend requires a VMEC geometry request. "
            "Pass request= to invoke the port."
        )
    return request


def _vmec_theta_grid(request: Any) -> _VmecThetaGrid:
    npol = float(request.npol)
    npol_min = None if request.npol_min is None else float(request.npol_min)
    ntheta_in = int(request.ntheta)
    ntheta = ntheta_in + 1  # imported-geometry convention: ntheta_in + 1 output points
    if npol_min is not None:
        theta = np.linspace(
            -2.0 * npol_min * np.pi, 2.0 * npol_min * np.pi, 2 * ntheta_in + 1
        )
    else:
        theta = np.linspace(-npol * np.pi, npol * np.pi, ntheta)
    return _VmecThetaGrid(
        npol=npol,
        npol_min=npol_min,
        ntheta_in=ntheta_in,
        ntheta=ntheta,
        theta=theta,
    )


def _vmec_flux_tube_cut_kind(request: Any) -> str:
    boundary = str(request.boundary).strip().lower()
    if boundary == "exact periodic":
        return "gds21"
    if boundary == "continuous drifts":
        return "gbdrift0"
    if boundary == "fix aspect":
        return "aspect"
    return "none"


def _vmec_flux_tube_cut(request: Any, *, grid: _VmecThetaGrid) -> _VmecFluxTubeCut:
    which_crossing = request.which_crossing
    if which_crossing is None:
        which_crossing = 0 if grid.npol_min is not None else -1
    y0 = float(request.y0)
    x0 = float(request.x0) if request.x0 is not None else y0
    return _VmecFluxTubeCut(
        kind=_vmec_flux_tube_cut_kind(request),
        which_crossing=int(which_crossing),
        y0=y0,
        x0=x0,
        jtwist_in=request.jtwist,
    )


def _vmec_betaprim(request: Any) -> float:
    if request.betaprim is not None:
        return float(request.betaprim)
    dens = np.asarray(request.dens, dtype=float)
    temp = np.asarray(request.temp, dtype=float)
    tprim = np.asarray(request.tprim, dtype=float)
    fprim = np.asarray(request.fprim, dtype=float)
    return -float(request.beta) * float(np.sum(dens * temp * (tprim + fprim)))


def _vmec_fieldline_geometry(
    *,
    request: Any,
    theta: np.ndarray,
    betaprim: float,
) -> Any:
    return _vmec_fieldlines(
        vmec_fname=str(request.vmec_file),
        s_val=float(request.torflux),
        betaprim=betaprim,
        alpha=float(request.alpha),
        include_shear_variation=bool(request.include_shear_variation),
        include_pressure_variation=bool(request.include_pressure_variation),
        theta1d=theta,
        isaxisym=bool(request.isaxisym),
        iota_input=None,
        s_hat_input=None,
    )


def _vmec_profiles_from_equal_arc(geo: Any, arrays_equal_arc: dict[str, Any]) -> dict[str, Any]:
    dpsidrho = float(geo.dpsidrho)
    qfac = abs(1.0 / float(geo.iota_input))
    shat = float(geo.s_hat_input)
    nfp = int(geo.nfp)
    alpha_out = float(geo.alpha)
    zeta_center_out = float(geo.zeta_center)

    R_arr = arrays_equal_arc["Rplot"]
    Rmaj = float((np.max(R_arr) + np.min(R_arr)) / 2.0)

    return {
        **arrays_equal_arc,
        "dpsidrho": dpsidrho,
        "kxfac": 1.0,
        "Rmaj": Rmaj,
        "q": qfac,
        "shat": shat,
        "alpha": alpha_out,
        "zeta_center": zeta_center_out,
        "nfp": nfp,
    }


def generate_vmec_eik_internal(
    *,
    output_path: str | Path,
    request: Any | None = None,
) -> Path:
    """Internal VMEC->EIK pipeline.

    Generate imported VMEC geometry from a runtime request. Accepts a
    runtime VMEC geometry request and writes an imported-geometry ``*.eik.nc`` file.
    """

    request = _require_vmec_request(request)
    grid = _vmec_theta_grid(request)
    cut = _vmec_flux_tube_cut(request, grid=grid)
    geo = _vmec_fieldline_geometry(
        request=request,
        theta=grid.theta,
        betaprim=_vmec_betaprim(request),
    )
    theta_cut, arrays_cut = _apply_flux_tube_cut(
        theta=grid.theta,
        geo=geo,
        ntheta=grid.ntheta,
        flux_tube_cut=cut.kind,
        npol_min=grid.npol_min,
        which_crossing=cut.which_crossing,
        y0=cut.y0,
        x0=cut.x0,
        jtwist_in=cut.jtwist_in,
    )
    _gradpar_eqarc, arrays_equal_arc = _equal_arc_remap(
        theta=theta_cut,
        arrays=arrays_cut,
        ntheta=grid.ntheta,
    )
    profiles = _vmec_profiles_from_equal_arc(geo, arrays_equal_arc)
    out = Path(output_path).expanduser().resolve()
    _write_vmec_eik_netcdf_atomically(out, profiles, request=request)
    return out
