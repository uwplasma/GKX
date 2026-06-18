"""High-level VMEC-to-imported-geometry pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.geometry_backends.vmec_fieldlines import _vmec_fieldlines
from spectraxgk.geometry_backends.vmec_io import _write_vmec_eik_netcdf_atomically
from spectraxgk.geometry_backends.vmec_remap import _apply_flux_tube_cut, _equal_arc_remap

def generate_vmec_eik_internal(
    *,
    output_path: str | Path,
    request: Any | None = None,
) -> Path:
    """Internal VMEC->EIK pipeline.

    Generate imported VMEC geometry from a runtime request. Accepts a
    runtime VMEC geometry request and writes an imported-geometry ``*.eik.nc`` file.
    """

    if request is None:
        raise NotImplementedError(
            "Internal VMEC geometry backend requires a VMEC geometry request. "
            "Pass request= to invoke the port."
        )

    npol = float(request.npol)
    npol_min = None if request.npol_min is None else float(request.npol_min)
    ntheta_in = int(request.ntheta)
    ntheta = ntheta_in + 1  # imported-geometry convention: ntheta_in + 1 output points

    # Map boundary string to flux-tube-cut type (imported-geometry convention)
    boundary = str(request.boundary).strip().lower()
    if boundary == "exact periodic":
        flux_tube_cut = "gds21"
    elif boundary == "continuous drifts":
        flux_tube_cut = "gbdrift0"
    elif boundary == "fix aspect":
        flux_tube_cut = "aspect"
    else:
        flux_tube_cut = "none"

    which_crossing = request.which_crossing
    if which_crossing is None:
        which_crossing = 0 if npol_min is not None else -1

    if request.betaprim is not None:
        betaprim = float(request.betaprim)
    else:
        z = np.asarray(request.z, dtype=float)
        dens = np.asarray(request.dens, dtype=float)
        temp = np.asarray(request.temp, dtype=float)
        tprim = np.asarray(request.tprim, dtype=float)
        fprim = np.asarray(request.fprim, dtype=float)
        _ = z  # charge does not enter the betaprim expression
        betaprim = -float(request.beta) * float(np.sum(dens * temp * (tprim + fprim)))
    y0 = float(request.y0)
    x0 = float(request.x0) if request.x0 is not None else y0
    jtwist_in = request.jtwist

    # Boozer-theta grid over the full npol range (extra points help the cut)
    if npol_min is not None:
        theta = np.linspace(
            -2.0 * npol_min * np.pi, 2.0 * npol_min * np.pi, 2 * ntheta_in + 1
        )
    else:
        theta = np.linspace(-npol * np.pi, npol * np.pi, ntheta)

    # Main Boozer-coordinate fieldline calculation
    geo = _vmec_fieldlines(
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

    dpsidrho = float(geo.dpsidrho)
    qfac = abs(1.0 / float(geo.iota_input))
    shat = float(geo.s_hat_input)
    nfp = int(geo.nfp)
    alpha_out = float(geo.alpha)
    zeta_center_out = float(geo.zeta_center)

    # Flux-tube cut
    theta_cut, arrays_cut = _apply_flux_tube_cut(
        theta=theta,
        geo=geo,
        ntheta=ntheta,
        flux_tube_cut=flux_tube_cut,
        npol_min=npol_min,
        which_crossing=which_crossing,
        y0=y0,
        x0=x0,
        jtwist_in=jtwist_in,
    )

    # Equal-arc remap onto uniform theta grid
    _gradpar_eqarc, arrays_gx = _equal_arc_remap(
        theta=theta_cut,
        arrays=arrays_cut,
        ntheta=ntheta,
    )

    R_arr = arrays_gx["Rplot"]
    Rmaj = float((np.max(R_arr) + np.min(R_arr)) / 2.0)

    profiles: dict[str, Any] = {
        **arrays_gx,
        "dpsidrho": dpsidrho,
        "kxfac": 1.0,
        "Rmaj": Rmaj,
        "q": qfac,
        "shat": shat,
        "alpha": alpha_out,
        "zeta_center": zeta_center_out,
        "nfp": nfp,
    }

    out = Path(output_path).expanduser().resolve()
    _write_vmec_eik_netcdf_atomically(out, profiles, request=request)
    return out
