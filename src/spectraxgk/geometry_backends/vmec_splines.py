"""Radial VMEC/Boozer spline construction for imported geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from spectraxgk.geometry_backends.vmec_types import _Struct


def _vmec_splines(nc_obj: Any, booz_obj: Any) -> _Struct:
    """Build radial splines from a VMEC netCDF object and a booz_xform object.

    Build VMEC spline data used by the imported-geometry pipeline.
    """

    r = _Struct()

    ns = int(nc_obj.variables["ns"][:].data)
    s_full = np.linspace(0.0, 1.0, ns)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])

    mnmax_b = int(booz_obj.mnboz)

    r.rmnc_b = [
        InterpolatedUnivariateSpline(s_half, booz_obj.rmnc_b.T[:, j])
        for j in range(mnmax_b)
    ]
    r.zmns_b = [
        InterpolatedUnivariateSpline(s_half, booz_obj.zmns_b.T[:, j])
        for j in range(mnmax_b)
    ]
    r.numns_b = [
        InterpolatedUnivariateSpline(s_half, booz_obj.numns_b.T[:, j])
        for j in range(mnmax_b)
    ]
    r.d_rmnc_b_d_s = [f.derivative() for f in r.rmnc_b]
    r.d_zmns_b_d_s = [f.derivative() for f in r.zmns_b]
    r.d_numns_b_d_s = [f.derivative() for f in r.numns_b]

    r.gmnc_b = [
        InterpolatedUnivariateSpline(s_half, booz_obj.gmnc_b.T[:, j])
        for j in range(mnmax_b)
    ]
    r.bmnc_b = [
        InterpolatedUnivariateSpline(s_half, booz_obj.bmnc_b.T[:, j])
        for j in range(mnmax_b)
    ]
    r.d_bmnc_b_d_s = [f.derivative() for f in r.bmnc_b]

    r.Gfun = InterpolatedUnivariateSpline(s_half, booz_obj.Boozer_G)
    r.Ifun = InterpolatedUnivariateSpline(s_half, booz_obj.Boozer_I)
    r.pressure = InterpolatedUnivariateSpline(
        s_half, np.asarray(nc_obj.variables["pres"][1:])
    )
    r.d_pressure_d_s = r.pressure.derivative()
    r.psi = InterpolatedUnivariateSpline(
        s_half, np.asarray(nc_obj.variables["phi"][1:]) / (2.0 * np.pi)
    )
    r.d_psi_d_s = r.psi.derivative()
    r.iota = InterpolatedUnivariateSpline(
        s_half, np.asarray(nc_obj.variables["iotas"][1:])
    )
    r.d_iota_d_s = r.iota.derivative()

    r.phiedge = float(nc_obj.variables["phi"][-1])
    r.Aminor_p = float(nc_obj.variables["Aminor_p"][:])
    r.nfp = int(nc_obj.variables["nfp"][:])
    r.raxis_cc = np.asarray(nc_obj.variables["raxis_cc"][:])

    r.xm_b = booz_obj.xm_b
    r.xn_b = booz_obj.xn_b
    r.mnbooz = mnmax_b
    r.mboz = booz_obj.mboz
    r.nboz = booz_obj.nboz

    return r

