"""Shared vmex state control helpers for differentiable geometry gates."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gkx.geometry.backend_discovery import (
    _booz_read_wout_square_layout_failure,
    _import_booz_backend,
    _new_booz_object,
)
from gkx.geometry.vmec_boozer_derivatives import (
    _BoozerFieldlineSamples,
    _VMECFieldlineScalars,
    _axisym_flip_required,
    _fieldline_boozer_coordinates,
    _input_iota_shear,
    _validated_reference_scales,
)
from gkx.geometry.vmec_field_line_sampling import (
    _boozer_mode_angle,
    _boozer_trig_basis,
    _fieldline_boozer_tensors,
    _sample_boozer_mode_table,
    _vmec_splines,
)


VMEC_BOOZER_STATE_PARAMETER_NAMES = ("Rcos_mid_surface_m1",)
VMEC_BOOZER_STATE_PARAMETER_FAMILIES = ("Rcos", "Rsin", "Zcos", "Zsin", "Lcos", "Lsin")


def _new_boozer_object_with_auto_fallback(
    primary_backend: Any, vmec_fname: str | Path, nc_obj: Any
) -> Any:
    """Create a Boozer transform object, using the classic reader if needed.

    Some vmex-written WOUT files expose a square ``(radius, mode)`` layout that
    old ``booz_xform_jax`` readers reject as ambiguous. In automatic backend
    mode, the imported-geometry path can safely fall back to the classic
    ``booz_xform`` reader; explicit backend selections remain fail-fast.
    """

    try:
        return _new_booz_object(primary_backend, str(vmec_fname))
    except Exception as exc:
        auto_backend = os.environ.get("GKX_BOOZ_BACKEND", "auto").strip().lower()
        if auto_backend in {"", "auto"} and _booz_read_wout_square_layout_failure(exc):
            try:
                fallback = _import_booz_backend("booz_xform")
                return _new_booz_object(fallback, str(vmec_fname))
            except Exception:
                nc_obj.close()
                raise
        nc_obj.close()
        raise


@dataclass(frozen=True)
class _BoozerModeProfiles:
    rmnc_b: np.ndarray
    zmns_b: np.ndarray
    numns_b: np.ndarray
    d_rmnc_b_d_s: np.ndarray
    d_zmns_b_d_s: np.ndarray
    d_numns_b_d_s: np.ndarray
    gmnc_b: np.ndarray
    bmnc_b: np.ndarray
    d_bmnc_b_d_s: np.ndarray


@dataclass(frozen=True)
class _BoozerFieldlineCoordinates:
    theta_b: np.ndarray
    phi_b: np.ndarray
    flipit: bool


@dataclass(frozen=True)
class _BoozerTrigSamples:
    cosangle_b: np.ndarray
    sinangle_b: np.ndarray
    mcosangle_b: np.ndarray
    msinangle_b: np.ndarray
    ncosangle_b: np.ndarray
    nsinangle_b: np.ndarray


def _load_vmec_boozer_splines(vmec_fname: str | Path) -> tuple[Any, Any]:
    """Open VMEC data, run Boozer transform, and build radial splines."""

    bxform = _import_booz_backend()
    from netCDF4 import Dataset as _NC

    nc_obj = _NC(str(vmec_fname), "r")
    try:
        mpol = int(nc_obj.variables["mpol"][:])
        ntor = int(nc_obj.variables["ntor"][:])
        booz_obj = _new_boozer_object_with_auto_fallback(bxform, vmec_fname, nc_obj)
        booz_obj.mboz = int(2 * mpol)
        booz_obj.nboz = int(2 * ntor)
        booz_obj.run()
        return nc_obj, _vmec_splines(nc_obj, booz_obj)
    except Exception:
        nc_obj.close()
        raise


def _fieldline_scalar_profiles(
    vs: Any,
    *,
    s_val: float,
    alpha: float,
    iota_input: float | None,
    s_hat_input: float | None,
) -> _VMECFieldlineScalars:
    """Sample scalar VMEC profiles and normalize reference scales."""

    s = np.array([s_val])
    alpha_arr = np.array([alpha])
    d_pressure_d_s = vs.d_pressure_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    shat = (-2.0 * s / iota) * d_iota_d_s
    edge_toroidal_flux_over_2pi = float(-vs.phiedge / (2.0 * np.pi))
    toroidal_flux_sign = float(np.sign(edge_toroidal_flux_over_2pi))
    L_reference, B_reference, R_mag_ax = _validated_reference_scales(
        vs, edge_toroidal_flux_over_2pi
    )
    iota_input_val, s_hat_input_val = _input_iota_shear(
        iota, shat, iota_input, s_hat_input
    )
    return _VMECFieldlineScalars(
        s=s,
        ns=1,
        alpha_arr=alpha_arr,
        d_pressure_d_s=d_pressure_d_s,
        iota=iota,
        d_iota_d_s=d_iota_d_s,
        shat=shat,
        nfp=vs.nfp,
        edge_toroidal_flux_over_2pi=edge_toroidal_flux_over_2pi,
        toroidal_flux_sign=toroidal_flux_sign,
        L_reference=float(L_reference),
        B_reference=float(B_reference),
        R_mag_ax=float(R_mag_ax),
        zeta_center=-alpha / float(iota[0]),
        iota_input_val=iota_input_val,
        s_hat_input_val=s_hat_input_val,
        G=vs.Gfun(s),
        boozer_i=vs.Ifun(s),
    )


def _sample_boozer_mode_profiles(
    vs: Any, scalars: _VMECFieldlineScalars
) -> _BoozerModeProfiles:
    """Sample Boozer coefficients and their radial derivatives."""

    (
        rmnc_b,
        zmns_b,
        numns_b,
        d_rmnc_b_d_s,
        d_zmns_b_d_s,
        d_numns_b_d_s,
        gmnc_b,
        bmnc_b,
        d_bmnc_b_d_s,
    ) = _sample_boozer_mode_table(vs, scalars.s, scalars.ns)
    return _BoozerModeProfiles(
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
        numns_b=numns_b,
        d_rmnc_b_d_s=d_rmnc_b_d_s,
        d_zmns_b_d_s=d_zmns_b_d_s,
        d_numns_b_d_s=d_numns_b_d_s,
        gmnc_b=gmnc_b,
        bmnc_b=bmnc_b,
        d_bmnc_b_d_s=d_bmnc_b_d_s,
    )


def _fieldline_coordinates_and_flip(
    *,
    theta1d: np.ndarray,
    scalars: _VMECFieldlineScalars,
    xm_b: np.ndarray,
    xn_b: np.ndarray,
    profiles: _BoozerModeProfiles,
    isaxisym: bool,
) -> _BoozerFieldlineCoordinates:
    """Return field-line Boozer coordinates and axisymmetric orientation."""

    theta_b, phi_b = _fieldline_boozer_coordinates(
        theta1d, scalars.alpha_arr, scalars.iota
    )
    flipit = _axisym_flip_required(
        isaxisym=isaxisym,
        xm_b=xm_b,
        xn_b=xn_b,
        theta_b=theta_b,
        phi_b=phi_b,
        rmnc_b=profiles.rmnc_b,
        zmns_b=profiles.zmns_b,
    )
    return _BoozerFieldlineCoordinates(
        theta_b=theta_b,
        phi_b=phi_b,
        flipit=bool(flipit),
    )


def _fieldline_trig_samples(
    xm_b: np.ndarray, xn_b: np.ndarray, coords: _BoozerFieldlineCoordinates
) -> _BoozerTrigSamples:
    """Return Boozer angle basis arrays used by tensor mode sums."""

    angle_b = _boozer_mode_angle(
        xm_b, xn_b, coords.theta_b, coords.phi_b, flipit=coords.flipit
    )
    (
        cosangle_b,
        sinangle_b,
        mcosangle_b,
        msinangle_b,
        ncosangle_b,
        nsinangle_b,
    ) = _boozer_trig_basis(xm_b, xn_b, angle_b)
    return _BoozerTrigSamples(
        cosangle_b=cosangle_b,
        sinangle_b=sinangle_b,
        mcosangle_b=mcosangle_b,
        msinangle_b=msinangle_b,
        ncosangle_b=ncosangle_b,
        nsinangle_b=nsinangle_b,
    )


def _sample_fieldline_boozer_state(
    vs: Any,
    scalars: _VMECFieldlineScalars,
    *,
    theta1d: np.ndarray,
    isaxisym: bool,
) -> _BoozerFieldlineSamples:
    """Build Boozer mode tables, field-line coordinates, and tensor sums."""

    xm_b = vs.xm_b
    xn_b = vs.xn_b
    profiles = _sample_boozer_mode_profiles(vs, scalars)
    coords = _fieldline_coordinates_and_flip(
        theta1d=theta1d,
        scalars=scalars,
        xm_b=xm_b,
        xn_b=xn_b,
        profiles=profiles,
        isaxisym=isaxisym,
    )
    trig = _fieldline_trig_samples(xm_b, xn_b, coords)
    tensors = _fieldline_boozer_tensors(
        rmnc_b=profiles.rmnc_b,
        zmns_b=profiles.zmns_b,
        numns_b=profiles.numns_b,
        d_rmnc_b_d_s=profiles.d_rmnc_b_d_s,
        d_zmns_b_d_s=profiles.d_zmns_b_d_s,
        d_numns_b_d_s=profiles.d_numns_b_d_s,
        gmnc_b=profiles.gmnc_b,
        bmnc_b=profiles.bmnc_b,
        d_bmnc_b_d_s=profiles.d_bmnc_b_d_s,
        cosangle_b=trig.cosangle_b,
        sinangle_b=trig.sinangle_b,
        mcosangle_b=trig.mcosangle_b,
        msinangle_b=trig.msinangle_b,
        ncosangle_b=trig.ncosangle_b,
        nsinangle_b=trig.nsinangle_b,
    )
    return _BoozerFieldlineSamples(
        xm_b=xm_b,
        xn_b=xn_b,
        rmnc_b=profiles.rmnc_b,
        zmns_b=profiles.zmns_b,
        numns_b=profiles.numns_b,
        d_rmnc_b_d_s=profiles.d_rmnc_b_d_s,
        d_zmns_b_d_s=profiles.d_zmns_b_d_s,
        d_numns_b_d_s=profiles.d_numns_b_d_s,
        gmnc_b=profiles.gmnc_b,
        bmnc_b=profiles.bmnc_b,
        d_bmnc_b_d_s=profiles.d_bmnc_b_d_s,
        theta_b=coords.theta_b,
        phi_b=coords.phi_b,
        flipit=coords.flipit,
        tensors=tensors,
        R_b=tensors.R_b,
        Z_b=tensors.Z_b,
        nu_b=tensors.nu_b,
        Vprime=profiles.gmnc_b[:, 0],
        mnmax_b=profiles.rmnc_b.shape[1],
    )


__all__ = [
    "VMEC_BOOZER_STATE_PARAMETER_FAMILIES",
    "VMEC_BOOZER_STATE_PARAMETER_NAMES",
]
