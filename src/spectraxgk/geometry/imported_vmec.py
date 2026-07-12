"""Imported VMEC/Boozer geometry backend and EIK-file generation helpers.

This module owns the in-repo VMEC-to-imported-geometry path: optional Boozer
backend discovery, radial spline construction, Boozer field-line sampling,
flux-tube cutting, equal-arc remapping, and NetCDF writeout. Keeping the VMEC
backend in the primary ``spectraxgk.geometry`` namespace makes the package
layout easier to navigate while preserving the same function-level test seams.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import uuid

import numpy as np
from scipy.integrate import cumulative_trapezoid as _ctrap
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline, PPoly, splrep

from spectraxgk.geometry.backend_discovery import (
    _booz_xform_jax_search_paths,
    _import_module_with_search_paths,
    _import_booz_xform_jax_backend,
    _import_booz_xform_backend,
    _import_booz_backend,
    _booz_read_wout_square_layout_failure,
    _new_booz_object,
    internal_vmec_backend_available,
)
from spectraxgk.geometry.vmec_field_line_sampling import (
    _Struct,
    _MU_0,
    _vmec_splines,
    nperiod_set,
    dermv,
    _sample_boozer_mode_table,
    _boozer_mode_angle,
    _boozer_mode_sum,
    _boozer_trig_basis,
    _fieldline_boozer_tensors,
    _fieldline_cartesian_derivatives,
    _fieldline_coordinate_gradients,
    _fieldline_alpha_gradients,
    _fieldline_local_shear,
    _fieldline_metric_drifts,
)
from spectraxgk.geometry.vmec_boozer_derivatives import (
    _axisym_flip_required,
    _centered_fieldline_integral,
    _fieldline_boozer_coordinates,
    _flux_surface_hngc_averages,
    _hngc_pressure_correction,
    _hngc_shear_correction,
    _input_iota_shear,
    _safe_mode_denominator,
    _validated_reference_scales,
)

def _new_boozer_object_with_auto_fallback(
    primary_backend: Any, vmec_fname: str | Path, nc_obj: Any
) -> Any:
    """Create a Boozer transform object, using the classic reader if needed.

    Some VMEC-JAX WOUT files expose a square ``(radius, mode)`` layout that old
    ``booz_xform_jax`` readers reject as ambiguous. In automatic backend mode,
    the imported-geometry path can safely fall back to the classic
    ``booz_xform`` reader; explicit backend selections remain fail-fast.
    """

    try:
        return _new_booz_object(primary_backend, str(vmec_fname))
    except Exception as exc:
        auto_backend = os.environ.get("SPECTRAX_BOOZ_BACKEND", "auto").strip().lower()
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
class _VMECFieldlineScalars:
    s: np.ndarray
    ns: int
    alpha_arr: np.ndarray
    d_pressure_d_s: np.ndarray
    iota: np.ndarray
    d_iota_d_s: np.ndarray
    shat: np.ndarray
    nfp: int
    edge_toroidal_flux_over_2pi: float
    toroidal_flux_sign: float
    L_reference: float
    B_reference: float
    R_mag_ax: float
    zeta_center: float
    iota_input_val: float
    s_hat_input_val: float
    G: np.ndarray
    boozer_i: np.ndarray


@dataclass(frozen=True)
class _BoozerFieldlineSamples:
    xm_b: np.ndarray
    xn_b: np.ndarray
    rmnc_b: np.ndarray
    zmns_b: np.ndarray
    numns_b: np.ndarray
    d_rmnc_b_d_s: np.ndarray
    d_zmns_b_d_s: np.ndarray
    d_numns_b_d_s: np.ndarray
    gmnc_b: np.ndarray
    bmnc_b: np.ndarray
    d_bmnc_b_d_s: np.ndarray
    theta_b: np.ndarray
    phi_b: np.ndarray
    flipit: bool
    tensors: Any
    R_b: np.ndarray
    Z_b: np.ndarray
    nu_b: np.ndarray
    Vprime: np.ndarray
    mnmax_b: int


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


@dataclass(frozen=True)
class _HNGCModeCorrections:
    beta_b: np.ndarray
    lambda_b: np.ndarray
    lambmnc_b: np.ndarray


@dataclass(frozen=True)
class _FieldlineMetricGeometry:
    gradients: Any
    alpha_gradients: Any


@dataclass(frozen=True)
class _FieldlineHNGCIntegrals:
    D1: float
    D2: float
    intinv_g: np.ndarray
    int_lam_div_g: np.ndarray


@dataclass(frozen=True)
class _FieldlineShearFactors:
    d_iota_d_s_1: np.ndarray
    d_pressure_d_s_1: np.ndarray
    sfac: float
    pfac: float


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


def _hngc_mode_corrections(
    scalars: _VMECFieldlineScalars, samples: _BoozerFieldlineSamples
) -> _HNGCModeCorrections:
    """Compute Lambda/beta Boozer-mode corrections for local equilibrium."""

    delmnc_b = np.zeros((scalars.ns, samples.mnmax_b))
    lambmnc_b = np.zeros((scalars.ns, samples.mnmax_b))
    betamns_b = np.zeros((scalars.ns, samples.mnmax_b))
    safe_denom_mn = _safe_mode_denominator(samples.xm_b, samples.xn_b, scalars.iota)
    delmnc_b[:, 1:] = samples.gmnc_b[:, 1:] / samples.Vprime[:, None]
    betamns_b[:, 1:] = (
        delmnc_b[:, 1:]
        / scalars.edge_toroidal_flux_over_2pi
        * _MU_0
        * scalars.d_pressure_d_s[:, None]
        * samples.Vprime[:, None]
        / safe_denom_mn
    )
    lambmnc_b[:, 1:] = (
        delmnc_b[:, 1:]
        * (samples.xm_b[1:] * scalars.G[:, None] + samples.xn_b[1:] * scalars.boozer_i[:, None])
        / (
            safe_denom_mn
            * (scalars.G[:, None] + scalars.iota[:, None] * scalars.boozer_i[:, None])
        )
    )
    angle_b = _boozer_mode_angle(
        samples.xm_b,
        samples.xn_b,
        samples.theta_b,
        samples.phi_b,
        flipit=samples.flipit,
    )
    cosangle_b, sinangle_b, *_ = _boozer_trig_basis(samples.xm_b, samples.xn_b, angle_b)
    return _HNGCModeCorrections(
        beta_b=_boozer_mode_sum(betamns_b, sinangle_b),
        lambda_b=_boozer_mode_sum(lambmnc_b, cosangle_b),
        lambmnc_b=lambmnc_b,
    )


def _fieldline_metric_geometry(
    scalars: _VMECFieldlineScalars,
    samples: _BoozerFieldlineSamples,
) -> _FieldlineMetricGeometry:
    """Return coordinate gradients needed for field-line metric coefficients."""

    cartesian = _fieldline_cartesian_derivatives(
        tensors=samples.tensors, phi_b=samples.phi_b
    )
    gradients = _fieldline_coordinate_gradients(
        tensors=samples.tensors,
        cartesian=cartesian,
        edge_toroidal_flux_over_2pi=scalars.edge_toroidal_flux_over_2pi,
    )
    alpha_gradients = _fieldline_alpha_gradients(
        gradients=gradients,
        phi_b=samples.phi_b,
        zeta_center=scalars.zeta_center,
        d_iota_d_s=scalars.d_iota_d_s,
        iota=scalars.iota,
        edge_toroidal_flux_over_2pi=scalars.edge_toroidal_flux_over_2pi,
    )
    return _FieldlineMetricGeometry(gradients=gradients, alpha_gradients=alpha_gradients)


def _fieldline_hngc_integrals(
    samples: _BoozerFieldlineSamples,
    hngc: _HNGCModeCorrections,
    alpha_gradients: Any,
    *,
    res_theta: int,
    res_phi: int,
) -> _FieldlineHNGCIntegrals:
    """Return flux-surface and field-line HNGC integrals."""

    D1, D2 = _flux_surface_hngc_averages(
        xm_b=samples.xm_b,
        xn_b=samples.xn_b,
        flipit=samples.flipit,
        lambmnc_b=hngc.lambmnc_b,
        rmnc_b=samples.rmnc_b,
        zmns_b=samples.zmns_b,
        numns_b=samples.numns_b,
        gmnc_b=samples.gmnc_b,
        res_theta=res_theta,
        res_phi=res_phi,
    )
    theta_1d = samples.theta_b[0, 0]
    intinv_g = _centered_fieldline_integral(
        1.0 / alpha_gradients.g_sup_psi_psi, samples.phi_b, theta_1d
    )
    int_lam_div_g = _centered_fieldline_integral(
        hngc.lambda_b / alpha_gradients.g_sup_psi_psi,
        samples.phi_b,
        theta_1d,
    )
    return _FieldlineHNGCIntegrals(
        D1=D1,
        D2=D2,
        intinv_g=intinv_g,
        int_lam_div_g=int_lam_div_g,
    )


def _fieldline_shear_factors(
    scalars: _VMECFieldlineScalars,
    *,
    s_val: float,
    betaprim: float,
    include_shear_variation: bool,
    include_pressure_variation: bool,
) -> _FieldlineShearFactors:
    """Return HNGC shear and pressure correction factors."""

    d_iota_d_s_1, sfac = _hngc_shear_correction(
        s_val=s_val,
        iota=scalars.iota,
        shat=scalars.shat,
        iota_input_val=scalars.iota_input_val,
        s_hat_input_val=scalars.s_hat_input_val,
        include_shear_variation=include_shear_variation,
    )
    d_pressure_d_s_1, pfac = _hngc_pressure_correction(
        s_val=s_val,
        betaprim=betaprim,
        B_reference=scalars.B_reference,
        d_pressure_d_s=scalars.d_pressure_d_s,
        include_pressure_variation=include_pressure_variation,
    )
    return _FieldlineShearFactors(
        d_iota_d_s_1=d_iota_d_s_1,
        d_pressure_d_s_1=d_pressure_d_s_1,
        sfac=sfac,
        pfac=pfac,
    )


def _fieldline_shear(
    scalars: _VMECFieldlineScalars,
    samples: _BoozerFieldlineSamples,
    geometry: _FieldlineMetricGeometry,
    integrals: _FieldlineHNGCIntegrals,
    factors: _FieldlineShearFactors,
) -> Any:
    """Return the local shear object used by metric/drift assembly."""

    return _fieldline_local_shear(
        edge_toroidal_flux_over_2pi=scalars.edge_toroidal_flux_over_2pi,
        d_iota_d_s=scalars.d_iota_d_s,
        d_iota_d_s_1=factors.d_iota_d_s_1,
        d_pressure_d_s_1=factors.d_pressure_d_s_1,
        Vprime=samples.Vprime,
        G=scalars.G,
        iota=scalars.iota,
        boozer_i=scalars.boozer_i,
        phi_b=samples.phi_b,
        zeta_center=scalars.zeta_center,
        intinv_g=integrals.intinv_g,
        int_lam_div_g=integrals.int_lam_div_g,
        D1=integrals.D1,
        D2=integrals.D2,
        g_sup_psi_psi=geometry.alpha_gradients.g_sup_psi_psi,
        grad_alpha_dot_grad_psi=geometry.alpha_gradients.grad_alpha_dot_grad_psi,
    )


def _fieldline_metric_coefficients(
    scalars: _VMECFieldlineScalars,
    samples: _BoozerFieldlineSamples,
    hngc: _HNGCModeCorrections,
    *,
    s_val: float,
    betaprim: float,
    include_shear_variation: bool,
    include_pressure_variation: bool,
    res_theta: int,
    res_phi: int,
) -> Any:
    """Assemble local shear, HNGC corrections, and metric/drift coefficients."""

    geometry = _fieldline_metric_geometry(scalars, samples)
    integrals = _fieldline_hngc_integrals(
        samples,
        hngc,
        geometry.alpha_gradients,
        res_theta=res_theta,
        res_phi=res_phi,
    )
    factors = _fieldline_shear_factors(
        scalars,
        s_val=s_val,
        betaprim=betaprim,
        include_shear_variation=include_shear_variation,
        include_pressure_variation=include_pressure_variation,
    )
    shear = _fieldline_shear(scalars, samples, geometry, integrals, factors)
    return _fieldline_metric_drifts(
        tensors=samples.tensors,
        gradients=geometry.gradients,
        alpha_gradients=geometry.alpha_gradients,
        shear=shear,
        s=scalars.s,
        shat=scalars.shat,
        sfac=factors.sfac,
        pfac=factors.pfac,
        L_reference=scalars.L_reference,
        B_reference=scalars.B_reference,
        toroidal_flux_sign=scalars.toroidal_flux_sign,
        edge_toroidal_flux_over_2pi=scalars.edge_toroidal_flux_over_2pi,
        d_pressure_d_s=scalars.d_pressure_d_s,
        beta_b=hngc.beta_b,
        G=scalars.G,
        iota=scalars.iota,
        boozer_i=scalars.boozer_i,
    )


def _assemble_fieldline_struct(
    scalars: _VMECFieldlineScalars,
    samples: _BoozerFieldlineSamples,
    coeffs: Any,
    *,
    s_val: float,
    alpha: float,
    betaprim: float,
) -> _Struct:
    theta_pest = samples.theta_b - scalars.iota[:, None, None] * samples.nu_b
    theta_geo = np.arctan2(samples.Z_b, samples.R_b - scalars.R_mag_ax)
    return _Struct(
        iota_input=scalars.iota_input_val,
        d_iota_d_s=scalars.d_iota_d_s,
        d_pressure_d_s=scalars.d_pressure_d_s,
        s_hat_input=scalars.s_hat_input_val,
        alpha=alpha,
        theta_b=samples.theta_b,
        phi_b=samples.phi_b,
        theta_PEST=theta_pest,
        theta_geo=theta_geo,
        edge_toroidal_flux_over_2pi=scalars.edge_toroidal_flux_over_2pi,
        R_b=samples.R_b,
        Z_b=samples.Z_b,
        betaprim=betaprim,
        bmag=coeffs.bmag,
        gradpar_theta_b=coeffs.gradpar_theta_b,
        gradpar_phi=coeffs.gradpar_phi,
        gds2=coeffs.gds2,
        gds21=coeffs.gds21,
        gds22=coeffs.gds22,
        gbdrift=coeffs.gbdrift,
        gbdrift0=coeffs.gbdrift0,
        cvdrift=coeffs.cvdrift,
        cvdrift0=coeffs.cvdrift0,
        grho=coeffs.grho,
        grad_y=coeffs.grad_y,
        grad_x=coeffs.grad_x,
        zeta_center=scalars.zeta_center,
        nfp=scalars.nfp,
        L_reference=scalars.L_reference,
        B_reference=scalars.B_reference,
        dpsidrho=2.0 * np.sqrt(s_val) * scalars.edge_toroidal_flux_over_2pi,
    )


def _vmec_fieldlines(
    vmec_fname: str | Path,
    s_val: float,
    betaprim: float,
    alpha: float,
    include_shear_variation: bool,
    include_pressure_variation: bool,
    theta1d: np.ndarray,
    isaxisym: bool,
    iota_input: float | None = None,
    s_hat_input: float | None = None,
    res_theta: int = 201,
    res_phi: int = 201,
) -> _Struct:
    """Compute VMEC flux-tube geometry coefficients from a VMEC equilibrium.

    Evaluate field-line geometry from VMEC and Boozer spline data.

    Parameters
    ----------
    vmec_fname:
        Path to a VMEC ``wout_*.nc`` file.
    s_val:
        Normalised toroidal flux (normalized toroidal-flux input).
    betaprim:
        Effective beta prime for pressure gradient variation.
    alpha:
        Field-line label (Boozer alpha = theta_b - iota * phi_b).
    include_shear_variation / include_pressure_variation:
        Whether to apply Hegna-Nakajima local-equilibrium corrections.
    theta1d:
        1-D Boozer-theta array defining the field line.
    isaxisym:
        Whether the equilibrium is axisymmetric (enables flipit logic).
    iota_input / s_hat_input:
        Override values; *None* means use VMEC values.
    res_theta / res_phi:
        Resolution of the 2-D (theta, phi) grid used for flux-surface
        integrals D1 and D2.
    """

    nc_obj, vs = _load_vmec_boozer_splines(vmec_fname)
    try:
        scalars = _fieldline_scalar_profiles(
            vs,
            s_val=s_val,
            alpha=alpha,
            iota_input=iota_input,
            s_hat_input=s_hat_input,
        )
        samples = _sample_fieldline_boozer_state(
            vs,
            scalars,
            theta1d=theta1d,
            isaxisym=isaxisym,
        )
        hngc = _hngc_mode_corrections(scalars, samples)
        coeffs = _fieldline_metric_coefficients(
            scalars,
            samples,
            hngc,
            s_val=s_val,
            betaprim=betaprim,
            include_shear_variation=include_shear_variation,
            include_pressure_variation=include_pressure_variation,
            res_theta=res_theta,
            res_phi=res_phi,
        )
        return _assemble_fieldline_struct(
            scalars,
            samples,
            coeffs,
            s_val=s_val,
            alpha=alpha,
            betaprim=betaprim,
        )
    finally:
        nc_obj.close()


def write_vmec_eik_netcdf(
    path: Path,
    profiles: dict[str, Any],
    *,
    request: Any,
) -> None:
    """Write an imported VMEC ``*.eik.nc`` file.

    The output format matches the imported-geometry NetCDF contract.
    Drift coefficients are stored at the pre-loader-normalization (2x) level;
    the loader (``load_imported_geometry_netcdf``) divides by 2 on read.
    """

    try:
        import netCDF4 as _nc
    except ImportError as exc:
        raise ImportError("netCDF4 is required for VMEC eik writeout") from exc

    theta = np.asarray(profiles["theta"], dtype=float)
    ntheta = int(theta.shape[0])

    dpsidrho = float(profiles["dpsidrho"])
    drhodpsi = 1.0 / abs(dpsidrho) if abs(dpsidrho) > 1.0e-30 else 1.0e30
    gradpar_val = float(profiles["gradpar"][0])
    bmag = np.asarray(profiles["bmag"], dtype=float)
    jacob = 1.0 / np.abs(drhodpsi * gradpar_val * bmag)

    with _nc.Dataset(path, "w") as ds:
        ds.createDimension("z", ntheta)
        ds.createDimension("3", 3)

        def _v(name: str, dtype: str = "f8", dims: tuple = ("z",)) -> Any:
            return ds.createVariable(name, dtype, dims)

        _v("theta")[:] = theta
        _v("theta_PEST")[:] = np.asarray(profiles["theta_PEST"], dtype=float)
        _v("bmag")[:] = bmag
        _v("gradpar")[:] = np.asarray(profiles["gradpar"], dtype=float)
        _v("grho")[:] = np.asarray(profiles["grho"], dtype=float)
        _v("gds2")[:] = np.asarray(profiles["gds2"], dtype=float)
        _v("gds21")[:] = np.asarray(profiles["gds21"], dtype=float)
        _v("gds22")[:] = np.asarray(profiles["gds22"], dtype=float)
        _v("gbdrift")[:] = np.asarray(profiles["gbdrift"], dtype=float)
        _v("gbdrift0")[:] = np.asarray(profiles["gbdrift0"], dtype=float)
        _v("cvdrift")[:] = np.asarray(profiles["cvdrift"], dtype=float)
        _v("cvdrift0")[:] = np.asarray(profiles["cvdrift0"], dtype=float)
        _v("jacob")[:] = jacob
        _v("Rplot")[:] = np.asarray(profiles["Rplot"], dtype=float)
        _v("Zplot")[:] = np.asarray(profiles["Zplot"], dtype=float)
        _v("grad_x", dims=("3", "z"))[:, :] = np.asarray(
            profiles["grad_x"], dtype=float
        )
        _v("grad_y", dims=("3", "z"))[:, :] = np.asarray(
            profiles["grad_y"], dtype=float
        )
        _v("b_vec", dims=("3", "z"))[:, :] = np.asarray(profiles["b_vec"], dtype=float)

        ds.createVariable("drhodpsi", "f8").assignValue(drhodpsi)
        ds.createVariable("kxfac", "f8").assignValue(float(profiles.get("kxfac", 1.0)))
        ds.createVariable("Rmaj", "f8").assignValue(float(profiles["Rmaj"]))
        ds.createVariable("q", "f8").assignValue(float(profiles["q"]))
        ds.createVariable("shat", "f8").assignValue(float(profiles["shat"]))
        ds.createVariable("scale", "f8").assignValue(float(profiles["scale"]))
        ds.createVariable("alpha", "f8").assignValue(float(profiles["alpha"]))
        ds.createVariable("zeta_center", "f8").assignValue(
            float(profiles["zeta_center"])
        )
        ds.createVariable("nfp", "i4").assignValue(int(profiles["nfp"]))


def _write_vmec_eik_netcdf_atomically(
    path: Path,
    profiles: dict[str, Any],
    *,
    request: Any,
) -> None:
    """Write a VMEC eik file through a unique temp file, then atomically replace.

    W7-X validation sweeps commonly launch multiple identical VMEC geometry
    requests at once. Writing the shared cache file directly can expose a
    partially written netCDF to another process. A per-process temp path keeps
    the final cache path all-or-nothing.
    """

    final_path = Path(path).expanduser().resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(
        f".{final_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        write_vmec_eik_netcdf(temp_path, profiles, request=request)
        os.replace(temp_path, final_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@dataclass(frozen=True)
class _FluxTubeCutRequest:
    """User/runtime policy for selecting a symmetric VMEC flux-tube cut."""

    flux_tube_cut: str
    npol_min: float | None
    which_crossing: int
    y0: float
    x0: float
    jtwist_in: int | None
    jtwist_max: int | None


@dataclass(frozen=True)
class _FluxTubeSamples:
    """One-dimensional VMEC arrays sampled on the candidate theta line."""

    bmag: np.ndarray
    gradpar: np.ndarray
    cvdrift: np.ndarray
    gbdrift: np.ndarray
    cvdrift0: np.ndarray
    gbdrift0: np.ndarray
    gds2: np.ndarray
    gds21: np.ndarray
    gds22: np.ndarray
    grho: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray


def _vmec_line_samples(geo: _Struct) -> _FluxTubeSamples:
    """Extract the single field-line arrays used by the VMEC remapping stage."""

    def _sl(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr[0, 0])

    return _FluxTubeSamples(
        bmag=_sl(geo.bmag),
        gradpar=np.abs(_sl(geo.gradpar_theta_b)),
        cvdrift=_sl(geo.cvdrift),
        gbdrift=_sl(geo.gbdrift),
        cvdrift0=_sl(geo.cvdrift0),
        gbdrift0=_sl(geo.gbdrift0),
        gds2=_sl(geo.gds2),
        gds21=_sl(geo.gds21),
        gds22=_sl(geo.gds22),
        grho=_sl(geo.grho),
        R=_sl(geo.R_b),
        Z=_sl(geo.Z_b),
        grad_x=geo.grad_x[:, 0, 0, :],
        grad_y=geo.grad_y[:, 0, 0, :],
    )


def _unit_b_from_gradients(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    """Construct the normalized magnetic-field direction from gradient vectors."""

    bv = np.cross(grad_x, grad_y, axis=0)
    bv_norm = np.linalg.norm(bv, axis=0, keepdims=True)
    bv_norm = np.where(bv_norm < 1.0e-300, 1.0e-300, bv_norm)
    return bv / bv_norm


def _remap_flux_tube_samples(
    theta: np.ndarray,
    theta_cut: np.ndarray,
    samples: _FluxTubeSamples,
) -> dict[str, np.ndarray]:
    """Interpolate VMEC field-line arrays onto the selected cut grid."""

    def _interp(arr: np.ndarray) -> np.ndarray:
        spl = InterpolatedUnivariateSpline(theta, arr)
        return spl(theta_cut)

    grad_x_arr = np.array(
        [
            InterpolatedUnivariateSpline(theta, samples.grad_x[i])(theta_cut)
            for i in range(3)
        ]
    )
    grad_y_arr = np.array(
        [
            InterpolatedUnivariateSpline(theta, samples.grad_y[i])(theta_cut)
            for i in range(3)
        ]
    )
    return {
        "theta": theta_cut,
        "theta_PEST": theta_cut,
        "bmag": _interp(samples.bmag),
        "gradpar": _interp(samples.gradpar),
        "cvdrift": _interp(samples.cvdrift),
        "gbdrift": _interp(samples.gbdrift),
        "cvdrift0": _interp(samples.cvdrift0),
        "gbdrift0": _interp(samples.gbdrift0),
        "gds2": _interp(samples.gds2),
        "gds21": _interp(samples.gds21),
        "gds22": _interp(samples.gds22),
        "grho": _interp(samples.grho),
        "Rplot": _interp(samples.R),
        "Zplot": _interp(samples.Z),
        "grad_x": grad_x_arr,
        "grad_y": grad_y_arr,
        "b_vec": _unit_b_from_gradients(grad_x_arr, grad_y_arr),
    }


def _select_flux_tube_crossing(
    crossings: np.ndarray,
    *,
    label: str,
    request: _FluxTubeCutRequest,
) -> float:
    """Pick the requested positive crossing after applying npol filtering."""

    crossings = np.asarray(crossings, dtype=float)
    crossings = np.sort(crossings[np.isfinite(crossings) & (crossings > 0.0)])
    if request.npol_min is not None:
        crossings = crossings[crossings > request.npol_min * np.pi]
    if crossings.size == 0:
        raise ValueError(
            f"No positive {label} flux-tube crossing was found for "
            f"flux_tube_cut={request.flux_tube_cut!r}, "
            f"npol_min={request.npol_min!r}. "
            "Try a different flux_tube_cut, npol_min, jtwist_in, or a larger "
            "theta/npol search range for this VMEC equilibrium."
        )
    try:
        return float(crossings[request.which_crossing])
    except IndexError as exc:
        raise ValueError(
            f"Requested which_crossing={request.which_crossing} for "
            f"flux_tube_cut={request.flux_tube_cut!r}, but only {crossings.size} "
            f"positive {label} crossings were found."
        ) from exc


def _aspect_cut_candidates(
    theta: np.ndarray,
    samples: _FluxTubeSamples,
    geo: _Struct,
    request: _FluxTubeCutRequest,
) -> np.ndarray:
    """Return candidate theta roots for the aspect-ratio twist cut."""

    jtwist_arr = 2.0 * geo.s_hat_input * samples.gds21 / samples.gds22
    jtwist_line = jtwist_arr / request.y0 * request.x0
    jtwist_spl = CubicSpline(theta, jtwist_line)
    if request.jtwist_in is not None:
        candidates = [float(-request.jtwist_in), float(request.jtwist_in)]
    else:
        n_max = request.jtwist_max if request.jtwist_max is not None else 30
        candidates = [v for v in range(-n_max, n_max + 1) if v != 0]
    return np.concatenate(
        [jtwist_spl.solve(float(v), extrapolate=False) for v in candidates]
    )


def _solve_flux_tube_cut(
    theta: np.ndarray,
    samples: _FluxTubeSamples,
    geo: _Struct,
    request: _FluxTubeCutRequest,
) -> float:
    """Solve the positive theta crossing selected by the flux-tube cut policy."""

    if request.flux_tube_cut == "gds21":
        tck = splrep(theta, samples.gds21, s=0)
        ppoly = PPoly.from_spline(tck)
        return _select_flux_tube_crossing(
            ppoly.roots(extrapolate=False),
            label="gds21",
            request=request,
        )
    if request.flux_tube_cut == "gbdrift0":
        tck = splrep(theta, samples.gbdrift0, s=0)
        ppoly = PPoly.from_spline(tck)
        return _select_flux_tube_crossing(
            ppoly.roots(extrapolate=False),
            label="gbdrift0",
            request=request,
        )
    if request.flux_tube_cut == "aspect":
        return _select_flux_tube_crossing(
            _aspect_cut_candidates(theta, samples, geo, request),
            label="jtwist",
            request=request,
        )
    raise ValueError(f"Unknown flux_tube_cut={request.flux_tube_cut!r}")


def _apply_flux_tube_cut(
    theta: np.ndarray,
    geo: _Struct,
    ntheta: int,
    flux_tube_cut: str,
    npol_min: float | None,
    which_crossing: int,
    y0: float,
    x0: float,
    jtwist_in: int | None,
    jtwist_max: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply flux-tube cut and return (theta_cut, arrays_cut).

    Apply the flux-tube cut used by the imported-geometry pipeline.
    """

    samples = _vmec_line_samples(geo)
    request = _FluxTubeCutRequest(
        flux_tube_cut=flux_tube_cut,
        npol_min=npol_min,
        which_crossing=which_crossing,
        y0=y0,
        x0=x0,
        jtwist_in=jtwist_in,
        jtwist_max=jtwist_max,
    )

    if flux_tube_cut == "none":
        return theta, _remap_flux_tube_samples(theta, theta, samples)

    cut = _solve_flux_tube_cut(theta, samples, geo, request)
    theta_cut = np.linspace(-cut, cut, ntheta)
    return theta_cut, _remap_flux_tube_samples(theta, theta_cut, samples)


def _equal_arc_remap(
    theta: np.ndarray,
    arrays: dict[str, np.ndarray],
    ntheta: int,
) -> tuple[float, dict[str, Any]]:
    """Remap all geometry arrays from the cut theta grid to equal-arc theta.

    Apply the equal-arc remap used by the imported-geometry pipeline.

    Returns ``(gradpar_eqarc, remapped_arrays_dict)`` where
    ``remapped_arrays_dict["scale"]`` holds the domain scaling factor.
    """

    gradpar = arrays["gradpar"]
    inv_gradpar_int = _ctrap(1.0 / gradpar, theta, initial=0)
    gradpar_eqarc = 2.0 * np.pi / float(inv_gradpar_int[-1])
    theta_eqarc = gradpar_eqarc * inv_gradpar_int - np.pi
    domain_scaling_factor = float(theta[-1]) / float(theta_eqarc[-1])

    theta_out = np.linspace(-np.pi, np.pi, ntheta)

    def _interp(arr: np.ndarray) -> np.ndarray:
        return np.interp(theta_out, theta_eqarc, arr)

    grad_x_arr = np.array([_interp(arrays["grad_x"][i]) for i in range(3)])
    grad_y_arr = np.array([_interp(arrays["grad_y"][i]) for i in range(3)])
    bv = np.cross(grad_x_arr, grad_y_arr, axis=0)
    bv_norm = np.linalg.norm(bv, axis=0, keepdims=True)
    bv_norm = np.where(bv_norm < 1.0e-300, 1.0e-300, bv_norm)

    out: dict[str, Any] = {
        "theta": theta_out,
        "theta_PEST": arrays["theta_PEST"],
        "bmag": _interp(arrays["bmag"]),
        "gradpar": gradpar_eqarc * np.ones(ntheta),
        "cvdrift": _interp(arrays["cvdrift"]),
        "gbdrift": _interp(arrays["gbdrift"]),
        "cvdrift0": _interp(arrays["cvdrift0"]),
        "gbdrift0": _interp(arrays["gbdrift0"]),
        "gds2": _interp(arrays["gds2"]),
        "gds21": _interp(arrays["gds21"]),
        "gds22": _interp(arrays["gds22"]),
        "grho": _interp(arrays["grho"]),
        "Rplot": _interp(arrays["Rplot"]),
        "Zplot": _interp(arrays["Zplot"]),
        "grad_x": grad_x_arr,
        "grad_y": grad_y_arr,
        "b_vec": bv / bv_norm,
        "scale": domain_scaling_factor,
    }
    return gradpar_eqarc, out


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


__all__ = [
    "_Struct",
    "_apply_flux_tube_cut",
    "_booz_read_wout_square_layout_failure",
    "_booz_xform_jax_search_paths",
    "_equal_arc_remap",
    "_import_booz_backend",
    "_import_booz_xform_backend",
    "_import_booz_xform_jax_backend",
    "_import_module_with_search_paths",
    "_new_booz_object",
    "_vmec_fieldlines",
    "_vmec_splines",
    "_write_vmec_eik_netcdf_atomically",
    "dermv",
    "generate_vmec_eik_internal",
    "internal_vmec_backend_available",
    "nperiod_set",
    "write_vmec_eik_netcdf",
]
