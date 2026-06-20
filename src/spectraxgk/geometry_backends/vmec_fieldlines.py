"""VMEC/Boozer field-line geometry assembly."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import numpy as np

from spectraxgk.geometry_backends.vmec_backend_discovery import (
    _booz_read_wout_square_layout_failure,
    _import_booz_backend,
    _new_booz_object,
)
from spectraxgk.geometry_backends.vmec_splines import _vmec_splines
from spectraxgk.geometry_backends.vmec_types import _Struct

from spectraxgk.geometry_backends.vmec_fieldline_numerics import (
    _MU_0,
    _axisym_flip_required,
    _boozer_mode_angle,
    _boozer_mode_sum,
    _boozer_trig_basis,
    _centered_fieldline_integral,
    _fieldline_alpha_gradients,
    _fieldline_boozer_coordinates,
    _fieldline_boozer_tensors,
    _fieldline_cartesian_derivatives,
    _fieldline_coordinate_gradients,
    _fieldline_local_shear,
    _fieldline_metric_drifts,
    _flux_surface_hngc_averages,
    _hngc_pressure_correction,
    _hngc_shear_correction,
    _input_iota_shear,
    _sample_boozer_mode_table,
    _safe_mode_denominator,
    _surface_average_2d,  # noqa: F401 - re-exported for helper-level tests.
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
    theta_b, phi_b = _fieldline_boozer_coordinates(
        theta1d, scalars.alpha_arr, scalars.iota
    )
    flipit = _axisym_flip_required(
        isaxisym=isaxisym,
        xm_b=xm_b,
        xn_b=xn_b,
        theta_b=theta_b,
        phi_b=phi_b,
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
    )
    angle_b = _boozer_mode_angle(xm_b, xn_b, theta_b, phi_b, flipit=flipit)
    (
        cosangle_b,
        sinangle_b,
        mcosangle_b,
        msinangle_b,
        ncosangle_b,
        nsinangle_b,
    ) = _boozer_trig_basis(xm_b, xn_b, angle_b)
    tensors = _fieldline_boozer_tensors(
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
        numns_b=numns_b,
        d_rmnc_b_d_s=d_rmnc_b_d_s,
        d_zmns_b_d_s=d_zmns_b_d_s,
        d_numns_b_d_s=d_numns_b_d_s,
        gmnc_b=gmnc_b,
        bmnc_b=bmnc_b,
        d_bmnc_b_d_s=d_bmnc_b_d_s,
        cosangle_b=cosangle_b,
        sinangle_b=sinangle_b,
        mcosangle_b=mcosangle_b,
        msinangle_b=msinangle_b,
        ncosangle_b=ncosangle_b,
        nsinangle_b=nsinangle_b,
    )
    return _BoozerFieldlineSamples(
        xm_b=xm_b,
        xn_b=xn_b,
        rmnc_b=rmnc_b,
        zmns_b=zmns_b,
        numns_b=numns_b,
        d_rmnc_b_d_s=d_rmnc_b_d_s,
        d_zmns_b_d_s=d_zmns_b_d_s,
        d_numns_b_d_s=d_numns_b_d_s,
        gmnc_b=gmnc_b,
        bmnc_b=bmnc_b,
        d_bmnc_b_d_s=d_bmnc_b_d_s,
        theta_b=theta_b,
        phi_b=phi_b,
        flipit=bool(flipit),
        tensors=tensors,
        R_b=tensors.R_b,
        Z_b=tensors.Z_b,
        nu_b=tensors.nu_b,
        Vprime=gmnc_b[:, 0],
        mnmax_b=rmnc_b.shape[1],
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
