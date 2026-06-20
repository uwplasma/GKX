"""Profile assembly for the internal Miller imported-geometry backend."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectraxgk.geometry_backends.miller_core import MillerCoreParams
from spectraxgk.geometry_backends.miller_numerics import (
    _safe_denom,
    cumulative_trapezoid,
    derm,
    nperiod_data_extend,
    to_ballooning,
)


@dataclass(frozen=True)
class _MillerCenterData:
    """Central-surface Miller arrays and scalar normalizations."""

    rho: np.ndarray
    qfac: np.ndarray
    r: np.ndarray
    z: np.ndarray
    dl: np.ndarray
    u_ml: np.ndarray
    r_c: np.ndarray
    dpsi_dr: np.ndarray
    bpol: np.ndarray
    bmag: np.ndarray
    psi_diff: np.ndarray
    diffq: np.ndarray
    diffrho: np.ndarray
    dpsidrho: float
    drhodpsi: float
    dpdpsi: float
    f_const: float
    center: int = 1


@dataclass(frozen=True)
class _MillerExtendedData:
    """Central-surface profiles extended over the requested field periods."""

    r: np.ndarray
    z: np.ndarray
    dl: np.ndarray
    b: np.ndarray
    b2: np.ndarray
    bpol: np.ndarray
    bpol_geo: np.ndarray
    dpsi_dr: np.ndarray
    u_ml: np.ndarray
    r_c: np.ndarray


@dataclass(frozen=True)
class _BishopProfiles:
    """Bishop integral coefficients used by metric and drift formulas."""

    a_s: np.ndarray
    b_s: np.ndarray
    c_s: np.ndarray
    dfdpsi: float
    dqdr: np.ndarray
    aprime: np.ndarray


@dataclass(frozen=True)
class _MetricProfiles:
    """Extended metric coefficients before interpolation to the target grid."""

    gds2: np.ndarray
    gds21: np.ndarray
    gds22: np.ndarray


@dataclass(frozen=True)
class _DriftProfiles:
    """Extended radial-gradient and magnetic-drift coefficients."""

    grho: np.ndarray
    gbdrift: np.ndarray
    cvdrift: np.ndarray
    gbdrift0: np.ndarray


def _collect_center_data(
    *,
    params: MillerCoreParams,
    state: dict[str, np.ndarray | float],
    straight_state: dict[str, np.ndarray],
    dpsidrho: float,
) -> _MillerCenterData:
    """Collect central Miller arrays and derivative normalizations."""

    rho = np.asarray(state["rho"], dtype=float)
    qfac = np.asarray(state["qfac"], dtype=float)
    r = np.asarray(state["r"], dtype=float)
    z = np.asarray(state["z"], dtype=float)
    drhodpsi = 1.0 / float(dpsidrho)
    dpdpsi = 0.5 * float(params.betaprim) * drhodpsi
    return _MillerCenterData(
        rho=rho,
        qfac=qfac,
        r=r,
        z=z,
        dl=np.asarray(straight_state["dl"], dtype=float),
        u_ml=np.asarray(straight_state["u_ml"], dtype=float),
        r_c=np.asarray(straight_state["r_c"], dtype=float),
        dpsi_dr=np.asarray(straight_state["dpsi_dr"], dtype=float),
        bpol=np.asarray(straight_state["bpol"], dtype=float),
        bmag=np.asarray(straight_state["bmag"], dtype=float),
        psi_diff=np.asarray(straight_state["psi_diff"], dtype=float),
        diffq=derm(qfac, "r"),
        diffrho=derm(rho, "r"),
        dpsidrho=float(dpsidrho),
        drhodpsi=drhodpsi,
        dpdpsi=dpdpsi,
        f_const=float(params.r_geo),
    )


def _extend_center_data(
    params: MillerCoreParams,
    center: _MillerCenterData,
    *,
    bpol_center: np.ndarray,
) -> _MillerExtendedData:
    """Extend central-surface profiles with the imported-geometry parity rules."""

    c = center.center
    b = nperiod_data_extend(center.bmag[c], params.nperiod, istheta=0, par="e")
    return _MillerExtendedData(
        r=nperiod_data_extend(center.r[c], params.nperiod, istheta=0, par="e"),
        z=nperiod_data_extend(center.z[c], params.nperiod, istheta=0, par="o"),
        dl=nperiod_data_extend(center.dl[c], params.nperiod, istheta=0, par="e"),
        b=b,
        b2=b * b,
        bpol=nperiod_data_extend(
            np.abs(center.bpol[c]), params.nperiod, istheta=0, par="e"
        ),
        bpol_geo=nperiod_data_extend(
            np.abs(bpol_center), params.nperiod, istheta=0, par="e"
        ),
        dpsi_dr=nperiod_data_extend(
            center.dpsi_dr[c], params.nperiod, istheta=0, par="e"
        ),
        u_ml=nperiod_data_extend(center.u_ml[c], params.nperiod, istheta=0, par="e"),
        r_c=nperiod_data_extend(center.r_c[c], params.nperiod, istheta=0, par="e"),
    )


def _bishop_profiles(
    params: MillerCoreParams,
    center: _MillerCenterData,
    ext: _MillerExtendedData,
    *,
    theta_st_ex: np.ndarray,
) -> _BishopProfiles:
    """Evaluate Bishop integral coefficients and radial theta derivative."""

    c = center.center
    q = center.qfac[c]
    a_s = -(
        2.0 * q / center.f_const * theta_st_ex
        + 2.0
        * center.f_const
        * q
        * cumulative_trapezoid(1.0 / _safe_denom(ext.r**2 * ext.bpol_geo**2), theta_st_ex)
    )
    b_s = -2.0 * q * cumulative_trapezoid(
        1.0 / _safe_denom(ext.bpol_geo**2), theta_st_ex
    )
    c_s = 2.0 * q * cumulative_trapezoid(
        (2.0 * np.sin(ext.u_ml) / _safe_denom(ext.r) - 2.0 / _safe_denom(ext.r_c))
        * (1.0 / _safe_denom(ext.r * ext.bpol_geo)),
        theta_st_ex,
    )
    prefac = (
        center.rho[c]
        * (center.psi_diff[c] / center.diffrho[c])
        * (1.0 / (2.0 * np.pi * q * (2 * params.nperiod - 1)))
    )
    dfdpsi = (-params.shat / _safe_denom(prefac) - (b_s[-1] * center.dpdpsi - c_s[-1])) / _safe_denom(a_s[-1])
    dqdr = center.diffq[c] * ext.dpsi_dr / center.psi_diff[c]
    aprime = -ext.r * ext.bpol * (a_s * dfdpsi + b_s * center.dpdpsi - c_s) / (
        2.0 * np.abs(center.drhodpsi)
    )
    return _BishopProfiles(
        a_s=a_s,
        b_s=b_s,
        c_s=c_s,
        dfdpsi=float(dfdpsi),
        dqdr=dqdr,
        aprime=aprime,
    )


def _metric_profiles(
    params: MillerCoreParams,
    center: _MillerCenterData,
    ext: _MillerExtendedData,
    bishop: _BishopProfiles,
    *,
    theta_st_ex: np.ndarray,
    straight_state: dict[str, np.ndarray],
) -> _MetricProfiles:
    """Compute extended Miller metric coefficients."""

    c = center.center
    q = center.qfac[c]
    gds22 = (center.diffq[c] / center.diffrho[c]) ** 2 * np.abs(ext.dpsi_dr) ** 2
    gds21 = center.diffq[c] / center.diffrho[c] * (-ext.dpsi_dr) * bishop.aprime
    dt_st_l_dl = nperiod_data_extend(
        np.asarray(straight_state["dt_st_l_dl_center"], dtype=float),
        params.nperiod,
        istheta=0,
        par="e",
    )
    dtdr_st = (bishop.aprime * center.drhodpsi - bishop.dqdr * theta_st_ex) / q
    gds2 = (center.psi_diff[c] / center.diffrho[c]) ** 2 * (
        1.0 / _safe_denom(ext.r**2)
        + (bishop.dqdr * theta_st_ex) ** 2
        + (q**2) * (dtdr_st**2 + dt_st_l_dl**2)
        + 2.0 * q * bishop.dqdr * theta_st_ex * dtdr_st
    )
    return _MetricProfiles(gds2=gds2, gds21=gds21, gds22=gds22)


def _drift_profiles(
    center: _MillerCenterData,
    ext: _MillerExtendedData,
    bishop: _BishopProfiles,
    *,
    dpsidrho: float,
) -> _DriftProfiles:
    """Compute extended grad-rho and magnetic-drift coefficients."""

    d_b2_l = np.asarray(derm(ext.b**2, "l", "e"), dtype=float)
    d_b2_l_over_dl = d_b2_l / _safe_denom(ext.dl)
    gbdrift0 = (
        (1.0 / _safe_denom(ext.b2**2))
        * dpsidrho
        * center.f_const
        / _safe_denom(ext.r)
        * (bishop.dqdr * d_b2_l_over_dl)
    )
    d_bdr_bish = (
        ext.bpol
        / _safe_denom(ext.b)
        * (
            -ext.bpol / _safe_denom(ext.r_c)
            + center.dpdpsi * ext.r
            - center.f_const**2 * np.sin(ext.u_ml) / _safe_denom(ext.r**3 * ext.bpol)
        )
    )
    gbdrift = (
        1.0
        / np.clip(np.abs(center.drhodpsi * ext.b**3), 1.0e-30, None)
        * (
            2.0 * ext.b2 * d_bdr_bish / _safe_denom(ext.dpsi_dr)
            + bishop.aprime
            * center.drhodpsi
            * center.f_const
            / _safe_denom(ext.r)
            * d_b2_l_over_dl
            / _safe_denom(ext.b)
        )
    )
    cvdrift = (
        1.0 / np.clip(np.abs(center.drhodpsi * ext.b**3), 1.0e-30, None)
    ) * (2.0 * ext.b * center.dpdpsi) + gbdrift
    grho = center.drhodpsi * ext.dpsi_dr
    return _DriftProfiles(
        grho=grho,
        gbdrift=gbdrift,
        cvdrift=cvdrift,
        gbdrift0=gbdrift0,
    )


def _interpolate_to_target_grid(
    *,
    theta_target_ex: np.ndarray,
    theta_source_ex: np.ndarray,
    ext: _MillerExtendedData,
    metrics: _MetricProfiles,
    drifts: _DriftProfiles,
) -> dict[str, np.ndarray]:
    """Interpolate extended profiles from source theta to target theta."""

    return {
        "r": np.interp(theta_target_ex, theta_source_ex, ext.r),
        "z": np.interp(theta_target_ex, theta_source_ex, ext.z),
        "b": np.interp(theta_target_ex, theta_source_ex, ext.b),
        "gds2": np.interp(theta_target_ex, theta_source_ex, metrics.gds2),
        "gds21": np.interp(theta_target_ex, theta_source_ex, metrics.gds21),
        "gds22": np.interp(theta_target_ex, theta_source_ex, metrics.gds22),
        "grho": np.interp(theta_target_ex, theta_source_ex, drifts.grho),
        "gbdrift": np.interp(theta_target_ex, theta_source_ex, drifts.gbdrift),
        "cvdrift": np.interp(theta_target_ex, theta_source_ex, drifts.cvdrift),
        "gbdrift0": np.interp(theta_target_ex, theta_source_ex, drifts.gbdrift0),
    }


def _to_ballooning_profiles(
    *,
    theta_target_ex: np.ndarray,
    gradpar_target_ex: np.ndarray,
    interpolated: dict[str, np.ndarray],
    theta_st_ex: np.ndarray,
    aprime: np.ndarray,
) -> dict[str, np.ndarray]:
    """Convert interpolated target-grid profiles to ballooning representation."""

    theta_ball, gradpar_ball = to_ballooning(
        theta_target_ex, gradpar_target_ex, parity="e"
    )
    _tb, b_ball = to_ballooning(theta_target_ex, interpolated["b"], parity="e")
    _tb, gds2_ball = to_ballooning(theta_target_ex, interpolated["gds2"], parity="e")
    _tb, gds21_ball = to_ballooning(theta_target_ex, interpolated["gds21"], parity="o")
    _tb, gds22_ball = to_ballooning(theta_target_ex, interpolated["gds22"], parity="e")
    _tb, grho_ball = to_ballooning(theta_target_ex, interpolated["grho"], parity="e")
    _tb, gbdrift_ball = to_ballooning(
        theta_target_ex, interpolated["gbdrift"], parity="e"
    )
    _tb, cvdrift_ball = to_ballooning(
        theta_target_ex, interpolated["cvdrift"], parity="e"
    )
    _tb, gbdrift0_ball = to_ballooning(
        theta_target_ex, interpolated["gbdrift0"], parity="o"
    )
    _tb, r_ball = to_ballooning(theta_target_ex, interpolated["r"], parity="e")
    _tb, z_ball = to_ballooning(theta_target_ex, interpolated["z"], parity="o")
    _tb, aprime_ball = to_ballooning(theta_st_ex, aprime, parity="o")
    return {
        "theta": theta_ball,
        "gradpar": gradpar_ball,
        "bmag": b_ball,
        "gds2": gds2_ball,
        "gds21": gds21_ball,
        "gds22": gds22_ball,
        "grho": grho_ball,
        "gbdrift": gbdrift_ball,
        "cvdrift": cvdrift_ball,
        "gbdrift0": gbdrift0_ball,
        "Rplot": r_ball,
        "Zplot": z_ball,
        "aprime": aprime_ball,
    }


def _pack_miller_profiles(
    params: MillerCoreParams,
    center: _MillerCenterData,
    balloon: dict[str, np.ndarray],
) -> dict[str, np.ndarray | float]:
    """Pack the imported Miller profile dictionary consumed by EIK writeout."""

    jacob = 1.0 / np.clip(
        np.abs(center.drhodpsi * balloon["gradpar"] * balloon["bmag"]),
        1.0e-30,
        None,
    )
    kxfac = abs(center.qfac[center.center] / params.rhoc * center.dpsidrho)
    rmaj_out = 0.5 * (np.max(balloon["Rplot"]) + np.min(balloon["Rplot"]))
    return {
        "theta": balloon["theta"],
        "bmag": balloon["bmag"],
        "gradpar": balloon["gradpar"],
        "grho": balloon["grho"],
        "gds2": balloon["gds2"],
        "gds21": balloon["gds21"],
        "gds22": balloon["gds22"],
        "gbdrift": balloon["gbdrift"],
        "gbdrift0": balloon["gbdrift0"],
        "cvdrift": balloon["cvdrift"],
        "cvdrift0": balloon["gbdrift0"],
        "jacob": jacob,
        "Rplot": balloon["Rplot"],
        "Zplot": balloon["Zplot"],
        "aprime": balloon["aprime"],
        "drhodpsi": abs(center.drhodpsi),
        "kxfac": float(kxfac),
        "Rmaj": float(rmaj_out),
        "q": float(center.qfac[center.center]),
        "shat": float(params.shat),
    }


def assemble_miller_profiles(
    *,
    params: MillerCoreParams,
    state: dict[str, np.ndarray | float],
    gradients: dict[str, np.ndarray],
    straight_state: dict[str, np.ndarray],
    theta_st_center: np.ndarray,
    theta_st_ex: np.ndarray,
    theta_source_ex: np.ndarray,
    theta_target_ex: np.ndarray,
    gradpar_target_ex: np.ndarray,
    bmag_center: np.ndarray,
    bpol_center: np.ndarray,
    dpsidrho: float,
) -> dict[str, np.ndarray | float]:
    """Assemble imported Miller profiles on the selected theta grid."""

    _ = gradients, theta_st_center, bmag_center
    center = _collect_center_data(
        params=params,
        state=state,
        straight_state=straight_state,
        dpsidrho=dpsidrho,
    )
    ext = _extend_center_data(params, center, bpol_center=bpol_center)
    bishop = _bishop_profiles(params, center, ext, theta_st_ex=theta_st_ex)
    metrics = _metric_profiles(
        params,
        center,
        ext,
        bishop,
        theta_st_ex=theta_st_ex,
        straight_state=straight_state,
    )
    drifts = _drift_profiles(center, ext, bishop, dpsidrho=dpsidrho)
    interpolated = _interpolate_to_target_grid(
        theta_target_ex=theta_target_ex,
        theta_source_ex=theta_source_ex,
        ext=ext,
        metrics=metrics,
        drifts=drifts,
    )
    balloon = _to_ballooning_profiles(
        theta_target_ex=theta_target_ex,
        gradpar_target_ex=gradpar_target_ex,
        interpolated=interpolated,
        theta_st_ex=theta_st_ex,
        aprime=bishop.aprime,
    )
    return _pack_miller_profiles(params, center, balloon)


__all__ = ["assemble_miller_profiles"]
