"""Imported Miller geometry backend and EIK-file generation helpers.

The Miller backend is kept in one module because its low-level finite
differences, surface construction, profile assembly, and NetCDF writeout are a
single imported-geometry pipeline. Shared JAX array kernels live in
``spectraxgk.geometry.kernels`` so VMEC and Miller use the same numerical
primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.geometry.analytic import MillerCoreParams, build_collocation_surfaces
from spectraxgk.geometry.kernels import (
    _safe_denom,
    cumulative_trapezoid,
    derm,
    dermv,
    nperiod_data_extend,
    reflect_n_append,
    to_ballooning,
)


@dataclass(frozen=True)
class _StraightThetaDerivatives:
    psi: np.ndarray
    psi_diff: np.ndarray
    jac: np.ndarray
    dpsid_r: np.ndarray
    dpsid_z: np.ndarray
    dt_d_r: np.ndarray
    dt_d_z: np.ndarray
    dl: np.ndarray


@dataclass(frozen=True)
class _StraightThetaSurfaceShape:
    theta_common_new: np.ndarray
    u_ml: np.ndarray
    r_c: np.ndarray


@dataclass(frozen=True)
class _StraightThetaMagneticProfiles:
    dpsi_dr: np.ndarray
    bpol: np.ndarray
    bmag: np.ndarray
    b2: np.ndarray


def compute_primary_gradients(
    state: dict[str, np.ndarray | float],
) -> dict[str, np.ndarray]:
    """Evaluate the primary Miller derivative block."""

    r = np.asarray(state["r"], dtype=float)
    z = np.asarray(state["z"], dtype=float)
    rho = np.asarray(state["rho"], dtype=float)
    theta_common = np.asarray(state["theta_common_mag_axis"], dtype=float)

    dl = np.sqrt(derm(r, "l", "e") ** 2 + derm(z, "l", "o") ** 2)
    dt = derm(theta_common, "l", "o")
    rho_diff = derm(rho, "r")

    d_r_drho = derm(r, "r") / rho_diff[:, None]
    d_r_dt = dermv(r, theta_common, "l", "e")
    d_z_drho = derm(z, "r") / rho_diff[:, None]
    d_z_dt = dermv(z, theta_common, "l", "o")

    jac = d_r_drho * d_z_dt - d_z_drho * d_r_dt
    drhod_r = d_z_dt / jac
    drhod_z = -d_r_dt / jac
    dt_d_r = -d_z_drho / jac
    dt_d_z = d_r_drho / jac

    return {
        "dl": dl,
        "dt": dt,
        "rho_diff": rho_diff,
        "d_r_drho": d_r_drho,
        "d_r_dt": d_r_dt,
        "d_z_drho": d_z_drho,
        "d_z_dt": d_z_dt,
        "jac": jac,
        "drhod_r": drhod_r,
        "drhod_z": drhod_z,
        "dt_d_r": dt_d_r,
        "dt_d_z": dt_d_z,
    }


def compute_straight_field_theta(
    *,
    f_const: float,
    dpsidrho: np.ndarray,
    jac: np.ndarray,
    r: np.ndarray,
    theta_common: np.ndarray,
) -> np.ndarray:
    """Construct the straight-field-line Miller theta grid."""

    out = np.zeros_like(theta_common, dtype=float)
    no_of_surfs = theta_common.shape[0]
    for i in range(no_of_surfs):
        integ = cumulative_trapezoid(
            np.abs(f_const * (1.0 / dpsidrho[i]) * jac[i] / r[i]), theta_common[i]
        )
        if np.abs(integ[-1]) < 1.0e-30:
            out[i] = theta_common[i]
        else:
            out[i] = np.pi * integ / integ[-1]
    return out


def compute_equal_arc_theta(
    *,
    theta_straight: np.ndarray,
    gradpar: np.ndarray,
    bmag: np.ndarray,
    bpol: np.ndarray,
    nperiod: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the equal-arc Miller mapping using central-surface quantities."""

    theta_st_ex = nperiod_data_extend(theta_straight, nperiod, istheta=1)
    gradpar_ex = nperiod_data_extend(gradpar, nperiod)
    bmag_ex = nperiod_data_extend(bmag, nperiod)
    bpol_ex = nperiod_data_extend(np.abs(bpol), nperiod)

    mask = theta_st_ex <= np.pi
    theta_lim = theta_st_ex[mask]
    gradpar_lim = gradpar_ex[mask]
    bmag_lim = bmag_ex[mask]
    bpol_lim = bpol_ex[mask]

    l_eqarc = cumulative_trapezoid(
        bpol_lim / _safe_denom(bmag_lim * gradpar_lim), theta_lim
    )
    denom = cumulative_trapezoid(1.0 / _safe_denom(gradpar_lim), theta_lim)[-1]
    if np.abs(denom) < 1.0e-30:
        # Degenerate fallback: keep geometric mapping and profile lengths consistent.
        theta_eqarc = theta_lim.copy()
        gradpar_eqarc = gradpar_lim.copy()
    else:
        # The equal-arc convention stores gradpar_eqarc as a scalar constant.
        gradpar_eqarc_const = np.pi / denom
        theta_eqarc = cumulative_trapezoid(
            bmag_lim / _safe_denom(bpol_lim) * gradpar_eqarc_const,
            l_eqarc,
        )
        gradpar_eqarc = np.full_like(theta_eqarc, gradpar_eqarc_const)

    ntgrid = int(theta_straight.shape[0])
    theta_uniform = np.linspace(0.0, np.pi, ntgrid, dtype=float)
    theta_uniform_ex = nperiod_data_extend(theta_uniform, nperiod, istheta=1)
    theta_eqarc_ex = nperiod_data_extend(theta_eqarc, nperiod, istheta=1)
    gradpar_eqarc_ex = nperiod_data_extend(gradpar_eqarc, nperiod, istheta=0)
    gradpar_uniform_ex = np.interp(theta_uniform_ex, theta_eqarc_ex, gradpar_eqarc_ex)
    return theta_uniform_ex, gradpar_uniform_ex, theta_eqarc_ex


def _straight_theta_derivatives(
    *,
    params: MillerCoreParams,
    r: np.ndarray,
    z: np.ndarray,
    theta_st: np.ndarray,
    dpsidrho: float,
) -> _StraightThetaDerivatives:
    drhodpsi = 1.0 / float(dpsidrho)
    psi = np.array(
        [1.0 - params.delrho / drhodpsi, 1.0, 1.0 + params.delrho / drhodpsi],
        dtype=float,
    )
    psi_diff = derm(psi, "r")
    d_r_dpsi = derm(r, "r") / psi_diff[:, None]
    d_r_dt = dermv(r, theta_st, "l", "e")
    d_z_dpsi = derm(z, "r") / psi_diff[:, None]
    d_z_dt = dermv(z, theta_st, "l", "o")
    jac = d_r_dpsi * d_z_dt - d_z_dpsi * d_r_dt
    return _StraightThetaDerivatives(
        psi=psi,
        psi_diff=psi_diff,
        jac=jac,
        dpsid_r=d_z_dt / jac,
        dpsid_z=-d_r_dt / jac,
        dt_d_r=-d_z_dpsi / jac,
        dt_d_z=d_r_dpsi / jac,
        dl=np.sqrt(derm(r, "l", "e") ** 2 + derm(z, "l", "o") ** 2),
    )


def _straight_theta_surface_shape(
    *, params: MillerCoreParams, r: np.ndarray, z: np.ndarray
) -> _StraightThetaSurfaceShape:
    d_r_l = derm(r, "l", "e")
    d_z_l = derm(z, "l", "o")
    phi_n = np.zeros_like(r)
    for i in range(r.shape[0]):
        phi = np.arctan2(d_z_l[i], d_r_l[i])
        phi_n[i] = np.concatenate(
            (phi[phi >= 0.0] - np.pi / 2.0, phi[phi < 0.0] + 3.0 * np.pi / 2.0)
        )
    return _StraightThetaSurfaceShape(
        theta_common_new=np.arctan2(z, r - params.rmaj),
        u_ml=np.arctan2(d_z_l, d_r_l),
        r_c=np.sqrt(d_r_l**2 + d_z_l**2) / derm(phi_n, "l", "o"),
    )


def _straight_theta_magnetic_profiles(
    *,
    r: np.ndarray,
    psi_diff: np.ndarray,
    dpsid_r: np.ndarray,
    dpsid_z: np.ndarray,
    f_const: float,
) -> _StraightThetaMagneticProfiles:
    dpsi_dr = np.sign(psi_diff)[:, None] * np.sqrt(dpsid_r**2 + dpsid_z**2)
    b_p = np.abs(dpsi_dr) / _safe_denom(r)
    b_t = f_const / _safe_denom(r)
    b2 = b_p**2 + b_t**2
    return _StraightThetaMagneticProfiles(
        dpsi_dr=dpsi_dr,
        bpol=b_p,
        bmag=np.sqrt(b2),
        b2=b2,
    )


def _straight_theta_arc_length(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    l_st = np.zeros_like(r)
    for i in range(r.shape[0]):
        l_st[i, 1:] = np.cumsum(np.sqrt(np.diff(r[i]) ** 2 + np.diff(z[i]) ** 2))
    return l_st


def _straight_theta_center_gradpar(
    *,
    r: np.ndarray,
    bmag: np.ndarray,
    dpsi_dr: np.ndarray,
    l_st: np.ndarray,
    theta_st: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c = 1
    dt_st_l_dl_center = np.asarray(
        1.0 / _safe_denom(dermv(l_st, theta_st, "l", "o")[c]),
        dtype=float,
    )
    gradpar_center = (
        -(1.0 / _safe_denom(r[c] * bmag[c])) * dpsi_dr[c] * dt_st_l_dl_center
    )
    return dt_st_l_dl_center, gradpar_center


def rebuild_straight_theta_state(
    *,
    params: MillerCoreParams,
    state: dict[str, np.ndarray | float],
    theta_st: np.ndarray,
    dpsidrho: float,
    f_const: float,
) -> dict[str, np.ndarray]:
    """Rebuild Miller geometric quantities with theta_st as the independent grid, mirroring the imported-geometry convention."""

    r = np.asarray(state["r"], dtype=float).copy()
    z = np.asarray(state["z"], dtype=float).copy()
    derivatives = _straight_theta_derivatives(
        params=params, r=r, z=z, theta_st=theta_st, dpsidrho=dpsidrho
    )
    shape = _straight_theta_surface_shape(params=params, r=r, z=z)
    magnetic = _straight_theta_magnetic_profiles(
        r=r,
        psi_diff=derivatives.psi_diff,
        dpsid_r=derivatives.dpsid_r,
        dpsid_z=derivatives.dpsid_z,
        f_const=f_const,
    )
    l_st = _straight_theta_arc_length(r, z)
    dt_st_l_dl_center, gradpar_center = _straight_theta_center_gradpar(
        r=r,
        bmag=magnetic.bmag,
        dpsi_dr=magnetic.dpsi_dr,
        l_st=l_st,
        theta_st=theta_st,
    )

    return {
        "psi": derivatives.psi,
        "psi_diff": derivatives.psi_diff,
        "jac": derivatives.jac,
        "dpsid_r": derivatives.dpsid_r,
        "dpsid_z": derivatives.dpsid_z,
        "dt_d_r": derivatives.dt_d_r,
        "dt_d_z": derivatives.dt_d_z,
        "dl": derivatives.dl,
        "theta_common_new": shape.theta_common_new,
        "u_ml": shape.u_ml,
        "r_c": shape.r_c,
        "dpsi_dr": magnetic.dpsi_dr,
        "bpol": magnetic.bpol,
        "bmag": magnetic.bmag,
        "b2": magnetic.b2,
        "l_st": l_st,
        "dt_st_l_dl_center": dt_st_l_dl_center,
        "gradpar_center": gradpar_center,
        "bmag_center": magnetic.bmag[1],
        "bpol_center": magnetic.bpol[1],
    }


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
        * cumulative_trapezoid(
            1.0 / _safe_denom(ext.r**2 * ext.bpol_geo**2), theta_st_ex
        )
    )
    b_s = (
        -2.0 * q * cumulative_trapezoid(1.0 / _safe_denom(ext.bpol_geo**2), theta_st_ex)
    )
    c_s = (
        2.0
        * q
        * cumulative_trapezoid(
            (2.0 * np.sin(ext.u_ml) / _safe_denom(ext.r) - 2.0 / _safe_denom(ext.r_c))
            * (1.0 / _safe_denom(ext.r * ext.bpol_geo)),
            theta_st_ex,
        )
    )
    prefac = (
        center.rho[c]
        * (center.psi_diff[c] / center.diffrho[c])
        * (1.0 / (2.0 * np.pi * q * (2 * params.nperiod - 1)))
    )
    dfdpsi = (
        -params.shat / _safe_denom(prefac) - (b_s[-1] * center.dpdpsi - c_s[-1])
    ) / _safe_denom(a_s[-1])
    dqdr = center.diffq[c] * ext.dpsi_dr / center.psi_diff[c]
    aprime = (
        -ext.r
        * ext.bpol
        * (a_s * dfdpsi + b_s * center.dpdpsi - c_s)
        / (2.0 * np.abs(center.drhodpsi))
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
    cvdrift = (1.0 / np.clip(np.abs(center.drhodpsi * ext.b**3), 1.0e-30, None)) * (
        2.0 * ext.b * center.dpdpsi
    ) + gbdrift
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
    straight_state: dict[str, np.ndarray],
    theta_st_ex: np.ndarray,
    theta_source_ex: np.ndarray,
    theta_target_ex: np.ndarray,
    gradpar_target_ex: np.ndarray,
    bpol_center: np.ndarray,
    dpsidrho: float,
) -> dict[str, np.ndarray | float]:
    """Assemble imported Miller profiles on the selected theta grid."""

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


def _request_attr(request: Any, *names: str) -> Any:
    """Return the first available attribute from a Miller request."""

    for name in names:
        if hasattr(request, name):
            return getattr(request, name)
    raise AttributeError(f"Miller request is missing all aliases: {', '.join(names)}")


@dataclass(frozen=True)
class _MillerGeometryNormalizations:
    dpsidrho_arr: np.ndarray
    dpsidrho: float
    bpol: np.ndarray
    bmag: np.ndarray


def _miller_params_from_request(request: Any) -> MillerCoreParams:
    return MillerCoreParams(
        ntgrid=int(int(request.ntheta) / 2 + 1),
        nperiod=int(request.nperiod),
        rhoc=float(_request_attr(request, "rhoc")),
        qinp=float(_request_attr(request, "qinp", "q")),
        shat=float(_request_attr(request, "shat", "s_hat")),
        rmaj=float(_request_attr(request, "Rmaj", "R0")),
        r_geo=float(_request_attr(request, "R_geo")),
        shift=float(_request_attr(request, "shift")),
        akappa=float(_request_attr(request, "akappa")),
        tri=float(_request_attr(request, "tri")),
        akappri=float(_request_attr(request, "akappri")),
        tripri=float(_request_attr(request, "tripri")),
        betaprim=float(_request_attr(request, "betaprim")),
    )


def _miller_geometry_normalizations(
    params: MillerCoreParams, state: dict[str, Any], gradients: dict[str, Any]
) -> _MillerGeometryNormalizations:
    r = np.asarray(state["r"], dtype=float)
    qfac = np.asarray(state["qfac"], dtype=float)
    theta_common = np.asarray(state["theta_common_mag_axis"], dtype=float)
    jac = np.asarray(gradients["jac"], dtype=float)
    drhod_r = np.asarray(gradients["drhod_r"], dtype=float)
    drhod_z = np.asarray(gradients["drhod_z"], dtype=float)

    jac_r_theta_arr = np.abs(
        2.0 * cumulative_trapezoid(jac / _safe_denom(r), theta_common, axis=1)[:, -1]
    )
    dpsidrho_arr = -(params.r_geo / np.abs(2.0 * np.pi * qfac)) * jac_r_theta_arr
    dpsidrho = float(dpsidrho_arr[1])
    bpol = (
        np.abs(dpsidrho)
        * np.sqrt(drhod_r[1] ** 2 + drhod_z[1] ** 2)
        / _safe_denom(r[1])
    )
    btor = params.r_geo / _safe_denom(r[1])
    return _MillerGeometryNormalizations(
        dpsidrho_arr=np.asarray(dpsidrho_arr, dtype=float),
        dpsidrho=dpsidrho,
        bpol=bpol,
        bmag=np.sqrt(bpol**2 + btor**2),
    )


def _miller_equal_arc_grid(
    params: MillerCoreParams, theta_st: np.ndarray, straight_state: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return compute_equal_arc_theta(
        theta_straight=theta_st[1],
        gradpar=np.asarray(straight_state["gradpar_center"], dtype=float),
        bmag=np.asarray(straight_state["bmag_center"], dtype=float),
        bpol=np.asarray(straight_state["bpol_center"], dtype=float),
        nperiod=params.nperiod,
    )


def _assemble_miller_profiles_for_request(request: Any) -> dict[str, Any]:
    params = _miller_params_from_request(request)
    state = build_collocation_surfaces(params)
    gradients = compute_primary_gradients(state)
    normalizations = _miller_geometry_normalizations(params, state, gradients)
    theta_common = np.asarray(state["theta_common_mag_axis"], dtype=float)
    r = np.asarray(state["r"], dtype=float)

    theta_st = compute_straight_field_theta(
        f_const=float(params.r_geo),
        dpsidrho=normalizations.dpsidrho_arr,
        jac=np.asarray(gradients["jac"], dtype=float),
        r=r,
        theta_common=theta_common,
    )
    straight_state = rebuild_straight_theta_state(
        params=params,
        state=state,
        theta_st=theta_st,
        dpsidrho=normalizations.dpsidrho,
        f_const=float(params.r_geo),
    )
    theta_target_ex, gradpar_target_ex, theta_source_ex = _miller_equal_arc_grid(
        params, theta_st, straight_state
    )
    return assemble_miller_profiles(
        params=params,
        state=state,
        straight_state=straight_state,
        theta_st_ex=nperiod_data_extend(theta_st[1], params.nperiod, istheta=1),
        theta_source_ex=theta_source_ex,
        theta_target_ex=theta_target_ex,
        gradpar_target_ex=gradpar_target_ex,
        bpol_center=normalizations.bpol,
        dpsidrho=normalizations.dpsidrho,
    )


def write_miller_eik_netcdf(
    path: Path, profiles: dict[str, np.ndarray | float]
) -> None:
    """Write root-level imported Miller ``*.eiknc.nc`` output."""

    try:
        import netCDF4 as nc
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError(
            "netCDF4 is required for internal Miller eik writeout"
        ) from exc

    theta = np.asarray(profiles["theta"], dtype=float)
    ntheta2 = int(theta.shape[0] - 1)
    if ntheta2 < 2:
        raise ValueError("Insufficient theta samples for Miller eik writeout")

    with nc.Dataset(path, "w") as ds:
        ds.createDimension("z", ntheta2)

        def _vec(name: str) -> Any:
            return ds.createVariable(name, "f8", ("z",))

        _vec("theta")[:] = np.asarray(profiles["theta"], dtype=float)[:-1]
        _vec("bmag")[:] = np.asarray(profiles["bmag"], dtype=float)[:-1]
        _vec("gradpar")[:] = np.asarray(profiles["gradpar"], dtype=float)[:-1]
        _vec("grho")[:] = np.asarray(profiles["grho"], dtype=float)[:-1]
        _vec("gds2")[:] = np.asarray(profiles["gds2"], dtype=float)[:-1]
        _vec("gds21")[:] = np.asarray(profiles["gds21"], dtype=float)[:-1]
        _vec("gds22")[:] = np.asarray(profiles["gds22"], dtype=float)[:-1]
        _vec("gbdrift")[:] = np.asarray(profiles["gbdrift"], dtype=float)[:-1]
        _vec("gbdrift0")[:] = np.asarray(profiles["gbdrift0"], dtype=float)[:-1]
        _vec("cvdrift")[:] = np.asarray(profiles["cvdrift"], dtype=float)[:-1]
        _vec("cvdrift0")[:] = np.asarray(profiles["cvdrift0"], dtype=float)[:-1]
        _vec("jacob")[:] = np.asarray(profiles["jacob"], dtype=float)[:-1]
        _vec("Rplot")[:] = np.asarray(profiles["Rplot"], dtype=float)[:-1]
        _vec("Zplot")[:] = np.asarray(profiles["Zplot"], dtype=float)[:-1]
        _vec("aprime")[:] = np.asarray(profiles["aprime"], dtype=float)[:-1]

        ds.createVariable("drhodpsi", "f8").assignValue(float(profiles["drhodpsi"]))
        ds.createVariable("kxfac", "f8").assignValue(float(profiles["kxfac"]))
        ds.createVariable("Rmaj", "f8").assignValue(float(profiles["Rmaj"]))
        ds.createVariable("q", "f8").assignValue(float(profiles["q"]))
        ds.createVariable("shat", "f8").assignValue(float(profiles["shat"]))


def generate_miller_eik_internal(*, output_path: str | Path, request: Any) -> Path:
    """Generate an EIK file from a complete Miller geometry request."""

    profiles = _assemble_miller_profiles_for_request(request)
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_miller_eik_netcdf(out, profiles)
    return out


__all__ = [
    "MillerCoreParams",
    "_request_attr",
    "_safe_denom",
    "assemble_miller_profiles",
    "build_collocation_surfaces",
    "compute_equal_arc_theta",
    "compute_primary_gradients",
    "compute_straight_field_theta",
    "cumulative_trapezoid",
    "derm",
    "dermv",
    "generate_miller_eik_internal",
    "nperiod_data_extend",
    "rebuild_straight_theta_state",
    "reflect_n_append",
    "to_ballooning",
    "write_miller_eik_netcdf",
]
