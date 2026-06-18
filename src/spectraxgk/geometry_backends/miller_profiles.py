"""Profile assembly for the internal Miller imported-geometry backend."""

from __future__ import annotations

import numpy as np

from spectraxgk.geometry_backends.miller_core import MillerCoreParams
from spectraxgk.geometry_backends.miller_numerics import (
    _safe_denom,
    cumulative_trapezoid,
    derm,
    nperiod_data_extend,
    to_ballooning,
)


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

    rho = np.asarray(state["rho"], dtype=float)
    qfac = np.asarray(state["qfac"], dtype=float)
    r = np.asarray(state["r"], dtype=float)
    z = np.asarray(state["z"], dtype=float)
    c = 1
    f_const = float(params.r_geo)
    drhodpsi = 1.0 / float(dpsidrho)
    dpdrho = 0.5 * float(params.betaprim)
    dpdpsi = dpdrho * drhodpsi

    dl = np.asarray(straight_state["dl"], dtype=float)
    u_ml = np.asarray(straight_state["u_ml"], dtype=float)
    r_c = np.asarray(straight_state["r_c"], dtype=float)
    dpsi_dr = np.asarray(straight_state["dpsi_dr"], dtype=float)
    bpol = np.asarray(straight_state["bpol"], dtype=float)
    bmag = np.asarray(straight_state["bmag"], dtype=float)
    psi_diff = np.asarray(straight_state["psi_diff"], dtype=float)

    r_ex = nperiod_data_extend(r[c], params.nperiod, istheta=0, par="e")
    z_ex = nperiod_data_extend(z[c], params.nperiod, istheta=0, par="o")
    dl_ex = nperiod_data_extend(dl[c], params.nperiod, istheta=0, par="e")
    b_ex = nperiod_data_extend(bmag[c], params.nperiod, istheta=0, par="e")
    b2_ex = b_ex * b_ex
    b_p_ex = nperiod_data_extend(np.abs(bpol[c]), params.nperiod, istheta=0, par="e")

    dpsi_dr_ex = nperiod_data_extend(dpsi_dr[c], params.nperiod, istheta=0, par="e")

    diffq = derm(qfac, "r")
    diffrho = derm(rho, "r")
    gds22_ex = (diffq[c] / diffrho[c]) ** 2 * np.abs(dpsi_dr_ex) ** 2

    u_ml_ex = nperiod_data_extend(u_ml[c], params.nperiod, istheta=0, par="e")
    r_c_ex = nperiod_data_extend(r_c[c], params.nperiod, istheta=0, par="e")

    # The imported-geometry convention uses two distinct B_p computations:
    # 1. Bishop integrals: B_p from the geometric-theta grid (bpol_center parameter).
    # 2. aprime formula + drifts: B_p from the theta_st Jacobian (b_p_ex above).
    b_p_geo_ex = nperiod_data_extend(
        np.abs(bpol_center), params.nperiod, istheta=0, par="e"
    )

    a_s = -(
        2.0 * qfac[c] / f_const * theta_st_ex
        + 2.0
        * f_const
        * qfac[c]
        * cumulative_trapezoid(1.0 / _safe_denom(r_ex**2 * b_p_geo_ex**2), theta_st_ex)
    )
    b_s = -(
        2.0
        * qfac[c]
        * cumulative_trapezoid(1.0 / _safe_denom(b_p_geo_ex**2), theta_st_ex)
    )
    c_s = (
        2.0
        * qfac[c]
        * cumulative_trapezoid(
            (2.0 * np.sin(u_ml_ex) / _safe_denom(r_ex) - 2.0 / _safe_denom(r_c_ex))
            * (1.0 / _safe_denom(r_ex * b_p_geo_ex)),
            theta_st_ex,
        )
    )

    prefac = (
        rho[c]
        * (psi_diff[c] / diffrho[c])
        * (1.0 / (2.0 * np.pi * qfac[c] * (2 * params.nperiod - 1)))
    )
    dfdpsi = (
        -params.shat / _safe_denom(prefac) - (b_s[-1] * dpdpsi - c_s[-1])
    ) / _safe_denom(a_s[-1])

    dqdr = diffq[c] * dpsi_dr_ex / psi_diff[c]
    aprime_bish = (
        -r_ex * b_p_ex * (a_s * dfdpsi + b_s * dpdpsi - c_s) / (2.0 * np.abs(drhodpsi))
    )
    gds21_ex = diffq[c] / diffrho[c] * (-dpsi_dr_ex) * aprime_bish

    dt_st_l_dl_center = np.asarray(straight_state["dt_st_l_dl_center"], dtype=float)
    dt_st_l_dl = nperiod_data_extend(
        dt_st_l_dl_center, params.nperiod, istheta=0, par="e"
    )

    dtdr_st_ex = (aprime_bish * drhodpsi - dqdr * theta_st_ex) / qfac[c]
    gds2_ex = (psi_diff[c] / diffrho[c]) ** 2 * (
        1.0 / _safe_denom(r_ex**2)
        + (dqdr * theta_st_ex) ** 2
        + (qfac[c] ** 2) * (dtdr_st_ex**2 + dt_st_l_dl**2)
        + 2.0 * qfac[c] * dqdr * theta_st_ex * dtdr_st_ex
    )

    d_b2_l_ex = np.asarray(derm(b_ex**2, "l", "e"), dtype=float)
    d_b2_l_over_dl_ex = d_b2_l_ex / _safe_denom(dl_ex)
    gbdrift0_ex = (
        (1.0 / _safe_denom(b2_ex**2))
        * dpsidrho
        * f_const
        / _safe_denom(r_ex)
        * (dqdr * d_b2_l_over_dl_ex)
    )

    d_bdr_bish = (
        b_p_ex
        / _safe_denom(b_ex)
        * (
            -b_p_ex / _safe_denom(r_c_ex)
            + dpdpsi * r_ex
            - f_const**2 * np.sin(u_ml_ex) / _safe_denom(r_ex**3 * b_p_ex)
        )
    )
    gbdrift_ex = (
        1.0
        / np.clip(np.abs(drhodpsi * b_ex**3), 1.0e-30, None)
        * (
            2.0 * b2_ex * d_bdr_bish / _safe_denom(dpsi_dr_ex)
            + aprime_bish
            * drhodpsi
            * f_const
            / _safe_denom(r_ex)
            * d_b2_l_over_dl_ex
            / _safe_denom(b_ex)
        )
    )
    cvdrift_ex = (
        1.0 / np.clip(np.abs(drhodpsi * b_ex**3), 1.0e-30, None) * (2.0 * b_ex * dpdpsi)
        + gbdrift_ex
    )
    grho_ex = drhodpsi * dpsi_dr_ex

    r_tgt = np.interp(theta_target_ex, theta_source_ex, r_ex)
    z_tgt = np.interp(theta_target_ex, theta_source_ex, z_ex)
    b_tgt = np.interp(theta_target_ex, theta_source_ex, b_ex)
    gds2_tgt = np.interp(theta_target_ex, theta_source_ex, gds2_ex)
    gds21_tgt = np.interp(theta_target_ex, theta_source_ex, gds21_ex)
    gds22_tgt = np.interp(theta_target_ex, theta_source_ex, gds22_ex)
    grho_tgt = np.interp(theta_target_ex, theta_source_ex, grho_ex)
    gbdrift_tgt = np.interp(theta_target_ex, theta_source_ex, gbdrift_ex)
    cvdrift_tgt = np.interp(theta_target_ex, theta_source_ex, cvdrift_ex)
    gbdrift0_tgt = np.interp(theta_target_ex, theta_source_ex, gbdrift0_ex)

    theta_ball, gradpar_ball = to_ballooning(
        theta_target_ex, gradpar_target_ex, parity="e"
    )
    _tb, b_ball = to_ballooning(theta_target_ex, b_tgt, parity="e")
    _tb, gds2_ball = to_ballooning(theta_target_ex, gds2_tgt, parity="e")
    _tb, gds21_ball = to_ballooning(theta_target_ex, gds21_tgt, parity="o")
    _tb, gds22_ball = to_ballooning(theta_target_ex, gds22_tgt, parity="e")
    _tb, grho_ball = to_ballooning(theta_target_ex, grho_tgt, parity="e")
    _tb, gbdrift_ball = to_ballooning(theta_target_ex, gbdrift_tgt, parity="e")
    _tb, cvdrift_ball = to_ballooning(theta_target_ex, cvdrift_tgt, parity="e")
    _tb, gbdrift0_ball = to_ballooning(theta_target_ex, gbdrift0_tgt, parity="o")
    _tb, r_ball = to_ballooning(theta_target_ex, r_tgt, parity="e")
    _tb, z_ball = to_ballooning(theta_target_ex, z_tgt, parity="o")
    _tb, aprime_ball = to_ballooning(theta_st_ex, aprime_bish, parity="o")

    jacob_ball = 1.0 / np.clip(np.abs(drhodpsi * gradpar_ball * b_ball), 1.0e-30, None)
    kxfac = abs(qfac[c] / params.rhoc * dpsidrho)
    rmaj_out = 0.5 * (np.max(r_ball) + np.min(r_ball))

    return {
        "theta": theta_ball,
        "bmag": b_ball,
        "gradpar": gradpar_ball,
        "grho": grho_ball,
        "gds2": gds2_ball,
        "gds21": gds21_ball,
        "gds22": gds22_ball,
        "gbdrift": gbdrift_ball,
        "gbdrift0": gbdrift0_ball,
        "cvdrift": cvdrift_ball,
        "cvdrift0": gbdrift0_ball,
        "jacob": jacob_ball,
        "Rplot": r_ball,
        "Zplot": z_ball,
        "aprime": aprime_ball,
        "drhodpsi": abs(drhodpsi),
        "kxfac": float(kxfac),
        "Rmaj": float(rmaj_out),
        "q": float(qfac[c]),
        "shat": float(params.shat),
    }


__all__ = ["assemble_miller_profiles"]
