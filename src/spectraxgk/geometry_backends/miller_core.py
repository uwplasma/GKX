"""Core Miller surface and theta-grid construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectraxgk.geometry_backends.miller_numerics import (
    _safe_denom,
    cumulative_trapezoid,
    derm,
    dermv,
    nperiod_data_extend,
)


@dataclass(frozen=True)
class MillerCoreParams:
    """Core Miller parameters needed by the low-level Miller geometry formulas."""

    ntgrid: int
    nperiod: int
    rhoc: float
    qinp: float
    shat: float
    rmaj: float
    r_geo: float
    shift: float
    akappa: float
    tri: float
    akappri: float
    tripri: float
    betaprim: float
    delrho: float = 1.0e-3


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


def build_collocation_surfaces(
    params: MillerCoreParams,
) -> dict[str, np.ndarray | float]:
    """Construct the Miller surface on the collocation grid."""

    no_of_surfs = 3
    theta = np.linspace(0.0, np.pi, int(params.ntgrid), dtype=float)
    r0 = np.array(
        [
            params.rmaj - params.shift * params.delrho,
            params.rmaj,
            params.rmaj + params.shift * params.delrho,
        ],
        dtype=float,
    )
    rho = np.array(
        [params.rhoc - params.delrho, params.rhoc, params.rhoc + params.delrho],
        dtype=float,
    )
    qfac = np.array(
        [
            params.qinp - params.shat * (params.qinp / params.rhoc) * params.delrho,
            params.qinp,
            params.qinp + params.shat * (params.qinp / params.rhoc) * params.delrho,
        ],
        dtype=float,
    )
    kappa = np.array(
        [
            params.akappa - params.akappri * params.delrho,
            params.akappa,
            params.akappa + params.akappri * params.delrho,
        ],
        dtype=float,
    )
    delta = np.array(
        [
            params.tri - params.tripri * params.delrho,
            params.tri,
            params.tri + params.tripri * params.delrho,
        ],
        dtype=float,
    )

    r = np.array(
        [
            r0[i] + rho[i] * np.cos(theta + np.arcsin(delta[i]) * np.sin(theta))
            for i in range(no_of_surfs)
        ],
        dtype=float,
    )
    z = np.array(
        [kappa[i] * rho[i] * np.sin(theta) for i in range(no_of_surfs)], dtype=float
    )
    theta_common_mag_axis = np.arctan2(z, r - params.rmaj)

    return {
        "theta": theta,
        "rho": rho,
        "qfac": qfac,
        "r": r,
        "z": z,
        "theta_common_mag_axis": theta_common_mag_axis,
        "dpdrho": 0.5 * float(params.betaprim),
        "no_of_surfs": float(no_of_surfs),
    }


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


__all__ = [
    "MillerCoreParams",
    "build_collocation_surfaces",
    "compute_equal_arc_theta",
    "compute_primary_gradients",
    "compute_straight_field_theta",
    "rebuild_straight_theta_state",
]
