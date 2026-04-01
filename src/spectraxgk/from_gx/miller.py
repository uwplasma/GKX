"""Miller geometry internals being ported from GX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spectraxgk.from_gx.kernels import gx_derm, gx_dermv, gx_nperiod_data_extend, gx_reflect_n_append


def _safe_denom(values: np.ndarray | float, eps: float = 1.0e-30) -> np.ndarray | float:
    """Avoid division by zero while preserving the sign of nonzero inputs."""

    arr = np.asarray(values, dtype=float)
    safe = np.where(np.abs(arr) < eps, np.where(arr < 0.0, -eps, eps), arr)
    if np.isscalar(values):
        return float(safe)
    return safe


def internal_miller_backend_available() -> bool:
    """Return True when internal Miller backend dependencies are present."""

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except Exception:
        return False
    return True


def generate_miller_eik_internal(*, output_path: str | Path, request: Any | None = None) -> Path:
    """Internal Miller->EIK pipeline entry point (in progress)."""

    if request is None:
        raise NotImplementedError(
            "Internal Miller geometry backend requires runtime request data. "
            "Current status: low-level GX numerics are ported; final EIK writeout is pending."
        )

    params = MillerCoreParams(
        ntgrid=int(int(request.ntheta) / 2 + 1),
        nperiod=int(request.nperiod),
        rhoc=float(request.rhoc),
        qinp=float(request.qinp),
        shat=float(request.shat),
        rmaj=float(request.Rmaj),
        r_geo=float(request.R_geo),
        shift=float(request.shift),
        akappa=float(request.akappa),
        tri=float(request.tri),
        akappri=float(request.akappri),
        tripri=float(request.tripri),
        betaprim=float(request.betaprim),
    )
    state = build_collocation_surfaces(params)
    gradients = compute_primary_gradients(state)

    r = np.asarray(state["r"], dtype=float)
    qfac = np.asarray(state["qfac"], dtype=float)
    theta_common = np.asarray(state["theta_common_mag_axis"], dtype=float)

    jac = np.asarray(gradients["jac"], dtype=float)
    drhod_r = np.asarray(gradients["drhod_r"], dtype=float)
    drhod_z = np.asarray(gradients["drhod_z"], dtype=float)
    dl = np.asarray(gradients["dl"], dtype=float)

    jac_r_theta_arr = np.abs(2.0 * cumulative_trapezoid(jac / _safe_denom(r), theta_common, axis=1)[:, -1])
    dpsidrho_arr = -(params.r_geo / np.abs(2.0 * np.pi * qfac)) * jac_r_theta_arr
    dpsidrho = float(dpsidrho_arr[1])
    bpol = np.abs(dpsidrho) * np.sqrt(drhod_r[1] ** 2 + drhod_z[1] ** 2) / _safe_denom(r[1])
    btor = params.r_geo / _safe_denom(r[1])
    bmag = np.sqrt(bpol**2 + btor**2)
    theta_st = compute_straight_field_theta(
        f_const=float(params.r_geo),
        dpsidrho=np.asarray(dpsidrho_arr, dtype=float),
        jac=jac,
        r=r,
        theta_common=theta_common,
    )
    straight_state = rebuild_straight_theta_state(
        params=params,
        state=state,
        theta_st=theta_st,
        dpsidrho=float(dpsidrho),
        f_const=float(params.r_geo),
    )
    theta_eqarc_target_ex, gradpar_eqarc_ex, theta_eqarc_source_ex = compute_equal_arc_theta(
        theta_straight=theta_st[1],
        gradpar=np.asarray(straight_state["gradpar_center"], dtype=float),
        bmag=np.asarray(straight_state["bmag_center"], dtype=float),
        bpol=np.asarray(straight_state["bpol_center"], dtype=float),
        nperiod=params.nperiod,
    )
    profiles = assemble_miller_profiles(
        params=params,
        state=state,
        gradients=gradients,
        straight_state=straight_state,
        theta_st_center=theta_st[1],
        theta_st_ex=nperiod_data_extend(theta_st[1], params.nperiod, istheta=1),
        theta_source_ex=theta_eqarc_source_ex,
        theta_target_ex=theta_eqarc_target_ex,
        gradpar_target_ex=gradpar_eqarc_ex,
        bmag_center=bmag,
        bpol_center=bpol,
        dpsidrho=float(dpsidrho),
    )
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_miller_eik_netcdf(out, profiles)
    return out


def derm(arr: np.ndarray, ch: str, par: str = "e") -> np.ndarray:
    """NumPy wrapper around JAX-backed GX Miller ``derm``."""

    axis = "l" if ch == "l" else "r"
    return np.asarray(gx_derm(np.asarray(arr), axis=axis, parity=par))


def dermv(arr: np.ndarray, brr: np.ndarray, ch: str, par: str = "e") -> np.ndarray:
    """NumPy wrapper around JAX-backed GX Miller ``dermv``."""

    axis = "l" if ch == "l" else "r"
    return np.asarray(gx_dermv(np.asarray(arr), np.asarray(brr), axis=axis, parity=par))


def nperiod_data_extend(arr: np.ndarray, nperiod: int, istheta: int = 0, par: str = "e") -> np.ndarray:
    """NumPy wrapper around JAX-backed GX Miller period-extension helper."""

    return np.asarray(gx_nperiod_data_extend(np.asarray(arr), int(nperiod), istheta=bool(istheta), parity=par))


def reflect_n_append(arr: np.ndarray, ch: str) -> np.ndarray:
    """NumPy wrapper around JAX-backed GX Miller reflection helper."""

    return np.asarray(gx_reflect_n_append(np.asarray(arr), parity=ch))


@dataclass(frozen=True)
class MillerCoreParams:
    """Core Miller parameters needed by the low-level GX geometry formulas."""

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


def build_collocation_surfaces(params: MillerCoreParams) -> dict[str, np.ndarray | float]:
    """Port of GX Miller surface construction on the collocation grid."""

    no_of_surfs = 3
    theta = np.linspace(0.0, np.pi, int(params.ntgrid), dtype=float)
    r0 = np.array(
        [
            params.rmaj + np.abs(params.shift) * params.delrho,
            params.rmaj,
            params.rmaj - np.abs(params.shift) * params.delrho,
        ],
        dtype=float,
    )
    rho = np.array([params.rhoc - params.delrho, params.rhoc, params.rhoc + params.delrho], dtype=float)
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
        [r0[i] + rho[i] * np.cos(theta + np.arcsin(delta[i]) * np.sin(theta)) for i in range(no_of_surfs)],
        dtype=float,
    )
    z = np.array([kappa[i] * rho[i] * np.sin(theta) for i in range(no_of_surfs)], dtype=float)
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


def compute_primary_gradients(state: dict[str, np.ndarray | float]) -> dict[str, np.ndarray]:
    """Port of primary derivative block from GX Miller geometry script."""

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


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy equivalent of SciPy cumulative trapezoid with initial=0.
    
    Supports 1D or 2D arrays. For 2D y with 1D x, broadcasts x and integrates per-row.
    """

    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)

    # Handle 1D case
    if yy.ndim == 1:
        if xx.ndim != 1:
            raise ValueError("x and y must have same number of dimensions")
        if yy.shape[0] != xx.shape[0]:
            raise ValueError("x and y must have the same length")
        if yy.shape[0] < 2:
            return np.zeros_like(yy)
        area = 0.5 * (yy[1:] + yy[:-1]) * (xx[1:] - xx[:-1])
        return np.concatenate(([0.0], np.cumsum(area)))

    # Handle 2D case (integrate along axis)
    if yy.ndim == 2:
        # Normalize axis
        if axis < 0:
            axis = yy.ndim + axis
        
        if axis == 1:
            # Integrate along columns (per-row integration) with 1D x broadcasting
            if xx.ndim not in (1, 2):
                raise ValueError("For 2D y with axis=1, x must be 1D or 2D")
            
            if xx.ndim == 1:
                # 1D x: broadcast across rows
                if yy.shape[1] != xx.shape[0]:
                    raise ValueError(f"x length ({xx.shape[0]}) must match y's column count ({yy.shape[1]})")
                result = np.zeros_like(yy)
                for i in range(yy.shape[0]):
                    area = 0.5 * (yy[i, 1:] + yy[i, :-1]) * (xx[1:] - xx[:-1])
                    result[i, 1:] = np.cumsum(area)
                return result
            else:
                # 2D x: element-wise matching
                if yy.shape != xx.shape:
                    raise ValueError(f"y shape {yy.shape} must match x shape {xx.shape}")
                result = np.zeros_like(yy)
                for i in range(yy.shape[0]):
                    area = 0.5 * (yy[i, 1:] + yy[i, :-1]) * (xx[i, 1:] - xx[i, :-1])
                    result[i, 1:] = np.cumsum(area)
                return result
        else:
            raise NotImplementedError(f"cumulative_trapezoid with axis={axis} not yet supported")

    raise ValueError(f"cumulative_trapezoid supports ndim=1 or 2, got {yy.ndim}")


def compute_straight_field_theta(
    *,
    f_const: float,
    dpsidrho: np.ndarray,
    jac: np.ndarray,
    r: np.ndarray,
    theta_common: np.ndarray,
) -> np.ndarray:
    """Port of GX Miller straight-field-line theta construction."""

    out = np.zeros_like(theta_common, dtype=float)
    no_of_surfs = theta_common.shape[0]
    for i in range(no_of_surfs):
        integ = cumulative_trapezoid(np.abs(f_const * (1.0 / dpsidrho[i]) * jac[i] / r[i]), theta_common[i])
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
    """Port of GX Miller equal-arc mapping using central-surface quantities."""

    theta_st_ex = nperiod_data_extend(theta_straight, nperiod, istheta=1)
    gradpar_ex = nperiod_data_extend(gradpar, nperiod)
    bmag_ex = nperiod_data_extend(bmag, nperiod)
    bpol_ex = nperiod_data_extend(np.abs(bpol), nperiod)

    mask = theta_st_ex <= np.pi
    theta_lim = theta_st_ex[mask]
    gradpar_lim = gradpar_ex[mask]
    bmag_lim = bmag_ex[mask]
    bpol_lim = bpol_ex[mask]

    l_eqarc = cumulative_trapezoid(bpol_lim / _safe_denom(bmag_lim * gradpar_lim), theta_lim)
    denom = cumulative_trapezoid(1.0 / _safe_denom(gradpar_lim), theta_lim)[-1]
    if np.abs(denom) < 1.0e-30:
        # Degenerate fallback: keep geometric mapping and profile lengths consistent.
        theta_eqarc = theta_lim.copy()
        gradpar_eqarc = gradpar_lim.copy()
    else:
        # GX treatment: gradpar_eqarc is a scalar constant on the equal-arc grid.
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


def rebuild_straight_theta_state(
    *,
    params: MillerCoreParams,
    state: dict[str, np.ndarray | float],
    theta_st: np.ndarray,
    dpsidrho: float,
    f_const: float,
) -> dict[str, np.ndarray]:
    """Rebuild Miller geometric quantities with theta_st as the independent grid, mirroring GX."""

    rho = np.asarray(state["rho"], dtype=float)
    r = np.asarray(state["r"], dtype=float).copy()
    z = np.asarray(state["z"], dtype=float).copy()

    drhodpsi = 1.0 / float(dpsidrho)
    psi = np.array([1.0 - params.delrho / drhodpsi, 1.0, 1.0 + params.delrho / drhodpsi], dtype=float)
    psi_diff = derm(psi, "r")

    d_r_dpsi = derm(r, "r") / psi_diff[:, None]
    d_r_dt = dermv(r, theta_st, "l", "e")
    d_z_dpsi = derm(z, "r") / psi_diff[:, None]
    d_z_dt = dermv(z, theta_st, "l", "o")

    jac = d_r_dpsi * d_z_dt - d_z_dpsi * d_r_dt
    dpsid_r = d_z_dt / jac
    dpsid_z = -d_r_dt / jac
    dt_d_r = -d_z_dpsi / jac
    dt_d_z = d_r_dpsi / jac

    dl = np.sqrt(derm(r, "l", "e") ** 2 + derm(z, "l", "o") ** 2)
    theta_common_new = np.arctan2(z, r - params.rmaj)

    d_r_l = derm(r, "l", "e")
    d_z_l = derm(z, "l", "o")
    phi_n = np.zeros_like(r)
    for i in range(r.shape[0]):
        phi = np.arctan2(d_z_l[i], d_r_l[i])
        phi_n[i] = np.concatenate((phi[phi >= 0.0] - np.pi / 2.0, phi[phi < 0.0] + 3.0 * np.pi / 2.0))
    u_ml = np.arctan2(d_z_l, d_r_l)
    r_c = dl / derm(phi_n, "l", "o")

    dpsi_dr = np.sign(psi_diff)[:, None] * np.sqrt(dpsid_r**2 + dpsid_z**2)
    b_p = np.abs(dpsi_dr) / _safe_denom(r)
    b_t = f_const / _safe_denom(r)
    b2 = b_p**2 + b_t**2
    b = np.sqrt(b2)

    l_st = np.zeros_like(r)
    for i in range(r.shape[0]):
        l_st[i, 1:] = np.cumsum(np.sqrt(np.diff(r[i]) ** 2 + np.diff(z[i]) ** 2))

    c = 1
    dt_st_l_dl_center = 1.0 / _safe_denom(dermv(l_st, theta_st, "l", "o")[c])
    gradpar_center = -(1.0 / _safe_denom(r[c] * b[c])) * dpsi_dr[c] * dt_st_l_dl_center

    return {
        "psi": psi,
        "psi_diff": psi_diff,
        "jac": jac,
        "dpsid_r": dpsid_r,
        "dpsid_z": dpsid_z,
        "dt_d_r": dt_d_r,
        "dt_d_z": dt_d_z,
        "dl": dl,
        "theta_common_new": theta_common_new,
        "u_ml": u_ml,
        "r_c": r_c,
        "dpsi_dr": dpsi_dr,
        "bpol": b_p,
        "bmag": b,
        "b2": b2,
        "l_st": l_st,
        "dt_st_l_dl_center": dt_st_l_dl_center,
        "gradpar_center": gradpar_center,
        "bmag_center": b[c],
        "bpol_center": b_p[c],
    }


def to_ballooning(theta_ex: np.ndarray, profile_ex: np.ndarray, *, parity: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert GX extended profiles to ballooning-space representation."""

    theta_ball = reflect_n_append(theta_ex, "o")
    prof_ball = reflect_n_append(profile_ex, parity)
    return theta_ball, prof_ball


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
    """Assemble GX-style Miller profiles on the selected theta grid."""

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
    l_st = np.asarray(straight_state["l_st"], dtype=float)
    psi = np.asarray(straight_state["psi"], dtype=float)
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

    # GX uses two distinct B_p computations:
    # 1. Bishop integrals: B_p from the geometric-theta grid (bpol_center parameter).
    # 2. aprime formula + drifts: B_p from the theta_st Jacobian (b_p_ex above).
    b_p_geo_ex = nperiod_data_extend(np.abs(bpol_center), params.nperiod, istheta=0, par="e")

    a_s = -(2.0 * qfac[c] / f_const * theta_st_ex + 2.0 * f_const * qfac[c] * cumulative_trapezoid(1.0 / _safe_denom(r_ex**2 * b_p_geo_ex**2), theta_st_ex))
    b_s = -(2.0 * qfac[c] * cumulative_trapezoid(1.0 / _safe_denom(b_p_geo_ex**2), theta_st_ex))
    c_s = 2.0 * qfac[c] * cumulative_trapezoid(
        (2.0 * np.sin(u_ml_ex) / _safe_denom(r_ex) - 2.0 / _safe_denom(r_c_ex))
        * (1.0 / _safe_denom(r_ex * b_p_geo_ex)),
        theta_st_ex,
    )

    prefac = rho[c] * (psi_diff[c] / diffrho[c]) * (1.0 / (2.0 * np.pi * qfac[c] * (2 * params.nperiod - 1)))
    dfdpsi = (-params.shat / _safe_denom(prefac) - (b_s[-1] * dpdpsi - c_s[-1])) / _safe_denom(a_s[-1])

    dqdr = diffq[c] * dpsi_dr_ex / psi_diff[c]
    aprime_bish = -r_ex * b_p_ex * (a_s * dfdpsi + b_s * dpdpsi - c_s) / (2.0 * np.abs(drhodpsi))
    gds21_ex = diffq[c] / diffrho[c] * (-dpsi_dr_ex) * aprime_bish

    dt_st_l_dl_center = np.asarray(straight_state["dt_st_l_dl_center"], dtype=float)
    dt_st_l_dl = nperiod_data_extend(dt_st_l_dl_center, params.nperiod, istheta=0, par="e")

    l_st_ex = np.concatenate(([0.0], np.cumsum(np.sqrt(np.diff(r_ex) ** 2 + np.diff(z_ex) ** 2))))
    dtdr_st_ex = (aprime_bish * drhodpsi - dqdr * theta_st_ex) / qfac[c]
    gds2_ex = (
        (psi_diff[c] / diffrho[c]) ** 2
        * (
            1.0 / _safe_denom(r_ex**2)
            + (dqdr * theta_st_ex) ** 2
            + (qfac[c] ** 2) * (dtdr_st_ex**2 + dt_st_l_dl**2)
            + 2.0 * qfac[c] * dqdr * theta_st_ex * dtdr_st_ex
        )
    )

    d_b2_l_ex = np.asarray(derm(b_ex**2, "l", "e"), dtype=float)
    d_b2_l_over_dl_ex = d_b2_l_ex / _safe_denom(dl_ex)
    gbdrift0_ex = (1.0 / _safe_denom(b2_ex**2)) * dpsidrho * f_const / _safe_denom(r_ex) * (dqdr * d_b2_l_over_dl_ex)

    d_bdr_bish = b_p_ex / _safe_denom(b_ex) * (
        -b_p_ex / _safe_denom(r_c_ex)
        + dpdpsi * r_ex
        - f_const**2 * np.sin(u_ml_ex) / _safe_denom(r_ex**3 * b_p_ex)
    )
    gbdrift_ex = (
        1.0
        / np.clip(np.abs(drhodpsi * b_ex**3), 1.0e-30, None)
        * (
            2.0 * b2_ex * d_bdr_bish / _safe_denom(dpsi_dr_ex)
            + aprime_bish * drhodpsi * f_const / _safe_denom(r_ex) * d_b2_l_over_dl_ex / _safe_denom(b_ex)
        )
    )
    cvdrift_ex = 1.0 / np.clip(np.abs(drhodpsi * b_ex**3), 1.0e-30, None) * (2.0 * b_ex * dpdpsi) + gbdrift_ex
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

    theta_ball, gradpar_ball = to_ballooning(theta_target_ex, gradpar_target_ex, parity="e")
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


def write_miller_eik_netcdf(path: Path, profiles: dict[str, np.ndarray | float]) -> None:
    """Write root-level GX-compatible Miller ``*.eiknc.nc`` output."""

    try:
        import netCDF4 as nc
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError("netCDF4 is required for internal Miller eik writeout") from exc

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
