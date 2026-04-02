"""VMEC geometry pipeline ported from GX pyvmec (gx_geo_vmec.py).

The primary public entry point is ``generate_vmec_eik_internal``.  All
physics calculations follow the original GX script as closely as possible
so that numerical parity can be verified variable by variable.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid as _ctrap
from scipy.integrate import simpson as _simps
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline, PPoly, splrep

from spectraxgk.from_gx.kernels import finite_diff_nonuniform, nperiod_contract

_MU_0 = 4.0 * np.pi * 1.0e-7


def _booz_xform_jax_search_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    raw_paths: list[Path] = []
    for env_name in ("SPECTRAX_BOOZ_XFORM_JAX_PATH", "BOOZ_XFORM_JAX_PATH"):
        raw = os.environ.get(env_name)
        if raw:
            raw_paths.append(Path(os.path.expandvars(raw)).expanduser())
    raw_paths.append(repo_root.parent / "booz_xform_jax")

    search_paths: list[Path] = []
    seen: set[Path] = set()
    for base in raw_paths:
        for candidate in (base, base / "src"):
            resolved = candidate.resolve(strict=False)
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)
            search_paths.append(resolved)
    return search_paths


def _import_module_with_search_paths(name: str, search_paths: list[Path]) -> Any:
    try:
        return importlib.import_module(name)
    except Exception:
        pass

    for path in search_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    raise ImportError(name)


def _import_booz_backend() -> Any:
    search_paths = _booz_xform_jax_search_paths()
    try:
        return _import_module_with_search_paths("booz_xform_jax", search_paths)
    except Exception:
        pass

    try:
        return importlib.import_module("booz_xform")
    except Exception as exc:
        raise ImportError("booz_xform_jax/booz_xform backend unavailable") from exc


# ---------------------------------------------------------------------------
# Public availability check
# ---------------------------------------------------------------------------

def internal_vmec_backend_available() -> bool:
    """Return True when the internal VMEC backend dependencies are present."""

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except Exception:
        return False

    try:
        _import_booz_backend()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Legacy helper wrappers (kept for back-compat)
# ---------------------------------------------------------------------------

def nperiod_set(
    values: np.ndarray, theta: np.ndarray, npol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Contract *values* / *theta* to theta in [-npol*pi, npol*pi]."""

    v = np.asarray(values)
    t = np.asarray(theta)
    if v.shape != t.shape:
        raise ValueError("values and theta must have the same shape")
    v_out, t_out = nperiod_contract(v, t, float(npol))
    return np.asarray(v_out), np.asarray(t_out)


def dermv(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Second-order non-uniform finite-difference derivative (1-D)."""

    v = np.asarray(values)
    x = np.asarray(grid)
    if v.ndim != 1 or x.ndim != 1:
        raise ValueError("dermv expects 1D arrays")
    if v.shape[0] != x.shape[0]:
        raise ValueError("values and grid must have identical lengths")
    return np.asarray(finite_diff_nonuniform(v, x))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _Struct:
    """Mutable attribute bag, mirroring the GX Struct helper."""


def _vmec_splines(nc_obj: Any, booz_obj: Any) -> _Struct:
    """Build radial splines from a VMEC netCDF object and a booz_xform object.

    Port of ``vmec_splines`` in GX's ``gx_geo_vmec.py``.
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
    """Compute GX flux-tube geometry coefficients from a VMEC equilibrium.

    Port of ``vmec_fieldlines`` in GX's ``gx_geo_vmec.py``.

    Parameters
    ----------
    vmec_fname:
        Path to a VMEC ``wout_*.nc`` file.
    s_val:
        Normalised toroidal flux (``torflux`` in GX config).
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

    bxform = _import_booz_backend()
    from netCDF4 import Dataset as _NC

    nc_obj = _NC(str(vmec_fname), "r")
    mpol = int(nc_obj.variables["mpol"][:])
    ntor = int(nc_obj.variables["ntor"][:])

    booz_obj = bxform.Booz_xform()
    booz_obj.verbose = 0
    booz_obj.read_wout(str(vmec_fname))
    booz_obj.mboz = int(2 * mpol)
    booz_obj.nboz = int(2 * ntor)
    booz_obj.run()

    vs = _vmec_splines(nc_obj, booz_obj)

    s = np.array([s_val])
    ns = 1
    alpha_arr = np.array([alpha])
    nalpha = 1
    nl = len(theta1d)

    d_pressure_d_s = vs.d_pressure_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    shat = (-2.0 * s / iota) * d_iota_d_s
    sqrt_s = np.sqrt(s)

    nfp = vs.nfp

    edge_toroidal_flux_over_2pi = -vs.phiedge / (2.0 * np.pi)
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    L_reference = vs.Aminor_p
    B_reference = 2.0 * abs(edge_toroidal_flux_over_2pi) / (L_reference ** 2)
    R_mag_ax = float(vs.raxis_cc[0])

    zeta_center = -alpha / float(iota[0])

    iota_input_val = float(iota[0]) if iota_input is None else float(iota_input)
    s_hat_input_val = float(shat[0]) if s_hat_input is None else float(s_hat_input)
    if abs(s_hat_input_val) < 1.0e-30:
        s_hat_input_val = 1.0e-8

    G = vs.Gfun(s)
    I = vs.Ifun(s)

    xm_b = vs.xm_b
    xn_b = vs.xn_b
    mnmax_b = vs.mnbooz

    rmnc_b = np.zeros((ns, mnmax_b))
    zmns_b = np.zeros((ns, mnmax_b))
    numns_b = np.zeros((ns, mnmax_b))
    d_rmnc_b_d_s = np.zeros((ns, mnmax_b))
    d_zmns_b_d_s = np.zeros((ns, mnmax_b))
    d_numns_b_d_s = np.zeros((ns, mnmax_b))
    gmnc_b = np.zeros((ns, mnmax_b))
    bmnc_b = np.zeros((ns, mnmax_b))
    d_bmnc_b_d_s = np.zeros((ns, mnmax_b))

    for jmn in range(mnmax_b):
        rmnc_b[:, jmn] = vs.rmnc_b[jmn](s)
        zmns_b[:, jmn] = vs.zmns_b[jmn](s)
        numns_b[:, jmn] = vs.numns_b[jmn](s)
        d_rmnc_b_d_s[:, jmn] = vs.d_rmnc_b_d_s[jmn](s)
        d_zmns_b_d_s[:, jmn] = vs.d_zmns_b_d_s[jmn](s)
        d_numns_b_d_s[:, jmn] = vs.d_numns_b_d_s[jmn](s)
        gmnc_b[:, jmn] = vs.gmnc_b[jmn](s)
        bmnc_b[:, jmn] = vs.bmnc_b[jmn](s)
        d_bmnc_b_d_s[:, jmn] = vs.d_bmnc_b_d_s[jmn](s)

    # Field line (theta_b, phi_b) along alpha = theta_b - iota * phi_b
    theta_b = np.zeros((ns, nalpha, nl))
    phi_b = np.zeros((ns, nalpha, nl))
    for js in range(ns):
        theta_b[js, :, :] = theta1d[None, :]
        phi_b[js, :, :] = (theta1d[None, :] - alpha_arr[:, None]) / iota[js]

    # Flipit check for axisymmetric equilibria
    angle_b_chk = (
        xm_b[:, None, None, None] * theta_b[None, :, :, :]
        - xn_b[:, None, None, None] * phi_b[None, :, :, :]
    )
    R_b_chk = np.einsum("ij,jikl->ikl", rmnc_b, np.cos(angle_b_chk))
    Z_b_chk = np.einsum("ij,jikl->ikl", zmns_b, np.sin(angle_b_chk))
    flipit = 0
    if isaxisym:
        if (
            R_b_chk[0, 0, 0] > R_b_chk[0, 0, 1]
            or Z_b_chk[0, 0, 1] > Z_b_chk[0, 0, 0]
        ):
            flipit = 1

    if flipit:
        angle_b = (
            xm_b[:, None, None, None] * (theta_b[None, :, :, :] + np.pi)
            - xn_b[:, None, None, None] * phi_b[None, :, :, :]
        )
    else:
        angle_b = (
            xm_b[:, None, None, None] * theta_b[None, :, :, :]
            - xn_b[:, None, None, None] * phi_b[None, :, :, :]
        )

    cosangle_b = np.cos(angle_b)
    sinangle_b = np.sin(angle_b)
    mcosangle_b = xm_b[:, None, None, None] * cosangle_b
    msinangle_b = xm_b[:, None, None, None] * sinangle_b
    ncosangle_b = xn_b[:, None, None, None] * cosangle_b
    nsinangle_b = xn_b[:, None, None, None] * sinangle_b

    R_b = np.einsum("ij,jikl->ikl", rmnc_b, cosangle_b)
    d_R_b_d_s = np.einsum("ij,jikl->ikl", d_rmnc_b_d_s, cosangle_b)
    d_R_b_d_theta_b = -np.einsum("ij,jikl->ikl", rmnc_b, msinangle_b)
    d_R_b_d_phi_b = np.einsum("ij,jikl->ikl", rmnc_b, nsinangle_b)

    Z_b = np.einsum("ij,jikl->ikl", zmns_b, sinangle_b)
    d_Z_b_d_s = np.einsum("ij,jikl->ikl", d_zmns_b_d_s, sinangle_b)
    d_Z_b_d_theta_b = np.einsum("ij,jikl->ikl", zmns_b, mcosangle_b)
    d_Z_b_d_phi_b = -np.einsum("ij,jikl->ikl", zmns_b, ncosangle_b)

    nu_b = np.einsum("ij,jikl->ikl", numns_b, sinangle_b)
    d_nu_b_d_s = np.einsum("ij,jikl->ikl", d_numns_b_d_s, sinangle_b)
    d_nu_b_d_theta_b = np.einsum("ij,jikl->ikl", numns_b, mcosangle_b)
    d_nu_b_d_phi_b = -np.einsum("ij,jikl->ikl", numns_b, ncosangle_b)

    sqrt_g_booz = np.einsum("ij,jikl->ikl", gmnc_b, cosangle_b)
    d_sqrt_g_booz_d_theta_b = -np.einsum("ij,jikl->ikl", gmnc_b, msinangle_b)
    d_sqrt_g_booz_d_phi_b = np.einsum("ij,jikl->ikl", gmnc_b, nsinangle_b)
    modB_b = np.einsum("ij,jikl->ikl", bmnc_b, cosangle_b)
    d_B_b_d_s = np.einsum("ij,jikl->ikl", d_bmnc_b_d_s, cosangle_b)

    Vprime = gmnc_b[:, 0]  # flux-surface volume element (m=0, n=0 Boozer mode)

    # Lambda / beta corrections (Hegna-Nakajima)
    delmnc_b = np.zeros((ns, mnmax_b))
    lambmnc_b = np.zeros((ns, mnmax_b))
    betamns_b = np.zeros((ns, mnmax_b))

    denom_mn = xm_b[1:] * iota[:, None] - xn_b[1:]
    safe_denom_mn = np.where(
        np.abs(denom_mn) < 1.0e-30,
        np.sign(denom_mn + 1.0e-300) * 1.0e-30,
        denom_mn,
    )

    delmnc_b[:, 1:] = gmnc_b[:, 1:] / Vprime[:, None]
    betamns_b[:, 1:] = (
        delmnc_b[:, 1:]
        / edge_toroidal_flux_over_2pi
        * _MU_0
        * d_pressure_d_s[:, None]
        * Vprime[:, None]
        / safe_denom_mn
    )
    lambmnc_b[:, 1:] = (
        delmnc_b[:, 1:]
        * (xm_b[1:] * G[:, None] + xn_b[1:] * I[:, None])
        / (safe_denom_mn * (G[:, None] + iota[:, None] * I[:, None]))
    )

    beta_b = np.einsum("ij,jikl->ikl", betamns_b, sinangle_b)
    lambda_b = np.einsum("ij,jikl->ikl", lambmnc_b, cosangle_b)

    # Cartesian coordinate derivatives for basis vectors
    phi_cyl = phi_b - nu_b
    sinphi = np.sin(phi_cyl)
    cosphi = np.cos(phi_cyl)

    d_X_d_theta_b = d_R_b_d_theta_b * cosphi - R_b * sinphi * (-d_nu_b_d_theta_b)
    d_X_d_phi_b = d_R_b_d_phi_b * cosphi - R_b * sinphi * (1.0 - d_nu_b_d_phi_b)
    d_X_d_s = d_R_b_d_s * cosphi - R_b * sinphi * (-d_nu_b_d_s)

    d_Y_d_theta_b = d_R_b_d_theta_b * sinphi + R_b * cosphi * (-d_nu_b_d_theta_b)
    d_Y_d_phi_b = d_R_b_d_phi_b * sinphi + R_b * cosphi * (1.0 - d_nu_b_d_phi_b)
    d_Y_d_s = d_R_b_d_s * sinphi + R_b * cosphi * (-d_nu_b_d_s)

    # Dual (contravariant) gradient vectors via cross products
    grad_psi_X = (
        d_Y_d_theta_b * d_Z_b_d_phi_b - d_Z_b_d_theta_b * d_Y_d_phi_b
    ) / sqrt_g_booz
    grad_psi_Y = (
        d_Z_b_d_theta_b * d_X_d_phi_b - d_X_d_theta_b * d_Z_b_d_phi_b
    ) / sqrt_g_booz
    grad_psi_Z = (
        d_X_d_theta_b * d_Y_d_phi_b - d_Y_d_theta_b * d_X_d_phi_b
    ) / sqrt_g_booz

    g_sup_psi_psi = grad_psi_X ** 2 + grad_psi_Y ** 2 + grad_psi_Z ** 2

    _etf = edge_toroidal_flux_over_2pi
    grad_theta_b_X = (
        d_Y_d_phi_b * d_Z_b_d_s - d_Z_b_d_phi_b * d_Y_d_s
    ) / (sqrt_g_booz * _etf)
    grad_theta_b_Y = (
        d_Z_b_d_phi_b * d_X_d_s - d_X_d_phi_b * d_Z_b_d_s
    ) / (sqrt_g_booz * _etf)
    grad_theta_b_Z = (
        d_X_d_phi_b * d_Y_d_s - d_Y_d_phi_b * d_X_d_s
    ) / (sqrt_g_booz * _etf)

    grad_phi_b_X = (
        d_Y_d_s * d_Z_b_d_theta_b - d_Z_b_d_s * d_Y_d_theta_b
    ) / (sqrt_g_booz * _etf)
    grad_phi_b_Y = (
        d_Z_b_d_s * d_X_d_theta_b - d_X_d_s * d_Z_b_d_theta_b
    ) / (sqrt_g_booz * _etf)
    grad_phi_b_Z = (
        d_X_d_s * d_Y_d_theta_b - d_Y_d_s * d_X_d_theta_b
    ) / (sqrt_g_booz * _etf)

    grad_alpha_X = (
        -(phi_b - zeta_center) * d_iota_d_s[:, None, None] * grad_psi_X / _etf
        + grad_theta_b_X
        - iota[:, None, None] * grad_phi_b_X
    )
    grad_alpha_Y = (
        -(phi_b - zeta_center) * d_iota_d_s[:, None, None] * grad_psi_Y / _etf
        + grad_theta_b_Y
        - iota[:, None, None] * grad_phi_b_Y
    )
    grad_alpha_Z = (
        -(phi_b - zeta_center) * d_iota_d_s[:, None, None] * grad_psi_Z / _etf
        + grad_theta_b_Z
        - iota[:, None, None] * grad_phi_b_Z
    )

    # ------------------------------------------------------------------
    # Flux-surface integrals D1, D2 (Hegna-Nakajima correction)
    # ------------------------------------------------------------------
    theta_b_grid = np.linspace(-np.pi, np.pi, res_theta)
    phi_b_grid = np.linspace(-np.pi, np.pi, res_phi)
    th_b_2D, ph_b_2D = np.meshgrid(theta_b_grid, phi_b_grid)  # (res_phi, res_theta)

    if flipit:
        angle_b_2D = (
            xm_b[:, None, None, None, None] * (th_b_2D[None, None, None, :, :] + np.pi)
            - xn_b[:, None, None, None, None] * ph_b_2D[None, None, None, :, :]
        )
    else:
        angle_b_2D = (
            xm_b[:, None, None, None, None] * th_b_2D[None, None, None, :, :]
            - xn_b[:, None, None, None, None] * ph_b_2D[None, None, None, :, :]
        )

    cosangle_b_2D = np.cos(angle_b_2D)
    sinangle_b_2D = np.sin(angle_b_2D)
    mcosangle_b_2D = xm_b[:, None, None, None, None] * cosangle_b_2D
    ncosangle_b_2D = xn_b[:, None, None, None, None] * cosangle_b_2D
    msinangle_b_2D = xm_b[:, None, None, None, None] * sinangle_b_2D
    nsinangle_b_2D = xn_b[:, None, None, None, None] * sinangle_b_2D

    lambda_b_2D = np.einsum("ij,jiklm->iklm", lambmnc_b, cosangle_b_2D)
    R_b_2D = np.einsum("ij,jiklm->iklm", rmnc_b, cosangle_b_2D)
    d_R_b_d_theta_b_2D = -np.einsum("ij,jiklm->iklm", rmnc_b, msinangle_b_2D)
    d_R_b_d_phi_b_2D = np.einsum("ij,jiklm->iklm", rmnc_b, ncosangle_b_2D)
    d_Z_b_d_theta_b_2D = np.einsum("ij,jiklm->iklm", zmns_b, mcosangle_b_2D)
    d_Z_b_d_phi_b_2D = -np.einsum("ij,jiklm->iklm", zmns_b, ncosangle_b_2D)
    nu_b_2D = np.einsum("ij,jiklm->iklm", numns_b, sinangle_b_2D)
    d_nu_b_d_theta_b_2D = np.einsum("ij,jiklm->iklm", numns_b, mcosangle_b_2D)
    d_nu_b_d_phi_b_2D = -np.einsum("ij,jiklm->iklm", numns_b, nsinangle_b_2D)
    sqrt_g_booz_2D = np.einsum("ij,jiklm->iklm", gmnc_b, cosangle_b_2D)
    Z_b_2D = np.einsum("ij,jiklm->iklm", zmns_b, sinangle_b_2D)  # noqa: F841

    ph_nat_2D = ph_b_2D - nu_b_2D
    sinphi_2D = np.sin(ph_nat_2D)
    cosphi_2D = np.cos(ph_nat_2D)

    d_X_d_th_b_2D = (
        d_R_b_d_theta_b_2D * cosphi_2D
        - R_b_2D * sinphi_2D * (-d_nu_b_d_theta_b_2D)
    )
    d_X_d_phi_2D = (
        d_R_b_d_phi_b_2D * cosphi_2D
        - R_b_2D * sinphi_2D * (1.0 - d_nu_b_d_phi_b_2D)
    )
    d_Y_d_th_b_2D = (
        d_R_b_d_theta_b_2D * sinphi_2D
        + R_b_2D * cosphi_2D * (-d_nu_b_d_theta_b_2D)
    )
    d_Y_d_phi_2D = (
        d_R_b_d_phi_b_2D * sinphi_2D
        + R_b_2D * cosphi_2D * (1.0 - d_nu_b_d_phi_b_2D)
    )

    grad_psi_X_2D = (
        d_Y_d_th_b_2D * d_Z_b_d_phi_b_2D - d_Z_b_d_theta_b_2D * d_Y_d_phi_2D
    ) / sqrt_g_booz_2D
    grad_psi_Y_2D = (
        d_Z_b_d_theta_b_2D * d_X_d_phi_2D - d_X_d_th_b_2D * d_Z_b_d_phi_b_2D
    ) / sqrt_g_booz_2D
    grad_psi_Z_2D = (
        d_X_d_th_b_2D * d_Y_d_phi_2D - d_Y_d_th_b_2D * d_X_d_phi_2D
    ) / sqrt_g_booz_2D

    g_sup_psi_psi_2D = grad_psi_X_2D ** 2 + grad_psi_Y_2D ** 2 + grad_psi_Z_2D ** 2
    g_sup_psi_psi_2D_inv = 1.0 / g_sup_psi_psi_2D
    lam_over_g_2D = lambda_b_2D * g_sup_psi_psi_2D_inv

    # Flux-surface averages over the (res_phi, res_theta) 2-D grid
    D1 = _simps(
        [_simps(row, x=theta_b_grid) for row in g_sup_psi_psi_2D_inv[0, 0]],
        x=phi_b_grid,
    ) / (2.0 * np.pi) ** 2

    D2 = _simps(
        [_simps(row, x=theta_b_grid) for row in lam_over_g_2D[0, 0]],
        x=phi_b_grid,
    ) / (2.0 * np.pi) ** 2

    # Cumulative integrals along the field line
    intinv_g = _ctrap(1.0 / g_sup_psi_psi, phi_b, initial=0)
    int_lam_div_g = _ctrap(lambda_b / g_sup_psi_psi, phi_b, initial=0)

    # Subtract the value at theta = 0 (field-line midpoint)
    theta_1d = theta_b[0, 0]
    spl0 = InterpolatedUnivariateSpline(theta_1d, intinv_g[0, 0])
    intinv_g = intinv_g - spl0(0.0)
    spl1 = InterpolatedUnivariateSpline(theta_1d, int_lam_div_g[0, 0])
    int_lam_div_g = int_lam_div_g - spl1(0.0)

    # HNGC correction factors
    d_iota_d_s_1 = (
        -(iota_input_val / (2.0 * s_val)) * s_hat_input_val
        + (float(iota[0]) / (2.0 * s_val)) * float(shat[0])
    ) * np.ones((ns,))
    sfac = float(shat[0]) / s_hat_input_val

    if not include_shear_variation:
        d_iota_d_s_1 = np.zeros((ns,))
        sfac = 1.0

    d_pressure_d_s_1 = (
        betaprim / (4.0 * np.sqrt(s_val)) * B_reference ** 2 * np.ones((ns,))
        - _MU_0 * d_pressure_d_s * np.ones((ns,))
    )

    dp_ds_safe = np.where(
        np.abs(d_pressure_d_s) < 1.0e-30, 1.0e-8, d_pressure_d_s
    )
    pfac = (
        betaprim * B_reference ** 2 / (4.0 * np.sqrt(s_val)) / (_MU_0 * float(dp_ds_safe[0]))
    )

    if not include_pressure_variation:
        pfac = 1.0
        d_pressure_d_s_1 = np.zeros((ns,))

    D_HNGC = (
        1.0 / _etf
        * (
            d_iota_d_s_1[:, None, None]
            * (intinv_g / D1 - phi_b + zeta_center)
            - d_pressure_d_s_1[:, None, None]
            * Vprime[:, None, None]
            * (G[:, None, None] + iota[:, None, None] * I[:, None, None])
            * (int_lam_div_g - D2 * intinv_g / D1)
        )
    )

    # Integrated local shear L0, L1
    grad_alpha_dot_grad_psi = (
        grad_alpha_X * grad_psi_X
        + grad_alpha_Y * grad_psi_Y
        + grad_alpha_Z * grad_psi_Z
    )

    L0 = -1.0 * (
        grad_alpha_dot_grad_psi / g_sup_psi_psi
        + 1.0 / _etf * d_iota_d_s[:, None, None] * (phi_b - zeta_center)
    )
    L1 = (
        -1.0 / _etf * d_iota_d_s_1[:, None, None] * (phi_b - zeta_center)
        + grad_alpha_dot_grad_psi / g_sup_psi_psi
        - D_HNGC
    )

    # Curvature components (normal and geodesic)
    kappa_n = (
        1.0 / modB_b ** 2
        * (modB_b * d_B_b_d_s + _MU_0 * d_pressure_d_s[:, None, None])
        / _etf
        - beta_b
        / (2.0 * sqrt_g_booz * (G[:, None, None] + iota[:, None, None] * I[:, None, None]))
        * d_sqrt_g_booz_d_phi_b
        + L0
        * (G[:, None] * d_sqrt_g_booz_d_theta_b - I[:, None] * d_sqrt_g_booz_d_phi_b)
        / (2.0 * sqrt_g_booz * (G[:, None] + iota[:, None] * I[:, None]))
    )
    kappa_g = (
        G[:, None] * d_sqrt_g_booz_d_theta_b - I[:, None] * d_sqrt_g_booz_d_phi_b
    ) / (2.0 * sqrt_g_booz * (G[:, None] + iota[:, None] * I[:, None]))

    B_cross_kappa_dot_grad_alpha = (kappa_n + kappa_g * L1) * modB_b ** 2
    B_cross_kappa_dot_grad_psi = kappa_g * modB_b ** 2

    # GX geometry coefficients -------------------------------------------
    bmag = modB_b / B_reference
    gradpar_theta_b = -L_reference / modB_b / sqrt_g_booz * iota[:, None, None]

    grad_alpha_dot_grad_alpha_b = modB_b ** 2 / g_sup_psi_psi + g_sup_psi_psi * L1 ** 2
    grad_alpha_dot_grad_psi_b = g_sup_psi_psi * L1

    gds2 = grad_alpha_dot_grad_alpha_b * L_reference ** 2 * s[:, None, None]
    gds21 = grad_alpha_dot_grad_psi_b * sfac * shat[:, None, None] / B_reference
    gds22 = (
        g_sup_psi_psi
        * (sfac * shat[:, None, None]) ** 2
        / (L_reference ** 2 * B_reference ** 2 * s[:, None, None])
    )
    grho = np.sqrt(
        g_sup_psi_psi / (L_reference ** 2 * B_reference ** 2 * s[:, None, None])
    )

    gbdrift0 = (
        -B_cross_kappa_dot_grad_psi
        * 2.0 * sfac * shat[:, None, None]
        / (modB_b ** 2 * sqrt_s[:, None, None])
        * toroidal_flux_sign
    )
    cvdrift0 = gbdrift0

    cvdrift = (
        -2.0
        * B_reference
        * L_reference ** 2
        * sqrt_s[:, None, None]
        * B_cross_kappa_dot_grad_alpha
        / modB_b ** 2
        * toroidal_flux_sign
    )
    gbdrift = cvdrift + (
        2.0
        * B_reference
        * L_reference ** 2
        * sqrt_s[:, None, None]
        * _MU_0
        * pfac
        * d_pressure_d_s[:, None, None]
        * toroidal_flux_sign
        / (_etf * modB_b ** 2)
    )

    theta_PEST = theta_b - iota[:, None, None] * nu_b
    theta_geo = np.arctan2(Z_b, R_b - R_mag_ax)

    grad_y = (
        L_reference
        * np.sqrt(s[:, None, None])
        * np.array([grad_alpha_X, grad_alpha_Y, grad_alpha_Z])
    )
    grad_x = (
        sfac
        * shat[:, None, None]
        * np.array([grad_psi_X, grad_psi_Y, grad_psi_Z])
        / (L_reference * B_reference * np.sqrt(s[:, None, None]))
    )

    nc_obj.close()

    # Pack results
    r = _Struct()
    r.iota_input = iota_input_val
    r.d_iota_d_s = d_iota_d_s
    r.d_pressure_d_s = d_pressure_d_s
    r.s_hat_input = float(shat[0])
    r.alpha = alpha
    r.theta_b = theta_b
    r.phi_b = phi_b
    r.theta_PEST = theta_PEST
    r.theta_geo = theta_geo
    r.edge_toroidal_flux_over_2pi = edge_toroidal_flux_over_2pi
    r.R_b = R_b
    r.Z_b = Z_b
    r.betaprim = betaprim
    r.bmag = bmag
    r.gradpar_theta_b = gradpar_theta_b
    r.gradpar_phi = L_reference / modB_b / sqrt_g_booz
    r.gds2 = gds2
    r.gds21 = gds21
    r.gds22 = gds22
    r.gbdrift = gbdrift
    r.gbdrift0 = gbdrift0
    r.cvdrift = cvdrift
    r.cvdrift0 = cvdrift0
    r.grho = grho
    r.grad_y = grad_y
    r.grad_x = grad_x
    r.zeta_center = zeta_center
    r.nfp = nfp
    r.L_reference = L_reference
    r.B_reference = B_reference
    r.dpsidrho = 2.0 * np.sqrt(s_val) * edge_toroidal_flux_over_2pi
    return r


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

    Port of the cut section in GX's ``gx_geo_vmec.py`` main script.
    """

    def _sl(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr[0, 0])

    bmag = _sl(geo.bmag)
    gradpar = np.abs(_sl(geo.gradpar_theta_b))
    cvdrift = _sl(geo.cvdrift)
    gbdrift = _sl(geo.gbdrift)
    cvdrift0 = _sl(geo.cvdrift0)
    gbdrift0 = _sl(geo.gbdrift0)
    gds2 = _sl(geo.gds2)
    gds21 = _sl(geo.gds21)
    gds22 = _sl(geo.gds22)
    grho = _sl(geo.grho)
    R = _sl(geo.R_b)
    Z = _sl(geo.Z_b)
    grad_x = geo.grad_x[:, 0, 0, :]  # shape (3, nl)
    grad_y = geo.grad_y[:, 0, 0, :]  # shape (3, nl)

    def _cut_and_remap(theta_cut: np.ndarray) -> dict[str, np.ndarray]:
        def _interp(arr: np.ndarray) -> np.ndarray:
            spl = InterpolatedUnivariateSpline(theta, arr)
            return spl(theta_cut)

        gx_arr = np.array([
            InterpolatedUnivariateSpline(theta, grad_x[i])(theta_cut) for i in range(3)
        ])
        gy_arr = np.array([
            InterpolatedUnivariateSpline(theta, grad_y[i])(theta_cut) for i in range(3)
        ])
        bv = np.cross(gx_arr, gy_arr, axis=0)
        bv_norm = np.linalg.norm(bv, axis=0, keepdims=True)
        bv_norm = np.where(bv_norm < 1.0e-300, 1.0e-300, bv_norm)
        return {
            "theta": theta_cut,
            "theta_PEST": theta_cut,
            "bmag": _interp(bmag),
            "gradpar": _interp(gradpar),
            "cvdrift": _interp(cvdrift),
            "gbdrift": _interp(gbdrift),
            "cvdrift0": _interp(cvdrift0),
            "gbdrift0": _interp(gbdrift0),
            "gds2": _interp(gds2),
            "gds21": _interp(gds21),
            "gds22": _interp(gds22),
            "grho": _interp(grho),
            "Rplot": _interp(R),
            "Zplot": _interp(Z),
            "grad_x": gx_arr,
            "grad_y": gy_arr,
            "b_vec": bv / bv_norm,
        }

    if flux_tube_cut == "none":
        return theta, _cut_and_remap(theta)

    jtwist_arr = 2.0 * geo.s_hat_input * gds21 / gds22  # twist_shift_geo_fac
    jtwist_line = jtwist_arr / y0 * x0

    if flux_tube_cut == "gds21":
        tck = splrep(theta, gds21, s=0)
        ppoly = PPoly.from_spline(tck)
        roots = ppoly.roots(extrapolate=False)
        roots = roots[roots > 0]
        if npol_min is not None:
            roots = roots[roots > npol_min * np.pi]
        cut = float(roots[which_crossing])

    elif flux_tube_cut == "gbdrift0":
        tck = splrep(theta, gbdrift0, s=0)
        ppoly = PPoly.from_spline(tck)
        roots = ppoly.roots(extrapolate=False)
        roots = roots[roots > 0]
        if npol_min is not None:
            roots = roots[roots > npol_min * np.pi]
        cut = float(roots[which_crossing])

    elif flux_tube_cut == "aspect":
        jtwist_spl = CubicSpline(theta, jtwist_line)
        if jtwist_in is not None:
            candidates = [float(-jtwist_in), float(jtwist_in)]
        else:
            n_max = jtwist_max if jtwist_max is not None else 30
            candidates = [v for v in range(-n_max, n_max + 1) if v != 0]
        crossings = np.concatenate([
            jtwist_spl.solve(float(v), extrapolate=False) for v in candidates
        ])
        crossings = np.sort(crossings[crossings > 0])
        if npol_min is not None:
            crossings = crossings[crossings > npol_min * np.pi]
        cut = float(crossings[which_crossing])

    else:
        raise ValueError(f"Unknown flux_tube_cut={flux_tube_cut!r}")

    theta_cut = np.linspace(-cut, cut, ntheta)
    return theta_cut, _cut_and_remap(theta_cut)


def _equal_arc_remap(
    theta: np.ndarray,
    arrays: dict[str, np.ndarray],
    ntheta: int,
) -> tuple[float, dict[str, Any]]:
    """Remap all geometry arrays from the cut theta grid to equal-arc theta.

    Port of the equal-arc section in GX's ``gx_geo_vmec.py`` main script.

    Returns ``(gradpar_eqarc, remapped_arrays_dict)`` where
    ``remapped_arrays_dict["scale"]`` holds the domain scaling factor.
    """

    gradpar = arrays["gradpar"]
    inv_gradpar_int = _ctrap(1.0 / gradpar, theta, initial=0)
    gradpar_eqarc = 2.0 * np.pi / float(inv_gradpar_int[-1])
    theta_eqarc = gradpar_eqarc * inv_gradpar_int - np.pi
    domain_scaling_factor = float(theta[-1]) / float(theta_eqarc[-1])

    theta_GX = np.linspace(-np.pi, np.pi, ntheta)

    def _interp(arr: np.ndarray) -> np.ndarray:
        return np.interp(theta_GX, theta_eqarc, arr)

    gx_arr = np.array([_interp(arrays["grad_x"][i]) for i in range(3)])
    gy_arr = np.array([_interp(arrays["grad_y"][i]) for i in range(3)])
    bv = np.cross(gx_arr, gy_arr, axis=0)
    bv_norm = np.linalg.norm(bv, axis=0, keepdims=True)
    bv_norm = np.where(bv_norm < 1.0e-300, 1.0e-300, bv_norm)

    out: dict[str, Any] = {
        "theta": theta_GX,
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
        "grad_x": gx_arr,
        "grad_y": gy_arr,
        "b_vec": bv / bv_norm,
        "scale": domain_scaling_factor,
    }
    return gradpar_eqarc, out


def write_vmec_eik_netcdf(
    path: Path,
    profiles: dict[str, Any],
    *,
    request: Any,
) -> None:
    """Write a GX-compatible VMEC ``*.eik.nc`` file.

    The output format matches GX's ``gx_geo_vmec.py`` netCDF section.
    Drift coefficients are stored at the pre-GX-normalisation (2x) level;
    the loader (``load_gx_geometry_netcdf``) divides by 2 on read.
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
        _v("grad_x", dims=("3", "z"))[:, :] = np.asarray(profiles["grad_x"], dtype=float)
        _v("grad_y", dims=("3", "z"))[:, :] = np.asarray(profiles["grad_y"], dtype=float)
        _v("b_vec", dims=("3", "z"))[:, :] = np.asarray(profiles["b_vec"], dtype=float)

        ds.createVariable("drhodpsi", "f8").assignValue(drhodpsi)
        ds.createVariable("kxfac", "f8").assignValue(float(profiles.get("kxfac", 1.0)))
        ds.createVariable("Rmaj", "f8").assignValue(float(profiles["Rmaj"]))
        ds.createVariable("q", "f8").assignValue(float(profiles["q"]))
        ds.createVariable("shat", "f8").assignValue(float(profiles["shat"]))
        ds.createVariable("scale", "f8").assignValue(float(profiles["scale"]))
        ds.createVariable("alpha", "f8").assignValue(float(profiles["alpha"]))
        ds.createVariable("zeta_center", "f8").assignValue(float(profiles["zeta_center"]))
        ds.createVariable("nfp", "i4").assignValue(int(profiles["nfp"]))


def generate_vmec_eik_internal(
    *,
    output_path: str | Path,
    request: Any | None = None,
) -> Path:
    """Internal VMEC->EIK pipeline.

    Port of GX's ``gx_geo_vmec.py`` main script.  Accepts a
    ``GXVmecGeometryRequest`` and writes a GX-compatible ``*.eik.nc`` file.
    """

    if request is None:
        raise NotImplementedError(
            "Internal VMEC geometry backend requires a GXVmecGeometryRequest. "
            "Pass request= to invoke the port."
        )

    npol = float(request.npol)
    npol_min = None if request.npol_min is None else float(request.npol_min)
    ntheta_in = int(request.ntheta)
    ntheta = ntheta_in + 1  # GX convention: ntheta_in + 1 output points

    # Map boundary string to flux-tube-cut type (GX convention)
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
        _ = z  # keep parity with GX species contract; charge does not enter betaprim expression
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
    out.parent.mkdir(parents=True, exist_ok=True)
    write_vmec_eik_netcdf(out, profiles, request=request)
    return out
