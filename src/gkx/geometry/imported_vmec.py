"""Imported VMEC/Boozer geometry backend and EIK-file generation helpers.

This module owns the in-repo VMEC-to-imported-geometry path: optional Boozer
backend discovery, radial spline construction, Boozer field-line sampling,
flux-tube cutting, equal-arc remapping, and NetCDF writeout. Keeping the VMEC
backend in the primary ``gkx.geometry`` namespace makes the package
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

from gkx.geometry.backend_discovery import (
    _booz_xform_jax_search_paths,
    _import_module_with_search_paths,
    _import_booz_xform_jax_backend,
    _import_booz_xform_backend,
    _import_booz_backend,
    _booz_read_wout_square_layout_failure,
    _new_booz_object,
    internal_vmec_backend_available,
)
from gkx.geometry.vmec_field_line_sampling import (
    _Struct,
    _vmec_splines,
    nperiod_set,
    dermv,
)
from gkx.geometry.vmec_boozer_derivatives import (
    _assemble_fieldline_struct,
    _fieldline_metric_coefficients,
    _hngc_mode_corrections,
)
from gkx.geometry.vmec_state_controls import (
    _fieldline_scalar_profiles,
    _load_vmec_boozer_splines,
    _sample_fieldline_boozer_state,
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


def _vmec_profiles_from_equal_arc(
    geo: Any, arrays_equal_arc: dict[str, Any]
) -> dict[str, Any]:
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
