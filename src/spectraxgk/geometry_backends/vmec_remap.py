"""Flux-tube cutting and equal-arc remapping for VMEC geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid as _ctrap
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline, PPoly, splrep

from spectraxgk.geometry_backends.vmec_splines import _Struct


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
