"""Flux-tube cutting and equal-arc remapping for VMEC geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import cumulative_trapezoid as _ctrap
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline, PPoly, splrep

from spectraxgk.geometry_backends.vmec_types import _Struct

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

        grad_x_arr = np.array(
            [
                InterpolatedUnivariateSpline(theta, grad_x[i])(theta_cut)
                for i in range(3)
            ]
        )
        grad_y_arr = np.array(
            [
                InterpolatedUnivariateSpline(theta, grad_y[i])(theta_cut)
                for i in range(3)
            ]
        )
        bv = np.cross(grad_x_arr, grad_y_arr, axis=0)
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
            "grad_x": grad_x_arr,
            "grad_y": grad_y_arr,
            "b_vec": bv / bv_norm,
        }

    if flux_tube_cut == "none":
        return theta, _cut_and_remap(theta)

    jtwist_arr = 2.0 * geo.s_hat_input * gds21 / gds22  # twist_shift_geo_fac
    jtwist_line = jtwist_arr / y0 * x0

    def _select_crossing(crossings: np.ndarray, *, label: str) -> float:
        crossings = np.asarray(crossings, dtype=float)
        crossings = np.sort(crossings[np.isfinite(crossings) & (crossings > 0.0)])
        if npol_min is not None:
            crossings = crossings[crossings > npol_min * np.pi]
        if crossings.size == 0:
            raise ValueError(
                f"No positive {label} flux-tube crossing was found for "
                f"flux_tube_cut={flux_tube_cut!r}, npol_min={npol_min!r}. "
                "Try a different flux_tube_cut, npol_min, jtwist_in, or a larger "
                "theta/npol search range for this VMEC equilibrium."
            )
        try:
            return float(crossings[which_crossing])
        except IndexError as exc:
            raise ValueError(
                f"Requested which_crossing={which_crossing} for "
                f"flux_tube_cut={flux_tube_cut!r}, but only {crossings.size} "
                f"positive {label} crossings were found."
            ) from exc

    if flux_tube_cut == "gds21":
        tck = splrep(theta, gds21, s=0)
        ppoly = PPoly.from_spline(tck)
        cut = _select_crossing(ppoly.roots(extrapolate=False), label="gds21")

    elif flux_tube_cut == "gbdrift0":
        tck = splrep(theta, gbdrift0, s=0)
        ppoly = PPoly.from_spline(tck)
        cut = _select_crossing(ppoly.roots(extrapolate=False), label="gbdrift0")

    elif flux_tube_cut == "aspect":
        jtwist_spl = CubicSpline(theta, jtwist_line)
        if jtwist_in is not None:
            candidates = [float(-jtwist_in), float(jtwist_in)]
        else:
            n_max = jtwist_max if jtwist_max is not None else 30
            candidates = [v for v in range(-n_max, n_max + 1) if v != 0]
        crossings = np.concatenate(
            [jtwist_spl.solve(float(v), extrapolate=False) for v in candidates]
        )
        cut = _select_crossing(crossings, label="jtwist")

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
