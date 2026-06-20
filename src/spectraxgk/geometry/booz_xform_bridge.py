"""Bounded VMEC/Boozer differentiable bridge helpers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.autodiff_checks import (
    _sensitivity_conditioning_metadata,
    finite_difference_jacobian,
)
from spectraxgk.geometry.backend_discovery import (
    _jax_float_dtype,
    discover_differentiable_geometry_backends,
)
from spectraxgk.geometry.sensitivity import geometry_sensitivity_report


def vmec_boundary_aspect_sensitivity_report(
    params: jnp.ndarray,
    *,
    fd_step: float = 2.0e-5,
    mpol: int = 2,
    ntor: int = 0,
    ntheta: int = 96,
    nphi: int = 1,
    nfp: int = 1,
) -> dict[str, object]:
    """Validate a real ``vmec_jax`` boundary-aspect derivative when available.

    The check intentionally stops at the boundary Fourier API. Full VMEC solves
    are too expensive and environment-sensitive for the default package tests,
    but the boundary-aspect path verifies that SPECTRAX-GK can discover a
    ``vmec_jax`` checkout and differentiate through its JAX-native boundary
    data structures before higher-cost optimization workflows are promoted.
    """

    p = jnp.asarray(params, dtype=_jax_float_dtype())
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a one-dimensional length-2 vector")
    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_boundary_api_available", False):
        return {
            "available": False,
            "backend_info": info,
            "aspect": None,
            "grad_ad": None,
            "grad_fd": None,
            "max_abs_ad_fd_error": None,
            "fd_step": float(fd_step),
        }

    import vmec_jax as vj  # type: ignore[import-untyped, import-not-found]

    modes = vj.vmec_mode_table(int(mpol), int(ntor))
    grid = vj.make_angle_grid(int(ntheta), int(nphi), int(nfp))
    basis = vj.build_helical_basis(modes, grid)

    def aspect_fn(x: jnp.ndarray) -> jnp.ndarray:
        ripple, elongation = x
        r0 = 1.0
        minor = 0.22 * (1.0 + 0.5 * ripple)
        r_cos = jnp.zeros(modes.K, dtype=p.dtype).at[0].set(r0).at[1].set(minor)
        z_sin = jnp.zeros(modes.K, dtype=p.dtype).at[1].set(minor * (1.0 + elongation))
        zeros = jnp.zeros_like(r_cos)
        boundary = vj.BoundaryCoeffs(R_cos=r_cos, R_sin=zeros, Z_cos=zeros, Z_sin=z_sin)
        return vj.boundary_aspect_ratio(boundary, basis)

    grad_ad = jax.grad(aspect_fn)(p)
    grad_fd = finite_difference_jacobian(
        lambda x: jnp.asarray([aspect_fn(x)]), p, step=fd_step
    )[0]
    diff = grad_ad - grad_fd
    conditioning = _sensitivity_conditioning_metadata(
        jnp.asarray(grad_ad)[None, :],
        jnp.asarray(grad_fd)[None, :],
        p,
        fd_step=float(fd_step),
        observable_names=("aspect_ratio",),
        param_names=("ripple", "elongation"),
    )
    return {
        "available": True,
        "backend_info": info,
        "aspect": float(aspect_fn(p)),
        "grad_ad": np.asarray(grad_ad).tolist(),
        "grad_fd": np.asarray(grad_fd).tolist(),
        "max_abs_ad_fd_error": float(np.max(np.abs(np.asarray(diff)))),
        "conditioning": conditioning,
        "fd_step": float(fd_step),
        "mpol": int(mpol),
        "ntor": int(ntor),
        "ntheta": int(ntheta),
        "nphi": int(nphi),
        "nfp": int(nfp),
    }


def _booz_xform_unavailable_report(
    *,
    backend_info: dict[str, object],
    fd_step: float,
    mboz: int,
    nboz: int,
    error: str | None = None,
) -> dict[str, object]:
    """Pack the fail-closed Boozer bridge report used when the backend is absent."""

    report: dict[str, object] = {
        "available": False,
        "backend_info": backend_info,
        "objective": None,
        "grad_ad": None,
        "grad_fd": None,
        "max_abs_ad_fd_error": None,
        "fd_step": float(fd_step),
        "mboz": int(mboz),
        "nboz": int(nboz),
    }
    if error is not None:
        report["error"] = error
    return report


def _booz_xform_demo_inputs(
    ripple_value: Any,
    *,
    xm: jnp.ndarray,
    xn: jnp.ndarray,
) -> SimpleNamespace:
    """Build a one-surface axisymmetric Boozer input bundle for derivative gates."""

    r = jnp.asarray(ripple_value)
    one = jnp.asarray(1.0, dtype=r.dtype)
    zero = jnp.asarray(0.0, dtype=r.dtype)
    minor = jnp.asarray(0.22, dtype=r.dtype)
    return SimpleNamespace(
        rmnc=jnp.asarray([[one, minor]], dtype=r.dtype),
        zmns=jnp.asarray([[zero, minor]], dtype=r.dtype),
        lmns=jnp.asarray([[zero, zero]], dtype=r.dtype),
        bmnc=jnp.asarray([[one, r]], dtype=r.dtype),
        bsubumnc=jnp.asarray([[0.1, 0.0]], dtype=r.dtype),
        bsubvmnc=jnp.asarray([[one, zero]], dtype=r.dtype),
        iota=jnp.asarray([0.41], dtype=r.dtype),
        xm=xm,
        xn=xn,
        xm_nyq=xm,
        xn_nyq=xn,
        nfp=1,
        bmns=None,
        bsubumns=None,
        bsubvmns=None,
    )


def _booz_xform_spectral_objective(
    bx: Any,
    *,
    ripple_value: jnp.ndarray,
    xm: jnp.ndarray,
    xn: jnp.ndarray,
    constants: Any,
    grids: Any,
) -> jnp.ndarray:
    """Return the small Boozer magnetic-spectrum norm used by the bridge gate."""

    out = bx.booz_xform_from_inputs(
        inputs=_booz_xform_demo_inputs(ripple_value, xm=xm, xn=xn),
        constants=constants,
        grids=grids,
        jit=False,
    )
    bmnc_b = jnp.asarray(out["bmnc_b"])
    return jnp.sum(bmnc_b * bmnc_b)


def _compute_booz_xform_spectral_sensitivity(
    bx: Any,
    *,
    ripple: float,
    fd_step: float,
    mboz: int,
    nboz: int,
) -> dict[str, object]:
    """Run the bounded Boozer spectral derivative and collect output arrays."""

    xm = jnp.asarray([0, 1], dtype=jnp.int32)
    xn = jnp.asarray([0, 0], dtype=jnp.int32)
    base_inputs = _booz_xform_demo_inputs(
        jnp.asarray(ripple, dtype=jnp.float64),
        xm=xm,
        xn=xn,
    )
    constants, grids = bx.prepare_booz_xform_constants_from_inputs(
        inputs=base_inputs,
        mboz=int(mboz),
        nboz=int(nboz),
        asym=False,
    )

    def objective_fn(ripple_value: jnp.ndarray) -> jnp.ndarray:
        return _booz_xform_spectral_objective(
            bx,
            ripple_value=ripple_value,
            xm=xm,
            xn=xn,
            constants=constants,
            grids=grids,
        )

    r0 = jnp.asarray(float(ripple), dtype=jnp.float64)
    grad_ad = jax.grad(objective_fn)(r0)
    h = jnp.asarray(float(fd_step), dtype=r0.dtype)
    grad_fd = (objective_fn(r0 + h) - objective_fn(r0 - h)) / (2.0 * h)
    out = bx.booz_xform_from_inputs(
        inputs=base_inputs,
        constants=constants,
        grids=grids,
        jit=False,
    )
    return {
        "objective": float(objective_fn(r0)),
        "grad_ad": float(grad_ad),
        "grad_fd": float(grad_fd),
        "max_abs_ad_fd_error": float(jnp.abs(grad_ad - grad_fd)),
        "bmnc_b": np.asarray(out["bmnc_b"]).tolist(),
        "rmnc_b": np.asarray(out["rmnc_b"]).tolist(),
        "zmns_b": np.asarray(out["zmns_b"]).tolist(),
        "iota_b": np.asarray(out["iota_b"]).tolist(),
        "ixm_b": np.asarray(out["ixm_b"]).tolist(),
        "ixn_b": np.asarray(out["ixn_b"]).tolist(),
    }


def booz_xform_spectral_sensitivity_report(  # pragma: no cover
    *,
    ripple: float = 0.05,
    fd_step: float = 2.0e-5,
    mboz: int = 2,
    nboz: int = 0,
) -> dict[str, object]:
    """Validate a real ``booz_xform_jax`` spectral derivative when available.

    This is a deliberately tiny Boozer-transform gate. It constructs an
    axisymmetric one-surface VMEC-to-Boozer input bundle, runs the real
    ``booz_xform_jax`` functional API, and checks the derivative of a Boozer
    magnetic-spectrum norm with respect to a magnetic-ripple coefficient against
    central finite differences.

    The gate strengthens the bridge beyond import discovery while remaining
    bounded enough for examples and optional local validation. It is not a full
    VMEC-state-to-flux-tube parity claim; that requires an equilibrium solve,
    field-line sampling, and comparison against the production imported-VMEC
    geometry path.
    """

    info = discover_differentiable_geometry_backends()
    if not info.get("booz_xform_jax_api_available", False):
        return _booz_xform_unavailable_report(
            backend_info=info,
            fd_step=fd_step,
            mboz=mboz,
            nboz=nboz,
        )

    bx = importlib.import_module("booz_xform_jax.jax_api")
    try:
        payload = _compute_booz_xform_spectral_sensitivity(
            bx,
            ripple=ripple,
            fd_step=fd_step,
            mboz=mboz,
            nboz=nboz,
        )
    except Exception as exc:
        return _booz_xform_unavailable_report(
            backend_info=info,
            fd_step=fd_step,
            mboz=mboz,
            nboz=nboz,
            error=f"{type(exc).__name__}: {exc}",
        )

    return {
        "available": True,
        "backend_info": info,
        "fd_step": float(fd_step),
        "mboz": int(mboz),
        "nboz": int(nboz),
        **payload,
    }


def evaluate_boozer_bmag_on_field_line(
    theta: jnp.ndarray,
    *,
    bmnc_b: jnp.ndarray,
    ixm_b: jnp.ndarray,
    ixn_b: jnp.ndarray,
    iota: jnp.ndarray | float,
    alpha: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate a Boozer ``|B|`` spectrum and theta derivative on a field line.

    The field-line label convention is :math:`\\alpha = \\theta - \\iota\\zeta`.
    This helper is intentionally small and JAX-native so that the
    ``booz_xform_jax`` spectral output can be differentiated all the way into
    the sampled SPECTRAX-GK geometry contract.
    """

    theta_arr = jnp.asarray(theta)
    modes_m = jnp.asarray(ixm_b, dtype=theta_arr.dtype)
    modes_n = jnp.asarray(ixn_b, dtype=theta_arr.dtype)
    coeffs = jnp.asarray(bmnc_b, dtype=theta_arr.dtype)
    iota_arr = jnp.asarray(iota, dtype=theta_arr.dtype)
    iota_safe = jnp.where(
        jnp.abs(iota_arr) < 1.0e-12, jnp.sign(iota_arr + 1.0e-30) * 1.0e-12, iota_arr
    )
    zeta = (theta_arr - jnp.asarray(float(alpha), dtype=theta_arr.dtype)) / iota_safe
    phase = theta_arr[:, None] * modes_m[None, :] - zeta[:, None] * modes_n[None, :]
    dphase_dtheta = modes_m[None, :] - modes_n[None, :] / iota_safe
    bmag = jnp.sum(coeffs[None, :] * jnp.cos(phase), axis=1)
    dbmag_dtheta = jnp.sum(-coeffs[None, :] * dphase_dtheta * jnp.sin(phase), axis=1)
    return bmag, dbmag_dtheta


def booz_xform_flux_tube_mapping_from_inputs(  # pragma: no cover
    inputs: Any,
    *,
    mboz: int = 2,
    nboz: int = 1,
    ntheta: int = 96,
    alpha: float = 0.0,
    surface_index: int = 0,
    magnetic_shear: float = 0.35,
    R0: float = 1.0,
    B0: float = 1.0,
    drift_scale: float = 1.0,
    jit: bool = False,
) -> dict[str, Any]:
    """Build a solver-ready flux-tube mapping from ``booz_xform_jax`` output.

    This is the first bounded production bridge step between JAX-native Boozer
    coordinates and SPECTRAX-GK. It uses the real Boozer magnetic-field
    spectrum for ``bmag``/``bgrad`` and supplies smooth metric/drift profiles
    with the same solver-ready names accepted by
    :func:`flux_tube_geometry_from_mapping`.

    Full VMEC/Boozer metric parity remains a separate promotion gate: a
    high-fidelity backend must replace the smooth metric/drift closure here
    with sampled VMEC/Boozer metric tensors before nonlinear optimization
    claims are made.
    """

    info = discover_differentiable_geometry_backends()
    if not info.get("booz_xform_jax_api_available", False):
        raise RuntimeError("booz_xform_jax functional API is not available")

    bx = importlib.import_module("booz_xform_jax.jax_api")
    constants, grids = bx.prepare_booz_xform_constants_from_inputs(
        inputs=inputs,
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(getattr(inputs, "bmns", None) is not None),
    )
    out = bx.booz_xform_from_inputs(
        inputs=inputs, constants=constants, grids=grids, jit=bool(jit)
    )
    idx = int(surface_index)
    theta = jnp.linspace(-jnp.pi, jnp.pi, int(ntheta), endpoint=False)
    bmnc_b = jnp.asarray(out["bmnc_b"])[idx]
    iota = jnp.asarray(out["iota_b"])[idx]
    bmag, dbmag_dtheta = evaluate_boozer_bmag_on_field_line(
        theta,
        bmnc_b=bmnc_b,
        ixm_b=jnp.asarray(out["ixm_b"]),
        ixn_b=jnp.asarray(out["ixn_b"]),
        iota=iota,
        alpha=float(alpha),
    )
    q = 1.0 / jnp.maximum(jnp.abs(iota), jnp.asarray(1.0e-12, dtype=theta.dtype))
    gradpar_value = 1.0 / (q * float(R0))
    gradpar = gradpar_value * jnp.ones_like(theta)
    shear = jnp.asarray(float(magnetic_shear), dtype=theta.dtype)
    field_line_shift = shear * theta
    gds2 = 1.0 + field_line_shift * field_line_shift
    gds21 = -shear * field_line_shift
    gds22 = (1.0 + shear * shear) * jnp.ones_like(theta)
    cv = (
        float(drift_scale)
        * (jnp.cos(theta) + field_line_shift * jnp.sin(theta))
        / float(R0)
    )
    cv0 = -float(drift_scale) * shear * jnp.sin(theta) / float(R0)
    bmag_safe = jnp.maximum(jnp.abs(bmag), jnp.asarray(1.0e-12, dtype=theta.dtype))
    return {
        "theta": theta,
        "gradpar": gradpar,
        "bmag": bmag,
        "bgrad": gradpar_value * dbmag_dtheta / bmag_safe,
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "cvdrift": cv,
        "gbdrift": cv,
        "cvdrift0": cv0,
        "gbdrift0": cv0,
        "jacobian": 1.0 / (gradpar * bmag_safe),
        "grho": jnp.ones_like(theta),
        "q": q,
        "s_hat": shear,
        "epsilon": jnp.sqrt(jnp.mean((bmag / jnp.mean(bmag) - 1.0) ** 2)),
        "R0": float(R0),
        "B0": float(B0),
        "alpha": float(alpha),
        "drift_scale": float(drift_scale),
        "nfp": int(jnp.asarray(getattr(inputs, "nfp", 1))),
        "booz_xform": {
            "bmnc_b": bmnc_b,
            "ixm_b": jnp.asarray(out["ixm_b"]),
            "ixn_b": jnp.asarray(out["ixn_b"]),
            "iota_b": iota,
        },
    }


def booz_xform_flux_tube_sensitivity_report(  # pragma: no cover
    *,
    params: jnp.ndarray | None = None,
    fd_step: float = 2.0e-5,
    mboz: int = 2,
    nboz: int = 1,
    ntheta: int = 64,
) -> dict[str, object]:
    """AD/FD-check a Boozer-spectrum-to-flux-tube geometry bridge.

    ``params = [axisymmetric_B_ripple, helical_B_ripple]`` perturbs a tiny
    one-surface VMEC-to-Boozer input bundle. The real ``booz_xform_jax``
    transform is run for each parameter vector; its Boozer ``|B|`` spectrum is
    sampled on a field line and converted into ``FluxTubeGeometryData``.
    """

    info = discover_differentiable_geometry_backends()
    if not info.get("booz_xform_jax_api_available", False):
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "mboz": int(mboz),
            "nboz": int(nboz),
            "ntheta": int(ntheta),
        }

    p = jnp.asarray([0.05, 0.02] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")

    xm = jnp.asarray([0, 1, 1], dtype=jnp.int32)
    xn = jnp.asarray([0, 0, 2], dtype=jnp.int32)

    def _inputs(x: jnp.ndarray) -> SimpleNamespace:
        axisym_ripple, helical_ripple = x
        one = jnp.asarray(1.0, dtype=x.dtype)
        zero = jnp.asarray(0.0, dtype=x.dtype)
        minor = jnp.asarray(0.22, dtype=x.dtype)
        helical_shape = jnp.asarray(0.02, dtype=x.dtype)
        return SimpleNamespace(
            rmnc=jnp.asarray([[one, minor, helical_shape]], dtype=x.dtype),
            zmns=jnp.asarray([[zero, minor, helical_shape]], dtype=x.dtype),
            lmns=jnp.asarray([[zero, zero, zero]], dtype=x.dtype),
            bmnc=jnp.asarray([[one, axisym_ripple, helical_ripple]], dtype=x.dtype),
            bsubumnc=jnp.asarray([[0.1, 0.0, 0.0]], dtype=x.dtype),
            bsubvmnc=jnp.asarray([[one, zero, zero]], dtype=x.dtype),
            iota=jnp.asarray([0.41], dtype=x.dtype),
            xm=xm,
            xn=xn,
            xm_nyq=xm,
            xn_nyq=xn,
            nfp=2,
            bmns=None,
            bsubumns=None,
            bsubvmns=None,
        )

    try:
        sensitivity = geometry_sensitivity_report(
            lambda x: booz_xform_flux_tube_mapping_from_inputs(
                _inputs(x),
                mboz=int(mboz),
                nboz=int(nboz),
                ntheta=int(ntheta),
                magnetic_shear=0.35,
                jit=False,
            ),
            p,
            fd_step=float(fd_step),
            source_model="booz_xform_jax:field-line-bmag",
        )
        mapping = booz_xform_flux_tube_mapping_from_inputs(
            _inputs(p),
            mboz=int(mboz),
            nboz=int(nboz),
            ntheta=int(ntheta),
            magnetic_shear=0.35,
            jit=False,
        )
        booz_meta = mapping["booz_xform"]
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "mboz": int(mboz),
            "nboz": int(nboz),
            "ntheta": int(ntheta),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "params": np.asarray(p).tolist(),
        "sensitivity": sensitivity,
        "fd_step": float(fd_step),
        "mboz": int(mboz),
        "nboz": int(nboz),
        "ntheta": int(ntheta),
        "bmnc_b": np.asarray(booz_meta["bmnc_b"]).tolist(),
        "ixm_b": np.asarray(booz_meta["ixm_b"]).tolist(),
        "ixn_b": np.asarray(booz_meta["ixn_b"]).tolist(),
        "iota_b": float(np.asarray(booz_meta["iota_b"])),
    }


__all__ = [
    "booz_xform_flux_tube_mapping_from_inputs",
    "booz_xform_flux_tube_sensitivity_report",
    "booz_xform_spectral_sensitivity_report",
    "evaluate_boozer_bmag_on_field_line",
    "vmec_boundary_aspect_sensitivity_report",
]
