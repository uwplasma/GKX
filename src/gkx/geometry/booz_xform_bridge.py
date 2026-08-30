"""Bounded VMEC/Boozer differentiable bridge helpers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gkx.geometry.autodiff_checks import (
    _sensitivity_conditioning_metadata,
    finite_difference_jacobian,
)
from gkx.geometry.backend_discovery import (
    _jax_float_dtype,
    discover_differentiable_geometry_backends,
)


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
    """Validate a real ``vmex`` boundary-aspect derivative when available.

    The check intentionally stops at the boundary Fourier API. Full VMEC solves
    are too expensive and environment-sensitive for the default package tests,
    but the boundary-aspect path verifies that GKX can discover a
    ``vmex`` checkout and differentiate through its JAX-native boundary
    data structures before higher-cost optimization workflows are promoted.
    """

    p = jnp.asarray(params, dtype=_jax_float_dtype())
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a one-dimensional length-2 vector")
    info = discover_differentiable_geometry_backends()
    if not info.get("vmex_boundary_api_available", False):
        return {
            "available": False,
            "backend_info": info,
            "aspect": None,
            "grad_ad": None,
            "grad_fd": None,
            "max_abs_ad_fd_error": None,
            "fd_step": float(fd_step),
        }

    import vmex as vj  # type: ignore[import-untyped, import-not-found]

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
    the sampled GKX geometry contract.
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


__all__ = [
    "booz_xform_spectral_sensitivity_report",
    "evaluate_boozer_bmag_on_field_line",
    "vmec_boundary_aspect_sensitivity_report",
]
