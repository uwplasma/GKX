"""Differentiable geometry bridge contracts for VMEC/JAX pipelines."""

from __future__ import annotations

from typing import Any
from collections.abc import Sequence
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import importlib
from types import SimpleNamespace

from gkx.geometry.autodiff_checks import (
    finite_difference_jacobian,
    observable_gradient_validation_report,
    _sensitivity_conditioning_metadata,
)
from gkx.geometry.backend_discovery import (
    discover_differentiable_geometry_backends,
    _jax_float_dtype,
)
from gkx.geometry.flux_tube_contract import (
    flux_tube_geometry_from_mapping,
    flux_tube_geometry_observables,
    _GEOMETRY_OBSERVABLE_NAMES,
)
from gkx.objectives.autodiff_validation import covariance_diagnostics


@dataclass(frozen=True)
class _GeometryInverseDesignProblem:
    """Validated inputs for a local geometry inverse-design solve."""

    params: jnp.ndarray
    target: jnp.ndarray
    indices_np: np.ndarray
    indices: jnp.ndarray


def _prepare_geometry_inverse_design_problem(
    initial_params: jnp.ndarray,
    target_observables: jnp.ndarray,
    observable_indices: Sequence[int] | None,
    *,
    max_steps: int,
    damping: float,
) -> _GeometryInverseDesignProblem:
    """Validate inverse-design inputs and construct selected observable indices."""

    params = jnp.asarray(initial_params, dtype=_jax_float_dtype())
    if params.ndim != 1:
        raise ValueError("initial_params must be one-dimensional")
    if int(max_steps) < 0:
        raise ValueError("max_steps must be non-negative")
    if float(damping) < 0.0:
        raise ValueError("damping must be non-negative")

    if observable_indices is None:
        indices_np = np.arange(len(_GEOMETRY_OBSERVABLE_NAMES), dtype=int)
    else:
        indices_np = np.asarray(list(observable_indices), dtype=int)
    if indices_np.ndim != 1 or indices_np.size == 0:
        raise ValueError(
            "observable_indices must be a non-empty one-dimensional sequence"
        )
    if np.any(indices_np < 0) or np.any(indices_np >= len(_GEOMETRY_OBSERVABLE_NAMES)):
        raise ValueError("observable_indices contains an out-of-range observable index")

    target = jnp.asarray(target_observables, dtype=params.dtype)
    if target.ndim != 1 or int(target.shape[0]) != int(indices_np.size):
        raise ValueError("target_observables length must match observable_indices")
    return _GeometryInverseDesignProblem(
        params=params,
        target=target,
        indices_np=indices_np,
        indices=jnp.asarray(indices_np, dtype=jnp.int32),
    )


def _geometry_observable_fn(
    mapping_fn: Any,
    indices: jnp.ndarray,
    *,
    source_model: str,
) -> Any:
    """Build the selected solver-geometry observable map used by AD/FD checks."""

    def observable_fn(x: jnp.ndarray) -> jnp.ndarray:
        geom = flux_tube_geometry_from_mapping(
            mapping_fn(x),
            source_model=source_model,
            validate_finite=False,
        )
        return flux_tube_geometry_observables(geom)[indices]

    return observable_fn


def _damped_gauss_newton_step(
    jac: jnp.ndarray,
    residual: jnp.ndarray,
    *,
    damping: float,
) -> jnp.ndarray:
    """Return one damped Gauss-Newton step, with the normal equations pinned.

    ``J^T J`` is the only matrix-times-matrix product in the loop, and XLA
    satisfies it with TF32 on Ampere and later NVIDIA GPUs. Forming the normal
    equations already squares the conditioning, so 10 mantissa bits land straight
    in the step: measured against a float64 reference on an RTX A4000 the entries
    carry 1.8e-04 and the step itself 1.7e-04, against the 1.0e-04 rtol this
    module's own AD-versus-FD report gates on. Pinned they are 4.0e-08 and
    4.7e-08. The matvec below is vector-shaped and stays exact unpinned.

    The step lives in its own function because the loop around it converts to
    host floats to build its history, so it cannot be traced -- and a CPU test
    can only see this pin in a jaxpr.
    """

    normal = jnp.matmul(jac.T, jac, precision=jax.lax.Precision.HIGHEST)
    normal = normal + float(damping) * jnp.eye(jac.shape[1], dtype=jac.dtype)
    return jnp.linalg.solve(normal, jac.T @ residual)


def _run_geometry_inverse_design_iterations(
    observable_fn: Any,
    params: jnp.ndarray,
    target: jnp.ndarray,
    *,
    max_steps: int,
    damping: float,
) -> tuple[jnp.ndarray, jnp.ndarray, list[dict[str, object]]]:
    """Run the small damped Gauss-Newton inverse-design loop."""

    history: list[dict[str, object]] = []
    p = params
    residual = observable_fn(p) - target
    for step in range(int(max_steps) + 1):
        obs = observable_fn(p)
        residual = obs - target
        objective = 0.5 * jnp.dot(residual, residual)
        history.append(
            {
                "step": int(step),
                "params": np.asarray(p).tolist(),
                "observables": np.asarray(obs).tolist(),
                "objective": float(objective),
                "residual_norm": float(jnp.linalg.norm(residual)),
            }
        )
        if step == int(max_steps):
            break
        jac = jax.jacfwd(observable_fn)(p)
        p = p - _damped_gauss_newton_step(jac, residual, damping=damping)
    return p, residual, history


def geometry_sensitivity_report(
    mapping_fn: Any,
    params: jnp.ndarray,
    *,
    fd_step: float = 1.0e-4,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    source_model: str = "vmex:in-memory",
) -> dict[str, object]:
    """Validate geometry-observable sensitivities by AD and finite differences.

    ``mapping_fn(params)`` must return the solver-ready field-line mapping
    accepted by :func:`flux_tube_geometry_from_mapping`. The report is strict
    JSON friendly so examples and CI gates can preserve the derivative
    contract without depending on large VMEC solves.
    """

    p = jnp.asarray(params, dtype=jnp.float64)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")

    def observable_fn(x: jnp.ndarray) -> jnp.ndarray:
        geom = flux_tube_geometry_from_mapping(
            mapping_fn(x),
            source_model=source_model,
            validate_finite=False,
        )
        return flux_tube_geometry_observables(geom)

    param_names = tuple(f"param_{idx}" for idx in range(int(p.shape[0])))

    report = observable_gradient_validation_report(
        observable_fn,
        p,
        fd_step=float(fd_step),
        rtol=float(rtol),
        atol=float(atol),
        observable_names=_GEOMETRY_OBSERVABLE_NAMES,
        param_names=param_names,
        relative_floor=1.0e-12,
        jacobian_chunk_size="auto",
        report_kind="geometry_sensitivity_ad_fd_gate",
    )
    report["source_model"] = str(source_model)
    return report


def _geometry_inverse_design_selected_names(indices_np: np.ndarray) -> list[str]:
    """Return stable names for the selected geometry observables."""

    return [str(_GEOMETRY_OBSERVABLE_NAMES[int(i)]) for i in indices_np]


def _geometry_inverse_design_derivative_report(
    observable_fn: Any,
    params: jnp.ndarray,
    residual: jnp.ndarray,
    *,
    fd_step: float,
    regularization: float,
    observable_names: list[str],
    param_names: tuple[str, ...],
) -> dict[str, object]:
    """Return AD/FD Jacobian, conditioning, and covariance diagnostics."""

    jac_ad = jax.jacfwd(observable_fn)(params)
    jac_fd = finite_difference_jacobian(observable_fn, params, step=fd_step)
    diff = jac_ad - jac_fd
    scale = jnp.maximum(jnp.abs(jac_fd), 1.0e-12)
    return {
        "jacobian_ad": np.asarray(jac_ad).tolist(),
        "jacobian_fd": np.asarray(jac_fd).tolist(),
        "max_abs_ad_fd_error": float(np.max(np.abs(np.asarray(diff)))),
        "max_rel_ad_fd_error": float(
            np.max(np.abs(np.asarray(diff) / np.asarray(scale)))
        ),
        "conditioning": _sensitivity_conditioning_metadata(
            jac_ad,
            jac_fd,
            params,
            fd_step=float(fd_step),
            observable_names=observable_names,
            param_names=param_names,
            relative_floor=1.0e-12,
        ),
        "uq": covariance_diagnostics(
            np.asarray(jac_ad),
            np.asarray(residual),
            regularization=regularization,
        ),
    }


def _pack_geometry_inverse_design_report(
    problem: _GeometryInverseDesignProblem,
    observable_fn: Any,
    final_params: jnp.ndarray,
    residual: jnp.ndarray,
    history: list[dict[str, object]],
    derivative_report: dict[str, object],
    *,
    fd_step: float,
    damping: float,
    regularization: float,
    source_model: str,
) -> dict[str, object]:
    """Pack the public inverse-design report schema."""

    observable_names = _geometry_inverse_design_selected_names(problem.indices_np)
    payload: dict[str, object] = {
        "observable_names": observable_names,
        "initial_params": np.asarray(problem.params).tolist(),
        "final_params": np.asarray(final_params).tolist(),
        "target_observables": np.asarray(problem.target).tolist(),
        "final_observables": np.asarray(observable_fn(final_params)).tolist(),
        "final_residual": np.asarray(residual).tolist(),
        "final_residual_norm": float(jnp.linalg.norm(residual)),
        "history": history,
        "fd_step": float(fd_step),
        "damping": float(damping),
        "regularization": float(regularization),
        "source_model": str(source_model),
        "backend_info": discover_differentiable_geometry_backends(),
    }
    payload.update(derivative_report)
    return payload


def geometry_inverse_design_report(
    mapping_fn: Any,
    initial_params: jnp.ndarray,
    target_observables: jnp.ndarray,
    *,
    observable_indices: Sequence[int] | None = None,
    max_steps: int = 8,
    damping: float = 1.0e-8,
    fd_step: float = 1.0e-4,
    regularization: float = 1.0e-8,
    source_model: str = "vmex:in-memory",
) -> dict[str, object]:
    """Run a small Gauss-Newton geometry inverse-design validation.

    ``mapping_fn(params)`` must be the same solver-ready field-line mapping
    accepted by :func:`flux_tube_geometry_from_mapping`. The routine is meant
    for differentiable ``vmex`` / ``booz_xform_jax`` workflows: it keeps
    the optimization, sensitivity check, and local UQ covariance in one
    JSON-friendly report so examples can validate the full AD contract without
    depending on a long equilibrium solve in CI.
    """

    problem = _prepare_geometry_inverse_design_problem(
        initial_params,
        target_observables,
        observable_indices,
        max_steps=max_steps,
        damping=damping,
    )
    observable_fn = _geometry_observable_fn(
        mapping_fn,
        problem.indices,
        source_model=source_model,
    )
    p, residual, history = _run_geometry_inverse_design_iterations(
        observable_fn,
        problem.params,
        problem.target,
        max_steps=max_steps,
        damping=damping,
    )

    observable_names = _geometry_inverse_design_selected_names(problem.indices_np)
    derivative_report = _geometry_inverse_design_derivative_report(
        observable_fn,
        p,
        residual,
        fd_step=fd_step,
        regularization=regularization,
        observable_names=observable_names,
        param_names=tuple(
            f"param_{idx}" for idx in range(int(problem.params.shape[0]))
        ),
    )
    return _pack_geometry_inverse_design_report(
        problem,
        observable_fn,
        p,
        residual,
        history,
        derivative_report,
        fd_step=fd_step,
        damping=damping,
        regularization=regularization,
        source_model=source_model,
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
