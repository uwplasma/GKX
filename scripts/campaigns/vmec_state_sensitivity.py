"""VMEC-state differentiable sensitivity reports (vmex-backed)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gkx.geometry.autodiff_checks import (
    _sensitivity_conditioning_metadata,
    finite_difference_jacobian,
)
from gkx.geometry.backend_discovery import (
    discover_differentiable_geometry_backends,
)
from gkx.geometry.flux_tube_contract import (
    _VMEC_FIELD_LINE_OBSERVABLE_NAMES,
    _VMEC_METRIC_OBSERVABLE_NAMES,
)
from gkx.geometry.vmec_field_line_sampling import _rms_with_floor
from gkx.geometry.vmec_tensor_mapping import _import_vmex_turbulence
from dataclasses import replace as dc_replace
from gkx.geometry.vmec_boozer_core import (
    load_solved_vmex_case,
    resolve_vmex_case_input_path,
)


#: Provenance marker: vmex equilibria are solved in memory, without a wout file.
VMEC_STATE_IN_MEMORY_WOUT_PATH = "in-memory:vmex.optimize.solve_equilibrium"


@dataclass(frozen=True)
class _VMECStateContext:
    """Solved vmex example state plus coefficient arrays used by AD gates.

    ``base_Rcos``/``base_Zsin`` mirror the vmex ``R_cos``/``Z_sin`` spectral
    tables; ``wout_path`` is the in-memory provenance marker because vmex
    solves the equilibrium directly instead of reading a wout file.
    """

    input_path: Any
    wout_path: Any
    inp: Any
    runtime: Any
    wout: Any
    state: Any
    base_Rcos: jnp.ndarray
    base_Zsin: jnp.ndarray


def _load_vmec_state_context(case_name: str) -> _VMECStateContext:
    """Solve a bundled vmex example and expose differentiable state arrays."""

    input_path = resolve_vmex_case_input_path(str(case_name))
    inp, state, runtime, wout = load_solved_vmex_case(str(case_name))
    base_Rcos = jnp.asarray(state.R_cos)
    base_Zsin = jnp.asarray(state.Z_sin)
    if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
        raise RuntimeError("vmex state R_cos/Z_sin arrays must be two-dimensional")
    return _VMECStateContext(
        input_path=input_path,
        wout_path=VMEC_STATE_IN_MEMORY_WOUT_PATH,
        inp=inp,
        runtime=runtime,
        wout=wout,
        state=state,
        base_Rcos=base_Rcos,
        base_Zsin=base_Zsin,
    )


def _resolve_vmec_state_indices(
    base_Rcos: jnp.ndarray,
    *,
    radial_index: int | None,
    mode_index: int,
    surface_index: int | None,
    surface_grid: str,
) -> tuple[int, int, int]:
    """Resolve coefficient and surface indices for VMEC-state sensitivity gates."""

    ns_full = int(base_Rcos.shape[0])
    ridx = ns_full // 2 if radial_index is None else int(radial_index)
    midx = int(mode_index)
    if not (0 <= ridx < ns_full):
        raise ValueError("radial_index is outside the VMEC state radial grid")
    if not (0 <= midx < int(base_Rcos.shape[1])):
        raise ValueError("mode_index is outside the VMEC state mode table")

    if surface_grid == "half_mesh":
        default_sidx = max(0, min(ridx - 1, ns_full - 2))
        surface_count = ns_full - 1
        error = "surface_index is outside the VMEC half-mesh Boozer surface grid"
    elif surface_grid == "field_line":
        default_sidx = max(1, min(ridx, ns_full - 2))
        surface_count = ns_full
        error = "surface_index is outside the VMEC metric radial grid"
    elif surface_grid == "metric":
        default_sidx = max(0, min(ridx - 1, ns_full - 1))
        surface_count = ns_full
        error = "surface_index is outside the VMEC metric radial grid"
    else:
        raise ValueError(f"unknown VMEC surface grid {surface_grid!r}")

    sidx = default_sidx if surface_index is None else int(surface_index)
    if not (0 <= sidx < surface_count):
        raise ValueError(error)
    return int(ridx), int(midx), int(sidx)


def _perturb_vmec_state(
    ctx: _VMECStateContext,
    x: jnp.ndarray,
    *,
    radial_index: int,
    mode_index: int,
) -> Any:
    """Return a VMEC state with two Fourier controls perturbed by ``x``."""

    return dc_replace(
        ctx.state,
        R_cos=ctx.base_Rcos.at[radial_index, mode_index].add(x[0]),
        Z_sin=ctx.base_Zsin.at[radial_index, mode_index].add(x[1]),
    )


def _length_two_params(params: jnp.ndarray | None, default: float) -> jnp.ndarray:
    """Normalize optional VMEC control perturbations to a length-two vector."""

    p = jnp.asarray([default, default] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")
    return p


def _array_parity_metrics(
    candidate: Any, reference: Any, *, floor: float = 1.0e-12
) -> dict[str, object]:
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    metrics: dict[str, object] = {
        "candidate_shape": [int(v) for v in cand.shape],
        "reference_shape": [int(v) for v in ref.shape],
        "shape_match": bool(cand.shape == ref.shape),
    }
    if cand.shape != ref.shape:
        return metrics
    diff = cand - ref
    ref_scale = max(float(np.nanmax(np.abs(ref))) if ref.size else 0.0, float(floor))
    local_scale = np.maximum(np.abs(ref), float(floor))
    metrics.update(
        {
            "max_abs": float(np.nanmax(np.abs(diff))) if diff.size else 0.0,
            "rms_abs": float(np.sqrt(np.nanmean(diff * diff))) if diff.size else 0.0,
            "max_rel_pointwise": float(np.nanmax(np.abs(diff) / local_scale))
            if diff.size
            else 0.0,
            "rms_rel_pointwise": float(np.sqrt(np.nanmean((diff / local_scale) ** 2)))
            if diff.size
            else 0.0,
            "reference_scale": ref_scale,
            "normalized_max_abs": float(np.nanmax(np.abs(diff)) / ref_scale)
            if diff.size
            else 0.0,
            "candidate_min": float(np.nanmin(cand)) if cand.size else 0.0,
            "candidate_max": float(np.nanmax(cand)) if cand.size else 0.0,
            "reference_min": float(np.nanmin(ref)) if ref.size else 0.0,
            "reference_max": float(np.nanmax(ref)) if ref.size else 0.0,
        }
    )
    return metrics


def _scalar_parity_metrics(
    candidate: Any, reference: Any, *, floor: float = 1.0e-12
) -> dict[str, float]:
    cand = float(np.asarray(candidate))
    ref = float(np.asarray(reference))
    diff = cand - ref
    scale = max(abs(ref), float(floor))
    return {
        "candidate": cand,
        "reference": ref,
        "abs": abs(diff),
        "rel": abs(diff) / scale,
    }


@dataclass(frozen=True)
class _VMECStateSensitivityReportRun:
    ctx: _VMECStateContext
    radial_index: int
    mode_index: int
    surface_index: int
    payload: dict[str, object]


def _unavailable_vmec_state_sensitivity_report(
    *,
    backend_info: Mapping[str, object],
    fd_step: float,
    case_name: str,
    reason: str,
) -> dict[str, object]:
    """Return a fail-closed optional-backend sensitivity report."""

    return {
        "available": False,
        "backend_info": dict(backend_info),
        "sensitivity": None,
        "fd_step": float(fd_step),
        "case_name": str(case_name),
        "reason": str(reason),
    }


def _failed_vmec_state_sensitivity_report(
    *,
    backend_info: Mapping[str, object],
    fd_step: float,
    case_name: str,
    exc: Exception,
) -> dict[str, object]:
    """Return a fail-closed report for exceptions raised inside backend probes."""

    return {
        "available": False,
        "backend_info": dict(backend_info),
        "sensitivity": None,
        "fd_step": float(fd_step),
        "case_name": str(case_name),
        "error": f"{type(exc).__name__}: {exc}",
    }


def _vmec_state_sensitivity_metadata(
    *,
    backend_info: Mapping[str, object],
    ctx: _VMECStateContext,
    case_name: str,
    params: jnp.ndarray,
    radial_index: int,
    mode_index: int,
    surface_index: int,
    fd_step: float,
) -> dict[str, object]:
    """Return shared metadata for VMEC-state sensitivity reports."""

    return {
        "available": True,
        "backend_info": dict(backend_info),
        "case_name": str(case_name),
        "input_path": str(ctx.input_path),
        "wout_path": str(ctx.wout_path),
        "param_names": ["delta_Rcos", "delta_Zsin"],
        "params": np.asarray(params).tolist(),
        "radial_index": int(radial_index),
        "mode_index": int(mode_index),
        "surface_index": int(surface_index),
        "state_shape": [int(ctx.base_Rcos.shape[0]), int(ctx.base_Rcos.shape[1])],
        "fd_step": float(fd_step),
    }


def _ad_fd_jacobian_diagnostics(
    observable_fn: Callable[[jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
    *,
    fd_step: float,
    observable_names: tuple[str, ...],
    relative_floor: float,
) -> dict[str, object]:
    """Return AD/finite-difference Jacobian diagnostics for sensitivity gates."""

    jac_ad = jax.jacfwd(observable_fn)(params)
    jac_fd = finite_difference_jacobian(observable_fn, params, step=float(fd_step))
    diff = jac_ad - jac_fd
    max_abs = jnp.max(jnp.abs(diff))
    max_rel = jnp.max(jnp.abs(diff) / (jnp.abs(jac_fd) + float(relative_floor)))
    return {
        "jacobian_ad": np.asarray(jac_ad).tolist(),
        "jacobian_fd": np.asarray(jac_fd).tolist(),
        "max_abs_ad_fd_error": float(np.asarray(max_abs)),
        "max_rel_ad_fd_error": float(np.asarray(max_rel)),
        "conditioning": _sensitivity_conditioning_metadata(
            jac_ad,
            jac_fd,
            params,
            fd_step=float(fd_step),
            observable_names=observable_names,
            param_names=("delta_Rcos", "delta_Zsin"),
            relative_floor=float(relative_floor),
        ),
    }


def _vmec_state_metric_grid_shape(runtime: Any) -> list[int]:
    """Return the vmex real-space evaluation grid shape ``[ns, ntheta, nzeta]``."""

    resolution = runtime.resolution
    return [int(resolution.ns), int(resolution.ntheta), int(resolution.nzeta)]


def _metric_tensor_observable_fn(
    *,
    ctx: _VMECStateContext,
    turbulence_mod: Any,
    radial_index: int,
    mode_index: int,
    surface_index: int,
    ntheta: int,
    rms_epsilon: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def metric_observables(x: jnp.ndarray) -> jnp.ndarray:
        traced_state = _perturb_vmec_state(
            ctx, x, radial_index=radial_index, mode_index=mode_index
        )
        mapping = turbulence_mod.gk_fieldline_geometry(
            traced_state,
            ctx.runtime,
            s_index=int(surface_index),
            ntheta=int(ntheta),
        )
        return jnp.asarray(
            [
                _rms_with_floor(jnp.asarray(mapping["jacobian"]), rms_epsilon),
                jnp.mean(jnp.asarray(mapping["gds2"])),
                jnp.mean(jnp.asarray(mapping["gds22"])),
                _rms_with_floor(jnp.asarray(mapping["gds21"]), rms_epsilon),
                jnp.mean(jnp.asarray(mapping["grho"])),
                jnp.mean(jnp.asarray(mapping["gradpar"])),
                jnp.asarray(mapping["s_hat"]),
            ]
        )

    return metric_observables


def _field_line_tensor_observable_fn(
    *,
    ctx: _VMECStateContext,
    turbulence_mod: Any,
    radial_index: int,
    mode_index: int,
    surface_index: int,
    alpha: float,
    ntheta: int,
    rms_epsilon: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def field_line_observables(x: jnp.ndarray) -> jnp.ndarray:
        traced_state = _perturb_vmec_state(
            ctx, x, radial_index=radial_index, mode_index=mode_index
        )
        mapping = turbulence_mod.gk_fieldline_geometry(
            traced_state,
            ctx.runtime,
            s_index=int(surface_index),
            alpha=float(alpha),
            ntheta=int(ntheta),
        )
        return jnp.asarray(
            [
                jnp.mean(jnp.asarray(mapping["bmag"])),
                jnp.asarray(mapping["epsilon"]),
                _rms_with_floor(jnp.asarray(mapping["bgrad"]), rms_epsilon),
                _rms_with_floor(jnp.asarray(mapping["cvdrift"]), rms_epsilon),
                _rms_with_floor(jnp.asarray(mapping["gbdrift"]), rms_epsilon),
                _rms_with_floor(jnp.asarray(mapping["gbdrift0"]), rms_epsilon),
                _rms_with_floor(jnp.asarray(mapping["jacobian"]), rms_epsilon),
            ]
        )

    return field_line_observables


def _tensor_sensitivity_payload(
    *,
    observable_fn: Callable[[jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
    fd_step: float,
    observable_names: tuple[str, ...],
    relative_floor: float,
) -> dict[str, object]:
    observables = observable_fn(params)
    return {
        "observable_names": list(observable_names),
        "observables": np.asarray(observables).tolist(),
        **_ad_fd_jacobian_diagnostics(
            observable_fn,
            params,
            fd_step=float(fd_step),
            observable_names=observable_names,
            relative_floor=float(relative_floor),
        ),
    }


def _vmec_state_sensitivity_report_from_run(
    *,
    backend_info: Mapping[str, object],
    run: _VMECStateSensitivityReportRun,
    case_name: str,
    params: jnp.ndarray,
    fd_step: float,
) -> dict[str, object]:
    """Pack shared VMEC-state metadata with a JSON-ready payload."""

    return {
        **_vmec_state_sensitivity_metadata(
            backend_info=backend_info,
            ctx=run.ctx,
            case_name=case_name,
            params=params,
            radial_index=run.radial_index,
            mode_index=run.mode_index,
            surface_index=run.surface_index,
            fd_step=fd_step,
        ),
        **run.payload,
    }


def _optional_vmec_state_sensitivity_report(
    *,
    params: jnp.ndarray | None,
    default_param: float,
    case_name: str,
    fd_step: float,
    backend_available: Callable[[Mapping[str, object]], bool],
    unavailable_reason: str,
    build_run: Callable[[jnp.ndarray], _VMECStateSensitivityReportRun],
) -> dict[str, object]:
    """Run an optional-backend VMEC-state gate with common fail-closed handling."""

    p = _length_two_params(params, default=default_param)
    info = discover_differentiable_geometry_backends()
    if not backend_available(info):
        return _unavailable_vmec_state_sensitivity_report(
            backend_info=info,
            fd_step=fd_step,
            case_name=case_name,
            reason=unavailable_reason,
        )

    try:
        run = build_run(p)
    except Exception as exc:
        return _failed_vmec_state_sensitivity_report(
            backend_info=info,
            fd_step=fd_step,
            case_name=case_name,
            exc=exc,
        )

    return _vmec_state_sensitivity_report_from_run(
        backend_info=info,
        run=run,
        case_name=case_name,
        params=p,
        fd_step=fd_step,
    )


def _load_vmec_geom_sensitivity_context(
    *,
    case_name: str,
    radial_index: int | None,
    mode_index: int,
    surface_index: int | None,
    surface_grid: str,
) -> tuple[_VMECStateContext, Any, int, int, int]:
    """Load the shared vmex geometry context for flux-tube AD/FD gates."""

    ctx = _load_vmec_state_context(str(case_name))
    turbulence_mod = _import_vmex_turbulence()
    ridx, midx, sidx = _resolve_vmec_state_indices(
        ctx.base_Rcos,
        radial_index=radial_index,
        mode_index=mode_index,
        surface_index=surface_index,
        surface_grid=surface_grid,
    )
    return ctx, turbulence_mod, ridx, midx, sidx


def _metric_tensor_report_payload(
    *,
    ctx: _VMECStateContext,
    turbulence_mod: Any,
    params: jnp.ndarray,
    radial_index: int,
    mode_index: int,
    surface_index: int,
    ntheta: int,
    fd_step: float,
    rms_epsilon: float,
) -> dict[str, object]:
    """Build the VMEC metric sensitivity payload from the PEST flux tube."""

    metric_observables = _metric_tensor_observable_fn(
        ctx=ctx,
        turbulence_mod=turbulence_mod,
        radial_index=radial_index,
        mode_index=mode_index,
        surface_index=surface_index,
        ntheta=int(ntheta),
        rms_epsilon=jnp.asarray(float(rms_epsilon), dtype=params.dtype),
    )
    tensor_payload = _tensor_sensitivity_payload(
        observable_fn=metric_observables,
        params=params,
        fd_step=float(fd_step),
        observable_names=_VMEC_METRIC_OBSERVABLE_NAMES,
        relative_floor=1.0e-12,
    )
    return {
        "source_model": "vmex:state->metric-tensors",
        **tensor_payload,
        "ntheta": int(ntheta),
        "metric_grid_shape": _vmec_state_metric_grid_shape(ctx.runtime),
        "rms_epsilon": float(rms_epsilon),
    }


def _run_vmec_metric_tensor_sensitivity(
    *,
    params: jnp.ndarray,
    case_name: str,
    radial_index: int | None,
    mode_index: int,
    surface_index: int | None,
    ntheta: int,
    fd_step: float,
    rms_epsilon: float,
) -> _VMECStateSensitivityReportRun:
    """Return a metadata-ready VMEC metric sensitivity run."""

    ctx, turbulence_mod, ridx, midx, sidx = _load_vmec_geom_sensitivity_context(
        case_name=case_name,
        radial_index=radial_index,
        mode_index=mode_index,
        surface_index=surface_index,
        surface_grid="metric",
    )
    return _VMECStateSensitivityReportRun(
        ctx=ctx,
        radial_index=ridx,
        mode_index=midx,
        surface_index=sidx,
        payload=_metric_tensor_report_payload(
            ctx=ctx,
            turbulence_mod=turbulence_mod,
            params=params,
            radial_index=ridx,
            mode_index=midx,
            surface_index=sidx,
            ntheta=int(ntheta),
            fd_step=float(fd_step),
            rms_epsilon=rms_epsilon,
        ),
    )


def _field_line_tensor_report_payload(
    *,
    ctx: _VMECStateContext,
    turbulence_mod: Any,
    params: jnp.ndarray,
    radial_index: int,
    mode_index: int,
    surface_index: int,
    alpha: float,
    ntheta: int,
    fd_step: float,
    b2_floor: float,
    rms_epsilon: float,
) -> dict[str, object]:
    """Build the VMEC field-line sensitivity payload from the PEST flux tube."""

    field_line_observables = _field_line_tensor_observable_fn(
        ctx=ctx,
        turbulence_mod=turbulence_mod,
        radial_index=radial_index,
        mode_index=mode_index,
        surface_index=surface_index,
        alpha=float(alpha),
        ntheta=int(ntheta),
        rms_epsilon=jnp.asarray(float(rms_epsilon), dtype=params.dtype),
    )
    tensor_payload = _tensor_sensitivity_payload(
        observable_fn=field_line_observables,
        params=params,
        fd_step=float(fd_step),
        observable_names=_VMEC_FIELD_LINE_OBSERVABLE_NAMES,
        relative_floor=1.0e-10,
    )
    mapping0 = turbulence_mod.gk_fieldline_geometry(
        ctx.state,
        ctx.runtime,
        s_index=int(surface_index),
        alpha=float(alpha),
        ntheta=int(ntheta),
    )
    vmex_meta = mapping0["vmex"]
    return {
        "source_model": "vmex:state->field-line-metric-and-b",
        "field_line_convention": str(vmex_meta["field_line_convention"]),
        **tensor_payload,
        "iota": float(np.asarray(vmex_meta["iota"])),
        "alpha": float(alpha),
        "ntheta": int(ntheta),
        "metric_grid_shape": _vmec_state_metric_grid_shape(ctx.runtime),
        # Retained for report-schema stability; the vmex spectral route needs
        # no |B|^2 floor.
        "b2_floor": float(b2_floor),
        "rms_epsilon": float(rms_epsilon),
    }


def _run_vmec_field_line_tensor_sensitivity(
    *,
    params: jnp.ndarray,
    case_name: str,
    radial_index: int | None,
    mode_index: int,
    surface_index: int | None,
    alpha: float,
    ntheta: int,
    fd_step: float,
    b2_floor: float,
    rms_epsilon: float,
) -> _VMECStateSensitivityReportRun:
    """Return a metadata-ready VMEC field-line sensitivity run."""

    ctx, turbulence_mod, ridx, midx, sidx = _load_vmec_geom_sensitivity_context(
        case_name=case_name,
        radial_index=radial_index,
        mode_index=mode_index,
        surface_index=surface_index,
        surface_grid="field_line",
    )
    return _VMECStateSensitivityReportRun(
        ctx=ctx,
        radial_index=ridx,
        mode_index=midx,
        surface_index=sidx,
        payload=_field_line_tensor_report_payload(
            ctx=ctx,
            turbulence_mod=turbulence_mod,
            params=params,
            radial_index=ridx,
            mode_index=midx,
            surface_index=sidx,
            alpha=alpha,
            ntheta=ntheta,
            fd_step=float(fd_step),
            b2_floor=b2_floor,
            rms_epsilon=rms_epsilon,
        ),
    )


def vmex_metric_tensor_sensitivity_report(  # pragma: no cover
    *,
    params: jnp.ndarray | None = None,
    case_name: str = "circular_tokamak",
    radial_index: int | None = None,
    mode_index: int = 1,
    surface_index: int | None = None,
    ntheta: int = 32,
    fd_step: float = 1.0e-5,
    rms_epsilon: float = 1.0e-24,
) -> dict[str, object]:
    """AD/FD-check flux-tube metric coefficients from a solved ``vmex`` state.

    The Boozer bridge validates the straight-field-line ``|B|`` spectrum, but
    GKX's production geometry contract also needs the perpendicular
    metric and Jacobian profiles. This gate perturbs two vmex Fourier
    coefficients, evaluates the PEST field-line metric arrays of
    ``vmex.core.turbulence.gk_fieldline_geometry`` (``gds2``/``gds21``/
    ``gds22``/``grho``/``jacobian``/``gradpar`` plus the magnetic shear), and
    checks metric-observable derivatives against central finite differences.
    """

    return _optional_vmec_state_sensitivity_report(
        params=params,
        default_param=1.0e-3,
        case_name=str(case_name),
        fd_step=float(fd_step),
        backend_available=lambda info: bool(info.get("vmex_available", False)),
        unavailable_reason="vmex is not available",
        build_run=lambda p: _run_vmec_metric_tensor_sensitivity(
            params=p,
            case_name=str(case_name),
            radial_index=radial_index,
            mode_index=mode_index,
            surface_index=surface_index,
            ntheta=int(ntheta),
            fd_step=float(fd_step),
            rms_epsilon=rms_epsilon,
        ),
    )


def vmex_field_line_tensor_sensitivity_report(  # pragma: no cover
    *,
    params: jnp.ndarray | None = None,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    surface_index: int | None = None,
    alpha: float = 0.0,
    ntheta: int = 32,
    fd_step: float = 1.0e-6,
    b2_floor: float = 1.0e-24,
    rms_epsilon: float = 1.0e-24,
) -> dict[str, object]:
    """AD/FD-check field-line ``|B|`` and drift arrays from a ``vmex`` state.

    This optional-backend gate perturbs two Fourier coefficients of a real
    solved stellarator ``vmex`` example state, samples the PEST field-line
    ``|B|``, its parallel derivative, and the projected grad-B/curvature drift
    arrays of ``vmex.core.turbulence.gk_fieldline_geometry``, and checks those
    observable derivatives against central finite differences.

    The gate proves differentiability from the vmex spectral state through the
    real field-line magnetic geometry. The later production gate must still
    parity-check the exact GKX ``gds*``/drift contract against the
    imported VMEC/EIK path.
    """

    return _optional_vmec_state_sensitivity_report(
        params=params,
        default_param=1.0e-4,
        case_name=str(case_name),
        fd_step=float(fd_step),
        backend_available=lambda info: bool(info.get("vmex_available", False)),
        unavailable_reason="vmex is not available",
        build_run=lambda p: _run_vmec_field_line_tensor_sensitivity(
            params=p,
            case_name=str(case_name),
            radial_index=radial_index,
            mode_index=mode_index,
            surface_index=surface_index,
            alpha=alpha,
            ntheta=ntheta,
            fd_step=float(fd_step),
            b2_floor=b2_floor,
            rms_epsilon=rms_epsilon,
        ),
    )


__all__ = [
    "vmex_field_line_tensor_sensitivity_report",
    "vmex_metric_tensor_sensitivity_report",
]
