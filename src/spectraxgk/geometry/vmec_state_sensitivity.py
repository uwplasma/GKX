"""VMEC-state differentiable sensitivity reports."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.autodiff_checks import (
    _sensitivity_conditioning_metadata,
    finite_difference_jacobian,
)
from spectraxgk.geometry.backend_discovery import (
    discover_differentiable_geometry_backends,
)
from spectraxgk.geometry.booz_xform_bridge import (
    booz_xform_flux_tube_mapping_from_inputs,
)
from spectraxgk.geometry.flux_tube_contract import (
    _VMEC_FIELD_LINE_OBSERVABLE_NAMES,
    _VMEC_METRIC_OBSERVABLE_NAMES,
)
from spectraxgk.geometry.vmec_field_line_sampling import (
    _rms_with_floor,
    _vmec_field_line_sampling_coordinates,
)
from spectraxgk.geometry.numerics import _periodic_bilinear_sample_2d
from spectraxgk.geometry.sensitivity import geometry_sensitivity_report
from spectraxgk.geometry.vmec_state_controls import (
    _VMECStateContext,
    _length_two_params,
    _load_vmec_state_context,
    _perturb_vmec_state,
    _resolve_vmec_state_indices,
)


@dataclass(frozen=True)
class _BoozerFluxTubeSensitivityRun:
    ctx: _VMECStateContext
    radial_index: int
    mode_index: int
    surface_index: int
    sensitivity: dict[str, object]
    booz_meta: Mapping[str, Any]


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


def _metric_tensor_observable_fn(
    *,
    ctx: _VMECStateContext,
    geom_mod: Any,
    radial_index: int,
    mode_index: int,
    surface_index: int,
    rms_epsilon: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def metric_observables(x: jnp.ndarray) -> jnp.ndarray:
        traced_state = _perturb_vmec_state(
            ctx, x, radial_index=radial_index, mode_index=mode_index
        )
        geom = geom_mod.eval_geom(traced_state, ctx.static)
        sqrtg, g_ss, g_st, g_sp, g_tt, g_tp, g_pp = (
            jnp.asarray(getattr(geom, name))[surface_index]
            for name in ("sqrtg", "g_ss", "g_st", "g_sp", "g_tt", "g_tp", "g_pp")
        )
        return jnp.asarray(
            [
                _rms_with_floor(sqrtg, rms_epsilon),
                jnp.mean(g_ss),
                jnp.mean(g_tt),
                jnp.mean(g_pp),
                _rms_with_floor(g_st, rms_epsilon),
                _rms_with_floor(g_sp, rms_epsilon),
                _rms_with_floor(g_tp, rms_epsilon),
            ]
        )

    return metric_observables


def _field_line_tensor_observable_fn(
    *,
    ctx: _VMECStateContext,
    geom_mod: Any,
    bcovar_mod: Any,
    field_mod: Any,
    radial_index: int,
    mode_index: int,
    surface_index: int,
    theta_vmec: jnp.ndarray,
    zeta_line: jnp.ndarray,
    b2_floor: jnp.ndarray,
    rms_epsilon: jnp.ndarray,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    def field_line_observables(x: jnp.ndarray) -> jnp.ndarray:
        traced_state = _perturb_vmec_state(
            ctx, x, radial_index=radial_index, mode_index=mode_index
        )
        geom = geom_mod.eval_geom(traced_state, ctx.static)
        bcovar = bcovar_mod.vmec_bcovar_half_mesh_from_wout(
            state=traced_state,
            static=ctx.static,
            wout=ctx.wout,
            pres=getattr(ctx.wout, "pres", None),
        )
        b2 = field_mod.b2_from_bsup(geom, bcovar.bsupu, bcovar.bsupv)
        def sample(values: jnp.ndarray) -> jnp.ndarray:
            return _periodic_bilinear_sample_2d(
                values[surface_index], theta_vmec, zeta_line
            )
        bmag = jnp.sqrt(jnp.maximum(sample(b2), b2_floor))
        sqrtg, g_tt, g_tp, g_pp, g_ss = (
            sample(getattr(geom, name))
            for name in ("sqrtg", "g_tt", "g_tp", "g_pp", "g_ss")
        )
        mean_b = jnp.mean(bmag)
        ripple = jnp.std(bmag) / jnp.maximum(
            jnp.abs(mean_b), jnp.asarray(1.0e-30, dtype=bmag.dtype)
        )
        return jnp.asarray(
            [
                mean_b,
                ripple,
                _rms_with_floor(sqrtg, rms_epsilon),
                jnp.mean(g_tt),
                jnp.mean(g_pp),
                _rms_with_floor(g_tp, rms_epsilon),
                jnp.mean(g_ss),
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


def _load_vmec_geom_sensitivity_context(
    *,
    case_name: str,
    radial_index: int | None,
    mode_index: int,
    surface_index: int | None,
    surface_grid: str,
) -> tuple[_VMECStateContext, Any, int, int, int]:
    """Load the shared VMEC geometry context for tensor AD/FD gates."""

    ctx = _load_vmec_state_context(str(case_name))
    geom_mod = importlib.import_module("vmec_jax.geom")
    ridx, midx, sidx = _resolve_vmec_state_indices(
        ctx.base_Rcos,
        radial_index=radial_index,
        mode_index=mode_index,
        surface_index=surface_index,
        surface_grid=surface_grid,
    )
    return ctx, geom_mod, ridx, midx, sidx


def _load_vmec_boozer_sensitivity_context(
    *,
    case_name: str,
    radial_index: int | None,
    mode_index: int,
    surface_index: int | None,
) -> tuple[_VMECStateContext, Any, int, int, int]:
    """Load VMEC state data and Boozer input conversion hooks."""

    ctx = _load_vmec_state_context(str(case_name))
    booz_input_mod = importlib.import_module("vmec_jax.booz_input")
    ridx, midx, sidx = _resolve_vmec_state_indices(
        ctx.base_Rcos,
        radial_index=radial_index,
        mode_index=mode_index,
        surface_index=surface_index,
        surface_grid="half_mesh",
    )
    return ctx, booz_input_mod, ridx, midx, sidx


def _vmec_to_boozer_mapping_fn(
    *,
    ctx: _VMECStateContext,
    booz_input_mod: Any,
    radial_index: int,
    mode_index: int,
    surface_index: int,
    mboz: int,
    nboz: int,
    ntheta: int,
) -> Callable[[jnp.ndarray], dict[str, Any]]:
    """Return the differentiable VMEC-state to Boozer field-line mapping."""

    def mapping_fn(x: jnp.ndarray) -> dict[str, Any]:
        traced_state = _perturb_vmec_state(
            ctx, x, radial_index=radial_index, mode_index=mode_index
        )
        inputs = booz_input_mod.booz_xform_inputs_from_state(
            state=traced_state,
            static=ctx.static,
            indata=ctx.indata,
            signgs=ctx.wout.signgs,
        )
        return booz_xform_flux_tube_mapping_from_inputs(
            inputs,
            mboz=int(mboz),
            nboz=int(nboz),
            ntheta=int(ntheta),
            surface_index=int(surface_index),
            magnetic_shear=0.35,
            jit=False,
        )

    return mapping_fn


def _boozer_flux_tube_report_payload(
    *,
    sensitivity: dict[str, object],
    booz_meta: Mapping[str, Any],
    mboz: int,
    nboz: int,
    ntheta: int,
) -> dict[str, object]:
    """Pack JSON-ready Boozer flux-tube payload fields."""

    return {
        "sensitivity": sensitivity,
        "mboz": int(mboz),
        "nboz": int(nboz),
        "ntheta": int(ntheta),
        "bmnc_b": np.asarray(booz_meta["bmnc_b"]).tolist(),
        "ixm_b": np.asarray(booz_meta["ixm_b"]).tolist(),
        "ixn_b": np.asarray(booz_meta["ixn_b"]).tolist(),
        "iota_b": float(np.asarray(booz_meta["iota_b"])),
    }


def _run_vmec_boozer_flux_tube_sensitivity(
    *,
    params: jnp.ndarray,
    case_name: str,
    radial_index: int | None,
    mode_index: int,
    surface_index: int | None,
    fd_step: float,
    mboz: int,
    nboz: int,
    ntheta: int,
) -> _BoozerFluxTubeSensitivityRun:
    ctx, booz_input_mod, ridx, midx, sidx = _load_vmec_boozer_sensitivity_context(
        case_name=str(case_name),
        radial_index=radial_index,
        mode_index=mode_index,
        surface_index=surface_index,
    )
    mapping_fn = _vmec_to_boozer_mapping_fn(
        ctx=ctx,
        booz_input_mod=booz_input_mod,
        radial_index=ridx,
        mode_index=midx,
        surface_index=sidx,
        mboz=mboz,
        nboz=nboz,
        ntheta=ntheta,
    )
    sensitivity = geometry_sensitivity_report(
        mapping_fn,
        params,
        fd_step=float(fd_step),
        source_model="vmec_jax:state->booz_xform_jax:field-line-bmag",
    )
    mapping = mapping_fn(params)
    return _BoozerFluxTubeSensitivityRun(
        ctx=ctx,
        radial_index=ridx,
        mode_index=midx,
        surface_index=sidx,
        sensitivity=sensitivity,
        booz_meta=mapping["booz_xform"],
    )


def _field_line_tensor_report_payload(
    *,
    ctx: _VMECStateContext,
    geom_mod: Any,
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
    """Build the VMEC field-line tensor sensitivity payload."""

    bcovar_mod = importlib.import_module("vmec_jax.vmec_bcovar")
    field_mod = importlib.import_module("vmec_jax.field")
    iota_line, _iota_safe, _theta_line, theta_vmec, zeta_line = (
        _vmec_field_line_sampling_coordinates(
            ctx.wout,
            surface_index=surface_index,
            alpha=alpha,
            ntheta=ntheta,
            dtype=params.dtype,
        )
    )
    field_line_observables = _field_line_tensor_observable_fn(
        ctx=ctx,
        geom_mod=geom_mod,
        bcovar_mod=bcovar_mod,
        field_mod=field_mod,
        radial_index=radial_index,
        mode_index=mode_index,
        surface_index=surface_index,
        theta_vmec=theta_vmec,
        zeta_line=zeta_line,
        b2_floor=jnp.asarray(float(b2_floor), dtype=params.dtype),
        rms_epsilon=jnp.asarray(float(rms_epsilon), dtype=params.dtype),
    )
    tensor_payload = _tensor_sensitivity_payload(
        observable_fn=field_line_observables,
        params=params,
        fd_step=float(fd_step),
        observable_names=_VMEC_FIELD_LINE_OBSERVABLE_NAMES,
        relative_floor=1.0e-10,
    )
    geom0 = geom_mod.eval_geom(ctx.state, ctx.static)
    return {
        "source_model": "vmec_jax:state->field-line-metric-and-b",
        "field_line_convention": "VMEC theta, zeta=(theta-alpha)/iota with periodic bilinear sampling",
        **tensor_payload,
        "iota": float(np.asarray(iota_line)),
        "alpha": float(alpha),
        "ntheta": int(ntheta),
        "metric_grid_shape": [int(v) for v in np.asarray(geom0.sqrtg).shape],
        "b2_floor": float(b2_floor),
        "rms_epsilon": float(rms_epsilon),
    }


def vmec_jax_boozer_flux_tube_sensitivity_report(  # pragma: no cover
    *,
    params: jnp.ndarray | None = None,
    case_name: str = "circular_tokamak",
    radial_index: int | None = None,
    mode_index: int = 1,
    surface_index: int | None = None,
    fd_step: float = 1.0e-5,
    mboz: int = 2,
    nboz: int = 0,
    ntheta: int = 32,
) -> dict[str, object]:
    """AD/FD-check ``vmec_jax`` state coefficients through the Boozer bridge.

    This is the first end-to-end optional-backend gate that starts from a real
    ``vmec_jax`` ``VMECState`` instead of a hand-built Boozer input bundle. It
    loads a small bundled VMEC example, perturbs two VMEC Fourier coefficients
    ``[Rcos(radial_index, mode_index), Zsin(radial_index, mode_index)]``,
    converts the perturbed state to ``booz_xform_jax`` inputs, samples the
    resulting Boozer ``|B|`` spectrum on a field line, and checks
    SPECTRAX-GK geometry-observable derivatives against central finite
    differences.

    The current metric/drift closure is still intentionally smooth and local to
    SPECTRAX-GK. Full production promotion requires replacing it with sampled
    VMEC/Boozer metric tensors and parity-checking those arrays against the
    imported VMEC/EIK path.
    """

    p = _length_two_params(params, default=1.0e-3)

    info = discover_differentiable_geometry_backends()
    if not (
        info.get("vmec_jax_available", False)
        and info.get("booz_xform_jax_api_available", False)
    ):
        return _unavailable_vmec_state_sensitivity_report(
            backend_info=info,
            fd_step=fd_step,
            case_name=case_name,
            reason="vmec_jax or booz_xform_jax functional API is not available",
        )

    try:
        run = _run_vmec_boozer_flux_tube_sensitivity(
            params=p,
            case_name=str(case_name),
            radial_index=radial_index,
            mode_index=mode_index,
            surface_index=surface_index,
            fd_step=float(fd_step),
            mboz=mboz,
            nboz=nboz,
            ntheta=ntheta,
        )
    except Exception as exc:
        return _failed_vmec_state_sensitivity_report(
            backend_info=info,
            fd_step=fd_step,
            case_name=case_name,
            exc=exc,
        )

    return {
        **_vmec_state_sensitivity_metadata(
            backend_info=info,
            ctx=run.ctx,
            case_name=case_name,
            params=p,
            radial_index=run.radial_index,
            mode_index=run.mode_index,
            surface_index=run.surface_index,
            fd_step=fd_step,
        ),
        **_boozer_flux_tube_report_payload(
            sensitivity=run.sensitivity,
            booz_meta=run.booz_meta,
            mboz=mboz,
            nboz=nboz,
            ntheta=ntheta,
        ),
    }


def vmec_jax_metric_tensor_sensitivity_report(  # pragma: no cover
    *,
    params: jnp.ndarray | None = None,
    case_name: str = "circular_tokamak",
    radial_index: int | None = None,
    mode_index: int = 1,
    surface_index: int | None = None,
    fd_step: float = 1.0e-5,
    rms_epsilon: float = 1.0e-24,
) -> dict[str, object]:
    """AD/FD-check real ``vmec_jax`` metric tensors from a ``VMECState``.

    The Boozer bridge validates the straight-field-line ``|B|`` spectrum, but
    SPECTRAX-GK's production geometry contract also needs sampled metric and
    drift tensors. This gate stays upstream of any reduced closure: it loads a
    real ``vmec_jax`` example state, perturbs two VMEC Fourier coefficients,
    evaluates ``vmec_jax.geom.eval_geom``, and checks metric-tensor observable
    derivatives against central finite differences.

    This is a prerequisite for replacing the smooth metric/drift closure in
    :func:`booz_xform_flux_tube_mapping_from_inputs`; it is not by itself the
    final Boozer-field-line metric parity gate.
    """

    p = _length_two_params(params, default=1.0e-3)

    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_available", False):
        return _unavailable_vmec_state_sensitivity_report(
            backend_info=info,
            fd_step=fd_step,
            case_name=case_name,
            reason="vmec_jax is not available",
        )

    try:
        ctx, geom_mod, ridx, midx, sidx = _load_vmec_geom_sensitivity_context(
            case_name=str(case_name),
            radial_index=radial_index,
            mode_index=mode_index,
            surface_index=surface_index,
            surface_grid="metric",
        )

        metric_observables = _metric_tensor_observable_fn(
            ctx=ctx,
            geom_mod=geom_mod,
            radial_index=ridx,
            mode_index=midx,
            surface_index=sidx,
            rms_epsilon=jnp.asarray(float(rms_epsilon), dtype=p.dtype),
        )
        tensor_payload = _tensor_sensitivity_payload(
            observable_fn=metric_observables,
            params=p,
            fd_step=float(fd_step),
            observable_names=_VMEC_METRIC_OBSERVABLE_NAMES,
            relative_floor=1.0e-12,
        )
        geom0 = geom_mod.eval_geom(ctx.state, ctx.static)
    except Exception as exc:
        return _failed_vmec_state_sensitivity_report(
            backend_info=info,
            fd_step=fd_step,
            case_name=case_name,
            exc=exc,
        )

    return {
        **_vmec_state_sensitivity_metadata(
            backend_info=info,
            ctx=ctx,
            case_name=case_name,
            params=p,
            radial_index=ridx,
            mode_index=midx,
            surface_index=sidx,
            fd_step=fd_step,
        ),
        "source_model": "vmec_jax:state->metric-tensors",
        **tensor_payload,
        "metric_grid_shape": [int(v) for v in np.asarray(geom0.sqrtg).shape],
        "rms_epsilon": float(rms_epsilon),
    }


def vmec_jax_field_line_tensor_sensitivity_report(  # pragma: no cover
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
    """AD/FD-check VMEC field-line metric and ``|B|`` tensors from ``vmec_jax``.

    This optional-backend gate is deliberately upstream of the production
    SPECTRAX-GK metric/drift closure. It loads a real stellarator
    ``vmec_jax`` example state, perturbs two VMEC Fourier coefficients,
    evaluates ``vmec_jax.geom.eval_geom`` and ``vmec_jax.vmec_bcovar``, samples
    raw metric/``|B|`` tensors on a fixed VMEC field-line convention, and checks
    those observable derivatives against central finite differences.

    The gate proves differentiability from ``VMECState`` through real VMEC
    metric and magnetic-field tensors. The later production gate must still
    convert those tensors into the exact SPECTRAX-GK ``gds*``/drift contract and
    compare against the imported VMEC/EIK path.
    """

    p = _length_two_params(params, default=1.0e-4)

    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_available", False):
        return _unavailable_vmec_state_sensitivity_report(
            backend_info=info,
            fd_step=fd_step,
            case_name=case_name,
            reason="vmec_jax is not available",
        )

    try:
        ctx, geom_mod, ridx, midx, sidx = _load_vmec_geom_sensitivity_context(
            case_name=str(case_name),
            radial_index=radial_index,
            mode_index=mode_index,
            surface_index=surface_index,
            surface_grid="field_line",
        )
        payload = _field_line_tensor_report_payload(
            ctx=ctx,
            geom_mod=geom_mod,
            params=p,
            radial_index=ridx,
            mode_index=midx,
            surface_index=sidx,
            alpha=alpha,
            ntheta=ntheta,
            fd_step=float(fd_step),
            b2_floor=b2_floor,
            rms_epsilon=rms_epsilon,
        )
    except Exception as exc:
        return _failed_vmec_state_sensitivity_report(
            backend_info=info,
            fd_step=fd_step,
            case_name=case_name,
            exc=exc,
        )

    return {
        **_vmec_state_sensitivity_metadata(
            backend_info=info,
            ctx=ctx,
            case_name=case_name,
            params=p,
            radial_index=ridx,
            mode_index=midx,
            surface_index=sidx,
            fd_step=fd_step,
        ),
        **payload,
    }


__all__ = [
    "vmec_jax_boozer_flux_tube_sensitivity_report",
    "vmec_jax_field_line_tensor_sensitivity_report",
    "vmec_jax_metric_tensor_sensitivity_report",
]
