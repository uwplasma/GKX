"""VMEC flux-tube sensitivity and parity reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.flux_tube_contract import flux_tube_geometry_from_mapping
from spectraxgk.geometry.backend_discovery import (
    discover_differentiable_geometry_backends,
)
from spectraxgk.geometry.numerics import (
    _array_parity_metrics,
    _scalar_parity_metrics,
)
from spectraxgk.geometry.sensitivity import geometry_sensitivity_report
from spectraxgk.geometry.vmec_boozer_core import (
    vmec_jax_boozer_equal_arc_core_profiles_from_state,
)
from spectraxgk.geometry.vmec_state_controls import (
    _length_two_params,
    _load_vmec_state_context,
    _perturb_vmec_state,
    _resolve_vmec_state_indices,
)
from spectraxgk.geometry.vmec_tensor_mapping import (
    vmec_jax_flux_tube_mapping_from_state,
)

_VMEC_BOOZER_PARITY_MIN_MODE_COUNT = 21
_EQUAL_ARC_CORE_FIELDS = (
    ("theta", "theta", "theta"),
    ("bmag", "bmag", "bmag_profile"),
    ("bgrad", "bgrad", "bgrad_profile"),
    ("jacobian", "jacobian", "jacobian_profile"),
)
_EQUAL_ARC_METRIC_FIELDS = (
    ("gds2", "gds2", "gds2_profile"),
    ("gds21", "gds21", "gds21_profile"),
    ("gds22", "gds22", "gds22_profile"),
    ("grho", "grho", "grho_profile"),
)
_EQUAL_ARC_DRIFT_FIELDS = (
    ("cvdrift", "cvdrift", "cv_profile"),
    ("gbdrift", "gbdrift", "gb_profile"),
    ("cvdrift0", "cvdrift0", "cv0_profile"),
    ("gbdrift0", "gbdrift0", "gb0_profile"),
)
_FLUX_TUBE_ARRAY_FIELDS = (
    ("theta", "theta", "theta"),
    ("bmag", "bmag_profile", "bmag_profile"),
    ("bgrad", "bgrad_profile", "bgrad_profile"),
    ("gds2", "gds2_profile", "gds2_profile"),
    ("gds21", "gds21_profile", "gds21_profile"),
    ("gds22", "gds22_profile", "gds22_profile"),
    ("cvdrift", "cv_profile", "cv_profile"),
    ("gbdrift", "gb_profile", "gb_profile"),
    ("cvdrift0", "cv0_profile", "cv0_profile"),
    ("gbdrift0", "gb0_profile", "gb0_profile"),
    ("jacobian", "jacobian_profile", "jacobian_profile"),
    ("grho", "grho_profile", "grho_profile"),
)
_PRODUCTION_CORE_ARRAY_NAMES = (
    "bmag",
    "gds2",
    "gds21",
    "gds22",
    "cvdrift",
    "gbdrift",
    "jacobian",
    "grho",
)
_VMEC_EIK_DEFAULT_REQUEST: dict[str, object] = {
    "y0": 10.0,
    "x0": None,
    "jtwist": None,
    "beta": 0.0,
    "npol": 1.0,
    "npol_min": None,
    "isaxisym": False,
    "which_crossing": None,
    "betaprim": None,
    "z": (1.0, -1.0),
    "mass": (1.0, 2.7e-4),
    "dens": (1.0, 1.0),
    "temp": (1.0, 1.0),
    "tprim": (3.0, 0.0),
    "fprim": (1.0, 0.0),
    "vnewk": (0.0, 0.0),
    "species_type": ("ion", "electron"),
}


def _normalized_max_abs(metrics: dict[str, object]) -> float:
    raw_value = metrics.get("normalized_max_abs")
    return float(raw_value) if isinstance(raw_value, int | float | np.floating) else np.inf


def _array_metrics_from_pairs(pairs: dict[str, tuple[Any, Any]]) -> dict[str, dict[str, object]]:
    return {
        name: _array_parity_metrics(candidate, reference)
        for name, (candidate, reference) in pairs.items()
    }


def _array_metrics_from_key_attrs(
    candidate: dict[str, Any],
    reference: Any,
    specs: tuple[tuple[str, str, str], ...],
) -> dict[str, dict[str, object]]:
    return _array_metrics_from_pairs(
        {name: (candidate[candidate_key], getattr(reference, reference_attr)) for name, candidate_key, reference_attr in specs}
    )


def _array_pairs_from_attrs(
    candidate: Any,
    reference: Any,
    specs: tuple[tuple[str, str, str], ...],
) -> dict[str, tuple[Any, Any]]:
    return {
        name: (getattr(candidate, candidate_attr), getattr(reference, reference_attr))
        for name, candidate_attr, reference_attr in specs
    }


def _worst_array_error(metrics_by_name: dict[str, dict[str, object]], names: tuple[str, ...] | None = None) -> float:
    selected_names = tuple(metrics_by_name) if names is None else names
    values = [
        _normalized_max_abs(metrics_by_name[name])
        for name in selected_names
        if bool(metrics_by_name[name].get("shape_match", False))
    ]
    return max(values) if values else np.inf


def _empty_equal_arc_parity(error: str | None) -> dict[str, object]:
    return {
        "equal_arc_core_array_metrics": {},
        "equal_arc_metric_array_metrics": {},
        "equal_arc_drift_array_metrics": {},
        "equal_arc_core_scalar_metrics": {},
        "equal_arc_core_worst_normalized_max_abs": np.inf,
        "equal_arc_core_worst_scalar_rel": np.inf,
        "equal_arc_derivative_worst_normalized_max_abs": np.inf,
        "equal_arc_metric_worst_normalized_max_abs": np.inf,
        "equal_arc_drift_worst_normalized_max_abs": np.inf,
        "equal_arc_core_passed": False,
        "equal_arc_derivative_passed": False,
        "equal_arc_metric_passed": False,
        "equal_arc_drift_passed": False,
        "equal_arc_core_error": error,
    }


def _equal_arc_core_profiles(
    *,
    ctx: Any,
    surface_index: int,
    torflux: float,
    alpha: float,
    ntheta: int,
    mboz: int,
    nboz: int,
) -> dict[str, Any]:
    return vmec_jax_boozer_equal_arc_core_profiles_from_state(
        ctx.state,
        ctx.static,
        ctx.indata,
        ctx.wout,
        surface_index=surface_index,
        torflux=torflux,
        alpha=float(alpha),
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        jit=False,
    )


def _equal_arc_array_metric_groups(
    equal_arc_core: dict[str, Any],
    imported: Any,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    return (
        _array_metrics_from_key_attrs(equal_arc_core, imported, _EQUAL_ARC_CORE_FIELDS),
        _array_metrics_from_key_attrs(equal_arc_core, imported, _EQUAL_ARC_METRIC_FIELDS),
        _array_metrics_from_key_attrs(equal_arc_core, imported, _EQUAL_ARC_DRIFT_FIELDS),
    )


def _equal_arc_scalar_metrics(equal_arc_core: dict[str, Any], imported: Any) -> dict[str, dict[str, Any]]:
    return {
        "gradpar": _scalar_parity_metrics(jnp.asarray(equal_arc_core["gradpar"])[0], imported.gradpar_value),
        "q": _scalar_parity_metrics(equal_arc_core["q"], imported.q),
        "s_hat": _scalar_parity_metrics(equal_arc_core["s_hat"], imported.s_hat),
    }


def _pack_equal_arc_parity(
    *,
    core_metrics: dict[str, dict[str, object]],
    metric_metrics: dict[str, dict[str, object]],
    drift_metrics: dict[str, dict[str, object]],
    scalar_metrics: dict[str, dict[str, Any]],
    core_tolerance: float,
    derivative_tolerance: float,
    metric_tolerance: float,
    drift_tolerance: float,
) -> dict[str, object]:
    core_worst = _worst_array_error(core_metrics, ("theta", "bmag", "jacobian"))
    core_worst_scalar = max(float(values["rel"]) for values in scalar_metrics.values())
    derivative_worst = _normalized_max_abs(core_metrics["bgrad"])
    metric_worst = _worst_array_error(metric_metrics)
    drift_worst = _worst_array_error(drift_metrics)
    return {
        "equal_arc_core_array_metrics": core_metrics,
        "equal_arc_metric_array_metrics": metric_metrics,
        "equal_arc_drift_array_metrics": drift_metrics,
        "equal_arc_core_scalar_metrics": scalar_metrics,
        "equal_arc_core_worst_normalized_max_abs": core_worst,
        "equal_arc_core_worst_scalar_rel": core_worst_scalar,
        "equal_arc_derivative_worst_normalized_max_abs": derivative_worst,
        "equal_arc_metric_worst_normalized_max_abs": metric_worst,
        "equal_arc_drift_worst_normalized_max_abs": drift_worst,
        "equal_arc_core_passed": bool(
            core_worst <= float(core_tolerance)
            and core_worst_scalar <= float(core_tolerance)
        ),
        "equal_arc_derivative_passed": bool(derivative_worst <= float(derivative_tolerance)),
        "equal_arc_metric_passed": bool(metric_worst <= float(metric_tolerance)),
        "equal_arc_drift_passed": bool(drift_worst <= float(drift_tolerance)),
        "equal_arc_core_error": None,
    }


def _equal_arc_parity_report(
    *,
    info: dict[str, object],
    ctx: Any,
    imported: Any,
    surface_index: int,
    torflux: float,
    alpha: float,
    ntheta: int,
    mboz: int,
    nboz: int,
    core_tolerance: float,
    derivative_tolerance: float,
    metric_tolerance: float,
    drift_tolerance: float,
) -> dict[str, object]:
    if not bool(info.get("booz_xform_jax_api_available", False)):
        return _empty_equal_arc_parity("booz_xform_jax functional API is not available")
    try:
        equal_arc_core = _equal_arc_core_profiles(
            ctx=ctx,
            surface_index=surface_index,
            torflux=torflux,
            alpha=alpha,
            ntheta=ntheta,
            mboz=mboz,
            nboz=nboz,
        )
    except Exception as exc:  # pragma: no cover - optional-backend diagnostic detail
        return _empty_equal_arc_parity(f"{type(exc).__name__}: {exc}")

    core_metrics, metric_metrics, drift_metrics = _equal_arc_array_metric_groups(
        equal_arc_core,
        imported,
    )
    return _pack_equal_arc_parity(
        core_metrics=core_metrics,
        metric_metrics=metric_metrics,
        drift_metrics=drift_metrics,
        scalar_metrics=_equal_arc_scalar_metrics(equal_arc_core, imported),
        core_tolerance=core_tolerance,
        derivative_tolerance=derivative_tolerance,
        metric_tolerance=metric_tolerance,
        drift_tolerance=drift_tolerance,
    )


def _vmec_sensitivity_unavailable_report(
    *,
    info: dict[str, object],
    case_name: str,
    fd_step: float,
    reason: str,
) -> dict[str, object]:
    return {
        "available": False,
        "backend_info": info,
        "sensitivity": None,
        "fd_step": float(fd_step),
        "case_name": str(case_name),
        "reason": reason,
    }


def _vmec_flux_tube_mapping_fn(
    *,
    ctx: Any,
    radial_index: int,
    mode_index: int,
    surface_index: int | None,
    alpha: float,
    ntheta: int,
) -> Any:
    def mapping_fn(x: jnp.ndarray) -> dict[str, Any]:
        traced_state = _perturb_vmec_state(
            ctx,
            x,
            radial_index=radial_index,
            mode_index=mode_index,
        )
        return vmec_jax_flux_tube_mapping_from_state(
            traced_state,
            ctx.static,
            ctx.wout,
            surface_index=surface_index,
            alpha=float(alpha),
            ntheta=int(ntheta),
        )

    return mapping_fn


def _pack_vmec_sensitivity_report(
    *,
    ctx: Any,
    sensitivity: dict[str, object],
    mapping: dict[str, Any],
    params: jnp.ndarray,
    case_name: str,
    radial_index: int,
    mode_index: int,
    alpha: float,
    ntheta: int,
    fd_step: float,
    info: dict[str, object],
) -> dict[str, object]:
    vmec_meta = mapping["vmec_jax"]
    return {
        "available": True,
        "backend_info": info,
        "case_name": str(case_name),
        "input_path": str(ctx.input_path),
        "wout_path": str(ctx.wout_path),
        "source_model": "vmec_jax:state->tensor-flux-tube",
        "param_names": ["delta_Rcos", "delta_Zsin"],
        "params": np.asarray(params).tolist(),
        "radial_index": int(radial_index),
        "mode_index": int(mode_index),
        "surface_index": int(vmec_meta["surface_index"]),
        "iota": float(np.asarray(vmec_meta["iota"])),
        "alpha": float(alpha),
        "ntheta": int(ntheta),
        "state_shape": [int(ctx.base_Rcos.shape[0]), int(ctx.base_Rcos.shape[1])],
        "sensitivity": sensitivity,
        "fd_step": float(fd_step),
        "reference_length": float(vmec_meta["reference_length"]),
        "reference_b": float(vmec_meta["reference_b"]),
        "field_line_convention": str(vmec_meta["field_line_convention"]),
        "scope": (
            "VMEC-native metric and grad-B flux-tube mapping; full imported-VMEC/EIK "
            "drift parity and production transport-gradient gates remain separate."
        ),
    }


def vmec_jax_flux_tube_sensitivity_report(  # pragma: no cover
    *,
    params: jnp.ndarray | None = None,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    surface_index: int | None = None,
    alpha: float = 0.0,
    ntheta: int = 24,
    fd_step: float = 2.0e-6,
) -> dict[str, object]:
    """AD/FD-check VMEC-state derivatives through a solver-ready flux tube.

    Unlike the Boozer-only bridge, this report starts from a real
    ``vmec_jax`` state, evaluates VMEC metric and magnetic-field tensors, emits
    the SPECTRAX-GK ``FluxTubeGeometryData`` mapping, and differentiates
    geometry observables through that path.
    """

    p = _length_two_params(params, default=1.0e-4)

    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_available", False):
        return _vmec_sensitivity_unavailable_report(
            info=info,
            case_name=case_name,
            fd_step=fd_step,
            reason="vmec_jax is not available",
        )

    try:
        ctx = _load_vmec_state_context(str(case_name))
        ridx, midx, _sidx = _resolve_vmec_state_indices(
            ctx.base_Rcos,
            radial_index=radial_index,
            mode_index=mode_index,
            surface_index=surface_index,
            surface_grid="field_line",
        )
        mapping_fn = _vmec_flux_tube_mapping_fn(
            ctx=ctx,
            radial_index=ridx,
            mode_index=midx,
            surface_index=surface_index,
            alpha=alpha,
            ntheta=ntheta,
        )
        sensitivity = geometry_sensitivity_report(
            mapping_fn,
            p,
            fd_step=float(fd_step),
            source_model="vmec_jax:state->tensor-flux-tube",
        )
        mapping = mapping_fn(p)
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return _pack_vmec_sensitivity_report(
        ctx=ctx,
        sensitivity=sensitivity,
        mapping=mapping,
        params=p,
        case_name=case_name,
        radial_index=ridx,
        mode_index=midx,
        alpha=alpha,
        ntheta=ntheta,
        fd_step=fd_step,
        info=info,
    )


def _validate_vmec_parity_inputs(ntheta: int, mboz: int, nboz: int) -> tuple[int, int, int]:
    ntheta_int = int(ntheta)
    if ntheta_int < 4:
        raise ValueError("ntheta must be >= 4")
    mboz_int = int(mboz)
    nboz_int = int(nboz)
    if (
        mboz_int < _VMEC_BOOZER_PARITY_MIN_MODE_COUNT
        or nboz_int < _VMEC_BOOZER_PARITY_MIN_MODE_COUNT
    ):
        raise ValueError(
            "mboz and nboz must both be >= "
            f"{_VMEC_BOOZER_PARITY_MIN_MODE_COUNT} for VMEC/Boozer parity gates"
        )
    return ntheta_int, mboz_int, nboz_int


def _vmec_array_parity_unavailable_report(
    *,
    info: dict[str, object],
    case_name: str,
    reason: str,
) -> dict[str, object]:
    return {
        "available": False,
        "backend_info": info,
        "case_name": str(case_name),
        "reason": reason,
    }


def _surface_index_and_torflux(ctx: Any, surface_index: int | None) -> tuple[int, float]:
    ns = int(ctx.base_Rcos.shape[0])
    sidx = max(1, min(ns // 2, ns - 2)) if surface_index is None else int(surface_index)
    torflux = float(sidx) / float(max(ns - 1, 1))
    return sidx, torflux


def _direct_vmec_flux_tube_geometry(
    *,
    ctx: Any,
    surface_index: int,
    alpha: float,
    ntheta: int,
) -> Any:
    direct_mapping = vmec_jax_flux_tube_mapping_from_state(
        ctx.state,
        ctx.static,
        ctx.wout,
        surface_index=surface_index,
        alpha=float(alpha),
        ntheta=ntheta,
    )
    return flux_tube_geometry_from_mapping(
        direct_mapping,
        source_model="vmec_jax:state->tensor-flux-tube",
        validate_finite=False,
    )


def _vmec_eik_request(
    *,
    ctx: Any,
    ntheta: int,
    boundary: str,
    alpha: float,
    torflux: float,
    include_shear_variation: bool,
    include_pressure_variation: bool,
) -> SimpleNamespace:
    payload = dict(_VMEC_EIK_DEFAULT_REQUEST)
    payload.update(
        {
            "vmec_file": str(ctx.wout_path),
            "ntheta": ntheta,
            "boundary": str(boundary),
            "alpha": float(alpha),
            "torflux": torflux,
            "include_shear_variation": bool(include_shear_variation),
            "include_pressure_variation": bool(include_pressure_variation),
        }
    )
    return SimpleNamespace(**payload)


def _imported_vmec_eik_geometry(*, case_name: str, request: SimpleNamespace) -> Any:
    from spectraxgk.geometry import load_imported_geometry_netcdf
    from spectraxgk.geometry_backends.vmec import generate_vmec_eik_internal

    with tempfile.TemporaryDirectory(prefix="spectrax_vmec_eik_parity_") as tmp:
        eik_path = Path(tmp) / f"{case_name}.eik.nc"
        generate_vmec_eik_internal(output_path=eik_path, request=request)
        return load_imported_geometry_netcdf(eik_path)


def _load_vmec_parity_geometries(
    *,
    case_name: str,
    surface_index: int | None,
    alpha: float,
    ntheta: int,
    boundary: str,
    include_shear_variation: bool,
    include_pressure_variation: bool,
) -> tuple[Any, int, float, Any, Any]:
    ctx = _load_vmec_state_context(str(case_name))
    sidx, torflux = _surface_index_and_torflux(ctx, surface_index)
    direct = _direct_vmec_flux_tube_geometry(
        ctx=ctx,
        surface_index=sidx,
        alpha=alpha,
        ntheta=ntheta,
    )
    request = _vmec_eik_request(
        ctx=ctx,
        ntheta=ntheta,
        boundary=boundary,
        alpha=alpha,
        torflux=torflux,
        include_shear_variation=include_shear_variation,
        include_pressure_variation=include_pressure_variation,
    )
    imported = _imported_vmec_eik_geometry(case_name=case_name, request=request)
    if imported.theta.shape[0] == direct.theta.shape[0] + 1:
        imported = imported.trim_terminal_theta_point()
    return ctx, sidx, torflux, direct, imported


def _flux_tube_array_pairs(direct: Any, imported: Any) -> dict[str, tuple[Any, Any]]:
    return _array_pairs_from_attrs(direct, imported, _FLUX_TUBE_ARRAY_FIELDS)


def _flux_tube_scalar_metrics(direct: Any, imported: Any) -> dict[str, dict[str, Any]]:
    return {
        "gradpar": _scalar_parity_metrics(direct.gradpar_value, imported.gradpar_value),
        "q": _scalar_parity_metrics(direct.q, imported.q),
        "s_hat": _scalar_parity_metrics(direct.s_hat, imported.s_hat),
    }


def _production_parity_metrics(
    *,
    direct: Any,
    imported: Any,
    core_tolerance: float,
    scalar_tolerance: float,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, Any]], float, float, bool]:
    array_metrics = _array_metrics_from_pairs(_flux_tube_array_pairs(direct, imported))
    scalar_metrics = _flux_tube_scalar_metrics(direct, imported)
    worst_core = _worst_array_error(array_metrics, _PRODUCTION_CORE_ARRAY_NAMES)
    worst_scalar = max(float(values["rel"]) for values in scalar_metrics.values())
    production_parity_passed = bool(
        worst_core <= float(core_tolerance)
        and worst_scalar <= float(scalar_tolerance)
    )
    return array_metrics, scalar_metrics, worst_core, worst_scalar, production_parity_passed


def _pack_vmec_array_parity_report(
    *,
    ctx: Any,
    info: dict[str, object],
    case_name: str,
    surface_index: int,
    torflux: float,
    alpha: float,
    ntheta: int,
    mboz: int,
    nboz: int,
    boundary: str,
    include_shear_variation: bool,
    include_pressure_variation: bool,
    array_metrics: dict[str, dict[str, object]],
    scalar_metrics: dict[str, dict[str, Any]],
    equal_arc_parity: dict[str, object],
    equal_arc_core_tolerance: float,
    equal_arc_derivative_tolerance: float,
    equal_arc_metric_tolerance: float,
    equal_arc_drift_tolerance: float,
    worst_core: float,
    worst_scalar: float,
    core_tolerance: float,
    scalar_tolerance: float,
    production_parity_passed: bool,
) -> dict[str, object]:
    return {
        "available": True,
        "backend_info": info,
        "case_name": str(case_name),
        "input_path": str(ctx.input_path),
        "wout_path": str(ctx.wout_path),
        "source_model": "vmec_jax:state->tensor-flux-tube vs imported-vmec-eik",
        "surface_index": int(surface_index),
        "torflux": float(torflux),
        "alpha": float(alpha),
        "ntheta": int(ntheta),
        "mboz": int(mboz),
        "nboz": int(nboz),
        "boundary": str(boundary),
        "include_shear_variation": bool(include_shear_variation),
        "include_pressure_variation": bool(include_pressure_variation),
        "array_metrics": array_metrics,
        "scalar_metrics": scalar_metrics,
        **equal_arc_parity,
        "equal_arc_core_tolerance": float(equal_arc_core_tolerance),
        "equal_arc_derivative_tolerance": float(equal_arc_derivative_tolerance),
        "equal_arc_metric_tolerance": float(equal_arc_metric_tolerance),
        "equal_arc_drift_tolerance": float(equal_arc_drift_tolerance),
        "worst_core_normalized_max_abs": float(worst_core),
        "worst_scalar_rel": float(worst_scalar),
        "core_tolerance": float(core_tolerance),
        "scalar_tolerance": float(scalar_tolerance),
        "production_parity_passed": production_parity_passed,
        "status": "passed" if production_parity_passed else "diagnostic_open",
        "interpretation": (
            "The direct VMEC tensor path proves state-level differentiability. "
            "The Boozer equal-arc core audit narrows the production gap to the "
            "full imported VMEC/EIK metric and drift convention when it passes."
        ),
    }


@dataclass(frozen=True)
class _VMECArrayParityOptions:
    case_name: str
    surface_index: int | None
    alpha: float
    ntheta: int
    mboz: int
    nboz: int
    boundary: str
    include_shear_variation: bool
    include_pressure_variation: bool
    core_tolerance: float
    scalar_tolerance: float
    equal_arc_core_tolerance: float
    equal_arc_derivative_tolerance: float
    equal_arc_metric_tolerance: float
    equal_arc_drift_tolerance: float


@dataclass(frozen=True)
class _VMECArrayParityResult:
    ctx: Any
    surface_index: int
    torflux: float
    array_metrics: dict[str, dict[str, object]]
    scalar_metrics: dict[str, dict[str, Any]]
    equal_arc_parity: dict[str, object]
    worst_core: float
    worst_scalar: float
    production_parity_passed: bool


def _pack_vmec_array_parity_result_report(
    *,
    result: _VMECArrayParityResult,
    info: dict[str, object],
    options: _VMECArrayParityOptions,
) -> dict[str, object]:
    """Pack computed VMEC parity data using the public report schema."""

    return _pack_vmec_array_parity_report(
        ctx=result.ctx,
        info=info,
        case_name=options.case_name,
        surface_index=result.surface_index,
        torflux=result.torflux,
        alpha=options.alpha,
        ntheta=options.ntheta,
        mboz=options.mboz,
        nboz=options.nboz,
        boundary=options.boundary,
        include_shear_variation=options.include_shear_variation,
        include_pressure_variation=options.include_pressure_variation,
        array_metrics=result.array_metrics,
        scalar_metrics=result.scalar_metrics,
        equal_arc_parity=result.equal_arc_parity,
        equal_arc_core_tolerance=options.equal_arc_core_tolerance,
        equal_arc_derivative_tolerance=options.equal_arc_derivative_tolerance,
        equal_arc_metric_tolerance=options.equal_arc_metric_tolerance,
        equal_arc_drift_tolerance=options.equal_arc_drift_tolerance,
        worst_core=result.worst_core,
        worst_scalar=result.worst_scalar,
        core_tolerance=options.core_tolerance,
        scalar_tolerance=options.scalar_tolerance,
        production_parity_passed=result.production_parity_passed,
    )


def _vmec_array_parity_options(
    *,
    case_name: str,
    surface_index: int | None,
    alpha: float,
    ntheta: int,
    mboz: int,
    nboz: int,
    boundary: str,
    include_shear_variation: bool,
    include_pressure_variation: bool,
    core_tolerance: float,
    scalar_tolerance: float,
    equal_arc_core_tolerance: float,
    equal_arc_derivative_tolerance: float,
    equal_arc_metric_tolerance: float,
    equal_arc_drift_tolerance: float,
) -> _VMECArrayParityOptions:
    ntheta_int, mboz_int, nboz_int = _validate_vmec_parity_inputs(ntheta, mboz, nboz)
    return _VMECArrayParityOptions(
        case_name=str(case_name),
        surface_index=surface_index,
        alpha=float(alpha),
        ntheta=ntheta_int,
        mboz=mboz_int,
        nboz=nboz_int,
        boundary=str(boundary),
        include_shear_variation=bool(include_shear_variation),
        include_pressure_variation=bool(include_pressure_variation),
        core_tolerance=float(core_tolerance),
        scalar_tolerance=float(scalar_tolerance),
        equal_arc_core_tolerance=float(equal_arc_core_tolerance),
        equal_arc_derivative_tolerance=float(equal_arc_derivative_tolerance),
        equal_arc_metric_tolerance=float(equal_arc_metric_tolerance),
        equal_arc_drift_tolerance=float(equal_arc_drift_tolerance),
    )


def _vmec_array_parity_backend_unavailable_reason(info: dict[str, object]) -> str | None:
    if not info.get("vmec_jax_available", False):
        return "vmec_jax is not available"
    from spectraxgk.geometry_backends.vmec import internal_vmec_backend_available

    if not internal_vmec_backend_available():
        return "internal VMEC/EIK backend is not available"
    return None


def _vmec_array_parity_error_report(
    *,
    info: dict[str, object],
    case_name: str,
    exc: Exception,
) -> dict[str, object]:
    return {
        "available": False,
        "backend_info": info,
        "case_name": str(case_name),
        "error": f"{type(exc).__name__}: {exc}",
    }


def _vmec_array_parity_result(
    *,
    info: dict[str, object],
    options: _VMECArrayParityOptions,
) -> _VMECArrayParityResult:
    ctx, sidx, torflux, direct, imported = _load_vmec_parity_geometries(
        case_name=options.case_name,
        surface_index=options.surface_index,
        alpha=options.alpha,
        ntheta=options.ntheta,
        boundary=options.boundary,
        include_shear_variation=options.include_shear_variation,
        include_pressure_variation=options.include_pressure_variation,
    )
    (
        array_metrics,
        scalar_metrics,
        worst_core,
        worst_scalar,
        production_parity_passed,
    ) = _production_parity_metrics(
        direct=direct,
        imported=imported,
        core_tolerance=options.core_tolerance,
        scalar_tolerance=options.scalar_tolerance,
    )
    equal_arc_parity = _equal_arc_parity_report(
        info=info,
        ctx=ctx,
        imported=imported,
        surface_index=sidx,
        torflux=torflux,
        alpha=options.alpha,
        ntheta=options.ntheta,
        mboz=options.mboz,
        nboz=options.nboz,
        core_tolerance=options.equal_arc_core_tolerance,
        derivative_tolerance=options.equal_arc_derivative_tolerance,
        metric_tolerance=options.equal_arc_metric_tolerance,
        drift_tolerance=options.equal_arc_drift_tolerance,
    )
    return _VMECArrayParityResult(
        ctx=ctx,
        surface_index=sidx,
        torflux=torflux,
        array_metrics=array_metrics,
        scalar_metrics=scalar_metrics,
        equal_arc_parity=equal_arc_parity,
        worst_core=worst_core,
        worst_scalar=worst_scalar,
        production_parity_passed=production_parity_passed,
    )


def vmec_jax_flux_tube_array_parity_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    surface_index: int | None = None,
    alpha: float = 0.0,
    ntheta: int = 16,
    mboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    nboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    boundary: str = "none",
    include_shear_variation: bool = True,
    include_pressure_variation: bool = True,
    core_tolerance: float = 5.0e-2,
    scalar_tolerance: float = 5.0e-3,
    equal_arc_core_tolerance: float = 1.0e-2,
    equal_arc_derivative_tolerance: float = 3.0e-2,
    equal_arc_metric_tolerance: float = 8.0e-2,
    equal_arc_drift_tolerance: float = 8.0e-2,
) -> dict[str, object]:
    """Compare the direct ``vmec_jax`` flux-tube arrays to imported VMEC/EIK.

    This is a diagnostic promotion gate, not a differentiability check.  It
    starts from the same real ``vmec_jax`` example state used by
    :func:`vmec_jax_flux_tube_sensitivity_report`, builds the direct
    VMEC-tensor-derived flux-tube mapping, then generates the existing imported
    VMEC/EIK geometry on the same surface and compares solver-facing arrays.

    The expected current result is that ``q`` and magnetic shear are close
    while metric/drift arrays remain open because the direct path still uses a
    VMEC-coordinate/equal-theta convention and a local grad-:math:`B` closure
    instead of the production Boozer equal-arc/Hegna-Nakajima convention.
    """

    options = _vmec_array_parity_options(
        case_name=case_name,
        surface_index=surface_index,
        alpha=alpha,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        boundary=boundary,
        include_shear_variation=include_shear_variation,
        include_pressure_variation=include_pressure_variation,
        core_tolerance=core_tolerance,
        scalar_tolerance=scalar_tolerance,
        equal_arc_core_tolerance=equal_arc_core_tolerance,
        equal_arc_derivative_tolerance=equal_arc_derivative_tolerance,
        equal_arc_metric_tolerance=equal_arc_metric_tolerance,
        equal_arc_drift_tolerance=equal_arc_drift_tolerance,
    )
    info = discover_differentiable_geometry_backends()
    unavailable_reason = _vmec_array_parity_backend_unavailable_reason(info)
    if unavailable_reason is not None:
        return _vmec_array_parity_unavailable_report(
            info=info,
            case_name=options.case_name,
            reason=unavailable_reason,
        )

    try:
        result = _vmec_array_parity_result(info=info, options=options)
    except Exception as exc:
        return _vmec_array_parity_error_report(info=info, case_name=options.case_name, exc=exc)

    return _pack_vmec_array_parity_result_report(
        result=result,
        info=info,
        options=options,
    )


__all__ = [
    "vmec_jax_flux_tube_array_parity_report",
    "vmec_jax_flux_tube_sensitivity_report",
]
