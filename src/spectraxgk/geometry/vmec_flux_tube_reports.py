"""VMEC flux-tube sensitivity and parity reports."""

from __future__ import annotations

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


def _normalized_max_abs(metrics: dict[str, object]) -> float:
    raw_value = metrics.get("normalized_max_abs")
    return float(raw_value) if isinstance(raw_value, int | float | np.floating) else np.inf


def _array_metrics_from_pairs(pairs: dict[str, tuple[Any, Any]]) -> dict[str, dict[str, object]]:
    return {
        name: _array_parity_metrics(candidate, reference)
        for name, (candidate, reference) in pairs.items()
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
        equal_arc_core = vmec_jax_boozer_equal_arc_core_profiles_from_state(
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
    except Exception as exc:  # pragma: no cover - optional-backend diagnostic detail
        return _empty_equal_arc_parity(f"{type(exc).__name__}: {exc}")

    core_metrics = _array_metrics_from_pairs(
        {
            "theta": (equal_arc_core["theta"], imported.theta),
            "bmag": (equal_arc_core["bmag"], imported.bmag_profile),
            "bgrad": (equal_arc_core["bgrad"], imported.bgrad_profile),
            "jacobian": (equal_arc_core["jacobian"], imported.jacobian_profile),
        }
    )
    metric_metrics = _array_metrics_from_pairs(
        {
            "gds2": (equal_arc_core["gds2"], imported.gds2_profile),
            "gds21": (equal_arc_core["gds21"], imported.gds21_profile),
            "gds22": (equal_arc_core["gds22"], imported.gds22_profile),
            "grho": (equal_arc_core["grho"], imported.grho_profile),
        }
    )
    drift_metrics = _array_metrics_from_pairs(
        {
            "cvdrift": (equal_arc_core["cvdrift"], imported.cv_profile),
            "gbdrift": (equal_arc_core["gbdrift"], imported.gb_profile),
            "cvdrift0": (equal_arc_core["cvdrift0"], imported.cv0_profile),
            "gbdrift0": (equal_arc_core["gbdrift0"], imported.gb0_profile),
        }
    )
    scalar_metrics = {
        "gradpar": _scalar_parity_metrics(jnp.asarray(equal_arc_core["gradpar"])[0], imported.gradpar_value),
        "q": _scalar_parity_metrics(equal_arc_core["q"], imported.q),
        "s_hat": _scalar_parity_metrics(equal_arc_core["s_hat"], imported.s_hat),
    }
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
        "equal_arc_core_passed": bool(core_worst <= float(core_tolerance) and core_worst_scalar <= float(core_tolerance)),
        "equal_arc_derivative_passed": bool(derivative_worst <= float(derivative_tolerance)),
        "equal_arc_metric_passed": bool(metric_worst <= float(metric_tolerance)),
        "equal_arc_drift_passed": bool(drift_worst <= float(drift_tolerance)),
        "equal_arc_core_error": None,
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
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "reason": "vmec_jax is not available",
        }

    try:
        ctx = _load_vmec_state_context(str(case_name))
        ridx, midx, _sidx = _resolve_vmec_state_indices(
            ctx.base_Rcos,
            radial_index=radial_index,
            mode_index=mode_index,
            surface_index=surface_index,
            surface_grid="field_line",
        )

        def mapping_fn(x: jnp.ndarray) -> dict[str, Any]:
            traced_state = _perturb_vmec_state(
                ctx,
                x,
                radial_index=ridx,
                mode_index=midx,
            )
            return vmec_jax_flux_tube_mapping_from_state(
                traced_state,
                ctx.static,
                ctx.wout,
                surface_index=surface_index,
                alpha=float(alpha),
                ntheta=int(ntheta),
            )

        sensitivity = geometry_sensitivity_report(
            mapping_fn,
            p,
            fd_step=float(fd_step),
            source_model="vmec_jax:state->tensor-flux-tube",
        )
        mapping = mapping_fn(p)
        vmec_meta = mapping["vmec_jax"]
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "case_name": str(case_name),
        "input_path": str(ctx.input_path),
        "wout_path": str(ctx.wout_path),
        "source_model": "vmec_jax:state->tensor-flux-tube",
        "param_names": ["delta_Rcos", "delta_Zsin"],
        "params": np.asarray(p).tolist(),
        "radial_index": int(ridx),
        "mode_index": int(midx),
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

    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_available", False):
        return {
            "available": False,
            "backend_info": info,
            "case_name": str(case_name),
            "reason": "vmec_jax is not available",
        }

    try:
        from spectraxgk.geometry_backends.vmec import (
            generate_vmec_eik_internal,
            internal_vmec_backend_available,
        )
        from spectraxgk.geometry import load_imported_geometry_netcdf

        if not internal_vmec_backend_available():
            return {
                "available": False,
                "backend_info": info,
                "case_name": str(case_name),
                "reason": "internal VMEC/EIK backend is not available",
            }

        ctx = _load_vmec_state_context(str(case_name))
        ns = int(ctx.base_Rcos.shape[0])
        sidx = (
            max(1, min(ns // 2, ns - 2))
            if surface_index is None
            else int(surface_index)
        )
        torflux = float(sidx) / float(max(ns - 1, 1))
        direct_mapping = vmec_jax_flux_tube_mapping_from_state(
            ctx.state,
            ctx.static,
            ctx.wout,
            surface_index=sidx,
            alpha=float(alpha),
            ntheta=ntheta_int,
        )
        direct = flux_tube_geometry_from_mapping(
            direct_mapping,
            source_model="vmec_jax:state->tensor-flux-tube",
            validate_finite=False,
        )

        request = SimpleNamespace(
            vmec_file=str(ctx.wout_path),
            ntheta=ntheta_int,
            boundary=str(boundary),
            y0=10.0,
            x0=None,
            jtwist=None,
            beta=0.0,
            alpha=float(alpha),
            torflux=torflux,
            npol=1.0,
            npol_min=None,
            isaxisym=False,
            which_crossing=None,
            include_shear_variation=bool(include_shear_variation),
            include_pressure_variation=bool(include_pressure_variation),
            betaprim=None,
            z=(1.0, -1.0),
            mass=(1.0, 2.7e-4),
            dens=(1.0, 1.0),
            temp=(1.0, 1.0),
            tprim=(3.0, 0.0),
            fprim=(1.0, 0.0),
            vnewk=(0.0, 0.0),
            species_type=("ion", "electron"),
        )
        with tempfile.TemporaryDirectory(prefix="spectrax_vmec_eik_parity_") as tmp:
            eik_path = Path(tmp) / f"{case_name}.eik.nc"
            generate_vmec_eik_internal(output_path=eik_path, request=request)
            imported = load_imported_geometry_netcdf(eik_path)
            if imported.theta.shape[0] == direct.theta.shape[0] + 1:
                imported = imported.trim_terminal_theta_point()

        array_pairs = {
            "theta": (direct.theta, imported.theta),
            "bmag": (direct.bmag_profile, imported.bmag_profile),
            "bgrad": (direct.bgrad_profile, imported.bgrad_profile),
            "gds2": (direct.gds2_profile, imported.gds2_profile),
            "gds21": (direct.gds21_profile, imported.gds21_profile),
            "gds22": (direct.gds22_profile, imported.gds22_profile),
            "cvdrift": (direct.cv_profile, imported.cv_profile),
            "gbdrift": (direct.gb_profile, imported.gb_profile),
            "cvdrift0": (direct.cv0_profile, imported.cv0_profile),
            "gbdrift0": (direct.gb0_profile, imported.gb0_profile),
            "jacobian": (direct.jacobian_profile, imported.jacobian_profile),
            "grho": (direct.grho_profile, imported.grho_profile),
        }
        array_metrics = _array_metrics_from_pairs(array_pairs)
        scalar_metrics = {
            "gradpar": _scalar_parity_metrics(
                direct.gradpar_value, imported.gradpar_value
            ),
            "q": _scalar_parity_metrics(direct.q, imported.q),
            "s_hat": _scalar_parity_metrics(direct.s_hat, imported.s_hat),
        }
        core_names = (
            "bmag",
            "gds2",
            "gds21",
            "gds22",
            "cvdrift",
            "gbdrift",
            "jacobian",
            "grho",
        )
        worst_core = _worst_array_error(array_metrics, core_names)
        worst_scalar = max(float(values["rel"]) for values in scalar_metrics.values())
        production_parity_passed = bool(
            worst_core <= float(core_tolerance)
            and worst_scalar <= float(scalar_tolerance)
        )

        equal_arc_parity = _equal_arc_parity_report(
            info=info,
            ctx=ctx,
            imported=imported,
            surface_index=sidx,
            torflux=torflux,
            alpha=alpha,
            ntheta=ntheta_int,
            mboz=mboz_int,
            nboz=nboz_int,
            core_tolerance=equal_arc_core_tolerance,
            derivative_tolerance=equal_arc_derivative_tolerance,
            metric_tolerance=equal_arc_metric_tolerance,
            drift_tolerance=equal_arc_drift_tolerance,
        )
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "case_name": str(case_name),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "case_name": str(case_name),
        "input_path": str(ctx.input_path),
        "wout_path": str(ctx.wout_path),
        "source_model": "vmec_jax:state->tensor-flux-tube vs imported-vmec-eik",
        "surface_index": int(sidx),
        "torflux": float(torflux),
        "alpha": float(alpha),
        "ntheta": int(ntheta_int),
        "mboz": mboz_int,
        "nboz": nboz_int,
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


__all__ = [
    "vmec_jax_flux_tube_array_parity_report",
    "vmec_jax_flux_tube_sensitivity_report",
]
