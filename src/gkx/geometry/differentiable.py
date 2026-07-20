"""Differentiable geometry bridge contracts for VMEC/JAX pipelines."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

from gkx.geometry import FluxTubeGeometryData
import gkx.geometry.booz_xform_bridge as _booz_bridge
import gkx.geometry.vmec_boozer_core as _vmec_boozer_core
import gkx.geometry.vmec_flux_tube_reports as _vmec_flux_tube_reports
import gkx.geometry.vmec_state_sensitivity as _vmec_state_sensitivity
import gkx.geometry.vmec_tensor_mapping as _vmec_tensor_mapping
from gkx.geometry.autodiff_checks import (
    _json_ready as _json_ready,
    _sensitivity_conditioning_metadata,
    finite_difference_jacobian,
    observable_gradient_validation_report,
)
from gkx.geometry.backend_discovery import (
    _candidate_paths as _candidate_paths,
    _find_importable_module as _find_importable_module,
    _is_traced as _is_traced,
    discover_differentiable_geometry_backends,
)
from gkx.geometry.booz_xform_bridge import (
    evaluate_boozer_bmag_on_field_line,
)
from gkx.geometry.flux_tube_contract import (
    _ARRAY_FIELDS as _ARRAY_FIELDS,
    _GEOMETRY_OBSERVABLE_NAMES as _GEOMETRY_OBSERVABLE_NAMES,
    _array as _array,
    _scalar as _scalar,
    flux_tube_geometry_from_mapping,
    flux_tube_geometry_observables,
    geometry_observable_names,
    vmec_field_line_tensor_observable_names,
    vmec_metric_tensor_observable_names,
)
from gkx.geometry.numerics import (
    _array_parity_metrics as _array_parity_metrics,
    _boozer_half_mesh_s_grid as _boozer_half_mesh_s_grid,
    _cumulative_trapezoid as _cumulative_trapezoid,
    _evaluate_boozer_cosine_series_on_field_line,
    _interp_equal_arc_profile as _interp_equal_arc_profile,
    _interp_radial as _interp_radial,
    _periodic_bilinear_sample_2d as _periodic_bilinear_sample_2d,
    _radial_derivative_array as _radial_derivative_array,
    _radial_derivative_profile as _radial_derivative_profile,
    _scalar_parity_metrics as _scalar_parity_metrics,
)
from gkx.geometry.sensitivity import (
    geometry_inverse_design_report,
    geometry_sensitivity_report,
)


_VMEC_BOOZER_PARITY_MIN_MODE_COUNT = 21
_DEFAULT_DISCOVER_DIFFERENTIABLE_GEOMETRY_BACKENDS = (
    discover_differentiable_geometry_backends
)


@contextmanager
def _patched_module_attrs(
    module: Any, replacements: dict[str, Any]
) -> Iterator[None]:
    """Temporarily patch module attributes and restore them after the call."""

    originals = {name: getattr(module, name) for name in replacements}
    for name, value in replacements.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def _call_with_facade_backend_discovery(func: Any, *args: Any, **kwargs: Any) -> Any:
    if (
        discover_differentiable_geometry_backends
        is _DEFAULT_DISCOVER_DIFFERENTIABLE_GEOMETRY_BACKENDS
    ):
        return func(*args, **kwargs)
    with _patched_module_attrs(
        _booz_bridge,
        {
            "discover_differentiable_geometry_backends": (
                discover_differentiable_geometry_backends
            )
        },
    ):
        return func(*args, **kwargs)


@wraps(_booz_bridge.vmec_boundary_aspect_sensitivity_report)
def vmec_boundary_aspect_sensitivity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_facade_backend_discovery(
        _booz_bridge.vmec_boundary_aspect_sensitivity_report, *args, **kwargs
    )


@wraps(_booz_bridge.booz_xform_spectral_sensitivity_report)
def booz_xform_spectral_sensitivity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_facade_backend_discovery(
        _booz_bridge.booz_xform_spectral_sensitivity_report, *args, **kwargs
    )


@wraps(_booz_bridge.booz_xform_flux_tube_mapping_from_inputs)
def booz_xform_flux_tube_mapping_from_inputs(*args: Any, **kwargs: Any) -> Any:
    return _call_with_facade_backend_discovery(
        _booz_bridge.booz_xform_flux_tube_mapping_from_inputs, *args, **kwargs
    )


@wraps(_booz_bridge.booz_xform_flux_tube_sensitivity_report)
def booz_xform_flux_tube_sensitivity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_facade_backend_discovery(
        _booz_bridge.booz_xform_flux_tube_sensitivity_report, *args, **kwargs
    )


def _call_with_vmec_state_facade_hooks(func: Any, *args: Any, **kwargs: Any) -> Any:
    with _patched_module_attrs(
        _vmec_state_sensitivity,
        {
            "discover_differentiable_geometry_backends": (
                discover_differentiable_geometry_backends
            ),
            "booz_xform_flux_tube_mapping_from_inputs": (
                booz_xform_flux_tube_mapping_from_inputs
            ),
            "geometry_sensitivity_report": geometry_sensitivity_report,
            "finite_difference_jacobian": finite_difference_jacobian,
            "_sensitivity_conditioning_metadata": (
                _sensitivity_conditioning_metadata
            ),
        },
    ):
        return func(*args, **kwargs)


@wraps(_vmec_state_sensitivity.vmex_boozer_flux_tube_sensitivity_report)
def vmex_boozer_flux_tube_sensitivity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_state_facade_hooks(
        _vmec_state_sensitivity.vmex_boozer_flux_tube_sensitivity_report,
        *args,
        **kwargs,
    )


@wraps(_vmec_state_sensitivity.vmex_metric_tensor_sensitivity_report)
def vmex_metric_tensor_sensitivity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_state_facade_hooks(
        _vmec_state_sensitivity.vmex_metric_tensor_sensitivity_report,
        *args,
        **kwargs,
    )


@wraps(_vmec_state_sensitivity.vmex_field_line_tensor_sensitivity_report)
def vmex_field_line_tensor_sensitivity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_state_facade_hooks(
        _vmec_state_sensitivity.vmex_field_line_tensor_sensitivity_report,
        *args,
        **kwargs,
    )


@wraps(_vmec_tensor_mapping.vmex_flux_tube_mapping_from_state)
def vmex_flux_tube_mapping_from_state(*args: Any, **kwargs: Any) -> Any:
    return _vmec_tensor_mapping.vmex_flux_tube_mapping_from_state(
        *args, **kwargs
    )


_cached_booz_xform_constants = _vmec_boozer_core._cached_booz_xform_constants


def _call_with_vmec_boozer_core_facade_hooks(
    func: Any, *args: Any, **kwargs: Any
) -> Any:
    with _patched_module_attrs(
        _vmec_boozer_core,
        {
            "_boozer_half_mesh_s_grid": _boozer_half_mesh_s_grid,
            "_cumulative_trapezoid": _cumulative_trapezoid,
            "_evaluate_boozer_cosine_series_on_field_line": (
                _evaluate_boozer_cosine_series_on_field_line
            ),
            "_interp_equal_arc_profile": _interp_equal_arc_profile,
            "_interp_radial": _interp_radial,
            "_radial_derivative_array": _radial_derivative_array,
            "_radial_derivative_profile": _radial_derivative_profile,
        },
    ):
        return func(*args, **kwargs)


@wraps(_vmec_boozer_core.prewarm_vmec_boozer_equal_arc_cache)
def prewarm_vmec_boozer_equal_arc_cache(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_boozer_core_facade_hooks(
        _vmec_boozer_core.prewarm_vmec_boozer_equal_arc_cache, *args, **kwargs
    )


@wraps(_vmec_boozer_core.vmex_boozer_equal_arc_core_profiles_from_state)
def vmex_boozer_equal_arc_core_profiles_from_state(
    *args: Any, **kwargs: Any
) -> Any:
    return _call_with_vmec_boozer_core_facade_hooks(
        _vmec_boozer_core.vmex_boozer_equal_arc_core_profiles_from_state,
        *args,
        **kwargs,
    )


def flux_tube_geometry_from_vmec_boozer_state(  # pragma: no cover
    state: Any,
    runtime: Any,
    inp: Any,
    wout: Any,
    *,
    surface_index: int | None = None,
    torflux: float | None = None,
    alpha: float = 0.0,
    ntheta: int = 32,
    mboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    nboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    jit: bool = False,
    surface_stencil_width: int | None = None,
    reference_length: float | None = None,
    reference_b: float | None = None,
    source_model: str = "mode21_vmec_boozer_state",
    validate_finite: bool = True,
) -> FluxTubeGeometryData:
    """Build solver-ready geometry directly from a solved ``vmex`` state.

    This is the production-facing in-memory bridge for differentiable
    optimization workflows. It keeps the path inside JAX-compatible objects:

    ``SpectralState -> boozer_input_tables -> booz_xform_jax ->
    FluxTubeGeometryData``.

    Runtime VMEC file generation can still use the NetCDF/EIK route, but
    differentiable stellarator optimization should call this function or a
    higher-level objective wrapper around it so gradients never pass through
    filesystem artifacts.
    """

    mapping = vmex_boozer_equal_arc_core_profiles_from_state(
        state,
        runtime,
        inp,
        wout,
        surface_index=surface_index,
        torflux=torflux,
        alpha=alpha,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        jit=jit,
        surface_stencil_width=surface_stencil_width,
        reference_length=reference_length,
        reference_b=reference_b,
    )
    return flux_tube_geometry_from_mapping(
        mapping,
        source_model=source_model,
        validate_finite=validate_finite,
    )


def _call_with_vmec_report_facade_hooks(func: Any, *args: Any, **kwargs: Any) -> Any:
    with _patched_module_attrs(
        _vmec_flux_tube_reports,
        {
            "discover_differentiable_geometry_backends": (
                discover_differentiable_geometry_backends
            ),
            "flux_tube_geometry_from_mapping": flux_tube_geometry_from_mapping,
            "geometry_sensitivity_report": geometry_sensitivity_report,
            "vmex_boozer_equal_arc_core_profiles_from_state": (
                vmex_boozer_equal_arc_core_profiles_from_state
            ),
            "vmex_flux_tube_mapping_from_state": (
                vmex_flux_tube_mapping_from_state
            ),
            "_array_parity_metrics": _array_parity_metrics,
            "_scalar_parity_metrics": _scalar_parity_metrics,
        },
    ):
        return func(*args, **kwargs)


@wraps(_vmec_flux_tube_reports.vmex_flux_tube_sensitivity_report)
def vmex_flux_tube_sensitivity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_report_facade_hooks(
        _vmec_flux_tube_reports.vmex_flux_tube_sensitivity_report,
        *args,
        **kwargs,
    )


@wraps(_vmec_flux_tube_reports.vmex_flux_tube_array_parity_report)
def vmex_flux_tube_array_parity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_report_facade_hooks(
        _vmec_flux_tube_reports.vmex_flux_tube_array_parity_report,
        *args,
        **kwargs,
    )


__all__ = [
    "booz_xform_flux_tube_mapping_from_inputs",
    "booz_xform_flux_tube_sensitivity_report",
    "booz_xform_spectral_sensitivity_report",
    "discover_differentiable_geometry_backends",
    "evaluate_boozer_bmag_on_field_line",
    "finite_difference_jacobian",
    "flux_tube_geometry_from_mapping",
    "flux_tube_geometry_from_vmec_boozer_state",
    "flux_tube_geometry_observables",
    "geometry_inverse_design_report",
    "geometry_observable_names",
    "geometry_sensitivity_report",
    "observable_gradient_validation_report",
    "vmex_boozer_flux_tube_sensitivity_report",
    "vmex_boozer_equal_arc_core_profiles_from_state",
    "vmex_field_line_tensor_sensitivity_report",
    "vmex_flux_tube_array_parity_report",
    "vmex_flux_tube_mapping_from_state",
    "vmex_flux_tube_sensitivity_report",
    "vmex_metric_tensor_sensitivity_report",
    "vmec_boundary_aspect_sensitivity_report",
    "vmec_field_line_tensor_observable_names",
    "vmec_metric_tensor_observable_names",
]
