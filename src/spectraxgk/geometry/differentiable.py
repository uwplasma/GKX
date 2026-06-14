"""Differentiable geometry bridge contracts for VMEC/JAX pipelines."""

from __future__ import annotations

from functools import wraps
from typing import Any

from spectraxgk.geometry import FluxTubeGeometryData
import spectraxgk.geometry.booz_xform_bridge as _booz_bridge
import spectraxgk.geometry.vmec_boozer_core as _vmec_boozer_core
import spectraxgk.geometry.vmec_flux_tube_reports as _vmec_flux_tube_reports
import spectraxgk.geometry.vmec_state_sensitivity as _vmec_state_sensitivity
import spectraxgk.geometry.vmec_tensor_mapping as _vmec_tensor_mapping
from spectraxgk.geometry.autodiff_checks import (
    _json_ready as _json_ready,
    _sensitivity_conditioning_metadata,
    finite_difference_jacobian,
    observable_gradient_validation_report,
)
from spectraxgk.geometry.backend_discovery import (
    _candidate_paths as _candidate_paths,
    _find_importable_module as _find_importable_module,
    _is_traced as _is_traced,
    discover_differentiable_geometry_backends,
)
from spectraxgk.geometry.booz_xform_bridge import (
    evaluate_boozer_bmag_on_field_line,
)
from spectraxgk.geometry.flux_tube_contract import (
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
from spectraxgk.geometry.numerics import (
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
from spectraxgk.geometry.sensitivity import (
    geometry_inverse_design_report,
    geometry_sensitivity_report,
)


_VMEC_BOOZER_PARITY_MIN_MODE_COUNT = 21
_DEFAULT_DISCOVER_DIFFERENTIABLE_GEOMETRY_BACKENDS = (
    discover_differentiable_geometry_backends
)


def _call_with_facade_backend_discovery(func: Any, *args: Any, **kwargs: Any) -> Any:
    if (
        discover_differentiable_geometry_backends
        is _DEFAULT_DISCOVER_DIFFERENTIABLE_GEOMETRY_BACKENDS
    ):
        return func(*args, **kwargs)
    original = _booz_bridge.discover_differentiable_geometry_backends
    _booz_bridge.discover_differentiable_geometry_backends = (
        discover_differentiable_geometry_backends
    )
    try:
        return func(*args, **kwargs)
    finally:
        _booz_bridge.discover_differentiable_geometry_backends = original


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
    originals = {
        "discover_differentiable_geometry_backends": (
            _vmec_state_sensitivity.discover_differentiable_geometry_backends
        ),
        "booz_xform_flux_tube_mapping_from_inputs": (
            _vmec_state_sensitivity.booz_xform_flux_tube_mapping_from_inputs
        ),
        "geometry_sensitivity_report": (
            _vmec_state_sensitivity.geometry_sensitivity_report
        ),
        "finite_difference_jacobian": (
            _vmec_state_sensitivity.finite_difference_jacobian
        ),
        "_periodic_bilinear_sample_2d": (
            _vmec_state_sensitivity._periodic_bilinear_sample_2d
        ),
        "_sensitivity_conditioning_metadata": (
            _vmec_state_sensitivity._sensitivity_conditioning_metadata
        ),
    }
    _vmec_state_sensitivity.discover_differentiable_geometry_backends = (
        discover_differentiable_geometry_backends
    )
    _vmec_state_sensitivity.booz_xform_flux_tube_mapping_from_inputs = (
        booz_xform_flux_tube_mapping_from_inputs
    )
    _vmec_state_sensitivity.geometry_sensitivity_report = geometry_sensitivity_report
    _vmec_state_sensitivity.finite_difference_jacobian = finite_difference_jacobian
    _vmec_state_sensitivity._periodic_bilinear_sample_2d = _periodic_bilinear_sample_2d
    _vmec_state_sensitivity._sensitivity_conditioning_metadata = (
        _sensitivity_conditioning_metadata
    )
    try:
        return func(*args, **kwargs)
    finally:
        for name, original in originals.items():
            setattr(_vmec_state_sensitivity, name, original)


@wraps(_vmec_state_sensitivity.vmec_jax_boozer_flux_tube_sensitivity_report)
def vmec_jax_boozer_flux_tube_sensitivity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_state_facade_hooks(
        _vmec_state_sensitivity.vmec_jax_boozer_flux_tube_sensitivity_report,
        *args,
        **kwargs,
    )


@wraps(_vmec_state_sensitivity.vmec_jax_metric_tensor_sensitivity_report)
def vmec_jax_metric_tensor_sensitivity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_state_facade_hooks(
        _vmec_state_sensitivity.vmec_jax_metric_tensor_sensitivity_report,
        *args,
        **kwargs,
    )


@wraps(_vmec_state_sensitivity.vmec_jax_field_line_tensor_sensitivity_report)
def vmec_jax_field_line_tensor_sensitivity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_state_facade_hooks(
        _vmec_state_sensitivity.vmec_jax_field_line_tensor_sensitivity_report,
        *args,
        **kwargs,
    )


@wraps(_vmec_tensor_mapping.vmec_jax_flux_tube_mapping_from_state)
def vmec_jax_flux_tube_mapping_from_state(*args: Any, **kwargs: Any) -> Any:
    original = _vmec_tensor_mapping._periodic_bilinear_sample_2d
    _vmec_tensor_mapping._periodic_bilinear_sample_2d = _periodic_bilinear_sample_2d
    try:
        return _vmec_tensor_mapping.vmec_jax_flux_tube_mapping_from_state(
            *args, **kwargs
        )
    finally:
        _vmec_tensor_mapping._periodic_bilinear_sample_2d = original


_cached_booz_xform_constants = _vmec_boozer_core._cached_booz_xform_constants


def _call_with_vmec_boozer_core_facade_hooks(
    func: Any, *args: Any, **kwargs: Any
) -> Any:
    originals = {
        "discover_differentiable_geometry_backends": (
            _vmec_boozer_core.discover_differentiable_geometry_backends
        ),
        "_boozer_half_mesh_s_grid": _vmec_boozer_core._boozer_half_mesh_s_grid,
        "_cumulative_trapezoid": _vmec_boozer_core._cumulative_trapezoid,
        "_evaluate_boozer_cosine_series_on_field_line": (
            _vmec_boozer_core._evaluate_boozer_cosine_series_on_field_line
        ),
        "_interp_equal_arc_profile": _vmec_boozer_core._interp_equal_arc_profile,
        "_interp_radial": _vmec_boozer_core._interp_radial,
        "_radial_derivative_array": _vmec_boozer_core._radial_derivative_array,
        "_radial_derivative_profile": _vmec_boozer_core._radial_derivative_profile,
    }
    _vmec_boozer_core.discover_differentiable_geometry_backends = (
        discover_differentiable_geometry_backends
    )
    _vmec_boozer_core._boozer_half_mesh_s_grid = _boozer_half_mesh_s_grid
    _vmec_boozer_core._cumulative_trapezoid = _cumulative_trapezoid
    _vmec_boozer_core._evaluate_boozer_cosine_series_on_field_line = (
        _evaluate_boozer_cosine_series_on_field_line
    )
    _vmec_boozer_core._interp_equal_arc_profile = _interp_equal_arc_profile
    _vmec_boozer_core._interp_radial = _interp_radial
    _vmec_boozer_core._radial_derivative_array = _radial_derivative_array
    _vmec_boozer_core._radial_derivative_profile = _radial_derivative_profile
    try:
        return func(*args, **kwargs)
    finally:
        for name, original in originals.items():
            setattr(_vmec_boozer_core, name, original)


@wraps(_vmec_boozer_core.prewarm_vmec_boozer_equal_arc_cache)
def prewarm_vmec_boozer_equal_arc_cache(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_boozer_core_facade_hooks(
        _vmec_boozer_core.prewarm_vmec_boozer_equal_arc_cache, *args, **kwargs
    )


@wraps(_vmec_boozer_core.vmec_jax_boozer_equal_arc_core_profiles_from_state)
def vmec_jax_boozer_equal_arc_core_profiles_from_state(
    *args: Any, **kwargs: Any
) -> Any:
    return _call_with_vmec_boozer_core_facade_hooks(
        _vmec_boozer_core.vmec_jax_boozer_equal_arc_core_profiles_from_state,
        *args,
        **kwargs,
    )


def flux_tube_geometry_from_vmec_boozer_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
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
    """Build solver-ready geometry directly from a solved ``vmec_jax`` state.

    This is the production-facing in-memory bridge for differentiable
    optimization workflows. It keeps the path inside JAX-compatible objects:

    ``VMECState -> BoozXformInputs -> booz_xform_jax -> FluxTubeGeometryData``.

    Runtime VMEC file generation can still use the NetCDF/EIK route, but
    differentiable stellarator optimization should call this function or a
    higher-level objective wrapper around it so gradients never pass through
    filesystem artifacts.
    """

    mapping = vmec_jax_boozer_equal_arc_core_profiles_from_state(
        state,
        static,
        indata,
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
    originals = {
        "discover_differentiable_geometry_backends": (
            _vmec_flux_tube_reports.discover_differentiable_geometry_backends
        ),
        "flux_tube_geometry_from_mapping": (
            _vmec_flux_tube_reports.flux_tube_geometry_from_mapping
        ),
        "geometry_sensitivity_report": (
            _vmec_flux_tube_reports.geometry_sensitivity_report
        ),
        "vmec_jax_boozer_equal_arc_core_profiles_from_state": (
            _vmec_flux_tube_reports.vmec_jax_boozer_equal_arc_core_profiles_from_state
        ),
        "vmec_jax_flux_tube_mapping_from_state": (
            _vmec_flux_tube_reports.vmec_jax_flux_tube_mapping_from_state
        ),
        "_array_parity_metrics": _vmec_flux_tube_reports._array_parity_metrics,
        "_scalar_parity_metrics": _vmec_flux_tube_reports._scalar_parity_metrics,
    }
    _vmec_flux_tube_reports.discover_differentiable_geometry_backends = (
        discover_differentiable_geometry_backends
    )
    _vmec_flux_tube_reports.flux_tube_geometry_from_mapping = (
        flux_tube_geometry_from_mapping
    )
    _vmec_flux_tube_reports.geometry_sensitivity_report = geometry_sensitivity_report
    _vmec_flux_tube_reports.vmec_jax_boozer_equal_arc_core_profiles_from_state = (
        vmec_jax_boozer_equal_arc_core_profiles_from_state
    )
    _vmec_flux_tube_reports.vmec_jax_flux_tube_mapping_from_state = (
        vmec_jax_flux_tube_mapping_from_state
    )
    _vmec_flux_tube_reports._array_parity_metrics = _array_parity_metrics
    _vmec_flux_tube_reports._scalar_parity_metrics = _scalar_parity_metrics
    try:
        return func(*args, **kwargs)
    finally:
        for name, original in originals.items():
            setattr(_vmec_flux_tube_reports, name, original)


@wraps(_vmec_flux_tube_reports.vmec_jax_flux_tube_sensitivity_report)
def vmec_jax_flux_tube_sensitivity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_report_facade_hooks(
        _vmec_flux_tube_reports.vmec_jax_flux_tube_sensitivity_report,
        *args,
        **kwargs,
    )


@wraps(_vmec_flux_tube_reports.vmec_jax_flux_tube_array_parity_report)
def vmec_jax_flux_tube_array_parity_report(*args: Any, **kwargs: Any) -> Any:
    return _call_with_vmec_report_facade_hooks(
        _vmec_flux_tube_reports.vmec_jax_flux_tube_array_parity_report,
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
    "vmec_jax_boozer_flux_tube_sensitivity_report",
    "vmec_jax_boozer_equal_arc_core_profiles_from_state",
    "vmec_jax_field_line_tensor_sensitivity_report",
    "vmec_jax_flux_tube_array_parity_report",
    "vmec_jax_flux_tube_mapping_from_state",
    "vmec_jax_flux_tube_sensitivity_report",
    "vmec_jax_metric_tensor_sensitivity_report",
    "vmec_boundary_aspect_sensitivity_report",
    "vmec_field_line_tensor_observable_names",
    "vmec_metric_tensor_observable_names",
]
