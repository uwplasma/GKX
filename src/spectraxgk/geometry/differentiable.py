"""Differentiable geometry bridge contracts for VMEC/JAX pipelines."""

from __future__ import annotations

import importlib
from dataclasses import replace as dc_replace
from functools import lru_cache, wraps
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from spectraxgk.geometry import FluxTubeGeometryData
import spectraxgk.geometry.booz_xform_bridge as _booz_bridge
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
    _VMEC_FIELD_LINE_OBSERVABLE_NAMES,
    _VMEC_METRIC_OBSERVABLE_NAMES,
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

    p = jnp.asarray([1.0e-3, 1.0e-3] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")

    info = discover_differentiable_geometry_backends()
    if not (
        info.get("vmec_jax_available", False)
        and info.get("booz_xform_jax_api_available", False)
    ):
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "case_name": str(case_name),
            "reason": "vmec_jax or booz_xform_jax functional API is not available",
        }

    try:
        driver = importlib.import_module("vmec_jax.driver")
        config_mod = importlib.import_module("vmec_jax.config")
        static_mod = importlib.import_module("vmec_jax.static")
        wout_mod = importlib.import_module("vmec_jax.wout")
        booz_input_mod = importlib.import_module("vmec_jax.booz_input")

        input_path, wout_path = driver.example_paths(str(case_name))
        if wout_path is None:
            raise RuntimeError(
                f"vmec_jax example {case_name!r} has no bundled wout reference"
            )

        cfg, indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        base_Rcos = jnp.asarray(state.Rcos)
        base_Zsin = jnp.asarray(state.Zsin)
        if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
            raise RuntimeError(
                "vmec_jax state Rcos/Zsin arrays must be two-dimensional"
            )

        ridx = (
            int(base_Rcos.shape[0] // 2) if radial_index is None else int(radial_index)
        )
        midx = int(mode_index)
        if not (0 <= ridx < int(base_Rcos.shape[0])):
            raise ValueError("radial_index is outside the VMEC state radial grid")
        if not (0 <= midx < int(base_Rcos.shape[1])):
            raise ValueError("mode_index is outside the VMEC state mode table")
        sidx = (
            max(0, min(ridx - 1, int(base_Rcos.shape[0]) - 2))
            if surface_index is None
            else int(surface_index)
        )
        if not (0 <= sidx < int(base_Rcos.shape[0]) - 1):
            raise ValueError(
                "surface_index is outside the VMEC half-mesh Boozer surface grid"
            )

        def mapping_fn(x: jnp.ndarray) -> dict[str, Any]:
            traced_state = dc_replace(
                state,
                Rcos=base_Rcos.at[ridx, midx].add(x[0]),
                Zsin=base_Zsin.at[ridx, midx].add(x[1]),
            )
            inputs = booz_input_mod.booz_xform_inputs_from_state(
                state=traced_state,
                static=static,
                indata=indata,
                signgs=wout.signgs,
            )
            return booz_xform_flux_tube_mapping_from_inputs(
                inputs,
                mboz=int(mboz),
                nboz=int(nboz),
                ntheta=int(ntheta),
                surface_index=int(sidx),
                magnetic_shear=0.35,
                jit=False,
            )

        sensitivity = geometry_sensitivity_report(
            mapping_fn,
            p,
            fd_step=float(fd_step),
            source_model="vmec_jax:state->booz_xform_jax:field-line-bmag",
        )
        mapping = mapping_fn(p)
        booz_meta = mapping["booz_xform"]
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
        "input_path": str(input_path),
        "wout_path": str(wout_path),
        "param_names": ["delta_Rcos", "delta_Zsin"],
        "params": np.asarray(p).tolist(),
        "radial_index": int(ridx),
        "mode_index": int(midx),
        "surface_index": int(sidx),
        "state_shape": [int(base_Rcos.shape[0]), int(base_Rcos.shape[1])],
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

    p = jnp.asarray([1.0e-3, 1.0e-3] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")

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
        driver = importlib.import_module("vmec_jax.driver")
        config_mod = importlib.import_module("vmec_jax.config")
        static_mod = importlib.import_module("vmec_jax.static")
        wout_mod = importlib.import_module("vmec_jax.wout")
        geom_mod = importlib.import_module("vmec_jax.geom")

        input_path, wout_path = driver.example_paths(str(case_name))
        if wout_path is None:
            raise RuntimeError(
                f"vmec_jax example {case_name!r} has no bundled wout reference"
            )

        cfg, _indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        base_Rcos = jnp.asarray(state.Rcos)
        base_Zsin = jnp.asarray(state.Zsin)
        if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
            raise RuntimeError(
                "vmec_jax state Rcos/Zsin arrays must be two-dimensional"
            )

        ridx = (
            int(base_Rcos.shape[0] // 2) if radial_index is None else int(radial_index)
        )
        midx = int(mode_index)
        if not (0 <= ridx < int(base_Rcos.shape[0])):
            raise ValueError("radial_index is outside the VMEC state radial grid")
        if not (0 <= midx < int(base_Rcos.shape[1])):
            raise ValueError("mode_index is outside the VMEC state mode table")
        sidx = (
            max(0, min(ridx - 1, int(base_Rcos.shape[0]) - 1))
            if surface_index is None
            else int(surface_index)
        )
        if not (0 <= sidx < int(base_Rcos.shape[0])):
            raise ValueError("surface_index is outside the VMEC metric radial grid")

        eps = jnp.asarray(float(rms_epsilon), dtype=p.dtype)

        def _rms(arr: jnp.ndarray) -> jnp.ndarray:
            arr = jnp.asarray(arr)
            return jnp.sqrt(jnp.mean(arr * arr) + eps)

        def metric_observables(x: jnp.ndarray) -> jnp.ndarray:
            traced_state = dc_replace(
                state,
                Rcos=base_Rcos.at[ridx, midx].add(x[0]),
                Zsin=base_Zsin.at[ridx, midx].add(x[1]),
            )
            geom = geom_mod.eval_geom(traced_state, static)
            sqrtg = jnp.asarray(geom.sqrtg)[sidx]
            g_ss = jnp.asarray(geom.g_ss)[sidx]
            g_st = jnp.asarray(geom.g_st)[sidx]
            g_sp = jnp.asarray(geom.g_sp)[sidx]
            g_tt = jnp.asarray(geom.g_tt)[sidx]
            g_tp = jnp.asarray(geom.g_tp)[sidx]
            g_pp = jnp.asarray(geom.g_pp)[sidx]
            return jnp.asarray(
                [
                    _rms(sqrtg),
                    jnp.mean(g_ss),
                    jnp.mean(g_tt),
                    jnp.mean(g_pp),
                    _rms(g_st),
                    _rms(g_sp),
                    _rms(g_tp),
                ]
            )

        observables = metric_observables(p)
        jac_ad = jax.jacfwd(metric_observables)(p)
        jac_fd = finite_difference_jacobian(metric_observables, p, step=float(fd_step))
        diff = jac_ad - jac_fd
        max_abs = jnp.max(jnp.abs(diff))
        max_rel = jnp.max(jnp.abs(diff) / (jnp.abs(jac_fd) + 1.0e-12))
        geom0 = geom_mod.eval_geom(state, static)
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
        "input_path": str(input_path),
        "wout_path": str(wout_path),
        "source_model": "vmec_jax:state->metric-tensors",
        "param_names": ["delta_Rcos", "delta_Zsin"],
        "observable_names": list(_VMEC_METRIC_OBSERVABLE_NAMES),
        "params": np.asarray(p).tolist(),
        "observables": np.asarray(observables).tolist(),
        "jacobian_ad": np.asarray(jac_ad).tolist(),
        "jacobian_fd": np.asarray(jac_fd).tolist(),
        "max_abs_ad_fd_error": float(np.asarray(max_abs)),
        "max_rel_ad_fd_error": float(np.asarray(max_rel)),
        "conditioning": _sensitivity_conditioning_metadata(
            jac_ad,
            jac_fd,
            p,
            fd_step=float(fd_step),
            observable_names=_VMEC_METRIC_OBSERVABLE_NAMES,
            param_names=("delta_Rcos", "delta_Zsin"),
            relative_floor=1.0e-12,
        ),
        "radial_index": int(ridx),
        "mode_index": int(midx),
        "surface_index": int(sidx),
        "state_shape": [int(base_Rcos.shape[0]), int(base_Rcos.shape[1])],
        "metric_grid_shape": [int(v) for v in np.asarray(geom0.sqrtg).shape],
        "fd_step": float(fd_step),
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

    p = jnp.asarray([1.0e-4, 1.0e-4] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")
    ntheta_int = int(ntheta)
    if ntheta_int < 4:
        raise ValueError("ntheta must be >= 4")

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
        driver = importlib.import_module("vmec_jax.driver")
        config_mod = importlib.import_module("vmec_jax.config")
        static_mod = importlib.import_module("vmec_jax.static")
        wout_mod = importlib.import_module("vmec_jax.wout")
        geom_mod = importlib.import_module("vmec_jax.geom")
        bcovar_mod = importlib.import_module("vmec_jax.vmec_bcovar")
        field_mod = importlib.import_module("vmec_jax.field")

        input_path, wout_path = driver.example_paths(str(case_name))
        if wout_path is None:
            raise RuntimeError(
                f"vmec_jax example {case_name!r} has no bundled wout reference"
            )

        cfg, _indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        base_Rcos = jnp.asarray(state.Rcos)
        base_Zsin = jnp.asarray(state.Zsin)
        if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
            raise RuntimeError(
                "vmec_jax state Rcos/Zsin arrays must be two-dimensional"
            )

        ridx = (
            int(base_Rcos.shape[0] // 2) if radial_index is None else int(radial_index)
        )
        midx = int(mode_index)
        if not (0 <= ridx < int(base_Rcos.shape[0])):
            raise ValueError("radial_index is outside the VMEC state radial grid")
        if not (0 <= midx < int(base_Rcos.shape[1])):
            raise ValueError("mode_index is outside the VMEC state mode table")
        sidx = (
            max(1, min(ridx, int(base_Rcos.shape[0]) - 2))
            if surface_index is None
            else int(surface_index)
        )
        if not (0 <= sidx < int(base_Rcos.shape[0])):
            raise ValueError("surface_index is outside the VMEC metric radial grid")

        iota_profile = jnp.asarray(getattr(wout, "iotas"))
        if iota_profile.ndim != 1 or int(iota_profile.shape[0]) <= sidx:
            raise RuntimeError(
                "vmec_jax wout iotas profile is missing or incompatible with the state grid"
            )
        iota_line = iota_profile[sidx]
        iota_safe = jnp.where(
            jnp.abs(iota_line) < 1.0e-12,
            jnp.sign(iota_line + 1.0e-30) * 1.0e-12,
            iota_line,
        )
        theta_line = jnp.linspace(
            -jnp.pi, jnp.pi, ntheta_int, endpoint=False, dtype=p.dtype
        )
        theta_vmec = jnp.mod(theta_line + jnp.pi, 2.0 * jnp.pi)
        zeta_line = jnp.mod(
            (theta_vmec - jnp.asarray(float(alpha), dtype=p.dtype)) / iota_safe,
            2.0 * jnp.pi,
        )
        b2_floor_arr = jnp.asarray(float(b2_floor), dtype=p.dtype)
        eps = jnp.asarray(float(rms_epsilon), dtype=p.dtype)

        def _rms(arr: jnp.ndarray) -> jnp.ndarray:
            arr = jnp.asarray(arr)
            return jnp.sqrt(jnp.mean(arr * arr) + eps)

        def field_line_observables(x: jnp.ndarray) -> jnp.ndarray:
            traced_state = dc_replace(
                state,
                Rcos=base_Rcos.at[ridx, midx].add(x[0]),
                Zsin=base_Zsin.at[ridx, midx].add(x[1]),
            )
            geom = geom_mod.eval_geom(traced_state, static)
            bcovar = bcovar_mod.vmec_bcovar_half_mesh_from_wout(
                state=traced_state,
                static=static,
                wout=wout,
                pres=getattr(wout, "pres", None),
            )
            b2 = field_mod.b2_from_bsup(geom, bcovar.bsupu, bcovar.bsupv)
            bmag = jnp.sqrt(
                jnp.maximum(
                    _periodic_bilinear_sample_2d(b2[sidx], theta_vmec, zeta_line),
                    b2_floor_arr,
                )
            )
            sqrtg = _periodic_bilinear_sample_2d(
                geom.sqrtg[sidx], theta_vmec, zeta_line
            )
            g_tt = _periodic_bilinear_sample_2d(geom.g_tt[sidx], theta_vmec, zeta_line)
            g_tp = _periodic_bilinear_sample_2d(geom.g_tp[sidx], theta_vmec, zeta_line)
            g_pp = _periodic_bilinear_sample_2d(geom.g_pp[sidx], theta_vmec, zeta_line)
            g_ss = _periodic_bilinear_sample_2d(geom.g_ss[sidx], theta_vmec, zeta_line)
            mean_b = jnp.mean(bmag)
            ripple = jnp.std(bmag) / jnp.maximum(
                jnp.abs(mean_b), jnp.asarray(1.0e-30, dtype=bmag.dtype)
            )
            return jnp.asarray(
                [
                    mean_b,
                    ripple,
                    _rms(sqrtg),
                    jnp.mean(g_tt),
                    jnp.mean(g_pp),
                    _rms(g_tp),
                    jnp.mean(g_ss),
                ]
            )

        observables = field_line_observables(p)
        jac_ad = jax.jacfwd(field_line_observables)(p)
        jac_fd = finite_difference_jacobian(
            field_line_observables, p, step=float(fd_step)
        )
        diff = jac_ad - jac_fd
        max_abs = jnp.max(jnp.abs(diff))
        max_rel = jnp.max(jnp.abs(diff) / (jnp.abs(jac_fd) + 1.0e-10))
        geom0 = geom_mod.eval_geom(state, static)
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
        "input_path": str(input_path),
        "wout_path": str(wout_path),
        "source_model": "vmec_jax:state->field-line-metric-and-b",
        "field_line_convention": "VMEC theta, zeta=(theta-alpha)/iota with periodic bilinear sampling",
        "param_names": ["delta_Rcos", "delta_Zsin"],
        "observable_names": list(_VMEC_FIELD_LINE_OBSERVABLE_NAMES),
        "params": np.asarray(p).tolist(),
        "observables": np.asarray(observables).tolist(),
        "jacobian_ad": np.asarray(jac_ad).tolist(),
        "jacobian_fd": np.asarray(jac_fd).tolist(),
        "max_abs_ad_fd_error": float(np.asarray(max_abs)),
        "max_rel_ad_fd_error": float(np.asarray(max_rel)),
        "conditioning": _sensitivity_conditioning_metadata(
            jac_ad,
            jac_fd,
            p,
            fd_step=float(fd_step),
            observable_names=_VMEC_FIELD_LINE_OBSERVABLE_NAMES,
            param_names=("delta_Rcos", "delta_Zsin"),
            relative_floor=1.0e-10,
        ),
        "radial_index": int(ridx),
        "mode_index": int(midx),
        "surface_index": int(sidx),
        "iota": float(np.asarray(iota_line)),
        "alpha": float(alpha),
        "ntheta": int(ntheta_int),
        "state_shape": [int(base_Rcos.shape[0]), int(base_Rcos.shape[1])],
        "metric_grid_shape": [int(v) for v in np.asarray(geom0.sqrtg).shape],
        "fd_step": float(fd_step),
        "b2_floor": float(b2_floor),
        "rms_epsilon": float(rms_epsilon),
    }


def vmec_jax_flux_tube_mapping_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    wout: Any,
    *,
    surface_index: int | None = None,
    alpha: float = 0.0,
    ntheta: int = 32,
    b2_floor: float = 1.0e-24,
    metric_floor: float = 1.0e-24,
    reference_length: float | None = None,
    reference_b: float | None = None,
    drift_scale: float = 1.0,
) -> dict[str, Any]:
    """Build a solver-ready flux-tube mapping directly from ``vmec_jax`` tensors.

    This is the VMEC-native bridge step: it evaluates ``vmec_jax.geom`` and
    ``vmec_jax.vmec_bcovar``, samples the covariant metric and ``|B|`` on a
    fixed field line, inverts the sampled metric to construct perpendicular
    flux-tube metric coefficients, and emits the
    :func:`flux_tube_geometry_from_mapping` contract.

    The metric and magnetic-field derivatives are differentiable with respect
    to the VMEC state.  The drift coefficients use a local grad-:math:`B`
    projection closure so that downstream solver contracts remain populated;
    the full Hegna-Nakajima/imported-VMEC drift parity gate remains a separate
    production promotion step.
    """

    ntheta_int = int(ntheta)
    if ntheta_int < 4:
        raise ValueError("ntheta must be >= 4")

    geom_mod = importlib.import_module("vmec_jax.geom")
    bcovar_mod = importlib.import_module("vmec_jax.vmec_bcovar")
    field_mod = importlib.import_module("vmec_jax.field")

    base_Rcos = jnp.asarray(state.Rcos)
    if base_Rcos.ndim != 2:
        raise RuntimeError("vmec_jax state Rcos array must be two-dimensional")
    ns = int(base_Rcos.shape[0])
    if ns < 3:
        raise RuntimeError("vmec_jax state needs at least three radial surfaces")
    sidx = max(1, min(ns // 2, ns - 2)) if surface_index is None else int(surface_index)
    if not (0 < sidx < ns - 1):
        raise ValueError("surface_index must be an interior VMEC radial index")

    iota_profile = jnp.asarray(getattr(wout, "iotas"))
    if iota_profile.ndim != 1 or int(iota_profile.shape[0]) <= sidx + 1:
        raise RuntimeError(
            "vmec_jax wout iotas profile is missing or incompatible with the state grid"
        )
    iota_line = iota_profile[sidx]
    iota_safe = jnp.where(
        jnp.abs(iota_line) < 1.0e-12, jnp.sign(iota_line + 1.0e-30) * 1.0e-12, iota_line
    )

    s_grid = jnp.linspace(0.0, 1.0, ns, dtype=base_Rcos.dtype)
    s_val = jnp.maximum(s_grid[sidx], jnp.asarray(1.0e-12, dtype=base_Rcos.dtype))
    ds = s_grid[1] - s_grid[0]
    d_iota_ds = (iota_profile[sidx + 1] - iota_profile[sidx - 1]) / (2.0 * ds)
    s_hat = -2.0 * s_val * d_iota_ds / iota_safe

    raw_length = (
        float(getattr(wout, "Aminor_p", 1.0))
        if reference_length is None
        else float(reference_length)
    )
    L_reference = (
        raw_length if np.isfinite(raw_length) and abs(raw_length) > 0.0 else 1.0
    )
    if reference_b is None:
        phi_profile = np.asarray(getattr(wout, "phi", [0.0, np.pi]), dtype=float)
        edge_toroidal_flux_over_2pi = abs(float(phi_profile[-1]) / (2.0 * np.pi))
        raw_b = 2.0 * edge_toroidal_flux_over_2pi / (L_reference * L_reference)
        B_reference = raw_b if np.isfinite(raw_b) and abs(raw_b) > 0.0 else 1.0
    else:
        B_reference = float(reference_b)
    B_reference = (
        B_reference if np.isfinite(B_reference) and abs(B_reference) > 0.0 else 1.0
    )

    theta_line = jnp.linspace(
        -jnp.pi, jnp.pi, ntheta_int, endpoint=False, dtype=base_Rcos.dtype
    )
    theta_vmec = jnp.mod(theta_line + jnp.pi, 2.0 * jnp.pi)
    zeta_line = jnp.mod(
        (theta_vmec - jnp.asarray(float(alpha), dtype=base_Rcos.dtype)) / iota_safe,
        2.0 * jnp.pi,
    )

    geom = geom_mod.eval_geom(state, static)
    bcovar = bcovar_mod.vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout,
        pres=getattr(wout, "pres", None),
    )
    b2 = field_mod.b2_from_bsup(geom, bcovar.bsupu, bcovar.bsupv)
    b2_floor_arr = jnp.asarray(float(b2_floor), dtype=base_Rcos.dtype)
    metric_floor_arr = jnp.asarray(float(metric_floor), dtype=base_Rcos.dtype)

    bmag_all = jnp.sqrt(jnp.maximum(jnp.asarray(b2), b2_floor_arr)) / float(B_reference)
    bmag_grid = bmag_all[sidx]
    dtheta_grid = 2.0 * jnp.pi / float(bmag_grid.shape[0])
    dzeta_grid = 2.0 * jnp.pi / float(bmag_grid.shape[1])
    db_dtheta_grid = (
        jnp.roll(bmag_grid, -1, axis=0) - jnp.roll(bmag_grid, 1, axis=0)
    ) / (2.0 * dtheta_grid)
    db_dzeta_grid = (
        jnp.roll(bmag_grid, -1, axis=1) - jnp.roll(bmag_grid, 1, axis=1)
    ) / (2.0 * dzeta_grid)
    db_ds_grid = (bmag_all[sidx + 1] - bmag_all[sidx - 1]) / (2.0 * ds)

    def _line(values: jnp.ndarray) -> jnp.ndarray:
        return _periodic_bilinear_sample_2d(values, theta_vmec, zeta_line)

    bmag = _line(bmag_grid)
    sqrtg = _line(jnp.asarray(geom.sqrtg)[sidx])
    g_ss = _line(jnp.asarray(geom.g_ss)[sidx])
    g_st = _line(jnp.asarray(geom.g_st)[sidx])
    g_sp = _line(jnp.asarray(geom.g_sp)[sidx])
    g_tt = _line(jnp.asarray(geom.g_tt)[sidx])
    g_tp = _line(jnp.asarray(geom.g_tp)[sidx])
    g_pp = _line(jnp.asarray(geom.g_pp)[sidx])
    db_ds = _line(db_ds_grid)
    db_dt = _line(db_dtheta_grid)
    db_dp = _line(db_dzeta_grid)

    cov_metric = jnp.stack(
        (
            jnp.stack((g_ss, g_st, g_sp), axis=-1),
            jnp.stack((g_st, g_tt, g_tp), axis=-1),
            jnp.stack((g_sp, g_tp, g_pp), axis=-1),
        ),
        axis=-2,
    )
    contra = jnp.linalg.inv(cov_metric)
    alpha_coeff = (
        jnp.asarray([0.0, 1.0, -1.0], dtype=base_Rcos.dtype).at[2].set(-iota_safe)
    )
    grad_alpha_metric = jnp.einsum("i,zij,j->z", alpha_coeff, contra, alpha_coeff)
    grad_s_metric = contra[:, 0, 0]
    grad_s_dot_alpha = contra[:, 0, 1] - iota_safe * contra[:, 0, 2]

    grad_alpha_metric = jnp.maximum(grad_alpha_metric, metric_floor_arr)
    grad_s_metric = jnp.maximum(grad_s_metric, metric_floor_arr)
    sqrt_s = jnp.sqrt(s_val)
    L = jnp.asarray(float(L_reference), dtype=base_Rcos.dtype)
    Bref = jnp.asarray(float(B_reference), dtype=base_Rcos.dtype)
    shat_safe = jnp.where(
        jnp.abs(s_hat) < 1.0e-12, jnp.asarray(0.0, dtype=base_Rcos.dtype), s_hat
    )
    gds2 = L * L * s_val * grad_alpha_metric
    gds21 = (
        shat_safe
        * grad_s_dot_alpha
        / jnp.maximum(Bref, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype))
    )
    gds22 = (
        shat_safe
        * shat_safe
        * grad_s_metric
        / jnp.maximum(
            L * L * Bref * Bref * s_val, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
        )
    )
    grho = jnp.sqrt(grad_s_metric) / jnp.maximum(
        L * Bref * sqrt_s, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
    )

    gradpar_profile = jnp.abs(
        L
        * iota_safe
        / jnp.maximum(
            jnp.abs(bmag * sqrtg), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
        )
    )
    gradpar_value = jnp.mean(gradpar_profile)
    gradpar = gradpar_value * jnp.ones_like(theta_line)
    dlogb_dtheta = (
        jnp.roll(jnp.log(jnp.maximum(bmag, 1.0e-30)), -1)
        - jnp.roll(jnp.log(jnp.maximum(bmag, 1.0e-30)), 1)
    ) / (2.0 * (2.0 * jnp.pi / float(ntheta_int)))
    bgrad = gradpar_value * dlogb_dtheta

    grad_b_cov = jnp.stack((db_ds, db_dt, db_dp), axis=-1)
    grad_b_dot_alpha = jnp.einsum("zi,zij,j->z", grad_b_cov, contra, alpha_coeff)
    grad_b_dot_s = jnp.einsum("zi,zi->z", grad_b_cov, contra[:, :, 0])
    bmag_sq = jnp.maximum(bmag * bmag, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype))
    gbdrift = -2.0 * float(drift_scale) * L * L * sqrt_s * grad_b_dot_alpha / bmag_sq
    gbdrift0 = (
        -2.0
        * float(drift_scale)
        * shat_safe
        * grad_b_dot_s
        / jnp.maximum(
            Bref * bmag_sq * sqrt_s, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
        )
    )

    return {
        "theta": theta_line,
        "gradpar": gradpar,
        "bmag": bmag,
        "bgrad": bgrad,
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "cvdrift": gbdrift,
        "gbdrift": gbdrift,
        "cvdrift0": gbdrift0,
        "gbdrift0": gbdrift0,
        "jacobian": jnp.abs(sqrtg)
        / jnp.maximum(
            jnp.mean(jnp.abs(sqrtg)), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
        ),
        "grho": grho,
        "q": 1.0
        / jnp.maximum(jnp.abs(iota_safe), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)),
        "s_hat": shat_safe,
        "epsilon": jnp.std(bmag)
        / jnp.maximum(jnp.mean(bmag), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)),
        "R0": float(L_reference),
        "B0": float(B_reference),
        "alpha": float(alpha),
        "drift_scale": float(drift_scale),
        "nfp": int(getattr(wout, "nfp", 1)),
        "vmec_jax": {
            "surface_index": int(sidx),
            "iota": iota_line,
            "reference_length": float(L_reference),
            "reference_b": float(B_reference),
            "field_line_convention": "VMEC theta, zeta=(theta-alpha)/iota with periodic bilinear sampling",
        },
    }


@lru_cache(maxsize=32)
def _cached_booz_xform_constants(
    *,
    nfp: int,
    mpol: int,
    ntor: int,
    ntheta: int,
    nzeta: int,
    mboz: int,
    nboz: int,
    asym: bool,
) -> tuple[Any, Any]:
    """Prepare Boozer constants outside traced VMEC-JAX residual callbacks."""

    modes_mod = importlib.import_module("vmec_jax.modes")
    bx = importlib.import_module("booz_xform_jax.jax_api")
    main_modes = modes_mod.vmec_mode_table(int(mpol), int(ntor))
    nyq_modes = modes_mod.nyquist_mode_table_from_grid(
        mpol=int(mpol),
        ntor=int(ntor),
        ntheta=int(ntheta),
        nzeta=int(nzeta),
    )
    return bx.prepare_booz_xform_constants(
        nfp=int(nfp),
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(asym),
        xm=np.asarray(main_modes.m, dtype=np.int32),
        xn=np.asarray(main_modes.n * int(nfp), dtype=np.int32),
        xm_nyq=np.asarray(nyq_modes.m, dtype=np.int32),
        xn_nyq=np.asarray(nyq_modes.n * int(nfp), dtype=np.int32),
    )


def prewarm_vmec_boozer_equal_arc_cache(
    static: Any,
    wout: Any,
    *,
    mboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    nboz: int = _VMEC_BOOZER_PARITY_MIN_MODE_COUNT,
    asym: bool | None = None,
) -> None:  # pragma: no cover - exercised by optional VMEC-JAX optimizer smoke tests.
    """Precompute Boozer constants before VMEC-JAX jits residual callbacks."""

    cfg = static.cfg
    nfp_raw = getattr(wout, "nfp", None)
    if nfp_raw is None:
        nfp_raw = getattr(cfg, "nfp", 1)
    nfp_int = 1 if nfp_raw is None else int(nfp_raw)
    _cached_booz_xform_constants(
        nfp=nfp_int,
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(getattr(cfg, "lasym", False) if asym is None else asym),
    )


def vmec_jax_boozer_equal_arc_core_profiles_from_state(  # pragma: no cover
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
) -> dict[str, Any]:
    """Return Boozer equal-arc core profiles from a real ``vmec_jax`` state.

    This bridge follows the same high-level convention as the imported VMEC/EIK
    runtime path for scalar/core field-line quantities and the zero-beta Boozer
    metric/drift terms that can be reconstructed directly from
    ``booz_xform_jax`` output: Boozer ``|B|``, equal-arc constant ``gradpar``,
    ``q``, magnetic shear, solver Jacobian normalization, ``gds*``/``grho``,
    and loaded-convention ``cvdrift``/``gbdrift`` coefficients.  General
    finite-beta pressure corrections and broader-equilibrium drift gates remain
    separate promotion steps.
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
    if not (
        info.get("vmec_jax_available", False)
        and info.get("booz_xform_jax_api_available", False)
    ):
        raise RuntimeError("vmec_jax and booz_xform_jax functional APIs are required")

    booz_input_mod = importlib.import_module("vmec_jax.booz_input")
    bx = importlib.import_module("booz_xform_jax.jax_api")

    base_Rcos = jnp.asarray(state.Rcos)
    if base_Rcos.ndim != 2:
        raise RuntimeError("vmec_jax state Rcos array must be two-dimensional")
    ns_full = int(base_Rcos.shape[0])
    if ns_full < 3:
        raise RuntimeError("vmec_jax state needs at least three radial surfaces")

    sidx = (
        max(1, min(ns_full // 2, ns_full - 2))
        if surface_index is None
        else int(surface_index)
    )
    if not (0 < sidx < ns_full - 1):
        raise ValueError("surface_index must be an interior VMEC radial index")
    s_value = (
        float(sidx) / float(max(ns_full - 1, 1)) if torflux is None else float(torflux)
    )
    if not (0.0 < s_value < 1.0):
        raise ValueError("torflux must lie inside (0, 1)")
    if surface_stencil_width is not None and int(surface_stencil_width) < 3:
        raise ValueError("surface_stencil_width must be >= 3 when provided")

    raw_length = (
        float(getattr(wout, "Aminor_p", 1.0))
        if reference_length is None
        else float(reference_length)
    )
    L_reference = (
        raw_length if np.isfinite(raw_length) and abs(raw_length) > 0.0 else 1.0
    )
    if reference_b is None:
        phi_profile = np.asarray(getattr(wout, "phi", [0.0, np.pi]), dtype=float)
        edge_toroidal_flux_over_2pi = -float(phi_profile[-1]) / (2.0 * np.pi)
        raw_b = 2.0 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)
        B_reference = raw_b if np.isfinite(raw_b) and abs(raw_b) > 0.0 else 1.0
    else:
        edge_toroidal_flux_over_2pi = -float(
            np.asarray(getattr(wout, "phi", [0.0, np.pi]))[-1]
        ) / (2.0 * np.pi)
        B_reference = float(reference_b)
    B_reference = (
        B_reference if np.isfinite(B_reference) and abs(B_reference) > 0.0 else 1.0
    )

    inputs = booz_input_mod.booz_xform_inputs_from_state(
        state=state,
        static=static,
        indata=indata,
        signgs=getattr(wout, "signgs", 1),
    )
    asym = bool(getattr(inputs, "bmns", None) is not None)
    cfg = getattr(static, "cfg", SimpleNamespace())
    nfp_raw = getattr(wout, "nfp", None)
    if nfp_raw is None:
        nfp_raw = getattr(cfg, "nfp", 1)
    nfp_int = 1 if nfp_raw is None else int(nfp_raw)
    try:
        constants, grids = _cached_booz_xform_constants(
            nfp=nfp_int,
            mpol=int(getattr(cfg, "mpol", max(2, base_Rcos.shape[1]))),
            ntor=int(getattr(cfg, "ntor", max(1, base_Rcos.shape[1] - 1))),
            ntheta=int(getattr(cfg, "ntheta", max(16, ntheta_int))),
            nzeta=int(getattr(cfg, "nzeta", max(16, 2 * ntheta_int))),
            mboz=mboz_int,
            nboz=nboz_int,
            asym=asym,
        )
    except (AttributeError, ModuleNotFoundError):
        constants, grids = bx.prepare_booz_xform_constants_from_inputs(
            inputs=inputs,
            mboz=mboz_int,
            nboz=nboz_int,
            asym=asym,
        )
    surface_indices = None
    if surface_stencil_width is not None:
        ns_b_est = max(1, ns_full - 1)
        width = min(int(surface_stencil_width), ns_b_est)
        center = int(round(s_value * float(ns_b_est) - 0.5))
        half_width = width // 2
        start = max(0, min(center - half_width, ns_b_est - width))
        surface_indices = jnp.arange(start, start + width, dtype=jnp.int32)
    out = bx.booz_xform_from_inputs(
        inputs=inputs,
        constants=constants,
        grids=grids,
        surface_indices=surface_indices,
        jit=bool(jit),
    )

    bmnc_b_all = jnp.asarray(out["bmnc_b"], dtype=base_Rcos.dtype)
    if bmnc_b_all.ndim != 2:
        raise RuntimeError(
            "booz_xform_jax bmnc_b output must have shape (surface, mode)"
        )
    ns_b = int(bmnc_b_all.shape[0])
    if ns_b < 2:
        raise RuntimeError("booz_xform_jax output needs at least two radial surfaces")
    ns_b_full = max(int(ns_full) - 1, int(ns_b))
    s_half = _boozer_half_mesh_s_grid(
        out.get("jlist"),
        ns_b=ns_b,
        ns_b_full=ns_b_full,
        dtype=base_Rcos.dtype,
    )

    radial_spacing = 1.0 / float(max(ns_b_full, 1))
    bmnc_b = _interp_radial(bmnc_b_all, s_half, s_value)
    rmnc_b = _interp_radial(
        jnp.asarray(out["rmnc_b"], dtype=base_Rcos.dtype), s_half, s_value
    )
    zmns_b = _interp_radial(
        jnp.asarray(out["zmns_b"], dtype=base_Rcos.dtype), s_half, s_value
    )
    numns_b = -_interp_radial(
        jnp.asarray(out["pmns_b"], dtype=base_Rcos.dtype), s_half, s_value
    )
    d_bmnc_b_d_s = _interp_radial(
        _radial_derivative_array(
            jnp.asarray(out["bmnc_b"], dtype=base_Rcos.dtype), radial_spacing
        ),
        s_half,
        s_value,
    )
    d_rmnc_b_d_s = _interp_radial(
        _radial_derivative_array(
            jnp.asarray(out["rmnc_b"], dtype=base_Rcos.dtype), radial_spacing
        ),
        s_half,
        s_value,
    )
    d_zmns_b_d_s = _interp_radial(
        _radial_derivative_array(
            jnp.asarray(out["zmns_b"], dtype=base_Rcos.dtype), radial_spacing
        ),
        s_half,
        s_value,
    )
    d_numns_b_d_s = -_interp_radial(
        _radial_derivative_array(
            jnp.asarray(out["pmns_b"], dtype=base_Rcos.dtype), radial_spacing
        ),
        s_half,
        s_value,
    )
    iota_profile = jnp.asarray(out["iota_b"], dtype=base_Rcos.dtype)
    iota = _interp_radial(iota_profile, s_half, s_value)
    d_iota_ds = _interp_radial(
        _radial_derivative_profile(iota_profile, radial_spacing), s_half, s_value
    )
    iota_safe = jnp.where(
        jnp.abs(iota) < 1.0e-12, jnp.sign(iota + 1.0e-30) * 1.0e-12, iota
    )
    s_hat = -2.0 * jnp.asarray(s_value, dtype=base_Rcos.dtype) * d_iota_ds / iota_safe

    boozer_i = _interp_radial(
        jnp.asarray(out["buco_b"], dtype=base_Rcos.dtype), s_half, s_value
    )
    boozer_g = _interp_radial(
        jnp.asarray(out["bvco_b"], dtype=base_Rcos.dtype), s_half, s_value
    )

    theta_closed = jnp.linspace(-jnp.pi, jnp.pi, ntheta_int + 1, dtype=base_Rcos.dtype)
    mod_b, _dmod_b_dtheta = _evaluate_boozer_cosine_series_on_field_line(
        theta_closed,
        coeffs=bmnc_b,
        ixm_b=jnp.asarray(out["ixm_b"]),
        ixn_b=jnp.asarray(out["ixn_b"]),
        iota=iota_safe,
        alpha=float(alpha),
    )
    mod_b_safe = jnp.maximum(
        jnp.abs(mod_b), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
    )
    sqrt_g_booz = (boozer_g + iota_safe * boozer_i) / (mod_b_safe * mod_b_safe)
    gradpar_raw = jnp.abs(
        jnp.asarray(float(L_reference), dtype=base_Rcos.dtype)
        * iota_safe
        / jnp.maximum(
            jnp.abs(mod_b_safe * sqrt_g_booz),
            jnp.asarray(1.0e-30, dtype=base_Rcos.dtype),
        )
    )
    inv_gradpar_int = _cumulative_trapezoid(1.0 / gradpar_raw, theta_closed)
    gradpar_eqarc = (
        2.0
        * jnp.pi
        / jnp.maximum(inv_gradpar_int[-1], jnp.asarray(1.0e-30, dtype=base_Rcos.dtype))
    )
    theta_eqarc = gradpar_eqarc * inv_gradpar_int - jnp.pi
    theta_uniform_closed = jnp.linspace(
        -jnp.pi, jnp.pi, ntheta_int + 1, dtype=base_Rcos.dtype
    )
    bmag_closed = jnp.asarray(
        _interp_equal_arc_profile(
            theta_uniform_closed,
            theta_eqarc,
            mod_b_safe / float(B_reference),
        )
    )
    theta = theta_uniform_closed[:-1]
    bmag = bmag_closed[:-1]
    gradpar = gradpar_eqarc * jnp.ones_like(theta)
    dtheta = 2.0 * jnp.pi / float(ntheta_int)
    bmag_safe = jnp.maximum(jnp.abs(bmag), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype))
    wave_number = 2.0 * jnp.pi * jnp.fft.fftfreq(ntheta_int, d=float(dtheta))
    dbmag_dtheta = jnp.real(jnp.fft.ifft(1j * wave_number * jnp.fft.fft(bmag)))
    bgrad = gradpar_eqarc * dbmag_dtheta / bmag_safe

    dpsidrho = (
        2.0
        * jnp.sqrt(jnp.asarray(s_value, dtype=base_Rcos.dtype))
        * jnp.asarray(
            edge_toroidal_flux_over_2pi,
            dtype=base_Rcos.dtype,
        )
    )
    drhodpsi = 1.0 / jnp.maximum(
        jnp.abs(dpsidrho), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
    )
    jacobian = 1.0 / jnp.maximum(
        jnp.abs(drhodpsi * gradpar_eqarc * bmag_safe),
        jnp.asarray(1.0e-30, dtype=base_Rcos.dtype),
    )

    m = jnp.asarray(out["ixm_b"], dtype=base_Rcos.dtype)
    n = jnp.asarray(out["ixn_b"], dtype=base_Rcos.dtype)
    phi_b = (
        theta_closed - jnp.asarray(float(alpha), dtype=base_Rcos.dtype)
    ) / iota_safe
    phase = m[:, None] * theta_closed[None, :] - n[:, None] * phi_b[None, :]
    cos_phase = jnp.cos(phase)
    sin_phase = jnp.sin(phase)
    m_cos = m[:, None] * cos_phase
    m_sin = m[:, None] * sin_phase
    n_cos = n[:, None] * cos_phase
    n_sin = n[:, None] * sin_phase

    r_b = jnp.sum(rmnc_b[:, None] * cos_phase, axis=0)
    d_mod_b_d_s = jnp.sum(d_bmnc_b_d_s[:, None] * cos_phase, axis=0)
    d_mod_b_d_theta = -jnp.sum(bmnc_b[:, None] * m_sin, axis=0)
    d_mod_b_d_phi = jnp.sum(bmnc_b[:, None] * n_sin, axis=0)
    d_r_b_d_s = jnp.sum(d_rmnc_b_d_s[:, None] * cos_phase, axis=0)
    d_r_b_d_theta = -jnp.sum(rmnc_b[:, None] * m_sin, axis=0)
    d_r_b_d_phi = jnp.sum(rmnc_b[:, None] * n_sin, axis=0)
    d_z_b_d_s = jnp.sum(d_zmns_b_d_s[:, None] * sin_phase, axis=0)
    d_z_b_d_theta = jnp.sum(zmns_b[:, None] * m_cos, axis=0)
    d_z_b_d_phi = -jnp.sum(zmns_b[:, None] * n_cos, axis=0)
    nu_b = jnp.sum(numns_b[:, None] * sin_phase, axis=0)
    d_nu_b_d_s = jnp.sum(d_numns_b_d_s[:, None] * sin_phase, axis=0)
    d_nu_b_d_theta = jnp.sum(numns_b[:, None] * m_cos, axis=0)
    d_nu_b_d_phi = -jnp.sum(numns_b[:, None] * n_cos, axis=0)

    phi_cyl = phi_b - nu_b
    sin_phi = jnp.sin(phi_cyl)
    cos_phi = jnp.cos(phi_cyl)
    d_x_d_theta = d_r_b_d_theta * cos_phi - r_b * sin_phi * (-d_nu_b_d_theta)
    d_x_d_phi = d_r_b_d_phi * cos_phi - r_b * sin_phi * (1.0 - d_nu_b_d_phi)
    d_x_d_s = d_r_b_d_s * cos_phi - r_b * sin_phi * (-d_nu_b_d_s)
    d_y_d_theta = d_r_b_d_theta * sin_phi + r_b * cos_phi * (-d_nu_b_d_theta)
    d_y_d_phi = d_r_b_d_phi * sin_phi + r_b * cos_phi * (1.0 - d_nu_b_d_phi)
    d_y_d_s = d_r_b_d_s * sin_phi + r_b * cos_phi * (-d_nu_b_d_s)

    grad_psi_x = (d_y_d_theta * d_z_b_d_phi - d_z_b_d_theta * d_y_d_phi) / sqrt_g_booz
    grad_psi_y = (d_z_b_d_theta * d_x_d_phi - d_x_d_theta * d_z_b_d_phi) / sqrt_g_booz
    grad_psi_z = (d_x_d_theta * d_y_d_phi - d_y_d_theta * d_x_d_phi) / sqrt_g_booz
    g_sup_psi_psi = grad_psi_x**2 + grad_psi_y**2 + grad_psi_z**2
    g_sup_psi_psi_safe = jnp.maximum(
        g_sup_psi_psi, jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
    )

    etf = jnp.asarray(edge_toroidal_flux_over_2pi, dtype=base_Rcos.dtype)
    grad_theta_x = (d_y_d_phi * d_z_b_d_s - d_z_b_d_phi * d_y_d_s) / (sqrt_g_booz * etf)
    grad_theta_y = (d_z_b_d_phi * d_x_d_s - d_x_d_phi * d_z_b_d_s) / (sqrt_g_booz * etf)
    grad_theta_z = (d_x_d_phi * d_y_d_s - d_y_d_phi * d_x_d_s) / (sqrt_g_booz * etf)
    grad_phi_x = (d_y_d_s * d_z_b_d_theta - d_z_b_d_s * d_y_d_theta) / (
        sqrt_g_booz * etf
    )
    grad_phi_y = (d_z_b_d_s * d_x_d_theta - d_x_d_s * d_z_b_d_theta) / (
        sqrt_g_booz * etf
    )
    grad_phi_z = (d_x_d_s * d_y_d_theta - d_y_d_s * d_x_d_theta) / (sqrt_g_booz * etf)
    zeta_center = -jnp.asarray(float(alpha), dtype=base_Rcos.dtype) / iota_safe
    shear_phase = phi_b - zeta_center
    grad_alpha_x = (
        -shear_phase * d_iota_ds * grad_psi_x / etf
        + grad_theta_x
        - iota_safe * grad_phi_x
    )
    grad_alpha_y = (
        -shear_phase * d_iota_ds * grad_psi_y / etf
        + grad_theta_y
        - iota_safe * grad_phi_y
    )
    grad_alpha_z = (
        -shear_phase * d_iota_ds * grad_psi_z / etf
        + grad_theta_z
        - iota_safe * grad_phi_z
    )
    grad_alpha_dot_grad_psi = (
        grad_alpha_x * grad_psi_x
        + grad_alpha_y * grad_psi_y
        + grad_alpha_z * grad_psi_z
    )
    local_shear_l1 = grad_alpha_dot_grad_psi / g_sup_psi_psi_safe
    s_arr = jnp.asarray(s_value, dtype=base_Rcos.dtype)
    L = jnp.asarray(float(L_reference), dtype=base_Rcos.dtype)
    Bref = jnp.asarray(float(B_reference), dtype=base_Rcos.dtype)
    shat_metric = s_hat
    metric_bmag_sq = mod_b_safe * mod_b_safe
    gds2_raw = (
        (metric_bmag_sq / g_sup_psi_psi_safe + g_sup_psi_psi_safe * local_shear_l1**2)
        * L
        * L
        * s_arr
    )
    gds21_raw = g_sup_psi_psi_safe * local_shear_l1 * shat_metric / Bref
    gds22_raw = (
        g_sup_psi_psi_safe * shat_metric * shat_metric / (L * L * Bref * Bref * s_arr)
    )
    grho_raw = jnp.sqrt(g_sup_psi_psi_safe / (L * L * Bref * Bref * s_arr))

    boozer_current_sum = boozer_g + iota_safe * boozer_i
    d_sqrt_g_booz_d_theta = (
        -2.0 * boozer_current_sum * d_mod_b_d_theta / (mod_b_safe**3)
    )
    d_sqrt_g_booz_d_phi = -2.0 * boozer_current_sum * d_mod_b_d_phi / (mod_b_safe**3)
    curvature_numerator = (
        boozer_g * d_sqrt_g_booz_d_theta - boozer_i * d_sqrt_g_booz_d_phi
    )
    curvature_denom = 2.0 * sqrt_g_booz * boozer_current_sum
    eps = jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)
    curvature_denom_safe = jnp.where(
        jnp.abs(curvature_denom) < eps,
        jnp.sign(curvature_denom + eps) * eps,
        curvature_denom,
    )
    etf_safe = jnp.where(jnp.abs(etf) < eps, jnp.sign(etf + eps) * eps, etf)
    kappa_g = curvature_numerator / curvature_denom_safe
    local_shear_l0 = -(local_shear_l1 + d_iota_ds / etf_safe * shear_phase)
    kappa_n = d_mod_b_d_s / (mod_b_safe * etf_safe) + local_shear_l0 * kappa_g
    b_cross_kappa_dot_grad_alpha = (kappa_n + kappa_g * local_shear_l1) * metric_bmag_sq
    b_cross_kappa_dot_grad_psi = kappa_g * metric_bmag_sq
    toroidal_flux_sign = jnp.sign(etf)
    sqrt_s = jnp.sqrt(s_arr)
    drift_cvdrift0_raw = (
        -b_cross_kappa_dot_grad_psi
        * 2.0
        * shat_metric
        / jnp.maximum(metric_bmag_sq * sqrt_s, eps)
        * toroidal_flux_sign
    )
    drift_cvdrift_raw = (
        -2.0
        * Bref
        * L
        * L
        * sqrt_s
        * b_cross_kappa_dot_grad_alpha
        / metric_bmag_sq
        * toroidal_flux_sign
    )
    # Root-level VMEC/EIK drift coefficients are stored at the pre-loader (2x)
    # level; SPECTRAX-GK compares against the loaded solver convention.
    drift_loader_factor = jnp.asarray(0.5, dtype=base_Rcos.dtype)
    gds2 = _interp_equal_arc_profile(theta_uniform_closed, theta_eqarc, gds2_raw)[:-1]
    gds21 = _interp_equal_arc_profile(theta_uniform_closed, theta_eqarc, gds21_raw)[:-1]
    gds22 = _interp_equal_arc_profile(theta_uniform_closed, theta_eqarc, gds22_raw)[:-1]
    grho = _interp_equal_arc_profile(theta_uniform_closed, theta_eqarc, grho_raw)[:-1]
    cvdrift = (
        drift_loader_factor
        * _interp_equal_arc_profile(
            theta_uniform_closed, theta_eqarc, drift_cvdrift_raw
        )[:-1]
    )
    gbdrift = cvdrift
    cvdrift0 = (
        drift_loader_factor
        * _interp_equal_arc_profile(
            theta_uniform_closed, theta_eqarc, drift_cvdrift0_raw
        )[:-1]
    )
    gbdrift0 = cvdrift0

    return {
        "theta": theta,
        "theta_equal_arc_closed": theta_eqarc,
        "theta_uniform_closed": theta_uniform_closed,
        "gradpar": gradpar,
        "bmag": bmag,
        "bgrad": bgrad,
        "jacobian": jacobian,
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "cvdrift": cvdrift,
        "gbdrift": gbdrift,
        "cvdrift0": cvdrift0,
        "gbdrift0": gbdrift0,
        "grho": grho,
        "q": 1.0
        / jnp.maximum(jnp.abs(iota_safe), jnp.asarray(1.0e-30, dtype=base_Rcos.dtype)),
        "s_hat": s_hat,
        "iota": iota,
        "torflux": float(s_value),
        "surface_index": int(sidx),
        "reference_length": float(L_reference),
        "reference_b": float(B_reference),
        "mboz": mboz_int,
        "nboz": nboz_int,
        "surface_stencil_width": None
        if surface_stencil_width is None
        else int(surface_stencil_width),
        "boozer_surface_indices": None
        if surface_indices is None
        else [int(x) for x in np.asarray(surface_indices)],
        "field_line_convention": "Boozer theta, alpha=theta-iota*zeta, equal-arc remap",
        "scope": (
            "Boozer equal-arc bmag/gradpar/Jacobian plus zero-beta metric/drift parity; "
            "finite-beta pressure corrections and broad-equilibrium drift gates remain open"
        ),
    }


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

    p = jnp.asarray([1.0e-4, 1.0e-4] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")

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
        driver = importlib.import_module("vmec_jax.driver")
        config_mod = importlib.import_module("vmec_jax.config")
        static_mod = importlib.import_module("vmec_jax.static")
        wout_mod = importlib.import_module("vmec_jax.wout")

        input_path, wout_path = driver.example_paths(str(case_name))
        if wout_path is None:
            raise RuntimeError(
                f"vmec_jax example {case_name!r} has no bundled wout reference"
            )

        cfg, _indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        base_Rcos = jnp.asarray(state.Rcos)
        base_Zsin = jnp.asarray(state.Zsin)
        if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
            raise RuntimeError(
                "vmec_jax state Rcos/Zsin arrays must be two-dimensional"
            )

        ridx = (
            int(base_Rcos.shape[0] // 2) if radial_index is None else int(radial_index)
        )
        midx = int(mode_index)
        if not (0 <= ridx < int(base_Rcos.shape[0])):
            raise ValueError("radial_index is outside the VMEC state radial grid")
        if not (0 <= midx < int(base_Rcos.shape[1])):
            raise ValueError("mode_index is outside the VMEC state mode table")

        def mapping_fn(x: jnp.ndarray) -> dict[str, Any]:
            traced_state = dc_replace(
                state,
                Rcos=base_Rcos.at[ridx, midx].add(x[0]),
                Zsin=base_Zsin.at[ridx, midx].add(x[1]),
            )
            return vmec_jax_flux_tube_mapping_from_state(
                traced_state,
                static,
                wout,
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
        "input_path": str(input_path),
        "wout_path": str(wout_path),
        "source_model": "vmec_jax:state->tensor-flux-tube",
        "param_names": ["delta_Rcos", "delta_Zsin"],
        "params": np.asarray(p).tolist(),
        "radial_index": int(ridx),
        "mode_index": int(midx),
        "surface_index": int(vmec_meta["surface_index"]),
        "iota": float(np.asarray(vmec_meta["iota"])),
        "alpha": float(alpha),
        "ntheta": int(ntheta),
        "state_shape": [int(base_Rcos.shape[0]), int(base_Rcos.shape[1])],
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
        from spectraxgk.from_gx.vmec import (
            generate_vmec_eik_internal,
            internal_vmec_backend_available,
        )
        from spectraxgk.geometry import load_gx_geometry_netcdf

        if not internal_vmec_backend_available():
            return {
                "available": False,
                "backend_info": info,
                "case_name": str(case_name),
                "reason": "internal VMEC/EIK backend is not available",
            }

        driver = importlib.import_module("vmec_jax.driver")
        config_mod = importlib.import_module("vmec_jax.config")
        static_mod = importlib.import_module("vmec_jax.static")
        wout_mod = importlib.import_module("vmec_jax.wout")

        input_path, wout_path = driver.example_paths(str(case_name))
        if wout_path is None:
            raise RuntimeError(
                f"vmec_jax example {case_name!r} has no bundled wout reference"
            )

        cfg, indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        ns = int(jnp.asarray(state.Rcos).shape[0])
        sidx = (
            max(1, min(ns // 2, ns - 2))
            if surface_index is None
            else int(surface_index)
        )
        torflux = float(sidx) / float(max(ns - 1, 1))
        direct_mapping = vmec_jax_flux_tube_mapping_from_state(
            state,
            static,
            wout,
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
            vmec_file=str(wout_path),
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
            imported = load_gx_geometry_netcdf(eik_path)
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
        array_metrics = {
            name: _array_parity_metrics(candidate, reference)
            for name, (candidate, reference) in array_pairs.items()
        }
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
        core_values: list[float] = []
        for name in core_names:
            metrics = array_metrics[name]
            if not bool(metrics.get("shape_match", False)):
                continue
            raw_value = metrics.get("normalized_max_abs")
            core_values.append(
                float(raw_value)
                if isinstance(raw_value, int | float | np.floating)
                else np.inf
            )
        worst_core = max(core_values) if core_values else np.inf
        worst_scalar = max(float(values["rel"]) for values in scalar_metrics.values())
        production_parity_passed = bool(
            worst_core <= float(core_tolerance)
            and worst_scalar <= float(scalar_tolerance)
        )

        equal_arc_core_error = None
        equal_arc_core_array_metrics: dict[str, dict[str, object]] = {}
        equal_arc_metric_array_metrics: dict[str, dict[str, object]] = {}
        equal_arc_drift_array_metrics: dict[str, dict[str, object]] = {}
        equal_arc_core_scalar_metrics: dict[str, dict[str, float]] = {}
        equal_arc_core_worst = np.inf
        equal_arc_core_worst_scalar = np.inf
        equal_arc_derivative_worst = np.inf
        equal_arc_metric_worst = np.inf
        equal_arc_drift_worst = np.inf
        equal_arc_core_passed = False
        equal_arc_derivative_passed = False
        equal_arc_metric_passed = False
        equal_arc_drift_passed = False
        if bool(info.get("booz_xform_jax_api_available", False)):
            try:
                equal_arc_core = vmec_jax_boozer_equal_arc_core_profiles_from_state(
                    state,
                    static,
                    indata,
                    wout,
                    surface_index=sidx,
                    torflux=torflux,
                    alpha=float(alpha),
                    ntheta=ntheta_int,
                    mboz=mboz_int,
                    nboz=nboz_int,
                    jit=False,
                )
                equal_arc_core_pairs = {
                    "theta": (equal_arc_core["theta"], imported.theta),
                    "bmag": (equal_arc_core["bmag"], imported.bmag_profile),
                    "bgrad": (equal_arc_core["bgrad"], imported.bgrad_profile),
                    "jacobian": (equal_arc_core["jacobian"], imported.jacobian_profile),
                }
                equal_arc_metric_pairs = {
                    "gds2": (equal_arc_core["gds2"], imported.gds2_profile),
                    "gds21": (equal_arc_core["gds21"], imported.gds21_profile),
                    "gds22": (equal_arc_core["gds22"], imported.gds22_profile),
                    "grho": (equal_arc_core["grho"], imported.grho_profile),
                }
                equal_arc_drift_pairs = {
                    "cvdrift": (equal_arc_core["cvdrift"], imported.cv_profile),
                    "gbdrift": (equal_arc_core["gbdrift"], imported.gb_profile),
                    "cvdrift0": (equal_arc_core["cvdrift0"], imported.cv0_profile),
                    "gbdrift0": (equal_arc_core["gbdrift0"], imported.gb0_profile),
                }
                equal_arc_core_array_metrics = {
                    name: _array_parity_metrics(candidate, reference)
                    for name, (candidate, reference) in equal_arc_core_pairs.items()
                }
                equal_arc_metric_array_metrics = {
                    name: _array_parity_metrics(candidate, reference)
                    for name, (candidate, reference) in equal_arc_metric_pairs.items()
                }
                equal_arc_drift_array_metrics = {
                    name: _array_parity_metrics(candidate, reference)
                    for name, (candidate, reference) in equal_arc_drift_pairs.items()
                }
                equal_arc_core_scalar_metrics = {
                    "gradpar": _scalar_parity_metrics(
                        jnp.asarray(equal_arc_core["gradpar"])[0],
                        imported.gradpar_value,
                    ),
                    "q": _scalar_parity_metrics(equal_arc_core["q"], imported.q),
                    "s_hat": _scalar_parity_metrics(
                        equal_arc_core["s_hat"], imported.s_hat
                    ),
                }
                equal_arc_core_names = ("theta", "bmag", "jacobian")
                equal_arc_values: list[float] = []
                for name in equal_arc_core_names:
                    metrics = equal_arc_core_array_metrics[name]
                    if not bool(metrics.get("shape_match", False)):
                        continue
                    raw_value = metrics.get("normalized_max_abs")
                    equal_arc_values.append(
                        float(raw_value)
                        if isinstance(raw_value, int | float | np.floating)
                        else np.inf
                    )
                equal_arc_core_worst = (
                    max(equal_arc_values) if equal_arc_values else np.inf
                )
                equal_arc_core_worst_scalar = max(
                    float(values["rel"])
                    for values in equal_arc_core_scalar_metrics.values()
                )
                derivative_metrics = equal_arc_core_array_metrics["bgrad"]
                raw_derivative = derivative_metrics.get("normalized_max_abs")
                equal_arc_derivative_worst = (
                    float(raw_derivative)
                    if isinstance(raw_derivative, int | float | np.floating)
                    else np.inf
                )
                equal_arc_core_passed = bool(
                    equal_arc_core_worst <= float(equal_arc_core_tolerance)
                    and equal_arc_core_worst_scalar <= float(equal_arc_core_tolerance)
                )
                equal_arc_derivative_passed = bool(
                    equal_arc_derivative_worst <= float(equal_arc_derivative_tolerance)
                )
                metric_values: list[float] = []
                for metrics in equal_arc_metric_array_metrics.values():
                    if not bool(metrics.get("shape_match", False)):
                        continue
                    raw_value = metrics.get("normalized_max_abs")
                    metric_values.append(
                        float(raw_value)
                        if isinstance(raw_value, int | float | np.floating)
                        else np.inf
                    )
                equal_arc_metric_worst = max(metric_values) if metric_values else np.inf
                equal_arc_metric_passed = bool(
                    equal_arc_metric_worst <= float(equal_arc_metric_tolerance)
                )
                drift_values: list[float] = []
                for metrics in equal_arc_drift_array_metrics.values():
                    if not bool(metrics.get("shape_match", False)):
                        continue
                    raw_value = metrics.get("normalized_max_abs")
                    drift_values.append(
                        float(raw_value)
                        if isinstance(raw_value, int | float | np.floating)
                        else np.inf
                    )
                equal_arc_drift_worst = max(drift_values) if drift_values else np.inf
                equal_arc_drift_passed = bool(
                    equal_arc_drift_worst <= float(equal_arc_drift_tolerance)
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - optional-backend diagnostic detail
                equal_arc_core_error = f"{type(exc).__name__}: {exc}"
        else:
            equal_arc_core_error = "booz_xform_jax functional API is not available"
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
        "input_path": str(input_path),
        "wout_path": str(wout_path),
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
        "equal_arc_core_array_metrics": equal_arc_core_array_metrics,
        "equal_arc_metric_array_metrics": equal_arc_metric_array_metrics,
        "equal_arc_drift_array_metrics": equal_arc_drift_array_metrics,
        "equal_arc_core_scalar_metrics": equal_arc_core_scalar_metrics,
        "equal_arc_core_worst_normalized_max_abs": float(equal_arc_core_worst),
        "equal_arc_core_worst_scalar_rel": float(equal_arc_core_worst_scalar),
        "equal_arc_derivative_worst_normalized_max_abs": float(
            equal_arc_derivative_worst
        ),
        "equal_arc_metric_worst_normalized_max_abs": float(equal_arc_metric_worst),
        "equal_arc_drift_worst_normalized_max_abs": float(equal_arc_drift_worst),
        "equal_arc_core_tolerance": float(equal_arc_core_tolerance),
        "equal_arc_derivative_tolerance": float(equal_arc_derivative_tolerance),
        "equal_arc_metric_tolerance": float(equal_arc_metric_tolerance),
        "equal_arc_drift_tolerance": float(equal_arc_drift_tolerance),
        "equal_arc_core_passed": bool(equal_arc_core_passed),
        "equal_arc_derivative_passed": bool(equal_arc_derivative_passed),
        "equal_arc_metric_passed": bool(equal_arc_metric_passed),
        "equal_arc_drift_passed": bool(equal_arc_drift_passed),
        "equal_arc_core_error": equal_arc_core_error,
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
