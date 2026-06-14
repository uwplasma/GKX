"""VMEC-state differentiable sensitivity reports."""

from __future__ import annotations

import importlib
from dataclasses import replace as dc_replace
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
from spectraxgk.geometry.numerics import _periodic_bilinear_sample_2d
from spectraxgk.geometry.sensitivity import geometry_sensitivity_report


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


__all__ = [
    "vmec_jax_boozer_flux_tube_sensitivity_report",
    "vmec_jax_field_line_tensor_sensitivity_report",
    "vmec_jax_metric_tensor_sensitivity_report",
]
