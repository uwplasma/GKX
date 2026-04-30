"""Differentiable geometry bridge contracts for VMEC/JAX pipelines."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace as dc_replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.autodiff_validation import covariance_diagnostics
from spectraxgk.geometry import FluxTubeGeometryData


_ARRAY_FIELDS = (
    "theta",
    "gradpar",
    "bmag",
    "bgrad",
    "gds2",
    "gds21",
    "gds22",
    "cvdrift",
    "gbdrift",
    "cvdrift0",
    "gbdrift0",
)
_GEOMETRY_OBSERVABLE_NAMES = (
    "mean_bmag",
    "relative_bmag_ripple",
    "metric_frobenius_rms",
    "drift_rms",
    "mean_jacobian",
    "mean_gradpar",
)
_VMEC_METRIC_OBSERVABLE_NAMES = (
    "sqrtg_rms",
    "mean_g_ss",
    "mean_g_tt",
    "mean_g_pp",
    "g_st_rms",
    "g_sp_rms",
    "g_tp_rms",
)
_VMEC_FIELD_LINE_OBSERVABLE_NAMES = (
    "mean_bmag",
    "relative_bmag_ripple",
    "sqrtg_rms",
    "mean_g_tt",
    "mean_g_pp",
    "g_tp_rms",
    "mean_g_ss",
)


def _candidate_paths(env_names: Sequence[str], defaults: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for name in env_names:
        raw = os.environ.get(name)
        if raw:
            paths.append(Path(os.path.expandvars(raw)).expanduser())
    paths.extend(defaults)

    out: list[Path] = []
    seen: set[Path] = set()
    for base in paths:
        for candidate in (base, base / "src"):
            resolved = candidate.resolve(strict=False)
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)
            out.append(resolved)
    return out


def _find_importable_module(name: str, paths: Sequence[Path]) -> Any | None:
    import sys

    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    try:
        return importlib.import_module(name)
    except Exception:
        pass
    return None


def _is_traced(value: Any) -> bool:
    """Return true when host NumPy validation would break JAX tracing."""

    if isinstance(value, jax.core.Tracer):
        return True
    if isinstance(value, (tuple, list)):
        return any(_is_traced(item) for item in value)
    if isinstance(value, Mapping):
        return any(_is_traced(item) for item in value.values())
    return False


def discover_differentiable_geometry_backends() -> dict[str, object]:
    """Discover optional ``vmec_jax`` and ``booz_xform_jax`` bridge APIs."""

    repo_parent = Path(__file__).resolve().parents[3].parent
    home = Path.home()
    vmec_paths = _candidate_paths(
        ("SPECTRAX_VMEC_JAX_PATH", "VMEC_JAX_PATH"),
        (
            repo_parent / "vmec_jax",
            home / "vmec_jax",
            home / "local" / "vmec_jax",
        ),
    )
    booz_paths = _candidate_paths(
        ("SPECTRAX_BOOZ_XFORM_JAX_PATH", "BOOZ_XFORM_JAX_PATH"),
        (
            repo_parent / "booz_xform_jax",
            home / "booz_xform_jax",
            home / "local" / "booz_xform_jax",
        ),
    )
    vmec = _find_importable_module("vmec_jax", vmec_paths)
    booz = _find_importable_module("booz_xform_jax", booz_paths)
    booz_jax_api = None if booz is None else _find_importable_module("booz_xform_jax.jax_api", booz_paths)

    vmec_boundary_api = vmec is not None and all(
        hasattr(vmec, name)
        for name in (
            "BoundaryCoeffs",
            "boundary_aspect_ratio",
            "build_helical_basis",
            "make_angle_grid",
            "vmec_mode_table",
        )
    )
    booz_api = (
        booz_jax_api is not None
        and hasattr(booz_jax_api, "prepare_booz_xform_constants_from_inputs")
        and hasattr(booz_jax_api, "booz_xform_from_inputs")
        and hasattr(booz_jax_api, "booz_xform_jax_impl")
    )

    return {
        "vmec_jax_available": vmec is not None,
        "vmec_jax_boundary_api_available": vmec_boundary_api,
        "booz_xform_jax_available": booz is not None,
        "booz_xform_jax_api_available": booz_api,
        "vmec_jax_paths": [str(path) for path in vmec_paths],
        "booz_xform_jax_paths": [str(path) for path in booz_paths],
    }


def _array(
    mapping: Mapping[str, Any],
    key: str,
    ntheta: int | None = None,
    *,
    validate_finite: bool = True,
) -> jnp.ndarray:
    if key not in mapping:
        raise ValueError(f"missing differentiable geometry field {key!r}")
    arr = jnp.asarray(mapping[key])
    if arr.ndim != 1:
        raise ValueError(f"{key} must be one-dimensional")
    if ntheta is not None and int(arr.shape[0]) != int(ntheta):
        raise ValueError(f"{key} length {arr.shape[0]} does not match theta length {ntheta}")
    if validate_finite and not _is_traced(arr) and not bool(np.all(np.isfinite(np.asarray(arr)))):
        raise ValueError(f"{key} contains non-finite values")
    return arr


def _scalar(mapping: Mapping[str, Any], key: str, default: float) -> Any:
    value = mapping.get(key, default)
    arr = jnp.asarray(value)
    if arr.ndim != 0:
        raise ValueError(f"{key} must be scalar")
    if _is_traced(arr):
        return arr
    return float(np.asarray(arr))


def flux_tube_geometry_from_mapping(
    data: Mapping[str, Any],
    *,
    source_model: str = "vmec_jax",
    validate_finite: bool = True,
) -> FluxTubeGeometryData:
    """Build ``FluxTubeGeometryData`` from an in-memory differentiable backend.

    The input is intentionally the solver-ready flux-tube contract, not a fake
    equilibrium. ``vmec_jax`` / ``booz_xform_jax`` pipelines should first
    produce the sampled field-line arrays named here, then this function
    validates shapes/finite values and hands them to the existing solver.
    """

    theta = _array(data, "theta", validate_finite=validate_finite)
    ntheta = int(theta.shape[0])
    arrays = {
        name: _array(data, name, ntheta, validate_finite=validate_finite)
        for name in _ARRAY_FIELDS
        if name != "theta"
    }
    jacobian = (
        _array(data, "jacobian", ntheta, validate_finite=validate_finite)
        if "jacobian" in data
        else 1.0 / arrays["gradpar"] / arrays["bmag"]
    )
    grho = _array(data, "grho", ntheta, validate_finite=validate_finite) if "grho" in data else jnp.ones_like(theta)

    gradpar_value: Any
    if _is_traced(arrays["gradpar"]):
        gradpar_value = jnp.mean(arrays["gradpar"])
    else:
        gradpar_values = np.asarray(arrays["gradpar"])
        gradpar_value = float(np.mean(gradpar_values))
        if validate_finite and not np.allclose(gradpar_values, gradpar_value, rtol=1.0e-5, atol=1.0e-7):
            raise ValueError("gradpar must be constant along the sampled field line")

    return FluxTubeGeometryData(
        theta=theta,
        gradpar_value=gradpar_value,
        bmag_profile=arrays["bmag"],
        bgrad_profile=arrays["bgrad"],
        gds2_profile=arrays["gds2"],
        gds21_profile=arrays["gds21"],
        gds22_profile=arrays["gds22"],
        cv_profile=arrays["cvdrift"],
        gb_profile=arrays["gbdrift"],
        cv0_profile=arrays["cvdrift0"],
        gb0_profile=arrays["gbdrift0"],
        jacobian_profile=jacobian,
        grho_profile=grho,
        q=_scalar(data, "q", 1.0),
        s_hat=_scalar(data, "s_hat", data.get("shat", 0.0)),
        epsilon=_scalar(data, "epsilon", 0.0),
        R0=_scalar(data, "R0", 1.0),
        B0=_scalar(data, "B0", 1.0),
        alpha=_scalar(data, "alpha", 0.0),
        drift_scale=_scalar(data, "drift_scale", 1.0),
        kxfac=_scalar(data, "kxfac", 1.0),
        theta_scale=_scalar(data, "theta_scale", 1.0),
        nfp=int(data.get("nfp", 1)),
        kperp2_bmag=bool(data.get("kperp2_bmag", True)),
        bessel_bmag_power=_scalar(data, "bessel_bmag_power", 0.0),
        source_model=str(source_model),
        theta_closed_interval=bool(data.get("theta_closed_interval", False)),
    )


def geometry_observable_names() -> tuple[str, ...]:
    """Return the ordered geometry observables used by bridge AD checks."""

    return _GEOMETRY_OBSERVABLE_NAMES


def vmec_metric_tensor_observable_names() -> tuple[str, ...]:
    """Return the ordered observables used by the VMEC metric-tensor gate."""

    return _VMEC_METRIC_OBSERVABLE_NAMES


def vmec_field_line_tensor_observable_names() -> tuple[str, ...]:
    """Return the ordered observables used by the VMEC field-line tensor gate."""

    return _VMEC_FIELD_LINE_OBSERVABLE_NAMES


def flux_tube_geometry_observables(geom: FluxTubeGeometryData) -> jnp.ndarray:
    """Return differentiable scalar observables from solver-ready geometry.

    The observables are intentionally geometry-level quantities: mean field
    strength, relative ripple, metric norm, drift norm, mean Jacobian, and mean
    parallel-gradient factor. They are used to validate the differentiable
    ``vmec_jax`` / ``booz_xform_jax`` bridge before any turbulence observable
    is promoted into an optimization claim.
    """

    bmag = jnp.asarray(geom.bmag_profile)
    jac = jnp.abs(jnp.asarray(geom.jacobian_profile))
    weights = jac / jnp.maximum(jnp.sum(jac), jnp.asarray(1.0e-300, dtype=jac.dtype))
    mean_b = jnp.sum(weights * bmag)
    ripple = jnp.sqrt(jnp.sum(weights * (bmag / jnp.maximum(jnp.abs(mean_b), 1.0e-300) - 1.0) ** 2))
    metric = jnp.sqrt(
        jnp.sum(
            weights
            * (
                jnp.asarray(geom.gds2_profile) ** 2
                + 2.0 * jnp.asarray(geom.gds21_profile) ** 2
                + jnp.asarray(geom.gds22_profile) ** 2
            )
        )
    )
    drift = jnp.sqrt(
        jnp.sum(
            weights
            * (
                jnp.asarray(geom.cv_profile) ** 2
                + jnp.asarray(geom.gb_profile) ** 2
                + jnp.asarray(geom.cv0_profile) ** 2
                + jnp.asarray(geom.gb0_profile) ** 2
            )
        )
    )
    return jnp.asarray(
        [
            mean_b,
            ripple,
            metric,
            drift,
            jnp.mean(jnp.asarray(geom.jacobian_profile)),
            jnp.mean(jnp.asarray(geom.gradpar_value)),
        ]
    )


def finite_difference_jacobian(fn: Any, params: jnp.ndarray, *, step: float = 1.0e-4) -> jnp.ndarray:
    """Central finite-difference Jacobian for small validation problems."""

    p = jnp.asarray(params, dtype=jnp.float64)
    h = float(step)
    if p.ndim != 1:
        raise ValueError("params must be one-dimensional")
    columns = []
    for idx in range(int(p.shape[0])):
        basis = jnp.zeros_like(p).at[idx].set(h)
        columns.append((jnp.asarray(fn(p + basis)) - jnp.asarray(fn(p - basis))) / (2.0 * h))
    return jnp.stack(columns, axis=1)


def _periodic_bilinear_sample_2d(values: jnp.ndarray, theta: jnp.ndarray, zeta: jnp.ndarray) -> jnp.ndarray:
    """Sample a uniform periodic ``(theta,zeta)`` grid with fixed bilinear weights."""

    arr = jnp.asarray(values)
    if arr.ndim != 2:
        raise ValueError("values must be a two-dimensional theta-zeta array")
    theta_arr = jnp.asarray(theta)
    zeta_arr = jnp.asarray(zeta)
    if theta_arr.shape != zeta_arr.shape:
        raise ValueError("theta and zeta samples must have the same shape")

    ntheta = int(arr.shape[0])
    nzeta = int(arr.shape[1])
    if ntheta < 1 or nzeta < 1:
        raise ValueError("values must have non-empty theta and zeta dimensions")

    twopi = jnp.asarray(2.0 * np.pi, dtype=theta_arr.dtype)
    theta_index = jnp.mod(theta_arr, twopi) * (float(ntheta) / twopi)
    zeta_index = jnp.mod(zeta_arr, twopi) * (float(nzeta) / twopi)
    i0 = jnp.floor(theta_index).astype(jnp.int32)
    j0 = jnp.floor(zeta_index).astype(jnp.int32)
    wi = theta_index - i0
    wj = zeta_index - j0
    i1 = (i0 + 1) % int(ntheta)
    j1 = (j0 + 1) % int(nzeta)

    v00 = arr[i0, j0]
    v10 = arr[i1, j0]
    v01 = arr[i0, j1]
    v11 = arr[i1, j1]
    return (
        (1.0 - wi) * (1.0 - wj) * v00
        + wi * (1.0 - wj) * v10
        + (1.0 - wi) * wj * v01
        + wi * wj * v11
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
    """Validate a real ``vmec_jax`` boundary-aspect derivative when available.

    The check intentionally stops at the boundary Fourier API. Full VMEC solves
    are too expensive and environment-sensitive for the default package tests,
    but the boundary-aspect path verifies that SPECTRAX-GK can discover a
    ``vmec_jax`` checkout and differentiate through its JAX-native boundary
    data structures before higher-cost optimization workflows are promoted.
    """

    p = jnp.asarray(params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a one-dimensional length-2 vector")
    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_boundary_api_available", False):
        return {
            "available": False,
            "backend_info": info,
            "aspect": None,
            "grad_ad": None,
            "grad_fd": None,
            "max_abs_ad_fd_error": None,
            "fd_step": float(fd_step),
        }

    import vmec_jax as vj  # type: ignore[import-untyped, import-not-found]

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
    grad_fd = finite_difference_jacobian(lambda x: jnp.asarray([aspect_fn(x)]), p, step=fd_step)[0]
    diff = grad_ad - grad_fd
    return {
        "available": True,
        "backend_info": info,
        "aspect": float(aspect_fn(p)),
        "grad_ad": np.asarray(grad_ad).tolist(),
        "grad_fd": np.asarray(grad_fd).tolist(),
        "max_abs_ad_fd_error": float(np.max(np.abs(np.asarray(diff)))),
        "fd_step": float(fd_step),
        "mpol": int(mpol),
        "ntor": int(ntor),
        "ntheta": int(ntheta),
        "nphi": int(nphi),
        "nfp": int(nfp),
    }


def booz_xform_spectral_sensitivity_report(
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
        return {
            "available": False,
            "backend_info": info,
            "objective": None,
            "grad_ad": None,
            "grad_fd": None,
            "max_abs_ad_fd_error": None,
            "fd_step": float(fd_step),
            "mboz": int(mboz),
            "nboz": int(nboz),
        }

    bx = importlib.import_module("booz_xform_jax.jax_api")

    xm = jnp.asarray([0, 1], dtype=jnp.int32)
    xn = jnp.asarray([0, 0], dtype=jnp.int32)

    def _inputs(ripple_value: Any) -> SimpleNamespace:
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

    try:
        base_inputs = _inputs(jnp.asarray(ripple, dtype=jnp.float64))
        constants, grids = bx.prepare_booz_xform_constants_from_inputs(
            inputs=base_inputs,
            mboz=int(mboz),
            nboz=int(nboz),
            asym=False,
        )

        def objective_fn(ripple_value: jnp.ndarray) -> jnp.ndarray:
            out = bx.booz_xform_from_inputs(
                inputs=_inputs(ripple_value),
                constants=constants,
                grids=grids,
                jit=False,
            )
            bmnc_b = jnp.asarray(out["bmnc_b"])
            return jnp.sum(bmnc_b * bmnc_b)

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
        diff = grad_ad - grad_fd
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "objective": None,
            "grad_ad": None,
            "grad_fd": None,
            "max_abs_ad_fd_error": None,
            "fd_step": float(fd_step),
            "mboz": int(mboz),
            "nboz": int(nboz),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "objective": float(objective_fn(r0)),
        "grad_ad": float(grad_ad),
        "grad_fd": float(grad_fd),
        "max_abs_ad_fd_error": float(jnp.abs(diff)),
        "fd_step": float(fd_step),
        "mboz": int(mboz),
        "nboz": int(nboz),
        "bmnc_b": np.asarray(out["bmnc_b"]).tolist(),
        "rmnc_b": np.asarray(out["rmnc_b"]).tolist(),
        "zmns_b": np.asarray(out["zmns_b"]).tolist(),
        "iota_b": np.asarray(out["iota_b"]).tolist(),
        "ixm_b": np.asarray(out["ixm_b"]).tolist(),
        "ixn_b": np.asarray(out["ixn_b"]).tolist(),
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

    The field-line label convention is :math:`\alpha = \theta - \iota\zeta`.
    This helper is intentionally small and JAX-native so that the
    ``booz_xform_jax`` spectral output can be differentiated all the way into
    the sampled SPECTRAX-GK geometry contract.
    """

    theta_arr = jnp.asarray(theta)
    modes_m = jnp.asarray(ixm_b, dtype=theta_arr.dtype)
    modes_n = jnp.asarray(ixn_b, dtype=theta_arr.dtype)
    coeffs = jnp.asarray(bmnc_b, dtype=theta_arr.dtype)
    iota_arr = jnp.asarray(iota, dtype=theta_arr.dtype)
    iota_safe = jnp.where(jnp.abs(iota_arr) < 1.0e-12, jnp.sign(iota_arr + 1.0e-30) * 1.0e-12, iota_arr)
    zeta = (theta_arr - jnp.asarray(float(alpha), dtype=theta_arr.dtype)) / iota_safe
    phase = theta_arr[:, None] * modes_m[None, :] - zeta[:, None] * modes_n[None, :]
    dphase_dtheta = modes_m[None, :] - modes_n[None, :] / iota_safe
    bmag = jnp.sum(coeffs[None, :] * jnp.cos(phase), axis=1)
    dbmag_dtheta = jnp.sum(-coeffs[None, :] * dphase_dtheta * jnp.sin(phase), axis=1)
    return bmag, dbmag_dtheta


def booz_xform_flux_tube_mapping_from_inputs(
    inputs: Any,
    *,
    mboz: int = 2,
    nboz: int = 1,
    ntheta: int = 96,
    alpha: float = 0.0,
    surface_index: int = 0,
    magnetic_shear: float = 0.35,
    R0: float = 1.0,
    B0: float = 1.0,
    drift_scale: float = 1.0,
    jit: bool = False,
) -> dict[str, Any]:
    """Build a solver-ready flux-tube mapping from ``booz_xform_jax`` output.

    This is the first bounded production bridge step between JAX-native Boozer
    coordinates and SPECTRAX-GK. It uses the real Boozer magnetic-field
    spectrum for ``bmag``/``bgrad`` and supplies smooth metric/drift profiles
    with the same solver-ready names accepted by
    :func:`flux_tube_geometry_from_mapping`.

    Full VMEC/Boozer metric parity remains a separate promotion gate: a
    high-fidelity backend must replace the smooth metric/drift closure here
    with sampled VMEC/Boozer metric tensors before nonlinear optimization
    claims are made.
    """

    info = discover_differentiable_geometry_backends()
    if not info.get("booz_xform_jax_api_available", False):
        raise RuntimeError("booz_xform_jax functional API is not available")

    bx = importlib.import_module("booz_xform_jax.jax_api")
    constants, grids = bx.prepare_booz_xform_constants_from_inputs(
        inputs=inputs,
        mboz=int(mboz),
        nboz=int(nboz),
        asym=bool(getattr(inputs, "bmns", None) is not None),
    )
    out = bx.booz_xform_from_inputs(inputs=inputs, constants=constants, grids=grids, jit=bool(jit))
    idx = int(surface_index)
    theta = jnp.linspace(-jnp.pi, jnp.pi, int(ntheta), endpoint=False)
    bmnc_b = jnp.asarray(out["bmnc_b"])[idx]
    iota = jnp.asarray(out["iota_b"])[idx]
    bmag, dbmag_dtheta = evaluate_boozer_bmag_on_field_line(
        theta,
        bmnc_b=bmnc_b,
        ixm_b=jnp.asarray(out["ixm_b"]),
        ixn_b=jnp.asarray(out["ixn_b"]),
        iota=iota,
        alpha=float(alpha),
    )
    q = 1.0 / jnp.maximum(jnp.abs(iota), jnp.asarray(1.0e-12, dtype=theta.dtype))
    gradpar_value = 1.0 / (q * float(R0))
    gradpar = gradpar_value * jnp.ones_like(theta)
    shear = jnp.asarray(float(magnetic_shear), dtype=theta.dtype)
    field_line_shift = shear * theta
    gds2 = 1.0 + field_line_shift * field_line_shift
    gds21 = -shear * field_line_shift
    gds22 = (1.0 + shear * shear) * jnp.ones_like(theta)
    cv = float(drift_scale) * (jnp.cos(theta) + field_line_shift * jnp.sin(theta)) / float(R0)
    cv0 = -float(drift_scale) * shear * jnp.sin(theta) / float(R0)
    bmag_safe = jnp.maximum(jnp.abs(bmag), jnp.asarray(1.0e-12, dtype=theta.dtype))
    return {
        "theta": theta,
        "gradpar": gradpar,
        "bmag": bmag,
        "bgrad": gradpar_value * dbmag_dtheta / bmag_safe,
        "gds2": gds2,
        "gds21": gds21,
        "gds22": gds22,
        "cvdrift": cv,
        "gbdrift": cv,
        "cvdrift0": cv0,
        "gbdrift0": cv0,
        "jacobian": 1.0 / (gradpar * bmag_safe),
        "grho": jnp.ones_like(theta),
        "q": q,
        "s_hat": shear,
        "epsilon": jnp.sqrt(jnp.mean((bmag / jnp.mean(bmag) - 1.0) ** 2)),
        "R0": float(R0),
        "B0": float(B0),
        "alpha": float(alpha),
        "drift_scale": float(drift_scale),
        "nfp": int(jnp.asarray(getattr(inputs, "nfp", 1))),
        "booz_xform": {
            "bmnc_b": bmnc_b,
            "ixm_b": jnp.asarray(out["ixm_b"]),
            "ixn_b": jnp.asarray(out["ixn_b"]),
            "iota_b": iota,
        },
    }


def booz_xform_flux_tube_sensitivity_report(
    *,
    params: jnp.ndarray | None = None,
    fd_step: float = 2.0e-5,
    mboz: int = 2,
    nboz: int = 1,
    ntheta: int = 64,
) -> dict[str, object]:
    """AD/FD-check a Boozer-spectrum-to-flux-tube geometry bridge.

    ``params = [axisymmetric_B_ripple, helical_B_ripple]`` perturbs a tiny
    one-surface VMEC-to-Boozer input bundle. The real ``booz_xform_jax``
    transform is run for each parameter vector; its Boozer ``|B|`` spectrum is
    sampled on a field line and converted into ``FluxTubeGeometryData``.
    """

    info = discover_differentiable_geometry_backends()
    if not info.get("booz_xform_jax_api_available", False):
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "mboz": int(mboz),
            "nboz": int(nboz),
            "ntheta": int(ntheta),
        }

    p = jnp.asarray([0.05, 0.02] if params is None else params, dtype=jnp.float64)
    if p.ndim != 1 or int(p.shape[0]) != 2:
        raise ValueError("params must be a length-2 vector")

    xm = jnp.asarray([0, 1, 1], dtype=jnp.int32)
    xn = jnp.asarray([0, 0, 2], dtype=jnp.int32)

    def _inputs(x: jnp.ndarray) -> SimpleNamespace:
        axisym_ripple, helical_ripple = x
        one = jnp.asarray(1.0, dtype=x.dtype)
        zero = jnp.asarray(0.0, dtype=x.dtype)
        minor = jnp.asarray(0.22, dtype=x.dtype)
        helical_shape = jnp.asarray(0.02, dtype=x.dtype)
        return SimpleNamespace(
            rmnc=jnp.asarray([[one, minor, helical_shape]], dtype=x.dtype),
            zmns=jnp.asarray([[zero, minor, helical_shape]], dtype=x.dtype),
            lmns=jnp.asarray([[zero, zero, zero]], dtype=x.dtype),
            bmnc=jnp.asarray([[one, axisym_ripple, helical_ripple]], dtype=x.dtype),
            bsubumnc=jnp.asarray([[0.1, 0.0, 0.0]], dtype=x.dtype),
            bsubvmnc=jnp.asarray([[one, zero, zero]], dtype=x.dtype),
            iota=jnp.asarray([0.41], dtype=x.dtype),
            xm=xm,
            xn=xn,
            xm_nyq=xm,
            xn_nyq=xn,
            nfp=2,
            bmns=None,
            bsubumns=None,
            bsubvmns=None,
        )

    try:
        sensitivity = geometry_sensitivity_report(
            lambda x: booz_xform_flux_tube_mapping_from_inputs(
                _inputs(x),
                mboz=int(mboz),
                nboz=int(nboz),
                ntheta=int(ntheta),
                magnetic_shear=0.35,
                jit=False,
            ),
            p,
            fd_step=float(fd_step),
            source_model="booz_xform_jax:field-line-bmag",
        )
        mapping = booz_xform_flux_tube_mapping_from_inputs(
            _inputs(p),
            mboz=int(mboz),
            nboz=int(nboz),
            ntheta=int(ntheta),
            magnetic_shear=0.35,
            jit=False,
        )
        booz_meta = mapping["booz_xform"]
    except Exception as exc:
        return {
            "available": False,
            "backend_info": info,
            "sensitivity": None,
            "fd_step": float(fd_step),
            "mboz": int(mboz),
            "nboz": int(nboz),
            "ntheta": int(ntheta),
            "error": f"{type(exc).__name__}: {exc}",
        }

    return {
        "available": True,
        "backend_info": info,
        "params": np.asarray(p).tolist(),
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


def vmec_jax_boozer_flux_tube_sensitivity_report(
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
    if not (info.get("vmec_jax_available", False) and info.get("booz_xform_jax_api_available", False)):
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
            raise RuntimeError(f"vmec_jax example {case_name!r} has no bundled wout reference")

        cfg, indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        base_Rcos = jnp.asarray(state.Rcos)
        base_Zsin = jnp.asarray(state.Zsin)
        if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
            raise RuntimeError("vmec_jax state Rcos/Zsin arrays must be two-dimensional")

        ridx = int(base_Rcos.shape[0] // 2) if radial_index is None else int(radial_index)
        midx = int(mode_index)
        if not (0 <= ridx < int(base_Rcos.shape[0])):
            raise ValueError("radial_index is outside the VMEC state radial grid")
        if not (0 <= midx < int(base_Rcos.shape[1])):
            raise ValueError("mode_index is outside the VMEC state mode table")
        sidx = max(0, min(ridx - 1, int(base_Rcos.shape[0]) - 2)) if surface_index is None else int(surface_index)
        if not (0 <= sidx < int(base_Rcos.shape[0]) - 1):
            raise ValueError("surface_index is outside the VMEC half-mesh Boozer surface grid")

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


def vmec_jax_metric_tensor_sensitivity_report(
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
            raise RuntimeError(f"vmec_jax example {case_name!r} has no bundled wout reference")

        cfg, _indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        base_Rcos = jnp.asarray(state.Rcos)
        base_Zsin = jnp.asarray(state.Zsin)
        if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
            raise RuntimeError("vmec_jax state Rcos/Zsin arrays must be two-dimensional")

        ridx = int(base_Rcos.shape[0] // 2) if radial_index is None else int(radial_index)
        midx = int(mode_index)
        if not (0 <= ridx < int(base_Rcos.shape[0])):
            raise ValueError("radial_index is outside the VMEC state radial grid")
        if not (0 <= midx < int(base_Rcos.shape[1])):
            raise ValueError("mode_index is outside the VMEC state mode table")
        sidx = max(0, min(ridx - 1, int(base_Rcos.shape[0]) - 1)) if surface_index is None else int(surface_index)
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
        "radial_index": int(ridx),
        "mode_index": int(midx),
        "surface_index": int(sidx),
        "state_shape": [int(base_Rcos.shape[0]), int(base_Rcos.shape[1])],
        "metric_grid_shape": [int(v) for v in np.asarray(geom0.sqrtg).shape],
        "fd_step": float(fd_step),
        "rms_epsilon": float(rms_epsilon),
    }


def vmec_jax_field_line_tensor_sensitivity_report(
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
            raise RuntimeError(f"vmec_jax example {case_name!r} has no bundled wout reference")

        cfg, _indata = config_mod.load_config(str(input_path))
        static = static_mod.build_static(cfg)
        wout = wout_mod.read_wout(wout_path)
        state = wout_mod.state_from_wout(wout)

        base_Rcos = jnp.asarray(state.Rcos)
        base_Zsin = jnp.asarray(state.Zsin)
        if base_Rcos.ndim != 2 or base_Zsin.ndim != 2:
            raise RuntimeError("vmec_jax state Rcos/Zsin arrays must be two-dimensional")

        ridx = int(base_Rcos.shape[0] // 2) if radial_index is None else int(radial_index)
        midx = int(mode_index)
        if not (0 <= ridx < int(base_Rcos.shape[0])):
            raise ValueError("radial_index is outside the VMEC state radial grid")
        if not (0 <= midx < int(base_Rcos.shape[1])):
            raise ValueError("mode_index is outside the VMEC state mode table")
        sidx = max(1, min(ridx, int(base_Rcos.shape[0]) - 2)) if surface_index is None else int(surface_index)
        if not (0 <= sidx < int(base_Rcos.shape[0])):
            raise ValueError("surface_index is outside the VMEC metric radial grid")

        iota_profile = jnp.asarray(getattr(wout, "iotas"))
        if iota_profile.ndim != 1 or int(iota_profile.shape[0]) <= sidx:
            raise RuntimeError("vmec_jax wout iotas profile is missing or incompatible with the state grid")
        iota_line = iota_profile[sidx]
        iota_safe = jnp.where(jnp.abs(iota_line) < 1.0e-12, jnp.sign(iota_line + 1.0e-30) * 1.0e-12, iota_line)
        theta_line = jnp.linspace(-jnp.pi, jnp.pi, ntheta_int, endpoint=False, dtype=p.dtype)
        theta_vmec = jnp.mod(theta_line + jnp.pi, 2.0 * jnp.pi)
        zeta_line = jnp.mod((theta_vmec - jnp.asarray(float(alpha), dtype=p.dtype)) / iota_safe, 2.0 * jnp.pi)
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
            bmag = jnp.sqrt(jnp.maximum(_periodic_bilinear_sample_2d(b2[sidx], theta_vmec, zeta_line), b2_floor_arr))
            sqrtg = _periodic_bilinear_sample_2d(geom.sqrtg[sidx], theta_vmec, zeta_line)
            g_tt = _periodic_bilinear_sample_2d(geom.g_tt[sidx], theta_vmec, zeta_line)
            g_tp = _periodic_bilinear_sample_2d(geom.g_tp[sidx], theta_vmec, zeta_line)
            g_pp = _periodic_bilinear_sample_2d(geom.g_pp[sidx], theta_vmec, zeta_line)
            g_ss = _periodic_bilinear_sample_2d(geom.g_ss[sidx], theta_vmec, zeta_line)
            mean_b = jnp.mean(bmag)
            ripple = jnp.std(bmag) / jnp.maximum(jnp.abs(mean_b), jnp.asarray(1.0e-30, dtype=bmag.dtype))
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
        jac_fd = finite_difference_jacobian(field_line_observables, p, step=float(fd_step))
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


def geometry_sensitivity_report(
    mapping_fn: Any,
    params: jnp.ndarray,
    *,
    fd_step: float = 1.0e-4,
    source_model: str = "vmec_jax:in-memory",
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

    obs = observable_fn(p)
    jac_ad = jax.jacfwd(observable_fn)(p)
    jac_fd = finite_difference_jacobian(observable_fn, p, step=fd_step)
    diff = jac_ad - jac_fd
    scale = jnp.maximum(jnp.abs(jac_fd), 1.0e-12)

    return {
        "observable_names": list(_GEOMETRY_OBSERVABLE_NAMES),
        "params": np.asarray(p).tolist(),
        "observables": np.asarray(obs).tolist(),
        "jacobian_ad": np.asarray(jac_ad).tolist(),
        "jacobian_fd": np.asarray(jac_fd).tolist(),
        "max_abs_ad_fd_error": float(np.max(np.abs(np.asarray(diff)))),
        "max_rel_ad_fd_error": float(np.max(np.abs(np.asarray(diff) / np.asarray(scale)))),
        "fd_step": float(fd_step),
        "source_model": str(source_model),
    }


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
    source_model: str = "vmec_jax:in-memory",
) -> dict[str, object]:
    """Run a small Gauss-Newton geometry inverse-design validation.

    ``mapping_fn(params)`` must be the same solver-ready field-line mapping
    accepted by :func:`flux_tube_geometry_from_mapping`. The routine is meant
    for differentiable ``vmec_jax`` / ``booz_xform_jax`` workflows: it keeps
    the optimization, sensitivity check, and local UQ covariance in one
    JSON-friendly report so examples can validate the full AD contract without
    depending on a long equilibrium solve in CI.
    """

    params = jnp.asarray(initial_params, dtype=jnp.float64)
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
        raise ValueError("observable_indices must be a non-empty one-dimensional sequence")
    if np.any(indices_np < 0) or np.any(indices_np >= len(_GEOMETRY_OBSERVABLE_NAMES)):
        raise ValueError("observable_indices contains an out-of-range observable index")

    target = jnp.asarray(target_observables, dtype=params.dtype)
    if target.ndim != 1 or int(target.shape[0]) != int(indices_np.size):
        raise ValueError("target_observables length must match observable_indices")
    indices = jnp.asarray(indices_np, dtype=jnp.int32)

    def observable_fn(x: jnp.ndarray) -> jnp.ndarray:
        geom = flux_tube_geometry_from_mapping(
            mapping_fn(x),
            source_model=source_model,
            validate_finite=False,
        )
        return flux_tube_geometry_observables(geom)[indices]

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
        normal = jac.T @ jac + float(damping) * jnp.eye(int(p.shape[0]), dtype=p.dtype)
        delta = jnp.linalg.solve(normal, jac.T @ residual)
        p = p - delta

    jac_ad = jax.jacfwd(observable_fn)(p)
    jac_fd = finite_difference_jacobian(observable_fn, p, step=fd_step)
    diff = jac_ad - jac_fd
    scale = jnp.maximum(jnp.abs(jac_fd), 1.0e-12)
    uq = covariance_diagnostics(np.asarray(jac_ad), np.asarray(residual), regularization=regularization)

    return {
        "observable_names": [str(_GEOMETRY_OBSERVABLE_NAMES[int(i)]) for i in indices_np],
        "initial_params": np.asarray(params).tolist(),
        "final_params": np.asarray(p).tolist(),
        "target_observables": np.asarray(target).tolist(),
        "final_observables": np.asarray(observable_fn(p)).tolist(),
        "final_residual": np.asarray(residual).tolist(),
        "final_residual_norm": float(jnp.linalg.norm(residual)),
        "history": history,
        "jacobian_ad": np.asarray(jac_ad).tolist(),
        "jacobian_fd": np.asarray(jac_fd).tolist(),
        "max_abs_ad_fd_error": float(np.max(np.abs(np.asarray(diff)))),
        "max_rel_ad_fd_error": float(np.max(np.abs(np.asarray(diff) / np.asarray(scale)))),
        "uq": uq,
        "fd_step": float(fd_step),
        "damping": float(damping),
        "regularization": float(regularization),
        "source_model": str(source_model),
        "backend_info": discover_differentiable_geometry_backends(),
    }


__all__ = [
    "booz_xform_flux_tube_mapping_from_inputs",
    "booz_xform_flux_tube_sensitivity_report",
    "booz_xform_spectral_sensitivity_report",
    "discover_differentiable_geometry_backends",
    "evaluate_boozer_bmag_on_field_line",
    "finite_difference_jacobian",
    "flux_tube_geometry_from_mapping",
    "flux_tube_geometry_observables",
    "geometry_inverse_design_report",
    "geometry_observable_names",
    "geometry_sensitivity_report",
    "vmec_jax_boozer_flux_tube_sensitivity_report",
    "vmec_jax_field_line_tensor_sensitivity_report",
    "vmec_jax_metric_tensor_sensitivity_report",
    "vmec_boundary_aspect_sensitivity_report",
    "vmec_field_line_tensor_observable_names",
    "vmec_metric_tensor_observable_names",
]
