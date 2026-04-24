"""Differentiable geometry bridge contracts for VMEC/JAX pipelines."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

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
    try:
        return importlib.import_module(name)
    except Exception:
        pass
    import sys

    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def discover_differentiable_geometry_backends() -> dict[str, object]:
    """Discover optional ``vmec_jax`` and ``booz_xform_jax`` bridge APIs."""

    repo_parent = Path(__file__).resolve().parents[3].parent
    vmec_paths = _candidate_paths(
        ("SPECTRAX_VMEC_JAX_PATH", "VMEC_JAX_PATH"),
        (repo_parent / "vmec_jax", Path("/Users/rogeriojorge/local/vmec_jax")),
    )
    booz_paths = _candidate_paths(
        ("SPECTRAX_BOOZ_XFORM_JAX_PATH", "BOOZ_XFORM_JAX_PATH"),
        (repo_parent / "booz_xform_jax", Path("/Users/rogeriojorge/local/booz_xform_jax")),
    )
    vmec = _find_importable_module("vmec_jax", vmec_paths)
    booz = _find_importable_module("booz_xform_jax", booz_paths)
    booz_jax_api = None if booz is None else _find_importable_module("booz_xform_jax.jax_api", booz_paths)

    return {
        "vmec_jax_available": vmec is not None,
        "booz_xform_jax_available": booz is not None,
        "booz_xform_jax_api_available": booz_jax_api is not None
        and hasattr(booz_jax_api, "prepare_booz_xform_constants_from_inputs")
        and hasattr(booz_jax_api, "booz_xform_jax_impl"),
        "vmec_jax_paths": [str(path) for path in vmec_paths],
        "booz_xform_jax_paths": [str(path) for path in booz_paths],
    }


def _array(mapping: Mapping[str, Any], key: str, ntheta: int | None = None) -> jnp.ndarray:
    if key not in mapping:
        raise ValueError(f"missing differentiable geometry field {key!r}")
    arr = jnp.asarray(mapping[key])
    if arr.ndim != 1:
        raise ValueError(f"{key} must be one-dimensional")
    if ntheta is not None and int(arr.shape[0]) != int(ntheta):
        raise ValueError(f"{key} length {arr.shape[0]} does not match theta length {ntheta}")
    if not bool(np.all(np.isfinite(np.asarray(arr)))):
        raise ValueError(f"{key} contains non-finite values")
    return arr


def flux_tube_geometry_from_mapping(
    data: Mapping[str, Any],
    *,
    source_model: str = "vmec_jax",
) -> FluxTubeGeometryData:
    """Build ``FluxTubeGeometryData`` from an in-memory differentiable backend.

    The input is intentionally the solver-ready flux-tube contract, not a fake
    equilibrium. ``vmec_jax`` / ``booz_xform_jax`` pipelines should first
    produce the sampled field-line arrays named here, then this function
    validates shapes/finite values and hands them to the existing solver.
    """

    theta = _array(data, "theta")
    ntheta = int(theta.shape[0])
    arrays = {name: _array(data, name, ntheta) for name in _ARRAY_FIELDS if name != "theta"}
    jacobian = _array(data, "jacobian", ntheta) if "jacobian" in data else 1.0 / arrays["gradpar"] / arrays["bmag"]
    grho = _array(data, "grho", ntheta) if "grho" in data else jnp.ones_like(theta)

    gradpar_values = np.asarray(arrays["gradpar"])
    gradpar_value = float(np.mean(gradpar_values))
    if not np.allclose(gradpar_values, gradpar_value, rtol=1.0e-5, atol=1.0e-7):
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
        q=float(data.get("q", 1.0)),
        s_hat=float(data.get("s_hat", data.get("shat", 0.0))),
        epsilon=float(data.get("epsilon", 0.0)),
        R0=float(data.get("R0", 1.0)),
        B0=float(data.get("B0", 1.0)),
        alpha=float(data.get("alpha", 0.0)),
        drift_scale=float(data.get("drift_scale", 1.0)),
        kxfac=float(data.get("kxfac", 1.0)),
        theta_scale=float(data.get("theta_scale", 1.0)),
        nfp=int(data.get("nfp", 1)),
        kperp2_bmag=bool(data.get("kperp2_bmag", True)),
        bessel_bmag_power=float(data.get("bessel_bmag_power", 0.0)),
        source_model=str(source_model),
        theta_closed_interval=bool(data.get("theta_closed_interval", False)),
    )


__all__ = ["discover_differentiable_geometry_backends", "flux_tube_geometry_from_mapping"]
