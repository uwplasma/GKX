"""Optional differentiable-geometry backend discovery helpers."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


def _jax_float_dtype() -> Any:
    """Return the active JAX floating dtype for small validation arrays."""

    return jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32


def _candidate_paths(env_names: Sequence[str], defaults: Sequence[Path]) -> list[Path]:
    """Return existing backend roots from environment variables and defaults."""

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
    """Import a backend module, preferring explicitly configured checkout paths."""

    import sys

    def _module_file(module_name: str) -> Path | None:
        module = sys.modules.get(module_name)
        raw = None if module is None else getattr(module, "__file__", None)
        if raw is None:
            return None
        return Path(str(raw)).resolve(strict=False)

    def _inside_candidates(path: Path | None) -> bool:
        if path is None:
            return False
        for root in paths:
            try:
                path.relative_to(root.resolve(strict=False))
                return True
            except ValueError:
                continue
        return False

    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    # Prefer explicitly configured/local differentiable-geometry checkouts over
    # globally installed packages. Some editable VMEC checkouts carry example
    # data that the site package does not ship, so importing the site package
    # first can silently disable real validation gates.
    root_name = name.split(".", maxsplit=1)[0]
    root_path = _module_file(root_name)
    if root_path is not None and not _inside_candidates(root_path):
        for module_name in list(sys.modules):
            if module_name == root_name or module_name.startswith(f"{root_name}."):
                sys.modules.pop(module_name, None)

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
    booz_jax_api = (
        None
        if booz is None
        else _find_importable_module("booz_xform_jax.jax_api", booz_paths)
    )

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


__all__ = [
    "_candidate_paths",
    "_find_importable_module",
    "_is_traced",
    "_jax_float_dtype",
    "discover_differentiable_geometry_backends",
]
