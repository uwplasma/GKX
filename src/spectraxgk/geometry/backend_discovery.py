"""Optional differentiable-geometry backend discovery helpers."""

from __future__ import annotations

import importlib
import os
import sys
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
    # first can silently disable real validation gates.  Only drop an already
    # imported package when a candidate checkout actually exists: re-importing
    # vmex/booz_xform_jax invalidates their pytree registrations and caches.
    root_name = name.split(".", maxsplit=1)[0]
    root_path = _module_file(root_name)
    if paths and root_path is not None and not _inside_candidates(root_path):
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
    """Discover the optional ``vmex`` and ``booz_xform_jax`` bridge APIs.

    ``vmex`` is normally the installed package; the filesystem fallbacks keep
    working for local checkouts (the historical ``vmex`` repository IS the
    vmex source tree).  The report KEY NAMES intentionally keep the legacy
    ``vmex_*`` spelling: they are string-coupled to frozen validation
    artifacts, and the coordinated identifier rename happens in a later phase.
    """

    repo_parent = Path(__file__).resolve().parents[3].parent
    home = Path.home()
    vmec_paths = _candidate_paths(
        ("SPECTRAX_VMEX_PATH", "VMEX_PATH"),
        (
            repo_parent / "vmex",
            home / "vmex",
            home / "local" / "vmex",
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
    vmec = _find_importable_module("vmex", vmec_paths)
    booz = _find_importable_module("booz_xform_jax", booz_paths)
    booz_jax_api = (
        None
        if booz is None
        else _find_importable_module("booz_xform_jax.jax_api", booz_paths)
    )

    # The boundary bridge still targets the retired vmex boundary helpers;
    # vmex does not expose them, so this reports False until that route is
    # ported (spectraxgk.geometry.booz_xform_bridge).
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
        "vmex_available": vmec is not None,
        "vmex_boundary_api_available": vmec_boundary_api,
        "booz_xform_jax_available": booz is not None,
        "booz_xform_jax_api_available": booz_api,
        "vmex_paths": [str(path) for path in vmec_paths],
        "booz_xform_jax_paths": [str(path) for path in booz_paths],
    }




def _booz_xform_jax_search_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    raw_paths: list[Path] = []
    for env_name in ("SPECTRAX_BOOZ_XFORM_JAX_PATH", "BOOZ_XFORM_JAX_PATH"):
        raw = os.environ.get(env_name)
        if raw:
            raw_paths.append(Path(os.path.expandvars(raw)).expanduser())
    raw_paths.append(repo_root.parent / "booz_xform_jax")

    search_paths: list[Path] = []
    seen: set[Path] = set()
    for base in raw_paths:
        for candidate in (base, base / "src"):
            resolved = candidate.resolve(strict=False)
            if resolved in seen or not resolved.exists():
                continue
            seen.add(resolved)
            search_paths.append(resolved)
    return search_paths


def _import_module_with_search_paths(
    name: str, search_paths: list[Path], *, required_attr: str | None = None
) -> Any:
    def _valid(module: Any) -> bool:
        return required_attr is None or hasattr(module, required_attr)

    try:
        module = importlib.import_module(name)
        if _valid(module):
            return module
    except Exception:
        pass

    for path in search_paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        cached = sys.modules.get(name)
        if cached is not None and not _valid(cached):
            del sys.modules[name]
        try:
            module = importlib.import_module(name)
            if _valid(module):
                return module
        except Exception:
            continue
    raise ImportError(name)


def _import_booz_xform_jax_backend() -> Any:
    search_paths = _booz_xform_jax_search_paths()
    return _import_module_with_search_paths(
        "booz_xform_jax", search_paths, required_attr="Booz_xform"
    )


def _import_booz_xform_backend() -> Any:
    module = importlib.import_module("booz_xform")
    if not hasattr(module, "Booz_xform"):
        raise ImportError("booz_xform backend does not expose Booz_xform")
    return module


def _import_booz_backend(preferred: str | None = None) -> Any:
    choice = (
        (preferred or os.environ.get("SPECTRAX_BOOZ_BACKEND", "auto")).strip().lower()
    )
    aliases = {
        "auto": "auto",
        "booz_xform_jax": "booz_xform_jax",
        "jax": "booz_xform_jax",
        "booz_xform": "booz_xform",
        "fortran": "booz_xform",
    }
    if choice not in aliases:
        raise ValueError(
            "SPECTRAX_BOOZ_BACKEND must be one of "
            "auto, booz_xform_jax, jax, booz_xform, or fortran"
        )
    choice = aliases[choice]
    if choice == "booz_xform_jax":
        return _import_booz_xform_jax_backend()
    if choice == "booz_xform":
        return _import_booz_xform_backend()

    try:
        return _import_booz_xform_jax_backend()
    except Exception:
        pass
    try:
        return _import_booz_xform_backend()
    except Exception as exc:
        raise ImportError("booz_xform_jax/booz_xform backend unavailable") from exc


def _booz_read_wout_square_layout_failure(exc: BaseException) -> bool:
    message = str(exc)
    return (
        isinstance(exc, ValueError)
        and "rmnc0 has unexpected shape" in message
        and "one dimension must equal ns=" in message
    )


def _new_booz_object(backend: Any, vmec_fname: str) -> Any:
    booz_obj = backend.Booz_xform()
    booz_obj.verbose = 0
    booz_obj.read_wout(str(vmec_fname))
    return booz_obj


# ---------------------------------------------------------------------------
# Public availability check
# ---------------------------------------------------------------------------


def internal_vmec_backend_available() -> bool:
    """Return True when the internal VMEC backend dependencies are present."""

    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
    except Exception:
        return False

    try:
        _import_booz_backend()
        return True
    except Exception:
        return False


__all__ = [
    "_booz_xform_jax_search_paths",
    "_import_module_with_search_paths",
    "_import_booz_xform_jax_backend",
    "_import_booz_xform_backend",
    "_import_booz_backend",
    "_booz_read_wout_square_layout_failure",
    "_new_booz_object",
    "internal_vmec_backend_available",
    "_candidate_paths",
    "_find_importable_module",
    "_is_traced",
    "_jax_float_dtype",
    "discover_differentiable_geometry_backends",
]
