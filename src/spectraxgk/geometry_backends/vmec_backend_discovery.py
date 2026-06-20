"""Dependency discovery for VMEC imported-geometry generation."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys
from typing import Any

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
