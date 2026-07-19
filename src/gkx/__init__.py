"""GKX: a JAX gyrokinetic solver with Hermite-Laguerre velocity space."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from gkx._version import __version__

_api = import_module("gkx.api")
__all__ = list(_api.__all__)
_EXPORT_TARGETS: dict[str, tuple[str, str]] = dict(_api._EXPORT_TARGETS)
__all__ = ["__version__", *__all__]


def __getattr__(name: str) -> Any:
    """Lazily resolve public API exports without importing the full solver stack."""

    if name == "__version__":
        return __version__
    try:
        module_name, attr_name = _EXPORT_TARGETS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
