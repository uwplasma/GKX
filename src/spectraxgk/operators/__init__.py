"""Public operator kernels with lazy domain imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["hermite_streaming"]


def __getattr__(name: str) -> Any:
    if name != "hermite_streaming":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module("spectraxgk.operators.linear.moments").hermite_streaming
    globals()[name] = value
    return value
