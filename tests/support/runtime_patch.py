"""Patch one runtime collaborator everywhere the solve layer looks it up.

The runtime solve layer has no injection records: each owner module imports
its collaborators and calls them directly. A test that replaces a
collaborator therefore patches the name in every solve-layer module that
binds it, which is what this helper does.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

SOLVE_LAYER_MODULES = (
    "gkx.runtime",
    "gkx.cli",
    "gkx.workflows.linear",
    "gkx.workflows.nonlinear",
    "gkx.workflows.runtime.artifacts",
    "gkx.workflows.runtime.commands",
    "gkx.workflows.runtime.diagnostics",
    "gkx.workflows.runtime.orchestration_scan",
    "gkx.workflows.runtime.startup",
    "gkx.workflows.runtime.toml",
    "gkx.artifacts.nonlinear_netcdf",
)


def patch_runtime(monkeypatch: pytest.MonkeyPatch, name: str, value: Any) -> None:
    """Replace ``name`` in every solve-layer module that binds it."""

    hits = 0
    for module_name in SOLVE_LAYER_MODULES:
        module = importlib.import_module(module_name)
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
            hits += 1
    if not hits:
        raise AttributeError(f"no solve-layer module binds {name!r}")
