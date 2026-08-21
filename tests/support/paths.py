"""Shared path helpers for tests.

Tests move across topology folders during refactors; keep repository paths here
instead of encoding parent-depth assumptions in individual test files.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
TOOLS_ROOT = REPO_ROOT / "tools"


def load_repo_script(
    relative_path: str | Path,
    module_name: str | None = None,
    *,
    write_bytecode: bool = True,
) -> ModuleType:
    """Load a repository script directly from the checkout."""

    path = REPO_ROOT / relative_path
    if module_name is None:
        module_name = f"test_loaded_{path.stem}"
    script_dir = path.parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = not write_bytecode
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def load_tool_script(tool_folder: str, script_name: str) -> ModuleType:
    """Load a script from a ``tools`` subfolder by file stem."""

    tools_dir = TOOLS_ROOT / tool_folder
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    return load_repo_script(tools_dir.relative_to(REPO_ROOT) / f"{script_name}.py")


def load_artifact_tool(script_name: str) -> ModuleType:
    """Load a ``tools/artifacts`` script directly from the checkout."""

    return load_tool_script("artifacts", script_name)


def load_release_tool(script_name: str) -> ModuleType:
    """Load a ``tools/release`` script directly from the checkout."""

    return load_tool_script("release", script_name)


def load_profiling_tool(script_name: str) -> ModuleType:
    """Load a ``tools/profiling`` script directly from the checkout."""

    return load_tool_script("profiling", script_name)


def load_comparison_tool(script_name: str) -> ModuleType:
    """Load a ``tools/comparison`` script directly from the checkout."""

    return load_tool_script("comparison", script_name)
