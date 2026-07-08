"""SPECTRAX-GK: a JAX gyrokinetic solver with Hermite-Laguerre velocity space."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
from typing import Any

from spectraxgk._version import __version__

_API_MODULE_ORDER = (
    "configuration",
    "geometry",
    "diagnostics",
    "runtime",
    "solvers",
    "benchmarks",
    "validation",
    "parallel",
    "objectives",
    "artifacts",
)


def _literal_all(path: Path) -> list[str]:
    """Read a module-level ``__all__`` list without importing the module."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            names = ast.literal_eval(node.value)
            return [str(name) for name in names]
    raise RuntimeError(f"{path} does not define a literal __all__")


def _api_exports() -> tuple[list[str], dict[str, str]]:
    api_dir = Path(__file__).with_name("api")
    public_names = _literal_all(api_dir / "__init__.py")
    export_modules: dict[str, str] = {}
    for module_name in _API_MODULE_ORDER:
        for name in _literal_all(api_dir / f"{module_name}.py"):
            export_modules.setdefault(name, module_name)
    missing = [name for name in public_names if name not in export_modules]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"public API export(s) are missing from API modules: {joined}"
        )
    return public_names, export_modules


__all__, _EXPORT_MODULES = _api_exports()
__all__ = ["__version__", *__all__]


def __getattr__(name: str) -> Any:
    """Lazily resolve public API exports without importing the full solver stack."""

    if name == "__version__":
        return __version__
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"spectraxgk.api.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
