"""Parallel execution, decomposition, and sharding helpers."""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path
from typing import Any

_PARALLEL_MODULE_ORDER = (
    "identity",
    "batch",
    "independent",
    "decomposition",
    "state",
    "velocity",
    "integrators",
)


def _literal_all(path: Path) -> list[str]:
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


def _parallel_exports() -> tuple[list[str], dict[str, str]]:
    package_dir = Path(__file__).parent
    public_names: list[str] = []
    export_modules: dict[str, str] = {}
    for module_name in _PARALLEL_MODULE_ORDER:
        for name in _literal_all(package_dir / f"{module_name}.py"):
            if name not in export_modules:
                public_names.append(name)
                export_modules[name] = module_name
    return public_names, export_modules


__all__, _EXPORT_MODULES = _parallel_exports()


def __getattr__(name: str) -> Any:
    """Lazily resolve parallel exports so pure contracts stay dependency-light."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"spectraxgk.parallel.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
