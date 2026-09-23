"""Shared runner behind the ``scripts/*.py`` developer commands.

Each command owns one or more module packages under ``scripts/`` and runs one
module per invocation, exactly as if that module had been executed directly:
the module sees the remaining arguments in ``sys.argv`` and keeps its own exit
status and output. Standard library only, so ``scripts/check.py`` can use it
before anything is installed.
"""

from __future__ import annotations

import ast
from pathlib import Path
import runpy
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent


def run_module(path: Path, argv: list[str]) -> int:
    """Run ``path`` as ``__main__`` with ``argv`` and return its exit status.

    ``sys.path[0]`` becomes the module's own directory, as for direct execution.
    That also takes ``scripts/`` off the path while the module runs, so the
    ``profile.py`` command cannot shadow the standard-library ``profile``
    module that ``cProfile`` imports.
    """

    saved_argv = sys.argv
    saved_path = list(sys.path)
    sys.argv = [str(path), *argv]
    sys.path[:] = [str(path.parent)] + [
        entry for entry in sys.path if not entry or Path(entry).resolve() != SCRIPTS_DIR
    ]
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = saved_argv
        sys.path[:] = saved_path
    return 0


def _summary(path: Path) -> str:
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    except (OSError, SyntaxError):
        doc = None
    return doc.strip().splitlines()[0] if doc else ""


def modules(packages: tuple[str, ...]) -> dict[str, Path]:
    """Return ``{stem: path}`` for every runnable module in ``packages``."""

    found: dict[str, Path] = {}
    for package in packages:
        for path in sorted((SCRIPTS_DIR / package).glob("*.py")):
            if path.name.startswith("_"):
                continue
            if path.stem in found:
                raise RuntimeError(f"duplicate module name {path.stem!r}")
            found[path.stem] = path
    return found


def main(command: str, packages: tuple[str, ...], argv: list[str]) -> int:
    """Dispatch ``argv[0]`` to the module of that name in ``packages``."""

    table = modules(packages)
    if not argv or argv[0] in {"-h", "--help", "--list"}:
        width = max(len(name) for name in table)
        rows = "\n".join(
            f"  {name.ljust(width)}  {_summary(path)}" for name, path in table.items()
        )
        where = ", ".join(f"scripts/{package}/" for package in packages)
        print(
            f"usage: python scripts/{command}.py <module> [arguments...]\n\n"
            f"modules ({where}):\n{rows}"
        )
        return 0
    name, rest = argv[0], argv[1:]
    if name.endswith(".py"):
        name = name[:-3]
    if name not in table:
        print(
            f"unknown module {name!r}; run `python scripts/{command}.py --list`",
            file=sys.stderr,
        )
        return 2
    return run_module(table[name], rest)
