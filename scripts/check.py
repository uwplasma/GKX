#!/usr/bin/env python3
"""Repository checks: one command, one subcommand per gate.

Usage::

    python scripts/check.py <subcommand> [arguments...]
    python scripts/check.py --list

Each subcommand runs one checker module from ``scripts/checks/`` with the
remaining arguments, exactly as if that module had been executed directly, so
every gate keeps its own arguments, exit status and output. The repo-hygiene CI
job runs these before anything is installed, so this file and every checker it
dispatches to must import with the standard library alone (``python -S``).
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

SCRIPTS_DIR = Path(__file__).resolve().parent
CHECKS_DIR = SCRIPTS_DIR / "checks"

# subcommand -> (checker module, one-line purpose)
SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "size": (
        "check_repository_size_manifest",
        "tracked-file size policy, release-artifact manifest, preview compression",
    ),
    "architecture": (
        "check_package_architecture_manifest",
        "package architecture, topology and line budgets; repository inventory",
    ),
    "parallel-scaling": (
        "check_parallel_scaling_artifacts",
        "performance manifest and parallel-scaling artifacts",
    ),
    "quasilinear": (
        "check_quasilinear_promotion_guardrails",
        "quasilinear promotion guardrails and calibration inputs",
    ),
    "vmec-boozer": (
        "check_vmec_boozer_gates",
        "VMEC/Boozer differentiability-claim and holdout gates",
    ),
    "readiness": (
        "check_release_readiness",
        "version, technical status and release readiness",
    ),
    "validation-coverage": (
        "check_validation_coverage_manifest",
        "validation coverage manifest and gate index",
    ),
    "test-gates": (
        "run_test_gates",
        "fast test gate and sharded wide-coverage runner",
    ),
    "nonlinear-transport": (
        "check_nonlinear_transport_gates",
        "nonlinear transport completion, convergence and ensemble gates",
    ),
}


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


def _usage() -> str:
    width = max(len(name) for name in SUBCOMMANDS)
    rows = "\n".join(
        f"  {name.ljust(width)}  {purpose}"
        for name, (_, purpose) in SUBCOMMANDS.items()
    )
    return (
        "usage: python scripts/check.py <subcommand> [arguments...]\n\n"
        f"subcommands:\n{rows}\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "--list"}:
        print(_usage(), end="")
        return 0
    name, rest = args[0], args[1:]
    if name not in SUBCOMMANDS:
        print(f"unknown subcommand {name!r}\n\n{_usage()}", end="", file=sys.stderr)
        return 2
    module, _ = SUBCOMMANDS[name]
    return run_module(CHECKS_DIR / f"{module}.py", rest)


if __name__ == "__main__":
    raise SystemExit(main())
