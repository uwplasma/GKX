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
import sys

from _command import run_module

CHECKS_DIR = Path(__file__).resolve().parent / "checks"

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
    "nonlinear-optimization": (
        "check_nonlinear_optimization_gates",
        "nonlinear optimization production-guard and gradient-evidence gates",
    ),
}


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
