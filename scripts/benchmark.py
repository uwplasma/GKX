#!/usr/bin/env python3
"""Benchmarks: one command, one module per benchmark driver.

Usage::

    python scripts/benchmark.py <module> [arguments...]
    python scripts/benchmark.py --list

Runs one module from ``scripts/benchmarks/`` with the remaining arguments, exactly as if
that module had been executed directly. The drivers run the literature
benchmark cases and the runtime/memory and integrator benchmarks; their
input decks live in ``benchmarks/cases/``.
"""

from __future__ import annotations

import sys

from _command import main

if __name__ == "__main__":
    raise SystemExit(main("benchmark", ("benchmarks",), sys.argv[1:]))
