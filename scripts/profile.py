#!/usr/bin/env python3
"""Profiling runs: one command, one module per profile.

Usage::

    python scripts/profile.py <module> [arguments...]
    python scripts/profile.py --list

Runs one module from ``scripts/profiling/`` with the remaining arguments, exactly as if
that module had been executed directly. The profiles time and trace the runtime,
startup cache, linear RHS terms, parallel workloads and nonlinear sharding;
the performance manifest names which profile backs which claim.
"""

from __future__ import annotations

import sys

from _command import main

if __name__ == "__main__":
    raise SystemExit(main("profile", ("profiling",), sys.argv[1:]))
