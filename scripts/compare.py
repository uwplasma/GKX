#!/usr/bin/env python3
"""External-code comparison: one command, one module per protocol.

Usage::

    python scripts/compare.py <module> [arguments...]
    python scripts/compare.py --list

Runs one module from ``scripts/comparison/`` with the remaining arguments, exactly as if
that module had been executed directly. The modules compare GKX against GX outputs
(linear, KBM, nonlinear, RHS terms, exact state) and build the parity matrix
and reference panels. They need the reference code's outputs locally.
"""

from __future__ import annotations

import sys

from _command import main

if __name__ == "__main__":
    raise SystemExit(main("compare", ("comparison",), sys.argv[1:]))
