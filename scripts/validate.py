#!/usr/bin/env python3
"""Scientific validation: one command for campaigns and evidence builders.

Usage::

    python scripts/validate.py <module> [arguments...]
    python scripts/validate.py --list

Runs one module from ``scripts/campaigns/`` and ``scripts/artifacts/`` with the remaining arguments, exactly as if
that module had been executed directly. ``campaigns`` holds the validation
campaign logic (convergence, replicates, gradient evidence, admission);
``artifacts`` holds the builders of the tracked gate JSONs and figures under
``docs/_static``.
"""

from __future__ import annotations

import sys

from _command import main

if __name__ == "__main__":
    raise SystemExit(main("validate", ("campaigns", "artifacts"), sys.argv[1:]))
