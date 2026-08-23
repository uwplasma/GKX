#!/usr/bin/env python3
"""Kinetic Alfvén wave linear frequency from a runtime TOML.

Integrates the config-backed electromagnetic KAW case (``runtime_kaw.toml``)
and prints the fitted ``ky``, ``gamma``, and ``omega``.  The grid is tiny, so
the run takes only a few seconds.
"""

from __future__ import annotations

from pathlib import Path

from gkx import run_linear_case

CONFIG = Path(__file__).resolve().parent / "runtime_kaw.toml"

# Overrides for the [run] block of CONFIG; None keeps the value in the TOML.
KY = None  # binormal wavenumber ky*rho_i (TOML: 0.01)
NL = None  # Laguerre moments (TOML: 16)
NM = None  # Hermite moments (TOML: 64)
SOLVER = None  # "krylov", "time", or "explicit_time" (TOML: "explicit_time")
DT = None  # time step
STEPS = None  # number of time steps
SAMPLE_STRIDE = None  # stride between stored time samples

run_linear_case(
    CONFIG,
    ky=KY,
    Nl=NL,
    Nm=NM,
    solver=SOLVER,
    dt=DT,
    steps=STEPS,
    sample_stride=SAMPLE_STRIDE,
)
