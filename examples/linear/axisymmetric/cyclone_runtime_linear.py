#!/usr/bin/env python3
"""Cyclone base case ITG linear growth rate from a runtime TOML.

Integrates the config-backed Cyclone ITG case (``cyclone.toml``: s-alpha
geometry, adiabatic electrons) and prints the fitted ``ky``, ``gamma``, and
``omega``.  Runs in about a minute on a laptop CPU.
"""

from __future__ import annotations

from pathlib import Path

from gkx import run_linear_case

CONFIG = Path(__file__).resolve().parent / "cyclone.toml"

# Overrides for the [run] block of CONFIG; None keeps the value in the TOML.
KY = None  # binormal wavenumber ky*rho_i (TOML: 0.3)
NL = None  # Laguerre moments (TOML: 16)
NM = None  # Hermite moments (TOML: 48)
SOLVER = None  # "krylov", "time", or "explicit_time" (TOML: "krylov")
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
