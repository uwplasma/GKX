#!/usr/bin/env python3
"""Full-GK ETG nonlinear pilot case from a runtime TOML.

Integrates the config-backed nonlinear electron-temperature-gradient pilot
(``runtime_etg_nonlinear.toml``: 24x24x48 grid, 500 steps), prints the final
energies and fluxes, and saves the configured runtime artifacts.  Takes a few
minutes on a CPU.
"""

from __future__ import annotations

from pathlib import Path

from gkx import run_nonlinear_case

CONFIG = Path(__file__).resolve().parent / "runtime_etg_nonlinear.toml"

# Overrides for the [run] block of CONFIG; None keeps the value in the TOML.
KY = None  # target ky*rho_e for the streamed diagnostics (TOML: 5.0)
NL = None  # Laguerre moments (TOML: 16)
NM = None  # Hermite moments (TOML: 8)
STEPS = None  # number of time steps (TOML: 500)
DT = None  # maximum time step; the runtime applies CFL control
SAMPLE_STRIDE = None  # stride between stored time samples
DIAGNOSTICS_STRIDE = None  # stride between streamed diagnostics

run_nonlinear_case(
    CONFIG,
    ky=KY,
    Nl=NL,
    Nm=NM,
    steps=STEPS,
    dt=DT,
    sample_stride=SAMPLE_STRIDE,
    diagnostics_stride=DIAGNOSTICS_STRIDE,
)
