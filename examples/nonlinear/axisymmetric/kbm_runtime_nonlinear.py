#!/usr/bin/env python3
"""KBM nonlinear turbulence (short window) from a runtime TOML.

Integrates the config-backed electromagnetic kinetic-ballooning-mode case
(``runtime_kbm_nonlinear_short.toml``: 32x32x96 grid, 200 steps), prints the
final energies and fluxes, and saves the configured runtime artifacts.  Takes
on the order of ten minutes on a CPU, about a minute on a GPU.
"""

from __future__ import annotations

from pathlib import Path

from gkx import run_nonlinear_case

CONFIG = Path(__file__).resolve().parent / "runtime_kbm_nonlinear_short.toml"

# Overrides for the [run] block of CONFIG; None keeps the value in the TOML.
KY = None  # target ky*rho_i for the streamed diagnostics (TOML: 0.3)
NL = None  # Laguerre moments (TOML: 4)
NM = None  # Hermite moments (TOML: 8)
STEPS = None  # number of time steps (TOML: 200)
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
