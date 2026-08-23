#!/usr/bin/env python3
"""W7-X nonlinear ITG turbulence from a VMEC equilibrium, via a runtime TOML.

Runs the config-backed W7-X nonlinear case
(``runtime_w7x_nonlinear_vmec_geometry.toml``: 96x96x48 grid, adiabatic
electrons).  The runtime samples the flux-tube geometry from the tracked W7-X
VMEC ``wout`` on first use and caches the generated ``*.eik.nc`` file, then
prints the final energies and fluxes and saves the configured artifacts.
Expect tens of minutes on a CPU, a few minutes on a GPU.
"""

from __future__ import annotations

from pathlib import Path

from gkx import run_nonlinear_case

CONFIG = Path(__file__).resolve().parent / "runtime_w7x_nonlinear_vmec_geometry.toml"

# Overrides for the [run] block of CONFIG; None keeps the value in the TOML.
KY = None  # target ky*rho_i for the streamed diagnostics
NL = None  # Laguerre moments (TOML: 4)
NM = None  # Hermite moments (TOML: 8)
STEPS = None  # number of time steps
DT = None  # maximum time step; the runtime applies CFL control (TOML: 0.1)

run_nonlinear_case(CONFIG, ky=KY, Nl=NL, Nm=NM, steps=STEPS, dt=DT)
