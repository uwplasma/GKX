#!/usr/bin/env python3
"""HSX nonlinear ITG turbulence from an imported sampled geometry file.

Builds the runtime configuration for the adiabatic-electron HSX flux tube
explicitly in this file, reads the field-line geometry from an imported
``*.eik.nc`` file, integrates the nonlinear case with adaptive CFL control,
and prints the final energies and fluxes.  Expect on the order of ten minutes
on a CPU for a 200-step window, about a minute on a GPU.
"""

from __future__ import annotations

from gkx.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from gkx.runtime import run_runtime_nonlinear
from gkx.workflows.runtime.config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)

# Path to the imported *.eik.nc geometry file (edit to point at your file).
GEOMETRY_FILE = "hsx_nonlinear.eik.nc"
KY = 1.0 / 21.0  # target ky*rho_i mode for the streamed diagnostics
NL = 4  # Laguerre moments
NM = 8  # Hermite moments
DT = 0.1  # maximum time step; the runtime applies adaptive CFL control
T_MAX = 200.0  # final time
STEPS = None  # optional explicit step-count override (None derives from T_MAX)

cfg = RuntimeConfig(
    grid=GridConfig(
        Nx=96,
        Ny=96,
        Nz=48,
        Lx=62.8,
        Ly=62.8,
        boundary="fix aspect",
        y0=21.0,
        ntheta=48,
        nperiod=1,
    ),
    time=TimeConfig(
        t_max=T_MAX,
        dt=DT,
        method="rk3",
        fixed_dt=False,
        sample_stride=50,
        diagnostics_stride=50,
        cfl=1.0,
    ),
    geometry=GeometryConfig(model="imported-netcdf", geometry_file=GEOMETRY_FILE),
    init=InitializationConfig(
        init_field="density",
        init_amp=1.0e-3,
        gaussian_init=False,
        init_single=False,
    ),
    species=(
        RuntimeSpeciesConfig(
            name="ion",
            charge=1.0,
            mass=1.0,
            density=1.0,
            temperature=1.0,
            tprim=3.0,
            fprim=1.0,
            nu=0.01,
        ),
    ),
    physics=RuntimePhysicsConfig(
        linear=False,
        nonlinear=True,
        adiabatic_electrons=True,
        tau_e=1.0,
        electrostatic=True,
        electromagnetic=False,
        use_apar=False,
        use_bpar=False,
        beta=0.0,
        collisions=True,
        hypercollisions=True,
    ),
    collisions=RuntimeCollisionConfig(
        damp_ends_amp=0.1,
        damp_ends_widthfrac=1.0 / 8.0,
        D_hyper=0.05,
    ),
    normalization=RuntimeNormalizationConfig(
        contract="kinetic", diagnostic_norm="rho_star"
    ),
    terms=RuntimeTermsConfig(
        apar=0.0,
        bpar=0.0,
        end_damping=1.0,
        hypercollisions=1.0,
        hyperdiffusion=1.0,
        nonlinear=1.0,
    ),
)

result = run_runtime_nonlinear(
    cfg,
    ky_target=float(KY),
    Nl=int(NL),
    Nm=int(NM),
    dt=float(DT),
    steps=None if STEPS is None else int(STEPS),
    resolved_diagnostics=False,
)
if result.diagnostics is None or result.ky_selected is None:
    raise RuntimeError("Nonlinear runtime did not produce diagnostics")
print(
    "ky={:.6f} Wg={:.8e} Wphi={:.8e} heat={:.8e} pflux={:.8e}".format(
        float(result.ky_selected),
        float(result.diagnostics.Wg_t[-1]),
        float(result.diagnostics.Wphi_t[-1]),
        float(result.diagnostics.heat_flux_t[-1]),
        float(result.diagnostics.particle_flux_t[-1]),
    )
)
