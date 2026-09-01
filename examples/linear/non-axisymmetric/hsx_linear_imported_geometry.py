#!/usr/bin/env python3
"""HSX linear ITG growth rate from an imported geometry file.

Builds the runtime configuration for the adiabatic-electron HSX flux tube
explicitly in this file, reads the field-line geometry from an imported
``*.eik.nc`` file, integrates to ``T_MAX``, and prints the fitted ``ky``,
``gamma``, and ``omega``.  About a minute on a laptop CPU at the default
resolution.
"""

from __future__ import annotations

from gkx.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from gkx.runtime import run_runtime_linear
from gkx.config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)

# Path to the HSX *.eik.nc geometry file (edit to point at your file).
GEOMETRY_FILE = "hsx_linear.eik.nc"
KY = 1.0 / 21.0  # target ky*rho_i mode
NL = 8  # Laguerre moments
NM = 8  # Hermite moments
SOLVER = "explicit_time"  # "explicit_time", "krylov", "time", or "auto"
DT = 0.005  # fixed time step
T_MAX = 2.0  # final time

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
        method="rk4",
        fixed_dt=True,
        sample_stride=1,
    ),
    geometry=GeometryConfig(model="imported-netcdf", geometry_file=GEOMETRY_FILE),
    init=InitializationConfig(
        init_field="density",
        init_amp=1.0e-3,
        gaussian_init=False,
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
        adiabatic_electrons=True,
        tau_e=1.0,
        electrostatic=True,
        electromagnetic=False,
        use_apar=False,
        use_bpar=False,
    ),
    collisions=RuntimeCollisionConfig(
        damp_ends_amp=0.1,
        damp_ends_widthfrac=1.0 / 8.0,
        D_hyper=0.05,
    ),
    normalization=RuntimeNormalizationConfig(
        contract="kinetic", diagnostic_norm="none"
    ),
    terms=RuntimeTermsConfig(
        apar=0.0,
        bpar=0.0,
        end_damping=1.0,
        hypercollisions=1.0,
        hyperdiffusion=1.0,
        nonlinear=0.0,
    ),
)

result = run_runtime_linear(
    cfg,
    ky_target=float(KY),
    Nl=int(NL),
    Nm=int(NM),
    solver=str(SOLVER),
    dt=float(DT),
    steps=int(round(float(T_MAX) / float(DT))),
)
print(f"ky={result.ky:.6f} gamma={result.gamma:.8f} omega={result.omega:.8f}")
