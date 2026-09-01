#!/usr/bin/env python3
"""W7-X linear ITG growth rate from an imported sampled geometry file.

Builds the runtime configuration for the adiabatic-electron W7-X flux tube
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

# Path to the imported *.eik.nc geometry file (edit to point at your file).
GEOMETRY_FILE = "itg_w7x_adiabatic_electrons_t2.eik.nc"
KY = 0.3  # target ky*rho_i mode
NL = 8  # Laguerre moments
NM = 16  # Hermite moments
SOLVER = "explicit_time"  # "explicit_time", "krylov", or "time"
DT = 0.005890226417991923  # time step; matches the tracked W7-X t=2 reference run
T_MAX = 2.0  # final time

cfg = RuntimeConfig(
    grid=GridConfig(Nx=1, Ny=82, Nz=256, Lx=62.8, Ly=62.8, boundary="linked", y0=10.0),
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
        init_amp=1.0e-10,
        gaussian_init=True,
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
    ),
    normalization=RuntimeNormalizationConfig(
        contract="kinetic", diagnostic_norm="none"
    ),
    terms=RuntimeTermsConfig(
        apar=0.0,
        bpar=0.0,
        end_damping=1.0,
        hypercollisions=1.0,
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
print(f"ky={result.ky:.4f} gamma={result.gamma:.8f} omega={result.omega:.8f}")
