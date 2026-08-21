#!/usr/bin/env python3
"""Run the HSX linear ITG case from an imported geometry file."""

from __future__ import annotations

import argparse

from gkx.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from gkx.runtime import run_runtime_linear
from gkx.workflows.runtime.config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)


def build_hsx_cfg(geometry_file: str, *, dt: float, t_max: float) -> RuntimeConfig:
    return RuntimeConfig(
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
            t_max=t_max,
            dt=dt,
            method="rk4",
            use_diffrax=False,
            fixed_dt=True,
            sample_stride=1,
        ),
        geometry=GeometryConfig(model="imported-netcdf", geometry_file=geometry_file),
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the imported-geometry HSX linear ITG example."
    )
    parser.add_argument(
        "--geometry-file", required=True, help="Path to the HSX *.eik.nc geometry file"
    )
    parser.add_argument("--ky", type=float, default=1.0 / 21.0, help="Target ky mode")
    parser.add_argument("--Nl", type=int, default=8)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument(
        "--solver",
        choices=["explicit_time", "krylov", "time", "auto"],
        default="explicit_time",
    )
    parser.add_argument("--dt", type=float, default=0.005, help="Fixed time step")
    parser.add_argument("--t-max", type=float, default=2.0, help="Final time")
    args = parser.parse_args()

    cfg = build_hsx_cfg(args.geometry_file, dt=float(args.dt), t_max=float(args.t_max))
    steps = int(round(float(args.t_max) / float(args.dt)))
    result = run_runtime_linear(
        cfg,
        ky_target=float(args.ky),
        Nl=int(args.Nl),
        Nm=int(args.Nm),
        solver=str(args.solver),
        dt=float(args.dt),
        steps=steps,
    )
    print(f"ky={result.ky:.6f} gamma={result.gamma:.8f} omega={result.omega:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
