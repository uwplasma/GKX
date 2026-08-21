#!/usr/bin/env python3
"""Run the W7-X linear ITG case from an imported sampled geometry file."""

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


def build_w7x_cfg(geometry_file: str, *, dt: float, t_max: float) -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(
            Nx=1, Ny=82, Nz=256, Lx=62.8, Ly=62.8, boundary="linked", y0=10.0
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the imported-geometry W7-X linear ITG example."
    )
    parser.add_argument(
        "--geometry-file",
        required=True,
        help="Path to the imported *.eik.nc geometry file",
    )
    parser.add_argument("--ky", type=float, default=0.3, help="Target ky mode")
    parser.add_argument("--Nl", type=int, default=8)
    parser.add_argument("--Nm", type=int, default=16)
    parser.add_argument(
        "--solver", choices=["explicit_time", "krylov", "time"], default="explicit_time"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005890226417991923,
        help="Time step. The default matches the tracked W7-X t=2 reference run.",
    )
    parser.add_argument("--t-max", type=float, default=2.0, help="Final time")
    args = parser.parse_args()

    cfg = build_w7x_cfg(args.geometry_file, dt=float(args.dt), t_max=float(args.t_max))
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
    print(f"ky={result.ky:.4f} gamma={result.gamma:.8f} omega={result.omega:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
