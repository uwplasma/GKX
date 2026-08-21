#!/usr/bin/env python3
"""Run the W7-X nonlinear ITG case from an imported sampled geometry file."""

from __future__ import annotations

import argparse

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


def build_w7x_nonlinear_cfg(
    geometry_file: str, *, dt: float, t_max: float
) -> RuntimeConfig:
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
            method="rk3",
            use_diffrax=False,
            fixed_dt=False,
            sample_stride=50,
            diagnostics_stride=50,
            cfl=1.0,
        ),
        geometry=GeometryConfig(model="vmec-eik", geometry_file=geometry_file),
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the imported-geometry W7-X nonlinear ITG example."
    )
    parser.add_argument(
        "--geometry-file",
        required=True,
        help="Path to the imported *.eik.nc geometry file",
    )
    parser.add_argument(
        "--ky", type=float, default=1.0 / 21.0, help="Target ky mode for diagnostics"
    )
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Maximum time step. The runtime uses adaptive CFL control by default.",
    )
    parser.add_argument("--t-max", type=float, default=200.0, help="Final time")
    parser.add_argument(
        "--steps", type=int, default=None, help="Optional explicit step count override"
    )
    args = parser.parse_args()

    cfg = build_w7x_nonlinear_cfg(
        args.geometry_file, dt=float(args.dt), t_max=float(args.t_max)
    )
    steps = int(args.steps) if args.steps is not None else None
    result = run_runtime_nonlinear(
        cfg,
        ky_target=float(args.ky),
        Nl=int(args.Nl),
        Nm=int(args.Nm),
        dt=float(args.dt),
        steps=steps,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
