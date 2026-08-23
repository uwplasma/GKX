#!/usr/bin/env python3
"""HSX nonlinear ITG turbulence from a VMEC ``wout`` file.

With ``VMEC_FILE = None`` (the default) the config-backed wrapper runs the
tracked ``runtime_hsx_nonlinear_vmec_geometry.toml`` case, which samples the
flux-tube geometry from the tracked HSX ``wout`` and caches the generated
``*.eik.nc`` file.  Point ``VMEC_FILE`` at a VMEC ``wout`` to build the same
runtime configuration explicitly in this file instead.  Either path prints the
final energies and fluxes; expect on the order of ten minutes on a CPU for a
200-step window, about a minute on a GPU.

``build_hsx_nonlinear_cfg`` and ``main`` are imported by the integration
tests, so the run itself stays under the ``__main__`` guard.
"""

from __future__ import annotations

from pathlib import Path

from gkx.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from gkx import run_nonlinear_case, run_runtime_nonlinear
from gkx.workflows.runtime.config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)

CONFIG = Path(__file__).resolve().parent / "runtime_hsx_nonlinear_vmec_geometry.toml"

VMEC_FILE = None  # VMEC wout path for the manual builder path; None runs CONFIG
GEOMETRY_FILE = None  # optional output/reuse path for the generated *.eik.nc file
GEOMETRY_HELPER_REPO = None  # optional helper repository for geometry generation
GEOMETRY_HELPER_PYTHON = None  # optional interpreter for geometry generation
TORFLUX = 0.64  # normalized toroidal flux surface label
ALPHA = 0.0  # field-line alpha label
NPOL = 1.0  # number of poloidal turns
KY = 1.0 / 21.0  # target ky*rho_i mode for the streamed diagnostics
NL = 4  # Laguerre moments
NM = 8  # Hermite moments
DT = 0.1  # maximum time step; the runtime applies adaptive CFL control
T_MAX = 200.0  # final time
STEPS = None  # optional explicit step-count override


def build_hsx_nonlinear_cfg(
    vmec_file: str,
    *,
    geometry_file: str | None,
    geometry_helper_repo: str | None,
    geometry_helper_python: str | None,
    torflux: float,
    alpha: float,
    npol: float,
    dt: float,
    t_max: float,
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
        geometry=GeometryConfig(
            model="vmec",
            vmec_file=vmec_file,
            geometry_file=geometry_file,
            geometry_helper_repo=geometry_helper_repo,
            geometry_helper_python=geometry_helper_python,
            torflux=torflux,
            alpha=alpha,
            npol=npol,
        ),
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
    if VMEC_FILE is None:
        return run_nonlinear_case(
            CONFIG,
            ky=KY,
            Nl=NL,
            Nm=NM,
            steps=STEPS,
            dt=DT,
        )

    cfg = build_hsx_nonlinear_cfg(
        VMEC_FILE,
        geometry_file=GEOMETRY_FILE,
        geometry_helper_repo=GEOMETRY_HELPER_REPO,
        geometry_helper_python=GEOMETRY_HELPER_PYTHON,
        torflux=float(TORFLUX),
        alpha=float(ALPHA),
        npol=float(NPOL),
        dt=float(DT),
        t_max=float(T_MAX),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
