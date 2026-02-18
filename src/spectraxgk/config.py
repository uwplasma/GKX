from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class GridConfig:
    """Spectral grid configuration in a flux-tube."""

    Nx: int = 48
    Ny: int = 48
    Nz: int = 64
    Lx: float = 62.8
    Ly: float = 62.8
    z_min: float = -3.141592653589793
    z_max: float = 3.141592653589793
    y0: float | None = None
    ntheta: int | None = None
    nperiod: int | None = None
    zp: int | None = None

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TimeConfig:
    """Time integration parameters."""

    t_max: float = 100.0
    dt: float = 0.1
    method: str = "rk2"
    sample_stride: int = 1
    checkpoint: bool = False
    implicit_restart: int = 20
    implicit_preconditioner: str | None = None
    implicit_solve_method: str = "batched"
    use_diffrax: bool = True
    diffrax_solver: str = "Heun"
    diffrax_adaptive: bool = False
    diffrax_rtol: float = 1.0e-5
    diffrax_atol: float = 1.0e-7
    diffrax_max_steps: int = 4096
    progress_bar: bool = True

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class GeometryConfig:
    """Simple analytic s-alpha geometry parameters."""

    q: float = 1.4
    s_hat: float = 0.8
    epsilon: float = 0.18
    R0: float = 1.0
    B0: float = 1.0
    alpha: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    """Dimensionless gradients for the Cyclone base case."""

    R_over_LTi: float = 2.49
    R_over_LTe: float = 0.0
    R_over_Ln: float = 0.8
    nu_i: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CycloneBaseCase:
    """Standard parameters for the Cyclone base case ITG benchmark."""

    grid: GridConfig = GridConfig(
        Nx=1,
        Ny=24,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig(t_max=8.0, dt=0.01, method="rk2")
    geometry: GeometryConfig = GeometryConfig(R0=2.77778)
    model: ModelConfig = ModelConfig()

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
        }


@dataclass(frozen=True)
class ETGModelConfig:
    """Dimensionless gradients and ratios for a canonical ETG setup."""

    R_over_LTi: float = 2.49
    R_over_LTe: float = 2.49
    R_over_Ln: float = 0.8
    Te_over_Ti: float = 1.0
    mass_ratio: float = 3670.0
    nu_i: float = 1.0e-2
    nu_e: float = 1.65e-4

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ETGBaseCase:
    """Parameters for a reduced ETG linear benchmark."""

    grid: GridConfig = GridConfig(
        Nx=1,
        Ny=24,
        Nz=96,
        Lx=6.28,
        Ly=6.28,
        y0=0.2,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig(
        t_max=2.0,
        dt=0.01,
        diffrax_solver="Tsit5",
        diffrax_adaptive=True,
        diffrax_rtol=1.0e-4,
        diffrax_atol=1.0e-7,
        diffrax_max_steps=20000,
    )
    geometry: GeometryConfig = GeometryConfig(R0=2.77778)
    model: ETGModelConfig = ETGModelConfig()

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
        }


@dataclass(frozen=True)
class KineticElectronModelConfig:
    """Gradients and ratios for a kinetic-electron Cyclone-base-case setup."""

    R_over_LTi: float = 2.49
    R_over_LTe: float = 2.49
    R_over_Ln: float = 0.8
    Te_over_Ti: float = 1.0
    mass_ratio: float = 3670.0
    nu_i: float = 1.0e-2
    nu_e: float = 1.65e-4
    beta: float = 1.0e-5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class KineticElectronBaseCase:
    """Parameters for kinetic-electron Cyclone benchmarks."""

    grid: GridConfig = GridConfig(
        Nx=1,
        Ny=24,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig(
        t_max=8.0,
        dt=0.01,
        diffrax_solver="Tsit5",
        diffrax_adaptive=True,
        diffrax_rtol=1.0e-4,
        diffrax_atol=1.0e-7,
        diffrax_max_steps=20000,
    )
    geometry: GeometryConfig = GeometryConfig(R0=2.77778)
    model: KineticElectronModelConfig = KineticElectronModelConfig()

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
        }


@dataclass(frozen=True)
class KBMBaseCase:
    """Parameters for an electromagnetic KBM benchmark."""

    grid: GridConfig = GridConfig(
        Nx=1,
        Ny=12,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        y0=10.0,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig(
        t_max=6.0,
        dt=0.01,
        diffrax_solver="Tsit5",
        diffrax_adaptive=True,
        diffrax_rtol=1.0e-4,
        diffrax_atol=1.0e-7,
        diffrax_max_steps=20000,
    )
    geometry: GeometryConfig = GeometryConfig(R0=2.77778)
    model: KineticElectronModelConfig = KineticElectronModelConfig(beta=0.015)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
        }


@dataclass(frozen=True)
class TEMModelConfig:
    """Parameters for a trapped-electron-mode benchmark."""

    R_over_LTi: float = 20.0
    R_over_LTe: float = 20.0
    R_over_Ln: float = 20.0
    Te_over_Ti: float = 1.0
    mass_ratio: float = 370.0
    nu_i: float = 0.0
    nu_e: float = 0.0
    beta: float = 1.0e-4

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TEMBaseCase:
    """Parameters for the TEM case in GX validation (s-alpha)."""

    grid: GridConfig = GridConfig(
        Nx=1,
        Ny=24,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig(
        t_max=8.0,
        dt=0.01,
        diffrax_solver="Tsit5",
        diffrax_adaptive=True,
        diffrax_rtol=1.0e-4,
        diffrax_atol=1.0e-7,
        diffrax_max_steps=20000,
    )
    geometry: GeometryConfig = GeometryConfig(q=2.7, s_hat=0.5, epsilon=0.18, R0=2.77778)
    model: TEMModelConfig = TEMModelConfig()

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
        }
