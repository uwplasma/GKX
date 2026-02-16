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
    time: TimeConfig = TimeConfig()
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

    R_over_LTe: float = 6.0
    R_over_Ln: float = 0.0
    Te_over_Ti: float = 1.0
    mass_ratio: float = 100.0

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
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig()
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
class MTMModelConfig:
    """Dimensionless gradients and ratios for a reduced MTM setup."""

    R_over_LTe: float = 5.0
    R_over_Ln: float = 0.0
    Te_over_Ti: float = 1.0
    mass_ratio: float = 100.0
    nu: float = 0.2

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MTMBaseCase:
    """Parameters for a reduced MTM linear benchmark."""

    grid: GridConfig = GridConfig(
        Nx=1,
        Ny=24,
        Nz=96,
        Lx=6.28,
        Ly=6.28,
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig()
    geometry: GeometryConfig = GeometryConfig(R0=2.77778)
    model: MTMModelConfig = MTMModelConfig()

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
        }
