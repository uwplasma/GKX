"""Benchmark-family configuration presets.

These dataclasses are public through :mod:`spectraxgk.config`, but their
implementation lives with the validation benchmark owners.  Keeping them here
separates user/runtime schema objects from literature benchmark operating
points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from spectraxgk.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    REFERENCE_MASS_RATIO,
    TimeConfig,
)


@dataclass(frozen=True)
class ModelConfig:
    """Dimensionless gradients for the Cyclone base case."""

    R_over_LTi: float = 2.49
    R_over_LTe: float = 0.0
    R_over_Ln: float = 0.8
    nu_i: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
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
        boundary="linked",
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig(
        t_max=150.0,
        dt=0.01,
        method="rk4",
        use_diffrax=True,
        diffrax_solver="Dopri8",
        diffrax_adaptive=True,
        diffrax_rtol=1.0e-6,
        diffrax_atol=1.0e-8,
        diffrax_max_steps=200000,
        fixed_dt=False,
        dt_max=0.05,
    )
    geometry: GeometryConfig = GeometryConfig(
        R0=2.77778,
        drift_scale=1.0,
    )
    model: ModelConfig = ModelConfig()
    init: InitializationConfig = InitializationConfig(
        init_field="density",
        init_amp=1.0e-10,
        gaussian_init=True,
        gaussian_width=0.5,
        gaussian_envelope_constant=1.0,
        gaussian_envelope_sine=0.0,
    )
    reference_aligned: bool = True

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
            "init": self.init.to_dict(),
            "reference_alignment": {"enabled": self.reference_aligned},
        }


@dataclass(frozen=True)
class ETGModelConfig:
    """Dimensionless gradients and ratios for a canonical ETG setup."""

    R_over_LTi: float = 2.49
    R_over_LTe: float = 2.49
    R_over_Ln: float = 0.8
    R_over_Lni: float | None = None
    R_over_Lne: float | None = None
    Te_over_Ti: float = 1.0
    mass_ratio: float = REFERENCE_MASS_RATIO
    nu_i: float = 0.0
    nu_e: float = 0.0
    beta: float = 1.0e-5
    adiabatic_ions: bool = True

    def to_dict(self) -> Dict[str, Any]:
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
        boundary="linked",
        y0=0.2,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig(
        t_max=10.0,
        dt=0.05,
        diffrax_solver="Dopri8",
        diffrax_adaptive=True,
        diffrax_rtol=1.0e-5,
        diffrax_atol=1.0e-7,
        diffrax_max_steps=200000,
    )
    geometry: GeometryConfig = GeometryConfig(R0=2.77778)
    model: ETGModelConfig = ETGModelConfig()
    init: InitializationConfig = InitializationConfig(
        init_field="density",
        init_amp=1.0e-10,
        gaussian_init=True,
    )

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
            "init": self.init.to_dict(),
        }


@dataclass(frozen=True)
class KineticElectronModelConfig:
    """Gradients and ratios for a kinetic-electron Cyclone-base-case setup."""

    R_over_LTi: float = 2.49
    R_over_LTe: float = 2.49
    R_over_Ln: float = 0.8
    Te_over_Ti: float = 1.0
    mass_ratio: float = REFERENCE_MASS_RATIO
    nu_i: float = 0.0
    nu_e: float = 0.0
    beta: float = 1.0e-5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KineticElectronBaseCase:
    """Parameters for kinetic-electron Cyclone benchmarks."""

    grid: GridConfig = GridConfig(
        Nx=1,
        Ny=16,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        boundary="linked",
        y0=10.0,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig(
        t_max=40.0,
        dt=0.01,
        method="rk4",
        diffrax_solver="Tsit5",
        diffrax_adaptive=True,
        diffrax_rtol=1.0e-4,
        diffrax_atol=1.0e-7,
        diffrax_max_steps=20000,
    )
    geometry: GeometryConfig = GeometryConfig(R0=2.77778)
    model: KineticElectronModelConfig = KineticElectronModelConfig()
    init: InitializationConfig = InitializationConfig(
        init_field="density",
        init_amp=1.0e-10,
        gaussian_init=True,
    )

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
            "init": self.init.to_dict(),
        }


@dataclass(frozen=True)
class KBMBaseCase:
    """Parameters for an electromagnetic KBM benchmark."""

    grid: GridConfig = GridConfig(
        Nx=1,
        Ny=16,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        boundary="linked",
        y0=10.0,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig(
        t_max=40.0,
        dt=0.01,
        method="rk4",
        diffrax_solver="Tsit5",
        diffrax_adaptive=True,
        diffrax_rtol=1.0e-4,
        diffrax_atol=1.0e-7,
        diffrax_max_steps=20000,
    )
    geometry: GeometryConfig = GeometryConfig(R0=2.77778)
    model: KineticElectronModelConfig = KineticElectronModelConfig(beta=0.015)
    init: InitializationConfig = InitializationConfig(
        init_field="all",
        init_amp=1.0e-10,
        gaussian_init=True,
    )

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
            "init": self.init.to_dict(),
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TEMBaseCase:
    """Parameters for the provisional literature-backed TEM stress case."""

    grid: GridConfig = GridConfig(
        Nx=1,
        Ny=24,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        boundary="linked",
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
    geometry: GeometryConfig = GeometryConfig(q=2.7, s_hat=0.5, epsilon=0.18, R0=1.0)
    model: TEMModelConfig = TEMModelConfig()
    init: InitializationConfig = InitializationConfig(
        init_field="density",
        init_amp=1.0e-10,
        gaussian_init=True,
    )

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
            "init": self.init.to_dict(),
        }


__all__ = [
    "CycloneBaseCase",
    "ETGBaseCase",
    "ETGModelConfig",
    "KBMBaseCase",
    "KineticElectronBaseCase",
    "KineticElectronModelConfig",
    "ModelConfig",
    "TEMBaseCase",
    "TEMModelConfig",
]
