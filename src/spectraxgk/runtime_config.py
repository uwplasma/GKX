"""Unified runtime configuration schema for linear/nonlinear GK runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig


@dataclass(frozen=True)
class RuntimeSpeciesConfig:
    """Single species definition for runtime-configured simulations."""

    name: str = "ion"
    charge: float = 1.0
    mass: float = 1.0
    density: float = 1.0
    temperature: float = 1.0
    tprim: float = 2.49
    fprim: float = 0.8
    nu: float = 0.0
    kinetic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimePhysicsConfig:
    """Physics-family toggles independent from benchmark case names."""

    reduced_model: str = "gyrokinetic"
    linear: bool = True
    nonlinear: bool = False
    electrostatic: bool = True
    electromagnetic: bool = False
    use_apar: bool = False
    use_bpar: bool = False
    adiabatic_electrons: bool = True
    adiabatic_ions: bool = False
    tau_e: float = 1.0
    tau_fac: float | None = None
    z_ion: float = 1.0
    beta: float = 0.0
    collisions: bool = True
    hypercollisions: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeCollisionConfig:
    """Collision and end-damping parameters."""

    nu_hermite: float = 1.0
    nu_laguerre: float = 2.0
    nu_hyper: float = 0.0
    p_hyper: float = 4.0
    nu_hyper_l: float = 0.0
    nu_hyper_m: float = 1.0
    nu_hyper_lm: float = 0.0
    p_hyper_l: float = 6.0
    p_hyper_m: float | None = None
    p_hyper_lm: float = 6.0
    D_hyper: float = 0.0
    p_hyper_kperp: float = 2.0
    # GX default nonlinear dissipation path: kz-proportional hypercollisions.
    hypercollisions_const: float = 0.0
    hypercollisions_kz: float = 1.0
    damp_ends_amp: float = 0.1
    damp_ends_widthfrac: float = 0.125
    damp_ends_scale_by_dt: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeNormalizationConfig:
    """Normalization contract selection + optional explicit overrides."""

    contract: str = "cyclone"
    rho_star: float | None = None
    omega_d_scale: float | None = None
    omega_star_scale: float | None = None
    diagnostic_norm: str = "gx"
    flux_scale: float = 1.0
    wphi_scale: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeTermsConfig:
    """Term toggles for assembly; applies to linear and nonlinear paths."""

    streaming: float = 1.0
    mirror: float = 1.0
    curvature: float = 1.0
    gradb: float = 1.0
    diamagnetic: float = 1.0
    collisions: float = 1.0
    hypercollisions: float = 1.0
    hyperdiffusion: float = 0.0
    end_damping: float = 1.0
    apar: float = 1.0
    bpar: float = 1.0
    nonlinear: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeExpertConfig:
    """Advanced runtime controls that should rarely be needed."""

    fixed_mode: bool = False
    iky_fixed: int | None = None
    ikx_fixed: int | None = None
    dealias_kz: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeOutputConfig:
    """Artifact-output controls for runtime CLI entry points."""

    path: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeConfig:
    """Unified simulation config for runtime-driven GK runs."""

    grid: GridConfig = GridConfig()
    time: TimeConfig = TimeConfig()
    geometry: GeometryConfig = GeometryConfig()
    init: InitializationConfig = InitializationConfig()
    species: Tuple[RuntimeSpeciesConfig, ...] = (RuntimeSpeciesConfig(),)
    physics: RuntimePhysicsConfig = RuntimePhysicsConfig()
    collisions: RuntimeCollisionConfig = RuntimeCollisionConfig()
    normalization: RuntimeNormalizationConfig = RuntimeNormalizationConfig()
    terms: RuntimeTermsConfig = RuntimeTermsConfig()
    expert: RuntimeExpertConfig = RuntimeExpertConfig()
    output: RuntimeOutputConfig = RuntimeOutputConfig()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "init": self.init.to_dict(),
            "species": [s.to_dict() for s in self.species],
            "physics": self.physics.to_dict(),
            "collisions": self.collisions.to_dict(),
            "normalization": self.normalization.to_dict(),
            "terms": self.terms.to_dict(),
            "expert": self.expert.to_dict(),
            "output": self.output.to_dict(),
        }
