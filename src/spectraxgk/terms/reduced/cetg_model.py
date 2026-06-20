"""Runtime contract and coefficient builder for the cETG reduced model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spectraxgk.geometry import FluxTubeGeometryLike, SlabGeometry
from spectraxgk.workflows.runtime.config import RuntimeConfig


@dataclass(frozen=True)
class CETGModelParams:
    """Collisional-slab ETG coefficients and normalization data."""

    tau_fac: float
    z_ion: float
    gradpar: float
    z0: float
    c1: float
    C12: float
    C23: float
    D_hyper: float
    nu_hyper: float
    pressure: float
    dealias_kz: bool


def _model_key(cfg: RuntimeConfig) -> str:
    return cfg.physics.reduced_model.strip().lower()


def validate_cetg_runtime_config(
    cfg: RuntimeConfig,
    geom: FluxTubeGeometryLike,
    *,
    Nl: int,
    Nm: int,
) -> None:
    """Validate that a runtime config matches the cETG model contract."""

    if _model_key(cfg) != "cetg":
        raise ValueError("cETG helpers require physics.reduced_model='cetg'")
    if int(Nl) != 2 or int(Nm) != 1:
        raise ValueError("cETG requires exactly Nl=2 and Nm=1")
    if not isinstance(geom, SlabGeometry):
        raise ValueError("cETG currently requires geometry.model='slab'")
    if not bool(cfg.physics.electrostatic) or bool(cfg.physics.electromagnetic):
        raise ValueError("cETG is electrostatic-only")
    if not bool(cfg.physics.adiabatic_ions):
        raise ValueError("cETG requires adiabatic_ions=true")
    kinetic = tuple(s for s in cfg.species if bool(s.kinetic))
    if len(kinetic) != 1:
        raise ValueError("cETG requires exactly one kinetic species")
    if float(kinetic[0].charge) >= 0.0:
        raise ValueError("cETG requires the kinetic species to be an electron")


def build_cetg_model_params(
    cfg: RuntimeConfig,
    geom: FluxTubeGeometryLike,
    *,
    Nl: int,
    Nm: int,
) -> CETGModelParams:
    """Build the cETG coefficient set from the runtime config."""

    validate_cetg_runtime_config(cfg, geom, Nl=Nl, Nm=Nm)
    if not isinstance(geom, SlabGeometry):
        raise ValueError("cETG currently requires geometry.model='slab'")
    kinetic = tuple(s for s in cfg.species if bool(s.kinetic))
    electron = kinetic[0]
    z_ion = float(cfg.physics.z_ion)
    tau_fac = float(
        cfg.physics.tau_fac if cfg.physics.tau_fac is not None else cfg.physics.tau_e
    )
    denom = 1.0 + 61.0 / (np.sqrt(128.0) * z_ion) + 9.0 / (2.0 * z_ion * z_ion)
    c1 = (
        217.0 / 64.0 + 151.0 / (np.sqrt(128.0) * z_ion) + 9.0 / (2.0 * z_ion * z_ion)
    ) / denom
    c2 = 2.5 * (33.0 / 16.0 + 45.0 / (np.sqrt(128.0) * z_ion)) / denom
    c3 = (
        25.0 / 4.0 * (13.0 / 4.0 + 45.0 / (np.sqrt(128.0) * z_ion)) / denom
        - c2 * c2 / c1
    )
    C12 = 1.0 + c2 / c1
    C23 = c3 / c1 + C12 * C12
    D_hyper = float(
        cfg.collisions.D_hyper if float(cfg.terms.hyperdiffusion) != 0.0 else 0.0
    )
    nu_hyper = (
        float(cfg.collisions.nu_hyper) if float(cfg.collisions.nu_hyper) > 0.0 else 2.0
    )
    return CETGModelParams(
        tau_fac=tau_fac,
        z_ion=z_ion,
        gradpar=float(geom.gradpar()),
        z0=float(geom.z0)
        if geom.z0 is not None
        else float(1.0 / float(geom.gradpar())),
        c1=float(c1),
        C12=float(C12),
        C23=float(C23),
        D_hyper=D_hyper,
        nu_hyper=float(nu_hyper),
        pressure=float(electron.density * electron.temperature),
        dealias_kz=bool(cfg.expert.dealias_kz),
    )


__all__ = [
    "CETGModelParams",
    "build_cetg_model_params",
    "validate_cetg_runtime_config",
]
