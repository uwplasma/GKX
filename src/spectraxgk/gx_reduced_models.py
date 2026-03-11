"""Helpers for auditing GX reduced-model benchmark inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spectraxgk.io import load_toml


@dataclass(frozen=True)
class GXReducedModelContract:
    """Minimal parsed contract for a GX reduced-model input file."""

    model: str
    nx: int
    ny: int
    nz: int
    Nl: int
    Nm: int
    x0: float
    y0: float
    z0: float | None
    boundary: str
    dt: float
    t_max: float | None
    cfl: float
    init_field: str
    init_amp: float
    ikpar_init: int
    adiabatic_species: str | None
    tau_fac: float | None
    z_ion: float | None
    zero_shat: bool
    hyper: bool
    D_hyper: float
    dealias_kz: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_gx_reduced_model_contract(path: str | Path) -> GXReducedModelContract:
    """Parse a GX reduced-model input file into a stable contract summary."""

    data = load_toml(path)
    dims = data.get("Dimensions", {})
    domain = data.get("Domain", {})
    time = data.get("Time", {})
    init = data.get("Initialization", {})
    boltz = data.get("Boltzmann", {})
    geo = data.get("Geometry", {})
    diss = data.get("Dissipation", {})
    expert = data.get("Expert", {})
    cetg = bool(data.get("Collisional_slab_ETG", {}).get("cetg", False))
    krehm = bool(data.get("KREHM", {}).get("krehm", False))

    if cetg:
        model = "cetg"
        nl = 2
        nm = 1
    elif krehm:
        model = "krehm"
        nl = int(dims.get("nlaguerre", 1))
        nm = int(dims.get("nhermite", 2))
    else:
        raise ValueError(f"{path} is not a GX reduced-model input (expected cETG or KREHM marker)")

    t_max = time.get("t_max")
    return GXReducedModelContract(
        model=model,
        nx=int(dims["nx"]),
        ny=int(dims["ny"]),
        nz=int(dims["ntheta"]),
        Nl=nl,
        Nm=nm,
        x0=float(domain["x0"]),
        y0=float(domain["y0"]),
        z0=float(domain["z0"]) if "z0" in domain else None,
        boundary=str(domain["boundary"]),
        dt=float(time["dt"]),
        t_max=None if t_max is None else float(t_max),
        cfl=float(time.get("cfl", 1.0)),
        init_field=str(init["init_field"]),
        init_amp=float(init["init_amp"]),
        ikpar_init=int(init.get("ikpar_init", 0)),
        adiabatic_species=str(boltz["Boltzmann_type"]) if "Boltzmann_type" in boltz else None,
        tau_fac=None if "tau_fac" not in boltz else float(boltz["tau_fac"]),
        z_ion=None if "Z_ion" not in boltz else float(boltz["Z_ion"]),
        zero_shat=bool(geo.get("zero_shat", False)),
        hyper=bool(diss.get("hyper", False)),
        D_hyper=float(diss.get("D_hyper", 0.0)),
        dealias_kz=bool(expert.get("dealias_kz", False)),
    )
