from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict


# Prefer stdlib tomllib; fallback to tomli for <=3.10
try:
    import tomllib # Python 3.11+
except ModuleNotFoundError: # pragma: no cover
    import pip._vendor.tomli as tomllib
    from tomllib import TOMLDecodeError


@dataclass
class SimConfig:
    mode: str = "fourier" # "fourier" | "dg"
    backend: str = "diffrax" # currently only diffrax
    tmax: float = 10.0
    nt: int = 201
    save_every: int = 1
    nonlinear: bool = False
    precision: str = "x64" # "x32" | "x64"


@dataclass
class GridConfig:
    Nn: int = 32 # Hermite order in v_||
    Nm: int = 8 # Laguerre order in v_perp
    kpar: float = 0.5
    vth: float = 1.0
    nu: float = 0.01 # LB collision frequency


@dataclass
class ICConfig:
    kind: str = "n0_mode" # "n0_mode" | "random"
    amp: float = 1e-3
    phase: float = 0.0


@dataclass
class PathsConfig:
    outdir: str = "outputs"
    outfile: str = "linear_slab_run.npz"


@dataclass
class FullConfig:
    sim: SimConfig
    grid: GridConfig
    ic: ICConfig
    paths: PathsConfig




def read_toml(path: str) -> FullConfig:
    with open(path, "rb") as f:
        try:
            raw = tomllib.load(f)
        except TOMLDecodeError as e:
            raise SystemExit(f"TOML parse error in {path}: {e}")
    sim = SimConfig(**raw.get("sim", {}))
    grid = GridConfig(**raw.get("grid", {}))
    ic = ICConfig(**raw.get("ic", {}))
    paths = PathsConfig(**raw.get("paths", {}))
    return FullConfig(sim=sim, grid=grid, ic=ic, paths=paths)