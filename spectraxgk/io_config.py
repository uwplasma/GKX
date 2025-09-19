# io_config.py
from dataclasses import dataclass
from typing import Optional, Any, Dict
import math
import re

# Prefer stdlib tomllib; fallback to tomli for <=3.10
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

@dataclass
class SimCfg:
    mode: str            # "fourier" | "dg"
    backend: str         # "eig" | "diffrax"
    tmax: float
    nt: int

@dataclass
class GridCfg:
    kmin: Optional[float] = None
    kmax: Optional[float] = None
    Nk:   Optional[int]   = None
    L:    Optional[float] = None
    Nx:   Optional[int]   = None

@dataclass
class HermiteCfg:
    N: int
    nu0: float = 0.0
    hyper_p: int = 0
    collide_cutoff: int = 3

@dataclass
class BCCfg:
    kind: str            # "periodic" | "dirichlet" | "neumann"

@dataclass
class InitCfg:
    type: str            # "landau" | "two_stream"
    amplitude: float
    k: Optional[float] = None
    shift: Optional[float] = None
    seed_c1: bool = False

@dataclass
class PlotCfg:
    nv: int = 257
    vmax: float = 6.0
    save_anim: Optional[str] = None
    fps: int = 30
    dpi: int = 150
    no_show: bool = False

@dataclass
class Config:
    sim: SimCfg
    grid: GridCfg
    hermite: HermiteCfg
    bc: BCCfg
    init: InitCfg
    plot: PlotCfg

# ---------- safe expression support: "2*pi" ----------
_ALLOWED_EXPR = re.compile(r"^[0-9\.\s\+\-\*\/\(\)piPI]+$")

def _coerce_constants(obj: Any) -> Any:
    """Recursively convert strings like '2*pi' (or 'PI/2') into floats, safely."""
    if isinstance(obj, dict):
        return {k: _coerce_constants(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_constants(x) for x in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if _ALLOWED_EXPR.match(s):
            try:
                return float(eval(s, {"__builtins__": None}, {"pi": math.pi, "PI": math.pi}))
            except Exception:
                return obj
    return obj

def read_toml(path: str) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    d = _coerce_constants(raw)

    sim     = SimCfg(**d["sim"])
    grid    = GridCfg(**d["grid"])
    hermite = HermiteCfg(**d["hermite"])
    bc      = BCCfg(**d["bc"])
    init    = InitCfg(**d["init"])
    plot    = PlotCfg(**d.get("plot", {}))  # defaults if missing

    return Config(sim=sim, grid=grid, hermite=hermite, bc=bc, init=init, plot=plot)
