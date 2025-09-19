# io_config.py
from dataclasses import dataclass, field
from typing import Optional, Any, List, Union
import math
import re

# Prefer stdlib tomllib; fallback to tomli for <=3.10
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

@dataclass
class SpeciesCfg:
    name: str = "species"
    q: float = -1.0
    m: float = 1.0
    n0: float = 1.0
    vth: float = 1.0      # reserved (basis is still centered at 0)
    u0: float = 0.0       # drift (in v_th units)
    nu0: float = 0.0
    hyper_p: int = 0
    collide_cutoff: int = 3
    amplitude: float = 0.0
    k: float = 0.0
    seed_c1: bool = False

@dataclass
class SimCfg:
    mode: str            # "fourier" | "dg"
    backend: str         # "eig" | "diffrax"
    tmax: float
    nt: int
    nonlinear: bool = False          # <-- NEW: linear (False) vs nonlinear (True)
    dealias_frac: float = 2.0/3.0    # <-- NEW: 2/3-rule cutoff for Fourier pseudo-spectral NL

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

@dataclass
class BCCfg:
    kind: str            # "periodic" | "dirichlet" | "neumann"

@dataclass
class PlotCfg:
    nv: int = 257
    vmax: float = 6.0
    save_anim: Optional[str] = None
    fps: int = 30
    dpi: int = 150
    no_show: bool = False
    fig_width: float = 12.0
    fig_row_height: float = 3.5

@dataclass
class Config:
    sim: SimCfg
    grid: GridCfg
    hermite: HermiteCfg
    bc: BCCfg
    plot: PlotCfg
    species: List[SpeciesCfg] = field(default_factory=list) 

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
    plot    = PlotCfg(**d.get("plot", {}))  # defaults if missing
    
    species = []
    for sp in d.get("species", []):
        species.append(SpeciesCfg(**sp))

    return Config(sim=sim, grid=grid, hermite=hermite, bc=bc, plot=plot, species=species)