# io_config.py
from dataclasses import dataclass
from typing import Optional, Any
import math
import re

# --- tomllib / tomli import with your requested fallback order ---
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    try:
        import pip._vendor.tomli as tomllib  # your preferred fallback
    except Exception:
        import tomli as tomllib  # final fallback if available


@dataclass
class SimCfg:
    mode: str
    backend: str
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
    kind: str

@dataclass
class InitCfg:
    type: str
    amplitude: float
    k: Optional[float] = None
    shift: Optional[float] = None
    seed_c1: bool = False

@dataclass
class Config:
    sim: SimCfg
    grid: GridCfg
    hermite: HermiteCfg
    bc: BCCfg
    init: InitCfg


# Allow strings like "2*pi" (but not arbitrary code)
_ALLOWED_EXPR = re.compile(r"^[0-9\.\s\+\-\*\/\(\)piPI]+$")

def _coerce_constants(obj: Any) -> Any:
    """Recursively convert strings like '2*pi' into floats (safe: only pi allowed)."""
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
    """Read a TOML config. Supports quoted expressions like "2*pi" -> float."""
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")

    # Parse TOML first (no pre-checks that can false-positive)
    d = tomllib.loads(text)

    # Optionally coerce any quoted expressions like "2*pi" -> float
    d = _coerce_constants(d)

    sim = SimCfg(**d["sim"])
    grid = GridCfg(**d["grid"])
    hermite = HermiteCfg(**d["hermite"])
    bc = BCCfg(**d["bc"])
    init = InitCfg(**d["init"])
    return Config(sim=sim, grid=grid, hermite=hermite, bc=bc, init=init)
