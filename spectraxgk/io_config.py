# io_config.py
import math
import re
from dataclasses import dataclass, field
from typing import Any

from jax import numpy as jnp

# Prefer stdlib tomllib; fallback to tomli for <=3.10
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import pip._vendor.tomli as tomllib
from tomllib import TOMLDecodeError

from spectraxgk.constants import (
    elementary_charge as e_charge,
)
from spectraxgk.constants import (
    mass_electron,
    mass_proton,
)
from spectraxgk.constants import (
    speed_of_light as c_light,
)


@dataclass
class SpeciesCfg:
    name: str = "species"

    # --- physical charge & density (as before) ---
    q: float = -1.0
    n0: float = 1.0

    # --- NEW: user-friendly inputs ---
    # choose which base mass to scale from: "electron" or "proton"
    mass_base: str = "electron"  # "electron" | "proton"
    mass_multiple: float = 1.0  # total mass = mass_multiple * base_mass

    temperature_eV: float | None = None  # noqa: N815 # replaces 'vth' as user input (T in eV)
    drift_c: float | None = None  # replaces 'u0' as user input (u/c)

    # --- collisions (per-species, unchanged) ---
    nu0: float = 0.0
    hyper_p: int = 0
    collide_cutoff: int = 3

    # --- IC knobs (per-species) ---
    amplitude: float = 0.0
    k: float = 0.0
    seed_c1: bool = False

    # --- derived values used by solvers (kept to avoid refactors) ---
    m: float = 9.10938371e-31  # SI mass set at load (default: m_e)
    vth: float = 1.0  # SI thermal speed set at load (sqrt(2 kB T / m))
    u0: float = 0.0  # SI drift speed set at load (drift_c * c)


@dataclass
class SimCfg:
    mode: str  # "fourier" | "dg"
    backend: str  # "eig" | "diffrax"
    tmax: float
    nt: int
    nonlinear: bool = False  # <-- NEW: linear (False) vs nonlinear (True)
    dealias_frac: float = 2.0 / 3.0  # <-- NEW: 2/3-rule cutoff for Fourier pseudo-spectral NL


@dataclass
class GridCfg:
    kmin: float | None = None
    kmax: float | None = None
    Nk: int | None = None
    Nx: int | None = None
    # Physical box length (meters) OR multiples of Debye length (dimensionless)
    L: float | None = None  # meters
    L_lambdaD: float | None = None  # dimensionless, multiples of λ_D (electron Debye length)
    debye_species: int | str | None = None  # which species to use for λ_D (default: first q<0)


@dataclass
class HermiteCfg:
    N: int


@dataclass
class BCCfg:
    kind: str  # "periodic" | "dirichlet" | "neumann"


@dataclass
class PlotCfg:
    nv: int = 257
    vmin_c: float | None = None  # lower bound in units of c
    vmax_c: float | None = None  # upper bound in units of c
    save_anim: str | None = None
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
    species: list[SpeciesCfg] = field(default_factory=list)


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


def _finalize_species_units(sp: SpeciesCfg) -> SpeciesCfg:
    # mass: base * multiple
    base = sp.mass_base.strip().lower()
    if base not in ("electron", "proton"):
        raise ValueError(f"species[{sp.name}].mass_base must be 'electron' or 'proton'")
    m0 = mass_electron if base == "electron" else mass_proton
    sp.m = float(sp.mass_multiple) * m0

    # vth: sqrt(2 T / m), where T (J) = temperature_eV * e_charge
    if sp.temperature_eV is None:
        sp.vth = 0.0
    else:
        T_J = float(sp.temperature_eV) * e_charge
        sp.vth = float(jnp.sqrt(2.0 * T_J / sp.m))

    # drift: u0 = drift_c * c
    if sp.drift_c is not None:
        if not (-1.0 < float(sp.drift_c) < 1.0):
            raise ValueError(f"{sp.name}.drift_c must be in (-1,1); got {sp.drift_c}")
        sp.u0 = float(sp.drift_c) * c_light
    else:
        sp.u0 = 0.0

    return sp


def _infer_mass_defaults_from_name(sp: SpeciesCfg) -> SpeciesCfg:
    nm = sp.name.strip().lower()
    if "electron" in nm or nm in ("e", "electrons"):
        sp.mass_base = "electron"
        if sp.mass_multiple == 1.0:
            sp.mass_multiple = 1.0
    if "ion" in nm or "proton" in nm:
        sp.mass_base = "proton"
        if sp.mass_multiple == 1.0:
            sp.mass_multiple = 1.0
    return sp


def _preprocess_toml_expressions(text: str) -> str:
    """
    Allow bare arithmetic like: n0 = 0.5*10000 or L = 2*pi by rewriting to strings:
      n0 = "0.5*10000"
      L  = "2*pi"
    Rules:
      - Only rewrites single-line assignments of the form: <key> = <value>
      - Skips lines that are tables ([...]), arrays ({...}/[...]), quoted strings, booleans
      - Leaves pure numbers alone (e.g., 1.23, 42)
      - Preserves end-of-line comments (# ...)

    NOTE: It does NOT rewrite inside arrays or multi-line values.
    """
    out_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or stripped.startswith("[") or "=" not in line:
            out_lines.append(line)
            continue

        # Split once on '='
        left, right = line.split("=", 1)

        # Separate trailing comment
        if "#" in right:
            rhs_raw, comment = right.split("#", 1)
            comment = "#" + comment  # keep '#'
        else:
            rhs_raw, comment = right, None

        rhs = rhs_raw.strip()

        # Conditions to skip rewriting
        skip = (
            not rhs
            or rhs[0] in "\"'"  # already quoted string
            or rhs[0] in "[{"  # array/table/dict
            or rhs.lower() in ("true", "false")  # boolean
        )

        if not skip:
            # If rhs is a pure number, leave it
            try:
                float(rhs)
                is_pure_number = True
            except Exception:
                is_pure_number = False

            # If rhs matches our safe arithmetic pattern and is NOT a pure number => quote it
            if (not is_pure_number) and _ALLOWED_EXPR.match(rhs):
                rhs = f'"{rhs}"'  # quote it
                # Rebuild the line with normalized spacing around '='
                new_line = f"{left.rstrip()} = {rhs}"
                if comment is not None:
                    new_line += f" {comment}"
                out_lines.append(new_line)
                continue

        # Default: keep original line
        out_lines.append(line)

    return "\n".join(out_lines)


def read_toml(path: str) -> Config:
    try:
        # Read raw text so we can preprocess arithmetic expressions
        with open(path, encoding="utf-8") as ftxt:
            raw_text = ftxt.read()
        pre_text = _preprocess_toml_expressions(raw_text)

        # Parse TOML from preprocessed text
        raw = tomllib.loads(pre_text)
    except TOMLDecodeError as e:
        raise SystemExit(
            f"TOML syntax error in '{path}': {e}\n\n"
            "Tip: You can write arithmetic like 0.5*10000 or 2*pi directly (unquoted); "
            "we’ll auto-convert it. If this still fails, check for arrays/multi-line values—"
            'those must quote expressions explicitly (e.g., ["2*pi", "3*pi"]).'
        ) from e

    # Coerce safe string expressions like "2*pi" -> float
    d = _coerce_constants(raw)

    sim = SimCfg(**d["sim"])
    grid = GridCfg(**d["grid"])
    hermite = HermiteCfg(**d["hermite"])
    bc = BCCfg(**d["bc"])
    plot = PlotCfg(**d.get("plot", {}))

    species = []
    for sp_raw in d.get("species", []):
        sp = SpeciesCfg(**sp_raw)
        sp = _infer_mass_defaults_from_name(sp)  # set sensible base by name (electron/proton)
        sp = _finalize_species_units(sp)  # compute m, vth (from temperature_eV), u0 (from drift_c)
        species.append(sp)

    return Config(sim=sim, grid=grid, hermite=hermite, bc=bc, plot=plot, species=species)
