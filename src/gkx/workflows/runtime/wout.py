"""Zero-config ``gkx wout_XXX.nc`` equilibrium shorthand helpers."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Callable, Sequence
from dataclasses import dataclass, replace
import numpy as np

from gkx.config import (
    deck_text,
    resolve_cfl_fac,
    RuntimeConfig,
)
from gkx.workflows.runtime.toml import (
    EXECUTABLE_TOML_SHORTHAND_COMMANDS,
    RUNTIME_TOML_SCHEMA_VERSION,
    load_toml,
    resolve_runtime_path,
    toml_shorthand_command,
    load_runtime_from_toml,
)
from gkx.core_grid import build_spectral_grid
from gkx.geometry import FluxTubeGeometryLike
from gkx.geometry.core import (
    apply_geometry_grid_defaults,
    ensure_flux_tube_geometry_data,
)
from gkx.solvers_time_explicit_cfl import _linear_frequency_bound
from gkx.workflows.runtime.startup import (
    build_runtime_geometry,
    build_runtime_linear_params,
)


#: Perpendicular grid rungs, extending the scan ladder upward.
PERP_LADDER = (32, 64, 96, 128, 192, 256)

#: ky_max*rho per (class, tier). At the default dky*rho = 0.071 these land on
#: rungs 64/96/128 (tokamak) and 96/128/192 (stellarator). Measured y0=14
#: ladder bias per tier -- tokamak: preview ~+8% above the converged 96/128
#: plateau, standard converged within SEM, cautious headroom; stellarator:
#: preview ~+15-25% and standard ~+8-19% (both still upper estimates, flux
#: falling at 128^2), cautious = published W7-X reach ~4.4 (unvalidated by
#: this scan until the 192^2 references land).
KY_TARGETS_BY_CLASS = {
    "tokamak": {"preview": 1.5, "standard": 2.2, "cautious": 2.9},
    "stellarator": {"preview": 2.2, "standard": 2.9, "cautious": 4.4},
}

# Anisotropy corroborates the nfp class split only (measured: DIII-D 0.138
# vs stellarators 0.55-0.80); its finer per-case ordering was falsified.
_TOKAMAK_ANISOTROPY_MAX = 0.30

_TARGET_ERRORS = ("preview", "standard", "cautious")
_WELL_PROMINENCE = 0.10  # fraction of the |B| range a well must dip to count
_PUBLISHED_STELLARATOR_DKY = 0.071  # published W7-X spacing; 0.100 also in use
_SATURATION_TIME = 50.0  # typical a/v_ti units to saturation in the scan
_T_MAX_FACTOR = 8.0  # scan cap 400 = 8 * t_sat; tokamak rungs stopped early


@dataclass(frozen=True)
class GeometryFeatures:
    """Host-side geometry scalars the estimator maps to a grid."""

    anisotropy: float
    shat: float
    q: float
    nfp: int
    bmag_wells: int
    zp: float


def _count_bmag_wells(bmag: np.ndarray) -> int:
    """Count ``|B|`` wells deeper than ``_WELL_PROMINENCE`` of the ``|B|`` range."""

    n, b_range = int(bmag.size), float(bmag.max() - bmag.min()) if bmag.size else 0.0
    if n < 4 or b_range <= 0.0:
        return 1
    prev, nxt = np.roll(bmag, 1), np.roll(bmag, -1)
    minima = np.flatnonzero((bmag < prev) & (bmag < nxt))
    maxima = np.flatnonzero((bmag > prev) & (bmag > nxt))
    if minima.size == 0 or maxima.size == 0:
        return 1
    wells = 0
    for i in minima:
        left = maxima[np.argmin((int(i) - maxima) % n)]
        right = maxima[np.argmin((maxima - int(i)) % n)]
        wells += min(bmag[left], bmag[right]) - bmag[i] >= _WELL_PROMINENCE * b_range
    return max(int(wells), 1)


def geometry_features(geom: Any, *, zp: float = 1.0) -> GeometryFeatures:
    """Extract estimator features from sampled flux-tube geometry data."""

    sl = slice(None, -1) if bool(geom.theta_closed_interval) else slice(None)
    gds2 = np.asarray(geom.gds2_profile, dtype=float)[sl]
    gds22 = np.asarray(geom.gds22_profile, dtype=float)[sl]
    bmag = np.asarray(geom.bmag_profile, dtype=float)[sl]
    shat = float(geom.s_hat)
    gradx_rms = float(np.sqrt(np.mean(gds22))) / max(abs(shat), 1.0e-8)
    grady_max = float(np.sqrt(np.max(gds2)))
    return GeometryFeatures(
        anisotropy=gradx_rms / grady_max,
        shat=shat,
        q=float(geom.q),
        nfp=int(geom.nfp),
        bmag_wells=_count_bmag_wells(bmag),
        zp=float(zp),
    )


def geometry_class(features: GeometryFeatures) -> str:
    """Classify the equilibrium; nfp decides, anisotropy corroborates."""

    if features.nfp > 1:
        return "stellarator"
    return (
        "tokamak" if features.anisotropy <= _TOKAMAK_ANISOTROPY_MAX else "stellarator"
    )


def ky_max_target(features: GeometryFeatures, target_error: str = "standard") -> float:
    """Required binormal reach ky_max*rho for this class and error tier."""

    if target_error not in _TARGET_ERRORS:
        raise ValueError(
            f"target_error must be one of {sorted(_TARGET_ERRORS)}, got {target_error!r}"
        )
    return KY_TARGETS_BY_CLASS[geometry_class(features)][target_error]


def perp_points_for(dky: float, ky_target: float) -> int:
    """Smallest ladder rung whose dealiased ky grid reaches ``ky_target``."""

    for rung in PERP_LADDER:
        if ((rung - 1) // 3) * dky >= 0.98 * ky_target:
            return rung
    return PERP_LADDER[-1]


def resolution_from_features(
    features: GeometryFeatures,
    *,
    dky: float,
    target_error: str = "standard",
    nz_default: int = 48,
    hypercollisions: bool = True,
    kinetic_electrons: bool = False,
) -> dict[str, Any]:
    """Map geometry features to grid hints with a rationale per number."""

    klass = geometry_class(features)
    ky_target = ky_max_target(features, target_error)
    nx = perp_points_for(dky, ky_target)
    floor = max(16.0 * features.zp, 6.0 * features.bmag_wells)
    nz_floor = int(-(-int(np.ceil(floor)) // 8) * 8)  # round up to multiple of 8
    nz = max(nz_default, nz_floor) if target_error == "cautious" else nz_default
    nl, nm = (4, 8) if hypercollisions else (6, 12)
    if kinetic_electrons:
        nm = max(nm, 16)
    notes = []
    if nz_floor > nz:
        notes.append(
            f"parallel floor unmet: {features.bmag_wells} |B| wells x 6 pts and 16/2pi-"
            f"period ask Nz >= {nz_floor}; the scan left Nz unvalidated here"
        )
    if dky < 0.9 * _PUBLISHED_STELLARATOR_DKY:
        alt = perp_points_for(_PUBLISHED_STELLARATOR_DKY, ky_target)
        notes.append(
            f"cheaper box: dky*rho = {_PUBLISHED_STELLARATOR_DKY} (y0 = 14, the "
            f"calibrated ladder's spacing) reaches ky_max {ky_target:g} at "
            f"Nx=Ny={alt}"
        )
    if klass == "stellarator":
        notes.append(
            "stellarator fluxes converge from above: the calibration ladder was "
            "still falling at 128^2, so this tier's flux is an upper estimate "
            "(measured bias: preview ~+15-25%, standard ~+8-19% vs the next rung)"
        )
    if abs(features.shat) < 0.3:
        notes.append(
            "low-shear tube: check npol>=2 and a second field line before "
            "promoting any flux (Faber 2018; Ajay 2020; Kim 2024)"
        )
    hyper = "hypercollisions" if hypercollisions else "no hypercollisions"
    rationale = {
        "nx": "square perpendicular box (Lx = Ly), so Nx tracks Ny",
        "ny": (
            f"{klass} class (nfp = {features.nfp}, anisotropy "
            f"{features.anisotropy:.3f}) asks ky_max*rho >= {ky_target:g} "
            f"({target_error}); reach ((Ny-1)//3)*dky = "
            f"{((nx - 1) // 3) * dky:.2f} at dky = {dky:.3f}"
        ),
        "nz": (
            f"scan found Nz weakly coupled (flux 8.47/8.78/8.39 at 24/32/48); "
            f"floors: 16/2pi-period x {features.zp:g}, 6 x {features.bmag_wells} |B| wells"
        ),
        "nl": f"Laguerre FLR floor with {hyper}; the scan converged at Nl=4",
        "nm": (
            f"{hyper}: t_quiet ~ 5.5*sqrt(Nm) recurrence sets the published floor "
            f"({'4,8' if hypercollisions else '6,12'})"
            + ("; kinetic electrons floor Nm at 16" if kinetic_electrons else "")
        ),
        "t_max": f"{_T_MAX_FACTOR:g} x t_sat ~ {_SATURATION_TIME:g} hard cap; "
        'run_to = "saturation" stops earlier',
    }
    t_max = _T_MAX_FACTOR * _SATURATION_TIME
    return {
        "nx": nx,
        "ny": nx,
        "nz": nz,
        "nl": nl,
        "nm": nm,
        "t_max": 0.5 * t_max if target_error == "preview" else t_max,
        "geometry_class": klass,
        "ky_max_target": ky_target,
        "rationale": rationale,
        "features": features,
        "notes": notes,
    }


def _dt_hint(
    cfg: RuntimeConfig, geom: FluxTubeGeometryLike, est: dict[str, Any]
) -> float:
    """Initial-dt bound from the solver's own linear CFL frequency estimator."""

    grid_cfg = replace(cfg.grid, Nx=int(est["nx"]), Ny=int(est["ny"]))
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, grid_cfg))
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    params = build_runtime_linear_params(cfg, Nm=int(est["nm"]), geom=geom_eff)
    nl, nm = int(est["nl"]), int(est["nm"])
    wmax = float(np.sum(_linear_frequency_bound(grid, geom_eff, params, nl, nm)))
    if wmax <= 0.0:
        return float(cfg.time.dt)
    fac = resolve_cfl_fac(cfg.time.method, cfg.time.cfl_fac)
    return min(float(cfg.time.dt), fac * float(cfg.time.cfl) / wmax)


def estimate_resolution(
    wout_path: str | Path,
    *,
    torflux: float | None = None,
    target_error: str = "standard",
    deck_path: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate the minimum adequate grid for one VMEC/VMEX equilibrium."""

    cfg, _ = load_runtime_from_toml(
        deck_path if deck_path is not None else default_wout_deck_path()
    )
    geometry = replace(cfg.geometry, vmec_file=str(Path(wout_path).resolve()))
    if torflux is not None:
        geometry = replace(geometry, torflux=float(torflux))
    cfg = replace(cfg, geometry=geometry)
    geom = build_runtime_geometry(cfg)

    grid = cfg.grid
    zp = float(grid.zp if grid.zp is not None else 2 * (grid.nperiod or 1) - 1)
    y0 = float(grid.y0) if grid.y0 is not None else grid.Ly / (2.0 * np.pi)
    est = resolution_from_features(
        geometry_features(geom, zp=zp),
        dky=1.0 / y0,
        target_error=target_error,
        nz_default=int(grid.ntheta if grid.ntheta is not None else grid.Nz),
        hypercollisions=bool(cfg.physics.hypercollisions),
        kinetic_electrons=any(
            sp.kinetic and float(sp.charge) < 0.0 for sp in cfg.species
        ),
    )
    est["dt"] = _dt_hint(cfg, geom, est)
    est["rationale"]["dt"] = (
        "explicit CFL bound cfl_fac*cfl/sum(omega_max) at this grid; "
        "the adaptive stepper raises it toward the measured ExB limit"
    )
    est["torflux"] = (
        None if cfg.geometry.torflux is None else float(cfg.geometry.torflux)
    )
    return est


WOUT_SIGNATURE_VARIABLES = ("rmnc", "zmns", "xm", "xn")
WOUT_FLAG_NAMES = ("--vmec", "--vmex")
DEFAULT_LINEAR_KY_VALUES = (0.1, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0)

#: Fixed step for the shorthand linear scan. Half the measured 0.019 bound
#: at the top of DEFAULT_LINEAR_KY_VALUES, so every rung integrates.
LINEAR_SCAN_DT = 0.01

# Path-valued deck fields that must survive relocating the resolved deck.
_DECK_PATH_FIELDS = (
    ("geometry", "vmec_file"),
    ("geometry", "geometry_file"),
    ("init", "init_file"),
    ("output", "path"),
    ("output", "restart_to_file"),
    ("output", "restart_from_file"),
    ("quasilinear", "output_path"),
)


def is_wout_file(path: str | Path) -> bool:
    """Return whether ``path`` is a NetCDF VMEC/VMEX ``wout`` equilibrium."""

    path = Path(path)
    if path.suffix.lower() != ".nc" or not path.is_file():
        return False
    from netCDF4 import Dataset

    try:
        with Dataset(path, "r") as ds:
            variables = set(ds.variables)
            if all(name in variables for name in WOUT_SIGNATURE_VARIABLES):
                return True
            attributes = {str(name).lower() for name in ds.ncattrs()}
            return "version_" in variables or any("vmec" in name for name in attributes)
    except OSError:
        return False


def _wout_nfp(path: Path) -> int | None:
    """Field periods from the wout header, or ``None`` when unreadable."""

    from netCDF4 import Dataset

    try:
        with Dataset(path, "r") as ds:
            return int(ds.variables["nfp"][()])
    except (OSError, KeyError, ValueError):
        return None


def _apply_class_resolution_defaults(data: dict[str, Any], wout_path: Path) -> None:
    """Shipped-deck preview grid per equilibrium class (2026-08 y0=14 ladder).

    Tokamaks (nfp = 1) saturated at every rung with 64^2 only ~8% above the
    converged 96/128 plateau, so their preview drops to 64^2. Stellarators
    keep the deck's 96^2, which the ladder measured as an upper estimate
    (flux still falling at 128^2). Never applied to a user-supplied deck.
    """

    nfp = _wout_nfp(wout_path)
    grid = dict(data.get("grid", {}))
    if nfp == 1:
        grid["Nx"], grid["Ny"] = 64, 64
        data["grid"] = grid
        print(
            "tokamak equilibrium (nfp = 1): preview grid 64x64 "
            "(measured ~+8% vs the converged 96/128 flux; run with "
            "--estimate for the standard/cautious tiers)"
        )
    elif nfp is not None:
        print(
            "stellarator equilibrium: preview grid "
            f"{grid.get('Nx', '?')}x{grid.get('Ny', '?')} -- an upper "
            "estimate (the calibration ladder was still falling at 128^2); "
            "run with --estimate for the standard/cautious tiers"
        )


def default_wout_deck_path() -> Path:
    """Return the single-source default deck used for bare equilibrium runs."""

    packaged = Path(str(resources.files("gkx").joinpath("data/common_input.toml")))
    if packaged.is_file():
        return packaged
    repo = Path(__file__).resolve().parents[4] / "examples" / "common_input.toml"
    if repo.is_file():
        return repo
    raise FileNotFoundError(
        "default deck common_input.toml not found in gkx package data or examples/"
    )


def extract_wout_flag_value(args: list[str]) -> str | None:
    """Pop ``--vmec``/``--vmex`` alias flags from ``args``, returning the file."""

    value: str | None = None
    index = 0
    while index < len(args):
        arg = args[index]
        flag = next(
            (f for f in WOUT_FLAG_NAMES if arg == f or arg.startswith(f + "=")),
            None,
        )
        if flag is None:
            index += 1
            continue
        if arg == flag:
            if index + 1 >= len(args):
                raise SystemExit(f"gkx: {flag} requires an equilibrium FILE argument")
            args.pop(index)
            value = args.pop(index)
        else:
            value = args.pop(index).split("=", 1)[1]
    return value


def resolved_deck_text(data: dict[str, Any], *, wout_path: Path) -> str:
    """Render resolved deck data as TOML, mirroring the demo reproducer."""

    header = (
        "# Fully-resolved GKX input written by the wout equilibrium shorthand.",
        f"# equilibrium: {wout_path}",
        "# Rerun with: gkx <this file>",
    )
    return deck_text(data, header=header)


def _resolve_deck_paths(data: dict[str, Any], *, base_dir: Path) -> None:
    """Resolve path-valued deck fields so the resolved deck can relocate."""

    for section, key in _DECK_PATH_FIELDS:
        table = data.get(section)
        if isinstance(table, dict) and table.get(key) is not None:
            table = dict(table)
            table[key] = resolve_runtime_path(str(table[key]), base_dir=base_dir)
            data[section] = table


def _force_vmec_geometry(data: dict[str, Any], wout_path: Path) -> None:
    """Point [geometry] at the wout file, forcing model="vmec" when needed."""

    geometry = dict(data.get("geometry", {}))
    if str(geometry.get("model", "")).strip().lower() != "vmec":
        geometry["model"] = "vmec"
        geometry.pop("geometry_file", None)
    geometry["vmec_file"] = str(wout_path)
    data["geometry"] = geometry


def _apply_linear_scan_defaults(data: dict[str, Any]) -> None:
    """Switch a deck to linear physics with a default ky-scan list.

    The step is reduced along with the physics. A nonlinear deck is written
    around its own low-``k_y`` box, while this scan reaches ``k_y rho = 1``,
    where the explicit bound is far tighter: on the shipped stellarator deck
    the CFL-stable step falls from 0.067 at the first finite ``k_y`` to 0.019
    at the top of the scan, so the nonlinear deck's 0.1 overflows every rung
    and the growth fit then refuses a non-finite history. The linear paths
    advance the whole RHS explicitly at a fixed step, so adaptivity in the
    deck does not rescue them.
    """

    data["physics"] = {**data.get("physics", {}), "linear": True, "nonlinear": False}
    data["terms"] = {**data.get("terms", {}), "nonlinear": 0.0}
    time_cfg = dict(data.get("time", {}))
    if float(time_cfg.get("dt", 0.0)) > LINEAR_SCAN_DT:
        time_cfg["dt"] = LINEAR_SCAN_DT
        data["time"] = time_cfg
    scan = dict(data.get("scan", {}))
    scan.setdefault("ky", list(DEFAULT_LINEAR_KY_VALUES))
    data["scan"] = scan


def _flag_value(args: list[str], flag: str) -> str | None:
    """Return the value of ``flag`` inside pass-through parser arguments."""

    for index, arg in enumerate(args):
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _resolved_output_prefix(
    data: dict[str, Any], extra: list[str], wout_path: Path
) -> tuple[Path, bool]:
    """Return the output prefix for wout-run artifacts, and whether it was asked for.

    The second element says the target came from ``--out`` or from the deck's
    own ``[output] path``; only the prefix this function invents is free to
    grow a suffix in :func:`_resolved_output_target`.
    """

    out_flag = _flag_value(extra, "--out")
    if out_flag is not None:
        return Path(str(resolve_runtime_path(out_flag, base_dir=Path.cwd()))), True
    configured = data.get("output", {}).get("path")
    if configured:
        return Path(configured), True
    return Path.cwd() / wout_path.stem / "gkx", False


def _resolved_output_target(prefix: Path, *, explicit: bool, linear: bool) -> Path:
    """Return the ``[output] path`` a bare equilibrium run writes to.

    A plain prefix makes the runtime write CSV/JSON sidecars, which carry time
    traces only -- so the spectra, the potential map, and the restart file all
    silently do not exist, and the figure set shrinks to one panel. The default
    is therefore the NetCDF bundle, which is the format the rest of the result
    set is read from. A linear ky scan has no NetCDF form, and a target the
    user named is theirs; both keep the prefix as given.
    """

    if linear or explicit:
        return prefix
    return Path(f"{prefix}.out.nc")


def _print_deck_header(
    *, wout_path: Path, deck_path: Path, resolved_path: Path, shipped: bool
) -> None:
    """Name the deck the defaults came from before the run starts.

    The resolved copy alone does not tell a first-time user what to edit: it is
    generated, it is inside the output directory, and it is overwritten by the
    next run. Naming the shipped deck -- and the command that runs an edited
    copy of it -- is what makes the next run theirs.
    """

    print(f"equilibrium: {wout_path}", flush=True)
    if shipped:
        # resolve() follows the packaged symlink back to examples/ in a repo
        # checkout, and is a no-op for an installed wheel: either way the
        # printed path is a file the user can copy.
        print(
            f"default deck: {deck_path.resolve()} "
            f"(copy, edit, then: gkx my_input.toml {wout_path.name})",
            flush=True,
        )
    else:
        print(f"input deck: {deck_path}", flush=True)
    print(f"wrote resolved input: {resolved_path}", flush=True)


def _pop_estimate_flag(args: list[str]) -> str | None:
    """Pop ``--estimate[=TARGET]``, returning the target-error tier if present."""

    value: str | None = None
    for index, arg in enumerate(args):
        if arg == "--estimate":
            value = "standard"
        elif arg.startswith("--estimate="):
            value = arg.split("=", 1)[1]
        else:
            continue
        args.pop(index)
        return value
    return None


def format_estimate_table(est: dict[str, Any]) -> str:
    """Render one resolution estimate as the table ``--estimate`` prints."""

    f = est["features"]
    lines = [
        f"geometry: shat={f.shat:+.4f} q={f.q:.3f} nfp={f.nfp} "
        f"|B| wells={f.bmag_wells} anisotropy={f.anisotropy:.3f} "
        f"-> ky_max*rho >= {est['ky_max_target']:g}",
    ]
    for key in ("nx", "ny", "nz", "nl", "nm", "dt", "t_max"):
        value = est[key]
        rendered = f"{value:.4g}" if isinstance(value, float) else str(value)
        lines.append(f"{key:>6} = {rendered:<8} {est['rationale'][key]}")
    lines.extend(f"note: {note}" for note in est["notes"])
    return "\n".join(lines)


def _print_resolution_estimate(
    wout_path: Path, config_arg: str | None, *, target_error: str
) -> None:
    """Print the minimum-grid estimate table for one equilibrium."""

    estimate = estimate_resolution(
        wout_path, target_error=target_error, deck_path=config_arg
    )
    print(f"equilibrium: {wout_path}", flush=True)
    print(format_estimate_table(estimate), flush=True)


def wout_shorthand_args(
    wout_arg: str,
    config_arg: str | None,
    extra_args: list[str],
    *,
    load_toml_func: Callable[[str | Path], dict[str, Any]] = load_toml,
) -> list[str]:
    """Return parser args for equilibrium shorthand, writing the resolved deck."""

    wout_path = Path(wout_arg).resolve()
    extra = [arg for arg in extra_args if arg != "--linear"]
    linear = len(extra) != len(extra_args)
    estimate_tier = _pop_estimate_flag(extra)
    if estimate_tier is not None:
        # Advisory mode: print the table and stop before any deck or output
        # directory is written; nothing about the eventual run is changed.
        _print_resolution_estimate(wout_path, config_arg, target_error=estimate_tier)
        raise SystemExit(0)

    deck_path = Path(config_arg) if config_arg is not None else default_wout_deck_path()
    data = dict(load_toml_func(deck_path))
    data.setdefault("schema_version", RUNTIME_TOML_SCHEMA_VERSION)
    _resolve_deck_paths(data, base_dir=deck_path.resolve().parent)
    _force_vmec_geometry(data, wout_path)
    if config_arg is None:
        _apply_class_resolution_defaults(data, wout_path)
    if linear:
        _apply_linear_scan_defaults(data)

    prefix, explicit = _resolved_output_prefix(data, extra, wout_path)
    target = _resolved_output_target(prefix, explicit=explicit, linear=linear)
    data["output"] = {**data.get("output", {}), "path": str(target)}
    resolved_path = Path(f"{prefix}.toml")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        resolved_deck_text(data, wout_path=wout_path), encoding="utf-8"
    )
    _print_deck_header(
        wout_path=wout_path,
        deck_path=deck_path,
        resolved_path=resolved_path,
        shipped=config_arg is None,
    )

    command = "scan-runtime-linear" if linear else "run"
    return [command, "--config", str(resolved_path), *extra]


def direct_config_shorthand_args(
    argv: Sequence[str],
    *,
    load_toml_func: Callable[[str | Path], dict[str, Any]] = load_toml,
) -> list[str] | None:
    """Return parser arguments for ``gkx case.toml`` / ``gkx wout_XXX.nc`` shorthand.

    Leading positionals may be a runtime TOML deck and/or a VMEC/VMEX wout
    equilibrium (in either order); ``--vmec FILE``/``--vmex FILE`` are explicit
    aliases for the wout positional. A wout argument routes through the
    equilibrium shorthand, which writes a fully-resolved deck next to the
    grouped outputs before dispatch.
    """

    if not argv or argv[0] in EXECUTABLE_TOML_SHORTHAND_COMMANDS:
        return None
    args = list(argv)
    wout_arg = extract_wout_flag_value(args)
    config_arg: str | None = None
    while args and not args[0].startswith("-") and Path(args[0]).exists():
        if is_wout_file(args[0]):
            if wout_arg is not None:
                break
            wout_arg = args.pop(0)
        elif config_arg is None:
            config_arg = args.pop(0)
        else:
            break
    if wout_arg is not None:
        return wout_shorthand_args(
            wout_arg, config_arg, args, load_toml_func=load_toml_func
        )
    if config_arg is None:
        return None
    command = toml_shorthand_command(load_toml_func(config_arg))
    return [command, "--config", config_arg, *args]
