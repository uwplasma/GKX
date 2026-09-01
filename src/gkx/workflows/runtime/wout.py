"""Zero-config ``gkx wout_XXX.nc`` equilibrium shorthand helpers."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Callable

from gkx.workflows.runtime.toml import (
    RUNTIME_TOML_SCHEMA_VERSION,
    load_toml,
    resolve_runtime_path,
)

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


def _toml_value(value: Any) -> str:
    """Render one scalar or flat array as TOML text."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported resolved-deck value: {value!r}")


def deck_text(data: dict[str, Any], *, header: tuple[str, ...] = ()) -> str:
    """Render deck data as TOML, with an optional comment header.

    This is the single renderer. ``resolved_deck_text`` supplies the equilibrium
    shorthand's header and delegates here, so a case written by ``Case.to_toml``
    and a deck written by the shorthand share one serializer and cannot drift
    apart in quoting or table ordering.
    """

    lines = list(header)
    tables = {
        key: value
        for key, value in data.items()
        if isinstance(value, dict)
        or (
            isinstance(value, list)
            and value
            and all(isinstance(v, dict) for v in value)
        )
    }
    for key, value in data.items():
        if key not in tables and value is not None:
            lines.append(f"{key} = {_toml_value(value)}")
    for key, value in tables.items():
        items = value if isinstance(value, list) else [value]
        marker = f"[[{key}]]" if isinstance(value, list) else f"[{key}]"
        for item in items:
            lines.append("")
            lines.append(marker)
            # TOML has no null. A field left as None is absent from the file,
            # which is exactly how the loader spells "use the default", so a
            # written case reloads to the case that wrote it.
            lines.extend(
                f"{k} = {_toml_value(v)}" for k, v in item.items() if v is not None
            )
    return "\n".join(lines) + "\n"


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

    from gkx.workflows.runtime.resolution import estimate_resolution

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
