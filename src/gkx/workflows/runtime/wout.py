"""Zero-config ``gkx wout_XXX.nc`` equilibrium shorthand helpers."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Callable

from gkx.workflows.runtime.toml import load_toml, resolve_runtime_path

WOUT_SIGNATURE_VARIABLES = ("rmnc", "zmns", "xm", "xn")
WOUT_FLAG_NAMES = ("--vmec", "--vmex")
DEFAULT_LINEAR_KY_VALUES = (0.1, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0)

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
            return "version_" in variables or any(
                "vmec" in name for name in attributes
            )
    except OSError:
        return False


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


def resolved_deck_text(data: dict[str, Any], *, wout_path: Path) -> str:
    """Render resolved deck data as TOML, mirroring the demo reproducer."""

    lines = [
        "# Fully-resolved GKX input written by the wout equilibrium shorthand.",
        f"# equilibrium: {wout_path}",
        "# Rerun with: gkx <this file>",
    ]
    tables = {
        key: value
        for key, value in data.items()
        if isinstance(value, dict)
        or (isinstance(value, list) and value and all(isinstance(v, dict) for v in value))
    }
    for key, value in data.items():
        if key not in tables:
            lines.append(f"{key} = {_toml_value(value)}")
    for key, value in tables.items():
        items = value if isinstance(value, list) else [value]
        marker = f"[[{key}]]" if isinstance(value, list) else f"[{key}]"
        for item in items:
            lines.append("")
            lines.append(marker)
            lines.extend(f"{k} = {_toml_value(v)}" for k, v in item.items())
    return "\n".join(lines) + "\n"


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
    """Switch a deck to linear physics with a default ky-scan list."""

    data["physics"] = {**data.get("physics", {}), "linear": True, "nonlinear": False}
    data["terms"] = {**data.get("terms", {}), "nonlinear": 0.0}
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

    deck_path = Path(config_arg) if config_arg is not None else default_wout_deck_path()
    data = dict(load_toml_func(deck_path))
    _resolve_deck_paths(data, base_dir=deck_path.resolve().parent)
    _force_vmec_geometry(data, wout_path)
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
