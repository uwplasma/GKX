"""TOML-based input helpers for the executable and driver scripts."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any, Callable, Sequence, cast
import os
from pathlib import Path

from gkx.utils import tomlcompat as tomllib
from gkx.workflows.runtime.config import (
    Case, RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeExpertConfig,
    RuntimeNormalizationConfig,
    RuntimeOutputConfig,
    RuntimeParallelConfig,
    RuntimePhysicsConfig,
    RuntimeQuasilinearConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)

RUNTIME_TOML_TOP_LEVEL_KEYS = {
    "species",
    "physics",
    "collisions",
    "normalization",
    "expert",
    "output",
    "quasilinear",
}
EXECUTABLE_TOML_SHORTHAND_COMMANDS = {
    "run",
    "run-runtime-linear",
    "scan-runtime-linear",
    "run-runtime-nonlinear",
}


# Leading bytes of the binary formats a user is most likely to hand the CLI by
# mistake, mapped to what that file actually is.
_BINARY_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"CDF\x01", "a NetCDF classic file"),
    (b"CDF\x02", "a NetCDF 64-bit-offset file"),
    (b"\x89HDF\r\n\x1a\n", "an HDF5 file, which is what NetCDF-4 uses"),
    (b"PK\x03\x04", "a zip archive"),
    (b"\x93NUMPY", "a NumPy array file"),
)


def describe_binary_input(path: Path) -> str | None:
    """Return what ``path`` actually is when it is not text, else ``None``."""

    try:
        with path.open("rb") as handle:
            head = handle.read(8)
    except OSError:
        return None
    for signature, description in _BINARY_SIGNATURES:
        if head.startswith(signature):
            return description
    return None


def load_toml(path: str | Path) -> dict:
    """Load a TOML file into a plain dictionary.

    A binary file reaches here whenever it was not recognised earlier -- most
    often an equilibrium whose name or contents did not match the wout
    signature -- and ``tomllib`` reports that as a decode error against a byte
    offset, which says nothing about what went wrong. Name the file and what it
    turned out to be instead.
    """

    path = Path(path)
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except UnicodeDecodeError as exc:
        binary = describe_binary_input(path)
        if binary is not None:
            raise ValueError(
                f"{path} is {binary}, not a TOML input file. If this is a VMEC "
                "equilibrium, it was not recognised as one: check that it "
                "carries the wout variables (rmnc, zmns, xm, xn), or pass it "
                "with --vmec to say so explicitly."
            ) from exc
        raise ValueError(
            f"{path} is not valid UTF-8, so it cannot be a TOML input file."
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path} is not valid TOML: {exc}") from exc


def is_runtime_toml(data: dict[str, Any]) -> bool:
    """Return whether a parsed input uses the supported runtime schema."""

    _ = data
    return True


def toml_shorthand_command(data: dict[str, Any]) -> str:
    """Return the executable command used for direct TOML path shorthand."""

    _ = data
    return "run"


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
    from gkx.workflows.runtime import wout as runtime_wout

    args = list(argv)
    wout_arg = runtime_wout.extract_wout_flag_value(args)
    config_arg: str | None = None
    while args and not args[0].startswith("-") and Path(args[0]).exists():
        if runtime_wout.is_wout_file(args[0]):
            if wout_arg is not None:
                break
            wout_arg = args.pop(0)
        elif config_arg is None:
            config_arg = args.pop(0)
        else:
            break
    if wout_arg is not None:
        return runtime_wout.wout_shorthand_args(
            wout_arg, config_arg, args, load_toml_func=load_toml_func
        )
    if config_arg is None:
        return None
    command = toml_shorthand_command(load_toml_func(config_arg))
    return [command, "--config", config_arg, *args]


def resolve_runtime_path(value: str | None, *, base_dir: Path) -> str | None:
    """Expand and resolve a runtime config path.

    Applies ``$VAR`` and ``~`` expansion, then resolves relative paths against
    ``base_dir``. If an unresolved ``$VAR`` remains after expansion (env var not
    set), the original value is returned unchanged so downstream code can raise
    a clearer error. ``None`` is passed through.

    Parameters
    ----------
    value : str or None
        Raw path string from a TOML config or CLI flag.
    base_dir : Path
        Directory used to resolve relative paths. Callers typically pass the
        config file's parent directory (TOML values) or ``Path.cwd()``
        (CLI-supplied values).

    Returns
    -------
    str or None
        Absolute resolved path as a string, or ``None`` when ``value`` is ``None``.
    """
    if value is None:
        return None
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "$" in expanded:
        return value
    path = Path(expanded)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def _merge_dataclass(base: Any, overrides: dict | None) -> Any:
    """Recursively merge a dict into a dataclass, returning a new instance."""

    if overrides is None:
        return base
    if not is_dataclass(base) or isinstance(base, type):
        raise TypeError("base must be a dataclass instance")
    updates = {}
    for field in base.__dataclass_fields__.values():  # type: ignore[attr-defined]
        name = field.name
        if name not in overrides:
            continue
        value = overrides[name]
        if value is None:
            continue
        current = getattr(base, name)
        if is_dataclass(current) and isinstance(value, dict):
            updates[name] = _merge_dataclass(current, value)
        else:
            updates[name] = value
    return cast(Any, replace(base, **updates))


def _normalize_geometry_overrides(overrides: dict | None) -> dict | None:
    """Return geometry overrides using the canonical runtime schema."""

    if not isinstance(overrides, dict):
        return overrides
    return dict(overrides)


def _runtime_base_config(data: dict[str, Any]) -> RuntimeConfig:
    """Return a runtime config after applying common dataclass sections."""

    return cast(
        RuntimeConfig,
        _merge_dataclass(
            RuntimeConfig(),
            {
                "grid": data.get("grid"),
                "time": data.get("time"),
                "geometry": _normalize_geometry_overrides(data.get("geometry")),
                "init": data.get("init"),
            },
        ),
    )


def _replace_runtime_section(
    cfg: RuntimeConfig,
    data: dict[str, Any],
    key: str,
    constructor: Callable[..., Any],
) -> RuntimeConfig:
    """Replace one runtime section when the TOML section is present."""

    raw = data.get(key)
    if not isinstance(raw, dict):
        return cfg
    return cast(RuntimeConfig, replace(cfg, **{key: constructor(**raw)}))


def _apply_runtime_section_overrides(
    cfg: RuntimeConfig,
    data: dict[str, Any],
) -> RuntimeConfig:
    """Apply non-nested runtime config sections from TOML data."""

    section_constructors: tuple[tuple[str, Callable[..., Any]], ...] = (
        ("physics", RuntimePhysicsConfig),
        ("collisions", RuntimeCollisionConfig),
        ("normalization", RuntimeNormalizationConfig),
        ("terms", RuntimeTermsConfig),
        ("expert", RuntimeExpertConfig),
        ("output", RuntimeOutputConfig),
        ("quasilinear", RuntimeQuasilinearConfig),
        ("parallel", RuntimeParallelConfig),
    )
    for key, constructor in section_constructors:
        cfg = _replace_runtime_section(cfg, data, key, constructor)
    return cfg


def _runtime_species_from_toml(
    species_raw: Any,
) -> tuple[RuntimeSpeciesConfig, ...] | None:
    """Parse optional ``[[species]]`` runtime entries."""

    if species_raw is None:
        return None
    if not isinstance(species_raw, list):
        raise TypeError("[[species]] entries must be provided as an array of tables")
    species: list[RuntimeSpeciesConfig] = []
    for item in species_raw:
        if not isinstance(item, dict):
            raise TypeError("Each [[species]] entry must be a table")
        species.append(RuntimeSpeciesConfig(**item))
    return tuple(species) if species else None


def _resolve_runtime_config_paths(cfg: RuntimeConfig, *, base_dir: Path) -> RuntimeConfig:
    """Resolve every path-valued runtime field against the TOML directory."""

    return replace(
        cfg,
        geometry=replace(
            cfg.geometry,
            vmec_file=resolve_runtime_path(cfg.geometry.vmec_file, base_dir=base_dir),
            geometry_file=resolve_runtime_path(
                cfg.geometry.geometry_file,
                base_dir=base_dir,
            ),
        ),
        init=replace(
            cfg.init,
            init_file=resolve_runtime_path(cfg.init.init_file, base_dir=base_dir),
        ),
        output=replace(
            cfg.output,
            path=resolve_runtime_path(cfg.output.path, base_dir=base_dir),
            restart_to_file=resolve_runtime_path(
                cfg.output.restart_to_file,
                base_dir=base_dir,
            ),
            restart_from_file=resolve_runtime_path(
                cfg.output.restart_from_file,
                base_dir=base_dir,
            ),
        ),
        quasilinear=replace(
            cfg.quasilinear,
            output_path=resolve_runtime_path(
                cfg.quasilinear.output_path,
                base_dir=base_dir,
            ),
        ),
    )


def load_runtime_from_toml(path: str | Path) -> tuple[RuntimeConfig, dict]:
    """Load unified runtime config from TOML, returning ``(cfg, data)``."""

    path = Path(path)
    data = load_toml(path)
    base_dir = path.resolve().parent
    cfg = _apply_runtime_section_overrides(_runtime_base_config(data), data)
    species = _runtime_species_from_toml(data.get("species"))
    if species is not None:
        cfg = replace(cfg, species=species)
    return _resolve_runtime_config_paths(cfg, base_dir=base_dir), data


def load(path: str | Path) -> Case:
    """Load a resolved immutable GKX case from TOML."""
    return load_runtime_from_toml(path)[0]
