"""TOML-based input helpers for the executable and driver scripts."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any, Callable, Sequence, cast
import os
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - only on Python <3.11
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

from gkx.workflows.runtime.config import (
    RuntimeCollisionConfig,
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


def load_toml(path: str | Path) -> dict:
    """Load a TOML file into a plain dictionary."""

    path = Path(path)
    with path.open("rb") as f:
        return tomllib.load(f)


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
    """Return parser arguments for ``gkx case.toml`` shorthand."""

    if not argv:
        return None
    config_arg = argv[0]
    if config_arg.startswith("-") or config_arg in EXECUTABLE_TOML_SHORTHAND_COMMANDS:
        return None
    if not Path(config_arg).exists():
        return None
    command = toml_shorthand_command(load_toml_func(config_arg))
    return [command, "--config", config_arg, *argv[1:]]


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
