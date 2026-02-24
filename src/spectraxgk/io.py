"""TOML-based input helpers for CLI and driver scripts."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any, cast
from pathlib import Path
import tomllib

from spectraxgk.config import (
    CycloneBaseCase,
    ETGBaseCase,
    KineticElectronBaseCase,
    KBMBaseCase,
    TEMBaseCase,
)
from spectraxgk.linear import LinearTerms
from spectraxgk.linear_krylov import KrylovConfig
from spectraxgk.runtime_config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)
from spectraxgk.terms.config import TermConfig


def load_toml(path: str | Path) -> dict:
    """Load a TOML file into a plain dictionary."""

    path = Path(path)
    with path.open("rb") as f:
        return tomllib.load(f)


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


def _case_registry():
    return {
        "cyclone": CycloneBaseCase,
        "etg": ETGBaseCase,
        "kinetic_itg": KineticElectronBaseCase,
        "kbm": KBMBaseCase,
        "tem": TEMBaseCase,
    }


def load_case_from_toml(path: str | Path, case_name: str | None = None):
    """Load a case config from TOML, returning (case_name, case_config, data)."""

    data = load_toml(path)
    if case_name is None:
        case_name = str(data.get("case", "cyclone")).lower()
    registry = _case_registry()
    if case_name not in registry:
        raise ValueError(f"Unknown case '{case_name}'. Available: {', '.join(registry)}")
    cfg = registry[case_name]()
    overrides = {
        "grid": data.get("grid"),
        "time": data.get("time"),
        "geometry": data.get("geometry"),
        "model": data.get("model"),
        "init": data.get("init"),
    }
    cfg = _merge_dataclass(cfg, overrides)
    return case_name, cfg, data


def load_runtime_from_toml(path: str | Path) -> tuple[RuntimeConfig, dict]:
    """Load unified runtime config from TOML, returning ``(cfg, data)``."""

    data = load_toml(path)
    cfg: RuntimeConfig = RuntimeConfig()
    cfg = _merge_dataclass(
        cfg,
        {
            "grid": data.get("grid"),
            "time": data.get("time"),
            "geometry": data.get("geometry"),
            "init": data.get("init"),
        },
    )
    physics = data.get("physics")
    if isinstance(physics, dict):
        cfg = replace(cfg, physics=RuntimePhysicsConfig(**physics))
    collisions = data.get("collisions")
    if isinstance(collisions, dict):
        cfg = replace(cfg, collisions=RuntimeCollisionConfig(**collisions))
    normalization = data.get("normalization")
    if isinstance(normalization, dict):
        cfg = replace(cfg, normalization=RuntimeNormalizationConfig(**normalization))
    terms = data.get("terms")
    if isinstance(terms, dict):
        cfg = replace(cfg, terms=RuntimeTermsConfig(**terms))
    species_raw = data.get("species")
    if species_raw is not None:
        if not isinstance(species_raw, list):
            raise TypeError("[[species]] entries must be provided as an array of tables")
        species: list[RuntimeSpeciesConfig] = []
        for item in species_raw:
            if not isinstance(item, dict):
                raise TypeError("Each [[species]] entry must be a table")
            species.append(RuntimeSpeciesConfig(**item))
        if species:
            cfg = replace(cfg, species=tuple(species))
    return cfg, data


def load_linear_terms_from_toml(data: dict) -> LinearTerms | None:
    """Parse LinearTerms from a TOML dict."""

    terms = data.get("terms")
    if not isinstance(terms, dict):
        return None
    return LinearTerms(**terms)


def load_krylov_from_toml(data: dict) -> KrylovConfig | None:
    """Parse KrylovConfig from a TOML dict."""

    krylov = data.get("krylov")
    if not isinstance(krylov, dict):
        return None
    return KrylovConfig(**krylov)


def load_term_config_from_toml(data: dict) -> TermConfig | None:
    """Parse TermConfig for nonlinear runs from TOML dict."""

    terms = data.get("terms")
    if not isinstance(terms, dict):
        return None
    return TermConfig(**terms)
