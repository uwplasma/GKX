"""Unified runtime configuration schema for linear/nonlinear GK runs."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

from gkx.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)


@dataclass(frozen=True)
class RuntimeSpeciesConfig:
    """Single species definition for runtime-configured simulations."""

    name: str = "ion"
    charge: float = 1.0
    mass: float = 1.0
    density: float = 1.0
    temperature: float = 1.0
    tprim: float = 2.49
    fprim: float = 0.8
    nu: float = 0.0
    kinetic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimePhysicsConfig:
    """Physics-family toggles independent from benchmark case names."""

    reduced_model: str = "gyrokinetic"
    linear: bool = True
    nonlinear: bool = False
    electrostatic: bool = True
    electromagnetic: bool = False
    use_apar: bool = False
    use_bpar: bool = False
    adiabatic_electrons: bool = True
    adiabatic_ions: bool = False
    tau_e: float = 1.0
    tau_fac: float | None = None
    z_ion: float = 1.0
    beta: float = 0.0
    collisions: bool = True
    hypercollisions: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeCollisionConfig:
    """Collision and end-damping parameters."""

    nu_hermite: float = 1.0
    nu_laguerre: float = 2.0
    nu_hyper: float = 0.0
    p_hyper: float = 4.0
    nu_hyper_l: float = 0.0
    nu_hyper_m: float = 1.0
    nu_hyper_lm: float = 0.0
    p_hyper_l: float = 6.0
    p_hyper_m: float | None = None
    p_hyper_lm: float = 6.0
    D_hyper: float = 0.0
    p_hyper_kperp: float = 2.0
    # Reference nonlinear dissipation path: kz-proportional hypercollisions.
    hypercollisions_const: float = 0.0
    hypercollisions_kz: float = 1.0
    damp_ends_amp: float = 0.1
    damp_ends_widthfrac: float = 0.125
    damp_ends_scale_by_dt: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeNormalizationConfig:
    """Normalization contract selection + optional explicit overrides."""

    contract: str = "cyclone"
    rho_star: float | None = None
    omega_d_scale: float | None = None
    omega_star_scale: float | None = None
    diagnostic_norm: str = "rho_star"
    flux_scale: float = 1.0
    wphi_scale: float = 1.0

    def __post_init__(self) -> None:
        diagnostic_norm = str(self.diagnostic_norm).strip().lower()
        object.__setattr__(self, "diagnostic_norm", diagnostic_norm)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeTermsConfig:
    """Term toggles for assembly; applies to linear and nonlinear paths."""

    streaming: float = 1.0
    mirror: float = 1.0
    curvature: float = 1.0
    gradb: float = 1.0
    diamagnetic: float = 1.0
    collisions: float = 1.0
    hypercollisions: float = 1.0
    hyperdiffusion: float = 0.0
    end_damping: float = 1.0
    apar: float = 1.0
    bpar: float = 1.0
    nonlinear: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeExpertConfig:
    """Advanced runtime controls that should rarely be needed."""

    fixed_mode: bool = False
    iky_fixed: int | None = None
    ikx_fixed: int | None = None
    dealias_kz: bool = False
    source: str = "default"
    phi_ext: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeOutputConfig:
    """Artifact-output controls for runtime executable entry points."""

    path: str | None = None
    restart: bool = False
    restart_if_exists: bool = False
    save_for_restart: bool = True
    restart_to_file: str | None = None
    restart_from_file: str | None = None
    restart_with_perturb: bool = False
    append_on_restart: bool = True
    # In-memory sibling of the restart controls above: carry a converged state
    # from one point of a repeated workload (ky scan, parameter scan) to the
    # next instead of restarting every point from the cold initial condition.
    # It has no effect on a single run, which never has a predecessor.
    # Off by default, on measurement rather than caution: the certified
    # adaptive eigensolver's cost is a fixed-size filtered Arnoldi that does
    # not shrink with a better starting vector, and on a fixed-horizon time
    # integration a warm seed removes a startup transient that the benchmark
    # parity decks are pinned to reproduce. See docs/performance.rst.
    warm_start: bool = False
    resolved_diagnostics: bool = True
    # Whether a completed run draws its own figures beside its output. On by
    # default: a saved bundle nobody plotted is a run nobody looked at. Set
    # false (or pass --no-plots) on batch surfaces that only want the data.
    plots: bool = True
    restart_scale: float = 1.0
    nsave: int = 10000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeQuasilinearConfig:
    """Quasilinear transport diagnostics computed from linear states."""

    enabled: bool = False
    mode: str = "weights"
    saturation_rule: str = "none"
    amplitude_normalization: str = "phi_rms"
    kperp_average: str = "phi_weighted"
    csat: float = 1.0
    gamma_floor: float = 0.0
    include_stable_modes: bool = False
    delta_ky: str | float = "auto"
    species: str = "all"
    channels: Tuple[str, ...] = ("es",)
    write_spectrum: bool = True
    output_path: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.channels, str):
            channels = (self.channels,)
        else:
            channels = tuple(self.channels)
        object.__setattr__(self, "channels", channels)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeParallelConfig:
    """Parallel-execution policy for independent scans and future sharded paths."""

    strategy: str = "serial"
    axis: str = "ky"
    batch_size: int | None = None
    num_devices: int | None = None
    strict_identity: bool = True
    profile: bool = False
    backend: str = "auto"
    auto: bool = False

    def __post_init__(self) -> None:
        strategy = str(self.strategy).strip().lower().replace("-", "_")
        strategy_aliases = {
            "none": "serial",
            "off": "serial",
            "batch_ky": "combined_ky",
            "combinedky": "combined_ky",
        }
        strategy = strategy_aliases.get(strategy, strategy)
        allowed_strategies = {
            "serial",
            "batch",
            "combined_ky",
            "device_batch",
            "pmap",
            "pjit",
            "shard_map",
            "state",
            "velocity",
        }
        if strategy not in allowed_strategies:
            raise ValueError(f"Unknown parallel strategy '{self.strategy}'")

        axis = str(self.axis).strip().lower().replace("-", "_")
        if not axis:
            raise ValueError("parallel axis must be nonempty")
        if bool(self.auto):
            # auto resolves strategy and axis from the visible devices, so an
            # explicit disagreeing request is a conflict to report, not a
            # preference to silently overrule.
            if strategy not in {"serial", "shard_map"}:
                raise ValueError(
                    f"[parallel] auto=true resolves to strategy='shard_map', "
                    f"which conflicts with strategy='{self.strategy}'"
                )
            strategy = "shard_map"
            if axis not in {"ky", "species_hermite", "velocity", "s_m"}:
                raise ValueError(
                    f"[parallel] auto=true resolves to axis='species_hermite', "
                    f"which conflicts with axis='{self.axis}'"
                )
            axis = "species_hermite"
        if self.batch_size is not None and int(self.batch_size) < 1:
            raise ValueError("parallel batch_size must be >= 1 when provided")
        if self.num_devices is not None and int(self.num_devices) < 1:
            raise ValueError("parallel num_devices must be >= 1 when provided")

        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "axis", axis)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeConfig:
    """Unified simulation config for runtime-driven GK runs."""

    grid: GridConfig = GridConfig()
    time: TimeConfig = TimeConfig()
    geometry: GeometryConfig = GeometryConfig()
    init: InitializationConfig = InitializationConfig()
    species: Tuple[RuntimeSpeciesConfig, ...] = (RuntimeSpeciesConfig(),)
    physics: RuntimePhysicsConfig = RuntimePhysicsConfig()
    collisions: RuntimeCollisionConfig = RuntimeCollisionConfig()
    normalization: RuntimeNormalizationConfig = RuntimeNormalizationConfig()
    terms: RuntimeTermsConfig = RuntimeTermsConfig()
    expert: RuntimeExpertConfig = RuntimeExpertConfig()
    output: RuntimeOutputConfig = RuntimeOutputConfig()
    quasilinear: RuntimeQuasilinearConfig = RuntimeQuasilinearConfig()
    parallel: RuntimeParallelConfig = RuntimeParallelConfig()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "init": self.init.to_dict(),
            "species": [s.to_dict() for s in self.species],
            "physics": self.physics.to_dict(),
            "collisions": self.collisions.to_dict(),
            "normalization": self.normalization.to_dict(),
            "terms": self.terms.to_dict(),
            "expert": self.expert.to_dict(),
            "output": self.output.to_dict(),
            "quasilinear": self.quasilinear.to_dict(),
            "parallel": self.parallel.to_dict(),
        }

    def replace(self, **changes: Any) -> "RuntimeConfig":
        """Return a validated copy with ``changes`` applied.

        The dataclass is frozen, so this is the supported way to derive a case.
        The copy is validated before it is returned: a scan that builds cases in
        a loop should fail on the case it built, not several stages later inside
        a compiled kernel.
        """

        updated = dataclasses.replace(self, **changes)
        updated.validate()
        return updated

    def validate(self) -> "RuntimeConfig":
        """Check cross-section consistency and return ``self``.

        Each section already validates its own fields in ``__post_init__``.
        What no section can see is a disagreement between sections, which is
        what this checks.
        """

        errors: list[str] = []
        if not self.species:
            errors.append("a case needs at least one species")
        kinetic = [s for s in self.species if getattr(s, "kinetic", False)]
        if not kinetic:
            errors.append("a case needs at least one kinetic species")
        if bool(self.physics.linear) and bool(self.physics.nonlinear):
            errors.append("physics.linear and physics.nonlinear cannot both be true")
        if float(self.time.dt) <= 0.0:
            errors.append("time.dt must be positive")
        if float(self.time.t_max) <= 0.0:
            errors.append("time.t_max must be positive")
        if self.geometry.model == "vmec" and not self.geometry.vmec_file:
            errors.append("geometry.model = 'vmec' requires geometry.vmec_file")
        if errors:
            raise ValueError("invalid case: " + "; ".join(errors))
        return self

    def summary(self) -> Dict[str, Any]:
        """Return the scalars that identify this case in a log or a report."""

        return {
            "geometry_model": self.geometry.model,
            "n_species": len(self.species),
            "n_kinetic_species": sum(
                1 for s in self.species if getattr(s, "kinetic", False)
            ),
            "linear": bool(self.physics.linear),
            "nonlinear": bool(self.physics.nonlinear),
            "electromagnetic": bool(getattr(self.physics, "electromagnetic", False)),
            "grid": {
                "Nx": self.grid.Nx,
                "Ny": self.grid.Ny,
                "Nz": self.grid.Nz,
                "y0": self.grid.y0,
            },
            "time": {
                "t_max": self.time.t_max,
                "dt": self.time.dt,
                "method": self.time.method,
                "fixed_dt": self.time.fixed_dt,
            },
        }

    def to_toml(self, path: str | Path) -> Path:
        """Write this case as TOML and return the path written.

        The text round-trips through ``gkx.load``: what is written is what the
        loader reads back, so a case saved from Python and a case edited by hand
        are the same object to the runtime.
        """

        from gkx.workflows.runtime.wout import deck_text

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(deck_text(self.to_dict()), encoding="utf-8")
        return target


Case = RuntimeConfig
