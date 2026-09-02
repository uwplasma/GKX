from __future__ import annotations

import json

from dataclasses import asdict, dataclass
from typing import Any, Dict
import dataclasses
from pathlib import Path
from typing import Tuple

REFERENCE_ELECTRON_MASS: float = 2.7e-4
REFERENCE_MASS_RATIO: float = 1.0 / REFERENCE_ELECTRON_MASS


def explicit_method_default_cfl_fac(method: str) -> float:
    """Return the reference explicit-method CFL prefactor for a given method."""

    method_key = method.strip().lower()
    if method_key in {"rk3", "sspx3"}:
        return 1.73
    if method_key == "rk4":
        return 2.82
    return 1.0


def resolve_cfl_fac(method: str, cfl_fac: float | None) -> float:
    """Resolve an explicit CFL prefactor, falling back to the method default."""

    if cfl_fac is None:
        return explicit_method_default_cfl_fac(method)
    return float(cfl_fac)


@dataclass(frozen=True)
class InitializationConfig:
    """Initialization options for linear runs."""

    init_field: str = "density"
    init_amp: float = 1.0e-5
    init_single: bool = True
    random_seed: int = 22
    gaussian_init: bool = False
    gaussian_width: float = 0.5
    gaussian_envelope_constant: float = 1.0
    gaussian_envelope_sine: float = 0.0
    kpar_init: float = 0.0
    init_file: str | None = None
    init_file_scale: float = 1.0
    init_file_mode: str = "replace"
    init_electrons_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GridConfig:
    """Spectral grid configuration in a flux-tube."""

    Nx: int = 48
    Ny: int = 48
    Nz: int = 64
    Lx: float = 62.8
    Ly: float = 62.8
    boundary: str = "periodic"
    jtwist: int | None = None
    non_twist: bool = False
    kxfac: float = 1.0
    z_min: float = -3.141592653589793
    z_max: float = 3.141592653589793
    y0: float | None = None
    ntheta: int | None = None
    nperiod: int | None = None
    zp: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimeConfig:
    """Time integration parameters."""

    t_max: float = 100.0
    dt: float = 0.1
    method: str = "rk2"
    sample_stride: int = 1
    diagnostics_stride: int = 1
    diagnostics: bool = True
    save_state: bool = False
    checkpoint: bool = False
    implicit_restart: int = 20
    implicit_preconditioner: str | None = None
    state_sharding: str | None = None
    progress_bar: bool = False
    fixed_dt: bool = True
    dt_min: float = 1.0e-7
    dt_max: float | None = None
    cfl: float = 0.9
    cfl_fac: float | None = None
    nstep_restart: int | None = None
    collision_split: bool = False
    collision_scheme: str = "implicit"
    collision_operator: str = "none"
    compressed_real_fft: bool = True
    nonlinear_dealias: bool = True
    laguerre_nonlinear_mode: str = "grid"
    # Nonlinear stop policy. "saturation" (the default) ends a diagnosed
    # nonlinear run once the post-spin-up heat-flux window statistics converge,
    # with t_max as the hard cap; "t_max" always integrates the full horizon.
    # Linear runs ignore these keys.
    run_to: str = "saturation"
    saturation_rel_sem: float = 0.05
    saturation_min_window: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, init=False)
class GeometryConfig:
    """Flux-tube geometry parameters or imported sampled geometry settings."""

    model: str = "s-alpha"
    geometry_backend: str = "auto"
    geometry_file: str | None = None
    vmec_file: str | None = None
    geometry_helper_python: str | None = None
    rhoc: float = 0.5
    R_geo: float | None = None
    shift: float = 0.0
    akappa: float = 1.0
    akappri: float = 0.0
    tri: float = 0.0
    tripri: float = 0.0
    torflux: float | None = None
    npol: float | None = None
    npol_min: float | None = None
    isaxisym: bool = False
    which_crossing: int | None = None
    include_shear_variation: bool = False
    include_pressure_variation: bool = False
    betaprim: float | None = None
    geometry_helper_repo: str | None = None
    q: float = 1.4
    s_hat: float = 0.8
    z0: float | None = None
    zero_shat: bool = False
    epsilon: float = 0.18
    R0: float = 1.0
    B0: float = 1.0
    alpha: float = 0.0
    drift_scale: float = 1.0
    kperp2_bmag: bool = True
    bessel_bmag_power: float = 0.0

    def __init__(
        self,
        model: str = "s-alpha",
        geometry_backend: str = "auto",
        geometry_file: str | None = None,
        vmec_file: str | None = None,
        geometry_helper_python: str | None = None,
        rhoc: float = 0.5,
        R_geo: float | None = None,
        shift: float = 0.0,
        akappa: float = 1.0,
        akappri: float = 0.0,
        tri: float = 0.0,
        tripri: float = 0.0,
        torflux: float | None = None,
        npol: float | None = None,
        npol_min: float | None = None,
        isaxisym: bool = False,
        which_crossing: int | None = None,
        include_shear_variation: bool = False,
        include_pressure_variation: bool = False,
        betaprim: float | None = None,
        geometry_helper_repo: str | None = None,
        q: float = 1.4,
        s_hat: float = 0.8,
        z0: float | None = None,
        zero_shat: bool = False,
        epsilon: float = 0.18,
        R0: float = 1.0,
        B0: float = 1.0,
        alpha: float = 0.0,
        drift_scale: float = 1.0,
        kperp2_bmag: bool = True,
        bessel_bmag_power: float = 0.0,
    ) -> None:
        values = {
            "model": model,
            "geometry_backend": geometry_backend,
            "geometry_file": geometry_file,
            "vmec_file": vmec_file,
            "geometry_helper_python": geometry_helper_python,
            "rhoc": rhoc,
            "R_geo": R_geo,
            "shift": shift,
            "akappa": akappa,
            "akappri": akappri,
            "tri": tri,
            "tripri": tripri,
            "torflux": torflux,
            "npol": npol,
            "npol_min": npol_min,
            "isaxisym": isaxisym,
            "which_crossing": which_crossing,
            "include_shear_variation": include_shear_variation,
            "include_pressure_variation": include_pressure_variation,
            "betaprim": betaprim,
            "geometry_helper_repo": geometry_helper_repo,
            "q": q,
            "s_hat": s_hat,
            "z0": z0,
            "zero_shat": zero_shat,
            "epsilon": epsilon,
            "R0": R0,
            "B0": B0,
            "alpha": alpha,
            "drift_scale": drift_scale,
            "kperp2_bmag": kperp2_bmag,
            "bessel_bmag_power": bessel_bmag_power,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Literature benchmark presets used by the executable, examples, and public API.
@dataclass(frozen=True)
class ModelConfig:
    r"""Dimensionless gradients for the Cyclone base case.

    These are :math:`a/L_T` and :math:`a/L_n` -- the same quantities the TOML
    ``tprim``/``fprim`` keys carry and the only ones the linear operator
    consumes. With the Cyclone :math:`R_0 = R/a = 2.77778`, ``tprim_i = 2.49``
    is the literature :math:`R/L_T = 6.92`.
    """

    tprim_i: float = 2.49
    tprim_e: float = 0.0
    fprim: float = 0.8
    nu_i: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CycloneBaseCase:
    """Standard parameters for the Cyclone base case ITG benchmark."""

    grid: GridConfig = GridConfig(
        Nx=1,
        Ny=24,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        boundary="linked",
        y0=20.0,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig(
        t_max=150.0,
        dt=0.01,
        method="rk4",
        fixed_dt=False,
        dt_max=0.05,
    )
    geometry: GeometryConfig = GeometryConfig(
        R0=2.77778,
        drift_scale=1.0,
    )
    model: ModelConfig = ModelConfig()
    init: InitializationConfig = InitializationConfig(
        init_field="density",
        init_amp=1.0e-10,
        gaussian_init=True,
        gaussian_width=0.5,
        gaussian_envelope_constant=1.0,
        gaussian_envelope_sine=0.0,
    )
    reference_aligned: bool = True

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
            "init": self.init.to_dict(),
            "reference_alignment": {"enabled": self.reference_aligned},
        }


@dataclass(frozen=True)
class KineticElectronModelConfig:
    r"""Gradients and ratios for a kinetic-electron Cyclone-base-case setup.

    Gradients are :math:`a/L_T` and :math:`a/L_n`, as in :class:`ModelConfig`.
    """

    tprim_i: float = 2.49
    tprim_e: float = 2.49
    fprim: float = 0.8
    Te_over_Ti: float = 1.0
    mass_ratio: float = REFERENCE_MASS_RATIO
    nu_i: float = 0.0
    nu_e: float = 0.0
    beta: float = 1.0e-5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KBMBaseCase:
    """Parameters for an electromagnetic KBM benchmark."""

    grid: GridConfig = GridConfig(
        Nx=1,
        Ny=16,
        Nz=96,
        Lx=62.8,
        Ly=62.8,
        boundary="linked",
        y0=10.0,
        ntheta=32,
        nperiod=2,
    )
    time: TimeConfig = TimeConfig(
        t_max=40.0,
        dt=0.01,
        method="rk4",
    )
    geometry: GeometryConfig = GeometryConfig(R0=2.77778)
    model: KineticElectronModelConfig = KineticElectronModelConfig(beta=0.015)
    init: InitializationConfig = InitializationConfig(
        init_field="all",
        init_amp=1.0e-10,
        gaussian_init=True,
    )

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            "grid": self.grid.to_dict(),
            "time": self.time.to_dict(),
            "geometry": self.geometry.to_dict(),
            "model": self.model.to_dict(),
            "init": self.init.to_dict(),
        }


__all__ = [
    "CycloneBaseCase",
    "GeometryConfig",
    "GridConfig",
    "InitializationConfig",
    "KBMBaseCase",
    "KineticElectronModelConfig",
    "ModelConfig",
    "REFERENCE_ELECTRON_MASS",
    "REFERENCE_MASS_RATIO",
    "TimeConfig",
    "explicit_method_default_cfl_fac",
    "resolve_cfl_fac",
]


# ---- deck configuration, lifted from workflows/runtime/config.py ----
# RuntimeConfig, aliased Case, is the type a user meets first, and it sat in
# the deepest layer while geometry imported it from the shallowest -- the worst
# inversion in the dependency graph. The deck model depended on nothing but
# this file, so it belongs here.


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


# Every strategy a deck may name. Named rather than inlined because consumers
# have to classify the whole set, not just the value in front of them: the ky
# scan decides per strategy whether it describes the scan or the solve, and it
# checks that decision against this set so a strategy added here cannot reach a
# worker unclassified.
PARALLEL_STRATEGIES = frozenset(
    {
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
)


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
        if strategy not in PARALLEL_STRATEGIES:
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

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(deck_text(self.to_dict()), encoding="utf-8")
        return target


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


Case = RuntimeConfig
