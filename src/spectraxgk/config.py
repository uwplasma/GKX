from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

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
    implicit_solve_method: str = "batched"
    use_diffrax: bool = True
    diffrax_solver: str = "Dopri8"
    diffrax_adaptive: bool = False
    diffrax_rtol: float = 1.0e-5
    diffrax_atol: float = 1.0e-7
    diffrax_max_steps: int = 4096
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
    compressed_real_fft: bool = True
    nonlinear_dealias: bool = True
    laguerre_nonlinear_mode: str = "grid"

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


from spectraxgk.validation.benchmarks.case_configs import (  # noqa: E402
    CycloneBaseCase,
    ETGBaseCase,
    ETGModelConfig,
    KBMBaseCase,
    KineticElectronBaseCase,
    KineticElectronModelConfig,
    ModelConfig,
    TEMBaseCase,
    TEMModelConfig,
)

__all__ = [
    "CycloneBaseCase",
    "ETGBaseCase",
    "ETGModelConfig",
    "GeometryConfig",
    "GridConfig",
    "InitializationConfig",
    "KBMBaseCase",
    "KineticElectronBaseCase",
    "KineticElectronModelConfig",
    "ModelConfig",
    "REFERENCE_ELECTRON_MASS",
    "REFERENCE_MASS_RATIO",
    "TEMBaseCase",
    "TEMModelConfig",
    "TimeConfig",
    "explicit_method_default_cfl_fac",
    "resolve_cfl_fac",
]
