"""Runtime startup and initialization helpers.

This module holds the geometry/loading/initial-condition logic used by the
public runtime entry points. It is intentionally kept separate from the solver
execution layer so startup behavior can be tested and refactored without
touching the time-integration control flow.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, NoReturn, Sequence

import jax.numpy as jnp

from gkx.geometry import FluxTubeGeometryLike, build_flux_tube_geometry
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from gkx.solvers.linear.krylov import KrylovConfig
from gkx.geometry.miller_eik import generate_runtime_miller_eik
from gkx.diagnostics.normalization import get_normalization_contract
from gkx.artifacts.io import load_netcdf_restart_state
from gkx.workflows.runtime.config import RuntimeConfig, RuntimeSpeciesConfig
from gkx.operators.linear.params import Species, build_linear_params
from gkx.terms.config import TermConfig
from gkx.geometry.vmec_eik import generate_runtime_vmec_eik
from dataclasses import dataclass
from pathlib import Path
from typing import cast
import numpy as np
from gkx.core_grid import SpectralGrid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.workflows.runtime.initial_phi import _density_moments_for_target_phi
from gkx.workflows.runtime.initial_phi import (
    _as_runtime_species_array,
)

__all__ = [
    "_build_gaussian_profile",
    "_build_initial_condition",
    "_build_single_phi_gaussian_profile",
    "_as_runtime_species_array",
    "_centered_glibc_random_pairs",
    "_dealiased_initial_mode_pairs",
    "_density_moments_for_target_phi",
    "_enforce_full_ky_hermitian",
    "_expand_ky",
    "_load_initial_state_from_file",
    "_periodic_zp_from_grid",
    "_reshape_netcdf_state",
    "build_runtime_geometry",
    "build_runtime_linear_params",
    "build_runtime_linear_terms",
    "build_runtime_term_config",
    "load_netcdf_restart_state",
    "runtime_geometry_config_for_builder",
]


def _species_to_linear(species_cfg: Sequence[RuntimeSpeciesConfig]) -> list[Species]:
    kinetic = [s for s in species_cfg if bool(s.kinetic)]
    if not kinetic:
        raise ValueError(
            "RuntimeConfig.species must include at least one kinetic species"
        )
    return [
        Species(
            charge=float(s.charge),
            mass=float(s.mass),
            density=float(s.density),
            temperature=float(s.temperature),
            tprim=float(s.tprim),
            fprim=float(s.fprim),
            nu=float(s.nu),
        )
        for s in kinetic
    ]


def _default_hermite_hypercollision_exponent(nhermite: int | None) -> float:
    """Return the default Hermite hypercollision exponent."""

    if nhermite is None:
        return 20.0
    return float(min(20, max(int(nhermite) // 2, 1)))


def _runtime_model_key(cfg: RuntimeConfig) -> str:
    return cfg.physics.reduced_model.strip().lower()


def _raise_unsupported_reduced_model(cfg: RuntimeConfig) -> NoReturn:
    """Fail closed unless the promoted full gyrokinetic runtime is selected."""

    raise ValueError(
        f"Unknown physics.reduced_model={cfg.physics.reduced_model!r}. "
        "Use physics.reduced_model='gyrokinetic' for promoted full-GK workflows."
    )


def _runtime_default_krylov_config(cfg: RuntimeConfig) -> KrylovConfig:
    """Return a model-aware Krylov default for runtime-configured linear runs."""

    contract = cfg.normalization.contract.strip().lower()
    kinetic_species = tuple(spec for spec in cfg.species if spec.kinetic)
    electron_only = len(kinetic_species) == 1 and float(kinetic_species[0].charge) < 0.0

    if contract == "etg" or (
        cfg.physics.adiabatic_ions
        and cfg.physics.electrostatic
        and not cfg.physics.electromagnetic
        and electron_only
    ):
        return KrylovConfig(
            method="shift_invert",
            krylov_dim=16,
            restarts=1,
            omega_min_factor=0.0,
            omega_target_factor=0.4,
            omega_cap_factor=1.5,
            omega_sign=-1,
            power_iters=80,
            power_dt=0.002,
            shift_source="target",
            shift_tol=1.0e-3,
            shift_maxiter=40,
            shift_restart=12,
            shift_solve_method="batched",
            shift_preconditioner="auto",
            shift_selection="targeted",
            mode_family="etg",
            fallback_method="arnoldi",
            fallback_real_floor=-1.0e-6,
        )

    if contract == "kbm":
        # Keep the benchmark-specific branch policy in its validation owner.
        from gkx.benchmarking_shared import KBM_KRYLOV_DEFAULT

        return KBM_KRYLOV_DEFAULT

    # Generic contracts (cyclone included) use the residual-certified adaptive
    # eigensolve. The raw propagator default has no certification gate and can
    # silently return an unconverged, unphysical branch at collisional runtime
    # resolutions; the adaptive path either certifies the dominant pair against
    # the continuous operator or fails closed with the measured residual.
    return KrylovConfig(method="adaptive")


def _resolve_runtime_hl_dims(
    cfg: RuntimeConfig,
    *,
    Nl: int | None,
    Nm: int | None,
) -> tuple[int, int]:
    """Resolve model-native Hermite/Laguerre dimensions."""

    model = _runtime_model_key(cfg)
    if model in {"", "gyrokinetic", "full", "full-gk"}:
        return int(24 if Nl is None else Nl), int(12 if Nm is None else Nm)
    _raise_unsupported_reduced_model(cfg)


def _require_full_gk_runtime_model(cfg: RuntimeConfig) -> None:
    """Reject non-promoted reduced-model configs before full-GK execution."""

    model = _runtime_model_key(cfg)
    if model in {"", "gyrokinetic", "full", "full-gk"}:
        return
    _raise_unsupported_reduced_model(cfg)


def runtime_geometry_config_for_builder(
    cfg: RuntimeConfig,
    *,
    vmec_eik_builder: Callable[[RuntimeConfig], Any],
    miller_eik_builder: Callable[[RuntimeConfig], Any],
) -> Any:
    """Return the geometry config consumed by the flux-tube builder."""

    model = cfg.geometry.model.strip().lower()
    if model == "vmec":
        eik_path = vmec_eik_builder(cfg)
        return replace(cfg.geometry, model="vmec-eik", geometry_file=str(eik_path))
    if model == "miller":
        eik_path = miller_eik_builder(cfg)
        return replace(cfg.geometry, model="imported-eik", geometry_file=str(eik_path))
    return cfg.geometry


def build_runtime_geometry(cfg: RuntimeConfig) -> FluxTubeGeometryLike:
    """Resolve runtime geometry, generating `*.eik.nc` geometry when needed."""

    return build_flux_tube_geometry(
        runtime_geometry_config_for_builder(
            cfg,
            vmec_eik_builder=generate_runtime_vmec_eik,
            miller_eik_builder=generate_runtime_miller_eik,
        )
    )


def build_runtime_linear_params(
    cfg: RuntimeConfig,
    *,
    Nm: int | None = None,
    geom: FluxTubeGeometryLike | None = None,
) -> LinearParams:
    """Build `LinearParams` from a unified runtime config."""

    _require_full_gk_runtime_model(cfg)
    if geom is None:
        geom = build_runtime_geometry(cfg)
    contract = get_normalization_contract(cfg.normalization.contract)
    rho_star = (
        contract.rho_star
        if cfg.normalization.rho_star is None
        else float(cfg.normalization.rho_star)
    )
    omega_d_scale = (
        contract.omega_d_scale
        if cfg.normalization.omega_d_scale is None
        else float(cfg.normalization.omega_d_scale)
    )
    omega_star_scale = (
        contract.omega_star_scale
        if cfg.normalization.omega_star_scale is None
        else float(cfg.normalization.omega_star_scale)
    )

    species = _species_to_linear(cfg.species)
    has_kinetic_electron = any(float(s.charge) < 0.0 for s in species)
    has_kinetic_ion = any(float(s.charge) > 0.0 for s in species)
    if cfg.physics.adiabatic_electrons and cfg.physics.adiabatic_ions:
        raise ValueError(
            "adiabatic_electrons and adiabatic_ions are mutually exclusive"
        )
    if cfg.physics.adiabatic_electrons and has_kinetic_electron:
        raise ValueError(
            "adiabatic_electrons=True conflicts with kinetic electron species"
        )
    if cfg.physics.adiabatic_ions and has_kinetic_ion:
        raise ValueError("adiabatic_ions=True conflicts with kinetic ion species")

    # ``tau_e`` is the field solver's historical name for the Boltzmann-species
    # quasineutrality coefficient; it applies to either adiabatic species.
    has_boltzmann_species = (
        cfg.physics.adiabatic_electrons or cfg.physics.adiabatic_ions
    )
    tau_e = float(cfg.physics.tau_e) if has_boltzmann_species else 0.0
    beta = float(cfg.physics.beta) if cfg.physics.electromagnetic else 0.0
    fapar = (
        1.0
        if (cfg.physics.electromagnetic and cfg.physics.use_apar and beta > 0.0)
        else 0.0
    )
    p_hyper_m = cfg.collisions.p_hyper_m
    if p_hyper_m is None:
        p_hyper_m = _default_hermite_hypercollision_exponent(Nm)

    params = build_linear_params(
        species,
        tau_e=tau_e,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=float(omega_d_scale),
        omega_star_scale=float(omega_star_scale),
        rho_star=float(rho_star),
        beta=beta,
        fapar=fapar,
        nu_hyper=float(cfg.collisions.nu_hyper),
        p_hyper=float(cfg.collisions.p_hyper),
        nu_hyper_l=float(cfg.collisions.nu_hyper_l),
        nu_hyper_m=float(cfg.collisions.nu_hyper_m),
        nu_hyper_lm=float(cfg.collisions.nu_hyper_lm),
        p_hyper_l=float(cfg.collisions.p_hyper_l),
        p_hyper_m=float(p_hyper_m),
        p_hyper_lm=float(cfg.collisions.p_hyper_lm),
        D_hyper=float(cfg.collisions.D_hyper),
        p_hyper_kperp=float(cfg.collisions.p_hyper_kperp),
        hypercollisions_const=float(cfg.collisions.hypercollisions_const),
        hypercollisions_kz=float(cfg.collisions.hypercollisions_kz),
    )
    return replace(
        params,
        nu_hermite=float(cfg.collisions.nu_hermite),
        nu_laguerre=float(cfg.collisions.nu_laguerre),
        damp_ends_amp=(
            float(cfg.collisions.damp_ends_amp) / float(cfg.time.dt)
            if cfg.collisions.damp_ends_scale_by_dt and float(cfg.time.dt) != 0.0
            else float(cfg.collisions.damp_ends_amp)
        ),
        damp_ends_widthfrac=float(cfg.collisions.damp_ends_widthfrac),
    )


def build_runtime_linear_terms(cfg: RuntimeConfig) -> LinearTerms:
    """Build `LinearTerms` from unified toggles."""

    em_on = bool(cfg.physics.electromagnetic)
    use_apar = em_on and bool(cfg.physics.use_apar)
    use_bpar = em_on and bool(cfg.physics.use_bpar)
    collisions_on = bool(cfg.physics.collisions) and any(
        float(sp.nu) != 0.0 for sp in cfg.species
    )
    hyper_on = bool(cfg.physics.hypercollisions)
    return LinearTerms(
        streaming=float(cfg.terms.streaming),
        mirror=float(cfg.terms.mirror),
        curvature=float(cfg.terms.curvature),
        gradb=float(cfg.terms.gradb),
        diamagnetic=float(cfg.terms.diamagnetic),
        collisions=float(cfg.terms.collisions if collisions_on else 0.0),
        hypercollisions=float(cfg.terms.hypercollisions if hyper_on else 0.0),
        hyperdiffusion=float(cfg.terms.hyperdiffusion),
        end_damping=float(cfg.terms.end_damping),
        apar=float(cfg.terms.apar if use_apar else 0.0),
        bpar=float(cfg.terms.bpar if use_bpar else 0.0),
    )


def build_runtime_term_config(cfg: RuntimeConfig) -> TermConfig:
    """Build nonlinear-ready `TermConfig` from unified toggles."""

    lin_terms = build_runtime_linear_terms(cfg)
    nonlinear_on = float(cfg.terms.nonlinear if cfg.physics.nonlinear else 0.0)
    return linear_terms_to_term_config(lin_terms, nonlinear=nonlinear_on)


def _build_initial_condition(
    grid,
    geom: FluxTubeGeometryLike,
    cfg: RuntimeConfig,
    *,
    ky_index: int,
    kx_index: int,
    Nl: int,
    Nm: int,
    nspecies: int,
) -> jnp.ndarray:
    """Build the runtime initial state using this module's patchable params builder."""

    return _build_initial_condition_impl(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=Nl,
        Nm=Nm,
        nspecies=nspecies,
        build_runtime_linear_params_fn=build_runtime_linear_params,
    )


# ---- merged from workflows/runtime/initial_conditions.py ----
# That module had exactly one consumer -- this one -- which imported ten of
# its private names. A boundary that wide is a split, not an interface.

_GLIBC_RAND_MAX = float((1 << 31) - 1)
_INITIAL_FIELD_MOMENTS: dict[str, tuple[int, int]] = {
    "density": (0, 0),
    "upar": (0, 1),
    "tpar": (0, 2),
    "tperp": (1, 0),
    "qpar": (0, 3),
    "qperp": (1, 1),
}
_ALL_FIELD_SCALES: dict[str, float] = {
    "density": 1.0,
    "upar": 1.0,
    "tpar": 1.0 / np.sqrt(2.0),
    "tperp": 1.0,
    "qpar": 1.0 / np.sqrt(6.0),
    "qperp": 1.0,
}
_VALID_INIT_FIELDS = {"all", "phi", *_INITIAL_FIELD_MOMENTS.keys()}


def _build_gaussian_profile(
    z: np.ndarray,
    *,
    kx: float,
    ky: float,
    s_hat: float,
    width: float,
    envelope_constant: float,
    envelope_sine: float,
) -> np.ndarray:
    if ky == 0.0:
        return np.zeros_like(z)
    theta0 = kx / (s_hat * ky)
    env = envelope_constant + envelope_sine * np.sin(z - theta0)
    return env * np.exp(-(((z - theta0) / width) ** 2))


def _build_single_phi_gaussian_profile(
    z: np.ndarray,
    *,
    kx: float,
    ky: float,
    s_hat: float,
    width: float,
    envelope_constant: float,
    envelope_sine: float,
) -> np.ndarray:
    """Return a single-mode Gaussian potential profile along the flux tube.

    The W7-X zonal-flow benchmark prescribes a Gaussian electrostatic-potential
    perturbation centered in the middle of the flux tube. A multi-mode
    ballooning-angle Gaussian initializer is undefined for ``ky=0`` because its
    center uses ``kx / (s_hat * ky)``; for the zonal case the physically stated
    center is therefore the tube midpoint, ``z=0``.
    """

    if ky != 0.0 and s_hat != 0.0:
        return _build_gaussian_profile(
            z,
            kx=kx,
            ky=ky,
            s_hat=s_hat,
            width=width,
            envelope_constant=envelope_constant,
            envelope_sine=envelope_sine,
        )
    center = 0.0
    env = envelope_constant + envelope_sine * np.sin(z - center)
    return env * np.exp(-(((z - center) / width) ** 2))


def _reshape_netcdf_state(
    raw: np.ndarray,
    *,
    nspec: int,
    nl: int,
    nm: int,
    nyc: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    nR = nyc * nx * nz
    arr = raw.reshape((nspec, nm, nl, nR)).transpose(0, 2, 1, 3)
    ky_idx = np.arange(nyc)[:, None, None]
    kx_idx = np.arange(nx)[None, :, None]
    z_idx = np.arange(nz)[None, None, :]
    idxyz = ky_idx + nyc * (kx_idx + nx * z_idx)
    arr_reordered = arr[..., idxyz.ravel()]
    return arr_reordered.reshape((nspec, nl, nm, nyc, nx, nz))


def _expand_ky(arr: np.ndarray, *, nyc: int) -> np.ndarray:
    ny_full = 2 * (nyc - 1)
    if ny_full <= 0 or arr.shape[-3] == ny_full:
        return arr
    if nyc <= 2:
        return arr
    pos = arr
    neg = np.conj(pos[..., 1 : nyc - 1, :, :])
    neg = neg[..., ::-1, :, :]
    nx = pos.shape[-2]
    if nx > 1:
        kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
        neg = neg[..., kx_neg, :]
    return np.concatenate([pos, neg], axis=-3)


def _enforce_full_ky_hermitian(arr: np.ndarray) -> np.ndarray:
    """Mirror positive-`ky` content into the negative branch for full FFT grids."""

    state = np.asarray(arr, dtype=np.complex64)
    ny = int(state.shape[-3])
    if ny <= 1:
        return state
    nyc = ny // 2 + 1
    neg_hi = nyc - 1 if (ny % 2) == 0 else nyc
    if neg_hi <= 1:
        return state
    neg = np.conj(state[..., 1:neg_hi, :, :])[..., ::-1, :, :]
    nx = int(state.shape[-2])
    if nx > 1:
        kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
        neg = neg[..., kx_neg, :]
    state[..., nyc:, :, :] = neg
    return state


def _load_initial_state_from_file(
    path: Path,
    *,
    nspecies: int,
    Nl: int,
    Nm: int,
    ny: int,
    nx: int,
    nz: int,
) -> np.ndarray:
    if path.suffix.lower() == ".nc":
        return load_netcdf_restart_state(
            path,
            nspecies=nspecies,
            Nl=Nl,
            Nm=Nm,
            ny=ny,
            nx=nx,
            nz=nz,
        )
    raw = np.fromfile(path, dtype=np.complex64)
    nyc = ny // 2 + 1
    expected_nyc = nspecies * Nl * Nm * nyc * nx * nz
    expected_full = nspecies * Nl * Nm * ny * nx * nz
    if raw.size == expected_nyc:
        arr = _reshape_netcdf_state(
            raw, nspec=nspecies, nl=Nl, nm=Nm, nyc=nyc, nx=nx, nz=nz
        )
        return _expand_ky(arr, nyc=nyc)
    if raw.size == expected_full:
        return raw.reshape((nspecies, Nl, Nm, ny, nx, nz))
    raise ValueError(
        f"init_file size {raw.size} does not match expected {expected_nyc} (nyc) or {expected_full} (full)"
    )


def _centered_glibc_random_pairs(seed: int, count: int) -> np.ndarray:
    """Return centered random pairs using glibc `rand()` semantics."""

    if count <= 0:
        return np.empty((0, 2), dtype=np.float64)

    seed_use = 1 if int(seed) == 0 else int(seed)
    state = np.zeros(344 + 2 * count, dtype=np.uint64)
    state[0] = np.uint64(seed_use)
    for i in range(1, 31):
        state[i] = np.uint64((16807 * int(state[i - 1])) % int(_GLIBC_RAND_MAX))
    for i in range(31, 34):
        state[i] = state[i - 31]
    for i in range(34, state.size):
        state[i] = (state[i - 31] + state[i - 3]) & np.uint64(0xFFFFFFFF)

    rand_vals = (state[344:] >> np.uint64(1)).astype(np.float64, copy=False)
    half = 0.5 * _GLIBC_RAND_MAX
    inv = 1.0 / _GLIBC_RAND_MAX
    pairs = np.empty((count, 2), dtype=np.float64)
    for i in range(count):
        pairs[i, 0] = (rand_vals[2 * i] - half) * inv
        pairs[i, 1] = (rand_vals[2 * i + 1] - half) * inv
    return pairs


def _dealiased_initial_mode_pairs(grid: SpectralGrid) -> list[tuple[int, int]]:
    """Return the dealiased startup-loop `(kx, ky)` pairs for multimode initial conditions.

    The binormal index 0 is skipped only when it really is the zonal mode. A
    linear run selects one ``k_y`` before seeding, so its grid holds a single
    nonzero binormal entry at index 0; skipping it by position left the whole
    state at zero and the eigensolver was handed a null seed.
    """

    nx = int(np.asarray(grid.kx).size)
    ky_values = np.asarray(grid.ky)
    ny = int(ky_values.size)
    kx_max = 1 + (nx - 1) // 3
    ky_max = 1 + (ny - 1) // 3
    ky_indices = [int(ky_i) for ky_i in range(ky_max) if float(ky_values[ky_i]) != 0.0]
    return [(int(kx_i), ky_i) for kx_i in range(kx_max) for ky_i in ky_indices]


def _periodic_zp_from_grid(z: np.ndarray) -> float:
    """Return periodic `Zp` from the discrete theta grid."""

    z_arr = np.asarray(z, dtype=float)
    if z_arr.size <= 1:
        return 1.0
    dz = float(z_arr[1] - z_arr[0])
    period = abs(dz) * float(z_arr.size)
    if period <= 0.0:
        return 1.0
    return period / (2.0 * np.pi)


def _validate_initialization(cfg: RuntimeConfig) -> tuple[str, str]:
    init_field = cfg.init.init_field.lower()
    if init_field not in _VALID_INIT_FIELDS:
        raise ValueError(
            "init_field must be one of {'density','upar','tpar','tperp','qpar','qperp','all','phi'}"
        )
    if cfg.init.gaussian_width <= 0.0:
        raise ValueError("gaussian_width must be > 0")
    init_file_mode = cfg.init.init_file_mode.strip().lower()
    if init_file_mode not in {"replace", "add"}:
        raise ValueError("init_file_mode must be one of {'replace', 'add'}")
    return init_field, init_file_mode


def _scaled_restart_state(
    cfg: RuntimeConfig,
    grid: SpectralGrid,
    *,
    Nl: int,
    Nm: int,
    nspecies: int,
) -> np.ndarray | None:
    if cfg.init.init_file is None:
        return None
    loaded_state = _load_initial_state_from_file(
        Path(cfg.init.init_file),
        nspecies=nspecies,
        Nl=Nl,
        Nm=Nm,
        ny=grid.ky.size,
        nx=grid.kx.size,
        nz=grid.z.size,
    )
    return np.asarray(loaded_state, dtype=np.complex64) * np.complex64(
        float(cfg.init.init_file_scale)
    )


def _species_targets(cfg: RuntimeConfig, nspecies: int) -> tuple[int, ...]:
    if nspecies == 1:
        return (0,)
    if not cfg.init.init_electrons_only:
        return tuple(range(nspecies))

    electron_indices = tuple(
        i for i, sp in enumerate(cfg.species[:nspecies]) if float(sp.charge) < 0.0
    )
    return electron_indices or (nspecies - 1,)


def _single_mode_values(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    cfg: RuntimeConfig,
    *,
    init_field: str,
    ky_index: int,
    kx_index: int,
) -> np.ndarray:
    z = np.asarray(grid.z)
    z_period = _periodic_zp_from_grid(z)
    z_phase = np.cos(float(cfg.init.kpar_init) * z / z_period)
    amp = float(cfg.init.init_amp)
    if cfg.init.init_single and cfg.init.gaussian_init:
        profile = _build_single_phi_gaussian_profile(
            z,
            kx=float(grid.kx[kx_index]),
            ky=float(grid.ky[ky_index]),
            s_hat=float(geom.s_hat),
            width=float(cfg.init.gaussian_width),
            envelope_constant=float(cfg.init.gaussian_envelope_constant),
            envelope_sine=float(cfg.init.gaussian_envelope_sine),
        )
        phase = 1.0 if init_field == "phi" else (1.0 + 1.0j)
        return amp * phase * profile.astype(np.complex64, copy=False)
    return amp * z_phase.astype(np.complex64, copy=False)


def _validate_named_moment_resolution(init_field: str, *, Nl: int, Nm: int) -> None:
    if init_field in {"all", "phi"}:
        return
    l_idx, m_idx = _INITIAL_FIELD_MOMENTS[init_field]
    if l_idx >= Nl or m_idx >= Nm:
        raise ValueError("init_field moment exceeds (Nl, Nm) resolution")


@dataclass
class _InitialConditionBuilder:
    grid: SpectralGrid
    geom: FluxTubeGeometryLike
    cfg: RuntimeConfig
    Nl: int
    Nm: int
    state: np.ndarray
    species_targets: tuple[int, ...]
    build_runtime_linear_params_fn: Callable[..., LinearParams]
    phi_seed_context: tuple[object, LinearParams] | None = None

    def set_mode(
        self, l_idx: int, m_idx: int, ky_i: int, kx_i: int, vals_k: np.ndarray
    ) -> None:
        if l_idx >= self.Nl or m_idx >= self.Nm:
            return
        for s_idx in self.species_targets:
            self.state[s_idx, l_idx, m_idx, ky_i, kx_i, :] = vals_k

    def set_named_mode_scaled(
        self, field_name: str, ky_i: int, kx_i: int, vals_k: np.ndarray
    ) -> None:
        l_idx, m_idx = _INITIAL_FIELD_MOMENTS[field_name]
        self.set_mode(l_idx, m_idx, ky_i, kx_i, vals_k * _ALL_FIELD_SCALES[field_name])

    def set_named_mode_raw(
        self, field_name: str, ky_i: int, kx_i: int, vals_k: np.ndarray
    ) -> None:
        l_idx, m_idx = _INITIAL_FIELD_MOMENTS[field_name]
        self.set_mode(l_idx, m_idx, ky_i, kx_i, vals_k)

    def set_phi_mode(self, ky_i: int, kx_i: int, vals_k: np.ndarray) -> None:
        if self.Nl < 1 or self.Nm < 1:
            raise ValueError(
                "init_field='phi' requires at least one Laguerre and one Hermite moment"
            )
        if self.phi_seed_context is None:
            phi_params = self.build_runtime_linear_params_fn(
                self.cfg, Nm=self.Nm, geom=self.geom
            )
            self.phi_seed_context = (
                build_linear_cache(self.grid, self.geom, phi_params, self.Nl, self.Nm),
                phi_params,
            )
        cache, phi_params = self.phi_seed_context
        seeds = _density_moments_for_target_phi(
            np.asarray(vals_k, dtype=np.complex64),
            cache=cache,
            params=phi_params,
            ky_i=int(ky_i),
            kx_i=int(kx_i),
            species_targets=self.species_targets,
        )
        for s_idx, seed_vals in seeds.items():
            self.state[s_idx, 0, 0, ky_i, kx_i, :] = seed_vals

    def seed_field(
        self, init_field: str, ky_i: int, kx_i: int, vals_k: np.ndarray
    ) -> None:
        if init_field == "all":
            for field_name in _INITIAL_FIELD_MOMENTS:
                self.set_named_mode_scaled(field_name, ky_i, kx_i, vals_k)
        elif init_field == "phi":
            self.set_phi_mode(ky_i, kx_i, vals_k)
        else:
            self.set_named_mode_raw(init_field, ky_i, kx_i, vals_k)


def _seed_gaussian_multimode(
    builder: _InitialConditionBuilder,
    *,
    init_field: str,
    amp: float,
) -> None:
    z = np.asarray(builder.grid.z)
    nx = builder.grid.kx.size
    for kx_i, ky_i in _dealiased_initial_mode_pairs(builder.grid):
        ky_k = float(builder.grid.ky[ky_i])
        if ky_k == 0.0:
            continue
        profile_k = _build_gaussian_profile(
            z,
            kx=abs(float(builder.grid.kx[kx_i])),
            ky=ky_k,
            s_hat=float(builder.geom.s_hat),
            width=float(builder.cfg.init.gaussian_width),
            envelope_constant=float(builder.cfg.init.gaussian_envelope_constant),
            envelope_sine=float(builder.cfg.init.gaussian_envelope_sine),
        )
        vals_k = amp * profile_k * (1.0 + 1.0j)
        builder.seed_field(init_field, ky_i, kx_i, vals_k)
        if kx_i != 0:
            builder.seed_field(init_field, ky_i, int(nx - kx_i), vals_k)


def _seed_random_multimode(
    builder: _InitialConditionBuilder,
    *,
    init_field: str,
    amp: float,
) -> None:
    _validate_named_moment_resolution(init_field, Nl=builder.Nl, Nm=builder.Nm)
    z = np.asarray(builder.grid.z)
    z_phase = np.cos(float(builder.cfg.init.kpar_init) * z / _periodic_zp_from_grid(z))
    nx = builder.grid.kx.size
    active_modes = _dealiased_initial_mode_pairs(builder.grid)
    rand_pairs = amp * _centered_glibc_random_pairs(
        int(builder.cfg.init.random_seed), len(active_modes)
    )
    for (kx_i, ky_i), (ra, rb) in zip(active_modes, rand_pairs, strict=True):
        vals_k = ((rb + 1j * ra) if kx_i == 0 else (ra + 1j * rb)) * z_phase
        builder.seed_field(init_field, ky_i, kx_i, vals_k)
        if kx_i != 0:
            vals_neg = (rb + 1j * ra) * z_phase
            builder.seed_field(init_field, ky_i, int(nx - kx_i), vals_neg)


def _finalize_initial_state(
    grid: SpectralGrid,
    state: np.ndarray,
    *,
    loaded_state: np.ndarray | None,
    init_file_mode: str,
) -> jnp.ndarray:
    if grid.ky.size > 1 and np.any(np.asarray(grid.ky) < 0.0):
        state = _enforce_full_ky_hermitian(state)
    if loaded_state is None:
        return jnp.asarray(state)
    if init_file_mode == "replace":
        return jnp.asarray(loaded_state)
    return jnp.asarray(cast(np.ndarray, loaded_state + state))


def _build_initial_condition_impl(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    cfg: RuntimeConfig,
    *,
    ky_index: int,
    kx_index: int,
    Nl: int,
    Nm: int,
    nspecies: int,
    build_runtime_linear_params_fn: Callable[..., LinearParams],
) -> jnp.ndarray:
    init_field, init_file_mode = _validate_initialization(cfg)
    state: np.ndarray = np.zeros(
        (nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size),
        dtype=np.complex64,
    )
    loaded_state = _scaled_restart_state(cfg, grid, Nl=Nl, Nm=Nm, nspecies=nspecies)
    amp = float(cfg.init.init_amp)
    builder = _InitialConditionBuilder(
        grid=grid,
        geom=geom,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        state=state,
        species_targets=_species_targets(cfg, nspecies),
        build_runtime_linear_params_fn=build_runtime_linear_params_fn,
    )

    if cfg.init.gaussian_init and not cfg.init.init_single:
        _seed_gaussian_multimode(builder, init_field=init_field, amp=amp)
    elif not cfg.init.init_single and not cfg.init.gaussian_init:
        _seed_random_multimode(builder, init_field=init_field, amp=amp)
    else:
        _validate_named_moment_resolution(init_field, Nl=Nl, Nm=Nm)
        vals = _single_mode_values(
            grid,
            geom,
            cfg,
            init_field=init_field,
            ky_index=ky_index,
            kx_index=kx_index,
        )
        builder.seed_field(init_field, ky_index, kx_index, vals)

    return _finalize_initial_state(
        grid,
        state,
        loaded_state=loaded_state,
        init_file_mode=init_file_mode,
    )
