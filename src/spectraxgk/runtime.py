"""Unified runtime-configured linear driver (case-agnostic core path)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from spectraxgk.cetg import (
    build_cetg_model_params,
    integrate_cetg_gx_diagnostics_state,
    validate_cetg_runtime_config,
)
from spectraxgk.config import resolve_cfl_fac
from spectraxgk.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
    select_ky_index,
)
from spectraxgk.diagnostics import GXDiagnostics, gx_energy_total
from spectraxgk.geometry import (
    apply_gx_geometry_grid_defaults,
    FluxTubeGeometryLike,
    build_flux_tube_geometry,
)
from spectraxgk.grids import SpectralGrid, build_spectral_grid, select_ky_grid
from spectraxgk.linear import (
    LinearParams,
    LinearTerms,
    build_linear_cache,
    integrate_linear_diagnostics,
    linear_terms_to_term_config,
)
from spectraxgk.nonlinear import integrate_nonlinear_gx_diagnostics_state
from spectraxgk.linear_krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.normalization import apply_diagnostic_normalization, get_normalization_contract
from spectraxgk.runtime_config import RuntimeConfig, RuntimeSpeciesConfig
from spectraxgk.runners import integrate_linear_from_config, integrate_nonlinear_from_config
from spectraxgk.species import Species, build_linear_params
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.vmec_eik import generate_runtime_vmec_eik


@dataclass(frozen=True)
class RuntimeLinearResult:
    """Result container for runtime linear runs."""

    ky: float
    gamma: float
    omega: float
    selection: ModeSelection
    t: np.ndarray | None = None
    signal: np.ndarray | None = None
    state: np.ndarray | None = None


@dataclass(frozen=True)
class RuntimeLinearScanResult:
    """Result container for runtime linear ky scans."""

    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray


@dataclass(frozen=True)
class RuntimeNonlinearResult:
    """Result container for runtime nonlinear runs."""

    t: np.ndarray
    diagnostics: GXDiagnostics | None
    phi2: np.ndarray | None = None
    fields: FieldState | None = None
    state: np.ndarray | None = None
    ky_selected: float | None = None
    kx_selected: float | None = None


_GX_RAND_MAX = float((1 << 31) - 1)


def _midplane_index(grid: SpectralGrid) -> int:
    if grid.z.size <= 1:
        return 0
    return min(int(grid.z.size // 2 + 1), int(grid.z.size) - 1)


def _zero_kx_index(grid: SpectralGrid) -> int:
    kx = np.asarray(grid.kx, dtype=float)
    return int(np.argmin(np.abs(kx)))


def _gx_centered_random_pairs(seed: int, count: int) -> np.ndarray:
    """Return GX-style centered random pairs using glibc ``rand()`` semantics."""

    if count <= 0:
        return np.empty((0, 2), dtype=np.float64)

    seed_use = 1 if int(seed) == 0 else int(seed)
    state = np.zeros(344 + 2 * count, dtype=np.uint64)
    state[0] = np.uint64(seed_use)
    for i in range(1, 31):
        state[i] = np.uint64((16807 * int(state[i - 1])) % int(_GX_RAND_MAX))
    for i in range(31, 34):
        state[i] = state[i - 31]
    for i in range(34, state.size):
        state[i] = (state[i - 31] + state[i - 3]) & np.uint64(0xFFFFFFFF)

    rand_vals = (state[344:] >> np.uint64(1)).astype(np.float64, copy=False)
    half = 0.5 * _GX_RAND_MAX
    inv = 1.0 / _GX_RAND_MAX
    pairs = np.empty((count, 2), dtype=np.float64)
    for i in range(count):
        pairs[i, 0] = (rand_vals[2 * i] - half) * inv
        pairs[i, 1] = (rand_vals[2 * i + 1] - half) * inv
    return pairs


def _gx_init_mode_pairs(grid: SpectralGrid) -> list[tuple[int, int]]:
    """Return the GX startup-loop ``(kx, ky)`` pairs for multimode initial conditions."""

    nx = int(np.asarray(grid.kx).size)
    ny = int(np.asarray(grid.ky).size)
    kx_max = 1 + (nx - 1) // 3
    ky_max = 1 + (ny - 1) // 3
    return [(int(kx_i), int(ky_i)) for kx_i in range(kx_max) for ky_i in range(1, ky_max)]


def _select_nonlinear_mode_indices(
    grid: SpectralGrid,
    *,
    ky_target: float,
    kx_target: float | None,
    use_dealias_mask: bool,
) -> tuple[int, int]:
    ky = np.asarray(grid.ky, dtype=float)
    kx = np.asarray(grid.kx, dtype=float)
    kx_pick_target = 0.0 if kx_target is None else float(kx_target)
    if not use_dealias_mask:
        ky_pick = select_ky_index(ky, ky_target)
        kx_pick = int(np.argmin(np.abs(kx - kx_pick_target)))
        return ky_pick, kx_pick

    mask = np.asarray(grid.dealias_mask, dtype=bool)
    ky_candidates = np.where(np.any(mask, axis=1))[0]
    if ky_candidates.size == 0:
        ky_candidates = np.arange(ky.size, dtype=int)
    ky_pick = ky_candidates[int(np.argmin(np.abs(ky[ky_candidates] - float(ky_target))))]
    kx_candidates = np.where(mask[ky_pick])[0]
    if kx_candidates.size == 0:
        kx_candidates = np.arange(kx.size, dtype=int)
    kx_pick = kx_candidates[int(np.argmin(np.abs(kx[kx_candidates] - kx_pick_target)))]
    return int(ky_pick), int(kx_pick)


def _infer_runtime_nonlinear_steps(
    cfg: RuntimeConfig,
    *,
    dt: float,
    steps: int | None,
) -> int:
    """Infer nonlinear explicit step counts with the same dt ceiling as the integrator."""

    if steps is not None:
        steps_val = int(steps)
    elif bool(cfg.time.fixed_dt):
        steps_val = int(np.round(float(cfg.time.t_max) / max(float(cfg.time.dt), 1.0e-12)))
    else:
        # Keep runtime inference aligned with GX-style adaptive stepping: when
        # dt_max is unset, the nonlinear integrator clamps at dt itself.
        dt_cap = float(cfg.time.dt_max) if cfg.time.dt_max is not None else float(dt)
        steps_val = int(np.ceil(float(cfg.time.t_max) / max(dt_cap, 1.0e-12)))
    if steps_val < 1:
        raise ValueError("steps must be >= 1")
    return steps_val


def _slice_gx_diagnostics(diag: GXDiagnostics, stop: int) -> GXDiagnostics:
    """Return the first ``stop`` diagnostic samples."""

    if stop < 0:
        raise ValueError("stop must be >= 0")

    def _slice_optional(arr: np.ndarray | jnp.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        return np.asarray(arr)[:stop, ...]

    dt_t = np.asarray(diag.dt_t)[:stop]
    Wg_t = np.asarray(diag.Wg_t)[:stop]
    Wphi_t = np.asarray(diag.Wphi_t)[:stop]
    Wapar_t = np.asarray(diag.Wapar_t)[:stop]
    if dt_t.size == 0:
        dt_mean = np.asarray(0.0, dtype=float)
    else:
        dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return GXDiagnostics(
        t=np.asarray(diag.t)[:stop],
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=np.asarray(diag.gamma_t)[:stop],
        omega_t=np.asarray(diag.omega_t)[:stop],
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=np.asarray(diag.heat_flux_t)[:stop],
        particle_flux_t=np.asarray(diag.particle_flux_t)[:stop],
        energy_t=np.asarray(gx_energy_total(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))),
        heat_flux_species_t=_slice_optional(diag.heat_flux_species_t),
        particle_flux_species_t=_slice_optional(diag.particle_flux_species_t),
        phi_mode_t=_slice_optional(diag.phi_mode_t),
    )


def _truncate_gx_diagnostics(diag: GXDiagnostics, *, t_max: float) -> GXDiagnostics:
    """Keep samples through the first entry that reaches ``t_max``."""

    t_arr = np.asarray(diag.t, dtype=float)
    if t_arr.size == 0:
        return diag
    stop = int(np.searchsorted(t_arr, float(t_max), side="left")) + 1
    stop = min(max(stop, 1), int(t_arr.size))
    return _slice_gx_diagnostics(diag, stop)


def _stride_gx_diagnostics(diag: GXDiagnostics, *, stride: int) -> GXDiagnostics:
    """Apply the GX runtime output stride after concatenating chunk diagnostics."""

    stride_use = int(max(stride, 1))
    if stride_use == 1:
        return diag

    def _stride_optional(arr: np.ndarray | jnp.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        return np.asarray(arr)[::stride_use, ...]

    dt_t = np.asarray(diag.dt_t)[::stride_use]
    Wg_t = np.asarray(diag.Wg_t)[::stride_use]
    Wphi_t = np.asarray(diag.Wphi_t)[::stride_use]
    Wapar_t = np.asarray(diag.Wapar_t)[::stride_use]
    if dt_t.size == 0:
        dt_mean = np.asarray(0.0, dtype=float)
    else:
        dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return GXDiagnostics(
        t=np.asarray(diag.t)[::stride_use],
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=np.asarray(diag.gamma_t)[::stride_use],
        omega_t=np.asarray(diag.omega_t)[::stride_use],
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=np.asarray(diag.heat_flux_t)[::stride_use],
        particle_flux_t=np.asarray(diag.particle_flux_t)[::stride_use],
        energy_t=np.asarray(gx_energy_total(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))),
        heat_flux_species_t=_stride_optional(diag.heat_flux_species_t),
        particle_flux_species_t=_stride_optional(diag.particle_flux_species_t),
        phi_mode_t=_stride_optional(diag.phi_mode_t),
    )


def _concat_gx_diagnostics(diags: Sequence[GXDiagnostics]) -> GXDiagnostics:
    """Concatenate one or more diagnostic chunks."""

    if not diags:
        raise ValueError("at least one diagnostic chunk is required")

    def _concat(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(diag, name)) for diag in diags], axis=0)

    def _concat_optional(name: str) -> np.ndarray | None:
        values = [getattr(diag, name) for diag in diags]
        if all(value is None for value in values):
            return None
        return np.concatenate([np.asarray(value) for value in values if value is not None], axis=0)

    dt_t = _concat("dt_t")
    Wg_t = _concat("Wg_t")
    Wphi_t = _concat("Wphi_t")
    Wapar_t = _concat("Wapar_t")
    dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return GXDiagnostics(
        t=_concat("t"),
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=_concat("gamma_t"),
        omega_t=_concat("omega_t"),
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=_concat("heat_flux_t"),
        particle_flux_t=_concat("particle_flux_t"),
        energy_t=np.asarray(gx_energy_total(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))),
        heat_flux_species_t=_concat_optional("heat_flux_species_t"),
        particle_flux_species_t=_concat_optional("particle_flux_species_t"),
        phi_mode_t=_concat_optional("phi_mode_t"),
    )


def _species_to_linear(species_cfg: Sequence[RuntimeSpeciesConfig]) -> list[Species]:
    kinetic = [s for s in species_cfg if bool(s.kinetic)]
    if not kinetic:
        raise ValueError("RuntimeConfig.species must include at least one kinetic species")
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


def _gx_default_p_hyper_m(nhermite: int | None) -> float:
    """Return the GX default Hermite hypercollision exponent."""

    if nhermite is None:
        return 20.0
    return float(min(20, max(int(nhermite) // 2, 1)))


def _runtime_model_key(cfg: RuntimeConfig) -> str:
    return cfg.physics.reduced_model.strip().lower()


def _resolve_runtime_hl_dims(
    cfg: RuntimeConfig,
    *,
    Nl: int | None,
    Nm: int | None,
) -> tuple[int, int]:
    """Resolve model-native Hermite/Laguerre dimensions."""

    model = _runtime_model_key(cfg)
    if model in {"", "gyrokinetic", "full", "full-gk", "gx"}:
        return int(24 if Nl is None else Nl), int(12 if Nm is None else Nm)
    if model == "cetg":
        Nl_use = 2 if Nl is None else int(Nl)
        Nm_use = 1 if Nm is None else int(Nm)
        if Nl_use != 2 or Nm_use != 1:
            raise ValueError("GX cETG requires exactly Nl=2 and Nm=1")
        return Nl_use, Nm_use
    if model == "krehm":
        raise NotImplementedError(
            "physics.reduced_model='krehm' requires the dedicated KREHM solver; "
            "the full-GK runtime path does not emulate the GX KREHM model."
        )
    raise ValueError(f"Unknown physics.reduced_model={cfg.physics.reduced_model!r}")


def _require_full_gk_runtime_model(cfg: RuntimeConfig) -> None:
    """Reject reduced-model configs until their dedicated solvers exist."""

    model = cfg.physics.reduced_model.strip().lower()
    if model in {"", "gyrokinetic", "full", "full-gk", "gx"}:
        return
    if model == "cetg":
        raise NotImplementedError(
            "physics.reduced_model='cetg' requires the dedicated collisional-slab ETG solver; "
            "the full-GK runtime path does not emulate the GX cETG model."
        )
    if model == "krehm":
        raise NotImplementedError(
            "physics.reduced_model='krehm' requires the dedicated KREHM solver; "
            "the full-GK runtime path does not emulate the GX KREHM model."
        )
    raise ValueError(f"Unknown physics.reduced_model={cfg.physics.reduced_model!r}")


def build_runtime_geometry(cfg: RuntimeConfig) -> FluxTubeGeometryLike:
    """Resolve runtime geometry, generating VMEC ``*.eik.nc`` files when requested."""

    if cfg.geometry.model.strip().lower() != "vmec":
        return build_flux_tube_geometry(cfg.geometry)
    eik_path = generate_runtime_vmec_eik(cfg)
    geom_cfg = replace(cfg.geometry, model="vmec-eik", geometry_file=str(eik_path))
    return build_flux_tube_geometry(geom_cfg)


def build_runtime_linear_params(
    cfg: RuntimeConfig,
    *,
    Nm: int | None = None,
    geom: FluxTubeGeometryLike | None = None,
) -> LinearParams:
    """Build ``LinearParams`` from a unified runtime config."""

    _require_full_gk_runtime_model(cfg)
    if geom is None:
        geom = build_runtime_geometry(cfg)
    contract = get_normalization_contract(cfg.normalization.contract)
    rho_star = contract.rho_star if cfg.normalization.rho_star is None else float(cfg.normalization.rho_star)
    omega_d_scale = (
        contract.omega_d_scale if cfg.normalization.omega_d_scale is None else float(cfg.normalization.omega_d_scale)
    )
    omega_star_scale = (
        contract.omega_star_scale
        if cfg.normalization.omega_star_scale is None
        else float(cfg.normalization.omega_star_scale)
    )

    species = _species_to_linear(cfg.species)
    has_kinetic_electron = any(float(s.charge) < 0.0 for s in species)
    if cfg.physics.adiabatic_electrons and has_kinetic_electron:
        raise ValueError("adiabatic_electrons=True conflicts with kinetic electron species")

    tau_e = float(cfg.physics.tau_e) if cfg.physics.adiabatic_electrons else 0.0
    beta = float(cfg.physics.beta) if cfg.physics.electromagnetic else 0.0
    fapar = 1.0 if (cfg.physics.electromagnetic and cfg.physics.use_apar and beta > 0.0) else 0.0
    p_hyper_m = cfg.collisions.p_hyper_m
    if p_hyper_m is None:
        p_hyper_m = _gx_default_p_hyper_m(Nm)

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
    """Build ``LinearTerms`` from unified toggles."""

    em_on = bool(cfg.physics.electromagnetic)
    use_apar = em_on and bool(cfg.physics.use_apar)
    use_bpar = em_on and bool(cfg.physics.use_bpar)
    collisions_on = bool(cfg.physics.collisions)
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
    """Build nonlinear-ready ``TermConfig`` from unified toggles."""

    lin_terms = build_runtime_linear_terms(cfg)
    nonlinear_on = float(cfg.terms.nonlinear if cfg.physics.nonlinear else 0.0)
    return linear_terms_to_term_config(lin_terms, nonlinear=nonlinear_on)


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
    return env * np.exp(-((z - theta0) / width) ** 2)


def _reshape_gx_state(
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
    arr = arr[..., idxyz.ravel()]
    return arr.reshape((nspec, nl, nm, nyc, nx, nz))


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
    raw = np.fromfile(path, dtype=np.complex64)
    nyc = ny // 2 + 1
    expected_nyc = nspecies * Nl * Nm * nyc * nx * nz
    expected_full = nspecies * Nl * Nm * ny * nx * nz
    if raw.size == expected_nyc:
        arr = _reshape_gx_state(raw, nspec=nspecies, nl=Nl, nm=Nm, nyc=nyc, nx=nx, nz=nz)
        return _expand_ky(arr, nyc=nyc)
    if raw.size == expected_full:
        return raw.reshape((nspecies, Nl, Nm, ny, nx, nz))
    raise ValueError(
        f"init_file size {raw.size} does not match expected {expected_nyc} (nyc) or {expected_full} (full)"
    )


def _build_initial_condition(
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    cfg: RuntimeConfig,
    *,
    ky_index: int,
    kx_index: int,
    Nl: int,
    Nm: int,
    nspecies: int,
) -> jnp.ndarray:
    field_map = {
        "density": (0, 0),
        "upar": (0, 1),
        "tpar": (0, 2),
        "tperp": (1, 0),
        "qpar": (0, 3),
        "qperp": (1, 1),
    }
    all_scales = {
        "density": 1.0,
        "upar": 1.0,
        "tpar": 1.0 / np.sqrt(2.0),
        "tperp": 1.0,
        "qpar": 1.0 / np.sqrt(6.0),
        "qperp": 1.0,
    }
    init_field = cfg.init.init_field.lower()
    if init_field != "all" and init_field not in field_map:
        raise ValueError(
            "init_field must be one of {'density','upar','tpar','tperp','qpar','qperp','all'}"
        )
    if cfg.init.gaussian_width <= 0.0:
        raise ValueError("gaussian_width must be > 0")
    init_file_mode = cfg.init.init_file_mode.strip().lower()
    if init_file_mode not in {"replace", "add"}:
        raise ValueError("init_file_mode must be one of {'replace', 'add'}")

    g0 = np.zeros((nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    loaded_state: np.ndarray | None = None
    if cfg.init.init_file is not None:
        loaded_state = _load_initial_state_from_file(
            Path(cfg.init.init_file),
            nspecies=nspecies,
            Nl=Nl,
            Nm=Nm,
            ny=grid.ky.size,
            nx=grid.kx.size,
            nz=grid.z.size,
        )
        loaded_state = np.asarray(loaded_state, dtype=np.complex64) * np.complex64(float(cfg.init.init_file_scale))
    amp = float(cfg.init.init_amp)
    ky_val = float(grid.ky[ky_index])

    z = np.asarray(grid.z)
    z_min = float(z.min())
    z_max = float(z.max())
    z_period = (z_max - z_min) / (2.0 * np.pi) if z_max > z_min else 1.0
    z_phase = np.cos(float(cfg.init.kpar_init) * z / z_period)
    if cfg.init.gaussian_init:
        profile = _build_gaussian_profile(
            z,
            kx=float(grid.kx[kx_index]),
            ky=ky_val,
            s_hat=float(geom.s_hat),
            width=float(cfg.init.gaussian_width),
            envelope_constant=float(cfg.init.gaussian_envelope_constant),
            envelope_sine=float(cfg.init.gaussian_envelope_sine),
        )
        vals = amp * profile * (1.0 + 1.0j)
    else:
        # GX seeds single non-Gaussian modes as purely real amplitudes.
        vals = amp * z_phase.astype(np.complex64, copy=False)

    species_targets: tuple[int, ...]
    if nspecies == 1:
        species_targets = (0,)
    elif cfg.init.init_electrons_only:
        electron_indices = tuple(
            i for i, sp in enumerate(cfg.species[:nspecies]) if float(sp.charge) < 0.0
        )
        species_targets = electron_indices or (nspecies - 1,)
    else:
        species_targets = tuple(range(nspecies))

    def _set_mode(
        l_idx: int,
        m_idx: int,
        ky_i: int,
        kx_i: int,
        vals_k: np.ndarray,
    ) -> None:
        if l_idx >= Nl or m_idx >= Nm:
            return
        for s_idx in species_targets:
            g0[s_idx, l_idx, m_idx, ky_i, kx_i, :] = vals_k

    def _set_named_mode(
        field_name: str,
        ky_i: int,
        kx_i: int,
        vals_k: np.ndarray,
    ) -> None:
        l_idx, m_idx = field_map[field_name]
        _set_mode(l_idx, m_idx, ky_i, kx_i, vals_k * all_scales[field_name])

    if cfg.init.gaussian_init and not cfg.init.init_single:
        nx = grid.kx.size
        for kx_i, ky_i in _gx_init_mode_pairs(grid):
            ky_k = float(grid.ky[ky_i])
            if ky_k == 0.0:
                continue
            kx_k = float(grid.kx[kx_i])
            profile_k = _build_gaussian_profile(
                z,
                kx=abs(kx_k),
                ky=ky_k,
                s_hat=float(geom.s_hat),
                width=float(cfg.init.gaussian_width),
                envelope_constant=float(cfg.init.gaussian_envelope_constant),
                envelope_sine=float(cfg.init.gaussian_envelope_sine),
            )
            vals_k = amp * profile_k * (1.0 + 1.0j)
            if init_field == "all":
                for field_name in field_map:
                    _set_named_mode(field_name, ky_i, kx_i, vals_k)
            else:
                l_idx, m_idx = field_map[init_field]
                _set_mode(l_idx, m_idx, ky_i, kx_i, vals_k)

            if kx_i == 0:
                continue
            kx_neg = int(nx - kx_i)
            if init_field == "all":
                for field_name in field_map:
                    _set_named_mode(field_name, ky_i, kx_neg, vals_k)
            else:
                l_idx, m_idx = field_map[init_field]
                _set_mode(l_idx, m_idx, ky_i, kx_neg, vals_k)
    elif not cfg.init.init_single and not cfg.init.gaussian_init:
        z_min = float(z.min())
        z_max = float(z.max())
        Zp = (z_max - z_min) / (2.0 * np.pi) if z_max > z_min else 1.0
        kpar = float(cfg.init.kpar_init)
        z_phase = np.cos(kpar * z / Zp)
        nx = grid.kx.size
        active_modes = _gx_init_mode_pairs(grid)
        rand_pairs = amp * _gx_centered_random_pairs(int(cfg.init.random_seed), len(active_modes))
        if init_field != "all":
            l_idx, m_idx = field_map[init_field]
            if l_idx >= Nl or m_idx >= Nm:
                raise ValueError("init_field moment exceeds (Nl, Nm) resolution")
        for (kx_i, ky_i), (ra, rb) in zip(active_modes, rand_pairs, strict=True):
            vals_k = ((rb + 1j * ra) if kx_i == 0 else (ra + 1j * rb)) * z_phase
            if init_field == "all":
                for field_name in field_map:
                    _set_named_mode(field_name, ky_i, kx_i, vals_k)
            else:
                for s_idx in species_targets:
                    g0[s_idx, l_idx, m_idx, ky_i, kx_i, :] = vals_k
            if kx_i != 0:
                kx_neg = nx - kx_i
                vals_neg = (rb + 1j * ra) * z_phase
                if init_field == "all":
                    for field_name in field_map:
                        _set_named_mode(field_name, ky_i, kx_neg, vals_neg)
                else:
                    for s_idx in species_targets:
                        g0[s_idx, l_idx, m_idx, ky_i, kx_neg, :] = vals_neg
    else:
        if ky_val == 0.0:
            if loaded_state is None:
                return jnp.asarray(g0)
            if init_file_mode == "replace":
                return jnp.asarray(loaded_state)
            return jnp.asarray(loaded_state + g0)
        if init_field == "all":
            for field_name in field_map:
                _set_named_mode(field_name, ky_index, kx_index, vals)
        else:
            l_idx, m_idx = field_map[init_field]
            if l_idx >= Nl or m_idx >= Nm:
                raise ValueError("init_field moment exceeds (Nl, Nm) resolution")
            for s_idx in species_targets:
                g0[s_idx, l_idx, m_idx, ky_index, kx_index, :] = vals
    if loaded_state is not None:
        if init_file_mode == "replace":
            return jnp.asarray(loaded_state)
        g0 = loaded_state + g0
    return jnp.asarray(g0)


def run_runtime_linear(
    cfg: RuntimeConfig,
    *,
    ky_target: float = 0.3,
    Nl: int | None = None,
    Nm: int | None = None,
    solver: str = "auto",
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    auto_window: bool = True,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 0.2,
    require_positive: bool = True,
    min_amp_fraction: float = 0.0,
    krylov_cfg: KrylovConfig | None = None,
    mode_method: str = "project",
    fit_signal: str = "auto",
    return_state: bool = False,
) -> RuntimeLinearResult:
    """Run one linear point from a case-agnostic runtime config."""

    Nl_use, Nm_use = _resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    if _runtime_model_key(cfg) == "cetg":
        geom = build_runtime_geometry(cfg)
        validate_cetg_runtime_config(cfg, geom, Nl=Nl_use, Nm=Nm_use)
        grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg.grid)
        grid_full = build_spectral_grid(grid_cfg)
        ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
        grid = select_ky_grid(grid_full, ky_index)
        sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
        g0 = _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=sel.ky_index,
            kx_index=sel.kx_index,
            Nl=Nl_use,
            Nm=Nm_use,
            nspecies=1,
        )
        cetg_terms = build_runtime_term_config(cfg)
        cetg_params = build_cetg_model_params(cfg, geom, Nl=Nl_use, Nm=Nm_use)
        solver_key = solver.strip().lower()
        if solver_key == "krylov":
            raise NotImplementedError("solver='krylov' is not implemented for physics.reduced_model='cetg'")
        if solver_key not in {"auto", "time", "gx_time"}:
            raise ValueError("solver must be one of {'auto', 'time', 'gx_time', 'krylov'}")
        dt_val = float(cfg.time.dt if dt is None else dt)
        if dt_val <= 0.0:
            raise ValueError("dt must be > 0")
        steps_val = int(steps) if steps is not None else int(round(float(cfg.time.t_max) / dt_val))
        if steps_val < 1:
            raise ValueError("steps must be >= 1")
        sample_stride_use = int(cfg.time.sample_stride if sample_stride is None else sample_stride)
        _t, diag, G_final, _fields = integrate_cetg_gx_diagnostics_state(
            g0,
            grid,
            cetg_params,
            cetg_terms,
            dt=dt_val,
            steps=steps_val,
            method=str(method or cfg.time.method),
            sample_stride=sample_stride_use,
            diagnostics_stride=1,
            gx_real_fft=bool(cfg.time.gx_real_fft),
            omega_ky_index=0,
            omega_kx_index=0,
            fixed_dt=bool(cfg.time.fixed_dt),
            dt_min=float(cfg.time.dt_min),
            dt_max=cfg.time.dt_max,
            cfl=float(cfg.time.cfl),
            cfl_fac=cfg.time.cfl_fac,
        )
        signal = np.asarray(diag.phi_mode_t if diag.phi_mode_t is not None else np.zeros_like(np.asarray(diag.t)))
        t_arr = np.asarray(diag.t, dtype=float)
        if t_arr.size < 2:
            gamma = float(np.asarray(diag.gamma_t)[-1]) if np.asarray(diag.gamma_t).size else 0.0
            omega = float(np.asarray(diag.omega_t)[-1]) if np.asarray(diag.omega_t).size else 0.0
        elif auto_window:
            gamma, omega, _fit_tmin, _fit_tmax = fit_growth_rate_auto(
                t_arr,
                signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
        else:
            gamma, omega = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
        return RuntimeLinearResult(
            ky=float(grid.ky[0]),
            gamma=float(gamma),
            omega=float(omega),
            selection=sel,
            t=t_arr,
            signal=np.asarray(signal),
            state=np.asarray(G_final) if return_state else None,
        )

    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg.grid)
    grid_full = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg, Nm=Nm_use, geom=geom)
    terms = build_runtime_linear_terms(cfg)

    ky_index = select_ky_index(np.asarray(grid_full.ky), ky_target)
    grid = select_ky_grid(grid_full, ky_index)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=_midplane_index(grid))
    g0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=sel.ky_index,
        kx_index=sel.kx_index,
        Nl=Nl_use,
        Nm=Nm_use,
        nspecies=max(len([s for s in cfg.species if s.kinetic]), 1),
    )

    solver_key = solver.strip().lower()
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")

    def _is_valid_growth(gamma_val: float, omega_val: float) -> bool:
        if not np.isfinite(gamma_val) or not np.isfinite(omega_val):
            return False
        if require_positive and gamma_val <= 0.0:
            return False
        return True

    def _run_krylov() -> tuple[float, float]:
        kcfg = krylov_cfg or KrylovConfig()
        cache = build_linear_cache(grid, geom, params, Nl_use, Nm_use)
        eig, _vec = dominant_eigenpair(
            g0,
            cache,
            params,
            terms=terms,
            krylov_dim=kcfg.krylov_dim,
            restarts=kcfg.restarts,
            omega_min_factor=kcfg.omega_min_factor,
            omega_target_factor=kcfg.omega_target_factor,
            omega_cap_factor=kcfg.omega_cap_factor,
            omega_sign=kcfg.omega_sign,
            method=kcfg.method,
            power_iters=kcfg.power_iters,
            power_dt=kcfg.power_dt,
            shift=kcfg.shift,
            shift_source=kcfg.shift_source,
            shift_tol=kcfg.shift_tol,
            shift_maxiter=kcfg.shift_maxiter,
            shift_restart=kcfg.shift_restart,
            shift_solve_method=kcfg.shift_solve_method,
            shift_preconditioner=kcfg.shift_preconditioner,
            shift_selection=kcfg.shift_selection,
            mode_family=kcfg.mode_family,
            fallback_method=kcfg.fallback_method,
            fallback_real_floor=kcfg.fallback_real_floor,
        )
        gamma = float(jnp.real(eig))
        omega = float(-jnp.imag(eig))
        gamma, omega = apply_diagnostic_normalization(
            gamma,
            omega,
            rho_star=float(np.asarray(params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
        return gamma, omega

    def _run_time() -> RuntimeLinearResult:
        tcfg = cfg.time
        if method is not None:
            tcfg = replace(tcfg, method=str(method))
        if dt is not None:
            tcfg = replace(tcfg, dt=float(dt))
        if steps is not None:
            tcfg = replace(tcfg, t_max=float(steps) * float(tcfg.dt))
        if sample_stride is not None:
            tcfg = replace(tcfg, sample_stride=int(sample_stride))
        if return_state and solver_key == "gx_time":
            raise ValueError("return_state is not supported with solver='gx_time'")
        if return_state:
            tcfg = replace(tcfg, save_state=True)

        need_density = fit_key in {"density", "auto"}
        g_last = None
        if tcfg.use_diffrax:
            save_field = "phi+density" if need_density else "phi"
            save_mode = None if need_density else sel
            g_last, saved = integrate_linear_from_config(
                g0,
                grid,
                geom,
                params,
                tcfg,
                terms=terms,
                save_mode=save_mode,
                mode_method=mode_method,
                save_field=save_field,
                density_species_index=0 if need_density else None,
            )
            if need_density:
                phi_t, density_t = saved
            else:
                phi_t, density_t = saved, None
        else:
            if need_density:
                _diag = integrate_linear_diagnostics(
                    g0,
                    grid,
                    geom,
                    params,
                    dt=tcfg.dt,
                    steps=int(round(tcfg.t_max / tcfg.dt)),
                    method=tcfg.method,
                    terms=terms,
                    sample_stride=tcfg.sample_stride,
                    species_index=0,
                    record_hl_energy=False,
                )
                g_last = _diag[0]
                phi_t = _diag[1]
                density_t = _diag[2] if len(_diag) > 2 else None
            else:
                g_last, phi_t = integrate_linear_from_config(
                    g0,
                    grid,
                    geom,
                    params,
                    tcfg,
                    terms=terms,
                    save_mode=sel,
                    mode_method=mode_method,
                    save_field="phi",
                )
                density_t = None

        phi_t_np = np.asarray(phi_t)
        t_arr = float(tcfg.dt) * float(tcfg.sample_stride) * (
            np.arange(phi_t_np.shape[0], dtype=float) + 1.0
        )
        density_np = None if density_t is None else np.asarray(density_t)

        if fit_key == "auto":
            phi_signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
            gamma_phi, omega_phi, _, _, r2_phi, r2p_phi = fit_growth_rate_auto_with_stats(
                t_arr,
                phi_signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
            best_gamma, best_omega = gamma_phi, omega_phi
            best_score = r2_phi + 0.2 * r2p_phi + growth_weight * gamma_phi
            if density_np is not None:
                dens_signal = extract_mode_time_series(density_np, sel, method=mode_method)
                gamma_den, omega_den, _, _, r2_den, r2p_den = fit_growth_rate_auto_with_stats(
                    t_arr,
                    dens_signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
                score_den = r2_den + 0.2 * r2p_den + growth_weight * gamma_den
                if score_den > best_score:
                    best_gamma, best_omega = gamma_den, omega_den
            gamma, omega = best_gamma, best_omega
        else:
            signal = extract_mode_time_series(
                density_np if fit_key == "density" and density_np is not None else phi_t_np,
                sel,
                method=mode_method,
            )
            if auto_window:
                gamma, omega, _tmin, _tmax = fit_growth_rate_auto(
                    t_arr,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
            else:
                gamma, omega = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)
        gamma, omega = apply_diagnostic_normalization(
            gamma,
            omega,
            rho_star=float(np.asarray(params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
        return RuntimeLinearResult(
            ky=float(grid.ky[sel.ky_index]),
            gamma=float(gamma),
            omega=float(omega),
            selection=sel,
            t=t_arr,
            signal=None,
            state=None if g_last is None or not return_state else np.asarray(g_last),
        )

    if solver_key == "krylov":
        gamma, omega = _run_krylov()
        return RuntimeLinearResult(
            ky=float(grid.ky[sel.ky_index]), gamma=gamma, omega=omega, selection=sel
        )
    if solver_key == "auto":
        result = _run_time()
        if not _is_valid_growth(result.gamma, result.omega):
            gamma, omega = _run_krylov()
            return RuntimeLinearResult(
                ky=float(grid.ky[sel.ky_index]), gamma=gamma, omega=omega, selection=sel
            )
        return result

    return _run_time()


def run_runtime_scan(
    cfg: RuntimeConfig,
    ky_values: Sequence[float],
    *,
    Nl: int | None = None,
    Nm: int | None = None,
    solver: str = "auto",
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    batch_ky: bool = False,
    auto_window: bool = True,
    tmin: float | None = None,
    tmax: float | None = None,
    window_fraction: float = 0.4,
    min_points: int = 40,
    start_fraction: float = 0.2,
    growth_weight: float = 0.2,
    require_positive: bool = True,
    min_amp_fraction: float = 0.0,
    krylov_cfg: KrylovConfig | None = None,
    mode_method: str = "project",
    fit_signal: str = "auto",
) -> RuntimeLinearScanResult:
    """Run a ky scan using the unified runtime config path.

    When ``batch_ky`` is enabled, all ky points are integrated together using
    the time integrator (Krylov is not supported in this mode).
    """

    ky_arr = np.asarray(ky_values, dtype=float)
    Nl_use, Nm_use = _resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    solver_key = solver.strip().lower()
    if batch_ky and solver_key == "krylov":
        raise ValueError("batch_ky is only supported for time integration")
    if batch_ky:
        return _run_runtime_scan_batch(
            cfg,
            ky_arr,
            Nl=Nl_use,
            Nm=Nm_use,
            method=method,
            dt=dt,
            steps=steps,
            sample_stride=sample_stride,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            mode_method=mode_method,
            fit_signal=fit_signal,
        )
    gamma = np.zeros_like(ky_arr)
    omega = np.zeros_like(ky_arr)
    for i, ky in enumerate(ky_arr):
        res = run_runtime_linear(
            cfg,
            ky_target=float(ky),
            Nl=Nl_use,
            Nm=Nm_use,
            solver=solver,
            method=method,
            dt=dt,
            steps=steps,
            sample_stride=sample_stride,
            auto_window=auto_window,
            tmin=tmin,
            tmax=tmax,
            window_fraction=window_fraction,
            min_points=min_points,
            start_fraction=start_fraction,
            growth_weight=growth_weight,
            require_positive=require_positive,
            min_amp_fraction=min_amp_fraction,
            krylov_cfg=krylov_cfg,
            mode_method=mode_method,
            fit_signal=fit_signal,
        )
        gamma[i] = float(res.gamma)
        omega[i] = float(res.omega)
    return RuntimeLinearScanResult(ky=ky_arr, gamma=gamma, omega=omega)


def _run_runtime_scan_batch(
    cfg: RuntimeConfig,
    ky_arr: np.ndarray,
    *,
    Nl: int,
    Nm: int,
    method: str | None,
    dt: float | None,
    steps: int | None,
    sample_stride: int | None,
    auto_window: bool,
    tmin: float | None,
    tmax: float | None,
    window_fraction: float,
    min_points: int,
    start_fraction: float,
    growth_weight: float,
    require_positive: bool,
    min_amp_fraction: float,
    mode_method: str,
    fit_signal: str,
) -> RuntimeLinearScanResult:
    """Batch a ky scan using one time integration over the full grid."""

    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
    terms = build_runtime_linear_terms(cfg)

    ky_indices = np.asarray([select_ky_index(np.asarray(grid.ky), ky) for ky in ky_arr], dtype=int)
    nspecies = max(len([s for s in cfg.species if s.kinetic]), 1)

    g0 = None
    for ky_idx in ky_indices:
        g0_local = _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=int(ky_idx),
            kx_index=0,
            Nl=Nl,
            Nm=Nm,
            nspecies=nspecies,
        )
        g0 = g0_local if g0 is None else g0 + g0_local
    if g0 is None:
        raise ValueError("No ky values provided for batch scan")

    tcfg = cfg.time
    if method is not None:
        tcfg = replace(tcfg, method=str(method))
    if dt is not None:
        tcfg = replace(tcfg, dt=float(dt))
    if steps is not None:
        tcfg = replace(tcfg, t_max=float(steps) * float(tcfg.dt))
    if sample_stride is not None:
        tcfg = replace(tcfg, sample_stride=int(sample_stride))

    steps_val = int(round(tcfg.t_max / tcfg.dt))
    diag = integrate_linear_diagnostics(
        g0,
        grid,
        geom,
        params,
        dt=tcfg.dt,
        steps=steps_val,
        method=tcfg.method,
        terms=terms,
        sample_stride=tcfg.sample_stride,
        species_index=0,
        record_hl_energy=False,
    )
    phi_t = diag[1]
    density_t = diag[2]
    phi_t_np = np.asarray(phi_t)
    dens_t_np = np.asarray(density_t)
    t_arr = float(tcfg.dt) * float(tcfg.sample_stride) * (
        np.arange(phi_t_np.shape[0], dtype=float) + 1.0
    )

    gamma = np.zeros_like(ky_arr, dtype=float)
    omega = np.zeros_like(ky_arr, dtype=float)
    fit_key = fit_signal.strip().lower()
    if fit_key not in {"phi", "density", "auto"}:
        raise ValueError("fit_signal must be 'phi', 'density', or 'auto'")

    for i, ky_idx in enumerate(ky_indices):
        sel = ModeSelection(ky_index=int(ky_idx), kx_index=0, z_index=_midplane_index(grid))
        if fit_key == "auto":
            phi_signal = extract_mode_time_series(phi_t_np, sel, method=mode_method)
            gamma_phi, omega_phi, _, _, r2_phi, r2p_phi = fit_growth_rate_auto_with_stats(
                t_arr,
                phi_signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
            dens_signal = extract_mode_time_series(dens_t_np, sel, method=mode_method)
            gamma_den, omega_den, _, _, r2_den, r2p_den = fit_growth_rate_auto_with_stats(
                t_arr,
                dens_signal,
                window_fraction=window_fraction,
                min_points=min_points,
                start_fraction=start_fraction,
                growth_weight=growth_weight,
                require_positive=require_positive,
                min_amp_fraction=min_amp_fraction,
            )
            score_phi = r2_phi + 0.2 * r2p_phi + growth_weight * gamma_phi
            score_den = r2_den + 0.2 * r2p_den + growth_weight * gamma_den
            g_val, o_val = (gamma_phi, omega_phi) if score_phi >= score_den else (gamma_den, omega_den)
        else:
            signal = extract_mode_time_series(
                dens_t_np if fit_key == "density" else phi_t_np, sel, method=mode_method
            )
            if auto_window:
                g_val, o_val, _tmin, _tmax = fit_growth_rate_auto(
                    t_arr,
                    signal,
                    window_fraction=window_fraction,
                    min_points=min_points,
                    start_fraction=start_fraction,
                    growth_weight=growth_weight,
                    require_positive=require_positive,
                    min_amp_fraction=min_amp_fraction,
                )
            else:
                g_val, o_val = fit_growth_rate(t_arr, signal, tmin=tmin, tmax=tmax)

        g_val, o_val = apply_diagnostic_normalization(
            g_val,
            o_val,
            rho_star=float(np.asarray(params.rho_star)),
            diagnostic_norm=cfg.normalization.diagnostic_norm,
        )
        gamma[i] = float(g_val)
        omega[i] = float(o_val)

    return RuntimeLinearScanResult(ky=ky_arr, gamma=gamma, omega=omega)


def run_runtime_nonlinear(
    cfg: RuntimeConfig,
    *,
    ky_target: float = 0.3,
    kx_target: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    dt: float | None = None,
    steps: int | None = None,
    method: str | None = None,
    sample_stride: int | None = None,
    diagnostics_stride: int | None = None,
    laguerre_mode: str | None = None,
    diagnostics: bool | None = None,
    return_state: bool = False,
) -> RuntimeNonlinearResult:
    """Run a nonlinear point using the unified runtime config path."""

    Nl_use, Nm_use = _resolve_runtime_hl_dims(cfg, Nl=Nl, Nm=Nm)
    if _runtime_model_key(cfg) == "cetg":
        geom = build_runtime_geometry(cfg)
        validate_cetg_runtime_config(cfg, geom, Nl=Nl_use, Nm=Nm_use)
        grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg.grid)
        grid = build_spectral_grid(grid_cfg)
        ky_index, kx_index = _select_nonlinear_mode_indices(
            grid,
            ky_target=ky_target,
            kx_target=kx_target,
            use_dealias_mask=bool(cfg.time.nonlinear_dealias),
        )
        G0 = _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=kx_index,
            Nl=Nl_use,
            Nm=Nm_use,
            nspecies=1,
        )
        dt_val = float(cfg.time.dt if dt is None else dt)
        if dt_val <= 0.0:
            raise ValueError("dt must be > 0")
        if steps is None:
            steps_val = int(round(float(cfg.time.t_max) / dt_val))
        else:
            steps_val = int(steps)
        if steps_val < 1:
            raise ValueError("steps must be >= 1")
        cetg_params = build_cetg_model_params(cfg, geom, Nl=Nl_use, Nm=Nm_use)
        cetg_term_cfg = build_runtime_term_config(cfg)
        sample_stride_use = cfg.time.sample_stride if sample_stride is None else int(sample_stride)
        diag_stride = cfg.time.diagnostics_stride if diagnostics_stride is None else int(diagnostics_stride)
        _t, diag, G_final, cetg_fields_final = integrate_cetg_gx_diagnostics_state(
            G0,
            grid,
            cetg_params,
            cetg_term_cfg,
            dt=dt_val,
            steps=steps_val,
            method=str(method or cfg.time.method),
            sample_stride=int(sample_stride_use),
            diagnostics_stride=int(diag_stride),
            gx_real_fft=bool(cfg.time.gx_real_fft),
            omega_ky_index=int(ky_index),
            omega_kx_index=int(kx_index),
            fixed_dt=bool(cfg.time.fixed_dt),
            dt_min=float(cfg.time.dt_min),
            dt_max=cfg.time.dt_max,
            cfl=float(cfg.time.cfl),
            cfl_fac=cfg.time.cfl_fac,
        )
        if diagnostics is False:
            phi2 = np.asarray(jnp.mean(jnp.abs(cetg_fields_final.phi) ** 2))
            return RuntimeNonlinearResult(
                t=np.asarray([]),
                diagnostics=None,
                phi2=phi2,
                fields=cetg_fields_final,
                state=np.asarray(G_final) if return_state else None,
                ky_selected=float(np.asarray(grid.ky[ky_index])),
                kx_selected=float(np.asarray(grid.kx[kx_index])),
            )
        return RuntimeNonlinearResult(
            t=np.asarray(diag.t),
            diagnostics=diag,
            phi2=None,
            fields=None,
            state=np.asarray(G_final) if return_state else None,
            ky_selected=float(np.asarray(grid.ky[ky_index])),
            kx_selected=float(np.asarray(grid.kx[kx_index])),
        )

    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    params = build_runtime_linear_params(cfg, Nm=Nm_use, geom=geom)
    term_cfg = build_runtime_term_config(cfg)

    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=ky_target,
        kx_target=kx_target,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )
    G0 = _build_initial_condition(
        grid,
        geom,
        cfg,
        ky_index=ky_index,
        kx_index=kx_index,
        Nl=Nl_use,
        Nm=Nm_use,
        nspecies=len(_species_to_linear(cfg.species)),
    )

    dt_val = float(cfg.time.dt if dt is None else dt)
    if dt_val <= 0.0:
        raise ValueError("dt must be > 0")
    adaptive_chunked = steps is None and not bool(cfg.time.fixed_dt)
    steps_val = _infer_runtime_nonlinear_steps(cfg, dt=dt_val, steps=steps)

    fixed_mode_on = bool(cfg.expert.fixed_mode)
    fixed_ky_index = cfg.expert.iky_fixed
    fixed_kx_index = cfg.expert.ikx_fixed
    fixed_ky_index_use: int | None = None
    fixed_kx_index_use: int | None = None
    if fixed_mode_on:
        if fixed_ky_index is None or fixed_kx_index is None:
            raise ValueError("expert.iky_fixed and expert.ikx_fixed must be set when expert.fixed_mode=true")
        fixed_ky_index_use = int(fixed_ky_index)
        fixed_kx_index_use = int(fixed_kx_index)

    diagnostics_on = cfg.time.diagnostics if diagnostics is None else bool(diagnostics)
    if diagnostics_on or fixed_mode_on or return_state or adaptive_chunked:
        sample_stride_use = cfg.time.sample_stride if sample_stride is None else int(sample_stride)
        diag_stride = cfg.time.diagnostics_stride if diagnostics_stride is None else int(diagnostics_stride)
        laguerre_mode_use = cfg.time.laguerre_nonlinear_mode if laguerre_mode is None else str(laguerre_mode)
        if adaptive_chunked:
            chunk_steps = min(steps_val, 1024)
            G_chunk = G0
            t_elapsed = 0.0
            diag_chunks: list[GXDiagnostics] = []
            fields_final: FieldState | None = None
            for _chunk in range(100000):
                _t_chunk, diag_chunk, G_chunk, fields_final = integrate_nonlinear_gx_diagnostics_state(
                    G_chunk,
                    grid,
                    geom,
                    params,
                    dt=dt_val,
                    steps=chunk_steps,
                    method=str(method or cfg.time.method),
                    terms=term_cfg,
                    sample_stride=1,
                    diagnostics_stride=1,
                    use_dealias_mask=bool(cfg.time.nonlinear_dealias),
                    laguerre_mode=laguerre_mode_use,
                    omega_ky_index=int(ky_index),
                    omega_kx_index=int(kx_index),
                    flux_scale=float(cfg.normalization.flux_scale),
                    wphi_scale=float(cfg.normalization.wphi_scale),
                    fixed_dt=False,
                    dt_min=float(cfg.time.dt_min),
                    dt_max=cfg.time.dt_max,
                    cfl=float(cfg.time.cfl),
                    cfl_fac=resolve_cfl_fac(str(method or cfg.time.method), cfg.time.cfl_fac),
                    collision_split=bool(cfg.time.collision_split),
                    collision_scheme=str(cfg.time.collision_scheme),
                    implicit_restart=int(cfg.time.implicit_restart),
                    implicit_solve_method=str(cfg.time.implicit_solve_method),
                    implicit_preconditioner=cfg.time.implicit_preconditioner,
                    fixed_mode_ky_index=fixed_ky_index_use,
                    fixed_mode_kx_index=fixed_kx_index_use,
                )
                diag_chunk = replace(diag_chunk, t=np.asarray(diag_chunk.t) + t_elapsed)
                diag_chunks.append(diag_chunk)
                t_next = float(np.asarray(diag_chunk.t)[-1])
                if t_next <= t_elapsed + 1.0e-12:
                    raise RuntimeError("adaptive nonlinear runtime made no time-step progress")
                t_elapsed = t_next
                if t_elapsed >= float(cfg.time.t_max):
                    break
            else:
                raise RuntimeError("adaptive nonlinear runtime exceeded chunk limit before reaching t_max")

            diag = _concat_gx_diagnostics(diag_chunks)
            diag = _truncate_gx_diagnostics(diag, t_max=float(cfg.time.t_max))
            diag = _stride_gx_diagnostics(diag, stride=max(int(sample_stride_use), int(diag_stride), 1))
            t = jnp.asarray(diag.t)
            G_final = G_chunk
        else:
            t, diag, G_final, fields_final = integrate_nonlinear_gx_diagnostics_state(
                G0,
                grid,
                geom,
                params,
                dt=dt_val,
                steps=steps_val,
                method=str(method or cfg.time.method),
                terms=term_cfg,
                sample_stride=int(sample_stride_use),
                diagnostics_stride=int(diag_stride),
                use_dealias_mask=bool(cfg.time.nonlinear_dealias),
                laguerre_mode=laguerre_mode_use,
                omega_ky_index=int(ky_index),
                omega_kx_index=int(kx_index),
                flux_scale=float(cfg.normalization.flux_scale),
                wphi_scale=float(cfg.normalization.wphi_scale),
                fixed_dt=bool(cfg.time.fixed_dt),
                dt_min=float(cfg.time.dt_min),
                dt_max=cfg.time.dt_max,
                cfl=float(cfg.time.cfl),
                cfl_fac=resolve_cfl_fac(str(method or cfg.time.method), cfg.time.cfl_fac),
                collision_split=bool(cfg.time.collision_split),
                collision_scheme=str(cfg.time.collision_scheme),
                implicit_restart=int(cfg.time.implicit_restart),
                implicit_solve_method=str(cfg.time.implicit_solve_method),
                implicit_preconditioner=cfg.time.implicit_preconditioner,
                fixed_mode_ky_index=fixed_ky_index_use,
                fixed_mode_kx_index=fixed_kx_index_use,
            )
        if diagnostics_on:
            state_out = np.asarray(G_final) if return_state else None
            return RuntimeNonlinearResult(
                t=np.asarray(t),
                diagnostics=diag,
                phi2=None,
                fields=None,
                state=state_out,
                ky_selected=float(np.asarray(grid.ky[ky_index])),
                kx_selected=float(np.asarray(grid.kx[kx_index])),
            )
        if fields_final is None:
            raise RuntimeError("adaptive nonlinear runtime did not produce final fields")
        phi2 = np.asarray(jnp.mean(jnp.abs(fields_final.phi) ** 2))
        return RuntimeNonlinearResult(
            t=np.asarray([]),
            diagnostics=None,
            phi2=phi2,
            fields=fields_final,
            state=np.asarray(G_final) if return_state else None,
            ky_selected=float(np.asarray(grid.ky[ky_index])),
            kx_selected=float(np.asarray(grid.kx[kx_index])),
        )

    # Diagnostics disabled: use the config-driven integrator for final state.
    t_cfg = replace(cfg.time, dt=dt_val, t_max=dt_val * steps_val)
    G_final, fields = integrate_nonlinear_from_config(
        G0,
        grid,
        geom,
        params,
        t_cfg,
        terms=term_cfg,
    )
    phi2 = np.asarray(jnp.mean(jnp.abs(fields.phi) ** 2))
    return RuntimeNonlinearResult(
        t=np.asarray([]),
        diagnostics=None,
        phi2=phi2,
        fields=fields,
        ky_selected=float(np.asarray(grid.ky[ky_index])),
        kx_selected=float(np.asarray(grid.kx[kx_index])),
    )
