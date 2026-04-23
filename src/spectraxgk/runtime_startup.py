"""Runtime startup and initialization helpers.

This module holds the geometry/loading/initial-condition logic used by the
public runtime entry points. It is intentionally kept separate from the solver
execution layer so startup behavior can be tested and refactored without
touching the time-integration control flow.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence, cast

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry import FluxTubeGeometryLike, build_flux_tube_geometry
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache, linear_terms_to_term_config
from spectraxgk.linear_krylov import KrylovConfig
from spectraxgk.miller_eik import generate_runtime_miller_eik
from spectraxgk.normalization import get_normalization_contract
from spectraxgk.restart import load_gx_restart_state
from spectraxgk.runtime_config import RuntimeConfig, RuntimeSpeciesConfig
from spectraxgk.species import Species, build_linear_params
from spectraxgk.terms.config import TermConfig
from spectraxgk.vmec_eik import generate_runtime_vmec_eik

_GX_RAND_MAX = float((1 << 31) - 1)


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
            shift_preconditioner="damping",
            shift_selection="targeted",
            mode_family="etg",
            fallback_method="arnoldi",
            fallback_real_floor=-1.0e-6,
        )

    return KrylovConfig()


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
    """Resolve runtime geometry, generating `*.eik.nc` geometry when needed."""

    model = cfg.geometry.model.strip().lower()
    if model == "vmec":
        eik_path = generate_runtime_vmec_eik(cfg)
        geom_cfg = replace(cfg.geometry, model="vmec-eik", geometry_file=str(eik_path))
        return build_flux_tube_geometry(geom_cfg)
    if model == "miller":
        eik_path = generate_runtime_miller_eik(cfg)
        geom_cfg = replace(cfg.geometry, model="gx-eik", geometry_file=str(eik_path))
        return build_flux_tube_geometry(geom_cfg)
    return build_flux_tube_geometry(cfg.geometry)


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
    """Build `LinearTerms` from unified toggles."""

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
    """Build nonlinear-ready `TermConfig` from unified toggles."""

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
        return load_gx_restart_state(
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
        arr = _reshape_gx_state(raw, nspec=nspecies, nl=Nl, nm=Nm, nyc=nyc, nx=nx, nz=nz)
        return _expand_ky(arr, nyc=nyc)
    if raw.size == expected_full:
        return raw.reshape((nspecies, Nl, Nm, ny, nx, nz))
    raise ValueError(
        f"init_file size {raw.size} does not match expected {expected_nyc} (nyc) or {expected_full} (full)"
    )


def _gx_centered_random_pairs(seed: int, count: int) -> np.ndarray:
    """Return GX-style centered random pairs using glibc `rand()` semantics."""

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
    """Return the GX startup-loop `(kx, ky)` pairs for multimode initial conditions."""

    nx = int(np.asarray(grid.kx).size)
    ny = int(np.asarray(grid.ky).size)
    kx_max = 1 + (nx - 1) // 3
    ky_max = 1 + (ny - 1) // 3
    return [(int(kx_i), int(ky_i)) for kx_i in range(kx_max) for ky_i in range(1, ky_max)]


def _gx_periodic_zp(z: np.ndarray) -> float:
    """Return GX's periodic `Zp` from the discrete theta grid."""

    z_arr = np.asarray(z, dtype=float)
    if z_arr.size <= 1:
        return 1.0
    dz = float(z_arr[1] - z_arr[0])
    period = abs(dz) * float(z_arr.size)
    if period <= 0.0:
        return 1.0
    return period / (2.0 * np.pi)


def _as_runtime_species_array(value: float | jnp.ndarray, nspecies: int, name: str) -> np.ndarray:
    """Return a length-``nspecies`` NumPy array for startup-only algebra."""

    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = arr.reshape(-1)
    if arr.size == 1:
        return np.full((int(nspecies),), float(arr[0]), dtype=float)
    if arr.size != int(nspecies):
        raise ValueError(f"{name} must have length {nspecies} (got {arr.size})")
    return arr.astype(float, copy=False)


def _density_moments_for_target_phi(
    phi_target: np.ndarray,
    *,
    cache,
    params: LinearParams,
    ky_i: int,
    kx_i: int,
    species_targets: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Invert the electrostatic field solve for a requested initial ``phi`` mode.

    The runtime evolves Hermite-Laguerre moments, not fields. For literature
    tests that prescribe an initial electrostatic-potential perturbation, seed
    only the density moment with the algebraic moment profile whose immediate
    quasineutrality solve returns ``phi_target``. The adiabatic-electron zonal
    branch includes the same flux-surface-average correction used in
    :mod:`spectraxgk.terms.fields`.
    """

    phi = np.asarray(phi_target, dtype=np.complex64)
    nspecies = int(np.asarray(cache.Jl).shape[0])
    if not species_targets:
        raise ValueError("init_field='phi' requires at least one kinetic target species")
    if min(species_targets) < 0 or max(species_targets) >= nspecies:
        raise ValueError("init_field='phi' target species index is out of range")

    mask0 = np.asarray(cache.mask0, dtype=bool)
    if np.all(mask0[int(ky_i), int(kx_i), :]):
        if np.any(np.abs(phi) > 0.0):
            raise ValueError("init_field='phi' cannot initialize the masked ky=0, kx=0 gauge mode")
        return {int(idx): np.zeros_like(phi) for idx in species_targets}

    Jl = np.asarray(cache.Jl, dtype=float)
    jacobian = np.asarray(cache.jacobian, dtype=float)
    charge = _as_runtime_species_array(params.charge_sign, nspecies, "charge_sign")
    density = _as_runtime_species_array(params.density, nspecies, "density")
    tz = _as_runtime_species_array(params.tz, nspecies, "tz")
    zt = np.where(tz == 0.0, 0.0, 1.0 / tz)
    tau_e_arr = np.asarray(params.tau_e, dtype=float).reshape(-1)
    tau_e = float(tau_e_arr[0]) if tau_e_arr.size else 0.0

    g0_species = np.sum(Jl[:, :, int(ky_i), int(kx_i), :] * Jl[:, :, int(ky_i), int(kx_i), :], axis=1)
    qneut = np.sum(density[:, None] * charge[:, None] * zt[:, None] * (1.0 - g0_species), axis=0)
    denom = tau_e + qneut
    if not np.all(np.isfinite(denom)):
        raise ValueError("init_field='phi' produced a non-finite quasineutrality denominator")

    nbar = denom.astype(np.complex64) * phi
    ky_is_zonal = np.isclose(float(np.asarray(cache.ky)[int(ky_i)]), 0.0)
    if tau_e > 0.0 and ky_is_zonal and int(kx_i) > 0:
        jac_sum = float(np.sum(jacobian))
        phi_avg = np.sum(jacobian.astype(np.complex64) * phi) / jac_sum if jac_sum != 0.0 else np.mean(phi)
        nbar = denom.astype(np.complex64) * phi - np.complex64(tau_e) * phi_avg

    target_indices = np.asarray(species_targets, dtype=int)
    coeff = density[target_indices, None] * charge[target_indices, None] * Jl[target_indices, 0, int(ky_i), int(kx_i), :]
    coeff_norm = np.sum(coeff * coeff, axis=0)
    tiny = 1.0e-30
    bad = (coeff_norm <= tiny) & (np.abs(nbar) > tiny)
    if np.any(bad):
        raise ValueError("init_field='phi' cannot be represented by the selected density moments")

    seed = np.where(
        coeff_norm[None, :] > tiny,
        (coeff / coeff_norm[None, :]).astype(np.complex64) * nbar[None, :],
        np.complex64(0.0),
    )
    return {int(s_idx): np.asarray(seed[i], dtype=np.complex64) for i, s_idx in enumerate(target_indices)}


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
    if init_field not in {"all", "phi", *field_map.keys()}:
        raise ValueError(
            "init_field must be one of {'density','upar','tpar','tperp','qpar','qperp','all','phi'}"
        )
    if cfg.init.gaussian_width <= 0.0:
        raise ValueError("gaussian_width must be > 0")
    init_file_mode = cfg.init.init_file_mode.strip().lower()
    if init_file_mode not in {"replace", "add"}:
        raise ValueError("init_file_mode must be one of {'replace', 'add'}")

    g0: np.ndarray = np.zeros((nspecies, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
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
    z_period = _gx_periodic_zp(z)
    z_phase = np.cos(float(cfg.init.kpar_init) * z / z_period)
    if cfg.init.init_single:
        vals = amp * z_phase.astype(np.complex64, copy=False)
    elif cfg.init.gaussian_init:
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
        vals = amp * z_phase.astype(np.complex64, copy=False)

    if nspecies == 1:
        species_targets: tuple[int, ...] = (0,)
    elif cfg.init.init_electrons_only:
        electron_indices = tuple(i for i, sp in enumerate(cfg.species[:nspecies]) if float(sp.charge) < 0.0)
        species_targets = electron_indices or (nspecies - 1,)
    else:
        species_targets = tuple(range(nspecies))

    def _set_mode(l_idx: int, m_idx: int, ky_i: int, kx_i: int, vals_k: np.ndarray) -> None:
        if l_idx >= Nl or m_idx >= Nm:
            return
        for s_idx in species_targets:
            g0[s_idx, l_idx, m_idx, ky_i, kx_i, :] = vals_k

    def _set_named_mode(field_name: str, ky_i: int, kx_i: int, vals_k: np.ndarray) -> None:
        l_idx, m_idx = field_map[field_name]
        _set_mode(l_idx, m_idx, ky_i, kx_i, vals_k * all_scales[field_name])

    phi_seed_context: tuple[object, LinearParams] | None = None

    def _set_phi_mode(ky_i: int, kx_i: int, vals_k: np.ndarray) -> None:
        nonlocal phi_seed_context
        if Nl < 1 or Nm < 1:
            raise ValueError("init_field='phi' requires at least one Laguerre and one Hermite moment")
        if phi_seed_context is None:
            phi_params = build_runtime_linear_params(cfg, Nm=Nm, geom=geom)
            phi_seed_context = (build_linear_cache(grid, geom, phi_params, Nl, Nm), phi_params)
        cache, phi_params = phi_seed_context
        seeds = _density_moments_for_target_phi(
            np.asarray(vals_k, dtype=np.complex64),
            cache=cache,
            params=phi_params,
            ky_i=int(ky_i),
            kx_i=int(kx_i),
            species_targets=species_targets,
        )
        for s_idx, seed_vals in seeds.items():
            g0[s_idx, 0, 0, ky_i, kx_i, :] = seed_vals

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
            elif init_field == "phi":
                _set_phi_mode(ky_i, kx_i, vals_k)
            else:
                l_idx, m_idx = field_map[init_field]
                _set_mode(l_idx, m_idx, ky_i, kx_i, vals_k)
            if kx_i == 0:
                continue
            kx_neg = int(nx - kx_i)
            if init_field == "all":
                for field_name in field_map:
                    _set_named_mode(field_name, ky_i, kx_neg, vals_k)
            elif init_field == "phi":
                _set_phi_mode(ky_i, kx_neg, vals_k)
            else:
                l_idx, m_idx = field_map[init_field]
                _set_mode(l_idx, m_idx, ky_i, kx_neg, vals_k)
    elif not cfg.init.init_single and not cfg.init.gaussian_init:
        Zp = _gx_periodic_zp(z)
        kpar = float(cfg.init.kpar_init)
        z_phase = np.cos(kpar * z / Zp)
        nx = grid.kx.size
        active_modes = _gx_init_mode_pairs(grid)
        rand_pairs = amp * _gx_centered_random_pairs(int(cfg.init.random_seed), len(active_modes))
        if init_field not in {"all", "phi"}:
            l_idx, m_idx = field_map[init_field]
            if l_idx >= Nl or m_idx >= Nm:
                raise ValueError("init_field moment exceeds (Nl, Nm) resolution")
        for (kx_i, ky_i), (ra, rb) in zip(active_modes, rand_pairs, strict=True):
            vals_k = ((rb + 1j * ra) if kx_i == 0 else (ra + 1j * rb)) * z_phase
            if init_field == "all":
                for field_name in field_map:
                    _set_named_mode(field_name, ky_i, kx_i, vals_k)
            elif init_field == "phi":
                _set_phi_mode(ky_i, kx_i, vals_k)
            else:
                for s_idx in species_targets:
                    g0[s_idx, l_idx, m_idx, ky_i, kx_i, :] = vals_k
            if kx_i != 0:
                kx_neg = nx - kx_i
                vals_neg = (rb + 1j * ra) * z_phase
                if init_field == "all":
                    for field_name in field_map:
                        _set_named_mode(field_name, ky_i, kx_neg, vals_neg)
                elif init_field == "phi":
                    _set_phi_mode(ky_i, kx_neg, vals_neg)
                else:
                    for s_idx in species_targets:
                        g0[s_idx, l_idx, m_idx, ky_i, kx_neg, :] = vals_neg
    else:
        if init_field == "all":
            for field_name in field_map:
                _set_named_mode(field_name, ky_index, kx_index, vals)
        elif init_field == "phi":
            _set_phi_mode(ky_index, kx_index, vals)
        else:
            l_idx, m_idx = field_map[init_field]
            if l_idx >= Nl or m_idx >= Nm:
                raise ValueError("init_field moment exceeds (Nl, Nm) resolution")
            for s_idx in species_targets:
                g0[s_idx, l_idx, m_idx, ky_index, kx_index, :] = vals
    if grid.ky.size > 1 and np.any(np.asarray(grid.ky) < 0.0):
        g0 = _enforce_full_ky_hermitian(g0)
    if loaded_state is not None:
        if init_file_mode == "replace":
            return jnp.asarray(loaded_state)
        g0 = cast(np.ndarray, loaded_state + g0)
    return jnp.asarray(g0)
