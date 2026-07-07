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

from spectraxgk.geometry import FluxTubeGeometryLike, build_flux_tube_geometry
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.solvers.linear.krylov import KrylovConfig
from spectraxgk.geometry.miller_eik import generate_runtime_miller_eik
from spectraxgk.diagnostics.normalization import get_normalization_contract
from spectraxgk.artifacts.restart import load_netcdf_restart_state
from spectraxgk.workflows.runtime.config import RuntimeConfig, RuntimeSpeciesConfig
from spectraxgk.core.species import Species, build_linear_params
from spectraxgk.terms.config import TermConfig
from spectraxgk.geometry.vmec_eik import generate_runtime_vmec_eik
from spectraxgk.workflows.runtime.initial_conditions import (
    _build_gaussian_profile,
    _build_initial_condition_impl,
    _build_single_phi_gaussian_profile,
    _centered_glibc_random_pairs,
    _dealiased_initial_mode_pairs,
    _enforce_full_ky_hermitian,
    _expand_ky,
    _load_initial_state_from_file,
    _periodic_zp_from_grid,
    _reshape_netcdf_state,
)
from spectraxgk.workflows.runtime.initial_phi import (
    _as_runtime_species_array,
    _density_moments_for_target_phi,
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
    """Fail closed for retired or non-promoted reduced-model contracts."""

    model = _runtime_model_key(cfg)
    if model in {"cetg", "krehm"}:
        raise NotImplementedError(
            f"physics.reduced_model={cfg.physics.reduced_model!r} is not supported "
            "by the maintained runtime. Use physics.reduced_model='gyrokinetic' "
            "for promoted full-GK workflows."
        )
    raise ValueError(f"Unknown physics.reduced_model={cfg.physics.reduced_model!r}")


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
    if cfg.physics.adiabatic_electrons and has_kinetic_electron:
        raise ValueError(
            "adiabatic_electrons=True conflicts with kinetic electron species"
        )

    tau_e = float(cfg.physics.tau_e) if cfg.physics.adiabatic_electrons else 0.0
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
