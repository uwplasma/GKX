"""Per-term RHS decomposition for diagnostics and parity audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.moments import build_H
from spectraxgk.operators.linear.params import LinearParams, _as_species_array
from spectraxgk.terms.assembly_helpers import (
    _collision_contribution_or_zero,
    _rhs_field_views,
)
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.fields import _solve_fields_impl, solve_fields
from spectraxgk.terms.linear_dissipation import (
    end_damping_contribution,
    hypercollisions_contribution,
    hyperdiffusion_contribution,
)
from spectraxgk.terms.linear_terms import (
    curvature_gradb_contribution,
    diamagnetic_contribution,
    linked_streaming_contribution,
    mirror_contribution,
)


@dataclass(frozen=True)
class _RHSState:
    G: jnp.ndarray
    out_dtype: Any
    real_dtype: Any
    imag: jnp.ndarray
    squeeze_species: bool


@dataclass(frozen=True)
class _SpeciesArrays:
    charge: jnp.ndarray
    density: jnp.ndarray
    mass: jnp.ndarray
    temp: jnp.ndarray
    tz: jnp.ndarray
    vth: jnp.ndarray
    tprim: jnp.ndarray
    fprim: jnp.ndarray
    nu: jnp.ndarray


@dataclass(frozen=True)
class _ScalarParams:
    omega_d_scale: jnp.ndarray
    omega_star_scale: jnp.ndarray
    kpar_scale: jnp.ndarray
    nu_hyper: jnp.ndarray
    nu_hyper_l: jnp.ndarray
    nu_hyper_m: jnp.ndarray
    nu_hyper_lm: jnp.ndarray
    hypercollisions_const: jnp.ndarray
    hypercollisions_kz: jnp.ndarray
    damp_amp: jnp.ndarray
    D_hyper: jnp.ndarray
    p_hyper_kperp: jnp.ndarray


@dataclass(frozen=True)
class _TermWeights:
    streaming: jnp.ndarray
    mirror: jnp.ndarray
    curvature: jnp.ndarray
    gradb: jnp.ndarray
    diamagnetic: jnp.ndarray
    collisions: jnp.ndarray
    hypercollisions: jnp.ndarray
    hyperdiffusion: jnp.ndarray
    end_damping: jnp.ndarray
    bpar: jnp.ndarray
    fapar: jnp.ndarray


@dataclass(frozen=True)
class _DiagnosticFields:
    fields: FieldState
    H: jnp.ndarray
    h_apar: jnp.ndarray | None
    h_bpar: jnp.ndarray | None


_TERM_SUM_ORDER = (
    "mirror",
    "curvature",
    "gradb",
    "diamagnetic",
    "collisions",
    "hypercollisions",
    "hyperdiffusion",
    "end_damping",
)


def _normalized_rhs_state(G: jnp.ndarray, cache: LinearCache) -> _RHSState:
    out_dtype = jnp.result_type(G, jnp.complex64)
    G_arr = jnp.asarray(G, dtype=out_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=out_dtype)).dtype
    squeeze_species = False
    if G_arr.ndim == 5:
        G_arr = G_arr[None, ...]
        squeeze_species = True
    if G_arr.ndim != 6:
        raise ValueError(
            "G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )
    if cache.Jl.shape[0] != G_arr.shape[0]:
        raise ValueError("Cache species dimension does not match G")
    return _RHSState(
        G=G_arr,
        out_dtype=out_dtype,
        real_dtype=real_dtype,
        imag=jnp.asarray(1j, dtype=out_dtype),
        squeeze_species=squeeze_species,
    )


def _species_arrays(params: LinearParams, ns: int, real_dtype: Any) -> _SpeciesArrays:
    return _SpeciesArrays(
        charge=_as_species_array(params.charge_sign, ns, "charge_sign").astype(
            real_dtype
        ),
        density=_as_species_array(params.density, ns, "density").astype(real_dtype),
        mass=_as_species_array(params.mass, ns, "mass").astype(real_dtype),
        temp=_as_species_array(params.temp, ns, "temp").astype(real_dtype),
        tz=_as_species_array(params.tz, ns, "tz").astype(real_dtype),
        vth=_as_species_array(params.vth, ns, "vth").astype(real_dtype),
        tprim=_as_species_array(params.R_over_LTi, ns, "R_over_LTi").astype(
            real_dtype
        ),
        fprim=_as_species_array(params.R_over_Ln, ns, "R_over_Ln").astype(
            real_dtype
        ),
        nu=_as_species_array(params.nu, ns, "nu").astype(real_dtype),
    )


def _scalar_params(
    params: LinearParams,
    real_dtype: Any,
    dt: jnp.ndarray | float | None,
) -> _ScalarParams:
    damp_amp = jnp.asarray(params.damp_ends_amp, dtype=real_dtype)
    if dt is not None:
        dt_arr = jnp.asarray(dt, dtype=real_dtype)
        damp_amp = jnp.where(dt_arr != 0.0, damp_amp / dt_arr, damp_amp)
    return _ScalarParams(
        omega_d_scale=jnp.asarray(params.omega_d_scale, dtype=real_dtype),
        omega_star_scale=jnp.asarray(params.omega_star_scale, dtype=real_dtype),
        kpar_scale=jnp.asarray(params.kpar_scale, dtype=real_dtype),
        nu_hyper=jnp.asarray(params.nu_hyper, dtype=real_dtype),
        nu_hyper_l=jnp.asarray(params.nu_hyper_l, dtype=real_dtype),
        nu_hyper_m=jnp.asarray(params.nu_hyper_m, dtype=real_dtype),
        nu_hyper_lm=jnp.asarray(params.nu_hyper_lm, dtype=real_dtype),
        hypercollisions_const=jnp.asarray(
            params.hypercollisions_const, dtype=real_dtype
        ),
        hypercollisions_kz=jnp.asarray(params.hypercollisions_kz, dtype=real_dtype),
        damp_amp=damp_amp,
        D_hyper=jnp.asarray(params.D_hyper, dtype=real_dtype),
        p_hyper_kperp=jnp.asarray(params.p_hyper_kperp, dtype=real_dtype),
    )


def _term_weights(
    params: LinearParams,
    term_cfg: TermConfig,
    real_dtype: Any,
) -> _TermWeights:
    w_apar = jnp.asarray(term_cfg.apar, dtype=real_dtype)
    return _TermWeights(
        streaming=jnp.asarray(term_cfg.streaming, dtype=real_dtype),
        mirror=jnp.asarray(term_cfg.mirror, dtype=real_dtype),
        curvature=jnp.asarray(term_cfg.curvature, dtype=real_dtype),
        gradb=jnp.asarray(term_cfg.gradb, dtype=real_dtype),
        diamagnetic=jnp.asarray(term_cfg.diamagnetic, dtype=real_dtype),
        collisions=jnp.asarray(term_cfg.collisions, dtype=real_dtype),
        hypercollisions=jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype),
        hyperdiffusion=jnp.asarray(term_cfg.hyperdiffusion, dtype=real_dtype),
        end_damping=jnp.asarray(term_cfg.end_damping, dtype=real_dtype),
        bpar=jnp.asarray(term_cfg.bpar, dtype=real_dtype),
        fapar=jnp.asarray(params.fapar, dtype=real_dtype) * w_apar,
    )


def _diagnostic_fields(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    species: _SpeciesArrays,
    weights: _TermWeights,
    *,
    use_custom_vjp: bool,
) -> _DiagnosticFields:
    fields_fn = solve_fields if use_custom_vjp else _solve_fields_impl
    fields = fields_fn(
        G,
        cache,
        params,
        charge=species.charge,
        density=species.density,
        temp=species.temp,
        mass=species.mass,
        tz=species.tz,
        vth=species.vth,
        fapar=weights.fapar,
        w_bpar=weights.bpar,
    )
    _, _, h_apar, h_bpar = _rhs_field_views(fields, term_cfg)
    H = build_H(
        G,
        cache.Jl,
        fields.phi,
        species.tz,
        apar=h_apar,
        vth=species.vth,
        bpar=h_bpar,
        JlB=cache.JlB,
    )
    return _DiagnosticFields(fields=fields, H=H, h_apar=h_apar, h_bpar=h_bpar)


def _streaming_contribution(
    G: jnp.ndarray,
    cache: LinearCache,
    species: _SpeciesArrays,
    scalars: _ScalarParams,
    weights: _TermWeights,
    diagnostic: _DiagnosticFields,
) -> jnp.ndarray:
    return linked_streaming_contribution(
        G,
        phi=diagnostic.fields.phi,
        apar=diagnostic.h_apar,
        bpar=diagnostic.h_bpar,
        Jl=cache.Jl,
        JlB=cache.JlB,
        tz=species.tz,
        kz=cache.kz,
        dz=cache.dz,
        vth=species.vth,
        sqrt_p=cache.sqrt_p,
        sqrt_m=cache.sqrt_m_ladder,
        kpar_scale=scalars.kpar_scale,
        weight=weights.streaming,
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
        linked_inverse_permutation=cache.linked_inverse_permutation,
        linked_full_cover=cache.linked_full_cover,
        linked_gather_map=cache.linked_gather_map,
        linked_gather_mask=cache.linked_gather_mask,
        linked_use_gather=cache.linked_use_gather,
        use_twist_shift=cache.use_twist_shift,
    )


def _drift_contributions(
    H: jnp.ndarray,
    cache: LinearCache,
    species: _SpeciesArrays,
    scalars: _ScalarParams,
    weights: _TermWeights,
    imag: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mirror = mirror_contribution(
        H,
        vth=species.vth,
        bgrad=cache.bgrad,
        ell=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        weight=weights.mirror,
    )
    zero_weight = jnp.asarray(0.0, dtype=weights.curvature.dtype)
    curvature = curvature_gradb_contribution(
        H,
        tz=species.tz,
        omega_d_scale=scalars.omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        imag=imag,
        weight_curv=weights.curvature,
        weight_gradb=zero_weight,
    )
    gradb = curvature_gradb_contribution(
        H,
        tz=species.tz,
        omega_d_scale=scalars.omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        ell=cache.l,
        m=cache.m,
        imag=imag,
        weight_curv=zero_weight,
        weight_gradb=weights.gradb,
    )
    return mirror, curvature, gradb


def _diamagnetic_contribution(
    G: jnp.ndarray,
    cache: LinearCache,
    species: _SpeciesArrays,
    scalars: _ScalarParams,
    weights: _TermWeights,
    diagnostic: _DiagnosticFields,
    imag: jnp.ndarray,
) -> jnp.ndarray:
    return diamagnetic_contribution(
        jnp.zeros_like(G),
        phi=diagnostic.fields.phi,
        apar=diagnostic.h_apar,
        bpar=diagnostic.h_bpar,
        Jl=cache.Jl,
        JlB=cache.JlB,
        l4=cache.l4,
        tprim=species.tprim,
        fprim=species.fprim,
        tz=species.tz,
        vth=species.vth,
        omega_star_scale=scalars.omega_star_scale,
        ky=cache.ky,
        imag=imag,
        weight=weights.diamagnetic,
    )


def _dissipation_contributions(
    G: jnp.ndarray,
    H: jnp.ndarray,
    cache: LinearCache,
    species: _SpeciesArrays,
    scalars: _ScalarParams,
    weights: _TermWeights,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    collisions = _collision_contribution_or_zero(
        H,
        G=G,
        Jl=cache.Jl,
        JlB=cache.JlB,
        b=cache.b,
        nu=species.nu,
        collision_lam=cache.collision_lam,
        lb_lam=cache.lb_lam,
        weight=weights.collisions,
    )
    hypercollisions = hypercollisions_contribution(
        G,
        vth=species.vth,
        nu_hyper=scalars.nu_hyper,
        nu_hyper_l=scalars.nu_hyper_l,
        nu_hyper_m=scalars.nu_hyper_m,
        nu_hyper_lm=scalars.nu_hyper_lm,
        hyper_ratio=cache.hyper_ratio,
        ratio_l=cache.ratio_l,
        ratio_m=cache.ratio_m,
        ratio_lm=cache.ratio_lm,
        mask_const=cache.mask_const,
        mask_kz=cache.mask_kz,
        m_pow=cache.m_pow,
        m_norm_kz_factor=cache.m_norm_kz_factor,
        kz=cache.kz,
        kpar_scale=scalars.kpar_scale,
        hypercollisions_const=scalars.hypercollisions_const,
        hypercollisions_kz=scalars.hypercollisions_kz,
        weight=weights.hypercollisions,
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
        linked_inverse_permutation=cache.linked_inverse_permutation,
        linked_full_cover=cache.linked_full_cover,
        linked_gather_map=cache.linked_gather_map,
        linked_gather_mask=cache.linked_gather_mask,
        linked_use_gather=cache.linked_use_gather,
    )
    hyperdiffusion = hyperdiffusion_contribution(
        G,
        kx=cache.kx,
        ky=cache.ky,
        dealias_mask=cache.dealias_mask,
        D_hyper=scalars.D_hyper,
        p_hyper_kperp=scalars.p_hyper_kperp,
        weight=weights.hyperdiffusion,
    )
    end_damping = end_damping_contribution(
        H,
        ky=cache.ky,
        damp_profile=cache.damp_profile,
        linked_damp_profile=cache.linked_damp_profile,
        damp_amp=scalars.damp_amp,
        weight=weights.end_damping,
    )
    return collisions, hypercollisions, hyperdiffusion, end_damping


def _diagnostic_contributions(
    state: _RHSState,
    cache: LinearCache,
    species: _SpeciesArrays,
    scalars: _ScalarParams,
    weights: _TermWeights,
    diagnostic: _DiagnosticFields,
) -> dict[str, jnp.ndarray]:
    mirror, curvature, gradb = _drift_contributions(
        diagnostic.H,
        cache,
        species,
        scalars,
        weights,
        state.imag,
    )
    collisions, hypercollisions, hyperdiffusion, end_damping = (
        _dissipation_contributions(
            state.G, diagnostic.H, cache, species, scalars, weights
        )
    )
    return {
        "streaming": _streaming_contribution(
            state.G, cache, species, scalars, weights, diagnostic
        ),
        "mirror": mirror,
        "curvature": curvature,
        "gradb": gradb,
        "diamagnetic": _diamagnetic_contribution(
            state.G, cache, species, scalars, weights, diagnostic, state.imag
        ),
        "collisions": collisions,
        "hypercollisions": hypercollisions,
        "hyperdiffusion": hyperdiffusion,
        "end_damping": end_damping,
    }


def _sum_diagnostic_terms(contrib: dict[str, jnp.ndarray]) -> jnp.ndarray:
    total = contrib["streaming"]
    for key in _TERM_SUM_ORDER:
        total = total + contrib[key]
    return total


def _squeeze_species_outputs(
    total: jnp.ndarray,
    contrib: dict[str, jnp.ndarray],
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    return total[0], {key: arr[0] for key, arr in contrib.items()}


def assemble_rhs_terms_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    terms: TermConfig | None = None,
    use_custom_vjp: bool = True,
    dt: jnp.ndarray | float | None = None,
) -> tuple[jnp.ndarray, FieldState, dict[str, jnp.ndarray]]:
    """Assemble per-term RHS contributions (debug/diagnostic path).

    Returns (total_rhs, fields, term_contributions).
    """

    term_cfg = terms or TermConfig()
    state = _normalized_rhs_state(G, cache)
    species = _species_arrays(params, state.G.shape[0], state.real_dtype)
    scalars = _scalar_params(params, state.real_dtype, dt)
    weights = _term_weights(params, term_cfg, state.real_dtype)
    diagnostic = _diagnostic_fields(
        state.G,
        cache,
        params,
        term_cfg,
        species,
        weights,
        use_custom_vjp=use_custom_vjp,
    )
    contrib = _diagnostic_contributions(
        state, cache, species, scalars, weights, diagnostic
    )
    total = _sum_diagnostic_terms(contrib)
    if state.squeeze_species:
        total, contrib = _squeeze_species_outputs(total, contrib)
    return total.astype(state.out_dtype), diagnostic.fields, contrib


__all__ = ["assemble_rhs_terms_cached"]
