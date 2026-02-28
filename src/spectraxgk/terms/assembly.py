"""RHS assembly for term-wise gyrokinetic evolution."""

from __future__ import annotations

from typing import Tuple

import functools
import jax

import jax.numpy as jnp

from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import SpectralGrid
from spectraxgk.linear import LinearCache, LinearParams, _as_species_array, build_H, build_linear_cache
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.fields import _solve_fields_impl, solve_fields
from spectraxgk.terms.linear_terms import (
    collisions_contribution,
    curvature_gradb_contribution,
    diamagnetic_contribution,
    end_damping_contribution,
    hypercollisions_contribution,
    hyperdiffusion_contribution,
    mirror_contribution,
    streaming_contribution_gx,
)


def assemble_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    terms: TermConfig | None = None,
    use_custom_vjp: bool = True,
) -> Tuple[jnp.ndarray, FieldState]:
    """Assemble the RHS from term-wise modules using a precomputed cache."""

    term_cfg = terms or TermConfig()

    out_dtype = jnp.result_type(G, jnp.complex64)
    G = jnp.asarray(G, dtype=out_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=out_dtype)).dtype
    imag = jnp.asarray(1j, dtype=out_dtype)
    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    if G.ndim != 6:
        raise ValueError("G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
    if cache.Jl.shape[0] != G.shape[0]:
        raise ValueError("Cache species dimension does not match G")

    ns = G.shape[0]
    charge = _as_species_array(params.charge_sign, ns, "charge_sign").astype(real_dtype)
    density = _as_species_array(params.density, ns, "density").astype(real_dtype)
    mass = _as_species_array(params.mass, ns, "mass").astype(real_dtype)
    temp = _as_species_array(params.temp, ns, "temp").astype(real_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tprim = _as_species_array(params.R_over_LTi, ns, "R_over_LTi").astype(real_dtype)
    fprim = _as_species_array(params.R_over_Ln, ns, "R_over_Ln").astype(real_dtype)
    nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)

    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    omega_star_scale = jnp.asarray(params.omega_star_scale, dtype=real_dtype)
    kpar_scale = jnp.asarray(params.kpar_scale, dtype=real_dtype)
    nu_hyper = jnp.asarray(params.nu_hyper, dtype=real_dtype)
    nu_hyper_l = jnp.asarray(params.nu_hyper_l, dtype=real_dtype)
    nu_hyper_m = jnp.asarray(params.nu_hyper_m, dtype=real_dtype)
    nu_hyper_lm = jnp.asarray(params.nu_hyper_lm, dtype=real_dtype)
    hypercollisions_const = jnp.asarray(params.hypercollisions_const, dtype=real_dtype)
    hypercollisions_kz = jnp.asarray(params.hypercollisions_kz, dtype=real_dtype)
    D_hyper = jnp.asarray(params.D_hyper, dtype=real_dtype)
    p_hyper_kperp = jnp.asarray(params.p_hyper_kperp, dtype=real_dtype)
    D_hyper = jnp.asarray(params.D_hyper, dtype=real_dtype)
    p_hyper_kperp = jnp.asarray(params.p_hyper_kperp, dtype=real_dtype)
    damp_amp = jnp.asarray(params.damp_ends_amp, dtype=real_dtype)

    w_stream = jnp.asarray(term_cfg.streaming, dtype=real_dtype)
    w_mirror = jnp.asarray(term_cfg.mirror, dtype=real_dtype)
    w_curv = jnp.asarray(term_cfg.curvature, dtype=real_dtype)
    w_gradb = jnp.asarray(term_cfg.gradb, dtype=real_dtype)
    w_dia = jnp.asarray(term_cfg.diamagnetic, dtype=real_dtype)
    w_coll = jnp.asarray(term_cfg.collisions, dtype=real_dtype)
    w_hyper = jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype)
    w_hyperdiff = jnp.asarray(term_cfg.hyperdiffusion, dtype=real_dtype)
    w_hyperdiff = jnp.asarray(term_cfg.hyperdiffusion, dtype=real_dtype)
    w_damp = jnp.asarray(term_cfg.end_damping, dtype=real_dtype)
    w_apar = jnp.asarray(term_cfg.apar, dtype=real_dtype)
    w_bpar = jnp.asarray(term_cfg.bpar, dtype=real_dtype)
    fapar = jnp.asarray(params.fapar, dtype=real_dtype) * w_apar

    fields_fn = solve_fields if use_custom_vjp else _solve_fields_impl
    fields = fields_fn(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=fapar,
        w_bpar=w_bpar,
    )

    Jl = cache.Jl
    JlB = cache.JlB
    apar = fields.apar if fields.apar is not None else jnp.zeros_like(fields.phi)
    bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(fields.phi)
    H = build_H(G, Jl, fields.phi, tz, apar=apar, vth=vth, bpar=bpar, JlB=JlB)

    dG = streaming_contribution_gx(
        G,
        phi=fields.phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        tz=tz,
        kz=cache.kz,
        dz=cache.dz,
        vth=vth,
        sqrt_p=cache.sqrt_p,
        sqrt_m=cache.sqrt_m_ladder,
        kpar_scale=kpar_scale,
        weight=w_stream,
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
        linked_inverse_permutation=cache.linked_inverse_permutation,
        use_twist_shift=cache.use_twist_shift,
    )
    dG = dG + mirror_contribution(
        H,
        vth=vth,
        bgrad=cache.bgrad,
        l=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        weight=w_mirror,
    )
    dG = dG + curvature_gradb_contribution(
        H,
        tz=tz,
        omega_d_scale=omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        l=cache.l,
        m=cache.m,
        imag=imag,
        weight_curv=w_curv,
        weight_gradb=w_gradb,
    )
    dG = diamagnetic_contribution(
        dG,
        phi=fields.phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        l4=cache.l4,
        tprim=tprim,
        fprim=fprim,
        tz=tz,
        vth=vth,
        omega_star_scale=omega_star_scale,
        ky=cache.ky,
        imag=imag,
        weight=w_dia,
    )
    dG = dG + collisions_contribution(
        H,
        nu=nu,
        lb_lam=cache.lb_lam,
        weight=w_coll,
    )
    dG = dG + hypercollisions_contribution(
        G,
        vth=vth,
        nu_hyper=nu_hyper,
        nu_hyper_l=nu_hyper_l,
        nu_hyper_m=nu_hyper_m,
        nu_hyper_lm=nu_hyper_lm,
        hyper_ratio=cache.hyper_ratio,
        ratio_l=cache.ratio_l,
        ratio_m=cache.ratio_m,
        ratio_lm=cache.ratio_lm,
        mask_const=cache.mask_const,
        mask_kz=cache.mask_kz,
        m_pow=cache.m_pow,
        m_norm_kz_factor=cache.m_norm_kz_factor,
        kz=cache.kz,
        kpar_scale=kpar_scale,
        hypercollisions_const=hypercollisions_const,
        hypercollisions_kz=hypercollisions_kz,
        weight=w_hyper,
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
        linked_inverse_permutation=cache.linked_inverse_permutation,
    )
    dG = dG + hyperdiffusion_contribution(
        G,
        kx=cache.kx,
        ky=cache.ky,
        dealias_mask=cache.dealias_mask,
        D_hyper=D_hyper,
        p_hyper_kperp=p_hyper_kperp,
        weight=w_hyperdiff,
    )
    dG = dG + end_damping_contribution(
        H,
        ky=cache.ky,
        damp_profile=cache.damp_profile,
        damp_amp=damp_amp,
        weight=w_damp,
    )

    if squeeze_species:
        dG = dG[0]
    return dG.astype(out_dtype), fields


def assemble_rhs_terms_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    terms: TermConfig | None = None,
    use_custom_vjp: bool = True,
) -> tuple[jnp.ndarray, FieldState, dict[str, jnp.ndarray]]:
    """Assemble per-term RHS contributions (debug/diagnostic path).

    Returns (total_rhs, fields, term_contributions).
    """

    term_cfg = terms or TermConfig()

    out_dtype = jnp.result_type(G, jnp.complex64)
    G = jnp.asarray(G, dtype=out_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=out_dtype)).dtype
    imag = jnp.asarray(1j, dtype=out_dtype)
    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    if G.ndim != 6:
        raise ValueError("G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
    if cache.Jl.shape[0] != G.shape[0]:
        raise ValueError("Cache species dimension does not match G")

    ns = G.shape[0]
    charge = _as_species_array(params.charge_sign, ns, "charge_sign").astype(real_dtype)
    density = _as_species_array(params.density, ns, "density").astype(real_dtype)
    mass = _as_species_array(params.mass, ns, "mass").astype(real_dtype)
    temp = _as_species_array(params.temp, ns, "temp").astype(real_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)
    tprim = _as_species_array(params.R_over_LTi, ns, "R_over_LTi").astype(real_dtype)
    fprim = _as_species_array(params.R_over_Ln, ns, "R_over_Ln").astype(real_dtype)
    nu = _as_species_array(params.nu, ns, "nu").astype(real_dtype)

    omega_d_scale = jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    omega_star_scale = jnp.asarray(params.omega_star_scale, dtype=real_dtype)
    kpar_scale = jnp.asarray(params.kpar_scale, dtype=real_dtype)
    nu_hyper = jnp.asarray(params.nu_hyper, dtype=real_dtype)
    nu_hyper_l = jnp.asarray(params.nu_hyper_l, dtype=real_dtype)
    nu_hyper_m = jnp.asarray(params.nu_hyper_m, dtype=real_dtype)
    nu_hyper_lm = jnp.asarray(params.nu_hyper_lm, dtype=real_dtype)
    hypercollisions_const = jnp.asarray(params.hypercollisions_const, dtype=real_dtype)
    hypercollisions_kz = jnp.asarray(params.hypercollisions_kz, dtype=real_dtype)
    damp_amp = jnp.asarray(params.damp_ends_amp, dtype=real_dtype)
    D_hyper = jnp.asarray(params.D_hyper, dtype=real_dtype)
    p_hyper_kperp = jnp.asarray(params.p_hyper_kperp, dtype=real_dtype)

    w_stream = jnp.asarray(term_cfg.streaming, dtype=real_dtype)
    w_mirror = jnp.asarray(term_cfg.mirror, dtype=real_dtype)
    w_curv = jnp.asarray(term_cfg.curvature, dtype=real_dtype)
    w_gradb = jnp.asarray(term_cfg.gradb, dtype=real_dtype)
    w_dia = jnp.asarray(term_cfg.diamagnetic, dtype=real_dtype)
    w_coll = jnp.asarray(term_cfg.collisions, dtype=real_dtype)
    w_hyper = jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype)
    w_hyperdiff = jnp.asarray(term_cfg.hyperdiffusion, dtype=real_dtype)
    w_damp = jnp.asarray(term_cfg.end_damping, dtype=real_dtype)
    w_apar = jnp.asarray(term_cfg.apar, dtype=real_dtype)
    w_bpar = jnp.asarray(term_cfg.bpar, dtype=real_dtype)
    fapar = jnp.asarray(params.fapar, dtype=real_dtype) * w_apar

    fields_fn = solve_fields if use_custom_vjp else _solve_fields_impl
    fields = fields_fn(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=fapar,
        w_bpar=w_bpar,
    )

    Jl = cache.Jl
    JlB = cache.JlB
    apar = fields.apar if fields.apar is not None else jnp.zeros_like(fields.phi)
    bpar = fields.bpar if fields.bpar is not None else jnp.zeros_like(fields.phi)
    H = build_H(G, Jl, fields.phi, tz, apar=apar, vth=vth, bpar=bpar, JlB=JlB)

    contrib: dict[str, jnp.ndarray] = {}
    contrib["streaming"] = streaming_contribution_gx(
        G,
        phi=fields.phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        tz=tz,
        kz=cache.kz,
        dz=cache.dz,
        vth=vth,
        sqrt_p=cache.sqrt_p,
        sqrt_m=cache.sqrt_m_ladder,
        kpar_scale=kpar_scale,
        weight=w_stream,
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
        linked_inverse_permutation=cache.linked_inverse_permutation,
        use_twist_shift=cache.use_twist_shift,
    )
    contrib["mirror"] = mirror_contribution(
        H,
        vth=vth,
        bgrad=cache.bgrad,
        l=cache.l,
        sqrt_m=cache.sqrt_m,
        sqrt_m_p1=cache.sqrt_m_p1,
        weight=w_mirror,
    )
    contrib["curvature"] = curvature_gradb_contribution(
        H,
        tz=tz,
        omega_d_scale=omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        l=cache.l,
        m=cache.m,
        imag=imag,
        weight_curv=w_curv,
        weight_gradb=jnp.asarray(0.0, dtype=real_dtype),
    )
    contrib["gradb"] = curvature_gradb_contribution(
        H,
        tz=tz,
        omega_d_scale=omega_d_scale,
        cv_d=cache.cv_d,
        gb_d=cache.gb_d,
        l=cache.l,
        m=cache.m,
        imag=imag,
        weight_curv=jnp.asarray(0.0, dtype=real_dtype),
        weight_gradb=w_gradb,
    )
    zero = jnp.zeros_like(G)
    contrib["diamagnetic"] = diamagnetic_contribution(
        zero,
        phi=fields.phi,
        apar=apar,
        bpar=bpar,
        Jl=Jl,
        JlB=JlB,
        l4=cache.l4,
        tprim=tprim,
        fprim=fprim,
        tz=tz,
        vth=vth,
        omega_star_scale=omega_star_scale,
        ky=cache.ky,
        imag=imag,
        weight=w_dia,
    )
    contrib["collisions"] = collisions_contribution(
        H,
        nu=nu,
        lb_lam=cache.lb_lam,
        weight=w_coll,
    )
    contrib["hypercollisions"] = hypercollisions_contribution(
        G,
        vth=vth,
        nu_hyper=nu_hyper,
        nu_hyper_l=nu_hyper_l,
        nu_hyper_m=nu_hyper_m,
        nu_hyper_lm=nu_hyper_lm,
        hyper_ratio=cache.hyper_ratio,
        ratio_l=cache.ratio_l,
        ratio_m=cache.ratio_m,
        ratio_lm=cache.ratio_lm,
        mask_const=cache.mask_const,
        mask_kz=cache.mask_kz,
        m_pow=cache.m_pow,
        m_norm_kz_factor=cache.m_norm_kz_factor,
        kz=cache.kz,
        kpar_scale=kpar_scale,
        hypercollisions_const=hypercollisions_const,
        hypercollisions_kz=hypercollisions_kz,
        weight=w_hyper,
        linked_indices=cache.linked_indices,
        linked_kz=cache.linked_kz,
        linked_inverse_permutation=cache.linked_inverse_permutation,
    )
    contrib["hyperdiffusion"] = hyperdiffusion_contribution(
        G,
        kx=cache.kx,
        ky=cache.ky,
        dealias_mask=cache.dealias_mask,
        D_hyper=D_hyper,
        p_hyper_kperp=p_hyper_kperp,
        weight=w_hyperdiff,
    )
    contrib["end_damping"] = end_damping_contribution(
        H,
        ky=cache.ky,
        damp_profile=cache.damp_profile,
        damp_amp=damp_amp,
        weight=w_damp,
    )

    total = contrib["streaming"]
    for key in (
        "mirror",
        "curvature",
        "gradb",
        "diamagnetic",
        "collisions",
        "hypercollisions",
        "end_damping",
    ):
        total = total + contrib[key]

    if squeeze_species:
        total = total[0]
        for key, arr in list(contrib.items()):
            contrib[key] = arr[0]
    return total.astype(out_dtype), fields, contrib


@functools.partial(jax.jit)
def assemble_rhs_cached_jit(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: TermConfig,
) -> Tuple[jnp.ndarray, FieldState]:
    """Jitted wrapper for cached RHS assembly."""

    return assemble_rhs_cached(G, cache, params, terms=terms)


def compute_fields_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    terms: TermConfig | None = None,
    use_custom_vjp: bool = True,
) -> FieldState:
    """Compute fields for a cached state without assembling the RHS."""

    term_cfg = terms or TermConfig()

    out_dtype = jnp.result_type(G, jnp.complex64)
    G = jnp.asarray(G, dtype=out_dtype)
    real_dtype = jnp.real(jnp.empty((), dtype=out_dtype)).dtype

    squeeze_species = False
    if G.ndim == 5:
        G = G[None, ...]
        squeeze_species = True
    if G.ndim != 6:
        raise ValueError("G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)")
    if cache.Jl.shape[0] != G.shape[0]:
        raise ValueError("Cache species dimension does not match G")

    ns = G.shape[0]
    charge = _as_species_array(params.charge_sign, ns, "charge_sign").astype(real_dtype)
    density = _as_species_array(params.density, ns, "density").astype(real_dtype)
    mass = _as_species_array(params.mass, ns, "mass").astype(real_dtype)
    temp = _as_species_array(params.temp, ns, "temp").astype(real_dtype)
    tz = _as_species_array(params.tz, ns, "tz").astype(real_dtype)
    vth = _as_species_array(params.vth, ns, "vth").astype(real_dtype)

    w_apar = jnp.asarray(term_cfg.apar, dtype=real_dtype)
    w_bpar = jnp.asarray(term_cfg.bpar, dtype=real_dtype)
    fapar = jnp.asarray(params.fapar, dtype=real_dtype) * w_apar

    fields_fn = solve_fields if use_custom_vjp else _solve_fields_impl
    fields = fields_fn(
        G,
        cache,
        params,
        charge=charge,
        density=density,
        temp=temp,
        mass=mass,
        tz=tz,
        vth=vth,
        fapar=fapar,
        w_bpar=w_bpar,
    )
    if squeeze_species:
        return FieldState(phi=fields.phi, apar=fields.apar, bpar=fields.bpar)
    return fields


def assemble_rhs(
    G: jnp.ndarray,
    grid: SpectralGrid,
    geom: SAlphaGeometry,
    params: LinearParams,
    *,
    Nl: int,
    Nm: int,
    terms: TermConfig | None = None,
    cache: LinearCache | None = None,
) -> Tuple[jnp.ndarray, FieldState]:
    """Assemble the RHS from term-wise modules."""

    cache = cache or build_linear_cache(grid, geom, params, Nl, Nm)
    return assemble_rhs_cached(G, cache, params, terms=terms)
