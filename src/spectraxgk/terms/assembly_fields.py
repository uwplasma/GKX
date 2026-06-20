"""Field-only cached solve helpers for runtime diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import LinearParams, _as_species_array
from spectraxgk.terms.assembly_helpers import _apply_external_phi_source
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.fields import _solve_fields_impl, solve_fields

def compute_fields_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    *,
    terms: TermConfig | None = None,
    use_custom_vjp: bool = True,
    external_phi: jnp.ndarray | float | None = None,
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
        raise ValueError(
            "G must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
        )
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
    fields = _apply_external_phi_source(fields, external_phi=external_phi)
    if squeeze_species:
        return FieldState(phi=fields.phi, apar=fields.apar, bpar=fields.bpar)
    return fields

__all__ = ["compute_fields_cached"]
