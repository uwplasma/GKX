"""Flux and turbulent-heating kernels evaluated inside the traced step."""

from __future__ import annotations

import jax.numpy as jnp

from gkx.core_grid import SpectralGrid
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import LinearParams
from gkx.operators.moments import (
    _heat_flux_channel_contrib_species,
    _particle_flux_channel_contrib_species,
    _turbulent_heating_contrib_species,
)

__all__ = [
    "heat_flux_species",
    "heat_flux_total",
    "particle_flux_species",
    "particle_flux_total",
    "turbulent_heating_species",
]


def heat_flux_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> jnp.ndarray:
    """Heat-flux diagnostic per species (gyroBohm units)."""

    es_contrib, apar_contrib, bpar_contrib = _heat_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return jnp.sum(es_contrib + apar_contrib + bpar_contrib, axis=(1, 2, 3))


def _summed(species_fn):
    def total(*args, **kwargs) -> jnp.ndarray:
        return jnp.sum(species_fn(*args, **kwargs))

    total.__doc__ = f"Sum over species of :func:`{species_fn.__name__}`."
    return total


heat_flux_total = _summed(heat_flux_species)


def particle_flux_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
    global_species: int | None = None,
) -> jnp.ndarray:
    """Particle-flux diagnostic per species."""

    es_contrib, apar_contrib, bpar_contrib = _particle_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
        global_species=global_species,
    )
    return jnp.sum(es_contrib + apar_contrib + bpar_contrib, axis=(1, 2, 3))


particle_flux_total = _summed(particle_flux_species)


def turbulent_heating_species(
    G: jnp.ndarray,
    G_old: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    phi_old: jnp.ndarray,
    apar_old: jnp.ndarray,
    bpar_old: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    dt: jnp.ndarray | float,
    *,
    use_dealias: bool = True,
) -> jnp.ndarray:
    """Turbulent-heating diagnostic per species."""

    contrib = _turbulent_heating_contrib_species(
        G,
        G_old,
        phi,
        apar,
        bpar,
        phi_old,
        apar_old,
        bpar_old,
        cache,
        grid,
        params,
        vol_fac,
        dt,
        use_dealias=use_dealias,
    )
    return jnp.sum(contrib, axis=(1, 2, 3))
