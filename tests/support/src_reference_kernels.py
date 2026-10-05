"""Reference kernels used only by the test suite."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from gkx.core_grid import SpectralGrid
from gkx.core_ky_layout import (
    KY_AXIS,
    _xp,
    symmetrize_self_conjugate_rows,
    to_full,
    to_half,
)
from gkx.diagnostics.analysis import ObservedOrderMetrics
from gkx.operators.fluxes import turbulent_heating_species
from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.dissipation import _species_collision_frequency
from gkx.operators.linear.params import LinearParams
from gkx.operators.moments import (
    _heat_flux_channel_contrib_species,
    _particle_flux_channel_contrib_species,
)
from gkx.operators.nonlinear.brackets import (
    _apply_mask_xy,
    _spectral_bracket,
    _spectral_bracket_multi_full,
    _spectral_bracket_multi_real_fft,
)


def drift_kinetic_dougherty_contribution(
    state: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    weight: jnp.ndarray = jnp.asarray(1.0),
) -> jnp.ndarray:
    """Apply the linearized drift-kinetic Dougherty moment operator.

    This is Appendix C, equation (C6), of Frei, Hoffmann & Ricci (2022),
    mapped to GKX's ``(species, ell, m, ky, kx, z)`` ordering and
    Laguerre-sign convention. The density and parallel-flow moments and the
    combined thermal moment ``sqrt(2) G[0, 2] + 2 G[1, 0]`` are exact null
    directions. Five-dimensional single-species states are also accepted.

    The kernel is both an independently auditable reference and a usable
    long-wavelength collision operator. It is not the finite-Larmor-radius
    Sugama or Coulomb operator.
    """

    value = jnp.asarray(state)
    if value.ndim not in {5, 6}:
        raise ValueError("collision state must have five or six dimensions")
    expanded = value[None, ...] if value.ndim == 5 else value
    if expanded.shape[1] < 2 or expanded.shape[2] < 3:
        raise ValueError("drift-kinetic Dougherty requires Nl >= 2 and Nm >= 3")

    ns, nl, nm = map(int, expanded.shape[:3])
    real_dtype = jnp.real(expanded).dtype
    nu_s = _species_collision_frequency(nu, ns=ns, dtype=real_dtype)
    rate = nu_s[:, None, None, None, None, None]
    ell = jnp.arange(nl, dtype=real_dtype)[None, :, None, None, None, None]
    hermite = jnp.arange(nm, dtype=real_dtype)[None, None, :, None, None, None]
    contribution = -rate * (2.0 * ell + hermite) * expanded

    temperature = (
        jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype)) * expanded[:, 0, 2]
        + 2.0 * expanded[:, 1, 0]
    ) / 3.0
    spatial_rate = nu_s[(slice(None),) + (None,) * (temperature.ndim - 1)]
    contribution = contribution.at[:, 0, 1].add(spatial_rate * expanded[:, 0, 1])
    contribution = contribution.at[:, 0, 2].add(
        spatial_rate * jnp.sqrt(jnp.asarray(2.0, dtype=real_dtype)) * temperature
    )
    contribution = contribution.at[:, 1, 0].add(2.0 * spatial_rate * temperature)
    result = jnp.asarray(weight, dtype=real_dtype) * contribution
    return result[0] if value.ndim == 5 else result


def heat_flux_channel_species(
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ES, Apar, and Bpar heat-flux channels per species."""

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
    return (
        jnp.sum(es_contrib, axis=(1, 2, 3)),
        jnp.sum(apar_contrib, axis=(1, 2, 3)),
        jnp.sum(bpar_contrib, axis=(1, 2, 3)),
    )


def particle_flux_channel_species(
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ES, Apar, and Bpar particle-flux channels per species."""

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
    )
    return (
        jnp.sum(es_contrib, axis=(1, 2, 3)),
        jnp.sum(apar_contrib, axis=(1, 2, 3)),
        jnp.sum(bpar_contrib, axis=(1, 2, 3)),
    )


def turbulent_heating_total(
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
    """Total turbulent-heating diagnostic."""

    return jnp.sum(
        turbulent_heating_species(
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
    )


def estimate_observed_order(
    step_sizes: np.ndarray, errors: np.ndarray
) -> ObservedOrderMetrics:
    """Estimate observed order from successive step-size refinements."""

    h = np.asarray(step_sizes, dtype=float)
    err = np.asarray(errors, dtype=float)
    if h.ndim != 1 or err.ndim != 1 or h.size != err.size or h.size < 2:
        raise ValueError(
            "step_sizes and errors must be one-dimensional arrays of equal length >= 2"
        )
    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(err)):
        raise ValueError("step_sizes and errors must be finite")
    if np.any(h <= 0.0):
        raise ValueError("step_sizes must be positive")
    if np.any(err <= 0.0):
        raise ValueError("errors must be positive")

    orders: list[float] = []
    for i in range(h.size - 1):
        if np.isclose(h[i], h[i + 1]):
            raise ValueError("successive step sizes must differ")
        orders.append(float(np.log(err[i] / err[i + 1]) / np.log(h[i] / h[i + 1])))
    orders_arr = np.asarray(orders, dtype=float)
    return ObservedOrderMetrics(
        step_sizes=h,
        errors=err,
        orders=orders_arr,
        asymptotic_order=float(orders_arr[-1]),
    )


def exb_nonlinear_contribution(
    G: jnp.ndarray,
    *,
    phi: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    kx_grid: jnp.ndarray,
    ky_grid: jnp.ndarray,
    weight: jnp.ndarray,
    compressed_real_fft: bool = True,
    ny_full: int | None = None,
    radial_phase: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return the nonlinear E×B contribution using a pseudospectral bracket."""
    phi = _apply_mask_xy(phi, dealias_mask)
    bracket_hat = _spectral_bracket(
        G,
        phi,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=dealias_mask,
        kxfac=jnp.asarray(1.0),
        radial_phase=radial_phase,
        compressed_real_fft=compressed_real_fft,
        ny_full=ny_full,
    )
    real_dtype = jnp.real(jnp.empty((), dtype=G.dtype)).dtype
    return jnp.asarray(weight, dtype=real_dtype) * bracket_hat


def reality_residual(state: Any, *, ny_full: int | None = None) -> Any:
    """Return ``max|F - H(F)| / max|F|`` for a two-sided array.

    ``H`` rebuilds the array from its own ``ky >= 0`` rows, so the residual is
    zero exactly when the stored negative rows agree with the reality
    condition.  The self-conjugate rows are included: their internal constraint
    is part of ``H``.
    """

    xp = _xp(state)
    rows = int(state.shape[KY_AXIS])
    ny = rows if ny_full is None else int(ny_full)
    if rows != ny:
        raise ValueError(
            f"reality_residual needs the full ky axis; got {rows} rows for ny_full={ny}"
        )
    rebuilt = to_full(
        symmetrize_self_conjugate_rows(to_half(state, ny_full=ny), ny_full=ny),
        ny_full=ny,
    )
    scale = xp.max(xp.abs(state))
    denominator = xp.where(scale > 0, scale, xp.ones_like(scale))
    return xp.max(xp.abs(state - rebuilt)) / denominator


def single_precision_factorial(m: jnp.ndarray) -> jnp.ndarray:
    """Return the single-precision factorial approximation."""

    m_arr = jnp.asarray(m)
    dtype = m_arr.dtype
    exact = jnp.asarray([1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0], dtype=dtype)
    m_int = m_arr.astype(jnp.int32)
    m_clamped = jnp.clip(m_int, 0, exact.shape[0] - 1)
    m_safe = jnp.where(m_arr > 0, m_arr, jnp.asarray(1.0, dtype=dtype))
    stirling = (
        jnp.sqrt(2.0 * jnp.asarray(jnp.pi, dtype=dtype) * m_safe)
        * (m_safe**m_safe)
        * jnp.exp(-m_safe)
        * (1.0 + 1.0 / (12.0 * m_safe) + 1.0 / (288.0 * m_safe * m_safe))
    )
    return jnp.where(m_int <= 6, exact[m_clamped], stirling)


def _spectral_bracket_multi(
    G_hat: jnp.ndarray,
    chi_hat_stack: jnp.ndarray,
    *,
    compressed_real_fft: bool = True,
    **kwargs,
) -> jnp.ndarray:
    kernel = (
        _spectral_bracket_multi_real_fft
        if compressed_real_fft
        else _spectral_bracket_multi_full
    )
    return kernel(G_hat, chi_hat_stack, **kwargs)
