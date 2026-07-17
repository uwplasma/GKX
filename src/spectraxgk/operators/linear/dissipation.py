"""Linear collisional, hypercollisional, and damping term contributions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

from spectraxgk.operators.collision import CollisionContext, CollisionOperator
from spectraxgk.terms.config import FieldState
from spectraxgk.terms.config import TermConfig
from spectraxgk.operators.linear.streaming import abs_z_linked_fft, shift_axis


@dataclass(frozen=True)
class _HypercollisionCoefficients:
    vth: jnp.ndarray
    nu_hyper: jnp.ndarray
    nu_hyper_l: jnp.ndarray
    nu_hyper_m: jnp.ndarray
    nu_hyper_lm: jnp.ndarray
    hyper_ratio: jnp.ndarray
    ratio_l: jnp.ndarray
    ratio_m: jnp.ndarray
    ratio_lm: jnp.ndarray
    hypercollisions_const: jnp.ndarray
    hypercollisions_kz: jnp.ndarray


@dataclass(frozen=True)
class _HypercollisionMasks:
    mask_const: jnp.ndarray
    mask_kz: jnp.ndarray
    m_pow: jnp.ndarray
    m_norm_kz_factor: jnp.ndarray
    kpar_scale: jnp.ndarray


@dataclass(frozen=True)
class _HypercollisionLinkedRoute:
    kz: jnp.ndarray
    linked_indices: tuple[jnp.ndarray, ...] | None
    linked_kz: tuple[jnp.ndarray, ...] | None
    linked_inverse_permutation: jnp.ndarray | None
    linked_full_cover: bool
    linked_gather_map: jnp.ndarray | None
    linked_gather_mask: jnp.ndarray | None
    linked_use_gather: bool


class CollisionInvariantRates(NamedTuple):
    """Long-wavelength collisional rates of the conserved fluid moments."""

    density: jnp.ndarray
    parallel_momentum: jnp.ndarray
    thermal_energy: jnp.ndarray


class MultispeciesCollisionRates(NamedTuple):
    """Physical rates used to gate a species-coupled collision model."""

    particle_density: jnp.ndarray
    total_parallel_momentum: jnp.ndarray
    total_thermal_energy: jnp.ndarray


def _is_static_zero(value: Any, dtype: jnp.dtype | None = None) -> bool:
    arr = jnp.asarray(value, dtype=dtype)
    if isinstance(arr, jax.core.Tracer):
        return False
    return bool(np.all(np.asarray(arr) == 0.0))


def _zeros_like_result(x: jnp.ndarray, *values: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros_like(x, dtype=jnp.result_type(x, *values))


def terms_without_builtin_collisions(
    terms: TermConfig,
    operator: CollisionOperator | None,
) -> TermConfig:
    """Disable built-in collisions when a custom operator owns the term."""

    if operator is None or _is_static_zero(terms.collisions):
        return terms
    return replace(terms, collisions=0.0)


def custom_collision_contribution(
    state: jnp.ndarray,
    fields: FieldState,
    cache: Any,
    parameters: Any,
    terms: TermConfig,
    operator: CollisionOperator | None,
    *,
    force_electrostatic_fields: bool = False,
) -> jnp.ndarray | None:
    """Evaluate a custom operator with the post-field Hamiltonian response."""

    if operator is None or _is_static_zero(terms.collisions):
        return None
    # Deferred to keep this low-level term module importable before operator facades.
    from spectraxgk.operators.linear.moments import build_H

    apar = (
        None
        if force_electrostatic_fields
        or fields.apar is None
        or _is_static_zero(terms.apar)
        else fields.apar
    )
    bpar = (
        None
        if force_electrostatic_fields
        or fields.bpar is None
        or _is_static_zero(terms.bpar)
        else fields.bpar
    )
    hamiltonian = build_H(
        state,
        cache.Jl,
        fields.phi,
        jnp.asarray(parameters.tz),
        apar=apar,
        vth=jnp.asarray(parameters.vth),
        bpar=bpar,
        JlB=cache.JlB,
    )
    context = CollisionContext(state, hamiltonian, fields, cache, parameters)
    contribution = jnp.asarray(operator.apply(context))
    if contribution.shape != state.shape:
        raise ValueError(
            "collision operator must return the same state shape "
            f"(expected {state.shape}, got {contribution.shape})"
        )
    real_dtype = jnp.real(jnp.empty((), dtype=state.dtype)).dtype
    return jnp.asarray(terms.collisions, dtype=real_dtype) * contribution


def _hermite_mode_drive(
    template: jnp.ndarray,
    mode: int,
    drive: jnp.ndarray,
) -> jnp.ndarray:
    """Embed a single-Hermite-mode drive into a full state-shaped array."""

    mask = jnp.arange(template.shape[2], dtype=jnp.int32)[
        None, None, :, None, None, None
    ]
    return (mask == int(mode)).astype(template.dtype) * drive[:, :, None, ...]


def _species_collision_frequency(
    nu: jnp.ndarray,
    *,
    ns: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Return one collision frequency per species."""

    nu_arr = jnp.asarray(nu, dtype=dtype).reshape(-1)
    if nu_arr.size == 1:
        return jnp.broadcast_to(nu_arr, (ns,))
    if int(nu_arr.size) != int(ns):
        raise ValueError(f"nu must have length {ns} (got {nu_arr.size})")
    return nu_arr


def _collision_magnetic_shift(
    b: jnp.ndarray | None,
    *,
    H_ndim: int,
    ns: int,
    dtype: jnp.dtype,
) -> jnp.ndarray | None:
    if b is None:
        return None
    b_s = jnp.asarray(b, dtype=dtype)
    if H_ndim == 6:
        if b_s.ndim == 3:
            b_s = jnp.broadcast_to(b_s, (ns,) + b_s.shape)
        return b_s[:, None, None, ...]
    if b_s.ndim == 4:
        b_s = b_s[0]
    return b_s[None, None, ...]


def _laguerre_collision_base(
    H: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    lb_lam: jnp.ndarray,
    b: jnp.ndarray | None,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    lb_arr = jnp.asarray(lb_lam, dtype=dtype)
    if lb_arr.ndim == 2 and H.ndim == 6:
        ns = int(H.shape[0])
        nu_s = _species_collision_frequency(nu, ns=ns, dtype=dtype)[
            :, None, None, None, None, None
        ]
        base = nu_s * lb_arr[None, :, :, None, None, None]
        b_shift = _collision_magnetic_shift(b, H_ndim=H.ndim, ns=ns, dtype=dtype)
        return base if b_shift is None else base + nu_s * b_shift
    if lb_arr.ndim == 2:
        nu0 = _species_collision_frequency(nu, ns=1, dtype=dtype)[0]
        base = nu0 * lb_arr[:, :, None, None, None]
        b_shift = _collision_magnetic_shift(b, H_ndim=H.ndim, ns=1, dtype=dtype)
        return base if b_shift is None else base + nu0 * b_shift
    if lb_arr.ndim == 6:
        ns = int(lb_arr.shape[0])
        base = (
            _species_collision_frequency(nu, ns=ns, dtype=dtype)[
                :, None, None, None, None, None
            ]
            * lb_arr
        )
        return base[0] if H.ndim == 5 else base
    return jnp.asarray(nu, dtype=dtype) * lb_arr


def _collision_base_operator(
    H: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    collision_lam: jnp.ndarray | None,
    lb_lam: jnp.ndarray | None,
    b: jnp.ndarray | None,
    dtype: jnp.dtype,
) -> jnp.ndarray | None:
    if collision_lam is not None:
        collision_arr = jnp.asarray(collision_lam, dtype=dtype)
        if collision_arr.size != 0:
            return collision_arr
    if _is_static_zero(nu, dtype):
        return None
    if lb_lam is None:
        return jnp.zeros_like(H, dtype=dtype)
    return _laguerre_collision_base(H, nu=nu, lb_lam=lb_lam, b=b, dtype=dtype)


def _laguerre_temperature_coupling(
    H_m0: jnp.ndarray,
    G_m2: jnp.ndarray,
    Jl: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    Jl_m1 = shift_axis(Jl, -1, axis=1)
    Jl_p1 = shift_axis(Jl, 1, axis=1)
    coeff_t = jnp.arange(Jl.shape[1], dtype=jnp.real(H_m0).dtype)[
        None, :, None, None, None
    ]
    coeff_t = coeff_t * Jl_m1 + 2.0 * coeff_t * Jl + (coeff_t + 1.0) * Jl_p1
    if int(Jl.shape[1]) == 1:
        t_bar = jnp.sqrt(2.0) * jnp.sum(Jl * G_m2, axis=1)
    else:
        t_bar = (jnp.sqrt(2.0) / 3.0) * jnp.sum(Jl * G_m2, axis=1) + (
            2.0 / 3.0
        ) * jnp.sum(coeff_t * H_m0, axis=1)
    return coeff_t, t_bar


def _collision_moment_correction(
    H: jnp.ndarray,
    *,
    G: jnp.ndarray,
    Jl: jnp.ndarray,
    JlB: jnp.ndarray,
    b: jnp.ndarray,
    nu: jnp.ndarray,
) -> jnp.ndarray:
    nu_s = nu[:, None, None, None, None]
    b_s = jnp.asarray(b, dtype=jnp.real(H).dtype)
    sqrt_b = jnp.sqrt(jnp.maximum(b_s, 0.0))
    H_m0 = H[:, :, 0, ...]
    Nm = H.shape[2]
    H_m1 = H[:, :, 1, ...] if Nm > 1 else jnp.zeros_like(H_m0)
    G_m2 = G[:, :, 2, ...] if Nm > 2 else jnp.zeros_like(H_m0)

    coeff_t, t_bar = _laguerre_temperature_coupling(H_m0, G_m2, Jl)
    uperp_bar = sqrt_b * jnp.sum(JlB * H_m0, axis=1)
    upar_bar = jnp.sum(Jl * H_m1, axis=1)

    corr = jnp.zeros_like(H)
    corr_m0 = (
        nu_s * sqrt_b[:, None, ...] * JlB * uperp_bar[:, None, ...]
        + nu_s * 2.0 * coeff_t * t_bar[:, None, ...]
    )
    corr = corr + _hermite_mode_drive(corr, 0, corr_m0)
    if Nm > 1:
        corr = corr + _hermite_mode_drive(corr, 1, nu_s * Jl * upar_bar[:, None, ...])
    if Nm > 2:
        corr = corr + _hermite_mode_drive(
            corr, 2, nu_s * jnp.sqrt(2.0) * Jl * t_bar[:, None, ...]
        )
    return corr


def drift_kinetic_dougherty_contribution(
    state: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    weight: jnp.ndarray = jnp.asarray(1.0),
) -> jnp.ndarray:
    """Apply the linearized drift-kinetic Dougherty moment operator.

    This is Appendix C, equation (C6), of Frei, Hoffmann & Ricci (2022),
    mapped to SPECTRAX-GK's ``(species, ell, m, ky, kx, z)`` ordering and
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


def _drift_kinetic_six_moment_contribution(
    state: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    weight: jnp.ndarray,
    thermal_coefficients: tuple[float, float, float],
    heat_coefficients: tuple[float, float, float],
) -> jnp.ndarray:
    value = jnp.asarray(state)
    if value.ndim not in {5, 6}:
        raise ValueError("collision state must have five or six dimensions")
    expanded = value[None, ...] if value.ndim == 5 else value
    if expanded.shape[1] < 2 or expanded.shape[2] < 4:
        raise ValueError("six-moment collision model requires Nl >= 2 and Nm >= 4")

    ns = int(expanded.shape[0])
    real_dtype = jnp.real(expanded).dtype
    rate = _species_collision_frequency(nu, ns=ns, dtype=real_dtype)
    rate = rate[(slice(None),) + (None,) * (expanded.ndim - 1)]
    inverse_sqrt_pi = 1.0 / jnp.sqrt(jnp.asarray(jnp.pi, dtype=real_dtype))
    thermal = jnp.asarray(thermal_coefficients, dtype=real_dtype) * inverse_sqrt_pi
    heat = jnp.asarray(heat_coefficients, dtype=real_dtype) * inverse_sqrt_pi

    result = jnp.zeros_like(expanded)
    result = result.at[:, 0, 2].set(
        thermal[0] * expanded[:, 0, 2] + thermal[1] * expanded[:, 1, 0]
    )
    result = result.at[:, 1, 0].set(
        thermal[1] * expanded[:, 0, 2] + thermal[2] * expanded[:, 1, 0]
    )
    result = result.at[:, 0, 3].set(
        heat[0] * expanded[:, 0, 3] + heat[1] * expanded[:, 1, 1]
    )
    result = result.at[:, 1, 1].set(
        heat[1] * expanded[:, 0, 3] + heat[2] * expanded[:, 1, 1]
    )
    result = jnp.asarray(weight, dtype=real_dtype) * rate * result
    return result[0] if value.ndim == 5 else result


def drift_kinetic_sugama_six_moment_contribution(
    state: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    weight: jnp.ndarray = jnp.asarray(1.0),
) -> jnp.ndarray:
    """Apply Frei, Ernst & Ricci (2022), equations (C6a)--(C6f)."""

    sqrt_two = 2.0**0.5
    return _drift_kinetic_six_moment_contribution(
        state,
        nu=nu,
        weight=weight,
        thermal_coefficients=(-64 * sqrt_two / 45, 64 / 45, -32 * sqrt_two / 45),
        heat_coefficients=(
            -361.0 * sqrt_two / 175.0,
            208.0 / (175.0 * 3.0**0.5),
            -1187.0 * sqrt_two / 525.0,
        ),
    )


def drift_kinetic_coulomb_six_moment_contribution(
    state: jnp.ndarray,
    *,
    nu: jnp.ndarray,
    weight: jnp.ndarray = jnp.asarray(1.0),
) -> jnp.ndarray:
    """Apply Frei, Ernst & Ricci (2022), equations (C9a)--(C9f)."""

    sqrt_two = 2.0**0.5
    return _drift_kinetic_six_moment_contribution(
        state,
        nu=nu,
        weight=weight,
        thermal_coefficients=(-16 * sqrt_two / 15, 16 / 15, -8 * sqrt_two / 15),
        heat_coefficients=(
            -8.0 * sqrt_two / 5.0,
            8.0 / (5.0 * 3.0**0.5),
            -28.0 * sqrt_two / 15.0,
        ),
    )


def collisions_contribution(
    H: jnp.ndarray,
    *,
    G: jnp.ndarray | None = None,
    Jl: jnp.ndarray | None = None,
    JlB: jnp.ndarray | None = None,
    b: jnp.ndarray | None = None,
    nu: jnp.ndarray,
    collision_lam: jnp.ndarray | None = None,
    lb_lam: jnp.ndarray | None = None,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    real_dtype = jnp.real(H).dtype
    if _is_static_zero(weight, real_dtype):
        return jnp.zeros_like(H)
    collision_base = _collision_base_operator(
        H,
        nu=nu,
        collision_lam=collision_lam,
        lb_lam=lb_lam,
        b=b,
        dtype=real_dtype,
    )
    if collision_base is None:
        return jnp.zeros_like(H)

    base = -(H * collision_base) * weight
    if G is None or Jl is None or JlB is None or b is None:
        return base

    corr = _collision_moment_correction(H, G=G, Jl=Jl, JlB=JlB, b=b, nu=nu)
    return base + weight * corr


def collision_invariant_rates(contribution: jnp.ndarray) -> CollisionInvariantRates:
    """Return density, parallel-momentum, and thermal-energy collision rates.

    The Hermite--Laguerre state must have ``(ell, m)`` axes first, optionally
    preceded by species.  These are the discrete long-wavelength invariants;
    finite-Larmor-radius operators require their full gyroaveraged moments.
    """

    value = jnp.asarray(contribution)
    if value.ndim not in {5, 6}:
        raise ValueError("collision state must have five or six dimensions")
    state = value[None, ...] if value.ndim == 5 else value
    if state.shape[1] < 2 or state.shape[2] < 3:
        raise ValueError("collision invariant rates require Nl >= 2 and Nm >= 3")
    return CollisionInvariantRates(
        density=state[:, 0, 0, ...],
        parallel_momentum=state[:, 0, 1, ...],
        thermal_energy=(
            jnp.sqrt(jnp.asarray(2.0, dtype=jnp.real(state).dtype))
            * state[:, 0, 2, ...]
            + 2.0 * state[:, 1, 0, ...]
        ),
    )


def multispecies_collision_invariant_rates(
    contribution: jnp.ndarray,
    *,
    density: jnp.ndarray,
    mass: jnp.ndarray,
    temperature: jnp.ndarray,
) -> MultispeciesCollisionRates:
    """Return particle, total-momentum, and total-energy collision rates.

    The Hermite--Laguerre coefficients are normalized separately for each
    species.  Particle density therefore carries a factor ``n_s``, parallel
    momentum carries ``n_s sqrt(m_s T_s)``, and thermal energy carries
    ``n_s T_s``.  A multispecies collision model must conserve particle number
    for every species and the sums of the latter two rates.
    """

    rates = collision_invariant_rates(contribution)
    ns = int(rates.density.shape[0])
    real_dtype = jnp.real(rates.density).dtype

    density_s = _species_vector(density, "density", ns=ns, dtype=real_dtype)
    mass_s = _species_vector(mass, "mass", ns=ns, dtype=real_dtype)
    temperature_s = _species_vector(temperature, "temperature", ns=ns, dtype=real_dtype)
    spatial_axes = (None,) * (rates.density.ndim - 1)
    particle_weight = density_s[(slice(None),) + spatial_axes]
    momentum_weight = (density_s * jnp.sqrt(mass_s * temperature_s))[
        (slice(None),) + spatial_axes
    ]
    energy_weight = (density_s * temperature_s)[(slice(None),) + spatial_axes]
    return MultispeciesCollisionRates(
        particle_density=particle_weight * rates.density,
        total_parallel_momentum=jnp.sum(
            momentum_weight * rates.parallel_momentum, axis=0
        ),
        total_thermal_energy=jnp.sum(energy_weight * rates.thermal_energy, axis=0),
    )


def _species_vector(
    value: jnp.ndarray, name: str, *, ns: int, dtype: jnp.dtype
) -> jnp.ndarray:
    vector = jnp.asarray(value, dtype=dtype).reshape(-1)
    if int(vector.size) != ns:
        raise ValueError(f"{name} must have length {ns} (got {vector.size})")
    return vector


def collision_quadratic_rate(
    state: jnp.ndarray,
    contribution: jnp.ndarray,
    *,
    weights: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return ``Re <state, C[state]>`` in the discrete moment norm.

    A dissipative collision model has a non-positive rate.  Optional
    broadcastable ``weights`` can supply species and spatial quadrature factors.
    """

    state_arr = jnp.asarray(state)
    contribution_arr = jnp.asarray(contribution)
    if state_arr.shape != contribution_arr.shape:
        raise ValueError("state and collision contribution must have the same shape")
    product = jnp.real(jnp.conj(state_arr) * contribution_arr)
    if weights is not None:
        product = product * jnp.asarray(weights, dtype=product.dtype)
    return jnp.sum(product)


def _hypercollision_operator_is_static_zero(
    *,
    weight: jnp.ndarray,
    nu_hyper: jnp.ndarray,
    nu_hyper_l: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    nu_hyper_lm: jnp.ndarray,
    hypercollisions_const: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
    dtype: jnp.dtype,
) -> bool:
    const_branch_zero = (
        _is_static_zero(weight * hypercollisions_const * nu_hyper_l, dtype)
        and _is_static_zero(weight * hypercollisions_const * nu_hyper_m, dtype)
        and _is_static_zero(weight * hypercollisions_const * nu_hyper_lm, dtype)
    )
    isotropic_branch_zero = _is_static_zero(weight * nu_hyper, dtype)
    kz_branch_zero = _is_static_zero(weight * hypercollisions_kz * nu_hyper_m, dtype)
    return const_branch_zero and isotropic_branch_zero and kz_branch_zero


def _constant_hypercollision_contribution(
    G: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    nu_hyper: jnp.ndarray,
    nu_hyper_l: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    nu_hyper_lm: jnp.ndarray,
    hyper_ratio: jnp.ndarray,
    ratio_l: jnp.ndarray,
    ratio_m: jnp.ndarray,
    ratio_lm: jnp.ndarray,
    mask_const: jnp.ndarray,
    hypercollisions_const: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    l_norm = jnp.asarray(max(G.shape[1], 1), dtype=ratio_l.dtype)
    m_norm = jnp.asarray(max(G.shape[2], 1), dtype=ratio_m.dtype)
    scaled_nu_l = l_norm * nu_hyper_l
    scaled_nu_m = m_norm * nu_hyper_m
    vth_s = vth[:, None, None, None, None, None]
    const_term = -(
        vth_s * (scaled_nu_l * ratio_l + scaled_nu_m * ratio_m) + nu_hyper_lm * ratio_lm
    )
    dG = weight * hypercollisions_const * jnp.where(mask_const, const_term, 0.0) * G
    return dG - weight * nu_hyper * hyper_ratio * G


def _hypercollision_kz_source(
    G: jnp.ndarray,
    *,
    weight: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    m_norm_kz_factor: jnp.ndarray,
    vth: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    mask_kz: jnp.ndarray,
    m_pow: jnp.ndarray,
) -> jnp.ndarray:
    """Return the pre-``|k_z|`` source used by hypercollisional damping."""

    vth_s = vth[:, None, None, None, None, None]
    kz_weight = jnp.asarray(weight) * jnp.asarray(hypercollisions_kz)
    nu_hyp_m = nu_hyper_m * m_norm_kz_factor * 2.3 * vth_s * jnp.abs(kpar_scale)
    return kz_weight * jnp.where(mask_kz, -nu_hyp_m * m_pow, 0.0) * G


def _apply_parallel_hypercollision(
    kz_source: jnp.ndarray,
    *,
    kz: jnp.ndarray,
    linked_indices: tuple[jnp.ndarray, ...] | None,
    linked_kz: tuple[jnp.ndarray, ...] | None,
    linked_inverse_permutation: jnp.ndarray | None,
    linked_full_cover: bool,
    linked_gather_map: jnp.ndarray | None,
    linked_gather_mask: jnp.ndarray | None,
    linked_use_gather: bool,
) -> jnp.ndarray:
    if linked_indices and linked_kz:
        return abs_z_linked_fft(
            kz_source,
            linked_indices=linked_indices,
            linked_kz=linked_kz,
            linked_inverse_permutation=linked_inverse_permutation,
            linked_full_cover=linked_full_cover,
            linked_gather_map=linked_gather_map,
            linked_gather_mask=linked_gather_mask,
            linked_use_gather=linked_use_gather,
        )
    abs_kz = jnp.abs(kz)[None, None, None, None, None, :]
    return abs_kz * kz_source


def _inactive_hypercollision_result(
    G: jnp.ndarray,
    *,
    weight: jnp.ndarray,
    nu_hyper: jnp.ndarray,
    nu_hyper_l: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    nu_hyper_lm: jnp.ndarray,
    hypercollisions_const: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
    dtype: jnp.dtype,
) -> jnp.ndarray | None:
    """Return a zero result when all hypercollision branches are statically off."""

    if _is_static_zero(weight, dtype) or _hypercollision_operator_is_static_zero(
        weight=weight,
        nu_hyper=nu_hyper,
        nu_hyper_l=nu_hyper_l,
        nu_hyper_m=nu_hyper_m,
        nu_hyper_lm=nu_hyper_lm,
        hypercollisions_const=hypercollisions_const,
        hypercollisions_kz=hypercollisions_kz,
        dtype=dtype,
    ):
        return _zeros_like_result(
            G,
            weight,
            nu_hyper,
            nu_hyper_l,
            nu_hyper_m,
            nu_hyper_lm,
            hypercollisions_const,
            hypercollisions_kz,
        )
    return None


def _parallel_hypercollision_contribution(
    G: jnp.ndarray,
    *,
    weight: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    m_norm_kz_factor: jnp.ndarray,
    vth: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    mask_kz: jnp.ndarray,
    m_pow: jnp.ndarray,
    kz: jnp.ndarray,
    linked_indices: tuple[jnp.ndarray, ...] | None,
    linked_kz: tuple[jnp.ndarray, ...] | None,
    linked_inverse_permutation: jnp.ndarray | None,
    linked_full_cover: bool,
    linked_gather_map: jnp.ndarray | None,
    linked_gather_mask: jnp.ndarray | None,
    linked_use_gather: bool,
) -> jnp.ndarray:
    """Compute the ``|k_z|`` hypercollision branch, including linked tubes."""

    kz_source = _hypercollision_kz_source(
        G,
        weight=weight,
        hypercollisions_kz=hypercollisions_kz,
        nu_hyper_m=nu_hyper_m,
        m_norm_kz_factor=m_norm_kz_factor,
        vth=vth,
        kpar_scale=kpar_scale,
        mask_kz=mask_kz,
        m_pow=m_pow,
    )
    return _apply_parallel_hypercollision(
        kz_source,
        kz=kz,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_inverse_permutation=linked_inverse_permutation,
        linked_full_cover=linked_full_cover,
        linked_gather_map=linked_gather_map,
        linked_gather_mask=linked_gather_mask,
        linked_use_gather=linked_use_gather,
    )


def hypercollisions_contribution(
    G: jnp.ndarray,
    *,
    vth: jnp.ndarray,
    nu_hyper: jnp.ndarray,
    nu_hyper_l: jnp.ndarray,
    nu_hyper_m: jnp.ndarray,
    nu_hyper_lm: jnp.ndarray,
    hyper_ratio: jnp.ndarray,
    ratio_l: jnp.ndarray,
    ratio_m: jnp.ndarray,
    ratio_lm: jnp.ndarray,
    mask_const: jnp.ndarray,
    mask_kz: jnp.ndarray,
    m_pow: jnp.ndarray,
    m_norm_kz_factor: jnp.ndarray,
    kz: jnp.ndarray,
    kpar_scale: jnp.ndarray,
    hypercollisions_const: jnp.ndarray,
    hypercollisions_kz: jnp.ndarray,
    weight: jnp.ndarray,
    linked_indices: tuple[jnp.ndarray, ...] | None = None,
    linked_kz: tuple[jnp.ndarray, ...] | None = None,
    linked_inverse_permutation: jnp.ndarray | None = None,
    linked_full_cover: bool = False,
    linked_gather_map: jnp.ndarray | None = None,
    linked_gather_mask: jnp.ndarray | None = None,
    linked_use_gather: bool = False,
) -> jnp.ndarray:
    coeffs = _HypercollisionCoefficients(
        vth=vth,
        nu_hyper=nu_hyper,
        nu_hyper_l=nu_hyper_l,
        nu_hyper_m=nu_hyper_m,
        nu_hyper_lm=nu_hyper_lm,
        hyper_ratio=hyper_ratio,
        ratio_l=ratio_l,
        ratio_m=ratio_m,
        ratio_lm=ratio_lm,
        hypercollisions_const=hypercollisions_const,
        hypercollisions_kz=hypercollisions_kz,
    )
    masks = _HypercollisionMasks(
        mask_const, mask_kz, m_pow, m_norm_kz_factor, kpar_scale
    )
    route = _HypercollisionLinkedRoute(
        kz=kz,
        linked_indices=linked_indices,
        linked_kz=linked_kz,
        linked_inverse_permutation=linked_inverse_permutation,
        linked_full_cover=linked_full_cover,
        linked_gather_map=linked_gather_map,
        linked_gather_mask=linked_gather_mask,
        linked_use_gather=linked_use_gather,
    )
    real_dtype = jnp.real(G).dtype
    inactive_result = _inactive_hypercollision_result(
        G,
        weight=weight,
        nu_hyper=coeffs.nu_hyper,
        nu_hyper_l=coeffs.nu_hyper_l,
        nu_hyper_m=coeffs.nu_hyper_m,
        nu_hyper_lm=coeffs.nu_hyper_lm,
        hypercollisions_const=coeffs.hypercollisions_const,
        hypercollisions_kz=coeffs.hypercollisions_kz,
        dtype=real_dtype,
    )
    if inactive_result is not None:
        return inactive_result

    dG = _constant_hypercollision_contribution(
        G,
        vth=coeffs.vth,
        nu_hyper=coeffs.nu_hyper,
        nu_hyper_l=coeffs.nu_hyper_l,
        nu_hyper_m=coeffs.nu_hyper_m,
        nu_hyper_lm=coeffs.nu_hyper_lm,
        hyper_ratio=coeffs.hyper_ratio,
        ratio_l=coeffs.ratio_l,
        ratio_m=coeffs.ratio_m,
        ratio_lm=coeffs.ratio_lm,
        mask_const=masks.mask_const,
        hypercollisions_const=coeffs.hypercollisions_const,
        weight=weight,
    )
    if _is_static_zero(weight * coeffs.hypercollisions_kz, real_dtype):
        return dG
    return dG + _parallel_hypercollision_contribution(
        G,
        weight=weight,
        hypercollisions_kz=coeffs.hypercollisions_kz,
        nu_hyper_m=coeffs.nu_hyper_m,
        m_norm_kz_factor=masks.m_norm_kz_factor,
        vth=coeffs.vth,
        kpar_scale=masks.kpar_scale,
        mask_kz=masks.mask_kz,
        m_pow=masks.m_pow,
        kz=route.kz,
        linked_indices=route.linked_indices,
        linked_kz=route.linked_kz,
        linked_inverse_permutation=route.linked_inverse_permutation,
        linked_full_cover=route.linked_full_cover,
        linked_gather_map=route.linked_gather_map,
        linked_gather_mask=route.linked_gather_mask,
        linked_use_gather=route.linked_use_gather,
    )


def hyperdiffusion_contribution(
    G: jnp.ndarray,
    *,
    kx: jnp.ndarray,
    ky: jnp.ndarray,
    dealias_mask: jnp.ndarray,
    D_hyper: jnp.ndarray,
    p_hyper_kperp: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    """Hyperdiffusion in k_perp following Laguerre-Hermite conventions."""

    real_dtype = jnp.real(G).dtype
    if _is_static_zero(weight, real_dtype) or _is_static_zero(D_hyper, real_dtype):
        return _zeros_like_result(G, weight, D_hyper)

    kx2 = kx * kx
    ky2 = ky * ky
    if kx2.ndim == 1:
        kperp2 = ky2[:, None] + kx2[None, :]
    elif kx2.ndim == 2 and tuple(kx2.shape) == (ky.size, dealias_mask.shape[1]):
        kperp2 = ky2[:, None] + kx2
    else:
        raise ValueError("kx must have shape (kx,) or (ky, kx)")

    nx = int(dealias_mask.shape[1])
    ny = ky.size
    kx_idx = max((nx - 1) // 3, 0)
    ky_idx = max((ny - 1) // 3, 0)
    kx2_max = kx2[kx_idx] if kx2.ndim == 1 else kx2[ky_idx, kx_idx]
    kperp2_max = kx2_max + ky2[ky_idx]
    kperp2_max = jnp.where(kperp2_max > 0.0, kperp2_max, 1.0)

    Dfac = D_hyper * (kperp2 / kperp2_max) ** p_hyper_kperp
    mask = dealias_mask.astype(Dfac.dtype)
    Dfac = Dfac * mask
    return -weight * Dfac[None, None, :, :, None] * G


def end_damping_contribution(
    H: jnp.ndarray,
    *,
    ky: jnp.ndarray,
    damp_profile: jnp.ndarray,
    linked_damp_profile: jnp.ndarray | None,
    damp_amp: jnp.ndarray,
    weight: jnp.ndarray,
) -> jnp.ndarray:
    real_dtype = jnp.real(H).dtype
    if _is_static_zero(weight, real_dtype) or _is_static_zero(damp_amp, real_dtype):
        return _zeros_like_result(H, weight, damp_amp)

    if linked_damp_profile is not None and getattr(linked_damp_profile, "size", 0) != 0:
        damp = weight * damp_amp * linked_damp_profile[None, None, None, ...]
        return -(damp * H)
    damp = weight * damp_amp * damp_profile[None, None, None, None, None, :]
    ky_mask = (ky > 0.0).astype(damp.dtype)[None, None, None, :, None, None]
    return -(ky_mask * damp * H)


__all__ = [
    "CollisionInvariantRates",
    "collision_invariant_rates",
    "collision_quadratic_rate",
    "collisions_contribution",
    "drift_kinetic_coulomb_six_moment_contribution",
    "drift_kinetic_dougherty_contribution",
    "drift_kinetic_sugama_six_moment_contribution",
    "end_damping_contribution",
    "hypercollisions_contribution",
    "hyperdiffusion_contribution",
]
