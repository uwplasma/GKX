"""Collision and hypercollision split helpers for nonlinear integrations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.cache_arrays import (
    hypercollision_damping,
)
from gkx.operators.linear.params import LinearParams
from gkx.terms.config import TermConfig

__all__ = [
    "NonlinearCollisionSplitPolicy",
    "_apply_collision_split",
    "_collision_damping",
    "build_nonlinear_collision_split_policy",
]


class FullFDoughertyCrossMoments(NamedTuple):
    """Pairwise primitive moments for the nonlinear full-f Dougherty model."""

    parallel_flow: jnp.ndarray
    thermal_speed_sq: jnp.ndarray


def _species_vector(
    value: jnp.ndarray, name: str, *, ns: int, dtype: jnp.dtype
) -> jnp.ndarray:
    vector = jnp.asarray(value, dtype=dtype).reshape(-1)
    if int(vector.size) != ns:
        raise ValueError(f"{name} must have length {ns} (got {vector.size})")
    return vector


def _validate_static_physical_array(
    value: jnp.ndarray, name: str, *, strictly_positive: bool
) -> None:
    if isinstance(value, jax.core.Tracer):
        return
    array = np.asarray(value)
    invalid = array <= 0.0 if strictly_positive else array < 0.0
    if np.any(invalid):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")


def conservative_full_f_dougherty_cross_moments(
    parallel_flow: jnp.ndarray,
    thermal_speed_sq: jnp.ndarray,
    *,
    density: jnp.ndarray,
    mass: jnp.ndarray,
    collision_frequency: jnp.ndarray,
    velocity_dimensions: int = 3,
) -> FullFDoughertyCrossMoments:
    """Return conservative pairwise full-f Dougherty primitive moments.

    This is equations (2.11)--(2.12) of Francisquez et al. (2022), whose
    collision operator evolves a nonlinear full distribution. Inputs use
    species as the leading axis; trailing axes are independent spatial samples.
    ``collision_frequency[s, r]`` is the directed rate for species ``s`` due to
    species ``r``. Diagonal and zero-rate pairs retain their self moments. These
    targets are not a field-particle closure for the linearized delta-f
    gyrokinetic operator.
    """

    flow = jnp.asarray(parallel_flow)
    thermal = jnp.asarray(thermal_speed_sq, dtype=jnp.real(flow).dtype)
    if flow.ndim < 1 or thermal.shape != flow.shape:
        raise ValueError(
            "parallel_flow and thermal_speed_sq must share (species, ...) shape"
        )
    if velocity_dimensions <= 0:
        raise ValueError("velocity_dimensions must be positive")
    ns = int(flow.shape[0])
    real_dtype = jnp.real(flow).dtype
    density_s = _species_vector(density, "density", ns=ns, dtype=real_dtype)
    mass_s = _species_vector(mass, "mass", ns=ns, dtype=real_dtype)
    nu = jnp.asarray(collision_frequency, dtype=real_dtype)
    if nu.shape != (ns, ns):
        raise ValueError(
            f"collision_frequency must have shape ({ns}, {ns}) (got {nu.shape})"
        )
    _validate_static_physical_array(density_s, "density", strictly_positive=True)
    _validate_static_physical_array(mass_s, "mass", strictly_positive=True)
    _validate_static_physical_array(
        thermal, "thermal_speed_sq", strictly_positive=False
    )
    _validate_static_physical_array(nu, "collision_frequency", strictly_positive=False)

    sample_axes = (None,) * (flow.ndim - 1)
    pair_axes = (None,) * (flow.ndim - 1)
    flow_s = flow[:, None, ...]
    flow_r = flow[None, :, ...]
    thermal_s = thermal[:, None, ...]
    thermal_r = thermal[None, :, ...]
    mass_s_pair = mass_s[:, None]
    mass_r_pair = mass_s[None, :]
    density_s_pair = density_s[:, None]
    density_r_pair = density_s[None, :]
    momentum_rate_sr = mass_s_pair * density_s_pair * nu
    momentum_rate_rs = mass_r_pair * density_r_pair * jnp.swapaxes(nu, 0, 1)
    momentum_denominator = momentum_rate_sr + momentum_rate_rs
    number_rate_sr = density_s_pair * nu
    number_rate_rs = density_r_pair * jnp.swapaxes(nu, 0, 1)
    thermal_denominator = (number_rate_sr + number_rate_rs) * mass_s_pair
    active = (~jnp.eye(ns, dtype=bool)) & (momentum_denominator > 0.0)
    active_samples = active[(slice(None), slice(None)) + sample_axes]
    safe_momentum_denominator = jnp.where(active, momentum_denominator, 1.0)
    safe_thermal_denominator = jnp.where(active, thermal_denominator, 1.0)

    flow_target = (
        momentum_rate_sr[(slice(None), slice(None)) + pair_axes] * flow_s
        + momentum_rate_rs[(slice(None), slice(None)) + pair_axes] * flow_r
    ) / safe_momentum_denominator[(slice(None), slice(None)) + pair_axes]
    relative_flow_sq = jnp.real((flow_s - flow_r) * jnp.conj(flow_s - flow_r))
    drift_energy = (momentum_rate_sr * momentum_rate_rs / safe_momentum_denominator)[
        (slice(None), slice(None)) + pair_axes
    ] * (relative_flow_sq / float(velocity_dimensions))
    thermal_numerator = (
        (mass_s_pair * number_rate_sr)[(slice(None), slice(None)) + pair_axes]
        * thermal_s
        + (mass_r_pair * number_rate_rs)[(slice(None), slice(None)) + pair_axes]
        * thermal_r
        + drift_energy
    )
    thermal_target = (
        thermal_numerator
        / safe_thermal_denominator[(slice(None), slice(None)) + pair_axes]
    )
    return FullFDoughertyCrossMoments(
        parallel_flow=jnp.where(active_samples, flow_target, flow_s),
        thermal_speed_sq=jnp.where(active_samples, thermal_target, thermal_s),
    )


@dataclass(frozen=True)
class NonlinearCollisionSplitPolicy:
    """Collision split settings shared by explicit and IMEX diagnostics."""

    active: bool
    rhs_terms: TermConfig
    damping: jnp.ndarray | None


def _collision_damping(
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    real_dtype: jnp.dtype,
    *,
    squeeze_species: bool,
) -> jnp.ndarray:
    """Assemble the diagonal hypercollision damping safe to split.

    The conserving collision operator includes non-diagonal field-particle
    corrections and must remain in the RHS unless an operator supplies its own
    mathematically valid split update.
    """

    hyper_damp = hypercollision_damping(cache, params, real_dtype)
    hyper_w = jnp.asarray(term_cfg.hypercollisions, dtype=real_dtype)
    if squeeze_species and hyper_damp.ndim == 6:
        hyper_damp = hyper_damp[0]
    return (hyper_w * hyper_damp).astype(real_dtype)


def build_nonlinear_collision_split_policy(
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    real_dtype: jnp.dtype,
    *,
    squeeze_species: bool,
    collision_split: bool,
    collision_damping_fn: Callable[..., jnp.ndarray] = _collision_damping,
) -> NonlinearCollisionSplitPolicy:
    """Build collision splitting weights and RHS terms for nonlinear scans."""

    active = bool(collision_split) and float(term_cfg.hypercollisions) != 0.0
    rhs_terms = replace(term_cfg, hypercollisions=0.0) if active else term_cfg
    damping = (
        collision_damping_fn(
            cache, params, term_cfg, real_dtype, squeeze_species=squeeze_species
        )
        if active
        else None
    )
    return NonlinearCollisionSplitPolicy(
        active=active,
        rhs_terms=rhs_terms,
        damping=damping,
    )


def _apply_collision_split(
    G: jnp.ndarray,
    damping: jnp.ndarray,
    dt_local: jnp.ndarray,
    scheme: str,
) -> jnp.ndarray:
    """Apply a diagonal collision/hypercollision split update."""

    scheme_key = scheme.strip().lower()
    if scheme_key in {"implicit", "imex"}:
        return G / (1.0 + dt_local * damping)
    if scheme_key in {"exp", "sts", "rkc", "rkc2"}:
        # For diagonal collision operators the exponential update is exact and
        # behaves like a stabilized explicit (STS/RKC) limit.
        return G * jnp.exp(-dt_local * damping)
    raise ValueError(
        "collision_scheme must be one of {'implicit', 'exp', 'sts', 'rkc'}"
    )
