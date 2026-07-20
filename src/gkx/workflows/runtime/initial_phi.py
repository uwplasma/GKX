"""Runtime electrostatic-potential initial-condition inversion."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from gkx.operators.linear.params import LinearParams


@dataclass(frozen=True)
class _PhiMomentAlgebra:
    Jl: np.ndarray
    jacobian: np.ndarray
    charge: np.ndarray
    density: np.ndarray
    denom: np.ndarray
    tau_e: float


def _as_runtime_species_array(
    value: float | jnp.ndarray, nspecies: int, name: str
) -> np.ndarray:
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


def _validate_phi_target_species(
    species_targets: tuple[int, ...],
    *,
    nspecies: int,
) -> None:
    if not species_targets:
        raise ValueError(
            "init_field='phi' requires at least one kinetic target species"
        )
    if min(species_targets) < 0 or max(species_targets) >= nspecies:
        raise ValueError("init_field='phi' target species index is out of range")


def _masked_gauge_phi_seed(
    phi: np.ndarray,
    *,
    cache,
    ky_i: int,
    kx_i: int,
    species_targets: tuple[int, ...],
) -> dict[int, np.ndarray] | None:
    mask0 = np.asarray(cache.mask0, dtype=bool)
    if not np.all(mask0[int(ky_i), int(kx_i), :]):
        return None
    if np.any(np.abs(phi) > 0.0):
        raise ValueError(
            "init_field='phi' cannot initialize the masked ky=0, kx=0 gauge mode"
        )
    return {int(idx): np.zeros_like(phi) for idx in species_targets}


def _phi_moment_algebra(
    *,
    cache,
    params: LinearParams,
    nspecies: int,
    ky_i: int,
    kx_i: int,
) -> _PhiMomentAlgebra:
    Jl = np.asarray(cache.Jl, dtype=float)
    jacobian = np.asarray(cache.jacobian, dtype=float)
    charge = _as_runtime_species_array(params.charge_sign, nspecies, "charge_sign")
    density = _as_runtime_species_array(params.density, nspecies, "density")
    tz = _as_runtime_species_array(params.tz, nspecies, "tz")
    zt = np.where(tz == 0.0, 0.0, 1.0 / tz)
    tau_e_arr = np.asarray(params.tau_e, dtype=float).reshape(-1)
    tau_e = float(tau_e_arr[0]) if tau_e_arr.size else 0.0
    g0_species = np.sum(
        Jl[:, :, int(ky_i), int(kx_i), :] * Jl[:, :, int(ky_i), int(kx_i), :], axis=1
    )
    qneut = np.sum(
        density[:, None] * charge[:, None] * zt[:, None] * (1.0 - g0_species), axis=0
    )
    denom = tau_e + qneut
    if not np.all(np.isfinite(denom)):
        raise ValueError(
            "init_field='phi' produced a non-finite quasineutrality denominator"
        )
    return _PhiMomentAlgebra(
        Jl=Jl,
        jacobian=jacobian,
        charge=charge,
        density=density,
        denom=denom,
        tau_e=tau_e,
    )


def _density_average_for_target_phi(
    phi: np.ndarray,
    *,
    algebra: _PhiMomentAlgebra,
    cache,
    ky_i: int,
    kx_i: int,
) -> np.ndarray:
    nbar = algebra.denom.astype(np.complex64) * phi
    ky_is_zonal = np.isclose(float(np.asarray(cache.ky)[int(ky_i)]), 0.0)
    if algebra.tau_e <= 0.0 or not ky_is_zonal or int(kx_i) <= 0:
        return nbar

    jac_sum = float(np.sum(algebra.jacobian))
    phi_avg = (
        np.sum(algebra.jacobian.astype(np.complex64) * phi) / jac_sum
        if jac_sum != 0.0
        else np.mean(phi)
    )
    return nbar - np.complex64(algebra.tau_e) * phi_avg


def _project_density_average_to_target_species(
    nbar: np.ndarray,
    *,
    algebra: _PhiMomentAlgebra,
    ky_i: int,
    kx_i: int,
    species_targets: tuple[int, ...],
) -> dict[int, np.ndarray]:
    target_indices = np.asarray(species_targets, dtype=int)
    coeff = (
        algebra.density[target_indices, None]
        * algebra.charge[target_indices, None]
        * algebra.Jl[target_indices, 0, int(ky_i), int(kx_i), :]
    )
    coeff_norm = np.sum(coeff * coeff, axis=0)
    tiny = 1.0e-30
    bad = (coeff_norm <= tiny) & (np.abs(nbar) > tiny)
    if np.any(bad):
        raise ValueError(
            "init_field='phi' cannot be represented by the selected density moments"
        )

    seed = np.where(
        coeff_norm[None, :] > tiny,
        (coeff / coeff_norm[None, :]).astype(np.complex64) * nbar[None, :],
        np.complex64(0.0),
    )
    return {
        int(s_idx): np.asarray(seed[i], dtype=np.complex64)
        for i, s_idx in enumerate(target_indices)
    }


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
    :mod:`gkx.terms.fields`.
    """

    phi = np.asarray(phi_target, dtype=np.complex64)
    nspecies = int(np.asarray(cache.Jl).shape[0])
    _validate_phi_target_species(species_targets, nspecies=nspecies)
    masked_seed = _masked_gauge_phi_seed(
        phi,
        cache=cache,
        ky_i=ky_i,
        kx_i=kx_i,
        species_targets=species_targets,
    )
    if masked_seed is not None:
        return masked_seed

    algebra = _phi_moment_algebra(
        cache=cache,
        params=params,
        nspecies=nspecies,
        ky_i=ky_i,
        kx_i=kx_i,
    )
    nbar = _density_average_for_target_phi(
        phi,
        algebra=algebra,
        cache=cache,
        ky_i=ky_i,
        kx_i=kx_i,
    )
    return _project_density_average_to_target_species(
        nbar,
        algebra=algebra,
        ky_i=ky_i,
        kx_i=kx_i,
        species_targets=species_targets,
    )
