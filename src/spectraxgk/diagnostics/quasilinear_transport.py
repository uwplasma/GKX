"""Quasilinear transport diagnostics from linear gyrokinetic states.

The routines in this module compute linear heat and particle flux weights
from an eigenstate or late-time linear state.  Saturation rules are kept
explicitly separate from the linear weights so calibration and uncertainty
metadata can be audited case by case.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics import (
    electrostatic_field_energy,
    heat_flux_species,
    particle_flux_species,
    fieldline_quadrature_weights,
)
from spectraxgk.geometry import FluxTubeGeometryLike
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.terms.assembly import compute_fields_cached
from spectraxgk.terms.config import TermConfig


_SUPPORTED_NORMALIZATIONS = {"phi_rms", "phi_midplane", "field_energy"}
_SUPPORTED_RULES = {
    "none",
    "mixing_length",
    "lapillonne_2011",
    "linear_weight",
    "absolute_growth_mixing_length",
    "abs_growth_mixing_length",
}
_SUPPORTED_MODES = {"weights", "saturated"}

__all__ = [
    "QuasilinearTransportResult",
    "compute_quasilinear_from_linear_state",
    "effective_kperp2",
    "mixing_length_amplitude2_jax",
    "normalize_quasilinear_channels",
    "phi_norm2",
    "quasilinear_feature_objective",
    "saturation_amplitude2",
    "saturated_flux_from_linear_weight",
    "shape_aware_power_law_objective",
    "spectral_phi_weights",
]


@dataclass(frozen=True)
class QuasilinearTransportResult:
    """JSON-friendly quasilinear diagnostic payload for one linear mode."""

    ky: float
    gamma: float
    omega: float
    mode: str
    saturation_rule: str
    amplitude_normalization: str
    channels: tuple[str, ...]
    kperp_average: str
    kperp_eff2: float
    phi_norm2: float
    amplitude2: float | None
    heat_flux_weight_species: tuple[float, ...]
    particle_flux_weight_species: tuple[float, ...]
    saturated_heat_flux_species: tuple[float, ...] | None
    saturated_particle_flux_species: tuple[float, ...] | None
    species: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""

        payload = asdict(self)
        payload["channels"] = list(self.channels)
        payload["species"] = list(self.species)
        payload["heat_flux_weight_species"] = list(self.heat_flux_weight_species)
        payload["particle_flux_weight_species"] = list(self.particle_flux_weight_species)
        if self.saturated_heat_flux_species is not None:
            payload["saturated_heat_flux_species"] = list(self.saturated_heat_flux_species)
        if self.saturated_particle_flux_species is not None:
            payload["saturated_particle_flux_species"] = list(self.saturated_particle_flux_species)
        payload["heat_flux_weight_total"] = float(sum(self.heat_flux_weight_species))
        payload["particle_flux_weight_total"] = float(sum(self.particle_flux_weight_species))
        if self.saturated_heat_flux_species is not None:
            payload["saturated_heat_flux_total"] = float(sum(self.saturated_heat_flux_species))
        else:
            payload["saturated_heat_flux_total"] = None
        if self.saturated_particle_flux_species is not None:
            payload["saturated_particle_flux_total"] = float(sum(self.saturated_particle_flux_species))
        else:
            payload["saturated_particle_flux_total"] = None
        return payload


@dataclass(frozen=True)
class _QuasilinearOptions:
    """Validated options for one quasilinear transport diagnostic."""

    mode: str
    channels: tuple[str, ...]
    kperp_average: str


@dataclass(frozen=True)
class _LinearFluxWeights:
    """Amplitude-normalized linear transport weights for all species."""

    kperp_eff2: float
    phi_norm2: float
    heat: np.ndarray
    particle: np.ndarray


def normalize_quasilinear_channels(channels: Iterable[str] | str) -> tuple[str, ...]:
    """Normalize and validate quasilinear field channels."""

    values: tuple[str, ...]
    if isinstance(channels, str):
        values = (channels,)
    else:
        values = tuple(str(ch).strip().lower() for ch in channels)
    values = tuple(dict.fromkeys(ch for ch in values if ch))
    if not values:
        values = ("es",)
    unsupported = [ch for ch in values if ch != "es"]
    if unsupported:
        raise NotImplementedError(
            "Only electrostatic quasilinear flux channels are validated so far; "
            f"unsupported channels: {unsupported}"
        )
    return values


def spectral_phi_weights(
    phi: jnp.ndarray,
    cache: LinearCache,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
) -> jnp.ndarray:
    """Return ``(ky, kx, z)`` weights used for ``|phi|^2`` averages."""

    ky = jnp.asarray(cache.ky)
    has_negative = jnp.any(ky < 0.0)
    fac = jnp.where(has_negative, 1.0, jnp.where(ky == 0.0, 1.0, 2.0))
    fac = fac[:, None] * jnp.ones((1, cache.kx.size), dtype=fac.dtype)
    if use_dealias:
        fac = fac * cache.dealias_mask.astype(fac.dtype)
    return (jnp.abs(phi) ** 2) * fac[:, :, None] * vol_fac[None, None, :]


def effective_kperp2(
    phi: jnp.ndarray,
    cache: LinearCache,
    vol_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    eps: float = 1.0e-30,
) -> jnp.ndarray:
    """Compute ``<k_perp^2 |phi|^2>/<|phi|^2>`` for a linear mode."""

    weights = spectral_phi_weights(phi, cache, vol_fac, use_dealias=use_dealias)
    denom = jnp.sum(weights)
    return jnp.sum(cache.kperp2 * weights) / jnp.maximum(denom, jnp.asarray(eps, dtype=denom.dtype))


def phi_norm2(
    phi: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    *,
    normalization: str = "phi_rms",
    use_dealias: bool = True,
    eps: float = 1.0e-30,
) -> jnp.ndarray:
    """Return the amplitude normalization used for quasilinear weights."""

    norm_key = normalization.strip().lower()
    if norm_key not in _SUPPORTED_NORMALIZATIONS:
        raise ValueError(f"Unknown quasilinear amplitude normalization '{normalization}'")
    if norm_key == "phi_rms":
        return jnp.maximum(
            jnp.sum(spectral_phi_weights(phi, cache, vol_fac, use_dealias=use_dealias)),
            jnp.asarray(eps, dtype=jnp.real(phi).dtype),
        )
    if norm_key == "phi_midplane":
        z_index = int(phi.shape[-1] // 2)
        return jnp.maximum(
            jnp.max(jnp.abs(phi[:, :, z_index]) ** 2),
            jnp.asarray(eps, dtype=jnp.real(phi).dtype),
        )
    return jnp.maximum(
        electrostatic_field_energy(phi, cache, params, vol_fac, use_dealias=use_dealias),
        jnp.asarray(eps, dtype=jnp.real(phi).dtype),
    )


def saturation_amplitude2(
    *,
    gamma: float,
    kperp_eff2_value: float,
    rule: str,
    csat: float = 1.0,
    gamma_floor: float = 0.0,
    include_stable_modes: bool = False,
) -> float | None:
    """Return the squared amplitude implied by a named saturation rule."""

    rule_key = rule.strip().lower()
    if rule_key not in _SUPPORTED_RULES:
        raise NotImplementedError(f"Quasilinear saturation rule '{rule}' is not implemented")
    if rule_key == "none":
        return None
    if kperp_eff2_value <= 0.0 or not np.isfinite(kperp_eff2_value):
        return 0.0
    drive = float(gamma) - float(gamma_floor)
    if not include_stable_modes:
        drive = max(drive, 0.0)
    if rule_key == "linear_weight":
        return float(csat)
    if rule_key in {"absolute_growth_mixing_length", "abs_growth_mixing_length"}:
        return float(csat) * abs(float(gamma)) / float(kperp_eff2_value)
    if rule_key in {"mixing_length", "lapillonne_2011"}:
        return float(csat) * drive / float(kperp_eff2_value)
    return None


def mixing_length_amplitude2_jax(
    gamma: jnp.ndarray | float,
    kperp_eff2_value: jnp.ndarray | float,
    *,
    csat: float = 1.0,
    gamma_floor: float = 0.0,
    include_stable_modes: bool = False,
    eps: float = 1.0e-30,
) -> jnp.ndarray:
    """JAX-differentiable mixing-length squared-amplitude rule."""

    gamma_arr = jnp.asarray(gamma)
    kperp_arr = jnp.asarray(kperp_eff2_value)
    drive = gamma_arr - jnp.asarray(gamma_floor, dtype=gamma_arr.dtype)
    if not include_stable_modes:
        drive = jnp.maximum(drive, jnp.asarray(0.0, dtype=gamma_arr.dtype))
    denom = jnp.maximum(kperp_arr, jnp.asarray(eps, dtype=kperp_arr.dtype))
    return jnp.asarray(csat, dtype=gamma_arr.dtype) * drive / denom


def saturated_flux_from_linear_weight(
    linear_flux_weight: jnp.ndarray | float,
    gamma: jnp.ndarray | float,
    kperp_eff2_value: jnp.ndarray | float,
    *,
    csat: float = 1.0,
    gamma_floor: float = 0.0,
    include_stable_modes: bool = False,
) -> jnp.ndarray:
    """Return a differentiable mixing-length saturated flux estimate."""

    amp2 = mixing_length_amplitude2_jax(
        gamma,
        kperp_eff2_value,
        csat=csat,
        gamma_floor=gamma_floor,
        include_stable_modes=include_stable_modes,
    )
    return jnp.asarray(linear_flux_weight) * amp2


def quasilinear_feature_objective(
    features: jnp.ndarray | Sequence[float],
    *,
    rule: str = "mixing_length",
    csat: float = 1.0,
    gamma_floor: float = 0.0,
    include_stable_modes: bool = False,
) -> jnp.ndarray:
    """Differentiable objective from ``[gamma, kperp_eff2, flux_weight]``.

    This helper is intentionally small: it is the reduced objective used by
    derivative validation tests and optimization examples once a linear scan has
    produced quasilinear weights.
    """

    x = jnp.asarray(features)
    if x.shape[-1] != 3:
        raise ValueError("features must end with [gamma, kperp_eff2, flux_weight]")
    rule_key = rule.strip().lower()
    if rule_key == "linear_weight":
        return jnp.asarray(csat, dtype=x.dtype) * x[..., 2]
    if rule_key in {"absolute_growth_mixing_length", "abs_growth_mixing_length"}:
        denom = jnp.maximum(x[..., 1], jnp.asarray(1.0e-30, dtype=x.dtype))
        return jnp.asarray(csat, dtype=x.dtype) * jnp.abs(x[..., 0]) * x[..., 2] / denom
    if rule_key not in {"mixing_length", "lapillonne_2011"}:
        raise NotImplementedError(f"Quasilinear feature rule '{rule}' is not implemented")
    return saturated_flux_from_linear_weight(
        x[..., 2],
        x[..., 0],
        x[..., 1],
        csat=csat,
        gamma_floor=gamma_floor,
        include_stable_modes=include_stable_modes,
    )


def shape_aware_power_law_objective(
    features: jnp.ndarray | Sequence[float],
    ky: jnp.ndarray | Sequence[float] | float,
    *,
    exponent: jnp.ndarray | float,
    csat: float = 1.0,
    ky_ref: float | None = None,
    eps: float = 1.0e-30,
) -> jnp.ndarray:
    """Differentiable shape-aware linear-weight objective.

    ``features`` must end with ``[gamma, kperp_eff2, flux_weight]``. The
    current low-dimensional shape model intentionally uses only the linear
    heat-flux weight and a power-law envelope in ``ky``:

    ``Q = C_sat * flux_weight * (ky / ky_ref)**exponent``.

    Growth-rate dependence is left to separately validated rules. This helper
    exists so the shape-aware saturation diagnostics and future optimization
    examples use one differentiable objective rather than plotting-only
    formulas.
    """

    dtype = jnp.result_type(features, ky, exponent, jnp.float32)
    x = jnp.asarray(features, dtype=dtype)
    if x.shape[-1] != 3:
        raise ValueError("features must end with [gamma, kperp_eff2, flux_weight]")
    ky_arr = jnp.asarray(ky, dtype=dtype)
    eps_arr = jnp.asarray(eps, dtype=dtype)
    positive_ky = jnp.maximum(ky_arr, eps_arr)
    if ky_ref is None:
        ref = jnp.exp(jnp.mean(jnp.log(positive_ky)))
    else:
        ref = jnp.maximum(jnp.asarray(ky_ref, dtype=dtype), eps_arr)
    envelope = (positive_ky / ref) ** jnp.asarray(exponent, dtype=dtype)
    return jnp.asarray(csat, dtype=dtype) * x[..., 2] * envelope


def _resolve_quasilinear_options(
    *,
    mode: str,
    channels: Sequence[str] | str,
    kperp_average: str,
) -> _QuasilinearOptions:
    """Validate model-selection knobs before computing linear weights."""

    mode_key = mode.strip().lower()
    if mode_key not in _SUPPORTED_MODES:
        raise ValueError(f"Unknown quasilinear mode '{mode}'")
    channels_use = normalize_quasilinear_channels(channels)
    kperp_key = kperp_average.strip().lower()
    if kperp_key != "phi_weighted":
        raise NotImplementedError("Only phi_weighted kperp averaging is validated so far")
    return _QuasilinearOptions(
        mode=mode_key,
        channels=channels_use,
        kperp_average=kperp_key,
    )


def _linear_flux_weights_from_state(
    state: jnp.ndarray | np.ndarray,
    *,
    cache: LinearCache,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    terms: TermConfig | None,
    amplitude_normalization: str,
    use_dealias: bool,
    flux_scale: float,
) -> _LinearFluxWeights:
    """Compute amplitude-invariant linear heat and particle flux weights."""

    G = jnp.asarray(state)
    fields = compute_fields_cached(G, cache, params, terms=terms)
    phi = fields.phi
    zero_field = jnp.zeros_like(phi)
    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)

    norm2 = phi_norm2(
        phi,
        cache,
        params,
        vol_fac,
        normalization=amplitude_normalization,
        use_dealias=use_dealias,
    )
    kperp_eff = effective_kperp2(phi, cache, vol_fac, use_dealias=use_dealias)
    heat = heat_flux_species(
        G,
        phi,
        zero_field,
        zero_field,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    particle = particle_flux_species(
        G,
        phi,
        zero_field,
        zero_field,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return _LinearFluxWeights(
        kperp_eff2=float(np.asarray(kperp_eff)),
        phi_norm2=float(np.asarray(norm2)),
        heat=np.asarray(jnp.real(heat / norm2), dtype=float).reshape(-1),
        particle=np.asarray(jnp.real(particle / norm2), dtype=float).reshape(-1),
    )


def _quasilinear_species_labels(
    species_names: Sequence[str] | None,
    n_species: int,
) -> tuple[str, ...]:
    """Return validated species labels for the diagnostic payload."""

    species = tuple(species_names or tuple(f"s{i}" for i in range(n_species)))
    if len(species) != n_species:
        return tuple(f"s{i}" for i in range(n_species))
    return species


def _saturated_species_fluxes(
    *,
    mode: str,
    heat_weights: np.ndarray,
    particle_weights: np.ndarray,
    amplitude2: float | None,
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    """Apply a selected saturation rule to linear weights when requested."""

    if mode != "saturated" or amplitude2 is None:
        return None, None
    return (
        tuple(float(x) for x in heat_weights * amplitude2),
        tuple(float(x) for x in particle_weights * amplitude2),
    )


def _quasilinear_metadata(
    metadata: dict[str, Any] | None,
    *,
    amplitude2: float | None,
    channels: tuple[str, ...],
) -> dict[str, Any]:
    """Attach explicit claim-scope metadata to a quasilinear payload."""

    meta = dict(metadata or {})
    meta.setdefault(
        "claim_level",
        "linear_weights" if amplitude2 is None else "uncalibrated_saturation_rule",
    )
    meta.setdefault("field_channels_validated", list(channels))
    meta.setdefault("electromagnetic_channels", "disabled_until_validated")
    return meta


def _saturation_outputs(
    *,
    options: _QuasilinearOptions,
    weights: _LinearFluxWeights,
    gamma: float,
    saturation_rule: str,
    csat: float,
    gamma_floor: float,
    include_stable_modes: bool,
) -> tuple[float | None, tuple[float, ...] | None, tuple[float, ...] | None]:
    """Return saturation amplitude and saturated species fluxes."""

    amp2 = saturation_amplitude2(
        gamma=gamma,
        kperp_eff2_value=weights.kperp_eff2,
        rule=saturation_rule,
        csat=csat,
        gamma_floor=gamma_floor,
        include_stable_modes=include_stable_modes,
    )
    saturated_heat, saturated_particle = _saturated_species_fluxes(
        mode=options.mode,
        heat_weights=weights.heat,
        particle_weights=weights.particle,
        amplitude2=amp2,
    )
    return amp2, saturated_heat, saturated_particle


def _build_quasilinear_transport_result(
    *,
    ky: float,
    gamma: float,
    omega: float,
    options: _QuasilinearOptions,
    saturation_rule: str,
    amplitude_normalization: str,
    weights: _LinearFluxWeights,
    species: tuple[str, ...],
    amplitude2: float | None,
    saturated_heat: tuple[float, ...] | None,
    saturated_particle: tuple[float, ...] | None,
    metadata: dict[str, Any] | None,
) -> QuasilinearTransportResult:
    """Pack one validated quasilinear diagnostic payload."""

    return QuasilinearTransportResult(
        ky=float(ky),
        gamma=float(gamma),
        omega=float(omega),
        mode=options.mode,
        saturation_rule=saturation_rule.strip().lower(),
        amplitude_normalization=amplitude_normalization.strip().lower(),
        channels=options.channels,
        kperp_average=options.kperp_average,
        kperp_eff2=weights.kperp_eff2,
        phi_norm2=weights.phi_norm2,
        amplitude2=None if amplitude2 is None else float(amplitude2),
        heat_flux_weight_species=tuple(float(x) for x in weights.heat),
        particle_flux_weight_species=tuple(float(x) for x in weights.particle),
        saturated_heat_flux_species=saturated_heat,
        saturated_particle_flux_species=saturated_particle,
        species=species,
        metadata=_quasilinear_metadata(
            metadata,
            amplitude2=amplitude2,
            channels=options.channels,
        ),
    )


def compute_quasilinear_from_linear_state(
    state: jnp.ndarray | np.ndarray,
    *,
    cache: LinearCache,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    ky: float,
    gamma: float,
    omega: float,
    terms: TermConfig | None = None,
    mode: str = "weights",
    saturation_rule: str = "none",
    amplitude_normalization: str = "phi_rms",
    kperp_average: str = "phi_weighted",
    csat: float = 1.0,
    gamma_floor: float = 0.0,
    include_stable_modes: bool = False,
    channels: Sequence[str] | str = ("es",),
    species_names: Sequence[str] | None = None,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
    metadata: dict[str, Any] | None = None,
) -> QuasilinearTransportResult:
    """Compute quasilinear transport weights from a linear state.

    The returned heat and particle flux weights are divided by the selected
    mode-amplitude normalization, so they are invariant under complex phase
    rotations and real amplitude rescalings of the eigenstate.
    """

    options = _resolve_quasilinear_options(
        mode=mode,
        channels=channels,
        kperp_average=kperp_average,
    )
    weights = _linear_flux_weights_from_state(
        state,
        cache=cache,
        grid=grid,
        geom=geom,
        params=params,
        terms=terms,
        amplitude_normalization=amplitude_normalization,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    species = _quasilinear_species_labels(species_names, weights.heat.size)
    amp2, saturated_heat, saturated_particle = _saturation_outputs(
        options=options,
        weights=weights,
        gamma=gamma,
        saturation_rule=saturation_rule,
        csat=csat,
        gamma_floor=gamma_floor,
        include_stable_modes=include_stable_modes,
    )

    return _build_quasilinear_transport_result(
        ky=ky,
        gamma=gamma,
        omega=omega,
        options=options,
        saturation_rule=saturation_rule,
        amplitude_normalization=amplitude_normalization,
        weights=weights,
        species=species,
        amplitude2=amp2,
        saturated_heat=saturated_heat,
        saturated_particle=saturated_particle,
        metadata=metadata,
    )
