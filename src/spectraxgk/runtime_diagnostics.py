"""Runtime diagnostic chunk helpers.

This module owns the GX-style diagnostic slicing, truncation, striding, and
concatenation helpers used by the runtime drivers. Keeping these utilities out
of ``runtime.py`` makes the execution/control-flow layer smaller while
preserving the existing public runtime behavior.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics import ResolvedDiagnostics, SimulationDiagnostics, total_energy


def slice_gx_diagnostics(diag: SimulationDiagnostics, stop: int) -> SimulationDiagnostics:
    """Return the first ``stop`` diagnostic samples."""

    if stop < 0:
        raise ValueError("stop must be >= 0")

    def _slice_optional(arr: np.ndarray | jnp.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        return np.asarray(arr)[:stop, ...]

    def _slice_resolved(resolved: ResolvedDiagnostics | None) -> ResolvedDiagnostics | None:
        if resolved is None:
            return None
        payload: dict[str, np.ndarray | None] = {}
        for field in dataclass_fields(ResolvedDiagnostics):
            value = getattr(resolved, field.name)
            payload[field.name] = None if value is None else np.asarray(value)[:stop, ...]
        return ResolvedDiagnostics(**payload)

    dt_t = np.asarray(diag.dt_t)[:stop]
    Wg_t = np.asarray(diag.Wg_t)[:stop]
    Wphi_t = np.asarray(diag.Wphi_t)[:stop]
    Wapar_t = np.asarray(diag.Wapar_t)[:stop]
    if dt_t.size == 0:
        dt_mean = np.asarray(0.0, dtype=float)
    else:
        dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return SimulationDiagnostics(
        t=np.asarray(diag.t)[:stop],
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=np.asarray(diag.gamma_t)[:stop],
        omega_t=np.asarray(diag.omega_t)[:stop],
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=np.asarray(diag.heat_flux_t)[:stop],
        particle_flux_t=np.asarray(diag.particle_flux_t)[:stop],
        energy_t=np.asarray(total_energy(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))),
        heat_flux_species_t=_slice_optional(diag.heat_flux_species_t),
        particle_flux_species_t=_slice_optional(diag.particle_flux_species_t),
        turbulent_heating_t=_slice_optional(diag.turbulent_heating_t),
        turbulent_heating_species_t=_slice_optional(diag.turbulent_heating_species_t),
        phi_mode_t=_slice_optional(diag.phi_mode_t),
        resolved=_slice_resolved(diag.resolved),
    )


def truncate_gx_diagnostics(diag: SimulationDiagnostics, *, t_max: float) -> SimulationDiagnostics:
    """Keep samples through the first entry that reaches ``t_max``."""

    t_arr = np.asarray(diag.t, dtype=float)
    if t_arr.size == 0:
        return diag
    stop = int(np.searchsorted(t_arr, float(t_max), side="left")) + 1
    stop = min(max(stop, 1), int(t_arr.size))
    return slice_gx_diagnostics(diag, stop)


def stride_gx_diagnostics(diag: SimulationDiagnostics, *, stride: int) -> SimulationDiagnostics:
    """Apply the GX runtime output stride after concatenating chunk diagnostics."""

    stride_use = int(max(stride, 1))
    if stride_use == 1:
        return diag

    def _stride_optional(arr: np.ndarray | jnp.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        return np.asarray(arr)[::stride_use, ...]

    def _stride_resolved(resolved: ResolvedDiagnostics | None) -> ResolvedDiagnostics | None:
        if resolved is None:
            return None
        payload: dict[str, np.ndarray | None] = {}
        for field in dataclass_fields(ResolvedDiagnostics):
            value = getattr(resolved, field.name)
            payload[field.name] = None if value is None else np.asarray(value)[::stride_use, ...]
        return ResolvedDiagnostics(**payload)

    dt_t = np.asarray(diag.dt_t)[::stride_use]
    Wg_t = np.asarray(diag.Wg_t)[::stride_use]
    Wphi_t = np.asarray(diag.Wphi_t)[::stride_use]
    Wapar_t = np.asarray(diag.Wapar_t)[::stride_use]
    if dt_t.size == 0:
        dt_mean = np.asarray(0.0, dtype=float)
    else:
        dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return SimulationDiagnostics(
        t=np.asarray(diag.t)[::stride_use],
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=np.asarray(diag.gamma_t)[::stride_use],
        omega_t=np.asarray(diag.omega_t)[::stride_use],
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=np.asarray(diag.heat_flux_t)[::stride_use],
        particle_flux_t=np.asarray(diag.particle_flux_t)[::stride_use],
        energy_t=np.asarray(total_energy(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))),
        heat_flux_species_t=_stride_optional(diag.heat_flux_species_t),
        particle_flux_species_t=_stride_optional(diag.particle_flux_species_t),
        turbulent_heating_t=_stride_optional(diag.turbulent_heating_t),
        turbulent_heating_species_t=_stride_optional(diag.turbulent_heating_species_t),
        phi_mode_t=_stride_optional(diag.phi_mode_t),
        resolved=_stride_resolved(diag.resolved),
    )


def concat_gx_diagnostics(diags: Sequence[SimulationDiagnostics]) -> SimulationDiagnostics:
    """Concatenate one or more diagnostic chunks."""

    if not diags:
        raise ValueError("at least one diagnostic chunk is required")

    def _concat(name: str) -> np.ndarray:
        return np.concatenate([np.asarray(getattr(diag, name)) for diag in diags], axis=0)

    def _concat_optional(name: str) -> np.ndarray | None:
        values = [getattr(diag, name) for diag in diags]
        if all(value is None for value in values):
            return None
        return np.concatenate([np.asarray(value) for value in values if value is not None], axis=0)

    def _concat_resolved() -> ResolvedDiagnostics | None:
        values = [diag.resolved for diag in diags]
        if all(value is None for value in values):
            return None
        payload: dict[str, np.ndarray | None] = {}
        for field in dataclass_fields(ResolvedDiagnostics):
            series = [None if value is None else getattr(value, field.name) for value in values]
            if all(item is None for item in series):
                payload[field.name] = None
            else:
                payload[field.name] = np.concatenate(
                    [np.asarray(item) for item in series if item is not None],
                    axis=0,
                )
        return ResolvedDiagnostics(**payload)

    dt_t = _concat("dt_t")
    Wg_t = _concat("Wg_t")
    Wphi_t = _concat("Wphi_t")
    Wapar_t = _concat("Wapar_t")
    dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return SimulationDiagnostics(
        t=_concat("t"),
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=_concat("gamma_t"),
        omega_t=_concat("omega_t"),
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=_concat("heat_flux_t"),
        particle_flux_t=_concat("particle_flux_t"),
        energy_t=np.asarray(total_energy(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))),
        heat_flux_species_t=_concat_optional("heat_flux_species_t"),
        particle_flux_species_t=_concat_optional("particle_flux_species_t"),
        turbulent_heating_t=_concat_optional("turbulent_heating_t"),
        turbulent_heating_species_t=_concat_optional("turbulent_heating_species_t"),
        phi_mode_t=_concat_optional("phi_mode_t"),
        resolved=_concat_resolved(),
    )
