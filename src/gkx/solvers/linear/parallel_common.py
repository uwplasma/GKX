"""Shared policies for opt-in velocity-parallel linear RHS routes."""

from __future__ import annotations

from typing import Any

import jax

from gkx.operators.linear.params import LinearTerms


def _is_streaming_only_terms(terms: LinearTerms | None) -> bool:
    term_weights = terms if terms is not None else LinearTerms()
    return (
        float(term_weights.streaming) == 1.0
        and float(term_weights.mirror) == 0.0
        and float(term_weights.curvature) == 0.0
        and float(term_weights.gradb) == 0.0
        and float(term_weights.diamagnetic) == 0.0
        and float(term_weights.collisions) == 0.0
        and float(term_weights.hypercollisions) == 0.0
        and float(term_weights.hyperdiffusion) == 0.0
        and float(term_weights.end_damping) == 0.0
        and float(term_weights.apar) == 0.0
        and float(term_weights.bpar) == 0.0
    )


def _is_electrostatic_slice_terms(terms: LinearTerms | None) -> bool:
    term_weights = terms if terms is not None else LinearTerms()
    return (
        float(term_weights.collisions) == 0.0
        and float(term_weights.hypercollisions) == 0.0
        and float(term_weights.hyperdiffusion) == 0.0
        and float(term_weights.end_damping) == 0.0
        and float(term_weights.apar) == 0.0
        and float(term_weights.bpar) == 0.0
    )


def _is_electrostatic_field_terms(terms: LinearTerms | None) -> bool:
    term_weights = terms if terms is not None else LinearTerms()
    return float(term_weights.apar) == 0.0 and float(term_weights.bpar) == 0.0


def _resolve_parallel_devices(
    *, num_devices: int | None = None, devices: Any | None = None
) -> list[Any]:
    """Return an explicit device list for opt-in parallel diagnostics."""

    if devices is None:
        device_list = list(jax.devices())
        if num_devices is not None:
            device_count = int(num_devices)
            if device_count < 1:
                raise ValueError("num_devices must be >= 1")
            if len(device_list) < device_count:
                raise ValueError(
                    f"requested {device_count} devices, but only {len(device_list)} are available"
                )
            device_list = device_list[:device_count]
    else:
        device_list = list(devices)
        if num_devices is not None and int(num_devices) != len(device_list):
            raise ValueError("num_devices must match the explicit devices list length")
    if not device_list:
        raise ValueError("at least one device is required")
    return device_list


__all__ = [
    "_is_electrostatic_field_terms",
    "_is_electrostatic_slice_terms",
    "_is_streaming_only_terms",
    "_resolve_parallel_devices",
]
