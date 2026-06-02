"""Acceptance gates for VMEC-JAX stellarator-optimization candidates.

The helpers here are intentionally independent of the SPECTRAX-GK time
integrator. They answer a narrower question: is a solved VMEC-JAX equilibrium
candidate physically acceptable enough to spend expensive nonlinear GK audit
time on it?
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np


def _finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if np.isfinite(result) else None


def _finite_gate(value: float | None, *, lower: float | None = None, upper: float | None = None) -> bool:
    if value is None:
        return False
    if lower is not None and value < float(lower):
        return False
    if upper is not None and value > float(upper):
        return False
    return True


def final_iota_profiles_from_vmec_result(result: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return final solved iota profiles from a VMEC-JAX result if available."""

    state = getattr(result, "final_state", None)
    optimizer = getattr(result, "final_optimizer", None)
    if state is None or optimizer is None:
        return None
    try:
        import vmec_jax as vj  # type: ignore[import-not-found]

        _chips, iotas, iotaf = vj.equilibrium_iota_profiles_from_state(
            state=state,
            static=getattr(optimizer, "_static"),
            indata=getattr(optimizer, "_indata"),
            signgs=int(getattr(optimizer, "_signgs")),
        )
    except Exception:
        return None
    return np.asarray(iotas, dtype=float), np.asarray(iotaf, dtype=float)


def _final_quasisymmetry_from_vmec_result(result: Any) -> float | None:
    """Return an independent final QS residual from a VMEC-JAX result."""

    optimizer = getattr(result, "final_optimizer", None)
    if optimizer is None:
        return None
    state = getattr(result, "final_state", None)
    if state is not None:
        try:
            residuals = getattr(optimizer, "_evaluate_residuals_from_state")(state)
            qs_total = getattr(optimizer, "_qs_total_from_state")(state, residuals)
            value = _finite_float_or_none(qs_total)
            if value is not None:
                return value
        except Exception:
            pass
    params = getattr(result, "final_params", None)
    if params is not None:
        try:
            return _finite_float_or_none(getattr(optimizer, "quasisymmetry_objective")(params))
        except Exception:
            return None
    return None


def _history_from_candidate(candidate: Any) -> Mapping[str, Any]:
    if isinstance(candidate, Mapping):
        return candidate
    history = getattr(candidate, "history", None)
    return history if isinstance(history, Mapping) else {}


def build_solved_vmec_candidate_gate(
    candidate: Any,
    *,
    target_aspect: float,
    aspect_atol: float,
    min_abs_mean_iota: float,
    qs_residual_max: float,
    iota_profile_floor: float | None,
    iota_profiles: tuple[np.ndarray, np.ndarray] | None = None,
    profile_source: str = "provided",
) -> dict[str, Any]:
    """Build a JSON-safe solved-equilibrium gate report.

    ``candidate`` may be a VMEC-JAX optimization result with a ``history``
    property or a history mapping loaded from ``history.json``.
    """

    history = _history_from_candidate(candidate)
    aspect = _finite_float_or_none(history.get("aspect_final"))
    mean_iota = _finite_float_or_none(history.get("iota_final"))
    mean_abs_iota = None if mean_iota is None else abs(mean_iota)
    qs_residual = None
    qs_source = "history"
    if not isinstance(candidate, Mapping):
        qs_residual = _final_quasisymmetry_from_vmec_result(candidate)
        if qs_residual is not None:
            qs_source = "vmec_jax_state"
    if qs_residual is None:
        qs_residual = _finite_float_or_none(history.get("qs_final"))
    aspect_error = None if aspect is None else abs(aspect - float(target_aspect))

    if iota_profiles is None and not isinstance(candidate, Mapping):
        profile_source = "vmec_jax_state"
        iota_profiles = final_iota_profiles_from_vmec_result(candidate)

    min_iota_profile: float | None = None
    min_iotaf_profile: float | None = None
    profile_passed = iota_profile_floor is None
    if iota_profiles is not None:
        iotas, iotaf = iota_profiles
        iotas = np.asarray(iotas, dtype=float)
        iotaf = np.asarray(iotaf, dtype=float)
        profile = iotas[1:] if iotas.size > 1 else iotas
        full_profile = iotaf[np.isfinite(iotaf)]
        min_iota_profile = _finite_float_or_none(np.nanmin(profile)) if profile.size else None
        min_iotaf_profile = _finite_float_or_none(np.nanmin(full_profile)) if full_profile.size else None
        if iota_profile_floor is not None:
            profile_passed = _finite_gate(min_iota_profile, lower=float(iota_profile_floor)) and _finite_gate(
                min_iotaf_profile,
                lower=float(iota_profile_floor),
            )
    elif iota_profile_floor is not None:
        profile_source = "missing"
        profile_passed = False

    checks = {
        "aspect": {
            "value": aspect,
            "target": float(target_aspect),
            "absolute_error": aspect_error,
            "absolute_tolerance": float(aspect_atol),
            "passed": _finite_gate(aspect_error, upper=float(aspect_atol)),
        },
        "mean_iota": {
            "value": mean_abs_iota,
            "minimum_abs": float(min_abs_mean_iota),
            "margin": None if mean_abs_iota is None else mean_abs_iota - float(min_abs_mean_iota),
            "passed": _finite_gate(mean_abs_iota, lower=float(min_abs_mean_iota)),
        },
        "quasisymmetry": {
            "value": qs_residual,
            "maximum": float(qs_residual_max),
            "margin": None if qs_residual is None else float(qs_residual_max) - qs_residual,
            "source": qs_source,
            "passed": _finite_gate(qs_residual, upper=float(qs_residual_max)),
        },
        "iota_profile": {
            "minimum_iotas_excluding_axis": min_iota_profile,
            "minimum_iotaf": min_iotaf_profile,
            "floor": None if iota_profile_floor is None else float(iota_profile_floor),
            "source": profile_source,
            "passed": bool(profile_passed),
        },
    }
    passed = all(bool(cast(Mapping[str, Any], check).get("passed")) for check in checks.values())
    return {
        "kind": "vmec_jax_solved_wout_candidate_gate",
        "passed": bool(passed),
        "checks": checks,
        "claim_level": "solved VMEC candidate gate before expensive SPECTRAX-GK nonlinear transport audit",
        "next_action": (
            "candidate may proceed to matched long-window nonlinear transport audits"
            if passed
            else "do not promote this candidate; refine constraints or reduce/re-scale the transport residual"
        ),
    }


__all__ = [
    "build_solved_vmec_candidate_gate",
    "final_iota_profiles_from_vmec_result",
]
