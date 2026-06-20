"""Acceptance gates for VMEC-JAX stellarator-optimization candidates.

The helpers here are intentionally independent of the SPECTRAX-GK time
integrator. They answer a narrower question: is a solved VMEC-JAX equilibrium
candidate physically acceptable enough to spend expensive nonlinear GK audit
time on it?
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


DEFAULT_QS_WOUT_SURFACES = tuple(float(x) for x in np.linspace(0.0, 1.0, 11))


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


def _checks_passed(checks: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(bool(check.get("passed")) for check in checks.values())


def _aspect_check(value: float | None, *, target: float, tolerance: float) -> dict[str, Any]:
    error = None if value is None else abs(value - float(target))
    return {
        "value": value,
        "target": float(target),
        "absolute_error": error,
        "absolute_tolerance": float(tolerance),
        "passed": _finite_gate(error, upper=float(tolerance)),
    }


def _mean_iota_check(value: float | None, *, minimum_abs: float) -> dict[str, Any]:
    magnitude = None if value is None else abs(value)
    return {
        "value": magnitude,
        "minimum_abs": float(minimum_abs),
        "margin": None if magnitude is None else magnitude - float(minimum_abs),
        "passed": _finite_gate(magnitude, lower=float(minimum_abs)),
    }


def _profile_floor_passed(
    min_iotas: float | None,
    min_iotaf: float | None,
    floor: float | None,
) -> bool:
    if floor is None:
        return True
    floor_value = float(floor)
    return _finite_gate(min_iotas, lower=floor_value) and _finite_gate(min_iotaf, lower=floor_value)


def _iota_profile_check(
    min_iotas: float | None,
    min_iotaf: float | None,
    *,
    floor: float | None,
    source: str | None = None,
) -> dict[str, Any]:
    check: dict[str, Any] = {
        "minimum_iotas_excluding_axis": min_iotas,
        "minimum_iotaf": min_iotaf,
        "floor": None if floor is None else float(floor),
        "passed": _profile_floor_passed(min_iotas, min_iotaf, floor),
    }
    if source is not None:
        check["source"] = source
    return check


def _profile_minima_from_arrays(iotas: np.ndarray, iotaf: np.ndarray) -> tuple[float | None, float | None]:
    iotas = np.asarray(iotas, dtype=float)
    iotaf = np.asarray(iotaf, dtype=float)
    profile = iotas[1:] if iotas.size > 1 else iotas
    full_profile = iotaf[np.isfinite(iotaf)]
    return (
        _finite_float_or_none(np.nanmin(profile)) if profile.size else None,
        _finite_float_or_none(np.nanmin(full_profile)) if full_profile.size else None,
    )


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
            import jax.numpy as jnp
            import vmec_jax as vj  # type: ignore[import-not-found]

            static = getattr(optimizer, "_static")
            qs = vj.QuasisymmetryRatioResidual(
                helicity_m=int(getattr(optimizer, "_helicity_m", 1) or 1),
                helicity_n=int(getattr(optimizer, "_helicity_n", 0) or 0),
                surfaces=np.arange(0.0, 1.01, 0.1),
            )
            ctx = SimpleNamespace(
                static=static,
                indata=getattr(optimizer, "_indata"),
                signgs=int(getattr(optimizer, "_signgs")),
                flux=getattr(optimizer, "_flux"),
                pressure=jnp.zeros_like(jnp.asarray(getattr(static, "s"))),
            )
            value = _finite_float_or_none(qs.total(ctx, state))
            if value is not None:
                return value
        except Exception:
            pass
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


def _wout_summary(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return {
            "source": str(source.get("source", "mapping")),
            "aspect": _finite_float_or_none(source.get("aspect")),
            "mean_iota": _finite_float_or_none(source.get("mean_iota")),
            "min_iotas_excluding_axis": _finite_float_or_none(source.get("min_iotas_excluding_axis")),
            "min_iotaf": _finite_float_or_none(source.get("min_iotaf")),
        }
    path = Path(source)
    try:
        import netCDF4  # type: ignore[import-not-found]

        with netCDF4.Dataset(path) as dataset:
            aspect = _finite_float_or_none(np.asarray(dataset.variables["aspect"][:]))
            iotas = np.asarray(dataset.variables["iotas"][:], dtype=float)
            iotaf = np.asarray(dataset.variables["iotaf"][:], dtype=float)
    except Exception as exc:
        return {
            "source": str(path),
            "error": f"{type(exc).__name__}: {exc}",
            "aspect": None,
            "mean_iota": None,
            "min_iotas_excluding_axis": None,
            "min_iotaf": None,
        }
    finite_iotas = iotas[np.isfinite(iotas)]
    finite_iotaf = iotaf[np.isfinite(iotaf)]
    profile = finite_iotas[1:] if finite_iotas.size > 1 else finite_iotas
    return {
        "source": str(path),
        "aspect": aspect,
        "mean_iota": _finite_float_or_none(np.nanmean(profile)) if profile.size else None,
        "min_iotas_excluding_axis": _finite_float_or_none(np.nanmin(profile)) if profile.size else None,
        "min_iotaf": _finite_float_or_none(np.nanmin(finite_iotaf)) if finite_iotaf.size else None,
    }


def _wout_quasisymmetry(
    source: str | Path | Mapping[str, Any],
    *,
    helicity_m: int,
    helicity_n: int,
    surfaces: tuple[float, ...],
    ntheta: int,
    nphi: int,
) -> tuple[float | None, str, str | None]:
    if isinstance(source, Mapping):
        value = _finite_float_or_none(source.get("qs_residual", source.get("quasisymmetry")))
        return value, str(source.get("qs_source", "mapping")), None if value is not None else "missing_qs_residual"
    try:
        import vmec_jax as vj  # type: ignore[import-not-found]

        wout = vj.load_wout(source)
        qs = vj.quasisymmetry_ratio_residual_from_wout(
            wout,
            surfaces=np.asarray(surfaces, dtype=float),
            helicity_m=int(helicity_m),
            helicity_n=int(helicity_n),
            ntheta=int(ntheta),
            nphi=int(nphi),
        )
        if isinstance(qs, Mapping):
            value = _finite_float_or_none(qs.get("total"))
        else:
            value = _finite_float_or_none(qs)
        return value, "vmec_jax_wout", None if value is not None else "nonfinite_qs_residual"
    except Exception as exc:
        return None, "vmec_jax_wout_error", f"{type(exc).__name__}: {exc}"


def build_authoritative_wout_candidate_gate(
    wout: str | Path | Mapping[str, Any],
    *,
    target_aspect: float,
    aspect_atol: float,
    min_abs_mean_iota: float,
    qs_residual_max: float,
    iota_profile_floor: float | None,
    helicity_m: int = 1,
    helicity_n: int = 0,
    qs_surfaces: tuple[float, ...] = DEFAULT_QS_WOUT_SURFACES,
    qs_ntheta: int = 63,
    qs_nphi: int = 64,
) -> dict[str, Any]:
    """Build a solved-equilibrium gate directly from a WOUT artifact.

    Use this when the deterministic replayed WOUT is the publication-facing
    equilibrium.  It does not assert that the replayed WOUT matches an
    optimizer-state WOUT; that remains the role of
    :func:`build_wout_reproducibility_gate`.
    """

    summary = _wout_summary(wout)
    aspect = _finite_float_or_none(summary.get("aspect"))
    mean_iota = _finite_float_or_none(summary.get("mean_iota"))
    min_iotas = _finite_float_or_none(summary.get("min_iotas_excluding_axis"))
    min_iotaf = _finite_float_or_none(summary.get("min_iotaf"))
    qs_value, qs_source, qs_error = _wout_quasisymmetry(
        wout,
        helicity_m=int(helicity_m),
        helicity_n=int(helicity_n),
        surfaces=tuple(float(x) for x in qs_surfaces),
        ntheta=int(qs_ntheta),
        nphi=int(qs_nphi),
    )
    checks = {
        "aspect": _aspect_check(aspect, target=target_aspect, tolerance=aspect_atol),
        "mean_iota": _mean_iota_check(mean_iota, minimum_abs=min_abs_mean_iota),
        "iota_profile": _iota_profile_check(
            min_iotas,
            min_iotaf,
            floor=iota_profile_floor,
            source="wout",
        ),
        "quasisymmetry": {
            "value": qs_value,
            "maximum": float(qs_residual_max),
            "margin": None if qs_value is None else float(qs_residual_max) - qs_value,
            "source": qs_source,
            "error": qs_error,
            "helicity_m": int(helicity_m),
            "helicity_n": int(helicity_n),
            "surfaces": [float(x) for x in qs_surfaces],
            "ntheta": int(qs_ntheta),
            "nphi": int(qs_nphi),
            "passed": _finite_gate(qs_value, upper=float(qs_residual_max)),
        },
    }
    passed = _checks_passed(checks)
    return {
        "kind": "vmec_jax_authoritative_wout_candidate_gate",
        "passed": bool(passed),
        "authoritative_wout": summary,
        "checks": checks,
        "claim_level": (
            "deterministic WOUT artifact passes solved-equilibrium admission; "
            "optimizer-state reproducibility must be reported separately"
        ),
        "next_action": (
            "this WOUT may be used as the authoritative equilibrium for downstream SPECTRAX-GK audits"
            if passed
            else "do not use this WOUT for downstream SPECTRAX-GK transport promotion"
        ),
    }


def build_wout_reproducibility_gate(
    reference_wout: str | Path | Mapping[str, Any],
    rerun_wout: str | Path | Mapping[str, Any],
    *,
    target_aspect: float,
    aspect_atol: float,
    min_abs_mean_iota: float,
    iota_profile_floor: float | None,
    mean_iota_repro_atol: float = 5.0e-4,
    aspect_repro_atol: float = 1.0e-6,
    profile_repro_atol: float = 5.0e-4,
) -> dict[str, Any]:
    """Check that a saved VMEC input reproduces the optimizer-state WOUT.

    VMEC-JAX can write both an optimizer-state ``wout_final.nc`` and an
    ``input.final`` deck.  For publication-facing transport claims, the deck
    must reproduce the WOUT when rerun; otherwise downstream SPECTRAX-GK
    metrics may be attached to a different equilibrium than the optimized
    state.  This gate compares the original WOUT against a fresh rerun WOUT and
    also applies the solved-equilibrium aspect/iota/profile admission checks to
    the rerun.
    """

    reference = _wout_summary(reference_wout)
    rerun = _wout_summary(rerun_wout)
    ref_aspect = _finite_float_or_none(reference.get("aspect"))
    rerun_aspect = _finite_float_or_none(rerun.get("aspect"))
    ref_iota = _finite_float_or_none(reference.get("mean_iota"))
    rerun_iota = _finite_float_or_none(rerun.get("mean_iota"))
    ref_min_iotas = _finite_float_or_none(reference.get("min_iotas_excluding_axis"))
    rerun_min_iotas = _finite_float_or_none(rerun.get("min_iotas_excluding_axis"))
    ref_min_iotaf = _finite_float_or_none(reference.get("min_iotaf"))
    rerun_min_iotaf = _finite_float_or_none(rerun.get("min_iotaf"))

    aspect_drift = None if ref_aspect is None or rerun_aspect is None else abs(rerun_aspect - ref_aspect)
    iota_drift = None if ref_iota is None or rerun_iota is None else abs(abs(rerun_iota) - abs(ref_iota))
    min_iotas_drift = (
        None
        if ref_min_iotas is None or rerun_min_iotas is None
        else abs(rerun_min_iotas - ref_min_iotas)
    )
    min_iotaf_drift = (
        None
        if ref_min_iotaf is None or rerun_min_iotaf is None
        else abs(rerun_min_iotaf - ref_min_iotaf)
    )
    checks = {
        "rerun_aspect_admission": _aspect_check(rerun_aspect, target=target_aspect, tolerance=aspect_atol),
        "rerun_mean_iota_admission": _mean_iota_check(rerun_iota, minimum_abs=min_abs_mean_iota),
        "rerun_iota_profile_admission": _iota_profile_check(
            rerun_min_iotas,
            rerun_min_iotaf,
            floor=iota_profile_floor,
        ),
        "aspect_reproducibility": {
            "reference": ref_aspect,
            "rerun": rerun_aspect,
            "absolute_drift": aspect_drift,
            "absolute_tolerance": float(aspect_repro_atol),
            "passed": _finite_gate(aspect_drift, upper=float(aspect_repro_atol)),
        },
        "mean_iota_reproducibility": {
            "reference": None if ref_iota is None else abs(ref_iota),
            "rerun": None if rerun_iota is None else abs(rerun_iota),
            "absolute_drift": iota_drift,
            "absolute_tolerance": float(mean_iota_repro_atol),
            "passed": _finite_gate(iota_drift, upper=float(mean_iota_repro_atol)),
        },
        "iota_profile_reproducibility": {
            "min_iotas_drift": min_iotas_drift,
            "min_iotaf_drift": min_iotaf_drift,
            "absolute_tolerance": float(profile_repro_atol),
            "passed": _finite_gate(min_iotas_drift, upper=float(profile_repro_atol))
            and _finite_gate(min_iotaf_drift, upper=float(profile_repro_atol)),
        },
    }
    passed = _checks_passed(checks)
    return {
        "kind": "vmec_jax_wout_reproducibility_gate",
        "passed": bool(passed),
        "reference_wout": reference,
        "rerun_wout": rerun,
        "checks": checks,
        "claim_level": "saved VMEC input must reproduce optimizer-state WOUT before SPECTRAX-GK transport promotion",
        "next_action": (
            "saved input/WOUT pair is reproducible enough for transport admission"
            if passed
            else "do not promote this saved input; rerun/refine the VMEC-JAX solve until input.final reproduces wout_final.nc"
        ),
    }


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
    qs_residual = None
    qs_source = "history"
    if not isinstance(candidate, Mapping):
        qs_residual = _final_quasisymmetry_from_vmec_result(candidate)
        if qs_residual is not None:
            qs_source = "vmec_jax_state"
    if qs_residual is None:
        qs_residual = _finite_float_or_none(history.get("qs_final"))

    if iota_profiles is None and not isinstance(candidate, Mapping):
        profile_source = "vmec_jax_state"
        iota_profiles = final_iota_profiles_from_vmec_result(candidate)

    min_iota_profile: float | None = None
    min_iotaf_profile: float | None = None
    if iota_profiles is not None:
        min_iota_profile, min_iotaf_profile = _profile_minima_from_arrays(*iota_profiles)
    elif iota_profile_floor is not None:
        profile_source = "missing"

    checks = {
        "aspect": _aspect_check(aspect, target=target_aspect, tolerance=aspect_atol),
        "mean_iota": _mean_iota_check(mean_iota, minimum_abs=min_abs_mean_iota),
        "quasisymmetry": {
            "value": qs_residual,
            "maximum": float(qs_residual_max),
            "margin": None if qs_residual is None else float(qs_residual_max) - qs_residual,
            "source": qs_source,
            "passed": _finite_gate(qs_residual, upper=float(qs_residual_max)),
        },
        "iota_profile": _iota_profile_check(
            min_iota_profile,
            min_iotaf_profile,
            floor=iota_profile_floor,
            source=profile_source,
        ),
    }
    passed = _checks_passed(checks)
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
    "DEFAULT_QS_WOUT_SURFACES",
    "build_authoritative_wout_candidate_gate",
    "build_solved_vmec_candidate_gate",
    "build_wout_reproducibility_gate",
    "final_iota_profiles_from_vmec_result",
]
