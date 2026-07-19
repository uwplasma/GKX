"""Acceptance gates for VMEC-JAX stellarator-optimization candidates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
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
    """Return final solved iota profiles from a vmex result bundle if available.

    ``result`` may carry ``final_wout`` (a ``vmex`` ``WoutData``-like object
    with ``iotas``/``iotaf`` tables) or ``final_equilibrium`` (a
    ``vmex.optimize.Equilibrium``, whose lazily built ``wout`` is used).
    """

    wout = getattr(result, "final_wout", None)
    if wout is None:
        equilibrium = getattr(result, "final_equilibrium", None)
        if equilibrium is not None:
            try:
                wout = equilibrium.wout
            except Exception:
                return None
    if wout is None:
        return None
    iotas = getattr(wout, "iotas", None)
    iotaf = getattr(wout, "iotaf", None)
    if iotas is None or iotaf is None:
        return None
    try:
        return np.asarray(iotas, dtype=float), np.asarray(iotaf, dtype=float)
    except Exception:
        return None


def _final_quasisymmetry_from_vmec_result(result: Any) -> float | None:
    """Return an independent final QS residual from a vmex result bundle.

    Preference order: the standalone ``vmex`` quasisymmetry residual on the
    solved state (``total_state``), then on the wout tables (``total``), then a
    caller-provided ``final_optimizer.quasisymmetry_objective(final_params)``
    fallback. The independent residual deliberately avoids any assembled
    transport-objective block.
    """

    equilibrium = getattr(result, "final_equilibrium", None)
    state = getattr(result, "final_state", None)
    runtime = getattr(result, "final_runtime", None)
    if equilibrium is not None:
        state = state if state is not None else getattr(equilibrium, "state", None)
        runtime = runtime if runtime is not None else getattr(equilibrium, "runtime", None)
    helicity_m = int(getattr(result, "helicity_m", 1) or 1)
    helicity_n = int(getattr(result, "helicity_n", 0) or 0)
    surfaces = np.arange(0.0, 1.01, 0.1)

    if state is not None and runtime is not None:
        try:
            import vmex as vj  # type: ignore[import-not-found]

            qs = vj.optimize.QuasisymmetryRatioResidual(
                surfaces,
                helicity_m=helicity_m,
                helicity_n=helicity_n,
            )
            value = _finite_float_or_none(qs.total_state(state, runtime))
            if value is not None:
                return value
        except Exception:
            pass

    wout = getattr(result, "final_wout", None)
    if wout is None and equilibrium is not None:
        try:
            wout = equilibrium.wout
        except Exception:
            wout = None
    if wout is not None:
        try:
            import vmex as vj  # type: ignore[import-not-found]

            qs = vj.optimize.QuasisymmetryRatioResidual(
                surfaces,
                helicity_m=helicity_m,
                helicity_n=helicity_n,
            )
            value = _finite_float_or_none(qs.total(wout))
            if value is not None:
                return value
        except Exception:
            pass

    optimizer = getattr(result, "final_optimizer", None)
    params = getattr(result, "final_params", None)
    if optimizer is not None and params is not None:
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
        import vmex as vj  # type: ignore[import-not-found]

        wout = vj.read_wout(source)
        qs = vj.optimize.QuasisymmetryRatioResidual(
            np.asarray(surfaces, dtype=float),
            helicity_m=int(helicity_m),
            helicity_n=int(helicity_n),
            ntheta=int(ntheta),
            nphi=int(nphi),
        )
        value = _finite_float_or_none(qs.total(wout))
        return value, "vmex_wout", None if value is not None else "nonfinite_qs_residual"
    except Exception as exc:
        return None, "vmex_wout_error", f"{type(exc).__name__}: {exc}"


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
        "kind": "vmex_authoritative_wout_candidate_gate",
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


def _wout_reproducibility_values(
    reference: Mapping[str, Any],
    rerun: Mapping[str, Any],
) -> dict[str, float | None]:
    return {
        "ref_aspect": _finite_float_or_none(reference.get("aspect")),
        "rerun_aspect": _finite_float_or_none(rerun.get("aspect")),
        "ref_iota": _finite_float_or_none(reference.get("mean_iota")),
        "rerun_iota": _finite_float_or_none(rerun.get("mean_iota")),
        "ref_min_iotas": _finite_float_or_none(reference.get("min_iotas_excluding_axis")),
        "rerun_min_iotas": _finite_float_or_none(rerun.get("min_iotas_excluding_axis")),
        "ref_min_iotaf": _finite_float_or_none(reference.get("min_iotaf")),
        "rerun_min_iotaf": _finite_float_or_none(rerun.get("min_iotaf")),
    }


def _wout_reproducibility_drifts(values: Mapping[str, float | None]) -> dict[str, float | None]:
    ref_aspect = values["ref_aspect"]
    rerun_aspect = values["rerun_aspect"]
    ref_iota = values["ref_iota"]
    rerun_iota = values["rerun_iota"]
    ref_min_iotas = values["ref_min_iotas"]
    rerun_min_iotas = values["rerun_min_iotas"]
    ref_min_iotaf = values["ref_min_iotaf"]
    rerun_min_iotaf = values["rerun_min_iotaf"]
    return {
        "aspect": None if ref_aspect is None or rerun_aspect is None else abs(rerun_aspect - ref_aspect),
        "iota": None if ref_iota is None or rerun_iota is None else abs(abs(rerun_iota) - abs(ref_iota)),
        "min_iotas": None
        if ref_min_iotas is None or rerun_min_iotas is None
        else abs(rerun_min_iotas - ref_min_iotas),
        "min_iotaf": None
        if ref_min_iotaf is None or rerun_min_iotaf is None
        else abs(rerun_min_iotaf - ref_min_iotaf),
    }


def _wout_reproducibility_checks(
    *,
    values: Mapping[str, float | None],
    drifts: Mapping[str, float | None],
    target_aspect: float,
    aspect_atol: float,
    min_abs_mean_iota: float,
    iota_profile_floor: float | None,
    mean_iota_repro_atol: float,
    aspect_repro_atol: float,
    profile_repro_atol: float,
) -> dict[str, dict[str, Any]]:
    return {
        "rerun_aspect_admission": _aspect_check(
            values["rerun_aspect"],
            target=target_aspect,
            tolerance=aspect_atol,
        ),
        "rerun_mean_iota_admission": _mean_iota_check(
            values["rerun_iota"],
            minimum_abs=min_abs_mean_iota,
        ),
        "rerun_iota_profile_admission": _iota_profile_check(
            values["rerun_min_iotas"],
            values["rerun_min_iotaf"],
            floor=iota_profile_floor,
        ),
        "aspect_reproducibility": {
            "reference": values["ref_aspect"],
            "rerun": values["rerun_aspect"],
            "absolute_drift": drifts["aspect"],
            "absolute_tolerance": float(aspect_repro_atol),
            "passed": _finite_gate(drifts["aspect"], upper=float(aspect_repro_atol)),
        },
        "mean_iota_reproducibility": {
            "reference": None if values["ref_iota"] is None else abs(values["ref_iota"]),
            "rerun": None if values["rerun_iota"] is None else abs(values["rerun_iota"]),
            "absolute_drift": drifts["iota"],
            "absolute_tolerance": float(mean_iota_repro_atol),
            "passed": _finite_gate(drifts["iota"], upper=float(mean_iota_repro_atol)),
        },
        "iota_profile_reproducibility": {
            "min_iotas_drift": drifts["min_iotas"],
            "min_iotaf_drift": drifts["min_iotaf"],
            "absolute_tolerance": float(profile_repro_atol),
            "passed": _finite_gate(drifts["min_iotas"], upper=float(profile_repro_atol))
            and _finite_gate(drifts["min_iotaf"], upper=float(profile_repro_atol)),
        },
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
    values = _wout_reproducibility_values(reference, rerun)
    drifts = _wout_reproducibility_drifts(values)
    checks = _wout_reproducibility_checks(
        values=values,
        drifts=drifts,
        target_aspect=target_aspect,
        aspect_atol=aspect_atol,
        min_abs_mean_iota=min_abs_mean_iota,
        iota_profile_floor=iota_profile_floor,
        mean_iota_repro_atol=mean_iota_repro_atol,
        aspect_repro_atol=aspect_repro_atol,
        profile_repro_atol=profile_repro_atol,
    )
    passed = _checks_passed(checks)
    return {
        "kind": "vmex_wout_reproducibility_gate",
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

    ``candidate`` may be a vmex optimization result bundle with a ``history``
    property (and ``final_equilibrium``/``final_wout``/``final_state`` fields)
    or a history mapping loaded from ``history.json``.
    """

    history = _history_from_candidate(candidate)
    aspect = _finite_float_or_none(history.get("aspect_final"))
    mean_iota = _finite_float_or_none(history.get("iota_final"))
    qs_residual = None
    qs_source = "history"
    if not isinstance(candidate, Mapping):
        qs_residual = _final_quasisymmetry_from_vmec_result(candidate)
        if qs_residual is not None:
            qs_source = "vmex_state"
    if qs_residual is None:
        qs_residual = _finite_float_or_none(history.get("qs_final"))

    if iota_profiles is None and not isinstance(candidate, Mapping):
        profile_source = "vmex_state"
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
        "kind": "vmex_solved_wout_candidate_gate",
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
