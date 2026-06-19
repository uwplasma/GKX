"""Boundary-chain diagnostics for VMEC-JAX/SPECTRAX-GK gradients.

These helpers classify the scalar contractions produced by the expensive
``tools/probe_vmec_jax_boundary_chain.py`` diagnostic.  The diagnostic compares
raw exact-solve finite differences, frozen-axis initial-state finite
differences, and VMEC-JAX exact-tape JVP/VJP contractions.  Keeping the
classification logic in the package makes the paper-facing convention explicit
and unit-testable without launching VMEC solves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import math

_BRANCH_SENSITIVE_BOUNDARY_CHAIN_CLASSES = frozenset(
    {
        "frozen_axis_replay_consistent_but_exact_fd_branch_sensitive",
        "frozen_axis_convention_verified_but_exact_fd_branch_sensitive",
    }
)

_BOUNDARY_CHAIN_COUNT_KEYS = (
    "n_total",
    "n_finite",
    "n_frozen_axis_internal_pass",
    "n_frozen_axis_convention_verified",
    "n_exact_fd_consistent",
    "n_branch_sensitive",
    "n_growth_branch_locality_checked",
    "n_growth_branch_locality_passed",
)


def _finite_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _relative_error(a: float, b: float, *, floor: float) -> float:
    scale = max(abs(float(a)), abs(float(b)), float(floor))
    return abs(float(a) - float(b)) / scale


def _norm_ratio(numerator: float | None, denominator: float | None) -> float | None:
    num = _finite_float(numerator)
    den = _finite_float(denominator)
    if num is None or den is None or den == 0.0:
        return None
    return abs(num) / abs(den)


def _passes_error(
    abs_error: float | None,
    rel_error: float | None,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> bool:
    if abs_error is None:
        return False
    return bool(
        abs_error <= absolute_tolerance
        or (rel_error is not None and rel_error <= relative_tolerance)
    )


def _error_pair(
    first: float | None,
    second: float | None,
    *,
    floor: float,
) -> tuple[float | None, float | None]:
    if first is None or second is None:
        return None, None
    return abs(first - second), _relative_error(first, second, floor=floor)


def _boundary_chain_error_metrics(
    *,
    exact: float,
    final: float,
    frozen_jvp: float,
    frozen_vjp: float,
    frozen_linear_jvp: float | None,
    frozen_linear_vjp: float | None,
    tangent_diff_abs: float | None,
    tangent_diff_rel: float | None,
    raw: float | None,
    absolute_tolerance: float,
) -> dict[str, float | None]:
    final_state_abs, final_state_rel = _error_pair(
        final, exact, floor=absolute_tolerance
    )
    frozen_axis_abs, frozen_axis_rel = _error_pair(
        frozen_jvp, exact, floor=absolute_tolerance
    )
    frozen_jvp_vjp_abs, frozen_jvp_vjp_rel = _error_pair(
        frozen_jvp, frozen_vjp, floor=absolute_tolerance
    )
    frozen_fd_jvp_vs_linear_abs, frozen_fd_jvp_vs_linear_rel = _error_pair(
        frozen_jvp, frozen_linear_jvp, floor=absolute_tolerance
    )
    frozen_linear_jvp_vjp_abs, frozen_linear_jvp_vjp_rel = _error_pair(
        frozen_linear_jvp, frozen_linear_vjp, floor=absolute_tolerance
    )
    frozen_fd_vjp_vs_linear_abs, frozen_fd_vjp_vs_linear_rel = _error_pair(
        frozen_vjp, frozen_linear_vjp, floor=absolute_tolerance
    )
    raw_abs, raw_rel = _error_pair(raw, exact, floor=absolute_tolerance)
    return {
        "final_state_vs_exact_fd_abs": final_state_abs,
        "final_state_vs_exact_fd_rel": final_state_rel,
        "frozen_axis_vs_exact_fd_abs": frozen_axis_abs,
        "frozen_axis_vs_exact_fd_rel": frozen_axis_rel,
        "frozen_axis_jvp_vjp_abs": frozen_jvp_vjp_abs,
        "frozen_axis_jvp_vjp_rel": frozen_jvp_vjp_rel,
        "frozen_axis_fd_jvp_vs_linear_jvp_abs": frozen_fd_jvp_vs_linear_abs,
        "frozen_axis_fd_jvp_vs_linear_jvp_rel": frozen_fd_jvp_vs_linear_rel,
        "frozen_axis_linear_jvp_vjp_abs": frozen_linear_jvp_vjp_abs,
        "frozen_axis_linear_jvp_vjp_rel": frozen_linear_jvp_vjp_rel,
        "frozen_axis_fd_vjp_vs_linear_vjp_abs": frozen_fd_vjp_vs_linear_abs,
        "frozen_axis_fd_vjp_vs_linear_vjp_rel": frozen_fd_vjp_vs_linear_rel,
        "frozen_axis_initial_fd_vs_linear_abs_norm": tangent_diff_abs,
        "frozen_axis_initial_fd_vs_linear_rel": tangent_diff_rel,
        "raw_initial_vs_exact_fd_abs": raw_abs,
        "raw_initial_vs_exact_fd_rel": raw_rel,
    }


def _boundary_chain_passes(
    errors: Mapping[str, float | None],
    *,
    raw: float | None,
    exact_relative_tolerance: float,
    internal_relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, bool]:
    tangent_abs = errors["frozen_axis_initial_fd_vs_linear_abs_norm"]
    tangent_rel = errors["frozen_axis_initial_fd_vs_linear_rel"]
    tangent_ok = bool(
        tangent_rel is not None
        and (
            (tangent_abs is not None and tangent_abs <= absolute_tolerance)
            or tangent_rel <= internal_relative_tolerance
        )
    )
    fd_jvp_linear_ok = _passes_error(
        errors["frozen_axis_fd_jvp_vs_linear_jvp_abs"],
        errors["frozen_axis_fd_jvp_vs_linear_jvp_rel"],
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=internal_relative_tolerance,
    )
    linear_jvp_vjp_ok = _passes_error(
        errors["frozen_axis_linear_jvp_vjp_abs"],
        errors["frozen_axis_linear_jvp_vjp_rel"],
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=internal_relative_tolerance,
    )
    fd_vjp_linear_ok = _passes_error(
        errors["frozen_axis_fd_vjp_vs_linear_vjp_abs"],
        errors["frozen_axis_fd_vjp_vs_linear_vjp_rel"],
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=internal_relative_tolerance,
    )
    return {
        "final_state_matches_exact_fd": _passes_error(
            errors["final_state_vs_exact_fd_abs"],
            errors["final_state_vs_exact_fd_rel"],
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=exact_relative_tolerance,
        ),
        "frozen_axis_matches_exact_fd": _passes_error(
            errors["frozen_axis_vs_exact_fd_abs"],
            errors["frozen_axis_vs_exact_fd_rel"],
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=exact_relative_tolerance,
        ),
        "frozen_axis_jvp_vjp_consistent": _passes_error(
            errors["frozen_axis_jvp_vjp_abs"],
            errors["frozen_axis_jvp_vjp_rel"],
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=internal_relative_tolerance,
        ),
        "frozen_axis_fd_matches_linear_tangent": tangent_ok,
        "frozen_axis_fd_jvp_matches_linear_jvp": fd_jvp_linear_ok,
        "frozen_axis_linear_jvp_vjp_consistent": linear_jvp_vjp_ok,
        "frozen_axis_fd_vjp_matches_linear_vjp": fd_vjp_linear_ok,
        "frozen_axis_convention_verified": bool(
            tangent_ok
            and fd_jvp_linear_ok
            and linear_jvp_vjp_ok
            and fd_vjp_linear_ok
        ),
        "raw_initial_matches_exact_fd": bool(
            raw is not None
            and _passes_error(
                errors["raw_initial_vs_exact_fd_abs"],
                errors["raw_initial_vs_exact_fd_rel"],
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=exact_relative_tolerance,
            )
        ),
    }


def _boundary_chain_summary_decision(
    passes: Mapping[str, bool], *, branch_sensitive: bool
) -> tuple[str, str]:
    """Classify one finite boundary-gradient probe from its pass flags."""

    if not passes["frozen_axis_jvp_vjp_consistent"]:
        return (
            "frozen_axis_replay_internally_inconsistent",
            "debug VMEC-JAX exact-tape JVP/VJP replay; the optimizer derivative "
            "is not internally transposed",
        )
    if passes["frozen_axis_matches_exact_fd"]:
        return (
            "exact_fd_and_frozen_axis_replay_consistent",
            "use the frozen-axis derivative as an optimization diagnostic; keep "
            "sparse FD checks and solved-equilibrium gates before promotion",
        )
    if passes["frozen_axis_convention_verified"] and branch_sensitive:
        return (
            "frozen_axis_convention_verified_but_exact_fd_branch_sensitive",
            "raw exact-solve FD is branch-sensitive, but the frozen-axis finite "
            "difference, explicit tangent column, tape JVP, and tape VJP agree; "
            "use only with solved-equilibrium, growth-branch, and projected "
            "line-search gates",
        )
    if passes["frozen_axis_convention_verified"]:
        return (
            "frozen_axis_convention_verified_but_exact_fd_inconsistent",
            "raw exact-solve FD is inconsistent with the optimizer convention, "
            "but the frozen-axis tangent convention is verified; require "
            "projected admission and matched nonlinear audits before promotion",
        )
    if branch_sensitive:
        return (
            "frozen_axis_replay_consistent_but_exact_fd_branch_sensitive",
            "tighten VMEC solve convergence and compare against the frozen-axis "
            "finite-difference convention; raw exact-solve FD is moving the "
            "magnetic-axis initialization branch",
        )
    if not passes["final_state_matches_exact_fd"]:
        return (
            "final_state_cotangent_mismatch",
            "audit the SPECTRAX final-state objective cotangent or the exact "
            "final-state finite-difference branch before blaming boundary replay",
        )
    return (
        "frozen_axis_replay_consistent_but_exact_fd_inconsistent",
        "treat the raw exact-solve FD as a convergence/branch diagnostic; "
        "increase VMEC iterations or reduce branch sensitivity before promotion",
    )


def build_boundary_chain_summary(
    *,
    exact_fd_cost_gradient: float,
    final_cot_dot_exact_final_fd: float,
    frozen_axis_replay_cost_gradient: float,
    frozen_axis_vjp_cost_gradient: float,
    frozen_axis_linear_replay_cost_gradient: float | None = None,
    frozen_axis_linear_vjp_cost_gradient: float | None = None,
    frozen_axis_initial_fd_vs_linear_abs_norm: float | None = None,
    frozen_axis_initial_fd_vs_linear_rel: float | None = None,
    raw_initial_replay_cost_gradient: float | None = None,
    raw_initial_fd_norm: float | None = None,
    frozen_axis_initial_fd_norm: float | None = None,
    exact_relative_tolerance: float = 1.0e-1,
    internal_relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Classify a boundary-gradient chain probe.

    Args:
        exact_fd_cost_gradient: Central finite difference through plus/minus
            exact VMEC solves.
        final_cot_dot_exact_final_fd: Final-state SPECTRAX-GK cotangent dotted
            into the exact final-state finite-difference direction.
        frozen_axis_replay_cost_gradient: VMEC-JAX tape JVP contraction using
            the frozen-axis initial-state tangent used by the optimizer.
        frozen_axis_vjp_cost_gradient: VMEC-JAX tape VJP contraction projected
            back through the same frozen initial-state map.
        frozen_axis_linear_replay_cost_gradient: Optional contraction using
            VMEC-JAX's explicit frozen-axis tangent column.
        frozen_axis_linear_vjp_cost_gradient: Optional VJP contraction using
            VMEC-JAX's explicit frozen-axis tangent column.
        frozen_axis_initial_fd_vs_linear_abs_norm: Optional norm of the
            frozen-axis finite-difference tangent minus the explicit tangent
            column.
        frozen_axis_initial_fd_vs_linear_rel: Optional relative norm of the
            frozen-axis finite-difference tangent minus the explicit tangent
            column.
        raw_initial_replay_cost_gradient: Optional tape JVP contraction using
            raw plus/minus initial-state finite differences. This diagnoses
            magnetic-axis branch sensitivity, but it is not the optimizer's
            advertised derivative.
    """

    exact = _finite_float(exact_fd_cost_gradient)
    final = _finite_float(final_cot_dot_exact_final_fd)
    frozen_jvp = _finite_float(frozen_axis_replay_cost_gradient)
    frozen_vjp = _finite_float(frozen_axis_vjp_cost_gradient)
    frozen_linear_jvp = _finite_float(frozen_axis_linear_replay_cost_gradient)
    frozen_linear_vjp = _finite_float(frozen_axis_linear_vjp_cost_gradient)
    tangent_diff_abs = _finite_float(frozen_axis_initial_fd_vs_linear_abs_norm)
    tangent_diff_rel = _finite_float(frozen_axis_initial_fd_vs_linear_rel)
    raw = _finite_float(raw_initial_replay_cost_gradient)
    finite = (
        exact is not None
        and final is not None
        and frozen_jvp is not None
        and frozen_vjp is not None
    )

    metrics: dict[str, float | None] = {
        "exact_fd_cost_gradient": exact,
        "final_cot_dot_exact_final_fd": final,
        "frozen_axis_replay_cost_gradient": frozen_jvp,
        "frozen_axis_vjp_cost_gradient": frozen_vjp,
        "frozen_axis_linear_replay_cost_gradient": frozen_linear_jvp,
        "frozen_axis_linear_vjp_cost_gradient": frozen_linear_vjp,
        "frozen_axis_initial_fd_vs_linear_abs_norm": tangent_diff_abs,
        "frozen_axis_initial_fd_vs_linear_rel": tangent_diff_rel,
        "raw_initial_replay_cost_gradient": raw,
        "raw_to_frozen_initial_norm_ratio": _norm_ratio(
            raw_initial_fd_norm, frozen_axis_initial_fd_norm
        ),
    }
    if not finite:
        return {
            "kind": "vmec_jax_boundary_chain_summary",
            "finite": False,
            "classification": "nonfinite_boundary_chain_probe",
            "metrics": metrics,
            "errors": {},
            "passes": {},
            "next_action": "repair nonfinite VMEC/Boozer/SPECTRAX derivatives before interpreting boundary gradients",
        }

    assert exact is not None
    assert final is not None
    assert frozen_jvp is not None
    assert frozen_vjp is not None
    errors = _boundary_chain_error_metrics(
        exact=exact,
        final=final,
        frozen_jvp=frozen_jvp,
        frozen_vjp=frozen_vjp,
        frozen_linear_jvp=frozen_linear_jvp,
        frozen_linear_vjp=frozen_linear_vjp,
        tangent_diff_abs=tangent_diff_abs,
        tangent_diff_rel=tangent_diff_rel,
        raw=raw,
        absolute_tolerance=absolute_tolerance,
    )
    passes = _boundary_chain_passes(
        errors,
        raw=raw,
        exact_relative_tolerance=exact_relative_tolerance,
        internal_relative_tolerance=internal_relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )

    norm_ratio = metrics["raw_to_frozen_initial_norm_ratio"]
    branch_sensitive = bool(norm_ratio is not None and norm_ratio > 10.0)
    classification, next_action = _boundary_chain_summary_decision(
        passes, branch_sensitive=branch_sensitive
    )

    return {
        "kind": "vmec_jax_boundary_chain_summary",
        "finite": True,
        "classification": classification,
        "exact_relative_tolerance": float(exact_relative_tolerance),
        "internal_relative_tolerance": float(internal_relative_tolerance),
        "absolute_tolerance": float(absolute_tolerance),
        "metrics": metrics,
        "errors": errors,
        "passes": passes,
        "next_action": next_action,
    }


def boundary_chain_summary_from_probe(
    payload: Mapping[str, Any], **kwargs: Any
) -> dict[str, Any]:
    """Build a chain summary from a probe JSON payload."""

    return build_boundary_chain_summary(
        exact_fd_cost_gradient=float(payload["exact_fd_cost_gradient"]),
        final_cot_dot_exact_final_fd=float(payload["final_cot_dot_exact_final_fd"]),
        frozen_axis_replay_cost_gradient=float(
            payload["final_cot_dot_tape_jvp_frozen_axis_fd"]
        ),
        frozen_axis_vjp_cost_gradient=float(payload["initial_cot_dot_frozen_axis_fd"]),
        frozen_axis_linear_replay_cost_gradient=payload.get(
            "final_cot_dot_tape_jvp_frozen_axis_linear"
        ),
        frozen_axis_linear_vjp_cost_gradient=payload.get(
            "initial_cot_dot_frozen_axis_linear"
        ),
        frozen_axis_initial_fd_vs_linear_abs_norm=payload.get(
            "frozen_axis_initial_fd_vs_linear_abs_norm"
        ),
        frozen_axis_initial_fd_vs_linear_rel=payload.get(
            "frozen_axis_initial_fd_vs_linear_rel"
        ),
        raw_initial_replay_cost_gradient=payload.get(
            "final_cot_dot_tape_jvp_raw_initial_fd"
        ),
        raw_initial_fd_norm=payload.get("raw_initial_fd_norm"),
        frozen_axis_initial_fd_norm=payload.get("frozen_axis_initial_fd_norm"),
        **kwargs,
    )


def _mapping_value(payload: Any, key: str) -> Any:
    return payload.get(key) if isinstance(payload, Mapping) else None


def _mapping_flag(payload: Any, key: str) -> bool:
    return bool(_mapping_value(payload, key))


def _collection_row(payload: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    """Build one JSON-safe row for a boundary-chain collection summary."""

    passes = summary.get("passes", {})
    errors = summary.get("errors", {})
    metrics = summary.get("metrics", {})
    growth_branch = payload.get("growth_branch_locality")
    growth_checked = isinstance(growth_branch, Mapping) and bool(
        growth_branch.get("enabled", True)
    )
    growth_passed = bool(
        isinstance(growth_branch, Mapping) and growth_branch.get("passed", False)
    )
    return {
        "index": payload.get("index"),
        "name": payload.get("name"),
        "classification": summary.get("classification"),
        "finite": bool(summary.get("finite", False)),
        "frozen_axis_jvp_vjp_consistent": _mapping_flag(
            passes, "frozen_axis_jvp_vjp_consistent"
        ),
        "frozen_axis_matches_exact_fd": _mapping_flag(
            passes, "frozen_axis_matches_exact_fd"
        ),
        "exact_fd_consistent": _mapping_flag(passes, "frozen_axis_matches_exact_fd"),
        "frozen_axis_convention_verified": _mapping_flag(
            passes, "frozen_axis_convention_verified"
        ),
        "final_state_matches_exact_fd": _mapping_flag(
            passes, "final_state_matches_exact_fd"
        ),
        "exact_fd_cost_gradient": _mapping_value(metrics, "exact_fd_cost_gradient"),
        "frozen_axis_replay_cost_gradient": _mapping_value(
            metrics, "frozen_axis_replay_cost_gradient"
        ),
        "frozen_axis_vs_exact_fd_rel": _mapping_value(
            errors, "frozen_axis_vs_exact_fd_rel"
        ),
        "frozen_axis_initial_fd_vs_linear_rel": _mapping_value(
            errors, "frozen_axis_initial_fd_vs_linear_rel"
        ),
        "frozen_axis_linear_jvp_vjp_rel": _mapping_value(
            errors, "frozen_axis_linear_jvp_vjp_rel"
        ),
        "raw_initial_vs_exact_fd_rel": _mapping_value(
            errors, "raw_initial_vs_exact_fd_rel"
        ),
        "growth_branch_locality_checked": growth_checked,
        "growth_branch_locality_passed": growth_passed,
        "growth_branch_locality_classification": _mapping_value(
            growth_branch, "classification"
        ),
    }


def _empty_boundary_chain_counts() -> dict[str, int]:
    """Return the JSON-stable zero-count payload for collection gates."""

    return {key: 0 for key in _BOUNDARY_CHAIN_COUNT_KEYS}


def _boundary_chain_collection_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count boundary-chain collection gates in one policy-owned place."""

    counts = _empty_boundary_chain_counts()
    counts["n_total"] = len(rows)
    counts["n_finite"] = sum(1 for row in rows if bool(row["finite"]))
    counts["n_frozen_axis_internal_pass"] = sum(
        1 for row in rows if bool(row["frozen_axis_jvp_vjp_consistent"])
    )
    counts["n_frozen_axis_convention_verified"] = sum(
        1 for row in rows if bool(row["frozen_axis_convention_verified"])
    )
    counts["n_exact_fd_consistent"] = sum(
        1 for row in rows if bool(row["frozen_axis_matches_exact_fd"])
    )
    counts["n_branch_sensitive"] = sum(
        1
        for row in rows
        if row["classification"] in _BRANCH_SENSITIVE_BOUNDARY_CHAIN_CLASSES
    )
    counts["n_growth_branch_locality_checked"] = sum(
        1 for row in rows if bool(row["growth_branch_locality_checked"])
    )
    counts["n_growth_branch_locality_passed"] = sum(
        1 for row in rows if bool(row["growth_branch_locality_passed"])
    )
    return counts


def _boundary_chain_collection_decision(
    counts: Mapping[str, int],
) -> tuple[bool, str, str]:
    """Classify a boundary-chain collection from precomputed gate counts."""

    n_total = int(counts["n_total"])
    n_finite = int(counts["n_finite"])
    n_internal = int(counts["n_frozen_axis_internal_pass"])
    n_exact = int(counts["n_exact_fd_consistent"])
    n_convention = int(counts["n_frozen_axis_convention_verified"])
    n_branch = int(counts["n_branch_sensitive"])
    finite = n_finite == n_total
    all_internal = finite and n_internal == n_total
    if not finite:
        return (
            finite,
            "nonfinite_boundary_chain_collection",
            "repair nonfinite VMEC/Boozer/SPECTRAX derivatives before using "
            "the boundary-gradient collection",
        )
    if not all_internal:
        return (
            finite,
            "internal_replay_failure",
            "debug VMEC-JAX exact-tape replay because at least one frozen-axis "
            "JVP/VJP contraction is not internally transposed",
        )
    if n_exact == n_total:
        return (
            finite,
            "all_components_exact_fd_and_frozen_axis_consistent",
            "promote the frozen-axis convention for these sparse components, "
            "while retaining solved-equilibrium and sparse-FD gates",
        )
    if n_exact > 0 and n_convention == 0 and n_branch > 0:
        return (
            finite,
            "mixed_exact_fd_consistency_with_branch_sensitive_modes",
            "use frozen-axis derivatives only as diagnostics; exclude or "
            "regularize branch-sensitive modes before projected VMEC updates",
        )
    if n_convention == n_total:
        return (
            finite,
            "all_components_frozen_axis_convention_verified",
            "raw exact-solve FD remains inconsistent, but every component "
            "passes the explicit frozen-axis tangent convention; projected "
            "updates may use these directions only with solved-equilibrium, "
            "growth-branch, and nonlinear-audit gates",
        )
    if n_exact + n_convention > 0 and n_branch > 0:
        return (
            finite,
            "mixed_exact_or_frozen_axis_convention_verified",
            "use only components with exact-FD consistency or explicit "
            "frozen-axis convention verification; unresolved branch-sensitive "
            "modes remain excluded",
        )
    return (
        finite,
        "branch_sensitive_boundary_chain_collection",
        "do not promote this boundary-gradient collection until exact-solve "
        "branch sensitivity is reduced or the frozen-axis convention is "
        "validated against a better-conditioned finite-difference protocol",
    )


def build_boundary_chain_collection_summary(
    probes: Sequence[Mapping[str, Any]],
    *,
    exact_relative_tolerance: float = 1.0e-1,
    internal_relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Summarize several boundary-chain probes as one promotion gate.

    A single coefficient can look well-conditioned while neighboring boundary
    modes still move the raw exact-solve initialization branch.  The collection
    summary keeps the stricter manuscript/release decision explicit: frozen-axis
    JVP/VJP replay must be internally transposed for every component, while
    exact finite-difference agreement is counted separately from branch
    sensitivity.
    """

    if not probes:
        return {
            "kind": "vmec_jax_boundary_chain_collection_summary",
            "finite": False,
            "classification": "empty_boundary_chain_collection",
            "rows": [],
            "counts": _empty_boundary_chain_counts(),
            "next_action": (
                "run at least one boundary-chain probe before interpreting the "
                "VMEC-JAX transport-gradient convention"
            ),
        }

    rows: list[dict[str, Any]] = []
    for payload in probes:
        summary_payload = payload.get("summary")
        summary = (
            dict(summary_payload)
            if isinstance(summary_payload, Mapping)
            else boundary_chain_summary_from_probe(
                payload,
                exact_relative_tolerance=exact_relative_tolerance,
                internal_relative_tolerance=internal_relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
        )
        rows.append(_collection_row(payload, summary))

    counts = _boundary_chain_collection_counts(rows)
    finite, classification, next_action = _boundary_chain_collection_decision(counts)

    return {
        "kind": "vmec_jax_boundary_chain_collection_summary",
        "finite": finite,
        "classification": classification,
        "exact_relative_tolerance": float(exact_relative_tolerance),
        "internal_relative_tolerance": float(internal_relative_tolerance),
        "absolute_tolerance": float(absolute_tolerance),
        "counts": counts,
        "rows": rows,
        "next_action": next_action,
    }


__all__ = [
    "boundary_chain_summary_from_probe",
    "build_boundary_chain_collection_summary",
    "build_boundary_chain_summary",
]
