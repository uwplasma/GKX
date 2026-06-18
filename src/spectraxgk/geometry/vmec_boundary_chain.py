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
    final_state_abs = abs(final - exact)
    final_state_rel = _relative_error(final, exact, floor=absolute_tolerance)
    frozen_axis_abs = abs(frozen_jvp - exact)
    frozen_axis_rel = _relative_error(frozen_jvp, exact, floor=absolute_tolerance)
    frozen_jvp_vjp_abs = abs(frozen_jvp - frozen_vjp)
    frozen_jvp_vjp_rel = _relative_error(
        frozen_jvp, frozen_vjp, floor=absolute_tolerance
    )
    frozen_fd_jvp_vs_linear_abs = (
        None if frozen_linear_jvp is None else abs(frozen_jvp - frozen_linear_jvp)
    )
    frozen_fd_jvp_vs_linear_rel = (
        None
        if frozen_linear_jvp is None
        else _relative_error(frozen_jvp, frozen_linear_jvp, floor=absolute_tolerance)
    )
    frozen_linear_jvp_vjp_abs = (
        None
        if frozen_linear_jvp is None or frozen_linear_vjp is None
        else abs(frozen_linear_jvp - frozen_linear_vjp)
    )
    frozen_linear_jvp_vjp_rel = (
        None
        if frozen_linear_jvp is None or frozen_linear_vjp is None
        else _relative_error(
            frozen_linear_jvp,
            frozen_linear_vjp,
            floor=absolute_tolerance,
        )
    )
    frozen_fd_vjp_vs_linear_abs = (
        None if frozen_linear_vjp is None else abs(frozen_vjp - frozen_linear_vjp)
    )
    frozen_fd_vjp_vs_linear_rel = (
        None
        if frozen_linear_vjp is None
        else _relative_error(frozen_vjp, frozen_linear_vjp, floor=absolute_tolerance)
    )
    errors: dict[str, float | None] = {
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
        "raw_initial_vs_exact_fd_abs": None if raw is None else abs(raw - exact),
        "raw_initial_vs_exact_fd_rel": (
            None
            if raw is None
            else _relative_error(raw, exact, floor=absolute_tolerance)
        ),
    }
    tangent_ok = bool(
        tangent_diff_rel is not None
        and (
            (tangent_diff_abs is not None and tangent_diff_abs <= absolute_tolerance)
            or tangent_diff_rel <= internal_relative_tolerance
        )
    )
    fd_jvp_linear_ok = _passes_error(
        frozen_fd_jvp_vs_linear_abs,
        frozen_fd_jvp_vs_linear_rel,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=internal_relative_tolerance,
    )
    linear_jvp_vjp_ok = _passes_error(
        frozen_linear_jvp_vjp_abs,
        frozen_linear_jvp_vjp_rel,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=internal_relative_tolerance,
    )
    fd_vjp_linear_ok = _passes_error(
        frozen_fd_vjp_vs_linear_abs,
        frozen_fd_vjp_vs_linear_rel,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=internal_relative_tolerance,
    )
    passes = {
        "final_state_matches_exact_fd": bool(
            final_state_abs <= absolute_tolerance
            or final_state_rel <= exact_relative_tolerance
        ),
        "frozen_axis_matches_exact_fd": bool(
            frozen_axis_abs <= absolute_tolerance
            or frozen_axis_rel <= exact_relative_tolerance
        ),
        "frozen_axis_jvp_vjp_consistent": bool(
            frozen_jvp_vjp_abs <= absolute_tolerance
            or frozen_jvp_vjp_rel <= internal_relative_tolerance
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
            and (
                (errors["raw_initial_vs_exact_fd_abs"] or 0.0) <= absolute_tolerance
                or (errors["raw_initial_vs_exact_fd_rel"] or math.inf)
                <= exact_relative_tolerance
            )
        ),
    }

    norm_ratio = metrics["raw_to_frozen_initial_norm_ratio"]
    branch_sensitive = bool(norm_ratio is not None and norm_ratio > 10.0)
    if not passes["frozen_axis_jvp_vjp_consistent"]:
        classification = "frozen_axis_replay_internally_inconsistent"
        next_action = (
            "debug VMEC-JAX exact-tape JVP/VJP replay; the optimizer derivative "
            "is not internally transposed"
        )
    elif passes["frozen_axis_matches_exact_fd"]:
        classification = "exact_fd_and_frozen_axis_replay_consistent"
        next_action = (
            "use the frozen-axis derivative as an optimization diagnostic; keep "
            "sparse FD checks and solved-equilibrium gates before promotion"
        )
    elif passes["frozen_axis_convention_verified"] and branch_sensitive:
        classification = "frozen_axis_convention_verified_but_exact_fd_branch_sensitive"
        next_action = (
            "raw exact-solve FD is branch-sensitive, but the frozen-axis finite "
            "difference, explicit tangent column, tape JVP, and tape VJP agree; "
            "use only with solved-equilibrium, growth-branch, and projected "
            "line-search gates"
        )
    elif passes["frozen_axis_convention_verified"]:
        classification = "frozen_axis_convention_verified_but_exact_fd_inconsistent"
        next_action = (
            "raw exact-solve FD is inconsistent with the optimizer convention, "
            "but the frozen-axis tangent convention is verified; require "
            "projected admission and matched nonlinear audits before promotion"
        )
    elif branch_sensitive:
        classification = "frozen_axis_replay_consistent_but_exact_fd_branch_sensitive"
        next_action = (
            "tighten VMEC solve convergence and compare against the frozen-axis "
            "finite-difference convention; raw exact-solve FD is moving the "
            "magnetic-axis initialization branch"
        )
    elif not passes["final_state_matches_exact_fd"]:
        classification = "final_state_cotangent_mismatch"
        next_action = (
            "audit the SPECTRAX final-state objective cotangent or the exact "
            "final-state finite-difference branch before blaming boundary replay"
        )
    else:
        classification = "frozen_axis_replay_consistent_but_exact_fd_inconsistent"
        next_action = (
            "treat the raw exact-solve FD as a convergence/branch diagnostic; "
            "increase VMEC iterations or reduce branch sensitivity before promotion"
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
            "counts": {
                "n_total": 0,
                "n_finite": 0,
                "n_frozen_axis_internal_pass": 0,
                "n_frozen_axis_convention_verified": 0,
                "n_exact_fd_consistent": 0,
                "n_branch_sensitive": 0,
                "n_growth_branch_locality_checked": 0,
                "n_growth_branch_locality_passed": 0,
            },
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

    n_total = len(rows)
    n_finite = sum(1 for row in rows if row["finite"])
    n_internal = sum(1 for row in rows if row["frozen_axis_jvp_vjp_consistent"])
    n_exact = sum(1 for row in rows if row["frozen_axis_matches_exact_fd"])
    n_convention = sum(1 for row in rows if row["frozen_axis_convention_verified"])
    n_branch = sum(
        1
        for row in rows
        if row["classification"]
        in {
            "frozen_axis_replay_consistent_but_exact_fd_branch_sensitive",
            "frozen_axis_convention_verified_but_exact_fd_branch_sensitive",
        }
    )
    n_growth_checked = sum(1 for row in rows if row["growth_branch_locality_checked"])
    n_growth_passed = sum(1 for row in rows if row["growth_branch_locality_passed"])
    finite = n_finite == n_total
    all_internal = finite and n_internal == n_total
    if not finite:
        classification = "nonfinite_boundary_chain_collection"
        next_action = (
            "repair nonfinite VMEC/Boozer/SPECTRAX derivatives before using "
            "the boundary-gradient collection"
        )
    elif not all_internal:
        classification = "internal_replay_failure"
        next_action = (
            "debug VMEC-JAX exact-tape replay because at least one frozen-axis "
            "JVP/VJP contraction is not internally transposed"
        )
    elif n_exact == n_total:
        classification = "all_components_exact_fd_and_frozen_axis_consistent"
        next_action = (
            "promote the frozen-axis convention for these sparse components, "
            "while retaining solved-equilibrium and sparse-FD gates"
        )
    elif n_exact > 0 and n_convention == 0 and n_branch > 0:
        classification = "mixed_exact_fd_consistency_with_branch_sensitive_modes"
        next_action = (
            "use frozen-axis derivatives only as diagnostics; exclude or "
            "regularize branch-sensitive modes before projected VMEC updates"
        )
    elif n_convention == n_total:
        classification = "all_components_frozen_axis_convention_verified"
        next_action = (
            "raw exact-solve FD remains inconsistent, but every component "
            "passes the explicit frozen-axis tangent convention; projected "
            "updates may use these directions only with solved-equilibrium, "
            "growth-branch, and nonlinear-audit gates"
        )
    elif n_exact + n_convention > 0 and n_branch > 0:
        classification = "mixed_exact_or_frozen_axis_convention_verified"
        next_action = (
            "use only components with exact-FD consistency or explicit "
            "frozen-axis convention verification; unresolved branch-sensitive "
            "modes remain excluded"
        )
    else:
        classification = "branch_sensitive_boundary_chain_collection"
        next_action = (
            "do not promote this boundary-gradient collection until exact-solve "
            "branch sensitivity is reduced or the frozen-axis convention is "
            "validated against a better-conditioned finite-difference protocol"
        )

    return {
        "kind": "vmec_jax_boundary_chain_collection_summary",
        "finite": finite,
        "classification": classification,
        "exact_relative_tolerance": float(exact_relative_tolerance),
        "internal_relative_tolerance": float(internal_relative_tolerance),
        "absolute_tolerance": float(absolute_tolerance),
        "counts": {
            "n_total": n_total,
            "n_finite": n_finite,
            "n_frozen_axis_internal_pass": n_internal,
            "n_frozen_axis_convention_verified": n_convention,
            "n_exact_fd_consistent": n_exact,
            "n_branch_sensitive": n_branch,
            "n_growth_branch_locality_checked": n_growth_checked,
            "n_growth_branch_locality_passed": n_growth_passed,
        },
        "rows": rows,
        "next_action": next_action,
    }


__all__ = [
    "boundary_chain_summary_from_probe",
    "build_boundary_chain_collection_summary",
    "build_boundary_chain_summary",
]
