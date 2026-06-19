"""Collection-level VMEC-JAX boundary-chain gate policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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


__all__ = [
    "_boundary_chain_collection_counts",
    "_boundary_chain_collection_decision",
    "_collection_row",
    "_empty_boundary_chain_counts",
]
