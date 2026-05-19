"""Follow-up planning for nonlinear replicate-spread blockers.

The planner consumes the replicate-spread diagnostic and already-known variant
metadata, then selects the smallest cross checks needed to separate seed
variability from timestep sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math


@dataclass(frozen=True)
class NonlinearReplicateFollowupConfig:
    """Options controlling the targeted replicate follow-up plan."""

    include_extra_nominal_seed: bool = True
    extra_seed_increment: int = 1
    max_runs_per_state: int = 3


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _finite_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _variant_key(state: str, label: str) -> tuple[str, str]:
    return str(state), str(label)


def _metadata_lookup(
    variant_metadata: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in variant_metadata:
        state = row.get("state")
        label = row.get("variant_label") or row.get("label")
        if state is None or label is None:
            continue
        seed = _finite_int(row.get("seed"))
        timestep = _finite_float(row.get("timestep") or row.get("dt"))
        if seed is None or timestep is None:
            continue
        out[_variant_key(str(state), str(label))] = {
            "state": str(state),
            "variant_label": str(label),
            "variant_axis": str(row.get("variant_axis") or row.get("axis") or "unknown"),
            "seed": seed,
            "timestep": timestep,
            "source_config": row.get("source_config"),
            "source_output": row.get("source_output"),
        }
    return out


def _row_by_state(rows: Sequence[Mapping[str, Any]], state: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("state")) == state:
            return dict(row)
    return None


def _next_seed(metadata_rows: Sequence[Mapping[str, Any]], *, increment: int) -> int | None:
    seeds = [
        seed
        for seed in (_finite_int(row.get("seed")) for row in metadata_rows)
        if seed is not None
    ]
    if not seeds:
        return None
    return max(seeds) + max(1, int(increment))


def _planned_run(
    *,
    state: str,
    seed: int,
    timestep: float,
    reason: str,
    source_labels: Sequence[str],
) -> dict[str, Any]:
    timestep_label = f"{float(timestep):.12g}".replace(".", "p").replace("-", "m")
    return {
        "state": state,
        "variant_axis": "seed_timestep",
        "variant_label": f"seed{int(seed)}_dt{timestep_label}",
        "seed": int(seed),
        "timestep": float(timestep),
        "reason": reason,
        "source_variant_labels": list(source_labels),
    }


def nonlinear_replicate_followup_plan(
    spread_report: Mapping[str, Any],
    *,
    variant_metadata: Sequence[Mapping[str, Any]],
    case: str = "nonlinear_replicate_followup_plan",
    config: NonlinearReplicateFollowupConfig | None = None,
) -> dict[str, Any]:
    """Return targeted cross-run follow-ups for failed replicate-spread states."""

    cfg = config or NonlinearReplicateFollowupConfig()
    if cfg.max_runs_per_state <= 0:
        raise ValueError("max_runs_per_state must be positive")
    if cfg.extra_seed_increment <= 0:
        raise ValueError("extra_seed_increment must be positive")

    lookup = _metadata_lookup(variant_metadata)
    state_rows_raw = spread_report.get("state_rows")
    state_rows = [row for row in state_rows_raw if isinstance(row, Mapping)] if isinstance(state_rows_raw, Sequence) else []
    failed_states = list(spread_report.get("summary", {}).get("failed_states", [])) if isinstance(spread_report.get("summary"), Mapping) else []
    planned: list[dict[str, Any]] = []
    state_plans: list[dict[str, Any]] = []
    missing_metadata: list[dict[str, Any]] = []

    for raw_state in failed_states:
        state = str(raw_state)
        state_row = _row_by_state(state_rows, state)
        if state_row is None:
            missing_metadata.append({"state": state, "reason": "missing state row in spread report"})
            continue
        classification = str(state_row.get("classification", ""))
        high_label = state_row.get("high_variant_label")
        low_label = state_row.get("low_variant_label")
        state_metadata = [row for key, row in lookup.items() if key[0] == state]
        runs: list[dict[str, Any]] = []
        if classification == "mixed_seed_timestep_spread" and high_label and low_label:
            high = lookup.get(_variant_key(state, str(high_label)))
            low = lookup.get(_variant_key(state, str(low_label)))
            if high is None or low is None:
                missing_metadata.append(
                    {
                        "state": state,
                        "high_variant_label": high_label,
                        "low_variant_label": low_label,
                        "reason": "missing seed/timestep metadata for high or low variant",
                    }
                )
            else:
                runs.append(
                    _planned_run(
                        state=state,
                        seed=int(low["seed"]),
                        timestep=float(high["timestep"]),
                        reason="test whether the low window follows the seed when the timestep is nominal",
                        source_labels=[str(low_label), str(high_label)],
                    )
                )
                runs.append(
                    _planned_run(
                        state=state,
                        seed=int(high["seed"]),
                        timestep=float(low["timestep"]),
                        reason="test whether the high window follows the seed when the timestep is refined",
                        source_labels=[str(high_label), str(low_label)],
                    )
                )
                if cfg.include_extra_nominal_seed:
                    extra_seed = _next_seed(state_metadata, increment=cfg.extra_seed_increment)
                    if extra_seed is not None:
                        runs.append(
                            _planned_run(
                                state=state,
                                seed=extra_seed,
                                timestep=float(high["timestep"]),
                                reason="add one independent nominal-timestep seed after the cross checks",
                                source_labels=[str(high_label)],
                            )
                        )
        elif classification == "seed_spread_limited":
            extra_seed = _next_seed(state_metadata, increment=cfg.extra_seed_increment)
            nominal_dt = None
            for row in state_metadata:
                if str(row.get("variant_axis")) == "seed":
                    nominal_dt = _finite_float(row.get("timestep"))
                    break
            if extra_seed is not None and nominal_dt is not None:
                runs.append(
                    _planned_run(
                        state=state,
                        seed=extra_seed,
                        timestep=nominal_dt,
                        reason="seed spread dominates; add one independent nominal-timestep seed",
                        source_labels=[],
                    )
                )
        elif classification == "timestep_spread_limited":
            nominal_seed = None
            refined_dt = None
            for row in state_metadata:
                if str(row.get("variant_axis")) == "timestep":
                    nominal_seed = _finite_int(row.get("seed"))
                    refined_dt = _finite_float(row.get("timestep"))
                    break
            if nominal_seed is not None and refined_dt is not None:
                runs.append(
                    _planned_run(
                        state=state,
                        seed=nominal_seed,
                        timestep=refined_dt,
                        reason="timestep spread dominates; repeat the refined-timestep replicate before promotion",
                        source_labels=[],
                    )
                )

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[int, float]] = set()
        for row in runs:
            key = (int(row["seed"]), float(row["timestep"]))
            if key not in seen:
                seen.add(key)
                deduped.append(row)
        deduped = deduped[: int(cfg.max_runs_per_state)]
        planned.extend(deduped)
        state_plans.append(
            {
                "state": state,
                "classification": classification,
                "planned_run_count": len(deduped),
                "planned_runs": deduped,
                "recommendation": (
                    "Run these targeted cross variants before rerunning the ensemble and central-FD gates."
                    if deduped
                    else "No runnable follow-up was selected; inspect metadata before spending more GPU time."
                ),
            }
        )

    return {
        "kind": "nonlinear_replicate_followup_plan",
        "claim_level": "targeted_replicate_disambiguation_launch_plan_not_simulation_claim",
        "case": str(case),
        "passed": not planned and not missing_metadata,
        "summary": {
            "failed_state_count": len(failed_states),
            "planned_run_count": len(planned),
            "missing_metadata_count": len(missing_metadata),
            "recommendation": (
                "Run planned cross variants, rebuild the failed ensemble, then rerun the central-FD gate."
                if planned
                else "No additional cross variants are currently required."
            ),
        },
        "state_plans": state_plans,
        "planned_runs": planned,
        "missing_metadata": missing_metadata,
        "config": {
            "include_extra_nominal_seed": bool(cfg.include_extra_nominal_seed),
            "extra_seed_increment": int(cfg.extra_seed_increment),
            "max_runs_per_state": int(cfg.max_runs_per_state),
        },
    }


__all__ = [
    "NonlinearReplicateFollowupConfig",
    "nonlinear_replicate_followup_plan",
]
